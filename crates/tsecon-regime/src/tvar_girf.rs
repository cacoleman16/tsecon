//! Regime-dependent **generalized impulse responses** of the two-regime
//! threshold VAR — the Koop-Pesaran-Potter (1996) simulation the
//! [`crate::threshold_var`] estimator declined to fake with linear tools.
//!
//! The engine lives in `tsecon-var` ([`tsecon_var::girf`]); this module
//! adapts the fitted TVAR to it and chooses the conditioning histories.
//!
//! # The model the engine simulates
//!
//! With `L = max(p, d)` rows of history (rows oldest first; row `L - d`
//! holds `y_{t-d}`, row `L - 1` holds `y_{t-1}`):
//!
//! * **state** — `0` (low) if `y_{threshold_index, t-d} <= threshold`,
//!   `1` (high) otherwise — the regime of period `t`, decided by the
//!   history alone, exactly as in the estimator;
//! * **covariance** — the fitted per-regime ML residual covariances
//!   `sigma_low` / `sigma_high`; every innovation along a simulated path is
//!   drawn from the covariance of the regime that path is in at that
//!   period, and the impact shock is scaled by the covariance of the regime
//!   the history is in at the shock date (Balke 2000's regime-by-regime
//!   draws; R `tsDyn`'s `GIRF` pools residuals because its `TVAR` fits one
//!   covariance — we could not run it, CRAN being unreachable from the
//!   build container, so its exact draw scheme is stated from its
//!   documentation, not verified);
//! * **step** — `y_t = A_s' x_t + u_t` with the regime-`s` coefficient
//!   block `[const?, y_{t-1}, …, y_{t-p}]`, switching regime period by
//!   period as the simulated threshold variable crosses `threshold`.
//!
//! # Histories
//!
//! The sample's actual lag windows, one per `t >= L` (the shock hits
//! period `t`), in time order: all of them ([`GirfRegime::All`]), only
//! those whose threshold variable puts period `t` in one regime
//! ([`GirfRegime::Low`] / [`GirfRegime::High`]), and optionally a seeded
//! subsample of `histories` of the selected windows
//! ([`tsecon_var::subsample_indices`]). `girf_low_regime` /
//! `girf_high_regime` average the *same* history-conditional responses
//! within each regime, so regime dependence is visible without a second
//! simulation.
//!
//! **Validation grade (honest):** the engine's exact linear reduction is
//! pinned at 1e-12 (`tsecon-var`: statsmodels' Cholesky IRF and the
//! Pesaran-Shin closed form); the regime-switching simulation is pinned
//! against an independent NumPy transcription of the documented draw
//! order (`fixtures/generate_girf_fixtures.py`, same Philox streams) at
//! 1e-10, and its statistical content — sign asymmetry, size
//! non-proportionality, regime dependence beyond the Monte Carlo error,
//! `1/sqrt(n_draws)` convergence, antithetic variance reduction — is
//! measured by the seeded property tests in `tests/tvar_girf_properties.rs`
//! and quoted in the model card. No third-party TVAR GIRF runs in this
//! container (no `tsDyn`), so there is no cross-package golden.

use crate::error::RegimeError;
use crate::linsolve::cholesky;
use crate::tvar::{threshold_var, TvarFit};
use tsecon_var::{girf, subsample_indices, Girf, GirfModel, GirfOptions, GirfShock, VarError};

/// Which histories condition the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GirfRegime {
    /// Every lag window of the sample.
    All,
    /// Windows whose shock-date regime is the low one (`z_t <= threshold`).
    Low,
    /// Windows whose shock-date regime is the high one (`z_t > threshold`).
    High,
}

/// Options of [`tvar_girf`] / [`threshold_var_girf`].
#[derive(Debug, Clone, PartialEq)]
pub struct TvarGirfOptions {
    /// The shock (see [`tsecon_var::girf`]).
    pub shock: GirfShock,
    /// Last horizon `H`.
    pub horizon: usize,
    /// Future-innovation draws per history.
    pub n_draws: usize,
    /// Reproducibility seed (simulation and history subsample).
    pub seed: u64,
    /// Antithetic `(+z, −z)` pairs (needs an even `n_draws`).
    pub antithetic: bool,
    /// Quantile levels of the across-history / across-draw bands.
    pub bands: (f64, f64),
    /// Which histories condition the response.
    pub regime: GirfRegime,
    /// Seeded subsample size of the selected histories (`None` = all).
    pub histories: Option<usize>,
}

/// Output of [`tvar_girf`]. Path arrays are `[h][variable]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TvarGirf {
    /// Mean over the used histories of the history-conditional GIRFs.
    pub girf: Vec<Vec<f64>>,
    /// `bands.0` quantile across the used histories.
    pub lower: Vec<Vec<f64>>,
    /// `bands.1` quantile across the used histories.
    pub upper: Vec<Vec<f64>>,
    /// History-conditional GIRFs, `[history][h][variable]`.
    pub per_history: Vec<Vec<Vec<f64>>>,
    /// Monte Carlo standard error of `girf` (`NaN` below two effective
    /// draws).
    pub mc_se: Vec<Vec<f64>>,
    /// Across-draw spread of one realized paired difference (root mean over
    /// histories; `NaN` when `n_draws < 2`).
    pub draw_sd: Vec<Vec<f64>>,
    /// Mean over histories of the per-history `bands.0` across-draw quantile.
    pub draw_lower: Vec<Vec<f64>>,
    /// Mean over histories of the per-history `bands.1` across-draw quantile.
    pub draw_upper: Vec<Vec<f64>>,
    /// Mean over the used low-regime histories (`None` if none).
    pub girf_low_regime: Option<Vec<Vec<f64>>>,
    /// Mean over the used high-regime histories (`None` if none).
    pub girf_high_regime: Option<Vec<Vec<f64>>>,
    /// Regime (0 low, 1 high) of each used history at the shock date.
    pub history_regimes: Vec<usize>,
    /// The shock date `t` (0-based sample row) of each used history.
    pub history_times: Vec<usize>,
    /// Number of histories used.
    pub n_histories: usize,
    /// Used histories in the low regime.
    pub n_low_histories: usize,
    /// Used histories in the high regime.
    pub n_high_histories: usize,
    /// Draws per history.
    pub n_draws: usize,
    /// Independent draw units per history.
    pub n_effective_draws: usize,
    /// The raw impact innovation vector for a history in the low regime
    /// (`[0]`) and the high regime (`[1]`).
    pub shock_by_regime: Vec<Vec<f64>>,
    /// The fitted threshold.
    pub threshold: f64,
    /// The fitted delay.
    pub delay: usize,
}

/// A fitted two-regime TVAR as a [`GirfModel`] (public so the reduction
/// tests can build one with equal regimes).
#[derive(Debug, Clone, PartialEq)]
pub struct TvarModel {
    k: usize,
    p: usize,
    delay: usize,
    threshold_index: usize,
    threshold: f64,
    constant: bool,
    len: usize,
    coefs: [Vec<Vec<f64>>; 2],
    sigma: [Vec<Vec<f64>>; 2],
}

impl TvarModel {
    /// Adapt a fit (its `p` and `constant` are read off `n_regressors`:
    /// `m = k p + constant` with `k >= 2` is unambiguous).
    ///
    /// # Errors
    ///
    /// [`RegimeError::InvalidSpec`] / [`RegimeError::DimensionMismatch`] /
    /// [`RegimeError::NonFinite`] for a fit whose blocks are inconsistent.
    pub fn from_fit(fit: &TvarFit) -> Result<Self, RegimeError> {
        let k = fit.neqs;
        if k < 2 {
            return Err(RegimeError::InvalidSpec {
                what: "a threshold VAR fit needs at least two series",
            });
        }
        let m = fit.n_regressors;
        let constant = m % k == 1;
        let p = m / k;
        if p == 0 || (m % k != 0 && m % k != 1) {
            return Err(RegimeError::InvalidSpec {
                what: "the fit's n_regressors is not k*p or k*p + 1 for some p >= 1",
            });
        }
        if fit.delay == 0 {
            return Err(RegimeError::InvalidParameter {
                name: "delay",
                value: 0.0,
                requirement: "delay >= 1 in the fit (every entry of delays is a \
                              candidate delay)",
            });
        }
        if fit.threshold_index >= k {
            return Err(RegimeError::InvalidParameter {
                name: "threshold_index",
                value: fit.threshold_index as f64,
                requirement: "threshold_index < n_series in the fit",
            });
        }
        if !fit.threshold.is_finite() {
            return Err(RegimeError::NonFinite {
                what: "the fitted threshold",
            });
        }
        for (name, block) in [
            ("coefs_low", &fit.coefs_low),
            ("coefs_high", &fit.coefs_high),
        ] {
            if block.len() != k {
                return Err(RegimeError::DimensionMismatch {
                    what: "a regime coefficient block must have k equation rows",
                    expected: k,
                    actual: block.len(),
                });
            }
            for row in block {
                if row.len() != m {
                    return Err(RegimeError::DimensionMismatch {
                        what: "a regime coefficient row must have n_regressors entries",
                        expected: m,
                        actual: row.len(),
                    });
                }
                if row.iter().any(|v| !v.is_finite()) {
                    return Err(RegimeError::NonFinite {
                        what: if name == "coefs_low" {
                            "the low-regime coefficients"
                        } else {
                            "the high-regime coefficients"
                        },
                    });
                }
            }
        }
        for (low, block) in [(true, &fit.sigma_low), (false, &fit.sigma_high)] {
            if block.len() != k || block.iter().any(|r| r.len() != k) {
                return Err(RegimeError::DimensionMismatch {
                    what: "a regime covariance must be k x k",
                    expected: k,
                    actual: block.len(),
                });
            }
            let mut flat = vec![0.0_f64; k * k];
            for i in 0..k {
                for j in 0..k {
                    let v = 0.5 * (block[i][j] + block[j][i]);
                    if !v.is_finite() {
                        return Err(RegimeError::NonFinite {
                            what: if low {
                                "the low-regime residual covariance"
                            } else {
                                "the high-regime residual covariance"
                            },
                        });
                    }
                    flat[i * k + j] = v;
                }
            }
            if cholesky(&flat, k).is_none() {
                return Err(RegimeError::Singular {
                    what: if low {
                        "the low-regime residual covariance (sigma_low) has no Cholesky \
                         factor — a degenerate regime; raise trim or lower p"
                    } else {
                        "the high-regime residual covariance (sigma_high) has no Cholesky \
                         factor — a degenerate regime; raise trim or lower p"
                    },
                });
            }
        }
        Ok(Self {
            k,
            p,
            delay: fit.delay,
            threshold_index: fit.threshold_index,
            threshold: fit.threshold,
            constant,
            len: p.max(fit.delay),
            coefs: [fit.coefs_low.clone(), fit.coefs_high.clone()],
            sigma: [fit.sigma_low.clone(), fit.sigma_high.clone()],
        })
    }

    /// The history length `max(p, delay)`.
    pub fn history_len(&self) -> usize {
        self.len
    }
}

impl GirfModel for TvarModel {
    fn n_vars(&self) -> usize {
        self.k
    }
    fn history_len(&self) -> usize {
        self.len
    }
    fn n_states(&self) -> usize {
        2
    }
    fn state(&self, history: &[f64]) -> usize {
        let z = history[(self.len - self.delay) * self.k + self.threshold_index];
        usize::from(z > self.threshold)
    }
    fn covariance(&self, state: usize) -> Vec<Vec<f64>> {
        self.sigma[state].clone()
    }
    fn step(&self, history: &[f64], innovation: &[f64], out: &mut [f64]) {
        let (k, p, len) = (self.k, self.p, self.len);
        let a = &self.coefs[self.state(history)];
        let c0 = usize::from(self.constant);
        for (r, row) in a.iter().enumerate() {
            let mut v = innovation[r];
            if self.constant {
                v += row[0];
            }
            for lag in 1..=p {
                let yrow = &history[(len - lag) * k..(len - lag + 1) * k];
                let arow = &row[c0 + (lag - 1) * k..c0 + lag * k];
                for c in 0..k {
                    v += arow[c] * yrow[c];
                }
            }
            out[r] = v;
        }
    }
}

/// Generalized impulse responses of a fitted TVAR from the lag windows of
/// `endog` (the sample it was fitted on: `T` rows of `k` series).
///
/// # Errors
///
/// * [`RegimeError::DimensionMismatch`] / [`RegimeError::NonFinite`] /
///   [`RegimeError::InsufficientData`] for a sample that does not match the
///   fit or is too short for one lag window;
/// * [`RegimeError::InvalidParameter`] for `histories = Some(0)`, an
///   invalid shock variable / size / horizon / draw count / bands (the
///   engine's teaching messages);
/// * [`RegimeError::InvalidSpec`] when the chosen regime holds no history;
/// * [`RegimeError::Singular`] for a regime covariance without a Cholesky
///   factor; [`RegimeError::NonFinite`] if the simulated paths overflow (an
///   explosive regime).
pub fn tvar_girf(
    fit: &TvarFit,
    endog: &[Vec<f64>],
    opts: &TvarGirfOptions,
) -> Result<TvarGirf, RegimeError> {
    let model = TvarModel::from_fit(fit)?;
    let k = model.k;
    let len = model.len;
    if endog.iter().any(|r| r.len() != k) {
        return Err(RegimeError::DimensionMismatch {
            what: "every row of the data matrix must have one entry per series of the fit",
            expected: k,
            actual: endog.iter().find(|r| r.len() != k).map_or(0, Vec::len),
        });
    }
    if endog.iter().flatten().any(|v| !v.is_finite()) {
        return Err(RegimeError::NonFinite {
            what: "the data matrix (the TVAR requires finite observations)",
        });
    }
    if endog.len() < len + 1 {
        return Err(RegimeError::InsufficientData {
            what: "data under the requested p and delay (a GIRF history is a window \
                   of max(p, delay) rows followed by the shock date)",
            needed: len + 1,
            got: endog.len(),
        });
    }
    if opts.histories == Some(0) {
        return Err(RegimeError::InvalidParameter {
            name: "histories",
            value: 0.0,
            requirement: "histories >= 1 (the number of lag windows to condition on) or \
                          None for every window of the chosen regime",
        });
    }

    // Every lag window with its shock-date regime, in time order.
    let mut windows: Vec<(usize, usize, Vec<f64>)> = Vec::with_capacity(endog.len() - len);
    for t in len..endog.len() {
        let mut h = Vec::with_capacity(len * k);
        for row in &endog[t - len..t] {
            h.extend_from_slice(row);
        }
        let regime = model.state(&h);
        let keep = match opts.regime {
            GirfRegime::All => true,
            GirfRegime::Low => regime == 0,
            GirfRegime::High => regime == 1,
        };
        if keep {
            windows.push((t, regime, h));
        }
    }
    if windows.is_empty() {
        return Err(RegimeError::InvalidSpec {
            what: "no history falls in the requested regime: the sample never visits it \
                   at a usable date — use regime = \"all\" or the other regime",
        });
    }
    let selected: Vec<(usize, usize, Vec<f64>)> = match opts.histories {
        Some(m) if m < windows.len() => subsample_indices(windows.len(), m, opts.seed)
            .into_iter()
            .map(|i| windows[i].clone())
            .collect(),
        _ => windows,
    };
    let histories: Vec<Vec<f64>> = selected.iter().map(|w| w.2.clone()).collect();
    let engine_opts = GirfOptions {
        shock: opts.shock.clone(),
        horizon: opts.horizon,
        n_draws: opts.n_draws,
        seed: opts.seed,
        antithetic: opts.antithetic,
        bands: opts.bands,
    };
    let g: Girf = girf(&model, &histories, &engine_opts).map_err(map_engine_error)?;

    let history_times: Vec<usize> = selected.iter().map(|w| w.0).collect();
    let history_regimes = g.history_states.clone();
    let regime_mean = |which: usize| -> Option<Vec<Vec<f64>>> {
        let idx: Vec<usize> = (0..g.n_histories)
            .filter(|&i| history_regimes[i] == which)
            .collect();
        if idx.is_empty() {
            return None;
        }
        let n = idx.len() as f64;
        let hh = g.horizon + 1;
        let mut acc = vec![vec![0.0_f64; k]; hh];
        for &i in &idx {
            for (row, src) in acc.iter_mut().zip(&g.per_history[i]) {
                for (v, s) in row.iter_mut().zip(src) {
                    *v += s;
                }
            }
        }
        for row in &mut acc {
            for v in row.iter_mut() {
                *v /= n;
            }
        }
        Some(acc)
    };
    let girf_low_regime = regime_mean(0);
    let girf_high_regime = regime_mean(1);
    let n_low_histories = history_regimes.iter().filter(|&&r| r == 0).count();
    let n_high_histories = history_regimes.len() - n_low_histories;

    Ok(TvarGirf {
        girf: g.girf,
        lower: g.lower,
        upper: g.upper,
        per_history: g.per_history,
        mc_se: g.mc_se,
        draw_sd: g.draw_sd,
        draw_lower: g.draw_lower,
        draw_upper: g.draw_upper,
        girf_low_regime,
        girf_high_regime,
        history_regimes,
        history_times,
        n_histories: g.n_histories,
        n_low_histories,
        n_high_histories,
        n_draws: g.n_draws,
        n_effective_draws: g.n_effective_draws,
        shock_by_regime: g.shock_by_state,
        threshold: fit.threshold,
        delay: fit.delay,
    })
}

/// Fit the TVAR ([`threshold_var`]) and simulate its generalized impulse
/// responses ([`tvar_girf`]) in one call.
///
/// # Errors
///
/// Those of [`threshold_var`] and [`tvar_girf`].
#[allow(clippy::too_many_arguments)]
pub fn threshold_var_girf(
    endog: &[Vec<f64>],
    p: usize,
    threshold_index: usize,
    delays: &[usize],
    trim: f64,
    constant: bool,
    opts: &TvarGirfOptions,
) -> Result<(TvarFit, TvarGirf), RegimeError> {
    let fit = threshold_var(endog, p, threshold_index, delays, trim, constant)?;
    let g = tvar_girf(&fit, endog, opts)?;
    Ok((fit, g))
}

/// The engine's errors in this crate's vocabulary. The engine validates the
/// shock, horizon, draw count and bands with teaching messages; the
/// structural cases are impossible after [`TvarModel::from_fit`] and the
/// history construction above.
fn map_engine_error(e: VarError) -> RegimeError {
    match e {
        VarError::InvalidParameter {
            name,
            value,
            requirement,
        } => RegimeError::InvalidParameter {
            name,
            value,
            requirement,
        },
        VarError::NonFinite { what, .. } => RegimeError::NonFinite { what },
        VarError::NotPositiveDefinite { what } => RegimeError::Singular { what },
        VarError::InvalidArgument { what } => RegimeError::InvalidSpec { what },
        VarError::MemoryBudget {
            what,
            bytes,
            budget,
        } => RegimeError::MemoryBudget {
            what,
            bytes,
            budget,
        },
        VarError::Dimension {
            what,
            expected,
            got,
        } => RegimeError::DimensionMismatch {
            what,
            expected,
            actual: got,
        },
        VarError::Linalg(_) | VarError::Stats(_) | VarError::InsufficientObservations { .. } => {
            RegimeError::InvalidSpec {
                what: "the GIRF engine rejected the fitted threshold VAR",
            }
        }
    }
}
