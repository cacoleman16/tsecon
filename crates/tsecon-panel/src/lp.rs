//! Panel local projections (Jordà 2005) with fixed effects.
//!
//! For each horizon `h`, a separate within (fixed-effects) regression of
//! the horizon-`h` outcome on a common observed shock plus lagged
//! controls:
//!
//! ```text
//! y_{i,t+h} = alpha_i^(h) + beta_h shock_t
//!           + sum_{l=1..Ls} gamma_l^(h) shock_{t-l}
//!           + sum_{l=1..Ly} delta_l^(h) y_{i,t-l} + e_{i,t+h}
//! ```
//!
//! `beta_h` is the impulse response at horizon `h` (Jordà 2005; the panel
//! variant is the workhorse of the Jordà-Schularick-Taylor macrohistory
//! literature). Standard errors come from the full [`PanelSeType`] menu;
//! Driscoll-Kraay is the natural default here because a common shock
//! induces cross-sectional dependence in the horizon-`h` errors by
//! construction.
//!
//! ## Cumulative multipliers (Ramey & Zubairy 2018)
//!
//! With [`PanelLpConfig::cumulative`] the regressand at horizon `h` is
//! the **cumulated outcome** `sum_{j=0..h} y_{i,t+j}`, estimated directly
//! per horizon. This is why the cumulative IRF is *not* produced by
//! cumulating the level point estimates: because the cumulated sum is
//! itself the regressand, `se[h]` is the correct standard error of the
//! cumulative response, whereas summing per-horizon point estimates would
//! require the full cross-horizon covariance of the level IRF (and
//! cumulating per-horizon standard errors is simply wrong — it ignores
//! the strong positive correlation of overlapping LP samples).
//!
//! ## Nickell bias — read this before trusting short-T dynamic panels
//!
//! The within transformation correlates the demeaned lagged outcome with
//! the demeaned error, so **fixed effects + lagged outcomes + short T
//! biases dynamic coefficients**: for an AR(1) panel the incidental-
//! parameter bias is approximately `-(1 + rho)/(T - 1)` (Nickell 1981) —
//! at `T = 20` and `rho = 0.5` that is a bias of about `-0.08`, and in
//! local projections the effect is horizon-amplified (roughly `O(h/T)`;
//! see Module 07 of the roadmap). It shrinks with the number of periods,
//! not the number of entities.
//!
//! Two half-panel corrections are offered. Both replace the point
//! estimates with the same affine combination of a full-sample fit and
//! two half-sample fits,
//!
//! ```text
//! theta_corrected = 2 theta_full - (theta_half1 + theta_half2) / 2
//! ```
//!
//! which removes the O(1/T) bias term because each half-panel estimate
//! carries roughly twice the bias of the full-panel one. They differ in
//! the half-sample bookkeeping and — decisively — in the standard error:
//!
//! * [`PanelLpConfig::jackknife`] — the half-panel jackknife of Dhaene &
//!   Jochmans (2015) as originally shipped: each half is re-estimated
//!   **strictly inside its own time window** (lags are re-burnt and leads
//!   truncated at the split, so no information crosses it), the two
//!   halves overlap by one period when `T` is odd (each of length
//!   `(T+1)/2`), and standard errors are kept from the full-sample fit —
//!   asymptotically justified because the jackknifed estimator has the
//!   same limiting variance as the uncorrected one (Dhaene & Jochmans
//!   2015, Theorem 3.1), but a finite-`T` approximation whose measured
//!   cost is documented on the panel model card.
//!
//! * [`PanelLpConfig::bias_correction`] = [`LpBiasCorrection::Spj`] — the
//!   split-panel jackknife for panel local projections of Mei, Sheng &
//!   Shi (2026, J. International Economics; arXiv 2302.13455), matching
//!   their reference implementation (the `pLP` R package,
//!   `github.com/zhentaoshi/panel-local-projection`). Leads and lags are
//!   computed on the **full panel** and only the per-horizon regression
//!   rows are split, at the floor of the median usable period (odd row
//!   counts give the extra row to the first half; halves never overlap),
//!   so no observations are burnt at the boundary. Standard errors are
//!   **recomputed for the corrected estimator**: residuals are evaluated
//!   at the SPJ coefficients and the sandwich meat uses the
//!   jackknife-adjusted scores `d_it = 2 x~_it - x~half_it` (the
//!   influence function of `2 theta_full - (theta_a + theta_b)/2`, since
//!   each half's `(X'X)` is asymptotically half the full one), with the
//!   full-sample bread. This is the SE the paper's implementation
//!   recommends; a bias correction with an unchanged plug-in SE costs
//!   coverage at short `T`.
//!
//!   Following `pLP`, the SPJ covariance conventions are Stata-flavoured
//!   and intentionally differ from the linearmodels conventions used by
//!   the uncorrected route: cluster-by-entity applies
//!   `(N/(N-1)) * ((n-1)/(n-k))` (absorbed effects not counted, group
//!   debias applied), and Driscoll-Kraay applies **no** small-sample
//!   factor. `pLP` hardcodes the Driscoll-Kraay lag truncation to
//!   `floor((T-h)^(1/4))`; here the user-supplied `bandwidth` is honoured
//!   (set it to that value to reproduce `pLP` exactly). No homoskedastic
//!   SPJ variance is provided in the reference implementation, so
//!   `bias_correction = Spj` with [`PanelSeType::NonRobust`] is refused
//!   rather than silently substituting a plug-in formula.
//!
//! The two corrections coincide **in the point estimates** when the two
//! half-samples are the same set of regression rows — at horizon 0 with
//! no lag controls on an even-`T` panel — and differ otherwise (the
//! window route burns `h` leads and `lag_max` lags around the split; the
//! MSS route does not). The SPJ standard errors differ by construction
//! even where the points coincide. Requesting both at once is refused as
//! ambiguous.
//!
//! // TODO(phase0): entity-varying shocks (an `N x T` impulse panel),
//! // user-supplied extra controls from `PanelData::regressor`, panel
//! // LP-IV, and the analytical (non-jackknife) Nickell corrections of
//! // Mei-Sheng-Shi.

use tsecon_hac::Kernel;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::data::PanelData;
use crate::error::PanelError;
use crate::fe::{fit_within, PanelSeType, WithinFit};

/// Nickell-bias correction applied to the per-horizon point estimates
/// (see the module docs for the two half-panel corrections, how they
/// relate, and their standard-error conventions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LpBiasCorrection {
    /// No correction: plain within (fixed-effects) estimates.
    #[default]
    None,
    /// Dhaene-Jochmans (2015) half-panel jackknife with windowed halves
    /// and full-sample plug-in standard errors — identical to setting
    /// [`PanelLpConfig::jackknife`], which remains as the original knob.
    DhaeneJochmans,
    /// Mei-Sheng-Shi (2026) split-panel jackknife for panel local
    /// projections: full-panel leads/lags with a median row split, and
    /// standard errors recomputed from the corrected residuals with
    /// jackknife-adjusted scores (the paper's reference implementation).
    Spj,
}

/// Configuration for [`panel_lp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelLpConfig {
    /// Maximum horizon `H`; responses are estimated for `h = 0..=H`.
    pub max_horizon: usize,
    /// Number of lagged-shock controls `shock_{t-1} .. shock_{t-Ls}`.
    pub shock_lags: usize,
    /// Number of lagged-outcome controls `y_{i,t-1} .. y_{i,t-Ly}`.
    /// Lagged outcomes soak up persistent noise but expose short-T
    /// panels to Nickell bias (see the module docs).
    pub outcome_lags: usize,
    /// Estimate the cumulated outcome `sum_{j<=h} y_{i,t+j}` per horizon
    /// (Ramey-Zubairy convention) instead of the level `y_{i,t+h}`.
    pub cumulative: bool,
    /// Apply the Dhaene-Jochmans (2015) half-panel jackknife bias
    /// correction to the point estimates (see the module docs). Kept for
    /// compatibility; equivalent to
    /// `bias_correction = LpBiasCorrection::DhaeneJochmans`. Combining
    /// `jackknife = true` with `bias_correction = Spj` is refused as
    /// ambiguous.
    pub jackknife: bool,
    /// Nickell-bias correction for the point estimates (and, for
    /// [`LpBiasCorrection::Spj`], the standard errors); see the module
    /// docs for how the two corrections relate.
    pub bias_correction: LpBiasCorrection,
    /// Covariance estimator for the per-horizon standard errors.
    pub cov: PanelSeType,
}

impl PanelLpConfig {
    /// A configuration with `n_lag_controls` lags of both the shock and
    /// the outcome as controls, level (non-cumulative) responses, and no
    /// jackknife. Adjust individual fields to taste.
    #[must_use]
    pub fn new(max_horizon: usize, n_lag_controls: usize, cov: PanelSeType) -> Self {
        Self {
            max_horizon,
            shock_lags: n_lag_controls,
            outcome_lags: n_lag_controls,
            cumulative: false,
            jackknife: false,
            bias_correction: LpBiasCorrection::None,
            cov,
        }
    }
}

/// Impulse responses from a panel local projection; produced by
/// [`panel_lp`]. All per-horizon vectors have `max_horizon + 1` entries,
/// indexed by horizon.
#[derive(Debug, Clone)]
pub struct PanelLpResult {
    /// Impulse response `beta_h` per horizon — the coefficient on
    /// `shock_t` (cumulated-outcome coefficient when `cumulative`;
    /// bias-corrected when a correction is requested).
    pub irf: Vec<f64>,
    /// Standard error of `irf[h]` under `se_type`. For
    /// [`LpBiasCorrection::None`] and
    /// [`LpBiasCorrection::DhaeneJochmans`] this is the full-sample
    /// plug-in SE (unchanged by the DJ jackknife; Dhaene & Jochmans 2015,
    /// Thm 3.1). For [`LpBiasCorrection::Spj`] it is the
    /// Mei-Sheng-Shi adjusted-score sandwich evaluated at the corrected
    /// coefficients (see the module docs for the exact conventions).
    pub se: Vec<f64>,
    /// Full per-horizon coefficient vectors, ordered
    /// `[shock_t, shock_{t-1..Ls}, y_{t-1..Ly}]` (bias-corrected when a
    /// correction is requested).
    pub params: Vec<Vec<f64>>,
    /// Stacked observations used at each horizon (the sample shrinks as
    /// `h` grows).
    pub nobs: Vec<usize>,
    /// Covariance estimator used for `se`.
    pub se_type: PanelSeType,
    /// Whether the regressand was the cumulated outcome.
    pub cumulative: bool,
    /// Whether the Dhaene-Jochmans half-panel jackknife was applied
    /// (kept for compatibility; equals
    /// `bias_correction == LpBiasCorrection::DhaeneJochmans`).
    pub jackknife: bool,
    /// Which Nickell-bias correction produced `irf`/`params`/`se`.
    pub bias_correction: LpBiasCorrection,
}

/// Estimates a panel local projection of the outcome in `data` on the
/// common shock series, horizon by horizon (see the module docs for the
/// regression, conventions, and the Nickell-bias warning).
///
/// `shock` must have `data.n_periods()` observations, aligned with the
/// outcome's periods. Regressors stored in `data` are currently ignored
/// (controls are generated internally from shock and outcome lags;
/// user-supplied controls are `// TODO(phase0)`).
///
/// # Errors
///
/// * [`PanelError::Dimension`] if `shock` is not `n_periods` long;
/// * [`PanelError::NonFinite`] if `shock` contains NaN/infinity;
/// * [`PanelError::InvalidArgument`] if `jackknife = true` is combined
///   with `bias_correction = Spj` (two different corrections — set
///   exactly one), or if `Spj` is combined with
///   [`PanelSeType::NonRobust`] (the reference implementation provides
///   no homoskedastic SPJ variance) or with fewer than two entities;
/// * [`PanelError::InsufficientObservations`] /
///   [`PanelError::DegreesOfFreedom`] when a horizon (or a jackknife /
///   split-panel half) leaves too small a sample;
/// * [`PanelError::SingularDesign`] for collinear controls.
pub fn panel_lp(
    data: &PanelData,
    shock: &[f64],
    config: &PanelLpConfig,
) -> Result<PanelLpResult, PanelError> {
    let t_len = data.n_periods();
    if shock.len() != t_len {
        return Err(PanelError::Dimension {
            what: "shock series must be aligned with the panel's periods",
            expected: t_len,
            got: shock.len(),
        });
    }
    if shock.iter().any(|v| !v.is_finite()) {
        return Err(PanelError::NonFinite { what: "shock" });
    }

    let hmax = config.max_horizon;
    // Reject an infeasible request up front: the last horizon's window is
    // `[lag_max, T - hmax)`, and if that is already empty the failure is a
    // sample-size problem — reported as such here, rather than as whatever
    // symptom (a rank-deficient shrunken design, an empty window) the
    // per-horizon loop would trip over first.
    let lag_max = config.shock_lags.max(config.outcome_lags);
    if t_len.saturating_sub(hmax) <= lag_max {
        return Err(PanelError::InsufficientObservations {
            what: "panel local projection: the largest horizon plus the lag order \
                   leaves no regression window inside the panel's periods",
            needed: hmax + lag_max + 1,
            got: t_len,
        });
    }
    // Resolve the two correction knobs into one effective choice; the
    // boolean flag is kept as sugar for the Dhaene-Jochmans route.
    let bc = match (config.jackknife, config.bias_correction) {
        (false, bc) => bc,
        (true, LpBiasCorrection::None | LpBiasCorrection::DhaeneJochmans) => {
            LpBiasCorrection::DhaeneJochmans
        }
        (true, LpBiasCorrection::Spj) => {
            return Err(PanelError::InvalidArgument {
                what: "jackknife = true requests the Dhaene-Jochmans half-panel \
                       jackknife while bias_correction = \"spj\" requests the \
                       Mei-Sheng-Shi split-panel jackknife; they split the sample \
                       and compute standard errors differently — set exactly one",
            });
        }
    };
    if bc == LpBiasCorrection::Spj {
        if matches!(config.cov, PanelSeType::NonRobust) {
            return Err(PanelError::InvalidArgument {
                what: "bias_correction = \"spj\" has no homoskedastic variance: the \
                       Mei-Sheng-Shi reference implementation provides only \
                       cluster-by-entity and Driscoll-Kraay adjusted-score \
                       sandwiches, and a nonrobust plug-in SE would ignore the \
                       correction — use se_type = \"cluster\" or \"driscoll_kraay\"",
            });
        }
        if data.n_entities() < 2 {
            return Err(PanelError::InvalidArgument {
                what: "bias_correction = \"spj\" clusters by entity in its \
                       covariance (the N/(N-1) group debias of the reference \
                       implementation) and needs at least two entities",
            });
        }
    }

    let mut irf = Vec::with_capacity(hmax + 1);
    let mut se = Vec::with_capacity(hmax + 1);
    let mut params = Vec::with_capacity(hmax + 1);
    let mut nobs = Vec::with_capacity(hmax + 1);

    // Half-panel windows for the Dhaene-Jochmans jackknife: overlapping
    // halves of length ceil(T/2) when T is odd (Dhaene-Jochmans 2015).
    let half = t_len.div_ceil(2);

    for h in 0..=hmax {
        let full = lp_fit_window(data, shock, config, h, 0, t_len)?;
        match bc {
            LpBiasCorrection::None => {
                let inference = full.inference(config.cov)?;
                irf.push(full.params[0]);
                se.push(inference.bse[0]);
                nobs.push(full.nobs);
                params.push(full.params.clone());
            }
            LpBiasCorrection::DhaeneJochmans => {
                let inference = full.inference(config.cov)?;
                let first = lp_fit_window(data, shock, config, h, 0, half)?;
                let second = lp_fit_window(data, shock, config, h, t_len - half, t_len)?;
                let coefs = half_panel_combine(&full.params, &first.params, &second.params);
                irf.push(coefs[0]);
                se.push(inference.bse[0]);
                nobs.push(full.nobs);
                params.push(coefs);
            }
            LpBiasCorrection::Spj => {
                let (coefs, se_h) = spj_horizon(data, shock, config, h, &full)?;
                irf.push(coefs[0]);
                se.push(se_h);
                nobs.push(full.nobs);
                params.push(coefs);
            }
        }
    }

    Ok(PanelLpResult {
        irf,
        se,
        params,
        nobs,
        se_type: config.cov,
        cumulative: config.cumulative,
        jackknife: bc == LpBiasCorrection::DhaeneJochmans,
        bias_correction: bc,
    })
}

/// The half-panel bias-corrected combination shared by both jackknife
/// routes: `2 theta_full - (theta_a + theta_b) / 2` (Dhaene & Jochmans
/// 2015; Mei, Sheng & Shi 2026).
fn half_panel_combine(full: &[f64], a: &[f64], b: &[f64]) -> Vec<f64> {
    full.iter()
        .zip(a.iter().zip(b.iter()))
        .map(|(&f, (&x, &y))| 2.0 * f - 0.5 * (x + y))
        .collect()
}

/// One horizon of the Mei-Sheng-Shi split-panel jackknife: the median
/// row split, the two half fits (full-panel leads/lags), the corrected
/// coefficients, and the adjusted-score sandwich SE of the impulse
/// response. `full` is the full-sample within fit for this horizon.
fn spj_horizon(
    data: &PanelData,
    shock: &[f64],
    config: &PanelLpConfig,
    h: usize,
    full: &WithinFit,
) -> Result<(Vec<f64>, f64), PanelError> {
    let t_len = data.n_periods();
    let n_ent = data.n_entities();
    let lag_max = config.shock_lags.max(config.outcome_lags);
    let k = 1 + config.shock_lags + config.outcome_lags;

    // Usable regression rows are t in [a0, b0] (0-based): every lag and
    // the horizon-h lead stay inside the panel. `pLP` splits them at the
    // floor of the median usable period: rows [a0, c0] form the first
    // half and rows [c0 + 1, b0] the second, so an odd row count gives
    // the extra row to the FIRST half and the halves never overlap.
    let a0 = lag_max;
    let b0 = t_len - h - 1; // t_len - h > lag_max was checked up front.
    let c0 = (a0 + b0) / 2;
    let s_a = c0 - a0 + 1;
    let s_b = b0 - c0;
    // Each half must leave residual degrees of freedom after its own
    // demeaning (fit_within requires n > k + N), and the COMMON shock
    // block needs rank: within-demeaned common columns live in an
    // (s_half - 1)-dimensional space, so `1 + shock_lags` of them need
    // `s_half >= shock_lags + 2` periods. Surface the failure as the
    // sample-size problem it is, with the smaller half's period count,
    // rather than as a downstream singular design.
    let min_periods = 2usize
        .max((k + n_ent) / n_ent + 1)
        .max(config.shock_lags + 2);
    if s_a.min(s_b) < min_periods {
        return Err(PanelError::InsufficientObservations {
            what: "split-panel jackknife half-panel: after the horizon's lead and \
                   the lag controls, the median split leaves too few periods in a \
                   half — reduce max_horizon or the lag order, or supply more \
                   periods (counts are periods per half)",
            needed: min_periods,
            got: s_a.min(s_b),
        });
    }

    let fit_a = lp_fit_rows(data, shock, config, h, a0, c0 + 1)?;
    let fit_b = lp_fit_rows(data, shock, config, h, c0 + 1, b0 + 1)?;
    let coefs = half_panel_combine(&full.params, &fit_a.params, &fit_b.params);

    // Residuals at the CORRECTED coefficients on the full demeaned data:
    // e_r = resid_full_r - x~_r' (theta_spj - theta_full).
    let s_full = b0 - a0 + 1;
    let n = n_ent * s_full;
    debug_assert_eq!(full.nobs, n);
    let mut resid = vec![0.0_f64; n];
    for (r, res) in resid.iter_mut().enumerate() {
        let mut v = full.resid[r];
        for (j, (&c, &p)) in coefs.iter().zip(full.params.iter()).enumerate() {
            v -= full.xd[(r, j)] * (c - p);
        }
        *res = v;
    }

    // Jackknife-adjusted scores d_it = 2 x~_it - x~half_it, where
    // x~half_it is the regressor demeaned within the half containing t.
    // Row layouts: full fit stacks (entity i, offset s = t - a0); half a
    // stacks offsets t - a0 over s_a periods; half b offsets t - c0 - 1
    // over s_b periods.
    let mut dd = Mat::<f64>::zeros(n, k);
    for i in 0..n_ent {
        for s in 0..s_full {
            let r = i * s_full + s;
            let (half_xd, half_r) = if s < s_a {
                (&fit_a.xd, i * s_a + s)
            } else {
                (&fit_b.xd, i * s_b + (s - s_a))
            };
            for j in 0..k {
                dd[(r, j)] = 2.0 * full.xd[(r, j)] - half_xd[(half_r, j)];
            }
        }
    }

    // Sandwich meat per the reference implementation (see module docs).
    let meat = match config.cov {
        PanelSeType::NonRobust => unreachable!("refused before the horizon loop"),
        PanelSeType::ClusterEntity => {
            let mut w = Mat::<f64>::zeros(k, k);
            let mut g = vec![0.0_f64; k];
            for i in 0..n_ent {
                g.iter_mut().for_each(|v| *v = 0.0);
                for s in 0..s_full {
                    let r = i * s_full + s;
                    let u = resid[r];
                    for (j, gj) in g.iter_mut().enumerate() {
                        *gj += dd[(r, j)] * u;
                    }
                }
                for a in 0..k {
                    for b in 0..k {
                        w[(a, b)] += g[a] * g[b];
                    }
                }
            }
            // pLP small-sample factor: (N/(N-1)) * ((n-1)/(n-k)) —
            // Stata-style, absorbed effects not counted, group debias on.
            let nf = n as f64;
            let nef = n_ent as f64;
            let scale = nef / (nef - 1.0) * (nf - 1.0) / (nf - k as f64);
            Mat::from_fn(k, k, |a, b| scale * w[(a, b)])
        }
        PanelSeType::DriscollKraay { bandwidth } => {
            if !bandwidth.is_finite() || bandwidth < 0.0 {
                return Err(PanelError::InvalidBandwidth { value: bandwidth });
            }
            // Per-period cross-sectional sums of the adjusted scores.
            let mut agg = Mat::<f64>::zeros(s_full, k);
            for i in 0..n_ent {
                for s in 0..s_full {
                    let r = i * s_full + s;
                    let u = resid[r];
                    for j in 0..k {
                        agg[(s, j)] += dd[(r, j)] * u;
                    }
                }
            }
            // Bartlett HAC on the aggregated scores, with NO
            // small-sample factor (pLP's dk_var applies none).
            let kernel = Kernel::Bartlett;
            let mut w = Mat::<f64>::zeros(k, k);
            for lag in 0..s_full {
                let wt = kernel.weight(lag, bandwidth);
                if lag > 0 && wt == 0.0 {
                    break; // Bartlett truncates.
                }
                for s in lag..s_full {
                    for a in 0..k {
                        for b in 0..k {
                            let gab = agg[(s, a)] * agg[(s - lag, b)];
                            if lag == 0 {
                                w[(a, b)] += gab;
                            } else {
                                w[(a, b)] += wt * gab;
                                w[(b, a)] += wt * gab;
                            }
                        }
                    }
                }
            }
            w
        }
    };
    let cov = &full.xtx_inv * &meat * &full.xtx_inv;
    Ok((coefs, cov[(0, 0)].max(0.0).sqrt()))
}

/// One per-horizon within regression restricted to the period window
/// `[w0, w1)`: regression index `t` runs over
/// `[w0 + max(Ls, Ly), w1 - h)` so every lag and lead stays inside the
/// window (the Dhaene-Jochmans jackknife half-panels therefore never
/// leak information across the split).
fn lp_fit_window(
    data: &PanelData,
    shock: &[f64],
    config: &PanelLpConfig,
    h: usize,
    w0: usize,
    w1: usize,
) -> Result<WithinFit, PanelError> {
    let lag_max = config.shock_lags.max(config.outcome_lags);
    let t_start = w0 + lag_max;
    let t_end = w1.saturating_sub(h);
    if t_end <= t_start {
        return Err(PanelError::InsufficientObservations {
            what: "panel local projection horizon window",
            needed: t_start + 1,
            got: t_end,
        });
    }
    lp_fit_rows(data, shock, config, h, t_start, t_end)
}

/// One per-horizon within regression on the exact regression rows
/// `t in [t_start, t_end)`, with lags and the horizon-`h` lead indexing
/// the **full panel** (the Mei-Sheng-Shi split-panel bookkeeping: a half
/// keeps the leads/lags that cross the split). The caller must guarantee
/// `t_start >= max(Ls, Ly)` and `t_end + h <= n_periods`.
fn lp_fit_rows(
    data: &PanelData,
    shock: &[f64],
    config: &PanelLpConfig,
    h: usize,
    t_start: usize,
    t_end: usize,
) -> Result<WithinFit, PanelError> {
    let n_ent = data.n_entities();
    let k = 1 + config.shock_lags + config.outcome_lags;
    debug_assert!(t_start >= config.shock_lags.max(config.outcome_lags));
    debug_assert!(t_end + h <= data.n_periods());
    let n_per = t_end - t_start;
    let n = n_ent * n_per;
    if n <= k + n_ent {
        return Err(PanelError::DegreesOfFreedom {
            n,
            k,
            n_entities: n_ent,
        });
    }

    let outcome: MatRef<'_, f64> = data.outcome();
    let mut y = vec![0.0_f64; n];
    let mut x_cols = vec![vec![0.0_f64; n]; k];
    for i in 0..n_ent {
        for (s, t) in (t_start..t_end).enumerate() {
            let r = i * n_per + s;
            y[r] = if config.cumulative {
                (0..=h).map(|j| outcome[(i, t + j)]).sum()
            } else {
                outcome[(i, t + h)]
            };
            x_cols[0][r] = shock[t];
            for l in 1..=config.shock_lags {
                x_cols[l][r] = shock[t - l];
            }
            for l in 1..=config.outcome_lags {
                x_cols[config.shock_lags + l][r] = outcome[(i, t - l)];
            }
        }
    }
    fit_within(&y, &x_cols, n_ent, n_per)
}
