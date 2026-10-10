//! Iterated point forecasts and asymptotic forecast-error intervals.
//!
//! Two kinds of interval live here. [`VarResults::forecast_interval`] is the
//! long-standing **marginal** one: for each horizon and each series separately,
//! `y_{T+h|T} ± z sqrt(MSE(h)_jj)`. [`VarResults::forecast_interval_simultaneous`]
//! adds a **joint** one over a declared family of `(horizon, series)` cells.
//!
//! The gap between the two is the largest measured in tsecon's own
//! interval-coverage audit: nominal 95% marginal `var_forecast` bands contained
//! every horizon *and* every series at once in 40.9% of samples at `T = 100`,
//! and still only 48.1% at `T = 800`. Joint coverage does not converge to the
//! marginal level as the sample grows — it is a different quantity.
//!
//! Separately from multiplicity, every interval here treats the estimated
//! coefficients as known by default. [`ForecastOptions::with_param_uncertainty`]
//! adds Lütkepohl's (2005, section 3.5) first-order parameter-estimation term
//! `Omega(h) / T` to the marginal covariance — see
//! [`VarResults::forecast_cov_with`].

use tsecon_linalg::faer::linalg::solvers::Solve;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::simultaneous;
use tsecon_stats::special::inv_norm_cdf;

use crate::error::VarError;
use crate::results::VarResults;

pub use crate::irf_asymptotic::{BandMethod, DEFAULT_N_SIM};

/// Point forecasts with symmetric asymptotic intervals, produced by
/// [`VarResults::forecast_interval`]. All matrices are `steps x k`,
/// row `h` being the `(h + 1)`-step-ahead values.
#[derive(Debug, Clone)]
pub struct ForecastInterval {
    /// Iterated point forecasts.
    pub point: Mat<f64>,
    /// Lower interval bounds, `point - z_{1 - alpha/2} * se`.
    pub lower: Mat<f64>,
    /// Upper interval bounds, `point + z_{1 - alpha/2} * se`.
    pub upper: Mat<f64>,
    /// Per-cell standard errors `sqrt(diag MSE(h))`, `steps x k` — the
    /// numbers `lower`/`upper` are built from. Reported so a caller can see
    /// that switching to a simultaneous band changes only the multiplier.
    pub se: Mat<f64>,
    /// The simultaneous band, when
    /// [`VarResults::forecast_interval_simultaneous`] produced this object;
    /// `None` from the plain [`VarResults::forecast_interval`].
    pub simultaneous: Option<ForecastSimultaneous>,
    /// The per-horizon forecast-error covariance matrices `se` is the root of
    /// the diagonal of (`steps` entries, each `k x k`, entry `h` for horizon
    /// `h + 1`): `MSE(h)` by default, `MSE(h) + Omega(h)/T` under
    /// [`ForecastOptions::with_param_uncertainty`].
    pub cov: Vec<Mat<f64>>,
}

/// Which uncertainty sources enter the forecast-error covariance of
/// [`VarResults::forecast_cov_with`] and [`VarResults::forecast_interval_with`].
///
/// The default is the statsmodels `forecast_interval` / `VARProcess.mse`
/// convention — innovation uncertainty only, coefficients treated as known —
/// so `ForecastOptions::default()` reproduces [`VarResults::forecast_cov`] and
/// [`VarResults::forecast_interval`] exactly.
///
/// Marked `#[non_exhaustive]`: build it with [`ForecastOptions::new`] and the
/// `with_*` setters so future uncertainty sources can be added without
/// breaking callers (the `tsecon-arima` pattern).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ForecastOptions {
    /// Add Lütkepohl's first-order parameter-estimation term `Omega(h)/T`
    /// to every horizon's covariance (default `false`).
    pub param_uncertainty: bool,
}

impl ForecastOptions {
    /// The statsmodels-parity defaults: no parameter uncertainty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggles Lütkepohl's parameter-uncertainty term `Omega(h)/T`.
    #[must_use]
    pub fn with_param_uncertainty(mut self, param_uncertainty: bool) -> Self {
        self.param_uncertainty = param_uncertainty;
        self
    }
}

/// Work budget of [`VarResults::forecast_param_uncertainty`], in units of
/// one multiply-add or one memory read of the trace table.
///
/// The term's dominant cost is the double sum over horizons: for every
/// horizon `h` an `h x h` block of the `steps x steps` trace table times an
/// `h x k^2` block of MA coefficients — `steps^3 k^2 / 3` multiply-adds over
/// `steps^3 / 3` reads of a table that outgrows the cache, so for small `k`
/// the reads, not the arithmetic, set the pace (measured: `k = 1` and `k = 3`
/// cost about the same per horizon cubed). The count is
/// `steps^3 (k^2 + 8) / 3` for it, plus `steps^2 k^3 / 2`, `2 steps m^3` (the
/// powers of `B`) and `steps^2 m^2` (the trace table), with `m = n_trend + k
/// lags` regressors per equation. On a small VAR the first term dominates and
/// the cost is cubic in `steps`: `2^34` is one to two seconds of one core
/// (measured 0.9-2.0 s at the cap for `k = 1, 2, 3`). On a wide one the
/// `2 steps m^3` term dominates — linear in `steps` but cubic in `k lags` —
/// and even a short horizon can exceed the budget. A request past it is
/// refused rather than left to run for minutes: naming `steps`, with the
/// remedy for whichever terms dominate, or naming `lags` when not even
/// `steps = 1` fits.
pub const PARAM_UNCERTAINTY_WORK_BUDGET: u128 = 1 << 34;

/// The work count the budget is checked against (see
/// [`PARAM_UNCERTAINTY_WORK_BUDGET`]), split into the terms driven by
/// `steps` and `k` (`steps^3 (k^2 + 8) / 3 + steps^2 k^3 / 2`) and the
/// terms driven by `m` (`2 steps m^3 + steps^2 m^2`). Saturating, so an
/// absurd `steps` (up to `usize::MAX`) cannot overflow the check that is
/// meant to refuse it.
fn param_uncertainty_terms(steps: usize, k: usize, m: usize) -> (u128, u128) {
    let (s, k, m) = (steps as u128, k as u128, m as u128);
    let prod = |xs: &[u128]| xs.iter().fold(1_u128, |acc, &x| acc.saturating_mul(x));
    let horizon = (prod(&[s, s, s, k.saturating_mul(k).saturating_add(8)]) / 3)
        .saturating_add(prod(&[s, s, k, k, k]) / 2);
    let regressors = prod(&[2, s, m, m, m]).saturating_add(prod(&[s, s, m, m]));
    (horizon, regressors)
}

fn param_uncertainty_work(steps: usize, k: usize, m: usize) -> u128 {
    let (horizon, regressors) = param_uncertainty_terms(steps, k, m);
    horizon.saturating_add(regressors)
}

/// The refusal when the parameter-uncertainty term for `steps` on a VAR with
/// `k` series, `lags` lags and `m` regressors per equation would exceed
/// [`PARAM_UNCERTAINTY_WORK_BUDGET`], or `None` when it fits. The message
/// quotes the whole count and names the driver: `lags` when not even one
/// step fits, otherwise `steps` with the remedy for whichever terms
/// dominate (a shorter horizon when the cubic-in-`steps` terms do, fewer
/// lags or series when the `m` terms do).
fn param_uncertainty_refusal(steps: usize, k: usize, lags: usize, m: usize) -> Option<VarError> {
    if param_uncertainty_work(steps, k, m) <= PARAM_UNCERTAINTY_WORK_BUDGET {
        return None;
    }
    if param_uncertainty_work(1, k, m) > PARAM_UNCERTAINTY_WORK_BUDGET {
        return Some(VarError::InvalidParameter {
            name: "lags",
            value: lags as f64,
            requirement: "a lag order whose parameter-uncertainty term fits the work budget \
                          of 2^34 operations at some horizon. The count is steps^3 (k^2 + 8) \
                          / 3 + steps^2 k^3 / 2 + 2 steps m^3 + steps^2 m^2, with k series and \
                          m = n_trend + k*lags regressors per equation, and on this VAR the \
                          2 steps m^3 term (CUBIC in k*lags) exceeds the budget even at \
                          steps = 1 — reduce lags or the number of series, or drop \
                          param_uncertainty",
        });
    }
    let (horizon, regressors) = param_uncertainty_terms(steps, k, m);
    let requirement = if regressors > horizon {
        "a horizon whose parameter-uncertainty term fits the work budget of 2^34 \
         operations. The count is steps^3 (k^2 + 8) / 3 + steps^2 k^3 / 2 + 2 steps m^3 + \
         steps^2 m^2, with k series and m = n_trend + k*lags regressors per equation, and \
         on this VAR the m terms dominate: 2 steps m^3 is linear in steps but CUBIC in \
         k*lags, so reduce lags or the number of series (a shorter horizon helps only in \
         proportion), or drop param_uncertainty"
    } else {
        "a horizon whose parameter-uncertainty term fits the work budget of 2^34 \
         operations. The count is steps^3 (k^2 + 8) / 3 + steps^2 k^3 / 2 + 2 steps m^3 + \
         steps^2 m^2, with k series and m = n_trend + k*lags regressors per equation, and \
         here the first term dominates: CUBIC in steps (measured 0.9-2.0 s at the budget \
         for k = 1, 2, 3). Forecast horizons are tens of periods; drop param_uncertainty \
         for a long innovation-only path"
    };
    Some(VarError::InvalidParameter {
        name: "steps",
        value: steps as f64,
        requirement,
    })
}

/// Which `(horizon, series)` cells a simultaneous forecast band covers jointly.
///
/// As with impulse responses, this is a real choice and not a detail: every
/// cell added to the family widens the band for every other cell in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastBandScope {
    /// One family per series: that series' whole forecast path.
    /// `K = steps`. The right scope for a single-series fan chart.
    Horizon,
    /// A single family: every horizon of every series at once.
    /// `K = steps * k`.
    ///
    /// **The default**, and the object the audit measured (12 horizons x 2
    /// series read jointly, 40.9% at `T = 100` against a nominal 95%). A
    /// multi-panel forecast figure is normally read as one statement.
    All,
}

impl ForecastBandScope {
    /// Parse the Python-facing spelling: `"horizon"` or `"all"`.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] naming the two accepted values.
    pub fn parse(s: &str) -> Result<Self, VarError> {
        match s {
            "horizon" => Ok(ForecastBandScope::Horizon),
            "all" => Ok(ForecastBandScope::All),
            _ => Err(VarError::InvalidArgument {
                what: "unknown band_scope; expected \"all\" (the default: joint over \
                       every horizon and every series) or \"horizon\" (joint over \
                       horizons, separately for each series)",
            }),
        }
    }

    /// The canonical Python-facing spelling, for echoing back.
    pub fn label(self) -> &'static str {
        match self {
            ForecastBandScope::Horizon => "horizon",
            ForecastBandScope::All => "all",
        }
    }
}

/// A simultaneous forecast band: the same point forecasts and the same
/// standard errors as the marginal interval, with a larger multiplier.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastSimultaneous {
    /// `critical_value[j]` is the multiplier applied to every cell of series
    /// `j`'s family. Under [`ForecastBandScope::All`] all entries are equal.
    pub critical_value: Vec<f64>,
    /// `point - c * se`, `steps x k`.
    pub lower: Mat<f64>,
    /// `point + c * se`, `steps x k`.
    pub upper: Mat<f64>,
    /// Number of cells in each family — the `K` the multiplier answers for.
    pub n_cells: usize,
    /// Cells of each family with a strictly positive standard error. Equal to
    /// [`Self::n_cells`] for any non-degenerate VAR (a forecast-error variance
    /// is zero only if the corresponding innovation variance is).
    pub n_cells_used: Vec<usize>,
    /// The marginal multiplier `z_{1 - alpha/2}`, for reference. Every entry of
    /// [`Self::critical_value`] is `>=` this.
    pub pointwise: f64,
    /// The method that produced [`Self::critical_value`].
    pub method: BandMethod,
    /// The cell family the band is simultaneous over.
    pub scope: ForecastBandScope,
}

impl VarResults {
    /// Iterated point forecasts `steps` periods past the estimation
    /// sample (Lütkepohl 2005, eq. 3.5.5):
    ///
    /// ```text
    /// y_{T+h|T} = c + sum_{i=1}^{p} A_i y_{T+h-i|T}
    /// ```
    ///
    /// with `y_{s|T} = y_s` for `s <= T`. The last `p` rows of the
    /// stored estimation sample seed the recursion, matching
    /// statsmodels `forecast(y[-p:], steps)`. Returns a `steps x k`
    /// matrix.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] if `steps == 0`.
    pub fn forecast(&self, steps: usize) -> Result<Mat<f64>, VarError> {
        if steps == 0 {
            return Err(VarError::InvalidArgument {
                what: "steps = 0: a forecast needs at least one step ahead; pass steps >= 1",
            });
        }
        let k = self.neqs;
        let n = self.endog.nrows();
        let mut out = Mat::<f64>::zeros(steps, k);
        for h in 0..steps {
            for r in 0..k {
                let mut v = self.intercept[r];
                for (i, a) in self.coefs.iter().enumerate() {
                    // y_{T + h - i} (1-based lag i = index i + 1).
                    for c in 0..k {
                        let lagged = if h > i {
                            out[(h - i - 1, c)]
                        } else {
                            self.endog[(n - (i + 1 - h), c)]
                        };
                        v += a[(r, c)] * lagged;
                    }
                }
                out[(h, r)] = v;
            }
        }
        Ok(out)
    }

    /// Asymptotic forecast-error covariance matrices for horizons
    /// `1..=steps` (Lütkepohl 2005, eq. 2.2.11):
    ///
    /// ```text
    /// MSE(h) = sum_{i=0}^{h-1} Psi_i sigma_u Psi_i'
    /// ```
    ///
    /// using the df-adjusted `sigma_u` and treating the coefficients as
    /// known — innovation uncertainty only, no parameter-estimation
    /// term, exactly like statsmodels `forecast_interval` /
    /// `VARProcess.mse`.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] if `steps == 0`;
    /// [`VarError::SizeBudget`] naming `steps` beyond the shared size budget
    /// ([`crate::budget::check_steps`]), checked before anything is
    /// allocated.
    pub fn forecast_cov(&self, steps: usize) -> Result<Vec<Mat<f64>>, VarError> {
        if steps == 0 {
            return Err(VarError::InvalidArgument {
                what: "steps = 0: a forecast covariance needs at least one step ahead; \
                       pass steps >= 1",
            });
        }
        crate::budget::check_steps(self.neqs, steps)?;
        let psi = self.ma_rep(steps - 1)?;
        let k = self.neqs;
        let mut covs = Vec::with_capacity(steps);
        let mut acc = Mat::<f64>::zeros(k, k);
        for phi in &psi {
            acc += phi * &self.sigma_u * phi.transpose();
            covs.push(acc.clone());
        }
        Ok(covs)
    }

    /// Point forecasts with symmetric `1 - alpha` asymptotic intervals
    /// (statsmodels `forecast_interval`):
    ///
    /// ```text
    /// y_{T+h|T} +/- z_{1 - alpha/2} sqrt(diag MSE(h))
    /// ```
    ///
    /// The intervals reflect innovation uncertainty only (see
    /// [`VarResults::forecast_cov`]); [`VarResults::forecast_interval_with`]
    /// adds the parameter-estimation term on request, and bootstrap
    /// intervals are `// TODO(phase0)` alongside the tsecon-bootstrap IRF
    /// bands.
    ///
    /// # Errors
    ///
    /// * [`VarError::InvalidArgument`] if `steps == 0` or `alpha` is
    ///   not strictly inside `(0, 1)`;
    /// * [`VarError::Stats`] if the normal quantile fails (impossible
    ///   for valid `alpha`).
    pub fn forecast_interval(
        &self,
        steps: usize,
        alpha: f64,
    ) -> Result<ForecastInterval, VarError> {
        self.forecast_interval_with(steps, alpha, ForecastOptions::default())
    }

    /// Point forecasts with symmetric `1 - alpha` asymptotic intervals built
    /// from [`VarResults::forecast_cov_with`]:
    ///
    /// ```text
    /// y_{T+h|T} +/- z_{1 - alpha/2} sqrt(diag cov(h))
    /// ```
    ///
    /// With the default options this *is* [`VarResults::forecast_interval`]
    /// (the same arithmetic in the same order — the default bands are
    /// bit-identical). With
    /// [`ForecastOptions::with_param_uncertainty(true)`](ForecastOptions::with_param_uncertainty)
    /// `cov(h) = MSE(h) + Omega(h)/T` (statsmodels
    /// `forecast_cov(method="auto")`), so the bands widen by Lütkepohl's
    /// first-order estimation term; the point forecasts are unchanged. The
    /// returned [`ForecastInterval::cov`] holds the matrices used.
    ///
    /// # Errors
    ///
    /// Everything [`VarResults::forecast_interval`] returns, plus the
    /// [`VarResults::forecast_param_uncertainty`] errors when the term is
    /// requested.
    pub fn forecast_interval_with(
        &self,
        steps: usize,
        alpha: f64,
        options: ForecastOptions,
    ) -> Result<ForecastInterval, VarError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(VarError::InvalidParameter {
                name: "alpha",
                value: alpha,
                requirement: "a value strictly inside (0, 1) — alpha = 0.05 gives a \
                              95% forecast interval",
            });
        }
        if options.param_uncertainty {
            // Before the point path allocates anything.
            self.check_param_uncertainty_budget(steps)?;
        }
        crate::budget::check_steps(self.neqs, steps)?;
        let point = self.forecast(steps)?;
        let covs = self.forecast_cov_with(steps, options)?;
        let z = inv_norm_cdf(1.0 - alpha / 2.0)?;
        let k = self.neqs;
        let mut lower = Mat::<f64>::zeros(steps, k);
        let mut upper = Mat::<f64>::zeros(steps, k);
        let mut ses = Mat::<f64>::zeros(steps, k);
        for h in 0..steps {
            for j in 0..k {
                let se = covs[h][(j, j)].max(0.0).sqrt();
                lower[(h, j)] = point[(h, j)] - z * se;
                upper[(h, j)] = point[(h, j)] + z * se;
                ses[(h, j)] = se;
            }
        }
        Ok(ForecastInterval {
            point,
            lower,
            upper,
            se: ses,
            simultaneous: None,
            cov: covs,
        })
    }

    /// Forecast-error covariance matrices for horizons `1..=steps` with a
    /// choice of uncertainty sources — see [`ForecastOptions`].
    ///
    /// Default options: exactly [`VarResults::forecast_cov`], `MSE(h)`.
    /// With [`ForecastOptions::with_param_uncertainty(true)`](ForecastOptions::with_param_uncertainty)
    /// (Lütkepohl 2005, section 3.5, eq. 3.5.13):
    ///
    /// ```text
    /// cov(h) = MSE(h) + Omega(h) / T
    /// ```
    ///
    /// with `T` = [`VarResults::nobs`] (not `T - m`) and `Omega(h)` from
    /// [`VarResults::forecast_param_uncertainty`] — statsmodels
    /// `VARResults.forecast_cov(steps, method="auto")` for a constant-only
    /// VAR. At `h = 1` the term is `m Sigma_u` (to rounding — the trace
    /// factor is the integer `m`), so `cov(1) = (1 + m/T) Sigma_u`.
    ///
    /// **A partial repair, by construction.** `Omega(h)/T` is the first-order
    /// `O(1/T)` effect of estimating the coefficients, derived for a
    /// *stationary* VAR. It moves measured coverage toward nominal but not to
    /// it: the remaining shortfall at `T = 100` is the OLS small-sample bias
    /// of the coefficients, which no variance term repairs. On a
    /// near-unit-root levels VAR the term works through an ill-conditioned
    /// `Z'Z` and is large; it is not a unit-root correction.
    ///
    /// Divergence from statsmodels on `trend = "n"`: its public
    /// `method="auto"` silently returns `MSE(h)` without a constant (a
    /// `k_exog == 1` gate). This crate applies the same formula to both
    /// trends; the golden for `"n"` is statsmodels'
    /// `mse + _omega_forc_cov / nobs`.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] if `steps == 0`; the
    /// [`VarResults::forecast_param_uncertainty`] errors under the option.
    pub fn forecast_cov_with(
        &self,
        steps: usize,
        options: ForecastOptions,
    ) -> Result<Vec<Mat<f64>>, VarError> {
        if !options.param_uncertainty {
            return self.forecast_cov(steps);
        }
        // Budget first, so an absurd `steps` is refused before anything
        // (the MSE recursion included) is allocated.
        self.check_param_uncertainty_budget(steps)?;
        let mut covs = self.forecast_cov(steps)?;
        let omega = self.forecast_param_uncertainty(steps)?;
        let tf = self.nobs as f64;
        let k = self.neqs;
        for (c, om) in covs.iter_mut().zip(&omega) {
            for j in 0..k {
                for i in 0..k {
                    c[(i, j)] += om[(i, j)] / tf;
                }
            }
        }
        Ok(covs)
    }

    /// Refuse, naming `steps` (or `lags`, when no horizon fits), a
    /// parameter-uncertainty request whose work would exceed
    /// [`PARAM_UNCERTAINTY_WORK_BUDGET`].
    fn check_param_uncertainty_budget(&self, steps: usize) -> Result<(), VarError> {
        match param_uncertainty_refusal(steps, self.neqs, self.spec.lags, self.df_model) {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Lütkepohl's (2005, pp. 96-97) parameter-uncertainty matrices
    /// `Omega(h)`, `h = 1..=steps` — the `T`-scaled first-order contribution of
    /// coefficient estimation error to the `h`-step forecast-error covariance:
    ///
    /// ```text
    /// Omega(h) = sum_{i,j<h} tr[(B')^{h-1-i} G^{-1} B^{h-1-j} G] Psi_i Sigma_u Psi_j'
    /// ```
    ///
    /// * `G = Z'Z` with `Z` the `T x m` regressor matrix rebuilt from the
    ///   estimation sample exactly as the fit built it (rows
    ///   `[1, y_{t-1}', ..., y_{t-p}']`, no `1` for `trend = "n"`); the trace
    ///   is invariant to scaling `G`, so `Z'Z` and `Z'Z / T` give the same
    ///   term;
    /// * `B` is `m x m`: the row `e_1'` (constant only), then `params'` (`k`
    ///   rows), then `[0 | I_{k(p-1)} | 0]` — the companion matrix extended
    ///   by the deterministic term (statsmodels `_bmat_forc_cov`);
    /// * `Psi_i` are the MA coefficients and `Sigma_u` the df-adjusted
    ///   residual covariance.
    ///
    /// Computed as `tau_ab = <B^a, G^{-1} B^b G>_F` (Frobenius; one matrix
    /// product of the stacked `vec(B^a)` and `vec(G^{-1} B^b G)`) with the
    /// `b = 0` factor set to the identity **exactly** (not formed as
    /// `G^{-1} I G`, which would carry roundoff), so the `h = 1` trace factor
    /// is the integer `m` and `Omega(1) = m Sigma_u` to rounding (asserted at
    /// 1e-14), then `Omega(h) = sum_i Psi_i Sigma_u (sum_j
    /// tau_{h-1-i, h-1-j} Psi_j)'` with the inner sums for all `i` as one
    /// `h x h` by `h x k^2` product. Each `Omega(h)` is symmetrized. Cost
    /// `O(steps^3 (k^2 + 8) + steps^2 m^2 + steps m^3)`, budgeted by
    /// [`PARAM_UNCERTAINTY_WORK_BUDGET`].
    ///
    /// # Errors
    ///
    /// * [`VarError::InvalidArgument`] if `steps == 0`;
    /// * [`VarError::InvalidParameter`] naming `steps` when the work would
    ///   exceed [`PARAM_UNCERTAINTY_WORK_BUDGET`], or naming `lags` when it
    ///   would at every horizon (a wide VAR, where `2 steps m^3` dominates);
    /// * [`VarError::NotPositiveDefinite`] if `Z'Z` is numerically singular
    ///   (impossible after a successful fit, which already factored it).
    pub fn forecast_param_uncertainty(&self, steps: usize) -> Result<Vec<Mat<f64>>, VarError> {
        if steps == 0 {
            return Err(VarError::InvalidArgument {
                what: "steps = 0: a forecast covariance needs at least one step ahead; \
                       pass steps >= 1",
            });
        }
        self.check_param_uncertainty_budget(steps)?;
        crate::budget::check_steps(self.neqs, steps)?;
        let k = self.neqs;
        let lags = self.spec.lags;
        let n_trend = self.spec.trend.n_terms();
        let m = self.df_model;

        // Z exactly as `estimate` builds it, then G = Z'Z.
        let t_eff = self.nobs;
        let y_all = self.endog.as_ref();
        let mut z = Mat::<f64>::zeros(t_eff, m);
        for t in 0..t_eff {
            if n_trend == 1 {
                z[(t, 0)] = 1.0;
            }
            for lag in 1..=lags {
                for j in 0..k {
                    z[(t, n_trend + (lag - 1) * k + j)] = y_all[(lags + t - lag, j)];
                }
            }
        }
        let g = z.transpose() * &z;
        let g_llt = g
            .llt(Side::Lower)
            .map_err(|_| VarError::NotPositiveDefinite {
                what: "Z'Z, the VAR regressor cross-product (parameter-uncertainty term)",
            })?;

        // B: deterministic row, the k equations, the lag shift.
        let mut b = Mat::<f64>::zeros(m, m);
        if n_trend == 1 {
            b[(0, 0)] = 1.0;
        }
        if lags >= 1 {
            for r in 0..k {
                for c in 0..m {
                    b[(n_trend + r, c)] = self.params[(c, r)];
                }
            }
            for i in 0..k * (lags - 1) {
                b[(n_trend + k + i, n_trend + i)] = 1.0;
            }
        }

        // Row a of `p_flat` / `q_flat` is vec(P_a) / vec(Q_a) with
        // P_a = B^a and Q_a = G^{-1} B^a G, a = 0..steps-1, and
        // P_0 = Q_0 = I exactly; then tau = p_flat q_flat', i.e.
        // tau_ab = <P_a, Q_b>_F = tr[(B')^a G^{-1} B^b G].
        let mm = m * m;
        let mut p_flat = Mat::<f64>::zeros(steps, mm);
        let mut q_flat = Mat::<f64>::zeros(steps, mm);
        let mut pa = Mat::<f64>::identity(m, m);
        for a in 0..steps {
            let qa = if a == 0 {
                Mat::<f64>::identity(m, m)
            } else {
                pa = &pa * &b;
                g_llt.solve(&(&pa * &g))
            };
            for c in 0..m {
                for r in 0..m {
                    p_flat[(a, c * m + r)] = pa[(r, c)];
                    q_flat[(a, c * m + r)] = qa[(r, c)];
                }
            }
        }
        let tau = &p_flat * q_flat.transpose();
        drop(p_flat);
        drop(q_flat);

        // psi_rev row s is Psi_{steps-1-s} flattened row-major (entry
        // e = r k + c), so for horizon h the h x k^2 block of rows
        // steps-h..steps holds Psi_{h-1}, ..., Psi_0 and
        // W = tau[0..h, 0..h] psi_rev[steps-h.., :] has row a equal to
        // sum_{b<h} tau_{a,b} Psi_{h-1-b} — the inner sum for i = h-1-a — as
        // one matrix product. D_i = Psi_i Sigma_u stays row-major.
        let psi = self.ma_rep(steps - 1)?;
        let kk = k * k;
        let mut psi_rev = Mat::<f64>::zeros(steps, kk);
        let mut d_flat = vec![0.0_f64; steps * kk];
        for (i, ps) in psi.iter().enumerate() {
            let d = ps * &self.sigma_u;
            for r in 0..k {
                for c in 0..k {
                    psi_rev[(steps - 1 - i, r * k + c)] = ps[(r, c)];
                    d_flat[i * kk + r * k + c] = d[(r, c)];
                }
            }
        }

        // Omega(h) = sum_{i<h} D_i W_{h-1-i}'.
        let mut out = Vec::with_capacity(steps);
        let mut om = vec![0.0_f64; kk];
        for h in 1..=steps {
            let w = tau.as_ref().submatrix(0, 0, h, h)
                * psi_rev.as_ref().submatrix(steps - h, 0, h, kk);
            om.fill(0.0);
            for i in 0..h {
                let a = h - 1 - i;
                let d = &d_flat[i * kk..(i + 1) * kk];
                for r in 0..k {
                    for c in 0..k {
                        let mut acc = 0.0;
                        for l in 0..k {
                            acc += d[r * k + l] * w[(a, c * k + l)];
                        }
                        om[r * k + c] += acc;
                    }
                }
            }
            out.push(Mat::from_fn(k, k, |r, c| {
                if r == c {
                    om[r * k + c]
                } else {
                    0.5 * (om[r * k + c] + om[c * k + r])
                }
            }));
        }
        Ok(out)
    }

    /// The **joint** forecast-error covariance of every `(horizon, series)`
    /// cell, row-major `(steps * k) x (steps * k)` with cell `(h, j)` at index
    /// `h * k + j` (`h` zero-based, so horizon `h + 1`).
    ///
    /// [`VarResults::forecast_cov`] gives the `steps` diagonal blocks — the
    /// per-horizon `MSE(h)` — and those are the only blocks a marginal interval
    /// needs. A simultaneous band needs the cross-horizon blocks too, and they
    /// are available in closed form: forecast errors at different horizons share
    /// innovations, so for `a <= b` and `d = b - a`,
    ///
    /// ```text
    /// Cov(e_{a+1}, e_{b+1}) = sum_{i=0}^{a} Psi_i Sigma_u Psi_{i+d}'
    /// ```
    ///
    /// (Lütkepohl 2005, eq. 2.2.11 generalized off the diagonal). Adjacent
    /// horizons are strongly positively correlated — the `h` and `h+1` forecast
    /// errors share all but one innovation — which is precisely why the marginal
    /// bands read jointly do so much worse than their nominal level, and why the
    /// sup-t multiplier here is far below the Bonferroni one.
    ///
    /// The diagonal blocks are taken verbatim from
    /// [`VarResults::forecast_cov`], with the same `.max(0.0)` clamp
    /// [`VarResults::forecast_interval`] applies, so the standard errors implied
    /// by this matrix are bit-identical to the marginal ones.
    ///
    /// Like the marginal interval, this treats the coefficients as known: it is
    /// innovation uncertainty only.
    ///
    /// # Errors
    ///
    /// [`VarError::InvalidArgument`] if `steps == 0`; propagates
    /// [`VarResults::ma_rep`] failures.
    pub fn forecast_error_cov_joint(&self, steps: usize) -> Result<Vec<f64>, VarError> {
        let covs = self.forecast_cov(steps)?;
        let psi = self.ma_rep(steps - 1)?;
        let k = self.neqs;
        let n = steps * k;
        let mut out = vec![0.0f64; n * n];
        for a in 0..steps {
            // Diagonal block: MSE(a + 1), symmetrized off the diagonal and
            // clamped on it exactly as `forecast_interval` clamps.
            for j1 in 0..k {
                for j2 in 0..k {
                    let v = if j1 == j2 {
                        covs[a][(j1, j1)].max(0.0)
                    } else {
                        0.5 * (covs[a][(j1, j2)] + covs[a][(j2, j1)])
                    };
                    out[(a * k + j1) * n + a * k + j2] = v;
                }
            }
            for b in (a + 1)..steps {
                let d = b - a;
                let mut acc = Mat::<f64>::zeros(k, k);
                for i in 0..=a {
                    acc += &psi[i] * &self.sigma_u * psi[i + d].transpose();
                }
                for j1 in 0..k {
                    for j2 in 0..k {
                        out[(a * k + j1) * n + b * k + j2] = acc[(j1, j2)];
                        out[(b * k + j2) * n + a * k + j1] = acc[(j1, j2)];
                    }
                }
            }
        }
        Ok(out)
    }

    /// Point forecasts with **both** the marginal interval and a
    /// **simultaneous** band over a declared family of `(horizon, series)`
    /// cells.
    ///
    /// The marginal `lower`/`upper` on the returned object are bit-identical to
    /// [`VarResults::forecast_interval`]'s; the simultaneous band reuses the
    /// same point forecasts and the same standard errors and changes only the
    /// multiplier:
    ///
    /// * [`BandMethod::Pointwise`] — `z_{1 - alpha/2}`, i.e. the simultaneous
    ///   band *is* the marginal band. Present so a caller can route all four
    ///   methods through one path.
    /// * [`BandMethod::SupT`] — the `1 - alpha` quantile of `max |t|` under the
    ///   exact joint forecast-error distribution
    ///   ([`VarResults::forecast_error_cov_joint`]), by `n_sim` Gaussian draws
    ///   from a `seed`-derived stream. Reproducible from `seed` alone; expose
    ///   that seed to the user. This is the route to prefer: the joint
    ///   covariance is known in closed form here, not estimated.
    /// * [`BandMethod::Sidak`] / [`BandMethod::Bonferroni`] — closed forms in
    ///   `K`, needing neither the covariance nor a seed. Both are markedly
    ///   conservative here, because forecast errors at adjacent horizons are
    ///   strongly positively correlated and neither method knows that.
    ///
    /// # What it fixes, and what it does not
    ///
    /// It fixes multiplicity, exactly: conditional on the coefficients, the
    /// sup-t band's joint coverage is its nominal level by construction. It
    /// inherits the marginal band's one approximation — coefficients treated as
    /// known — so in finite samples it under-covers jointly by about as much as
    /// the marginal band under-covers marginally, and no more.
    ///
    /// # Budget
    ///
    /// The joint covariance is `(steps k)^2` values and its cross-horizon
    /// blocks cost about `steps^3 / 6` products of `k x k` matrices (CUBIC in
    /// `steps`: 2.3 s at `steps = 400` on a bivariate VAR). Both are checked
    /// before anything is allocated, against [`crate::budget::BUDGET_VALUES`]
    /// and [`JOINT_COV_WORK_BUDGET`], and refused naming `steps`. The sup-t
    /// simulation is checked against the shared sup-t budgets
    /// ([`simultaneous::sup_t_n_sim_fits`]) and refused naming `band_n_sim`.
    /// Each refusal states the largest value that fits.
    ///
    /// # Errors
    ///
    /// * [`VarError::InvalidArgument`] if `steps == 0`, or `n_sim < 2` under
    ///   [`BandMethod::SupT`];
    /// * [`VarError::SizeBudget`] naming `steps` or `band_n_sim` beyond the
    ///   budgets above;
    /// * [`VarError::InvalidParameter`] if `alpha` is not strictly inside
    ///   `(0, 1)`;
    /// * [`VarError::Stats`] from the simultaneous-band layer.
    pub fn forecast_interval_simultaneous(
        &self,
        steps: usize,
        alpha: f64,
        method: BandMethod,
        scope: ForecastBandScope,
        seed: u64,
        n_sim: usize,
    ) -> Result<ForecastInterval, VarError> {
        if method == BandMethod::SupT && n_sim < 2 {
            return Err(VarError::InvalidArgument {
                what: "band_n_sim must be at least 2 to simulate a sup-t critical value; \
                       100000 is the recommended default and 50000 the practical \
                       floor (this is a quantile in the tail of a maximum)",
            });
        }
        if let Some(refusal) = joint_band_refusal(self.neqs, steps, method, scope, n_sim) {
            return Err(refusal);
        }
        let mut fc = self.forecast_interval(steps, alpha)?;
        let k = self.neqs;
        let z = simultaneous::pointwise_critical_value(alpha).map_err(VarError::Stats)?;

        // Cell families, in the (horizon-major, series-minor) order the joint
        // covariance uses. `families[f]` lists flat cell indices h * k + j.
        let families: Vec<Vec<usize>> = match scope {
            ForecastBandScope::Horizon => (0..k)
                .map(|j| (0..steps).map(|h| h * k + j).collect())
                .collect(),
            ForecastBandScope::All => vec![(0..steps * k).collect()],
        };
        let n_cells = families[0].len();

        let full = self.forecast_error_cov_joint(steps)?;
        let n = steps * k;
        let mut streams = if method == BandMethod::SupT {
            tsecon_rng::Stream::substreams(seed, families.len()).map_err(|_| {
                VarError::InvalidArgument {
                    what: "cannot spawn one reproducible RNG substream per band family",
                }
            })?
        } else {
            Vec::new()
        };
        let mut uniforms = if method == BandMethod::SupT {
            vec![0.0f64; simultaneous::required_uniforms(n_cells, n_sim)]
        } else {
            Vec::new()
        };

        let mut critical_value = vec![z; k];
        let mut n_cells_used = vec![0usize; k];
        for (f, cells) in families.iter().enumerate() {
            let mut sigma = Vec::with_capacity(n_cells * n_cells);
            for &a in cells {
                for &b in cells {
                    sigma.push(full[a * n + b]);
                }
            }
            let se = simultaneous::std_errors_from_cov(&sigma, n_cells).map_err(VarError::Stats)?;
            let n_used = se.iter().filter(|s| **s > 0.0).count();
            let c = if n_used == 0 {
                z
            } else {
                match method {
                    BandMethod::Pointwise => z,
                    BandMethod::SupT => {
                        if let Some(stream) = streams.get_mut(f) {
                            stream.fill_uniform_f64(&mut uniforms);
                        }
                        simultaneous::sup_t_from_cov(&sigma, n_cells, alpha, &uniforms)
                            .map_err(VarError::Stats)?
                    }
                    BandMethod::Sidak => simultaneous::sidak_critical_value(alpha, n_used)
                        .map_err(VarError::Stats)?,
                    BandMethod::Bonferroni => {
                        simultaneous::bonferroni_critical_value(alpha, n_used)
                            .map_err(VarError::Stats)?
                    }
                }
            };
            match scope {
                ForecastBandScope::Horizon => {
                    critical_value[f] = c;
                    n_cells_used[f] = n_used;
                }
                ForecastBandScope::All => {
                    critical_value = vec![c; k];
                    n_cells_used = vec![n_used; k];
                }
            }
        }

        let lower = Mat::from_fn(steps, k, |h, j| {
            fc.point[(h, j)] - critical_value[j] * fc.se[(h, j)]
        });
        let upper = Mat::from_fn(steps, k, |h, j| {
            fc.point[(h, j)] + critical_value[j] * fc.se[(h, j)]
        });
        fc.simultaneous = Some(ForecastSimultaneous {
            critical_value,
            lower,
            upper,
            n_cells,
            n_cells_used,
            pointwise: z,
            method,
            scope,
        });
        Ok(fc)
    }
}

/// The work cap of the joint forecast-error covariance a simultaneous band
/// builds: its cross-horizon blocks are `steps (steps^2 - 1) / 6` products
/// of `k x k` matrices, counted at `k^3 + 32` each (the 32 stands for the
/// fixed cost of a small product), against `2^31`: about ten seconds
/// (measured on a bivariate VAR(1) at 2.3 s for `steps = 400`, where the
/// count is about `2^28.7`, and 11 s for `steps = 685`, the largest that fits;
/// `lab/audit/round16/var_budgets/joint_cov_cost.py`).
pub const JOINT_COV_WORK_BUDGET: u128 = 1 << 31;

const JOINT_STEPS_WHAT: &str = "steps: a simultaneous forecast band builds the joint \
    forecast-error covariance of every (horizon, series) cell, (steps k)^2 values whose \
    cross-horizon blocks cost about steps^3 / 6 products of k x k matrices (CUBIC in steps), \
    and at this many steps it exceeds the 2^28-value (2 GiB) memory budget or the 2^31 work \
    budget (about ten seconds), checked before anything is allocated; reduce steps, or use \
    band=\"pointwise\"";

const BAND_N_SIM_WHAT: &str = "band_n_sim: the sup-t simulation draws band_n_sim Gaussian \
    vectors per family of K cells (K * band_n_sim uniforms, and n_families * band_n_sim * K^2 \
    multiply-adds to colour them), which exceeds the 2^28-value (2 GiB) memory budget or the \
    2^36 work budget, checked before anything is allocated; lower band_n_sim (100000 is the \
    default) or narrow band_scope";

/// The refusal, if any, of a simultaneous forecast band's joint covariance
/// (naming `steps`) or of its sup-t simulation (naming `band_n_sim`), with
/// the largest value that fits at the other arguments.
fn joint_band_refusal(
    k: usize,
    steps: usize,
    method: BandMethod,
    scope: ForecastBandScope,
    n_sim: usize,
) -> Option<VarError> {
    use crate::budget::{largest_fitting, BUDGET_VALUES};
    let family = |s: usize| -> (usize, usize) {
        match scope {
            ForecastBandScope::Horizon => (k, s),
            ForecastBandScope::All => (1, s.saturating_mul(k)),
        }
    };
    let joint_fits = |s: usize| {
        let (_, kc) = family(s);
        let values = s
            .checked_mul(k)
            .and_then(|n| n.checked_mul(n))
            .and_then(|nn| kc.checked_mul(kc)?.checked_mul(2)?.checked_add(nn));
        let (su, ku) = (s as u128, k as u128);
        let work = su
            .checked_mul(su)
            .and_then(|s2| s2.checked_mul(su))
            .map(|s3| s3 / 6)
            .and_then(|terms| {
                terms.checked_mul(ku.checked_mul(ku)?.checked_mul(ku)?.checked_add(32)?)
            });
        matches!(values, Some(v) if v <= BUDGET_VALUES)
            && matches!(work, Some(w) if w <= JOINT_COV_WORK_BUDGET)
    };
    let sim_fits = |s: usize, n: usize| {
        let (f, kc) = family(s);
        method != BandMethod::SupT || simultaneous::sup_t_n_sim_fits(f, kc, n)
    };
    if joint_fits(steps) && sim_fits(steps, n_sim) {
        return None;
    }
    if joint_fits(steps) {
        let (f, kc) = family(steps);
        let max = simultaneous::sup_t_max_n_sim(f, kc);
        if max >= 2 {
            return Some(VarError::SizeBudget {
                what: BAND_N_SIM_WHAT,
                name: "band_n_sim",
                value: n_sim,
                max: Some(max),
            });
        }
    }
    Some(VarError::SizeBudget {
        what: JOINT_STEPS_WHAT,
        name: "steps",
        value: steps,
        max: largest_fitting(1, steps, |s| joint_fits(s) && sim_fits(s, n_sim)),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn refusal(steps: usize, k: usize, lags: usize) -> Option<(&'static str, String)> {
        let m = 1 + k * lags;
        param_uncertainty_refusal(steps, k, lags, m).map(|e| match e {
            VarError::InvalidParameter { name, .. } => (name, e.to_string()),
            other => panic!("expected InvalidParameter, got {other:?}"),
        })
    }

    /// Audit round 15: the refusal quoted only `steps^3 (k^2 + 8) / 3` and
    /// named `steps`, while on a wide VAR the `2 steps m^3` term is what
    /// trips the budget (k = 60, lags = 20: m = 1201, so a 10-step horizon
    /// is 2.0x the budget although the quoted term is 0.00007x).
    #[test]
    fn the_refusal_names_the_terms_that_trip_the_budget() {
        // A small VAR past its horizon cap: CUBIC in steps, name steps.
        let (name, msg) = refusal(5000, 3, 2).expect("refused");
        assert_eq!(name, "steps");
        assert!(
            msg.contains("CUBIC in steps") && msg.contains("tens of periods"),
            "{msg}"
        );
        assert!(
            msg.contains("2 steps m^3") && msg.contains("m = n_trend + k*lags"),
            "{msg}"
        );
        // k = 60, lags = 20 (m = 1201): steps <= 4 fits, 5 and 10 do not, and
        // the m terms dominate, so the remedy is lags or series, not steps.
        assert!(refusal(4, 60, 20).is_none());
        for steps in [5, 10] {
            let (name, msg) = refusal(steps, 60, 20).expect("refused");
            assert_eq!(name, "steps");
            assert!(
                msg.contains("the m terms dominate") && msg.contains("reduce lags"),
                "{msg}"
            );
            assert!(!msg.contains("tens of periods"), "{msg}");
        }
        // k = 30, lags = 10 (m = 301) at steps = 200: the auditor's second case.
        let (name, msg) = refusal(200, 30, 10).expect("refused");
        assert_eq!(name, "steps");
        assert!(msg.contains("the m terms dominate"), "{msg}");
        // m = 2049: 2 m^3 alone exceeds 2^34, so no horizon fits — name lags.
        let (name, msg) = refusal(1, 64, 32).expect("refused");
        assert_eq!(name, "lags");
        assert!(msg.starts_with("invalid lags = 32: requires"), "{msg}");
        assert!(
            msg.contains("even at steps = 1") && msg.contains("reduce lags"),
            "{msg}"
        );
        // The boundary: 2 * 2048^3 is 2^34 exactly, so with the small terms
        // m = 2048 is over the budget at one step and m = 2047 is under.
        assert!(param_uncertainty_refusal(1, 1, 2046, 2047).is_none());
        assert_eq!(
            param_uncertainty_refusal(1, 1, 2047, 2048).map(|e| e.to_string()[..14].to_string()),
            Some("invalid lags =".to_string())
        );
        // Saturating at absurd steps, never overflowing.
        assert!(refusal(usize::MAX, 3, 2).is_some());
    }
}
