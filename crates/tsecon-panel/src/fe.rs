//! Fixed-effects (within) panel OLS with panel-robust covariances.
//!
//! The within estimator removes entity fixed effects by demeaning every
//! variable entity by entity (Wooldridge 2010, ch. 10; Baltagi 2021):
//!
//! ```text
//! y_it - ybar_i = (x_it - xbar_i)' beta + (u_it - ubar_i)
//! ```
//!
//! and runs OLS on the stacked demeaned data. Because the demeaning
//! absorbs one mean per entity, the residual degrees of freedom are
//! `nobs - k - N` — the `N` absorbed effects are subtracted alongside the
//! `k` slopes.
//!
//! Every numeric convention matches `linearmodels.panel.PanelOLS` with
//! `entity_effects=True` (version 7.0; the golden fixture
//! `fixtures/panel.json` arbitrates). With `n = nobs`, `k` slopes, `N`
//! entities, within residuals `u` and demeaned design `X`:
//!
//! * [`PanelSeType::NonRobust`] — `s2 (X'X)^{-1}` with
//!   `s2 = u'u / (n - k - N)` (linearmodels `cov_type="unadjusted"`,
//!   `debiased=True`, effects counted);
//! * [`PanelSeType::ClusterEntity`] — the Arellano (1987) / Liang-Zeger
//!   (1986) cluster sandwich
//!   `c (X'X)^{-1} [ sum_i g_i g_i' ] (X'X)^{-1}` with per-entity score
//!   sums `g_i = sum_t x_it u_it`. The small-sample factor is
//!   `c = n / (n - k)`: linearmodels' `auto_df` rule does **not** count
//!   the absorbed entity effects when the effect is nested inside the
//!   cluster variable (the Stata `areg`/`xtreg` convention; see Cameron &
//!   Miller 2015, section VI.B, on nested fixed effects), and it applies
//!   no `G/(G-1)` group factor by default (`group_debias=False`).
//!   Verified against linearmodels 7.0 at machine precision;
//! * [`PanelSeType::DriscollKraay`] — Driscoll & Kraay (1998):
//!   cross-sectional sums of scores per period, `a_t = sum_i x_it u_it`,
//!   then a Bartlett-kernel HAC on the `T` aggregated scores,
//!   `S = Gamma_0 + sum_{j>=1} w_j (Gamma_j + Gamma_j')` with
//!   `Gamma_j = sum_{t>j} a_t a_{t-j}'`, and
//!   `cov = c (X'X)^{-1} S (X'X)^{-1}` with `c = n / (n - k - N)`
//!   (effects counted — the cluster nesting exemption does not apply).
//!   The `bandwidth` argument is the **lag-truncation** parameter: weights
//!   `w_j = 1 - j/(bandwidth + 1)` come from
//!   [`tsecon_hac::Kernel::Bartlett`], so `bandwidth = 4` includes lags
//!   1..=4 — exactly linearmodels `cov_type="kernel"`,
//!   `kernel="bartlett"`, `bandwidth=4` (their "bandwidth" is maxlags,
//!   not Andrews' continuous scale). Driscoll-Kraay is consistent under
//!   arbitrary cross-sectional dependence but needs a long time dimension;
//!   prefer entity clustering when `T` is short.
//!
//! ## Which effects are swept out ([`FixedEffects`])
//!
//! [`panel_ols_fe`] sweeps out entity effects only (the historical
//! surface). [`panel_ols_fe_with`] takes a [`FixedEffects`] menu: entity
//! effects, time effects, and entity-specific linear trends. On a
//! balanced panel every combination has an exact closed-form within
//! transformation, because the per-entity annihilator `M_T` (demeaning,
//! or residualising on `[1, t]` when trends are requested) and the
//! cross-sectional annihilator `I_N - J_N/N` act on different Kronecker
//! factors of the `N x T` layout and therefore commute:
//!
//! ```text
//! entity                 : (I_N)          (x) (I_T - J_T/T)
//! time                   : (I_N - J_N/N)  (x) (I_T)
//! entity + time          : (I_N - J_N/N)  (x) (I_T - J_T/T)   = y - ybar_i - ybar_t + ybar
//! entity + trends        : (I_N)          (x) M_T,  M_T = I - W(W'W)^{-1}W',  W = [1, t]
//! entity + trends + time : (I_N - J_N/N)  (x) M_T
//! ```
//!
//! (the last line is the joint projection, not an approximation: the
//! time dummies residualised on the per-entity trends are the same `T`
//! vectors in every entity, so their projection collapses to the
//! cross-sectional mean of the detrended data). The degrees-of-freedom
//! bookkeeping follows linearmodels: the absorbed **effects** count
//! `N` (entity), `T` (time) or `N + T - 1` (both, one redundant
//! constant), the trend slopes count like ordinary regressors (`N`, or
//! `N - 1` alongside time effects, whose span already holds the common
//! linear trend), and
//! the cluster-by-entity nesting exemption above applies **only** when
//! entity effects are the sole absorbed effects (linearmodels' `auto_df`
//! counts every effect under two-way or time-only specifications). The
//! golden fixture `fixtures/panel_dl.json` pins each of these
//! combinations against `PanelOLS` (the trends variant via explicit
//! entity x trend regressors).

use tsecon_hac::Kernel;
use tsecon_linalg::faer::linalg::solvers::{DenseSolveCore, SolveLstsq};
use tsecon_linalg::faer::{Mat, Side};

use crate::data::PanelData;
use crate::error::PanelError;

/// Covariance-estimator choice for the within estimator, mirroring
/// `linearmodels.panel.PanelOLS.fit(cov_type=...)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PanelSeType {
    /// Classical spherical-errors covariance `s2 (X'X)^{-1}` with
    /// `s2 = u'u / (n - k - N)` (linearmodels `cov_type="unadjusted"`).
    NonRobust,
    /// One-way cluster-robust sandwich, clustered by entity (Arellano
    /// 1987; Liang & Zeger 1986; linearmodels `cov_type="clustered"`,
    /// `cluster_entity=True`). Robust to arbitrary within-entity serial
    /// correlation and heteroskedasticity; requires many entities.
    ClusterEntity,
    /// Driscoll & Kraay (1998) kernel covariance (linearmodels
    /// `cov_type="kernel"`, `kernel="bartlett"`). Robust to
    /// cross-sectional dependence and serial correlation; requires a
    /// long time dimension.
    DriscollKraay {
        /// Bartlett lag-truncation bandwidth (linearmodels `bandwidth` /
        /// statsmodels `maxlags`): lags `1..=bandwidth` receive weight
        /// `1 - j/(bandwidth + 1)`.
        bandwidth: f64,
    },
}

/// Which fixed effects the within transformation sweeps out (see the
/// module docs for the closed-form balanced-panel projections and the
/// degrees-of-freedom conventions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedEffects {
    /// Entity (unit) effects `alpha_i`.
    pub entity: bool,
    /// Period effects `delta_t`, common to every entity.
    pub time: bool,
    /// Entity-specific linear trends `g_i * t` (requires `entity`; the
    /// trend needs its own intercept).
    pub entity_trends: bool,
}

impl FixedEffects {
    /// Entity effects only — the [`panel_ols_fe`] specification.
    pub const ENTITY: Self = Self {
        entity: true,
        time: false,
        entity_trends: false,
    };
    /// Entity and time effects (two-way fixed effects).
    pub const TWO_WAY: Self = Self {
        entity: true,
        time: true,
        entity_trends: false,
    };

    /// Rejects specifications that are not a panel model: no effect at
    /// all (that is pooled OLS without a constant — use a plain OLS), or
    /// entity trends without entity effects.
    ///
    /// # Errors
    ///
    /// [`PanelError::InvalidArgument`] for either case.
    pub fn validate(&self) -> Result<(), PanelError> {
        if !self.entity && !self.time && !self.entity_trends {
            return Err(PanelError::InvalidArgument {
                what: "no fixed effects requested (entity_effects=false, \
                       time_effects=false, entity_trends=false): the within \
                       estimator then reduces to pooled OLS without a constant, \
                       which is not a panel model — request entity and/or time \
                       effects, or run a plain OLS on the stacked data",
            });
        }
        if self.entity_trends && !self.entity {
            return Err(PanelError::InvalidArgument {
                what: "entity_trends=true requires entity_effects=true: an \
                       entity-specific linear trend g_i * t needs its own \
                       intercept alpha_i, otherwise the trend lines are forced \
                       through a common origin — pass entity_effects=true",
            });
        }
        Ok(())
    }

    /// Number of absorbed effects in linearmodels' accounting: `N`
    /// (entity), `T` (time), `N + T - 1` (both).
    pub(crate) fn n_effects(&self, n_entities: usize, n_periods: usize) -> usize {
        let mut n = 0;
        let mut drop_first = false;
        if self.entity {
            n += n_entities;
            drop_first = true;
        }
        if self.time {
            n += n_periods - usize::from(drop_first);
        }
        n
    }

    /// Number of absorbed entity-trend slopes: `N` when requested, or
    /// `N - 1` when time effects are also present (the common linear
    /// trend `sum_i g_i t` already lies in the span of the time dummies,
    /// so one slope is redundant — linearmodels needs one explicit
    /// entity x trend column dropped for the same reason).
    pub(crate) fn n_trend_params(&self, n_entities: usize) -> usize {
        if self.entity_trends {
            n_entities - usize::from(self.time)
        } else {
            0
        }
    }

    /// Total absorbed parameters `n_effects + n_trend_params`.
    pub(crate) fn n_absorbed(&self, n_entities: usize, n_periods: usize) -> usize {
        self.n_effects(n_entities, n_periods) + self.n_trend_params(n_entities)
    }

    /// linearmodels' `auto_df` rule for entity clusters: the absorbed
    /// effects are exempt from the cluster small-sample factor only when
    /// entity effects are the sole absorbed effects (nested in the
    /// clusters); two-way and time-only specifications count them.
    pub(crate) fn cluster_counts_effects(&self) -> bool {
        !(self.entity && !self.time)
    }
}

impl Default for FixedEffects {
    fn default() -> Self {
        Self::ENTITY
    }
}

/// Standard errors, t-statistics, and the full parameter covariance for
/// one [`PanelSeType`].
#[derive(Debug, Clone)]
pub struct PanelInference {
    /// Which covariance estimator produced this inference.
    pub se_type: PanelSeType,
    /// Parameter covariance matrix, `k x k`.
    pub cov: Mat<f64>,
    /// Standard errors `sqrt(diag(cov))`, one per slope.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`, one per slope.
    pub tvalues: Vec<f64>,
}

/// A fitted fixed-effects (within) panel regression; produced by
/// [`panel_ols_fe`].
#[derive(Debug, Clone)]
pub struct FePanelOls {
    /// Slope estimates, in the order the regressors were supplied.
    pub params: Vec<f64>,
    /// Regressor names, aligned with `params`.
    pub names: Vec<String>,
    /// Total stacked observations `n = N * T`.
    pub nobs: usize,
    /// Number of slope regressors `k`.
    pub nparams: usize,
    /// Number of entities `N` (absorbed fixed effects).
    pub n_entities: usize,
    /// Number of periods `T` per entity.
    pub n_periods: usize,
    /// Residual degrees of freedom of the within estimator,
    /// `n - k - n_absorbed` (`n - k - N` for entity effects only: the
    /// absorbed entity means are counted).
    pub df_resid: usize,
    /// Which fixed effects were swept out.
    pub effects: FixedEffects,
    /// Number of absorbed parameters (effects plus entity-trend slopes).
    pub n_absorbed: usize,
    /// The shared within-fit internals (demeaned design, residuals,
    /// bread) consumed by [`FePanelOls::inference`].
    within: WithinFit,
}

/// Fits the fixed-effects (within) estimator on a balanced panel: entity
/// demeaning of the outcome and every regressor, then OLS on the stacked
/// demeaned data (Wooldridge 2010, eq. 10.41).
///
/// Standard errors are computed on demand by [`FePanelOls::inference`].
/// This is [`panel_ols_fe_with`] at [`FixedEffects::ENTITY`].
///
/// # Errors
///
/// * [`PanelError::InvalidArgument`] if the panel has no regressors;
/// * [`PanelError::DegreesOfFreedom`] unless `nobs > k + N`;
/// * [`PanelError::SingularDesign`] if the within-transformed design is
///   collinear (e.g. a regressor constant within every entity).
pub fn panel_ols_fe(data: &PanelData) -> Result<FePanelOls, PanelError> {
    panel_ols_fe_with(data, FixedEffects::ENTITY)
}

/// Fits the within estimator sweeping out the requested [`FixedEffects`]
/// (entity effects, time effects, entity-specific linear trends — see the
/// module docs for the closed-form balanced-panel projections and the
/// linearmodels degrees-of-freedom conventions), then OLS on the stacked
/// transformed data. With [`FixedEffects::ENTITY`] this is
/// [`panel_ols_fe`] operation for operation.
///
/// # Errors
///
/// * [`PanelError::InvalidArgument`] if the panel has no regressors, or
///   the effects menu is not a panel model ([`FixedEffects::validate`]);
/// * [`PanelError::DegreesOfFreedom`] /
///   [`PanelError::DegreesOfFreedomAbsorbed`] unless
///   `nobs > k + n_absorbed`;
/// * [`PanelError::SingularDesign`] if the within-transformed design is
///   collinear (a regressor the effects absorb, or dependent columns).
pub fn panel_ols_fe_with(
    data: &PanelData,
    effects: FixedEffects,
) -> Result<FePanelOls, PanelError> {
    effects.validate()?;
    let k = data.n_regressors();
    if k == 0 {
        return Err(PanelError::InvalidArgument {
            what: "the panel has no regressors; the within estimator needs \
                   at least one slope (entity effects alone are not a model)",
        });
    }
    let (n_ent, n_per) = (data.n_entities(), data.n_periods());
    let nobs = data.nobs();
    // Stack entity-major: row index r = i * T + t.
    let mut y = vec![0.0_f64; nobs];
    let mut x_cols = vec![vec![0.0_f64; nobs]; k];
    let outcome = data.outcome();
    for i in 0..n_ent {
        for t in 0..n_per {
            y[i * n_per + t] = outcome[(i, t)];
        }
    }
    for (j, col) in x_cols.iter_mut().enumerate() {
        // The regressor index is in range by construction (j < k).
        if let Some(m) = data.regressor(j) {
            for i in 0..n_ent {
                for t in 0..n_per {
                    col[i * n_per + t] = m[(i, t)];
                }
            }
        }
    }
    let within = fit_within_with(&y, &x_cols, n_ent, n_per, effects)?;
    let n_absorbed = effects.n_absorbed(n_ent, n_per);
    Ok(FePanelOls {
        params: within.params.clone(),
        names: data.names().to_vec(),
        nobs,
        nparams: k,
        n_entities: n_ent,
        n_periods: n_per,
        df_resid: nobs - k - n_absorbed,
        effects,
        n_absorbed,
        within,
    })
}

impl FePanelOls {
    /// Standard errors, t-statistics, and the parameter covariance under
    /// the requested [`PanelSeType`] (see the module docs for the exact
    /// formulas, references, and the linearmodels degrees-of-freedom
    /// conventions).
    ///
    /// # Errors
    ///
    /// [`PanelError::InvalidBandwidth`] for a negative/non-finite
    /// Driscoll-Kraay bandwidth.
    pub fn inference(&self, se_type: PanelSeType) -> Result<PanelInference, PanelError> {
        self.within.inference(se_type)
    }

    /// The within (demeaned-scale) residuals, stacked entity-major
    /// (`r = entity * n_periods + period`).
    #[must_use]
    pub fn within_residuals(&self) -> &[f64] {
        &self.within.resid
    }
}

/// Internal within-OLS fit on stacked entity-major data, shared between
/// [`panel_ols_fe`] and the per-horizon regressions in `lp.rs`.
#[derive(Debug, Clone)]
pub(crate) struct WithinFit {
    /// Slope estimates.
    pub(crate) params: Vec<f64>,
    /// Within residuals, stacked entity-major.
    pub(crate) resid: Vec<f64>,
    /// Demeaned design (`n x k`), stacked entity-major. `pub(crate)` so
    /// the split-panel jackknife in `lp.rs` can build its adjusted-score
    /// sandwich from the same demeaned data.
    pub(crate) xd: Mat<f64>,
    /// `(X'X)^{-1}` of the demeaned design — the sandwich bread.
    pub(crate) xtx_inv: Mat<f64>,
    pub(crate) nobs: usize,
    pub(crate) nparams: usize,
    pub(crate) n_entities: usize,
    pub(crate) n_periods: usize,
    /// Absorbed effects in linearmodels' accounting (`N`, `T`, or
    /// `N + T - 1`).
    pub(crate) n_effects: usize,
    /// Absorbed entity-trend slopes (`N` when trends are requested),
    /// counted like regressors in every covariance.
    pub(crate) n_trend_params: usize,
    /// Whether the cluster-by-entity small-sample factor counts the
    /// absorbed effects (false only for entity-only effects, which are
    /// nested in the clusters).
    pub(crate) count_effects_in_cluster: bool,
}

/// Entity-demeans `y` and the design columns (stacked entity-major, `T`
/// contiguous periods per entity), then solves the within OLS by
/// Householder QR least squares (the same solver idiom as
/// `tsecon-var`). This is [`fit_within_with`] at
/// [`FixedEffects::ENTITY`].
pub(crate) fn fit_within(
    y: &[f64],
    x_cols: &[Vec<f64>],
    n_entities: usize,
    n_periods: usize,
) -> Result<WithinFit, PanelError> {
    fit_within_with(y, x_cols, n_entities, n_periods, FixedEffects::ENTITY)
}

/// Applies the within transformation for `effects` to `y` and the design
/// columns (stacked entity-major, `T` contiguous periods per entity),
/// then solves the within OLS by Householder QR least squares. The
/// entity-only path performs exactly the operations the pre-0.9.0
/// `fit_within` performed, in the same order, so its results are
/// bit-identical.
pub(crate) fn fit_within_with(
    y: &[f64],
    x_cols: &[Vec<f64>],
    n_entities: usize,
    n_periods: usize,
    effects: FixedEffects,
) -> Result<WithinFit, PanelError> {
    let n = n_entities * n_periods;
    let k = x_cols.len();
    debug_assert_eq!(y.len(), n);
    debug_assert!(x_cols.iter().all(|c| c.len() == n));
    let n_absorbed = effects.n_absorbed(n_entities, n_periods);
    if n <= k + n_absorbed {
        return Err(if effects == FixedEffects::ENTITY {
            PanelError::DegreesOfFreedom { n, k, n_entities }
        } else {
            PanelError::DegreesOfFreedomAbsorbed { n, k, n_absorbed }
        });
    }

    let mut yd = y.to_vec();
    let mut xd = Mat::<f64>::zeros(n, k);
    for (j, col) in x_cols.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            xd[(r, j)] = v;
        }
    }
    let tf = n_periods as f64;
    if effects.entity && !effects.entity_trends {
        // Entity demeaning (the original code path, kept operation for
        // operation so `panel_ols_fe` stays bit-identical).
        for i in 0..n_entities {
            let base = i * n_periods;
            let ymean = yd[base..base + n_periods].iter().sum::<f64>() / tf;
            for v in &mut yd[base..base + n_periods] {
                *v -= ymean;
            }
            for j in 0..k {
                let mut xmean = 0.0;
                for t in 0..n_periods {
                    xmean += xd[(base + t, j)];
                }
                xmean /= tf;
                for t in 0..n_periods {
                    xd[(base + t, j)] -= xmean;
                }
            }
        }
    }
    if effects.entity_trends {
        // Per-entity residualisation on [1, t]: with centred time
        // c_t = t - (T-1)/2 the annihilator is v - vbar - b c_t,
        // b = sum_t c_t v_t / sum_t c_t^2 (the intercept absorbs the
        // entity effect, so no separate demeaning step is needed).
        let centre = (n_periods as f64 - 1.0) / 2.0;
        let css: f64 = (0..n_periods)
            .map(|t| {
                let c = t as f64 - centre;
                c * c
            })
            .sum();
        let detrend = |v: &mut [f64]| {
            let mean = v.iter().sum::<f64>() / tf;
            let mut cross = 0.0;
            for (t, x) in v.iter().enumerate() {
                cross += (t as f64 - centre) * (x - mean);
            }
            let slope = if css > 0.0 { cross / css } else { 0.0 };
            for (t, x) in v.iter_mut().enumerate() {
                *x -= mean + slope * (t as f64 - centre);
            }
        };
        let mut col = vec![0.0_f64; n_periods];
        for i in 0..n_entities {
            let base = i * n_periods;
            detrend(&mut yd[base..base + n_periods]);
            for j in 0..k {
                for t in 0..n_periods {
                    col[t] = xd[(base + t, j)];
                }
                detrend(&mut col);
                for t in 0..n_periods {
                    xd[(base + t, j)] = col[t];
                }
            }
        }
    }
    if effects.time {
        // Cross-sectional demeaning per period, (I_N - J_N/N) (x) I_T,
        // applied after the per-entity step (the two Kronecker factors
        // commute, so this is the joint projection).
        let nf = n_entities as f64;
        for t in 0..n_periods {
            let mut ymean = 0.0;
            for i in 0..n_entities {
                ymean += yd[i * n_periods + t];
            }
            ymean /= nf;
            for i in 0..n_entities {
                yd[i * n_periods + t] -= ymean;
            }
            for j in 0..k {
                let mut xmean = 0.0;
                for i in 0..n_entities {
                    xmean += xd[(i * n_periods + t, j)];
                }
                xmean /= nf;
                for i in 0..n_entities {
                    xd[(i * n_periods + t, j)] -= xmean;
                }
            }
        }
    }

    // Rank guard on the within-transformed design. The Cholesky below is a
    // positive-definiteness test, not a rank test: it passes on a design
    // the within transformation has annihilated unless the residue is
    // bit-exactly zero — an entity-constant regressor stored as ordinary
    // doubles (log land area, a share in [0, 1]) demeans to an O(1e-16)
    // residue that survives it, and the cluster covariance then launders
    // the blown-up bread into a publishable t-statistic. linearmodels
    // deliberately refuses this design (`AbsorbingEffectError`) and the
    // docstring promises its conventions. Two scale-invariant checks:
    //
    // * per-column absorption: the demeaned column norm collapses
    //   relative to the raw column norm exactly when the fixed effects
    //   absorb the regressor;
    // * joint rank: the numpy/linearmodels singular-value criterion
    //   (`sigma_min <= sigma_max * max(n, k) * eps`) catches duplicated
    //   and linearly dependent demeaned columns — and, unlike the old
    //   guard, accepts a merely ill-conditioned design, so the check is
    //   monotone in the perturbation.
    const ABSORPTION_TOL: f64 = 1e-10;
    for (j, col) in x_cols.iter().enumerate() {
        let raw = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        let within = (0..n).map(|r| xd[(r, j)] * xd[(r, j)]).sum::<f64>().sqrt();
        if within <= ABSORPTION_TOL * raw {
            return Err(PanelError::SingularDesign {
                what: if effects == FixedEffects::ENTITY {
                    "within (fixed-effects) OLS: a regressor is constant within \
                     every entity, so the fixed effects absorb it and no within \
                     variation remains to identify its coefficient"
                } else {
                    "within (fixed-effects) OLS: the requested effects absorb a \
                     regressor entirely (constant within every entity, common to \
                     every entity in each period under time effects, or an exact \
                     linear trend per entity under entity trends), so no within \
                     variation remains to identify its coefficient"
                },
            });
        }
    }
    let svd = xd.thin_svd().map_err(|_| PanelError::SingularDesign {
        what: "within (fixed-effects) OLS: the singular-value decomposition of the \
               demeaned design failed",
    })?;
    let sv: Vec<f64> = svd.S().column_vector().iter().copied().collect();
    let smax = sv.iter().copied().fold(0.0_f64, f64::max);
    let rank_tol = smax * (n.max(k) as f64) * f64::EPSILON;
    if smax <= 0.0 || sv.iter().any(|&v| v <= rank_tol) {
        return Err(PanelError::SingularDesign {
            what: "within (fixed-effects) OLS: the demeaned design is rank-deficient \
                   (duplicated or linearly dependent regressors)",
        });
    }

    let ymat = Mat::from_fn(n, 1, |r, _| yd[r]);
    let xtx = xd.transpose() * &xd;
    let xtx_inv = xtx
        .llt(Side::Lower)
        .map_err(|_| PanelError::SingularDesign {
            what: "within (fixed-effects) OLS",
        })?
        .inverse();
    let params_mat = xd.qr().solve_lstsq(&ymat);
    let params: Vec<f64> = (0..k).map(|j| params_mat[(j, 0)]).collect();
    let fitted = &xd * &params_mat;
    let resid: Vec<f64> = (0..n).map(|r| yd[r] - fitted[(r, 0)]).collect();

    Ok(WithinFit {
        params,
        resid,
        xd,
        xtx_inv,
        nobs: n,
        nparams: k,
        n_entities,
        n_periods,
        n_effects: effects.n_effects(n_entities, n_periods),
        n_trend_params: effects.n_trend_params(n_entities),
        count_effects_in_cluster: effects.cluster_counts_effects(),
    })
}

impl WithinFit {
    /// Covariance/SE computation shared by the public entry points; see
    /// the module docs for formulas and the linearmodels conventions.
    pub(crate) fn inference(&self, se_type: PanelSeType) -> Result<PanelInference, PanelError> {
        let (n, k) = (self.nobs, self.nparams);
        let nf = n as f64;
        // linearmodels' `extra_df` accounting: the absorbed effects are
        // counted in every covariance except the entity-cluster sandwich
        // under entity-only effects (nested); trend slopes always count.
        let k_all = k + self.n_trend_params;
        let df_counted = n - k_all - self.n_effects;
        let cov = match se_type {
            PanelSeType::NonRobust => {
                let rss: f64 = self.resid.iter().map(|u| u * u).sum();
                let s2 = rss / df_counted as f64;
                Mat::from_fn(k, k, |i, j| s2 * self.xtx_inv[(i, j)])
            }
            PanelSeType::ClusterEntity => {
                // Meat: sum over entities of outer products of the
                // per-entity score sums g_i = sum_t x_it u_it.
                let mut meat = Mat::<f64>::zeros(k, k);
                let mut g = vec![0.0_f64; k];
                for i in 0..self.n_entities {
                    g.iter_mut().for_each(|v| *v = 0.0);
                    for t in 0..self.n_periods {
                        let r = i * self.n_periods + t;
                        let u = self.resid[r];
                        for (j, gj) in g.iter_mut().enumerate() {
                            *gj += self.xd[(r, j)] * u;
                        }
                    }
                    for a in 0..k {
                        for b in 0..k {
                            meat[(a, b)] += g[a] * g[b];
                        }
                    }
                }
                // Nested-cluster df convention: n / (n - k), effects NOT
                // counted (linearmodels auto_df; Stata areg/xtreg) when
                // entity effects are the only absorbed effects; counted
                // otherwise.
                let scale = if self.count_effects_in_cluster {
                    nf / df_counted as f64
                } else {
                    nf / (n - k_all) as f64
                };
                let sw = &self.xtx_inv * &meat * &self.xtx_inv;
                Mat::from_fn(k, k, |i, j| scale * sw[(i, j)])
            }
            PanelSeType::DriscollKraay { bandwidth } => {
                if !bandwidth.is_finite() || bandwidth < 0.0 {
                    return Err(PanelError::InvalidBandwidth { value: bandwidth });
                }
                // Per-period cross-sectional score sums a_t.
                let t_len = self.n_periods;
                let mut agg = Mat::<f64>::zeros(t_len, k);
                for i in 0..self.n_entities {
                    for t in 0..t_len {
                        let r = i * t_len + t;
                        let u = self.resid[r];
                        for j in 0..k {
                            agg[(t, j)] += self.xd[(r, j)] * u;
                        }
                    }
                }
                // Bartlett-kernel HAC on the aggregated scores; weights
                // from the library's single kernel owner (tsecon-hac).
                let kernel = Kernel::Bartlett;
                let mut meat = Mat::<f64>::zeros(k, k);
                for lag in 0..t_len {
                    let w = kernel.weight(lag, bandwidth);
                    if lag > 0 && w == 0.0 {
                        break; // Bartlett truncates.
                    }
                    for t in lag..t_len {
                        for a in 0..k {
                            for b in 0..k {
                                let gab = agg[(t, a)] * agg[(t - lag, b)];
                                if lag == 0 {
                                    meat[(a, b)] += gab;
                                } else {
                                    meat[(a, b)] += w * gab;
                                    meat[(b, a)] += w * gab;
                                }
                            }
                        }
                    }
                }
                // Effects counted: n / (n - k - n_effects) (linearmodels
                // kernel covariance with count_effects=True).
                let scale = nf / df_counted as f64;
                let sw = &self.xtx_inv * &meat * &self.xtx_inv;
                Mat::from_fn(k, k, |i, j| scale * sw[(i, j)])
            }
        };

        let mut bse = Vec::with_capacity(k);
        for i in 0..k {
            // The Bartlett kernel and the cluster/nonrobust meats are
            // positive semi-definite, so the diagonal cannot go negative
            // in exact arithmetic; clamp roundoff.
            bse.push(cov[(i, i)].max(0.0).sqrt());
        }
        let tvalues = self
            .params
            .iter()
            .zip(bse.iter())
            .map(|(p, s)| p / s)
            .collect();
        Ok(PanelInference {
            se_type,
            cov,
            bse,
            tvalues,
        })
    }
}
