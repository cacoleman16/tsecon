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
//!
//! ## Unbalanced panels (the observation mask)
//!
//! On an unbalanced panel ([`PanelData::unbalanced`]) the two Kronecker
//! factors no longer commute, so the transformation is built the way
//! `PanelOLS` builds it (`PanelData._demean_both`: the Frisch-Waugh-Lovell
//! route): the per-entity projection (demeaning, or residualising on
//! `[1, t]` over the entity's observed periods) is exact on its own, and
//! time effects are then partialled out **exactly** by residualising the
//! projected data on the projected time dummies (one per period that
//! carries an observation, the first dropped) through a rank-revealing
//! least-squares step — the joint projection, not an alternating
//! approximation. Time-only effects demean per period over the entities
//! observed in it. Effect counts use the entities and periods that carry
//! at least one observation (`N_obs + T_obs - 1` under two-way effects),
//! the entity-cluster score sums run over each entity's observed cells,
//! and the Driscoll-Kraay per-period sums over the entities observed in
//! each period, with lags measured in **calendar** periods of the layout
//! (a period nobody observes contributes a zero score and still counts as
//! elapsed time). The golden fixture `fixtures/panel_unbalanced.json`
//! pins every effects menu and covariance against `PanelOLS` on the
//! Arellano-Bond `EmplUK` panel and on a seeded panel with entry, exit
//! and internal gaps; a fully observed mask takes the balanced code path
//! and is bit-identical to it.

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
    /// (entity), `T` (time), `N + T - 1` (both) — with `N` and `T` the
    /// entities and periods that carry an observation.
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
    /// Stacked observations `n`: `N * T` on a balanced panel, the number
    /// of observed cells on an unbalanced one.
    pub nobs: usize,
    /// Number of slope regressors `k`.
    pub nparams: usize,
    /// Number of entities `N` in the layout (rows of the panel).
    pub n_entities: usize,
    /// Number of periods `T` in the layout (columns of the panel).
    pub n_periods: usize,
    /// Residual degrees of freedom of the within estimator,
    /// `n - k - n_absorbed` (`n - k - N` for entity effects only: the
    /// absorbed entity means are counted).
    pub df_resid: usize,
    /// Which fixed effects were swept out.
    pub effects: FixedEffects,
    /// Number of absorbed parameters (effects plus entity-trend slopes;
    /// on an unbalanced panel only entities and periods with an
    /// observation count).
    pub n_absorbed: usize,
    /// The shared within-fit internals (demeaned design, residuals,
    /// bread) consumed by [`FePanelOls::inference`].
    within: WithinFit,
}

/// Fits the fixed-effects (within) estimator: entity demeaning of the
/// outcome and every regressor, then OLS on the stacked demeaned data
/// (Wooldridge 2010, eq. 10.41). On an unbalanced panel the demeaning
/// runs over each entity's observed cells.
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
/// module docs for the closed-form balanced-panel projections, the
/// unbalanced-panel route, and the linearmodels degrees-of-freedom
/// conventions), then OLS on the stacked transformed data. With
/// [`FixedEffects::ENTITY`] this is [`panel_ols_fe`] operation for
/// operation.
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
    let within = if data.is_balanced() {
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
        fit_within_with(&y, &x_cols, n_ent, n_per, effects)?
    } else {
        // Stack the observed cells only, entity-major.
        let cells = data.observed_cells();
        let n = cells.len();
        let mut y = vec![0.0_f64; n];
        let mut x_cols = vec![vec![0.0_f64; n]; k];
        let outcome = data.outcome();
        for (r, &(i, t)) in cells.iter().enumerate() {
            y[r] = outcome[(i, t)];
        }
        for (j, col) in x_cols.iter_mut().enumerate() {
            if let Some(m) = data.regressor(j) {
                for (r, &(i, t)) in cells.iter().enumerate() {
                    col[r] = m[(i, t)];
                }
            }
        }
        fit_within_masked(&y, &x_cols, &cells, n_ent, n_per, effects)?
    };
    let nobs = within.nobs;
    let n_absorbed = within.n_effects + within.n_trend_params;
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
    /// (`r = entity * n_periods + period` on a balanced panel; the
    /// observed cells in entity-major order otherwise — see
    /// [`FePanelOls::row_entities`] / [`FePanelOls::row_periods`]).
    #[must_use]
    pub fn within_residuals(&self) -> &[f64] {
        &self.within.resid
    }

    /// The entity of each stacked row of [`FePanelOls::within_residuals`].
    #[must_use]
    pub fn row_entities(&self) -> &[usize] {
        &self.within.row_entity
    }

    /// The period of each stacked row of [`FePanelOls::within_residuals`].
    #[must_use]
    pub fn row_periods(&self) -> &[usize] {
        &self.within.row_period
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
    /// Entity of each stacked row (`r / T` on a balanced panel). Rows are
    /// entity-major, so each entity's rows are contiguous.
    pub(crate) row_entity: Vec<usize>,
    /// Period of each stacked row (`r % T` on a balanced panel), in
    /// `0..n_periods`.
    pub(crate) row_period: Vec<usize>,
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
/// bit-identical; the whole balanced path is untouched by the 0.10.0
/// observation mask (which lives in [`fit_within_masked`]).
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

    let row_entity: Vec<usize> = (0..n).map(|r| r / n_periods).collect();
    let row_period: Vec<usize> = (0..n).map(|r| r % n_periods).collect();
    solve_within(
        yd,
        xd,
        x_cols,
        effects,
        WithinShape {
            n_periods,
            n_effects: effects.n_effects(n_entities, n_periods),
            n_trend_params: effects.n_trend_params(n_entities),
            row_entity,
            row_period,
        },
    )
}

/// The bookkeeping [`solve_within`] stamps on a [`WithinFit`].
struct WithinShape {
    n_periods: usize,
    n_effects: usize,
    n_trend_params: usize,
    row_entity: Vec<usize>,
    row_period: Vec<usize>,
}

/// Applies the within transformation for `effects` to the stacked
/// **observed** cells of an unbalanced panel (`cells[r] = (entity,
/// period)` of row `r`, entity-major so each entity's rows are
/// contiguous), then solves the within OLS exactly as
/// [`fit_within_with`] does. See the module docs for the projection
/// route (per-entity projection first, time effects partialled out by a
/// rank-revealing least-squares step) and the effect counts (entities
/// and periods with an observation).
pub(crate) fn fit_within_masked(
    y: &[f64],
    x_cols: &[Vec<f64>],
    cells: &[(usize, usize)],
    n_entities: usize,
    n_periods: usize,
    effects: FixedEffects,
) -> Result<WithinFit, PanelError> {
    let n = cells.len();
    let k = x_cols.len();
    debug_assert_eq!(y.len(), n);
    debug_assert!(x_cols.iter().all(|c| c.len() == n));
    debug_assert!(cells.windows(2).all(|w| w[0].0 <= w[1].0));

    // Entities and periods that carry an observation, and each entity's
    // contiguous row segment.
    let mut ent_count = vec![0usize; n_entities];
    let mut per_count = vec![0usize; n_periods];
    for &(i, t) in cells {
        ent_count[i] += 1;
        per_count[t] += 1;
    }
    let n_ent_obs = ent_count.iter().filter(|&&c| c > 0).count();
    let n_per_obs = per_count.iter().filter(|&&c| c > 0).count();
    let n_effects = effects.n_effects(n_ent_obs, n_per_obs);
    let n_trend_params = effects.n_trend_params(n_ent_obs);
    let n_absorbed = n_effects + n_trend_params;
    if n <= k + n_absorbed {
        return Err(if effects == FixedEffects::ENTITY {
            PanelError::DegreesOfFreedom {
                n,
                k,
                n_entities: n_ent_obs,
            }
        } else {
            PanelError::DegreesOfFreedomAbsorbed { n, k, n_absorbed }
        });
    }
    let mut segments: Vec<(usize, usize)> = Vec::with_capacity(n_ent_obs);
    let mut r = 0;
    while r < n {
        let i = cells[r].0;
        let start = r;
        while r < n && cells[r].0 == i {
            r += 1;
        }
        segments.push((start, r));
    }

    let mut yd = y.to_vec();
    let mut xd = Mat::<f64>::zeros(n, k);
    for (j, col) in x_cols.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            xd[(r, j)] = v;
        }
    }

    // The per-entity projection: demeaning, or residualisation on [1, t]
    // over the entity's observed periods (centred time keeps the two
    // steps orthogonal, as in the balanced path). Applied to y, every
    // regressor column, and — under time effects — every time dummy.
    let project_entity = |v: &mut [f64], times: &[f64]| {
        let cnt = v.len() as f64;
        let mean = v.iter().sum::<f64>() / cnt;
        if effects.entity_trends {
            let centre = times.iter().sum::<f64>() / cnt;
            let css: f64 = times
                .iter()
                .map(|t| {
                    let c = t - centre;
                    c * c
                })
                .sum();
            let mut cross = 0.0;
            for (t, x) in times.iter().zip(v.iter()) {
                cross += (t - centre) * (x - mean);
            }
            let slope = if css > 0.0 { cross / css } else { 0.0 };
            for (t, x) in times.iter().zip(v.iter_mut()) {
                *x -= mean + slope * (t - centre);
            }
        } else {
            for x in v.iter_mut() {
                *x -= mean;
            }
        }
    };
    let entity_projection = effects.entity || effects.entity_trends;
    let mut col = vec![0.0_f64; n];
    if entity_projection {
        for &(a, b) in &segments {
            let times: Vec<f64> = cells[a..b].iter().map(|c| c.1 as f64).collect();
            project_entity(&mut yd[a..b], &times);
            for j in 0..k {
                for r in a..b {
                    col[r - a] = xd[(r, j)];
                }
                project_entity(&mut col[..b - a], &times);
                for r in a..b {
                    xd[(r, j)] = col[r - a];
                }
            }
        }
    }
    if effects.time && !entity_projection {
        // Time-only effects: per-period demeaning over the entities
        // observed in that period (exact on its own).
        let mut sums = vec![0.0_f64; n_periods];
        for (r, &(_, t)) in cells.iter().enumerate() {
            sums[t] += yd[r];
        }
        for (r, &(_, t)) in cells.iter().enumerate() {
            yd[r] -= sums[t] / per_count[t] as f64;
        }
        for j in 0..k {
            sums.iter_mut().for_each(|s| *s = 0.0);
            for (r, &(_, t)) in cells.iter().enumerate() {
                sums[t] += xd[(r, j)];
            }
            for (r, &(_, t)) in cells.iter().enumerate() {
                xd[(r, j)] -= sums[t] / per_count[t] as f64;
            }
        }
    } else if effects.time {
        // Frisch-Waugh-Lovell: the time dummies of the observed periods
        // (first dropped), put through the same per-entity projection,
        // then y and every regressor residualised on them by a
        // rank-revealing least-squares step (thin SVD; components below
        // the numpy/linearmodels rank tolerance are dropped, so a
        // disconnected panel still gets the exact projection onto the
        // span of the projected dummies).
        let present: Vec<usize> = (0..n_periods).filter(|&t| per_count[t] > 0).collect();
        let m = present.len() - 1;
        if m > 0 {
            let mut d = Mat::<f64>::zeros(n, m);
            let mut which = vec![usize::MAX; n_periods];
            for (c, &t) in present.iter().enumerate().skip(1) {
                which[t] = c - 1;
            }
            for (r, &(_, t)) in cells.iter().enumerate() {
                if which[t] != usize::MAX {
                    d[(r, which[t])] = 1.0;
                }
            }
            for &(a, b) in &segments {
                let times: Vec<f64> = cells[a..b].iter().map(|c| c.1 as f64).collect();
                for c in 0..m {
                    for r in a..b {
                        col[r - a] = d[(r, c)];
                    }
                    project_entity(&mut col[..b - a], &times);
                    for r in a..b {
                        d[(r, c)] = col[r - a];
                    }
                }
            }
            let svd = d.thin_svd().map_err(|_| PanelError::SingularDesign {
                what: "within (fixed-effects) OLS on the unbalanced panel: the \
                       singular-value decomposition of the projected time dummies \
                       failed",
            })?;
            let u = svd.U();
            let sv: Vec<f64> = svd.S().column_vector().iter().copied().collect();
            let smax = sv.iter().copied().fold(0.0_f64, f64::max);
            let tol = smax * (n.max(m) as f64) * f64::EPSILON;
            let kept: Vec<usize> = (0..sv.len()).filter(|&c| sv[c] > tol).collect();
            let mut coef = vec![0.0_f64; kept.len()];
            let mut residualise = |v: &mut [f64]| {
                for (q, &c) in kept.iter().enumerate() {
                    let mut s = 0.0;
                    for (r, x) in v.iter().enumerate() {
                        s += u[(r, c)] * x;
                    }
                    coef[q] = s;
                }
                for (r, x) in v.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for (q, &c) in kept.iter().enumerate() {
                        s += u[(r, c)] * coef[q];
                    }
                    *x -= s;
                }
            };
            residualise(&mut yd);
            for j in 0..k {
                for r in 0..n {
                    col[r] = xd[(r, j)];
                }
                residualise(&mut col);
                for r in 0..n {
                    xd[(r, j)] = col[r];
                }
            }
        }
    }

    let row_entity: Vec<usize> = cells.iter().map(|c| c.0).collect();
    let row_period: Vec<usize> = cells.iter().map(|c| c.1).collect();
    solve_within(
        yd,
        xd,
        x_cols,
        effects,
        WithinShape {
            n_periods,
            n_effects,
            n_trend_params,
            row_entity,
            row_period,
        },
    )
}

/// The rank guard, the OLS solve and the residuals on the transformed
/// data — the tail every within fit shares (the balanced path calls it
/// with exactly the operations it performed before the mask existed).
fn solve_within(
    yd: Vec<f64>,
    xd: Mat<f64>,
    x_cols: &[Vec<f64>],
    effects: FixedEffects,
    shape: WithinShape,
) -> Result<WithinFit, PanelError> {
    let n = yd.len();
    let k = x_cols.len();
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
                   (duplicated or linearly dependent columns among the regressors)",
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
        n_periods: shape.n_periods,
        n_effects: shape.n_effects,
        n_trend_params: shape.n_trend_params,
        count_effects_in_cluster: effects.cluster_counts_effects(),
        row_entity: shape.row_entity,
        row_period: shape.row_period,
    })
}

impl WithinFit {
    /// Covariance/SE computation shared by the public entry points; see
    /// the module docs for formulas and the linearmodels conventions.
    /// Rows are visited in their stacked (entity-major) order, so on a
    /// balanced panel every sum accumulates exactly as it did before the
    /// observation mask existed.
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
                // per-entity score sums g_i = sum_t x_it u_it (each
                // entity's rows are contiguous).
                let mut meat = Mat::<f64>::zeros(k, k);
                let mut g = vec![0.0_f64; k];
                let mut r = 0;
                while r < n {
                    let entity = self.row_entity[r];
                    g.iter_mut().for_each(|v| *v = 0.0);
                    while r < n && self.row_entity[r] == entity {
                        let u = self.resid[r];
                        for (j, gj) in g.iter_mut().enumerate() {
                            *gj += self.xd[(r, j)] * u;
                        }
                        r += 1;
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
                // Per-period cross-sectional score sums a_t (calendar
                // periods of the layout).
                let t_len = self.n_periods;
                let mut agg = Mat::<f64>::zeros(t_len, k);
                for r in 0..n {
                    let t = self.row_period[r];
                    let u = self.resid[r];
                    for j in 0..k {
                        agg[(t, j)] += self.xd[(r, j)] * u;
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
