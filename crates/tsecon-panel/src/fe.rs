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
    /// `n - k - N` (the absorbed entity means are counted).
    pub df_resid: usize,
    /// The shared within-fit internals (demeaned design, residuals,
    /// bread) consumed by [`FePanelOls::inference`].
    within: WithinFit,
}

/// Fits the fixed-effects (within) estimator on a balanced panel: entity
/// demeaning of the outcome and every regressor, then OLS on the stacked
/// demeaned data (Wooldridge 2010, eq. 10.41).
///
/// Standard errors are computed on demand by [`FePanelOls::inference`].
///
/// # Errors
///
/// * [`PanelError::InvalidArgument`] if the panel has no regressors;
/// * [`PanelError::DegreesOfFreedom`] unless `nobs > k + N`;
/// * [`PanelError::SingularDesign`] if the within-transformed design is
///   collinear (e.g. a regressor constant within every entity).
pub fn panel_ols_fe(data: &PanelData) -> Result<FePanelOls, PanelError> {
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
    let within = fit_within(&y, &x_cols, n_ent, n_per)?;
    Ok(FePanelOls {
        params: within.params.clone(),
        names: data.names().to_vec(),
        nobs,
        nparams: k,
        n_entities: n_ent,
        n_periods: n_per,
        df_resid: nobs - k - n_ent,
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
    /// Demeaned design (`n x k`), stacked entity-major.
    xd: Mat<f64>,
    /// `(X'X)^{-1}` of the demeaned design — the sandwich bread.
    xtx_inv: Mat<f64>,
    pub(crate) nobs: usize,
    pub(crate) nparams: usize,
    pub(crate) n_entities: usize,
    pub(crate) n_periods: usize,
}

/// Entity-demeans `y` and the design columns (stacked entity-major, `T`
/// contiguous periods per entity), then solves the within OLS by
/// Householder QR least squares (the same solver idiom as
/// `tsecon-var`).
pub(crate) fn fit_within(
    y: &[f64],
    x_cols: &[Vec<f64>],
    n_entities: usize,
    n_periods: usize,
) -> Result<WithinFit, PanelError> {
    let n = n_entities * n_periods;
    let k = x_cols.len();
    debug_assert_eq!(y.len(), n);
    debug_assert!(x_cols.iter().all(|c| c.len() == n));
    if n <= k + n_entities {
        return Err(PanelError::DegreesOfFreedom {
            n,
            k,
            n_entities,
        });
    }

    // Entity demeaning.
    let mut yd = y.to_vec();
    let mut xd = Mat::<f64>::zeros(n, k);
    for (j, col) in x_cols.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            xd[(r, j)] = v;
        }
    }
    let tf = n_periods as f64;
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
    })
}

impl WithinFit {
    /// Covariance/SE computation shared by the public entry points; see
    /// the module docs for formulas and the linearmodels conventions.
    pub(crate) fn inference(&self, se_type: PanelSeType) -> Result<PanelInference, PanelError> {
        let (n, k) = (self.nobs, self.nparams);
        let nf = n as f64;
        let cov = match se_type {
            PanelSeType::NonRobust => {
                let rss: f64 = self.resid.iter().map(|u| u * u).sum();
                let s2 = rss / (n - k - self.n_entities) as f64;
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
                // counted (linearmodels auto_df; Stata areg/xtreg).
                let scale = nf / (n - k) as f64;
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
                // Effects counted: n / (n - k - N) (linearmodels kernel
                // covariance with count_effects=True).
                let scale = nf / (n - k - self.n_entities) as f64;
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
