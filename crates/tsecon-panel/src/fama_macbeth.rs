//! The Fama-MacBeth (1973) two-pass estimator: one cross-sectional OLS per
//! calendar period, then the time-series mean of the per-period
//! coefficients with the dispersion-based standard error, following
//! `linearmodels.panel.FamaMacBeth` (version 7.0; the golden fixture
//! `fixtures/panel_static.json` arbitrates).
//!
//! **Per-period regressions.** For each calendar period `t` the design
//! `X_t` (`n_t x K`) stacks the entities observed in `t`. The period is
//! skipped (a `NaN` row of `params_per_period`, `periods_used[t] = false`)
//! when `n_t < K` or `X_t` is rank-deficient. tsecon decides the rank by
//! the numpy singular-value criterion
//! `sigma_min <= sigma_max * max(n_t, K) * eps`; linearmodels uses LAPACK
//! `gelsy` with `cond = max(n_t, K) * eps` (a column-pivoted QR condition
//! estimate). The two agree away from the numerical boundary (a design
//! perturbed at `1e-14` is rank 3 for `gelsy` and rank 2 for numpy), and
//! the fixtures stay clear of it.
//!
//! **Pooling.** With `V` the `T_v >= 2` valid periods in calendar order,
//! `beta = mean_{s in V} beta_s` and `e_s = beta_s - beta` over the
//! **compressed** sequence `V` (linearmodels' convention: a skipped period
//! does not count as elapsed time for the HAC lags).
//!
//! * [`FamaMacBethSe::Standard`] — `cov = sum_s e_s e_s' / (T_v (T_v - 1))`,
//!   i.e. `bse = sd(beta_s, ddof = 1) / sqrt(T_v)` (linearmodels
//!   `cov_type="unadjusted"`; its `"robust"` is the same number);
//! * [`FamaMacBethSe::Hac`] — the Newey-West (1987) Bartlett HAC of the
//!   `beta_s` series, `cov = [G_0 + sum_{j>=1} w_j (G_j + G_j')] /
//!   (T_v (T_v - 1))` with `G_j = sum_{s>j} e_s e_{s-j}'` and
//!   `w_j = 1 - j/(bandwidth + 1)` (linearmodels `cov_type="kernel"`,
//!   `kernel="bartlett"`; it truncates `int(bandwidth)`, so the two agree
//!   for integer bandwidths). `bandwidth + 1 > T_v` is refused, as
//!   linearmodels refuses it.
//!
//! **Caveats.** The standard error treats the `beta_s` as independent
//! draws: persistent firm effects make them correlated across periods and
//! the Fama-MacBeth SE too small (Petersen 2009), and the HAC variant does
//! not repair that — the correlation is the same at every lag, so a
//! truncated kernel cannot capture it (measured on the panel model card);
//! with generated regressors (estimated betas in asset pricing) there is no
//! Shanken (1992) errors-in-variables correction.

use tsecon_hac::Kernel;
use tsecon_linalg::faer::linalg::solvers::SolveLstsq;
use tsecon_linalg::faer::Mat;

use crate::data::PanelData;
use crate::error::PanelError;
use crate::pooled::{
    check_finite_cov, design_names, n_params, numerical_rank, refuse_duplicate_constant,
};

/// Covariance choice for [`FamaMacBethFit::inference`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FamaMacBethSe {
    /// `sd(beta_s, ddof = 1) / sqrt(T_v)` (the Fama-MacBeth SE).
    Standard,
    /// Newey-West (Bartlett) HAC on the `beta_s` series.
    Hac {
        /// Bartlett lag truncation: lags `1..=bandwidth` get weight
        /// `1 - j/(bandwidth + 1)`.
        bandwidth: f64,
    },
}

/// Standard errors, t-statistics and the covariance of a Fama-MacBeth fit.
#[derive(Debug, Clone)]
pub struct FamaMacBethInference {
    /// Which covariance produced this inference.
    pub se_type: FamaMacBethSe,
    /// Parameter covariance, `K x K`.
    pub cov: Mat<f64>,
    /// Standard errors `sqrt(diag(cov))`.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`.
    pub tvalues: Vec<f64>,
}

/// A fitted Fama-MacBeth regression; produced by [`fama_macbeth`].
#[derive(Debug, Clone)]
pub struct FamaMacBethFit {
    /// Time-series means of the per-period coefficients, `const` first
    /// when the constant is included.
    pub params: Vec<f64>,
    /// Parameter names aligned with `params`.
    pub names: Vec<String>,
    /// Observed cells (every period, skipped ones included).
    pub nobs: usize,
    /// Per-period coefficients, `T x K` in calendar order (`NaN` rows for
    /// skipped periods).
    pub params_per_period: Vec<Vec<f64>>,
    /// Whether each calendar period's cross-section was used.
    pub periods_used: Vec<bool>,
    /// Number of periods used `T_v`.
    pub n_periods_used: usize,
    /// Whether the constant was included.
    pub constant: bool,
    /// `beta_s - beta` over the compressed valid periods (`T_v x K`).
    deviations: Vec<Vec<f64>>,
}

/// Fits the Fama-MacBeth estimator (see the module docs).
///
/// # Errors
///
/// * [`PanelError::InsufficientData`] for an empty design (naming
///   `regressors`) or fewer than two periods whose cross-section
///   identifies every parameter (naming `regressors`);
/// * [`PanelError::RankDeficient`] for a regressor duplicating the
///   constant (naming `regressors`).
pub fn fama_macbeth(data: &PanelData, constant: bool) -> Result<FamaMacBethFit, PanelError> {
    let k = data.n_regressors();
    let kk = n_params(k, constant, "panel_fama_macbeth")?;
    let cells = data.observed_cells();
    refuse_duplicate_constant(data, &cells, constant)?;
    let off = usize::from(constant);
    let (n_ent, t_len) = (data.n_entities(), data.n_periods());
    let outcome = data.outcome();
    let mut params_per_period = vec![vec![f64::NAN; kk]; t_len];
    let mut periods_used = vec![false; t_len];
    let mut valid: Vec<Vec<f64>> = Vec::new();
    for t in 0..t_len {
        let rows: Vec<usize> = (0..n_ent).filter(|&i| data.observed(i, t)).collect();
        let nt = rows.len();
        if nt < kk {
            continue;
        }
        let x = Mat::from_fn(nt, kk, |r, j| {
            if j < off {
                1.0
            } else {
                data.regressor(j - off).map_or(0.0, |m| m[(rows[r], t)])
            }
        });
        if numerical_rank(&x) != Some(kk) {
            continue;
        }
        let y = Mat::from_fn(nt, 1, |r, _| outcome[(rows[r], t)]);
        let b = x.qr().solve_lstsq(&y);
        let beta: Vec<f64> = (0..kk).map(|j| b[(j, 0)]).collect();
        if beta.iter().any(|v| !v.is_finite()) {
            return Err(PanelError::InsufficientData {
                what: format!(
                    "outcome: the period-{t} cross-section regression overflowed to a \
                     non-finite coefficient; rescale the outcome and regressors"
                ),
            });
        }
        params_per_period[t].clone_from(&beta);
        periods_used[t] = true;
        valid.push(beta);
    }
    let tv = valid.len();
    if tv < 2 {
        return Err(PanelError::InsufficientData {
            what: format!(
                "regressors: Fama-MacBeth needs at least 2 periods whose cross-section \
                 identifies all K = {kk} parameters (n_t >= K observed entities and a \
                 full-rank design), but only {tv} of the T = {t_len} periods qualify; a \
                 regressor common to every entity within a period is collinear with \
                 the constant in each cross-section — check the mask, or drop such \
                 regressors"
            ),
        });
    }
    let tvf = tv as f64;
    let params: Vec<f64> = (0..kk)
        .map(|j| valid.iter().map(|b| b[j]).sum::<f64>() / tvf)
        .collect();
    if params.iter().any(|v| !v.is_finite()) {
        return Err(crate::pooled::overflow_error());
    }
    let deviations: Vec<Vec<f64>> = valid
        .iter()
        .map(|b| b.iter().zip(&params).map(|(v, m)| v - m).collect())
        .collect();
    Ok(FamaMacBethFit {
        params,
        names: design_names(data.names(), constant),
        nobs: cells.len(),
        params_per_period,
        periods_used,
        n_periods_used: tv,
        constant,
        deviations,
    })
}

impl FamaMacBethFit {
    /// Standard errors, t-statistics and the covariance (see the module
    /// docs).
    ///
    /// # Errors
    ///
    /// [`PanelError::KernelBandwidth`] for a negative or non-finite HAC
    /// bandwidth, [`PanelError::InsufficientData`] (naming `bandwidth`)
    /// when `bandwidth + 1` exceeds the number of periods used.
    pub fn inference(&self, se_type: FamaMacBethSe) -> Result<FamaMacBethInference, PanelError> {
        let kk = self.params.len();
        let tv = self.n_periods_used;
        let e = &self.deviations;
        let mut s = Mat::<f64>::zeros(kk, kk);
        for row in e {
            for a in 0..kk {
                for b in 0..kk {
                    s[(a, b)] += row[a] * row[b];
                }
            }
        }
        if let FamaMacBethSe::Hac { bandwidth } = se_type {
            if !bandwidth.is_finite() || bandwidth < 0.0 {
                return Err(PanelError::KernelBandwidth {
                    what: "the Newey-West (Bartlett) HAC on the Fama-MacBeth beta_t series",
                    value: bandwidth,
                });
            }
            if bandwidth + 1.0 > tv as f64 {
                return Err(PanelError::InsufficientData {
                    what: format!(
                        "bandwidth: the Newey-West (Bartlett) HAC on the Fama-MacBeth \
                         beta_t series needs bandwidth + 1 <= T_v, the number of periods \
                         used; got bandwidth = {bandwidth} with T_v = {tv} (pass a \
                         smaller bandwidth, or se_type=\"fama_macbeth\")"
                    ),
                });
            }
            let kernel = Kernel::Bartlett;
            for lag in 1..tv {
                let w = kernel.weight(lag, bandwidth);
                if w == 0.0 {
                    break; // Bartlett truncates.
                }
                for t in lag..tv {
                    for a in 0..kk {
                        for b in 0..kk {
                            let g = e[t][a] * e[t - lag][b];
                            s[(a, b)] += w * g;
                            s[(b, a)] += w * g;
                        }
                    }
                }
            }
        }
        let denom = (tv * (tv - 1)) as f64;
        let cov = Mat::from_fn(kk, kk, |a, b| s[(a, b)] / denom);
        check_finite_cov(&cov)?;
        let bse: Vec<f64> = (0..kk).map(|i| cov[(i, i)].max(0.0).sqrt()).collect();
        let tvalues = self.params.iter().zip(&bse).map(|(p, s)| p / s).collect();
        Ok(FamaMacBethInference {
            se_type,
            cov,
            bse,
            tvalues,
        })
    }
}
