//! Shared machinery of the static-panel menu ([`crate::panel_random_effects`],
//! [`crate::panel_between`], [`crate::panel_first_difference`]): the
//! full-rank pooled least-squares solve on a stacked design, its covariance
//! menu ([`StaticSeType`]), and the rank-revealing residual projection the
//! random-effects auxiliary regressions need.
//!
//! The covariance conventions are linearmodels 7.0's for `RandomEffects`,
//! `BetweenOLS`, `FirstDifferenceOLS` and `PooledOLS` with `debiased=True`
//! (the default): with `n` stacked rows, `K` parameters (the constant
//! counted), residuals `e` and the final design `X`, bread
//! `B = (X'X)^{-1}`,
//!
//! * [`StaticSeType::NonRobust`] — `s2 B`, `s2 = e'e / (n - K)`;
//! * [`StaticSeType::Hc1`] — `n/(n - K) B [sum_r x_r x_r' e_r^2] B` (the
//!   White sandwich with the MacKinnon-White `n/(n - K)` factor: the house
//!   `"hc1"` of `ols`; linearmodels `cov_type="robust"`);
//! * [`StaticSeType::ClusterEntity`] — `n/(n - K) B [sum_i g_i g_i'] B`,
//!   `g_i = sum_t x_it e_it`, with no `G/(G - 1)` group factor
//!   (linearmodels `group_debias=False`);
//! * [`StaticSeType::DriscollKraay`] — `n/(n - K) B S B` with `S` the
//!   Bartlett-kernel HAC of the per-period score sums over the **calendar**
//!   periods of the layout (a period nobody observes contributes a zero
//!   score and still counts as elapsed time — linearmodels instead lags over
//!   the periods present in the data; the two agree whenever every period
//!   carries an observation, which the golden fixtures keep).
//!
//! The nonrobust, cluster and Driscoll-Kraay sandwiches are the panel-FE
//! code itself: the fit is stored as a `WithinFit` with no absorbed
//! effects, so `WithinFit::inference` (whose arithmetic is shared with
//! `panel_fe`) reduces exactly to the formulas above.

use tsecon_linalg::faer::linalg::solvers::{DenseSolveCore, SolveLstsq};
use tsecon_linalg::faer::{Mat, MatRef, Side};

use crate::error::PanelError;
use crate::fe::{PanelSeType, WithinFit};

/// Covariance-estimator choice for the static-panel menu (random effects,
/// between, first difference), mirroring linearmodels'
/// `fit(cov_type=...)` with `debiased=True`. See the module docs of
/// `pooled.rs` for the formulas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StaticSeType {
    /// Classical `s2 (X'X)^{-1}` with `s2 = e'e / (n - K)`
    /// (linearmodels `cov_type="unadjusted"`).
    NonRobust,
    /// White heteroskedasticity-robust sandwich with the `n/(n - K)`
    /// factor (house `"hc1"`; linearmodels `cov_type="robust"`).
    Hc1,
    /// One-way cluster-robust sandwich by entity with the `n/(n - K)`
    /// factor and no `G/(G - 1)` (linearmodels `cov_type="clustered"`,
    /// `cluster_entity=True`).
    ClusterEntity,
    /// Driscoll-Kraay (1998) Bartlett-kernel HAC on per-period score sums
    /// over the calendar periods (linearmodels `cov_type="kernel"`,
    /// `kernel="bartlett"`).
    DriscollKraay {
        /// Bartlett lag truncation: lags `1..=bandwidth` get weight
        /// `1 - j/(bandwidth + 1)`.
        bandwidth: f64,
    },
}

/// Standard errors, t-statistics and the full parameter covariance of a
/// static-panel fit under one [`StaticSeType`].
#[derive(Debug, Clone)]
pub struct StaticInference {
    /// Which covariance estimator produced this inference.
    pub se_type: StaticSeType,
    /// Parameter covariance, `K x K` (the constant first when present).
    pub cov: Mat<f64>,
    /// Standard errors `sqrt(diag(cov))`.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`.
    pub tvalues: Vec<f64>,
}

/// The stacked rows of a pooled regression: outcome, design columns, and
/// the entity / calendar period of every row (entity-major, so each
/// entity's rows are contiguous).
pub(crate) struct StackedRows {
    pub(crate) y: Vec<f64>,
    pub(crate) x: Mat<f64>,
    pub(crate) row_entity: Vec<usize>,
    pub(crate) row_period: Vec<usize>,
    pub(crate) n_periods: usize,
}

/// Full-rank OLS of `y` on `x` with the numpy/linearmodels singular-value
/// rank criterion `sigma_min <= sigma_max * max(n, K) * eps` (the guard
/// `panel_fe` uses), the bread by Cholesky inverse and the coefficients by
/// Householder QR — the same solver idiom as the within estimator. The fit
/// is stored as a `WithinFit` with no absorbed effects so the covariance
/// menu is the panel-FE code.
///
/// `rank_msg` is the refusal text for a rank-deficient design; it must
/// start with the parameter name.
pub(crate) fn pooled_ols(rows: StackedRows, rank_msg: &str) -> Result<WithinFit, PanelError> {
    let StackedRows {
        y,
        x,
        row_entity,
        row_period,
        n_periods,
    } = rows;
    let n = y.len();
    let k = x.ncols();
    debug_assert_eq!(x.nrows(), n);
    // Callers guard n > K with a parameter-naming message; this is the
    // backstop that keeps the usize arithmetic of the covariance safe.
    if k == 0 || n <= k {
        return Err(PanelError::InsufficientData {
            what: format!(
                "regressors: the pooled regression has {n} rows for K = {k} parameters \
                 and needs more rows than parameters"
            ),
        });
    }
    let rank_err = || PanelError::RankDeficient {
        what: rank_msg.to_string(),
    };
    let svd = x.thin_svd().map_err(|_| rank_err())?;
    let sv: Vec<f64> = svd.S().column_vector().iter().copied().collect();
    let smax = sv.iter().copied().fold(0.0_f64, f64::max);
    let tol = smax * (n.max(k) as f64) * f64::EPSILON;
    if smax.is_nan() || smax <= 0.0 || sv.iter().any(|&v| v <= tol) {
        return Err(rank_err());
    }
    let xtx = x.transpose() * &x;
    check_finite_matrix(&xtx)?;
    let xtx_inv = xtx.llt(Side::Lower).map_err(|_| rank_err())?.inverse();
    let ymat = Mat::from_fn(n, 1, |r, _| y[r]);
    let params_mat = x.qr().solve_lstsq(&ymat);
    let params: Vec<f64> = (0..k).map(|j| params_mat[(j, 0)]).collect();
    let fitted = &x * &params_mat;
    let resid: Vec<f64> = (0..n).map(|r| y[r] - fitted[(r, 0)]).collect();
    if params.iter().any(|v| !v.is_finite()) {
        return Err(scale_error());
    }
    check_sum_of_squares(&resid)?;
    Ok(WithinFit {
        params,
        resid,
        xd: x,
        xtx_inv,
        nobs: n,
        nparams: k,
        n_periods,
        n_effects: 0,
        n_trend_params: 0,
        count_effects_in_cluster: true,
        row_entity,
        row_period,
    })
}

/// The covariance menu on a pooled fit (see the module docs). The
/// nonrobust, cluster and Driscoll-Kraay estimators are
/// `WithinFit::inference` with no absorbed effects; HC1 is the White
/// sandwich times `n/(n - K)`.
///
/// # Errors
///
/// [`PanelError::InvalidBandwidth`] for a negative or non-finite
/// Driscoll-Kraay bandwidth; [`PanelError::InsufficientData`] (naming
/// `se_type`) for an entity-clustered covariance whose rows all belong to
/// one entity, or a Driscoll-Kraay covariance whose rows all fall in one
/// period.
pub(crate) fn pooled_inference(
    fit: &WithinFit,
    se_type: StaticSeType,
) -> Result<StaticInference, PanelError> {
    refuse_single_score_group(fit, se_type)?;
    let panel = match se_type {
        StaticSeType::NonRobust => Some(PanelSeType::NonRobust),
        StaticSeType::ClusterEntity => Some(PanelSeType::ClusterEntity),
        StaticSeType::DriscollKraay { bandwidth } => Some(PanelSeType::DriscollKraay { bandwidth }),
        StaticSeType::Hc1 => None,
    };
    if let Some(p) = panel {
        let inf = fit.inference(p)?;
        check_finite_matrix(&inf.cov)?;
        return Ok(StaticInference {
            se_type,
            cov: inf.cov,
            bse: inf.bse,
            tvalues: inf.tvalues,
        });
    }
    let (n, k) = (fit.nobs, fit.nparams);
    let mut meat = Mat::<f64>::zeros(k, k);
    for r in 0..n {
        let u2 = fit.resid[r] * fit.resid[r];
        for a in 0..k {
            let xa = fit.xd[(r, a)] * u2;
            for b in 0..k {
                meat[(a, b)] += xa * fit.xd[(r, b)];
            }
        }
    }
    let scale = n as f64 / (n - k) as f64;
    let sw = &fit.xtx_inv * &meat * &fit.xtx_inv;
    let cov = Mat::from_fn(k, k, |i, j| scale * sw[(i, j)]);
    check_finite_matrix(&cov)?;
    Ok(finish(se_type, cov, &fit.params))
}

/// Refuses an entity-clustered covariance whose rows all belong to one
/// entity, and a Driscoll-Kraay covariance whose rows all fall in one
/// period: the single group's score sum is then `X'e`, which the normal
/// equations make zero, so the covariance is rounding noise and every
/// t-statistic astronomically large (linearmodels returns those numbers;
/// `lp_did` refuses a single cluster the same way).
fn refuse_single_score_group(fit: &WithinFit, se_type: StaticSeType) -> Result<(), PanelError> {
    let one_group = |ids: &[usize]| ids.iter().all(|&g| Some(g) == ids.first().copied());
    let what = match se_type {
        StaticSeType::ClusterEntity if one_group(&fit.row_entity) => {
            "se_type: the entity-clustered covariance needs rows from at least two \
             entities, but every row of this fit belongs to one entity — the single \
             cluster's score sum is X'e = 0 by the normal equations, so the covariance \
             would be identically zero; use se_type=\"hc1\" or \"nonrobust\""
        }
        StaticSeType::DriscollKraay { .. } if one_group(&fit.row_period) => {
            "se_type: the Driscoll-Kraay covariance needs scores in at least two \
             periods, but every row of this fit falls in one period (first differences \
             of a two-period panel, say) — that period's score sum is X'e = 0 by the \
             normal equations, so the covariance would be identically zero; use \
             se_type=\"cluster\", \"hc1\" or \"nonrobust\""
        }
        _ => return Ok(()),
    };
    Err(PanelError::InsufficientData {
        what: what.to_string(),
    })
}

/// The refusal for sums of squares that leave the range of double
/// precision: data of magnitude ~1e154 and above square past `f64::MAX`,
/// and residuals of magnitude ~1e-154 and below square into the subnormal
/// range, where the variance estimates lose their digits or flush to zero.
/// Returning a clamped, NaN-laundered or zero standard error would be
/// silently wrong.
pub(crate) fn scale_error() -> PanelError {
    PanelError::InsufficientData {
        what: "outcome: the estimator's sums of squares left the range of double \
               precision (the outcome or regressors are of magnitude ~1e154 or more, so \
               their squares overflow, or the residuals are of magnitude ~1e-154 or \
               less, so their squares underflow); rescale the outcome and regressors"
            .to_string(),
    }
}

/// Refuses a square matrix with a non-finite entry (see [`scale_error`]).
pub(crate) fn check_finite_matrix(m: &Mat<f64>) -> Result<(), PanelError> {
    let k = m.nrows();
    if (0..k).any(|a| (0..k).any(|b| !m[(a, b)].is_finite())) {
        return Err(scale_error());
    }
    Ok(())
}

/// Refuses residuals whose sum of squares is non-finite, or underflows
/// below `f64::MIN_POSITIVE` while the residuals are not all zero (see
/// [`scale_error`]).
pub(crate) fn check_sum_of_squares(resid: &[f64]) -> Result<(), PanelError> {
    let ss: f64 = resid.iter().map(|e| e * e).sum();
    if !ss.is_finite() || (ss < f64::MIN_POSITIVE && resid.iter().any(|&e| e != 0.0)) {
        return Err(scale_error());
    }
    Ok(())
}

/// Refuses an outcome or a regressor whose sum of squares over the
/// observed cells leaves the normal range of double precision: it
/// overflows (magnitude ~1e154 or more) or, for a column that is not all
/// zero, underflows below `f64::MIN_POSITIVE` (magnitude ~1e-154 or less).
/// Every cross-product the static estimators form is bounded by these
/// sums, so checking them first keeps the problem from surfacing later as
/// a spurious rank, SVD or time-invariance refusal, or as a digit-less
/// standard error.
pub(crate) fn check_data_scale(
    data: &crate::data::PanelData,
    cells: &[(usize, usize)],
) -> Result<(), PanelError> {
    let out_of_range = |m: MatRef<'_, f64>| {
        let ss: f64 = cells.iter().map(|&c| m[c] * m[c]).sum();
        !ss.is_finite() || (ss < f64::MIN_POSITIVE && cells.iter().any(|&c| m[c] != 0.0))
    };
    if out_of_range(data.outcome()) {
        return Err(scale_error());
    }
    for j in 0..data.n_regressors() {
        if let Some(m) = data.regressor(j) {
            if out_of_range(m) {
                return Err(PanelError::InsufficientData {
                    what: format!(
                        "regressors: column {j} ({}) is of magnitude ~1e154 or more (or \
                         ~1e-154 or less) over the observed cells, so its sum of squares \
                         overflows (or underflows) double precision; rescale it",
                        data.names()[j]
                    ),
                });
            }
        }
    }
    Ok(())
}

/// `bse = sqrt(max(diag, 0))` and `tvalues = params / bse`.
pub(crate) fn finish(se_type: StaticSeType, cov: Mat<f64>, params: &[f64]) -> StaticInference {
    let k = params.len();
    let bse: Vec<f64> = (0..k).map(|i| cov[(i, i)].max(0.0).sqrt()).collect();
    let tvalues = params.iter().zip(&bse).map(|(p, s)| p / s).collect();
    StaticInference {
        se_type,
        cov,
        bse,
        tvalues,
    }
}

/// Numerical rank of `x` under the numpy/linearmodels singular-value
/// criterion `sigma > sigma_max * max(n, K) * eps` (0 for an all-zero or
/// empty matrix). `None` if the decomposition fails.
pub(crate) fn numerical_rank(x: &Mat<f64>) -> Option<usize> {
    let (n, k) = (x.nrows(), x.ncols());
    if n == 0 || k == 0 {
        return Some(0);
    }
    let sv: Vec<f64> = if n >= k {
        let svd = x.thin_svd().ok()?;
        svd.S().column_vector().iter().copied().collect()
    } else {
        let xt = x.transpose().to_owned();
        let svd = xt.thin_svd().ok()?;
        svd.S().column_vector().iter().copied().collect()
    };
    let smax = sv.iter().copied().fold(0.0_f64, f64::max);
    if smax.is_nan() || smax <= 0.0 {
        return Some(0);
    }
    let tol = smax * (n.max(k) as f64) * f64::EPSILON;
    Some(sv.iter().filter(|&&v| v > tol).count())
}

/// The residual of the minimum-norm least-squares fit of `y` on `x`
/// (`n >= K` rows), rank-revealing: `y - U_r U_r' y` over the left singular
/// vectors whose singular values pass the numpy/linearmodels criterion
/// `sigma > sigma_max * max(n, K) * eps`. Columns the transformation has
/// annihilated (the demeaned constant, a time-invariant regressor in the
/// within step) and an exactly collinear between design (time dummies on
/// an unbalanced panel) are therefore legal here: the residual is the
/// projection off the column span, which every least-squares solution
/// shares. linearmodels uses LAPACK `gelsy` (rank by a column-pivoted QR
/// condition estimate at `cond = max(n, K) * eps`); the two rank decisions
/// agree away from the numerical boundary.
pub(crate) fn residual_projection(
    y: &[f64],
    x: &Mat<f64>,
    what: &'static str,
) -> Result<Vec<f64>, PanelError> {
    let n = y.len();
    let k = x.ncols();
    debug_assert_eq!(x.nrows(), n);
    if k == 0 {
        return Ok(y.to_vec());
    }
    let svd = x.thin_svd().map_err(|_| PanelError::RankDeficient {
        what: format!("{what}: the singular-value decomposition of the design failed"),
    })?;
    let u = svd.U();
    let sv: Vec<f64> = svd.S().column_vector().iter().copied().collect();
    let smax = sv.iter().copied().fold(0.0_f64, f64::max);
    let tol = smax * (n.max(k) as f64) * f64::EPSILON;
    let mut out = y.to_vec();
    if smax.is_nan() || smax <= 0.0 {
        return Ok(out);
    }
    for (c, &s) in sv.iter().enumerate() {
        if s <= tol {
            continue;
        }
        let mut coef = 0.0;
        for (r, v) in y.iter().enumerate() {
            coef += u[(r, c)] * v;
        }
        for (r, o) in out.iter_mut().enumerate() {
            *o -= u[(r, c)] * coef;
        }
    }
    Ok(out)
}

/// `K = k + constant`, refusing an empty design (`k = 0` without the
/// constant) by naming `regressors`.
pub(crate) fn n_params(k: usize, constant: bool, caller: &str) -> Result<usize, PanelError> {
    let kk = k + usize::from(constant);
    if kk == 0 {
        return Err(PanelError::InsufficientData {
            what: format!(
                "regressors: {caller} got an empty design (no regressors and \
                 constant=False); pass at least one regressor, or constant=True for \
                 the intercept-only model"
            ),
        });
    }
    Ok(kk)
}

/// Design-column names: `const` first when the constant is included, then
/// the regressor names.
pub(crate) fn design_names(names: &[String], constant: bool) -> Vec<String> {
    let mut out = Vec::with_capacity(names.len() + usize::from(constant));
    if constant {
        out.push("const".to_string());
    }
    out.extend(names.iter().cloned());
    out
}

/// Refuses a regressor that is constant over every observed cell when the
/// estimator also adds the intercept (`constant=true`): the column
/// duplicates the constant, so the design is collinear.
pub(crate) fn refuse_duplicate_constant(
    data: &crate::data::PanelData,
    cells: &[(usize, usize)],
    constant: bool,
) -> Result<(), PanelError> {
    if !constant || cells.is_empty() {
        return Ok(());
    }
    for j in 0..data.n_regressors() {
        if let Some(m) = data.regressor(j) {
            let first = m[cells[0]];
            if cells.iter().all(|&c| m[c] == first) {
                return Err(PanelError::RankDeficient {
                    what: format!(
                        "regressors: column {j} ({}) is constant over every observed \
                         cell, so it duplicates the intercept that constant=True adds \
                         and the design is collinear; drop the column, or pass \
                         constant=False to keep it as the intercept",
                        data.names()[j]
                    ),
                });
            }
        }
    }
    Ok(())
}
