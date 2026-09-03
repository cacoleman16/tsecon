//! Structured penalties and post-selection: `group_lasso` (group / sparse-
//! group LASSO), `post_lasso` (post-LASSO OLS refit, no standard errors by
//! design) and `pds_lasso` (post-double-selection with Newey-West HAC
//! inference). Thin marshalling over `tsecon_ml`; every numeric convention
//! is documented on the Rust side and restated in the docstrings below.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{to_faer, to_py, vec1};

/// Parse `group_weights`: `"sqrt_size"`, `"none"`, or one positive weight
/// per distinct group label (ascending label order).
fn parse_group_weights(obj: &Bound<'_, PyAny>) -> PyResult<tsecon_ml::GroupWeights> {
    if let Ok(s) = obj.extract::<String>() {
        return match s.as_str() {
            "sqrt_size" => Ok(tsecon_ml::GroupWeights::SqrtSize),
            "none" => Ok(tsecon_ml::GroupWeights::Uniform),
            other => Err(PyValueError::new_err(format!(
                "group_weights: unknown option {other:?}; accepted values are \
                 \"sqrt_size\" (w_g = sqrt(group size), the Yuan-Lin default), \
                 \"none\" (w_g = 1), or an array with one positive weight per \
                 distinct group label in ascending label order"
            ))),
        };
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(tsecon_ml::GroupWeights::Custom(vec1(&arr)));
    }
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(tsecon_ml::GroupWeights::Custom(v));
    }
    Err(PyValueError::new_err(
        "group_weights must be \"sqrt_size\", \"none\", or a 1-D float array with one \
         positive weight per distinct group label (ascending label order)",
    ))
}

/// Group LASSO (Yuan & Lin 2006) and sparse-group LASSO (Simon, Friedman,
/// Hastie & Tibshirani 2013) by block coordinate descent with exact
/// per-block Lipschitz constants and the two-level prox.
///
/// Objective (the `1/(2n)` data-fit scaling of the crate's `lasso`, so
/// `alpha` lives on scikit-learn's scale):
///
/// `(1/(2n))||y - Xb||^2 + alpha * [ (1 - l1_ratio) * sum_g w_g ||b_g||_2 + l1_ratio * ||b||_1 ]`.
///
/// `x` is the (n, p) design and `y` the length-n target — no intercept is
/// fitted and nothing is standardized inside: center `y` and standardize
/// the columns of `x` first, exactly as for `lasso`. `groups` gives one
/// integer label per column (any integers, contiguous or not; columns
/// sharing a label form a group — an integer array passes through the
/// coercion layer untouched). `alpha >= 0` is the penalty. `l1_ratio` in
/// [0, 1] is the within-group L1 share: `0.0` (default) is the group LASSO,
/// `1.0` is exactly `lasso(x, y, alpha)`, values between are the
/// sparse-group LASSO. `group_weights` is `"sqrt_size"` (`w_g =
/// sqrt(|g|)`, the Yuan-Lin default), `"none"` (`w_g = 1`), or an array
/// with one positive weight per distinct label in ascending label order.
/// `tol` is the dimensionless stopping tolerance shared with `lasso`
/// (`max_j |Δb_j| ||x_j|| <= tol ||y||`) and also bounds the returned KKT
/// residual relative to `max_j |x_j'y|/n`; `max_iter` caps the number of
/// block sweeps.
///
/// Returns `coef` (length p), `n_iter`, `converged` (True only when both the
/// sweep-change rule and the KKT certificate were met — when False the
/// last iterate is returned and `kkt_violation` says how far it is from
/// optimal), `active_groups` (labels with a nonzero block), `active_set`
/// (column indices with nonzero coefficients), `objective` (the value of
/// the objective above at `coef`), `kkt_violation` (the largest subgradient
/// Karush-Kuhn-Tucker residual at `coef`: for an inactive group
/// `||S(-grad_g, alpha*l1_ratio)||_2 - alpha*(1-l1_ratio)*w_g`, for a
/// nonzero coordinate `|grad_j + alpha*(1-l1_ratio)*w_g*b_j/||b_g|| +
/// alpha*l1_ratio*sign(b_j)|`, for a zero coordinate of an active group
/// `|grad_j| - alpha*l1_ratio`, all clipped at 0 — a self-certificate you
/// can read: the problem is convex, so a residual near zero proves the
/// answer is near the global optimum whatever the solver did),
/// `max_rel_change` (last full sweep's largest coefficient move relative to
/// the problem's scale), and `alpha_max` (the smallest `alpha` at which the
/// solution is identically zero for this design, `l1_ratio` and weights —
/// the top of a regularization path).
///
/// Validation: every fixture case is certified by an independent
/// evaluation of the KKT conditions (residual <= 1e-8 asserted, ~2e-13
/// achieved) — the rigorous grade for a convex problem — and cross-checked
/// against the independent `skglm` package (GroupLasso /
/// WeightedL1GroupL2, same 1/(2n) objective) at 1e-8 (~1.5e-12 achieved);
/// `l1_ratio=1` reproduces `lasso` (scikit-learn-pinned) and singleton
/// groups with `l1_ratio=0` reproduce it too. Fixture:
/// `fixtures/structured.json`.
#[pyfunction]
#[pyo3(signature = (x, y, groups, alpha, l1_ratio = 0.0, group_weights = None, tol = 1e-8, max_iter = 10000))]
#[allow(clippy::too_many_arguments)]
fn group_lasso<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    groups: Vec<i64>,
    alpha: f64,
    l1_ratio: f64,
    group_weights: Option<Bound<'py, PyAny>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let weights = match &group_weights {
        None => tsecon_ml::GroupWeights::SqrtSize,
        Some(obj) => parse_group_weights(obj)?,
    };
    let m = to_faer(&x);
    let opts = tsecon_ml::CoordDescentOptions { tol, max_iter };
    let fit = tsecon_ml::group_lasso(
        m.as_ref(),
        &vec1(&y),
        &groups,
        alpha,
        l1_ratio,
        &weights,
        opts,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("coef", fit.coef.into_pyarray(py))?;
    d.set_item("n_iter", fit.n_iter)?;
    d.set_item("converged", fit.converged)?;
    d.set_item("active_groups", fit.active_groups)?;
    d.set_item("active_set", fit.active_set)?;
    d.set_item("objective", fit.objective)?;
    d.set_item("kkt_violation", fit.kkt_violation)?;
    d.set_item("max_rel_change", fit.max_rel_change)?;
    d.set_item("alpha_max", fit.alpha_max)?;
    Ok(d)
}

/// Post-LASSO OLS (Belloni & Chernozhukov 2013): fit the LASSO (or the
/// elastic net for `l1_ratio < 1`) with the crate's scikit-learn objective
/// `(1/(2n))||y - Xb||^2 + alpha*l1_ratio*||b||_1 +
/// 0.5*alpha*(1-l1_ratio)*||b||^2`, take the nonzero support, and refit
/// ordinary least squares on those columns alone to remove the shrinkage
/// bias. `x` is the (n, p) design and `y` the target — no intercept, no
/// standardization inside (center `y`, standardize `x` first). `alpha`,
/// `l1_ratio`, `tol`, `max_iter` are exactly `elastic_net`'s.
///
/// Returns `support` (selected column indices, ascending), `coef_lasso`
/// (the first-stage coefficients, length p), `coef_ols` (the refit, length
/// p, exactly zero off-support; the minimum-norm least-squares solution on
/// the selected columns), `n_selected`, and `rss` (residual sum of squares
/// of the refit).
///
/// **No standard errors, deliberately.** Naive OLS standard errors on a
/// data-selected support are invalid: the selection event depends on the
/// same sample, so the refit's sampling distribution is not the textbook
/// one and `n - |S|` residual degrees of freedom overstate what is left
/// after searching over p columns. For valid inference on a
/// low-dimensional target coefficient use `pds_lasso`.
///
/// Validation: the refit matches scikit-learn
/// `LinearRegression(fit_intercept=False)` on the scikit-learn `Lasso` /
/// `ElasticNet` support at 1e-10 (~8e-15 achieved), support exact.
/// Fixture: `fixtures/structured.json`.
#[pyfunction]
#[pyo3(signature = (x, y, alpha, l1_ratio = 1.0, tol = 1e-8, max_iter = 100000))]
fn post_lasso<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    l1_ratio: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let m = to_faer(&x);
    let opts = tsecon_ml::CoordDescentOptions { tol, max_iter };
    let fit = tsecon_ml::post_lasso(m.as_ref(), &vec1(&y), alpha, l1_ratio, opts).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("support", fit.support)?;
    d.set_item("coef_lasso", fit.coef_lasso.into_pyarray(py))?;
    d.set_item("coef_ols", fit.coef_ols.into_pyarray(py))?;
    d.set_item("n_selected", fit.n_selected)?;
    d.set_item("rss", fit.rss)?;
    Ok(d)
}

/// Post-double-selection LASSO (Belloni, Chernozhukov & Hansen 2014) for
/// the coefficient on a treatment `d` with high-dimensional controls `x`,
/// with Newey-West HAC inference from the library's single HAC engine.
///
/// Steps: LASSO of `y` on `x` (support S_y); LASSO of `d` on `x` (support
/// S_d); OLS of `y` on `[d, x[:, S_y ∪ S_d]]`, reading the effect off `d`.
/// Selecting on the treatment equation too is what makes the interval
/// robust to first-stage selection mistakes — single selection on `y`
/// alone drops controls that matter for `d` but only modestly for `y`, and
/// the omitted-variable bias is of the same order as the standard error
/// (measured below). The treatment is never penalized. `y` and `d` are
/// length-n vectors and `x` the (n, p) control matrix (p > n allowed) — no
/// intercept is fitted and nothing is standardized inside: center `y` and
/// `d`, standardize `x`.
///
/// `alpha` is either a float applied to both LASSOs or `"bic"` (default):
/// the per-equation BIC minimizer along `lasso_path`'s default grid (100
/// log-spaced points from lambda_max down three decades, `BIC = n ln(RSS/n)
/// + ln(n) df`). `hac_lags` is the Bartlett lag truncation for the
/// Newey-West sandwich: `None` (default) resolves to the Newey-West rule
/// `floor(4 (n/100)^(2/9))`, a positive integer is used as given, and `0`
/// switches to the classical spherical-errors covariance
/// `sigma2 (X'X)^{-1}` (statsmodels `cov_type="nonrobust"`). The HAC
/// covariance carries the finite-sample factor `n/(n-k)` (statsmodels
/// `cov_type="HAC"`, `cov_kwds={"maxlags": hac_lags, "use_correction":
/// True}`; statsmodels' own default is `use_correction=False`, so pass it
/// explicitly to compare). `p_value` and `conf_int` use the standard normal
/// in both modes (statsmodels `use_t=False`, its HAC default; its nonrobust
/// default would use t(n-k)) because the post-double-selection theory is
/// asymptotic. `tol` and `max_iter` are the coordinate-descent controls of
/// the selection LASSOs.
///
/// Returns `coef` (the treatment effect), `se`, `t_stat`, `p_value`
/// (two-sided, normal), `conf_int` (95% `(lo, hi)`), `support_y`,
/// `support_d`, `union_support` (all ascending column indices),
/// `n_controls_selected`, `alpha_y`, `alpha_d` (the penalties actually
/// used), and `hac_lags_resolved` (the lag truncation actually used, 0 for
/// classical standard errors).
///
/// Validation: Monte-Carlo grade for the statistical claim — R `hdm` and
/// Stata `pdslasso` are not runnable in the reference environment — with a
/// seeded design (n = 400, p = 40 AR(1) controls, AR(1) errors in both
/// equations, four confounders that load strongly on `d` and weakly on
/// `y`) in `crates/tsecon-ml/tests/structured_properties.rs`; the PDS
/// interval's measured coverage and the single-selection interval's
/// measured undercoverage are quoted on the model card. Exact leg: with the
/// union forced to every control (tiny `alpha`) and with the BIC-selected
/// union, `coef`/`se`/`t_stat`/`conf_int` match statsmodels HAC and
/// nonrobust OLS at 1e-8 relative (~1e-14 achieved), `p_value` at 1e-12.
/// Fixture: `fixtures/structured.json`.
#[pyfunction]
#[pyo3(signature = (y, d, x, alpha = None, hac_lags = None, tol = 1e-8, max_iter = 100000))]
#[allow(clippy::too_many_arguments)]
fn pds_lasso<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    d: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    alpha: Option<Bound<'py, PyAny>>,
    hac_lags: Option<i64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let alpha = match &alpha {
        None => tsecon_ml::PdsAlpha::Bic,
        Some(obj) => {
            if let Ok(s) = obj.extract::<String>() {
                match s.as_str() {
                    "bic" => tsecon_ml::PdsAlpha::Bic,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "alpha: unknown option {other:?}; accepted values are \"bic\" \
                             (the per-equation BIC pick along the crate's lasso_path grid) \
                             or a non-negative float applied to both selection LASSOs"
                        )))
                    }
                }
            } else if let Ok(a) = obj.extract::<f64>() {
                tsecon_ml::PdsAlpha::Fixed(a)
            } else {
                return Err(PyValueError::new_err(
                    "alpha must be \"bic\" or a non-negative float applied to both \
                     selection LASSOs",
                ));
            }
        }
    };
    let hac_lags = match hac_lags {
        None => None,
        Some(l) if l < 0 => {
            return Err(PyValueError::new_err(format!(
                "hac_lags must be a non-negative integer (got {l}): pass hac_lags=None for \
                 the Newey-West rule floor(4 (n/100)^(2/9)), a positive lag truncation for \
                 Bartlett HAC, or hac_lags=0 for classical (non-robust) standard errors"
            )))
        }
        Some(l) => Some(l as usize),
    };
    let m = to_faer(&x);
    let opts = tsecon_ml::CoordDescentOptions { tol, max_iter };
    let fit = tsecon_ml::pds_lasso(&vec1(&y), &vec1(&d), m.as_ref(), alpha, hac_lags, opts)
        .map_err(to_py)?;
    let out = PyDict::new(py);
    out.set_item("coef", fit.coef)?;
    out.set_item("se", fit.se)?;
    out.set_item("t_stat", fit.t_stat)?;
    out.set_item("p_value", fit.p_value)?;
    out.set_item("conf_int", fit.conf_int)?;
    out.set_item("support_y", fit.support_y)?;
    out.set_item("support_d", fit.support_d)?;
    out.set_item("union_support", fit.union_support)?;
    out.set_item("n_controls_selected", fit.n_controls_selected)?;
    out.set_item("alpha_y", fit.alpha_y)?;
    out.set_item("alpha_d", fit.alpha_d)?;
    out.set_item("hac_lags_resolved", fit.hac_lags_resolved)?;
    Ok(out)
}

/// Registers the structured-penalty and post-selection functions on the
/// `_core` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(group_lasso, m)?)?;
    m.add_function(wrap_pyfunction!(post_lasso, m)?)?;
    m.add_function(wrap_pyfunction!(pds_lasso, m)?)?;
    Ok(())
}
