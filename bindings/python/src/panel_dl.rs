//! Python binding for the distributed-lag panel regression of
//! `tsecon-panel` (`panel_distributed_lag`): the climate-impact
//! specification of Dell-Jones-Olken (2012) and Burke-Hsiang-Miguel
//! (2015). Registered into `_core` through [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tsecon_panel::{DistributedLagConfig, FixedEffects};

use crate::{panel_se, to_py};

/// Distributed-lag panel regression — the climate-impact specification
/// of Dell-Jones-Olken (2012, AEJ:Macro) and Burke-Hsiang-Miguel (2015,
/// Nature):
///
/// `y_it = sum_{l=0..L} beta_l x_{i,t-l} [+ sum_l gamma_l x^2_{i,t-l}]
/// + alpha_i + delta_t [+ g_i t] + e_it`
///
/// `outcome` is `N x T`; `regressors` is `k x N x T` (weather variables:
/// strictly exogenous, no lagged outcome — a lagged dependent variable
/// would put Nickell bias back into the within estimator; use
/// `panel_lp` with a bias correction for dynamic panels). `lags` is `L`;
/// lags `0..L` of every regressor enter and the first `L` periods of
/// each entity are dropped so the panel stays balanced (unbalanced
/// panels and NaN are refused). `powers=1` is the linear response,
/// `powers=2` adds the lags of the square (the BHM quadratic response).
/// `entity_effects` (True), `time_effects` (True) and `entity_trends`
/// (False; requires entity effects) choose the fixed effects; at least
/// one effect is required. `se_type` is "nonrobust", "cluster" (by
/// entity — the DJO default) or "driscoll_kraay" (the BHM robustness
/// choice; needs a long T). `bandwidth` is the Driscoll-Kraay lag
/// truncation and acts ONLY under `se_type="driscoll_kraay"` (4.0 when
/// omitted there); passing it explicitly with any other `se_type`
/// **raises** instead of being silently absorbed. `eval_points`
/// (1-D, default None) are the points at which the marginal effect of
/// the cumulative quadratic response is evaluated for every regressor;
/// it acts ONLY under `powers=2` (None there means each regressor's
/// pooled sample mean) and passing it under `powers=1` **raises** —
/// the linear cumulative response has one constant marginal effect,
/// `cumulative_effect` itself.
///
/// Design columns are ordered regressor-major, then power, then lag
/// (`names` lists them, e.g. `x0_L0`, `x0_L1`, `x0^2_L0`, ...). Returned
/// keys: `params`, `names`, `bse`, `tvalues`, `cov` (K x K nested
/// lists), `lag_effects` and `lag_se` (`[regressor][power-1][lag]`),
/// `cumulative_effect`, `cumulative_se` (delta method, `sqrt(1' V 1)`),
/// `cumulative_ci_low`, `cumulative_ci_high` (normal 95%,
/// `[regressor][power-1]`), and under `powers=2` `eval_points`,
/// `marginal_effect`, `marginal_se` (`[regressor][point]`; the
/// marginal effect `B_1 + 2 B_2 x` of the cumulative response),
/// `turning_point`, `turning_point_se` (`[regressor]`; `-B_1/(2 B_2)`,
/// NaN when `B_2` is exactly zero) — all five are None under `powers=1`;
/// plus `nobs` (`N (T - L)`), `n_entities`, `n_periods_used` (`T - L`),
/// `lags`, `powers`, `df_resid`, `se_type`, `entity_effects`,
/// `time_effects`, `entity_trends`.
///
/// Validation (fixtures/panel_dl.json): nine cases x three covariance
/// estimators pinned at 1e-10 against linearmodels PanelOLS (slopes,
/// SEs, t-statistics, full covariance; the trends variant via explicit
/// entity x trend regressors), the delta-method quantities against the
/// documented NumPy transcription at 1e-10; the cumulative-effect
/// interval's coverage is measured in seeded Monte Carlo and quoted on
/// the panel model card. The `lags=0`, `time_effects=False` call is
/// bit-identical to `panel_fe`.
#[pyfunction]
#[pyo3(signature = (outcome, regressors, lags, powers = 1, entity_effects = true, time_effects = true, entity_trends = false, se_type = "cluster", bandwidth = None, eval_points = None))]
#[allow(clippy::too_many_arguments)]
fn panel_distributed_lag<'py>(
    py: Python<'py>,
    outcome: PyReadonlyArray2<'py, f64>,
    regressors: PyReadonlyArray3<'py, f64>,
    lags: i64,
    powers: i64,
    entity_effects: bool,
    time_effects: bool,
    entity_trends: bool,
    se_type: &str,
    bandwidth: Option<f64>,
    eval_points: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    if lags < 0 {
        return Err(PyValueError::new_err(format!(
            "lags must be a non-negative integer (the number of lags L of each \
             regressor; 0 is the contemporaneous regression); got {lags}"
        )));
    }
    if powers != 1 && powers != 2 {
        return Err(PyValueError::new_err(format!(
            "powers must be 1 (linear response: lags of x) or 2 (quadratic response: \
             lags of x and of x^2, the Burke-Hsiang-Miguel form); got {powers}"
        )));
    }
    if powers == 1 && eval_points.is_some() {
        return Err(PyValueError::new_err(
            "eval_points was given but powers=1 ignores it: the marginal effect of a \
             linear cumulative response is the constant cumulative_effect itself, so \
             there is nothing to evaluate at a point; pass powers=2 for the quadratic \
             response or drop eval_points",
        ));
    }
    if entity_trends && !entity_effects {
        return Err(PyValueError::new_err(
            "entity_trends=True requires entity_effects=True: an entity-specific linear \
             trend needs its own intercept, otherwise the trend lines are forced through \
             a common origin; pass entity_effects=True",
        ));
    }
    if !entity_effects && !time_effects {
        return Err(PyValueError::new_err(
            "entity_effects=False with time_effects=False leaves no fixed effects: the \
             within estimator then reduces to pooled OLS without a constant, which is not \
             a panel model — request entity and/or time effects, or run tsecon.ols on the \
             stacked data",
        ));
    }
    let o = outcome.as_array();
    let outcome_m = Mat::from_fn(o.nrows(), o.ncols(), |i, j| o[(i, j)]);
    let r = regressors.as_array();
    let (k, n, t) = (r.shape()[0], r.shape()[1], r.shape()[2]);
    let regs: Vec<(String, Mat<f64>)> = (0..k)
        .map(|c| (format!("x{c}"), Mat::from_fn(n, t, |i, j| r[[c, i, j]])))
        .collect();
    let data = tsecon_panel::PanelData::balanced(outcome_m, regs).map_err(to_py)?;
    let cfg = DistributedLagConfig {
        lags: lags as usize,
        powers: powers as usize,
        effects: FixedEffects {
            entity: entity_effects,
            time: time_effects,
            entity_trends,
        },
        se_type: panel_se("panel_distributed_lag", se_type, bandwidth)?,
        eval_points: eval_points.map(|p| p.as_array().to_vec()),
    };
    let res = tsecon_panel::panel_distributed_lag(&data, &cfg).map_err(to_py)?;
    let kk = res.params.len();
    let cov: Vec<Vec<f64>> = (0..kk)
        .map(|a| (0..kk).map(|b| res.cov[(a, b)]).collect())
        .collect();
    let d = PyDict::new(py);
    d.set_item("params", res.params.into_pyarray(py))?;
    d.set_item("names", res.names)?;
    d.set_item("bse", res.bse.into_pyarray(py))?;
    d.set_item("tvalues", res.tvalues.into_pyarray(py))?;
    d.set_item("cov", cov)?;
    d.set_item("lag_effects", res.lag_effects)?;
    d.set_item("lag_se", res.lag_se)?;
    d.set_item("cumulative_effect", res.cumulative_effect)?;
    d.set_item("cumulative_se", res.cumulative_se)?;
    d.set_item("cumulative_ci_low", res.cumulative_ci_low)?;
    d.set_item("cumulative_ci_high", res.cumulative_ci_high)?;
    d.set_item("eval_points", res.eval_points)?;
    d.set_item("marginal_effect", res.marginal_effect)?;
    d.set_item("marginal_se", res.marginal_se)?;
    d.set_item("turning_point", res.turning_point)?;
    d.set_item("turning_point_se", res.turning_point_se)?;
    d.set_item("nobs", res.nobs)?;
    d.set_item("n_entities", res.n_entities)?;
    d.set_item("n_periods_used", res.n_periods_used)?;
    d.set_item("lags", res.lags)?;
    d.set_item("powers", res.powers)?;
    d.set_item("df_resid", res.df_resid)?;
    d.set_item("se_type", se_type)?;
    d.set_item("entity_effects", entity_effects)?;
    d.set_item("time_effects", time_effects)?;
    d.set_item("entity_trends", entity_trends)?;
    Ok(d)
}

/// Adds the module's functions to the `_core` extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(panel_distributed_lag, m)?)?;
    Ok(())
}
