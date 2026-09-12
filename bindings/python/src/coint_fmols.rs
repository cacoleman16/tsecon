//! Python bindings for the single-equation cointegrating regressions of
//! `tsecon-coint`: `fmols` (Phillips-Hansen 1990 fully modified OLS),
//! `dols` (Stock-Watson 1993 dynamic OLS) and `ccr` (Park 1992 canonical
//! cointegrating regression). Registered into `_core` through
//! [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tsecon_coint::tsecon_hac::Kernel;
use tsecon_coint::{
    BandwidthRule, CointRegOptions, CointRegResult, CointTrend, DolsCovType, DolsIc, DolsOptions,
};

use crate::{to_py, vec1};

fn x_to_faer(x: &numpy::PyReadonlyArray2<'_, f64>) -> tsecon_coint::tsecon_linalg::faer::Mat<f64> {
    let a = x.as_array();
    tsecon_coint::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
}

/// A bool spelled the way the Python caller wrote it.
fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

fn parse_trend(code: &str, name: &str) -> PyResult<CointTrend> {
    CointTrend::parse(code).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown {name} {code:?}; expected \"n\" (no deterministics), \"c\" (constant), \
             \"ct\" (constant and trend) or \"ctt\" (constant, trend and quadratic trend)"
        ))
    })
}

fn parse_kernel(code: &str) -> PyResult<Kernel> {
    tsecon_coint::parse_kernel(code).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown kernel {code:?}; expected \"bartlett\", \"parzen\" or \
             \"quadratic-spectral\""
        ))
    })
}

/// The `bandwidth` / `bandwidth_rule` pair: the rule only acts when the
/// bandwidth is selected automatically, so passing both is refused.
fn parse_bandwidth_rule(
    bandwidth: Option<f64>,
    bandwidth_rule: Option<&str>,
) -> PyResult<BandwidthRule> {
    match bandwidth_rule {
        None => Ok(BandwidthRule::NeweyWest),
        Some(code) => {
            let rule = BandwidthRule::parse(code).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown bandwidth_rule {code:?}; expected \"newey-west\" (arch's \
                     automatic rule, the default) or \"andrews\" (the Andrews 1991 AR(1) \
                     plug-in)"
                ))
            })?;
            if let Some(b) = bandwidth {
                return Err(PyValueError::new_err(format!(
                    "bandwidth_rule = {code:?} was given together with an explicit bandwidth \
                     = {b}: the rule only chooses the bandwidth when bandwidth is None, so it \
                     would be inert; drop bandwidth_rule or pass bandwidth = None"
                )));
            }
            Ok(rule)
        }
    }
}

fn nested<'py>(py: Python<'py>, m: &[Vec<f64>]) -> PyResult<Bound<'py, PyList>> {
    let outer = PyList::empty(py);
    for row in m {
        outer.append(PyList::new(py, row.iter().copied())?)?;
    }
    Ok(outer)
}

fn coint_reg_dict<'py>(py: Python<'py>, r: CointRegResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("estimator", r.estimator.name())?;
    d.set_item("params", r.params.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    d.set_item("tvalues", r.tvalues.into_pyarray(py))?;
    d.set_item("pvalues", r.pvalues.into_pyarray(py))?;
    d.set_item("cov", nested(py, &r.cov)?)?;
    d.set_item("param_names", PyList::new(py, r.param_names)?)?;
    d.set_item("resid", r.resid.into_pyarray(py))?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("n_x", r.n_x)?;
    d.set_item("n_det", r.n_det)?;
    d.set_item("trend", r.trend.code())?;
    d.set_item("x_trend", r.x_trend.code())?;
    d.set_item("kernel", tsecon_coint::kernel_code(r.kernel))?;
    d.set_item("bandwidth", r.bandwidth)?;
    match r.bandwidth_rule {
        Some(rule) => d.set_item("bandwidth_rule", rule.code())?,
        None => d.set_item("bandwidth_rule", py.None())?,
    }
    d.set_item("force_int", r.force_int)?;
    d.set_item("diff", r.diff)?;
    d.set_item("df_adjust", r.df_adjust)?;
    d.set_item("long_run_variance", r.long_run_variance)?;
    d.set_item("omega", nested(py, &r.lrcov.omega)?)?;
    d.set_item("lambda", nested(py, &r.lrcov.lambda)?)?;
    d.set_item("sigma", nested(py, &r.lrcov.sigma)?)?;
    d.set_item("n_lags", r.lrcov.n_lags)?;
    d.set_item("rsquared", r.rsquared)?;
    d.set_item("rsquared_adj", r.rsquared_adj)?;
    d.set_item("ols_params", r.ols_params.into_pyarray(py))?;
    d.set_item("ols_se", r.ols_se.into_pyarray(py))?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
fn coint_reg_options(
    trend: &str,
    kernel: &str,
    bandwidth: Option<f64>,
    bandwidth_rule: Option<&str>,
    force_int: bool,
    df_adjust: bool,
    diff: Option<bool>,
    x_trend: Option<&str>,
) -> PyResult<CointRegOptions> {
    let trend = parse_trend(trend, "trend")?;
    let x_trend = x_trend.map(|c| parse_trend(c, "x_trend")).transpose()?;
    let effective = x_trend.unwrap_or(trend);
    if let Some(d) = diff {
        if effective.n_det() < 2 {
            return Err(PyValueError::new_err(format!(
                "diff = {} was given but the regressors are detrended with {} = \"{}\", which \
                 has no trend term: diff only decides whether the trend is removed from the \
                 differences or from the levels, so it is inert here; drop diff, or pass a \
                 trend (\"ct\" / \"ctt\") through trend or x_trend",
                py_bool(d),
                if x_trend.is_some() {
                    "x_trend"
                } else {
                    "trend"
                },
                effective.code()
            )));
        }
    }
    Ok(CointRegOptions {
        trend,
        x_trend,
        kernel: parse_kernel(kernel)?,
        bandwidth,
        bandwidth_rule: parse_bandwidth_rule(bandwidth, bandwidth_rule)?,
        force_int,
        diff: diff.unwrap_or(false),
        df_adjust,
    })
}

/// Phillips-Hansen (1990) fully modified OLS (FM-OLS) of one cointegrating
/// vector, with asymptotically valid (mixed-normal) inference.
///
/// `y` is the regressand (length T), `x` the (T, k) matrix of I(1)
/// regressors (do NOT add your own constant: deterministics come from
/// `trend`). The static OLS `y = x'beta + d'delta + e` is super-consistent
/// but its t-statistics are invalid — serial correlation in `e` and
/// correlation between `e` and the regressor innovations `dx` leave a
/// second-order bias and a nuisance-parameter limit. FM-OLS corrects the
/// regressand for endogeneity (`y+ = y - omega_12 Omega_22^-1 eta_2`) and
/// subtracts the serial-correlation bias `lambda+_12` from the moment
/// equations, using the kernel long-run covariance of the residual system
/// `eta = (OLS residual, detrended dx)`; the corrected estimator has
/// covariance `omega_1.2 (Z'Z)^-1` and standard-normal t-statistics.
///
/// `trend`: "n", "c" (default), "ct", "ctt" (constant, trend, quadratic
/// trend; the trend runs 1..T). `kernel`: "bartlett" (default), "parzen",
/// "quadratic-spectral". `bandwidth`: an explicit kernel bandwidth (>= 0;
/// Bartlett/Parzen weight lag j by k(j/(bandwidth+1)) for j <=
/// floor(bandwidth), quadratic spectral by k(j/bandwidth)), or None
/// (default) to select it by `bandwidth_rule`: "newey-west" (default;
/// arch's rule — the Newey-West 1994 plug-in on the unit-weighted sum of
/// the residual system with ceil(4 (T/100)^rate) pilot lags) or
/// "andrews" (the Andrews 1991 AR(1) parametric plug-in on the same
/// series). Passing `bandwidth_rule` together with an explicit `bandwidth`
/// RAISES (the rule would be inert). `force_int` (default True, as arch)
/// ceils the bandwidth — automatic or explicit; the automatic one is also
/// capped at T - 1. `df_adjust` (default False) scales the covariance by
/// (T-1)/(T-1-k). `x_trend` (default None = `trend`; must carry at least
/// the terms of `trend`) sets the deterministics the regressors are
/// detrended with before differencing; `diff` (default False) removes the
/// trend from the differences instead of the levels and RAISES when the
/// effective x_trend has no trend term (it would be inert).
///
/// Keys: `estimator`, `params` (x columns first, then the deterministics —
/// see `param_names`), `se`, `tvalues`, `pvalues` (two-sided normal),
/// `cov`, `param_names`, `resid` (length `nobs` = T, the full sample),
/// `nobs`, `n_x`, `n_det`, `trend`, `x_trend`, `kernel`, `bandwidth` (the
/// one actually used), `bandwidth_rule` (None when explicit), `force_int`,
/// `diff`, `df_adjust`, `long_run_variance` (`omega_1.2`, df-scaled),
/// `omega` / `lambda` / `sigma` (the (1+k)x(1+k) long-run, one-sided
/// long-run and short-run covariances of the residual system, nested
/// lists), `n_lags` (positive lags the window covered), `rsquared`,
/// `rsquared_adj`, `ols_params` and `ols_se` (the plain static OLS for
/// comparison — its SEs are NOT valid for inference).
///
/// Validation: arch 8.0 `FullyModifiedOLS` at 1e-10 across every trend,
/// kernel and option (fixtures/fmols.json); t-statistic size and
/// super-consistency measured by seeded Monte Carlo (model card).
///
/// Further arguments, with defaults: `trend` ("c"), `kernel` ("bartlett"),
/// `bandwidth` (None), `bandwidth_rule` (None: "newey-west"), `force_int`
/// (True), `df_adjust` (False), `diff` (None: False), `x_trend` (None:
/// `trend`).
#[pyfunction]
#[pyo3(signature = (y, x, trend = "c", kernel = "bartlett", bandwidth = None, bandwidth_rule = None, force_int = true, df_adjust = false, diff = None, x_trend = None))]
#[allow(clippy::too_many_arguments)]
fn fmols<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    trend: &str,
    kernel: &str,
    bandwidth: Option<f64>,
    bandwidth_rule: Option<&str>,
    force_int: bool,
    df_adjust: bool,
    diff: Option<bool>,
    x_trend: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = coint_reg_options(
        trend,
        kernel,
        bandwidth,
        bandwidth_rule,
        force_int,
        df_adjust,
        diff,
        x_trend,
    )?;
    let xm = x_to_faer(&x);
    let r = tsecon_coint::fmols(&vec1(&y), xm.as_ref(), &opts).map_err(to_py)?;
    coint_reg_dict(py, r)
}

/// Park (1992) canonical cointegrating regression (CCR) of one
/// cointegrating vector, with asymptotically valid inference.
///
/// Same inputs, options and keys as `fmols`. Where FM-OLS corrects the
/// regressand and the moment equations, CCR transforms the DATA: with
/// `Sigma`, `Lambda`, `Omega` the short-run, one-sided and two-sided
/// long-run covariances of the residual system `eta` and `beta_ols` the
/// static OLS coefficients, `x* = x - (Sigma^-1 Lambda_2)'eta` and `y* =
/// y - (Sigma^-1 Lambda_2 beta_ols + kappa)'eta` with `kappa = (0,
/// Omega_22^-1 omega_21)`, and `params` is the OLS of `y*` on `[x*, d]`
/// over t = 2..T, with covariance `omega_1.2 (Z*'Z*)^-1`. Asymptotically
/// equivalent to FM-OLS; the two differ in finite samples.
///
/// `df_adjust` scales the covariance by (T-1)/(T-1-k) as documented —
/// arch 8.0's `CanonicalCointegratingReg.fit` scales only `omega_11`
/// (an operator-precedence slip); everything else is arch-exact.
///
/// Keys: `estimator`, `params`, `se`, `tvalues`, `pvalues`, `cov`,
/// `param_names`, `resid`, `nobs`, `n_x`, `n_det`, `trend`, `x_trend`,
/// `kernel`, `bandwidth`, `bandwidth_rule`, `force_int`, `diff`,
/// `df_adjust`, `long_run_variance`, `omega`, `lambda`, `sigma`, `n_lags`,
/// `rsquared`, `rsquared_adj`, `ols_params`, `ols_se`.
///
/// Validation: arch 8.0 `CanonicalCointegratingReg` at 1e-10
/// (fixtures/fmols.json), the `df_adjust` scaling as documented.
///
/// Further arguments, with defaults: `trend` ("c"), `kernel` ("bartlett"),
/// `bandwidth` (None), `bandwidth_rule` (None: "newey-west"), `force_int`
/// (True), `df_adjust` (False), `diff` (None: False), `x_trend` (None:
/// `trend`).
#[pyfunction]
#[pyo3(signature = (y, x, trend = "c", kernel = "bartlett", bandwidth = None, bandwidth_rule = None, force_int = true, df_adjust = false, diff = None, x_trend = None))]
#[allow(clippy::too_many_arguments)]
fn ccr<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    trend: &str,
    kernel: &str,
    bandwidth: Option<f64>,
    bandwidth_rule: Option<&str>,
    force_int: bool,
    df_adjust: bool,
    diff: Option<bool>,
    x_trend: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = coint_reg_options(
        trend,
        kernel,
        bandwidth,
        bandwidth_rule,
        force_int,
        df_adjust,
        diff,
        x_trend,
    )?;
    let xm = x_to_faer(&x);
    let r = tsecon_coint::ccr(&vec1(&y), xm.as_ref(), &opts).map_err(to_py)?;
    coint_reg_dict(py, r)
}

/// Stock-Watson (1993) / Saikkonen (1991) dynamic OLS (DOLS) of one
/// cointegrating vector: the static regression augmented with `lags` lags
/// and `leads` leads of the regressor differences (the contemporaneous
/// difference is always included), so the augmented error is orthogonal
/// to the regressor innovations and OLS on the augmented design is
/// asymptotically mixed normal.
///
/// `y` (length T) and `x` (T, k) as for `fmols`; `trend` as there. The
/// regression runs over the T - 1 - lags - leads rows every term is
/// defined on, design `[x, deterministics, dx_{t-lags}, ..., dx_t, ...,
/// dx_{t+leads}]` (k columns per block; the trend runs 1..nobs over that
/// sample). `lags` / `leads` (default None) fix the counts; when either is
/// None it is chosen by minimising `ic` — "bic" (default), "aic" or
/// "hqic": `ln(RSS/nobs) + n_params c/nobs` — over 0..`max_lag` /
/// 0..`max_lead` (default None = ceil(12 (T/100)^(1/4))) on the COMMON
/// sample of the largest candidate (ties to the smaller lag, then lead),
/// the chosen model then refit on its own sample; `common` (default None
/// = False) restricts the search to lags == leads. The sentinel rule:
/// `ic` passed with both `lags` and `leads` fixed RAISES, `max_lag` passed
/// with `lags` fixed RAISES, `max_lead` with `leads` RAISES, and `common`
/// with both fixed RAISES (each would be inert). A search whose largest
/// candidate has no residual degrees of freedom is refused (arch runs it
/// underdetermined).
///
/// `cov_type`: "unadjusted" (default) — `sigma2_HAC (Z'Z/n)^-1 / n` with
/// `sigma2_HAC` the kernel long-run variance of the residuals; "robust" —
/// the kernel-HAC sandwich `(Z'Z/n)^-1 S_HAC (Z'Z/n)^-1 / n` on the scores.
/// `kernel`, `bandwidth`, `bandwidth_rule` and `force_int` (default False,
/// arch's DOLS default) as for `fmols`, the automatic bandwidth chosen on
/// the residuals ("unadjusted") or the scores ("robust"); `df_adjust`
/// (default False) scales the covariance by nobs/(nobs - n_params).
///
/// Keys: `params` (the cointegrating vector: x columns then the
/// deterministics — `param_names`), `se`, `tvalues`, `pvalues`, `cov`,
/// `param_names`, `full_params` / `full_se` / `full_cov` /
/// `full_param_names` (every coefficient incl. the difference blocks),
/// `resid` (length `nobs`), `nobs` (the augmented regression's rows),
/// `n_total` (T), `n_x`, `n_det`, `n_params`, `trend`, `lags`, `leads`,
/// `selected` (False when both were fixed), `ic`, `ic_value` (the
/// minimised criterion; NaN when both were fixed), `max_lag` / `max_lead`
/// (the caps the search used), `common`, `cov_type`, `kernel`,
/// `bandwidth`, `bandwidth_rule`, `force_int`, `df_adjust`,
/// `long_run_variance` (kernel LRV of the residuals at `bandwidth`,
/// df-scaled — the `sigma2_HAC` of the unadjusted covariance),
/// `rsquared`, `rsquared_adj`, `ols_params`, `ols_se` (the plain static
/// OLS on the full sample — SEs NOT valid for inference).
///
/// Validation: arch 8.0 `DynamicOLS` at 1e-10 across every trend, both
/// covariance types, the three criteria, fixed/searched/common/capped
/// leads and lags (fixtures/fmols.json).
///
/// Further arguments, with defaults: `trend` ("c"), `lags` (None), `leads`
/// (None), `ic` (None: "bic"), `common` (None: False), `max_lag` (None),
/// `max_lead` (None), `cov_type` ("unadjusted"), `kernel` ("bartlett"),
/// `bandwidth` (None), `bandwidth_rule` (None: "newey-west"), `force_int`
/// (False), `df_adjust` (False).
#[pyfunction]
#[pyo3(signature = (y, x, trend = "c", lags = None, leads = None, ic = None, common = None, max_lag = None, max_lead = None, cov_type = "unadjusted", kernel = "bartlett", bandwidth = None, bandwidth_rule = None, force_int = false, df_adjust = false))]
#[allow(clippy::too_many_arguments)]
fn dols<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    trend: &str,
    lags: Option<usize>,
    leads: Option<usize>,
    ic: Option<&str>,
    common: Option<bool>,
    max_lag: Option<usize>,
    max_lead: Option<usize>,
    cov_type: &str,
    kernel: &str,
    bandwidth: Option<f64>,
    bandwidth_rule: Option<&str>,
    force_int: bool,
    df_adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let both_fixed = lags.is_some() && leads.is_some();
    let ic_rule = match ic {
        None => DolsIc::Bic,
        Some(code) => {
            let rule = DolsIc::parse(code).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown ic {code:?}; expected \"aic\", \"bic\" or \"hqic\""
                ))
            })?;
            if both_fixed {
                return Err(PyValueError::new_err(format!(
                    "ic = {code:?} was given but both lags = {} and leads = {} are fixed: the \
                     criterion only acts when a lead or lag count is searched, so it would be \
                     inert; drop ic or leave lags/leads as None",
                    lags.unwrap_or(0),
                    leads.unwrap_or(0)
                )));
            }
            rule
        }
    };
    if let (Some(m), Some(p)) = (max_lag, lags) {
        return Err(PyValueError::new_err(format!(
            "max_lag = {m} was given but lags = {p} is fixed: the cap only bounds a searched \
             lag count, so it would be inert; drop max_lag or pass lags = None"
        )));
    }
    if let (Some(m), Some(q)) = (max_lead, leads) {
        return Err(PyValueError::new_err(format!(
            "max_lead = {m} was given but leads = {q} is fixed: the cap only bounds a \
             searched lead count, so it would be inert; drop max_lead or pass leads = None"
        )));
    }
    if let (Some(c), true) = (common, both_fixed) {
        return Err(PyValueError::new_err(format!(
            "common = {} was given but both lags = {} and leads = {} are fixed: common only \
             restricts the search to lags == leads, so it would be inert; drop common or \
             leave one of lags/leads as None",
            py_bool(c),
            lags.unwrap_or(0),
            leads.unwrap_or(0)
        )));
    }
    let cov = DolsCovType::parse(cov_type).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown cov_type {cov_type:?}; expected \"unadjusted\" or \"robust\""
        ))
    })?;
    let opts = DolsOptions {
        trend: parse_trend(trend, "trend")?,
        lags,
        leads,
        common: common.unwrap_or(false),
        max_lag,
        max_lead,
        ic: ic_rule,
        cov_type: cov,
        kernel: parse_kernel(kernel)?,
        bandwidth,
        bandwidth_rule: parse_bandwidth_rule(bandwidth, bandwidth_rule)?,
        force_int,
        df_adjust,
    };
    let xm = x_to_faer(&x);
    let r = tsecon_coint::dols(&vec1(&y), xm.as_ref(), &opts).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("params", r.params.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    d.set_item("tvalues", r.tvalues.into_pyarray(py))?;
    d.set_item("pvalues", r.pvalues.into_pyarray(py))?;
    d.set_item("cov", nested(py, &r.cov)?)?;
    d.set_item("param_names", PyList::new(py, r.param_names)?)?;
    d.set_item("full_params", r.full_params.into_pyarray(py))?;
    d.set_item("full_se", r.full_se.into_pyarray(py))?;
    d.set_item("full_cov", nested(py, &r.full_cov)?)?;
    d.set_item("full_param_names", PyList::new(py, r.full_param_names)?)?;
    d.set_item("resid", r.resid.into_pyarray(py))?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("n_total", r.n_total)?;
    d.set_item("n_x", r.n_x)?;
    d.set_item("n_det", r.n_det)?;
    d.set_item("n_params", r.n_params)?;
    d.set_item("trend", r.trend.code())?;
    d.set_item("lags", r.lags)?;
    d.set_item("leads", r.leads)?;
    d.set_item("selected", r.selected)?;
    d.set_item("ic", r.ic.code())?;
    d.set_item("ic_value", r.ic_value)?;
    d.set_item("max_lag", r.max_lag)?;
    d.set_item("max_lead", r.max_lead)?;
    d.set_item("common", r.common)?;
    d.set_item("cov_type", r.cov_type.code())?;
    d.set_item("kernel", tsecon_coint::kernel_code(r.kernel))?;
    d.set_item("bandwidth", r.bandwidth)?;
    match r.bandwidth_rule {
        Some(rule) => d.set_item("bandwidth_rule", rule.code())?,
        None => d.set_item("bandwidth_rule", py.None())?,
    }
    d.set_item("force_int", r.force_int)?;
    d.set_item("df_adjust", r.df_adjust)?;
    d.set_item("long_run_variance", r.long_run_variance)?;
    d.set_item("rsquared", r.rsquared)?;
    d.set_item("rsquared_adj", r.rsquared_adj)?;
    d.set_item("ols_params", r.ols_params.into_pyarray(py))?;
    d.set_item("ols_se", r.ols_se.into_pyarray(py))?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fmols, m)?)?;
    m.add_function(wrap_pyfunction!(dols, m)?)?;
    m.add_function(wrap_pyfunction!(ccr, m)?)?;
    Ok(())
}
