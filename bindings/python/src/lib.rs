//! Python bindings for the tsecon foundation crates (v0 smoke-test surface).
//!
//! This is deliberately a thin, flat function surface proving the
//! Rust-to-Python pipeline end to end; the ergonomic model-object API
//! (Spec -> fit() -> Results) arrives with the model crates.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Autocorrelation function with Bartlett standard errors.
///
/// Matches statsmodels `acf` conventions exactly (validated at 1e-12).
#[pyfunction]
#[pyo3(signature = (y, nlags = 20, adjusted = false))]
fn acf<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    nlags: usize,
    adjusted: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_diag::acf(y.as_slice()?, nlags, adjusted).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("acf", r.acf.into_pyarray(py))?;
    d.set_item("bartlett_se", r.bartlett_se.into_pyarray(py))?;
    Ok(d)
}

/// Partial autocorrelation function.
///
/// `method`: "yw" (Yule-Walker, statsmodels 'ywm') or "ols".
#[pyfunction]
#[pyo3(signature = (y, nlags = 20, method = "yw"))]
fn pacf<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    nlags: usize,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = y.as_slice()?;
    let v = match method {
        "yw" => tsecon_diag::pacf_yw(y, nlags).map_err(to_py)?,
        "ols" => tsecon_diag::pacf_ols(y, nlags).map_err(to_py)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown pacf method {other:?}; expected \"yw\" or \"ols\""
            )))
        }
    };
    Ok(v.into_pyarray(py))
}

/// Ljung-Box and Box-Pierce portmanteau tests for lags 1..=nlags.
#[pyfunction]
#[pyo3(signature = (y, nlags = 10))]
fn ljung_box<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_diag::ljung_box(y.as_slice()?, nlags).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("lags", r.lags.into_iter().map(|x| x as u64).collect::<Vec<_>>().into_pyarray(py))?;
    d.set_item("lb_stat", r.lb_stat.into_pyarray(py))?;
    d.set_item("lb_pvalue", r.lb_pvalue.into_pyarray(py))?;
    d.set_item("bp_stat", r.bp_stat.into_pyarray(py))?;
    d.set_item("bp_pvalue", r.bp_pvalue.into_pyarray(py))?;
    Ok(d)
}

/// Jarque-Bera normality test.
#[pyfunction]
fn jarque_bera<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_diag::jarque_bera(x.as_slice()?).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("skewness", r.skewness)?;
    d.set_item("kurtosis", r.kurtosis)?;
    d.set_item("n", r.n)?;
    Ok(d)
}

/// Engle's ARCH-LM test (statsmodels `het_arch` convention).
#[pyfunction]
#[pyo3(signature = (resid, nlags = 4))]
fn arch_lm<'py>(
    py: Python<'py>,
    resid: PyReadonlyArray1<'py, f64>,
    nlags: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_diag::arch_lm(resid.as_slice()?, nlags).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("df", r.df)?;
    d.set_item("nobs", r.nobs)?;
    Ok(d)
}

/// Uniform draws from the tsecon Philox stream.
///
/// Bit-identical to `np.random.Generator(np.random.Philox(seed)).random(n)`.
#[pyfunction]
fn philox_uniforms(py: Python<'_>, seed: u64, n: usize) -> PyResult<Bound<'_, PyArray1<f64>>> {
    let mut stream = tsecon_rng::Stream::new(seed);
    let mut buf = vec![0.0_f64; n];
    stream.fill_uniform_f64(&mut buf);
    Ok(buf.into_pyarray(py))
}

/// Bootstrap resampling indices.
///
/// `scheme`: "iid", "moving" or "circular" (require `block_length`), or
/// "stationary" (requires `p`, the restart probability; expected block
/// length is 1/p). Same seed always yields the same indices.
#[pyfunction]
#[pyo3(signature = (n, scheme = "stationary", seed = 0, block_length = None, p = None))]
fn bootstrap_indices<'py>(
    py: Python<'py>,
    n: usize,
    scheme: &str,
    seed: u64,
    block_length: Option<usize>,
    p: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    use tsecon_bootstrap::BlockScheme;
    let need = |o: Option<usize>, what: &str| {
        o.ok_or_else(|| PyValueError::new_err(format!("scheme {scheme:?} requires {what}")))
    };
    let s = match scheme {
        "iid" => BlockScheme::Iid,
        "moving" => BlockScheme::MovingBlock { block_length: need(block_length, "block_length")? },
        "circular" => BlockScheme::CircularBlock { block_length: need(block_length, "block_length")? },
        "stationary" => BlockScheme::Stationary {
            p: p.ok_or_else(|| PyValueError::new_err("scheme \"stationary\" requires p"))?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scheme {other:?}; expected iid/moving/circular/stationary"
            )))
        }
    };
    let mut stream = tsecon_rng::Stream::new(seed);
    let idx = tsecon_bootstrap::indices(s, n, &mut stream).map_err(to_py)?;
    Ok(idx.into_iter().map(|i| i as u64).collect::<Vec<_>>().into_pyarray(py))
}

/// Politis-White (2004) automatic block length with the Patton-Politis-White
/// (2009) correction. Returns optimal lengths for the stationary and
/// circular block bootstraps.
#[pyfunction]
fn optimal_block_length<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_bootstrap::optimal_block_length(y.as_slice()?).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("stationary", r.stationary)?;
    d.set_item("circular", r.circular)?;
    Ok(d)
}

/// Fit-free local-level pass: exact-diffuse Kalman filter + smoother at
/// fixed variances. NaN entries in `y` are treated as missing.
///
/// Returns loglik plus filtered/smoothed level and variances; matches
/// statsmodels `UnobservedComponents(..., use_exact_diffuse=True)` at
/// fixed params (validated ~1e-11).
#[pyfunction]
fn local_level_smooth<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    sigma2_eps: f64,
    sigma2_eta: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_ssm::tsecon_linalg::faer::Mat;
    let ys = y.as_slice()?;
    let obs = Mat::from_fn(ys.len(), 1, |i, _| ys[i]);
    let model = tsecon_ssm::LinearGaussianSSM::local_level(sigma2_eps, sigma2_eta).map_err(to_py)?;
    let out = model.smooth(obs.as_ref()).map_err(to_py)?;
    let n = ys.len();
    let d = PyDict::new(py);
    d.set_item("loglik", out.filter.loglik)?;
    d.set_item("d_diffuse", out.filter.d_diffuse)?;
    let filt: Vec<f64> = out.filter.filtered_state.iter().take(n).map(|s| s[0]).collect();
    let filt_var: Vec<f64> = out.filter.filtered_state_cov.iter().take(n).map(|p| p[(0, 0)]).collect();
    let smo: Vec<f64> = out.smoothed_state.iter().take(n).map(|s| s[0]).collect();
    let smo_var: Vec<f64> = out.smoothed_state_cov.iter().take(n).map(|p| p[(0, 0)]).collect();
    d.set_item("filtered_state", filt.into_pyarray(py))?;
    d.set_item("filtered_state_var", filt_var.into_pyarray(py))?;
    d.set_item("smoothed_state", smo.into_pyarray(py))?;
    d.set_item("smoothed_state_var", smo_var.into_pyarray(py))?;
    Ok(d)
}

/// Gaussian log-likelihood of an AR(p) model with intercept at fixed
/// parameters, evaluated exactly via the state-space form with stationary
/// initialization (matches statsmodels SARIMAX `trend='c'` conventions).
#[pyfunction]
#[pyo3(signature = (y, coeffs, sigma2, intercept = 0.0))]
fn ar_loglik(
    y: PyReadonlyArray1<'_, f64>,
    coeffs: Vec<f64>,
    sigma2: f64,
    intercept: f64,
) -> PyResult<f64> {
    use tsecon_ssm::tsecon_linalg::faer::Mat;
    let ys = y.as_slice()?;
    let obs = Mat::from_fn(ys.len(), 1, |i, _| ys[i]);
    let model = tsecon_ssm::LinearGaussianSSM::ar(&coeffs, sigma2, intercept).map_err(to_py)?;
    model.loglike(obs.as_ref()).map_err(to_py)
}

fn adf_regression(s: &str) -> PyResult<tsecon_diag::AdfRegression> {
    use tsecon_diag::AdfRegression::*;
    match s {
        "n" => Ok(NoConstant),
        "c" => Ok(Constant),
        "ct" => Ok(ConstantTrend),
        other => Err(PyValueError::new_err(format!(
            "unknown regression {other:?}; expected \"n\", \"c\", or \"ct\""
        ))),
    }
}

/// Augmented Dickey-Fuller unit-root test with MacKinnon p-values.
///
/// `regression`: "n", "c" (default), "ct". `autolag`: "aic" (default),
/// "bic", "t-stat", or None to use `maxlag` as a fixed lag.
/// Matches statsmodels `adfuller` (validated at 1e-8).
#[pyfunction]
#[pyo3(signature = (y, regression = "c", autolag = Some("aic"), maxlag = None))]
fn adf<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    regression: &str,
    autolag: Option<&str>,
    maxlag: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_diag::AdfLagSelection as L;
    let sel = match autolag {
        Some("aic") | Some("AIC") => L::Aic(maxlag),
        Some("bic") | Some("BIC") => L::Bic(maxlag),
        Some("t-stat") => L::TStat(maxlag),
        None => L::Fixed(maxlag.ok_or_else(|| {
            PyValueError::new_err("autolag=None requires an explicit maxlag (used as fixed lag)")
        })?),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown autolag {other:?}; expected \"aic\", \"bic\", \"t-stat\", or None"
            )))
        }
    };
    let r = tsecon_diag::adf(y.as_slice()?, adf_regression(regression)?, sel).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("used_lag", r.used_lag)?;
    d.set_item("nobs", r.nobs)?;
    let crit = PyDict::new(py);
    crit.set_item("1%", r.crit.pct1)?;
    crit.set_item("5%", r.crit.pct5)?;
    crit.set_item("10%", r.crit.pct10)?;
    d.set_item("crit", crit)?;
    Ok(d)
}

/// KPSS stationarity test (null: stationary).
///
/// `regression`: "c" (level-stationary, default) or "ct" (trend-stationary).
/// `nlags`: "auto" (Hobijn-Franses-Ooms, default), "legacy", or an integer.
/// P-value is interpolated and bounded to [0.01, 0.10], statsmodels-style.
#[pyfunction]
#[pyo3(signature = (y, regression = "c", nlags = None))]
fn kpss<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    regression: &str,
    nlags: Option<Bound<'py, pyo3::PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_diag::{KpssLags, KpssRegression};
    let reg = match regression {
        "c" => KpssRegression::Constant,
        "ct" => KpssRegression::ConstantTrend,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown regression {other:?}; expected \"c\" or \"ct\""
            )))
        }
    };
    let lags = match &nlags {
        None => KpssLags::Auto,
        Some(v) => {
            if let Ok(s) = v.extract::<String>() {
                match s.as_str() {
                    "auto" => KpssLags::Auto,
                    "legacy" => KpssLags::Legacy,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown nlags {other:?}; expected \"auto\", \"legacy\", or an integer"
                        )))
                    }
                }
            } else {
                KpssLags::Fixed(v.extract::<usize>()?)
            }
        }
    };
    let r = tsecon_diag::kpss(y.as_slice()?, reg, lags).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("lags", r.lags)?;
    Ok(d)
}

/// The stationarity decision workflow: ADF and KPSS run together and
/// classified into the confirmatory quadrant, with a teaching
/// interpretation and a concrete recommendation.
#[pyfunction]
#[pyo3(signature = (y, alpha = 0.05))]
fn check_stationarity<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_diag::check_stationarity_at(y.as_slice()?, alpha).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("quadrant", format!("{:?}", r.quadrant))?;
    d.set_item("recommendation", format!("{:?}", r.recommendation))?;
    d.set_item("interpretation", &r.interpretation)?;
    d.set_item("adf_statistic", r.adf.statistic)?;
    d.set_item("adf_p_value", r.adf.p_value)?;
    d.set_item("kpss_statistic", r.kpss.statistic)?;
    d.set_item("kpss_p_value", r.kpss.p_value)?;
    d.set_item("alpha", r.alpha)?;
    Ok(d)
}

fn hac_kernel(s: &str) -> PyResult<tsecon_hac::Kernel> {
    use tsecon_hac::Kernel::*;
    match s {
        "bartlett" | "newey-west" => Ok(Bartlett),
        "parzen" => Ok(Parzen),
        "qs" | "quadratic-spectral" => Ok(QuadraticSpectral),
        "truncated" => Ok(Truncated),
        other => Err(PyValueError::new_err(format!(
            "unknown kernel {other:?}; expected bartlett/parzen/qs/truncated"
        ))),
    }
}

/// Kernel long-run variance of a series (demeaned internally).
///
/// `bandwidth=None` uses the Newey-West rule-of-thumb maxlags
/// floor(4*(n/100)^(2/9)) for Bartlett/Parzen, matching common practice.
#[pyfunction]
#[pyo3(signature = (x, kernel = "bartlett", bandwidth = None))]
fn long_run_variance(
    x: PyReadonlyArray1<'_, f64>,
    kernel: &str,
    bandwidth: Option<f64>,
) -> PyResult<f64> {
    let xs = x.as_slice()?;
    let mean = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
    let z: Vec<f64> = xs.iter().map(|v| v - mean).collect();
    let k = hac_kernel(kernel)?;
    let bw = bandwidth.unwrap_or_else(|| tsecon_hac::newey_west_maxlags(z.len()) as f64);
    tsecon_hac::lrv(&z, k, bw).map_err(to_py)
}

/// OLS with robust standard-error options.
///
/// `x` is a 2-D design matrix used as-is (add your own constant column).
/// `se_type`: "nonrobust", "hc0", "hc1", or "hac" (Bartlett kernel;
/// `maxlags=None` uses the Newey-West rule of thumb). HAC results match
/// statsmodels `cov_type="HAC"` at 1e-10.
#[pyfunction]
#[pyo3(signature = (y, x, se_type = "hac", maxlags = None, use_correction = true))]
fn ols<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    se_type: &str,
    maxlags: Option<usize>,
    use_correction: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let ys = y.as_slice()?;
    let xa = x.as_array();
    let cols: Vec<Vec<f64>> = (0..xa.ncols()).map(|j| xa.column(j).to_vec()).collect();
    let fit = tsecon_hac::ols(ys, &cols).map_err(to_py)?;
    let se = match se_type {
        "nonrobust" => tsecon_hac::SeType::NonRobust,
        "hc0" => tsecon_hac::SeType::Hc0,
        "hc1" => tsecon_hac::SeType::Hc1,
        "hac" => tsecon_hac::SeType::Hac {
            kernel: tsecon_hac::Kernel::Bartlett,
            bandwidth: maxlags.unwrap_or_else(|| tsecon_hac::newey_west_maxlags(ys.len())) as f64,
            use_correction,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown se_type {other:?}; expected nonrobust/hc0/hc1/hac"
            )))
        }
    };
    let inf = fit.inference(se).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("params", fit.params.clone().into_pyarray(py))?;
    d.set_item("bse", inf.bse.into_pyarray(py))?;
    d.set_item("tvalues", inf.tvalues.into_pyarray(py))?;
    d.set_item("se_type", se_type)?;
    Ok(d)
}

fn var_results(
    data: &numpy::PyReadonlyArray2<'_, f64>,
    lags: usize,
    trend: &str,
) -> PyResult<tsecon_var::VarResults> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    let a = data.as_array();
    let m = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let tr = match trend {
        "c" => tsecon_var::Trend::Constant,
        "n" => tsecon_var::Trend::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown trend {other:?}; expected \"c\" or \"n\""
            )))
        }
    };
    tsecon_var::VarSpec { lags, trend: tr }
        .fit(m.as_ref())
        .map_err(to_py)
}

fn mat_to_vec2(m: &tsecon_var::tsecon_linalg::faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows()).map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect()).collect()
}

/// Fit a VAR(p) by OLS and return estimates, fit statistics, and stability.
///
/// Matches statsmodels `VAR(...).fit(lags, trend)` at 1e-8.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, trend = "c"))]
fn var_fit<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    trend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let r = var_results(&data, lags, trend)?;
    let d = PyDict::new(py);
    d.set_item("params", mat_to_vec2(&r.params))?;
    d.set_item("sigma_u", mat_to_vec2(&r.sigma_u))?;
    d.set_item("llf", r.llf)?;
    d.set_item("aic", r.aic)?;
    d.set_item("bic", r.bic)?;
    d.set_item("hqic", r.hqic)?;
    let roots = r.roots_moduli().map_err(to_py)?;
    d.set_item("max_root", roots.iter().cloned().fold(0.0_f64, f64::max))?;
    Ok(d)
}

/// Impulse responses of a fitted VAR: `irfs[h][i][j]` is the response of
/// variable i to a shock in variable j at horizon h (orthogonalized via
/// the Cholesky factor of sigma_u when `orth=True`).
#[pyfunction]
#[pyo3(signature = (data, lags = 2, horizon = 10, orth = true, trend = "c"))]
fn var_irf<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    orth: bool,
    trend: &str,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let r = var_results(&data, lags, trend)?;
    let irf = r.irf(horizon).map_err(to_py)?;
    let mats = if orth { &irf.orth_irfs } else { &irf.irfs };
    let out: Vec<Vec<Vec<f64>>> = mats.iter().map(mat_to_vec2).collect();
    pyo3::types::PyList::new(py, out.iter().map(|m| m.clone()))
}

/// Forecast-error variance decomposition: `fevd[h][i][j]` is the share of
/// variable i's h-step forecast-error variance attributed to shock j.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, horizon = 10, trend = "c"))]
fn var_fevd<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    trend: &str,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let r = var_results(&data, lags, trend)?;
    let fevd = r.fevd(horizon).map_err(to_py)?;
    let out: Vec<Vec<Vec<f64>>> = fevd.decomp.iter().map(mat_to_vec2).collect();
    pyo3::types::PyList::new(py, out.iter().map(|m| m.clone()))
}

/// Iterated VAR point forecasts with (innovation-uncertainty) intervals.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, steps = 8, alpha = 0.05, trend = "c"))]
fn var_forecast<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    steps: usize,
    alpha: f64,
    trend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let r = var_results(&data, lags, trend)?;
    let fc = r.forecast_interval(steps, alpha).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("point", mat_to_vec2(&fc.point))?;
    d.set_item("lower", mat_to_vec2(&fc.lower))?;
    d.set_item("upper", mat_to_vec2(&fc.upper))?;
    Ok(d)
}

/// Granger-causality F test: do the `causing` variables help predict the
/// `caused` variables? Matches statsmodels `test_causality(kind="f")`.
#[pyfunction]
#[pyo3(signature = (data, caused, causing, lags = 2, trend = "c"))]
fn var_granger<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    caused: Vec<usize>,
    causing: Vec<usize>,
    lags: usize,
    trend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let r = var_results(&data, lags, trend)?;
    let t = r.test_causality(&caused, &causing).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("statistic", t.statistic)?;
    d.set_item("p_value", t.pvalue)?;
    d.set_item("df_num", t.df_num)?;
    d.set_item("df_den", t.df_den)?;
    Ok(d)
}

fn decomposition_dict<'py>(
    py: Python<'py>,
    dec: &tsecon_filters::Decomposition,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    if let Some(tr) = &dec.trend {
        d.set_item("trend", tr.clone().into_pyarray(py))?;
    }
    d.set_item("cycle", dec.cycle.clone().into_pyarray(py))?;
    d.set_item("first_index", dec.alignment.first_index())?;
    Ok(d)
}

/// Hodrick-Prescott filter (O(n) pentadiagonal solve). `one_sided=True`
/// gives the real-time variant. Matches statsmodels `hpfilter` at 1e-8.
#[pyfunction]
#[pyo3(signature = (y, lamb = 1600.0, one_sided = false))]
fn hp_filter<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    lamb: f64,
    one_sided: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let dec = if one_sided {
        tsecon_filters::hp_filter_one_sided(y.as_slice()?, lamb)
    } else {
        tsecon_filters::hp_filter(y.as_slice()?, lamb)
    }
    .map_err(to_py)?;
    decomposition_dict(py, &dec)
}

/// Baxter-King band-pass filter (loses `k` observations at each end —
/// `first_index` reports the alignment).
#[pyfunction]
#[pyo3(signature = (y, low = 6.0, high = 32.0, k = 12))]
fn bk_filter<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    k: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let dec = tsecon_filters::bk_filter(y.as_slice()?, low, high, k).map_err(to_py)?;
    decomposition_dict(py, &dec)
}

/// Christiano-Fitzgerald asymmetric band-pass filter (full sample).
#[pyfunction]
#[pyo3(signature = (y, low = 6.0, high = 32.0, drift = true))]
fn cf_filter<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    low: f64,
    high: f64,
    drift: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let dec = tsecon_filters::cf_filter(y.as_slice()?, low, high, drift).map_err(to_py)?;
    decomposition_dict(py, &dec)
}

/// Hamilton (2018) regression filter — the modern HP alternative.
#[pyfunction]
#[pyo3(signature = (y, h = 8, p = 4))]
fn hamilton_filter<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    h: usize,
    p: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_filters::hamilton_filter(y.as_slice()?, h, p).map_err(to_py)?;
    let d = decomposition_dict(py, &r.decomposition)?;
    d.set_item("beta", r.beta.clone().into_pyarray(py))?;
    Ok(d)
}

/// Diebold-Mariano test of equal predictive accuracy with the
/// Harvey-Leybourne-Newbold small-sample correction.
#[pyfunction]
#[pyo3(signature = (e1, e2, h = 1, loss = "squared"))]
fn dm_test<'py>(
    py: Python<'py>,
    e1: PyReadonlyArray1<'py, f64>,
    e2: PyReadonlyArray1<'py, f64>,
    h: usize,
    loss: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let l = match loss {
        "squared" => tsecon_forecast::DmLoss::Squared,
        "absolute" => tsecon_forecast::DmLoss::Absolute,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown loss {other:?}; expected \"squared\" or \"absolute\""
            )))
        }
    };
    let r = tsecon_forecast::dm_test(e1.as_slice()?, e2.as_slice()?, h, l).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("dm_stat", r.dm_stat)?;
    d.set_item("hln_stat", r.hln_stat)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("mean_loss_diff", r.mean_loss_diff)?;
    Ok(d)
}

/// Forecast accuracy measures in one call.
#[pyfunction]
#[pyo3(signature = (actual, forecast, insample = None, period = 1))]
fn accuracy<'py>(
    py: Python<'py>,
    actual: PyReadonlyArray1<'py, f64>,
    forecast: PyReadonlyArray1<'py, f64>,
    insample: Option<PyReadonlyArray1<'py, f64>>,
    period: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_forecast as f;
    let a = actual.as_slice()?;
    let p = forecast.as_slice()?;
    let d = PyDict::new(py);
    d.set_item("me", f::me(a, p).map_err(to_py)?)?;
    d.set_item("rmse", f::rmse(a, p).map_err(to_py)?)?;
    d.set_item("mae", f::mae(a, p).map_err(to_py)?)?;
    if let Ok(v) = f::mape(a, p) {
        d.set_item("mape", v)?;
    }
    if let Ok(v) = f::smape(a, p) {
        d.set_item("smape", v)?;
    }
    if let Some(ins) = insample {
        d.set_item("mase", f::mase(a, p, ins.as_slice()?, period).map_err(to_py)?)?;
        d.set_item("rmsse", f::rmsse(a, p, ins.as_slice()?, period).map_err(to_py)?)?;
    }
    Ok(d)
}

/// The Theta method (Assimakopoulos-Nikolopoulos 2000) — a stubbornly hard
/// benchmark to beat. Matches statsmodels ThetaModel.
#[pyfunction]
#[pyo3(signature = (y, steps, period = 1))]
fn theta_forecast<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    steps: usize,
    period: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let r = tsecon_forecast::theta_forecast(y.as_slice()?, period, steps).map_err(to_py)?;
    Ok(r.forecast.into_pyarray(py))
}

/// Fit a univariate volatility model by QMLE.
///
/// `vol`: "garch", "gjr", or "egarch"; `mean`: "zero" or "constant";
/// `dist`: "normal" or "t". Conventions and results match the `arch`
/// package (fixed-parameter logliks at machine precision). Returns both
/// MLE and Bollerslev-Wooldridge robust standard errors.
#[pyfunction]
#[pyo3(signature = (y, vol = "garch", mean = "zero", dist = "normal", p = 1, o = 1, q = 1, forecast_horizon = 0))]
#[allow(clippy::too_many_arguments)]
fn garch_fit<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    vol: &str,
    mean: &str,
    dist: &str,
    p: usize,
    o: usize,
    q: usize,
    forecast_horizon: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_garch::{DistSpec, GarchModel, GarchSpec, MeanSpec, VolSpec};
    let spec = GarchSpec {
        mean: match mean {
            "zero" => MeanSpec::Zero,
            "constant" => MeanSpec::Constant,
            other => return Err(PyValueError::new_err(format!("unknown mean {other:?}; expected zero/constant"))),
        },
        vol: match vol {
            "garch" => VolSpec::Garch { p, q },
            "gjr" => VolSpec::Gjr { p, o, q },
            "egarch" => VolSpec::Egarch { p, o, q },
            other => return Err(PyValueError::new_err(format!("unknown vol {other:?}; expected garch/gjr/egarch"))),
        },
        dist: match dist {
            "normal" => DistSpec::Normal,
            "t" => DistSpec::StudentT,
            other => return Err(PyValueError::new_err(format!("unknown dist {other:?}; expected normal/t"))),
        },
    };
    let model = GarchModel::new(y.as_slice()?, spec).map_err(to_py)?;
    let r = model.fit().map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("params", r.params.clone().into_pyarray(py))?;
    d.set_item("param_names", r.param_names.clone())?;
    d.set_item("loglik", r.loglik)?;
    d.set_item("aic", r.aic)?;
    d.set_item("bic", r.bic)?;
    d.set_item("se_mle", r.se_mle.clone().into_pyarray(py))?;
    d.set_item("se_robust", r.se_robust.clone().into_pyarray(py))?;
    d.set_item("conditional_volatility", r.conditional_volatility.clone().into_pyarray(py))?;
    d.set_item("std_residuals", r.std_residuals.clone().into_pyarray(py))?;
    if forecast_horizon > 0 {
        d.set_item("variance_forecast", r.forecast_variance(forecast_horizon).map_err(to_py)?.into_pyarray(py))?;
    }
    Ok(d)
}

/// Fit a Bayesian VAR with the Minnesota / Normal-inverse-Wishart
/// conjugate prior (closed-form posterior — no MCMC needed).
///
/// `delta` is the own-first-lag prior mean (0 for growth rates, 1 for
/// levels/random-walk shrinkage). Returns the posterior coefficient
/// mean, posterior mean of Sigma, and the log marginal likelihood (the
/// evidence — compare across lambda settings to tune tightness).
#[pyfunction]
#[pyo3(signature = (data, lags = 2, lambda0 = 100.0, lambda1 = 0.2, lambda3 = 1.0, delta = 0.0))]
fn bvar_fit<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    lambda0: f64,
    lambda1: f64,
    lambda3: f64,
    delta: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let a = data.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let prior = tsecon_bayes::MinnesotaNiwPrior::new(m.as_ref(), lags, lambda0, lambda1, lambda3, delta)
        .map_err(to_py)?;
    let post = prior.posterior(m.as_ref()).map_err(to_py)?;
    let d = PyDict::new(py);
    let bb = post.b_bar();
    d.set_item("posterior_mean_coefs", (0..bb.nrows()).map(|i| (0..bb.ncols()).map(|j| bb[(i, j)]).collect::<Vec<_>>()).collect::<Vec<_>>())?;
    let sm = post.sigma_posterior_mean().map_err(to_py)?;
    d.set_item("sigma_posterior_mean", (0..sm.nrows()).map(|i| (0..sm.ncols()).map(|j| sm[(i, j)]).collect::<Vec<_>>()).collect::<Vec<_>>())?;
    d.set_item("log_marginal_likelihood", post.log_marginal_likelihood())?;
    Ok(d)
}

/// Posterior draws of Cholesky-orthogonalized impulse responses from the
/// Minnesota-NIW BVAR: returns a list [draw][horizon][variable][shock] —
/// take numpy quantiles across draws for credible bands.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, horizon = 16, n_draws = 500, seed = 0, lambda0 = 100.0, lambda1 = 0.2, lambda3 = 1.0, delta = 0.0))]
#[allow(clippy::too_many_arguments)]
fn bvar_irf_draws<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    n_draws: usize,
    seed: u64,
    lambda0: f64,
    lambda1: f64,
    lambda3: f64,
    delta: f64,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let a = data.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let prior = tsecon_bayes::MinnesotaNiwPrior::new(m.as_ref(), lags, lambda0, lambda1, lambda3, delta)
        .map_err(to_py)?;
    let post = prior.posterior(m.as_ref()).map_err(to_py)?;
    let mut stream = tsecon_rng::Stream::new(seed);
    let draws = post.irf_draws(n_draws, horizon, &mut stream).map_err(to_py)?;
    let out: Vec<Vec<Vec<Vec<f64>>>> = draws
        .iter()
        .map(|dr| dr.iter().map(mat_to_vec2_bayes).collect())
        .collect();
    pyo3::types::PyList::new(py, out.iter().map(|d| d.clone()))
}

fn mat_to_vec2_bayes(m: &tsecon_var::tsecon_linalg::faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows()).map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect()).collect()
}

/// MCMC convergence diagnostics (Vehtari et al. 2021, ArviZ-exact):
/// rank-normalized split R-hat and bulk/tail effective sample sizes.
/// `chains` is (n_chains, n_draws).
#[pyfunction]
fn mcmc_diagnostics<'py>(
    py: Python<'py>,
    chains: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let a = chains.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let d = PyDict::new(py);
    d.set_item("rhat", tsecon_bayes::convergence::rhat_rank(m.as_ref()).map_err(to_py)?)?;
    d.set_item("ess_bulk", tsecon_bayes::convergence::ess_bulk(m.as_ref()).map_err(to_py)?)?;
    d.set_item("ess_tail", tsecon_bayes::convergence::ess_tail(m.as_ref()).map_err(to_py)?)?;
    Ok(d)
}

#[pymodule]
fn tsecon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(acf, m)?)?;
    m.add_function(wrap_pyfunction!(pacf, m)?)?;
    m.add_function(wrap_pyfunction!(ljung_box, m)?)?;
    m.add_function(wrap_pyfunction!(jarque_bera, m)?)?;
    m.add_function(wrap_pyfunction!(arch_lm, m)?)?;
    m.add_function(wrap_pyfunction!(philox_uniforms, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_indices, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_block_length, m)?)?;
    m.add_function(wrap_pyfunction!(local_level_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(ar_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(adf, m)?)?;
    m.add_function(wrap_pyfunction!(kpss, m)?)?;
    m.add_function(wrap_pyfunction!(check_stationarity, m)?)?;
    m.add_function(wrap_pyfunction!(long_run_variance, m)?)?;
    m.add_function(wrap_pyfunction!(ols, m)?)?;
    m.add_function(wrap_pyfunction!(var_fit, m)?)?;
    m.add_function(wrap_pyfunction!(var_irf, m)?)?;
    m.add_function(wrap_pyfunction!(var_fevd, m)?)?;
    m.add_function(wrap_pyfunction!(var_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(var_granger, m)?)?;
    m.add_function(wrap_pyfunction!(hp_filter, m)?)?;
    m.add_function(wrap_pyfunction!(bk_filter, m)?)?;
    m.add_function(wrap_pyfunction!(cf_filter, m)?)?;
    m.add_function(wrap_pyfunction!(hamilton_filter, m)?)?;
    m.add_function(wrap_pyfunction!(dm_test, m)?)?;
    m.add_function(wrap_pyfunction!(accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(theta_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(garch_fit, m)?)?;
    m.add_function(wrap_pyfunction!(bvar_fit, m)?)?;
    m.add_function(wrap_pyfunction!(bvar_irf_draws, m)?)?;
    m.add_function(wrap_pyfunction!(mcmc_diagnostics, m)?)?;
    Ok(())
}
