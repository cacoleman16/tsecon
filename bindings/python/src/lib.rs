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
    d.set_item(
        "lags",
        r.lags
            .into_iter()
            .map(|x| x as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("lb_stat", r.lb_stat.into_pyarray(py))?;
    d.set_item("lb_pvalue", r.lb_pvalue.into_pyarray(py))?;
    d.set_item("bp_stat", r.bp_stat.into_pyarray(py))?;
    d.set_item("bp_pvalue", r.bp_pvalue.into_pyarray(py))?;
    Ok(d)
}

/// Jarque-Bera normality test.
#[pyfunction]
fn jarque_bera<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
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
        "moving" => BlockScheme::MovingBlock {
            block_length: need(block_length, "block_length")?,
        },
        "circular" => BlockScheme::CircularBlock {
            block_length: need(block_length, "block_length")?,
        },
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
    Ok(idx
        .into_iter()
        .map(|i| i as u64)
        .collect::<Vec<_>>()
        .into_pyarray(py))
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
    let model =
        tsecon_ssm::LinearGaussianSSM::local_level(sigma2_eps, sigma2_eta).map_err(to_py)?;
    let out = model.smooth(obs.as_ref()).map_err(to_py)?;
    let n = ys.len();
    let d = PyDict::new(py);
    d.set_item("loglik", out.filter.loglik)?;
    d.set_item("d_diffuse", out.filter.d_diffuse)?;
    let filt: Vec<f64> = out
        .filter
        .filtered_state
        .iter()
        .take(n)
        .map(|s| s[0])
        .collect();
    let filt_var: Vec<f64> = out
        .filter
        .filtered_state_cov
        .iter()
        .take(n)
        .map(|p| p[(0, 0)])
        .collect();
    let smo: Vec<f64> = out.smoothed_state.iter().take(n).map(|s| s[0]).collect();
    let smo_var: Vec<f64> = out
        .smoothed_state_cov
        .iter()
        .take(n)
        .map(|p| p[(0, 0)])
        .collect();
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
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
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
#[pyo3(signature = (data, lags = 2, horizon = 10, orth = true, trend = "c", cumulative = false))]
fn var_irf<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    orth: bool,
    trend: &str,
    cumulative: bool,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let r = var_results(&data, lags, trend)?;
    let irf = r.irf(horizon).map_err(to_py)?;
    let mats = if orth { &irf.orth_irfs } else { &irf.irfs };
    let mut out: Vec<Vec<Vec<f64>>> = mats.iter().map(mat_to_vec2).collect();
    if cumulative {
        // Running total over horizons: the level response to a shock when the
        // VAR is estimated in differences. Point path only — correct cumulative
        // BANDS need the joint covariance across horizons (delta method or
        // bootstrap; Bayesian: use bvar_irf_draws(cumulative=True)).
        for h in 1..out.len() {
            for i in 0..out[h].len() {
                for j in 0..out[h][i].len() {
                    out[h][i][j] += out[h - 1][i][j];
                }
            }
        }
    }
    pyo3::types::PyList::new(py, out.iter().cloned())
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
    pyo3::types::PyList::new(py, out.iter().cloned())
}

/// Iterated VAR point forecasts with (innovation-uncertainty) intervals.
///
/// `alpha` sets the interval coverage: the bands are the symmetric
/// `1 - alpha` asymptotic intervals `point +/- z_{1-alpha/2} * se`
/// (normal quantile; e.g. the default `alpha=0.05` is a 95% interval
/// with z = 1.96, `alpha=0.32` a 68% interval with z ~= 0.994).
/// Intervals reflect innovation uncertainty only (coefficients treated
/// as known), matching statsmodels `forecast_interval`.
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
        d.set_item(
            "mase",
            f::mase(a, p, ins.as_slice()?, period).map_err(to_py)?,
        )?;
        d.set_item(
            "rmsse",
            f::rmsse(a, p, ins.as_slice()?, period).map_err(to_py)?,
        )?;
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
///
/// When `forecast_horizon > 0`, `variance_forecast` is the analytic
/// *point* path of conditional variances `E[sigma2_{T+m} | F_T]`,
/// m = 1..horizon — it carries no interval or coverage level, and none
/// is implied (forecast distributions for GARCH variance paths require
/// simulation, which is not yet exposed).
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
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown mean {other:?}; expected zero/constant"
                )))
            }
        },
        vol: match vol {
            "garch" => VolSpec::Garch { p, q },
            "gjr" => VolSpec::Gjr { p, o, q },
            "egarch" => VolSpec::Egarch { p, o, q },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown vol {other:?}; expected garch/gjr/egarch"
                )))
            }
        },
        dist: match dist {
            "normal" => DistSpec::Normal,
            "t" => DistSpec::StudentT,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown dist {other:?}; expected normal/t"
                )))
            }
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
    d.set_item(
        "conditional_volatility",
        r.conditional_volatility.clone().into_pyarray(py),
    )?;
    d.set_item("std_residuals", r.std_residuals.clone().into_pyarray(py))?;
    if forecast_horizon > 0 {
        d.set_item(
            "variance_forecast",
            r.forecast_variance(forecast_horizon)
                .map_err(to_py)?
                .into_pyarray(py),
        )?;
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
    let prior =
        tsecon_bayes::MinnesotaNiwPrior::new(m.as_ref(), lags, lambda0, lambda1, lambda3, delta)
            .map_err(to_py)?;
    let post = prior.posterior(m.as_ref()).map_err(to_py)?;
    let d = PyDict::new(py);
    let bb = post.b_bar();
    d.set_item(
        "posterior_mean_coefs",
        (0..bb.nrows())
            .map(|i| (0..bb.ncols()).map(|j| bb[(i, j)]).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    let sm = post.sigma_posterior_mean().map_err(to_py)?;
    d.set_item(
        "sigma_posterior_mean",
        (0..sm.nrows())
            .map(|i| (0..sm.ncols()).map(|j| sm[(i, j)]).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("log_marginal_likelihood", post.log_marginal_likelihood())?;
    Ok(d)
}

/// Posterior draws of Cholesky-orthogonalized impulse responses from the
/// Minnesota-NIW BVAR: returns a list [draw][horizon][variable][shock].
///
/// Raw draws are returned exactly so credible-band coverage is
/// configurable by construction: form pointwise bands with numpy
/// quantiles across the draw axis, choosing the quantile pair to match
/// the stated coverage — e.g. a 90% band is
/// `np.quantile(draws, [0.05, 0.95], axis=0)`, a 68% band
/// `np.quantile(draws, [0.16, 0.84], axis=0)`.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, horizon = 16, n_draws = 500, seed = 0, lambda0 = 100.0, lambda1 = 0.2, lambda3 = 1.0, delta = 0.0, cumulative = false))]
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
    cumulative: bool,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let a = data.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let prior =
        tsecon_bayes::MinnesotaNiwPrior::new(m.as_ref(), lags, lambda0, lambda1, lambda3, delta)
            .map_err(to_py)?;
    let post = prior.posterior(m.as_ref()).map_err(to_py)?;
    let mut stream = tsecon_rng::Stream::new(seed);
    let draws = post
        .irf_draws(n_draws, horizon, &mut stream)
        .map_err(to_py)?;
    let mut out: Vec<Vec<Vec<Vec<f64>>>> = draws
        .iter()
        .map(|dr| dr.iter().map(mat_to_vec2_bayes).collect())
        .collect();
    if cumulative {
        // Cumulate WITHIN each draw, then the caller's quantiles across draws
        // give correctly cumulated credible bands (the summed responses are
        // correlated across horizons, so cumulating the bands would be wrong).
        for draw in out.iter_mut() {
            for h in 1..draw.len() {
                for i in 0..draw[h].len() {
                    for j in 0..draw[h][i].len() {
                        draw[h][i][j] += draw[h - 1][i][j];
                    }
                }
            }
        }
    }
    pyo3::types::PyList::new(py, out.iter().cloned())
}

fn mat_to_vec2_bayes(m: &tsecon_var::tsecon_linalg::faer::Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
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
    d.set_item(
        "rhat",
        tsecon_bayes::convergence::rhat_rank(m.as_ref()).map_err(to_py)?,
    )?;
    d.set_item(
        "ess_bulk",
        tsecon_bayes::convergence::ess_bulk(m.as_ref()).map_err(to_py)?,
    )?;
    d.set_item(
        "ess_tail",
        tsecon_bayes::convergence::ess_tail(m.as_ref()).map_err(to_py)?,
    )?;
    Ok(d)
}

/// Fit an ARIMA(p,d,q) by exact Gaussian maximum likelihood on the
/// state-space engine (Monahan-transformed L-BFGS with Nelder-Mead
/// polish; Hannan-Rissanen starting values). `d > 0` uses simple
/// differencing (the statsmodels simple_differencing=True convention),
/// and forecasts are undifferenced with exact cumulative variance.
///
/// With `forecast_steps > 0`, `conf_alpha` (default None) additionally
/// returns `forecast_lower`/`forecast_upper`: the symmetric Gaussian
/// `1 - conf_alpha` intervals `mean +/- z_{1-conf_alpha/2} * se`
/// (statsmodels `get_forecast(...).conf_int(alpha)` convention; e.g.
/// `conf_alpha=0.05` gives 95% bands with z = 1.96). Standard errors
/// reflect innovation and filtering uncertainty only (parameters
/// treated as known).
#[pyfunction]
#[pyo3(signature = (y, p = 1, d = 0, q = 0, constant = true, forecast_steps = 0, conf_alpha = None))]
#[allow(clippy::too_many_arguments)]
fn arima_fit<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    p: usize,
    d: usize,
    q: usize,
    constant: bool,
    forecast_steps: usize,
    conf_alpha: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if conf_alpha.is_some() && forecast_steps == 0 {
        return Err(PyValueError::new_err(
            "conf_alpha requires forecast_steps >= 1 (there is no forecast to band)",
        ));
    }
    let spec = tsecon_arima::ArimaSpec::new(p, d, q)
        .map_err(to_py)?
        .with_constant(constant);
    let r = spec.fit(y.as_slice()?).map_err(to_py)?;
    let dct = PyDict::new(py);
    dct.set_item("params", r.params().to_vec().into_pyarray(py))?;
    dct.set_item("param_names", r.param_names().to_vec())?;
    dct.set_item("loglik", r.loglik)?;
    dct.set_item("aic", r.aic)?;
    dct.set_item("bic", r.bic)?;
    if forecast_steps > 0 {
        let fc = r.forecast(forecast_steps).map_err(to_py)?;
        if let Some(alpha) = conf_alpha {
            let ci = fc.conf_int(alpha).map_err(to_py)?;
            let (lower, upper): (Vec<f64>, Vec<f64>) = ci.into_iter().unzip();
            dct.set_item("forecast_lower", lower.into_pyarray(py))?;
            dct.set_item("forecast_upper", upper.into_pyarray(py))?;
            dct.set_item("conf_alpha", alpha)?;
        }
        dct.set_item("forecast_mean", fc.mean.into_pyarray(py))?;
        dct.set_item("forecast_se", fc.se.into_pyarray(py))?;
    }
    dct.set_item("residuals", r.residuals().map_err(to_py)?.into_pyarray(py))?;
    Ok(dct)
}

/// Local projection impulse responses (Jordà 2005).
///
/// `se`: "lag_augmented" (Montiel Olea-Plagborg-Møller 2021, the default) or
/// "hac" (Newey-West; `maxlags=None` grows with the horizon). `cumulative`
/// regresses the cumulated outcome (Ramey-Zubairy). Returns per-horizon irf
/// and standard errors.
#[pyfunction]
#[pyo3(signature = (y, shock, horizons = 12, n_lag_controls = 4, se = "lag_augmented", maxlags = None, cumulative = false))]
#[allow(clippy::too_many_arguments)]
fn lp<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    shock: PyReadonlyArray1<'py, f64>,
    horizons: usize,
    n_lag_controls: usize,
    se: &str,
    maxlags: Option<usize>,
    cumulative: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut spec = tsecon_lp::LpSpec::new(horizons, n_lag_controls).cumulative(cumulative);
    spec = match se {
        "lag_augmented" => spec,
        "hac" => spec.with_hac(maxlags),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown se {other:?}; expected \"lag_augmented\" or \"hac\""
            )))
        }
    };
    let r = tsecon_lp::lp(y.as_slice()?, shock.as_slice()?, spec).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item(
        "horizons",
        r.horizons
            .iter()
            .map(|&h| h as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("irf", r.irf.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    Ok(d)
}

/// LP-IV: instrumental-variable local projections (Stock-Watson 2018,
/// Ramey-Zubairy 2018). The `impulse` is instrumented by `instrument`;
/// kernel-HAC standard errors match linearmodels IV2SLS. Returns per-horizon
/// irf, se, and the first-stage effective F diagnostic.
#[pyfunction]
#[pyo3(signature = (y, impulse, instrument, horizons = 8, n_lag_controls = 4, cumulative = false))]
fn lp_iv<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    impulse: PyReadonlyArray1<'py, f64>,
    instrument: PyReadonlyArray1<'py, f64>,
    horizons: usize,
    n_lag_controls: usize,
    cumulative: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = tsecon_lp::LpSpec::new(horizons, n_lag_controls).cumulative(cumulative);
    let r = tsecon_lp::lp_iv(
        y.as_slice()?,
        impulse.as_slice()?,
        instrument.as_slice()?,
        spec,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item(
        "horizons",
        r.horizons
            .iter()
            .map(|&h| h as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("irf", r.irf.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    d.set_item("first_stage_f", r.first_stage_f.into_pyarray(py))?;
    Ok(d)
}

fn to_faer(x: &numpy::PyReadonlyArray2<'_, f64>) -> tsecon_var::tsecon_linalg::faer::Mat<f64> {
    let a = x.as_array();
    tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
}

/// Ridge regression (closed form). Minimizes ||y - Xb||^2 + alpha*||b||^2,
/// matching scikit-learn's `Ridge` objective. Add your own intercept column
/// to X if you want one.
#[pyfunction]
fn ridge<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m = to_faer(&x);
    let coef = tsecon_ml::ridge(m.as_ref(), y.as_slice()?, alpha).map_err(to_py)?;
    Ok(coef.into_pyarray(py))
}

/// Elastic-net regression via coordinate descent. Minimizes
/// (1/2n)||y-Xb||^2 + alpha*l1_ratio*||b||_1 + (alpha/2)(1-l1_ratio)||b||^2,
/// matching scikit-learn. `l1_ratio=1.0` is the lasso. Returns coefficients,
/// iterations, and the final max coefficient change.
#[pyfunction]
#[pyo3(signature = (x, y, alpha, l1_ratio = 0.5, tol = 1e-8, max_iter = 100000))]
fn elastic_net<'py>(
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
    let fit =
        tsecon_ml::elastic_net(m.as_ref(), y.as_slice()?, alpha, l1_ratio, opts).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("coef", fit.coef.into_pyarray(py))?;
    d.set_item("n_iter", fit.n_iter)?;
    d.set_item("max_change", fit.max_change)?;
    Ok(d)
}

/// Lasso regression (elastic net with l1_ratio = 1.0).
#[pyfunction]
#[pyo3(signature = (x, y, alpha, tol = 1e-8, max_iter = 100000))]
fn lasso<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    elastic_net(py, x, y, alpha, 1.0, tol, max_iter)
}

/// Sign-restricted Bayesian SVAR (Uhlig 2005; Rubio-Ramirez-Waggoner-Zha
/// 2010) on the Minnesota-NIW posterior.
///
/// `restrictions` is a list of `(variable, shock, horizon, sign)` tuples with
/// `sign` in {"+", "-"}. Returns, per (horizon, variable, shock), the
/// identified-set envelope (`set_min`/`set_max`) and posterior `quantiles` at
/// `probs = [0.05, 0.16, 0.50, 0.84, 0.95]` (median + 68/90% credible bands),
/// plus the mandatory acceptance `diagnostics` — in set-identified settings
/// the diagnostics are the inference.
#[pyfunction]
#[pyo3(signature = (data, restrictions, lags = 2, horizon = 12, n_draws = 500, max_tries = 400, seed = 0, lambda1 = 0.2))]
#[allow(clippy::too_many_arguments)]
fn sign_restricted_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    restrictions: Vec<(usize, usize, usize, String)>,
    lags: usize,
    horizon: usize,
    n_draws: usize,
    max_tries: usize,
    seed: u64,
    lambda1: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_ident::{Sign, SignRestriction, SignRestrictionSet, SignSampler};
    let a = data.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let n_vars = a.ncols();
    let prior = tsecon_bayes::MinnesotaNiwPrior::new(m.as_ref(), lags, 100.0, lambda1, 1.0, 0.0)
        .map_err(to_py)?;
    let posterior = prior.posterior(m.as_ref()).map_err(to_py)?;

    let mut rs = Vec::with_capacity(restrictions.len());
    for (v, s, h, sign) in restrictions {
        let sg = match sign.as_str() {
            "+" | "positive" => Sign::Positive,
            "-" | "negative" => Sign::Negative,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sign {other:?}; expected \"+\" or \"-\""
                )))
            }
        };
        rs.push(SignRestriction::at(v, s, h, sg));
    }
    let restr = SignRestrictionSet::new(rs, n_vars, horizon).map_err(to_py)?;
    let result = SignSampler::new(horizon, n_draws, max_tries)
        .map_err(to_py)?
        .run(&posterior, &restr, seed)
        .map_err(to_py)?;

    let summary = result.summary();
    let hs = horizon + 1;
    // quantiles[h][var][shock][prob], set_min/set_max[h][var][shock]
    let mut quantiles = vec![vec![vec![Vec::<f64>::new(); n_vars]; n_vars]; hs];
    let mut set_min = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
    let mut set_max = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
    for h in 0..hs {
        for i in 0..n_vars {
            for j in 0..n_vars {
                let bp = summary.point(i, j, h).map_err(to_py)?;
                quantiles[h][i][j] = bp.quantiles.clone();
                set_min[h][i][j] = bp.min;
                set_max[h][i][j] = bp.max;
            }
        }
    }
    let d = PyDict::new(py);
    d.set_item("probs", summary.probs().to_vec())?;
    d.set_item("quantiles", quantiles)?;
    d.set_item("set_min", set_min)?;
    d.set_item("set_max", set_max)?;
    let diag = result.diagnostics();
    let dd = PyDict::new(py);
    dd.set_item("posterior_draws_used", diag.posterior_draws_used)?;
    dd.set_item("rotations_tried", diag.rotations_tried)?;
    dd.set_item("accepted", diag.accepted)?;
    dd.set_item("acceptance_rate", diag.acceptance_rate)?;
    d.set_item("diagnostics", dd)?;
    Ok(d)
}

fn panel_se(se_type: &str, bandwidth: f64) -> PyResult<tsecon_panel::PanelSeType> {
    use tsecon_panel::PanelSeType::*;
    match se_type {
        "nonrobust" => Ok(NonRobust),
        "cluster" | "cluster_entity" => Ok(ClusterEntity),
        "driscoll_kraay" | "dk" => Ok(DriscollKraay { bandwidth }),
        other => Err(PyValueError::new_err(format!(
            "unknown se_type {other:?}; expected nonrobust/cluster/driscoll_kraay"
        ))),
    }
}

/// Fixed-effects (within) panel OLS with panel-robust standard errors.
///
/// `outcome` is `N x T`; `regressors` is `k x N x T`. `se_type`:
/// "nonrobust", "cluster" (by entity), or "driscoll_kraay" (uses
/// `bandwidth`). Matches linearmodels PanelOLS conventions.
#[pyfunction]
#[pyo3(signature = (outcome, regressors, se_type = "cluster", bandwidth = 4.0))]
fn panel_fe<'py>(
    py: Python<'py>,
    outcome: numpy::PyReadonlyArray2<'py, f64>,
    regressors: numpy::PyReadonlyArray3<'py, f64>,
    se_type: &str,
    bandwidth: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    let o = outcome.as_array();
    let outcome_m = Mat::from_fn(o.nrows(), o.ncols(), |i, j| o[(i, j)]);
    let r = regressors.as_array();
    let (k, n, t) = (r.shape()[0], r.shape()[1], r.shape()[2]);
    let regs: Vec<(String, Mat<f64>)> = (0..k)
        .map(|c| (format!("x{c}"), Mat::from_fn(n, t, |i, j| r[[c, i, j]])))
        .collect();
    let data = tsecon_panel::PanelData::balanced(outcome_m, regs).map_err(to_py)?;
    let fit = tsecon_panel::panel_ols_fe(&data).map_err(to_py)?;
    let inf = fit
        .inference(panel_se(se_type, bandwidth)?)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("params", fit.params.clone().into_pyarray(py))?;
    d.set_item("bse", inf.bse.into_pyarray(py))?;
    d.set_item("tvalues", inf.tvalues.into_pyarray(py))?;
    d.set_item("se_type", se_type)?;
    Ok(d)
}

/// Panel local projection of a common shock (Jordà 2005 for panels), with
/// fixed effects and panel-robust standard errors, the Ramey-Zubairy
/// `cumulative` option, and the Dhaene-Jochmans half-panel `jackknife`
/// Nickell-bias correction.
#[pyfunction]
#[pyo3(signature = (outcome, shock, horizon = 8, n_lag_controls = 2, se_type = "driscoll_kraay", bandwidth = 4.0, cumulative = false, jackknife = false))]
#[allow(clippy::too_many_arguments)]
fn panel_lp<'py>(
    py: Python<'py>,
    outcome: numpy::PyReadonlyArray2<'py, f64>,
    shock: PyReadonlyArray1<'py, f64>,
    horizon: usize,
    n_lag_controls: usize,
    se_type: &str,
    bandwidth: f64,
    cumulative: bool,
    jackknife: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    let o = outcome.as_array();
    let outcome_m = Mat::from_fn(o.nrows(), o.ncols(), |i, j| o[(i, j)]);
    let data = tsecon_panel::PanelData::balanced(outcome_m, vec![]).map_err(to_py)?;
    let mut cfg =
        tsecon_panel::PanelLpConfig::new(horizon, n_lag_controls, panel_se(se_type, bandwidth)?);
    cfg.cumulative = cumulative;
    cfg.jackknife = jackknife;
    let r = tsecon_panel::panel_lp(&data, shock.as_slice()?, &cfg).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("irf", r.irf.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    d.set_item(
        "nobs",
        r.nobs
            .iter()
            .map(|&x| x as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

/// Clark-West test for nested-model equal predictive accuracy (Clark-West
/// 2007). One-sided; the null is that the small (nested) model is as good.
#[pyfunction]
#[pyo3(signature = (e_small, e_large, yhat_small, yhat_large, lrv_lags = 0))]
fn cw_test<'py>(
    py: Python<'py>,
    e_small: PyReadonlyArray1<'py, f64>,
    e_large: PyReadonlyArray1<'py, f64>,
    yhat_small: PyReadonlyArray1<'py, f64>,
    yhat_large: PyReadonlyArray1<'py, f64>,
    lrv_lags: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_forecast::cw_test(
        e_small.as_slice()?,
        e_large.as_slice()?,
        yhat_small.as_slice()?,
        yhat_large.as_slice()?,
        lrv_lags,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("cw_stat", r.cw_stat)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("mean_adj_diff", r.mean_adj_diff)?;
    Ok(d)
}

/// Giacomini-White unconditional test of equal predictive ability
/// (Giacomini-White 2006), chi-squared(1) on a loss differential.
#[pyfunction]
#[pyo3(signature = (loss1, loss2, lrv_lags = 0))]
fn gw_test<'py>(
    py: Python<'py>,
    loss1: PyReadonlyArray1<'py, f64>,
    loss2: PyReadonlyArray1<'py, f64>,
    lrv_lags: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r =
        tsecon_forecast::gw_test(loss1.as_slice()?, loss2.as_slice()?, lrv_lags).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("gw_stat", r.gw_stat)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("df", r.df)?;
    Ok(d)
}

fn spectral_window(w: &str) -> PyResult<tsecon_spectral::Window> {
    match w {
        "boxcar" => Ok(tsecon_spectral::Window::Boxcar),
        "hann" => Ok(tsecon_spectral::Window::Hann),
        other => Err(PyValueError::new_err(format!(
            "unknown window {other:?}; expected \"boxcar\" or \"hann\""
        ))),
    }
}

fn spectral_detrend(d: &str) -> PyResult<tsecon_spectral::Detrend> {
    match d {
        "none" => Ok(tsecon_spectral::Detrend::None),
        "constant" => Ok(tsecon_spectral::Detrend::Constant),
        "linear" => Ok(tsecon_spectral::Detrend::Linear),
        other => Err(PyValueError::new_err(format!(
            "unknown detrend {other:?}; expected none/constant/linear"
        ))),
    }
}

/// Periodogram power spectral density (one FFT). Matches
/// `scipy.signal.periodogram` to ~1e-15. Returns `freqs` and `psd`.
#[pyfunction]
#[pyo3(signature = (x, fs = 1.0, window = "boxcar", detrend = "none"))]
fn periodogram<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    fs: f64,
    window: &str,
    detrend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_spectral::periodogram(
        x.as_slice()?,
        fs,
        spectral_window(window)?,
        tsecon_spectral::Scaling::Density,
        spectral_detrend(detrend)?,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("freqs", r.freqs.into_pyarray(py))?;
    d.set_item("psd", r.psd.into_pyarray(py))?;
    Ok(d)
}

/// Welch's averaged-periodogram PSD (periodic Hann, 50% overlap by
/// default). Matches `scipy.signal.welch`. Returns `freqs` and `psd`.
#[pyfunction]
#[pyo3(signature = (x, nperseg = 256, fs = 1.0, noverlap = None, window = "hann", detrend = "none"))]
fn welch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    nperseg: usize,
    fs: f64,
    noverlap: Option<usize>,
    window: &str,
    detrend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_spectral::welch(
        x.as_slice()?,
        fs,
        nperseg,
        noverlap,
        spectral_window(window)?,
        tsecon_spectral::Scaling::Density,
        spectral_detrend(detrend)?,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("freqs", r.freqs.into_pyarray(py))?;
    d.set_item("psd", r.psd.into_pyarray(py))?;
    Ok(d)
}

/// Magnitude-squared coherence between two series via Welch cross-spectra.
/// Matches `scipy.signal.coherence`. Returns `freqs` and `coherence` in [0, 1].
#[pyfunction]
#[pyo3(signature = (x, y, nperseg = 256, fs = 1.0, noverlap = None, window = "hann", detrend = "none"))]
#[allow(clippy::too_many_arguments)]
fn coherence<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    nperseg: usize,
    fs: f64,
    noverlap: Option<usize>,
    window: &str,
    detrend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_spectral::coherence(
        x.as_slice()?,
        y.as_slice()?,
        fs,
        nperseg,
        noverlap,
        spectral_window(window)?,
        spectral_detrend(detrend)?,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("freqs", r.freqs.into_pyarray(py))?;
    d.set_item("coherence", r.coherence.into_pyarray(py))?;
    Ok(d)
}

fn data_to_faer(
    data: &numpy::PyReadonlyArray2<'_, f64>,
) -> tsecon_var::tsecon_linalg::faer::Mat<f64> {
    let a = data.as_array();
    tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
}

/// Johansen cointegration test (Johansen 1991). `data` is T x k (rows are
/// observations, oldest first). Returns eigenvalues, trace and
/// max-eigenvalue statistics with their 90/95/99% critical values, and the
/// selected rank at 5%. Matches statsmodels `coint_johansen` (det_order=0).
#[pyfunction]
#[pyo3(signature = (data, k_ar_diff = 1))]
fn johansen<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    k_ar_diff: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let m = data_to_faer(&data);
    let r = tsecon_coint::johansen(m.as_ref(), k_ar_diff).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("eig", r.eig.clone().into_pyarray(py))?;
    d.set_item("trace_stat", r.trace_stat.clone().into_pyarray(py))?;
    d.set_item("max_eig_stat", r.max_eig_stat.clone().into_pyarray(py))?;
    let tc: Vec<Vec<f64>> = r.trace_crit.iter().map(|c| c.to_vec()).collect();
    let mc: Vec<Vec<f64>> = r.max_eig_crit.iter().map(|c| c.to_vec()).collect();
    d.set_item("trace_crit_90_95_99", tc)?;
    d.set_item("max_eig_crit_90_95_99", mc)?;
    d.set_item(
        "rank_trace_5pct",
        r.rank_trace(tsecon_coint::SignificanceLevel::Five),
    )?;
    d.set_item(
        "rank_max_eig_5pct",
        r.rank_max_eig(tsecon_coint::SignificanceLevel::Five),
    )?;
    Ok(d)
}

/// VECM maximum-likelihood estimation at a given cointegrating rank
/// (Johansen). Returns the loadings alpha, cointegrating vectors beta
/// (normalized beta[:r,:r] = I), short-run Gamma, and the log-likelihood.
/// Matches statsmodels VECM.
#[pyfunction]
#[pyo3(signature = (data, k_ar_diff = 1, coint_rank = 1))]
fn vecm<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    k_ar_diff: usize,
    coint_rank: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let m = data_to_faer(&data);
    let r = tsecon_coint::fit_vecm(m.as_ref(), k_ar_diff, coint_rank).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("alpha", mat_to_vec2_bayes(&r.alpha))?;
    d.set_item("beta", mat_to_vec2_bayes(&r.beta))?;
    d.set_item("gamma", mat_to_vec2_bayes(&r.gamma))?;
    d.set_item("sigma_u", mat_to_vec2_bayes(&r.sigma_u))?;
    d.set_item("llf", r.llf)?;
    Ok(d)
}

/// Markov-switching autoregression (Hamilton 1989), fitted by EM.
///
/// Estimates a `k_regimes`-state MS-AR(`order`) with a common AR and
/// (optionally) switching variances, starting from an internal
/// quantile-based initialization. Returns the estimated transition matrix,
/// per-regime means and variances, log-likelihood, smoothed regime
/// probabilities, the MAP regime path, and expected regime durations.
#[pyfunction]
#[pyo3(signature = (y, k_regimes = 2, order = 1, switching_variance = true, max_iter = 500, tol = 1e-6))]
fn markov_switching_ar<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    k_regimes: usize,
    order: usize,
    switching_variance: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_regime::{MarkovSwitchingAr, MsarParams, MsarSpec};
    let ys = y.as_slice()?;
    let spec = MsarSpec {
        k_regimes,
        order,
        switching_ar: false,
        switching_variance,
    };
    let model = MarkovSwitchingAr::new(ys, spec).map_err(to_py)?;

    // A quantile-based default start: regime means at evenly spaced quantiles,
    // a diagonal-heavy transition, a shared small AR, and the sample variance.
    let mut sorted: Vec<f64> = ys.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let means: Vec<f64> = (0..k_regimes)
        .map(|r| sorted[((r as f64 + 0.5) / k_regimes as f64 * n as f64) as usize % n])
        .collect();
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let var_y = ys.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64;
    let transition: Vec<Vec<f64>> = (0..k_regimes)
        .map(|i| {
            (0..k_regimes)
                .map(|j| {
                    if i == j {
                        0.9
                    } else {
                        0.1 / (k_regimes as f64 - 1.0)
                    }
                })
                .collect()
        })
        .collect();
    let ar = vec![vec![0.1; order]];
    let variances = if switching_variance {
        vec![var_y; k_regimes]
    } else {
        vec![var_y]
    };
    let start = MsarParams::new(transition, means, ar, variances).map_err(to_py)?;

    let fit = model.fit(&start, max_iter, tol).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("transition", fit.params.transition_matrix())?;
    d.set_item("means", fit.params.means().to_vec())?;
    d.set_item("variances", fit.params.variances().to_vec())?;
    d.set_item("loglik", fit.loglik)?;
    d.set_item("iterations", fit.iterations)?;
    d.set_item("converged", fit.converged)?;
    d.set_item("expected_durations", fit.params.expected_durations())?;
    let prob1: Vec<f64> = fit.smoothed_prob.iter().map(|p| p[k_regimes - 1]).collect();
    d.set_item("smoothed_prob_last_regime", prob1.into_pyarray(py))?;
    d.set_item(
        "regimes",
        fit.classified()
            .iter()
            .map(|&r| r as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

/// MIDAS weight function (normalized to sum 1). `scheme`: "exp_almon"
/// (uses theta1, theta2) or "beta" (uses theta1, theta2 as the two shape
/// parameters). `k` is the number of high-frequency lags.
#[pyfunction]
#[pyo3(signature = (scheme, theta1, theta2, k))]
fn midas_weights<'py>(
    py: Python<'py>,
    scheme: &str,
    theta1: f64,
    theta2: f64,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let w = match scheme {
        "exp_almon" => tsecon_midas::exp_almon_weights(theta1, theta2, k),
        "beta" => tsecon_midas::beta_weights(theta1, theta2, k),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scheme {other:?}; expected \"exp_almon\" or \"beta\""
            )))
        }
    }
    .map_err(to_py)?;
    Ok(w.into_pyarray(py))
}

/// U-MIDAS: unrestricted mixed-frequency regression (= OLS of `y` on a
/// constant plus the `hf_lags` columns). `hf_lags` is `nobs x K` (each
/// column a high-frequency lag). Returns params, HAC standard errors, and R².
#[pyfunction]
#[pyo3(signature = (y, hf_lags, se_type = "hac", maxlags = None))]
fn umidas<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    hf_lags: numpy::PyReadonlyArray2<'py, f64>,
    se_type: &str,
    maxlags: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let a = hf_lags.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let se = match se_type {
        "nonrobust" => tsecon_midas::SeType::NonRobust,
        "hac" => tsecon_midas::SeType::Hac {
            kernel: tsecon_hac::Kernel::Bartlett,
            bandwidth: maxlags.unwrap_or_else(|| tsecon_hac::newey_west_maxlags(a.nrows())) as f64,
            use_correction: true,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown se_type {other:?}; expected nonrobust/hac"
            )))
        }
    };
    let r = tsecon_midas::umidas(y.as_slice()?, &cols, se).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("params", r.params.into_pyarray(py))?;
    d.set_item("bse", r.bse.into_pyarray(py))?;
    d.set_item("rsquared", r.rsquared)?;
    Ok(d)
}

fn garch11_spec() -> tsecon_garch::GarchSpec {
    tsecon_garch::GarchSpec {
        mean: tsecon_garch::MeanSpec::Zero,
        vol: tsecon_garch::VolSpec::Garch { p: 1, q: 1 },
        dist: tsecon_garch::DistSpec::Normal,
    }
}

fn returns_to_series(r: &numpy::PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    let a = r.as_array();
    (0..a.ncols()).map(|j| a.column(j).to_vec()).collect()
}

/// CCC-GARCH (Bollerslev 1990): a GARCH(1,1) per series with a constant
/// conditional correlation. `returns` is `T x k`. Returns the correlation
/// matrix and the log-likelihood.
#[pyfunction]
fn ccc_garch<'py>(
    py: Python<'py>,
    returns: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let series = returns_to_series(&returns);
    let fit = tsecon_mgarch::CccGarch::new(garch11_spec())
        .fit(&series)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("correlation", mat_to_vec2_bayes(&fit.correlation))?;
    d.set_item("loglik", fit.loglik)?;
    Ok(d)
}

/// DCC-GARCH (Engle 2002): GARCH(1,1) per series with dynamic conditional
/// correlations Q_t = (1-a-b)Qbar + a z z' + b Q_{t-1}. `returns` is `T x k`.
/// Returns the DCC parameters (a, b), the targeted Qbar, the log-likelihood,
/// convergence, and the final-period correlation matrix.
#[pyfunction]
fn dcc_garch<'py>(
    py: Python<'py>,
    returns: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let series = returns_to_series(&returns);
    let fit = tsecon_mgarch::DccGarch::new(garch11_spec())
        .fit(&series)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("a", fit.a)?;
    d.set_item("b", fit.b)?;
    d.set_item("qbar", mat_to_vec2_bayes(&fit.qbar))?;
    d.set_item("loglik", fit.loglik)?;
    d.set_item("converged", fit.converged)?;
    if let Some(last) = fit.correlation_path.last() {
        d.set_item("correlation_last", mat_to_vec2_bayes(last))?;
    }
    Ok(d)
}

/// Realized volatility measures on a vector of high-frequency returns.
///
/// Returns realized variance (`rv`), bipower variation (`bipower`, the
/// jump-robust integrated-variance estimator of Barndorff-Nielsen &
/// Shephard 2004), and the truncated jump component (`jump = max(rv -
/// bipower, 0)`). Validated against the documented BNS formulas at 1e-12.
#[pyfunction]
fn realized_measures<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = returns.as_slice()?;
    let rv = tsecon_realized::realized_variance(r).map_err(to_py)?;
    let bv = tsecon_realized::bipower_variation(r).map_err(to_py)?;
    let jump = tsecon_realized::jump_component(r).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("rv", rv)?;
    d.set_item("bipower", bv)?;
    d.set_item("jump", jump)?;
    Ok(d)
}

/// HAR-RV heterogeneous autoregression of realized variance (Corsi 2009).
///
/// Regresses `RV_t` on `[const, RV_{t-1}, RV_week, RV_month]`, where the
/// weekly/monthly regressors are trailing averages known at `t-1`. The
/// `variant` transforms the series first: `"level"`, `"log"`, or `"sqrt"`.
/// Standard errors are Newey-West HAC with `hac_maxlags` lags. Matches
/// statsmodels OLS-HAC at 1e-8.
#[pyfunction]
#[pyo3(signature = (rv, start = 22, variant = "level", hac_maxlags = 5, use_correction = false))]
fn har_rv<'py>(
    py: Python<'py>,
    rv: PyReadonlyArray1<'py, f64>,
    start: usize,
    variant: &str,
    hac_maxlags: usize,
    use_correction: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_realized::{HarConfig, HarVariant};
    let v = match variant {
        "level" => HarVariant::Level,
        "log" => HarVariant::Log,
        "sqrt" => HarVariant::Sqrt,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown variant {other:?}; expected \"level\", \"log\", or \"sqrt\""
            )))
        }
    };
    let cfg = HarConfig {
        start,
        variant: v,
        hac_maxlags,
        use_correction,
    };
    let fit = tsecon_realized::har_rv(rv.as_slice()?, &cfg).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("params", fit.params.into_pyarray(py))?;
    d.set_item("bse", fit.bse.into_pyarray(py))?;
    d.set_item("tvalues", fit.tvalues.into_pyarray(py))?;
    d.set_item("rsquared", fit.rsquared)?;
    d.set_item("nobs", fit.nobs)?;
    Ok(d)
}

/// Diebold-Yilmaz connectedness from a VAR's generalized FEVD.
///
/// Fits a VAR(`lags`, trend) then builds the spillover table from the
/// row-normalized Pesaran-Shin GFEVD at the given `horizon`: total
/// (system-wide spillover index), directional `to_others`/`from_others`,
/// `net`, and the antisymmetric `pairwise_net` matrix — all in percent.
/// Matches the documented GFEVD golden to ~1e-13.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, horizon = 10, trend = "c"))]
fn connectedness<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    trend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let res = var_results(&data, lags, trend)?;
    let table = tsecon_connect::ConnectednessTable::from_var(&res, horizon).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("total", table.total)?;
    d.set_item("to_others", table.to_others.clone().into_pyarray(py))?;
    d.set_item("from_others", table.from_others.clone().into_pyarray(py))?;
    d.set_item("net", table.net.clone().into_pyarray(py))?;
    d.set_item("gfevd", mat_to_vec2(&table.gfevd))?;
    d.set_item("pairwise_net", mat_to_vec2(&table.pairwise_net))?;
    Ok(d)
}

/// Static approximate factor model (PCA) with Bai-Ng factor selection.
///
/// Extracts `n_factors` principal components from `data` (T x N; the
/// caller standardizes if desired) via SVD: `factors` (T x r), `loadings`
/// (N x r), and the full `eigenvalues` vector. Also runs the Bai-Ng (2002)
/// information criteria up to `kmax` and returns the selected factor counts
/// (`icp1`/`icp2`/`pcp1`/`pcp2`). Matches numpy PCA to 1e-6 (up to sign).
#[pyfunction]
#[pyo3(signature = (data, n_factors = 2, kmax = 8))]
fn factor_model<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    n_factors: usize,
    kmax: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    let a = data.as_array();
    let (n, big_n) = (a.nrows(), a.ncols());
    let m = Mat::from_fn(n, big_n, |i, j| a[(i, j)]);
    let model = tsecon_favar::FactorModel::fit(m.as_ref()).map_err(to_py)?;
    let factors = model.factors(n_factors).map_err(to_py)?;
    let loadings = model.loadings(n_factors).map_err(to_py)?;
    let eigenvalues = model.eigenvalues().to_vec();
    let bn = tsecon_favar::bai_ng(&eigenvalues, n, big_n, kmax).map_err(to_py)?;
    let (er, er_ratios) = tsecon_favar::eigenvalue_ratio(&eigenvalues, kmax).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("factors", mat_to_vec2(&factors))?;
    d.set_item("loadings", mat_to_vec2(&loadings))?;
    d.set_item("eigenvalues", eigenvalues.into_pyarray(py))?;
    d.set_item("icp1", bn.icp1_hat)?;
    d.set_item("icp2", bn.icp2_hat)?;
    d.set_item("pcp1", bn.pcp1_hat)?;
    d.set_item("pcp2", bn.pcp2_hat)?;
    // Ahn-Horenstein (2013) eigenvalue ratio: robust in small cross-sections
    // where the Bai-Ng criteria over-select.
    d.set_item("er", er)?;
    d.set_item("er_ratios", er_ratios.into_pyarray(py))?;
    Ok(d)
}

/// Linear IV-GMM (Hansen 1982) with a robust or HAC weighting matrix.
///
/// Estimates `y = X beta + u` where the columns of `X` may be endogenous,
/// using instrument matrix `Z` (which must include the exogenous regressor
/// columns — exogenous regressors instrument themselves). `method`:
/// `"2sls"` (one-step with the 2SLS weight), `"2step"` (two-step efficient),
/// or `"iterated"`. `weight`: `"robust"` (heteroskedasticity-robust White)
/// or `"hac"` (Newey-West at `bandwidth`). Returns `params`, robust-sandwich
/// `bse`, the parameter covariance `cov`, `residuals`, and — when the model
/// is over-identified — the Hansen `j_stat`/`j_pval`/`j_dof` test of the
/// over-identifying restrictions. Matches linearmodels IVGMM to machine
/// precision.
#[pyfunction]
#[pyo3(signature = (x, z, y, method = "2step", weight = "robust", bandwidth = 0.0,
                    tol = 1e-8, max_iter = 100))]
#[allow(clippy::too_many_arguments)]
fn iv_gmm<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    z: numpy::PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    method: &str,
    weight: &str,
    bandwidth: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_gmm::GmmWeight;
    // faer/ndarray are column-major-agnostic; the crate takes column vectors.
    let cols = |a: &numpy::PyReadonlyArray2<'_, f64>| -> Vec<Vec<f64>> {
        let arr = a.as_array();
        (0..arr.ncols())
            .map(|j| (0..arr.nrows()).map(|i| arr[(i, j)]).collect())
            .collect()
    };
    let (x_cols, z_cols) = (cols(&x), cols(&z));
    let yv = y.as_slice()?;
    let cov_weight = match weight {
        "robust" => GmmWeight::Robust,
        "hac" => GmmWeight::Hac {
            kernel: tsecon_hac::Kernel::Bartlett,
            bandwidth,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown weight {other:?}; expected \"robust\" or \"hac\""
            )))
        }
    };
    let fit = match method {
        "2sls" => tsecon_gmm::two_stage_least_squares(&x_cols, &z_cols, yv).map_err(to_py)?,
        "2step" => tsecon_gmm::two_step_gmm(&x_cols, &z_cols, yv, cov_weight).map_err(to_py)?,
        "iterated" => tsecon_gmm::iterated_gmm(&x_cols, &z_cols, yv, cov_weight, tol, max_iter)
            .map_err(to_py)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}; expected \"2sls\", \"2step\", or \"iterated\""
            )))
        }
    };
    let d = PyDict::new(py);
    d.set_item("params", fit.params.clone().into_pyarray(py))?;
    d.set_item("bse", fit.bse.clone().into_pyarray(py))?;
    d.set_item("residuals", fit.residuals.clone().into_pyarray(py))?;
    d.set_item("nobs", fit.nobs)?;
    d.set_item("nmoments", fit.nmoments)?;
    d.set_item("nparams", fit.nparams)?;
    d.set_item("steps", fit.steps)?;
    if let Some(j) = fit.jtest {
        d.set_item("j_stat", j.stat)?;
        d.set_item("j_dof", j.dof)?;
        d.set_item("j_pval", j.pval)?;
    }
    Ok(d)
}

/// Leakage-safe cross-validation splits for time-series / sequential data.
///
/// Returns a list of `{train, test}` index dicts. `scheme`:
/// - `"expanding"`: expanding-origin (recursive) CV; `train` is the first
///   training size, each fold forecasts the next `horizon` steps, advancing
///   by `step`.
/// - `"rolling"`: fixed-width rolling-origin CV; `train` is the window
///   width.
/// - `"purged_kfold"`: López de Prado purged K-fold with a `purge` gap and
///   an `embargo` after each test fold, to prevent train/test leakage from
///   serial correlation (`k` folds; `train` is ignored).
#[pyfunction]
#[pyo3(signature = (n, scheme = "expanding", train = 0, horizon = 1, step = 1,
                    k = 5, purge = 0, embargo = 0))]
#[allow(clippy::too_many_arguments)]
fn cv_splits<'py>(
    py: Python<'py>,
    n: usize,
    scheme: &str,
    train: usize,
    horizon: usize,
    step: usize,
    k: usize,
    purge: usize,
    embargo: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let splits = match scheme {
        "expanding" => {
            tsecon_ml::expanding_origin_splits(n, train, horizon, step).map_err(to_py)?
        }
        "rolling" => tsecon_ml::rolling_origin_splits(n, train, horizon, step).map_err(to_py)?,
        "purged_kfold" => tsecon_ml::purged_kfold_splits(n, k, purge, embargo).map_err(to_py)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scheme {other:?}; expected \"expanding\", \"rolling\", or \
                 \"purged_kfold\""
            )))
        }
    };
    splits
        .into_iter()
        .map(|s| {
            let d = PyDict::new(py);
            d.set_item("train", s.train)?;
            d.set_item("test", s.test)?;
            Ok(d)
        })
        .collect()
}

/// Adaptive LASSO of Zou (2006): a weighted-L1 penalty with data-driven
/// weights `w_j = 1 / |b_j^ols|^gamma`, which restores the oracle property
/// the plain lasso lacks. `alpha` is the overall penalty, `l1_ratio` mixes
/// L1/L2 (elastic-net weighting of the penalty), `gamma > 0` controls how
/// hard small OLS coefficients are penalized. Returns the coefficients and
/// convergence info.
#[pyfunction]
#[pyo3(signature = (x, y, alpha, l1_ratio = 1.0, gamma = 1.0, tol = 1e-7, max_iter = 100000))]
#[allow(clippy::too_many_arguments)]
fn adaptive_lasso<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    l1_ratio: f64,
    gamma: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let m = to_faer(&x);
    let opts = tsecon_ml::CoordDescentOptions { tol, max_iter };
    let fit = tsecon_ml::adaptive_lasso(m.as_ref(), y.as_slice()?, alpha, l1_ratio, gamma, opts)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("coef", fit.coef.into_pyarray(py))?;
    d.set_item("n_iter", fit.n_iter)?;
    d.set_item("max_change", fit.max_change)?;
    Ok(d)
}

/// Elastic-net regularization path over an automatic lambda grid.
///
/// Fits the penalized regression at `n_lambdas` values descending from the
/// smallest lambda that zeros all coefficients down to `eps` times it (the
/// glmnet convention), at fixed `l1_ratio` in (0, 1]. Returns the `lambdas`
/// grid, the `coefs` at each (one row per lambda), residual sums of squares
/// `rss`, degrees of freedom `df` (nonzero count), the `aic`/`bic` along the
/// path, and the `aic_best`/`bic_best` indices selecting the minimizing
/// lambda.
#[pyfunction]
#[pyo3(signature = (x, y, l1_ratio = 1.0, n_lambdas = 100, eps = 1e-3, tol = 1e-7, max_iter = 100000))]
#[allow(clippy::too_many_arguments)]
fn lasso_path<'py>(
    py: Python<'py>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    l1_ratio: f64,
    n_lambdas: usize,
    eps: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let m = to_faer(&x);
    let opts = tsecon_ml::PathOptions {
        n_lambdas,
        eps,
        cd: tsecon_ml::CoordDescentOptions { tol, max_iter },
    };
    let path =
        tsecon_ml::regularization_path(m.as_ref(), y.as_slice()?, l1_ratio, opts).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("lambdas", path.lambdas.clone().into_pyarray(py))?;
    d.set_item("coefs", path.coefs.clone())?;
    d.set_item("rss", path.rss.clone().into_pyarray(py))?;
    d.set_item("df", path.df.clone())?;
    d.set_item("aic", path.aic.clone().into_pyarray(py))?;
    d.set_item("bic", path.bic.clone().into_pyarray(py))?;
    d.set_item("aic_best", path.aic_best())?;
    d.set_item("bic_best", path.bic_best())?;
    Ok(d)
}

/// Pseudo-out-of-sample backtest over a rolling or expanding window.
///
/// Re-estimates `forecaster` along the series and evaluates horizons
/// `1..=horizon` at every origin. `window` is `"expanding"` (training set
/// grows from `train` observations) or `"rolling"` (fixed width `train`).
/// `refit_every` sets the refit cadence. Built-in forecasters: `"naive"`,
/// `"drift"`, `"mean"`, `"seasonal_naive"`, `"theta"` (`period` is used by
/// the seasonal ones). Returns the origin indices, per-horizon `forecasts`
/// and `targets`, and an `accuracy` table (ME/MSE/RMSE/MAE/MdAE, plus
/// MAPE/sMAPE/MASE/RMSSE where defined) whose scaled measures use the
/// first training window at `insample_period` — never the test sample.
#[pyfunction]
#[pyo3(signature = (y, window = "expanding", train = 20, horizon = 1, refit_every = 1,
                    forecaster = "naive", period = 1, insample_period = 1))]
#[allow(clippy::too_many_arguments)]
fn backtest<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    window: &str,
    train: usize,
    horizon: usize,
    refit_every: usize,
    forecaster: &str,
    period: usize,
    insample_period: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_forecast::{
        backtest::Window, drift, historical_mean, naive, seasonal_naive, theta_forecast, Backtest,
        ForecastError,
    };
    let win = match window {
        "expanding" => Window::Expanding { min_train: train },
        "rolling" => Window::Rolling { width: train },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown window {other:?}; expected \"expanding\" or \"rolling\""
            )))
        }
    };
    let bt = Backtest::new(win, horizon, refit_every).map_err(to_py)?;
    // Dispatch the forecaster string to a built-in point forecaster.
    let fc = forecaster.to_string();
    let point = |train: &[f64], h: usize| -> Result<Vec<f64>, ForecastError> {
        match fc.as_str() {
            "naive" => Ok(naive(train, h, 0.95)?.mean),
            "drift" => Ok(drift(train, h, 0.95)?.mean),
            "mean" => Ok(historical_mean(train, h, 0.95)?.mean),
            "seasonal_naive" => Ok(seasonal_naive(train, period, h, 0.95)?.mean),
            "theta" => Ok(theta_forecast(train, period, h)?.forecast),
            // Unreachable: the forecaster name is validated before `run`.
            _ => Err(ForecastError::NonFinite {
                what: "forecaster",
                index: 0,
                value: f64::NAN,
            }),
        }
    };
    if !matches!(
        fc.as_str(),
        "naive" | "drift" | "mean" | "seasonal_naive" | "theta"
    ) {
        return Err(PyValueError::new_err(format!(
            "unknown forecaster {forecaster:?}; expected one of naive, drift, mean, \
             seasonal_naive, theta"
        )));
    }
    let res = bt.run(y.as_slice()?, point).map_err(to_py)?;

    let mut forecasts = Vec::with_capacity(horizon);
    let mut targets = Vec::with_capacity(horizon);
    for h in 1..=horizon {
        forecasts.push(res.forecasts(h).map_err(to_py)?.to_vec());
        targets.push(res.targets(h).map_err(to_py)?.to_vec());
    }
    let table = res.accuracy_table(insample_period).map_err(to_py)?;
    let rows = table
        .iter()
        .map(|r| {
            let row = PyDict::new(py);
            row.set_item("name", &r.name)?;
            row.set_item("me", r.me)?;
            row.set_item("mse", r.mse)?;
            row.set_item("rmse", r.rmse)?;
            row.set_item("mae", r.mae)?;
            row.set_item("mdae", r.mdae)?;
            row.set_item("mape", r.mape)?;
            row.set_item("smape", r.smape)?;
            row.set_item("mase", r.mase)?;
            row.set_item("rmsse", r.rmsse)?;
            Ok(row)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let d = PyDict::new(py);
    d.set_item("origins", res.origins().to_vec())?;
    d.set_item("n_origins", res.n_origins())?;
    d.set_item("horizon", res.horizon())?;
    d.set_item("forecasts", forecasts)?;
    d.set_item("targets", targets)?;
    d.set_item("accuracy", rows)?;
    Ok(d)
}

/// Nelson-Siegel yield-curve fit (Diebold-Li 2006).
///
/// Cross-sectional OLS of `yields` on the three Nelson-Siegel loadings at
/// the given `decay` (lambda), recovering `[level, slope, curvature]`
/// factors. If `optimal_lambda` is true, `decay` is treated as a starting
/// value and the decay is estimated by NLS (profiling the linear factors
/// out). Returns `factors`, the fitted `lambda`, `residuals`, and centered
/// `rsquared` (matches statsmodels' constant-included R^2 to 1e-8).
#[pyfunction]
#[pyo3(signature = (maturities, yields, decay = 0.0609, optimal_lambda = false))]
fn nelson_siegel<'py>(
    py: Python<'py>,
    maturities: PyReadonlyArray1<'py, f64>,
    yields: PyReadonlyArray1<'py, f64>,
    decay: f64,
    optimal_lambda: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (mat, yld) = (maturities.as_slice()?, yields.as_slice()?);
    let fit = if optimal_lambda {
        tsecon_termstructure::fit_nelson_siegel_optimal_lambda(mat, yld, decay).map_err(to_py)?
    } else {
        tsecon_termstructure::fit_nelson_siegel(mat, yld, decay).map_err(to_py)?
    };
    let d = PyDict::new(py);
    d.set_item("level", fit.factors[0])?;
    d.set_item("slope", fit.factors[1])?;
    d.set_item("curvature", fit.factors[2])?;
    d.set_item("factors", fit.factors.to_vec().into_pyarray(py))?;
    d.set_item("lambda", fit.lambda)?;
    d.set_item("residuals", fit.residuals.into_pyarray(py))?;
    d.set_item("rsquared", fit.rsquared)?;
    Ok(d)
}

/// Svensson (1994) four-factor yield-curve fit.
///
/// The Nelson-Siegel extension with a second curvature term at decay
/// `lambda2`; cross-sectional OLS at fixed `lambda1`, `lambda2` returns
/// the four `factors` and centered `rsquared`. Nests Nelson-Siegel.
#[pyfunction]
fn svensson<'py>(
    py: Python<'py>,
    maturities: PyReadonlyArray1<'py, f64>,
    yields: PyReadonlyArray1<'py, f64>,
    lambda1: f64,
    lambda2: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (mat, yld) = (maturities.as_slice()?, yields.as_slice()?);
    let fit = tsecon_termstructure::fit_svensson(mat, yld, lambda1, lambda2).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("factors", fit.factors.to_vec().into_pyarray(py))?;
    d.set_item("lambda1", lambda1)?;
    d.set_item("lambda2", lambda2)?;
    d.set_item("residuals", fit.residuals.into_pyarray(py))?;
    d.set_item("rsquared", fit.rsquared)?;
    Ok(d)
}

/// Nonlinear GMM driver (Hansen 1982) minimizing `gbar(theta)' W gbar(theta)`
/// by the derivative-free Nelder-Mead simplex, with a Python moment function.
///
/// `moments_fn(theta)` is a Python callable mapping a parameter vector (passed
/// as a NumPy 1-D `float64` array) to the `n x m` matrix of per-observation
/// moment contributions (rows = observations, cols = moments); it may return a
/// NumPy 2-D array or a list of lists, and its shape must be the same at every
/// `theta`. `initial` is the starting parameter vector. `weight` is the
/// flattened `m x m` GMM weighting matrix (row-major) or `None` for the
/// identity (the natural choice when exactly identified). A Python exception
/// raised inside `moments_fn` is captured and re-raised. Returns `params`,
/// `objective`, `gbar`, `converged`, `iterations`, `fevals`, `nmoments`, and
/// `nparams`.
#[pyfunction]
#[pyo3(signature = (moments_fn, initial, weight = None))]
fn gmm_nonlinear<'py>(
    py: Python<'py>,
    moments_fn: Bound<'py, PyAny>,
    initial: Vec<f64>,
    weight: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    // The driver's moment closure is `FnMut(&[f64]) -> Vec<Vec<f64>>` and so
    // cannot return a Result. On a failed callback (Python raised, or the
    // return value did not coerce to an n-by-m float matrix) we stash the
    // PyErr here and yield an empty matrix; the driver then rejects the empty
    // / shape-inconsistent moments with a crate error that masks the true
    // cause, so we re-raise the stashed PyErr first once the driver returns.
    let err_slot: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
    let moments_ref = &moments_fn;
    let err_ref = &err_slot;
    let closure = move |theta: &[f64]| -> Vec<Vec<f64>> {
        let params = theta.to_vec().into_pyarray(py);
        match moments_ref.call1((params,)) {
            Ok(ret) => match ret.extract::<Vec<Vec<f64>>>() {
                Ok(mat) => mat,
                Err(e) => {
                    *err_ref.borrow_mut() = Some(e);
                    Vec::new()
                }
            },
            Err(e) => {
                *err_ref.borrow_mut() = Some(e);
                Vec::new()
            }
        }
    };

    let result = tsecon_gmm::gmm_nonlinear(closure, &initial, weight.as_deref());
    // Surface a Python exception raised inside the callback first; it is the
    // true cause and carries the original traceback.
    if let Some(pyerr) = err_slot.into_inner() {
        return Err(pyerr);
    }
    let fit = result.map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("params", fit.params.into_pyarray(py))?;
    d.set_item("objective", fit.objective)?;
    d.set_item("gbar", fit.gbar.into_pyarray(py))?;
    d.set_item("converged", fit.converged)?;
    d.set_item("iterations", fit.iterations)?;
    d.set_item("fevals", fit.fevals)?;
    d.set_item("nmoments", fit.nmoments)?;
    d.set_item("nparams", fit.nparams)?;
    Ok(d)
}

/// Weighted MIDAS regression fit by nonlinear least squares (Ghysels,
/// Santa-Clara & Valkanov 2004; Ghysels, Sinko & Valkanov 2007). Restricts the
/// `K` high-frequency lag coefficients to a two-parameter weight shape and
/// estimates `(alpha, beta, psi_1, psi_2)` by minimizing the residual sum of
/// squares. `hf_lags` is `nobs x K` (each column a high-frequency lag,
/// most-recent-first, aligned to `y`). `scheme`: "exp_almon" (unconstrained
/// hyperparameters) or "beta" (strictly positive shapes; needs `K >= 2`).
/// `weight_start` optionally overrides the starting hyperparameters
/// `(psi_1, psi_2)` in natural space; `None` uses the scheme default. Because
/// the weights sum to one, `slope` is the aggregate slope on a proper weighted
/// average, comparable to the sum of the U-MIDAS lag coefficients. Returns dict
/// keys: `scheme`, `intercept`, `slope`, `weight_params`, `weights`, `fitted`,
/// `residuals`, `ssr`, `rsquared`, `converged`, `iterations`.
#[pyfunction]
#[pyo3(signature = (y, hf_lags, scheme = "exp_almon", weight_start = None))]
fn weighted_midas<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    hf_lags: numpy::PyReadonlyArray2<'py, f64>,
    scheme: &str,
    weight_start: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyDict>> {
    let a = hf_lags.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let sch = match scheme {
        "exp_almon" => tsecon_midas::WeightScheme::ExpAlmon,
        "beta" => tsecon_midas::WeightScheme::Beta,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown scheme {other:?}; expected \"exp_almon\" or \"beta\""
            )))
        }
    };
    let start = weight_start.map(|(p1, p2)| [p1, p2]);
    let fit = tsecon_midas::weighted_midas(y.as_slice()?, &cols, sch, start).map_err(to_py)?;
    let scheme_name = match fit.scheme {
        tsecon_midas::WeightScheme::ExpAlmon => "exp_almon",
        tsecon_midas::WeightScheme::Beta => "beta",
    };
    let d = PyDict::new(py);
    d.set_item("scheme", scheme_name)?;
    d.set_item("intercept", fit.intercept)?;
    d.set_item("slope", fit.slope)?;
    d.set_item("weight_params", fit.weight_params.to_vec().into_pyarray(py))?;
    d.set_item("weights", fit.weights.into_pyarray(py))?;
    d.set_item("fitted", fit.fitted.into_pyarray(py))?;
    d.set_item("residuals", fit.residuals.into_pyarray(py))?;
    d.set_item("ssr", fit.ssr)?;
    d.set_item("rsquared", fit.rsquared)?;
    d.set_item("converged", fit.converged)?;
    d.set_item("iterations", fit.iterations)?;
    Ok(d)
}

/// State-dependent (interacted) local projections (Ramey & Zubairy 2018).
///
/// The impulse and every control are interacted with the *lagged* binary
/// state indicator `I_{t-1}` and its complement, so the regime is
/// predetermined (not itself moved by the shock). Two response paths are
/// returned. `se`: "lag_augmented" (Montiel Olea-Plagborg-Møller 2021, the
/// default) or "hac" (Newey-West; `maxlags=None` grows with the horizon).
/// `cumulative` regresses the cumulated outcome (Ramey-Zubairy). Returns dict
/// keys `horizons`, `irf_state1`, `se_state1`, `irf_state0`, `se_state0` (the
/// per-regime impulse responses and their standard errors at each horizon).
#[pyfunction]
#[pyo3(signature = (y, shock, state_indicator, horizons = 12, n_lag_controls = 4, se = "lag_augmented", maxlags = None, cumulative = false))]
#[allow(clippy::too_many_arguments)]
fn lp_state<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    shock: PyReadonlyArray1<'py, f64>,
    state_indicator: PyReadonlyArray1<'py, f64>,
    horizons: usize,
    n_lag_controls: usize,
    se: &str,
    maxlags: Option<usize>,
    cumulative: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut spec = tsecon_lp::LpSpec::new(horizons, n_lag_controls).cumulative(cumulative);
    spec = match se {
        "lag_augmented" => spec,
        "hac" => spec.with_hac(maxlags),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown se {other:?}; expected \"lag_augmented\" or \"hac\""
            )))
        }
    };
    let r = tsecon_lp::lp_state(
        y.as_slice()?,
        shock.as_slice()?,
        state_indicator.as_slice()?,
        spec,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item(
        "horizons",
        r.horizons
            .iter()
            .map(|&h| h as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("irf_state1", r.irf_state1.into_pyarray(py))?;
    d.set_item("se_state1", r.se_state1.into_pyarray(py))?;
    d.set_item("irf_state0", r.irf_state0.into_pyarray(py))?;
    d.set_item("se_state0", r.se_state0.into_pyarray(py))?;
    Ok(d)
}

/// Pesaran-Smith (1995) mean-group panel VAR: fit a reduced-form VAR(p)
/// to every entity (equation-by-equation OLS via `tsecon-var`) and average,
/// with dispersion-based cross-entity standard errors `sd(theta_i)/sqrt(N)`.
///
/// `entities` is a list of per-entity `T_i x k` data matrices (rows are
/// observations, oldest first; the time dimensions may differ but the `k`
/// variables must match). `trend` is "c" (a constant in every equation) or
/// "n" (no deterministic term). IRFs are Cholesky-orthogonalized entity by
/// entity, then averaged. Requires `N >= 2` (the dispersion SE needs it).
///
/// Returns a dict with `intercept`/`intercept_se` (length k), `coefs`/
/// `coefs_se` (the mean-group lag matrices `[A_1, ..., A_p]`, each k x k),
/// `orth_irfs`/`orth_irfs_se` (the mean-group IRF path, `(horizon + 1)` of
/// k x k matrices), `irf_path`/`irf_path_se` (the `mg_irf_path` response=
/// `response`, impulse=`impulse` series over horizons), and the scalars
/// `n_entities`, `neqs`, `lags`.
#[pyfunction]
#[pyo3(signature = (entities, lags = 1, trend = "c", horizon = 10, response = 0, impulse = 0))]
fn mean_group_var<'py>(
    py: Python<'py>,
    entities: Vec<numpy::PyReadonlyArray2<'py, f64>>,
    lags: usize,
    trend: &str,
    horizon: usize,
    response: usize,
    impulse: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let tr = match trend {
        "c" => tsecon_var::Trend::Constant,
        "n" => tsecon_var::Trend::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown trend {other:?}; expected \"c\" or \"n\""
            )))
        }
    };
    let mats: Vec<_> = entities.iter().map(to_faer).collect();
    let mg = tsecon_panel::mean_group_var(&mats, lags, tr, horizon).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("intercept", mg.intercept.clone().into_pyarray(py))?;
    d.set_item("intercept_se", mg.intercept_se.clone().into_pyarray(py))?;
    let coefs: Vec<Vec<Vec<f64>>> = mg.coefs.iter().map(mat_to_vec2).collect();
    d.set_item("coefs", coefs)?;
    let coefs_se: Vec<Vec<Vec<f64>>> = mg.coefs_se.iter().map(mat_to_vec2).collect();
    d.set_item("coefs_se", coefs_se)?;
    let orth_irfs: Vec<Vec<Vec<f64>>> = mg.orth_irfs.iter().map(mat_to_vec2).collect();
    d.set_item("orth_irfs", orth_irfs)?;
    let orth_irfs_se: Vec<Vec<Vec<f64>>> = mg.orth_irfs_se.iter().map(mat_to_vec2).collect();
    d.set_item("orth_irfs_se", orth_irfs_se)?;
    match tsecon_panel::mg_irf_path(&mg, response, impulse) {
        Some((path, path_se)) => {
            d.set_item("irf_path", path.into_pyarray(py))?;
            d.set_item("irf_path_se", path_se.into_pyarray(py))?;
        }
        None => {
            return Err(PyValueError::new_err(format!(
                "mg_irf_path indices out of range: response={response}, impulse={impulse}, neqs={}",
                mg.neqs
            )))
        }
    }
    d.set_item("n_entities", mg.n_entities)?;
    d.set_item("neqs", mg.neqs)?;
    d.set_item("lags", mg.lags)?;
    Ok(d)
}

/// Dynamic Nelson-Siegel factors and one-step curve forecast
/// (Diebold & Li 2006, two-step estimator).
///
/// Step one fits the three Nelson-Siegel factors `[level, slope, curvature]`
/// cross-sectionally for every date in `panel` (a `T x n_maturities` matrix of
/// yield curves, one curve per row) at the fixed `decay` (lambda). Step two
/// fits an independent AR(1) to each factor series and maps the one-step-ahead
/// factor forecast back through the loadings to a forecast curve.
///
/// Returns `maturities`, `lambda`, the `T x 3` `factors`, per-date `rsquared`,
/// the `level`/`slope`/`curvature` factor series, and a `forecast` sub-dict
/// with the one-step forecast `factors`, forecast `yields`, and the per-factor
/// AR(1) `ar1_intercept`/`ar1_phi`.
#[pyfunction]
#[pyo3(signature = (panel, maturities, decay = 0.0609))]
fn dynamic_ns<'py>(
    py: Python<'py>,
    panel: numpy::PyReadonlyArray2<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    decay: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let rows = mat_to_vec2(&to_faer(&panel));
    let mat = maturities.as_slice()?;
    let fit = tsecon_termstructure::fit_dynamic_ns(&rows, mat, decay).map_err(to_py)?;

    let factors: Vec<Vec<f64>> = fit.factors.iter().map(|f| f.to_vec()).collect();
    let level = fit.level();
    let slope = fit.slope();
    let curvature = fit.curvature();
    let fc = fit.forecast().map_err(to_py)?;

    let fdict = PyDict::new(py);
    fdict.set_item("factors", fc.factors.to_vec().into_pyarray(py))?;
    fdict.set_item("yields", fc.yields.into_pyarray(py))?;
    let ar1_intercept: Vec<f64> = fc.factor_ar1.iter().map(|a| a.intercept).collect();
    let ar1_phi: Vec<f64> = fc.factor_ar1.iter().map(|a| a.phi).collect();
    fdict.set_item("ar1_intercept", ar1_intercept.into_pyarray(py))?;
    fdict.set_item("ar1_phi", ar1_phi.into_pyarray(py))?;

    let d = PyDict::new(py);
    d.set_item("maturities", fit.maturities.into_pyarray(py))?;
    d.set_item("lambda", fit.lambda)?;
    d.set_item("factors", factors)?;
    d.set_item("rsquared", fit.rsquared.into_pyarray(py))?;
    d.set_item("level", level.into_pyarray(py))?;
    d.set_item("slope", slope.into_pyarray(py))?;
    d.set_item("curvature", curvature.into_pyarray(py))?;
    d.set_item("forecast", fdict)?;
    Ok(d)
}

/// Two-step factor-augmented VAR (Bernanke-Boivin-Eliasz 2005, QJE).
///
/// Step 1 extracts `n_factors` principal-component factors from the large
/// standardized information panel `panel` (`T x N`, observations in rows);
/// step 2 fits a VAR(`lags`) with deterministic `trend` on `[factors,
/// policy]`, the observed `policy` series (length `T`) ordered last, so a
/// recursive/Cholesky scheme identifies the policy innovation as the
/// structural monetary shock. Pass `slow_indices` (column positions of the
/// slow-moving series) to use the slow/fast factor rotation that purges the
/// contemporaneous policy component (`Favar::two_step_slow_fast`); omit it
/// for the plain `Favar::two_step`. Returns `factors` (`T x n_factors`), the
/// factor-VAR `params` and `sigma_u`, `n_factors`, `n_endog`, `policy_index`,
/// and the recursive policy-shock IRFs `irf_panel` (`N x (horizon + 1)`) and
/// `irf_policy`.
#[pyfunction]
#[pyo3(signature = (panel, policy, n_factors = 2, lags = 2, trend = "c",
                    slow_indices = None, horizon = 20, orth = true))]
#[allow(clippy::too_many_arguments)]
fn favar<'py>(
    py: Python<'py>,
    panel: numpy::PyReadonlyArray2<'py, f64>,
    policy: PyReadonlyArray1<'py, f64>,
    n_factors: usize,
    lags: usize,
    trend: &str,
    slow_indices: Option<Vec<usize>>,
    horizon: usize,
    orth: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let m = to_faer(&panel);
    let pol = policy.as_slice()?;
    let tr = match trend {
        "c" => tsecon_favar::Trend::Constant,
        "n" => tsecon_favar::Trend::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown trend {other:?}; expected \"c\" or \"n\""
            )))
        }
    };
    let fit = match slow_indices {
        Some(ref idx) => {
            tsecon_favar::Favar::two_step_slow_fast(m.as_ref(), pol, idx, n_factors, lags, tr)
                .map_err(to_py)?
        }
        None => {
            tsecon_favar::Favar::two_step(m.as_ref(), pol, n_factors, lags, tr).map_err(to_py)?
        }
    };
    let d = PyDict::new(py);
    d.set_item("factors", mat_to_vec2(&fit.factors().to_owned()))?;
    d.set_item("params", mat_to_vec2(&fit.var().params))?;
    d.set_item("sigma_u", mat_to_vec2(&fit.var().sigma_u))?;
    d.set_item("n_factors", fit.n_factors())?;
    d.set_item("n_endog", fit.n_endog())?;
    d.set_item("policy_index", fit.policy_index())?;
    // Impulse responses to the recursive (Cholesky) policy shock, mapped onto
    // the panel through the loadings (BBE observation equation X_t = L F_t).
    let shock = fit.policy_index();
    let irf_panel: Vec<Vec<f64>> = (0..fit.factor_model().n_series())
        .map(|s| fit.series_response(s, shock, horizon, orth).map_err(to_py))
        .collect::<PyResult<Vec<Vec<f64>>>>()?;
    let irf_policy = fit.policy_response(shock, horizon, orth).map_err(to_py)?;
    d.set_item("irf_panel", irf_panel)?;
    d.set_item("irf_policy", irf_policy.into_pyarray(py))?;
    Ok(d)
}

/// Realized quarticity `RQ = (n/3) sum r_i^4` (Barndorff-Nielsen &
/// Shephard 2002), the non-jump-robust estimator of integrated quarticity
/// `int sigma^4 ds` (the asymptotic-variance scale of realized variance).
/// For a jump-robust version use `tripower_quarticity`.
#[pyfunction]
fn realized_quarticity(returns: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let r = returns.as_slice()?;
    tsecon_realized::realized_quarticity(r).map_err(to_py)
}

/// Tripower quarticity
/// `TQ = n mu_{4/3}^{-3} sum |r_i|^{4/3}|r_{i-1}|^{4/3}|r_{i-2}|^{4/3}`
/// (Barndorff-Nielsen & Shephard 2004), the jump-robust estimator of
/// integrated quarticity `int sigma^4 ds` used to studentize the BNS ratio
/// jump test.
#[pyfunction]
fn tripower_quarticity(returns: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let r = returns.as_slice()?;
    tsecon_realized::tripower_quarticity(r).map_err(to_py)
}

/// Barndorff-Nielsen-Shephard ratio jump test (BNS 2004; Huang & Tauchen
/// 2005). Returns a dict with `ratio`, the studentized relative-jump
/// z-statistic; compare against a standard-normal critical value (larger =
/// stronger evidence of a jump).
#[pyfunction]
fn bns_jump_test<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = returns.as_slice()?;
    let ratio = tsecon_realized::bns_jump_ratio(r).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("ratio", ratio)?;
    Ok(d)
}

/// Range-based daily variance from OHLC bars, summed across the supplied
/// bars. `method="parkinson"` gives the Parkinson (1980) high-low estimator
/// `(1/(4 ln 2)) sum (ln(H/L))^2`; `method="garman_klass"` gives Garman &
/// Klass (1980), which additionally requires the `open` and `close` series.
#[pyfunction]
#[pyo3(signature = (high, low, method = "parkinson", open = None, close = None))]
fn realized_range(
    high: PyReadonlyArray1<'_, f64>,
    low: PyReadonlyArray1<'_, f64>,
    method: &str,
    open: Option<PyReadonlyArray1<'_, f64>>,
    close: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<f64> {
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    match method {
        "parkinson" => tsecon_realized::parkinson(h, l).map_err(to_py),
        "garman_klass" => {
            let open = open.ok_or_else(|| {
                PyValueError::new_err("garman_klass requires the open and close series")
            })?;
            let close = close.ok_or_else(|| {
                PyValueError::new_err("garman_klass requires the open and close series")
            })?;
            let o = open.as_slice()?;
            let c = close.as_slice()?;
            tsecon_realized::garman_klass(o, h, l, c).map_err(to_py)
        }
        other => Err(PyValueError::new_err(format!(
            "unknown method {other:?}; expected \"parkinson\" or \"garman_klass\""
        ))),
    }
}

/// GAS(1,1) score-driven volatility (Creal-Koopman-Lucas 2013).
///
/// Fits a time-varying-variance model by maximum likelihood, driving the
/// variance by the scaled score of the observation density with the
/// inverse-information scaling. `density`: `"gaussian"` or `"student_t"`
/// (heavy-tailed, robust to outliers; estimates the degrees of freedom
/// `nu`). Returns the fitted `omega`/`a`/`b` (and `nu` for Student-t), the
/// filtered conditional `variance` path, `std_resid`, `loglik`, `aic`,
/// `bic`, the one-step-ahead `next_variance`, convergence info, and — if
/// `horizon > 0` — an `h`-step variance `forecast`.
#[pyfunction]
#[pyo3(signature = (y, density = "gaussian", horizon = 0))]
fn gas_volatility<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    density: &str,
    horizon: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let dens = match density {
        "gaussian" => tsecon_gas::Density::Gaussian,
        "student_t" | "t" => tsecon_gas::Density::StudentT,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown density {other:?}; expected \"gaussian\" or \"student_t\""
            )))
        }
    };
    let model = tsecon_gas::GasModel::new(y.as_slice()?, dens).map_err(to_py)?;
    let res = model.fit().map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("omega", res.params.omega)?;
    d.set_item("a", res.params.a)?;
    d.set_item("b", res.params.b)?;
    if matches!(dens, tsecon_gas::Density::StudentT) {
        d.set_item("nu", res.params.nu)?;
    }
    d.set_item("variance", res.variance.clone().into_pyarray(py))?;
    d.set_item("std_resid", res.std_resid.clone().into_pyarray(py))?;
    d.set_item("loglik", res.loglik)?;
    d.set_item("aic", res.aic())?;
    d.set_item("bic", res.bic())?;
    d.set_item("next_variance", res.next_variance)?;
    d.set_item("converged", res.converged)?;
    d.set_item("iterations", res.iterations)?;
    if horizon > 0 {
        d.set_item("forecast", res.forecast(horizon).map_err(to_py)?.into_pyarray(py))?;
    }
    Ok(d)
}

/// Heterogeneous-panel mean-group estimator (Pesaran-Smith 1995) and its
/// common-correlated-effects variant (Pesaran 2006, CCE-MG).
///
/// `ys` is a list of per-unit response vectors and `xs` the matching list
/// of per-unit `T_i x k` regressor matrices (one entry per cross-sectional
/// unit; the time lengths may differ for `"mg"`). `method`: `"mg"` averages
/// the per-unit OLS slopes with dispersion-based cross-unit SEs; `"cce"`
/// augments each unit with the cross-section averages of `y` and `X` to
/// purge unobserved common factors before averaging (requires a balanced
/// panel). Returns the mean-group `coef`, `se`, `tstat`, the per-unit
/// `coef_per_unit`, `n_units`, and `k`.
#[pyfunction]
#[pyo3(signature = (ys, xs, method = "mg"))]
fn panel_mean_group<'py>(
    py: Python<'py>,
    ys: Vec<PyReadonlyArray1<'py, f64>>,
    xs: Vec<numpy::PyReadonlyArray2<'py, f64>>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    if ys.len() != xs.len() {
        return Err(PyValueError::new_err(format!(
            "ys and xs must have the same number of units; got {} and {}",
            ys.len(),
            xs.len()
        )));
    }
    let mut units = Vec::with_capacity(ys.len());
    for (yi, xi) in ys.iter().zip(xs.iter()) {
        let a = xi.as_array();
        let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
        units.push(tsecon_panelts::PanelUnit::new(yi.as_slice()?.to_vec(), cols));
    }
    let mg = match method {
        "mg" => tsecon_panelts::mean_group(&units).map_err(to_py)?,
        "cce" => tsecon_panelts::cce_mean_group(&units).map_err(to_py)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}; expected \"mg\" or \"cce\""
            )))
        }
    };
    let d = PyDict::new(py);
    d.set_item("coef", mg.coef.clone().into_pyarray(py))?;
    d.set_item("se", mg.se.clone().into_pyarray(py))?;
    d.set_item("tstat", mg.tstat.clone().into_pyarray(py))?;
    d.set_item("coef_per_unit", mg.coef_per_unit.clone())?;
    d.set_item("n_units", mg.n_units)?;
    d.set_item("k", mg.k)?;
    Ok(d)
}

/// Dynamic-factor-model nowcast (Doz-Giannone-Reichlin 2011 two-step).
///
/// Fits `n_factors` common factors with an order-`factor_order` factor VAR
/// to the `T x N` panel `data`, then Kalman-smooths and reads the nowcast
/// off the sample edge. `data` may carry `NaN` in the last rows of the
/// faster-arriving series (the ragged edge): the two-step model is
/// estimated on the leading balanced block (rows before the first row with
/// any missing value) and the Kalman filter then runs over the full panel,
/// using exactly the observations that are present at the edge. Returns the
/// edge `nowcast` (one level per series), the `edge_factor`, the Gaussian
/// `loglik`, the `smoothed_factors` (`T x r`), and `n_factors`/`factor_order`.
#[pyfunction]
#[pyo3(signature = (data, n_factors = 1, factor_order = 2))]
fn dfm_nowcast<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    n_factors: usize,
    factor_order: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    let arr = data.as_array();
    let (t, n) = (arr.nrows(), arr.ncols());
    // Training block: the leading rows before the ragged edge (all finite).
    let mut first_ragged = t;
    for i in 0..t {
        if (0..n).any(|j| !arr[(i, j)].is_finite()) {
            first_ragged = i;
            break;
        }
    }
    let train = Mat::from_fn(first_ragged, n, |r, j| arr[(r, j)]);
    let full = to_faer(&data);
    let nc = tsecon_nowcast::Nowcaster::fit_two_step(train.as_ref(), n_factors, factor_order)
        .map_err(to_py)?;
    let res = nc.nowcast_panel(full.as_ref()).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("nowcast", res.values.clone().into_pyarray(py))?;
    d.set_item("edge_factor", res.edge_factor.clone().into_pyarray(py))?;
    d.set_item("loglik", res.smoothing.loglik)?;
    let factors: Vec<Vec<f64>> = nc.smoothed_factors().to_vec();
    d.set_item("smoothed_factors", factors)?;
    d.set_item("n_factors", nc.n_factors())?;
    d.set_item("factor_order", nc.factor_order())?;
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
    m.add_function(wrap_pyfunction!(arima_fit, m)?)?;
    m.add_function(wrap_pyfunction!(lp, m)?)?;
    m.add_function(wrap_pyfunction!(lp_iv, m)?)?;
    m.add_function(wrap_pyfunction!(ridge, m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net, m)?)?;
    m.add_function(wrap_pyfunction!(lasso, m)?)?;
    m.add_function(wrap_pyfunction!(sign_restricted_svar, m)?)?;
    m.add_function(wrap_pyfunction!(panel_fe, m)?)?;
    m.add_function(wrap_pyfunction!(panel_lp, m)?)?;
    m.add_function(wrap_pyfunction!(cw_test, m)?)?;
    m.add_function(wrap_pyfunction!(gw_test, m)?)?;
    m.add_function(wrap_pyfunction!(periodogram, m)?)?;
    m.add_function(wrap_pyfunction!(welch, m)?)?;
    m.add_function(wrap_pyfunction!(coherence, m)?)?;
    m.add_function(wrap_pyfunction!(johansen, m)?)?;
    m.add_function(wrap_pyfunction!(vecm, m)?)?;
    m.add_function(wrap_pyfunction!(markov_switching_ar, m)?)?;
    m.add_function(wrap_pyfunction!(midas_weights, m)?)?;
    m.add_function(wrap_pyfunction!(umidas, m)?)?;
    m.add_function(wrap_pyfunction!(ccc_garch, m)?)?;
    m.add_function(wrap_pyfunction!(dcc_garch, m)?)?;
    m.add_function(wrap_pyfunction!(realized_measures, m)?)?;
    m.add_function(wrap_pyfunction!(har_rv, m)?)?;
    m.add_function(wrap_pyfunction!(connectedness, m)?)?;
    m.add_function(wrap_pyfunction!(factor_model, m)?)?;
    m.add_function(wrap_pyfunction!(nelson_siegel, m)?)?;
    m.add_function(wrap_pyfunction!(svensson, m)?)?;
    m.add_function(wrap_pyfunction!(backtest, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_lasso, m)?)?;
    m.add_function(wrap_pyfunction!(lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(cv_splits, m)?)?;
    m.add_function(wrap_pyfunction!(iv_gmm, m)?)?;
    m.add_function(wrap_pyfunction!(gmm_nonlinear, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_midas, m)?)?;
    m.add_function(wrap_pyfunction!(lp_state, m)?)?;
    m.add_function(wrap_pyfunction!(mean_group_var, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_ns, m)?)?;
    m.add_function(wrap_pyfunction!(favar, m)?)?;
    m.add_function(wrap_pyfunction!(realized_quarticity, m)?)?;
    m.add_function(wrap_pyfunction!(tripower_quarticity, m)?)?;
    m.add_function(wrap_pyfunction!(bns_jump_test, m)?)?;
    m.add_function(wrap_pyfunction!(realized_range, m)?)?;
    m.add_function(wrap_pyfunction!(gas_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(panel_mean_group, m)?)?;
    m.add_function(wrap_pyfunction!(dfm_nowcast, m)?)?;
    Ok(())
}
