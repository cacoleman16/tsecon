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

/// A 1-D `f64` array as an owned `Vec`. Unlike `as_slice()`, this accepts
/// non-contiguous input — a column view (`data[:, 0]`), a transposed slice, a
/// strided view (`x[::2]`), or a pandas column — copying only when the array
/// is not already a contiguous buffer. Every 1-D input goes through this so
/// users never hit "the given array is not contiguous or is misaligned".
fn vec1(a: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    match a.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => a.as_array().to_vec(),
    }
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
    let r = tsecon_diag::acf(&vec1(&y), nlags, adjusted).map_err(to_py)?;
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
    let y = vec1(&y);
    let y = y.as_slice();
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
    let r = tsecon_diag::ljung_box(&vec1(&y), nlags).map_err(to_py)?;
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
    let r = tsecon_diag::jarque_bera(&vec1(&x)).map_err(to_py)?;
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
    let r = tsecon_diag::arch_lm(&vec1(&resid), nlags).map_err(to_py)?;
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
    let r = tsecon_bootstrap::optimal_block_length(&vec1(&y)).map_err(to_py)?;
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
    let ys = vec1(&y);
    let ys = ys.as_slice();
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
    let ys = vec1(&y);
    let ys = ys.as_slice();
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
    let r = tsecon_diag::adf(&vec1(&y), adf_regression(regression)?, sel).map_err(to_py)?;
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
    let r = tsecon_diag::kpss(&vec1(&y), reg, lags).map_err(to_py)?;
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
    let r = tsecon_diag::check_stationarity_at(&vec1(&y), alpha).map_err(to_py)?;
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

fn po_trend(s: &str) -> PyResult<tsecon_diag::PoTrend> {
    use tsecon_diag::PoTrend::*;
    match s {
        "n" => Ok(None),
        "c" => Ok(Constant),
        "ct" => Ok(ConstantTrend),
        other => Err(PyValueError::new_err(format!(
            "unknown trend {other:?}; expected \"n\", \"c\", or \"ct\""
        ))),
    }
}

/// Phillips-Perron unit-root test (semiparametric; null: unit root).
///
/// `regression`: "n", "c" (default), "ct". `test_type`: "tau" (Z-tau,
/// default) or "rho" (Z-alpha). `lags`: Bartlett LRV bandwidth; None uses
/// ceil(12*(n/100)^(1/4)). Matches arch.unitroot.PhillipsPerron (< 1e-10).
#[pyfunction]
#[pyo3(signature = (y, regression = "c", test_type = "tau", lags = None))]
fn phillips_perron<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    regression: &str,
    test_type: &str,
    lags: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_diag::PpTestType;
    let tt = match test_type {
        "tau" => PpTestType::Tau,
        "rho" => PpTestType::Rho,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown test_type {other:?}; expected \"tau\" or \"rho\""
            )))
        }
    };
    let r = tsecon_diag::phillips_perron(&vec1(&y), adf_regression(regression)?, tt, lags)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("stat", r.stat)?;
    d.set_item("ztau", r.ztau)?;
    d.set_item("zalpha", r.zalpha)?;
    d.set_item("pvalue", r.p_value)?;
    d.set_item("lags", r.lags)?;
    d.set_item("nobs", r.nobs)?;
    let crit = PyDict::new(py);
    crit.set_item("1%", r.crit.pct1)?;
    crit.set_item("5%", r.crit.pct5)?;
    crit.set_item("10%", r.crit.pct10)?;
    d.set_item("crit", crit)?;
    Ok(d)
}

/// Phillips-Ouliaris residual cointegration test (null: no cointegration).
///
/// `x` is a 2-D (T, m) matrix of the m stochastic regressors, used as-is
/// (deterministics come from `trend`; do NOT add your own constant column).
/// `trend`: "n", "c" (default), "ct". `test_type`: "Zt" (default) or "Za".
/// `bandwidth`: Bartlett LRV bandwidth of the AR(1) residual; None uses the
/// Newey-West rule floor(4*((T-1)/100)^(2/9)). Za is statistic-only
/// (pvalue None/crit None). Zt p-value/crit use the MacKinnon N-surfaces
/// (statsmodels `coint` route). N = 1 + m.
#[pyfunction]
#[pyo3(signature = (y, x, trend = "c", test_type = "Zt", bandwidth = None))]
fn phillips_ouliaris<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    trend: &str,
    test_type: &str,
    bandwidth: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_diag::PoTestType;
    let tt = match test_type {
        "Zt" | "zt" => PoTestType::Zt,
        "Za" | "za" => PoTestType::Za,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown test_type {other:?}; expected \"Zt\" or \"Za\""
            )))
        }
    };
    let xa = x.as_array();
    if xa.ncols() < 1 {
        return Err(PyValueError::new_err(
            "phillips_ouliaris requires at least one regressor column in x \
             (n_vars = 1 + ncols(x) must be >= 2)",
        ));
    }
    let cols: Vec<Vec<f64>> = (0..xa.ncols()).map(|j| xa.column(j).to_vec()).collect();
    let r = tsecon_diag::phillips_ouliaris(&vec1(&y), &cols, po_trend(trend)?, tt, bandwidth)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("stat", r.stat)?;
    d.set_item("pvalue", r.p_value)?; // f64::NAN -> Python nan for Za / N>6
    match r.crit {
        Some(c) => {
            let crit = PyDict::new(py);
            crit.set_item("1%", c.pct1)?;
            crit.set_item("5%", c.pct5)?;
            crit.set_item("10%", c.pct10)?;
            d.set_item("crit", crit)?;
        }
        None => d.set_item("crit", py.None())?, // Za, N>12, or no-constant N>1
    }
    d.set_item("lags", r.lags)?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("n_vars", r.n_vars)?;
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
    let xs = vec1(&x);
    let xs = xs.as_slice();
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
    let ys = vec1(&y);
    let ys = ys.as_slice();
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
/// Stability: read `is_stable` (bool). `min_root`/`max_root` are the smallest
/// and largest moduli of the reciprocal characteristic roots — the system is
/// stable iff `min_root > 1`, so `max_root` alone is NOT a verdict.
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
    // `roots_moduli` are the RECIPROCAL characteristic roots (the statsmodels
    // convention): the VAR is stable iff EVERY modulus exceeds 1, i.e. iff
    // `min_root > 1`. `max_root` is therefore the root FARTHEST from the unit
    // circle and is NOT a stability verdict on its own — read `is_stable` (or
    // compare `min_root` to 1). `max_root` is kept for backwards compatibility.
    let roots = r.roots_moduli().map_err(to_py)?;
    d.set_item("max_root", roots.iter().cloned().fold(0.0_f64, f64::max))?;
    d.set_item(
        "min_root",
        roots.iter().cloned().fold(f64::INFINITY, f64::min),
    )?;
    d.set_item("is_stable", r.is_stable().map_err(to_py)?)?;
    Ok(d)
}

/// Impulse responses of a fitted VAR: `irfs[h][i][j]` is the response of
/// variable i to a shock in variable j at horizon h (orthogonalized via
/// the Cholesky factor of sigma_u when `orth=True`).
///
/// This returns the point path only. For frequentist confidence bands
/// (delta-method or bootstrap) use `var_irf_bands`.
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

/// Frequentist confidence bands on VAR impulse responses — the banded
/// companion to `var_irf` (which stays a bare nested list; this returns a
/// dict). Keys: `point`/`se`/`lower`/`upper`, each `[h][i][j]` (response of
/// variable i to a shock in variable j at horizon h, matching `var_irf`),
/// plus echoed `method`/`alpha`/`n_boot` (`n_boot` is `None` for the
/// asymptotic branch).
///
/// `method="asymptotic"` (default) uses the Lütkepohl (1990) delta-method
/// standard errors (statsmodels `irf.stderr`) and symmetric Wald bands
/// `point ± z_{1-alpha/2}·se`. `method="bootstrap"` uses a residual
/// (Efron/Kilian) recursive-design bootstrap with percentile bands, an
/// optional Kilian (1998) bias correction (`bias_correct`), `n_boot`
/// replications and a reproducible `seed`. `orth` toggles orthogonalized
/// (Cholesky) vs reduced-form responses and `cumulative` puts the bands on
/// the cumulated IRF — both exactly as in `var_irf`.
#[pyfunction]
#[pyo3(signature = (
    data,
    lags = 2,
    horizon = 10,
    orth = true,
    method = "asymptotic",
    alpha = 0.1,
    cumulative = false,
    n_boot = 1000,
    seed = 0,
    trend = "c",
    bias_correct = false,
))]
#[allow(clippy::too_many_arguments)]
fn var_irf_bands<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    orth: bool,
    method: &str,
    alpha: f64,
    cumulative: bool,
    n_boot: usize,
    seed: u64,
    trend: &str,
    bias_correct: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(PyValueError::new_err(format!(
            "alpha must lie strictly in (0, 1), got {alpha}"
        )));
    }
    let d = PyDict::new(py);
    match method {
        "asymptotic" => {
            let r = var_results(&data, lags, trend)?;
            // Point path: exactly what `var_irf` returns, cumulated when asked
            // so the bands are anchored on the same estimate the SE describes.
            let irf = r.irf(horizon).map_err(to_py)?;
            let mats = if orth { &irf.orth_irfs } else { &irf.irfs };
            let mut point: Vec<Vec<Vec<f64>>> = mats.iter().map(mat_to_vec2).collect();
            if cumulative {
                for h in 1..point.len() {
                    for i in 0..point[h].len() {
                        for j in 0..point[h][i].len() {
                            point[h][i][j] += point[h - 1][i][j];
                        }
                    }
                }
            }
            // Delta-method standard errors, same [h][i][j] layout as `point`.
            let se_mats =
                tsecon_var::irf_asymptotic::irf_asymptotic_se(&r, horizon, orth, cumulative)
                    .map_err(to_py)?;
            let se: Vec<Vec<Vec<f64>>> = se_mats.iter().map(mat_to_vec2).collect();
            // Symmetric Wald bands: point ± z_{1-alpha/2} · se.
            let z = tsecon_stats::special::inv_norm_cdf(1.0 - alpha / 2.0).map_err(to_py)?;
            let mut lower = point.clone();
            let mut upper = point.clone();
            for h in 0..point.len() {
                for i in 0..point[h].len() {
                    for j in 0..point[h][i].len() {
                        let half = z * se[h][i][j];
                        lower[h][i][j] = point[h][i][j] - half;
                        upper[h][i][j] = point[h][i][j] + half;
                    }
                }
            }
            d.set_item("point", point)?;
            d.set_item("se", se)?;
            d.set_item("lower", lower)?;
            d.set_item("upper", upper)?;
            d.set_item("method", "asymptotic")?;
            d.set_item("alpha", alpha)?;
            d.set_item("n_boot", py.None())?;
        }
        "bootstrap" => {
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
            let bands = tsecon_var::bootstrap_irf_bands(
                m.as_ref(),
                lags,
                tr,
                horizon,
                orth,
                cumulative,
                alpha,
                n_boot,
                seed,
                bias_correct,
            )
            .map_err(to_py)?;
            let point: Vec<Vec<Vec<f64>>> = bands.point.iter().map(mat_to_vec2).collect();
            let se: Vec<Vec<Vec<f64>>> = bands.se.iter().map(mat_to_vec2).collect();
            let lower: Vec<Vec<Vec<f64>>> = bands.lower.iter().map(mat_to_vec2).collect();
            let upper: Vec<Vec<Vec<f64>>> = bands.upper.iter().map(mat_to_vec2).collect();
            d.set_item("point", point)?;
            d.set_item("se", se)?;
            d.set_item("lower", lower)?;
            d.set_item("upper", upper)?;
            d.set_item("method", "bootstrap")?;
            d.set_item("alpha", bands.alpha)?;
            d.set_item("n_boot", bands.n_boot)?;
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}; expected \"asymptotic\" or \"bootstrap\""
            )))
        }
    }
    Ok(d)
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
        tsecon_filters::hp_filter_one_sided(&vec1(&y), lamb)
    } else {
        tsecon_filters::hp_filter(&vec1(&y), lamb)
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
    let dec = tsecon_filters::bk_filter(&vec1(&y), low, high, k).map_err(to_py)?;
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
    let dec = tsecon_filters::cf_filter(&vec1(&y), low, high, drift).map_err(to_py)?;
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
    let r = tsecon_filters::hamilton_filter(&vec1(&y), h, p).map_err(to_py)?;
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
    let r = tsecon_forecast::dm_test(&vec1(&e1), &vec1(&e2), h, l).map_err(to_py)?;
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
    let a = vec1(&actual);
    let a = a.as_slice();
    let p = vec1(&forecast);
    let p = p.as_slice();
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
        d.set_item("mase", f::mase(a, p, &vec1(&ins), period).map_err(to_py)?)?;
        d.set_item("rmsse", f::rmsse(a, p, &vec1(&ins), period).map_err(to_py)?)?;
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
    let r = tsecon_forecast::theta_forecast(&vec1(&y), period, steps).map_err(to_py)?;
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
    let model = GarchModel::new(&vec1(&y), spec).map_err(to_py)?;
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

/// Hierarchical (empirical-Bayes / ML-II) Minnesota-BVAR: select the prior
/// tightness lambda1 by maximizing the closed-form marginal likelihood
/// (Giannone-Lenza-Primiceri 2015), then refit the conjugate posterior at
/// the optimum — a drop-in richer `bvar_fit` that tunes its own shrinkage.
///
/// `optimize` is "lambda1" (default) or "lambda1+lambda3"; `hyperprior` is
/// "none" (pure ML-II) or "glp" (GLP Gamma, mode 0.2, sd 0.4 — MAP-II).
/// Returns the selected lambdas, the log marginal likelihood and log
/// posterior, the posterior coefficient/Sigma means, the pre-scan ML
/// profile, and the fixed-lambda reference the optimum dominates.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, delta = 0.0, lambda0 = 100.0, lambda3 = 1.0, lambda1_init = 0.2, lambda1_lo = 1e-4, lambda1_hi = 10.0, optimize = "lambda1", hyperprior = "none", n_grid = 25, max_iter = 200, tol = 1e-8))]
#[allow(clippy::too_many_arguments)]
fn bvar_hierarchical<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    delta: f64,
    lambda0: f64,
    lambda3: f64,
    lambda1_init: f64,
    lambda1_lo: f64,
    lambda1_hi: f64,
    optimize: &str,
    hyperprior: &str,
    n_grid: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let a = data.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);

    let optimize_lambda3 = match optimize {
        "lambda1" => false,
        "lambda1+lambda3" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "optimize must be 'lambda1' or 'lambda1+lambda3', got '{other}'"
            )))
        }
    };
    let hyper = match hyperprior {
        "none" => tsecon_bayes::Hyperprior::None,
        "glp" => tsecon_bayes::Hyperprior::Glp,
        other => {
            return Err(PyValueError::new_err(format!(
                "hyperprior must be 'none' or 'glp', got '{other}'"
            )))
        }
    };

    let cfg = tsecon_bayes::HierarchicalConfig {
        lambda0,
        lambda3,
        delta,
        lambda1_init,
        lambda1_lo,
        lambda1_hi,
        optimize_lambda3,
        hyperprior: hyper,
        n_grid,
        max_iter,
        tol,
    };
    let fit = tsecon_bayes::bvar_hierarchical(m.as_ref(), lags, &cfg).map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("lambda1_opt", fit.lambda1)?;
    d.set_item("lambda3_opt", fit.lambda3)?;
    d.set_item("log_marginal_likelihood", fit.log_ml)?;
    d.set_item("log_posterior", fit.log_posterior)?;
    d.set_item(
        "posterior_mean_coefs",
        mat_to_vec2_bayes(&fit.posterior.b_bar().to_owned()),
    )?;
    d.set_item(
        "sigma_posterior_mean",
        mat_to_vec2_bayes(&fit.posterior.sigma_posterior_mean().map_err(to_py)?),
    )?;
    d.set_item("grid_lambda1", fit.grid_lambda1)?;
    d.set_item("grid_log_ml", fit.grid_log_ml)?;
    d.set_item("lambda1_fixed_log_ml", fit.lambda1_fixed_log_ml)?;
    d.set_item("converged", fit.converged)?;
    d.set_item("n_evals", fit.n_evals)?;
    Ok(d)
}

/// SSVS-BVAR (George, Sun & Ni 2008): spike-and-slab stochastic-search
/// variable selection on the VAR coefficients (and, optionally, the
/// off-diagonal error precision), estimated by a 4-block Gibbs sampler with
/// semi-automatic prior scales from the OLS standard errors.
///
/// Returns posterior inclusion probabilities (`inclusion_prob`, k x n, the
/// intercept row pinned to 1), the coefficient/covariance posterior means
/// (`coef_mean` k x n same layout as `bvar_fit["posterior_mean_coefs"]`,
/// `sigma_mean` n x n), Cholesky-orthogonalized `irf_draws`
/// [draw][h][variable][shock] for credible bands, the off-diagonal precision
/// inclusion probabilities `inclusion_prob_cov` (only when `ssvs_cov`), and a
/// `diagnostics` dict.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, n_draws = 10000, burn = 2000, seed = 0, c0 = 0.1, c1 = 10.0, prior_inclusion = 0.5, ssvs_cov = false, kappa0 = 0.1, kappa1 = 10.0, prior_inclusion_cov = 0.5, gamma_a = 0.01, gamma_b = 0.01, horizon = 16, thin = 1, n_chains = 1))]
#[allow(clippy::too_many_arguments)]
fn bvar_ssvs<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    n_draws: usize,
    burn: usize,
    seed: u64,
    c0: f64,
    c1: f64,
    prior_inclusion: f64,
    ssvs_cov: bool,
    kappa0: f64,
    kappa1: f64,
    prior_inclusion_cov: f64,
    gamma_a: f64,
    gamma_b: f64,
    horizon: usize,
    thin: usize,
    n_chains: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let a = data.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let cfg = tsecon_bayes::SsvsConfig {
        lags,
        n_draws,
        burn,
        c0,
        c1,
        prior_inclusion,
        ssvs_cov,
        kappa0,
        kappa1,
        prior_inclusion_cov,
        gamma_a,
        gamma_b,
        horizon,
        thin,
        n_chains,
    };
    let res = tsecon_bayes::bvar_ssvs(m.as_ref(), &cfg, seed).map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("inclusion_prob", mat_to_vec2_bayes(&res.inclusion_prob))?;
    d.set_item("coef_mean", mat_to_vec2_bayes(&res.coef_mean))?;
    d.set_item("sigma_mean", mat_to_vec2_bayes(&res.sigma_mean))?;
    let irf: Vec<Vec<Vec<Vec<f64>>>> = res
        .irf_draws
        .iter()
        .map(|dr| dr.iter().map(mat_to_vec2_bayes).collect())
        .collect();
    d.set_item("irf_draws", irf)?;
    if let Some(cov) = &res.inclusion_prob_cov {
        d.set_item("inclusion_prob_cov", mat_to_vec2_bayes(cov))?;
    }
    let diag = PyDict::new(py);
    diag.set_item("n_draws_kept", res.n_draws_kept)?;
    diag.set_item("burn", burn)?;
    diag.set_item("thin", thin)?;
    diag.set_item("mean_model_size", res.mean_model_size)?;
    diag.set_item(
        "log_marginal_likelihood_median",
        res.log_marginal_likelihood_median,
    )?;
    if let Some(rhat) = res.rhat {
        diag.set_item("rhat", rhat)?;
    }
    if let Some(ess) = res.ess_bulk {
        diag.set_item("ess_bulk", ess)?;
    }
    d.set_item("diagnostics", diag)?;
    Ok(d)
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
    let r = spec.fit(&vec1(&y)).map_err(to_py)?;
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

/// Parse the `cumulative=` argument shared by the LP entry points.
///
/// Accepts the historical booleans (`False` -> no cumulation, `True` ->
/// outcome-only, unchanged in meaning) and the explicit spellings `"none"`,
/// `"outcome"`, `"both"`. `"both"` accumulates the impulse as well, turning
/// the coefficient into an integral multiplier.
fn parse_cumulation(arg: Option<&Bound<'_, PyAny>>) -> PyResult<tsecon_lp::Cumulation> {
    use tsecon_lp::Cumulation;
    let Some(obj) = arg else {
        return Ok(Cumulation::None);
    };
    if obj.is_none() {
        return Ok(Cumulation::None);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(if b {
            Cumulation::Outcome
        } else {
            Cumulation::None
        });
    }
    let s: String = obj.extract().map_err(|_| {
        PyValueError::new_err("cumulative must be a bool or one of \"none\", \"outcome\", \"both\"")
    })?;
    match s.as_str() {
        "none" => Ok(Cumulation::None),
        "outcome" => Ok(Cumulation::Outcome),
        "both" => Ok(Cumulation::Both),
        other => Err(PyValueError::new_err(format!(
            "unknown cumulative {other:?}; expected \"none\", \"outcome\" or \"both\"              (or a bool). Note \"outcome\" is a cumulative impulse response, not a              multiplier -- for a multiplier use tsecon.lp_multiplier"
        ))),
    }
}

/// Local projection impulse responses (Jordà 2005).
///
/// `se`: "lag_augmented" (Montiel Olea-Plagborg-Møller 2021, the default) or
/// "hac" (Newey-West; `maxlags=None` grows with the horizon).
///
/// `cumulative` selects which side(s) accumulate over the horizon:
/// `False`/`"none"` (level response), `True`/`"outcome"` (the Ramey-Zubairy
/// cumulative impulse response: cumulated outcome on the *contemporaneous*
/// impulse), or `"both"` (cumulated outcome on cumulated impulse, an OLS
/// integral multiplier). `True` keeps its historical meaning exactly.
///
/// A cumulative-outcome response is NOT a multiplier: its denominator never
/// grows with the horizon, so it rises roughly linearly in `h` by
/// construction. For an identified multiplier use `tsecon.lp_multiplier`.
///
/// Returns per-horizon irf and standard errors.
#[pyfunction]
#[pyo3(signature = (y, shock, horizons = 12, n_lag_controls = 4, se = "lag_augmented", maxlags = None, cumulative = None))]
#[allow(clippy::too_many_arguments)]
fn lp<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    shock: PyReadonlyArray1<'py, f64>,
    horizons: usize,
    n_lag_controls: usize,
    se: &str,
    maxlags: Option<usize>,
    cumulative: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut spec = tsecon_lp::LpSpec::new(horizons, n_lag_controls)
        .with_cumulation(parse_cumulation(cumulative)?);
    spec = match se {
        "lag_augmented" => spec,
        "hac" => spec.with_hac(maxlags),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown se {other:?}; expected \"lag_augmented\" or \"hac\""
            )))
        }
    };
    let r = tsecon_lp::lp(&vec1(&y), &vec1(&shock), spec).map_err(to_py)?;
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
///
/// `cumulative` takes `False`/`"none"`, `True`/`"outcome"` or `"both"`, with
/// the same meaning as in `tsecon.lp`; the instrument stays contemporaneous
/// in every mode. `True`/`"outcome"` cumulates ONLY the outcome, so it is a
/// cumulative impulse response (cumulated outcome per unit of contemporaneous
/// impulse) and NOT a multiplier -- it grows without bound in the horizon
/// because its denominator does not accumulate. For the Ramey-Zubairy
/// integral multiplier use `tsecon.lp_multiplier`.
#[pyfunction]
#[pyo3(signature = (y, impulse, instrument, horizons = 8, n_lag_controls = 4, cumulative = None))]
fn lp_iv<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    impulse: PyReadonlyArray1<'py, f64>,
    instrument: PyReadonlyArray1<'py, f64>,
    horizons: usize,
    n_lag_controls: usize,
    cumulative: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = tsecon_lp::LpSpec::new(horizons, n_lag_controls)
        .with_cumulation(parse_cumulation(cumulative)?);
    let r =
        tsecon_lp::lp_iv(&vec1(&y), &vec1(&impulse), &vec1(&instrument), spec).map_err(to_py)?;
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

/// Ramey-Zubairy (2018) integral multiplier by one-step LP-IV.
///
/// At each horizon `h` this regresses the CUMULATED outcome
/// `sum_{j=0..h} y_{t+j}` on the CUMULATED impulse `sum_{j=0..h} x_{t+j}`,
/// instrumented by the contemporaneous `instrument`, controlling for a
/// constant and `n_lag_controls` lags of BOTH the outcome and the impulse.
/// Because both sides accumulate over the same window, the coefficient is a
/// multiplier: extra cumulated outcome per extra cumulated impulse through
/// horizon `h`.
///
/// This is the estimator you want for a fiscal (or any integral) multiplier.
/// `lp_iv(..., cumulative=True)` is NOT: it accumulates only the outcome, so
/// it reports cumulated output per unit of CONTEMPORANEOUS spending, which
/// rises roughly linearly in the horizon by construction.
///
/// Standard errors: `se` is the kernel-HAC standard error of the multiplier
/// coefficient itself. The multiplier is estimated as a single 2SLS
/// parameter, not as a ratio of two separately estimated responses, so this
/// is honest inference on the reported number -- not a delta-method
/// approximation and not one leg's SE relabelled.
///
/// Returns a dict with `horizons`, `multiplier`, `se`, `first_stage_f`
/// (weak-instrument concern below 10), and the two reduced-form legs
/// `cumulative_outcome` / `cumulative_impulse` (no SEs; their ratio equals
/// `multiplier` by the just-identified IV algebra).
#[pyfunction]
#[pyo3(signature = (y, impulse, instrument, horizons = 20, n_lag_controls = 4, maxlags = None))]
fn lp_multiplier<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    impulse: PyReadonlyArray1<'py, f64>,
    instrument: PyReadonlyArray1<'py, f64>,
    horizons: usize,
    n_lag_controls: usize,
    maxlags: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut spec = tsecon_lp::LpSpec::new(horizons, n_lag_controls);
    if maxlags.is_some() {
        spec = spec.with_hac(maxlags);
    }
    let r = tsecon_lp::lp_multiplier(&vec1(&y), &vec1(&impulse), &vec1(&instrument), spec)
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
    d.set_item("multiplier", r.multiplier.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    d.set_item("first_stage_f", r.first_stage_f.into_pyarray(py))?;
    d.set_item("cumulative_outcome", r.cumulative_outcome.into_pyarray(py))?;
    d.set_item("cumulative_impulse", r.cumulative_impulse.into_pyarray(py))?;
    d.set_item(
        "nobs_per_h",
        r.nobs_per_h
            .iter()
            .map(|&v| v as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
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
    let coef = tsecon_ml::ridge(m.as_ref(), &vec1(&y), alpha).map_err(to_py)?;
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
        tsecon_ml::elastic_net(m.as_ref(), &vec1(&y), alpha, l1_ratio, opts).map_err(to_py)?;
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

/// Zero + sign restricted Bayesian SVAR (Rubio-Ramirez-Waggoner-Zha 2010
/// exact-zero column recursion; Arias-Rubio-Ramirez-Waggoner 2018 importance
/// weighting) on the Minnesota-NIW posterior. A superset of
/// `sign_restricted_svar`.
///
/// `sign_restrictions` are (variable, shock, horizon, sign) tuples with sign in
/// {"+","-"} (may be empty); `zero_restrictions` are (variable, shock, horizon)
/// tuples imposing Theta_h[(variable,shock)] = 0 exactly (horizon 0 = impact).
/// At least one list must be non-empty. Returns per-(horizon, variable, shock)
/// `set_min`/`set_max` (weight-invariant identified-set envelope) and
/// ARW-weighted `quantiles` at probs=[0.05,0.16,0.50,0.84,0.95], plus per-draw
/// `weights` (normalized), `ess`, and acceptance `diagnostics`. With
/// strict-upper-triangle impact zeros and no signs it reproduces
/// `var_irf(orth=True)` deterministically.
#[pyfunction]
#[pyo3(signature = (data, sign_restrictions, zero_restrictions, lags = 2, horizon = 12, n_draws = 500, max_tries = 400, seed = 0, lambda1 = 0.2, weighted = true))]
#[allow(clippy::too_many_arguments)]
fn zero_sign_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    sign_restrictions: Vec<(usize, usize, usize, String)>,
    zero_restrictions: Vec<(usize, usize, usize)>,
    lags: usize,
    horizon: usize,
    n_draws: usize,
    max_tries: usize,
    seed: u64,
    lambda1: f64,
    weighted: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_ident::{
        Sign, SignRestriction, SignRestrictionSet, ZeroRestriction, ZeroRestrictionSet,
        ZeroSignSampler,
    };
    if sign_restrictions.is_empty() && zero_restrictions.is_empty() {
        return Err(PyValueError::new_err(
            "at least one of sign_restrictions / zero_restrictions must be non-empty",
        ));
    }
    let a = data.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let n_vars = a.ncols();
    let prior = tsecon_bayes::MinnesotaNiwPrior::new(m.as_ref(), lags, 100.0, lambda1, 1.0, 0.0)
        .map_err(to_py)?;
    let posterior = prior.posterior(m.as_ref()).map_err(to_py)?;

    // Sign restrictions (optional; None => pure-zero / recursive identification).
    let signs = if sign_restrictions.is_empty() {
        None
    } else {
        let mut rs = Vec::with_capacity(sign_restrictions.len());
        for (v, s, h, sign) in sign_restrictions {
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
        Some(SignRestrictionSet::new(rs, n_vars, horizon).map_err(to_py)?)
    };

    // Zero restrictions (may be empty => degenerates to the sign-only sampler).
    let zero_rs: Vec<ZeroRestriction> = zero_restrictions
        .into_iter()
        .map(|(v, s, h)| ZeroRestriction::at(v, s, h))
        .collect();
    let zeros = ZeroRestrictionSet::new(zero_rs, n_vars, horizon).map_err(to_py)?;

    let result = ZeroSignSampler::new(horizon, n_draws, max_tries)
        .map_err(to_py)?
        .with_weighting(weighted)
        .run(&posterior, signs.as_ref(), &zeros, seed)
        .map_err(to_py)?;

    // quantiles[h][var][shock][prob], set_min/set_max[h][var][shock]
    let hs = horizon + 1;
    let mut quantiles = vec![vec![vec![Vec::<f64>::new(); n_vars]; n_vars]; hs];
    let mut set_min = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
    let mut set_max = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
    for h in 0..hs {
        for i in 0..n_vars {
            for j in 0..n_vars {
                let bp = result.summary_point(i, j, h).map_err(to_py)?;
                quantiles[h][i][j] = bp.quantiles.clone();
                set_min[h][i][j] = bp.min;
                set_max[h][i][j] = bp.max;
            }
        }
    }
    let d = PyDict::new(py);
    d.set_item("probs", result.probs().to_vec())?;
    d.set_item("quantiles", quantiles)?;
    d.set_item("set_min", set_min)?;
    d.set_item("set_max", set_max)?;
    d.set_item("weights", result.weights().to_vec())?;
    d.set_item("ess", result.ess())?;
    let diag = result.diagnostics();
    let dd = PyDict::new(py);
    dd.set_item("posterior_draws_used", diag.posterior_draws_used)?;
    dd.set_item("rotations_tried", diag.rotations_tried)?;
    dd.set_item("accepted", diag.accepted)?;
    dd.set_item("acceptance_rate", diag.acceptance_rate)?;
    d.set_item("diagnostics", dd)?;
    Ok(d)
}

/// Blanchard-Quah long-run SVAR: closed-form structural IRFs under the
/// recursive frequency-zero restriction (the analog of R `vars::BQ`).
///
/// Returns `impact` (B), `long_run` (LR), `irf`, `cumulative_irf`, `fevd`,
/// and `long_run_multiplier` (C(1)). Point estimates, no RNG.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, horizon = 12, trend = "c", restrictions = None, normalize = "long_run"))]
fn long_run_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    trend: &str,
    restrictions: Option<Vec<(usize, usize)>>,
    normalize: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let normalize_impact = match normalize {
        "long_run" => false,
        "impact" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown normalize {other:?}; expected \"long_run\" or \"impact\""
            )))
        }
    };
    // Reduced form via the existing OLS VAR path (this is the only use of
    // tsecon-var here; r.coefs is already the split [A_1..A_p] form).
    let r = var_results(&data, lags, trend)?;
    let coefs: Vec<_> = r.coefs.iter().map(|m| m.as_ref()).collect();
    let res = tsecon_ident::long_run::long_run_svar(
        &coefs,
        r.sigma_u.as_ref(),
        horizon,
        restrictions.as_deref(),
        normalize_impact,
    )
    .map_err(to_py)?;

    // irf[h][i][j] and its running-sum cumulative (level response in differences).
    let irf: Vec<Vec<Vec<f64>>> = res.irf.iter().map(mat_to_vec2).collect();
    let mut cum = irf.clone();
    for h in 1..cum.len() {
        for i in 0..cum[h].len() {
            for j in 0..cum[h][i].len() {
                cum[h][i][j] += cum[h - 1][i][j];
            }
        }
    }

    // Structural FEVD: shares of variable i's (h+1)-step FEV due to shock j.
    // Numerator = sum_{s<=h} Theta_s[i][j]^2; denominator = i-th diagonal of
    // the cumulative Theta Theta'. Rows sum to 1 (orthonormal structural shocks).
    let k = res.impact.nrows();
    let hs = res.irf.len();
    let mut fevd = vec![vec![vec![0.0_f64; k]; k]; hs];
    let mut contrib = vec![vec![0.0_f64; k]; k];
    for (th, fevd_h) in res.irf.iter().zip(fevd.iter_mut()) {
        for i in 0..k {
            for j in 0..k {
                contrib[i][j] += th[(i, j)] * th[(i, j)];
            }
        }
        for i in 0..k {
            let total: f64 = (0..k).map(|j| contrib[i][j]).sum();
            for j in 0..k {
                fevd_h[i][j] = if total > 0.0 {
                    contrib[i][j] / total
                } else {
                    0.0
                };
            }
        }
    }

    let d = PyDict::new(py);
    d.set_item("impact", mat_to_vec2(&res.impact))?;
    d.set_item("long_run", mat_to_vec2(&res.long_run))?;
    d.set_item("long_run_multiplier", mat_to_vec2(&res.long_run_multiplier))?;
    d.set_item("irf", irf)?;
    d.set_item("cumulative_irf", cum)?;
    d.set_item("fevd", fevd)?;
    Ok(d)
}

/// Max-share / maximum-FEV structural shock (Uhlig 2004; Francis, Owyang,
/// Roush & DiCecio 2014 main-business-cycle shock; Barsky & Sims 2011 news
/// shock). Identifies the single unit-variance structural shock whose share of
/// the `target` variable's forecast-error variance, accumulated over the
/// horizon window `[h0, h1]`, is maximal — the leading eigenvector of a small
/// symmetric PSD matrix built from the orthogonalized MA coefficients.
///
/// `weighting="window"` (Uhlig/Francis, default) maximizes the incremental
/// windowed FEV (its `share_window` is an exact accumulated-FEV fraction);
/// `"cumulative"` (Barsky-Sims) maximizes the window-mean cumulative FEV share.
/// `exclude_impact=True` imposes zero impact on the target (Barsky-Sims news
/// shock). `sign` pins the identified shock's sign
/// ("cumsum"|"impact"|"none"). Deterministic — no seed.
///
/// Keys: `irf` [horizon+1][k], `impact` [k], `q` [k], `share_window` (float),
/// `fev_share` [horizon+1], `eigenvalues` (ascending; length k, or k-1 when
/// `exclude_impact`).
#[pyfunction]
#[pyo3(signature = (
    data,
    lags = 2,
    target = 0,
    h0 = 0,
    h1 = 40,
    horizon = 40,
    trend = "c",
    exclude_impact = false,
    weighting = "window",
    sign = "cumsum",
))]
#[allow(clippy::too_many_arguments)]
fn max_share_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    target: usize,
    h0: usize,
    h1: usize,
    horizon: usize,
    trend: &str,
    exclude_impact: bool,
    weighting: &str,
    sign: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_ident::{max_share_shock, MaxShareSign, MaxShareWeighting};

    if h0 > h1 {
        return Err(PyValueError::new_err(format!(
            "window bounds must satisfy h0 <= h1, got h0={h0}, h1={h1}"
        )));
    }
    if h1 > horizon {
        return Err(PyValueError::new_err(format!(
            "window end h1={h1} must not exceed horizon={horizon}"
        )));
    }
    let weighting_kind = match weighting {
        "window" => MaxShareWeighting::Window,
        "cumulative" => MaxShareWeighting::Cumulative,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown weighting {other:?}; expected \"window\" or \"cumulative\""
            )))
        }
    };
    let sign_kind = match sign {
        "cumsum" => MaxShareSign::Cumsum,
        "impact" => MaxShareSign::Impact,
        "none" => MaxShareSign::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown sign {other:?}; expected \"cumsum\", \"impact\", or \"none\""
            )))
        }
    };

    // Reduced form via the shared VAR OLS helper (matches var_fit at 1e-8),
    // then the orthogonalized MA Theta_s = Psi_s P (identical to the array
    // behind var_fevd / var_irf orth=True).
    let r = var_results(&data, lags, trend)?;
    let theta = r.orth_ma_rep(horizon).map_err(to_py)?;

    let out = max_share_shock(
        &theta,
        target,
        h0,
        h1,
        exclude_impact,
        weighting_kind,
        sign_kind,
    )
    .map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("irf", out.irf)?;
    d.set_item("impact", out.impact)?;
    d.set_item("q", out.q)?;
    d.set_item("share_window", out.share_window)?;
    d.set_item("fev_share", out.fev_share)?;
    d.set_item("eigenvalues", out.eigenvalues)?;
    Ok(d)
}

/// Proxy SVAR / external-instrument identification (SVAR-IV): one structural
/// shock from a single external instrument (Stock-Watson 2018; Mertens-Ravn
/// 2013; Gertler-Karadi 2015; Montiel-Olea-Stock-Watson 2021).
///
/// The residual-instrument covariance pins the target shock's impact column up
/// to scale; the unit-effect normalization fixes scale and sign so a positive
/// shock raises `norm_var` by `unit` on impact. `proxy` aligns to `data` rows
/// (pass length n_obs -- the first `lags` presample rows are dropped -- or the
/// residual length T directly); NaN entries outside the instrument's
/// availability window are dropped from the moments and the first stage.
///
/// Returns `irf` (horizon+1, n), `impact`/`relative_impact`/`cov_um` (n),
/// `first_stage_f` (weak below 10), `reliability` = Corr(m, u_norm)^2,
/// `n_proxy` (effective obs), and the estimated structural `shock` (T). Point
/// estimate only: valid bands need the Jentsch-Lunsford (2019) moving-block
/// bootstrap (documented v2 extension).
#[pyfunction]
#[pyo3(signature = (data, proxy, lags = 2, horizon = 12, norm_var = 0, unit = 1.0, trend = "c", robust_f = true))]
#[allow(clippy::too_many_arguments)]
fn proxy_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    proxy: PyReadonlyArray1<'py, f64>,
    lags: usize,
    horizon: usize,
    norm_var: usize,
    unit: f64,
    trend: &str,
    robust_f: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let r = var_results(&data, lags, trend)?;
    let psi = r.ma_rep(horizon).map_err(to_py)?;
    let n_obs = data.as_array().nrows();
    let t = r.resid.nrows();

    // Align the proxy to the residual sample: accept the full n_obs series
    // (drop the first `lags` presample rows) or a series already of length T.
    let pv = vec1(&proxy);
    let proxy_aligned: Vec<f64> = if pv.len() == t {
        pv
    } else if pv.len() == n_obs {
        pv[lags..].to_vec()
    } else {
        return Err(PyValueError::new_err(format!(
            "proxy length {} must equal the number of observations {} or the residual sample length {}",
            pv.len(),
            n_obs,
            t
        )));
    };

    let res = tsecon_ident::proxy_svar(
        r.resid.as_ref(),
        &proxy_aligned,
        &psi,
        r.sigma_u.as_ref(),
        norm_var,
        unit,
        robust_f,
    )
    .map_err(to_py)?;

    let d = PyDict::new(py);
    // irf is (H+1, n); returned as a nested list, matching var_fit's `params`
    // (Vec<Vec<f64>> -> list-of-lists; users np.asarray it).
    d.set_item("irf", res.irf)?;
    d.set_item("impact", res.impact.into_pyarray(py))?;
    d.set_item("relative_impact", res.relative_impact.into_pyarray(py))?;
    d.set_item("first_stage_f", res.first_stage_f)?;
    d.set_item("reliability", res.reliability)?;
    d.set_item("cov_um", res.cov_um.into_pyarray(py))?;
    d.set_item("n_proxy", res.n_proxy)?;
    d.set_item("shock", res.shock.into_pyarray(py))?;
    Ok(d)
}

/// Non-Gaussian / independent-component SVAR identification (Lanne-Meitz-
/// Saikkonen 2017; Gourieroux-Monfort-Renne 2017; FastICA/Hyvarinen).
///
/// Point-identifies the structural impact matrix `B` in `u_t = B eps_t` from
/// the reduced-form residuals ALONE -- no sign, zero, long-run, or proxy
/// restriction -- by exploiting the statistical INDEPENDENCE and NON-
/// GAUSSIANITY of the structural shocks (at most one Gaussian). Whitens by
/// `Sigma_u^{-1/2}`, finds the orthogonal rotation maximizing non-Gaussianity
/// via a deterministic symmetric FastICA fixed point (log-cosh contrast,
/// identity init -- bit-reproducible), then `B = Sigma_u^{1/2} Q`. Columns are
/// ordered by `order_by` ("kurtosis"|"colnorm") and signed max-abs-positive;
/// both are CONVENTIONS. STATISTICAL identification: it FAILS if the shocks are
/// Gaussian, and a `shock_kurtosis` near zero flags a weakly identified column.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, horizon = 12, trend = "c", contrast = "logcosh", max_iter = 200, tol = 1e-8, order_by = "kurtosis"))]
#[allow(clippy::too_many_arguments)]
fn nongaussian_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    trend: &str,
    contrast: &str,
    max_iter: usize,
    tol: f64,
    order_by: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_ident::{nongaussian_svar as ng, Contrast, OrderBy};
    use tsecon_var::tsecon_linalg::faer::Mat;

    let contrast_kind = match contrast {
        "logcosh" => Contrast::LogCosh,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown contrast {other:?}; expected \"logcosh\""
            )))
        }
    };
    let order_kind = match order_by {
        "kurtosis" => OrderBy::Kurtosis,
        "colnorm" => OrderBy::ColumnNorm,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown order_by {other:?}; expected \"kurtosis\" or \"colnorm\""
            )))
        }
    };

    // Reduced form via the shared VAR OLS helper (matches var_fit at 1e-8):
    // residuals U and covariance Sigma_u for whitening.
    let r = var_results(&data, lags, trend)?;
    let k = r.neqs;

    // Pack coefficients as (1 + k*lags) x k -- intercept row then lag blocks,
    // the layout structural_ma expects -- so the IRF path is correct for any
    // trend (the intercept row is zero when trend = "n").
    let mut b_coefs = Mat::<f64>::zeros(1 + k * lags, k);
    for i in 0..k {
        b_coefs[(0, i)] = r.intercept[i];
    }
    for (l, a) in r.coefs.iter().enumerate() {
        // structural_ma reads A_l[(i,j)] = coef of y_{t-l,j} in eq i at
        // b[(1 + l*k + j, i)].
        for i in 0..k {
            for j in 0..k {
                b_coefs[(1 + l * k + j, i)] = a[(i, j)];
            }
        }
    }

    let out = ng(
        r.resid.as_ref(),
        r.sigma_u.as_ref(),
        b_coefs.as_ref(),
        lags,
        horizon,
        contrast_kind,
        max_iter,
        tol,
        order_kind,
    )
    .map_err(to_py)?;

    let irf: Vec<Vec<Vec<f64>>> = out.irf.iter().map(mat_to_vec2).collect();

    let d = PyDict::new(py);
    d.set_item("impact", mat_to_vec2(&out.impact))?; // [var][shock]
    d.set_item("irf", irf)?; // [h][var][shock]
    d.set_item("rotation", mat_to_vec2(&out.rotation))?; // Q [whitened][shock]
    d.set_item("shock_kurtosis", out.shock_kurtosis)?; // [shock]
    d.set_item("converged", out.converged)?;
    d.set_item("n_iter", out.n_iter)?;
    d.set_item("order", out.order)?; // [position] -> raw index
    Ok(d)
}

/// Identification through heteroskedasticity (Rigobon 2003; Lanne-Lutkepohl
/// 2008), exactly two known variance regimes. Recovers the constant SVAR
/// impact matrix B (up to column sign/order) from the two within-regime
/// reduced-form residual covariances via a generalized eigendecomposition;
/// point-identified iff the variance ratios are pairwise distinct.
#[pyfunction]
#[pyo3(signature = (data, regime_labels, lags = 2, horizon = 12, trend = "c", base_regime = None, sign_normalization = "max"))]
#[allow(clippy::too_many_arguments)]
fn hetero_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    regime_labels: Vec<i64>,
    lags: usize,
    horizon: usize,
    trend: &str,
    base_regime: Option<i64>,
    sign_normalization: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    if lags < 1 {
        return Err(PyValueError::new_err("lags must be at least 1"));
    }
    let sign = match sign_normalization {
        "max" => tsecon_ident::SignConvention::MaxAbs,
        "diag" => tsecon_ident::SignConvention::Diagonal,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown sign_normalization {other:?}; expected \"max\" or \"diag\""
            )))
        }
    };
    let a = data.as_array();
    let t_total = a.nrows();
    let n = a.ncols();
    if regime_labels.len() != t_total {
        return Err(PyValueError::new_err(format!(
            "regime_labels length {} != number of observations {}",
            regime_labels.len(),
            t_total
        )));
    }

    // One pooled reduced-form VAR on the full sample.
    let r = var_results(&data, lags, trend)?;
    let resid = &r.resid; // (T - p) x n
    let t_eff = resid.nrows();

    // Residual row rr <-> regime_labels[lags + rr]; require exactly 2 labels.
    let mut distinct: Vec<i64> = Vec::new();
    for rr in 0..t_eff {
        let lab = regime_labels[lags + rr];
        if !distinct.contains(&lab) {
            distinct.push(lab);
        }
    }
    if distinct.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "expected exactly 2 distinct regime labels among the {t_eff} \
             residual-aligned observations, found {}",
            distinct.len()
        )));
    }
    // Regime 1 (Lambda = I base): base_regime if given, else the smaller label.
    let base = match base_regime {
        Some(b) => {
            if !distinct.contains(&b) {
                return Err(PyValueError::new_err(format!(
                    "base_regime {b} is not among the regime labels"
                )));
            }
            b
        }
        None => distinct[0].min(distinct[1]),
    };
    let other = if base == distinct[0] {
        distinct[1]
    } else {
        distinct[0]
    };

    let mut rows1: Vec<usize> = Vec::new();
    let mut rows2: Vec<usize> = Vec::new();
    for rr in 0..t_eff {
        if regime_labels[lags + rr] == base {
            rows1.push(rr);
        } else {
            rows2.push(rr);
        }
    }
    let n1 = rows1.len();
    let n2 = rows2.len();
    if n1 < n || n2 < n {
        return Err(PyValueError::new_err(format!(
            "each regime needs at least n={n} residuals for a nonsingular \
             within-regime covariance (got n1={n1}, n2={n2})"
        )));
    }

    // Decomposition covariance: ML divisor, raw residuals (mean ~ 0).
    let sigma_ml = |rows: &[usize]| -> Mat<f64> {
        let ns = rows.len() as f64;
        Mat::from_fn(n, n, |i, j| {
            let mut acc = 0.0;
            for &rr in rows {
                acc += resid[(rr, i)] * resid[(rr, j)];
            }
            acc / ns
        })
    };
    // Box's M covariance: unbiased divisor, mean-subtracted.
    let sigma_boxm = |rows: &[usize]| -> Mat<f64> {
        let ns = rows.len();
        let mut mean = vec![0.0_f64; n];
        for &rr in rows {
            for i in 0..n {
                mean[i] += resid[(rr, i)];
            }
        }
        for m in &mut mean {
            *m /= ns as f64;
        }
        Mat::from_fn(n, n, |i, j| {
            let mut acc = 0.0;
            for &rr in rows {
                acc += (resid[(rr, i)] - mean[i]) * (resid[(rr, j)] - mean[j]);
            }
            acc / (ns as f64 - 1.0)
        })
    };
    let sigma1 = sigma_ml(&rows1);
    let sigma2 = sigma_ml(&rows2);
    let s1_boxm = sigma_boxm(&rows1);
    let s2_boxm = sigma_boxm(&rows2);

    let decomp =
        tsecon_ident::hetero_decompose(sigma1.as_ref(), sigma2.as_ref(), sign).map_err(to_py)?;
    let bm = tsecon_ident::box_m_test(&[(s1_boxm.as_ref(), n1), (s2_boxm.as_ref(), n2)])
        .map_err(to_py)?;

    // Structural IRF: Theta_h = Psi_h @ B.
    let psi = r.ma_rep(horizon).map_err(to_py)?;
    let structural_irf: Vec<Vec<Vec<f64>>> = psi
        .iter()
        .map(|p| mat_to_vec2(&(p.as_ref() * decomp.b.as_ref())))
        .collect();

    // identified heuristic (documented tol; the honest inference is the
    // returned numbers + covariance_equality, not this flag alone).
    let max_lam = decomp.lambda.iter().fold(0.0_f64, |m, &l| m.max(l.abs()));
    let tol = 1e-6 * max_lam.max(1.0);
    let min_dist_unity = decomp
        .ratio_dist_from_unity
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let identified = decomp.min_ratio_gap > tol && min_dist_unity > tol;

    let d = PyDict::new(py);
    d.set_item("B", mat_to_vec2(&decomp.b))?;
    d.set_item("variance_ratios", decomp.lambda.clone())?;
    d.set_item("structural_irf", structural_irf)?;
    d.set_item("min_ratio_gap", decomp.min_ratio_gap)?;
    d.set_item(
        "ratio_dist_from_unity",
        decomp.ratio_dist_from_unity.clone(),
    )?;
    d.set_item("identified", identified)?;
    let ce = PyDict::new(py);
    ce.set_item("statistic", bm.statistic)?;
    ce.set_item("dof", bm.dof)?;
    ce.set_item("pvalue", bm.pvalue)?;
    ce.set_item("distinct_regimes", bm.pvalue < 0.05)?;
    d.set_item("covariance_equality", ce)?;
    d.set_item("sigma_regime1", mat_to_vec2(&sigma1))?;
    d.set_item("sigma_regime2", mat_to_vec2(&sigma2))?;
    d.set_item("regime1_label", base)?;
    d.set_item("regime2_label", other)?;
    d.set_item("regime_sizes", vec![n1 as u64, n2 as u64])?;
    d.set_item("n_vars", n)?;
    d.set_item("horizon", horizon)?;
    d.set_item("lags", lags)?;
    d.set_item("sign_convention", sign_normalization)?;
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
    let r = tsecon_panel::panel_lp(&data, &vec1(&shock), &cfg).map_err(to_py)?;
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
        &vec1(&e_small),
        &vec1(&e_large),
        &vec1(&yhat_small),
        &vec1(&yhat_large),
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
    let r = tsecon_forecast::gw_test(&vec1(&loss1), &vec1(&loss2), lrv_lags).map_err(to_py)?;
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
        &vec1(&x),
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
        &vec1(&x),
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
        &vec1(&x),
        &vec1(&y),
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
    let ys = vec1(&y);
    let ys = ys.as_slice();
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
    let r = tsecon_midas::umidas(&vec1(&y), &cols, se).map_err(to_py)?;
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
    let r = vec1(&returns);
    let r = r.as_slice();
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
    let fit = tsecon_realized::har_rv(&vec1(&rv), &cfg).map_err(to_py)?;
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
    let yv = vec1(&y);
    let yv = yv.as_slice();
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
    let fit = tsecon_ml::adaptive_lasso(m.as_ref(), &vec1(&y), alpha, l1_ratio, gamma, opts)
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
        tsecon_ml::regularization_path(m.as_ref(), &vec1(&y), l1_ratio, opts).map_err(to_py)?;
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
    let res = bt.run(&vec1(&y), point).map_err(to_py)?;

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
    let (mat, yld) = (vec1(&maturities), vec1(&yields));
    let mat = mat.as_slice();
    let yld = yld.as_slice();
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
    let (mat, yld) = (vec1(&maturities), vec1(&yields));
    let mat = mat.as_slice();
    let yld = yld.as_slice();
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
    let fit = tsecon_midas::weighted_midas(&vec1(&y), &cols, sch, start).map_err(to_py)?;
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
#[pyo3(signature = (y, shock, state_indicator, horizons = 12, n_lag_controls = 4, se = "lag_augmented", maxlags = None, cumulative = None))]
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
    cumulative: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut spec = tsecon_lp::LpSpec::new(horizons, n_lag_controls)
        .with_cumulation(parse_cumulation(cumulative)?);
    spec = match se {
        "lag_augmented" => spec,
        "hac" => spec.with_hac(maxlags),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown se {other:?}; expected \"lag_augmented\" or \"hac\""
            )))
        }
    };
    let r = tsecon_lp::lp_state(&vec1(&y), &vec1(&shock), &vec1(&state_indicator), spec)
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
    let mat = vec1(&maturities);
    let mat = mat.as_slice();
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
    let pol = vec1(&policy);
    let pol = pol.as_slice();
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
    let r = vec1(&returns);
    let r = r.as_slice();
    tsecon_realized::realized_quarticity(r).map_err(to_py)
}

/// Tripower quarticity
/// `TQ = n mu_{4/3}^{-3} sum |r_i|^{4/3}|r_{i-1}|^{4/3}|r_{i-2}|^{4/3}`
/// (Barndorff-Nielsen & Shephard 2004), the jump-robust estimator of
/// integrated quarticity `int sigma^4 ds` used to studentize the BNS ratio
/// jump test.
#[pyfunction]
fn tripower_quarticity(returns: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let r = vec1(&returns);
    let r = r.as_slice();
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
    let r = vec1(&returns);
    let r = r.as_slice();
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
    let h = vec1(&high);
    let h = h.as_slice();
    let l = vec1(&low);
    let l = l.as_slice();
    match method {
        "parkinson" => tsecon_realized::parkinson(h, l).map_err(to_py),
        "garman_klass" => {
            let open = open.ok_or_else(|| {
                PyValueError::new_err("garman_klass requires the open and close series")
            })?;
            let close = close.ok_or_else(|| {
                PyValueError::new_err("garman_klass requires the open and close series")
            })?;
            let o = vec1(&open);
            let o = o.as_slice();
            let c = vec1(&close);
            let c = c.as_slice();
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
    let yv = vec1(&y);
    let model = tsecon_gas::GasModel::new(&yv, dens).map_err(to_py)?;
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
        d.set_item(
            "forecast",
            res.forecast(horizon).map_err(to_py)?.into_pyarray(py),
        )?;
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
        units.push(tsecon_panelts::PanelUnit::new(vec1(yi), cols));
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
#[pyo3(signature = (data, n_factors = 1, factor_order = 2, method = "two_step"))]
fn dfm_nowcast<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    n_factors: usize,
    factor_order: usize,
    method: &str,
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
    // "two_step" is the Doz-Giannone-Reichlin estimator; "mle" is the exact
    // one-step Gaussian MLE (single factor only, r = 1).
    let nc = match method {
        "two_step" => {
            tsecon_nowcast::Nowcaster::fit_two_step(train.as_ref(), n_factors, factor_order)
                .map_err(to_py)?
        }
        "mle" => {
            if n_factors != 1 {
                return Err(PyValueError::new_err(
                    "method=\"mle\" supports a single factor only (n_factors=1)",
                ));
            }
            tsecon_nowcast::Nowcaster::fit_mle(train.as_ref(), factor_order).map_err(to_py)?
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}; expected \"two_step\" or \"mle\""
            )))
        }
    };
    let res = nc.nowcast_panel(full.as_ref()).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("nowcast", res.values.clone().into_pyarray(py))?;
    d.set_item("edge_factor", res.edge_factor.clone().into_pyarray(py))?;
    d.set_item("loglik", res.smoothing.loglik)?;
    d.set_item("fit_loglik", nc.loglik())?;
    let factors: Vec<Vec<f64>> = nc.smoothed_factors().to_vec();
    d.set_item("smoothed_factors", factors)?;
    d.set_item("n_factors", nc.n_factors())?;
    d.set_item("factor_order", nc.factor_order())?;
    Ok(d)
}

/// Pooled Mean Group (PMG) ARDL(1,1) panel estimator (Pesaran-Shin-Smith
/// 1999).
///
/// Estimates a panel error-correction model in which the LONG-RUN
/// coefficients are pooled (common across units) by maximum likelihood,
/// while the error-correction speed and short-run dynamics stay
/// unit-specific. `ys`/`xs` are per-unit response vectors and `T_i x k`
/// regressor matrices (as for `panel_mean_group`). Returns the pooled
/// long-run `theta` and its `theta_se`, the average adjustment speed
/// `phi_bar`, the per-unit speeds `phi`, per-unit innovation variances
/// `sigma2`, the `loglik`, and iteration/shape info. Complements the
/// mean-group and CCE-MG estimators: PMG pools the long run, they do not.
#[pyfunction]
fn panel_pmg<'py>(
    py: Python<'py>,
    ys: Vec<PyReadonlyArray1<'py, f64>>,
    xs: Vec<numpy::PyReadonlyArray2<'py, f64>>,
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
        units.push(tsecon_panelts::PanelUnit::new(vec1(yi), cols));
    }
    let r = tsecon_panelts::pmg(&units).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("theta", r.theta.clone().into_pyarray(py))?;
    d.set_item("theta_se", r.theta_se.clone().into_pyarray(py))?;
    d.set_item("phi_bar", r.phi_bar)?;
    d.set_item("phi", r.phi.clone().into_pyarray(py))?;
    d.set_item("sigma2", r.sigma2.clone().into_pyarray(py))?;
    d.set_item("loglik", r.loglik)?;
    d.set_item("iterations", r.iterations)?;
    d.set_item("n_units", r.n_units)?;
    d.set_item("k", r.k)?;
    Ok(d)
}

/// First-generation panel unit-root tests: Levin-Lin-Chu, Im-Pesaran-Shin,
/// and the Fisher-type (Maddala-Wu / Choi) p-value combinations.
///
/// `data` is a balanced `N x T` array (each ROW a unit) or a list of 1-D
/// per-unit series (unbalanced is fine for "ips"/"fisher"; "llc" needs a
/// common length). `test` is "ips" (default), "llc", or "fisher". `lags` is
/// None (per-unit auto AIC), an int (fixed common lag), or "aic"/"bic"/
/// "t-stat". `regression` is "c" (default), "ct", or "n" ("n" is invalid for
/// "ips"). `lrv_kernel`/`lrv_bandwidth` configure the LLC long-run variance.
/// Conventions match plm::purtest (validated to plm, and for "fisher" to statsmodels).
#[pyfunction]
#[pyo3(signature = (data, test = "ips", lags = None, regression = "c", max_lags = None, lrv_kernel = "bartlett", lrv_bandwidth = None))]
#[allow(clippy::too_many_arguments)]
fn panel_unit_root<'py>(
    py: Python<'py>,
    data: Bound<'py, pyo3::PyAny>,
    test: &str,
    lags: Option<Bound<'py, pyo3::PyAny>>,
    regression: &str,
    max_lags: Option<usize>,
    lrv_kernel: &str,
    lrv_bandwidth: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_diag::AdfLagSelection as L;
    use tsecon_panelroot::{panel_unit_root as run, PanelRootDetail, PanelRootOpts, PanelRootTest};

    // Accept an N x T array (rows = units) or a list of 1-D per-unit series.
    let units: Vec<Vec<f64>> = if let Ok(arr) = data.extract::<numpy::PyReadonlyArray2<'py, f64>>()
    {
        let a = arr.as_array();
        (0..a.nrows()).map(|i| a.row(i).to_vec()).collect()
    } else {
        let list = data
            .extract::<Vec<PyReadonlyArray1<'py, f64>>>()
            .map_err(|_| {
                PyValueError::new_err(
                    "data must be a 2-D array (rows = units) or a list of 1-D per-unit arrays",
                )
            })?;
        list.iter().map(vec1).collect()
    };

    let test_enum = match test {
        "ips" => PanelRootTest::Ips,
        "llc" => PanelRootTest::Llc,
        "fisher" => PanelRootTest::Fisher,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown test {other:?}; expected \"ips\", \"llc\", or \"fisher\""
            )))
        }
    };
    let reg = adf_regression(regression)?;

    let sel = match lags {
        None => L::Aic(max_lags),
        Some(obj) => {
            if let Ok(k) = obj.extract::<usize>() {
                L::Fixed(k)
            } else if let Ok(s) = obj.extract::<String>() {
                match s.as_str() {
                    "aic" | "AIC" => L::Aic(max_lags),
                    "bic" | "BIC" => L::Bic(max_lags),
                    "t-stat" => L::TStat(max_lags),
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown lags {other:?}; expected None, an int, or \"aic\"/\"bic\"/\"t-stat\""
                        )))
                    }
                }
            } else {
                return Err(PyValueError::new_err(
                    "lags must be None, an int, or one of \"aic\"/\"bic\"/\"t-stat\"",
                ));
            }
        }
    };

    let opts = PanelRootOpts {
        lrv_kernel: hac_kernel(lrv_kernel)?,
        lrv_bandwidth,
    };
    let r = run(&units, test_enum, reg, sel, &opts).map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("test", test)?;
    d.set_item("statistic", r.statistic)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("per_unit_tstat", r.per_unit_tstat.clone().into_pyarray(py))?;
    d.set_item(
        "per_unit_pvalue",
        r.per_unit_pvalue.clone().into_pyarray(py),
    )?;
    d.set_item(
        "per_unit_lags",
        r.per_unit_lags
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "per_unit_nobs",
        r.per_unit_nobs
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>()
            .into_pyarray(py),
    )?;
    d.set_item("n_units", r.n_units)?;
    d.set_item("regression", regression)?;
    match r.detail {
        PanelRootDetail::Ips { t_bar } => {
            d.set_item("t_bar", t_bar)?;
        }
        PanelRootDetail::Llc {
            delta_hat,
            t_delta,
            s_n,
            t_bar_periods,
        } => {
            d.set_item("delta_hat", delta_hat)?;
            d.set_item("t_delta", t_delta)?;
            d.set_item("s_n", s_n)?;
            d.set_item("t_bar_periods", t_bar_periods)?;
        }
        PanelRootDetail::Fisher {
            choi_z,
            choi_z_pvalue,
        } => {
            d.set_item("maddala_wu", r.statistic)?;
            d.set_item("choi_z", choi_z)?;
            d.set_item("choi_z_pvalue", choi_z_pvalue)?;
        }
    }
    Ok(d)
}

/// News / update decomposition of a DFM nowcast revision (Bańbura-Modugno
/// 2014).
///
/// Fits the two-step DFM on the balanced leading block of `old_vintage`,
/// then decomposes the revision of the `target_series` nowcast at
/// `target_period` between the `old_vintage` and `new_vintage` data panels
/// (same shape; the new one reveals additional ragged-edge observations)
/// into a per-newly-observed-datapoint breakdown. `target_period` defaults
/// to the last row. Returns `old_nowcast`, `new_nowcast`, `total_revision`,
/// and `contributions` — a list of `{series, period, actual, forecast,
/// news, weight, contribution}` where `contribution = weight * news` and
/// the contributions sum exactly to the total revision.
#[pyfunction]
#[pyo3(signature = (old_vintage, new_vintage, target_series = 0, target_period = None,
                    n_factors = 1, factor_order = 2))]
#[allow(clippy::too_many_arguments)]
fn dfm_news<'py>(
    py: Python<'py>,
    old_vintage: numpy::PyReadonlyArray2<'py, f64>,
    new_vintage: numpy::PyReadonlyArray2<'py, f64>,
    target_series: usize,
    target_period: Option<usize>,
    n_factors: usize,
    factor_order: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    let old_arr = old_vintage.as_array();
    let (t, n) = (old_arr.nrows(), old_arr.ncols());
    // Fit on the leading balanced block of the old vintage.
    let mut first_ragged = t;
    for i in 0..t {
        if (0..n).any(|j| !old_arr[(i, j)].is_finite()) {
            first_ragged = i;
            break;
        }
    }
    let train = Mat::from_fn(first_ragged, n, |r, j| old_arr[(r, j)]);
    let old_m = to_faer(&old_vintage);
    let new_m = to_faer(&new_vintage);
    let nc = tsecon_nowcast::Nowcaster::fit_two_step(train.as_ref(), n_factors, factor_order)
        .map_err(to_py)?;
    let period = target_period.unwrap_or(t.saturating_sub(1));
    let nd = nc
        .news_decomposition(old_m.as_ref(), new_m.as_ref(), target_series, period)
        .map_err(to_py)?;
    let contribs = nd
        .contributions
        .iter()
        .map(|c| {
            let row = PyDict::new(py);
            row.set_item("series", c.series)?;
            row.set_item("period", c.period)?;
            row.set_item("actual", c.actual)?;
            row.set_item("forecast", c.forecast)?;
            row.set_item("news", c.news)?;
            row.set_item("weight", c.weight)?;
            row.set_item("contribution", c.contribution)?;
            Ok(row)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let d = PyDict::new(py);
    d.set_item("target_series", nd.target_series)?;
    d.set_item("target_period", nd.target_period)?;
    d.set_item("old_nowcast", nd.old_nowcast)?;
    d.set_item("new_nowcast", nd.new_nowcast)?;
    d.set_item("total_revision", nd.total_revision)?;
    d.set_item("contributions", contribs)?;
    Ok(d)
}

/// Predictive regression of `r_{t+1}` on a persistent predictor `x_t`, with
/// inference robust to the predictor's persistence.
///
/// Returns three views of `r_{t+1} = alpha + beta*x_t + u_{t+1}` where `x` is
/// near-integrated and its innovation correlates with `u` (the Stambaugh
/// setting, which biases OLS and makes the naive t-test over-reject):
/// `ols` (plain OLS `beta`/`se`/`tstat`), `stambaugh` (the Stambaugh 1999
/// bias-corrected `beta_corrected` with the `bias_term` and estimated
/// `rho`), and `ivx` (the Kostakis-Magdalinos-Stamatogiannis 2015 estimator
/// `beta_ivx` and its Wald test `wald`/`pvalue`, asymptotically chi-square
/// uniformly over the persistence of `x`). `cz`/`alpha` tune the IVX
/// instrument (defaults -1, 0.95).
#[pyfunction]
#[pyo3(signature = (r, x, cz = -1.0, alpha = 0.95))]
fn predictive_regression<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    cz: f64,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (rs, xs) = (vec1(&r), vec1(&x));
    let rs = rs.as_slice();
    let xs = xs.as_slice();
    let ols = tsecon_predreg::ols_predictive(rs, xs).map_err(to_py)?;
    let stb = tsecon_predreg::stambaugh(rs, xs).map_err(to_py)?;
    let cfg = tsecon_predreg::IvxConfig { cz, alpha };
    let iv = tsecon_predreg::ivx(rs, xs, cfg).map_err(to_py)?;

    let ols_d = PyDict::new(py);
    ols_d.set_item("alpha", ols.alpha)?;
    ols_d.set_item("beta", ols.beta)?;
    ols_d.set_item("se", ols.se)?;
    ols_d.set_item("tstat", ols.tstat)?;

    let stb_d = PyDict::new(py);
    stb_d.set_item("beta_ols", stb.beta_ols)?;
    stb_d.set_item("beta_corrected", stb.beta_corrected)?;
    stb_d.set_item("bias_term", stb.bias_term)?;
    stb_d.set_item("rho_ols", stb.rho_ols)?;
    stb_d.set_item("se", stb.se)?;

    let iv_d = PyDict::new(py);
    iv_d.set_item("beta_ivx", iv.beta_ivx)?;
    iv_d.set_item("wald", iv.wald)?;
    iv_d.set_item("pvalue", iv.pvalue)?;
    iv_d.set_item("rz", iv.rz)?;

    let d = PyDict::new(py);
    d.set_item("ols", ols_d)?;
    d.set_item("stambaugh", stb_d)?;
    d.set_item("ivx", iv_d)?;
    d.set_item("nobs", iv.nobs)?;
    Ok(d)
}

/// Joint IVX predictability test for several persistent predictors at once
/// (Kostakis-Magdalinos-Stamatogiannis 2015).
///
/// `xs` is a `T x k` matrix of persistent predictors; tests `H0: beta = 0`
/// jointly with a chi-square(`k`) Wald statistic whose validity is uniform
/// over the predictors' persistence. Returns the IVX slope vector
/// `beta_ivx`, the joint `wald`/`pvalue`, the instrument decay `rz`, and
/// shape info.
#[pyfunction]
#[pyo3(signature = (r, xs, cz = -1.0, alpha = 0.95))]
fn ivx_test<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    xs: numpy::PyReadonlyArray2<'py, f64>,
    cz: f64,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let a = xs.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let cfg = tsecon_predreg::IvxConfig { cz, alpha };
    let iv = tsecon_predreg::ivx_multi(&vec1(&r), &cols, cfg).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("beta_ivx", iv.beta_ivx.clone().into_pyarray(py))?;
    d.set_item("wald", iv.wald)?;
    d.set_item("pvalue", iv.pvalue)?;
    d.set_item("rz", iv.rz)?;
    d.set_item("nregressors", iv.nregressors)?;
    d.set_item("nobs", iv.nobs)?;
    Ok(d)
}

/// Recession-probability model: probit or logit of a binary recession
/// indicator on leading variables (e.g. the term spread).
///
/// `y` is the 0/1 recession indicator; `x` is a `T x k` design (include a
/// constant column). `link` is `"probit"` or `"logit"`. With
/// `dynamic=True` the Kauppi-Saikkonen (2008) dynamic probit is fit instead
/// (an autoregressive index; probit only, and `x` must NOT contain a
/// constant — the model supplies its own). Returns `params`, `bse`,
/// `zstats`, the fitted `probabilities`, `loglik`, McFadden `pseudo_r2`,
/// and `converged` (plus `rho` for the dynamic model).
#[pyfunction]
#[pyo3(signature = (y, x, link = "probit", dynamic = false))]
fn recession_probit<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    link: &str,
    dynamic: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let a = x.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let ys = vec1(&y);
    let ys = ys.as_slice();
    let d = PyDict::new(py);
    if dynamic {
        let f = tsecon_recession::fit_dynamic_probit(ys, &cols).map_err(to_py)?;
        d.set_item("params", f.params.clone().into_pyarray(py))?;
        d.set_item("bse", f.bse.clone().into_pyarray(py))?;
        d.set_item("zstats", f.zstats.clone().into_pyarray(py))?;
        d.set_item("w", f.w)?;
        d.set_item("beta", f.beta.clone().into_pyarray(py))?;
        d.set_item("rho", f.rho)?;
        d.set_item("probabilities", f.fitted.clone().into_pyarray(py))?;
        d.set_item("loglik", f.loglik)?;
        d.set_item("pseudo_r2", f.pseudo_r2)?;
        d.set_item("converged", f.converged)?;
    } else {
        let lk = match link {
            "probit" => tsecon_recession::Link::Probit,
            "logit" => tsecon_recession::Link::Logit,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown link {other:?}; expected \"probit\" or \"logit\""
                )))
            }
        };
        let f = tsecon_recession::fit_static(ys, &cols, lk).map_err(to_py)?;
        d.set_item("params", f.params.clone().into_pyarray(py))?;
        d.set_item("bse", f.bse.clone().into_pyarray(py))?;
        d.set_item("zstats", f.zstats.clone().into_pyarray(py))?;
        d.set_item("probabilities", f.fitted.clone().into_pyarray(py))?;
        d.set_item("loglik", f.loglik)?;
        d.set_item("pseudo_r2", f.pseudo_r2)?;
        d.set_item("converged", f.converged)?;
    }
    Ok(d)
}

/// Coibion-Gorodnichenko (2015) information-rigidity regression.
///
/// Regresses the mean forecast `errors` on the mean forecast `revisions` by
/// OLS with Newey-West HAC standard errors (`maxlags=None` uses the
/// automatic rule). The `slope` measures information rigidity — 0 under
/// full-information rational expectations, positive under sticky/noisy
/// information — and maps to `implied_rigidity = slope/(1+slope)`. Returns
/// `intercept`/`slope` with HAC `se`/`t`/`p`, `r_squared`, `implied_rigidity`,
/// and `maxlags`/`nobs`.
#[pyfunction]
#[pyo3(signature = (errors, revisions, maxlags = None, use_correction = true))]
fn cg_regression<'py>(
    py: Python<'py>,
    errors: PyReadonlyArray1<'py, f64>,
    revisions: PyReadonlyArray1<'py, f64>,
    maxlags: Option<usize>,
    use_correction: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let bw = match maxlags {
        Some(l) => tsecon_survey::HacBandwidth::Lags(l),
        None => tsecon_survey::HacBandwidth::Auto,
    };
    let r = tsecon_survey::cg_regression(&vec1(&errors), &vec1(&revisions), bw, use_correction)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("intercept", r.intercept)?;
    d.set_item("slope", r.slope)?;
    d.set_item("se_intercept", r.se_intercept)?;
    d.set_item("se_slope", r.se_slope)?;
    d.set_item("t_slope", r.t_slope)?;
    d.set_item("p_slope", r.p_slope)?;
    d.set_item("r_squared", r.r_squared)?;
    d.set_item("implied_rigidity", r.implied_rigidity)?;
    d.set_item("maxlags", r.maxlags)?;
    d.set_item("nobs", r.nobs)?;
    Ok(d)
}

/// Mincer-Zarnowitz forecast-efficiency (rationality) test.
///
/// Regresses the forecast `errors` on a constant and `regressors` (a
/// `T x k` matrix, e.g. the forecast itself or lagged information) with HAC
/// SEs, and jointly Wald-tests that all coefficients are zero — the null of
/// forecast efficiency. Returns `params`/`bse`/`tvalues`/`pvalues`,
/// `r_squared`, and the `wald`/`wald_df`/`wald_pvalue`.
#[pyfunction]
#[pyo3(signature = (errors, regressors, maxlags = None, use_correction = true))]
fn forecast_efficiency<'py>(
    py: Python<'py>,
    errors: PyReadonlyArray1<'py, f64>,
    regressors: numpy::PyReadonlyArray2<'py, f64>,
    maxlags: Option<usize>,
    use_correction: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let a = regressors.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let bw = match maxlags {
        Some(l) => tsecon_survey::HacBandwidth::Lags(l),
        None => tsecon_survey::HacBandwidth::Auto,
    };
    let r =
        tsecon_survey::efficiency_test(&vec1(&errors), &cols, bw, use_correction).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("params", r.params.clone().into_pyarray(py))?;
    d.set_item("bse", r.bse.clone().into_pyarray(py))?;
    d.set_item("tvalues", r.tvalues.clone().into_pyarray(py))?;
    d.set_item("pvalues", r.pvalues.clone().into_pyarray(py))?;
    d.set_item("r_squared", r.r_squared)?;
    d.set_item("wald", r.wald)?;
    d.set_item("wald_df", r.wald_df)?;
    d.set_item("wald_pvalue", r.wald_pvalue)?;
    Ok(d)
}

/// Fractional differencing `(1 - L)^d x` via the binomial expansion.
///
/// The long-memory generalization of integer differencing: `d = 1` is the
/// ordinary first difference, `0 < d < 0.5` a stationary long-memory filter.
#[pyfunction]
fn frac_diff<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    d: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out = tsecon_longmemory::frac_diff(&vec1(&x), d).map_err(to_py)?;
    Ok(out.into_pyarray(py))
}

/// Estimate the fractional-integration (long-memory) parameter `d`.
///
/// `method="gph"` is the Geweke-Porter-Hudak (1983) log-periodogram
/// regression; `method="local_whittle"` is the Robinson (1995) local Whittle
/// estimator. `m` is the number of low Fourier frequencies used (bandwidth);
/// `None` uses the standard `n^0.5` default. Returns the estimate `d` and its
/// asymptotic `se`.
#[pyfunction]
#[pyo3(signature = (x, m = None, method = "gph"))]
fn long_memory_d<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    m: Option<usize>,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let xs = vec1(&x);
    let xs = xs.as_slice();
    let bw = m.unwrap_or_else(|| tsecon_longmemory::default_bandwidth(xs.len()));
    let d = PyDict::new(py);
    match method {
        "gph" => {
            let r = tsecon_longmemory::gph(xs, bw).map_err(to_py)?;
            d.set_item("d", r.d)?;
            d.set_item("se", r.se)?;
            d.set_item("m", r.m)?;
        }
        "local_whittle" | "whittle" => {
            let r = tsecon_longmemory::local_whittle(xs, bw).map_err(to_py)?;
            d.set_item("d", r.d)?;
            d.set_item("se", r.se)?;
            d.set_item("m", r.m)?;
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}; expected \"gph\" or \"local_whittle\""
            )))
        }
    }
    Ok(d)
}

/// Forecast-disagreement measures from a panel of forecasters.
///
/// `panel` is a list of per-period cross-sections (one array of the
/// individual forecasts made for each period; the cross-sections may be
/// ragged). Returns, per period, the dispersion `std` (with `ddof`), the
/// quartiles `p25`/`p50`/`p75`, the inter-quartile range `iqr`, and the
/// forecaster `counts`.
#[pyfunction]
#[pyo3(signature = (panel, ddof = 1))]
fn forecast_disagreement<'py>(
    py: Python<'py>,
    panel: Vec<PyReadonlyArray1<'py, f64>>,
    ddof: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let cross: Vec<Vec<f64>> = panel.iter().map(vec1).collect();
    let dg = tsecon_survey::disagreement(&cross, ddof).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("std", dg.std.clone().into_pyarray(py))?;
    d.set_item("p25", dg.p25.clone().into_pyarray(py))?;
    d.set_item("p50", dg.p50.clone().into_pyarray(py))?;
    d.set_item("p75", dg.p75.clone().into_pyarray(py))?;
    d.set_item("iqr", dg.iqr.clone().into_pyarray(py))?;
    d.set_item("counts", dg.counts.clone())?;
    Ok(d)
}

/// Fractional integration `(1 - L)^{-d} x` — the inverse of `frac_diff`.
#[pyfunction]
fn frac_integrate<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    d: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let out = tsecon_longmemory::frac_integrate(&vec1(&x), d).map_err(to_py)?;
    Ok(out.into_pyarray(py))
}

/// Heteroskedasticity test on an OLS regression of `y` on `x` (a `T x k`
/// design; include a constant column). `test`: `"white"` (White 1980, the
/// squares-and-cross-products auxiliary) or `"breusch_pagan"` (Koenker
/// studentized). Returns the LM `statistic`/`df`/`pvalue` and the F-form.
#[pyfunction]
#[pyo3(signature = (y, x, test = "white"))]
fn heteroskedasticity_test<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    test: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let a = x.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let ys = vec1(&y);
    let r = match test {
        "white" => tsecon_spectest::white_test(&ys, &cols).map_err(to_py)?,
        "breusch_pagan" | "bp" => tsecon_spectest::breusch_pagan_test(&ys, &cols).map_err(to_py)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown test {other:?}; expected \"white\" or \"breusch_pagan\""
            )))
        }
    };
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("df", r.df)?;
    d.set_item("pvalue", r.pvalue)?;
    d.set_item("fstat", r.fstat)?;
    d.set_item("f_pvalue", r.f_pvalue)?;
    Ok(d)
}

/// Ramsey RESET functional-form test: F-test of powers of the fitted values
/// (`yhat^2 .. yhat^max_power`) added to the OLS of `y` on `x` (`T x k`).
#[pyfunction]
#[pyo3(signature = (y, x, max_power = 3))]
fn reset_test<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    max_power: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let a = x.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let r = tsecon_spectest::reset_test(&vec1(&y), &cols, max_power).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("fstat", r.fstat)?;
    d.set_item("df_num", r.df_num)?;
    d.set_item("df_den", r.df_den)?;
    d.set_item("pvalue", r.pvalue)?;
    Ok(d)
}

/// Chow structural-break test at a known 0-indexed `split`: F-test that the
/// regression of `y` on `x` (`T x k`) has the same coefficients before and
/// after `split`. Returns the F stat, dfs, p-value, and the sub-sample SSRs.
#[pyfunction]
fn chow_test<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    split: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let a = x.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let r = tsecon_spectest::chow_test(&vec1(&y), &cols, split).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("fstat", r.fstat)?;
    d.set_item("df_num", r.df_num)?;
    d.set_item("df_den", r.df_den)?;
    d.set_item("pvalue", r.pvalue)?;
    d.set_item("ssr_pooled", r.ssr_pooled)?;
    d.set_item("ssr1", r.ssr1)?;
    d.set_item("ssr2", r.ssr2)?;
    Ok(d)
}

/// CUSUM parameter-stability test (Brown-Durbin-Evans 1975) on the recursive
/// residuals of the OLS of `y` on `x` (`T x k`). Returns the standardized
/// `path` and the 5% significance `bound_upper`/`bound_lower` lines: the
/// coefficients are unstable if the path crosses a bound.
#[pyfunction]
fn cusum_test<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let a = x.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let r = tsecon_spectest::cusum_test(&vec1(&y), &cols).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("path", r.path.clone().into_pyarray(py))?;
    d.set_item("bound_upper", r.bound_upper.clone().into_pyarray(py))?;
    d.set_item("bound_lower", r.bound_lower.clone().into_pyarray(py))?;
    d.set_item("sigma", r.sigma)?;
    Ok(d)
}

/// Arbitrage-free Nelson-Siegel yield adjustment (Christensen-Diebold-
/// Rudebusch 2011): the deterministic term `-C(τ)/τ` that makes the
/// three-factor curve arbitrage-free, given the factor-volatility diagonal
/// `sigma` (three values) and the decay. Add it to a Nelson-Siegel fit.
#[pyfunction]
#[pyo3(signature = (maturities, sigma, decay = 0.0609))]
fn afns_adjustment<'py>(
    py: Python<'py>,
    maturities: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    decay: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sv = vec1(&sigma);
    if sv.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "sigma must have 3 elements (the factor-volatility diagonal); got {}",
            sv.len()
        )));
    }
    let sig = [sv[0], sv[1], sv[2]];
    let out = tsecon_termstructure::afns_yield_adjustment(&vec1(&maturities), decay, sig)
        .map_err(to_py)?;
    Ok(out.into_pyarray(py))
}

/// Solve a linear rational-expectations (DSGE-lite) model by Blanchard-Kahn
/// (1980): `A E_t[y_{t+1}] = B y_t + C z_{t+1}`, where `y` stacks the
/// `n_predetermined` backward-looking variables then the forward-looking
/// ones. Returns the decision rule `g` (jump = g·predetermined), the law of
/// motion `p`/`q` (predetermined_{t+1} = p·predetermined + q·z), the
/// `eigenvalue_moduli`, and the Blanchard-Kahn `verdict` (unique /
/// indeterminate / no stable solution).
#[pyfunction]
fn dsge_solve<'py>(
    py: Python<'py>,
    a: numpy::PyReadonlyArray2<'py, f64>,
    b: numpy::PyReadonlyArray2<'py, f64>,
    c: numpy::PyReadonlyArray2<'py, f64>,
    n_predetermined: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let rows = |m: &numpy::PyReadonlyArray2<'_, f64>| -> Vec<Vec<f64>> {
        let arr = m.as_array();
        (0..arr.nrows())
            .map(|i| (0..arr.ncols()).map(|j| arr[(i, j)]).collect())
            .collect()
    };
    let model = tsecon_dsge::LinearReModel::new(&rows(&a), &rows(&b), &rows(&c), n_predetermined)
        .map_err(to_py)?;
    let sol = tsecon_dsge::solve(&model).map_err(to_py)?;
    let mat = |m: &tsecon_var::tsecon_linalg::faer::Mat<f64>| -> Vec<Vec<f64>> {
        (0..m.nrows())
            .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
            .collect()
    };
    let d = PyDict::new(py);
    d.set_item("g", mat(sol.g()))?;
    d.set_item("p", mat(sol.p()))?;
    d.set_item("q", mat(sol.q()))?;
    let moduli: Vec<f64> = sol
        .eigenvalues()
        .iter()
        .map(|z| (z.re * z.re + z.im * z.im).sqrt())
        .collect();
    d.set_item("eigenvalue_moduli", moduli.into_pyarray(py))?;
    d.set_item("verdict", format!("{}", sol.verdict()))?;
    Ok(d)
}

// --------------------------------------------------------------------------
// quantile bindings (assembled from the quantile builder's draft)
// --------------------------------------------------------------------------
/// Linear quantile regression at one or many quantile levels.
///
/// IRLS check-loss estimation with Powell kernel-sandwich standard errors
/// (Epanechnikov kernel, Hall-Sheather bandwidth). Matches statsmodels
/// `QuantReg(y, x).fit(q=tau)` (all defaults): `params` at 1e-6 (the shared
/// IRLS stopping tolerance) and `bse`/`bandwidth`/`sparsity` at 1e-6
/// relative. Include the constant column in `x` yourself (statsmodels exog
/// convention). `taus` defaults to [0.05, 0.25, 0.5, 0.75, 0.95].
#[pyfunction]
#[pyo3(signature = (y, x, taus = None, se = "robust"))]
fn quantile_regression<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    taus: Option<Vec<f64>>,
    se: &str,
) -> PyResult<Bound<'py, PyDict>> {
    if se != "robust" {
        return Err(PyValueError::new_err(format!(
            "unknown se {se:?}; only \"robust\" (the Powell kernel sandwich, \
             statsmodels' default) is implemented"
        )));
    }
    let taus = taus.unwrap_or_else(|| vec![0.05, 0.25, 0.5, 0.75, 0.95]);
    let ys = vec1(&y);
    let xa = x.as_array();
    let cols: Vec<Vec<f64>> = (0..xa.ncols()).map(|j| xa.column(j).to_vec()).collect();
    let fits = tsecon_quantile::quantile_regression(&ys, &cols, &taus).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("taus", taus.into_pyarray(py))?;
    d.set_item(
        "params",
        fits.iter().map(|f| f.params.clone()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "bse",
        fits.iter().map(|f| f.bse.clone()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "tvalues",
        fits.iter().map(|f| f.tvalues.clone()).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "iterations",
        fits.iter().map(|f| f.iterations as u64).collect::<Vec<_>>(),
    )?;
    d.set_item("converged", fits.iter().all(|f| f.converged))?;
    d.set_item(
        "bandwidth",
        fits.iter()
            .map(|f| f.bandwidth)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "sparsity",
        fits.iter()
            .map(|f| f.sparsity)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

/// Quantile local projections: per-horizon check-loss IRFs of `y` to `shock`
/// at each tau, controlling for a constant and `n_lag_controls` lags of BOTH
/// `y` and `shock` (tsecon-lp design conventions; the impulse coefficient is
/// design column 0). Matches statsmodels `QuantReg` per (tau, horizon) on
/// the identical design at 1e-6; `se` is the Powell kernel sandwich.
/// `taus` defaults to [0.1, 0.5, 0.9]. Returns `irf[tau][h]`, `se[tau][h]`.
#[pyfunction]
#[pyo3(signature = (y, shock, taus = None, horizons = 12, n_lag_controls = 4))]
fn quantile_lp<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    shock: PyReadonlyArray1<'py, f64>,
    taus: Option<Vec<f64>>,
    horizons: usize,
    n_lag_controls: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let taus = taus.unwrap_or_else(|| vec![0.1, 0.5, 0.9]);
    let r = tsecon_quantile::quantile_lp(&vec1(&y), &vec1(&shock), &taus, horizons, n_lag_controls)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("taus", r.taus.into_pyarray(py))?;
    d.set_item(
        "horizons",
        r.horizons
            .iter()
            .map(|&h| h as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("irf", r.irf)?;
    d.set_item("se", r.se)?;
    Ok(d)
}

/// Growth-at-risk (Adrian-Boyarchenko-Giannone 2019 AER): conditional
/// quantiles of the `horizon`-ahead outcome on `[const, conditions, y_t]`
/// (canonically GDP growth on NFCI + own growth), fitted at EVERY
/// observation — `current` is the risk read at the latest one. `rearrange`
/// applies the Chernozhukov-Fernandez-Val-Galichon monotone rearrangement;
/// `crossing` reports whether the raw quantile paths crossed either way.
/// Matches statsmodels `QuantReg` per tau plus a numpy sort at 1e-6.
/// `taus` must be strictly increasing; defaults to
/// [0.05, 0.25, 0.5, 0.75, 0.95]. Requires `horizon >= 1`.
#[pyfunction]
#[pyo3(signature = (y, conditions, horizon = 1, taus = None, rearrange = true))]
fn growth_at_risk<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    conditions: numpy::PyReadonlyArray2<'py, f64>,
    horizon: usize,
    taus: Option<Vec<f64>>,
    rearrange: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let taus = taus.unwrap_or_else(|| vec![0.05, 0.25, 0.5, 0.75, 0.95]);
    let ca = conditions.as_array();
    let cond_cols: Vec<Vec<f64>> = (0..ca.ncols()).map(|j| ca.column(j).to_vec()).collect();
    let r = tsecon_quantile::growth_at_risk(&vec1(&y), &cond_cols, horizon, &taus, rearrange)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("taus", r.taus.into_pyarray(py))?;
    d.set_item("horizon", r.horizon as u64)?;
    d.set_item("params", r.params)?;
    d.set_item("bse", r.bse)?;
    d.set_item("fitted", r.fitted)?;
    d.set_item("fitted_raw", r.fitted_raw)?;
    d.set_item("crossing", r.crossing)?;
    d.set_item("current", r.current.into_pyarray(py))?;
    Ok(d)
}

// --------------------------------------------------------------------------
// funcshock bindings (assembled from the funcshock builder's draft)
// --------------------------------------------------------------------------
// ------------------------------------------------------------------
// tsecon-funcshock: functional shocks (Inoue-Rossi 2021)
// ------------------------------------------------------------------

/// A 2-D `f64` array as owned rows (Vec of row Vecs). Like `vec1`, accepts
/// non-contiguous input (transposed slices, strided views, pandas frames).
fn rows2(a: &numpy::PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    let v = a.as_array();
    (0..v.nrows())
        .map(|i| (0..v.ncols()).map(|j| v[(i, j)]).collect())
        .collect()
}

/// Per-horizon K*K row-major covariances -> (H+1) x K x K nested lists.
fn flp_covs_nested(covs: &[Vec<f64>], k: usize) -> Vec<Vec<Vec<f64>>> {
    covs.iter()
        .map(|c| (0..k).map(|i| c[i * k..(i + 1) * k].to_vec()).collect())
        .collect()
}

/// Functional PCA of a T x M panel of curve observations (e.g. daily
/// yield-curve changes on a maturity grid): demean, eigendecompose the M x M
/// covariance Xc'Xc/T, keep the leading `n_factors` eigenfunctions, scores,
/// eigenvalues, and explained-variance shares. Sign convention: each
/// eigenfunction's largest-|.| entry is positive.
///
/// Matches numpy.linalg.eigh of the same covariance at 1e-10.
#[pyfunction]
#[pyo3(signature = (curves, n_factors = 3))]
fn functional_pca<'py>(
    py: Python<'py>,
    curves: numpy::PyReadonlyArray2<'py, f64>,
    n_factors: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_funcshock::functional_pca(&rows2(&curves), n_factors).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("mean_curve", r.mean_curve.into_pyarray(py))?;
    d.set_item("eigenfunctions", r.eigenfunctions)?; // K rows, each length M
    d.set_item("scores", r.scores)?; // T rows, each length K
    d.set_item("eigenvalues", r.eigenvalues.into_pyarray(py))?;
    d.set_item("explained", r.explained.into_pyarray(py))?;
    d.set_item("total_variance", r.total_variance)?;
    Ok(d)
}

/// Functional local projection (Inoue-Rossi 2021): at each horizon h regress
/// y_{t+h} JOINTLY on all K scores + a constant + `n_lag_controls` lags of y,
/// with Newey-West Bartlett HAC standard errors (maxlags = h + n_lag_controls
/// unless `hac_maxlags` is given; statsmodels use_correction=True). `covs` is
/// the per-horizon JOINT K x K coefficient covariance — whole-curve scenarios
/// need its off-diagonals.
///
/// Matches statsmodels OLS(...).fit(cov_type="HAC") per horizon at 1e-8.
#[pyfunction]
#[pyo3(signature = (y, scores, horizons = 8, n_lag_controls = 2, hac_maxlags = None))]
fn flp<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    scores: numpy::PyReadonlyArray2<'py, f64>,
    horizons: usize,
    n_lag_controls: usize,
    hac_maxlags: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = tsecon_funcshock::flp(
        &vec1(&y),
        &rows2(&scores),
        horizons,
        n_lag_controls,
        hac_maxlags,
    )
    .map_err(to_py)?;
    let covs = flp_covs_nested(&r.covs, r.n_factors);
    let d = PyDict::new(py);
    d.set_item(
        "horizons",
        r.horizons
            .iter()
            .map(|h| *h as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("n_factors", r.n_factors)?;
    d.set_item("betas", r.betas)?; // (H+1) x K
    d.set_item("covs", covs)?; // (H+1) x K x K
    d.set_item("se", r.se)?; // (H+1) x K
    d.set_item(
        "nobs",
        r.nobs
            .iter()
            .map(|n| *n as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    Ok(d)
}

/// One-call functional-shock IRF: functional PCA of the curves, joint FLP of
/// y on the scores, then the response of y to the whole-curve scenario
/// `delta` (length M, same grid as the curves): weights w_k = <phi_k, delta>,
/// response_h = w'beta_h, se_h = sqrt(w' Cov_h w). This is the response of y
/// to the ENTIRE curve moving by delta — the functional deliverable.
///
/// Matches the numpy/statsmodels composition (eigh + OLS-HAC + closed form)
/// at 1e-8.
#[pyfunction]
#[pyo3(signature = (y, curves, delta, n_factors = 3, horizons = 8, n_lag_controls = 2, hac_maxlags = None))]
#[allow(clippy::too_many_arguments)] // py + 7 user args; the composite scenario call genuinely needs them
fn flp_scenario<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    curves: numpy::PyReadonlyArray2<'py, f64>,
    delta: PyReadonlyArray1<'py, f64>,
    n_factors: usize,
    horizons: usize,
    n_lag_controls: usize,
    hac_maxlags: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let fpca = tsecon_funcshock::functional_pca(&rows2(&curves), n_factors).map_err(to_py)?;
    let fit = tsecon_funcshock::flp(
        &vec1(&y),
        &fpca.scores,
        horizons,
        n_lag_controls,
        hac_maxlags,
    )
    .map_err(to_py)?;
    let irf = tsecon_funcshock::flp_scenario(&fpca, &fit, &vec1(&delta)).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item(
        "horizons",
        irf.horizons
            .iter()
            .map(|h| *h as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("weights", irf.weights.into_pyarray(py))?;
    d.set_item("response", irf.response.into_pyarray(py))?;
    d.set_item("se", irf.se.into_pyarray(py))?;
    d.set_item("betas", fit.betas)?; // per-horizon joint score coefficients
    d.set_item("explained", fpca.explained.into_pyarray(py))?;
    Ok(d)
}

/// FVAR whole-curve scenario (Inoue-Rossi 2021): fit a VAR to [scores, y]
/// (scores ordered FIRST, outcome last, constant included), set the
/// reduced-form score innovation to w = phi'delta and the outcome's own
/// structural shock to zero (recursive/Cholesky identification), and read the
/// response off the orthogonalized IRFs. IDENTIFICATION CAVEAT: the impact
/// response of y is the in-sample regression of its innovation on the score
/// innovations — credible under announcement-day timing, an assumption
/// otherwise. `responses[h]` lists the K score responses then the outcome;
/// at h=0 the score responses equal the weights exactly.
///
/// Matches statsmodels VAR(...).fit(lags, trend="c") + orth_ma_rep + scipy
/// triangular solve at 1e-8.
#[pyfunction]
#[pyo3(signature = (y, curves, delta, n_factors = 3, lags = 2, horizon = 10))]
fn fvar_scenario<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    curves: numpy::PyReadonlyArray2<'py, f64>,
    delta: PyReadonlyArray1<'py, f64>,
    n_factors: usize,
    lags: usize,
    horizon: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let fpca = tsecon_funcshock::functional_pca(&rows2(&curves), n_factors).map_err(to_py)?;
    let w =
        tsecon_funcshock::scenario_weights(&fpca.eigenfunctions, &vec1(&delta)).map_err(to_py)?;
    let r = tsecon_funcshock::fvar_scenario(&fpca.scores, &vec1(&y), &w, lags, horizon)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item(
        "horizons",
        r.horizons
            .iter()
            .map(|h| *h as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("weights", r.weights.into_pyarray(py))?;
    d.set_item("response_outcome", r.response_outcome.into_pyarray(py))?;
    d.set_item("responses", r.responses)?; // (H+1) x (K+1): scores first, outcome last
    d.set_item("implied_outcome_innovation", r.implied_outcome_innovation)?;
    Ok(d)
}

// --------------------------------------------------------------------------
// breaks bindings (assembled from the breaks builder's draft)
// --------------------------------------------------------------------------
// Compile-checked verbatim against pyo3 0.29 / numpy 0.29 and the real crate
// (scratch harness). Uses only the existing helpers `to_py` and `vec1`.

/// Bai-Perron multiple structural breaks: global partitions by dynamic
/// programming, number of breaks selected by sequential supF(l+1|l) tests
/// at 5% (published Bai-Perron critical values), per-regime OLS, and
/// Bai (1997) break-date confidence intervals.
///
/// `x` is the T x q design whose coefficients ALL switch at each break
/// (add your own constant column). `trim` must be one of 0.05, 0.10,
/// 0.15, 0.20, 0.25 (the published critical-value grid); q <= 10. Break
/// dates are 0-indexed last observations of each regime. Validated
/// against exact brute-force enumeration + numpy segment OLS at 1e-8
/// (dates exact); CIs assume homogeneous regressor moments and error
/// variance across regimes.
#[pyfunction]
#[pyo3(signature = (y, x, max_breaks = 5, trim = 0.15))]
fn bai_perron<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    max_breaks: usize,
    trim: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let a = x.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let r = tsecon_breaks::bai_perron(
        &vec1(&y),
        &cols,
        tsecon_breaks::BaiPerronConfig { max_breaks, trim },
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("n_breaks", r.n_breaks)?;
    d.set_item(
        "break_dates",
        r.break_dates
            .iter()
            .map(|v| *v as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("sup_f_seq", r.sup_f_seq.into_pyarray(py))?;
    d.set_item("sup_f_crit", r.sup_f_crit.into_pyarray(py))?;
    d.set_item("ssr_path", r.ssr_path.into_pyarray(py))?;
    d.set_item(
        "break_dates_by_m",
        r.break_dates_by_m
            .iter()
            .map(|dates| dates.iter().map(|v| *v as u64).collect())
            .collect::<Vec<Vec<u64>>>(),
    )?;
    d.set_item(
        "regime_starts",
        r.regimes
            .iter()
            .map(|s| s.start as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "regime_ends",
        r.regimes
            .iter()
            .map(|s| s.end as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "params",
        r.regimes
            .iter()
            .map(|s| s.params.clone())
            .collect::<Vec<Vec<f64>>>(),
    )?;
    d.set_item(
        "bse",
        r.regimes
            .iter()
            .map(|s| s.se.clone())
            .collect::<Vec<Vec<f64>>>(),
    )?;
    d.set_item(
        "regime_ssr",
        r.regimes
            .iter()
            .map(|s| s.ssr)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "ci_lower_90",
        r.ci.iter()
            .map(|c| c.lower90 as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "ci_upper_90",
        r.ci.iter()
            .map(|c| c.upper90 as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "ci_lower_95",
        r.ci.iter()
            .map(|c| c.lower95 as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "ci_upper_95",
        r.ci.iter()
            .map(|c| c.upper95 as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "ci_scale",
        r.ci.iter()
            .map(|c| c.scale)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("h", r.h)?;
    Ok(d)
}

/// Andrews (1993) sup-F (Quandt) test for a single structural break at an
/// unknown date, with Hansen (1997) approximate asymptotic p-value from
/// his published response surfaces.
///
/// `x` is the T x q design (add your own constant column, q <= 10);
/// candidate dates leave at least `h = ceil(trim * T)` observations per
/// regime. The statistic is the Wald-form sup of the Chow path (matches
/// R strucchange `Fstats`/`sctest` at 1e-8; p-value 1e-10 against the
/// transcribed surface). Returns the full `f_path` over `dates` plus the
/// argmax `break_date`.
#[pyfunction]
#[pyo3(signature = (y, x, trim = 0.15))]
fn sup_f_test<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    trim: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let a = x.as_array();
    let cols: Vec<Vec<f64>> = (0..a.ncols()).map(|j| a.column(j).to_vec()).collect();
    let r = tsecon_breaks::sup_f_test(&vec1(&y), &cols, trim).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("stat", r.stat)?;
    d.set_item("p_value", r.p_value)?;
    d.set_item("break_date", r.break_date)?;
    d.set_item(
        "dates",
        r.dates
            .iter()
            .map(|v| *v as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item("f_path", r.f_path.into_pyarray(py))?;
    d.set_item("h", r.h)?;
    Ok(d)
}

// --------------------------------------------------------------------------
// smoothlp bindings (assembled from the smoothlp builder's draft)
// --------------------------------------------------------------------------
/// Smooth local projections (Barnichon-Brownlees 2019): the IRF path is
/// estimated jointly across horizons as a B-spline in `h` with a ridge
/// penalty on the `penalty_order`-th difference of the basis coefficients
/// (default 2: shrink toward a straight line). Closed form
/// `theta = (X'X + lam*P)^{-1} X'y` over the stacked per-horizon design.
///
/// `lam`: a float fixes the smoothing parameter (`0.0` reproduces the
/// per-horizon `tsecon.lp(se="hac")` point estimates exactly with the
/// default basis); `"cv"` or `None` selects it by leave-h-block-out
/// cross-validation over `lambda_grid` (or a default log-spaced grid) with
/// `n_folds` contiguous folds and a dependence buffer of
/// `horizons + n_lag_controls` periods around each held-out block.
///
/// `se` is the delta method through the basis over a stacked Bartlett-HAC
/// sandwich (bandwidth `hac_maxlags`, default `horizons + n_lag_controls`).
/// It conditions on `lam` (even when cross-validated) and describes the
/// penalized estimator's own sampling variability — shrinkage bias is not
/// accounted for. `irf_raw`/`se_raw` are the unsmoothed per-horizon HAC LP
/// for comparison.
///
/// Matches fixtures/smoothlp.json: basis vs scipy BSpline.design_matrix at
/// 1e-10, theta/irf/se and CV scores vs NumPy normal equations at ~1e-8
/// relative, lambda=0 IRF vs statsmodels per-horizon OLS at 1e-8.
#[pyfunction]
#[pyo3(signature = (y, shock, horizons = 12, n_lag_controls = 4, lam = None, degree = 3, n_basis = None, penalty_order = 2, lambda_grid = None, n_folds = 5, hac_maxlags = None))]
#[allow(clippy::too_many_arguments)]
fn smooth_lp<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    shock: PyReadonlyArray1<'py, f64>,
    horizons: usize,
    n_lag_controls: usize,
    lam: Option<&Bound<'py, PyAny>>,
    degree: usize,
    n_basis: Option<usize>,
    penalty_order: usize,
    lambda_grid: Option<Vec<f64>>,
    n_folds: usize,
    hac_maxlags: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut spec = tsecon_lp::SmoothLpSpec::new(horizons, n_lag_controls)
        .with_degree(degree)
        .with_penalty_order(penalty_order);
    if let Some(k) = n_basis {
        spec = spec.with_n_basis(k);
    }
    if let Some(ml) = hac_maxlags {
        spec = spec.with_hac_maxlags(ml);
    }
    // lam: None / "cv" -> cross-validation; a number -> fixed lambda.
    spec = match lam {
        None => spec.with_cv(lambda_grid, n_folds),
        Some(obj) if obj.is_none() => spec.with_cv(lambda_grid, n_folds),
        Some(obj) => {
            if let Ok(s) = obj.extract::<String>() {
                if s == "cv" {
                    spec.with_cv(lambda_grid, n_folds)
                } else {
                    return Err(PyValueError::new_err(format!(
                        "unknown lam {s:?}; expected a non-negative float, \"cv\", or None"
                    )));
                }
            } else if let Ok(v) = obj.extract::<f64>() {
                if lambda_grid.is_some() {
                    return Err(PyValueError::new_err(
                        "lambda_grid was given together with a fixed lam; pass lam=\"cv\" to \
                         cross-validate over the grid, or drop lambda_grid to use the fixed value",
                    ));
                }
                spec.with_lambda(v)
            } else {
                return Err(PyValueError::new_err(
                    "lam must be a non-negative float, \"cv\", or None",
                ));
            }
        }
    };
    let r = tsecon_lp::smooth_lp(&vec1(&y), &vec1(&shock), &spec).map_err(to_py)?;
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
    d.set_item("lambda_used", r.lambda)?;
    d.set_item("cv_grid", r.cv_grid.into_pyarray(py))?;
    d.set_item("cv_scores", r.cv_scores.into_pyarray(py))?;
    d.set_item("theta", r.theta.into_pyarray(py))?;
    d.set_item("irf_raw", r.irf_raw.into_pyarray(py))?;
    d.set_item("se_raw", r.se_raw.into_pyarray(py))?;
    Ok(d)
}

/// Structural forecast-error variance decomposition for an arbitrary structural
/// impact matrix `A0` — the gap `var_fevd` (recursive-Cholesky only) leaves.
///
/// `impact` is an optional (n x n) structural impact matrix A0 (columns =
/// one-standard-deviation structural shocks, A0 A0' = Sigma; from any
/// identification — sign, zero, proxy, max-share, long-run). If None, A0 is the
/// lower Cholesky of the innovation covariance and the result equals `var_fevd`
/// exactly. `sigma` ('dfadj'|'mle') selects the default Cholesky's df scaling;
/// the FEVD shares are INVARIANT to it (numerator and denominator scale
/// together) — it only rescales the reported `impact`.
///
/// Returns `fevd` [horizon+1][variable][shock] (each row sums to 1) and the
/// `impact` [n][n] A0 used.
#[pyfunction]
#[pyo3(signature = (data, lags = 2, horizon = 10, trend = "c", impact = None, sigma = "dfadj"))]
fn structural_fevd<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    trend: &str,
    impact: Option<numpy::PyReadonlyArray2<'py, f64>>,
    sigma: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_bayes::cholesky_irf;
    use tsecon_var::tsecon_linalg::faer::Mat;

    let n_vars = data.as_array().ncols();
    let r = var_results(&data, lags, trend)?;

    let (a0, fevd) = match impact {
        Some(arr) => {
            let a = arr.as_array();
            if a.nrows() != n_vars || a.ncols() != n_vars {
                return Err(PyValueError::new_err(format!(
                    "impact must be {n_vars}x{n_vars}, got {}x{}",
                    a.nrows(),
                    a.ncols()
                )));
            }
            let a0 = Mat::from_fn(n_vars, n_vars, |i, j| a[(i, j)]);
            let fevd = tsecon_ident::structural_fevd::structural_fevd(
                r.params.as_ref(),
                a0.as_ref(),
                lags,
                horizon,
            )
            .map_err(to_py)?;
            (a0, fevd)
        }
        None => {
            // Default recursive impact A0 = chol_lower(Sigma). cholesky_irf returns
            // Theta_0 = A0, so we lift A0 straight from theta[0] (no extra factor).
            // Shares are scale-invariant; `sigma` only rescales the reported A0.
            let sig = match sigma {
                "dfadj" => r.sigma_u.clone(),
                "mle" => r.sigma_u_mle.clone(),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown sigma {other:?}; expected \"dfadj\" or \"mle\""
                    )))
                }
            };
            let theta =
                cholesky_irf(r.params.as_ref(), sig.as_ref(), lags, horizon).map_err(to_py)?;
            let a0 = theta[0].clone();
            let fevd =
                tsecon_ident::structural_fevd::structural_fevd_from_theta(&theta).map_err(to_py)?;
            (a0, fevd)
        }
    };

    let out: Vec<Vec<Vec<f64>>> = fevd.iter().map(mat_to_vec2).collect();
    let d = PyDict::new(py);
    d.set_item("fevd", out)?;
    d.set_item("impact", mat_to_vec2(&a0))?;
    Ok(d)
}

/// Parse a list of narrative-restriction dicts into a NarrativeRestrictionSet.
/// period/start/end are 0-based EFFECTIVE-sample indices (= data_row - lags).
fn parse_narrative_restrictions(
    restrictions: &[Bound<'_, PyDict>],
    n_vars: usize,
    t_eff: usize,
) -> PyResult<tsecon_ident::NarrativeRestrictionSet> {
    use tsecon_ident::{ContributionRule, NarrativeRestriction, Sign};
    let parse_sign = |v: &Bound<'_, pyo3::PyAny>| -> PyResult<Sign> {
        match v.extract::<String>()?.as_str() {
            "+" | "positive" => Ok(Sign::Positive),
            "-" | "negative" => Ok(Sign::Negative),
            other => Err(PyValueError::new_err(format!(
                "unknown sign {other:?}; expected \"+\" or \"-\""
            ))),
        }
    };
    fn get<'a>(d: &Bound<'a, PyDict>, k: &str) -> PyResult<Bound<'a, pyo3::PyAny>> {
        d.get_item(k)?.ok_or_else(|| {
            PyValueError::new_err(format!("narrative restriction missing key {k:?}"))
        })
    }
    let mut items = Vec::with_capacity(restrictions.len());
    for d in restrictions {
        match get(d, "type")?.extract::<String>()?.as_str() {
            "shock_sign" => items.push(NarrativeRestriction::ShockSign {
                shock: get(d, "shock")?.extract()?,
                period: get(d, "period")?.extract()?,
                sign: parse_sign(&get(d, "sign")?)?,
            }),
            "contribution" => {
                let rule = match get(d, "rule")?.extract::<String>()?.as_str() {
                    "most" => ContributionRule::Most,
                    "least" => ContributionRule::Least,
                    o => {
                        return Err(PyValueError::new_err(format!(
                            "unknown rule {o:?}; expected \"most\"|\"least\""
                        )))
                    }
                };
                let strong = d
                    .get_item("strong")?
                    .map(|v| v.extract::<bool>())
                    .transpose()?
                    .unwrap_or(false);
                items.push(NarrativeRestriction::Contribution {
                    variable: get(d, "variable")?.extract()?,
                    shock: get(d, "shock")?.extract()?,
                    start: get(d, "start")?.extract()?,
                    end: get(d, "end")?.extract()?,
                    rule,
                    strong,
                });
            }
            "contribution_sign" => items.push(NarrativeRestriction::ContributionSign {
                variable: get(d, "variable")?.extract()?,
                shock: get(d, "shock")?.extract()?,
                start: get(d, "start")?.extract()?,
                end: get(d, "end")?.extract()?,
                sign: parse_sign(&get(d, "sign")?)?,
            }),
            o => {
                return Err(PyValueError::new_err(format!(
                    "unknown narrative restriction type {o:?}"
                )))
            }
        }
    }
    tsecon_ident::NarrativeRestrictionSet::new(items, n_vars, t_eff).map_err(to_py)
}

/// Historical decomposition (Kilian & Lütkepohl 2017, ch.4): per-(time, variable,
/// shock) structural-shock contributions.
#[pyfunction]
#[pyo3(signature = (data, restrictions = vec![], lags = 2, horizon = None, identification = "cholesky",
                    n_draws = 500, max_tries = 400, seed = 0, lambda1 = 0.2,
                    narrative_restrictions = None, n_weight_draws = 200))]
#[allow(clippy::too_many_arguments)]
fn historical_decomposition<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    restrictions: Vec<(usize, usize, usize, String)>,
    lags: usize,
    horizon: Option<usize>,
    identification: &str,
    n_draws: usize,
    max_tries: usize,
    seed: u64,
    lambda1: f64,
    narrative_restrictions: Option<Vec<Bound<'py, PyDict>>>,
    n_weight_draws: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    let a = data.as_array();
    let m = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let n = a.ncols();
    if a.nrows() <= lags {
        return Err(PyValueError::new_err("need more than `lags` observations"));
    }
    let t_eff = a.nrows() - lags;
    let d = PyDict::new(py);
    d.set_item("times", (0..t_eff).collect::<Vec<usize>>())?;

    match identification {
        "cholesky" => {
            let vr = var_results(&data, lags, "c")?;
            let sigma = vr.sigma_u_mle.clone();
            let eye = Mat::<f64>::identity(n, n);
            let h = horizon.unwrap_or(t_eff - 1);
            let hd = tsecon_ident::decompose(
                m.as_ref(),
                vr.params.as_ref(),
                sigma.as_ref(),
                eye.as_ref(),
                lags,
                h,
            )
            .map_err(to_py)?;
            let ref_to_vec2 =
                |r: tsecon_var::tsecon_linalg::faer::MatRef<'_, f64>| -> Vec<Vec<f64>> {
                    (0..r.nrows())
                        .map(|i| (0..r.ncols()).map(|j| r[(i, j)]).collect())
                        .collect()
                };
            let hd_tensor: Vec<Vec<Vec<f64>>> = hd
                .hd()
                .iter()
                .map(|mm| {
                    (0..n)
                        .map(|i| (0..n).map(|j| mm[(i, j)]).collect())
                        .collect()
                })
                .collect();
            d.set_item("baseline", ref_to_vec2(hd.baseline()))?;
            d.set_item("hd", hd_tensor)?;
            d.set_item("shocks", ref_to_vec2(hd.shocks()))?;
        }
        "sign" => {
            use tsecon_ident::{NarrativeSampler, Sign, SignRestriction, SignRestrictionSet};
            let prior =
                tsecon_bayes::MinnesotaNiwPrior::new(m.as_ref(), lags, 100.0, lambda1, 1.0, 0.0)
                    .map_err(to_py)?;
            let posterior = prior.posterior(m.as_ref()).map_err(to_py)?;
            let sign_h = horizon.unwrap_or(12);
            let signs = if restrictions.is_empty() {
                None
            } else {
                let mut rs = Vec::new();
                for (v, s, hh, sign) in restrictions {
                    let sg = match sign.as_str() {
                        "+" | "positive" => Sign::Positive,
                        "-" | "negative" => Sign::Negative,
                        o => return Err(PyValueError::new_err(format!("unknown sign {o:?}"))),
                    };
                    rs.push(SignRestriction::at(v, s, hh, sg));
                }
                Some(SignRestrictionSet::new(rs, n, sign_h).map_err(to_py)?)
            };
            let narrative = match &narrative_restrictions {
                Some(l) if !l.is_empty() => Some(parse_narrative_restrictions(l, n, t_eff)?),
                _ => None,
            };
            let result = NarrativeSampler::new(sign_h, n_draws, max_tries, n_weight_draws)
                .map_err(to_py)?
                .run(
                    &posterior,
                    m.as_ref(),
                    signs.as_ref(),
                    narrative.as_ref(),
                    seed,
                )
                .map_err(to_py)?;
            let hd = result.hd_summary().map_err(to_py)?;
            let mut q = vec![vec![vec![Vec::<f64>::new(); n]; n]; t_eff];
            let (mut lo, mut hi) = (
                vec![vec![vec![0.0; n]; n]; t_eff],
                vec![vec![vec![0.0; n]; n]; t_eff],
            );
            for t in 0..t_eff {
                for i in 0..n {
                    for j in 0..n {
                        let bp = hd.point(i, j, t).map_err(to_py)?;
                        q[t][i][j] = bp.quantiles.clone();
                        lo[t][i][j] = bp.min;
                        hi[t][i][j] = bp.max;
                    }
                }
            }
            let bl = hd.baseline();
            d.set_item("probs", hd.probs().to_vec())?;
            d.set_item(
                "baseline",
                (0..bl.nrows())
                    .map(|i| (0..bl.ncols()).map(|j| bl[(i, j)]).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )?;
            d.set_item("hd_quantiles", q)?;
            d.set_item("hd_set_min", lo)?;
            d.set_item("hd_set_max", hi)?;
            d.set_item("weights", result.weights().to_vec())?;
            let dg = result.diagnostics();
            let dd = PyDict::new(py);
            dd.set_item("accepted", dg.accepted)?;
            dd.set_item("acceptance_rate", dg.acceptance_rate)?;
            dd.set_item("ess", dg.ess)?;
            dd.set_item("narrative_acceptance_rate", dg.narrative_acceptance_rate)?;
            dd.set_item("min_ptilde", dg.min_ptilde)?;
            d.set_item("diagnostics", dd)?;
        }
        o => {
            return Err(PyValueError::new_err(format!(
                "unknown identification {o:?}; expected \"cholesky\"|\"sign\""
            )))
        }
    }
    Ok(d)
}

/// Narrative sign-restricted Bayesian SVAR (Antolín-Díaz & Rubio-Ramírez 2018).
/// Superset of sign_restricted_svar; narrative None + weights all 1 reproduces it.
#[pyfunction]
#[pyo3(signature = (data, sign_restrictions = vec![], narrative_restrictions = None, lags = 2,
                    horizon = 12, n_draws = 500, max_tries = 400, seed = 0, lambda1 = 0.2, n_weight_draws = 200))]
#[allow(clippy::too_many_arguments)]
fn narrative_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    sign_restrictions: Vec<(usize, usize, usize, String)>,
    narrative_restrictions: Option<Vec<Bound<'py, PyDict>>>,
    lags: usize,
    horizon: usize,
    n_draws: usize,
    max_tries: usize,
    seed: u64,
    lambda1: f64,
    n_weight_draws: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_ident::{NarrativeSampler, Sign, SignRestriction, SignRestrictionSet};
    let a = data.as_array();
    let m = tsecon_var::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let n = a.ncols();
    if a.nrows() <= lags {
        return Err(PyValueError::new_err("need more than `lags` observations"));
    }
    let t_eff = a.nrows() - lags;
    let prior = tsecon_bayes::MinnesotaNiwPrior::new(m.as_ref(), lags, 100.0, lambda1, 1.0, 0.0)
        .map_err(to_py)?;
    let posterior = prior.posterior(m.as_ref()).map_err(to_py)?;

    let signs = if sign_restrictions.is_empty() {
        None
    } else {
        let mut rs = Vec::new();
        for (v, s, h, sign) in sign_restrictions {
            let sg = match sign.as_str() {
                "+" | "positive" => Sign::Positive,
                "-" | "negative" => Sign::Negative,
                o => return Err(PyValueError::new_err(format!("unknown sign {o:?}"))),
            };
            rs.push(SignRestriction::at(v, s, h, sg));
        }
        Some(SignRestrictionSet::new(rs, n, horizon).map_err(to_py)?)
    };
    let narrative = match &narrative_restrictions {
        Some(l) if !l.is_empty() => Some(parse_narrative_restrictions(l, n, t_eff)?),
        _ => None,
    };
    let result = NarrativeSampler::new(horizon, n_draws, max_tries, n_weight_draws)
        .map_err(to_py)?
        .run(
            &posterior,
            m.as_ref(),
            signs.as_ref(),
            narrative.as_ref(),
            seed,
        )
        .map_err(to_py)?;

    let summary = result.irf_summary(horizon).map_err(to_py)?;
    let hs = horizon + 1;
    let mut quantiles = vec![vec![vec![Vec::<f64>::new(); n]; n]; hs];
    let (mut set_min, mut set_max) = (
        vec![vec![vec![0.0; n]; n]; hs],
        vec![vec![vec![0.0; n]; n]; hs],
    );
    for h in 0..hs {
        for i in 0..n {
            for j in 0..n {
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
    d.set_item("weights", result.weights().to_vec())?;
    let dg = result.diagnostics();
    let dd = PyDict::new(py);
    dd.set_item("posterior_draws_used", dg.posterior_draws_used)?;
    dd.set_item("rotations_tried", dg.rotations_tried)?;
    dd.set_item("accepted", dg.accepted)?;
    dd.set_item("acceptance_rate", dg.acceptance_rate)?;
    dd.set_item("narrative_accepted", dg.narrative_accepted)?;
    dd.set_item("narrative_acceptance_rate", dg.narrative_acceptance_rate)?;
    dd.set_item("ess", dg.ess)?;
    dd.set_item("mean_weight", dg.mean_weight)?;
    dd.set_item("min_ptilde", dg.min_ptilde)?;
    d.set_item("diagnostics", dd)?;
    Ok(d)
}

/// Fry-Pagan (2011) median-target SVAR: the single accepted sign-restricted
/// draw whose structural IRFs are jointly closest to the pointwise median.
#[pyfunction]
#[pyo3(signature = (data, restrictions, lags = 2, horizon = 12, n_draws = 500, max_tries = 400, seed = 0, lambda1 = 0.2, target = "restricted"))]
#[allow(clippy::too_many_arguments)]
fn fry_pagan_svar<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    restrictions: Vec<(usize, usize, usize, String)>,
    lags: usize,
    horizon: usize,
    n_draws: usize,
    max_tries: usize,
    seed: u64,
    lambda1: f64,
    target: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_ident::{median_target, Sign, SignRestriction, SignRestrictionSet, SignSampler};
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

    let shocks: Vec<usize> = match target {
        "restricted" => restr.restricted_shocks().to_vec(),
        "all" => (0..n_vars).collect(),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown target {other:?}; expected \"restricted\" or \"all\""
            )))
        }
    };
    let hs = horizon + 1;
    let mut cells = Vec::with_capacity(hs * n_vars * shocks.len());
    for h in 0..hs {
        for i in 0..n_vars {
            for &j in &shocks {
                cells.push((i, j, h));
            }
        }
    }

    let draws = result.draws();
    let mt = median_target(draws, &cells).map_err(to_py)?;

    let to_nested = |irf: &[tsecon_var::tsecon_linalg::faer::Mat<f64>]| {
        let mut out = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
        for (h, mh) in irf.iter().enumerate() {
            for i in 0..n_vars {
                for j in 0..n_vars {
                    out[h][i][j] = mh[(i, j)];
                }
            }
        }
        out
    };
    let median_target_irf = to_nested(&draws[mt.index]);
    let median_irf = to_nested(&mt.median_irf);

    let d = PyDict::new(py);
    d.set_item("median_target_irf", median_target_irf)?;
    d.set_item("median_irf", median_irf)?;
    d.set_item("mt_index", mt.index)?;
    d.set_item("mt_statistic", mt.statistic)?;
    d.set_item("n_accepted", draws.len())?;
    let diag = result.diagnostics();
    let dd = PyDict::new(py);
    dd.set_item("posterior_draws_used", diag.posterior_draws_used)?;
    dd.set_item("rotations_tried", diag.rotations_tried)?;
    dd.set_item("accepted", diag.accepted)?;
    dd.set_item("acceptance_rate", diag.acceptance_rate)?;
    d.set_item("diagnostics", dd)?;
    Ok(d)
}

/// Giacomini-Kitagawa (2021) prior-robust identified-set bounds for a sign-
/// restricted SVAR on the Minnesota-NIW posterior.
#[pyfunction]
#[pyo3(signature = (data, restrictions, lags = 2, horizon = 12, n_draws = 500, seed = 0, lambda1 = 0.2, alpha = 0.10))]
#[allow(clippy::too_many_arguments)]
fn robust_svar_bounds<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    restrictions: Vec<(usize, usize, usize, String)>,
    lags: usize,
    horizon: usize,
    n_draws: usize,
    seed: u64,
    lambda1: f64,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_ident::{robust_svar_bounds_default, Sign, SignRestriction, SignRestrictionSet};
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
    let result = robust_svar_bounds_default(&posterior, &restr, horizon, n_draws, seed, alpha)
        .map_err(to_py)?;

    let hs = horizon + 1;
    let mut set_lower_mean = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
    let mut set_upper_mean = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
    let mut robust_ci_lower = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
    let mut robust_ci_upper = vec![vec![vec![0.0_f64; n_vars]; n_vars]; hs];
    let mut lower_quantiles = vec![vec![vec![Vec::<f64>::new(); n_vars]; n_vars]; hs];
    let mut upper_quantiles = vec![vec![vec![Vec::<f64>::new(); n_vars]; n_vars]; hs];
    for h in 0..hs {
        for i in 0..n_vars {
            for j in 0..n_vars {
                let bp = result.point(i, j, h).map_err(to_py)?;
                set_lower_mean[h][i][j] = bp.set_lower_mean;
                set_upper_mean[h][i][j] = bp.set_upper_mean;
                robust_ci_lower[h][i][j] = bp.robust_ci_lower;
                robust_ci_upper[h][i][j] = bp.robust_ci_upper;
                lower_quantiles[h][i][j] = bp.lower_quantiles.clone();
                upper_quantiles[h][i][j] = bp.upper_quantiles.clone();
            }
        }
    }

    let d = PyDict::new(py);
    d.set_item("set_lower_mean", set_lower_mean)?;
    d.set_item("set_upper_mean", set_upper_mean)?;
    d.set_item("robust_ci_lower", robust_ci_lower)?;
    d.set_item("robust_ci_upper", robust_ci_upper)?;
    d.set_item("lower_quantiles", lower_quantiles)?;
    d.set_item("upper_quantiles", upper_quantiles)?;
    d.set_item("probs", result.probs().to_vec())?;
    d.set_item("alpha", result.alpha())?;
    d.set_item("restricted_shocks", result.restricted_shocks().to_vec())?;
    let diag = result.diagnostics();
    let dd = PyDict::new(py);
    dd.set_item("posterior_draws_used", diag.posterior_draws_used)?;
    dd.set_item("nonempty_draws", diag.nonempty_draws)?;
    dd.set_item("empty_set_rate", diag.empty_set_rate)?;
    d.set_item("diagnostics", dd)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(var_irf_bands, m)?)?;
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
    m.add_function(wrap_pyfunction!(lp_multiplier, m)?)?;
    m.add_function(wrap_pyfunction!(ridge, m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net, m)?)?;
    m.add_function(wrap_pyfunction!(lasso, m)?)?;
    m.add_function(wrap_pyfunction!(sign_restricted_svar, m)?)?;
    m.add_function(wrap_pyfunction!(zero_sign_svar, m)?)?;
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
    m.add_function(wrap_pyfunction!(panel_pmg, m)?)?;
    m.add_function(wrap_pyfunction!(panel_unit_root, m)?)?;
    m.add_function(wrap_pyfunction!(dfm_news, m)?)?;
    m.add_function(wrap_pyfunction!(predictive_regression, m)?)?;
    m.add_function(wrap_pyfunction!(ivx_test, m)?)?;
    m.add_function(wrap_pyfunction!(recession_probit, m)?)?;
    m.add_function(wrap_pyfunction!(cg_regression, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_efficiency, m)?)?;
    m.add_function(wrap_pyfunction!(frac_diff, m)?)?;
    m.add_function(wrap_pyfunction!(long_memory_d, m)?)?;
    m.add_function(wrap_pyfunction!(forecast_disagreement, m)?)?;
    m.add_function(wrap_pyfunction!(frac_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(heteroskedasticity_test, m)?)?;
    m.add_function(wrap_pyfunction!(reset_test, m)?)?;
    m.add_function(wrap_pyfunction!(chow_test, m)?)?;
    m.add_function(wrap_pyfunction!(cusum_test, m)?)?;
    m.add_function(wrap_pyfunction!(afns_adjustment, m)?)?;
    m.add_function(wrap_pyfunction!(dsge_solve, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_regression, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_lp, m)?)?;
    m.add_function(wrap_pyfunction!(growth_at_risk, m)?)?;
    m.add_function(wrap_pyfunction!(functional_pca, m)?)?;
    m.add_function(wrap_pyfunction!(flp, m)?)?;
    m.add_function(wrap_pyfunction!(flp_scenario, m)?)?;
    m.add_function(wrap_pyfunction!(fvar_scenario, m)?)?;
    m.add_function(wrap_pyfunction!(bai_perron, m)?)?;
    m.add_function(wrap_pyfunction!(sup_f_test, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_lp, m)?)?;
    m.add_function(wrap_pyfunction!(phillips_perron, m)?)?;
    m.add_function(wrap_pyfunction!(phillips_ouliaris, m)?)?;
    m.add_function(wrap_pyfunction!(long_run_svar, m)?)?;
    m.add_function(wrap_pyfunction!(max_share_svar, m)?)?;
    m.add_function(wrap_pyfunction!(proxy_svar, m)?)?;
    m.add_function(wrap_pyfunction!(nongaussian_svar, m)?)?;
    m.add_function(wrap_pyfunction!(hetero_svar, m)?)?;
    m.add_function(wrap_pyfunction!(bvar_hierarchical, m)?)?;
    m.add_function(wrap_pyfunction!(bvar_ssvs, m)?)?;
    m.add_function(wrap_pyfunction!(structural_fevd, m)?)?;
    m.add_function(wrap_pyfunction!(historical_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(narrative_svar, m)?)?;
    m.add_function(wrap_pyfunction!(fry_pagan_svar, m)?)?;
    m.add_function(wrap_pyfunction!(robust_svar_bounds, m)?)?;
    Ok(())
}
