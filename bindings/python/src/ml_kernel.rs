//! Python bindings for the kernel-methods slice of `tsecon-ml`:
//! `kernel_ridge` (exact / random-Fourier-feature kernel ridge regression)
//! and `kernel_regression` (Nadaraya-Watson and local-linear smoothing
//! with dependence-aware bandwidth selection).

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tsecon_var::tsecon_linalg::faer::Mat;

use crate::{to_faer, to_py, vec1};

/// A design matrix from a 2-D `(n, k)` array or a 1-D `(n,)` array (one
/// regressor), with a teaching error for anything else.
fn design<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Mat<f64>> {
    if let Ok(a) = obj.extract::<PyReadonlyArray2<'py, f64>>() {
        return Ok(to_faer(&a));
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'py, f64>>() {
        let v = vec1(&a);
        return Ok(Mat::from_fn(v.len(), 1, |i, _| v[i]));
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a float64 NumPy array shaped (n, k) — one row per observation, one \
         column per regressor — or 1-D (n,) for a single regressor; got {}. A pandas \
         DataFrame/Series, an integer array, or a plain list of numbers is converted \
         automatically; a nested list is not — wrap it with np.array(...)",
        obj.get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_default()
    )))
}

/// Kernel ridge regression: exact dual solve, or the Rahimi-Recht
/// random-Fourier-feature approximation of the rbf kernel.
///
/// Minimizes `sum_i (y_i - f(x_i))^2 + alpha * ||f||_H^2` over the RKHS of
/// the kernel — scikit-learn's `KernelRidge` objective (no `1/n`, no
/// intercept: center `y` if the kernel does not model a level). The exact
/// solution is `(K + alpha I) a = y` by Cholesky. Kernels in scikit-learn's
/// exact parameterization: `kernel="rbf"` `exp(-gamma ||x-y||^2)`,
/// `"laplacian"` `exp(-gamma ||x-y||_1)`, `"polynomial"`
/// `(gamma <x,y> + coef0)^degree`, `"linear"` `<x,y>`. `gamma=None`
/// resolves to `1 / n_features` (scikit-learn's default); the linear
/// kernel has no gamma and refuses one. `degree`/`coef0` act on the
/// polynomial kernel only and are refused at non-default values elsewhere
/// (nothing is silently ignored).
///
/// `rff_features=D` switches to the random-Fourier-feature primal
/// approximation (Rahimi & Recht 2007): `z(x) = sqrt(2/D) cos(Wx + b)`
/// with `W ~ N(0, 2 gamma I)`, `b ~ U[0, 2 pi)` drawn from a Philox stream
/// keyed by `seed` (same seed, bit-identical features), then ridge on `z`
/// — `O(n D^2)` instead of `O(n^3)`, converging to the exact fit as `D`
/// grows. rbf only; `seed` is refused in exact mode. `x_test` (`(m, k)`)
/// adds `predicted`. `alpha=0` is the interpolating fit and is refused
/// when `K` is not positive definite (scikit-learn silently falls back to
/// least squares there; tsecon raises and names `alpha`).
///
/// Keys: `dual_coef` (exact mode; the `a` of `f(x) = sum_i a_i k(x, x_i)`,
/// scikit-learn's `dual_coef_`) or `coef` (RFF mode; the `D` primal
/// weights), `fitted`, `predicted` (only when `x_test` is given),
/// `kernel`, `gamma` (resolved; None for linear), `n_rff_features` (None
/// in exact mode).
///
/// Validated against scikit-learn 1.9.0 `KernelRidge` — `dual_coef_`,
/// `predict(X)` and `predict(X_test)` for all four kernels at 1e-8
/// (independent package). The RFF approximation is a Monte-Carlo object
/// and is property-tested (seeded determinism; error against the exact
/// fit falling with `D`), not golden-pinned.
#[pyfunction]
#[pyo3(signature = (x, y, alpha = 1.0, kernel = "rbf", gamma = None, degree = 3.0, coef0 = 1.0, x_test = None, rff_features = None, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn kernel_ridge<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: PyReadonlyArray1<'py, f64>,
    alpha: f64,
    kernel: &str,
    gamma: Option<f64>,
    degree: f64,
    coef0: f64,
    x_test: Option<&Bound<'py, PyAny>>,
    rff_features: Option<usize>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let xm = design(x, "x")?;
    let xt = x_test.map(|t| design(t, "x_test")).transpose()?;
    let opts = tsecon_ml::KernelRidgeOptions {
        alpha,
        kernel: tsecon_ml::KernelType::parse(kernel).map_err(to_py)?,
        gamma,
        degree,
        coef0,
        rff_features,
        seed,
    };
    let fit = tsecon_ml::kernel_ridge(
        xm.as_ref(),
        &vec1(&y),
        xt.as_ref().map(|m| m.as_ref()),
        &opts,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    if let Some(a) = fit.dual_coef {
        d.set_item("dual_coef", a.into_pyarray(py))?;
    }
    if let Some(c) = fit.coef {
        d.set_item("coef", c.into_pyarray(py))?;
    }
    d.set_item("fitted", fit.fitted.into_pyarray(py))?;
    if let Some(p) = fit.predicted {
        d.set_item("predicted", p.into_pyarray(py))?;
    }
    d.set_item("kernel", fit.kernel.as_str())?;
    d.set_item("gamma", fit.gamma)?;
    d.set_item("n_rff_features", fit.n_rff_features)?;
    Ok(d)
}

/// A fixed bandwidth: a positive scalar (broadcast to every column) or a
/// length-`k` sequence / 1-D array.
fn parse_bandwidth<'py>(obj: &Bound<'py, PyAny>, k: usize) -> PyResult<Vec<f64>> {
    if let Ok(h) = obj.extract::<f64>() {
        return Ok(vec![h; k]);
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray1<'py, f64>>() {
        return Ok(vec1(&a));
    }
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(v);
    }
    Err(PyTypeError::new_err(format!(
        "bandwidth must be a positive float (one Gaussian bandwidth for every column of x) \
         or a sequence of {k} floats (one per column); got {}",
        obj.get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_default()
    )))
}

/// Nadaraya-Watson or local-linear kernel regression of `y` on `x`
/// (`(n, k)`, `k <= 3`, or 1-D for one regressor) with a product Gaussian
/// kernel, at a fixed or cross-validated bandwidth.
///
/// Conventions are statsmodels `KernelReg(reg_type="lc" | "ll",
/// var_type="c"*k)` exactly: `kind="nadaraya_watson"` is the local
/// constant `sum_i K_h(x_i - x) y_i / sum_i K_h(x_i - x)`;
/// `kind="local_linear"` (default — no boundary bias) is the intercept of
/// the kernel-weighted least squares of `y` on `[1, x_i - x]`, solved
/// through the pseudoinverse as statsmodels does. `kernel`: `"gaussian"`
/// only (the one statsmodels validates against; compact-support kernels
/// are deferred). The bandwidth is the kernel's standard deviation per
/// column, in the column's units.
///
/// `bandwidth_method="fixed"` (default) uses `bandwidth` (a positive
/// scalar broadcast to every column, or one value per column) as given.
/// `"loo_cv"` minimizes the leave-one-out least-squares criterion
/// `n^-1 sum_i (y_i - g_{-i}(x_i))^2` (statsmodels `cv_loo`).
/// `"block_cv"` minimizes the leave-block-out criterion (Chu & Marron
/// 1991): predicting `y_i` drops the `2*block + 1` observations with
/// `|j - i| <= block` (default `block = ceil(n^(1/3))`), so serially
/// correlated neighbours never vote on their own errors — leave-one-out
/// undersmooths badly under autocorrelated errors, and this is the method
/// to use for time-series regressors. Selection is a 21-point log grid
/// on a common multiple of the Scott reference `1.06 sd(x_j) n^(-1/(4+k))`
/// over `[0.05, 20]`, golden-section refinement, then per-column
/// coordinate refinement for `k >= 2`; deterministic, and not statsmodels'
/// Nelder-Mead path (the criterion value at any bandwidth matches
/// statsmodels at 1e-10; the search reaches a criterion no worse than
/// fmin's). Under the CV methods `bandwidth` must be omitted and under
/// `"fixed"`/`"loo_cv"` `block` must be omitted — a conflicting argument
/// raises rather than being ignored.
///
/// Keys: `fitted` (at the training rows), `predicted` (only when `x_test`
/// is given; NaN where every training weight underflows), `bandwidth`
/// (resolved, one per column), `bandwidth_method`, `block` (resolved
/// half-width under `"block_cv"`, else None), `cv_criterion` (the
/// leave-one-out criterion under `"fixed"`/`"loo_cv"`, the leave-block-out
/// criterion under `"block_cv"`, at the reported bandwidth), `effective_df`
/// (`tr(S)` of the linear smoother: from `k+1` (local linear) or `1`
/// (Nadaraya-Watson) at huge bandwidths up to `n` at tiny ones), `kind`,
/// `kernel`, `bandwidth_at_boundary` (True when a selected bandwidth sits
/// on a wall of the search range — the criterion was still falling, so
/// the reported value is the search's limit, not an interior optimum;
/// typically a target with no detectable signal), and
/// `n_criterion_evaluations` (0 under `"fixed"`).
///
/// Validated against statsmodels 0.15.0 `KernelReg.fit()` at fixed
/// bandwidths (`k = 1, 2`, both estimators) at 1e-8 and `cv_loo` at 1e-10
/// (independent package); the leave-block-out criterion and
/// `effective_df` are documented-formula transcriptions (no package
/// computes them) pinned at 1e-10.
#[pyfunction]
#[pyo3(signature = (x, y, bandwidth = None, kind = "local_linear", kernel = "gaussian", bandwidth_method = "fixed", block = None, x_test = None))]
#[allow(clippy::too_many_arguments)]
fn kernel_regression<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: PyReadonlyArray1<'py, f64>,
    bandwidth: Option<&Bound<'py, PyAny>>,
    kind: &str,
    kernel: &str,
    bandwidth_method: &str,
    block: Option<usize>,
    x_test: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let xm = design(x, "x")?;
    let xt = x_test.map(|t| design(t, "x_test")).transpose()?;
    let k = xm.ncols();
    let kind_p = tsecon_ml::RegressionKind::parse(kind).map_err(to_py)?;
    let kernel_p = tsecon_ml::RegressionKernel::parse(kernel).map_err(to_py)?;

    // Resolve the bandwidth specification with the inert-argument rule:
    // an argument the chosen method cannot use is refused, never dropped.
    let spec = match bandwidth_method {
        "fixed" => {
            if let Some(b) = block {
                return Err(PyValueError::new_err(format!(
                    "block={b} has no effect under bandwidth_method=\"fixed\": block is the \
                     leave-block-out exclusion half-width of the cross-validation criterion, \
                     and a fixed bandwidth runs no cross-validation, so it would be silently \
                     ignored. Pass bandwidth_method=\"block_cv\" (and drop bandwidth) to \
                     select the bandwidth by leave-block-out CV, or drop block"
                )));
            }
            let Some(bw) = bandwidth else {
                return Err(PyValueError::new_err(
                    "bandwidth is required under bandwidth_method=\"fixed\": pass a positive \
                     float (one Gaussian bandwidth, in the units of x, for every column) or \
                     one value per column — or choose the bandwidth by cross-validation with \
                     bandwidth_method=\"loo_cv\" (leave-one-out) or \"block_cv\" (leave-block-\
                     out, the dependence-aware choice for time-series regressors)",
                ));
            };
            tsecon_ml::BandwidthSpec::Fixed(parse_bandwidth(bw, k)?)
        }
        "loo_cv" | "block_cv" => {
            if let Some(bw) = bandwidth {
                return Err(PyValueError::new_err(format!(
                    "bandwidth={} conflicts with bandwidth_method={bandwidth_method:?}: that \
                     method selects the bandwidth by cross-validation and would silently \
                     discard the value you passed. Drop bandwidth to let the criterion choose \
                     it, or pass bandwidth_method=\"fixed\" to use your value as given",
                    bw.repr().map(|r| r.to_string()).unwrap_or_default()
                )));
            }
            if bandwidth_method == "loo_cv" {
                if let Some(b) = block {
                    return Err(PyValueError::new_err(format!(
                        "block={b} has no effect under bandwidth_method=\"loo_cv\": \
                         leave-one-out drops exactly the observation being predicted, so \
                         the block half-width would be silently ignored. Pass \
                         bandwidth_method=\"block_cv\" for the leave-block-out criterion \
                         that uses it, or drop block"
                    )));
                }
                tsecon_ml::BandwidthSpec::LooCv
            } else {
                tsecon_ml::BandwidthSpec::BlockCv { block }
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown bandwidth_method {other:?}; accepted values are \"fixed\" (use \
                 bandwidth as given), \"loo_cv\" (leave-one-out least-squares CV) and \
                 \"block_cv\" (leave-block-out CV with half-width block, the dependence-aware \
                 choice)"
            )))
        }
    };

    let opts = tsecon_ml::KernelRegressionOptions {
        kind: kind_p,
        kernel: kernel_p,
        bandwidth: spec,
    };
    let fit = tsecon_ml::kernel_regression(
        xm.as_ref(),
        &vec1(&y),
        xt.as_ref().map(|m| m.as_ref()),
        &opts,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("fitted", fit.fitted.into_pyarray(py))?;
    if let Some(p) = fit.predicted {
        d.set_item("predicted", p.into_pyarray(py))?;
    }
    d.set_item("bandwidth", fit.bandwidth.into_pyarray(py))?;
    d.set_item("bandwidth_method", fit.bandwidth_method)?;
    d.set_item("block", fit.block)?;
    d.set_item("cv_criterion", fit.cv_criterion)?;
    d.set_item("effective_df", fit.effective_df)?;
    d.set_item("kind", fit.kind.as_str())?;
    d.set_item("kernel", fit.kernel.as_str())?;
    d.set_item("bandwidth_at_boundary", fit.bandwidth_at_boundary)?;
    d.set_item("n_criterion_evaluations", fit.n_criterion_evaluations)?;
    Ok(d)
}

/// Registers the kernel-methods functions on the `_core` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kernel_ridge, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_regression, m)?)?;
    Ok(())
}
