//! Python bindings for equal-weighted cosine (EWC) HAR inference in
//! `tsecon-hac` — the Lazarus-Lewis-Stock-Watson (2018) long-run variance
//! `ewc_lrv` and its recommended degrees of freedom `ewc_default_b`, both
//! validated in Rust since 0.1 and bound in 0.11. The OLS sandwich that
//! uses them is `ols(se_type="ewc")` in `lib.rs`. Registered into `_core`
//! through [`register`].

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{to_py, vec1};

/// Refuse a `b` outside `[1, n - 1]` naming the Python keyword (`b` for
/// `ewc_lrv`, `ewc_b` for `ols`). The crate's own `InvalidDof` message says
/// "B =", which is not a name the caller typed, so the range is checked here
/// first.
pub(crate) fn check_ewc_b(func: &str, name: &str, b: usize, n: usize) -> PyResult<()> {
    if b == 0 || b > n.saturating_sub(1) {
        return Err(PyValueError::new_err(format!(
            "{func}: {name} = {b} is outside [1, n - 1] = [1, {}] for n = {n} observations. \
             {name} counts the cosine frequencies averaged — it is also the Student-t \
             degrees of freedom the t-statistics are referred to — and the basis has only \
             n - 1 non-constant vectors; the LLSW (2018) default is ewc_default_b(n) = \
             round(0.4 * n^(2/3)) = {}. Pass None for the default",
            n.saturating_sub(1),
            tsecon_hac::ewc_default_b(n),
        )));
    }
    Ok(())
}

/// Equal-weighted cosine (EWC) long-run variance of a series — the
/// Lazarus-Lewis-Stock-Watson (2018) HAR estimator (demeaned internally).
///
/// `Omega_hat = (1/b) * sum_{j=1..b} Lambda_j^2` with
/// `Lambda_j = sqrt(2/n) * sum_t cos(pi * j * (t - 1/2) / n) * x_t`: the
/// average of the series' first `b` squared type-II cosine coefficients. The
/// `j = 0` (mean) frequency is excluded, so the estimate is invariant to the
/// mean in exact arithmetic; demeaning first only helps conditioning, as in
/// `long_run_variance`. At `b = n - 1` it collapses to the plain sample
/// variance `sum (x - mean)^2 / (n - 1)`.
///
/// PAIR IT WITH STUDENT-t CRITICAL VALUES WITH `b` DEGREES OF FREEDOM, not the
/// normal: `Omega_hat` is asymptotically `Omega * chi2_b / b`, so
/// `t = (mean - mu0) / sqrt(Omega_hat / n)` is `t_b` under the null
/// (fixed-b inference). That is the whole point of EWC — at the default `b`
/// it is what keeps the size close to nominal where Newey-West with normal
/// critical values over-rejects.
///
/// `b=None` uses `ewc_default_b(len(x))` = round(0.4 n^(2/3)). An explicit
/// `b` must lie in `[1, n - 1]`.
///
/// Validated against the documented closed form evaluated with
/// `scipy.fft.dct(type=2, norm="ortho")` (no reference package implements
/// EWC) at 1e-12 — `fixtures/ewc.json`.
///
/// Further arguments, with defaults: `b` (None: `ewc_default_b(len(x))`).
#[pyfunction]
#[pyo3(signature = (x, b = None))]
fn ewc_lrv(x: PyReadonlyArray1<'_, f64>, b: Option<usize>) -> PyResult<f64> {
    let xs = vec1(&x);
    let n = xs.len();
    if n < 2 {
        return Err(PyValueError::new_err(format!(
            "ewc_lrv: x has {n} observation(s) but a long-run variance needs at least 2 \
             (the cosine basis beyond the mean has n - 1 vectors)"
        )));
    }
    if let Some((i, v)) = xs.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "ewc_lrv: x contains a non-finite value ({v}) at index {i}; HAC estimators do not \
             skip missing values silently — drop or impute NaN/inf observations first"
        )));
    }
    let b = b.unwrap_or_else(|| tsecon_hac::ewc_default_b(n));
    check_ewc_b("ewc_lrv", "b", b, n)?;
    let mean = xs.iter().sum::<f64>() / n as f64;
    let z: Vec<f64> = xs.iter().map(|v| v - mean).collect();
    tsecon_hac::ewc_lrv(&z, b).map_err(to_py)
}

/// The Lazarus-Lewis-Stock-Watson (2018) recommended EWC degrees of freedom
/// for a sample of `n` observations: `round(0.4 * n^(2/3))`, clamped into the
/// valid range `[1, n - 1]`.
///
/// It is the default `b` of `ewc_lrv` and the default `ewc_b` of
/// `ols(se_type="ewc")`, and it is also the Student-t degrees of freedom to
/// refer the resulting t-statistics to. (Exact .5 ties cannot occur:
/// `n^(2/3)` is rational only for perfect cubes `n = m^3`, where `0.4 m^2`
/// never ends in .5.) `n < 2` is refused: no long-run variance exists.
#[pyfunction]
fn ewc_default_b(n: usize) -> PyResult<usize> {
    if n < 2 {
        return Err(PyValueError::new_err(format!(
            "ewc_default_b: n = {n} observations is too few; an EWC long-run variance needs \
             n >= 2 (the b cosine frequencies come from the n - 1 non-constant basis vectors)"
        )));
    }
    Ok(tsecon_hac::ewc_default_b(n))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ewc_lrv, m)?)?;
    m.add_function(wrap_pyfunction!(ewc_default_b, m)?)?;
    Ok(())
}
