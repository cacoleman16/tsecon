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

/// Work budget of the direct EWC cosine transform, in units of about 1.6 ns
/// of one core. The crate evaluates every cosine `cos(pi j (t - 1/2) / n)`
/// directly — about 8 units — then one multiply-add per column, so a call
/// costs `n * b * (k + 8)` units: SUPERLINEAR in `n`, like `n^(5/3)` at the
/// default `b = round(0.4 n^(2/3))`. Measured for one series at the default
/// `b` (lab/audit/wave11/ewc_varcov_measurements.py --ewc-cost): 0.03 s at
/// n = 10^4, 1.3 s at 10^5, 57 s at n = 972427 (`b` = 3926), the largest `n`
/// this budget admits there. Past it the call is refused before any work,
/// naming the keyword, rather than left to run for many minutes.
const EWC_WORK_BUDGET: u128 = 1 << 35;

/// Refuse, naming `name` (`b` / `ewc_b`), an EWC request over `n`
/// observations and `k` columns whose cost exceeds [`EWC_WORK_BUDGET`],
/// pointing at the largest `b` the budget admits.
pub(crate) fn check_ewc_work(func: &str, name: &str, b: usize, n: usize, k: usize) -> PyResult<()> {
    let per_b = (n as u128).saturating_mul(k as u128 + 8);
    let work = per_b.saturating_mul(b as u128);
    if work > EWC_WORK_BUDGET {
        return Err(PyValueError::new_err(format!(
            "{func}: {name} = {b} cosine frequencies over n = {n} observations is past the \
             work budget of the direct EWC transform: it evaluates n * {name} cosines, about \
             n * {name} * (k + 8) = {work:.2e} operations for k = {k} column(s), against a \
             budget of 2^35 (about a minute; the cost grows like n^(5/3) at the default \
             {name}). Pass {name} <= {} — EWC stays valid at a smaller {name}, paired with \
             Student-t critical values with that many degrees of freedom — or a shorter series",
            EWC_WORK_BUDGET / per_b.max(1),
            work = work as f64,
        )));
    }
    Ok(())
}

/// Two-sided Student-`t_b` p-values `2 * sf(|t|)` for `ols(se_type="ewc")`.
/// A NaN t-statistic (no residual variation, or an overflowed covariance)
/// stays a NaN p-value: `f64::min(NaN, 1.0)` would report it as 1.
pub(crate) fn t_b_pvalues(b: usize, tvalues: &[f64]) -> PyResult<Vec<f64>> {
    use tsecon_stats::ContinuousDist;
    let t_b = tsecon_stats::StudentT::new(b as f64).map_err(to_py)?;
    Ok(tvalues
        .iter()
        .map(|t| {
            let p = 2.0 * t_b.sf(t.abs());
            if p > 1.0 {
                1.0
            } else {
                p
            }
        })
        .collect())
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
/// COST: the cosines are evaluated directly, `n * b` of them, so the work is
/// SUPERLINEAR in `n` — like `n^(5/3)` at the default `b` (measured 0.03 s at
/// n = 10^4, 1.3 s at 10^5, 57 s at n = 972427, the largest series whose
/// default `b` fits the budget). A request past a work budget of 2^35
/// (`n * b * 9` operations, about a minute) is refused naming `b`, with the
/// largest `b` it admits; EWC stays valid at a smaller `b`.
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
    check_ewc_work("ewc_lrv", "b", b, n, 1)?;
    let mean = xs.iter().sum::<f64>() / n as f64;
    let z: Vec<f64> = xs.iter().map(|v| v - mean).collect();
    // Finite input can still overflow double precision in the mean or the
    // squared cosine coefficients (|x| near 1e308); name `x` rather than
    // returning inf or surfacing the crate's message about its own input.
    let overflow = || {
        PyValueError::new_err(
            "ewc_lrv: x overflows double precision (its mean or its squared cosine \
             coefficients exceed ~1.8e308); rescale x, e.g. divide it by its largest \
             absolute value, and multiply the result by that value squared",
        )
    };
    if !mean.is_finite() || z.iter().any(|v| !v.is_finite()) {
        return Err(overflow());
    }
    let lrv = tsecon_hac::ewc_lrv(&z, b).map_err(to_py)?;
    if lrv.is_finite() {
        Ok(lrv)
    } else {
        Err(overflow())
    }
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
