//! Python bindings for the threshold-inference layer of `tsecon-regime`:
//! `setar_threshold_ci`, the Hansen (1997/2000) likelihood-ratio
//! confidence set for the SETAR threshold. Registered into `_core` through
//! [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{to_py, vec1};

/// Hansen (1997/2000) likelihood-ratio confidence set for the threshold of
/// a two-regime SETAR(p), built on exactly the `setar` fit.
///
/// The fit is `setar(y, p, delay/delays, trim, constant)` itself — the
/// returned `threshold`, `delay`, `thresholds` and `ssr_path` are
/// bit-identical to its (`delays` overrides `delay`, as there). Over the
/// candidate grid the profile `LR_n(gamma) = nobs * (S(gamma) - S_min) /
/// S_min` (Hansen 2000; Hansen 1997 for the TAR) is inverted against the
/// closed-form critical value `c = -2 ln(1 - sqrt(level))` — the `level`
/// quantile of `P(xi <= x) = (1 - exp(-x/2))^2`, Hansen (2000) Table 1:
/// 4.50 / 5.94 / 7.35 / 10.59 at 80 / 90 / 95 / 99% — giving the set
/// `{gamma : LR_n(gamma) <= eta2 * c}`. The set always contains the
/// estimate (`LR_n = 0` there), is typically asymmetric, and CAN BE
/// DISJOINT, so it is returned as a list of closed `[low, high]`
/// `intervals` (maximal runs of in-set candidates, grid endpoints as in
/// Hansen's own programs) with `is_connected`, `n_intervals`, and the
/// convex hull `ci_low` / `ci_high`; `in_set` flags each candidate and
/// `n_in_set` counts them. `LR_n` is a step function, constant between
/// adjacent candidates, so a `null_threshold` gamma_0 (any value inside
/// `[thresholds[0], thresholds[-1]]`) is evaluated at the largest candidate
/// `<= gamma_0` (`null_threshold_used`): `lr_at_null` and the p-value
/// `pvalue_at_threshold = p(lr_at_null / eta2)`, `p(x) = 1 - (1 -
/// exp(-x/2))^2` — test inversion at one point; all three are None when no
/// null is passed.
///
/// `het_robust=True` applies Hansen's (2000, section 3.4)
/// heteroskedasticity correction: the critical value is scaled by
/// `eta2 = E[e^2 (x'delta)^2 | q = gamma] / (sigma^2 E[(x'delta)^2 | q =
/// gamma])`, estimated as his programs do — at the threshold estimate,
/// regress `(x'delta_hat)^2` and `e_hat^2 (x'delta_hat)^2` each on a
/// quadratic polynomial in the threshold variable `y_{t-d}` (with
/// intercept), take the ratio of the fitted values at the estimate, and
/// divide by `sigma_hat^2 = S_min / nobs`. `eta2` is exactly 1 otherwise.
/// With no threshold effect the fitted ratio can be non-positive; that is
/// refused with a teaching error rather than returned as a negative scale.
///
/// `slope_level=0.95` (say) adds Hansen's (2000, section 3.3) CONSERVATIVE
/// slope intervals: the union, over every candidate in the
/// `slope_region_level` threshold set (default 0.80, his applied
/// convention), of the conventional per-regime intervals `b_j(gamma) +/-
/// z se_j(gamma)` — classical per-regime SEs as `setar` reports (at the
/// estimate exactly `bse_low` / `bse_high`), or White HC0 SEs under
/// `het_robust`. They come back as `slope_ci_low` and `slope_ci_high`,
/// each a 2 x k nested list `[[low-regime coefficients], [high-regime
/// coefficients]]` (constant first, then lags 1..p), with the region used
/// in `slope_region_low` / `slope_region_high` / `slope_n_region`; all are
/// None without `slope_level`, and `slope_region_level` passed without
/// `slope_level` RAISES (it would be inert).
///
/// Validation: closed-form critical values and p-values pinned at 1e-14
/// (Table 1 reproduced to the printed decimals); the LR profile, the
/// `eta2` convention, the intervals and the slope unions pinned at 1e-10
/// against an independent NumPy transcription (fixtures/setar_ci.json —
/// no third-party threshold-CI code runs in the fixture container);
/// coverage MEASURED by seeded Monte Carlo in the crate's property tests
/// and quoted in the model card (asymptotically conservative for a fixed
/// threshold effect, as Hansen's theory says).
///
/// Further arguments, with defaults: `delay` (1), `trim` (0.15), `delays`
/// (None), `constant` (True), `level` (0.95), `het_robust` (False),
/// `slope_level` (None), `slope_region_level` (None: 0.80 when slope
/// intervals are requested), `null_threshold` (None).
///
/// Returned keys: `threshold`, `delay`, `nobs`, `k`, `level`, `lr_crit`,
/// `lr_crit_scaled`, `eta2`, `het_robust`, `thresholds`, `ssr_path`,
/// `lr_stat`, `in_set`, `intervals`, `n_intervals`, `is_connected`,
/// `ci_low`, `ci_high`, `n_in_set`, `null_threshold_used`, `lr_at_null`,
/// `pvalue_at_threshold`, `slope_level`, `slope_region_level`,
/// `slope_region_low`, `slope_region_high`, `slope_n_region`,
/// `slope_ci_low`, `slope_ci_high`.
#[pyfunction]
#[pyo3(signature = (y, p, delay = 1, trim = 0.15, delays = None, constant = true, level = 0.95, het_robust = false, slope_level = None, slope_region_level = None, null_threshold = None))]
#[allow(clippy::too_many_arguments)]
fn setar_threshold_ci<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    p: usize,
    delay: usize,
    trim: f64,
    delays: Option<Vec<usize>>,
    constant: bool,
    level: f64,
    het_robust: bool,
    slope_level: Option<f64>,
    slope_region_level: Option<f64>,
    null_threshold: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if slope_level.is_none() && slope_region_level.is_some() {
        return Err(PyValueError::new_err(
            "slope_region_level was given but slope_level is None: the region level \
             only sets which threshold confidence set the conservative slope \
             intervals are unioned over, so without slope_level it is inert; pass \
             slope_level (e.g. 0.95) or drop slope_region_level",
        ));
    }
    let opts = tsecon_regime::ThresholdCiOptions {
        level,
        het_robust,
        slope_level,
        slope_region_level: slope_region_level.unwrap_or(0.80),
        null_threshold,
    };
    let dl: Vec<usize> = delays.unwrap_or_else(|| vec![delay]);
    let ys = vec1(&y);
    let r = tsecon_regime::setar_threshold_ci(&ys, p, &dl, trim, constant, &opts).map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("threshold", r.threshold)?;
    d.set_item("delay", r.delay.unwrap_or(0))?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("k", r.k)?;
    d.set_item("level", r.level)?;
    d.set_item("lr_crit", r.lr_crit)?;
    d.set_item("lr_crit_scaled", r.lr_crit_scaled)?;
    d.set_item("eta2", r.eta2)?;
    d.set_item("het_robust", r.het_robust)?;
    d.set_item("thresholds", r.thresholds.into_pyarray(py))?;
    d.set_item("ssr_path", r.ssr_path.into_pyarray(py))?;
    d.set_item("lr_stat", r.lr_stat.into_pyarray(py))?;
    d.set_item("in_set", r.in_set.into_pyarray(py))?;
    let intervals = PyList::empty(py);
    for (lo, hi) in &r.intervals {
        intervals.append(PyList::new(py, [*lo, *hi])?)?;
    }
    d.set_item("n_intervals", r.intervals.len())?;
    d.set_item("intervals", intervals)?;
    d.set_item("is_connected", r.is_connected)?;
    d.set_item("ci_low", r.ci_low)?;
    d.set_item("ci_high", r.ci_high)?;
    d.set_item("n_in_set", r.n_in_set)?;
    d.set_item("null_threshold_used", r.null_threshold_used)?;
    d.set_item("lr_at_null", r.lr_at_null)?;
    d.set_item("pvalue_at_threshold", r.pvalue_at_threshold)?;
    match r.slope {
        None => {
            d.set_item("slope_level", py.None())?;
            d.set_item("slope_region_level", py.None())?;
            d.set_item("slope_region_low", py.None())?;
            d.set_item("slope_region_high", py.None())?;
            d.set_item("slope_n_region", py.None())?;
            d.set_item("slope_ci_low", py.None())?;
            d.set_item("slope_ci_high", py.None())?;
        }
        Some(s) => {
            d.set_item("slope_level", s.level)?;
            d.set_item("slope_region_level", s.region_level)?;
            d.set_item("slope_region_low", s.region_low)?;
            d.set_item("slope_region_high", s.region_high)?;
            d.set_item("slope_n_region", s.n_region)?;
            d.set_item(
                "slope_ci_low",
                PyList::new(
                    py,
                    [
                        PyList::new(py, s.low_lower)?,
                        PyList::new(py, s.high_lower)?,
                    ],
                )?,
            )?;
            d.set_item(
                "slope_ci_high",
                PyList::new(
                    py,
                    [
                        PyList::new(py, s.low_upper)?,
                        PyList::new(py, s.high_upper)?,
                    ],
                )?,
            )?;
        }
    }
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(setar_threshold_ci, m)?)?;
    Ok(())
}
