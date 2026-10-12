//! Python binding for the Lo-MacKinlay (1988) variance-ratio test and the
//! Chow-Denning (1993) multiple-horizon joint test of `tsecon-diag`
//! (`variance_ratio`). Registered into `_core` through [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyMemoryView, PySet,
    PyString,
};
use tsecon_diag::{DiagError, VarianceRatioSpec, VrTrend};

use crate::{to_py, vec1};

const LAGS_HINT: &str = "pass an int horizon q >= 2 (e.g. lags=4) or a sequence of \
                         distinct horizons (e.g. lags=[2, 4, 8, 16])";

fn type_name(v: &Bound<'_, PyAny>) -> String {
    v.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_default()
}

/// A NumPy bool scalar or bool array (`dtype.kind == "b"`).
fn is_numpy_bool(v: &Bound<'_, PyAny>) -> bool {
    v.getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .is_ok_and(|k| k == "b")
}

/// The refusal for a value that is not an integer horizon or a sequence of
/// them, naming the value and its type.
fn not_a_horizon(h: &Bound<'_, PyAny>) -> PyErr {
    let what = if h.extract::<f64>().is_ok() {
        format!("{h} of type {}", type_name(h))
    } else {
        type_name(h)
    };
    PyValueError::new_err(format!(
        "variance_ratio: lags must be an integer horizon q >= 2 or a 1-D sequence of \
         them; got {what} — {LAGS_HINT}"
    ))
}

/// A NumPy floating scalar or array (`dtype.kind == "f"`).
fn is_numpy_float(v: &Bound<'_, PyAny>) -> bool {
    v.getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .is_ok_and(|k| k == "f")
}

/// One horizon from a sequence (`lags[j]`): an integer >= 2 — a Python or
/// NumPy integer. A float is refused even when integral, exactly as the
/// scalar `lags=4.0` is, so `[4.0]`, `(4.0,)` and a float array follow the
/// same rule as `4.0` (and as `ardl_bounds_test`'s `order`).
fn one_horizon(item: &Bound<'_, PyAny>, j: usize) -> PyResult<usize> {
    let shown = item
        .repr()
        .map(|r| r.to_string())
        .unwrap_or_else(|_| "?".to_string());
    if item.is_instance_of::<PyBool>() || is_numpy_bool(item) {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags[{j}] = {shown} is a bool, not a horizon — {LAGS_HINT}"
        )));
    }
    if item.is_instance_of::<PyFloat>() || is_numpy_float(item) {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags[{j}] = {shown} is a float, not an integer horizon: a \
             float is refused inside a sequence exactly as the scalar lags=4.0 is (a float \
             array too) — {LAGS_HINT}"
        )));
    }
    if item.is_instance_of::<PyString>() || item.is_instance_of::<PyBytes>() {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags[{j}] = {shown} is a string, not a horizon — {LAGS_HINT}"
        )));
    }
    if let Ok(v) = item.extract::<i64>() {
        return horizon_from_i64(v);
    }
    if let Ok(v) = item.extract::<u64>() {
        // Above i64::MAX: refused downstream by the length rule.
        return Ok(usize::try_from(v).unwrap_or(usize::MAX));
    }
    if item.is_instance_of::<PyInt>() {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags[{j}] = {shown} is out of range; a horizon must be an \
             integer between 2 and n - 2 — {LAGS_HINT}"
        )));
    }
    Err(PyValueError::new_err(format!(
        "variance_ratio: lags[{j}] = {shown} (of type {}) is not an integer horizon — \
         {LAGS_HINT}",
        type_name(item)
    )))
}

fn below_two(v: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!(
        "variance_ratio: lags must be >= 2; got {v}. Each horizon q compares the \
         variance of q-period changes with q times the one-period variance; at q = 1 \
         that is the one-period variance itself, so the ratio is identically 1 — \
         {LAGS_HINT}"
    ))
}

fn horizon_from_i64(v: i64) -> PyResult<usize> {
    if v < 2 {
        return Err(below_two(v));
    }
    Ok(usize::try_from(v).unwrap_or(usize::MAX))
}

/// `lags` accepts `None` (-> 2), an int (Python int, NumPy integer, or a 0-d
/// integer array, via `__index__`), or a sequence of ints (a list, a tuple,
/// an integer array, a pandas Series — `lags` is in the coercion layer's
/// `_EXEMPT` table, so each reaches this parser as given). One rule for
/// both forms: a float is refused whether it is the scalar `4.0` or an
/// element of `[4.0]`, `(4.0,)` or a float array. Returns the horizons and
/// whether a scalar was given (scalar in -> scalars out). Anything else is a
/// teaching error that names `lags` (or `lags[j]`). A series of length `n`
/// admits at most `n - 3` distinct horizons, so reading stops past
/// `max(n, 16)` elements: an endless or enormous iterable is refused
/// instead of being collected.
fn parse_vr_lags(lags: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<(Vec<usize>, bool)> {
    let Some(h) = lags else {
        return Ok((vec![2], true));
    };
    if h.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags must be an integer horizon, not a bool (got {h}) — \
             {LAGS_HINT}"
        )));
    }
    // A NumPy bool is not a horizon either. Older NumPy releases (2.0 among
    // them) still convert it to an int through `__index__`, so catch it
    // before the integer paths to refuse it the same way on every version.
    if is_numpy_bool(h) {
        return Err(not_a_horizon(h));
    }
    // bytearray and memoryview iterate as ints like bytes (b'3' would read as
    // the horizon 51): refused like it
    if h.is_instance_of::<PyString>()
        || h.is_instance_of::<PyBytes>()
        || h.is_instance_of::<PyByteArray>()
        || h.is_instance_of::<PyMemoryView>()
    {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags must be an integer horizon or a sequence of them, not a \
             string or bytes-like object (got {}) — {LAGS_HINT}",
            h.repr().map(|r| r.to_string()).unwrap_or_default()
        )));
    }
    if let Ok(v) = h.extract::<i64>() {
        return Ok((vec![horizon_from_i64(v)?], true));
    }
    if let Ok(v) = h.extract::<u64>() {
        // Above i64::MAX: refused downstream by the length rule.
        return Ok((vec![usize::try_from(v).unwrap_or(usize::MAX)], true));
    }
    if h.is_instance_of::<PyInt>() {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags = {h} is out of range; a horizon must be an integer \
             between 2 and n - 2 — {LAGS_HINT}"
        )));
    }
    if h.is_instance_of::<PyDict>()
        || h.is_instance_of::<PySet>()
        || h.is_instance_of::<PyFrozenSet>()
    {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags = {} is a mapping or set, which has no order for the \
             per-horizon outputs to follow — {LAGS_HINT}",
            h.repr().map(|r| r.to_string()).unwrap_or_default()
        )));
    }
    // A 2-D array or a DataFrame iterates over rows or column LABELS, which
    // would pass silently as horizons when the labels are ints: only a 1-D
    // sequence is read.
    if let Ok(ndim) = h.getattr("ndim").and_then(|d| d.extract::<usize>()) {
        if ndim > 1 {
            return Err(PyValueError::new_err(format!(
                "variance_ratio: lags is a {ndim}-D {}, but horizons are a 1-D sequence \
                 (a DataFrame would be read by its column labels) — {LAGS_HINT}",
                type_name(h)
            )));
        }
    }
    // A float (even an integral one such as 4.0), a 0-d float array or a
    // Fraction is not iterable and not an int horizon: name the value and
    // its type.
    let Ok(items) = h.try_iter() else {
        return Err(not_a_horizon(h));
    };
    let cap = n.max(16);
    let mut out = Vec::new();
    for (j, item) in items.enumerate() {
        if j == cap {
            return Err(PyValueError::new_err(format!(
                "variance_ratio: lags has more than {cap} horizons, but a series of length \
                 {n} admits at most n - 3 distinct horizons (2..n-2) — {LAGS_HINT}"
            )));
        }
        out.push(one_horizon(&item?, j)?);
    }
    Ok((out, false))
}

/// Resolve the `debiased` / `robust` sentinels. Both default to True under
/// `overlap=True`; under `overlap=False` they have no effect (the
/// non-overlapping ratio has neither a debiasing correction nor a robust
/// variance), so an explicit True there is refused instead of being
/// silently ignored as arch does, and None/False resolve to False.
fn resolve_vr_flags(
    overlap: bool,
    debiased: Option<bool>,
    robust: Option<bool>,
) -> PyResult<(bool, bool)> {
    if overlap {
        return Ok((debiased.unwrap_or(true), robust.unwrap_or(true)));
    }
    if debiased == Some(true) {
        return Err(PyValueError::new_err(
            "variance_ratio: debiased=True has no effect under overlap=False: the \
             non-overlapping variance ratio uses the plain block variances (arch silently \
             ignores the flag). Use overlap=True for the Lo-MacKinlay debiased estimator, \
             or drop debiased.",
        ));
    }
    if robust == Some(true) {
        return Err(PyValueError::new_err(
            "variance_ratio: robust=True has no effect under overlap=False: the \
             non-overlapping variance ratio has only the iid variance 2(q - 1) (arch \
             silently ignores the flag). Use overlap=True for the heteroskedasticity-robust \
             statistic, or drop robust.",
        ));
    }
    Ok((false, false))
}

/// Name the offending Python parameter: the core's `InvalidLags` prints the
/// shared `nlags = ...` field name (the keyword here is `lags`), and its
/// `NonFinite` / `SeriesTooShort` messages say "input" / "series" rather
/// than `y`.
fn vr_err(e: DiagError) -> PyErr {
    match e {
        DiagError::NonFinite { index, value } => PyValueError::new_err(format!(
            "variance_ratio: y contains a non-finite value ({value}) at index {index}; \
             the test does not skip missing values silently — drop or impute NaN/inf \
             observations before testing"
        )),
        DiagError::SeriesTooShort { n, needed, .. } => PyValueError::new_err(format!(
            "variance_ratio: y has {n} observations but needs at least {needed} (and at \
             least lags + 2 for a horizon q = lags with overlap=True); pass the level \
             series"
        )),
        DiagError::InvalidLags {
            nlags,
            n,
            requirement,
            ..
        } => PyValueError::new_err(format!(
            "variance_ratio: lags = {nlags} is invalid for a series of length {n}: \
             requires {requirement}"
        )),
        other => to_py(other),
    }
}

/// Lo-MacKinlay (1988) variance-ratio test of the random-walk hypothesis,
/// with the Chow-Denning (1993) joint test over several horizons.
///
/// Tests whether the INCREMENTS of `y` are serially uncorrelated (a
/// martingale difference sequence): the variance of q-period changes is
/// compared with q times the one-period variance, `vr = sigma2_q /
/// sigma2_1`, which is 1 under the null, above 1 under positive
/// autocorrelation (momentum) and below 1 under mean reversion. It is NOT
/// a unit-root or stationarity test — `y` is assumed to have a unit root
/// either way. Pass the LEVEL series (log prices), not returns: returns
/// passed as `y` test the wrong thing (on iid returns, vr ~ 0.5 at q = 2).
/// Matches `arch.unitroot.VarianceRatio` (arch 8.0.0) in its conventions.
///
/// Arguments: `lags` (2) is one horizon q >= 2, or a sequence of distinct
/// horizons (a list, tuple, integer array or pandas Series); `None` means 2.
/// Horizons are integers: a float is refused alike as the scalar `4.0` and
/// inside a sequence (`[4.0]`, `(2, 4.0)`, a float array).
/// `trend`: "c" (random walk with drift, default; the increments are
/// centred at the sample drift) or "n" (no drift). `overlap` (True) uses
/// all overlapping q-period sums; `overlap=False` uses non-overlapping
/// blocks and drops the trailing `(n - 1) % q` observations (`n_dropped`).
/// `debiased` and `robust` are sentinels: None means True under
/// `overlap=True` (the Lo-MacKinlay unbiased variances and the
/// heteroskedasticity-robust z*, valid under ARCH); under `overlap=False`
/// neither exists, so an explicit True there **raises** (arch silently
/// ignores it) and None/False are accepted. Every horizon needs at least
/// two q-period sums (`n >= q + 2` with overlap, `n >= 2q + 1` without) —
/// arch accepts one sum and returns a meaningless ratio — and constant
/// increments (a constant series, or an exact linear trend under "c") are
/// refused.
///
/// Statistic: `stat = sqrt(nobs) (vr - 1) / sqrt(stat_variance)`, N(0, 1)
/// under the null, with `stat_variance` the robust theta, `2(2q - 1)(q -
/// 1)/(3q)` (homoskedastic), or `2(q - 1)` (non-overlapping). `pvalue` is
/// two-sided, `2 Phi(-|stat|)` computed through erfc — NOT arch's `2 - 2
/// Phi(|stat|)`, so it differs from arch by up to ~1e-9 relative at
/// moderate |stat| and stays nonzero in the deep tail where arch's
/// underflows to 0. Chow-Denning: `cd_stat = max |stat_i|` over the
/// horizons, `cd_pvalue = 1 - (1 - 2 Phi(-cd_stat))^m` and `cd_crit` the
/// studentized-maximum-modulus SMM(alpha; m, inf) = Šidák critical values
/// — conservative, because the horizons' statistics are positively
/// correlated. With one horizon it is exactly the single test.
///
/// Returned keys: `lags`, `vr`, `stat`, `pvalue`, `stat_variance`, `nobs`,
/// `n_dropped`, `crit`, `cd_stat`, `cd_lag`, `cd_pvalue`, `cd_crit`,
/// `trend`, `se_type`, `debiased`. The per-horizon keys (`lags`, `vr`,
/// `stat`, `pvalue`, `stat_variance`, `nobs` — increments used, `n - 1 -
/// n_dropped` — and `n_dropped`) are Python scalars when `lags` is an int or
/// None, and 1-D arrays aligned with the given order when it is a sequence
/// (`lags` then echoes a list). `crit` is arch's one-sided N(0, 1) quantile
/// dict {"1%", "5%", "10%", "90%", "95%", "99%"}; `cd_lag` is the horizon
/// attaining `cd_stat` (the first on a tie); `cd_crit` is {"1%", "5%",
/// "10%"}; `se_type` is "robust", "homoskedastic" or "nonoverlapping";
/// `debiased` is whether the correction was applied.
///
/// COST: the robust variance is O(n q) per horizon (every lag 1..q-1 of
/// the squared increments); the homoskedastic and non-overlapping variances
/// are O(n). With several horizons and `overlap=True` the lag products are
/// computed once, up to the longest horizon, so a whole profile costs about
/// as much as its longest horizon alone. Every horizon is checked before
/// any of that work starts. Measured on n = 20,001 levels
/// (scripts/variance_ratio_cost.py): 0.02 s at q = 1,000 and 0.18 s at
/// q = 10,000 (robust), under 1 ms with `robust=False`, and 0.05 s for all
/// 999 horizons q = 2..1,000 in one call.
///
/// Validation (fixtures/variance_ratio.json): 250 single-horizon cases
/// against arch.unitroot.VarianceRatio (vr 1e-12, stat 1e-12 + 1e-10|stat|,
/// pvalue against scipy's 2*norm.sf at 1e-9 relative with no absolute
/// floor, |z| up to 34.6 and p down to 2.1e-262), 45 Chow-Denning cases
/// against the closed form.
#[pyfunction]
#[pyo3(
    signature = (y, lags = None, trend = "c", debiased = None, robust = None, overlap = true),
    text_signature = "(y, lags=2, trend='c', debiased=None, robust=None, overlap=True)"
)]
fn variance_ratio<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    lags: Option<&Bound<'py, PyAny>>,
    trend: &str,
    debiased: Option<bool>,
    robust: Option<bool>,
    overlap: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (horizons, scalar) = parse_vr_lags(lags, y.as_array().len())?;
    if horizons.is_empty() {
        return Err(PyValueError::new_err(format!(
            "variance_ratio: lags is empty — {LAGS_HINT}"
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    for &q in &horizons {
        if !seen.insert(q) {
            return Err(PyValueError::new_err(format!(
                "variance_ratio: lags contains the horizon {q} more than once; each horizon \
                 may appear once (a repeat would count the same statistic twice in the \
                 Chow-Denning multiplicity correction)"
            )));
        }
    }
    let vr_trend = match trend {
        "c" => VrTrend::Drift,
        "n" => VrTrend::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "variance_ratio: unknown trend {other:?}; expected \"c\" (random walk with \
                 drift, the default: increments centred at the sample drift) or \"n\" (no \
                 drift). A deterministic trend in the drift (\"ct\") is not part of the \
                 variance-ratio null"
            )))
        }
    };
    let (debiased_v, robust_v) = resolve_vr_flags(overlap, debiased, robust)?;
    let spec = VarianceRatioSpec {
        trend: vr_trend,
        overlap,
        debiased: debiased_v,
        robust: robust_v,
    };
    let data = vec1(&y);
    let cd = tsecon_diag::chow_denning(&data, &horizons, spec).map_err(vr_err)?;
    let crit = tsecon_diag::variance_ratio_crit().map_err(to_py)?;

    let d = PyDict::new(py);
    let ind = &cd.individual;
    if scalar {
        let r = &ind[0];
        d.set_item("lags", r.lags)?;
        d.set_item("vr", r.vr)?;
        d.set_item("stat", r.stat)?;
        d.set_item("pvalue", r.p_value)?;
        d.set_item("stat_variance", r.stat_variance)?;
        d.set_item("nobs", r.nobs)?;
        d.set_item("n_dropped", r.n_dropped)?;
    } else {
        let col = |f: fn(&tsecon_diag::VarianceRatioResult) -> f64| -> Vec<f64> {
            ind.iter().map(f).collect()
        };
        let icol = |f: fn(&tsecon_diag::VarianceRatioResult) -> usize| -> Vec<i64> {
            ind.iter().map(|r| f(r) as i64).collect()
        };
        d.set_item("lags", ind.iter().map(|r| r.lags).collect::<Vec<_>>())?;
        d.set_item("vr", col(|r| r.vr).into_pyarray(py))?;
        d.set_item("stat", col(|r| r.stat).into_pyarray(py))?;
        d.set_item("pvalue", col(|r| r.p_value).into_pyarray(py))?;
        d.set_item("stat_variance", col(|r| r.stat_variance).into_pyarray(py))?;
        d.set_item("nobs", icol(|r| r.nobs).into_pyarray(py))?;
        d.set_item("n_dropped", icol(|r| r.n_dropped).into_pyarray(py))?;
    }
    let c = PyDict::new(py);
    c.set_item("1%", crit.pct1)?;
    c.set_item("5%", crit.pct5)?;
    c.set_item("10%", crit.pct10)?;
    c.set_item("90%", crit.pct90)?;
    c.set_item("95%", crit.pct95)?;
    c.set_item("99%", crit.pct99)?;
    d.set_item("crit", c)?;
    d.set_item("cd_stat", cd.stat)?;
    d.set_item("cd_lag", cd.lag)?;
    d.set_item("cd_pvalue", cd.p_value)?;
    let cc = PyDict::new(py);
    cc.set_item("1%", cd.crit.pct1)?;
    cc.set_item("5%", cd.crit.pct5)?;
    cc.set_item("10%", cd.crit.pct10)?;
    d.set_item("cd_crit", cc)?;
    d.set_item("trend", trend)?;
    d.set_item("se_type", ind[0].se_type.as_str())?;
    d.set_item("debiased", ind[0].debiased)?;
    Ok(d)
}

/// Adds the module's functions to the `_core` extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(variance_ratio, m)?)?;
    Ok(())
}
