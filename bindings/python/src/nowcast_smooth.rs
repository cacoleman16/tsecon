//! Python binding for the fixed-parameter dynamic-factor Kalman smoother of
//! `tsecon-nowcast` — `dfm_smooth`, the vintage re-smoothing entry point
//! (`smooth_standardized`, which `dfm_nowcast`'s edge nowcast is computed
//! by). Registered into `_core` through [`register`].
//!
//! Every input is validated here first so a refusal names the keyword the
//! caller typed: the state-space engine's own messages speak of `Q`, `H`
//! and "the transition matrix", which are not arguments of this function.

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tsecon_nowcast::tsecon_ssm::tsecon_linalg::LinalgError;
use tsecon_nowcast::tsecon_ssm::SsmError;
use tsecon_nowcast::{smooth_standardized, DfmParams, NowcastError};
use tsecon_var::tsecon_linalg::faer::Mat;

use crate::{to_faer, to_py, vec1};

/// Largest accepted `T * m^2` (`m = r * p` states) and `N^2`.
///
/// The filter and smoother keep sequences of `m x m` covariance matrices,
/// one per period (predicted, filtered, smoothed, plus the returned
/// `smoothed_state_cov`), and the state-space form stores the `N x N`
/// observation covariance densely, once. `2^26` cells is half a GiB per
/// sequence or for that one matrix — the bound `unobserved_components` uses
/// for the same engine. (The engine accepts the diagonal `N x N` matrix by
/// inspection; until audit round 16 it confirmed it with a dense jittered
/// Cholesky, about six `N x N` buffers and `O(N^3)` work that no budget
/// priced: 3.1 GB and 18-20 s at `N = 8192`.)
const MAX_CELLS: u128 = 1 << 26;

/// Work budget of one smoothing pass, in units of about a nanosecond:
/// `2 m^3 + 20 N m^2 + N^2 + 1000 N + 12000` per period plus `16 m^3` once.
/// Per period the filter and smoother form a few `m x m` triple products,
/// each observed series costs an `O(m^2)` update on the way forward and
/// back, the `N^2` term prices building and checking the dense `N x N`
/// observation covariance (a one-off that the per-period term over-covers
/// for `T > 1`), and the last two terms are the per-series and per-period
/// overhead (about 1
/// and 12 microseconds); the one-off term is the stationary initialization
/// (the companion eigenvalue check and the Lyapunov doubling). The doubling
/// runs until `rho^(2^k)` drops below machine precision, so its iteration
/// count grows with the factors' persistence: about 6 doublings at a
/// spectral radius of 0.5, 22 at 0.99999 and 58 just below one, where a
/// one-period call costs about 12 ns per `m^3` (10.9 s at `m = 960`) — the
/// `16 m^3` term prices that worst case. The per-period coefficients were
/// chosen from a non-negative least-squares fit to the timings that
/// `lab/audit/wave11/dfm_smooth_cost.py` records, which also times the
/// one-off term at the edge of stationarity; the measured range per unit
/// and the resulting time at the cap are quoted in the docstring. Past the
/// budget the call is refused before any work, naming the arguments that
/// size it.
const WORK_BUDGET: u128 = 1 << 34;

fn err(msg: String) -> PyErr {
    PyValueError::new_err(format!("dfm_smooth: {msg}"))
}

/// A 2-D float argument: a NumPy array (pandas/array-likes are coerced to
/// one by the Python layer) or a nested list of rows — the shape
/// `dfm_nowcast` returns `loadings`/`factor_ar`/`factor_cov` in. A complex
/// array or cell is refused by name first (the shared real-numeric gate).
fn matrix_arg(name: &str, obj: &Bound<'_, PyAny>) -> PyResult<Mat<f64>> {
    // Before any float extraction: a complex array or cell would otherwise
    // be read through its `__float__`, which drops the imaginary part.
    if let Some(why) = crate::real_numeric::non_real_reason(obj, name, true)? {
        return Err(PyTypeError::new_err(format!(
            "dfm_smooth: {name} must hold real numbers, {why}"
        )));
    }
    if let Ok(a) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(to_faer(&a));
    }
    if let Ok(rows) = obj.extract::<Vec<Vec<f64>>>() {
        let ncols = rows.first().map_or(0, Vec::len);
        if let Some((i, row)) = rows.iter().enumerate().find(|(_, r)| r.len() != ncols) {
            return Err(err(format!(
                "{name} is a ragged nested list: row 0 has {ncols} entries but row {i} has {}; \
                 pass a rectangular 2-D array",
                row.len()
            )));
        }
        return Ok(Mat::from_fn(rows.len(), ncols, |i, j| rows[i][j]));
    }
    if obj.extract::<PyReadonlyArray1<'_, f64>>().is_ok() {
        return Err(err(format!(
            "{name} is 1-D but must be a 2-D array (rows x columns); for a single factor \
             pass it as a column or a single row, e.g. np.reshape({name}, (-1, 1)) for \
             loadings or np.reshape({name}, (1, -1)) for factor_ar"
        )));
    }
    Err(err(format!(
        "{name} must be a 2-D float array or a nested list of rows (got {})",
        obj.get_type().name()?
    )))
}

fn check_finite_mat(name: &str, m: &Mat<f64>) -> PyResult<()> {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            let v = m[(i, j)];
            if !v.is_finite() {
                return Err(err(format!(
                    "{name} contains a non-finite value ({v}) at [{i}, {j}]; model parameters \
                     must be finite"
                )));
            }
        }
    }
    Ok(())
}

/// Rewrites the engine's errors in the caller's vocabulary.
fn rewrite(e: NowcastError) -> PyErr {
    match &e {
        NowcastError::Ssm(SsmError::NotPsd { what }) if *what == "Q" => err(
            "factor_cov is not a symmetric positive semidefinite matrix; it is the covariance \
             of the factor innovations (statsmodels' DynamicFactor fixes it to the identity, \
             which is also what factor_cov=None gives)"
                .to_owned(),
        ),
        NowcastError::Ssm(SsmError::NotPsd { what }) if *what == "H" => err(
            "idiosyncratic must hold non-negative variances (the diagonal of the observation \
             covariance)"
                .to_owned(),
        ),
        NowcastError::Ssm(SsmError::Linalg(LinalgError::Unstable { spectral_radius })) => {
            err(format!(
                "factor_ar describes a non-stationary factor VAR: the companion matrix of \
                 [A_1 | ... | A_p] has spectral radius {spectral_radius} >= 1, so the factors \
                 have no stationary distribution to initialize the Kalman filter from. Pass \
                 stationary coefficients (every root of det(I - A_1 z - ... - A_p z^p) \
                 outside the unit circle)"
            ))
        }
        NowcastError::Ssm(SsmError::NoInformation) => err(
            "no observed element of data carries usable information about the factors: \
             for every observed series the prediction variance loadings_i P loadings_i' + \
             idiosyncratic_i is either zero to roundoff (a zero row of loadings with a \
             zero idiosyncratic variance) or not finite (loadings, factor_cov or \
             idiosyncratic so large that it overflows double precision), so the \
             log-likelihood is an empty sum. An all-NaN data panel is not refused: it \
             returns loglik 0.0 and the unconditional moments"
                .to_owned(),
        ),
        // Overflow inside the recursions: the stationary factor covariance,
        // the state covariances and the prediction variances are products of
        // the parameters, so an extreme magnitude surfaces here as the
        // engine's "non-finite value in m" — which reads like missing data.
        NowcastError::Ssm(
            SsmError::NonFinite { .. } | SsmError::Linalg(LinalgError::NonFinite { .. }),
        ) => err(
            "the Kalman recursion overflowed to a non-finite value: loadings, factor_cov \
             or idiosyncratic hold magnitudes too large for double precision (the \
             stationary factor covariance and every prediction variance are products of \
             them). Rescale the model; on the standardized scale dfm_nowcast estimates it \
             on, these are of order one"
                .to_owned(),
        ),
        _ => to_py(e),
    }
}

/// Fixed-parameter dynamic-factor Kalman smoother: runs the Kalman
/// filter/smoother of a DFM at GIVEN parameters over a (possibly ragged)
/// `T x N` panel — no estimation. The vintage re-smoothing workflow: fit
/// once with `dfm_nowcast`, then re-smooth each new data release with the
/// fitted model,
/// `dfm_smooth(new_data, loadings=fit["loadings"], factor_ar=fit["factor_ar"],
/// idiosyncratic=fit["idiosyncratic"], factor_cov=fit["factor_cov"],
/// center=fit["center"], scale=fit["scale"])`, which reproduces
/// `dfm_nowcast`'s `loglik`, `edge_factor` and `nowcast` on the same panel
/// bit for bit (`dfm_nowcast`'s edge nowcast is computed by this same
/// routine).
///
/// Model (`r` factors following a VAR(`p`), white-noise idiosyncratic
/// errors), on the standardized panel `z = (data - center) / scale`:
/// `z_t = loadings @ f_t + e_t`, `e_t ~ N(0, diag(idiosyncratic))`;
/// `f_t = A_1 f_{t-1} + ... + A_p f_{t-p} + eta_t`, `eta_t ~ N(0,
/// factor_cov)`. State `[f_t', ..., f_{t-p+1}']'` (`m = r p`), stationary
/// (Lyapunov) initialization — exactly statsmodels'
/// `DynamicFactor(k_factors=r, factor_order=p, error_order=0)`, which fixes
/// `factor_cov` to the identity. `NaN` in `data` marks a missing element
/// and is skipped (the ragged edge, or any gap).
///
/// Inputs: `data` (`T x N`); `loadings` (`N x r`); `factor_ar` (`r x r p`,
/// the stacked `[A_1 | ... | A_p]`: `factor_ar[i][k*r + j]` is the effect of
/// factor `j` at lag `k + 1` on factor `i`); `idiosyncratic` (length `N`,
/// variances `>= 0`); `factor_cov` (`r x r` symmetric PSD; None = identity);
/// `center` (length `N`; None = zeros) and `scale` (length `N`, `> 0`;
/// None = ones). The matrices may be NumPy arrays or nested lists — the
/// forms `dfm_nowcast` returns them in.
///
/// Returned keys: `loglik` (the Gaussian prediction-error-decomposition
/// log-likelihood of the STANDARDIZED panel `z` — no Jacobian term for the
/// standardization; with the default `center`/`scale` it is that of `data`
/// itself), `smoothed_factors` (`(T, r)` ndarray over the FULL panel: row
/// `t` is `E[f_t | data]`. Not the same object as `dfm_nowcast`'s
/// `smoothed_factors`, which is a `(T_b, r)` nested list from the balanced
/// TRAINING block only), `smoothed_state` (`(T, r p)` ndarray, the whole
/// lag-stacked state; its first `r` columns are `smoothed_factors`),
/// `smoothed_state_cov` (`(T, r p, r p)` ndarray, `Var[state_t | data]`;
/// its leading `r x r` block is the factors' posterior covariance),
/// `edge_factor` (length `r`, the smoothed factor at the last period),
/// `nowcast` (length `N`, `center + scale * (loadings @ edge_factor)` — the
/// model's level of every series at the last period, observed or not),
/// `n_factors` (`r`) and `factor_order` (`p`).
///
/// ZERO IDIOSYNCRATIC VARIANCES. `idiosyncratic >= 0` admits exact zeros.
/// When more than `r` series with zero idiosyncratic variance are observed
/// in the same period, the first ones pin the factors and the rest are
/// deterministic functions of them: the joint density is singular, and
/// `loglik` is the density on its support — an element an earlier series of
/// the same period has pinned is skipped exactly like a missing value (the
/// value with that cell set to NaN). Which series pins depends on the
/// column order of `data`; two orders differ by the Jacobian
/// `ln |loading_j / loading_i|` per such period. (statsmodels takes the
/// roundoff variance of the redundant element as information there.)
///
/// Refusals name the argument: shapes (`data`'s columns vs `loadings`' rows,
/// `factor_ar` rows vs `r` and width a multiple of `r`, `factor_cov`
/// `r x r`, lengths of `idiosyncratic`/`center`/`scale`); complex entries in
/// `loadings`/`factor_ar`/`factor_cov` (refused, not truncated to their real
/// part); non-finite parameters; `inf` in `data` (NaN is missing, not an
/// error); zero rows;
/// negative `idiosyncratic`; `scale <= 0`; a non-stationary `factor_ar`; a
/// non-PSD `factor_cov`; and finite inputs that overflow double precision —
/// a standardized entry `(data - center) / scale` that is not finite (named
/// by cell), a log-likelihood that overflows to `-inf` (the standardized
/// data far off the model's scale), or parameters so large that the Kalman
/// recursion overflows. COST: the smoother stores `T * (r p)^2` covariance
/// cells per sequence and the state space an `N x N` observation
/// covariance, each capped at 2^26; the work is about
/// `T (2 m^3 + 20 N m^2 + N^2 + 1000 N + 12000) + 16 m^3` units with
/// `m = r p` (per period the triple products, the per-series updates,
/// building the dense observation covariance — a one-off the per-period
/// term over-covers — and the per-series / per-period overhead; once, the
/// stationary initialization, priced at its worst case: a factor VAR at the
/// edge of stationarity, where the Lyapunov doubling runs about 58 steps),
/// capped at 2^34 — measured at 0.14-1.07 ns per unit across nineteen
/// shapes from 3% of the cap up, so a call at the cap takes about 18 s at
/// most (17.0 s at N = 8192, m = 60, 0.97 of the cap; 10.1 s at T = 8000,
/// N = 1000, m = 2; 10.9 s for one period at m = 960 with the spectral
/// radius the largest double below one, 0.93 of the cap; one period at
/// N = 8192, 0.4% of the cap, takes 1.0 s, building the dense N x N
/// matrix;
/// `lab/audit/wave11/dfm_smooth_cost.py` records the timings, the
/// nowcasting model card quotes them). Past either cap the call is refused
/// before any work.
///
/// Validated against statsmodels 0.15.0 `DynamicFactor.smooth(params)` (two
/// factors, VAR(2), ragged edge) and its generic `KalmanSmoother` (a
/// non-identity `factor_cov`; a standardized panel with its edge nowcast):
/// log-likelihood 1e-8 relative, states and covariances 1e-8 absolute
/// (achieved over the three: log-likelihood 5.2e-13, states 2.6e-10 and
/// covariances 1.2e-10, the latter two on the standardized panel; 6.0e-11
/// and 1.7e-11 on the raw ones); zero idiosyncratic variances observed
/// together against statsmodels on the panel with the redundant column
/// missing and an exact rational / 80-digit Kalman filter (1e-12) —
/// `fixtures/dfm_smooth.json`.
///
/// Further arguments, with defaults: `factor_cov` (None: identity), `center`
/// (None: zeros), `scale` (None: ones).
#[pyfunction]
#[pyo3(signature = (data, loadings, factor_ar, idiosyncratic, factor_cov = None, center = None, scale = None))]
#[allow(clippy::too_many_arguments)]
fn dfm_smooth<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    loadings: Bound<'py, PyAny>,
    factor_ar: Bound<'py, PyAny>,
    idiosyncratic: PyReadonlyArray1<'py, f64>,
    factor_cov: Option<Bound<'py, PyAny>>,
    center: Option<PyReadonlyArray1<'py, f64>>,
    scale: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let loadings = matrix_arg("loadings", &loadings)?;
    let (n, r) = (loadings.nrows(), loadings.ncols());
    if n == 0 || r == 0 {
        return Err(err(format!(
            "loadings has shape ({n}, {r}) but must be N x r with at least one series (row) \
             and one factor (column)"
        )));
    }
    check_finite_mat("loadings", &loadings)?;

    let factor_ar = matrix_arg("factor_ar", &factor_ar)?;
    if factor_ar.nrows() != r {
        return Err(err(format!(
            "factor_ar has {} row(s) but loadings has r = {r} factor column(s); factor_ar \
             is the r x (r p) stacked [A_1 | ... | A_p], one row per factor",
            factor_ar.nrows()
        )));
    }
    let width = factor_ar.ncols();
    if width == 0 || width % r != 0 {
        return Err(err(format!(
            "factor_ar has {width} column(s), which is not a positive multiple of r = {r}: \
             it must be the r x (r p) stacked [A_1 | ... | A_p] for a factor VAR of order \
             p >= 1"
        )));
    }
    let p = width / r;
    check_finite_mat("factor_ar", &factor_ar)?;

    let factor_cov = match factor_cov {
        None => Mat::from_fn(r, r, |i, j| if i == j { 1.0 } else { 0.0 }),
        Some(obj) => {
            let q = matrix_arg("factor_cov", &obj)?;
            if q.nrows() != r || q.ncols() != r {
                return Err(err(format!(
                    "factor_cov has shape ({}, {}) but must be r x r = ({r}, {r}), the \
                     covariance of the r factor innovations",
                    q.nrows(),
                    q.ncols()
                )));
            }
            check_finite_mat("factor_cov", &q)?;
            q
        }
    };

    let idio = vec1(&idiosyncratic);
    if idio.len() != n {
        return Err(err(format!(
            "idiosyncratic has {} entries but loadings has N = {n} rows (series); pass one \
             variance per series",
            idio.len()
        )));
    }
    if let Some((i, v)) = idio
        .iter()
        .enumerate()
        .find(|(_, v)| !(v.is_finite() && **v >= 0.0))
    {
        return Err(err(format!(
            "idiosyncratic[{i}] = {v}; the idiosyncratic variances must be finite and \
             non-negative"
        )));
    }

    let moments =
        |name: &str, v: Option<PyReadonlyArray1<'py, f64>>, default: f64| -> PyResult<Vec<f64>> {
            match v {
                None => Ok(vec![default; n]),
                Some(a) => {
                    let v = vec1(&a);
                    if v.len() != n {
                        return Err(err(format!(
                            "{name} has {} entries but loadings has N = {n} rows (series); pass \
                         one value per series",
                            v.len()
                        )));
                    }
                    Ok(v)
                }
            }
        };
    let center = moments("center", center, 0.0)?;
    let scale = moments("scale", scale, 1.0)?;
    if let Some((i, v)) = center.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(err(format!(
            "center[{i}] = {v}; the centering values must be finite"
        )));
    }
    if let Some((i, v)) = scale
        .iter()
        .enumerate()
        .find(|(_, v)| !(v.is_finite() && **v > 0.0))
    {
        return Err(err(format!(
            "scale[{i}] = {v}; the scales must be finite and strictly positive (the panel \
             is standardized as (data - center) / scale)"
        )));
    }

    let arr = data.as_array();
    let (t_len, n_data) = (arr.nrows(), arr.ncols());
    if t_len == 0 {
        return Err(err(
            "data has zero rows; pass a T x N panel with at least one period".to_owned(),
        ));
    }
    if n_data != n {
        return Err(err(format!(
            "data has {n_data} column(s) but loadings has N = {n} row(s); data is T x N with \
             one column per series, in the order of the loadings' rows"
        )));
    }
    // The smoother sees z = (data - center) / scale: refuse an infinite
    // entry, and one whose standardization overflows (the engine would
    // report that as "non-finite value in panel data", which the caller's
    // finite data does not hold), naming the cell and the moments that
    // produce it. The largest |z| is kept for the log-likelihood check below.
    let mut z_max = (0.0f64, 0usize, 0usize);
    for ((i, j), v) in arr.indexed_iter() {
        if v.is_infinite() {
            return Err(err(format!(
                "data contains an infinite value ({v}) at [{i}, {j}]; NaN marks a missing \
                 observation, infinity is not admissible"
            )));
        }
        if v.is_nan() {
            continue;
        }
        let z = (v - center[j]) / scale[j];
        if !z.is_finite() {
            return Err(err(format!(
                "data[{i}, {j}] = {v:e} standardizes to (data - center) / scale = {z} with \
                 center[{j}] = {c:e} and scale[{j}] = {s:e}, which overflows double \
                 precision; pass a scale on the order of the series' spread",
                c = center[j],
                s = scale[j]
            )));
        }
        if z.abs() > z_max.0 {
            z_max = (z.abs(), i, j);
        }
    }

    // Size guards, before anything of that size is built.
    let m = (r as u128) * (p as u128);
    let cells = (t_len as u128) * m * m;
    if cells > MAX_CELLS {
        return Err(err(format!(
            "the state has m = r * p = {r} * {p} = {m} entries over T = {t_len} periods: the \
             smoother stores sequences of T * m^2 = {cells} covariance cells, beyond the \
             {MAX_CELLS} accepted. Lower the factor order (the width of factor_ar) or the \
             number of factors, or smooth a shorter span of data"
        )));
    }
    if (n as u128) * (n as u128) > MAX_CELLS {
        return Err(err(format!(
            "loadings has N = {n} series: the state-space form stores the N x N observation \
             covariance densely ({} cells, beyond the {MAX_CELLS} accepted). Smooth at most \
             8192 series at a time",
            (n as u128) * (n as u128)
        )));
    }
    let n128 = n as u128;
    let per_period = 2 * m * m * m + 20 * n128 * m * m + n128 * n128 + 1000 * n128 + 12_000;
    let work = (t_len as u128) * per_period + 16 * m * m * m;
    if work > WORK_BUDGET {
        return Err(err(format!(
            "T = {t_len} periods, N = {n} series (the rows of loadings) and m = r * p = {m} \
             states (factor_ar's width) cost about T * (2 m^3 + 20 N m^2 + N^2 + 1000 N + \
             12000) + 16 m^3 = {work:.2e} work units (about a nanosecond each), beyond the \
             budget of 2^34 (under half a minute). Lower the factor order or the number of \
             factors, or smooth fewer periods or series",
            work = work as f64
        )));
    }

    let full = to_faer(&data);
    let params = DfmParams {
        loadings,
        factor_ar,
        factor_cov,
        idiosyncratic: idio,
    };
    let res = smooth_standardized(&params, full.as_ref(), &center, &scale).map_err(rewrite)?;
    let sm = res.smoothing;
    // v^2 / F overflows when the standardized data are extreme next to the
    // model's prediction variances; the engine then sums a -inf rather than
    // refusing. A log-likelihood that is not a number is not a result.
    if !sm.loglik.is_finite() {
        return Err(err(format!(
            "the log-likelihood overflowed to {ll}: the standardized panel (data - center) \
             / scale reaches |z| = {z:e} (at data[{i}, {j}]), too large next to the model's \
             prediction variances for v^2 / F to stay finite in double precision. Pass \
             center and scale on the data's own location and spread (as dfm_nowcast's are), \
             or rescale loadings and idiosyncratic to the data",
            ll = sm.loglik,
            z = z_max.0,
            i = z_max.1,
            j = z_max.2
        )));
    }
    let m = r * p;
    let flat_factors: Vec<f64> = sm.smoothed_factors.iter().flatten().copied().collect();
    let flat_state: Vec<f64> = sm.smoothed_state.iter().flatten().copied().collect();
    let mut flat_cov = Vec::with_capacity(t_len * m * m);
    for c in &sm.smoothed_state_cov {
        for i in 0..m {
            for j in 0..m {
                flat_cov.push(c[(i, j)]);
            }
        }
    }
    let d = PyDict::new(py);
    d.set_item("loglik", sm.loglik)?;
    d.set_item(
        "smoothed_factors",
        flat_factors.into_pyarray(py).reshape([t_len, r])?,
    )?;
    d.set_item(
        "smoothed_state",
        flat_state.into_pyarray(py).reshape([t_len, m])?,
    )?;
    d.set_item(
        "smoothed_state_cov",
        flat_cov.into_pyarray(py).reshape([t_len, m, m])?,
    )?;
    d.set_item("edge_factor", res.edge_factor.into_pyarray(py))?;
    d.set_item("nowcast", res.values.into_pyarray(py))?;
    d.set_item("n_factors", r)?;
    d.set_item("factor_order", p)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dfm_smooth, m)?)?;
    Ok(())
}
