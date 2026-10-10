//! Python binding for the Blanchard-Quah long-run SVAR confidence bands of
//! `tsecon-var` (`long_run_svar_bands`): the residual bootstrap that
//! re-identifies inside every draw (optional Kilian bias correction and
//! simultaneous bands) and the delta-method arm with an analytic
//! Blanchard-Quah Jacobian. Registered into `_core` through [`register`].

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tsecon_var::long_run_bands::{
    long_run_asymptotic_bands, long_run_bootstrap_bands, LongRunBandSpec,
};
use tsecon_var::tsecon_linalg::faer::Mat;

use crate::{mat_to_vec2, parse_trend, set_irf_band_items, to_py, var_band_method, var_results};

const DEFAULT_N_BOOT: usize = 1000;
const DEFAULT_SEED: u64 = 0;
const DEFAULT_BAND_SEED: u64 = 20_260_807;
const DEFAULT_BAND_N_SIM: usize = 100_000;

fn cube(c: &[Mat<f64>]) -> Vec<Vec<Vec<f64>>> {
    c.iter().map(mat_to_vec2).collect()
}

/// Extracts an optional list of `(variable, shock)` index pairs, naming
/// `name` in every refusal. A `TypeError` (a string, a list where a tuple
/// goes, a float index) passes through for PyO3 to prefix with the argument
/// name; the failures PyO3 leaves unnamed — a pair of the wrong length, and
/// a negative or beyond-64-bit index nested inside a pair — become a
/// `ValueError` that names it.
fn index_pairs_arg(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<Vec<(usize, usize)>>> {
    if obj.is_none() {
        return Ok(None);
    }
    obj.extract::<Vec<(usize, usize)>>().map(Some).map_err(|e| {
        if e.is_instance_of::<PyTypeError>(obj.py()) {
            e
        } else {
            PyValueError::new_err(format!(
                "{name} must be a list of (variable, shock) pairs of nonnegative integer \
                 indices below the number of variables, e.g. [(0, 1)]; converting it \
                 failed: {e}"
            ))
        }
    })
}

/// `from_py_with` extractor for `impact_signs=` (shared with `long_run_svar`).
pub(crate) fn impact_signs_arg(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<(usize, usize)>>> {
    index_pairs_arg(obj, "impact_signs")
}

/// `from_py_with` extractor for the long-run zero `restrictions=`.
fn restrictions_arg(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<(usize, usize)>>> {
    index_pairs_arg(obj, "restrictions")
}

/// Confidence bands for Blanchard-Quah long-run SVAR impulse responses — the
/// banded companion to `long_run_svar`, with the identification re-run inside
/// every bootstrap draw.
///
/// The structural model, `restrictions`, `normalize` and `impact_signs` are
/// exactly `long_run_svar`'s (`impact_signs=[(0, 1)]` signs the
/// Blanchard-Quah demand shock so output rises on impact). Two arms:
///
/// `method="bootstrap"` (the DEFAULT — unlike `var_irf_bands`, whose default
/// is "asymptotic") is the Runkle (1987) / Kilian (1998) residual recursive
/// design on the same engine and stream as `var_irf_bands`: replication b
/// draws iid residual rows from substream b of
/// `SeedSequence(seed).spawn(n_boot)`, rebuilds the pseudo-sample from the
/// observed first `lags` rows (no burn-in), refits the VAR and re-runs the
/// Blanchard-Quah map with the same `restrictions` / `normalize` /
/// `impact_signs`. Re-identifying is not optional: holding the impact matrix
/// at its full-sample value would drop exactly the uncertainty in the
/// long-run multiplier C(1) that dominates long-run schemes. `lower` /
/// `upper` are Efron percentiles at alpha/2 and 1-alpha/2 (NumPy "linear"),
/// `se` the across-draw SD (ddof=1); with `cumulative=True` each draw is
/// cumulated before the percentiles. `bias_correct=True` applies Kilian's
/// correction (an inner `n_boot` bootstrap with seed `seed +
/// 0x9E3779B97F4A7C15`, shrunk to stay stationary); the pseudo-samples are
/// then generated from the corrected slopes (the intercept is not corrected,
/// so the bootstrap DGP's implied mean moves), and `point` becomes the map at
/// the corrected coefficients and the full-sample df-adjusted Sigma_u — it no
/// longer equals `long_run_svar`, and it raises the estimated persistence.
/// Without it, `point` is bit-identical to `long_run_svar(...)["irf"]` (or
/// `["cumulative_irf"]`) on the same arguments.
///
/// `method="asymptotic"` is the Lütkepohl (2005) delta method with the
/// statsmodels conventions of `var_irf_bands` (Sigma_alpha = (Z'Z)^-1 kron
/// Sigma_u on the lag block, Sigma_sigma = 2 D+ (Sigma_u kron Sigma_u) D+' /
/// T, T = nobs) and an analytic Jacobian of the Blanchard-Quah map (column
/// signs held at the point estimate's); bands are `point ± z_{1-alpha/2}·se`.
///
/// COVERAGE IS NOT UNIFORM, and no arm covers every cell: read the per-cell
/// Monte Carlo table in the structural-identification model card before
/// choosing. In short, the Kilian bootstrap is the better choice for the
/// long-run matrix and on average, the delta arm for the output-level hump,
/// and plain Efron (the default) is the weakest.
///
/// Returned keys, impulse responses indexed [h][i][j] = response of variable i
/// to structural shock j at horizon h (as in `long_run_svar`): `point`
/// (cumulated when `cumulative`), `se`, `lower`, `upper`; the long-run matrix
/// LR = C(1) B as [i][j]: `long_run_point`, `long_run_se`, `long_run_lower`,
/// `long_run_upper` (the imposed zeros come out exactly ±0 with zero width
/// and `long_run_se` == 0 — a non-degenerate imposed cell would mean the
/// identification had been hoisted out of the loop); echoes `method`,
/// `alpha`, `n_boot` (None on the asymptotic arm), `bias_correct`, `band`,
/// `cumulative`; `min_root` (the full-sample VAR's smallest
/// reverse-characteristic root modulus, the `var_fit` convention: stable iff
/// > 1; it is 1 / companion spectral radius, the INVERSE of the radius that
/// `n_unstable_draws` thresholds); `n_unstable_draws` (raw bootstrap refits,
/// before any Kilian correction, whose companion spectral radius is >= 1;
/// None on the asymptotic arm); `sign_anchor_mismatches` (one count per
/// `impact_signs` entry: draws whose pre-anchor sign of impact[v][s] differs
/// from the point estimate's — a large value flags bimodal bands because the
/// anchored cell is near zero; None without `impact_signs` or on the
/// asymptotic arm). With `return_draws=True` (bootstrap only): `irf_draws`,
/// an ndarray (n_boot, horizon+1, k, k) of the REPORTED object (cumulated iff
/// `cumulative`, bias-corrected iff `bias_correct`), and `long_run_draws`
/// (n_boot, k, k). With `band` set to "sup-t", "sidak" or "bonferroni":
/// `sim_lower`, `sim_upper` (`point ± c·se`), `critical_value` (k x k),
/// `pointwise_critical_value`, `band_scope`, `n_cells` and `n_cells_used`,
/// exactly as in `var_irf_bands`; long-run cells take no part. Bootstrap
/// sup-t reads its quantile off the draws (use n_boot >= 999); asymptotic
/// sup-t simulates from `band_seed` / `band_n_sim`.
///
/// Refused, naming the parameter: `alpha` outside (0, 1); an unknown
/// `method`, `normalize`, `trend`, `band` or `band_scope`; `n_boot` < 2;
/// `lags` == 0; `band_n_sim` < 2 under asymptotic sup-t; `impact_signs` out of
/// range or anchoring a shock twice; `restrictions` that are not a column
/// permutation of the recursive pattern; `bias_correct=True` with
/// method="asymptotic"; a singular I - A_1 - ... - A_p at the full sample
/// (`data` / `lags`). Inert keywords are refused rather than ignored: `n_boot`,
/// `seed` and `return_draws` on the asymptotic arm; `band_seed` and
/// `band_n_sim` anywhere but the asymptotic sup-t band (`var_irf_bands`
/// accepts both silently on its bootstrap arm); a non-default `band_scope`
/// with band="pointwise". The working set, counted as real storage (padded
/// matrices and the Python lists included), is budgeted at 2^28 values
/// (2 GiB) BEFORE anything is allocated: the n_boot·(horizon+1) k x k draw
/// matrices on the bootstrap arm (refused naming `n_boot` and `horizon`), and
/// the joint
/// covariance of a simultaneous band's family — ((horizon+1)·k²)² under
/// band_scope="all" — on the asymptotic arm (refused naming `horizon` and
/// `band_scope`). The asymptotic arm's Jacobians cost horizon·(horizon+1)/2 ·
/// k² · lags·k² multiply-adds, QUADRATIC in horizon, and are capped at 2^28
/// (a few seconds; refused naming `horizon`). A replication whose
/// pseudo-sample cannot be refitted (a singular residual covariance, which
/// very small samples hit with positive probability because the iid resample
/// can draw too few distinct residual rows) or whose re-estimated VAR cannot
/// be identified is never dropped: the first one is reported, by index, as an
/// error.
///
/// Further arguments, with defaults: `lags` (2), `horizon` (12), `trend`
/// ("c"), `restrictions` (None), `normalize` ("long_run"), `impact_signs`
/// (None), `method` ("bootstrap"), `alpha` (0.1), `cumulative` (False),
/// `n_boot` (1000), `seed` (0), `bias_correct` (False), `band`
/// ("pointwise"), `band_scope` ("horizon"), `band_seed` (20260807),
/// `band_n_sim` (100000), `return_draws` (False).
#[pyfunction]
#[pyo3(signature = (
    data,
    lags = 2,
    horizon = 12,
    trend = "c",
    restrictions = None,
    normalize = "long_run",
    impact_signs = None,
    method = "bootstrap",
    alpha = 0.1,
    cumulative = false,
    n_boot = 1000,
    seed = 0,
    bias_correct = false,
    band = "pointwise",
    band_scope = "horizon",
    band_seed = 20260807,
    band_n_sim = 100000,
    return_draws = false,
))]
#[allow(clippy::too_many_arguments)]
fn long_run_svar_bands<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    horizon: usize,
    trend: &str,
    #[pyo3(from_py_with = restrictions_arg)] restrictions: Option<Vec<(usize, usize)>>,
    normalize: &str,
    #[pyo3(from_py_with = impact_signs_arg)] impact_signs: Option<Vec<(usize, usize)>>,
    method: &str,
    alpha: f64,
    cumulative: bool,
    n_boot: usize,
    seed: u64,
    bias_correct: bool,
    band: &str,
    band_scope: &str,
    band_seed: u64,
    band_n_sim: usize,
    return_draws: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::irf_asymptotic::{BandMethod, IrfBandScope};

    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(PyValueError::new_err(format!(
            "alpha must lie strictly in (0, 1), got {alpha} (alpha = 0.1 gives 90% bands)"
        )));
    }
    let asymptotic = match method {
        "bootstrap" => false,
        "asymptotic" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method {other:?}; expected \"bootstrap\" (the default: the \
                 residual bootstrap, re-identified in every draw) or \"asymptotic\" (the \
                 delta method)"
            )))
        }
    };
    let normalize_impact = match normalize {
        "long_run" => false,
        "impact" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown normalize {other:?}; expected \"long_run\" (positive long-run \
                 diagonal) or \"impact\" (positive impact diagonal)"
            )))
        }
    };
    let tr = parse_trend(trend)?;
    let band_method = var_band_method(band)?;
    // Parsed even at band="pointwise", so a typo is caught.
    let scope = IrfBandScope::parse(band_scope).map_err(to_py)?;
    if lags == 0 {
        return Err(PyValueError::new_err(
            "lags must be at least 1: a VAR(0) has no dynamics, hence no long-run \
             multiplier C(1) to identify from",
        ));
    }

    // Kilian's correction re-centres bootstrap draws; the delta method draws
    // none. Accepting the flag and discarding it would be a silent no-op.
    if bias_correct && asymptotic {
        return Err(PyValueError::new_err(
            "bias_correct=True has no effect with method=\"asymptotic\": Kilian's \
             correction re-centres bootstrap draws, and the delta-method arm draws \
             none. Pass method=\"bootstrap\" to get it, or drop bias_correct to keep \
             the asymptotic band.",
        ));
    }
    // Inert keywords, compared to their defaults.
    let supt_sim = asymptotic && band_method == BandMethod::SupT;
    let inert: [(&str, bool, &str); 6] = [
        (
            "n_boot",
            asymptotic && n_boot != DEFAULT_N_BOOT,
            "method=\"asymptotic\" draws nothing",
        ),
        (
            "seed",
            asymptotic && seed != DEFAULT_SEED,
            "method=\"asymptotic\" draws nothing (the sup-t simulation is keyed by band_seed)",
        ),
        (
            "return_draws",
            asymptotic && return_draws,
            "method=\"asymptotic\" has no draws to return",
        ),
        (
            "band_seed",
            !supt_sim && band_seed != DEFAULT_BAND_SEED,
            "only the asymptotic sup-t band simulates (bootstrap sup-t reads its quantile \
             off the draws, keyed by seed)",
        ),
        (
            "band_n_sim",
            !supt_sim && band_n_sim != DEFAULT_BAND_N_SIM,
            "only the asymptotic sup-t band simulates (bootstrap sup-t reads its quantile \
             off the draws, so use a larger n_boot instead)",
        ),
        (
            "band_scope",
            band_method == BandMethod::Pointwise && scope != IrfBandScope::Horizon,
            "band=\"pointwise\" has no family to be simultaneous over; pass band=\"sup-t\", \
             \"sidak\" or \"bonferroni\"",
        ),
    ];
    if let Some((name, _, why)) = inert.iter().find(|(_, hit, _)| *hit) {
        return Err(PyValueError::new_err(format!(
            "{name} is not read here and would be silently ignored: {why}. Drop {name} \
             (method={method:?}, band={band:?})."
        )));
    }

    let spec = LongRunBandSpec {
        lags,
        trend: tr,
        horizon,
        long_run_zeros: restrictions,
        normalize_impact,
        impact_signs,
        cumulative,
        alpha,
        n_boot,
        seed,
        bias_correct,
        band: if band_method.is_simultaneous() {
            Some((band_method, scope))
        } else {
            None
        },
        band_seed,
        band_n_sim,
        keep_draws: return_draws,
    };
    let r = if asymptotic {
        let res = var_results(&data, lags, trend)?;
        long_run_asymptotic_bands(&res, &spec).map_err(to_py)?
    } else {
        let a = data.as_array();
        let m = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
        long_run_bootstrap_bands(m.as_ref(), &spec).map_err(to_py)?
    };

    let d = PyDict::new(py);
    d.set_item("point", cube(&r.point))?;
    d.set_item("se", cube(&r.se))?;
    d.set_item("lower", cube(&r.lower))?;
    d.set_item("upper", cube(&r.upper))?;
    d.set_item("long_run_point", mat_to_vec2(&r.long_run_point))?;
    d.set_item("long_run_se", mat_to_vec2(&r.long_run_se))?;
    d.set_item("long_run_lower", mat_to_vec2(&r.long_run_lower))?;
    d.set_item("long_run_upper", mat_to_vec2(&r.long_run_upper))?;
    d.set_item("method", method)?;
    d.set_item("alpha", r.alpha)?;
    d.set_item("n_boot", r.n_boot)?;
    d.set_item("bias_correct", r.bias_correct)?;
    d.set_item("band", band_method.label())?;
    d.set_item("cumulative", r.cumulative)?;
    d.set_item("min_root", r.min_root)?;
    d.set_item("n_unstable_draws", r.n_unstable_draws)?;
    d.set_item("sign_anchor_mismatches", r.sign_anchor_mismatches.clone())?;
    if let (Some(cv), Some(lo), Some(hi)) = (&r.simultaneous, &r.sim_lower, &r.sim_upper) {
        set_irf_band_items(&d, cv, lo, hi)?;
    }
    if let (Some(draws), Some(lr)) = (&r.irf_draws, &r.long_run_draws) {
        let k = r.long_run_point.nrows();
        let hh = r.point.len();
        let nb = draws.len();
        let mut flat = Vec::with_capacity(nb * hh * k * k);
        for c in draws {
            for m in c {
                for i in 0..k {
                    for j in 0..k {
                        flat.push(m[(i, j)]);
                    }
                }
            }
        }
        d.set_item("irf_draws", flat.into_pyarray(py).reshape([nb, hh, k, k])?)?;
        let mut lflat = Vec::with_capacity(nb * k * k);
        for m in lr {
            for i in 0..k {
                for j in 0..k {
                    lflat.push(m[(i, j)]);
                }
            }
        }
        d.set_item(
            "long_run_draws",
            lflat.into_pyarray(py).reshape([nb, k, k])?,
        )?;
    }
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(long_run_svar_bands, m)?)?;
    Ok(())
}
