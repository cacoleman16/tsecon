//! Python bindings for the multiple-forecast-comparison slice of
//! `tsecon-forecast`: `spa_test` (White's Reality Check / Hansen's test for
//! Superior Predictive Ability), `model_confidence_set` (Hansen-Lunde-Nason)
//! and `stepm_test` (Romano-Wolf StepM). Registered into `_core` through
//! [`register`].

use numpy::ndarray::Ix2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{to_py, vec1};
use tsecon_forecast::{McsMethod, McsOptions, ResampleScheme, SpaOptions, SpaResult, StepmOptions};

fn scheme(bootstrap: &str) -> PyResult<ResampleScheme> {
    match bootstrap {
        "stationary" => Ok(ResampleScheme::Stationary),
        "circular" => Ok(ResampleScheme::CircularBlock),
        "moving_block" => Ok(ResampleScheme::MovingBlock),
        other => Err(PyValueError::new_err(format!(
            "bootstrap = {other:?} is invalid: expected \"stationary\" (Politis-Romano \
             1994, geometric block lengths — the default), \"circular\" (fixed blocks \
             with wrap-around) or \"moving_block\" (fixed blocks, no wrap-around)"
        ))),
    }
}

/// A loss panel as columns: a 1-D array is one model, a 2-D `T x m` array
/// has one column per model.
fn loss_columns(what: &str, a: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<Vec<Vec<f64>>> {
    let v = a.as_array();
    match v.ndim() {
        1 => Ok(vec![v.iter().copied().collect()]),
        2 => {
            let v2 = v.into_dimensionality::<Ix2>().map_err(to_py)?;
            Ok((0..v2.ncols())
                .map(|j| v2.column(j).iter().copied().collect())
                .collect())
        }
        d => Err(PyValueError::new_err(format!(
            "{what} must be a 1-D loss series (one model) or a 2-D T x m array with one \
             column per model, index-aligned with the benchmark; got {d} dimensions"
        ))),
    }
}

fn spa_dict<'py>(py: Python<'py>, r: &SpaResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("best_model", r.best_model)?;
    d.set_item("p_value", r.p_value_consistent)?;
    d.set_item("p_value_lower", r.p_value_lower)?;
    d.set_item("p_value_consistent", r.p_value_consistent)?;
    d.set_item("p_value_upper", r.p_value_upper)?;
    d.set_item("crit_levels", r.crit_levels.to_vec().into_pyarray(py))?;
    d.set_item("crit_lower", r.crit_lower.to_vec().into_pyarray(py))?;
    d.set_item(
        "crit_consistent",
        r.crit_consistent.to_vec().into_pyarray(py),
    )?;
    d.set_item("crit_upper", r.crit_upper.to_vec().into_pyarray(py))?;
    d.set_item("mean_loss_diff", r.mean_loss_diff.clone().into_pyarray(py))?;
    d.set_item("loss_diff_var", r.loss_diff_var.clone().into_pyarray(py))?;
    d.set_item("recentered", r.recentered.clone().into_pyarray(py))?;
    d.set_item("boot_lower", r.boot_lower.clone().into_pyarray(py))?;
    d.set_item(
        "boot_consistent",
        r.boot_consistent.clone().into_pyarray(py),
    )?;
    d.set_item("boot_upper", r.boot_upper.clone().into_pyarray(py))?;
    d.set_item("n", r.n)?;
    d.set_item("m", r.m)?;
    d.set_item("block_size", r.block_size)?;
    d.set_item("block_size_auto", r.block_size_auto)?;
    d.set_item("reps", r.reps)?;
    d.set_item("bootstrap", r.scheme.name())?;
    d.set_item("studentize", r.studentize)?;
    d.set_item("nested", r.nested)?;
    Ok(d)
}

/// White's (2000) Reality Check and Hansen's (2005) test for Superior
/// Predictive Ability: does the BEST of `m` competing models beat the
/// benchmark once the search over all of them is accounted for?
///
/// `benchmark_losses` is the benchmark's loss series over the `n`
/// evaluation periods and `model_losses` a `T x m` array with one loss
/// column per competing model (a 1-D array is one model), index-aligned —
/// e.g. squared errors from `backtest` runs under the same scheme. With
/// `d_{t,k} = benchmark_t - model_{k,t}` (positive favours the model) the
/// null is `max_k E[d_k] <= 0`, and the statistic is `sqrt(n) max_k dbar_k /
/// omega_k` (`studentize=True`, Hansen's SPA) or `sqrt(n) max_k dbar_k`
/// (`studentize=False`, White's RC), where `omega_k^2` is the
/// stationary-bootstrap long-run variance of Hansen (2005, eq. 9) with
/// restart probability `1/block_size` (or, with `nested=True`, the bootstrap
/// variance of the resampled mean over the same resamples). The null
/// distribution is a block bootstrap of the whole loss-differential panel
/// (rows resampled together), re-centred three ways: `p_value_upper`
/// re-centres every model (White's original Reality Check, conservative when
/// poor models pad the comparison), `p_value_consistent` leaves models
/// significantly worse than the benchmark — by Hansen's `sqrt(2 log log n)`
/// threshold — un-centred (the recommended p-value, also returned as
/// `p_value`), and `p_value_lower` re-centres none of the models with a
/// negative sample mean (the liberal bound); always `lower <= consistent <=
/// upper`. Each p-value is the fraction of the `reps` replicate statistics
/// above the observed one.
///
/// `block_size=None` uses the Politis-White (2004) / Patton-Politis-White
/// (2009) optimal length of each loss-differential column, averaged over the
/// columns and rounded (reported in `block_size`, with `block_size_auto`
/// True). The schemes are the library's stationary (geometric blocks, mean
/// `block_size`), circular-block and moving-block bootstraps, one Philox
/// substream per replication, bit-identical at any thread count.
///
/// Validation: `arch.bootstrap.SPA`/`RealityCheck` reproduced EXACTLY
/// (means, variances, every replicate statistic, p-values, critical values)
/// when the Rust core is fed arch's own resample indices, with
/// `studentize=False` — arch 8.0's `studentize` flag is inert (measured:
/// identical output on/off), so arch computes the un-studentized statistic
/// with Hansen's re-centrings; the studentized path is pinned at 1e-12
/// against a NumPy transcription of Hansen's formulas on the same
/// resamples. The public seeded path lands within 0.05 of arch at 4000
/// replications. Size and power are measured by seeded Monte Carlo: the
/// un-studentized rejection rates under the least favourable null match
/// arch's own on the same design, and power against a dominated benchmark
/// is 0.985 at 5%. `studentize=True` (the default, Hansen's statistic)
/// divides the observed statistic and every bootstrap replicate by the
/// SAME estimated omega_k, which over-rejects in short samples — measured
/// 0.123 at a nominal 0.05 with n=200 and AR(0.5) losses, shrinking to
/// 0.093 at n=800. No package computes that statistic, so it is measured,
/// not validated; prefer `studentize=False` in a short evaluation sample
/// (see the forecasting model card for the full tables).
///
/// Further arguments, with defaults: `block_size` (None: Politis-White),
/// `reps` (1000), `bootstrap` ("stationary"; or "circular",
/// "moving_block"), `studentize` (True), `nested` (False), `seed` (0).
///
/// Returned keys: `statistic` (sqrt(n)-scaled), `best_model` (index
/// attaining the maximum), `p_value` (= `p_value_consistent`),
/// `p_value_lower`, `p_value_consistent`, `p_value_upper`, `crit_levels`
/// ([0.90, 0.95, 0.99]), `crit_lower`, `crit_consistent`, `crit_upper`
/// (critical values at those levels, same scale as `statistic`),
/// `mean_loss_diff` (`dbar_k`), `loss_diff_var` (`omega_k^2`),
/// `recentered` (bool per model: re-centred under the consistent p-value),
/// `boot_lower`, `boot_consistent`, `boot_upper` (the `reps` replicate
/// statistics), `n`, `m`, `block_size`, `block_size_auto`, `reps`,
/// `bootstrap`, `studentize`, `nested`.
#[pyfunction]
#[pyo3(signature = (benchmark_losses, model_losses, block_size = None, reps = 1000, bootstrap = "stationary", studentize = true, nested = false, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn spa_test<'py>(
    py: Python<'py>,
    benchmark_losses: PyReadonlyArray1<'py, f64>,
    model_losses: PyReadonlyArrayDyn<'py, f64>,
    block_size: Option<usize>,
    reps: usize,
    bootstrap: &str,
    studentize: bool,
    nested: bool,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = SpaOptions {
        block_size,
        reps,
        scheme: scheme(bootstrap)?,
        studentize,
        nested,
        seed,
    };
    let bench = vec1(&benchmark_losses);
    let models = loss_columns("model_losses", &model_losses)?;
    let r = tsecon_forecast::spa_test(&bench, &models, &opts).map_err(to_py)?;
    spa_dict(py, &r)
}

/// The Hansen-Lunde-Nason (2011) Model Confidence Set: which of `m` models
/// are statistically indistinguishable from the best?
///
/// `losses` is a `T x m` array with one loss column per model (`m >= 2`),
/// index-aligned over the same evaluation periods. Starting from all
/// models, each step tests equal predictive ability across the models still
/// in the set with the range statistic `T_R = max_{i,j} |dbar_ij| /
/// sqrt(var*(dbar_ij))` (`method="R"`, HLN's recommended default) or the max
/// statistic `T_max = max_i dbar_i. / sqrt(var*(dbar_i.))` (`method="max"`),
/// against a block bootstrap of the loss panel (the same resamples reused at
/// every step, re-centred at the sample means); the worst model — the row of
/// the maximizing pair under `T_R`, every model attaining the maximum under
/// `T_max` — is eliminated with the step's p-value, until one model remains.
/// A model's MCS p-value is the running maximum of the step p-values along
/// the elimination path, so the set at any `size` is `{k : p_MCS(k) >
/// size}` (`included`) and the sets are nested in `size`; the p-values do
/// not depend on `size`. Hansen-Lunde-Nason's guarantee — the set contains
/// the best model(s) with probability at least `1 - size` — is ASYMPTOTIC
/// and about the whole best set. Measured on a design with two
/// exactly-equally-best models it holds at about 0.87 against a nominal 0.90
/// and does not improve from n=150 to n=600, matching `arch`'s own rates on
/// the same design; the easier event "a best model is in the set" does hold
/// at the nominal level. The model card has the table.
///
/// `block_size=None` uses the Politis-White optimal length of each loss
/// column, averaged and rounded (`block_size`, `block_size_auto`). Identical
/// loss columns are degenerate under `method="R"` only — their pairwise
/// bootstrap variance is exactly zero, so the panel is refused by name —
/// while `method="max"` standardizes against the cross-sectional mean, gives
/// the duplicates one statistic and eliminates them together in one step.
/// `arch` 8.0.0 handles neither: measured, its `method="R"` raises after
/// warning about the 0/0 division and its `method="max"` does not return.
///
/// Validation: `arch.bootstrap.MCS` reproduced EXACTLY (mean losses,
/// elimination order, included/excluded sets, MCS p-values, the pairwise
/// variance matrix) when fed arch's own resample indices; the public seeded
/// path reproduces arch's set and lands within 0.05 of its p-values at 4000
/// replications; coverage of the best set is measured by seeded Monte
/// Carlo and cross-checked against arch's own frequencies on the same
/// design (see the forecasting model card).
///
/// Further arguments, with defaults: `size` (0.10), `method` ("R"; or
/// "max"), `block_size` (None: Politis-White), `reps` (1000), `bootstrap`
/// ("stationary"; or "circular", "moving_block"), `seed` (0).
///
/// Returned keys: `included` (model indices in the set, ascending),
/// `excluded`, `mcs_p_values` (per model), `elimination_order` (every model
/// in the order eliminated, survivor last), `step_p_values` (the raw
/// p-value of the step that eliminated each model, aligned with
/// `elimination_order`; 1.0 for the survivor), `statistics` (observed `T_R`
/// / `T_max` per step), `n_steps`, `mean_losses`, `n`, `m`, `size`,
/// `method`, `block_size`, `block_size_auto`, `reps`, `bootstrap`.
#[pyfunction]
#[pyo3(signature = (losses, size = 0.10, method = "R", block_size = None, reps = 1000, bootstrap = "stationary", seed = 0))]
#[allow(clippy::too_many_arguments)]
fn model_confidence_set<'py>(
    py: Python<'py>,
    losses: PyReadonlyArrayDyn<'py, f64>,
    size: f64,
    method: &str,
    block_size: Option<usize>,
    reps: usize,
    bootstrap: &str,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let method = match method {
        "R" => McsMethod::Range,
        "max" => McsMethod::Max,
        other => {
            return Err(PyValueError::new_err(format!(
                "method = {other:?} is invalid: expected \"R\" (the range statistic T_R, \
                 Hansen-Lunde-Nason's recommended default) or \"max\" (the T_max statistic)"
            )))
        }
    };
    let opts = McsOptions {
        size,
        method,
        block_size,
        reps,
        scheme: scheme(bootstrap)?,
        seed,
    };
    let cols = loss_columns("losses", &losses)?;
    let r = tsecon_forecast::model_confidence_set(&cols, &opts).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("included", PyList::new(py, &r.included)?)?;
    d.set_item("excluded", PyList::new(py, &r.excluded)?)?;
    d.set_item("mcs_p_values", r.mcs_p_values.into_pyarray(py))?;
    d.set_item("elimination_order", PyList::new(py, &r.elimination_order)?)?;
    d.set_item("step_p_values", r.step_p_values.into_pyarray(py))?;
    d.set_item("statistics", r.statistics.into_pyarray(py))?;
    d.set_item("n_steps", r.n_steps)?;
    d.set_item("mean_losses", r.mean_losses.into_pyarray(py))?;
    d.set_item("n", r.n)?;
    d.set_item("m", r.m)?;
    d.set_item("size", r.size)?;
    d.set_item("method", r.method.name())?;
    d.set_item("block_size", r.block_size)?;
    d.set_item("block_size_auto", r.block_size_auto)?;
    d.set_item("reps", r.reps)?;
    d.set_item("bootstrap", r.scheme.name())?;
    Ok(d)
}

/// The Romano-Wolf (2005) StepM procedure: WHICH models beat the benchmark,
/// controlling the family-wise error rate at `size`, on the SPA bootstrap.
///
/// Arguments as `spa_test`. Step 1 declares superior every model whose
/// statistic (`sqrt(n) dbar_k / omega_k`, or un-studentized) exceeds the
/// `1 - size` quantile of the bootstrap maximum over ALL models under the
/// consistent re-centring; each later step recomputes that quantile over
/// the models not yet declared superior and adds those now exceeding it,
/// until a step adds nothing (or every model is superior — `arch` raises
/// there; this stops). The full-set SPA result is returned alongside.
/// Validation: reproduces `arch.bootstrap.StepM`'s superior set exactly on
/// arch's own resamples (`studentize=False`), the studentized rule against
/// the NumPy transcription.
///
/// Further arguments, with defaults: `size` (0.05), `block_size` (None:
/// Politis-White), `reps` (1000), `bootstrap` ("stationary"), `studentize`
/// (True), `nested` (False), `seed` (0).
///
/// Returned keys: `superior_models` (indices, ascending), `n_superior`,
/// `steps` (models declared superior at each step; the last entry is empty
/// when the procedure stopped because a step added nothing), `n_steps`,
/// `step_crit_values` (the `1 - size` bootstrap quantile compared against at
/// each step, on the `statistic` scale), `size`, plus every key of
/// `spa_test` for the full-set test: `statistic`, `best_model`, `p_value`,
/// `p_value_lower`, `p_value_consistent`, `p_value_upper`, `crit_levels`,
/// `crit_lower`, `crit_consistent`, `crit_upper`, `mean_loss_diff`,
/// `loss_diff_var`, `recentered`, `boot_lower`, `boot_consistent`,
/// `boot_upper`, `n`, `m`, `block_size`, `block_size_auto`, `reps`,
/// `bootstrap`, `studentize`, `nested`.
#[pyfunction]
#[pyo3(signature = (benchmark_losses, model_losses, size = 0.05, block_size = None, reps = 1000, bootstrap = "stationary", studentize = true, nested = false, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn stepm_test<'py>(
    py: Python<'py>,
    benchmark_losses: PyReadonlyArray1<'py, f64>,
    model_losses: PyReadonlyArrayDyn<'py, f64>,
    size: f64,
    block_size: Option<usize>,
    reps: usize,
    bootstrap: &str,
    studentize: bool,
    nested: bool,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = StepmOptions {
        size,
        spa: SpaOptions {
            block_size,
            reps,
            scheme: scheme(bootstrap)?,
            studentize,
            nested,
            seed,
        },
    };
    let bench = vec1(&benchmark_losses);
    let models = loss_columns("model_losses", &model_losses)?;
    let r = tsecon_forecast::stepm_test(&bench, &models, &opts).map_err(to_py)?;
    let d = spa_dict(py, &r.spa)?;
    d.set_item("superior_models", PyList::new(py, &r.superior_models)?)?;
    d.set_item("n_superior", r.superior_models.len())?;
    let steps = PyList::empty(py);
    for s in &r.steps {
        steps.append(PyList::new(py, s)?)?;
    }
    d.set_item("steps", steps)?;
    d.set_item("n_steps", r.steps.len())?;
    d.set_item("step_crit_values", r.step_crit_values.into_pyarray(py))?;
    d.set_item("size", r.size)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spa_test, m)?)?;
    m.add_function(wrap_pyfunction!(model_confidence_set, m)?)?;
    m.add_function(wrap_pyfunction!(stepm_test, m)?)?;
    Ok(())
}
