//! Neural bindings (tsecon-ml, roadmap Module 10 Tier 2 "Neural" and the
//! contrib-tier echo state network): `mlp_regression` and
//! `echo_state_network`.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::{to_faer, to_py, vec1};

/// `hidden[i]`: a non-negative Python or NumPy integer, or an integral float
/// (a float array of widths reads as floats). A width at or beyond 2**48 is
/// refused here by name, before the core's budget, so a float such as 1e300
/// is quoted as given rather than as a saturated cast.
fn one_width(item: &Bound<'_, PyAny>, i: usize) -> PyResult<usize> {
    let shown = item
        .repr()
        .map(|r| r.to_string())
        .unwrap_or_else(|_| "?".to_string());
    if let Ok(v) = item.extract::<usize>() {
        return Ok(v);
    }
    if let Ok(v) = item.extract::<f64>() {
        if v.is_finite() && v >= 0.0 && v.fract() == 0.0 {
            if v < ABSURD_WIDTH {
                return Ok(v as usize);
            }
            return Err(PyValueError::new_err(format!(
                "mlp_regression: hidden[{i}] = {shown} is at or beyond 2**48 units, a layer \
                 no machine can hold; pass widths that fit the data, e.g. hidden=(16,) or \
                 hidden=[32, 16]"
            )));
        }
    }
    Err(PyValueError::new_err(format!(
        "hidden must list positive integer layer widths; hidden[{i}] = {shown} is not a \
         non-negative integer. Pass e.g. hidden=(16,) or hidden=[32, 16]"
    )))
}

/// 2**48: the house bound beyond which a count cannot be allocated at all.
const ABSURD_WIDTH: f64 = 281_474_976_710_656.0;

/// Longest `hidden` sequence read element by element; the core names the
/// depth limit for anything from 3 up to here.
const MAX_LISTED_LAYERS: usize = 64;

/// `hidden` accepts a bare int (one layer) or a tuple, list or 1-D NumPy
/// array of widths (ints, or integral floats). It is in the coercion
/// layer's `_EXEMPT` table, so it reaches this parser as given: a list
/// stays a list of Python ints (the 2**48 pre-flight sees it) and an
/// integer array keeps its dtype. Each bad element is refused as
/// `hidden[i]`, and the core then budgets the network the widths imply.
/// Anything else is a teaching error.
fn parse_hidden(hidden: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<usize>> {
    let Some(h) = hidden else {
        return Ok(vec![16]);
    };
    if let Ok(v) = h.extract::<usize>() {
        return Ok(vec![v]);
    }
    // A NumPy array is read through its Python-scalar list.
    let listed;
    let h = if h.hasattr("dtype")? && h.hasattr("tolist")? {
        listed = h.call_method0("tolist")?;
        &listed
    } else {
        h
    };
    if h.is_instance_of::<PyList>() || h.is_instance_of::<PyTuple>() {
        let len = h.len()?;
        if len > MAX_LISTED_LAYERS {
            // The core refuses any net deeper than two hidden layers; do not
            // copy an absurdly long sequence element by element to say so.
            return Err(PyValueError::new_err(format!(
                "invalid argument: hidden lists {len} layers but this native net is \
                 limited to 2 hidden layers by design; pass one or two widths, e.g. \
                 hidden=(32, 16)"
            )));
        }
        return h
            .try_iter()?
            .enumerate()
            .map(|(i, e)| one_width(&e?, i))
            .collect();
    }
    Err(PyValueError::new_err(format!(
        "hidden must be a tuple or list of positive integers (one or two hidden-layer \
         widths), e.g. hidden=(16,) or hidden=[32, 16]; got {}",
        h.get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_default()
    )))
}

fn mat_to_py<'py>(
    py: Python<'py>,
    m: &tsecon_ml::faer::Mat<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows: Vec<Vec<f64>> = (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect();
    PyArray2::from_vec2(py, &rows).map_err(to_py)
}

/// Feed-forward neural regressor with a seed ensemble, early stopping on a
/// TEMPORAL validation split, and scikit-learn's exact objective — the "NN"
/// entry of the macro-forecasting horse races (Medeiros et al. 2021; Goulet
/// Coulombe et al. 2022), and the only neural net implemented natively in
/// tsecon (torch / N-BEATS / foundation-model adapters are out of core by
/// scope ruling; this adds no framework dependency).
///
/// Model: one or two hidden layers of widths `hidden` (a tuple or list of
/// ints, default `(16,)`; an int is taken as one layer; at most two
/// layers by design), hidden `activation` in {"tanh", "relu", "logistic"},
/// identity output. Objective, exactly scikit-learn MLPRegressor's:
/// `(1/(2n)) sum (y - f(x))^2 + (alpha/(2n)) sum_l ||W_l||_F^2`,
/// intercepts unpenalized (`alpha` on sklearn's scale, default 1e-4).
///
/// Training. `solver="adam"` (default): Adam with sklearn's constants,
/// `learning_rate` (None -> 1e-3), `max_epochs` (500), `batch_size` (None
/// -> one full-batch step per epoch; an int -> seeded shuffled mini-batches
/// every epoch), and early stopping: the LAST
/// `floor(validation_fraction * n)` rows (default 0.2; 0 disables early
/// stopping; at most 0.5) are held out — never a random split, which on a
/// time series leaks the future into the training set — and training
/// stops once the validation loss (`0.5 * mean((y - yhat)^2)` on those
/// rows) has failed to improve on its best by a relative 1e-4 for
/// `patience` (None -> 20) consecutive epochs; the best epoch's weights
/// are kept. `solver="lbfgs"` minimizes the full training objective with
/// tsecon's L-BFGS (strong-Wolfe line search, analytic gradient;
/// `max_epochs` caps its iterations); it has no learning rate, batches, or
/// patience, so passing `learning_rate`, `batch_size`, or `patience`
/// explicitly under lbfgs RAISES rather than being silently ignored
/// (leave them None). `standardize=True` fits column means/scales of `x`
/// and the mean/scale of `y` on the TRAINING rows only and replays them
/// on the validation rows and on `x_test` (a test perturbs the validation
/// rows and checks the scaler is bit-identical). `n_seeds` (5) members
/// are trained from independent Philox substreams of `seed` (0) —
/// Glorot-uniform initializations and, under mini-batches, shuffles —
/// and averaged. `x_test` (optional, `n_test x p`) is predicted by every
/// member.
///
/// Returns a dict: `fitted` (ensemble-mean prediction for every row of
/// `x`, training and validation rows alike, on the original y scale),
/// `predicted` (ensemble mean on `x_test`, else None),
/// `member_predictions` (`n_seeds x n_test` array, else None),
/// `train_loss_path` and `validation_loss_path` (lists, one array per
/// member: the objective / the validation loss per epoch — under lbfgs
/// two entries, at the initial and the returned weights; the validation
/// path is empty when validation_fraction=0), `best_epoch` (per member,
/// 1-based; the iteration count under lbfgs; the epochs run with no
/// validation split), `converged` (per member: True if early stopping
/// fired, or L-BFGS met its convergence test; False if the member ran
/// out of max_epochs), `n_parameters`, `weights` (per member a dict
/// {"coefs": [fan_in x fan_out arrays], "intercepts": [arrays]} in
/// scikit-learn's layout, on the standardized scale when standardizing),
/// `n_train`, `n_validation`, `x_mean`, `x_scale`, `y_mean`, `y_scale`
/// (the training-row scaler; identity when standardize=False), `solver`,
/// `activation`. Keys: fitted, predicted, member_predictions,
/// train_loss_path, validation_loss_path, best_epoch, converged,
/// n_parameters, weights, n_train, n_validation, x_mean, x_scale,
/// y_mean, y_scale, solver, activation.
///
/// Validation (fixtures/neural.json, independent package: scikit-learn
/// 1.9.0 MLPRegressor): the forward pass reproduces sklearn `predict` at
/// its fitted weights (1e-12), the objective equals the sklearn-convention
/// loss (1e-10), the analytic gradient equals sklearn's own backprop at
/// random and fitted weights (1e-10) and a central finite difference
/// (1e-6 relative), and the gradient norm at sklearn's converged L-BFGS
/// weights reproduces the norm measured there (1e-8). The optimizer
/// trajectory is deliberately NOT pinned (no two Adam/L-BFGS runs share
/// one). The estimator itself is property/Monte-Carlo graded: recovers
/// y_t = sin(2 y_{t-1}) + e_t out of sample (R^2 0.94 mini-batch Adam,
/// 0.95 lbfgs, 0.80 all-defaults, vs 0.75 linear), the ensemble beats the
/// mean member in every replication (Jensen) and the median member in a
/// majority (7/10 Rust draws, 10/10 NumPy draws) on a documented
/// overfitting DGP, early stopping fires on an easy problem and cannot at
/// max_epochs=1. Reproducibility: single-threaded Rust, every
/// draw a pure function of `seed` — bit-identical on the same build;
/// across platforms the libm tanh/exp may differ in the last ulp, so the
/// cross-platform promise is statistical (seed-ensemble) reproducibility.
/// Errors name the array (NaN/inf in x, y, x_test), list the accepted
/// activation/solver names, name the hidden-layer limit, and report
/// `insufficient data: {got} observations, at least {needed} required`
/// counting the validation split.
///
/// `hidden` may also be a 1-D array of widths. A layer width, an
/// `n_seeds`, or (under Adam with `validation_fraction=0`, when every
/// epoch runs and is recorded) a `max_epochs` whose working set exceeds
/// the 2**28-value (2 GiB) memory budget is refused by name (`hidden[i]`,
/// `n_seeds`, `max_epochs`), with the largest value that fits, before
/// anything is allocated; with a validation split a large `max_epochs`
/// simply runs until early stopping.
#[pyfunction]
#[pyo3(signature = (x, y, hidden = None, activation = "tanh", alpha = 1e-4, solver = "adam",
                    learning_rate = None, batch_size = None, max_epochs = 500,
                    validation_fraction = 0.2, patience = None, n_seeds = 5, seed = 0,
                    standardize = true, x_test = None))]
#[allow(clippy::too_many_arguments)]
fn mlp_regression<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    hidden: Option<&Bound<'py, PyAny>>,
    activation: &str,
    alpha: f64,
    solver: &str,
    learning_rate: Option<f64>,
    batch_size: Option<usize>,
    max_epochs: usize,
    validation_fraction: f64,
    patience: Option<usize>,
    n_seeds: usize,
    seed: u64,
    standardize: bool,
    x_test: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = tsecon_ml::MlpOptions {
        hidden: parse_hidden(hidden)?,
        activation: tsecon_ml::Activation::parse(activation).map_err(to_py)?,
        alpha,
        solver: tsecon_ml::Solver::parse(solver).map_err(to_py)?,
        learning_rate,
        batch_size,
        max_epochs,
        validation_fraction,
        patience,
        n_seeds,
        seed,
        standardize,
    };
    let xm = to_faer(&x);
    let xt = x_test.as_ref().map(to_faer);
    let r = tsecon_ml::mlp_regression(
        xm.as_ref(),
        &vec1(&y),
        xt.as_ref().map(|m| m.as_ref()),
        &opts,
    )
    .map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("fitted", r.fitted.into_pyarray(py))?;
    match r.predicted {
        Some(p) => d.set_item("predicted", p.into_pyarray(py))?,
        None => d.set_item("predicted", py.None())?,
    }
    match r.member_predictions {
        Some(mp) => d.set_item(
            "member_predictions",
            PyArray2::from_vec2(py, &mp).map_err(to_py)?,
        )?,
        None => d.set_item("member_predictions", py.None())?,
    }
    let train_paths = PyList::empty(py);
    for p in r.train_loss_path {
        train_paths.append(p.into_pyarray(py))?;
    }
    d.set_item("train_loss_path", train_paths)?;
    let val_paths = PyList::empty(py);
    for p in r.validation_loss_path {
        val_paths.append(p.into_pyarray(py))?;
    }
    d.set_item("validation_loss_path", val_paths)?;
    d.set_item("best_epoch", r.best_epoch)?;
    d.set_item("converged", r.converged)?;
    d.set_item("n_parameters", r.n_parameters)?;
    let weights = PyList::empty(py);
    for w in &r.weights {
        let wd = PyDict::new(py);
        let coefs = PyList::empty(py);
        for c in &w.coefs {
            coefs.append(mat_to_py(py, c)?)?;
        }
        let intercepts = PyList::empty(py);
        for b in &w.intercepts {
            intercepts.append(b.clone().into_pyarray(py))?;
        }
        wd.set_item("coefs", coefs)?;
        wd.set_item("intercepts", intercepts)?;
        weights.append(wd)?;
    }
    d.set_item("weights", weights)?;
    d.set_item("n_train", r.n_train)?;
    d.set_item("n_validation", r.n_validation)?;
    d.set_item("x_mean", r.x_mean.into_pyarray(py))?;
    d.set_item("x_scale", r.x_scale.into_pyarray(py))?;
    d.set_item("y_mean", r.y_mean)?;
    d.set_item("y_scale", r.y_scale)?;
    d.set_item("solver", opts.solver.name())?;
    d.set_item("activation", opts.activation.name())?;
    Ok(d)
}

/// Echo state network (reservoir computing; Jaeger 2001; Lukosevicius
/// 2012): a fixed sparse random recurrent reservoir, a leaky-integrator
/// tanh state recursion, and a ridge-trained linear readout — nonlinear
/// dynamics at linear-regression cost, deterministic given `seed`.
///
/// Model: with inputs `u_t` (the rows of `x`, `n x p`), `N =
/// reservoir_size` (200) units, `W_in` (`N x p`, uniform on
/// `[-input_scaling, input_scaling]`, default 1.0) and `W` (`N x N`; each
/// entry nonzero with probability `sparsity` — the connectivity, default
/// 0.1 = 10% — with standard-normal values, then rescaled to spectral
/// radius `spectral_radius`, default 0.9), and leak rate `leak_rate` in
/// (0, 1] (default 1.0 = the plain ESN):
/// `s_t = (1 - a) s_{t-1} + a tanh(W s_{t-1} + W_in u_t)`, `s_0 = 0`, no
/// reservoir bias (reservoirpy's Reservoir with bias=0). The first
/// `washout` (50) rows are discarded and the readout is the ridge
/// regression of `y_t` on `[1, u_t, s_t]`, minimizing
/// `||y - Z b||^2 + ridge_alpha ||b||^2` (`ridge_alpha` 1e-6, scikit-learn
/// `Ridge(fit_intercept=False)` scale — the constant column is penalized
/// like every coefficient, Lukosevicius eq. 9). The spectral radius is
/// the leading-eigenvalue modulus from a dense eigenvalue decomposition
/// (not a power iteration, which does not converge on the complex leading
/// pair of a random reservoir); the value recomputed on the scaled matrix
/// is returned. Radii above 1 are accepted (the echo state property is
/// neither guaranteed below 1 nor excluded above it; the washout is what
/// removes the initial-state transient). `x_test` (optional, `n_test x p`)
/// is treated as the CONTINUATION of `x`: the state recursion carries on
/// from the last training state, no washout re-applied.
///
/// Returns a dict: `fitted` (readout predictions on the rows that entered
/// the fit, `t >= washout`, so length `n - washout`), `predicted` (on
/// `x_test`, else None), `readout` (coefficients on `[1, u, s]`, length
/// `1 + p + N`), `spectral_radius_achieved`, `reservoir_size`,
/// `n_washout`, `n_train` (`n - washout`). Keys: fitted, predicted,
/// readout, spectral_radius_achieved, reservoir_size, n_washout, n_train.
///
/// Validation (fixtures/neural.json): the state recursion on an explicit
/// small reservoir is pinned at 1e-12 against a NumPy transcription that
/// reservoirpy 0.4.2's Reservoir (same explicit W / Win / lr) reproduced
/// exactly at generation time — a third-party pin of the mechanics — and
/// the readout at 1e-10 against the closed-form ridge, itself
/// cross-checked against scikit-learn Ridge; the spectral radius against
/// numpy.linalg.eigvals (1e-6). The public estimator is property graded:
/// NARMA-10 out-of-sample NRMSE 0.32 (mean over four data seeds) with
/// input_scaling=0.3 and otherwise default settings on 1000 training rows
/// (0.19 with reservoir_size=400 on 2000 rows; the all-defaults call
/// averages 0.43, its input_scaling=1 over-driving tanh for NARMA's u in
/// [0, 0.5]), the achieved radius within 1e-6 of the target, and the seed
/// contract (same seed bit-identical, different seeds differ).
/// Reproducibility: single-threaded, every draw a pure function of
/// `seed`; bit-identical on the same build, last-ulp libm/eigenvalue
/// differences across platforms. Errors name the array (NaN/inf in x, y,
/// x_test); `washout >= n` names the fix; fewer than two rows after the
/// washout reports `insufficient data: {got} observations, at least
/// {needed} required` with the washout counted.
#[pyfunction]
#[pyo3(signature = (x, y, reservoir_size = 200, spectral_radius = 0.9, leak_rate = 1.0,
                    input_scaling = 1.0, sparsity = 0.1, washout = 50, ridge_alpha = 1e-6,
                    seed = 0, x_test = None))]
#[allow(clippy::too_many_arguments)]
fn echo_state_network<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    reservoir_size: usize,
    spectral_radius: f64,
    leak_rate: f64,
    input_scaling: f64,
    sparsity: f64,
    washout: usize,
    ridge_alpha: f64,
    seed: u64,
    x_test: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = tsecon_ml::EsnOptions {
        reservoir_size,
        spectral_radius,
        leak_rate,
        input_scaling,
        sparsity,
        washout,
        ridge_alpha,
        seed,
    };
    let xm = to_faer(&x);
    let xt = x_test.as_ref().map(to_faer);
    let r = tsecon_ml::echo_state_network(
        xm.as_ref(),
        &vec1(&y),
        xt.as_ref().map(|m| m.as_ref()),
        &opts,
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("fitted", r.fitted.into_pyarray(py))?;
    match r.predicted {
        Some(p) => d.set_item("predicted", p.into_pyarray(py))?,
        None => d.set_item("predicted", py.None())?,
    }
    d.set_item("readout", r.readout.into_pyarray(py))?;
    d.set_item("spectral_radius_achieved", r.spectral_radius_achieved)?;
    d.set_item("reservoir_size", r.reservoir_size)?;
    d.set_item("n_washout", r.n_washout)?;
    d.set_item("n_train", r.n_train)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mlp_regression, m)?)?;
    m.add_function(wrap_pyfunction!(echo_state_network, m)?)?;
    Ok(())
}
