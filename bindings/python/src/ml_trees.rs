//! Python bindings for the tree and forest slice of `tsecon-ml`:
//! `regression_tree` (CART, scikit-learn conventions, golden-pinned) and
//! `random_forest` (time-series-aware resampling, out-of-bag error,
//! quantile regression forests, grouped block-permutation importance).
//!
//! String options are parsed here into the crate's enums with teaching
//! errors that list the accepted values; the sentinel-refusal convention
//! (audit round 10) applies to every kwarg that is inert outside its mode:
//! `block_length` under `bootstrap="iid"`/`"none"`, `importance_groups`
//! under `importance="none"`, and `permutation_block`/`n_permutations`
//! under anything but `importance="block_permutation"`.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Borrowed;

use crate::{to_faer, to_py, vec1};

/// CART regression tree (Breiman et al. 1984) with scikit-learn's best-split
/// conventions — reproduces `DecisionTreeRegressor(criterion="squared_error",
/// splitter="best", max_features=None)`: test predictions at 1e-12,
/// `n_leaves`/`depth` exact, `feature_importances_` at 1e-10, and the sorted
/// (feature, threshold) multiset at 1e-12 on fixtures/trees.json (an
/// independent-package golden; the fixture proves each case tie-free by
/// refitting under five sklearn random_states — sklearn breaks exact ties
/// between features by its private RNG's visit order, this tree by the
/// lowest feature index, and only two-row nodes make such ties likely).
///
/// Conventions: squared-error criterion, threshold = midpoint of the two
/// adjacent sorted distinct values (values within 1e-7 count as one), a
/// split must leave at least `min_samples_leaf` rows on both sides, a
/// node with fewer than `min_samples_split` rows (or at `max_depth`, or
/// pure) is a leaf, leaves predict the training mean. sklearn works in
/// float32; this tree works in float64, so on general data the two can
/// differ wherever float32 rounding merges neighbouring values.
///
/// Arguments: `x` (n, p) design, `y` (n) target, `max_depth` (None =
/// unbounded), `min_samples_leaf` (>= 1), `min_samples_split` (>= 2),
/// `x_test` (m, p) rows to predict (optional).
///
/// Returns `fitted` (n, the leaf mean of every training row), `predicted`
/// (m, or None without `x_test`), `n_nodes`, `n_leaves`, `depth` (root =
/// 0; a stump has depth 1), `feature_importance` (p, impurity-based,
/// normalized to sum to one; zeros for a tree that never split), and
/// `splits` (a list of [feature, threshold] pairs over the internal nodes,
/// sorted by (feature, threshold)). Keys: fitted, predicted, n_nodes,
/// n_leaves, depth, feature_importance, splits.
///
/// Raises ValueError for NaN/inf in `x`/`y`/`x_test` (naming the array),
/// `x_test` with a different column count, `min_samples_leaf < 1`,
/// `min_samples_split < 2`, and `insufficient data: {got} observations,
/// at least {needed} required` when n < max(min_samples_split,
/// 2 * min_samples_leaf), i.e. when no split could ever be made.
#[pyfunction]
#[pyo3(signature = (x, y, max_depth = None, min_samples_leaf = 1, min_samples_split = 2, x_test = None))]
fn regression_tree<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    min_samples_split: usize,
    x_test: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = to_faer(&x);
    let mt = x_test.as_ref().map(to_faer);
    let opts = tsecon_ml::TreeOptions {
        max_depth,
        min_samples_leaf,
        min_samples_split,
    };
    let fit =
        tsecon_ml::regression_tree(m.as_ref(), &vec1(&y), opts, mt.as_ref().map(|t| t.as_ref()))
            .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("fitted", fit.fitted.into_pyarray(py))?;
    d.set_item("predicted", fit.predicted.map(|v| v.into_pyarray(py)))?;
    d.set_item("n_nodes", fit.n_nodes)?;
    d.set_item("n_leaves", fit.n_leaves)?;
    d.set_item("depth", fit.depth)?;
    d.set_item(
        "feature_importance",
        fit.feature_importance.into_pyarray(py),
    )?;
    let splits = PyList::empty(py);
    for (f, t) in fit.splits {
        splits.append(PyList::new(
            py,
            [
                f.into_pyobject(py)?.into_any(),
                t.into_pyobject(py)?.into_any(),
            ],
        )?)?;
    }
    d.set_item("splits", splits)?;
    Ok(d)
}

/// `max_features` as passed from Python — a scheme name or an integer
/// count — so the signature can carry the documented default `"third"`.
/// The count is validated against the column count once `x` is known.
enum MaxFeaturesArg {
    Name(String),
    Count(i64),
}

impl<'a, 'py> FromPyObject<'a, 'py> for MaxFeaturesArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.extract::<String>() {
            return Ok(Self::Name(s));
        }
        if let Ok(k) = obj.extract::<i64>() {
            return Ok(Self::Count(k));
        }
        Err(PyValueError::new_err(format!(
            "max_features must be \"sqrt\", \"third\", \"all\", or a positive integer; got an \
             instance of {}",
            obj.get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<unknown type>".to_string())
        )))
    }
}

fn parse_max_features(v: MaxFeaturesArg, p: usize) -> PyResult<tsecon_ml::MaxFeatures> {
    match v {
        MaxFeaturesArg::Name(s) => match s.as_str() {
            "sqrt" => Ok(tsecon_ml::MaxFeatures::Sqrt),
            "third" => Ok(tsecon_ml::MaxFeatures::Third),
            "all" => Ok(tsecon_ml::MaxFeatures::All),
            other => Err(PyValueError::new_err(format!(
                "unknown max_features {other:?}; expected \"sqrt\", \"third\", \"all\", or an \
                 integer in 1..={p} (the number of columns of x)"
            ))),
        },
        MaxFeaturesArg::Count(k) => {
            if k < 1 || k as usize > p {
                return Err(PyValueError::new_err(format!(
                    "max_features={k} is outside 1..={p} (x has {p} columns); pass an integer \
                     in that range or one of \"sqrt\", \"third\", \"all\""
                )));
            }
            Ok(tsecon_ml::MaxFeatures::Count(k as usize))
        }
    }
}

fn parse_bootstrap(
    bootstrap: &str,
    block_length: Option<usize>,
) -> PyResult<tsecon_ml::Resampling> {
    let needs_block = |what: &str| -> PyResult<usize> {
        block_length.ok_or_else(|| {
            PyValueError::new_err(format!(
                "bootstrap=\"{bootstrap}\" needs block_length ({what}); pass e.g. block_length=10, \
                 or optimal_block_length(y)[\"stationary\"] rounded up"
            ))
        })
    };
    let refuse_block = |mode: &str| -> PyResult<()> {
        match block_length {
            Some(b) => Err(PyValueError::new_err(format!(
                "block_length={b} has no effect under bootstrap=\"{bootstrap}\": {mode}; drop \
                 block_length, or pass bootstrap=\"block\" or \"stationary\" to resample in \
                 dependence-preserving blocks"
            ))),
            None => Ok(()),
        }
    };
    match bootstrap {
        "iid" => {
            refuse_block("the iid bootstrap draws single rows")?;
            Ok(tsecon_ml::Resampling::Iid)
        }
        "none" => {
            refuse_block("no resampling happens at all")?;
            Ok(tsecon_ml::Resampling::None)
        }
        "block" => Ok(tsecon_ml::Resampling::MovingBlock {
            block_length: needs_block("the moving-block length in rows")?,
        }),
        "stationary" => Ok(tsecon_ml::Resampling::Stationary {
            block_length: needs_block("the mean geometric block length in rows")?,
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown bootstrap {other:?}; expected \"iid\", \"block\", \"stationary\", or \"none\""
        ))),
    }
}

fn parse_importance(
    importance: &str,
    importance_groups: &Option<Vec<usize>>,
    permutation_block: Option<usize>,
    n_permutations: Option<usize>,
) -> PyResult<tsecon_ml::Importance> {
    let refuse_perm_knobs = || -> PyResult<()> {
        if let Some(b) = permutation_block {
            return Err(PyValueError::new_err(format!(
                "permutation_block={b} has no effect under importance=\"{importance}\": it is the \
                 block length of the block-permutation importance; pass \
                 importance=\"block_permutation\" or drop permutation_block"
            )));
        }
        if let Some(k) = n_permutations {
            return Err(PyValueError::new_err(format!(
                "n_permutations={k} has no effect under importance=\"{importance}\": it is the \
                 number of block permutations averaged by the block-permutation importance; \
                 pass importance=\"block_permutation\" or drop n_permutations"
            )));
        }
        Ok(())
    };
    match importance {
        "none" => {
            if let Some(g) = importance_groups {
                return Err(PyValueError::new_err(format!(
                    "importance_groups ({} labels) has no effect under importance=\"none\": it \
                     names the unit each column belongs to when an importance is computed; pass \
                     importance=\"impurity\" or \"block_permutation\", or drop importance_groups",
                    g.len()
                )));
            }
            refuse_perm_knobs()?;
            Ok(tsecon_ml::Importance::None)
        }
        "impurity" => {
            refuse_perm_knobs()?;
            Ok(tsecon_ml::Importance::Impurity)
        }
        "block_permutation" => Ok(tsecon_ml::Importance::BlockPermutation {
            permutation_block,
            n_permutations: n_permutations.unwrap_or(10),
        }),
        other => Err(PyValueError::new_err(format!(
            "unknown importance {other:?}; expected \"none\", \"impurity\", or \
             \"block_permutation\""
        ))),
    }
}

/// Random forest for regression (Breiman 2001) with time-series-aware
/// resampling, out-of-bag error, quantile regression forests (Meinshausen
/// 2006), and grouped block-permutation importance. Each tree is the CART
/// tree of `regression_tree` grown on a row resample (drawn rows act as
/// multiplicity weights; rows never drawn are the tree's out-of-bag rows)
/// visiting `max_features` random columns per node, and the forest
/// averages the trees.
///
/// Validation grade (honest): the deterministic tree is golden-pinned to
/// scikit-learn 1.9.0, and `random_forest(bootstrap="none",
/// max_features="all", n_trees=1, min_samples_leaf=1)` reproduces
/// `regression_tree` bit-for-bit, which is how the forest inherits that
/// golden. The full forest's randomness is tsecon's own Philox stream (one
/// SeedSequence substream per tree, so the result is bit-identical at any
/// thread count; same `seed` same forest, different `seed` different
/// forest), so it is validated by seeded Monte-Carlo property tests whose
/// measured numbers the model card quotes: out-of-sample R^2 on Friedman #1
/// above a documented bar, block/stationary resampling preserving the
/// lag-1 autocorrelation of the resampled rows, out-of-bag optimism under
/// AR errors, quantile-band coverage, and importance recovery.
///
/// Arguments: `x` (n, p), `y` (n); `n_trees`; `max_features` in {"sqrt",
/// "third" (max(1, p // 3), Breiman's regression default), "all", or an
/// integer in 1..=p}; `max_depth` (None = unbounded); `min_samples_leaf`;
/// `bootstrap` in {"iid" (Efron), "block" (Künsch moving block),
/// "stationary" (Politis-Romano, geometric blocks of mean `block_length`),
/// "none" (every tree sees every row; no out-of-bag rows)} — `block_length`
/// is REQUIRED for "block"/"stationary" and refused for "iid"/"none";
/// `seed`; `x_test` (m, p) rows to predict; `quantiles` (strictly inside
/// (0, 1), strictly increasing; requires `x_test`) turns on the quantile
/// regression forest; `importance` in {"none", "impurity",
/// "block_permutation"}; `importance_groups` (one integer label per column
/// — all lags of one variable get one label so they are permuted and
/// credited as one unit; needs `importance` != "none"; an integer array,
/// pass it as a list or int array — it is a label vector, not data);
/// `permutation_block` (rows per permuted block; None = ceil(n ** (1/3));
/// 1 = single-row permutation) and `n_permutations` (None = 10) act only
/// under importance="block_permutation" and are refused elsewhere.
///
/// Returns `fitted` (n, in-sample forest prediction — every tree, in-bag
/// rows included), `predicted` (m, or None without `x_test`),
/// `oob_prediction` (n; NaN where a row was never out-of-bag; None under
/// bootstrap="none"), `oob_mse` (over the rows with an out-of-bag
/// prediction; None under bootstrap="none"), `importance` (one entry per
/// unit — impurity: mean normalized impurity decrease, sums to one;
/// block_permutation: mean out-of-bag MSE increase in units of y^2,
/// negative values possible; None under importance="none"),
/// `importance_groups_resolved` (the unit label each `importance` entry
/// refers to: the sorted distinct labels, or 0..p-1), `quantile_predictions`
/// ((m, len(quantiles)) conditional quantiles, never crossing; None without
/// `quantiles`), `n_trees`, and `max_features_resolved` (columns visited
/// per node). Keys: fitted, predicted, oob_prediction, oob_mse, importance,
/// importance_groups_resolved, quantile_predictions, n_trees,
/// max_features_resolved.
///
/// Gotchas measured in the test suite and quoted on the model card. (1)
/// OUT-OF-BAG ERROR IS OPTIMISTIC ON TIME SERIES: an out-of-bag row's
/// temporal neighbours are in-bag in the trees that score it, and with
/// persistent predictors and autocorrelated errors they carry its error —
/// the property test measures OOB/POOS MSE ratios of about 0.70 under AR(0.9)
/// errors versus about 0.84 under iid errors on the same persistent design.
/// Report pseudo-out-of-sample metrics (fit on the past, score the
/// future), and prefer bootstrap="block"/"stationary". (2) IMPORTANCE OF A
/// PERSISTENT IRRELEVANT PREDICTOR IS INFLATED when the relevant predictors
/// are persistent too (the forest uses it as a time proxy); grouping the
/// lags of a variable is what keeps permuted rows dynamically possible and
/// stops its importance being diluted across collinear lags, but block
/// permutation does NOT remove that inflation — for a row-wise forest
/// scored row-wise, single-row and block permutation give the same mean
/// importance (measured within noise); compare against a control instead.
/// (3) Impurity importance favours columns with many distinct values.
///
/// Raises ValueError for NaN/inf (naming the array), `insufficient data:
/// {got} observations, at least {needed} required` when n < 2 *
/// min_samples_leaf, unknown string options (listing the accepted values),
/// `quantiles` outside (0, 1) or unsorted (naming the fix),
/// `importance_groups` of the wrong length (naming both lengths), a block
/// length outside 1..=n, and every inert-kwarg combination above.
#[pyfunction]
#[pyo3(signature = (x, y, n_trees = 500, max_features = MaxFeaturesArg::Name("third".to_string()), max_depth = None, min_samples_leaf = 5, bootstrap = "iid", block_length = None, seed = 0, x_test = None, quantiles = None, importance = "none", importance_groups = None, permutation_block = None, n_permutations = None))]
#[allow(clippy::too_many_arguments)]
fn random_forest<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    n_trees: usize,
    max_features: MaxFeaturesArg,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    bootstrap: &str,
    block_length: Option<usize>,
    seed: u64,
    x_test: Option<PyReadonlyArray2<'py, f64>>,
    quantiles: Option<Vec<f64>>,
    importance: &str,
    importance_groups: Option<Vec<usize>>,
    permutation_block: Option<usize>,
    n_permutations: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = to_faer(&x);
    let p = m.ncols();
    let mt = x_test.as_ref().map(to_faer);
    let max_features = parse_max_features(max_features, p)?;
    let resampling = parse_bootstrap(bootstrap, block_length)?;
    let importance = parse_importance(
        importance,
        &importance_groups,
        permutation_block,
        n_permutations,
    )?;
    let opts = tsecon_ml::ForestOptions {
        n_trees,
        max_features,
        max_depth,
        min_samples_leaf,
        resampling,
        seed,
        quantiles,
        importance,
        importance_groups,
    };
    let fit = tsecon_ml::random_forest(
        m.as_ref(),
        &vec1(&y),
        &opts,
        mt.as_ref().map(|t| t.as_ref()),
    )
    .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("fitted", fit.fitted.into_pyarray(py))?;
    d.set_item("predicted", fit.predicted.map(|v| v.into_pyarray(py)))?;
    d.set_item(
        "oob_prediction",
        fit.oob_prediction.map(|v| v.into_pyarray(py)),
    )?;
    d.set_item("oob_mse", fit.oob_mse)?;
    d.set_item("importance", fit.importance.map(|v| v.into_pyarray(py)))?;
    d.set_item(
        "importance_groups_resolved",
        fit.importance_groups_resolved.map(|v| {
            v.into_iter()
                .map(|g| g as u64)
                .collect::<Vec<_>>()
                .into_pyarray(py)
        }),
    )?;
    let qp: Option<Bound<'py, PyArray2<f64>>> = match fit.quantile_predictions {
        None => None,
        Some(rows) => {
            Some(PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))?)
        }
    };
    d.set_item("quantile_predictions", qp)?;
    d.set_item("n_trees", fit.n_trees)?;
    d.set_item("max_features_resolved", fit.max_features_resolved)?;
    Ok(d)
}

/// Registers the tree and forest functions on the `_core` module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regression_tree, m)?)?;
    m.add_function(wrap_pyfunction!(random_forest, m)?)?;
    Ok(())
}
