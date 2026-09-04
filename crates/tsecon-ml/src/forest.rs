//! Random forests for regression (Breiman 2001) with time-series-aware
//! resampling, out-of-bag error, quantile regression forests (Meinshausen
//! 2006), and grouped block-permutation importance.
//!
//! # What a tree in the forest is
//!
//! Every tree is a [`crate::tree`] CART tree grown on a resample of the
//! rows with per-row multiplicity weights (scikit-learn's convention: a
//! bootstrap draw becomes `sample_weight`, rows drawn zero times are
//! excluded and are the tree's *out-of-bag* rows), visiting a random
//! subset of `max_features` columns at every node. With
//! `Resampling::None`, `MaxFeatures::All`, and `n_trees = 1` the forest is
//! bit-for-bit the deterministic tree [`crate::regression_tree`] grows
//! (the property suite asserts this bridge, which is how the forest
//! inherits the scikit-learn golden).
//!
//! # Resampling
//!
//! * [`Resampling::Iid`] — Efron's bootstrap; right for independent rows,
//!   and the scheme every other forest library uses.
//! * [`Resampling::MovingBlock`] — Künsch's (1989) moving-block bootstrap:
//!   blocks of `block_length` consecutive rows starting at uniform
//!   positions; preserves within-block serial dependence.
//! * [`Resampling::Stationary`] — Politis & Romano's (1994) stationary
//!   bootstrap with geometric block lengths of mean `block_length`
//!   (restart probability `1 / block_length`), wrapping circularly.
//! * [`Resampling::None`] — every tree sees every row (no out-of-bag
//!   rows; the randomness is feature subsampling only).
//!
//! The index conventions mirror `tsecon_bootstrap::indices` exactly (same
//! draw order, same bitmask-rejection bounded uniforms), documented on
//! [`resample_indices`].
//!
//! # Reproducibility
//!
//! Tree `b` is grown from substream `b` of `Stream::substreams(seed,
//! n_trees + 1)` (the last substream drives the permutation importance),
//! and every aggregation runs in tree-index order after the parallel fit,
//! so the result is bit-identical at any rayon thread count. The same
//! `seed` gives the same forest; a different `seed` gives a different one.
//!
//! # Out-of-bag error — read this before quoting it
//!
//! `oob_prediction[i]` averages the trees in which row `i` was never
//! drawn (NaN if there were none; `oob_mse` averages over the rows that
//! have one). On independent rows this is an honest estimate of test
//! error. **On a time series it is optimistic**: a row's temporal
//! neighbours are in-bag in most of the trees that score it, and when
//! both the predictors and the errors are persistent those neighbours
//! carry the very error the out-of-bag row is supposed to be blind to.
//! The property suite measures the out-of-bag MSE against a
//! pseudo-out-of-sample MSE on a held-out final segment of the same
//! forest and asserts the sign; the model card quotes the ratio. Report
//! pseudo-out-of-sample metrics for time series, and use block or
//! stationary resampling so that each tree at least sees dependence-
//! preserving blocks.
//!
//! # Quantile regression forests
//!
//! Meinshausen (2006): the forest defines weights over the training
//! targets, `w_i(x) = (1/B) sum_b 1{i in leaf_b(x)} / |leaf_b(x)|`, where
//! every training row (in-bag or not) is dropped down tree `b` to find
//! its leaf, and the conditional distribution estimate is
//! `F(y | x) = sum_i w_i(x) 1{y_i <= y}`. The `q`-quantile is
//! `inf{y : F(y | x) >= q}`, read off one pass over the sorted targets,
//! so quantiles at increasing `q` never cross. Coverage of the
//! `[q_lo, q_hi]` band on iid data is measured in the property suite.
//!
//! # Importance
//!
//! See [`crate::importance`] for the two schemes and, in particular, for
//! what grouping does and what block permutation does not.

use rayon::prelude::*;
use tsecon_linalg::faer::MatRef;
use tsecon_rng::Stream;

use crate::error::MlError;
use crate::importance::{self, Groups};
use crate::tree::{check_tree_inputs, grow, RegressionTree, TreeOptions};
use crate::util::columns;

/// How many columns each node of a forest tree considers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxFeatures {
    /// `max(1, floor(sqrt(p)))`.
    Sqrt,
    /// `max(1, floor(p / 3))` — Breiman's regression default.
    Third,
    /// Every column (bagging; ties then break to the lowest index, as in
    /// [`crate::regression_tree`]).
    All,
    /// An explicit count in `1..=p`.
    Count(usize),
}

impl MaxFeatures {
    /// The number of columns visited per node for a `p`-column design.
    ///
    /// # Errors
    ///
    /// [`MlError::InvalidArgument`] if an explicit count is `0` or exceeds
    /// `p`.
    pub fn resolve(self, p: usize) -> Result<usize, MlError> {
        let m = match self {
            Self::Sqrt => (p as f64).sqrt().floor() as usize,
            Self::Third => p / 3,
            Self::All => p,
            Self::Count(k) => {
                if k == 0 || k > p {
                    return Err(MlError::InvalidArgument {
                        what: "max_features must lie in 1..=p (the number of columns of x)",
                    });
                }
                k
            }
        };
        Ok(m.max(1))
    }
}

/// The row-resampling scheme each tree is grown on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resampling {
    /// Efron's iid bootstrap.
    Iid,
    /// Künsch's moving-block bootstrap with fixed block length.
    MovingBlock {
        /// Block length in rows, `1..=n`.
        block_length: usize,
    },
    /// Politis-Romano stationary bootstrap with geometric blocks of this
    /// mean length.
    Stationary {
        /// Mean block length in rows, `1..=n`.
        block_length: usize,
    },
    /// No resampling: every tree sees every row once.
    None,
}

/// Which importance to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Importance {
    /// None.
    None,
    /// Mean normalized impurity decrease (scikit-learn's
    /// `feature_importances_`), summed within groups.
    Impurity,
    /// Grouped block-permutation importance on the out-of-bag rows.
    BlockPermutation {
        /// Rows per permuted block; `None` uses `ceil(n^(1/3))`. `Some(1)`
        /// is single-row permutation.
        permutation_block: Option<usize>,
        /// Block permutations averaged per unit (`>= 1`).
        n_permutations: usize,
    },
}

/// Configuration of [`random_forest`].
#[derive(Debug, Clone, PartialEq)]
pub struct ForestOptions {
    /// Number of trees (`>= 1`).
    pub n_trees: usize,
    /// Columns considered per node.
    pub max_features: MaxFeatures,
    /// Maximum tree depth (`None` = unbounded).
    pub max_depth: Option<usize>,
    /// Smallest number of rows a split may leave on either side.
    pub min_samples_leaf: usize,
    /// Row-resampling scheme.
    pub resampling: Resampling,
    /// Seed of the reproducible stream family.
    pub seed: u64,
    /// Quantile levels for quantile regression forests, strictly inside
    /// `(0, 1)` and strictly increasing; requires `x_test`.
    pub quantiles: Option<Vec<f64>>,
    /// Importance scheme.
    pub importance: Importance,
    /// One integer label per column of `x` naming its importance unit;
    /// `None` makes every column its own unit.
    pub importance_groups: Option<Vec<usize>>,
}

impl Default for ForestOptions {
    fn default() -> Self {
        Self {
            n_trees: 500,
            max_features: MaxFeatures::Third,
            max_depth: None,
            min_samples_leaf: 5,
            resampling: Resampling::Iid,
            seed: 0,
            quantiles: None,
            importance: Importance::None,
            importance_groups: None,
        }
    }
}

/// Result of [`random_forest`].
#[derive(Debug, Clone, PartialEq)]
pub struct ForestFit {
    /// Forest prediction for every training row (all trees, in-bag
    /// included — an in-sample fit).
    pub fitted: Vec<f64>,
    /// Forest prediction for every row of `x_test`, when given.
    pub predicted: Option<Vec<f64>>,
    /// Out-of-bag prediction per training row (NaN where the row was
    /// never out-of-bag); `None` under [`Resampling::None`].
    pub oob_prediction: Option<Vec<f64>>,
    /// Mean squared out-of-bag error over the rows with an out-of-bag
    /// prediction; `None` under [`Resampling::None`]. Optimistic on time
    /// series — see the [module docs](self).
    pub oob_mse: Option<f64>,
    /// Importance per unit, `None` under [`Importance::None`].
    pub importance: Option<Vec<f64>>,
    /// The unit label each entry of `importance` refers to (the sorted
    /// distinct `importance_groups` labels, or `0..p`).
    pub importance_groups_resolved: Option<Vec<usize>>,
    /// `n_test x n_quantiles` conditional quantiles, when requested.
    pub quantile_predictions: Option<Vec<Vec<f64>>>,
    /// Number of trees grown.
    pub n_trees: usize,
    /// Columns visited per node after resolving `max_features`.
    pub max_features_resolved: usize,
}

/// Exactly uniform draw from `0..n` by bitmask rejection sampling on the
/// raw 64-bit output (the convention of `tsecon_bootstrap`); `n == 1`
/// consumes no randomness.
#[inline]
pub(crate) fn uniform_index(stream: &mut Stream, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let bound = n as u64;
    let mask = u64::MAX >> (bound - 1).leading_zeros();
    loop {
        let v = stream.next_u64() & mask;
        if v < bound {
            return v as usize;
        }
    }
}

/// Draws one length-`n` index resample under `scheme`, mirroring the
/// conventions of `tsecon_bootstrap::indices`:
///
/// * `Iid`: `n` bounded uniform draws;
/// * `MovingBlock`: one bounded uniform start in `0..=n-l` per block,
///   blocks concatenated and the last one truncated;
/// * `Stationary`: a bounded uniform first index, then per step one
///   `[0, 1)` uniform restart coin (probability `1/l`) and, on restart,
///   one bounded uniform draw; otherwise the next index modulo `n`;
/// * `None`: `0..n`, no draws.
///
/// # Errors
///
/// [`MlError::EmptyInput`] if `n == 0`; [`MlError::InvalidBlockLength`]
/// if a block length is outside `1..=n`.
pub fn resample_indices(
    scheme: Resampling,
    n: usize,
    stream: &mut Stream,
) -> Result<Vec<usize>, MlError> {
    if n == 0 {
        return Err(MlError::EmptyInput { what: "x" });
    }
    let check = |l: usize| -> Result<(), MlError> {
        if l == 0 || l > n {
            Err(MlError::InvalidBlockLength {
                what: "block_length",
                block_length: l,
                n,
            })
        } else {
            Ok(())
        }
    };
    match scheme {
        Resampling::None => Ok((0..n).collect()),
        Resampling::Iid => Ok((0..n).map(|_| uniform_index(stream, n)).collect()),
        Resampling::MovingBlock { block_length } => {
            check(block_length)?;
            let n_starts = n - block_length + 1;
            let mut out = Vec::with_capacity(n);
            while out.len() < n {
                let start = uniform_index(stream, n_starts);
                let take = block_length.min(n - out.len());
                out.extend(start..start + take);
            }
            Ok(out)
        }
        Resampling::Stationary { block_length } => {
            check(block_length)?;
            let p = 1.0 / block_length as f64;
            let mut out = Vec::with_capacity(n);
            let mut idx = uniform_index(stream, n);
            out.push(idx);
            for _ in 1..n {
                if stream.uniform_f64() < p {
                    idx = uniform_index(stream, n);
                } else {
                    idx = (idx + 1) % n;
                }
                out.push(idx);
            }
            Ok(out)
        }
    }
}

/// Everything one tree contributes, computed inside the parallel fit.
struct TreeOutput {
    tree: RegressionTree,
    /// Resample multiplicity per training row (0 = out-of-bag).
    weights: Vec<f64>,
    pred_train: Vec<f64>,
    pred_test: Option<Vec<f64>>,
    /// Leaf id of every training row and of every test row, plus a CSR
    /// index (`offsets`, `rows`) of training rows by leaf — only when
    /// quantiles were requested.
    leaves: Option<LeafIndex>,
}

struct LeafIndex {
    offsets: Vec<usize>,
    rows: Vec<usize>,
    leaf_test: Vec<usize>,
}

fn leaf_index(
    tree: &RegressionTree,
    cols: &[Vec<f64>],
    n: usize,
    x_test: MatRef<'_, f64>,
) -> LeafIndex {
    let n_nodes = tree.n_nodes();
    let leaf_train: Vec<usize> = (0..n).map(|i| tree.leaf_for(|j| cols[j][i])).collect();
    let mut counts = vec![0usize; n_nodes + 1];
    for &l in &leaf_train {
        counts[l + 1] += 1;
    }
    for k in 0..n_nodes {
        counts[k + 1] += counts[k];
    }
    let offsets = counts.clone();
    let mut fill = counts;
    let mut rows = vec![0usize; n];
    for (i, &l) in leaf_train.iter().enumerate() {
        rows[fill[l]] = i;
        fill[l] += 1;
    }
    let leaf_test = (0..x_test.nrows())
        .map(|i| tree.leaf_for(|j| x_test[(i, j)]))
        .collect();
    LeafIndex {
        offsets,
        rows,
        leaf_test,
    }
}

/// Validates the quantile levels.
fn check_quantiles(q: &[f64]) -> Result<(), MlError> {
    if q.is_empty() {
        return Err(MlError::InvalidArgument {
            what: "quantiles must be nonempty — e.g. quantiles=[0.1, 0.5, 0.9]",
        });
    }
    if q.iter().any(|v| !v.is_finite() || *v <= 0.0 || *v >= 1.0) {
        return Err(MlError::InvalidArgument {
            what: "quantiles must lie strictly inside (0, 1) — e.g. quantiles=[0.1, 0.5, 0.9]",
        });
    }
    if q.windows(2).any(|w| w[1] <= w[0]) {
        return Err(MlError::InvalidArgument {
            what: "quantiles must be strictly increasing (sorted ascending, no duplicates) \
                   — e.g. quantiles=[0.1, 0.5, 0.9]",
        });
    }
    Ok(())
}

/// Meinshausen's quantile regression forest for the test rows.
fn quantile_predictions(outputs: &[TreeOutput], y: &[f64], quantiles: &[f64]) -> Vec<Vec<f64>> {
    let n = y.len();
    let n_trees = outputs.len() as f64;
    let n_test = outputs
        .first()
        .and_then(|o| o.leaves.as_ref())
        .map_or(0, |l| l.leaf_test.len());
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]));

    (0..n_test)
        .into_par_iter()
        .map(|i| {
            let mut w = vec![0.0f64; n];
            for o in outputs {
                if let Some(l) = &o.leaves {
                    let leaf = l.leaf_test[i];
                    let start = l.offsets[leaf];
                    let end = l.offsets[leaf + 1];
                    if end > start {
                        let share = 1.0 / ((end - start) as f64 * n_trees);
                        for &r in &l.rows[start..end] {
                            w[r] += share;
                        }
                    }
                }
            }
            // One pass over the sorted targets; increasing q gives
            // non-decreasing quantiles by construction.
            let mut out = Vec::with_capacity(quantiles.len());
            let mut cum = 0.0;
            let mut k = 0usize;
            for &q in quantiles {
                while k < n && cum < q {
                    cum += w[order[k]];
                    k += 1;
                }
                let idx = if k == 0 { 0 } else { k - 1 };
                out.push(y[order[idx.min(n - 1)]]);
            }
            out
        })
        .collect()
}

/// Fits a regression forest to `x` (`n x p`) and `y` (`n`); see the
/// [module docs](self) for the resampling schemes, the out-of-bag caveat,
/// quantile forests, and importance.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on malformed `x`, `y`, `x_test`, or
///   `importance_groups`;
/// * [`MlError::InsufficientData`] if `n < max(2, 2 * min_samples_leaf)`;
/// * [`MlError::InvalidBlockLength`] for a block or permutation block
///   outside `1..=n`;
/// * [`MlError::InvalidArgument`] for `n_trees == 0`, an explicit
///   `max_features` outside `1..=p`, malformed `quantiles`, `quantiles`
///   without `x_test`, `n_permutations == 0`, or block-permutation
///   importance under [`Resampling::None`] (no out-of-bag rows to score).
pub fn random_forest(
    x: MatRef<'_, f64>,
    y: &[f64],
    opts: &ForestOptions,
    x_test: Option<MatRef<'_, f64>>,
) -> Result<ForestFit, MlError> {
    let tree_opts = TreeOptions {
        max_depth: opts.max_depth,
        min_samples_leaf: opts.min_samples_leaf,
        min_samples_split: 2,
    };
    let (n, p) = check_tree_inputs(x, y, x_test, tree_opts)?;
    if opts.n_trees == 0 {
        return Err(MlError::InvalidArgument {
            what: "n_trees must be at least 1",
        });
    }
    let m = opts.max_features.resolve(p)?;
    match opts.resampling {
        Resampling::MovingBlock { block_length } | Resampling::Stationary { block_length } => {
            if block_length == 0 || block_length > n {
                return Err(MlError::InvalidBlockLength {
                    what: "block_length",
                    block_length,
                    n,
                });
            }
        }
        Resampling::Iid | Resampling::None => {}
    }
    if let Some(q) = &opts.quantiles {
        check_quantiles(q)?;
        if x_test.is_none() {
            return Err(MlError::InvalidArgument {
                what: "quantiles were given but x_test was not: quantile predictions are \
                       computed for the rows of x_test; pass x_test=... or drop quantiles",
            });
        }
    }
    let groups: Option<Groups> = match opts.importance {
        Importance::None => None,
        Importance::Impurity | Importance::BlockPermutation { .. } => Some(
            importance::resolve_groups(p, opts.importance_groups.as_deref())?,
        ),
    };
    let permutation = match opts.importance {
        Importance::BlockPermutation {
            permutation_block,
            n_permutations,
        } => {
            if opts.resampling == Resampling::None {
                return Err(MlError::InvalidArgument {
                    what: "importance='block_permutation' scores the out-of-bag rows and \
                           bootstrap='none' has none; use bootstrap='iid', 'block' or \
                           'stationary', or importance='impurity'",
                });
            }
            if n_permutations == 0 {
                return Err(MlError::InvalidArgument {
                    what: "n_permutations must be at least 1",
                });
            }
            let block = match permutation_block {
                None => ((n as f64).cbrt().ceil() as usize).clamp(1, n),
                Some(b) => {
                    if b == 0 || b > n {
                        return Err(MlError::InvalidBlockLength {
                            what: "permutation_block",
                            block_length: b,
                            n,
                        });
                    }
                    b
                }
            };
            Some((block, n_permutations))
        }
        Importance::None | Importance::Impurity => None,
    };

    let cols = columns(x);
    let mut streams =
        Stream::substreams(opts.seed, opts.n_trees + 1).map_err(|_| MlError::InvalidArgument {
            what: "n_trees exceeds the reproducible substream limit",
        })?;
    let mut perm_stream = streams.pop().unwrap_or_else(|| Stream::new(opts.seed));
    let want_leaves = opts.quantiles.is_some();
    let resampling = opts.resampling;

    let outputs: Vec<TreeOutput> = streams
        .into_par_iter()
        .map(|mut stream| -> Result<TreeOutput, MlError> {
            let idx = resample_indices(resampling, n, &mut stream)?;
            let mut weights = vec![0.0f64; n];
            for i in idx {
                weights[i] += 1.0;
            }
            let tree = if m >= p {
                let mut in_order = |k: usize, _p: usize| k;
                grow(&cols, y, &weights, tree_opts, p, &mut in_order)
            } else {
                let mut shuffled = |k: usize, p: usize| k + uniform_index(&mut stream, p - k);
                grow(&cols, y, &weights, tree_opts, m, &mut shuffled)
            };
            let pred_train = tree.predict_cols(&cols, n);
            let pred_test = x_test.map(|xt| tree.predict(xt));
            let leaves = match (want_leaves, x_test) {
                (true, Some(xt)) => Some(leaf_index(&tree, &cols, n, xt)),
                _ => None,
            };
            Ok(TreeOutput {
                tree,
                weights,
                pred_train,
                pred_test,
                leaves,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let n_trees = outputs.len();
    let inv_b = 1.0 / n_trees as f64;
    let mut fitted = vec![0.0; n];
    for o in &outputs {
        for (f, v) in fitted.iter_mut().zip(&o.pred_train) {
            *f += v;
        }
    }
    for f in &mut fitted {
        *f *= inv_b;
    }
    let predicted = x_test.map(|xt| {
        let mut acc = vec![0.0; xt.nrows()];
        for o in &outputs {
            if let Some(pt) = &o.pred_test {
                for (a, v) in acc.iter_mut().zip(pt) {
                    *a += v;
                }
            }
        }
        for a in &mut acc {
            *a *= inv_b;
        }
        acc
    });

    // Out-of-bag aggregation in tree order.
    let (oob_prediction, oob_mse, oob_trees) = if resampling == Resampling::None {
        (None, None, Vec::new())
    } else {
        let mut oob_trees: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sum = vec![0.0; n];
        for (b, o) in outputs.iter().enumerate() {
            for i in 0..n {
                if o.weights[i] == 0.0 {
                    sum[i] += o.pred_train[i];
                    oob_trees[i].push(b);
                }
            }
        }
        let mut sse = 0.0;
        let mut count = 0usize;
        let pred: Vec<f64> = (0..n)
            .map(|i| {
                let k = oob_trees[i].len();
                if k == 0 {
                    f64::NAN
                } else {
                    let v = sum[i] / k as f64;
                    let e = y[i] - v;
                    sse += e * e;
                    count += 1;
                    v
                }
            })
            .collect();
        let mse = if count > 0 {
            sse / count as f64
        } else {
            f64::NAN
        };
        (Some(pred), Some(mse), oob_trees)
    };

    let quantile_predictions = opts
        .quantiles
        .as_ref()
        .map(|q| quantile_predictions(&outputs, y, q));

    let trees: Vec<RegressionTree> = outputs.into_iter().map(|o| o.tree).collect();
    let (importance, importance_groups_resolved) = match (&groups, opts.importance) {
        (Some(g), Importance::Impurity) => (
            Some(importance::impurity_importance(&trees, g)),
            Some(g.labels.clone()),
        ),
        (Some(g), Importance::BlockPermutation { .. }) => {
            let (block, n_perm) = permutation.unwrap_or((1, 1));
            let base = oob_mse.unwrap_or(f64::NAN);
            let imp = importance::block_permutation_importance(
                &trees,
                &oob_trees,
                &cols,
                y,
                g,
                block,
                n_perm,
                &mut perm_stream,
                base,
            );
            (Some(imp), Some(g.labels.clone()))
        }
        _ => (None, None),
    };

    Ok(ForestFit {
        fitted,
        predicted,
        oob_prediction,
        oob_mse,
        importance,
        importance_groups_resolved,
        quantile_predictions,
        n_trees,
        max_features_resolved: m,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn max_features_resolution() {
        assert_eq!(MaxFeatures::Sqrt.resolve(10).unwrap(), 3);
        assert_eq!(MaxFeatures::Third.resolve(10).unwrap(), 3);
        assert_eq!(MaxFeatures::Third.resolve(2).unwrap(), 1);
        assert_eq!(MaxFeatures::All.resolve(7).unwrap(), 7);
        assert_eq!(MaxFeatures::Count(4).resolve(7).unwrap(), 4);
        assert!(MaxFeatures::Count(0).resolve(7).is_err());
        assert!(MaxFeatures::Count(8).resolve(7).is_err());
    }

    #[test]
    fn resample_schemes_have_length_n_and_in_range_indices() {
        let mut s = Stream::new(5);
        for scheme in [
            Resampling::Iid,
            Resampling::MovingBlock { block_length: 4 },
            Resampling::Stationary { block_length: 4 },
            Resampling::None,
        ] {
            let idx = resample_indices(scheme, 17, &mut s).unwrap();
            assert_eq!(idx.len(), 17);
            assert!(idx.iter().all(|&i| i < 17));
        }
        assert_eq!(
            resample_indices(Resampling::None, 5, &mut s).unwrap(),
            vec![0, 1, 2, 3, 4]
        );
        let err =
            resample_indices(Resampling::MovingBlock { block_length: 9 }, 8, &mut s).unwrap_err();
        assert_eq!(
            err.to_string(),
            "block_length=9 is outside 1..=8: a block cannot be empty or longer than the \
             8-row sample"
        );
    }

    #[test]
    fn quantile_checks_name_the_fix() {
        assert!(check_quantiles(&[]).is_err());
        let e = check_quantiles(&[0.1, 1.0]).unwrap_err().to_string();
        assert!(e.contains("strictly inside (0, 1)"), "{e}");
        let e = check_quantiles(&[0.9, 0.1]).unwrap_err().to_string();
        assert!(e.contains("strictly increasing"), "{e}");
        assert!(check_quantiles(&[0.1, 0.5, 0.9]).is_ok());
    }
}
