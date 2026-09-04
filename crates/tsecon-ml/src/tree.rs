//! CART regression trees (Breiman, Friedman, Olshen & Stone 1984) with
//! scikit-learn's best-split conventions, so a tree grown here is the same
//! tree `DecisionTreeRegressor(criterion="squared_error", splitter="best")`
//! grows — the golden fixture `fixtures/trees.json` pins that at 1e-12.
//!
//! # Split search
//!
//! Every internal node searches every visited feature for the threshold
//! that maximizes the squared-error improvement. With `S_L`, `S_R` the
//! weighted target sums and `W_L`, `W_R` the weight totals of the two
//! sides, the candidate is ranked by the proxy
//!
//! ```text
//! S_L^2 / W_L + S_R^2 / W_R ,
//! ```
//!
//! which is the weighted between-group sum of squares up to a constant and
//! therefore orders splits exactly as the impurity decrease does (the
//! quantity scikit-learn's `MSE.proxy_impurity_improvement` computes).
//! The threshold is the **midpoint of the two adjacent sorted distinct
//! values**, `v[p-1]/2 + v[p]/2`, and — scikit-learn's `FEATURE_THRESHOLD`
//! — two values within `1e-7` of each other count as one value. A split
//! must leave at least `min_samples_leaf` rows (distinct rows, not weight)
//! on each side. A node becomes a leaf when it has fewer than
//! `min_samples_split` rows, fewer than `2 * min_samples_leaf` rows, sits
//! at `max_depth`, is pure (impurity `<= f64::EPSILON`), or admits no
//! valid split. Leaves predict the weighted training mean.
//!
//! # Tie-break
//!
//! scikit-learn visits features in an order drawn from its private RNG and
//! keeps the first strictly-best split, so when two features induce the
//! *same partition* of a node (certain in a two-row node, measure-zero in a
//! large one with continuous data) the winner depends on that RNG. This
//! implementation visits features in index order `0, 1, ..., p-1` (the
//! forest shuffles them with its own stream) and also keeps the first
//! strictly-best split, so ties resolve to the lowest feature index. The
//! fixture generator proves each stored case is tie-free by refitting
//! under five `random_state` values and asserting the same tree, which is
//! why exact matching is possible there and why unbounded depth with
//! `min_samples_leaf = 1` is not stored.
//!
//! # Float32
//!
//! scikit-learn casts features to float32 before growing and predicting.
//! This crate works in float64; the fixture stores float32-representable
//! values so both see identical numbers. On general float64 data the two
//! can differ wherever float32 rounding merges or reorders neighbours.
//!
//! # Feature importance
//!
//! Impurity-based, as scikit-learn's `feature_importances_`: for every
//! internal node `W * impurity - W_L * impurity_L - W_R * impurity_R` is
//! credited to the split feature, the totals are divided by the root
//! weight, and the vector is normalized to sum to one (all zeros for a
//! tree that never split). It is fast and exact, and it is biased toward
//! features with many distinct values and toward whichever member of a
//! correlated group happens to be picked — read
//! [`crate::importance`] before interpreting it.

use tsecon_linalg::faer::MatRef;

use crate::error::MlError;
use crate::util::{check_xy, columns};

/// scikit-learn's `FEATURE_THRESHOLD`: two feature values closer than this
/// are one value when enumerating candidate thresholds, and a feature whose
/// node range is no wider than this is constant at that node.
pub const FEATURE_THRESHOLD: f64 = 1e-7;

/// Stopping controls for a regression tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeOptions {
    /// Maximum depth (root = depth 0); `None` grows until every leaf is
    /// pure or too small to split.
    pub max_depth: Option<usize>,
    /// Smallest number of rows a split may leave on either side (`>= 1`).
    pub min_samples_leaf: usize,
    /// Smallest number of rows a node needs to be considered for a split
    /// (`>= 2`).
    pub min_samples_split: usize,
}

impl Default for TreeOptions {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
        }
    }
}

/// The split stored on an internal node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Split {
    /// Column of `x` the node splits on.
    pub feature: usize,
    /// Rows with `x[feature] <= threshold` go left, the rest right.
    pub threshold: f64,
    /// Node id of the left child.
    pub left: usize,
    /// Node id of the right child.
    pub right: usize,
}

/// One node of a fitted tree (ids are depth-first pre-order, left first;
/// node 0 is the root).
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// `Some` for an internal node, `None` for a leaf.
    pub split: Option<Split>,
    /// Weighted mean of the training targets in the node — the prediction
    /// if it is a leaf.
    pub value: f64,
    /// Number of distinct training rows in the node.
    pub n_samples: usize,
    /// Total training weight in the node (equals `n_samples` for unit
    /// weights; the bootstrap multiplicity total inside a forest).
    pub weighted_n_samples: f64,
    /// Weighted mean squared error of the node's targets around `value`.
    pub impurity: f64,
    /// Depth of the node (root = 0).
    pub depth: usize,
}

/// A fitted regression tree.
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionTree {
    /// Nodes in depth-first pre-order (left child before right).
    pub nodes: Vec<Node>,
    /// Number of columns the tree was grown on.
    pub n_features: usize,
    /// Depth of the deepest leaf (a single split gives depth 1; a tree that
    /// never split has depth 0).
    pub depth: usize,
    /// Number of leaves.
    pub n_leaves: usize,
    /// Unnormalized impurity-based importance: the summed weighted impurity
    /// decrease per feature divided by the root weight (scikit-learn's
    /// `compute_feature_importances(normalize=False)`).
    pub raw_importance: Vec<f64>,
}

impl RegressionTree {
    /// Number of nodes (internal plus leaves).
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Id of the leaf a row falls into, where `feature(j)` returns the
    /// row's value on column `j`.
    pub fn leaf_for<F: Fn(usize) -> f64>(&self, feature: F) -> usize {
        let mut id = 0usize;
        loop {
            match self.nodes.get(id).and_then(|n| n.split) {
                None => return id,
                Some(s) => {
                    id = if feature(s.feature) <= s.threshold {
                        s.left
                    } else {
                        s.right
                    };
                }
            }
        }
    }

    /// Prediction for a row given as a feature accessor.
    pub fn predict_with<F: Fn(usize) -> f64>(&self, feature: F) -> f64 {
        let id = self.leaf_for(feature);
        self.nodes.get(id).map_or(f64::NAN, |n| n.value)
    }

    /// Prediction for a contiguous row slice of length `n_features`.
    pub fn predict_row(&self, row: &[f64]) -> f64 {
        self.predict_with(|j| row.get(j).copied().unwrap_or(f64::NAN))
    }

    /// Predictions for every row of an `m x n_features` matrix.
    pub fn predict(&self, x: MatRef<'_, f64>) -> Vec<f64> {
        (0..x.nrows())
            .map(|i| self.predict_with(|j| x[(i, j)]))
            .collect()
    }

    /// Predictions for every row of a column-major design (`cols[j][i]`).
    pub(crate) fn predict_cols(&self, cols: &[Vec<f64>], n: usize) -> Vec<f64> {
        (0..n).map(|i| self.predict_with(|j| cols[j][i])).collect()
    }

    /// Normalized impurity-based feature importance (sums to one; all
    /// zeros for a tree that never split), scikit-learn's
    /// `feature_importances_`.
    pub fn feature_importance(&self) -> Vec<f64> {
        let total: f64 = self.raw_importance.iter().sum();
        if total > 0.0 {
            self.raw_importance.iter().map(|v| v / total).collect()
        } else {
            vec![0.0; self.raw_importance.len()]
        }
    }

    /// The multiset of `(feature, threshold)` pairs over the internal
    /// nodes, sorted by `(feature, threshold)`.
    pub fn splits(&self) -> Vec<(usize, f64)> {
        let mut out: Vec<(usize, f64)> = self
            .nodes
            .iter()
            .filter_map(|n| n.split.map(|s| (s.feature, s.threshold)))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        out
    }
}

/// Weighted sum, weight total, and weighted sum of squares of `y` over the
/// rows in `idx`.
fn node_stats(idx: &[usize], y: &[f64], w: &[f64]) -> (f64, f64, f64) {
    let mut sum = 0.0;
    let mut wt = 0.0;
    let mut sq = 0.0;
    for &i in idx {
        let wi = w[i];
        let yi = y[i];
        sum += wi * yi;
        wt += wi;
        sq += wi * yi * yi;
    }
    (sum, wt, sq)
}

/// Weighted mean squared error around the weighted mean,
/// `sq/w - (sum/w)^2` (scikit-learn's `MSE.node_impurity`).
fn impurity(sum: f64, w: f64, sq: f64) -> f64 {
    if w > 0.0 {
        sq / w - (sum / w) * (sum / w)
    } else {
        0.0
    }
}

/// The best split found at a node (before the children are materialized).
struct Candidate {
    feature: usize,
    threshold: f64,
    proxy: f64,
}

/// Best-split search over the node's rows `idx`, visiting features in the
/// order produced by `draw` (see [`grow`]) until `max_features` have been
/// visited and at least one of them was non-constant.
#[allow(clippy::too_many_arguments)]
fn best_split(
    idx: &[usize],
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    min_samples_leaf: usize,
    sum_total: f64,
    w_total: f64,
    max_features: usize,
    draw: &mut dyn FnMut(usize, usize) -> usize,
) -> Option<Candidate> {
    let p = cols.len();
    let m = idx.len();
    let mut features: Vec<usize> = (0..p).collect();
    let mut best: Option<Candidate> = None;
    let mut visited = 0usize;
    let mut non_constant = 0usize;
    let mut k = 0usize;
    // (value, row) pairs sorted per feature; one allocation reused.
    let mut pairs: Vec<(f64, usize)> = Vec::with_capacity(m);

    while k < p && (visited < max_features || non_constant == 0) {
        let j = draw(k, p).clamp(k, p - 1);
        features.swap(k, j);
        let f = features[k];
        k += 1;
        visited += 1;

        let col = &cols[f];
        pairs.clear();
        pairs.extend(idx.iter().map(|&i| (col[i], i)));
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Constant at this node (scikit-learn: max <= min + FEATURE_THRESHOLD).
        if pairs[m - 1].0 <= pairs[0].0 + FEATURE_THRESHOLD {
            continue;
        }
        non_constant += 1;

        let mut sum_left = 0.0;
        let mut w_left = 0.0;
        let mut added = 0usize; // rows already folded into the left sums
        let mut pos = 0usize;
        while pos < m {
            // Skip over rows whose values are within FEATURE_THRESHOLD of
            // each other: they are one value and cannot be separated.
            while pos + 1 < m && pairs[pos + 1].0 <= pairs[pos].0 + FEATURE_THRESHOLD {
                pos += 1;
            }
            let prev = pos;
            pos += 1;
            if pos >= m {
                break;
            }
            while added < pos {
                let i = pairs[added].1;
                sum_left += w[i] * y[i];
                w_left += w[i];
                added += 1;
            }
            let n_left = pos;
            let n_right = m - pos;
            if n_left < min_samples_leaf || n_right < min_samples_leaf {
                continue;
            }
            let sum_right = sum_total - sum_left;
            let w_right = w_total - w_left;
            if w_left <= 0.0 || w_right <= 0.0 {
                continue;
            }
            let proxy = sum_left * sum_left / w_left + sum_right * sum_right / w_right;
            let improves = match &best {
                None => true,
                Some(b) => proxy > b.proxy,
            };
            if improves {
                let lo = pairs[prev].0;
                let hi = pairs[pos].0;
                let mut threshold = lo / 2.0 + hi / 2.0;
                if threshold == hi || !threshold.is_finite() {
                    threshold = lo;
                }
                best = Some(Candidate {
                    feature: f,
                    threshold,
                    proxy,
                });
            }
        }
    }
    best
}

/// In-place partition of `idx` by `col[i] <= threshold`; returns the number
/// of rows that went left (they occupy `idx[..pos]`).
fn partition(idx: &mut [usize], col: &[f64], threshold: f64) -> usize {
    let mut pos = 0usize;
    for i in 0..idx.len() {
        if col[idx[i]] <= threshold {
            idx.swap(i, pos);
            pos += 1;
        }
    }
    pos
}

/// A node waiting to be materialized during the depth-first build.
struct Pending {
    start: usize,
    end: usize,
    depth: usize,
    /// `(parent id, is_left_child)`; `None` for the root.
    parent: Option<(usize, bool)>,
    impurity: f64,
}

/// Grows a tree on the column-major design `cols` (each `cols[j]` has one
/// entry per row of the *full* sample), targets `y`, and per-row weights
/// `w` — rows with zero weight are excluded, positive weights act as
/// multiplicities in the sums while `min_samples_leaf` /
/// `min_samples_split` count distinct rows (scikit-learn's convention for
/// a bootstrap sample passed as `sample_weight`).
///
/// `draw(k, p)` returns the index in `k..p` of the feature to visit in
/// slot `k` of a lazy Fisher-Yates shuffle; `|k, _| k` visits features in
/// index order. `max_features` bounds how many features a node visits
/// (a node keeps visiting past the bound only while every visited feature
/// was constant).
///
/// Node ids are assigned depth-first with the left child before the right,
/// as scikit-learn's `DepthFirstTreeBuilder` does; the node sums are
/// accumulated over rows in the order the parent's partition left them,
/// so the same call sequence gives bit-identical trees.
pub(crate) fn grow(
    cols: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    opts: TreeOptions,
    max_features: usize,
    draw: &mut dyn FnMut(usize, usize) -> usize,
) -> RegressionTree {
    let p = cols.len();
    let max_depth = opts.max_depth.unwrap_or(usize::MAX);
    let mut samples: Vec<usize> = (0..y.len()).filter(|&i| w[i] > 0.0).collect();
    let n_root = samples.len();
    let mut nodes: Vec<Node> = Vec::new();
    let mut raw_importance = vec![0.0; p];

    let (root_sum, root_w, root_sq) = node_stats(&samples, y, w);
    let root_impurity = impurity(root_sum, root_w, root_sq);
    let mut stack = vec![Pending {
        start: 0,
        end: n_root,
        depth: 0,
        parent: None,
        impurity: root_impurity,
    }];

    while let Some(pending) = stack.pop() {
        let Pending {
            start,
            end,
            depth,
            parent,
            impurity: node_impurity,
        } = pending;
        let node_id = nodes.len();
        let n_node = end - start;
        let (sum_total, w_total, _) = node_stats(&samples[start..end], y, w);
        let value = if w_total > 0.0 {
            sum_total / w_total
        } else {
            0.0
        };

        let is_leaf = depth >= max_depth
            || n_node < opts.min_samples_split
            || n_node < 2 * opts.min_samples_leaf
            || node_impurity <= f64::EPSILON;
        let candidate = if is_leaf {
            None
        } else {
            best_split(
                &samples[start..end],
                cols,
                y,
                w,
                opts.min_samples_leaf,
                sum_total,
                w_total,
                max_features,
                draw,
            )
        };

        if let Some((pid, is_left)) = parent {
            if let Some(Some(s)) = nodes.get_mut(pid).map(|n| n.split.as_mut()) {
                if is_left {
                    s.left = node_id;
                } else {
                    s.right = node_id;
                }
            }
        }

        match candidate {
            None => nodes.push(Node {
                split: None,
                value,
                n_samples: n_node,
                weighted_n_samples: w_total,
                impurity: node_impurity,
                depth,
            }),
            Some(c) => {
                let n_left = partition(&mut samples[start..end], &cols[c.feature], c.threshold);
                let pos = start + n_left;
                let (sl, wl, sql) = node_stats(&samples[start..pos], y, w);
                let (sr, wr, sqr) = node_stats(&samples[pos..end], y, w);
                let imp_l = impurity(sl, wl, sql);
                let imp_r = impurity(sr, wr, sqr);
                raw_importance[c.feature] += w_total * node_impurity - wl * imp_l - wr * imp_r;
                nodes.push(Node {
                    split: Some(Split {
                        feature: c.feature,
                        threshold: c.threshold,
                        left: 0,
                        right: 0,
                    }),
                    value,
                    n_samples: n_node,
                    weighted_n_samples: w_total,
                    impurity: node_impurity,
                    depth,
                });
                // Right first so the left child is materialized first.
                stack.push(Pending {
                    start: pos,
                    end,
                    depth: depth + 1,
                    parent: Some((node_id, false)),
                    impurity: imp_r,
                });
                stack.push(Pending {
                    start,
                    end: pos,
                    depth: depth + 1,
                    parent: Some((node_id, true)),
                    impurity: imp_l,
                });
            }
        }
    }

    if root_w > 0.0 {
        for v in &mut raw_importance {
            *v /= root_w;
        }
    }
    let depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);
    let n_leaves = nodes.iter().filter(|n| n.split.is_none()).count();
    RegressionTree {
        nodes,
        n_features: p,
        depth,
        n_leaves,
        raw_importance,
    }
}

/// Validates `x`/`y`/`x_test` and the stopping controls shared by the tree
/// and the forest; returns `(n, p)`.
pub(crate) fn check_tree_inputs(
    x: MatRef<'_, f64>,
    y: &[f64],
    x_test: Option<MatRef<'_, f64>>,
    opts: TreeOptions,
) -> Result<(usize, usize), MlError> {
    let (n, p) = check_xy(x, y)?;
    if let Some(xt) = x_test {
        if xt.nrows() == 0 {
            return Err(MlError::EmptyInput { what: "x_test" });
        }
        if xt.ncols() != p {
            return Err(MlError::DimensionMismatch {
                what: "x_test must have the same number of columns as x",
                expected: p,
                got: xt.ncols(),
            });
        }
        for j in 0..p {
            for i in 0..xt.nrows() {
                if !xt[(i, j)].is_finite() {
                    return Err(MlError::NonFinite { what: "x_test" });
                }
            }
        }
    }
    if opts.min_samples_leaf == 0 {
        return Err(MlError::InvalidArgument {
            what: "min_samples_leaf must be at least 1",
        });
    }
    if opts.min_samples_split < 2 {
        return Err(MlError::InvalidArgument {
            what: "min_samples_split must be at least 2",
        });
    }
    let needed = 2usize
        .max(opts.min_samples_split)
        .max(2 * opts.min_samples_leaf);
    if n < needed {
        return Err(MlError::InsufficientData {
            got: n,
            needed,
            what: "tree fit",
        });
    }
    Ok((n, p))
}

/// Result of [`regression_tree`].
#[derive(Debug, Clone, PartialEq)]
pub struct TreeFit {
    /// Prediction for every training row (its leaf mean).
    pub fitted: Vec<f64>,
    /// Predictions for the rows of `x_test`, when given.
    pub predicted: Option<Vec<f64>>,
    /// Number of nodes, internal plus leaves.
    pub n_nodes: usize,
    /// Number of leaves.
    pub n_leaves: usize,
    /// Depth of the deepest leaf (root = 0).
    pub depth: usize,
    /// Normalized impurity-based importance per column of `x`.
    pub feature_importance: Vec<f64>,
    /// `(feature, threshold)` over the internal nodes, sorted.
    pub splits: Vec<(usize, f64)>,
    /// The fitted tree itself.
    pub tree: RegressionTree,
}

/// Fits a CART regression tree to `x` (`n x p`) and `y` (`n`) under the
/// scikit-learn best-split conventions described in the [module docs](self),
/// optionally predicting `x_test`.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] / [`MlError::DimensionMismatch`] /
///   [`MlError::NonFinite`] on malformed inputs (`x_test` must have `p`
///   columns and be finite);
/// * [`MlError::InvalidArgument`] if `min_samples_leaf < 1` or
///   `min_samples_split < 2`;
/// * [`MlError::InsufficientData`] if `n < max(2, min_samples_split,
///   2 * min_samples_leaf)` — a tree that could never split.
pub fn regression_tree(
    x: MatRef<'_, f64>,
    y: &[f64],
    opts: TreeOptions,
    x_test: Option<MatRef<'_, f64>>,
) -> Result<TreeFit, MlError> {
    let (n, p) = check_tree_inputs(x, y, x_test, opts)?;
    let cols = columns(x);
    let w = vec![1.0; n];
    let mut in_order = |k: usize, _p: usize| k;
    let tree = grow(&cols, y, &w, opts, p, &mut in_order);
    let fitted = tree.predict_cols(&cols, n);
    let predicted = x_test.map(|xt| tree.predict(xt));
    Ok(TreeFit {
        fitted,
        predicted,
        n_nodes: tree.n_nodes(),
        n_leaves: tree.n_leaves,
        depth: tree.depth,
        feature_importance: tree.feature_importance(),
        splits: tree.splits(),
        tree,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tsecon_linalg::faer::Mat;

    #[test]
    fn stump_splits_at_the_midpoint_and_predicts_side_means() {
        // x = 0,1,2,3 ; y = 0,0,10,10 -> split at 1.5, leaves 0 and 10.
        let x = Mat::from_fn(4, 1, |i, _| i as f64);
        let y = [0.0, 0.0, 10.0, 10.0];
        let fit = regression_tree(x.as_ref(), &y, TreeOptions::default(), None).unwrap();
        assert_eq!(fit.n_leaves, 2);
        assert_eq!(fit.depth, 1);
        assert_eq!(fit.splits, vec![(0, 1.5)]);
        assert_eq!(fit.fitted, vec![0.0, 0.0, 10.0, 10.0]);
        assert_eq!(fit.feature_importance, vec![1.0]);
    }

    #[test]
    fn constant_target_is_a_single_leaf() {
        let x = Mat::from_fn(5, 2, |i, j| (i * 2 + j) as f64);
        let y = [3.0; 5];
        let fit = regression_tree(x.as_ref(), &y, TreeOptions::default(), None).unwrap();
        assert_eq!(fit.n_nodes, 1);
        assert_eq!(fit.depth, 0);
        assert!(fit.splits.is_empty());
        assert_eq!(fit.feature_importance, vec![0.0, 0.0]);
    }

    #[test]
    fn insufficient_data_uses_house_wording() {
        let x = Mat::from_fn(3, 1, |i, _| i as f64);
        let y = [1.0, 2.0, 3.0];
        let opts = TreeOptions {
            max_depth: None,
            min_samples_leaf: 2,
            min_samples_split: 2,
        };
        let err = regression_tree(x.as_ref(), &y, opts, None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "insufficient data: 3 observations, at least 4 required (tree fit)"
        );
    }

    #[test]
    fn x_test_column_mismatch_names_both_counts() {
        let x = Mat::from_fn(6, 2, |i, j| (i + j) as f64);
        let y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let xt = Mat::from_fn(2, 3, |_, _| 0.0);
        let err =
            regression_tree(x.as_ref(), &y, TreeOptions::default(), Some(xt.as_ref())).unwrap_err();
        assert_eq!(
            err,
            MlError::DimensionMismatch {
                what: "x_test must have the same number of columns as x",
                expected: 2,
                got: 3
            }
        );
    }
}
