//! Variable importance for forests: impurity-based and grouped
//! block-permutation importance.
//!
//! # Impurity importance
//!
//! scikit-learn's `feature_importances_`: each tree's normalized impurity
//! decrease (see [`crate::tree`]) averaged over the trees and renormalized
//! to sum to one. On a single deterministic tree it equals scikit-learn's
//! number exactly (golden-pinned at 1e-10). It is biased toward columns
//! with many distinct values and credits whichever member of a correlated
//! group the split happened to pick.
//!
//! # Grouped block-permutation importance
//!
//! Breiman's (2001) permutation importance evaluated on the out-of-bag
//! rows, with two changes for serially dependent designs:
//!
//! * **Grouping** — `importance_groups` maps every column to a unit (all
//!   the lags of one variable, say), and a unit's columns are permuted
//!   *together*, row-wise. This is what keeps the permuted rows dynamically
//!   possible: permuting one lag of a persistent variable on its own
//!   creates `(x_{t-1}, x_{t-2})` pairs that never occur, and it also
//!   *dilutes* the variable's importance across its near-collinear lags —
//!   the forest simply routes through another lag when one is scrambled.
//! * **Block permutation** — the rows of a unit are permuted in
//!   contiguous blocks (blocks of `permutation_block` consecutive rows are
//!   shuffled as wholes), so the permuted columns keep their
//!   autocorrelation and the counterfactual design is a plausible series
//!   rather than white noise.
//!
//! What block permutation does **not** do, measured honestly: for a
//! row-wise model scored by a row-wise loss, the expected importance
//! depends only on which row each permuted value comes from, and both a
//! single-row and a block permutation pair row `t` with an essentially
//! uniform other row — so the *mean* importance is the same under both
//! (the property suite measures the two within noise of each other). What
//! changes is the variance of the estimate and the plausibility of the
//! counterfactual, not the level. In particular, block permutation does
//! **not** deflate the inflated importance a persistent but irrelevant
//! predictor picks up when the relevant predictors are persistent too:
//! that inflation lives in the fitted forest, which uses the irrelevant
//! series as a time proxy (in-sample, two persistent series are
//! correlated). The remedy there is a control comparison or conditional
//! importance, not a different permutation. The model card quotes the
//! measured numbers.
//!
//! The importance of a unit is the mean, over `n_permutations` block
//! permutations, of the out-of-bag MSE with the unit permuted minus the
//! out-of-bag MSE of the unpermuted forest — in the units of `y^2`, and
//! negative values are possible (an irrelevant unit whose permutation
//! happened to help).

use rayon::prelude::*;
use tsecon_rng::Stream;

use crate::error::MlError;
use crate::forest::uniform_index;
use crate::tree::RegressionTree;

/// Importance units: the resolved (sorted, distinct) group labels and the
/// member columns of each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Groups {
    /// Distinct labels in ascending order; `importance[k]` refers to
    /// `labels[k]`.
    pub labels: Vec<usize>,
    /// Columns of `x` belonging to each label.
    pub members: Vec<Vec<usize>>,
}

/// Resolves `importance_groups` (one integer label per column of `x`) into
/// importance units; `None` makes every column its own unit.
///
/// # Errors
///
/// [`MlError::DimensionMismatch`] if the label vector does not have one
/// entry per column.
pub(crate) fn resolve_groups(p: usize, groups: Option<&[usize]>) -> Result<Groups, MlError> {
    match groups {
        None => Ok(Groups {
            labels: (0..p).collect(),
            members: (0..p).map(|j| vec![j]).collect(),
        }),
        Some(g) => {
            if g.len() != p {
                return Err(MlError::DimensionMismatch {
                    what: "importance_groups must have one label per feature (column of x)",
                    expected: p,
                    got: g.len(),
                });
            }
            let mut labels: Vec<usize> = g.to_vec();
            labels.sort_unstable();
            labels.dedup();
            let members = labels
                .iter()
                .map(|&lab| (0..p).filter(|&j| g[j] == lab).collect())
                .collect();
            Ok(Groups { labels, members })
        }
    }
}

/// Mean of the trees' normalized impurity importances, renormalized to sum
/// to one, then summed within each unit. A single-tree forest returns the
/// tree's own normalized vector untouched (the renormalization of an
/// already-normalized vector would only add a rounding ulp, and the
/// single-tree bridge to `regression_tree` is held bit-for-bit).
pub(crate) fn impurity_importance(trees: &[RegressionTree], groups: &Groups) -> Vec<f64> {
    let p = trees.first().map_or(0, |t| t.n_features);
    let mut acc = vec![0.0; p];
    for t in trees {
        for (a, v) in acc.iter_mut().zip(t.feature_importance()) {
            *a += v;
        }
    }
    let total: f64 = acc.iter().sum();
    if trees.len() > 1 && total > 0.0 {
        for a in &mut acc {
            *a /= total;
        }
    }
    groups
        .members
        .iter()
        .map(|m| m.iter().map(|&j| acc[j]).sum())
        .collect()
}

/// A contiguous-block permutation of `0..n`: the rows are cut into blocks
/// `[b*block, (b+1)*block)` (the last one shorter when `block` does not
/// divide `n`), the block order is shuffled by Fisher-Yates, and the blocks
/// are concatenated. `block = 1` is an ordinary single-row permutation;
/// `block >= n` is the identity.
///
/// Draw order: one bounded uniform draw per Fisher-Yates step over the
/// `ceil(n / block)` blocks, from the last block down to the second.
pub fn block_permutation(n: usize, block: usize, stream: &mut Stream) -> Vec<usize> {
    let block = block.clamp(1, n.max(1));
    let n_blocks = n.div_ceil(block);
    let mut order: Vec<usize> = (0..n_blocks).collect();
    for k in (1..n_blocks).rev() {
        let j = uniform_index(stream, k + 1);
        order.swap(k, j);
    }
    let mut perm = Vec::with_capacity(n);
    for b in order {
        let start = b * block;
        let end = (start + block).min(n);
        perm.extend(start..end);
    }
    perm
}

/// Out-of-bag MSE of the forest with the columns in `members` replaced by
/// their `perm`-permuted values: row `t` reads `cols[j][perm[t]]` for a
/// member column and `cols[j][t]` otherwise. `oob_trees[t]` lists the
/// trees in which row `t` was out-of-bag; rows with none are skipped.
fn oob_mse_permuted(
    trees: &[RegressionTree],
    oob_trees: &[Vec<usize>],
    cols: &[Vec<f64>],
    y: &[f64],
    members: &[usize],
    perm: &[usize],
) -> f64 {
    let p = cols.len();
    let mut is_member = vec![false; p];
    for &j in members {
        if j < p {
            is_member[j] = true;
        }
    }
    let mut sse = 0.0;
    let mut count = 0usize;
    for (t, own) in oob_trees.iter().enumerate() {
        if own.is_empty() {
            continue;
        }
        let src = perm[t];
        let mut sum = 0.0;
        for &b in own {
            sum += trees[b].predict_with(|j| {
                if is_member[j] {
                    cols[j][src]
                } else {
                    cols[j][t]
                }
            });
        }
        let e = y[t] - sum / own.len() as f64;
        sse += e * e;
        count += 1;
    }
    if count > 0 {
        sse / count as f64
    } else {
        f64::NAN
    }
}

/// Grouped block-permutation importance (see the [module docs](self)):
/// for every unit, the mean over `n_permutations` block permutations of
/// the out-of-bag MSE increase over `base_mse` (the unpermuted out-of-bag
/// MSE, computed over the same rows).
///
/// All permutations are drawn sequentially from `stream` first; the
/// `(unit, replicate)` evaluations then run in parallel and are collected
/// in index order, so the result is independent of the thread count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_permutation_importance(
    trees: &[RegressionTree],
    oob_trees: &[Vec<usize>],
    cols: &[Vec<f64>],
    y: &[f64],
    groups: &Groups,
    block: usize,
    n_permutations: usize,
    stream: &mut Stream,
    base_mse: f64,
) -> Vec<f64> {
    let n = y.len();
    let n_units = groups.members.len();
    let perms: Vec<Vec<usize>> = (0..n_units * n_permutations)
        .map(|_| block_permutation(n, block, stream))
        .collect();
    let mses: Vec<f64> = perms
        .par_iter()
        .enumerate()
        .map(|(k, perm)| {
            let unit = k / n_permutations;
            oob_mse_permuted(trees, oob_trees, cols, y, &groups.members[unit], perm)
        })
        .collect();
    (0..n_units)
        .map(|u| {
            let slice = &mses[u * n_permutations..(u + 1) * n_permutations];
            slice.iter().map(|m| m - base_mse).sum::<f64>() / n_permutations as f64
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn block_permutation_is_a_permutation_with_intact_blocks() {
        let mut s = Stream::new(3);
        let n = 23;
        let block = 5;
        let perm = block_permutation(n, block, &mut s);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
        // Every block is placed whole: at each block position the run
        // starts on a block boundary and continues for the block's length
        // (the last block of the original order is the short one).
        let mut blocks = 0;
        let mut i = 0;
        while i < n {
            let start = perm[i];
            assert_eq!(start % block, 0, "block must start at a block boundary");
            let len = block.min(n - start);
            assert_eq!(
                &perm[i..i + len],
                &(start..start + len).collect::<Vec<_>>()[..]
            );
            blocks += 1;
            i += len;
        }
        assert_eq!(blocks, n.div_ceil(block));
    }

    #[test]
    fn single_row_block_is_a_plain_permutation_and_full_block_is_identity() {
        let mut s = Stream::new(11);
        let p1 = block_permutation(10, 1, &mut s);
        let mut sorted = p1.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
        let id = block_permutation(10, 10, &mut s);
        assert_eq!(id, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn groups_resolve_sorted_distinct_labels() {
        let g = resolve_groups(5, Some(&[7, 2, 7, 2, 9])).unwrap();
        assert_eq!(g.labels, vec![2, 7, 9]);
        assert_eq!(g.members, vec![vec![1, 3], vec![0, 2], vec![4]]);
        let err = resolve_groups(5, Some(&[1, 2])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "dimension mismatch: importance_groups must have one label per feature \
             (column of x) (expected 5, got 2)"
        );
    }
}
