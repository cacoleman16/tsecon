//! Fry-Pagan (2011) median-target rotation.
//!
//! Sign restrictions **set-identify**: the accepted rotations trace out a set
//! of mutually observationally-equivalent structural models, not a point. The
//! pointwise median band that [`crate::SignSampler`] reports is a convenient
//! summary of that set, but — as Fry & Pagan (2011, *J. Monetary Economics*)
//! stress — it is **not itself a model**: at each `(variable, shock, horizon)`
//! it takes the median *across draws*, so the "median IRF" stitches together
//! responses drawn from different, mutually inconsistent rotations. No single
//! structural model need produce it.
//!
//! The median-target answer is to report, instead of the synthetic median, the
//! single *accepted* draw whose structural IRFs come **jointly closest** to
//! that median — a genuine, internally-coherent model that is as central as the
//! accepted set allows. "Closest" is the Fry-Pagan criterion: the sum, over a
//! chosen set of target cells, of the squared deviations of a draw's responses
//! from the pointwise median, each standardized by that cell's dispersion
//! across draws so cells on different scales contribute comparably.
//!
//! # The estimand is a descriptive summary, not a point estimate
//!
//! The selected draw is one interior point of the identified set; which point
//! it is depends on the (Haar-)informative sampling prior that generated the
//! accepted set (Baumeister & Hamilton 2015; see the crate-level docs). The
//! *selection rule* here is exact and reproducible, but the object it selects
//! inherits the set-identification caveat: read it as the most-central coherent
//! model in the accepted set, not as a prior-free structural estimate.

use tsecon_linalg::faer::Mat;

use crate::error::IdentError;
use crate::summary::quantile_sorted;

/// The outcome of a Fry-Pagan median-target selection.
#[derive(Debug, Clone)]
pub struct MedianTargetResult {
    /// Zero-based index, into the supplied accepted-draw set, of the selected
    /// median-target draw `d*`.
    pub index: usize,
    /// The median-target criterion at the winner,
    /// `MT(d*) = sum_cells ((Theta^(d*)[cell] - med[cell]) / sd[cell])^2`,
    /// summed over the target cells with non-degenerate dispersion.
    pub statistic: f64,
    /// The pointwise (per-cell) median structural IRF across all accepted
    /// draws, indexed `[horizon]`, each `n x n`. This is the **incoherent**
    /// median band (it mixes models); it is returned only for a side-by-side
    /// comparison with the coherent selected draw.
    pub median_irf: Vec<Mat<f64>>,
}

/// Fry-Pagan (2011) median-target selection over a set of accepted structural
/// IRF draws.
///
/// `draws` is indexed `[draw][horizon]`, each an `n x n` matrix whose `(i, j)`
/// entry is the sign-normalized response of variable `i` to structural shock
/// `j` at that horizon — exactly the accepted set
/// [`crate::SignSampleResult::draws`] emits. `target_cells` lists the
/// `(variable i, shock j, horizon h)` cells the criterion is summed over
/// (typically every response variable and horizon of the sign-restricted
/// shocks).
///
/// For each target cell the routine forms the pointwise median `med` (NumPy
/// type-7, via [`quantile_sorted`]) and the pointwise population standard
/// deviation `sd` across the draws. Draw `d`'s criterion is
/// `MT(d) = sum_cells ((Theta^(d)[cell] - med) / sd)^2`; a cell with `sd == 0`
/// contributes nothing (its standardized deviation is undefined and, since
/// every draw carries the identical value there, uninformative). The selected
/// draw is `d* = argmin_d MT(d)`, ties resolved to the lowest index. The
/// returned draw is therefore always a member of the accepted set and always
/// minimizes the criterion — a single coherent model, not a mix.
///
/// The population (rather than sample) standard deviation is used, matching
/// `numpy.std`'s default; because every cell is standardized by the same draw
/// count, the choice of divisor rescales every `MT(d)` by one common constant
/// and so never changes `argmin` — it only fixes the reported `statistic` value.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `draws` is empty, `target_cells` is
///   empty, or a draw has zero horizon matrices;
/// * [`IdentError::Dimension`] if the draws disagree on their horizon length or
///   any IRF matrix is not `n x n`;
/// * [`IdentError::RestrictionOutOfRange`] if a target cell references a
///   variable, shock, or horizon outside the draws' dimensions.
pub fn median_target(
    draws: &[Vec<Mat<f64>>],
    target_cells: &[(usize, usize, usize)],
) -> Result<MedianTargetResult, IdentError> {
    if draws.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "median_target requires at least one accepted draw",
        });
    }
    if target_cells.is_empty() {
        return Err(IdentError::InvalidArgument {
            what: "median_target requires at least one target cell",
        });
    }
    let n_h = draws[0].len();
    if n_h == 0 {
        return Err(IdentError::InvalidArgument {
            what: "each draw must contain at least one IRF horizon matrix",
        });
    }
    let n = draws[0][0].nrows();
    if draws[0][0].ncols() != n {
        return Err(IdentError::Dimension {
            what: "structural IRF matrices must be square",
            expected: n,
            got: draws[0][0].ncols(),
        });
    }

    // Every draw must share the (horizon length, n, n) shape so the pointwise
    // stack is well defined.
    for d in draws {
        if d.len() != n_h {
            return Err(IdentError::Dimension {
                what: "all draws must share the same horizon length",
                expected: n_h,
                got: d.len(),
            });
        }
        for m in d {
            if m.nrows() != n {
                return Err(IdentError::Dimension {
                    what: "all IRF matrices must have n rows",
                    expected: n,
                    got: m.nrows(),
                });
            }
            if m.ncols() != n {
                return Err(IdentError::Dimension {
                    what: "all IRF matrices must have n columns",
                    expected: n,
                    got: m.ncols(),
                });
            }
        }
    }

    for &(i, j, h) in target_cells {
        if i >= n {
            return Err(IdentError::RestrictionOutOfRange {
                what: "response variable",
                index: i,
                bound: n,
            });
        }
        if j >= n {
            return Err(IdentError::RestrictionOutOfRange {
                what: "structural shock",
                index: j,
                bound: n,
            });
        }
        if h >= n_h {
            return Err(IdentError::RestrictionOutOfRange {
                what: "horizon",
                index: h,
                bound: n_h,
            });
        }
    }

    let d_count = draws.len();
    let cell = |h: usize, i: usize, j: usize| (h * n + i) * n + j;

    // Pointwise median over EVERY cell — the returned (incoherent) band, and
    // the standardization centre for the target cells.
    let mut med_all = vec![0.0f64; n_h * n * n];
    let mut buf: Vec<f64> = Vec::with_capacity(d_count);
    for h in 0..n_h {
        for i in 0..n {
            for j in 0..n {
                buf.clear();
                for d in draws {
                    buf.push(d[h][(i, j)]);
                }
                buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                med_all[cell(h, i, j)] = quantile_sorted(&buf, 0.5);
            }
        }
    }
    let median_irf: Vec<Mat<f64>> = (0..n_h)
        .map(|h| Mat::from_fn(n, n, |i, j| med_all[cell(h, i, j)]))
        .collect();

    // Precompute (i, j, h, med, sd) for each target cell that carries genuine
    // dispersion; degenerate (sd == 0) cells drop out of the criterion.
    let mut active: Vec<(usize, usize, usize, f64, f64)> = Vec::with_capacity(target_cells.len());
    for &(i, j, h) in target_cells {
        let med = med_all[cell(h, i, j)];
        let mut mean = 0.0f64;
        for d in draws {
            mean += d[h][(i, j)];
        }
        mean /= d_count as f64;
        let mut var = 0.0f64;
        for d in draws {
            let e = d[h][(i, j)] - mean;
            var += e * e;
        }
        var /= d_count as f64;
        let sd = var.sqrt();
        if sd > 0.0 {
            active.push((i, j, h, med, sd));
        }
    }

    // MT(d) = sum of squared standardized deviations from the pointwise median.
    // Strict `<` keeps the lowest index on a tie.
    let mut best_idx = 0usize;
    let mut best_mt = f64::INFINITY;
    for (d_idx, d) in draws.iter().enumerate() {
        let mut mt = 0.0f64;
        for &(i, j, h, med, sd) in &active {
            let z = (d[h][(i, j)] - med) / sd;
            mt += z * z;
        }
        if mt < best_mt {
            best_mt = mt;
            best_idx = d_idx;
        }
    }
    // With no active cell every draw scores 0; `best_mt` then settles at 0.
    if !best_mt.is_finite() {
        best_mt = 0.0;
    }

    Ok(MedianTargetResult {
        index: best_idx,
        statistic: best_mt,
        median_irf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a draw set from a `[D][H+1]` list of flat row-major `n x n`
    /// blocks.
    fn draws_from(blocks: &[Vec<Vec<f64>>], n: usize) -> Vec<Vec<Mat<f64>>> {
        blocks
            .iter()
            .map(|draw| {
                draw.iter()
                    .map(|flat| Mat::from_fn(n, n, |i, j| flat[i * n + j]))
                    .collect()
            })
            .collect()
    }

    /// A single 1x1 cell across three draws: median-target must pick the draw
    /// nearest the pointwise median.
    #[test]
    fn picks_the_draw_nearest_the_median() -> Result<(), IdentError> {
        // Values 0, 1, 2 -> median 1 -> the middle draw (index 1) is the winner.
        let blocks = vec![vec![vec![0.0]], vec![vec![1.0]], vec![vec![2.0]]];
        let draws = draws_from(&blocks, 1);
        let out = median_target(&draws, &[(0, 0, 0)])?;
        assert_eq!(out.index, 1);
        assert!(out.statistic.abs() < 1e-15, "central draw scores 0");
        Ok(())
    }

    /// The returned draw always minimizes the criterion, recomputed here
    /// independently of the core.
    #[test]
    fn winner_minimizes_the_criterion() -> Result<(), IdentError> {
        let n = 2;
        // Four draws, two horizons, arbitrary spread.
        let blocks = vec![
            vec![vec![0.5, -1.0, 0.2, 0.3], vec![0.1, 0.0, -0.4, 0.9]],
            vec![vec![1.5, -0.2, 0.9, 0.1], vec![0.6, 0.2, -0.1, 0.5]],
            vec![vec![-0.3, 0.7, 0.4, 0.8], vec![0.2, -0.5, 0.3, 0.0]],
            vec![vec![0.4, -0.4, 0.5, 0.35], vec![0.15, -0.1, -0.2, 0.6]],
        ];
        let draws = draws_from(&blocks, n);
        // Target every cell of shock 0 across both horizons.
        let mut cells = Vec::new();
        for h in 0..2 {
            for i in 0..n {
                cells.push((i, 0usize, h));
            }
        }
        let out = median_target(&draws, &cells)?;

        // Independent recomputation of MT for every draw.
        let mt_of = |d_idx: usize| -> f64 {
            let mut mt = 0.0;
            for &(i, j, h) in &cells {
                let mut vals: Vec<f64> = draws.iter().map(|d| d[h][(i, j)]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                let med = quantile_sorted(&vals, 0.5);
                let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
                let var: f64 =
                    vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
                let sd = var.sqrt();
                if sd > 0.0 {
                    let z = (draws[d_idx][h][(i, j)] - med) / sd;
                    mt += z * z;
                }
            }
            mt
        };
        let best = (0..draws.len()).map(mt_of).fold(f64::INFINITY, f64::min);
        assert!(
            (mt_of(out.index) - best).abs() < 1e-12,
            "returned index {} does not attain the minimum criterion",
            out.index
        );
        // No other draw beats it.
        for d in 0..draws.len() {
            assert!(mt_of(d) >= mt_of(out.index) - 1e-12);
        }
        assert!((out.statistic - mt_of(out.index)).abs() < 1e-12);
        Ok(())
    }

    /// Ties resolve to the lowest index.
    #[test]
    fn ties_go_to_the_lowest_index() -> Result<(), IdentError> {
        // Two symmetric draws about the median: MT ties -> index 0.
        let blocks = vec![vec![vec![1.0]], vec![vec![-1.0]]];
        let draws = draws_from(&blocks, 1);
        let out = median_target(&draws, &[(0, 0, 0)])?;
        assert_eq!(out.index, 0, "equal criteria must select the lowest index");
        // Both draws sit one sd from the median (sd = 1) -> MT = 1.
        assert!((out.statistic - 1.0).abs() < 1e-12);
        Ok(())
    }

    /// The pointwise median band is exactly the per-cell median.
    #[test]
    fn median_irf_is_the_pointwise_median() -> Result<(), IdentError> {
        let n = 2;
        let blocks = vec![
            vec![vec![0.0, 1.0, 2.0, 3.0]],
            vec![vec![4.0, 5.0, 6.0, 7.0]],
            vec![vec![8.0, 9.0, 10.0, 11.0]],
        ];
        let draws = draws_from(&blocks, n);
        let out = median_target(&draws, &[(0, 0, 0)])?;
        // Odd count -> median is the middle draw's value, cell by cell.
        for i in 0..n {
            for j in 0..n {
                assert!((out.median_irf[0][(i, j)] - draws[1][0][(i, j)]).abs() < 1e-15);
            }
        }
        Ok(())
    }

    /// Determinism: identical inputs give identical results.
    #[test]
    fn is_deterministic() -> Result<(), IdentError> {
        let n = 2;
        let blocks = vec![
            vec![vec![0.5, -1.0, 0.2, 0.3]],
            vec![vec![1.5, -0.2, 0.9, 0.1]],
            vec![vec![-0.3, 0.7, 0.4, 0.8]],
        ];
        let draws = draws_from(&blocks, n);
        let cells = vec![(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
        let a = median_target(&draws, &cells)?;
        let b = median_target(&draws, &cells)?;
        assert_eq!(a.index, b.index);
        assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
        Ok(())
    }

    #[test]
    fn rejects_bad_inputs() {
        let n = 2;
        let blocks = vec![
            vec![vec![0.5, -1.0, 0.2, 0.3]],
            vec![vec![1.5, -0.2, 0.9, 0.1]],
        ];
        let draws = draws_from(&blocks, n);

        // Empty draws.
        assert!(matches!(
            median_target(&[], &[(0, 0, 0)]),
            Err(IdentError::InvalidArgument { .. })
        ));
        // Empty target cells.
        assert!(matches!(
            median_target(&draws, &[]),
            Err(IdentError::InvalidArgument { .. })
        ));
        // Response variable out of range.
        assert!(matches!(
            median_target(&draws, &[(2, 0, 0)]),
            Err(IdentError::RestrictionOutOfRange { .. })
        ));
        // Shock out of range.
        assert!(matches!(
            median_target(&draws, &[(0, 2, 0)]),
            Err(IdentError::RestrictionOutOfRange { .. })
        ));
        // Horizon out of range (only horizon 0 exists here).
        assert!(matches!(
            median_target(&draws, &[(0, 0, 1)]),
            Err(IdentError::RestrictionOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_ragged_draws() {
        let n = 2;
        // Second draw has two horizon blocks, first has one -> inconsistent.
        let good: Vec<Mat<f64>> = vec![Mat::from_fn(n, n, |_, _| 1.0)];
        let ragged: Vec<Mat<f64>> = vec![
            Mat::from_fn(n, n, |_, _| 1.0),
            Mat::from_fn(n, n, |_, _| 2.0),
        ];
        let draws = vec![good, ragged];
        assert!(matches!(
            median_target(&draws, &[(0, 0, 0)]),
            Err(IdentError::Dimension { .. })
        ));
    }
}
