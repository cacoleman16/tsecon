//! Shared post-identification helpers: structural IRF construction from a
//! rotation or an arbitrary impact matrix, per-shock sign normalization, and
//! pointwise (optionally importance-weighted) band summaries of a set of
//! accepted structural IRF draws.
//!
//! These were lifted out of `sampler.rs` so every identification scheme that
//! produces structural IRF draws — the Uhlig/ARWZ sign sampler, the
//! forthcoming FEVD/historical-decomposition/robust-bounds tools, and the
//! narrative sign-restriction sampler — reuses one vetted NumPy type-7
//! quantile and one summary code path instead of re-deriving them (and
//! silently disagreeing at the last ULP). The unweighted [`summarize`] path
//! is byte-for-byte the original sampler summary; the [`summarize_weighted`]
//! path adds importance weights on the quantiles (the identified-set min/max
//! stay weight-free) and reduces to the unweighted path when every weight is
//! equal.

use tsecon_linalg::companion_from_var;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::IdentError;
use crate::sampler::{IrfBandPoint, StructuralIrfSummary};

/// Candidate structural IRF `Theta_h = Theta^chol_h Q` (post-multiplying each
/// horizon matrix by the rotation `Q`).
pub(crate) fn structural_irf(base: &[Mat<f64>], q: MatRef<'_, f64>) -> Vec<Mat<f64>> {
    base.iter().map(|m| m.as_ref() * q).collect()
}

/// Applies per-shock sign orientations in place: column `j` is scaled by
/// `orient[j]` at every horizon.
pub(crate) fn normalize(mut irf: Vec<Mat<f64>>, orient: &[f64]) -> Vec<Mat<f64>> {
    let n = orient.len();
    for m in irf.iter_mut() {
        for (j, &s) in orient.iter().enumerate().take(n) {
            if s != 1.0 {
                for i in 0..n {
                    m[(i, j)] *= s;
                }
            }
        }
    }
    irf
}

/// General-impact reduced-form-to-structural MA weights: `Theta_h = Psi_h A0`
/// for an *arbitrary* structural impact matrix `A0 = impact` (columns are
/// one-standard-deviation structural impulse vectors).
///
/// This generalizes `tsecon_bayes::cholesky_irf`, which hardcodes
/// `impact = chol(Sigma)`; passing `impact = chol(Sigma)` here reproduces it
/// exactly. The reduced-form MA weights `Psi_h = J F^h J'` are built from the
/// companion matrix `F` with the same regressor convention `cholesky_irf`
/// uses (`b` is `k x n`, `k = 1 + n p`, intercept row then lag blocks;
/// `A_l[(i, j)]` is the coefficient of `y_{t-l, j}` in equation `i`).
///
/// The returned vector has length `horizon + 1` with `Theta_0 = impact`.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `p == 0`;
/// * [`IdentError::Dimension`] if `impact` is not square or `b` is not
///   `(1 + n p) x n`;
/// * [`IdentError::NonFinite`] on any NaN/infinite entry in `b` or `impact`;
/// * [`IdentError::Linalg`] if the companion construction fails.
// Shared prerequisite for the forthcoming structural_fevd / historical_-
// decomposition / robust_svar_bounds modules, which supply the first live
// callers; wired ahead of them per the recommended build order.
#[allow(dead_code)]
pub(crate) fn structural_ma(
    b: MatRef<'_, f64>,
    impact: MatRef<'_, f64>,
    p: usize,
    horizon: usize,
) -> Result<Vec<Mat<f64>>, IdentError> {
    let n = impact.nrows();
    if impact.ncols() != n {
        return Err(IdentError::Dimension {
            what: "structural impact matrix must be square",
            expected: n,
            got: impact.ncols(),
        });
    }
    if p == 0 {
        return Err(IdentError::InvalidArgument {
            what: "lag length p must be at least 1",
        });
    }
    let k = 1 + n * p;
    if b.nrows() != k {
        return Err(IdentError::Dimension {
            what: "b must have 1 + n*p rows (intercept plus lag blocks)",
            expected: k,
            got: b.nrows(),
        });
    }
    if b.ncols() != n {
        return Err(IdentError::Dimension {
            what: "b must have one column per variable",
            expected: n,
            got: b.ncols(),
        });
    }
    for j in 0..n {
        for i in 0..k {
            if !b[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "b" });
            }
        }
    }
    for j in 0..n {
        for i in 0..n {
            if !impact[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "impact" });
            }
        }
    }

    // A_l[(i, j)] = coefficient of y_{t-l, j} in equation i.
    let coef_mats: Vec<Mat<f64>> = (1..=p)
        .map(|l| Mat::from_fn(n, n, |i, j| b[(1 + (l - 1) * n + j, i)]))
        .collect();
    let coef_refs: Vec<MatRef<'_, f64>> = coef_mats.iter().map(|m| m.as_ref()).collect();
    let companion = companion_from_var(&coef_refs)?;

    let mut out = Vec::with_capacity(horizon + 1);
    let np = n * p;
    let mut f_pow = Mat::<f64>::identity(np, np);
    for _h in 0..=horizon {
        let psi = Mat::from_fn(n, n, |i, j| f_pow[(i, j)]);
        out.push(psi.as_ref() * impact);
        f_pow = companion.as_ref() * f_pow.as_ref();
    }
    Ok(out)
}

/// Type-7 (NumPy default) linear-interpolation quantile of a sorted slice.
pub(crate) fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let len = sorted.len();
    if len == 0 {
        return f64::NAN;
    }
    if len == 1 {
        return sorted[0];
    }
    let pos = p * (len - 1) as f64;
    let lo = pos.floor() as usize;
    if lo >= len - 1 {
        return sorted[len - 1];
    }
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
}

/// Weighted type-7 quantile of paired (value, weight) data, with `sorted`
/// ascending and `weights` aligned entry-for-entry.
///
/// Uses cumulative-midpoint plotting positions `m_k = S_k - w_k / 2` (the
/// Hazen convention, `S_k` the cumulative weight) rescaled to `[0, 1]` by
/// `P_k = (m_k - m_1) / (m_n - m_1)`. This construction reduces **exactly** to
/// the type-7 positions `(k - 1) / (len - 1)` of [`quantile_sorted`] when all
/// weights are equal, so importance-weighted summaries collapse to the
/// unweighted path (to rounding) as every weight tends to uniform — the
/// regression contract the sign sampler and the narrative sampler share.
///
/// Degenerate inputs (non-positive total weight, or a zero-width position
/// span) fall back to the unweighted [`quantile_sorted`].
// First live caller is the forthcoming narrative sign-restriction sampler;
// wired ahead of it (shared-summary prerequisite).
#[allow(dead_code)]
pub(crate) fn weighted_quantile_sorted(sorted: &[f64], weights: &[f64], p: f64) -> f64 {
    let len = sorted.len();
    if len == 0 {
        return f64::NAN;
    }
    if len == 1 {
        return sorted[0];
    }
    if weights.len() != len {
        return quantile_sorted(sorted, p);
    }

    let mut cum = 0.0f64;
    let mut mids = Vec::with_capacity(len);
    for &w in weights {
        let w = if w > 0.0 { w } else { 0.0 };
        cum += w;
        mids.push(cum - 0.5 * w);
    }
    if cum <= 0.0 {
        return quantile_sorted(sorted, p);
    }
    let m_first = mids[0];
    let span = mids[len - 1] - m_first;
    if span <= 0.0 {
        return quantile_sorted(sorted, p);
    }
    if p <= 0.0 {
        return sorted[0];
    }
    if p >= 1.0 {
        return sorted[len - 1];
    }

    let target = m_first + p * span;
    let mut k = 0usize;
    for (idx, &m) in mids.iter().enumerate() {
        if m <= target {
            k = idx;
        } else {
            break;
        }
    }
    if k >= len - 1 {
        return sorted[len - 1];
    }
    let denom = mids[k + 1] - mids[k];
    if denom <= 0.0 {
        return sorted[k + 1];
    }
    let frac = (target - mids[k]) / denom;
    sorted[k] + frac * (sorted[k + 1] - sorted[k])
}

/// Builds the per-cell min/max/quantile summary from the accepted draws
/// (equal-weight; the sign sampler's original path, preserved byte-for-byte).
pub(crate) fn summarize(
    draws: &[Vec<Mat<f64>>],
    n_vars: usize,
    horizon: usize,
    probs: &[f64],
) -> StructuralIrfSummary {
    let n_cells = (horizon + 1) * n_vars * n_vars;
    let mut points = Vec::with_capacity(n_cells);

    if draws.is_empty() {
        for _ in 0..n_cells {
            points.push(IrfBandPoint {
                min: f64::NAN,
                max: f64::NAN,
                quantiles: vec![f64::NAN; probs.len()],
            });
        }
        return StructuralIrfSummary::from_parts(n_vars, horizon, probs.to_vec(), points);
    }

    let mut values = Vec::with_capacity(draws.len());
    for h in 0..=horizon {
        for i in 0..n_vars {
            for j in 0..n_vars {
                values.clear();
                for d in draws {
                    values.push(d[h][(i, j)]);
                }
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                let min = values[0];
                let max = values[values.len() - 1];
                let quantiles = probs.iter().map(|&p| quantile_sorted(&values, p)).collect();
                points.push(IrfBandPoint {
                    min,
                    max,
                    quantiles,
                });
            }
        }
    }

    StructuralIrfSummary::from_parts(n_vars, horizon, probs.to_vec(), points)
}

/// Importance-weighted per-cell summary: the identified-set min/max are
/// **weight-free** (envelope over the accepted support), while the pointwise
/// quantiles are the [`weighted_quantile_sorted`] bands using `weights` (one
/// per draw, aligned to `draws`). Missing weights default to `1.0`; passing
/// all-equal weights reproduces [`summarize`] to rounding.
// First live caller is the forthcoming narrative sign-restriction sampler;
// wired ahead of it (shared-summary prerequisite).
#[allow(dead_code)]
pub(crate) fn summarize_weighted(
    draws: &[Vec<Mat<f64>>],
    weights: &[f64],
    n_vars: usize,
    horizon: usize,
    probs: &[f64],
) -> StructuralIrfSummary {
    let n_cells = (horizon + 1) * n_vars * n_vars;
    let mut points = Vec::with_capacity(n_cells);

    if draws.is_empty() {
        for _ in 0..n_cells {
            points.push(IrfBandPoint {
                min: f64::NAN,
                max: f64::NAN,
                quantiles: vec![f64::NAN; probs.len()],
            });
        }
        return StructuralIrfSummary::from_parts(n_vars, horizon, probs.to_vec(), points);
    }

    let d = draws.len();
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(d);
    let mut vals: Vec<f64> = Vec::with_capacity(d);
    let mut wts: Vec<f64> = Vec::with_capacity(d);
    for h in 0..=horizon {
        for i in 0..n_vars {
            for j in 0..n_vars {
                pairs.clear();
                for (idx, draw) in draws.iter().enumerate() {
                    let w = weights.get(idx).copied().unwrap_or(1.0);
                    pairs.push((draw[h][(i, j)], w));
                }
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
                let min = pairs[0].0;
                let max = pairs[pairs.len() - 1].0;
                vals.clear();
                wts.clear();
                for &(v, w) in &pairs {
                    vals.push(v);
                    wts.push(w);
                }
                let quantiles = probs
                    .iter()
                    .map(|&p| weighted_quantile_sorted(&vals, &wts, p))
                    .collect();
                points.push(IrfBandPoint {
                    min,
                    max,
                    quantiles,
                });
            }
        }
    }

    StructuralIrfSummary::from_parts(n_vars, horizon, probs.to_vec(), points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsecon_bayes::cholesky_irf;

    /// A small stable VAR(1) coefficient matrix in the crate's regressor
    /// layout (`k = 1 + n` rows: intercept then the lag-1 block) and a
    /// positive-definite covariance `Sigma = A A'` with `A` lower triangular.
    fn toy_var() -> (Mat<f64>, Mat<f64>) {
        let n = 3;
        let phi = [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]];
        let mut b = Mat::<f64>::zeros(1 + n, n);
        for i in 0..n {
            for v in 0..n {
                b[(1 + v, i)] = phi[i][v];
            }
        }
        let a = [[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.3, 0.7]];
        let sigma = Mat::from_fn(n, n, |i, j| {
            a[i].iter().zip(a[j].iter()).map(|(x, y)| x * y).sum()
        });
        (b, sigma)
    }

    /// The lower-triangular Cholesky factor `A` behind [`toy_var`]'s `Sigma`.
    fn toy_chol() -> Mat<f64> {
        let a = [[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.3, 0.7]];
        Mat::from_fn(3, 3, |i, j| a[i][j])
    }

    #[test]
    fn identity_rotation_reproduces_cholesky_irf() -> Result<(), IdentError> {
        // Q = I must leave the Cholesky IRF unchanged to 1e-12.
        let (b, sigma) = toy_var();
        let horizon = 10;
        let base = cholesky_irf(b.as_ref(), sigma.as_ref(), 1, horizon)?;
        let eye = Mat::<f64>::identity(3, 3);
        let rotated = structural_irf(&base, eye.as_ref());
        assert_eq!(rotated.len(), base.len());
        for (r, o) in rotated.iter().zip(base.iter()) {
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (r[(i, j)] - o[(i, j)]).abs() < 1e-12,
                        "identity-rotated IRF differs from cholesky_irf at ({i},{j})"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn normalize_flips_only_marked_columns() -> Result<(), IdentError> {
        let (b, sigma) = toy_var();
        let base = cholesky_irf(b.as_ref(), sigma.as_ref(), 1, 3)?;
        let orient = [-1.0, 1.0, -1.0];
        let flipped = normalize(base.clone(), &orient);
        for h in 0..base.len() {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((flipped[h][(i, j)] - orient[j] * base[h][(i, j)]).abs() < 1e-15);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn quantile_matches_numpy_type7() {
        // NumPy: np.quantile([1,2,3,4], [0,0.25,0.5,0.75,1]) =
        // [1, 1.75, 2.5, 3.25, 4].
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile_sorted(&xs, 0.0) - 1.0).abs() < 1e-15);
        assert!((quantile_sorted(&xs, 0.25) - 1.75).abs() < 1e-15);
        assert!((quantile_sorted(&xs, 0.5) - 2.5).abs() < 1e-15);
        assert!((quantile_sorted(&xs, 0.75) - 3.25).abs() < 1e-15);
        assert!((quantile_sorted(&xs, 1.0) - 4.0).abs() < 1e-15);
    }

    #[test]
    fn weighted_quantile_reduces_to_type7_under_equal_weights() {
        let xs = [1.0, 2.0, 3.5, 4.0, 9.0];
        let w = [1.0; 5];
        for &p in &[0.0, 0.05, 0.16, 0.25, 0.5, 0.84, 0.95, 1.0] {
            let unw = quantile_sorted(&xs, p);
            let wtd = weighted_quantile_sorted(&xs, &w, p);
            assert!(
                (unw - wtd).abs() < 1e-12,
                "weighted quantile at p={p} ({wtd}) must match type-7 ({unw})"
            );
        }
    }

    #[test]
    fn weighted_quantile_golden_and_direction() {
        // Symmetric: the middle value carrying double weight pulls the
        // quartiles inward to a hand-computed golden.
        let xs = [1.0, 2.0, 3.0];
        let w = [1.0, 2.0, 1.0];
        assert!((weighted_quantile_sorted(&xs, &w, 0.25) - 1.5).abs() < 1e-12);
        assert!((weighted_quantile_sorted(&xs, &w, 0.5) - 2.0).abs() < 1e-12);
        assert!((weighted_quantile_sorted(&xs, &w, 0.75) - 2.5).abs() < 1e-12);
        // Heavy weight on the largest value drags the median above 2.0.
        let w2 = [1.0, 1.0, 10.0];
        let med = weighted_quantile_sorted(&xs, &w2, 0.5);
        assert!(
            med > 2.0,
            "up-weighting the largest value must raise the median (got {med})"
        );
    }

    #[test]
    fn structural_ma_matches_cholesky_when_impact_is_chol() -> Result<(), IdentError> {
        // Theta_h = Psi_h * chol(Sigma) is exactly cholesky_irf.
        let (b, sigma) = toy_var();
        let a_mat = toy_chol();
        let horizon = 8;
        let chol = cholesky_irf(b.as_ref(), sigma.as_ref(), 1, horizon)?;
        let ma = structural_ma(b.as_ref(), a_mat.as_ref(), 1, horizon)?;
        assert_eq!(chol.len(), ma.len());
        for (c, m) in chol.iter().zip(ma.iter()) {
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (c[(i, j)] - m[(i, j)]).abs() < 1e-10,
                        "structural_ma with impact=chol(Sigma) must match cholesky_irf at ({i},{j})"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn structural_ma_identity_impact_gives_reduced_form_ma() -> Result<(), IdentError> {
        // impact = I yields the raw reduced-form MA weights Psi_h; with
        // Sigma = I, cholesky_irf returns the same thing.
        let (b, _) = toy_var();
        let eye = Mat::<f64>::identity(3, 3);
        let horizon = 6;
        let ma = structural_ma(b.as_ref(), eye.as_ref(), 1, horizon)?;
        let chol = cholesky_irf(b.as_ref(), eye.as_ref(), 1, horizon)?;
        for (c, m) in chol.iter().zip(ma.iter()) {
            for i in 0..3 {
                for j in 0..3 {
                    assert!((c[(i, j)] - m[(i, j)]).abs() < 1e-12);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn structural_ma_rejects_bad_shapes() {
        let (b, _) = toy_var(); // b is (1 + 3) x 3, valid for p = 1.
        let bad_impact = Mat::<f64>::zeros(3, 2);
        assert!(matches!(
            structural_ma(b.as_ref(), bad_impact.as_ref(), 1, 4),
            Err(IdentError::Dimension { .. })
        ));
        let impact = Mat::<f64>::identity(3, 3);
        // p = 2 needs 1 + 3*2 = 7 rows; b has 4.
        assert!(matches!(
            structural_ma(b.as_ref(), impact.as_ref(), 2, 4),
            Err(IdentError::Dimension { .. })
        ));
        assert!(matches!(
            structural_ma(b.as_ref(), impact.as_ref(), 0, 4),
            Err(IdentError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn summarize_weighted_uniform_matches_unweighted() -> Result<(), IdentError> {
        let n = 2;
        let horizon = 1;
        let make = |vals: [[f64; 4]; 2]| -> Vec<Mat<f64>> {
            vals.iter()
                .map(|m| Mat::from_fn(n, n, |i, j| m[i * n + j]))
                .collect()
        };
        let draws = vec![
            make([[0.5, -1.0, 0.2, 0.3], [0.1, 0.0, -0.4, 0.9]]),
            make([[1.5, -0.2, 0.9, 0.1], [0.6, 0.2, -0.1, 0.5]]),
            make([[-0.3, 0.7, 0.4, 0.8], [0.2, -0.5, 0.3, 0.0]]),
        ];
        let probs = [0.05, 0.5, 0.95];
        let unw = summarize(&draws, n, horizon, &probs);
        let weights = vec![1.0; draws.len()];
        let wtd = summarize_weighted(&draws, &weights, n, horizon, &probs);
        for h in 0..=horizon {
            for i in 0..n {
                for j in 0..n {
                    let a = unw.point(i, j, h)?;
                    let b = wtd.point(i, j, h)?;
                    assert!((a.min - b.min).abs() < 1e-15);
                    assert!((a.max - b.max).abs() < 1e-15);
                    for (qa, qb) in a.quantiles.iter().zip(b.quantiles.iter()) {
                        assert!(
                            (qa - qb).abs() < 1e-12,
                            "quantile mismatch at ({i},{j},{h})"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn summarize_weighted_shifts_bands_toward_heavy_draws() -> Result<(), IdentError> {
        // A single cell across three draws; heavily weighting the largest
        // draw must push the median above the equal-weight median.
        let n = 1;
        let horizon = 0;
        let one = |v: f64| -> Vec<Mat<f64>> { vec![Mat::from_fn(n, n, |_, _| v)] };
        let draws = vec![one(0.0), one(1.0), one(2.0)];
        let probs = [0.5];
        let unw = summarize(&draws, n, horizon, &probs);
        let wtd = summarize_weighted(&draws, &[1.0, 1.0, 20.0], n, horizon, &probs);
        let unw_med = unw.point(0, 0, 0)?.quantiles[0];
        let wtd_med = wtd.point(0, 0, 0)?.quantiles[0];
        assert!((unw_med - 1.0).abs() < 1e-12, "equal-weight median is 1.0");
        assert!(
            wtd_med > unw_med,
            "weighting the top draw must raise the median ({wtd_med} vs {unw_med})"
        );
        // Min/max stay weight-free.
        assert!((wtd.point(0, 0, 0)?.min - 0.0).abs() < 1e-15);
        assert!((wtd.point(0, 0, 0)?.max - 2.0).abs() < 1e-15);
        Ok(())
    }
}
