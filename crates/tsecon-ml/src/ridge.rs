//! Ridge regression in closed form via the thin singular value decomposition.
//!
//! # Objective (scikit-learn convention)
//!
//! [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
//! minimizes, over the coefficient vector `b`,
//!
//! ```text
//! ||y - X b||_2^2 + alpha * ||b||_2^2
//! ```
//!
//! There is **no** `1/n` factor on the data-fit term (unlike the LASSO /
//! elastic-net family in [`crate::coordinate_descent`], which carry the
//! `1/(2n)` scikit-learn uses). The golden fixture `fixtures/ml.json`
//! documents both conventions in its `_meta.objective_note`; ridge and the
//! penalized-`L1` solvers therefore take `alpha` on different scales, and
//! that is intentional — each matches its scikit-learn namesake exactly.
//!
//! No intercept is fitted: callers pass a centered `y` and a design whose
//! columns have been centered (and, in the fixture, standardized), so the
//! intercept is identically zero.
//!
//! # Closed form
//!
//! The normal equations `(X'X + alpha I) b = X'y` have the unique solution
//! `b = (X'X + alpha I)^{-1} X'y`. Writing the thin SVD `X = U S V'`
//! (`U` is `n x p`, `S = diag(s_1, ..., s_p)`, `V` is `p x p` orthogonal
//! for `n >= p`), and using `V'V = I`,
//!
//! ```text
//! X'X + alpha I = V (S^2 + alpha I) V',
//! b = V diag( s_k / (s_k^2 + alpha) ) U' y.
//! ```
//!
//! This form (Hastie, Tibshirani & Friedman 2009, *Elements of Statistical
//! Learning*, section 3.4.1) is numerically stable — the ill-conditioning
//! that ridge is meant to tame appears only through the benign shrinkage
//! factor `s_k / (s_k^2 + alpha)`, never as an explicit inverse of the
//! near-singular `X'X`.

use tsecon_linalg::faer::MatRef;

use crate::error::MlError;
use crate::util::check_xy;

/// Ridge regression coefficients, computed in closed form via the thin SVD.
///
/// Solves `min_b ||y - X b||^2 + alpha * ||b||^2` (scikit-learn's `Ridge`
/// objective — no `1/n` factor; see the [module docs](self)). `x` is the
/// `n x p` design (no intercept column; pass centered/standardized data),
/// `y` the length-`n` centered target, and `alpha >= 0` the penalty.
///
/// With `alpha = 0` this returns the minimum-norm ordinary-least-squares
/// solution.
///
/// # Errors
///
/// * [`MlError::EmptyInput`] if `x` has no rows or columns;
/// * [`MlError::DimensionMismatch`] if `y.len() != x.nrows()`;
/// * [`MlError::NonFinite`] on any NaN/infinite entry;
/// * [`MlError::InvalidArgument`] if `alpha` is negative or non-finite;
/// * [`MlError::DecompositionFailed`] if the SVD iteration does not
///   converge.
pub fn ridge(x: MatRef<'_, f64>, y: &[f64], alpha: f64) -> Result<Vec<f64>, MlError> {
    let (_n, p) = check_xy(x, y)?;
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(MlError::InvalidArgument {
            what: "alpha must be finite and non-negative",
        });
    }

    let svd = x
        .thin_svd()
        .map_err(|_| MlError::DecompositionFailed { what: "ridge SVD" })?;
    let u = svd.U();
    let v = svd.V();
    let s: Vec<f64> = svd.S().column_vector().iter().copied().collect();
    let r = s.len(); // = min(n, p); for n >= p this equals p.

    // d_k = (U' y)_k
    let d: Vec<f64> = (0..r)
        .map(|k| (0..u.nrows()).map(|i| u[(i, k)] * y[i]).sum::<f64>())
        .collect();

    // filtered coefficients c_k = s_k / (s_k^2 + alpha) * d_k
    let c: Vec<f64> = (0..r)
        .map(|k| {
            let sk = s[k];
            sk / (sk * sk + alpha) * d[k]
        })
        .collect();

    // b_j = sum_k V[j, k] * c_k
    let beta: Vec<f64> = (0..p)
        .map(|j| (0..r).map(|k| v[(j, k)] * c[k]).sum::<f64>())
        .collect();

    Ok(beta)
}

/// Ordinary least squares via the thin SVD (the `alpha = 0` ridge limit,
/// but expressed through the pseudoinverse so exact or near-zero singular
/// values are dropped rather than amplified). Used internally to build the
/// adaptive-LASSO weights of Zou (2006).
///
/// Singular values at or below `rcond * s_max` are treated as zero.
///
/// # Errors
///
/// As [`ridge`], minus the `alpha` domain check.
pub(crate) fn ols_svd(x: MatRef<'_, f64>, y: &[f64]) -> Result<Vec<f64>, MlError> {
    let (n, p) = check_xy(x, y)?;
    let svd = x
        .thin_svd()
        .map_err(|_| MlError::DecompositionFailed { what: "OLS SVD" })?;
    let u = svd.U();
    let v = svd.V();
    let s: Vec<f64> = svd.S().column_vector().iter().copied().collect();
    let r = s.len();
    let s_max = s.iter().copied().fold(0.0f64, f64::max);
    // Relative cutoff scaled by the larger dimension, matching NumPy's
    // `lstsq` default rcond behaviour closely enough for the well-posed
    // full-rank designs this crate's weights are built from.
    let cutoff = s_max * (n.max(p) as f64) * f64::EPSILON;

    let d: Vec<f64> = (0..r)
        .map(|k| (0..u.nrows()).map(|i| u[(i, k)] * y[i]).sum::<f64>())
        .collect();
    let c: Vec<f64> = (0..r)
        .map(|k| if s[k] > cutoff { d[k] / s[k] } else { 0.0 })
        .collect();
    let beta: Vec<f64> = (0..p)
        .map(|j| (0..r).map(|k| v[(j, k)] * c[k]).sum::<f64>())
        .collect();
    Ok(beta)
}
