//! Small dense-matrix helpers shared by the CCC and DCC estimators.
//!
//! Everything routes covariance/correlation factorizations through
//! [`tsecon_linalg::jittered_cholesky`] so the whole library shares one
//! positive-definiteness-hygiene path (symmetrize, then an `L L'` Cholesky
//! with a bounded, logged jitter ladder).

use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_linalg::{jittered_cholesky, JitteredCholesky};

use crate::error::MgarchError;

/// The uncentered second-moment matrix of the standardized residuals,
/// `Qbar = (1 / T) sum_t z_t z_t'`.
///
/// This is Engle's (2002) correlation-targeting estimator: because the
/// standardized residuals `z_t` are (approximately) unit-variance and
/// mean-zero, `Qbar` is their sample correlation matrix, and — crucially —
/// the sample mean of the recursion's driving term `z_t z_t'` equals `Qbar`
/// exactly by construction, which is what makes DCC targeting reproduce the
/// unconditional level of `Q_t` (see [`crate::dcc`]).
///
/// `z` is `T` rows (time) of `k` columns (series).
pub(crate) fn moment_matrix(z: &[Vec<f64>], k: usize) -> Mat<f64> {
    let t = z.len();
    let mut m = Mat::<f64>::zeros(k, k);
    for row in z {
        for i in 0..k {
            for j in 0..k {
                m[(i, j)] += row[i] * row[j];
            }
        }
    }
    let inv_t = 1.0 / t as f64;
    for i in 0..k {
        for j in 0..k {
            m[(i, j)] *= inv_t;
        }
    }
    m
}

/// The correlation matrix implied by a covariance-like matrix `q`:
/// `R = diag(q)^{-1/2} q diag(q)^{-1/2}`.
///
/// This is exactly the map `Q_t -> R_t` of the DCC recursion (Engle 2002).
/// The diagonal of the result is exactly one (set explicitly, not divided,
/// so it is bit-exact).
pub(crate) fn corr_from_cov(q: MatRef<'_, f64>) -> Mat<f64> {
    let k = q.nrows();
    let d: Vec<f64> = (0..k).map(|i| q[(i, i)].sqrt()).collect();
    Mat::from_fn(k, k, |i, j| {
        if i == j {
            1.0
        } else {
            q[(i, j)] / (d[i] * d[j])
        }
    })
}

/// Factorizes a correlation matrix `R = L L'` through the shared jitter
/// ladder (symmetrize, then `L L'` Cholesky, escalating a bounded diagonal
/// jitter only if the clean factorization fails).
///
/// The returned [`JitteredCholesky`] both certifies positive-definiteness
/// (a successful factorization *is* the PD check) and supplies the
/// log-determinant `ln|R| = 2 sum_i ln L_ii`.
///
/// # Errors
///
/// [`MgarchError::Linalg`] if `R` is genuinely indefinite (jitter ladder
/// exhausted), non-finite, or non-square.
pub(crate) fn cholesky(r: MatRef<'_, f64>) -> Result<JitteredCholesky, MgarchError> {
    Ok(jittered_cholesky(r)?)
}

/// The Gaussian quadratic form `x' R^{-1} x` from a Cholesky factor `L`
/// (`R = L L'`).
///
/// Solving `L y = x` by forward substitution gives
/// `x' R^{-1} x = x' L^{-T} L^{-1} x = y' y` — never forming `R^{-1}`.
pub(crate) fn quad_form(l: MatRef<'_, f64>, x: &[f64]) -> f64 {
    let n = l.nrows();
    let mut y = vec![0.0_f64; n];
    let mut quad = 0.0;
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= l[(i, j)] * y[j];
        }
        let yi = s / l[(i, i)];
        y[i] = yi;
        quad += yi * yi;
    }
    quad
}
