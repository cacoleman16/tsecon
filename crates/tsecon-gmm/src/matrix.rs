//! Small dense linear-algebra plumbing for the GMM normal equations,
//! moment-covariance inverses, and the sandwich bread.
//!
//! Everything here delegates the actual factorizations to `faer` (re-exported
//! through `tsecon-linalg`, the workspace's single dense backend), matching
//! the idiom used by the sibling IV/cointegration crates
//! (`tsecon-lp::iv`, `tsecon-coint::linalg`). GMM never hand-rolls a Cholesky
//! or an inverse of its own.

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, MatRef, Side};

use crate::error::GmmError;

/// Assemble a column-major slice-of-columns into an `n x k` dense matrix.
///
/// Each element of `cols` is one column (length `n`); this is the
/// statsmodels/linearmodels "exog as explicit columns" convention used
/// throughout the library.
pub(crate) fn mat_from_cols(cols: &[Vec<f64>], n: usize) -> Mat<f64> {
    let k = cols.len();
    Mat::from_fn(n, k, |i, j| cols[j][i])
}

/// Column vector (`n x 1`) from a slice.
pub(crate) fn col_vec(v: &[f64]) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

/// Inverse of a symmetric positive definite matrix via its Cholesky factor,
/// tagging the offending matrix on failure (indefinite/singular input).
pub(crate) fn inv_spd(m: MatRef<'_, f64>, what: &'static str) -> Result<Mat<f64>, GmmError> {
    Ok(m.llt(Side::Lower)
        .map_err(|_| GmmError::SingularMatrix { what })?
        .inverse())
}

/// Solve the symmetric positive definite system `A beta = b` (single
/// right-hand side), tagging `A` on factorization failure.
pub(crate) fn solve_spd(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    what: &'static str,
) -> Result<Mat<f64>, GmmError> {
    let inv = inv_spd(a, what)?;
    Ok(&inv * b)
}

/// Copy a `k x k` dense matrix into a row-major `Vec<f64>` (the public
/// covariance representation, matching `tsecon-hac::OlsInference::cov`).
pub(crate) fn mat_to_rowmajor(m: MatRef<'_, f64>) -> Vec<f64> {
    let (nr, nc) = (m.nrows(), m.ncols());
    let mut out = vec![0.0_f64; nr * nc];
    for i in 0..nr {
        for j in 0..nc {
            out[i * nc + j] = m[(i, j)];
        }
    }
    out
}

/// Wrap a row-major `p x p` slice as a dense matrix, validating that it is
/// square of the requested order.
pub(crate) fn mat_from_rowmajor(
    data: &[f64],
    p: usize,
    what: &'static str,
) -> Result<Mat<f64>, GmmError> {
    if data.len() != p * p {
        return Err(GmmError::DimensionMismatch {
            what,
            expected: p * p,
            got: data.len(),
        });
    }
    Ok(Mat::from_fn(p, p, |i, j| data[i * p + j]))
}
