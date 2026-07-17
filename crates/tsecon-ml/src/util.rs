//! Small shared helpers: input validation and a column-major view of the
//! design matrix. Not part of the public API.

use tsecon_linalg::faer::MatRef;

use crate::error::MlError;

/// Validates that `x` (`n x p`) and `y` (`n`) are nonempty, finite, and
/// conformable. Returns `(n, p)` on success.
pub(crate) fn check_xy(x: MatRef<'_, f64>, y: &[f64]) -> Result<(usize, usize), MlError> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return Err(MlError::EmptyInput { what: "x" });
    }
    if y.len() != n {
        return Err(MlError::DimensionMismatch {
            what: "y length must equal the number of rows of x",
            expected: n,
            got: y.len(),
        });
    }
    for j in 0..p {
        for i in 0..n {
            if !x[(i, j)].is_finite() {
                return Err(MlError::NonFinite { what: "x" });
            }
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(MlError::NonFinite { what: "y" });
    }
    Ok((n, p))
}

/// Copies the columns of `x` into contiguous `Vec<f64>` buffers so the
/// coordinate-descent inner loops touch cache-friendly column slices
/// (faer stores column-major, but exposing `&[f64]` lets the hot loops
/// avoid per-element bounds bookkeeping through the `Mat` indexer).
pub(crate) fn columns(x: MatRef<'_, f64>) -> Vec<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let mut cols = Vec::with_capacity(p);
    for j in 0..p {
        let mut c = Vec::with_capacity(n);
        for i in 0..n {
            c.push(x[(i, j)]);
        }
        cols.push(c);
    }
    cols
}

/// `sum_i a_i * b_i`.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
