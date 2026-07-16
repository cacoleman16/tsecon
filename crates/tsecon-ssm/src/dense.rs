//! Small crate-internal dense kernels shared by the filter and smoother.
//!
//! These are deliberately plain loops over `faer` matrices: the state and
//! observation dimensions of the models this engine serves are small, the
//! recursions are rank-one dominated, and explicit loops keep every update
//! auditable against the Durbin & Koopman (2012) formulas. Heavier products
//! (`T P T'`-style sandwiches) go through `faer`'s matmul.

use tsecon_linalg::faer::{Mat, MatRef};

/// Dot product of two equal-length slices.
#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y += alpha * x` on slices.
#[inline]
pub(crate) fn axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
    debug_assert_eq!(y.len(), x.len());
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// Matrix-vector product `M x`.
pub(crate) fn mat_vec(m: MatRef<'_, f64>, x: &[f64]) -> Vec<f64> {
    debug_assert_eq!(m.ncols(), x.len());
    let mut out = vec![0.0; m.nrows()];
    for (j, xj) in x.iter().enumerate() {
        if *xj != 0.0 {
            for (i, oi) in out.iter_mut().enumerate() {
                *oi += m[(i, j)] * xj;
            }
        }
    }
    out
}

/// Transposed matrix-vector product `M' x`.
pub(crate) fn mat_t_vec(m: MatRef<'_, f64>, x: &[f64]) -> Vec<f64> {
    debug_assert_eq!(m.nrows(), x.len());
    let mut out = vec![0.0; m.ncols()];
    for (j, oj) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (i, xi) in x.iter().enumerate() {
            s += m[(i, j)] * xi;
        }
        *oj = s;
    }
    out
}

/// Rank-one downdate `P -= a b'`.
pub(crate) fn outer_sub(p: &mut Mat<f64>, a: &[f64], b: &[f64]) {
    debug_assert_eq!(p.nrows(), a.len());
    debug_assert_eq!(p.ncols(), b.len());
    for (j, bj) in b.iter().enumerate() {
        if *bj != 0.0 {
            for (i, ai) in a.iter().enumerate() {
                p[(i, j)] -= ai * bj;
            }
        }
    }
}

/// Scaled outer product `s * a b'` as a new matrix.
pub(crate) fn outer_scaled(a: &[f64], b: &[f64], s: f64) -> Mat<f64> {
    Mat::from_fn(a.len(), b.len(), |i, j| s * a[i] * b[j])
}

/// Sandwich product `T P T'`.
pub(crate) fn sandwich(t: MatRef<'_, f64>, p: MatRef<'_, f64>) -> Mat<f64> {
    let tp = t * p;
    tp.as_ref() * t.transpose()
}

/// Transposed sandwich product `T' N T`.
pub(crate) fn sandwich_t(t: MatRef<'_, f64>, n: MatRef<'_, f64>) -> Mat<f64> {
    let tn = t.transpose() * n;
    tn.as_ref() * t
}

/// Squared Frobenius norm, `sum_{ij} M_{ij}^2`.
pub(crate) fn frob_sq(m: MatRef<'_, f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            s += m[(i, j)] * m[(i, j)];
        }
    }
    s
}

/// In-place exact symmetrization `M <- (M + M') / 2` of a square matrix.
///
/// Cheap hygiene applied after covariance updates so downstream code never
/// sees the (tiny, roundoff-level) asymmetry the rank-one recursions leave.
pub(crate) fn symmetrize_in_place(m: &mut Mat<f64>) {
    let n = m.nrows();
    debug_assert_eq!(n, m.ncols());
    for j in 0..n {
        for i in 0..j {
            let v = 0.5 * (m[(i, j)] + m[(j, i)]);
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
}

/// Row `i` of `M` as an owned vector.
pub(crate) fn row_to_vec(m: MatRef<'_, f64>, i: usize) -> Vec<f64> {
    (0..m.ncols()).map(|j| m[(i, j)]).collect()
}

/// Solves `L L' x = b` given a lower-triangular Cholesky factor `L`
/// (forward then backward substitution). The factor comes from a
/// successful Cholesky, so its diagonal is strictly positive.
pub(crate) fn chol_solve(l: &Mat<f64>, b: &[f64]) -> Vec<f64> {
    let n = l.nrows();
    debug_assert_eq!(b.len(), n);
    let mut x = b.to_vec();
    // Forward: L y = b.
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= l[(i, j)] * x[j];
        }
        x[i] = s / l[(i, i)];
    }
    // Backward: L' x = y.
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= l[(j, i)] * x[j];
        }
        x[i] = s / l[(i, i)];
    }
    x
}
