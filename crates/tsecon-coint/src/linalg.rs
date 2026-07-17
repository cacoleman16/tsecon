//! Private linear-algebra helpers shared by the Johansen test and the
//! VECM estimator: partialled-out (auxiliary-regression) residuals, the
//! symmetric-definite reduced-rank eigenproblem at the heart of Johansen's
//! canonical-correlation analysis, and small positive-definite utilities.

use tsecon_linalg::faer::linalg::solvers::{DenseSolveCore, SolveLstsq};
use tsecon_linalg::faer::{Mat, MatRef, Side};

use crate::error::CointError;

/// Verifies that every entry of a matrix is finite, tagging the argument.
pub(crate) fn check_finite(m: MatRef<'_, f64>, what: &'static str) -> Result<(), CointError> {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m[(i, j)].is_finite() {
                return Err(CointError::NonFinite { what });
            }
        }
    }
    Ok(())
}

/// Subtracts each column's mean in place (statsmodels `detrend(x, 0)` —
/// the deterministic-constant projection used by `coint_johansen` with
/// `det_order = 0`).
pub(crate) fn demean_columns(m: &mut Mat<f64>) {
    let n = m.nrows();
    if n == 0 {
        return;
    }
    for j in 0..m.ncols() {
        let mut mean = 0.0;
        for i in 0..n {
            mean += m[(i, j)];
        }
        mean /= n as f64;
        for i in 0..n {
            m[(i, j)] -= mean;
        }
    }
}

/// Residuals `Y - X (X'X)^{-1} X' Y` of the multivariate regression of
/// `y` on `x` (statsmodels `resid(y, x) = y - x @ pinv(x) @ y`). When `x`
/// has no columns the residuals are `y` itself.
///
/// The projection is computed by Householder QR least squares, which
/// reproduces the SVD-based `pinv` residuals to machine precision on the
/// well-conditioned lagged-difference designs the VECM layer builds.
pub(crate) fn partial_out(y: MatRef<'_, f64>, x: MatRef<'_, f64>) -> Mat<f64> {
    if x.ncols() == 0 {
        return y.to_owned();
    }
    let b = x.qr().solve_lstsq(y);
    y - x * b
}

/// Inverse of a symmetric positive definite matrix via its Cholesky
/// factor.
pub(crate) fn inv_spd(m: MatRef<'_, f64>, what: &'static str) -> Result<Mat<f64>, CointError> {
    Ok(m.llt(Side::Lower)
        .map_err(|_| CointError::NotPositiveDefinite { what })?
        .inverse())
}

/// Inverse of a general square matrix via LU with partial pivoting; the
/// result is screened for non-finite entries (a numerically singular
/// input) and rejected with [`CointError::Singular`].
pub(crate) fn inv_general(m: MatRef<'_, f64>, what: &'static str) -> Result<Mat<f64>, CointError> {
    let inv = m.partial_piv_lu().inverse();
    check_finite(inv.as_ref(), what).map_err(|_| CointError::Singular { what })?;
    Ok(inv)
}

/// `ln det(M)` of a symmetric positive definite matrix from its Cholesky
/// factor, `2 sum_i ln L_ii`.
pub(crate) fn ln_det_spd(m: MatRef<'_, f64>, what: &'static str) -> Result<f64, CointError> {
    let l = m
        .llt(Side::Lower)
        .map_err(|_| CointError::NotPositiveDefinite { what })?
        .L()
        .to_owned();
    let mut ld = 0.0;
    for i in 0..l.nrows() {
        ld += l[(i, i)].ln();
    }
    Ok(2.0 * ld)
}

/// Inverse of a lower-triangular matrix by forward substitution (the
/// factors here are `k x k` with `k` the number of series, so the cubic
/// cost is negligible and an explicit inverse keeps the reduced-rank
/// transform readable).
fn lower_tri_inverse(l: MatRef<'_, f64>) -> Mat<f64> {
    let n = l.nrows();
    let mut inv = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        inv[(j, j)] = 1.0 / l[(j, j)];
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in j..i {
                s += l[(i, k)] * inv[(k, j)];
            }
            inv[(i, j)] = -s / l[(i, i)];
        }
    }
    inv
}

/// Solution of Johansen's canonical-correlation eigenproblem
///
/// ```text
/// S_10 S_00^{-1} S_01 v_i = lambda_i S_11 v_i,     v_i' S_11 v_j = delta_ij
/// ```
///
/// (Johansen 1991; Lütkepohl 2005, eq. 7.2.9–7.2.12), where `s01 = R_0'
/// R_1 / T`, `s00 = R_0' R_0 / T`, `s11 = R_1' R_1 / T` are the auxiliary
/// residual second moments (`R_0` the partialled-out differences, `R_1`
/// the partialled-out lagged levels).
///
/// It is solved as a *symmetric* problem: with `S_11 = L L'` the
/// eigenvalues are those of `C = L^{-1} B L^{-T}`, `B = S_10 S_00^{-1}
/// S_01`, and the `S_11`-orthonormal generalized eigenvectors are
/// `v = L^{-T} q`. Returns the eigenvalues and the matching eigenvector
/// columns, both sorted in **decreasing** eigenvalue order (statsmodels'
/// `np.argsort(lambd)[::-1]`).
///
/// # Errors
///
/// * [`CointError::NotPositiveDefinite`] if `s00` or `s11` is not SPD;
/// * [`CointError::Linalg`] if the symmetric eigensolver fails to
///   converge.
pub(crate) fn reduced_rank_eig(
    s00: MatRef<'_, f64>,
    s01: MatRef<'_, f64>,
    s11: MatRef<'_, f64>,
) -> Result<(Vec<f64>, Mat<f64>), CointError> {
    let s00_inv = inv_spd(s00, "S_00")?;
    // B = S_10 S_00^{-1} S_01 = S_01' S_00^{-1} S_01 (symmetric PSD).
    let s10 = s01.transpose().to_owned();
    let b = &s10 * &s00_inv * s01;

    let l = s11
        .llt(Side::Lower)
        .map_err(|_| CointError::NotPositiveDefinite { what: "S_11" })?
        .L()
        .to_owned();
    let l_inv = lower_tri_inverse(l.as_ref());
    // C = L^{-1} B L^{-T}, symmetrized to kill the tiny asymmetry that
    // floating-point matrix products leave behind.
    let c_raw = &l_inv * &b * l_inv.transpose();
    let k = c_raw.nrows();
    let c = Mat::from_fn(k, k, |i, j| 0.5 * (c_raw[(i, j)] + c_raw[(j, i)]));

    let eigen =
        c.self_adjoint_eigen(Side::Lower)
            .map_err(|_| tsecon_linalg::LinalgError::EigenFailed {
                what: "Johansen reduced-rank eigenproblem",
            })?;
    // faer returns eigenvalues in nondecreasing order; reverse to match
    // Johansen's convention (largest canonical correlation first).
    let s = eigen.S();
    let ascending: Vec<f64> = s.column_vector().iter().copied().collect();
    let q = eigen.U();
    // Generalized eigenvectors v = L^{-T} q, reordered to decreasing eigenvalue.
    let lt_inv = l_inv.transpose().to_owned();
    let v_ascending = &lt_inv * q;

    let mut eigvals = Vec::with_capacity(k);
    let mut vecs = Mat::<f64>::zeros(k, k);
    for (out_col, src_col) in (0..k).rev().enumerate() {
        eigvals.push(ascending[src_col]);
        for i in 0..k {
            vecs[(i, out_col)] = v_ascending[(i, src_col)];
        }
    }
    Ok((eigvals, vecs))
}
