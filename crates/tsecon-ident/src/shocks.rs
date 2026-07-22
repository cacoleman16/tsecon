//! Structural-shock extraction primitives — the greenfield `epsilon_t` home
//! that the historical decomposition and the narrative sign-restriction
//! sampler both rest on.
//!
//! Given a reduced-form draw `(B, Sigma)` and a rotation `Q`, the structural
//! shocks are recovered without ever inverting a covariance:
//!
//! ```text
//! U = Y - X B                 (reduced-form residuals, T_eff x n)
//! W = U P^{-T}                 (orthogonalized residuals; rows w_t = P^{-1} u_t)
//! E = W Q                      (structural shocks; row t = eps_t' = w_t' Q)
//! ```
//!
//! with `P = chol(Sigma)` lower-triangular (the same factor
//! [`tsecon_bayes::cholesky_irf`] uses) and `Q` orthogonal. The identity
//! `u_t = P Q eps_t` round-trips `U` exactly, because
//! `E Q' P' = W Q Q' P' = W P' = U P^{-T} P' = U` (`Q` orthogonal, `P`
//! lower-triangular). The orthogonalized residuals are obtained by forward
//! substitution against `P`, never by forming `P^{-1}`.
//!
//! The `(Y, X)` regressor layout matches the crate-wide Normal-inverse-Wishart
//! convention exactly (intercept column, then lag blocks; see
//! `tsecon_bayes::niw`), so residuals built here line up with the coefficients
//! `B` produced anywhere else in the stack.

use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::IdentError;

/// Builds the VAR(`p`) regressor matrices `(Y, X)` from a `T x n` data matrix.
///
/// `Y` is `T_eff x n` with `Y[(t, j)] = data[(p + t, j)]`; `X` is
/// `T_eff x (1 + n p)` with column `0` the intercept and column
/// `1 + (l - 1) n + v` equal to `data[(p + t - l, v)]` (lag `l` in `1..=p`,
/// variable `v`). `T_eff = T - p`. This is byte-for-byte the layout
/// `tsecon_bayes::NiwPrior::posterior` builds, so the residuals `Y - X B`
/// match the reduced form the posterior coefficients describe.
///
/// # Errors
///
/// * [`IdentError::InvalidArgument`] if `p == 0`;
/// * [`IdentError::Dimension`] if `T < p + 1` (no complete regression row);
/// * [`IdentError::NonFinite`] on any NaN/infinite data entry.
pub(crate) fn build_regressors(
    data: MatRef<'_, f64>,
    p: usize,
) -> Result<(Mat<f64>, Mat<f64>), IdentError> {
    if p == 0 {
        return Err(IdentError::InvalidArgument {
            what: "lag length p must be at least 1",
        });
    }
    let n = data.ncols();
    let t = data.nrows();
    if t < p + 1 {
        return Err(IdentError::Dimension {
            what: "data must have at least p + 1 rows to form one regression row",
            expected: p + 1,
            got: t,
        });
    }
    for j in 0..n {
        for i in 0..t {
            if !data[(i, j)].is_finite() {
                return Err(IdentError::NonFinite { what: "data" });
            }
        }
    }
    let t_eff = t - p;
    let k = 1 + n * p;
    let y = Mat::from_fn(t_eff, n, |i, j| data[(p + i, j)]);
    let x = Mat::from_fn(t_eff, k, |i, j| {
        if j == 0 {
            1.0
        } else {
            let l = (j - 1) / n + 1; // lag 1..=p
            let v = (j - 1) % n; // variable within the lag block
            data[(p + i - l, v)]
        }
    });
    Ok((y, x))
}

/// Reduced-form residuals `U = Y - X B` (`T_eff x n`).
pub(crate) fn reduced_form_residuals(
    y: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
) -> Mat<f64> {
    let fitted = x * b;
    Mat::from_fn(y.nrows(), y.ncols(), |i, j| y[(i, j)] - fitted[(i, j)])
}

/// Orthogonalized residuals `W = U P^{-T}` (rows `w_t = P^{-1} u_t`).
///
/// Computed by forward substitution against the lower-triangular Cholesky
/// factor `P` — `P w_t = u_t` solved per row — so `P` is never inverted. The
/// result feeds [`structural_shocks`]; together they give
/// `eps_t = Q' P^{-1} u_t` for any rotation `Q`.
///
/// # Errors
///
/// * [`IdentError::Dimension`] if `p_chol` is not `n x n` with `n` the number
///   of residual columns;
/// * [`IdentError::InvalidArgument`] if `p_chol` has a zero or non-finite
///   diagonal entry (a singular triangular factor).
pub(crate) fn orthogonalized_residuals(
    u: MatRef<'_, f64>,
    p_chol: MatRef<'_, f64>,
) -> Result<Mat<f64>, IdentError> {
    let t = u.nrows();
    let n = u.ncols();
    if p_chol.nrows() != n || p_chol.ncols() != n {
        return Err(IdentError::Dimension {
            what: "Cholesky factor must be n x n with n the residual column count",
            expected: n,
            got: p_chol.nrows(),
        });
    }
    for i in 0..n {
        let d = p_chol[(i, i)];
        if !d.is_finite() || d == 0.0 {
            return Err(IdentError::InvalidArgument {
                what: "Cholesky factor has a zero or non-finite diagonal entry",
            });
        }
    }
    let mut w = Mat::<f64>::zeros(t, n);
    for row in 0..t {
        for i in 0..n {
            let mut s = u[(row, i)];
            for jj in 0..i {
                s -= p_chol[(i, jj)] * w[(row, jj)];
            }
            w[(row, i)] = s / p_chol[(i, i)];
        }
    }
    Ok(w)
}

/// Structural shocks `E = W Q` (`T_eff x n`), row `t` = `eps_t' = w_t' Q`.
pub(crate) fn structural_shocks(w: MatRef<'_, f64>, q: MatRef<'_, f64>) -> Mat<f64> {
    w * q
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// A 2x2 rotation `Q(theta)` (orthogonal, det +1).
    fn rot2(theta: f64) -> Mat<f64> {
        let (s, c) = theta.sin_cos();
        Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => c,
            (0, 1) => -s,
            (1, 0) => s,
            (1, 1) => c,
            _ => 0.0,
        })
    }

    #[test]
    fn regressor_layout_matches_niw_convention() -> Result<(), IdentError> {
        // 4 obs, 2 vars, p = 1 => T_eff = 3, k = 3.
        let data = Mat::from_fn(4, 2, |i, j| (10 * (i + 1) + j) as f64);
        let (y, x) = build_regressors(data.as_ref(), 1)?;
        assert_eq!(y.nrows(), 3);
        assert_eq!(x.ncols(), 3);
        for t in 0..3 {
            for j in 0..2 {
                assert_eq!(y[(t, j)], data[(1 + t, j)]);
            }
            assert_eq!(x[(t, 0)], 1.0);
            // column 1 => lag 1 var 0; column 2 => lag 1 var 1.
            assert_eq!(x[(t, 1)], data[(t, 0)]);
            assert_eq!(x[(t, 2)], data[(t, 1)]);
        }
        Ok(())
    }

    #[test]
    fn structural_shocks_round_trip_recovers_residuals() -> Result<(), IdentError> {
        // Arbitrary residuals, an arbitrary lower-triangular P, and an
        // orthogonal Q. E Q' P' must reconstruct U to machine precision.
        let t = 5usize;
        let n = 2usize;
        let u = Mat::from_fn(t, n, |i, j| {
            ((i as f64) - 2.0) * 0.7 + (j as f64) * 1.3 - 0.4
        });
        let p_chol = Mat::from_fn(n, n, |i, j| match (i, j) {
            (0, 0) => 1.5,
            (1, 0) => -0.6,
            (1, 1) => 0.9,
            _ => 0.0,
        });
        let q = rot2(0.7);
        let w = orthogonalized_residuals(u.as_ref(), p_chol.as_ref())?;
        let e = structural_shocks(w.as_ref(), q.as_ref());
        // U_recon = E Q' P'.
        let eqt = e.as_ref() * q.as_ref().transpose();
        let recon = eqt.as_ref() * p_chol.as_ref().transpose();
        for i in 0..t {
            for j in 0..n {
                assert!(
                    (recon[(i, j)] - u[(i, j)]).abs() < 1e-12,
                    "round-trip residual mismatch at ({i},{j})"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn orthogonalized_residuals_solve_the_triangular_system() -> Result<(), IdentError> {
        // W must satisfy P W' = U' row-by-row (P w_t = u_t).
        let t = 3usize;
        let n = 3usize;
        let u = Mat::from_fn(t, n, |i, j| (i as f64 + 1.0) * (j as f64 - 1.0) + 0.5);
        let p_chol = Mat::from_fn(n, n, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (1, 0) => 0.5,
            (1, 1) => 1.1,
            (2, 0) => -0.3,
            (2, 1) => 0.2,
            (2, 2) => 0.7,
            _ => 0.0,
        });
        let w = orthogonalized_residuals(u.as_ref(), p_chol.as_ref())?;
        for row in 0..t {
            for i in 0..n {
                let mut s = 0.0;
                for jj in 0..=i {
                    s += p_chol[(i, jj)] * w[(row, jj)];
                }
                assert!((s - u[(row, i)]).abs() < 1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn build_regressors_rejects_bad_input() {
        let data = Mat::<f64>::zeros(3, 2);
        assert!(matches!(
            build_regressors(data.as_ref(), 0),
            Err(IdentError::InvalidArgument { .. })
        ));
        // p = 5 needs at least 6 rows; only 3 given.
        assert!(matches!(
            build_regressors(data.as_ref(), 5),
            Err(IdentError::Dimension { .. })
        ));
        let mut bad = Mat::<f64>::zeros(4, 2);
        bad[(2, 1)] = f64::NAN;
        assert!(matches!(
            build_regressors(bad.as_ref(), 1),
            Err(IdentError::NonFinite { .. })
        ));
    }
}
