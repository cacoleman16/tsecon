//! Discrete Lyapunov equation solver via the doubling algorithm.

use faer::{Mat, MatRef};

use crate::companion::spectral_radius;
use crate::error::LinalgError;

/// Iteration budget for the doubling recursion. Because the transition
/// matrix is squared every step, the contribution of the truncated tail
/// after `k` steps is of order `rho(A)^(2^k)`, so even `rho = 1 - 1e-12`
/// converges in fewer than 60 doublings; the budget is a hard safety net,
/// not a tuning parameter.
const MAX_DOUBLING_ITER: usize = 100;

/// Solves the discrete Lyapunov (Stein) equation `X = A X A' + Q`.
///
/// Uses the Smith (1968) doubling iteration: with `A_0 = A`, `X_0 = Q`,
///
/// ```text
/// X_{k+1} = X_k + A_k X_k A_k',    A_{k+1} = A_k A_k,
/// ```
///
/// so that `X_k = sum_{j=0}^{2^k - 1} A^j Q (A')^j`, converging
/// quadratically to the unique solution `X = sum_{j>=0} A^j Q (A')^j`
/// whenever the spectral radius `rho(A) < 1`. This is the
/// stationary-covariance initializer for state-space models (the
/// `E[alpha alpha']` of a stationary VAR(1) transition), cf. Lütkepohl
/// (2005, section 2.1.4).
///
/// Convergence criterion: the iteration stops once the increment satisfies
/// `||A_k X_k A_k'||_max <= eps * max(||X_k||_max, f64::MIN_POSITIVE)`
/// with `eps = f64::EPSILON` (machine precision), i.e. when adding further
/// tail terms can no longer change the solution in double precision.
///
/// Failure mode: if `rho(A) >= 1` the series diverges and no stationary
/// solution exists. The spectral radius is checked *before* iterating
/// (via `faer` eigenvalues) and [`LinalgError::Unstable`] is returned —
/// the routine never spins on an explosive input. A defensive
/// [`LinalgError::NoConvergence`] guards the (unreachable in practice)
/// case of the budget running out, and non-finite intermediates abort
/// with [`LinalgError::NonFinite`].
///
/// If `Q` is symmetric the result is symmetrized (`0.5 (X + X')`) before
/// being returned so downstream Cholesky factorizations see an exactly
/// symmetric matrix; an asymmetric `Q` is accepted and passed through
/// unsymmetrized, matching `scipy.linalg.solve_discrete_lyapunov`.
///
/// # Errors
///
/// * [`LinalgError::NotSquare`] / [`LinalgError::DimensionMismatch`] on
///   shape violations;
/// * [`LinalgError::NonFinite`] on NaN/infinite input or intermediates;
/// * [`LinalgError::Unstable`] when `rho(A) >= 1`;
/// * [`LinalgError::EigenFailed`] if the eigenvalue pre-check fails to
///   converge;
/// * [`LinalgError::NoConvergence`] if the iteration budget is exhausted.
pub fn solve_discrete_lyapunov(
    a: MatRef<'_, f64>,
    q: MatRef<'_, f64>,
) -> Result<Mat<f64>, LinalgError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::NotSquare {
            what: "a",
            rows: a.nrows(),
            cols: a.ncols(),
        });
    }
    if q.nrows() != n || q.ncols() != q.nrows() {
        return Err(LinalgError::DimensionMismatch {
            what: "q must be square with the same dimension as a",
            expected: n,
            got: if q.nrows() != n { q.nrows() } else { q.ncols() },
        });
    }
    if n == 0 {
        return Err(LinalgError::EmptyInput { what: "a" });
    }
    if !is_finite_mat(a) || !is_finite_mat(q) {
        return Err(LinalgError::NonFinite { what: "a / q" });
    }

    // Stability pre-check: rho(A) >= 1 means the geometric series diverges.
    let rho = spectral_radius(a)?;
    if rho >= 1.0 {
        return Err(LinalgError::Unstable {
            spectral_radius: rho,
        });
    }

    let mut x = q.to_owned();
    let mut ak = a.to_owned();
    let mut iterations = 0usize;
    let mut last_rel = f64::INFINITY;
    let mut converged = false;
    while iterations < MAX_DOUBLING_ITER {
        iterations += 1;
        // increment = A_k X_k A_k'
        let increment = &ak * &x * ak.transpose();
        let inc_norm = increment.norm_max();
        let x_norm = x.as_ref().norm_max().max(f64::MIN_POSITIVE);
        if !inc_norm.is_finite() {
            return Err(LinalgError::NonFinite {
                what: "doubling iterate (intermediate overflow)",
            });
        }
        x += &increment;
        last_rel = inc_norm / x_norm;
        if inc_norm <= f64::EPSILON * x_norm {
            converged = true;
            break;
        }
        ak = &ak * &ak;
    }
    if !converged {
        // Unreachable for rho(A) < 1 in exact arithmetic (see budget note
        // above); kept as a defensive guard against pathological rounding.
        return Err(LinalgError::NoConvergence {
            iterations,
            residual: last_rel,
        });
    }

    // Preserve exact symmetry when Q is symmetric (the covariance case).
    if is_symmetric(q) {
        x = crate::hygiene::symmetrize(x.as_ref())?;
    }
    Ok(x)
}

/// True when every entry of `m` is finite.
fn is_finite_mat(m: MatRef<'_, f64>) -> bool {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if !m[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}

/// Exact (bitwise) symmetry check.
fn is_symmetric(m: MatRef<'_, f64>) -> bool {
    if m.nrows() != m.ncols() {
        return false;
    }
    for j in 0..m.ncols() {
        for i in 0..j {
            if m[(i, j)] != m[(j, i)] {
                return false;
            }
        }
    }
    true
}
