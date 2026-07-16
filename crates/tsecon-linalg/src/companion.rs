//! Companion-form utilities for AR/VAR lag polynomials.
//!
//! The companion form rewrites an order-`p` autoregression as a first-order
//! system; it is the workhorse for stability checks, stationary-covariance
//! initialization, and MA(infinity) / impulse-response expansions
//! (Lütkepohl 2005, section 2.1; Hamilton 1994, section 1.2).

use faer::{Mat, MatRef};

use crate::error::LinalgError;

/// Builds the `p x p` companion matrix of a scalar AR(p) polynomial
/// `x_t = phi_1 x_{t-1} + ... + phi_p x_{t-p} + e_t`:
///
/// ```text
/// [ phi_1  phi_2 ... phi_{p-1}  phi_p ]
/// [   1      0   ...     0        0   ]
/// [   0      1   ...     0        0   ]
/// [   .      .    .      .        .   ]
/// [   0      0   ...     1        0   ]
/// ```
///
/// The process is (covariance) stationary iff all eigenvalues of this
/// matrix lie strictly inside the unit circle (Hamilton 1994, prop. 1.1).
///
/// # Errors
///
/// * [`LinalgError::EmptyInput`] if `phi` is empty;
/// * [`LinalgError::NonFinite`] if `phi` contains NaN/infinity.
pub fn companion_from_ar(phi: &[f64]) -> Result<Mat<f64>, LinalgError> {
    if phi.is_empty() {
        return Err(LinalgError::EmptyInput { what: "phi" });
    }
    if phi.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite { what: "phi" });
    }
    let p = phi.len();
    let mut c = Mat::<f64>::zeros(p, p);
    for (j, &v) in phi.iter().enumerate() {
        c[(0, j)] = v;
    }
    for i in 1..p {
        c[(i, i - 1)] = 1.0;
    }
    Ok(c)
}

/// Builds the `kp x kp` companion matrix of a VAR(p) lag polynomial with
/// `k x k` coefficient matrices `coefs = [A_1, ..., A_p]`:
///
/// ```text
/// [ A_1  A_2 ... A_{p-1}  A_p ]
/// [  I    0  ...    0      0  ]
/// [  0    I  ...    0      0  ]
/// [  .    .   .     .      .  ]
/// [  0    0  ...    I      0  ]
/// ```
///
/// (Lütkepohl 2005, eq. 2.1.8). For `k = 1` this reduces to
/// [`companion_from_ar`].
///
/// # Errors
///
/// * [`LinalgError::EmptyInput`] if `coefs` is empty or the matrices are
///   `0 x 0`;
/// * [`LinalgError::NotSquare`] / [`LinalgError::DimensionMismatch`] if
///   any `A_i` is not square or the sizes disagree;
/// * [`LinalgError::NonFinite`] on NaN/infinite entries.
pub fn companion_from_var(coefs: &[MatRef<'_, f64>]) -> Result<Mat<f64>, LinalgError> {
    if coefs.is_empty() {
        return Err(LinalgError::EmptyInput { what: "coefs" });
    }
    let k = coefs[0].nrows();
    if k == 0 {
        return Err(LinalgError::EmptyInput { what: "coefs[0]" });
    }
    for a in coefs {
        if a.nrows() != a.ncols() {
            return Err(LinalgError::NotSquare {
                what: "VAR coefficient matrix",
                rows: a.nrows(),
                cols: a.ncols(),
            });
        }
        if a.nrows() != k {
            return Err(LinalgError::DimensionMismatch {
                what: "all VAR coefficient matrices must share one dimension",
                expected: k,
                got: a.nrows(),
            });
        }
        for j in 0..k {
            for i in 0..k {
                if !a[(i, j)].is_finite() {
                    return Err(LinalgError::NonFinite { what: "coefs" });
                }
            }
        }
    }
    let p = coefs.len();
    let kp = k * p;
    let mut c = Mat::<f64>::zeros(kp, kp);
    for (block, a) in coefs.iter().enumerate() {
        for j in 0..k {
            for i in 0..k {
                c[(i, block * k + j)] = a[(i, j)];
            }
        }
    }
    for i in k..kp {
        c[(i, i - k)] = 1.0;
    }
    Ok(c)
}

/// Spectral radius `max_i |lambda_i(A)|` via `faer`'s dense (real Schur)
/// eigenvalue solver.
///
/// # Errors
///
/// * [`LinalgError::NotSquare`] / [`LinalgError::EmptyInput`] on shape
///   violations;
/// * [`LinalgError::NonFinite`] on NaN/infinite entries;
/// * [`LinalgError::EigenFailed`] if the QR eigenvalue iteration does not
///   converge.
pub fn spectral_radius(a: MatRef<'_, f64>) -> Result<f64, LinalgError> {
    if a.nrows() != a.ncols() {
        return Err(LinalgError::NotSquare {
            what: "a",
            rows: a.nrows(),
            cols: a.ncols(),
        });
    }
    if a.nrows() == 0 {
        return Err(LinalgError::EmptyInput { what: "a" });
    }
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            if !a[(i, j)].is_finite() {
                return Err(LinalgError::NonFinite { what: "a" });
            }
        }
    }
    let eigs = a
        .eigenvalues()
        .map_err(|_| LinalgError::EigenFailed {
            what: "spectral_radius",
        })?;
    Ok(eigs
        .iter()
        .map(|c| c.re.hypot(c.im))
        .fold(0.0f64, f64::max))
}

/// Stability check for a companion (or any transition) matrix:
/// returns `true` iff `rho(A) < 1 - tol`.
///
/// `tol >= 0` shrinks the acceptance region away from the unit circle;
/// `tol = 0` is the exact theoretical condition, while e.g. `tol = 1e-7`
/// rejects processes numerically indistinguishable from unit roots (the
/// regime where Lyapunov initialization and Levinson recursions lose
/// accuracy).
///
/// # Errors
///
/// * [`LinalgError::InvalidArgument`] if `tol` is negative, NaN, or
///   `>= 1`;
/// * plus everything [`spectral_radius`] can return.
pub fn is_stable(a: MatRef<'_, f64>, tol: f64) -> Result<bool, LinalgError> {
    if !tol.is_finite() || !(0.0..1.0).contains(&tol) {
        return Err(LinalgError::InvalidArgument {
            what: "tol must satisfy 0 <= tol < 1",
        });
    }
    Ok(spectral_radius(a)? < 1.0 - tol)
}

/// Psi-weights (MA(infinity) coefficients) of a scalar AR(p) polynomial,
/// `psi_0, psi_1, ..., psi_horizon`.
///
/// For `x_t = phi_1 x_{t-1} + ... + phi_p x_{t-p} + e_t`, the moving
/// average representation `x_t = sum_j psi_j e_{t-j}` has weights given by
/// the recursion (Brockwell & Davis 1991, section 3.3; `1/phi(z)` power
/// series)
///
/// ```text
/// psi_0 = 1,    psi_j = sum_{i=1}^{min(j, p)} phi_i psi_{j-i}
/// ```
///
/// These are exactly the impulse responses of the AR process to a unit
/// innovation — the IRF primitive. The recursion is well defined for any
/// coefficients; for a non-stationary polynomial the weights simply do not
/// decay (no error is raised, since finite-horizon responses of explosive
/// processes are legitimate objects).
///
/// The returned vector has length `horizon + 1`.
///
/// # Errors
///
/// * [`LinalgError::EmptyInput`] if `phi` is empty;
/// * [`LinalgError::NonFinite`] if `phi` contains NaN/infinity.
pub fn ar_psi_weights(phi: &[f64], horizon: usize) -> Result<Vec<f64>, LinalgError> {
    if phi.is_empty() {
        return Err(LinalgError::EmptyInput { what: "phi" });
    }
    if phi.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NonFinite { what: "phi" });
    }
    let p = phi.len();
    let mut psi = Vec::with_capacity(horizon + 1);
    psi.push(1.0);
    for j in 1..=horizon {
        let mut s = 0.0;
        for i in 1..=j.min(p) {
            s += phi[i - 1] * psi[j - i];
        }
        psi.push(s);
    }
    Ok(psi)
}
