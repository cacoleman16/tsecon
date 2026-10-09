//! Companion-form utilities for AR/VAR lag polynomials.
//!
//! The companion form rewrites an order-`p` autoregression as a first-order
//! system; it is the workhorse for stability checks, stationary-covariance
//! initialization, and MA(infinity) / impulse-response expansions
//! (Lütkepohl 2005, section 2.1; Hamilton 1994, section 1.2).

use faer::{c64, Mat, MatRef};

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

/// Radix of the balancing scale factors: a power-of-two similarity is exact
/// in binary floating point.
const BALANCE_RADIX: f64 = 2.0;

/// LAPACK `dgebal`'s acceptance factor: a rescaling of index `i` is applied
/// only when it lowers `c + r` (the column and row norms) below 95% of its
/// value.
const BALANCE_FACTOR: f64 = 0.95;

/// Cap on the balancing sweeps. LAPACK runs to convergence; every accepted
/// step cuts one `c + r` by at least 5%, so a handful of sweeps suffice in
/// practice (each sweep can move a factor by many powers of two). Stopping
/// early only leaves the matrix less balanced: the similarity is exact
/// either way, so the eigenvalues are unaffected.
const BALANCE_MAX_SWEEPS: usize = 100;

/// Overflow-safe Euclidean norm.
fn norm2(x: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = x.clone().fold(0.0f64, |acc, v| acc.max(v.abs()));
    if m == 0.0 || !m.is_finite() {
        return m;
    }
    m * x.map(|v| (v / m) * (v / m)).sum::<f64>().sqrt()
}

/// Balances a square matrix by a diagonal similarity `D^{-1} A D` whose
/// entries are powers of two — LAPACK `dgebal` with `job = 'S'` (scaling
/// only, no permutation; the Parlett-Reinsch 1969 iteration with the
/// 2-norms of LAPACK 3.5 and later). Returns the balanced matrix and the
/// diagonal of `D`.
///
/// A power-of-two scaling is EXACT in binary floating point (the safe-range
/// guards keep every entry away from overflow and the subnormal range), so
/// the balanced matrix has exactly the eigenvalues of `a`. What changes is
/// their computability: a QR eigenvalue iteration is backward stable
/// relative to the norm of the matrix it runs on, so on a matrix whose
/// entries span many orders of magnitude it can lose most of the digits of
/// the eigenvalues that the small entries determine. Balancing makes each
/// row and its column comparable in norm, which brings that error back to
/// the size of the eigenvalues' own sensitivity.
///
/// The VAR companion matrix is the canonical case: rescaling series `i` by
/// `s_i` is the similarity `A_l -> S A_l S^{-1}` of every lag matrix, so
/// the roots of a VAR do not depend on the units of its variables — but
/// series in units `1e14` apart give a companion whose entries span about
/// `1e-15 .. 1e14`, and an unbalanced solve then puts a stable root on the
/// wrong side of the unit circle (`fixtures/companion_roots.json`).
///
/// An index whose off-diagonal column or row is zero (an isolated
/// eigenvalue, e.g. a zero column of `A_p`) is left unscaled, as in LAPACK.
///
/// # Errors
///
/// * [`LinalgError::NotSquare`] / [`LinalgError::EmptyInput`] on shape
///   violations;
/// * [`LinalgError::NonFinite`] on NaN/infinite entries.
pub fn balance(a: MatRef<'_, f64>) -> Result<(Mat<f64>, Vec<f64>), LinalgError> {
    if a.nrows() != a.ncols() {
        return Err(LinalgError::NotSquare {
            what: "a",
            rows: a.nrows(),
            cols: a.ncols(),
        });
    }
    let n = a.nrows();
    if n == 0 {
        return Err(LinalgError::EmptyInput { what: "a" });
    }
    for j in 0..n {
        for i in 0..n {
            if !a[(i, j)].is_finite() {
                return Err(LinalgError::NonFinite { what: "a" });
            }
        }
    }
    // LAPACK's safe range: SFMIN1 = safe minimum / precision.
    let sfmin1 = f64::MIN_POSITIVE / f64::EPSILON;
    let sfmax1 = 1.0 / sfmin1;
    let sfmin2 = sfmin1 * BALANCE_RADIX;
    let sfmax2 = 1.0 / sfmin2;

    let mut b = a.to_owned();
    let mut scale = vec![1.0f64; n];
    for _ in 0..BALANCE_MAX_SWEEPS {
        let mut converged = true;
        for i in 0..n {
            let mut c = norm2((0..n).map(|r| b[(r, i)]));
            let mut r = norm2((0..n).map(|col| b[(i, col)]));
            let mut ca = (0..n).fold(0.0f64, |m, row| m.max(b[(row, i)].abs()));
            let mut ra = (0..n).fold(0.0f64, |m, col| m.max(b[(i, col)].abs()));
            if c == 0.0 || r == 0.0 || !(c + ca + r + ra).is_finite() {
                continue;
            }
            let mut g = r / BALANCE_RADIX;
            let mut f = 1.0f64;
            let s = c + r;
            while c < g && f.max(c).max(ca) < sfmax2 && r.min(g).min(ra) > sfmin2 {
                f *= BALANCE_RADIX;
                c *= BALANCE_RADIX;
                ca *= BALANCE_RADIX;
                r /= BALANCE_RADIX;
                g /= BALANCE_RADIX;
                ra /= BALANCE_RADIX;
            }
            g = c / BALANCE_RADIX;
            while g >= r && r.max(ra) < sfmax2 && f.min(c).min(g).min(ca) > sfmin2 {
                f /= BALANCE_RADIX;
                c /= BALANCE_RADIX;
                g /= BALANCE_RADIX;
                ca /= BALANCE_RADIX;
                r *= BALANCE_RADIX;
                ra *= BALANCE_RADIX;
            }
            if c + r >= BALANCE_FACTOR * s {
                continue;
            }
            if f < 1.0 && scale[i] < 1.0 && f * scale[i] <= sfmin1 {
                continue;
            }
            if f > 1.0 && scale[i] > 1.0 && scale[i] >= sfmax1 / f {
                continue;
            }
            let g = 1.0 / f;
            scale[i] *= f;
            converged = false;
            for col in 0..n {
                b[(i, col)] *= g;
            }
            for row in 0..n {
                b[(row, i)] *= f;
            }
        }
        if converged {
            break;
        }
    }
    Ok((b, scale))
}

/// Eigenvalues of a real square matrix, computed by `faer`'s dense (real
/// Schur) solver on the [`balance`]d matrix — the order of operations of
/// LAPACK `dgeev` (and so of `numpy.linalg.eigvals`).
///
/// Every eigenvalue solve on a companion matrix in the library goes
/// through here, so the roots of a VAR do not depend on the units of its
/// variables (see [`balance`]).
///
/// # Errors
///
/// * everything [`balance`] can return;
/// * [`LinalgError::EigenFailed`] if the QR eigenvalue iteration does not
///   converge.
pub fn balanced_eigenvalues(a: MatRef<'_, f64>) -> Result<Vec<c64>, LinalgError> {
    let (b, _) = balance(a)?;
    b.eigenvalues().map_err(|_| LinalgError::EigenFailed {
        what: "the eigenvalues of the balanced matrix",
    })
}

/// Spectral radius `max_i |lambda_i(A)|` via [`balanced_eigenvalues`]
/// (`faer`'s dense real Schur solver after a power-of-two balancing).
///
/// # Errors
///
/// * [`LinalgError::NotSquare`] / [`LinalgError::EmptyInput`] on shape
///   violations;
/// * [`LinalgError::NonFinite`] on NaN/infinite entries;
/// * [`LinalgError::EigenFailed`] if the QR eigenvalue iteration does not
///   converge.
pub fn spectral_radius(a: MatRef<'_, f64>) -> Result<f64, LinalgError> {
    let eigs = balanced_eigenvalues(a).map_err(|e| match e {
        LinalgError::EigenFailed { .. } => LinalgError::EigenFailed {
            what: "spectral_radius",
        },
        other => other,
    })?;
    Ok(eigs.iter().map(|c| c.re.hypot(c.im)).fold(0.0f64, f64::max))
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
