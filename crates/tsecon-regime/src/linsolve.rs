//! Small dense linear solvers used by the estimation routines.
//!
//! The Markov-switching machinery only needs to solve a handful of very
//! small systems: the `k`-by-`k` stationary-distribution system for the
//! regime chain and the `k`-by-`k` / `order`-by-`order` normal-equation
//! systems of the EM M-step. Rather than pull the full `tsecon-linalg`
//! factorization surface into this crate, we carry a self-contained
//! Gaussian elimination with partial pivoting (Golub & Van Loan 2013,
//! Algorithm 3.4.1), plus the Cholesky pair the SETAR and threshold-VAR
//! grid scans factor their thousands of tiny Gram matrices with.

use crate::error::RegimeError;

/// Lower-triangular Cholesky factor of the symmetric positive-definite
/// row-major `k x k` matrix `a`; `None` if a pivot is not strictly
/// positive.
pub(crate) fn cholesky(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i * k + j];
            for m in 0..j {
                sum -= l[i * k + m] * l[j * k + m];
            }
            if i == j {
                if !(sum > 0.0 && sum.is_finite()) {
                    return None;
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }
    Some(l)
}

/// Solve `L L' x = b` given the lower Cholesky factor `l`.
pub(crate) fn chol_solve(l: &[f64], k: usize, b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    for i in 0..k {
        for j in 0..i {
            x[i] -= l[i * k + j] * x[j];
        }
        x[i] /= l[i * k + i];
    }
    for i in (0..k).rev() {
        for j in (i + 1)..k {
            x[i] -= l[j * k + i] * x[j];
        }
        x[i] /= l[i * k + i];
    }
    x
}

/// Solves `a x = b` for `x` by Gaussian elimination with partial pivoting.
///
/// `a` is a row-major `n`-by-`n` matrix (`a[i * n + j]`) and `b` has length
/// `n`; both are consumed as scratch space. Returns
/// [`RegimeError::Singular`] (tagged with `what`) if a pivot is numerically
/// zero.
pub(crate) fn solve(
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    n: usize,
    what: &'static str,
) -> Result<Vec<f64>, RegimeError> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);

    for col in 0..n {
        // Partial pivot: largest magnitude entry at or below the diagonal.
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let mag = a[row * n + col].abs();
            if mag > best {
                best = mag;
                pivot = row;
            }
        }
        if best < 1e-300 {
            return Err(RegimeError::Singular { what });
        }
        if pivot != col {
            for j in 0..n {
                a.swap(pivot * n + j, col * n + j);
            }
            b.swap(pivot, col);
        }

        let diag = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / diag;
            if factor != 0.0 {
                for j in col..n {
                    a[row * n + j] -= factor * a[col * n + j];
                }
                b[row] -= factor * b[col];
            }
        }
    }

    // Back-substitution.
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = b[row];
        for j in (row + 1)..n {
            acc -= a[row * n + j] * x[j];
        }
        x[row] = acc / a[row * n + row];
    }
    Ok(x)
}
