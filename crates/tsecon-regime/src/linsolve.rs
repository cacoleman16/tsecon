//! A small dense linear solver used by the estimation routines.
//!
//! The Markov-switching machinery only needs to solve a handful of very
//! small systems: the `k`-by-`k` stationary-distribution system for the
//! regime chain and the `k`-by-`k` / `order`-by-`order` normal-equation
//! systems of the EM M-step. Rather than pull the full `tsecon-linalg`
//! factorization surface into this crate, we carry a self-contained
//! Gaussian elimination with partial pivoting (Golub & Van Loan 2013,
//! Algorithm 3.4.1).

use crate::error::RegimeError;

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
