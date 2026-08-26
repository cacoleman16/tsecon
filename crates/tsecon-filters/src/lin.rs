//! Crate-internal dense linear algebra for the regression-based filters.
//!
//! `tsecon-filters` deliberately depends on no linear-algebra crate (see
//! the workspace dependency notes in `tsecon-diag`'s manifest); the
//! problems here are tiny (at most `p^2 x p^2` with `p ~ 12`), so three
//! textbook dense routines suffice:
//!
//! * [`householder_lstsq`] — least squares by Householder QR (moved
//!   verbatim from `hamilton.rs`, so the Hamilton filter's results are
//!   bit-identical to every release since the filter landed);
//! * [`cholesky_solve_spd`] — solve for a symmetric positive-definite
//!   matrix (the Bayesian ridge posterior of the BN filter);
//! * [`lu_solve`] — partial-pivoting LU solve for a general square
//!   system (the `vec`'d Lyapunov equation behind the BN filter's cycle
//!   standard error, and the `(I - F)' w = phi` companion solve).

use crate::error::FiltersError;

/// Least squares `min_beta ||A beta - b||_2` by Householder QR without
/// pivoting (Golub & Van Loan 2013, algorithm 5.2.1).
///
/// `cols` holds the columns of `A` (each of length `m = b.len()`); the
/// factorization overwrites them. Rank deficiency is detected by
/// comparing each diagonal of `R` against a scaled tolerance
/// `m * eps * max_j ||a_j||`.
pub(crate) fn householder_lstsq(
    mut cols: Vec<Vec<f64>>,
    mut b: Vec<f64>,
    what: &'static str,
) -> Result<Vec<f64>, FiltersError> {
    let k = cols.len();
    let m = b.len();
    debug_assert!(m >= k);

    let max_colnorm = cols
        .iter()
        .map(|c| c.iter().map(|v| v * v).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    let tol = m as f64 * f64::EPSILON * max_colnorm;

    let mut v = vec![0.0_f64; m];
    for j in 0..k {
        // Householder vector annihilating rows j+1.. of column j.
        let norm = cols[j][j..].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= tol {
            return Err(FiltersError::RankDeficient { what });
        }
        let alpha = if cols[j][j] >= 0.0 { -norm } else { norm };
        v[j] = cols[j][j] - alpha;
        v[(j + 1)..m].copy_from_slice(&cols[j][(j + 1)..m]);
        let vtv: f64 = v[j..].iter().map(|x| x * x).sum();
        cols[j][j] = alpha; // R[j][j]
        for x in &mut cols[j][(j + 1)..] {
            *x = 0.0;
        }
        // Reflect the remaining columns and the right-hand side:
        // c <- c - 2 v (v'c) / (v'v).
        for col in cols.iter_mut().skip(j + 1) {
            let dot: f64 = v[j..].iter().zip(&col[j..]).map(|(a, c)| a * c).sum();
            let fac = 2.0 * dot / vtv;
            for (ci, vi) in col[j..].iter_mut().zip(&v[j..]) {
                *ci -= fac * vi;
            }
        }
        let dot: f64 = v[j..].iter().zip(&b[j..]).map(|(a, c)| a * c).sum();
        let fac = 2.0 * dot / vtv;
        for (bi, vi) in b[j..].iter_mut().zip(&v[j..]) {
            *bi -= fac * vi;
        }
    }

    // Back substitution R beta = (Q'b)[..k]; R[i][j] = cols[j][i], j >= i.
    let mut beta = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut s = b[i];
        for j in (i + 1)..k {
            s -= cols[j][i] * beta[j];
        }
        let rii = cols[i][i];
        if rii.abs() <= tol {
            return Err(FiltersError::RankDeficient { what });
        }
        beta[i] = s / rii;
    }
    Ok(beta)
}

/// Solve `S x = b` for a symmetric positive-definite `n x n` matrix `S`
/// (row-major) by Cholesky factorization. Errors with
/// [`FiltersError::RankDeficient`] when a pivot is not numerically
/// positive relative to the original diagonal.
pub(crate) fn cholesky_solve_spd(
    s: &[f64],
    n: usize,
    b: &[f64],
    what: &'static str,
) -> Result<Vec<f64>, FiltersError> {
    debug_assert_eq!(s.len(), n * n);
    debug_assert_eq!(b.len(), n);
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut acc = s[i * n + j];
            for m in 0..j {
                acc -= l[i * n + m] * l[j * n + m];
            }
            if i == j {
                let tol = s[i * n + i].abs() * 1e-13;
                if acc <= tol.max(f64::MIN_POSITIVE) {
                    return Err(FiltersError::RankDeficient { what });
                }
                l[i * n + i] = acc.sqrt();
            } else {
                l[i * n + j] = acc / l[j * n + j];
            }
        }
    }
    // Forward substitution: L z = b.
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut acc = b[i];
        for m in 0..i {
            acc -= l[i * n + m] * z[m];
        }
        z[i] = acc / l[i * n + i];
    }
    // Back substitution: L' x = z.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = z[i];
        for m in (i + 1)..n {
            acc -= l[m * n + i] * x[m];
        }
        x[i] = acc / l[i * n + i];
    }
    Ok(x)
}

/// Solve `A x = b` for a general square `n x n` matrix (row-major) by LU
/// factorization with partial pivoting. `A` and `b` are consumed. Errors
/// with [`FiltersError::RankDeficient`] on a numerically zero pivot.
pub(crate) fn lu_solve(
    mut a: Vec<f64>,
    n: usize,
    mut b: Vec<f64>,
    what: &'static str,
) -> Result<Vec<f64>, FiltersError> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);
    let max_abs = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let tol = n as f64 * f64::EPSILON * max_abs;
    for col in 0..n {
        // Partial pivoting: bring the largest remaining entry to the diagonal.
        let mut piv = col;
        let mut piv_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > piv_val {
                piv = row;
                piv_val = v;
            }
        }
        if piv_val <= tol {
            return Err(FiltersError::RankDeficient { what });
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / d;
            if factor == 0.0 {
                continue;
            }
            a[row * n + col] = 0.0;
            for j in (col + 1)..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution on the upper triangle.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= a[i * n + j] * x[j];
        }
        x[i] = acc / a[i * n + i];
    }
    Ok(x)
}
