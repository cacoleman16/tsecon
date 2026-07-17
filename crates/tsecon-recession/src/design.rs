//! Input validation and small dense-linear-algebra plumbing shared by the
//! static and dynamic estimators.
//!
//! Matrix factorizations delegate to `faer` (re-exported through
//! `tsecon-linalg`, the workspace's single dense backend), matching the idiom
//! of the sibling `tsecon-gmm::matrix` / `tsecon-predreg` crates. This crate
//! never hand-rolls a Cholesky or an inverse.

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, MatRef, Side};

use crate::error::RecessionError;

/// Validate the binary response `y` and the design columns `x`, returning the
/// sample size `n` and the number of parameters `k = x.len()`.
///
/// Checks, in order: `y` non-empty; at least one design column; every column
/// the same length as `y`; every value finite; `y` coded in `{0, 1}`; `y`
/// not degenerate (all-zero / all-one); and `n > k` degrees of freedom.
pub(crate) fn validate(y: &[f64], x: &[Vec<f64>]) -> Result<(usize, usize), RecessionError> {
    let n = y.len();
    if n == 0 {
        return Err(RecessionError::EmptyInput { what: "y" });
    }
    if x.is_empty() {
        return Err(RecessionError::NoRegressors);
    }
    let k = x.len();
    for (j, col) in x.iter().enumerate() {
        if col.len() != n {
            return Err(RecessionError::DimensionMismatch {
                what: leak_col_name(j),
                expected: n,
                got: col.len(),
            });
        }
        for &v in col.iter() {
            if !v.is_finite() {
                return Err(RecessionError::NonFinite { what: "x" });
            }
        }
    }
    let mut ones = 0usize;
    for (t, &yt) in y.iter().enumerate() {
        if !yt.is_finite() {
            return Err(RecessionError::NonFinite { what: "y" });
        }
        if yt != 0.0 && yt != 1.0 {
            return Err(RecessionError::NonBinaryResponse {
                index: t,
                value: yt,
            });
        }
        if yt == 1.0 {
            ones += 1;
        }
    }
    if ones == 0 || ones == n {
        return Err(RecessionError::Degenerate { ones, n });
    }
    if n <= k {
        return Err(RecessionError::DegreesOfFreedom { n, k });
    }
    Ok((n, k))
}

/// A stable static name for the j-th design column, used only in error
/// messages. Falls back to a generic label past the first few columns.
fn leak_col_name(j: usize) -> &'static str {
    match j {
        0 => "x column 0",
        1 => "x column 1",
        2 => "x column 2",
        _ => "an x column",
    }
}

/// The linear index `x_t' beta` for every observation `t`.
pub(crate) fn linear_index(x: &[Vec<f64>], beta: &[f64], n: usize) -> Vec<f64> {
    let mut idx = vec![0.0_f64; n];
    for (j, col) in x.iter().enumerate() {
        let bj = beta[j];
        for (t, &xtj) in col.iter().enumerate() {
            idx[t] += bj * xtj;
        }
    }
    idx
}

/// Invert a symmetric positive-definite `k x k` matrix (given row-major) via
/// its Cholesky factor, returning the inverse row-major. Fails with
/// [`RecessionError::SingularInformation`] if the factorization rejects the
/// matrix (indefinite or singular).
pub(crate) fn inv_spd_rowmajor(info: &[f64], k: usize) -> Result<Vec<f64>, RecessionError> {
    let m = Mat::from_fn(k, k, |i, j| info[i * k + j]);
    let inv = m
        .llt(Side::Lower)
        .map_err(|_| RecessionError::SingularInformation)?
        .inverse();
    Ok(mat_to_rowmajor(inv.as_ref()))
}

/// Copy a dense matrix into a row-major `Vec<f64>`.
fn mat_to_rowmajor(m: MatRef<'_, f64>) -> Vec<f64> {
    let (nr, nc) = (m.nrows(), m.ncols());
    let mut out = vec![0.0_f64; nr * nc];
    for i in 0..nr {
        for j in 0..nc {
            out[i * nc + j] = m[(i, j)];
        }
    }
    out
}

/// The intercept-only log-likelihood, whose MLE probability is
/// `ybar = ones / n`, giving `LL_null = n [ybar ln ybar + (1-ybar) ln(1-ybar)]`.
/// This is McFadden's denominator and is identical for the probit and logit
/// (both intercept-only models fit `p = ybar` exactly).
pub(crate) fn loglik_null(ones: usize, n: usize) -> f64 {
    let ybar = ones as f64 / n as f64;
    n as f64 * (ybar * ybar.ln() + (1.0 - ybar) * (1.0 - ybar).ln())
}

/// Standard errors and z-statistics from a coefficient vector and its
/// row-major covariance: `se_j = sqrt(cov_jj)`, `z_j = beta_j / se_j`.
pub(crate) fn se_and_z(params: &[f64], cov: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut se = vec![0.0_f64; k];
    let mut z = vec![0.0_f64; k];
    for j in 0..k {
        let var = cov[j * k + j];
        se[j] = if var > 0.0 { var.sqrt() } else { f64::NAN };
        z[j] = params[j] / se[j];
    }
    (se, z)
}
