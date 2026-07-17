//! Parameter covariance estimators: classical MLE (inverse Hessian) and
//! the Bollerslev-Wooldridge (1992) robust sandwich, from numerical
//! derivatives of the log-likelihood.
//!
//! With `A = (1/T) * Hessian of -loglik` and `B = Cov(per-observation
//! scores)` (sample covariance, `ddof = 1`),
//!
//! ```text
//! Cov_mle    = A^{-1} / T                    (information equality)
//! Cov_robust = A^{-1} B A^{-1} / T           (Bollerslev-Wooldridge 1992)
//! ```
//!
//! The finite-difference steps replicate `statsmodels.tools.numdiff` (which
//! is what `arch.compute_param_cov` calls), so the numbers reproduce the
//! `arch` package:
//!
//! * Hessian: four-point central cross differences with
//!   `h_i = eps^(1/4) * max(|theta_i|, 0.1)` (`approx_hess3`);
//! * scores: forward differences with
//!   `h_i = eps^(1/2) * max(|theta_i|, 0.1)` (`approx_fprime`);
//! * `B` demeans the scores and normalizes by `T - 1` (`np.cov`).
//!
//! Measured agreement with the `arch` fixture's robust standard errors is
//! ~1e-6 relative on the normal-innovation cases; near-flat directions
//! (e.g. a Student-t `nu` in the hundreds) are step-size sensitive in *any*
//! implementation and are documented at their achieved tolerance in the
//! golden tests.

use crate::error::GarchError;

/// Standard errors of the parameter vector under both covariance
/// estimators, in parameter order.
///
/// Entries are NaN when the corresponding covariance diagonal is negative
/// (a non-positive-definite numerical Hessian at a flat or boundary
/// optimum) — reported honestly rather than clipped.
#[derive(Debug, Clone, PartialEq)]
pub struct StdErrors {
    /// Classical MLE standard errors, `sqrt(diag(A^{-1} / T))`.
    pub mle: Vec<f64>,
    /// Bollerslev-Wooldridge (1992) robust standard errors,
    /// `sqrt(diag(A^{-1} B A^{-1} / T))` — `arch`'s default (`robust`)
    /// covariance.
    pub robust: Vec<f64>,
}

/// Four-point central-difference Hessian of `f` (the *negative* total
/// log-likelihood), statsmodels `approx_hess3` steps.
fn numerical_hessian<F>(mut f: F, x: &[f64]) -> Result<Vec<Vec<f64>>, GarchError>
where
    F: FnMut(&[f64]) -> Result<f64, GarchError>,
{
    let n = x.len();
    let h: Vec<f64> = x
        .iter()
        .map(|&v| f64::EPSILON.powf(0.25) * v.abs().max(0.1))
        .collect();
    let mut hess = vec![vec![0.0; n]; n];
    let mut probe = x.to_vec();
    let mut eval = |probe: &mut Vec<f64>, di: (usize, f64), dj: (usize, f64)| {
        probe.copy_from_slice(x);
        probe[di.0] += di.1;
        probe[dj.0] += dj.1;
        f(probe)
    };
    for i in 0..n {
        for j in i..n {
            let fpp = eval(&mut probe, (i, h[i]), (j, h[j]))?;
            let fpm = eval(&mut probe, (i, h[i]), (j, -h[j]))?;
            let fmp = eval(&mut probe, (i, -h[i]), (j, h[j]))?;
            let fmm = eval(&mut probe, (i, -h[i]), (j, -h[j]))?;
            let v = ((fpp - fpm) - (fmp - fmm)) / (4.0 * h[i] * h[j]);
            hess[i][j] = v;
            hess[j][i] = v;
        }
    }
    Ok(hess)
}

/// Forward-difference per-observation score matrix (`T x k`) of the
/// negative log-likelihood contributions, statsmodels `approx_fprime`
/// steps. (The sign is irrelevant for the score covariance.)
fn numerical_scores<G>(mut g: G, x: &[f64]) -> Result<Vec<Vec<f64>>, GarchError>
where
    G: FnMut(&[f64]) -> Result<Vec<f64>, GarchError>,
{
    let k = x.len();
    let base = g(x)?;
    let nobs = base.len();
    let mut scores = vec![vec![0.0; k]; nobs];
    let mut probe = x.to_vec();
    for i in 0..k {
        let h = f64::EPSILON.sqrt() * x[i].abs().max(0.1);
        probe.copy_from_slice(x);
        probe[i] += h;
        let shifted = g(&probe)?;
        if shifted.len() != nobs {
            return Err(GarchError::DimensionMismatch {
                what: "per-observation log-likelihood",
                expected: nobs,
                actual: shifted.len(),
            });
        }
        for (t, row) in scores.iter_mut().enumerate() {
            row[i] = (shifted[t] - base[t]) / h;
        }
    }
    Ok(scores)
}

/// Inverts a small symmetric matrix by Gauss-Jordan elimination with
/// partial pivoting.
///
/// # Errors
///
/// [`GarchError::SingularHessian`] when a pivot is (numerically) zero.
fn invert(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, GarchError> {
    let n = a.len();
    // Augmented [A | I], reduced in place.
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for col in 0..n {
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                m[r1][col]
                    .abs()
                    .partial_cmp(&m[r2][col].abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .ok_or(GarchError::SingularHessian)?;
        let pivot = m[pivot_row][col];
        if !pivot.is_finite() || pivot.abs() < 1e-300 {
            return Err(GarchError::SingularHessian);
        }
        m.swap(col, pivot_row);
        for v in m[col].iter_mut() {
            *v /= pivot;
        }
        let pivot_vals = m[col].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == col {
                continue;
            }
            let factor = row[col];
            if factor != 0.0 {
                for (v, &pv) in row.iter_mut().zip(&pivot_vals) {
                    *v -= factor * pv;
                }
            }
        }
    }
    Ok(m.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// `A * B` for small square matrices.
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for (ci, ai) in c.iter_mut().zip(a) {
        for (k, &aik) in ai.iter().enumerate() {
            for (cij, &bkj) in ci.iter_mut().zip(&b[k]) {
                *cij += aik * bkj;
            }
        }
    }
    c
}

/// Computes both standard-error vectors at `params`.
///
/// `total` is the negative total log-likelihood; `per_obs` its
/// per-observation contributions (length `nobs`).
///
/// # Errors
///
/// [`GarchError::SingularHessian`] if the numerical Hessian cannot be
/// inverted; any error the likelihood evaluations raise at a probe point
/// (e.g. a boundary optimum whose finite-difference probe leaves the
/// admissible region).
pub(crate) fn std_errors<F, G>(
    total: F,
    per_obs: G,
    params: &[f64],
    nobs: usize,
) -> Result<StdErrors, GarchError>
where
    F: FnMut(&[f64]) -> Result<f64, GarchError>,
    G: FnMut(&[f64]) -> Result<Vec<f64>, GarchError>,
{
    let k = params.len();
    let t = nobs as f64;
    let mut hess = numerical_hessian(total, params)?;
    for row in &mut hess {
        for v in row.iter_mut() {
            *v /= t;
        }
    }
    let a_inv = invert(&hess)?;

    let scores = numerical_scores(per_obs, params)?;
    // Demeaned sample covariance of the scores, ddof = 1 (np.cov).
    let mut mean = vec![0.0; k];
    for row in &scores {
        for (m, &s) in mean.iter_mut().zip(row) {
            *m += s;
        }
    }
    for m in &mut mean {
        *m /= t;
    }
    let mut b = vec![vec![0.0; k]; k];
    for row in &scores {
        for i in 0..k {
            let di = row[i] - mean[i];
            for j in 0..k {
                b[i][j] += di * (row[j] - mean[j]);
            }
        }
    }
    let ddof = (nobs.saturating_sub(1)).max(1) as f64;
    for row in &mut b {
        for v in row.iter_mut() {
            *v /= ddof;
        }
    }

    let sandwich = matmul(&matmul(&a_inv, &b), &a_inv);
    let se = |m: &[Vec<f64>]| -> Vec<f64> {
        m.iter()
            .enumerate()
            .map(|(i, row)| (row[i] / t).sqrt()) // negative diag -> NaN, kept.
            .collect()
    };
    Ok(StdErrors {
        mle: se(&a_inv),
        robust: se(&sandwich),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn invert_recovers_identity() {
        let a = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let inv = invert(&a).unwrap();
        let prod = matmul(&a, &inv);
        for (i, row) in prod.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((v - target).abs() < 1e-12, "prod[{i}][{j}] = {v}");
            }
        }
    }

    #[test]
    fn invert_rejects_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(matches!(invert(&a), Err(GarchError::SingularHessian)));
    }
}
