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
//! The finite-difference formulas are `statsmodels.tools.numdiff`'s (which
//! is what `arch.compute_param_cov` calls):
//!
//! * Hessian: four-point central cross differences with
//!   `h_i = eps^(1/4) * s_i` (`approx_hess3`);
//! * scores: forward differences with `h_i = eps^(1/2) * s_i`
//!   (`approx_fprime`);
//! * `B` demeans the scores and normalizes by `T - 1` (`np.cov`).
//!
//! `s_i` is the *step scale* of coordinate `i`: the length over which the
//! log-likelihood varies appreciably in that direction. statsmodels, which
//! knows nothing about the model, hardcodes `s_i = max(|theta_i|, 0.1)` —
//! an absolute floor. That floor is a bug the moment a parameter carries
//! the units of the data: `omega` for daily equity returns quoted in
//! decimals is around `1e-6`, so a step floored at `0.1` is *larger than
//! the parameter*, the probe leaves `omega > 0`, and the whole covariance
//! comes back NaN. Just above that scale the floor does not fail outright,
//! it silently biases: at `omega = 8e-5` the robust standard errors are
//! ~15% off.
//!
//! So the caller supplies `s_i` per parameter, in the units that parameter
//! actually has — see `GarchModel::step_scales`. At the
//! percent-return scale `arch` is fitted on, those scales reduce to
//! statsmodels' `max(|theta_i|, 0.1)` to within a small factor, and the
//! golden fixture still pins agreement with `arch`'s robust standard
//! errors; unlike statsmodels', they are equivariant under `y -> c * y`.
//!
//! Near-flat directions (e.g. a Student-t `nu` in the hundreds) remain
//! step-size sensitive in *any* implementation and are documented at their
//! achieved tolerance in the golden tests.
//!
//! **Active boundaries** (audit round 7). When the QMLE lands on a
//! constraint — `alpha` at its sign bound, persistence at 1 — the observed
//! information is singular in the constrained direction *by construction*,
//! and a central-difference probe there leaves the admissible region, so
//! no full-vector covariance exists. The caller passes a `free` mask and
//! this module computes the reduced Hessian/score covariance over the
//! interior directions only, reporting finite standard errors for those
//! and NaN + `se_valid = false` + `boundary = true` for the constrained
//! ones. That is the same honesty contract as `tsecon-evt`'s `se_valid`.

use crate::error::GarchError;

/// Standard errors of the parameter vector under both covariance
/// estimators, in parameter order.
///
/// Entries are NaN when the parameter sits at an active constraint
/// boundary (see [`StdErrors::boundary`]) or when the corresponding
/// covariance diagonal is negative (a non-positive-definite numerical
/// Hessian at a flat optimum) — reported honestly rather than clipped,
/// and flagged per parameter in [`StdErrors::se_valid`].
#[derive(Debug, Clone, PartialEq)]
pub struct StdErrors {
    /// Classical MLE standard errors, `sqrt(diag(A^{-1} / T))`.
    pub mle: Vec<f64>,
    /// Bollerslev-Wooldridge (1992) robust standard errors,
    /// `sqrt(diag(A^{-1} B A^{-1} / T))` — `arch`'s default (`robust`)
    /// covariance.
    pub robust: Vec<f64>,
    /// Per parameter: `true` when the parameter was interior (not at a
    /// constraint boundary) and both standard errors came out finite. A
    /// `false` entry means the reported NaN (or a non-finite value) is a
    /// statement — standard QMLE asymptotics do not back a standard error
    /// for that parameter at this point.
    pub se_valid: Vec<bool>,
    /// Per parameter: `true` when the parameter sits at an active
    /// constraint boundary at the resolution of the finite-difference
    /// probes (a coefficient at its sign constraint, or every coefficient
    /// in an active persistence/`|sum(beta)|` constraint). The observed
    /// information is singular in the constrained direction by
    /// construction, so such parameters are excluded from the Hessian and
    /// their standard errors reported as NaN with `se_valid = false`.
    pub boundary: Vec<bool>,
}

impl StdErrors {
    /// The all-NaN, nothing-valid report for `k` parameters with the given
    /// boundary flags — used when even the reduced (interior-direction)
    /// Hessian is singular or a probe fails.
    pub(crate) fn all_invalid(k: usize, boundary: Vec<bool>) -> Self {
        Self {
            mle: vec![f64::NAN; k],
            robust: vec![f64::NAN; k],
            se_valid: vec![false; k],
            boundary,
        }
    }
}

/// Validates the caller's step scales: one strictly positive, finite
/// entry per parameter.
fn check_step_scales(step_scale: &[f64], k: usize) -> Result<(), GarchError> {
    if step_scale.len() != k {
        return Err(GarchError::DimensionMismatch {
            what: "finite-difference step scales",
            expected: k,
            actual: step_scale.len(),
        });
    }
    if let Some(i) = step_scale.iter().position(|s| !(*s > 0.0 && s.is_finite())) {
        return Err(GarchError::NonFinite {
            what: "a finite-difference step scale (each must be strictly positive and finite)",
            at: Some(i),
        });
    }
    Ok(())
}

/// Four-point central-difference Hessian of `f` (the *negative* total
/// log-likelihood) over the coordinates in `idx`, `approx_hess3` formula
/// with caller-supplied step scales: `h_i = eps^(1/4) * step_scale[i]`.
/// Coordinates of `x` not in `idx` are held fixed at their values (an
/// active-boundary parameter is never probed). With `idx = 0..n` this is
/// the full Hessian, evaluated in exactly the order (and arithmetic) of
/// the unmasked implementation.
fn numerical_hessian<F>(
    mut f: F,
    x: &[f64],
    step_scale: &[f64],
    idx: &[usize],
) -> Result<Vec<Vec<f64>>, GarchError>
where
    F: FnMut(&[f64]) -> Result<f64, GarchError>,
{
    check_step_scales(step_scale, x.len())?;
    let m = idx.len();
    let h: Vec<f64> = step_scale
        .iter()
        .map(|&s| f64::EPSILON.powf(0.25) * s)
        .collect();
    let mut hess = vec![vec![0.0; m]; m];
    let mut probe = x.to_vec();
    let mut eval = |probe: &mut Vec<f64>, di: (usize, f64), dj: (usize, f64)| {
        probe.copy_from_slice(x);
        probe[di.0] += di.1;
        probe[dj.0] += dj.1;
        f(probe)
    };
    for (a, &i) in idx.iter().enumerate() {
        for (b, &j) in idx.iter().enumerate().skip(a) {
            let fpp = eval(&mut probe, (i, h[i]), (j, h[j]))?;
            let fpm = eval(&mut probe, (i, h[i]), (j, -h[j]))?;
            let fmp = eval(&mut probe, (i, -h[i]), (j, h[j]))?;
            let fmm = eval(&mut probe, (i, -h[i]), (j, -h[j]))?;
            let v = ((fpp - fpm) - (fmp - fmm)) / (4.0 * h[i] * h[j]);
            hess[a][b] = v;
            hess[b][a] = v;
        }
    }
    Ok(hess)
}

/// Forward-difference per-observation score matrix (`T x len(idx)`) of the
/// negative log-likelihood contributions over the coordinates in `idx`,
/// `approx_fprime` formula with caller-supplied step scales:
/// `h_i = eps^(1/2) * step_scale[i]`. (The sign is irrelevant for the
/// score covariance.)
fn numerical_scores<G>(
    mut g: G,
    x: &[f64],
    step_scale: &[f64],
    idx: &[usize],
) -> Result<Vec<Vec<f64>>, GarchError>
where
    G: FnMut(&[f64]) -> Result<Vec<f64>, GarchError>,
{
    check_step_scales(step_scale, x.len())?;
    let base = g(x)?;
    let nobs = base.len();
    let mut scores = vec![vec![0.0; idx.len()]; nobs];
    let mut probe = x.to_vec();
    for (a, &i) in idx.iter().enumerate() {
        let h = f64::EPSILON.sqrt() * step_scale[i];
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
            row[a] = (shifted[t] - base[t]) / h;
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

/// Computes both standard-error vectors at `params`, over the free
/// (interior) coordinates only.
///
/// `total` is the negative total log-likelihood; `per_obs` its
/// per-observation contributions (length `nobs`). `step_scale` gives the
/// finite-difference step scale of each coordinate — one strictly
/// positive, finite entry per parameter, in that parameter's own units
/// (see the module docs). `free[i] = false` marks a coordinate at an
/// active constraint boundary: it is held fixed (never probed — a central
/// difference there would leave the admissible region), excluded from the
/// Hessian and score covariance, and reported as `se = NaN`,
/// `se_valid = false`, `boundary = true`. With every coordinate free the
/// computation is bit-identical to the classical full-vector one.
///
/// # Errors
///
/// [`GarchError::DimensionMismatch`] / [`GarchError::NonFinite`] if
/// `step_scale`/`free` is the wrong length or `step_scale` holds a
/// non-positive entry; [`GarchError::SingularHessian`] if the (reduced)
/// numerical Hessian cannot be inverted, or if no coordinate is free at
/// all; any error the likelihood evaluations raise at a probe point.
pub(crate) fn std_errors<F, G>(
    total: F,
    per_obs: G,
    params: &[f64],
    nobs: usize,
    step_scale: &[f64],
    free: &[bool],
) -> Result<StdErrors, GarchError>
where
    F: FnMut(&[f64]) -> Result<f64, GarchError>,
    G: FnMut(&[f64]) -> Result<Vec<f64>, GarchError>,
{
    let k = params.len();
    if free.len() != k {
        return Err(GarchError::DimensionMismatch {
            what: "free-coordinate mask",
            expected: k,
            actual: free.len(),
        });
    }
    let idx: Vec<usize> = (0..k).filter(|&i| free[i]).collect();
    if idx.is_empty() {
        // Every parameter at a boundary: there is no interior direction to
        // differentiate along.
        return Err(GarchError::SingularHessian);
    }
    let m = idx.len();
    let t = nobs as f64;
    let mut hess = numerical_hessian(total, params, step_scale, &idx)?;
    for row in &mut hess {
        for v in row.iter_mut() {
            *v /= t;
        }
    }
    let a_inv = invert(&hess)?;

    let scores = numerical_scores(per_obs, params, step_scale, &idx)?;
    // Demeaned sample covariance of the scores, ddof = 1 (np.cov).
    let mut mean = vec![0.0; m];
    for row in &scores {
        for (mn, &s) in mean.iter_mut().zip(row) {
            *mn += s;
        }
    }
    for mn in &mut mean {
        *mn /= t;
    }
    let mut b = vec![vec![0.0; m]; m];
    for row in &scores {
        for i in 0..m {
            let di = row[i] - mean[i];
            for j in 0..m {
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
    // Scatter the reduced diagonals back to full parameter order; fixed
    // (boundary) coordinates stay NaN.
    let se = |mat: &[Vec<f64>]| -> Vec<f64> {
        let mut out = vec![f64::NAN; k];
        for (a, &i) in idx.iter().enumerate() {
            out[i] = (mat[a][a] / t).sqrt(); // negative diag -> NaN, kept.
        }
        out
    };
    let mle = se(&a_inv);
    let robust = se(&sandwich);
    let se_valid: Vec<bool> = (0..k)
        .map(|i| free[i] && mle[i].is_finite() && robust[i].is_finite())
        .collect();
    Ok(StdErrors {
        mle,
        robust,
        se_valid,
        boundary: free.iter().map(|f| !f).collect(),
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

    #[test]
    fn step_scales_are_validated() {
        assert!(matches!(
            check_step_scales(&[1.0, 1.0], 3),
            Err(GarchError::DimensionMismatch { expected: 3, .. })
        ));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    check_step_scales(&[1.0, bad], 2),
                    Err(GarchError::NonFinite { at: Some(1), .. })
                ),
                "step scale {bad} should be rejected"
            );
        }
        assert!(check_step_scales(&[1.0, 1e-9], 2).is_ok());
    }

    /// The step scale is honored, and a coordinate many orders of
    /// magnitude below an absolute floor is differentiated correctly.
    ///
    /// `f` is a quadratic in `(x0, x1)` whose second coordinate lives at
    /// `1e-6` and which is *undefined for `x1 <= 0`* — exactly the shape of
    /// the GARCH likelihood in `omega`. A step scaled to `x1` succeeds; the
    /// `max(|x1|, 0.1)` floor statsmodels would use pushes the probe past
    /// zero and the evaluation refuses the point.
    #[test]
    fn step_scale_reaches_a_tiny_positive_coordinate() {
        let x = [1.0, 1e-6];
        let f = |p: &[f64]| -> Result<f64, GarchError> {
            if p[1] <= 0.0 {
                return Err(GarchError::InvalidParameter {
                    name: "x1",
                    value: p[1],
                    requirement: "x1 > 0",
                });
            }
            Ok(0.5 * p[0] * p[0] + 0.5 * (p[1] / 1e-6).powi(2))
        };
        let good = numerical_hessian(f, &x, &[1.0, 1e-6], &[0, 1]).unwrap();
        assert!((good[0][0] - 1.0).abs() < 1e-6, "d2/dx0^2 = {}", good[0][0]);
        assert!(
            (good[1][1] - 1e12).abs() < 1e12 * 1e-6,
            "d2/dx1^2 = {}",
            good[1][1]
        );

        let floored = numerical_hessian(f, &x, &[1.0, 0.1], &[0, 1]);
        assert!(
            matches!(floored, Err(GarchError::InvalidParameter { .. })),
            "an absolute floor should drive x1 non-positive, got {floored:?}"
        );

        // Masking the boundary coordinate makes the same floored step
        // harmless: x1 is never probed, and the reduced Hessian over x0
        // alone is exact.
        let masked = numerical_hessian(f, &x, &[1.0, 0.1], &[0]).unwrap();
        assert_eq!(masked.len(), 1);
        assert!(
            (masked[0][0] - 1.0).abs() < 1e-6,
            "reduced d2/dx0^2 = {}",
            masked[0][0]
        );
    }

    /// End-to-end: for a Gaussian log-likelihood the inverse Hessian is the
    /// exact covariance, so the MLE standard errors are known in closed
    /// form even when a coordinate sits at `1e-6`.
    #[test]
    fn std_errors_recover_a_known_covariance() {
        let nobs = 100usize;
        let x = [2.0, 1e-6];
        // Per-observation negative log-likelihood, curvature (4, 1e12).
        let per_obs = |p: &[f64]| -> Result<Vec<f64>, GarchError> {
            let v = 0.5 * 4.0 * (p[0] - 2.0).powi(2) + 0.5 * 1e12 * (p[1] - 1e-6).powi(2);
            Ok(vec![v / nobs as f64 + p[0] * 1e-9; nobs])
        };
        let total = |p: &[f64]| -> Result<f64, GarchError> { Ok(per_obs(p)?.iter().sum()) };
        let se = std_errors(total, per_obs, &x, nobs, &[2.0, 1e-6], &[true, true]).unwrap();
        // A = H/T, Cov = A^{-1}/T = H^{-1}: se = 1/sqrt(diag(H)).
        assert!((se.mle[0] - 0.5).abs() < 1e-6, "se[0] = {}", se.mle[0]);
        assert!((se.mle[1] - 1e-6).abs() < 1e-12, "se[1] = {}", se.mle[1]);
        assert!(se.robust.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert_eq!(se.se_valid, vec![true, true]);
        assert_eq!(se.boundary, vec![false, false]);
        // The step-scale and mask contracts are enforced through this
        // entry point too.
        assert!(std_errors(total, per_obs, &x, nobs, &[2.0], &[true, true]).is_err());
        assert!(std_errors(total, per_obs, &x, nobs, &[2.0, 1e-6], &[true]).is_err());
        assert!(matches!(
            std_errors(total, per_obs, &x, nobs, &[2.0, 1e-6], &[false, false]),
            Err(GarchError::SingularHessian)
        ));
    }

    /// Masking a coordinate at a boundary: the free coordinate keeps its
    /// exact closed-form standard error, the fixed one reports NaN with
    /// `se_valid = false` and `boundary = true`.
    #[test]
    fn std_errors_masked_scatter_boundary_coordinate() {
        let nobs = 100usize;
        let x = [2.0, 0.0]; // x1 pinned at its (hypothetical) lower bound
        let per_obs = |p: &[f64]| -> Result<Vec<f64>, GarchError> {
            if p[1] < 0.0 {
                return Err(GarchError::InvalidParameter {
                    name: "x1",
                    value: p[1],
                    requirement: "x1 >= 0",
                });
            }
            let v = 0.5 * 4.0 * (p[0] - 2.0).powi(2) + p[1];
            Ok(vec![v / nobs as f64 + p[0] * 1e-9; nobs])
        };
        let total = |p: &[f64]| -> Result<f64, GarchError> { Ok(per_obs(p)?.iter().sum()) };
        // Unmasked, the central difference on x1 crosses zero and fails.
        assert!(matches!(
            std_errors(total, per_obs, &x, nobs, &[2.0, 0.1], &[true, true]),
            Err(GarchError::InvalidParameter { .. })
        ));
        // Masked, the interior coordinate is differentiated exactly.
        let se = std_errors(total, per_obs, &x, nobs, &[2.0, 0.1], &[true, false]).unwrap();
        assert!((se.mle[0] - 0.5).abs() < 1e-6, "se[0] = {}", se.mle[0]);
        assert!(se.mle[1].is_nan() && se.robust[1].is_nan());
        assert_eq!(se.se_valid, vec![true, false]);
        assert_eq!(se.boundary, vec![false, true]);
    }
}
