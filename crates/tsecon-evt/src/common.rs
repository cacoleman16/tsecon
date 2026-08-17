//! Shared numerical pieces: the `ln(1 + xi x) / xi` kernel with its
//! documented Gumbel/exponential-limit branch, the numpy-convention
//! empirical quantile, the observed-information standard errors, and input
//! validation.

use crate::error::EvtError;

/// The `|xi|` cutoff below which the likelihood kernels switch to the
/// exponential/Gumbel-limit expansion.
///
/// For `|xi| >= XI_CUTOFF` the exact form `ln_1p(xi * x) / xi` is used —
/// numerically healthy down to tiny `xi` because `ln_1p` is accurate near
/// zero and the division introduces no cancellation. Below the cutoff the
/// second-order expansion `x (1 - xi x / 2 + (xi x)^2 / 3)` takes over,
/// removing the `xi = 0` singularity; its truncation error is
/// `O(xi^3 x^4)`, which at the cutoff is below `1e-16` for any
/// standardized residual `|x| < 100`, so the two branches agree to
/// floating-point noise across the seam (pinned by the property tests).
pub const XI_CUTOFF: f64 = 1e-8;

/// `ln(1 + xi * x) / xi`, the kernel both the GPD and GEV log-likelihoods
/// are built from, with the [`XI_CUTOFF`] limit branch.
///
/// Returns `+infinity` when the support constraint `1 + xi x > 0` is
/// violated (the caller maps that to a `-infinity` log-likelihood /
/// `+infinity` negative log-likelihood, matching scipy's `logpdf`
/// convention of `-inf` outside the support).
#[inline]
pub(crate) fn ln1p_over_xi(xi: f64, x: f64) -> f64 {
    if xi.abs() < XI_CUTOFF {
        let t = xi * x;
        x * (1.0 - 0.5 * t + t * t / 3.0)
    } else {
        let t = xi * x;
        if t <= -1.0 {
            return f64::INFINITY;
        }
        t.ln_1p() / xi
    }
}

/// The empirical quantile in numpy's default convention
/// (`np.quantile(..., method="linear")`, Hyndman-Fan type 7): virtual
/// index `h = q (n - 1)` on the sorted sample, linearly interpolated —
/// including numpy's two-sided `_lerp` (interpolate from the upper order
/// statistic once the fraction reaches 0.5) so the fixture generator's
/// `np.quantile` threshold is reproduced to the last bit in the common
/// case.
///
/// `sorted_y` must be non-empty and ascending; `q` strictly inside (0, 1)
/// (both enforced by the callers).
pub(crate) fn np_quantile_linear(sorted_y: &[f64], q: f64) -> f64 {
    let n = sorted_y.len();
    if n == 1 {
        return sorted_y[0];
    }
    let h = q * (n - 1) as f64;
    let i = h.floor();
    let g = h - i;
    let idx = i as usize;
    let lo = sorted_y[idx];
    if g == 0.0 {
        return lo;
    }
    let hi = sorted_y[(idx + 1).min(n - 1)];
    let d = hi - lo;
    if g < 0.5 {
        lo + d * g
    } else {
        hi - d * (1.0 - g)
    }
}

/// Observed-information standard errors: the numerical Hessian of the
/// negative log-likelihood at the MLE, inverted, square-rooted on the
/// diagonal.
///
/// The Hessian uses the four-point central cross-difference formula
/// (statsmodels `approx_hess3`, the same scheme `tsecon-garch` uses and
/// the fixture generator mirrors) with steps `h_i = eps^(1/4) *
/// step_scale[i]`, where `step_scale` carries each parameter's own units
/// (shape: `max(|xi|, 0.1)`; scale/location: the fitted scale) so the
/// probes are equivariant under `y -> c y`.
///
/// Returns `None` when the Hessian cannot be formed (a probe left the
/// support, so `nll` returned non-finite) or is not positive definite
/// (flat or boundary optimum) — the callers report NaN standard errors
/// and `se_valid = false` rather than fabricating numbers.
pub(crate) fn observed_info_ses<F>(mut nll: F, x: &[f64], step_scale: &[f64]) -> Option<Vec<f64>>
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x.len();
    let h: Vec<f64> = step_scale
        .iter()
        .map(|&s| f64::EPSILON.powf(0.25) * s)
        .collect();
    let mut hess = vec![vec![0.0_f64; n]; n];
    let mut probe = vec![0.0_f64; n];
    let mut eval = |probe: &mut Vec<f64>, di: (usize, f64), dj: (usize, f64)| {
        probe.copy_from_slice(x);
        probe[di.0] += di.1;
        probe[dj.0] += dj.1;
        nll(probe)
    };
    for i in 0..n {
        for j in i..n {
            let fpp = eval(&mut probe, (i, h[i]), (j, h[j]));
            let fpm = eval(&mut probe, (i, h[i]), (j, -h[j]));
            let fmp = eval(&mut probe, (i, -h[i]), (j, h[j]));
            let fmm = eval(&mut probe, (i, -h[i]), (j, -h[j]));
            let v = ((fpp - fpm) - (fmp - fmm)) / (4.0 * h[i] * h[j]);
            if !v.is_finite() {
                return None;
            }
            hess[i][j] = v;
            hess[j][i] = v;
        }
    }
    let cov = invert_spd(&hess)?;
    let mut ses = Vec::with_capacity(n);
    for (i, row) in cov.iter().enumerate() {
        let d = row[i];
        if !(d > 0.0 && d.is_finite()) {
            return None;
        }
        ses.push(d.sqrt());
    }
    Some(ses)
}

/// Inverts a small symmetric matrix by Gauss-Jordan elimination with
/// partial pivoting; `None` on a (numerically) singular pivot.
fn invert_spd(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
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
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            m[r1][col]
                .abs()
                .partial_cmp(&m[r2][col].abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        })?;
        let pivot = m[pivot_row][col];
        if !pivot.is_finite() || pivot.abs() < 1e-300 {
            return None;
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
    Some(m.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// Validates a series: non-empty and all-finite.
pub(crate) fn check_series(y: &[f64], what: &'static str) -> Result<(), EvtError> {
    if y.is_empty() {
        return Err(EvtError::EmptyInput { what });
    }
    if let Some(index) = y.iter().position(|v| !v.is_finite()) {
        return Err(EvtError::NonFinite { what, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_matches_definition() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(np_quantile_linear(&s, 0.5), 3.0);
        assert!((np_quantile_linear(&s, 0.9) - 4.6).abs() < 1e-12);
        assert!((np_quantile_linear(&s, 0.1) - 1.4).abs() < 1e-12);
    }

    #[test]
    fn ln1p_over_xi_branches_agree_near_cutoff() {
        // The exact form is numerically healthy at any nonzero xi, so the
        // expansion branch can be checked against it *inside* its region.
        for &x in &[0.03, 0.7, 5.0, 30.0] {
            for &xi in &[1e-9, -1e-9, 3e-12, -3e-12] {
                let expansion = ln1p_over_xi(xi, x);
                let exact = (xi * x).ln_1p() / xi;
                assert!(
                    (expansion - exact).abs() <= 1e-12 * x.max(1.0),
                    "xi={xi}, x={x}: {expansion} vs {exact}"
                );
            }
        }
    }
}
