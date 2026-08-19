//! Shared pieces: input validation, the average-rank pseudo-observation
//! transform, Kendall's tau-b, and the observed-information standard
//! errors (the same central-4 stencil `tsecon-evt`/`tsecon-garch` use and
//! the fixture generator mirrors).

use crate::error::CopulaError;

/// Minimum number of paired observations required by
/// [`crate::copula_fit`].
///
/// Below this a one/two-parameter dependence MLE and its
/// observed-information standard errors are noise; the value is a
/// documented floor, not a statistical guarantee — serious dependence
/// work wants far more.
pub const MIN_OBS: usize = 20;

/// Validates a raw column: non-empty and all-finite.
pub(crate) fn check_series(y: &[f64], what: &'static str) -> Result<(), CopulaError> {
    if y.is_empty() {
        return Err(CopulaError::EmptyInput { what });
    }
    if let Some(index) = y.iter().position(|v| !v.is_finite()) {
        return Err(CopulaError::NonFinite { what, index });
    }
    Ok(())
}

/// Validates a pseudo-observation pair of columns for fitting: equal
/// lengths, at least [`MIN_OBS`] pairs, all finite, all strictly inside
/// `(0, 1)`.
pub(crate) fn check_u(u1: &[f64], u2: &[f64]) -> Result<(), CopulaError> {
    check_series(u1, "u[:, 0]")?;
    check_series(u2, "u[:, 1]")?;
    if u1.len() != u2.len() {
        return Err(CopulaError::LengthMismatch {
            n1: u1.len(),
            n2: u2.len(),
        });
    }
    if u1.len() < MIN_OBS {
        return Err(CopulaError::TooFewObservations {
            n: u1.len(),
            min: MIN_OBS,
        });
    }
    for (what, col) in [("u[:, 0]", u1), ("u[:, 1]", u2)] {
        if let Some(index) = col.iter().position(|&v| !(v > 0.0 && v < 1.0)) {
            return Err(CopulaError::OutOfUnitInterval {
                what,
                index,
                value: col[index],
            });
        }
    }
    Ok(())
}

/// The average-rank pseudo-observation transform of one column:
/// `u_i = rank_i / (n + 1)` with ties assigned their average rank —
/// exactly `scipy.stats.rankdata(x, method="average") / (n + 1)`
/// (golden-pinned, ties included). The `n + 1` denominator keeps every
/// value strictly inside `(0, 1)`, which the quantile transforms of the
/// elliptical copulas require.
pub(crate) fn pseudo_obs_col(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut u = vec![0.0; n];
    let denom = (n + 1) as f64;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        // 1-based ranks i+1 ..= j+1 share the average (i + j + 2) / 2.
        let avg_rank = (i + j + 2) as f64 / 2.0;
        for &k in &idx[i..=j] {
            u[k] = avg_rank / denom;
        }
        i = j + 1;
    }
    u
}

/// Kendall's tau-b of two equal-length columns, with tie corrections —
/// the same estimand and the same floating-point evaluation order as
/// `scipy.stats.kendalltau` (`(P - Q) / sqrt(n0 - n1) / sqrt(n0 - n2)`),
/// so a continuous sample matches scipy to the last bit and tied data
/// matches its tau-b convention. O(n^2) pair enumeration — exact and
/// fast enough for the sample sizes this crate is for.
///
/// Errors with [`CopulaError::Degenerate`] when a column is constant
/// (tau's denominator is zero there).
pub(crate) fn kendall_tau(x: &[f64], y: &[f64]) -> Result<f64, CopulaError> {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    let mut con_minus_dis: i64 = 0;
    let mut tie_x: u64 = 0;
    let mut tie_y: u64 = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            if dx == 0.0 {
                tie_x += 1;
                if dy == 0.0 {
                    tie_y += 1;
                }
            } else if dy == 0.0 {
                tie_y += 1;
            } else if (dx > 0.0) == (dy > 0.0) {
                con_minus_dis += 1;
            } else {
                con_minus_dis -= 1;
            }
        }
    }
    let n0 = (n as u64) * (n as u64 - 1) / 2;
    if tie_x == n0 {
        return Err(CopulaError::Degenerate { what: "u[:, 0]" });
    }
    if tie_y == n0 {
        return Err(CopulaError::Degenerate { what: "u[:, 1]" });
    }
    Ok(con_minus_dis as f64 / ((n0 - tie_x) as f64).sqrt() / ((n0 - tie_y) as f64).sqrt())
}

/// Observed-information standard errors: the numerical Hessian of the
/// negative log-likelihood at the MLE, inverted, square-rooted on the
/// diagonal.
///
/// The Hessian uses the four-point central cross-difference formula
/// (statsmodels `approx_hess3`, the same scheme `tsecon-evt` uses and the
/// fixture generator mirrors) with steps `h_i = eps^(1/4) *
/// step_scale[i]`, where `step_scale` carries each parameter's own units
/// (rho: `max(1 - rho^2, 0.01)`; nu: `nu`; theta: `max(|theta|, 0.1)`).
///
/// Returns `None` when the Hessian cannot be formed (a probe left the
/// parameter domain, so `nll` returned non-finite) or is not positive
/// definite (flat or boundary optimum) — the caller reports NaN standard
/// errors and `se_valid = false` rather than fabricating numbers.
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

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_obs_ranks_with_ties() {
        // x = [3, 1, 3, 2]: ranks (average) = [3.5, 1, 3.5, 2], n+1 = 5.
        let u = pseudo_obs_col(&[3.0, 1.0, 3.0, 2.0]);
        assert_eq!(u, vec![0.7, 0.2, 0.7, 0.4]);
    }

    #[test]
    fn kendall_tau_hand_case() {
        // Perfectly concordant and perfectly discordant.
        // (The sqrt-division evaluation order mirrors scipy, which rounds
        // the perfectly concordant case to 1 + 2e-16 identically.)
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.1, 0.2, 0.3, 0.4];
        assert!((kendall_tau(&x, &y).expect("tau") - 1.0).abs() < 1e-12);
        let yr = [0.4, 0.3, 0.2, 0.1];
        assert!((kendall_tau(&x, &yr).expect("tau") + 1.0).abs() < 1e-12);
        // Constant column degenerates.
        assert!(matches!(
            kendall_tau(&[1.0, 1.0, 1.0], &[0.1, 0.2, 0.3]),
            Err(CopulaError::Degenerate { .. })
        ));
    }
}
