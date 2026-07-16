//! The Monahan (1984) PACF-based stationarity transform for AR
//! coefficients.

use crate::error::OptimError;
use crate::transform::{check_finite, check_lengths, Transform};

/// Bijection between `R^p` and the stationarity region of AR(p)
/// coefficients, via partial autocorrelations (Monahan 1984;
/// Barndorff-Nielsen-Schou 1973).
///
/// Coefficients follow the convention
/// `x_t = phi_1 x_{t-1} + ... + phi_p x_{t-p} + e_t`; the process is
/// stationary iff all roots of `1 - phi_1 L - ... - phi_p L^p` lie outside
/// the unit circle, which holds iff all partial autocorrelations `r_k` lie
/// in `(-1, 1)` (Barndorff-Nielsen-Schou 1973).
///
/// **Forward** (`z -> phi`, exact): squash `r_k = tanh(z_k)` into
/// `(-1, 1)`, then run the Levinson-Durbin recursion
///
/// ```text
/// phi^(k)_k = r_k,
/// phi^(k)_j = phi^(k-1)_j - r_k phi^(k-1)_{k-j},   j = 1..k-1,
/// ```
///
/// so the output is stationary *by construction* for every `z` in `R^p`.
/// For `p = 1` this is exactly `phi_1 = tanh(z_1)`.
///
/// **Inverse** (`phi -> z`, exact): the reverse recursion
///
/// ```text
/// r_k = phi^(k)_k,
/// phi^(k-1)_j = (phi^(k)_j + r_k phi^(k)_{k-j}) / (1 - r_k^2),
/// ```
///
/// then `z_k = atanh(r_k)`; it fails with [`OptimError::NotStationary`]
/// (naming the offending lag) iff `phi` is not stationary.
///
/// **Log-Jacobian** of the forward map (derived from the recursion's
/// triangular composition; each stage multiplies the determinant by
/// `det(I - r_k P_{k-1})` with `P` the index-reversal permutation, and the
/// `tanh` layer adds `1 - r_k^2` per coordinate):
///
/// ```text
/// log |det J| = sum_k [ (1 + floor((k-1)/2)) log(1 - r_k^2)
///                       + ((k-1) mod 2) log(1 - r_k) ]
/// ```
///
/// computed stably in `z` for large `|z_k|`.
///
/// **MA invertibility by duality**: `1 + theta_1 L + ... + theta_q L^q` is
/// invertible iff the AR polynomial with coefficients `-theta_j` is
/// stationary, so model crates obtain invertible MA coefficients as
/// `theta = -forward(z)` and map back with `inverse(-theta)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StationaryAr;

/// Stable `log(1 - tanh(z)^2)`.
fn log_one_minus_tanh_sq(z: f64) -> f64 {
    let a = z.abs();
    2.0 * (core::f64::consts::LN_2 - a - (-2.0 * a).exp().ln_1p())
}

/// Stable `log(1 - tanh(z)) = log 2 - softplus(2 z)`.
fn log_one_minus_tanh(z: f64) -> f64 {
    let t = 2.0 * z;
    let softplus = if t >= 0.0 {
        t + (-t).exp().ln_1p()
    } else {
        t.exp().ln_1p()
    };
    core::f64::consts::LN_2 - softplus
}

impl Transform for StationaryAr {
    fn forward(&self, z: &[f64], theta: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("theta", z.len(), theta.len())?;
        check_finite("z", z)?;
        let p = z.len();
        for s in 1..=p {
            let r = z[s - 1].tanh();
            // Update the order-(s-1) coefficients in theta[..s-1] pairwise:
            // new_j = old_j - r * old_{m-1-j} over m = s-1 old entries.
            let m = s - 1;
            for j in 0..m / 2 {
                let a = theta[j];
                let b = theta[m - 1 - j];
                theta[j] = a - r * b;
                theta[m - 1 - j] = b - r * a;
            }
            if m % 2 == 1 {
                theta[(m - 1) / 2] *= 1.0 - r;
            }
            theta[s - 1] = r;
        }
        Ok(())
    }

    fn inverse(&self, theta: &[f64], z: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("z", theta.len(), z.len())?;
        check_finite("phi", theta)?;
        let p = theta.len();
        let mut w = theta.to_vec();
        for s in (1..=p).rev() {
            let r = w[s - 1];
            if r.abs() >= 1.0 || r.is_nan() {
                return Err(OptimError::NotStationary { order: s, pacf: r });
            }
            z[s - 1] = r.atanh();
            let m = s - 1;
            let d = 1.0 - r * r;
            for j in 0..m / 2 {
                let a = w[j];
                let b = w[m - 1 - j];
                w[j] = (a + r * b) / d;
                w[m - 1 - j] = (b + r * a) / d;
            }
            if m % 2 == 1 {
                w[(m - 1) / 2] /= 1.0 - r;
            }
        }
        Ok(())
    }

    fn log_jacobian(&self, z: &[f64]) -> Result<f64, OptimError> {
        check_finite("z", z)?;
        let mut lj = 0.0;
        for (k1, &zk) in z.iter().enumerate() {
            // k1 = k - 1 for lag k = 1..p.
            let pow_sq = 1.0 + (k1 / 2) as f64;
            lj += pow_sq * log_one_minus_tanh_sq(zk);
            if k1 % 2 == 1 {
                lj += log_one_minus_tanh(zk);
            }
        }
        Ok(lj)
    }
}
