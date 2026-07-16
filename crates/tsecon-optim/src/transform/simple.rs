//! Elementwise and ordering transforms: [`Positive`], [`Bounded`],
//! [`UnitInterval`], [`Ordered`].

use crate::error::OptimError;
use crate::transform::{check_finite, check_lengths, Transform};

/// Numerically stable logistic function `1 / (1 + e^{-z})`.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable `log(sigmoid(z)) = -log(1 + e^{-z})`.
fn log_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        -(-z).exp().ln_1p()
    } else {
        z - z.exp().ln_1p()
    }
}

/// Elementwise positivity: `theta_i = exp(z_i)` — the standard
/// reparameterization for variances and GARCH intercepts.
///
/// * forward: `theta = exp(z)`;
/// * inverse: `z = log(theta)`, requiring `theta > 0`;
/// * `log |det J| = sum_i z_i` (the Jacobian is diagonal with entries
///   `exp(z_i)`).
///
/// For `z` beyond ~709 the forward map overflows to `+infinity`;
/// [`TransformedObjective`](crate::TransformedObjective) treats such points
/// as infeasible.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Positive;

impl Transform for Positive {
    fn forward(&self, z: &[f64], theta: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("theta", z.len(), theta.len())?;
        check_finite("z", z)?;
        for (t, &zi) in theta.iter_mut().zip(z) {
            *t = zi.exp();
        }
        Ok(())
    }

    fn inverse(&self, theta: &[f64], z: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("z", theta.len(), z.len())?;
        check_finite("theta", theta)?;
        for (zi, &t) in z.iter_mut().zip(theta) {
            if t <= 0.0 {
                return Err(OptimError::Domain {
                    name: "theta",
                    value: t,
                    requirement: "theta > 0 for Positive::inverse",
                });
            }
            *zi = t.ln();
        }
        Ok(())
    }

    fn log_jacobian(&self, z: &[f64]) -> Result<f64, OptimError> {
        check_finite("z", z)?;
        Ok(z.iter().sum())
    }
}

/// Elementwise box constraint `lo < theta_i < hi` via the scaled logistic:
/// `theta_i = lo + (hi - lo) * sigmoid(z_i)` — for GARCH persistences,
/// damped-trend parameters, smoothing constants.
///
/// * inverse: `z = log(theta - lo) - log(hi - theta)` (the logit of the
///   rescaled coordinate);
/// * `log |det J| = sum_i [log(hi - lo) + log sigmoid(z_i)
///   + log sigmoid(-z_i)]`, computed stably for large `|z|`.
///
/// For `|z|` beyond ~745 the forward map saturates at the bounds in f64;
/// such points are on the boundary and `inverse` rejects them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounded {
    lo: f64,
    hi: f64,
}

impl Bounded {
    /// A box `(lo, hi)` applied to every element.
    ///
    /// # Errors
    ///
    /// [`OptimError::InvalidOption`] unless `lo < hi`, both finite.
    pub fn new(lo: f64, hi: f64) -> Result<Self, OptimError> {
        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
            return Err(OptimError::InvalidOption {
                name: "(lo, hi)",
                value: hi - lo,
                requirement: "finite bounds with lo < hi",
            });
        }
        Ok(Self { lo, hi })
    }

    /// Lower bound.
    pub fn lo(&self) -> f64 {
        self.lo
    }

    /// Upper bound.
    pub fn hi(&self) -> f64 {
        self.hi
    }
}

impl Transform for Bounded {
    fn forward(&self, z: &[f64], theta: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("theta", z.len(), theta.len())?;
        check_finite("z", z)?;
        let width = self.hi - self.lo;
        for (t, &zi) in theta.iter_mut().zip(z) {
            *t = self.lo + width * sigmoid(zi);
        }
        Ok(())
    }

    fn inverse(&self, theta: &[f64], z: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("z", theta.len(), z.len())?;
        check_finite("theta", theta)?;
        for (zi, &t) in z.iter_mut().zip(theta) {
            if !(t > self.lo && t < self.hi) {
                return Err(OptimError::Domain {
                    name: "theta",
                    value: t,
                    requirement: "lo < theta < hi for Bounded::inverse",
                });
            }
            *zi = (t - self.lo).ln() - (self.hi - t).ln();
        }
        Ok(())
    }

    fn log_jacobian(&self, z: &[f64]) -> Result<f64, OptimError> {
        check_finite("z", z)?;
        let lw = (self.hi - self.lo).ln();
        Ok(z.iter()
            .map(|&zi| lw + log_sigmoid(zi) + log_sigmoid(-zi))
            .sum())
    }
}

/// Elementwise `0 < theta_i < 1` via the logistic — [`Bounded`] with the
/// unit box, without the redundant `log(hi - lo)` term: probabilities,
/// mixing weights, discount factors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitInterval;

impl Transform for UnitInterval {
    fn forward(&self, z: &[f64], theta: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("theta", z.len(), theta.len())?;
        check_finite("z", z)?;
        for (t, &zi) in theta.iter_mut().zip(z) {
            *t = sigmoid(zi);
        }
        Ok(())
    }

    fn inverse(&self, theta: &[f64], z: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("z", theta.len(), z.len())?;
        check_finite("theta", theta)?;
        for (zi, &t) in z.iter_mut().zip(theta) {
            if !(t > 0.0 && t < 1.0) {
                return Err(OptimError::Domain {
                    name: "theta",
                    value: t,
                    requirement: "0 < theta < 1 for UnitInterval::inverse",
                });
            }
            *zi = t.ln() - (1.0 - t).ln();
        }
        Ok(())
    }

    fn log_jacobian(&self, z: &[f64]) -> Result<f64, OptimError> {
        check_finite("z", z)?;
        Ok(z.iter().map(|&zi| log_sigmoid(zi) + log_sigmoid(-zi)).sum())
    }
}

/// Strictly increasing vectors — the constraint for regime thresholds
/// (TAR/SETAR models) and ordered cut points:
///
/// ```text
/// theta_1 = z_1,    theta_k = theta_{k-1} + exp(z_k),  k = 2..n
/// ```
///
/// * inverse: `z_1 = theta_1`, `z_k = log(theta_k - theta_{k-1})`,
///   requiring `theta` strictly increasing;
/// * `log |det J| = sum_{k=2}^n z_k` (the Jacobian is lower triangular
///   with diagonal `(1, exp(z_2), ..., exp(z_n))`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ordered;

impl Transform for Ordered {
    fn forward(&self, z: &[f64], theta: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("theta", z.len(), theta.len())?;
        check_finite("z", z)?;
        let mut prev = 0.0;
        for (k, (t, &zi)) in theta.iter_mut().zip(z).enumerate() {
            *t = if k == 0 { zi } else { prev + zi.exp() };
            prev = *t;
        }
        Ok(())
    }

    fn inverse(&self, theta: &[f64], z: &mut [f64]) -> Result<(), OptimError> {
        check_lengths("z", theta.len(), z.len())?;
        check_finite("theta", theta)?;
        let mut prev = 0.0;
        for (k, (zi, &t)) in z.iter_mut().zip(theta).enumerate() {
            if k == 0 {
                *zi = t;
            } else {
                let gap = t - prev;
                if gap <= 0.0 {
                    return Err(OptimError::Domain {
                        name: "theta",
                        value: t,
                        requirement: "strictly increasing theta for Ordered::inverse",
                    });
                }
                *zi = gap.ln();
            }
            prev = t;
        }
        Ok(())
    }

    fn log_jacobian(&self, z: &[f64]) -> Result<f64, OptimError> {
        check_finite("z", z)?;
        Ok(z.iter().skip(1).sum())
    }
}
