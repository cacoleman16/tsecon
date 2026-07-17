//! Generalized error distribution (GED).

use super::ContinuousDist;
use crate::error::StatsError;
use crate::special::{gamma_p, gamma_q, inv_gamma_p, ln_gamma};

/// The generalized error distribution (GED, also "generalized normal" or
/// "exponential power"), in the **scipy `gennorm` parameterization** with
/// shape `nu > 0`:
///
/// ```text
/// f(x) = nu / (2 Γ(1/nu)) · exp(-|x|^nu)
/// ```
///
/// `nu = 2` gives a normal with variance 1/2, `nu = 1` the unit-scale
/// Laplace. The variance is `Γ(3/nu) / Γ(1/nu)`; use
/// [`super::Standardized::ged`] for the unit-variance version used as a
/// GARCH innovation.
///
/// CDF and quantiles go through the regularized incomplete gamma function:
/// for `x > 0`, `F(x) = 1/2 + P(1/nu, x^nu)/2`, and by symmetry
/// `F(-x) = Q(1/nu, x^nu)/2` (computed from `Q` directly for tail accuracy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ged {
    nu: f64,
}

impl Ged {
    /// Create a GED with shape parameter `nu > 0`.
    pub fn new(nu: f64) -> Result<Self, StatsError> {
        if !(nu > 0.0 && nu.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "nu",
                value: nu,
                requirement: "0 < nu < inf",
            });
        }
        Ok(Self { nu })
    }

    /// The shape parameter.
    pub fn nu(&self) -> f64 {
        self.nu
    }
}

impl ContinuousDist for Ged {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        (0.5 * self.nu).ln() - ln_gamma(1.0 / self.nu) - x.abs().powf(self.nu)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x == 0.0 {
            return 0.5;
        }
        let a = 1.0 / self.nu;
        if x > 0.0 {
            match gamma_p(a, x.powf(self.nu)) {
                Ok(p) => 0.5 + 0.5 * p,
                Err(_) => f64::NAN,
            }
        } else {
            match gamma_q(a, (-x).powf(self.nu)) {
                Ok(q) => 0.5 * q,
                Err(_) => f64::NAN,
            }
        }
    }

    fn sf(&self, x: f64) -> f64 {
        // Symmetry keeps relative accuracy in the right tail.
        self.cdf(-x)
    }

    fn ppf(&self, u: f64) -> Result<f64, StatsError> {
        if !(u > 0.0 && u < 1.0) {
            return Err(StatsError::Domain {
                name: "u",
                value: u,
                requirement: "0 < u < 1",
            });
        }
        if u == 0.5 {
            return Ok(0.0);
        }
        let a = 1.0 / self.nu;
        if u > 0.5 {
            Ok(inv_gamma_p(a, 2.0 * u - 1.0)?.powf(a))
        } else {
            Ok(-inv_gamma_p(a, 1.0 - 2.0 * u)?.powf(a))
        }
    }
}
