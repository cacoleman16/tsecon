//! Hansen (1994) skewed Student t distribution.

use super::{ContinuousDist, StudentT};
use crate::error::StatsError;
use crate::special::ln_gamma;
use core::f64::consts::PI;

/// Hansen's skewed Student t distribution (Hansen 1994, "Autoregressive
/// conditional density estimation", *International Economic Review* 35),
/// **standardized to mean 0 and variance 1 by construction** — the standard
/// skewed innovation density for GARCH-type models.
///
/// Parameters: tail thickness `eta` (`2 < eta < inf`) and skewness `lambda`
/// (`-1 < lambda < 1`); `lambda = 0` recovers the unit-variance
/// (standardized) Student t with `eta` degrees of freedom.
///
/// Density (Hansen 1994, eq. 10):
///
/// ```text
/// g(z) = b c (1 + (1/(η-2)) ((b z + a)/(1 - λ))²)^(-(η+1)/2),  z < -a/b
/// g(z) = b c (1 + (1/(η-2)) ((b z + a)/(1 + λ))²)^(-(η+1)/2),  z ≥ -a/b
///
/// c = Γ((η+1)/2) / (√(π(η-2)) Γ(η/2)),
/// a = 4 λ c (η-2)/(η-1),   b² = 1 + 3λ² - a²
/// ```
///
/// The constants `a` and `b` are exactly what enforce `E[z] = 0` and
/// `Var[z] = 1`. CDF and quantiles are expressed through the CDF/quantile of
/// an ordinary Student t with `eta` degrees of freedom: with
/// `k = sqrt(η/(η-2))` and `w = (b z + a)/(1 ∓ λ)`,
///
/// ```text
/// F(z) = (1-λ) T_η(k w),         z < -a/b
/// F(z) = (1+λ) T_η(k w) - λ,     z ≥ -a/b
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HansenSkewT {
    eta: f64,
    lambda: f64,
    /// Location constant `a` above.
    a: f64,
    /// Scale constant `b` above.
    b: f64,
    /// `ln c`.
    ln_c: f64,
    /// Student t with `eta` degrees of freedom, used for CDF/quantiles.
    t: StudentT,
}

impl HansenSkewT {
    /// Create a Hansen skew-t with tail parameter `eta > 2` and skewness
    /// `-1 < lambda < 1`.
    pub fn new(eta: f64, lambda: f64) -> Result<Self, StatsError> {
        if !(eta > 2.0 && eta.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "eta",
                value: eta,
                requirement: "2 < eta < inf",
            });
        }
        if !(lambda > -1.0 && lambda < 1.0) {
            return Err(StatsError::InvalidParameter {
                name: "lambda",
                value: lambda,
                requirement: "-1 < lambda < 1",
            });
        }
        let ln_c =
            ln_gamma(0.5 * (eta + 1.0)) - ln_gamma(0.5 * eta) - 0.5 * (PI * (eta - 2.0)).ln();
        let c = ln_c.exp();
        let a = 4.0 * lambda * c * (eta - 2.0) / (eta - 1.0);
        let b2 = 1.0 + 3.0 * lambda * lambda - a * a;
        if b2 <= 0.0 || b2.is_nan() {
            // Mathematically b² > 0 on the whole valid (eta, lambda) region;
            // defensive check against pathological rounding.
            return Err(StatsError::InvalidParameter {
                name: "eta,lambda",
                value: b2,
                requirement: "1 + 3λ² - a² > 0",
            });
        }
        Ok(Self {
            eta,
            lambda,
            a,
            b: b2.sqrt(),
            ln_c,
            t: StudentT::new(eta)?,
        })
    }

    /// The tail-thickness parameter `eta`.
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// The skewness parameter `lambda`.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The mode-side switch point `-a/b` (the density kink location).
    fn threshold(&self) -> f64 {
        -self.a / self.b
    }

    /// `sqrt(eta / (eta - 2))`: converts the skew-t "w" scale to the
    /// unstandardized Student t scale.
    fn k(&self) -> f64 {
        (self.eta / (self.eta - 2.0)).sqrt()
    }

    /// `(1 - λ)` on the left branch, `(1 + λ)` on the right.
    fn side(&self, z: f64) -> f64 {
        if z < self.threshold() {
            1.0 - self.lambda
        } else {
            1.0 + self.lambda
        }
    }
}

impl ContinuousDist for HansenSkewT {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        let w = (self.b * x + self.a) / self.side(x);
        self.b.ln() + self.ln_c - 0.5 * (self.eta + 1.0) * (w * w / (self.eta - 2.0)).ln_1p()
    }

    fn cdf(&self, x: f64) -> f64 {
        let s = self.side(x);
        let w = (self.b * x + self.a) / s;
        let tv = self.t.cdf(self.k() * w);
        if x < self.threshold() {
            s * tv
        } else {
            s * tv - self.lambda
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x < self.threshold() {
            1.0 - self.cdf(x)
        } else {
            // (1+λ)(1 - T(kw)) — keeps relative accuracy in the right tail.
            let s = 1.0 + self.lambda;
            let w = (self.b * x + self.a) / s;
            s * self.t.sf(self.k() * w)
        }
    }

    fn ppf(&self, u: f64) -> Result<f64, StatsError> {
        if !(u > 0.0 && u < 1.0) {
            return Err(StatsError::Domain {
                name: "u",
                value: u,
                requirement: "0 < u < 1",
            });
        }
        let half_left = 0.5 * (1.0 - self.lambda); // = F(-a/b)
        if u == half_left {
            return Ok(self.threshold());
        }
        let (s, v) = if u < half_left {
            let s = 1.0 - self.lambda;
            (s, u / s)
        } else {
            let s = 1.0 + self.lambda;
            (s, (u + self.lambda) / s)
        };
        let tq = self.t.ppf(v)?;
        let w = tq / self.k();
        Ok((s * w - self.a) / self.b)
    }
}
