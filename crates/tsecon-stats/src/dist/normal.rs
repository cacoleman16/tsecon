//! Standard and general normal distributions.

#![allow(clippy::excessive_precision)]

use super::ContinuousDist;
use crate::error::StatsError;
use crate::special::{erfc, inv_norm_cdf};

/// `1 / sqrt(2*pi)`.
const FRAC_1_SQRT_2PI: f64 = 0.39894228040143267793994605993438;
/// `ln(2*pi)`.
const LN_2PI: f64 = 1.8378770664093454835606594728112353;

/// The standard normal distribution `N(0, 1)`.
///
/// * pdf: `φ(x) = exp(-x²/2) / sqrt(2π)`
/// * cdf: `Φ(x) = erfc(-x/√2) / 2` (Cody-accurate in both tails)
/// * ppf: Wichura's AS241 (see [`inv_norm_cdf`])
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StdNormal;

impl ContinuousDist for StdNormal {
    fn pdf(&self, x: f64) -> f64 {
        FRAC_1_SQRT_2PI * (-0.5 * x * x).exp()
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        -0.5 * x * x - 0.5 * LN_2PI
    }

    fn cdf(&self, x: f64) -> f64 {
        0.5 * erfc(-x * core::f64::consts::FRAC_1_SQRT_2)
    }

    fn sf(&self, x: f64) -> f64 {
        // By symmetry; erfc keeps relative accuracy in the right tail.
        0.5 * erfc(x * core::f64::consts::FRAC_1_SQRT_2)
    }

    fn ppf(&self, u: f64) -> Result<f64, StatsError> {
        inv_norm_cdf(u)
    }
}

/// A normal distribution `N(mean, sd²)` parameterized by its mean and
/// standard deviation.
///
/// All methods delegate to [`StdNormal`] through the affine map
/// `z = (x - mean) / sd`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    mean: f64,
    sd: f64,
}

impl Normal {
    /// Create a normal distribution with the given mean and standard
    /// deviation.
    ///
    /// Errors if `mean` is not finite or `sd` is not strictly positive and
    /// finite.
    pub fn new(mean: f64, sd: f64) -> Result<Self, StatsError> {
        if !mean.is_finite() {
            return Err(StatsError::InvalidParameter {
                name: "mean",
                value: mean,
                requirement: "finite",
            });
        }
        if !(sd > 0.0 && sd.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "sd",
                value: sd,
                requirement: "0 < sd < inf",
            });
        }
        Ok(Self { mean, sd })
    }

    /// The mean parameter.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// The standard deviation parameter.
    pub fn sd(&self) -> f64 {
        self.sd
    }

    #[inline]
    fn z(&self, x: f64) -> f64 {
        (x - self.mean) / self.sd
    }
}

impl ContinuousDist for Normal {
    fn pdf(&self, x: f64) -> f64 {
        StdNormal.pdf(self.z(x)) / self.sd
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        StdNormal.ln_pdf(self.z(x)) - self.sd.ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        StdNormal.cdf(self.z(x))
    }

    fn sf(&self, x: f64) -> f64 {
        StdNormal.sf(self.z(x))
    }

    fn ppf(&self, u: f64) -> Result<f64, StatsError> {
        Ok(self.mean + self.sd * inv_norm_cdf(u)?)
    }
}
