//! Student's t distribution.

use super::ContinuousDist;
use crate::error::StatsError;
use crate::special::{beta_inc, inv_beta_inc, ln_gamma};
use core::f64::consts::PI;

/// Student's t distribution with `df` degrees of freedom (non-integer `df`
/// supported).
///
/// * pdf: `Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(-(ν+1)/2)`
/// * cdf via the regularized incomplete beta function: for `t <= 0`,
///   `F(t) = I_x(ν/2, 1/2) / 2` with `x = ν/(ν + t²)`; symmetry otherwise
///   (Abramowitz & Stegun 26.7.1).
/// * ppf by inverting the incomplete beta ([`inv_beta_inc`]).
///
/// Note the *unstandardized* t has variance `ν/(ν-2)` for `ν > 2`; use
/// [`super::Standardized::student_t`] for the unit-variance version used as
/// a GARCH innovation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentT {
    df: f64,
}

impl StudentT {
    /// Create a Student t distribution with `df > 0` degrees of freedom
    /// (need not be an integer).
    pub fn new(df: f64) -> Result<Self, StatsError> {
        if !(df > 0.0 && df.is_finite()) {
            return Err(StatsError::InvalidParameter {
                name: "df",
                value: df,
                requirement: "0 < df < inf",
            });
        }
        Ok(Self { df })
    }

    /// The degrees-of-freedom parameter.
    pub fn df(&self) -> f64 {
        self.df
    }
}

impl ContinuousDist for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        let v = self.df;
        ln_gamma(0.5 * (v + 1.0))
            - ln_gamma(0.5 * v)
            - 0.5 * (v * PI).ln()
            - 0.5 * (v + 1.0) * (x * x / v).ln_1p()
    }

    fn cdf(&self, x: f64) -> f64 {
        if x == 0.0 {
            return 0.5;
        }
        if x.is_infinite() {
            return if x > 0.0 { 1.0 } else { 0.0 };
        }
        let v = self.df;
        let z = v / (v + x * x);
        // Half of the two-sided tail probability P(|T| > |x|).
        let half_tail = match beta_inc(0.5 * v, 0.5, z) {
            Ok(i) => 0.5 * i,
            Err(_) => return f64::NAN,
        };
        if x < 0.0 {
            half_tail
        } else {
            1.0 - half_tail
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
        let v = self.df;
        let tail = 2.0 * u.min(1.0 - u); // two-sided tail probability
        let z = inv_beta_inc(0.5 * v, 0.5, tail)?; // z = ν/(ν + t²)
        let t = (v * (1.0 - z) / z).sqrt();
        Ok(if u < 0.5 { -t } else { t })
    }
}
