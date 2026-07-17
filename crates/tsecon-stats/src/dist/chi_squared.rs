//! Chi-squared distribution (p-values for the diagnostics crate).

use super::ContinuousDist;
use crate::error::StatsError;
use crate::special::{gamma_p, gamma_q, inv_gamma_p, ln_gamma};
use core::f64::consts::LN_2;

fn check_df(df: f64) -> Result<(), StatsError> {
    if !(df > 0.0 && df.is_finite()) {
        return Err(StatsError::InvalidParameter {
            name: "df",
            value: df,
            requirement: "0 < df < inf",
        });
    }
    Ok(())
}

/// Chi-squared CDF `P(X <= x)` for `X ~ χ²(df)`, via the regularized lower
/// incomplete gamma function: `F(x) = P(df/2, x/2)`.
///
/// Free-function form for the diagnostics crate's test statistics;
/// `df` need not be an integer. `x <= 0` returns 0.
pub fn chi2_cdf(x: f64, df: f64) -> Result<f64, StatsError> {
    check_df(df)?;
    if x.is_nan() {
        return Err(StatsError::Domain {
            name: "x",
            value: x,
            requirement: "not NaN",
        });
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    gamma_p(0.5 * df, 0.5 * x)
}

/// Chi-squared survival function `P(X > x)` — the p-value of a χ² test
/// statistic — via the regularized upper incomplete gamma function:
/// `S(x) = Q(df/2, x/2)`, accurate in the right tail (relative accuracy is
/// preserved for tiny p-values).
///
/// `x <= 0` returns 1.
pub fn chi2_sf(x: f64, df: f64) -> Result<f64, StatsError> {
    check_df(df)?;
    if x.is_nan() {
        return Err(StatsError::Domain {
            name: "x",
            value: x,
            requirement: "not NaN",
        });
    }
    if x <= 0.0 {
        return Ok(1.0);
    }
    gamma_q(0.5 * df, 0.5 * x)
}

/// The chi-squared distribution with `df > 0` degrees of freedom (need not
/// be an integer).
///
/// * pdf: `x^(df/2 - 1) e^(-x/2) / (2^(df/2) Γ(df/2))` for `x > 0`
/// * cdf/sf via the regularized incomplete gamma function (see
///   [`chi2_cdf`] / [`chi2_sf`])
/// * ppf: `2 · P^{-1}(df/2, u)` via [`inv_gamma_p`]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    df: f64,
}

impl ChiSquared {
    /// Create a chi-squared distribution with `df > 0` degrees of freedom.
    pub fn new(df: f64) -> Result<Self, StatsError> {
        check_df(df)?;
        Ok(Self { df })
    }

    /// The degrees-of-freedom parameter.
    pub fn df(&self) -> f64 {
        self.df
    }
}

impl ContinuousDist for ChiSquared {
    fn pdf(&self, x: f64) -> f64 {
        self.ln_pdf(x).exp()
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        let k2 = 0.5 * self.df;
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 {
            // Boundary limit of the density.
            return if self.df < 2.0 {
                f64::INFINITY
            } else if self.df == 2.0 {
                0.5_f64.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
        (k2 - 1.0) * x.ln() - 0.5 * x - ln_gamma(k2) - k2 * LN_2
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return if x.is_nan() { f64::NAN } else { 0.0 };
        }
        if x.is_infinite() {
            return 1.0;
        }
        gamma_p(0.5 * self.df, 0.5 * x).unwrap_or(f64::NAN)
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return if x.is_nan() { f64::NAN } else { 1.0 };
        }
        if x.is_infinite() {
            return 0.0;
        }
        gamma_q(0.5 * self.df, 0.5 * x).unwrap_or(f64::NAN)
    }

    fn ppf(&self, u: f64) -> Result<f64, StatsError> {
        if !(u > 0.0 && u < 1.0) {
            return Err(StatsError::Domain {
                name: "u",
                value: u,
                requirement: "0 < u < 1",
            });
        }
        Ok(2.0 * inv_gamma_p(0.5 * self.df, u)?)
    }
}
