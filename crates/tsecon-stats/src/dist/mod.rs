//! The innovation-distribution zoo.
//!
//! One trait, [`ContinuousDist`], implemented by every continuous univariate
//! distribution used as a model innovation or for diagnostics p-values:
//! [`StdNormal`], [`Normal`], [`StudentT`], [`Ged`], [`HansenSkewT`],
//! [`ChiSquared`], and the unit-variance [`Standardized`] wrappers used for
//! GARCH innovations.
//!
//! # NaN policy
//!
//! Parameters are validated at construction (`new` returns
//! `Result<_, StatsError>`), so `pdf`/`ln_pdf`/`cdf`/`sf` are total functions
//! of `x` and return `f64` directly: `NaN` in yields `NaN` out, and an
//! internal special-function convergence failure (not observed for valid
//! parameters) also surfaces as `NaN`. Quantile functions (`ppf`) return
//! `Result` because `u` outside `(0, 1)` is a domain error.

mod chi_squared;
mod ged;
mod hansen;
mod normal;
mod standardized;
mod student_t;

pub use chi_squared::{chi2_cdf, chi2_sf, ChiSquared};
pub use ged::Ged;
pub use hansen::HansenSkewT;
pub use normal::{Normal, StdNormal};
pub use standardized::Standardized;
pub use student_t::StudentT;

use crate::error::StatsError;

/// A continuous univariate distribution.
///
/// This is the contract every innovation distribution satisfies. Sampling is
/// deliberately expressed through [`ContinuousDist::sample_from_uniform`]
/// (inverse-transform from a `U(0,1)` variate) so the zoo composes with the
/// `tsecon-rng` uniform streams later without a crate dependency now.
pub trait ContinuousDist {
    /// Probability density function `f(x)`.
    fn pdf(&self, x: f64) -> f64;

    /// Natural logarithm of the density, `ln f(x)`; `-inf` where the density
    /// is zero. Computed directly (not as `pdf(x).ln()`) so it stays finite
    /// and accurate far into the tails.
    fn ln_pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function `F(x) = P(X <= x)`.
    fn cdf(&self, x: f64) -> f64;

    /// Survival function `S(x) = P(X > x) = 1 - F(x)`, computed to preserve
    /// relative accuracy in the right tail where possible.
    fn sf(&self, x: f64) -> f64;

    /// Quantile function (inverse CDF): the `x` with `F(x) = u`.
    ///
    /// Errors with [`StatsError::Domain`] unless `0 < u < 1`, and with
    /// [`StatsError::NoConvergence`] if an internal root find fails (not
    /// observed for valid parameters).
    fn ppf(&self, u: f64) -> Result<f64, StatsError>;

    /// Map a uniform variate `u ~ U(0,1)` to a draw from this distribution.
    ///
    /// The default implementation is the inverse-transform `ppf(u)`;
    /// distributions may override with a cheaper exact transform. RNG
    /// integration composes on top of this: feed uniforms from any stream.
    fn sample_from_uniform(&self, u: f64) -> Result<f64, StatsError> {
        self.ppf(u)
    }
}
