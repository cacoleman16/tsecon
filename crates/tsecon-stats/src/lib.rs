//! # tsecon-stats
//!
//! Special functions and the innovation-distribution zoo for the `tsecon`
//! time-series econometrics library (foundations layer; see ROADMAP §5).
//!
//! Two public surfaces:
//!
//! * [`special`] — scalar special functions: log-gamma (Lanczos), the error
//!   function pair (Cody rational approximations), regularized incomplete
//!   gamma and beta functions (series + modified-Lentz continued fractions),
//!   the inverse standard normal CDF (Wichura AS241), and safeguarded-Newton
//!   inverses of the incomplete gamma/beta functions.
//! * [`dist`] — the [`ContinuousDist`] trait and the distributions consumed
//!   by the volatility, Bayesian, and forecasting modules: [`StdNormal`],
//!   [`Normal`], [`StudentT`], [`Ged`] (scipy `gennorm` parameterization),
//!   [`HansenSkewT`] (Hansen 1994, unit variance by construction),
//!   unit-variance [`Standardized`] wrappers for GARCH innovations, and
//!   [`ChiSquared`] plus the [`chi2_cdf`]/[`chi2_sf`] p-value helpers for
//!   the diagnostics crate.
//!
//! Sampling composes through
//! [`ContinuousDist::sample_from_uniform`] (inverse transform), so the crate
//! has no RNG dependency: feed it uniforms from `tsecon-rng` substreams.
//!
//! Accuracy: golden-value tests pin densities/CDFs at `1e-12` relative and
//! quantiles at `1e-9` against SciPy fixtures; the special functions are
//! individually accurate to near machine precision (documented per
//! function).

#![warn(missing_docs)]

pub mod dist;
pub mod error;
pub mod special;

pub use dist::{
    chi2_cdf, chi2_sf, ChiSquared, ContinuousDist, Ged, HansenSkewT, Normal, Standardized,
    StdNormal, StudentT,
};
pub use error::StatsError;
