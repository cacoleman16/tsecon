//! # tsecon-termstructure — yield-curve / term-structure econometrics
//!
//! Extension E2 of the `tsecon` library (ROADMAP `docs/roadmap/12-extensions.md`
//! E2, a named central-bank requirement): the Nelson-Siegel family of
//! parametric yield-curve models and its Diebold-Li dynamic extension. Every
//! cross-sectional fit is a linear regression routed through the library's
//! single OLS owner ([`tsecon_hac::ols`]); the one nonlinear piece — estimating
//! the decay parameter `lambda` — is profiled onto that linear fit and handed
//! to [`tsecon_optim`].
//!
//! ## The Nelson-Siegel curve
//!
//! Nelson & Siegel (1987) write the zero-coupon yield at maturity `t` as three
//! factors on maturity-dependent loadings governed by one decay rate `lambda`:
//!
//! ```text
//! y(t) = beta0
//!      + beta1 * (1 - e^{-lambda t}) / (lambda t)
//!      + beta2 * [ (1 - e^{-lambda t}) / (lambda t) - e^{-lambda t} ].
//! ```
//!
//! The three factors have a standing interpretation (Diebold & Li 2006):
//! `beta0` is the **level** (the long rate — its loading is a flat `1`),
//! `beta1` is the **slope** (its loading starts at `1` and decays to `0`, so
//! `-beta1` is the long-minus-short spread), and `beta2` is the **curvature**
//! (its loading is a hump, `0` at both ends). Holding `lambda` fixed makes the
//! curve linear in the factors, so a cross-sectional OLS of an observed curve
//! on the loadings recovers `[level, slope, curvature]`
//! ([`nelson_siegel_loadings`], [`fit_nelson_siegel`]).
//!
//! ## What this crate provides
//!
//! - [`nelson_siegel_loadings`] / [`svensson_loadings`] /
//!   [`nelson_siegel_forward_loadings`] — the loading columns, with the
//!   `maturity -> 0` limits handled analytically.
//! - [`fit_nelson_siegel`] — cross-sectional OLS fit at a **fixed** `lambda`
//!   (the Diebold-Li convention), returning [`NsFit`] with the factors, the
//!   centered R^2, and [`NsFit::yield_at`] / [`NsFit::forward_at`] curve
//!   utilities.
//! - [`fit_nelson_siegel_optimal_lambda`] — the same fit with `lambda`
//!   **estimated** by nonlinear least squares (profiling out the linear
//!   factors), the curve-fitting convention.
//! - [`fit_svensson`] — the Svensson (1994) four-factor extension (a second
//!   curvature with its own decay).
//! - [`fit_dynamic_ns`] — the dynamic Nelson-Siegel (Diebold-Li 2006): the
//!   three factors for every date in a panel, with [`DynamicNsFit::forecast`]
//!   giving an AR(1)-based one-step factor-and-curve forecast.
//!
//! ```
//! use tsecon_termstructure::{fit_nelson_siegel, nelson_siegel_loadings};
//!
//! let maturities = [3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0];
//! let yields = [4.10, 3.99, 3.98, 4.09, 4.10, 4.25, 4.31, 4.43];
//!
//! // Diebold-Li fix lambda = 0.0609 for monthly maturities.
//! let fit = fit_nelson_siegel(&maturities, &yields, 0.0609).unwrap();
//! let [level, slope, curvature] = fit.factors;
//! assert!(level > 4.0 && level < 5.0); // the long-rate level
//! assert!(fit.rsquared > 0.9);
//! let _ = (slope, curvature);
//!
//! // The fitted curve interpolates/extrapolates to any maturity.
//! let y_10y = fit.yield_at(120.0).unwrap();
//! assert!((y_10y - yields[7]).abs() < 0.2);
//!
//! // Loadings on their own:
//! let [l0, l1, l2] = nelson_siegel_loadings(&maturities, 0.0609).unwrap();
//! assert_eq!(l0, vec![1.0; 8]);
//! assert!(l1[0] > l1[7]); // slope decays with maturity
//! let _ = l2;
//! ```
//!
//! ## References
//!
//! - Nelson, C. R., & Siegel, A. F. (1987). "Parsimonious Modeling of Yield
//!   Curves." *Journal of Business*, 60(4), 473-489.
//! - Svensson, L. E. O. (1994). "Estimating and Interpreting Forward Interest
//!   Rates: Sweden 1992-1994." NBER Working Paper 4871.
//! - Diebold, F. X., & Li, C. (2006). "Forecasting the Term Structure of
//!   Government Bond Yields." *Journal of Econometrics*, 130(2), 337-364.
//! - Diebold, F. X., Rudebusch, G. D., & Aruoba, S. B. (2006). "The
//!   Macroeconomy and the Yield Curve: A Dynamic Latent Factor Approach."
//!   *Journal of Econometrics*, 131(1-2), 309-338.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod dynamic;
mod error;
mod fit;
mod loadings;
mod optlambda;
mod svensson;

pub use dynamic::{ar1_fit, fit_dynamic_ns, Ar1, DynamicNsFit, DynamicNsForecast};
pub use error::TermStructureError;
pub use fit::{fit_nelson_siegel, NsFit};
pub use loadings::{nelson_siegel_forward_loadings, nelson_siegel_loadings, svensson_loadings};
pub use optlambda::fit_nelson_siegel_optimal_lambda;
pub use svensson::{fit_svensson, SvenssonFit};
