//! # tsecon-gas
//!
//! Score-driven (Generalized Autoregressive Score / Dynamic Conditional
//! Score) models for the `tsecon` time-series econometrics library (Creal,
//! Koopman & Lucas 2013, *J. Appl. Econometrics* 28; Harvey 2013,
//! *Dynamic Models for Volatility and Heavy Tails*).
//!
//! This crate implements the **GAS(1,1) time-varying-variance** model. A
//! single latent variance `f_t` evolves by the score-driven recursion
//!
//! ```text
//! f_{t+1} = omega + a * s_t + b * f_t,
//! s_t     = S_t * nabla_t,
//! nabla_t = d/df_t log p(y_t | f_t),        (score of the density)
//! S_t     = I_t^{-1},                        (inverse Fisher information)
//! ```
//!
//! with `I_t = E[nabla_t^2]` the conditional Fisher information — the
//! **inverse-information scaling**, the Creal-Koopman-Lucas default. Two
//! observation densities are provided ([`Density`]):
//!
//! * **Gaussian**, `y_t ~ N(0, f_t)`. Here `nabla_t = 0.5(y_t^2 - f_t)/f_t^2`
//!   and `I_t = 0.5/f_t^2`, so `S_t = 2 f_t^2` and the scaled score
//!   collapses to `s_t = y_t^2 - f_t`; the recursion becomes the GARCH-like
//!   update `f_{t+1} = omega + a(y_t^2 - f_t) + b f_t`.
//!
//! * **Standardized Student-t** with `nu > 2` degrees of freedom, unit
//!   variance (so it nests toward the Gaussian as `nu -> inf`). Writing
//!   `eps^2 = y_t^2/f_t`, the scaled score is
//!   `s_t = ((nu+3)/nu) f_t (nu y_t^2 - (nu-2) f_t) / ((nu-2) f_t + y_t^2)`,
//!   which **down-weights** large `|y_t|` by the factor
//!   `(1 + y_t^2/((nu-2) f_t))^{-1}` — the outlier robustness that
//!   distinguishes GAS-t from GARCH. The `I_t = (nu/(nu+3))/(2 f_t^2)`
//!   scaling constant `E[g^2] = 2 nu/(nu+3)` is derived in
//!   [`kernel::scaled_score`] and validated by direct numerical integration
//!   in the fixture generator.
//!
//! ## Estimation, filtering, forecasting
//!
//! [`GasModel::filter`] runs the recursion at fixed parameters (returning
//! the filtered variance path, scaled scores, one-step-ahead variance, and
//! log-likelihood); [`GasModel::loglike`] returns the likelihood alone;
//! [`GasModel::fit`] maximizes it by `tsecon-optim` Nelder-Mead in a
//! reparameterized working space enforcing `omega > 0`, `a >= 0`,
//! `0 <= b < 1`, `nu > 2`. [`GasResults`] carries the fitted conditional
//! variances, standardized residuals, information criteria, and an
//! `h`-step variance forecast ([`GasResults::forecast`]).
//!
//! The recursion is initialized at the stationary mean
//! `f_1 = omega / (1 - b)` (the score is conditionally mean-zero, so `a`
//! does not shift the mean of `f_t`).
//!
//! ## Cross-package validation
//!
//! The golden fixture `fixtures/tsecon-gas.json` fixes the parameters and a
//! return series and computes the filtered `f_t` path and total
//! log-likelihood by literally applying the documented recursion and
//! density in NumPy (with the Student-t density additionally cross-checked
//! against `scipy.stats.t`). Rust reproduces both to `~1e-10`. A property
//! test on a longer simulated series checks maximum-likelihood parameter
//! recovery and that the likelihood at the MLE exceeds that at a perturbed
//! start.
//!
//! ```
//! use tsecon_gas::{Density, GasModel, GasParams};
//!
//! let y = [0.4, -1.2, 0.3, 0.8, -0.5, 1.4, -0.9, 0.2, -0.1, 0.6];
//! let model = GasModel::new(&y, Density::Gaussian).unwrap();
//! let out = model.filter(&GasParams::gaussian(0.05, 0.04, 0.92)).unwrap();
//! assert_eq!(out.variance.len(), y.len());
//! assert!(out.loglik.is_finite());
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
pub mod kernel;
mod model;
mod results;

pub use error::GasError;
pub use kernel::Density;
pub use model::{forecast_from, GasModel, GasParams};
pub use results::{GasFiltered, GasResults};
