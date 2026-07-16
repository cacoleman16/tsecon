//! # tsecon-hac — the single HAC / long-run variance engine of tsecon
//!
//! Every regression-based estimator in the library draws its
//! heteroskedasticity-and-autocorrelation-robust inference from this crate
//! (ROADMAP §5: one owner per capability — identical settings must never
//! produce different p-values in different modules). It provides:
//!
//! - [`Kernel`]: Bartlett (Newey-West 1987), Parzen, quadratic spectral
//!   (Andrews 1991), and truncated lag windows, with the Andrews plug-in
//!   constants attached.
//! - [`lrv`]: kernel long-run variance of a mean-zero univariate series,
//!   with explicit bandwidth.
//! - Automatic bandwidths: [`newey_west_bandwidth`] (Newey-West 1994
//!   nonparametric plug-in), [`andrews_bandwidth_ar1`] (Andrews 1991 AR(1)
//!   parametric plug-in), and the ubiquitous rule of thumb
//!   [`newey_west_maxlags`] `= floor(4 (n/100)^(2/9))`.
//! - [`lrv_prewhitened_ar1`]: AR(1) prewhitened-and-recolored LRV
//!   (Andrews-Monahan 1992).
//! - [`ewc_lrv`] / [`ewc_default_b`]: the equal-weighted cosine estimator
//!   of Lazarus-Lewis-Stock-Watson (2018).
//! - [`ols`] + [`OlsFit::inference`]: OLS with nonrobust, HC0/HC1, and HAC
//!   sandwich standard errors matching statsmodels `cov_type="HAC"`
//!   (`maxlags`, `use_correction`) to golden-fixture precision.
//!
//! ## The library-wide default policy (ROADMAP §5; LLSW 2018)
//!
//! Following Lazarus, Lewis, Stock & Watson (2018), the library's default
//! HAR inference is **EWC with `B = round(0.4 n^(2/3))` degrees of freedom
//! and Student-t critical values with `B` degrees of freedom** (fixed-b,
//! not normal, critical values). Kernel HAC — Bartlett with the
//! [`newey_west_maxlags`] rule, optionally prewhitened — is retained as
//! the statsmodels/R-compatibility option. Consumers must take both from
//! this crate rather than reimplementing.
//!
//! ```
//! use tsecon_hac::{andrews_bandwidth_ar1, ewc_default_b, ewc_lrv, lrv, Kernel};
//!
//! // A mean-zero series (demean your data or pass residuals/scores).
//! let x: Vec<f64> = (0..200).map(|t| (0.3 * t as f64).sin()).collect();
//!
//! // Kernel LRV with an Andrews (1991) automatic bandwidth:
//! let kernel = Kernel::QuadraticSpectral;
//! let scale = andrews_bandwidth_ar1(&x, kernel).unwrap();
//! let omega = lrv(&x, kernel, kernel.bandwidth_from_scale(scale)).unwrap();
//!
//! // The library default: EWC with t_B inference.
//! let b = ewc_default_b(x.len());
//! let omega_ewc = ewc_lrv(&x, b).unwrap();
//! assert!(omega.is_finite() && omega_ewc.is_finite());
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
mod ewc;
mod kernel;
mod lrv;
mod ols;
mod validate;

pub use error::HacError;
pub use ewc::{ewc_default_b, ewc_lrv};
pub use kernel::Kernel;
pub use lrv::{
    andrews_bandwidth_ar1, lrv, lrv_prewhitened_ar1, newey_west_bandwidth, newey_west_maxlags,
    PrewhitenedLrv,
};
pub use ols::{ols, OlsFit, OlsInference, SeType};
