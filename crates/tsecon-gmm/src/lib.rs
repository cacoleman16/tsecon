//! # tsecon-gmm — generalized method of moments (Hansen 1982)
//!
//! The GMM estimation layer of the `tsecon` time-series econometrics
//! library. It provides the linear instrumental-variables GMM family used in
//! applied work, plus a general driver for user-supplied nonlinear moment
//! conditions. Every numeric result on the linear path is validated against a
//! `linearmodels` golden fixture to machine precision (`tests/golden.rs`).
//!
//! ## Module map
//!
//! - [`one_step_gmm`] — linear GMM with a caller-supplied weighting matrix,
//!   `beta(W) = (X'Z W Z'X)^{-1} X'Z W Z'y`, with the robust sandwich
//!   covariance and the Hansen J-test.
//! - [`two_stage_least_squares`] — the `W = (Z'Z/n)^{-1}` special case (2SLS),
//!   which for an exactly identified model equals the simple IV estimator for
//!   any weight.
//! - [`two_step_gmm`] — the efficient two-step estimator (Hansen 1982):
//!   step 1 is 2SLS, step 2 re-weights by the inverse moment covariance
//!   `S(u1)^{-1}`. With [`GmmWeight::Robust`] this reproduces
//!   `linearmodels` `IVGMM(...).fit()` (`weight_type="robust"`,
//!   `cov_type="robust"`) exactly.
//! - [`iterated_gmm`] — iterate the (re-weight, re-estimate) loop to a fixed
//!   point.
//! - [`gmm_nonlinear`] — minimize `gbar(theta)' W gbar(theta)` over
//!   parameters for an arbitrary moment function, via the `tsecon-optim`
//!   Nelder-Mead simplex.
//!
//! ## Weighting and covariance
//!
//! The moment-score covariance `S` (used both for the efficient weight
//! `W = S^{-1}` and for the sandwich covariance meat) is estimated either
//! heteroskedasticity-robustly ([`GmmWeight::Robust`], White 1980) or with a
//! HAC kernel ([`GmmWeight::Hac`], Newey-West 1987 via the library's single
//! kernel owner [`tsecon_hac::Kernel`]). The robust parameter covariance is
//! the general GMM sandwich; see [`crate::linear`] for the exact conventions
//! (pinned empirically to the golden fixture) and their tolerances.
//!
//! ```
//! use tsecon_gmm::{two_step_gmm, GmmWeight};
//!
//! // y ~ [const, w] exogenous + x endogenous; instruments [const, w, z].
//! let n = 200;
//! let cst = vec![1.0; n];
//! let w: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
//! let z: Vec<f64> = (0..n).map(|t| (0.17 * t as f64).cos()).collect();
//! let x: Vec<f64> = (0..n).map(|t| 0.6 * z[t] + 0.2 * w[t]).collect();
//! let y: Vec<f64> = (0..n).map(|t| 1.0 - 0.5 * w[t] + 0.5 * x[t]).collect();
//!
//! let xcols = vec![cst.clone(), w.clone(), x];
//! let zcols = vec![cst, w, z];
//! let fit = two_step_gmm(&xcols, &zcols, &y, GmmWeight::Robust).unwrap();
//! assert_eq!(fit.params.len(), 3);
//! assert!(fit.jtest.is_none()); // exactly identified here (3 instruments, 3 params)
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
pub mod linear;
mod matrix;
pub mod nonlinear;

pub use error::GmmError;
pub use linear::{
    iterated_gmm, one_step_gmm, two_stage_least_squares, two_step_gmm, GmmFit, GmmWeight, HansenJ,
};
pub use nonlinear::{gmm_nonlinear, NonlinearGmmFit};
