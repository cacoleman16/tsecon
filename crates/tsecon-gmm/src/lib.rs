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
//! A HAC bandwidth of zero is a **hard error**, not a computation: at
//! bandwidth 0 every kernel drops every lag and `S` reduces to the White
//! estimator, so `Hac { bandwidth: 0.0 }` silently returns exactly the
//! `Robust` answer with none of the serial-correlation robustness the caller
//! asked for. Pass a positive bandwidth, or [`GmmWeight::HacAuto`] for the
//! documented Newey-West (1994) rule
//! [`GmmWeight::auto_bandwidth`]`= floor(4 * (n/100)^(2/9))`, whose realized
//! value comes back in [`GmmFit::hac_bandwidth`].
//!
//! **That fixes a no-op, not the coverage.** Do not read `HacAuto` as the
//! remedy for HAC under-coverage. This library's interval-coverage audit
//! measured `iv_gmm(weight="hac")` under an AR(1) error with `phi = 0.8` at
//! `T = 250` and an explicit `bandwidth = 10` covering **0.868 ± 0.006
//! against a nominal 0.95**, an 8.2-point shortfall
//! (`docs/examples/interval-coverage.md`, Table 1) — and at that same
//! `T = 250` the automatic rule returns `floor(4 * 2.5^(2/9)) = 4` lags,
//! *fewer* than the setting that produced 0.868. With persistent moments a
//! nominal-95% GMM interval is narrower than its label at every bandwidth
//! this crate offers; the gap is open, not closed.
//!
//! ## Weak instruments
//!
//! Every fit reports [`GmmFit::first_stage`]: the robust first-stage F on the
//! excluded instruments, one per instrumented regressor. GMM's sandwich
//! standard errors are an asymptotic approximation that silently degrades as
//! the instruments weaken, and this library's own coverage audit found
//! nominal-95% GMM intervals covering only 0.915 at a *median first-stage F
//! of 10.5* — read [`FirstStageF`] before trusting `F > 10`.
//!
//! With **two or more endogenous regressors** this per-regressor F is not a
//! weak-identification test: every one of them can clear 10 while the system
//! is under-identified, because the instruments may predict only a single
//! common combination of the endogenous regressors. The statistics that do
//! answer that question — Angrist-Pischke per-regressor F, Cragg-Donald /
//! Kleibergen-Paap against Stock-Yogo critical values — are not implemented
//! here. See [`FirstStageF`].
//!
//! ```
//! use tsecon_gmm::{two_step_gmm, GmmWeight};
//!
//! // y ~ [const, w] exogenous + x endogenous; instruments [const, w, z].
//! // `x` carries its own variation `v` on top of the instrument, so the
//! // first stage is a real regression rather than an identity: instruments
//! // that reproduce a regressor exactly get no F at all (see `FirstStageF`).
//! let n = 200;
//! let cst = vec![1.0; n];
//! let w: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
//! let z: Vec<f64> = (0..n).map(|t| (0.17 * t as f64).cos()).collect();
//! let v: Vec<f64> = (0..n).map(|t| (1.1 * t as f64).sin()).collect(); // endogeneity shock
//! let x: Vec<f64> = (0..n).map(|t| 0.6 * z[t] + 0.2 * w[t] + 0.5 * v[t]).collect();
//! let y: Vec<f64> = (0..n).map(|t| 1.0 - 0.5 * w[t] + 0.5 * x[t] + 0.4 * v[t]).collect();
//!
//! let xcols = vec![cst.clone(), w.clone(), x];
//! let zcols = vec![cst, w, z];
//! let fit = two_step_gmm(&xcols, &zcols, &y, GmmWeight::Robust).unwrap();
//! assert_eq!(fit.params.len(), 3);
//! assert!(fit.jtest.is_none()); // exactly identified here (3 instruments, 3 params)
//!
//! // Weak-instrument diagnostic: one entry, for the one instrumented
//! // regressor (`x`, column 2). `const` and `w` instrument themselves.
//! assert_eq!(fit.first_stage.len(), 1);
//! assert_eq!(fit.first_stage[0].regressor, 2);
//! assert_eq!(fit.first_stage[0].dof_num, 1); // one excluded instrument, `z`
//! assert!(fit.first_stage[0].fstat.is_finite());
//! assert_eq!(fit.hac_bandwidth, None); // no HAC weighting was requested
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
pub mod linear;
mod matrix;
pub mod nonlinear;

pub use error::GmmError;
pub use linear::{
    iterated_gmm, one_step_gmm, two_stage_least_squares, two_step_gmm, FirstStageF, GmmFit,
    GmmWeight, HansenJ,
};
pub use nonlinear::{gmm_nonlinear, NonlinearGmmFit};
