//! # tsecon-bayes
//!
//! The Bayesian foundations of the tsecon time-series econometrics
//! library (roadmap module 05, Tier 1): the pieces every Bayesian model
//! in the library is built from.
//!
//! Contents:
//!
//! * [`MinnesotaNiwPrior`] / [`NiwPosterior`] — the natural-conjugate
//!   Minnesota / Normal-inverse-Wishart BVAR: closed-form posterior
//!   moments, the matrix-variate-t log marginal likelihood (the anchor
//!   for hierarchical hyperparameter selection à la Giannone-Lenza-
//!   Primiceri 2015), joint `(B, Sigma)` posterior sampling through the
//!   Kronecker structure (never forming the `nk x nk` covariance), and
//!   posterior draws of Cholesky-orthogonalized impulse responses
//!   ([`cholesky_irf`]);
//! * [`FfbsSampler`] — the Carter-Kohn (1994) forward-filter
//!   backward-sampling simulation smoother on the shared `tsecon-ssm`
//!   univariate filter, with rank-aware (pseudo-inverse) handling of
//!   singular `R Q R'` selection-form state covariances;
//! * [`rhat_rank`] / [`ess_bulk`] / [`ess_tail`] / [`ess_mean`] —
//!   convergence diagnostics per Vehtari, Gelman, Simpson, Carpenter &
//!   Bürkner (2021), numerically matching ArviZ / the R `posterior`
//!   package on shared draws.
//!
//! Randomness enters exclusively through [`tsecon_rng::Stream`] uniforms
//! mapped by inverse CDFs (`tsecon-stats`), so every draw is reproducible
//! under the library-wide substream contract. A ziggurat normal sampler
//! is a planned replacement for the inverse-CDF transform
//! (`TODO(phase0)` in `dense.rs`); the transform is exact, only slower.
//!
//! All fallible routines return [`BayesError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod dense;

pub mod convergence;
pub mod error;
pub mod ffbs;
pub mod hierarchical;
pub mod niw;

pub use convergence::{ess_bulk, ess_mean, ess_tail, ess_tail_prob, rhat_rank};
pub use error::BayesError;
pub use ffbs::FfbsSampler;
pub use hierarchical::{bvar_hierarchical, HierarchicalConfig, HierarchicalFit, Hyperprior};
pub use niw::{cholesky_irf, MinnesotaNiwPrior, NiwDraw, NiwPosterior};

// Re-export the state-space engine (and, through it, the shared
// linear-algebra layer and faer) so Gibbs-block consumers see one stack.
pub use tsecon_ssm;
