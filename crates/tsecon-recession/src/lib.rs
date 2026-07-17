//! # tsecon-recession — recession-probability models
//!
//! Static and dynamic binary-choice models for the probability of recession:
//! a `{0, 1}` recession indicator `y_t` regressed on leading predictors `X_t`
//! (a constant plus, typically, the term spread), estimated by exact-likelihood
//! maximum likelihood. This is the roadmap E8 crate.
//!
//! ## The models
//!
//! **Static probit / logit.** With linear index `index_t = X_t' beta`,
//!
//! ```text
//! P(y_t = 1 | X_t) = F(index_t),   F = Phi (probit)  or  Lambda (logit),
//! ```
//!
//! fit by maximizing the exact log-likelihood
//! `sum_t [ y_t ln F(index_t) + (1 - y_t) ln(1 - F(index_t)) ]`. The estimator
//! reports the coefficients, their standard errors (from the inverse observed
//! information — the negative analytic Hessian at the MLE), z-statistics, the
//! log-likelihood, McFadden's pseudo-R^2, and the in-sample fitted probability
//! path. See [`fit_static`] / [`RecessionFit`] and [`Link`].
//!
//! **Dynamic probit (Kauppi & Saikkonen 2008).** The index carries an
//! autoregressive term,
//!
//! ```text
//! index_t = w + X_t' b + rho * index_{t-1},   P(y_t = 1) = Phi(index_t),
//! ```
//!
//! fit by ML over `(w, b, rho)`, with the index initialized at its stationary
//! mean and `rho` held in `(-1, 1)`. See [`fit_dynamic_probit`] /
//! [`DynamicProbitFit`].
//!
//! The normal and logistic CDF/pdf come from [`tsecon_stats`], the optimizer
//! from [`tsecon_optim`], and the dense factorizations from `tsecon-linalg`'s
//! `faer` backend. This crate reimplements none of them.
//!
//! ## Module map
//!
//! - [`Link`] — the probit (`Phi`) or logit (`Lambda`) link and its
//!   per-observation likelihood, score, and information quantities.
//! - [`fit_static`] / [`RecessionFit`] — the static probit/logit MLE.
//! - [`fit_dynamic_probit`] / [`DynamicProbitFit`] — the dynamic probit MLE.
//! - [`RecessionError`] — the crate's error type ("errors that teach").
//!
//! ## Validation
//!
//! The **static** probit and logit are pinned to an INDEPENDENT reference:
//! `fixtures/tsecon-recession.json` is produced by fitting statsmodels'
//! `sm.Probit` / `sm.Logit` on a fixed simulated dataset, and the Rust
//! reproduces its `params`, `bse`, `llf`, McFadden pseudo-R^2, and the fitted
//! probability path to ~1e-6 (`tests/golden.rs`). Because statsmodels computes
//! these by an entirely separate code path, the match is a genuine
//! cross-implementation check.
//!
//! The **dynamic** probit has no statsmodels reference and is validated
//! PROPERTY-ONLY (`tests/properties.rs`): on data simulated from a known
//! dynamic-probit DGP it recovers `rho` and `b` within Monte-Carlo bands, and
//! its log-likelihood exceeds the static model's on persistent data. This is
//! documented as property-only, never reference-matched.
//!
//! ```
//! use tsecon_recession::{fit_static, Link};
//!
//! // A constant and one leading predictor.
//! let y = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
//! let c = vec![1.0; 8];
//! let spread = vec![1.2, -0.3, -0.5, 0.4, 0.3, -0.9, -1.3, 0.6];
//! let fit = fit_static(&y, &[c, spread], Link::Probit).unwrap();
//! assert_eq!(fit.params.len(), 2);
//! assert!(fit.fitted.iter().all(|&p| (0.0..=1.0).contains(&p)));
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod design;
mod dynamic;
mod error;
mod link;
mod static_model;

pub use dynamic::{fit_dynamic_probit, DynamicProbitFit};
pub use error::RecessionError;
pub use link::Link;
pub use static_model::{fit_static, RecessionFit};
