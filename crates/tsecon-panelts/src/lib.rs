//! # tsecon-panelts
//!
//! Heterogeneous-panel time-series estimators for the `tsecon` econometrics
//! library — estimators for panels in which every unit has its *own* slope
//! coefficients rather than a single pooled slope. Two public surfaces, both
//! validated against an independent statsmodels golden (`fixtures/tsecon-panelts.json`):
//!
//! * [`mean_group`] — the Pesaran & Smith (1995) **mean-group (MG)** estimator.
//!   Fit a separate OLS per unit, drop the intercept, and report the simple
//!   cross-unit average of the slope vectors; the standard error is the
//!   cross-unit sample covariance of the per-unit slopes divided by `N`
//!   (`SE_k = sd_i(b_ik) / sqrt(N)`), the t-statistic uses a `t_{N-1}`
//!   reference. See [`mg`].
//! * [`cce_mean_group`] — the Pesaran (2006) **common-correlated-effects
//!   mean-group (CCE-MG)** estimator. Augment each unit with the per-period
//!   cross-section averages of `y` and of every `x` (which span the space of
//!   any unobserved common factor), fit the augmented per-unit OLS, and
//!   MG-average only the own-`x` slopes. This purges a common factor that
//!   would otherwise bias plain MG. See [`cce`].
//!
//! Both estimators are exact, deterministic maps from the per-unit OLS fits, so
//! the golden reproduces the coefficient vectors, standard errors, t-statistics,
//! and per-unit slopes to machine precision (`1e-10`). The per-unit OLS is
//! delegated to [`tsecon_hac::ols`]; the t-distribution p-values and confidence
//! bands ([`MeanGroup::pvalues`], [`MeanGroup::conf_int`]) use `tsecon-stats`.
//! Nothing in this crate reimplements least squares.
//!
//! ## Not (yet) implemented: PMG
//!
//! The Pesaran, Shin & Smith (1999) **pooled mean-group (PMG)** ARDL estimator
//! — which pools the *long-run* coefficients across units by maximum likelihood
//! while leaving the short-run dynamics and error-correction speeds free — is a
//! deliberate `TODO`. It is a constrained pooled-ML problem (a Newton /
//! back-substitution iteration over the concentrated likelihood), not a simple
//! average of independent OLS fits, so it has a different validation surface and
//! is out of scope for this deliverable. MG and CCE-MG are the required,
//! reference-validated estimators; PMG would be layered on the shared
//! [`PanelUnit`] input when added.
//!
//! All fallible routines return [`PanelTsError`]; nothing panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod cce;
pub mod error;
pub mod mg;

pub use cce::cce_mean_group;
pub use error::PanelTsError;
pub use mg::{mean_group, MeanGroup, PanelUnit};

// Re-export the OLS backend so downstream users see one `tsecon-hac` version
// and can inspect the per-unit fits with the same types this crate consumes.
pub use tsecon_hac;
