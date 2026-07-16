//! # tsecon-var
//!
//! Reduced-form VAR(p) estimation and analysis — the macro workhorse of
//! the `tsecon` time-series econometrics library (see ROADMAP). Every
//! numeric convention follows statsmodels' `VAR` (which in turn follows
//! Lütkepohl 2005), and the golden fixture `fixtures/var.json`
//! arbitrates:
//!
//! * [`VarSpec`] / [`VarSpec::fit`] — equation-by-equation OLS
//!   (multivariate least squares) with a [`Trend::Constant`] or
//!   [`Trend::None`] deterministic term; [`VarResults`] exposes the
//!   coefficient matrices, intercept, residuals, the df-adjusted and ML
//!   residual covariances, the Gaussian log-likelihood, and
//!   AIC/BIC/HQIC/FPE;
//! * [`select_order`] — lag-order selection over a common sample, per
//!   criterion (statsmodels `select_order` conventions);
//! * [`VarResults::companion`] / [`VarResults::roots_moduli`] /
//!   [`VarResults::is_stable`] — companion-form stability analysis via
//!   `tsecon-linalg` eigenvalues;
//! * [`VarResults::test_causality`] — Granger-causality F tests
//!   (statsmodels `test_causality(kind="f")`);
//! * [`ma_rep`] / [`VarResults::irf`] — non-orthogonalized and
//!   Cholesky-orthogonalized impulse responses;
//! * [`VarResults::fevd`] — forecast-error variance decomposition;
//! * [`VarResults::forecast`] / [`VarResults::forecast_interval`] —
//!   iterated point forecasts with asymptotic (innovation-uncertainty
//!   only) intervals.
//!
//! The results object deliberately exposes `sigma_u`, `coefs`, `resid`,
//! `params`, and `zz_inv` as public fields so the structural
//! identification (SVAR) layer can consume the reduced form directly.
//!
//! Bootstrap IRF confidence bands are `// TODO(phase0)` and will consume
//! `tsecon-bootstrap` (see `src/irf.rs`).
//!
//! All fallible routines return [`VarError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod causality;
pub mod error;
mod estimate;
pub mod fevd;
pub mod forecast;
pub mod irf;
pub mod results;
pub mod select;
pub mod spec;

pub use causality::CausalityTest;
pub use error::VarError;
pub use fevd::Fevd;
pub use forecast::ForecastInterval;
pub use irf::{ma_rep, Irf};
pub use results::VarResults;
pub use select::{select_order, LagOrderCandidate, LagOrderSelection};
pub use spec::{Trend, VarSpec};

// Re-export the shared linear-algebra layer (and, through it, the dense
// backend) so downstream crates see one faer version.
pub use tsecon_linalg;
