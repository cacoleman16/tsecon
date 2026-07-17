//! # tsecon-coint
//!
//! Cointegration and vector error-correction models — the multivariate
//! long-run layer of the `tsecon` time-series econometrics library (see
//! ROADMAP §04). Every numeric convention follows statsmodels 0.14.6
//! (`coint_johansen`, `VECM`), and the golden fixture `fixtures/coint.json`
//! arbitrates:
//!
//! * [`johansen`] — the Johansen (1991) cointegration-rank test: the two
//!   auxiliary regressions, the canonical-correlation eigenproblem, and the
//!   trace and maximum-eigenvalue likelihood-ratio statistics, with the
//!   MacKinnon-Haug-Michelis (1999) critical values ([`critical_values`])
//!   for the constant-in-data case (`det_order = 0`). [`JohansenResult`]
//!   exposes the eigenvalues, eigenvectors, both statistics, and sequential
//!   rank selection ([`JohansenResult::rank_trace`],
//!   [`JohansenResult::rank_max_eig`]).
//! * [`fit_vecm`] — Johansen maximum-likelihood VECM estimation at a fixed
//!   rank: the cointegrating vectors `beta` (normalized as statsmodels
//!   does, `beta[:r, :r] = I`), the loadings `alpha`, the short-run
//!   `Gamma` matrices, the residual covariance, and the log-likelihood.
//!   [`VecmResult`] also maps the fit to the equivalent level VAR
//!   ([`VecmResult::var_coefs`], [`VecmResult::companion`]) for downstream
//!   impulse responses.
//! * [`engle_granger`] — the Engle-Granger (1987) two-step residual-based
//!   test, delegating the residual unit-root step to `tsecon-diag`'s ADF.
//!
//! All fallible routines return [`CointError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod critvals;
pub mod engle_granger;
pub mod error;
pub mod johansen;
mod linalg;
pub mod vecm;

pub use critvals::{critical_values, DetOrder};
pub use engle_granger::{engle_granger, EngleGrangerResult, EngleGrangerTrend};
pub use error::CointError;
pub use johansen::{johansen, JohansenResult, SignificanceLevel};
pub use vecm::{fit_vecm, VecmResult};

// Re-export the shared linear-algebra layer (and, through it, the dense
// backend) so downstream crates see one faer version.
pub use tsecon_linalg;
