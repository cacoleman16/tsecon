//! # tsecon-ssm
//!
//! The linear-Gaussian state-space engine: the single representation and
//! filter/smoother stack on which the library's model classes (ARIMA,
//! unobserved components, dynamic factor models, TVP regressions,
//! nowcasting, ...) are built.
//!
//! Model form (Durbin & Koopman 2012 notation, plus intercepts):
//!
//! ```text
//! y_t         = d + Z alpha_t + eps_t,      eps_t ~ N(0, H)
//! alpha_{t+1} = c + T alpha_t + R eta_t,    eta_t ~ N(0, Q)
//! alpha_1 ~ N(a_1, P_* + kappa P_inf),      kappa -> infinity (exact)
//! ```
//!
//! Contents:
//!
//! * [`LinearGaussianSSM`] — validated model storage built through
//!   [`SsmBuilder`] (dimension checks, symmetric-PSD hygiene on `H` and
//!   `Q` via the shared jitter-ladder Cholesky), with convenience
//!   constructors [`LinearGaussianSSM::local_level`] and
//!   [`LinearGaussianSSM::ar`];
//! * [`Initialization`] — known, stationary (discrete-Lyapunov
//!   unconditional moments), exact diffuse (Koopman 1997), or mixed
//!   diffuse/stationary via a per-state flag;
//! * [`filter_univariate`] — the primary filter: univariate (sequential)
//!   processing (Koopman & Durbin 2000) with the exact-diffuse
//!   `(P_inf, P_*)` two-matrix recursions (Koopman & Durbin 2003), NaN
//!   missing-value handling, and the prediction-error-decomposition
//!   log-likelihood with exact-diffuse corrections;
//! * [`filter_matrix`] — the standard matrix Kalman filter with
//!   Joseph-form covariance update, kept as an independent cross-check
//!   path;
//! * [`smooth_univariate`] — the Durbin-Koopman backward state smoother
//!   in univariate form, exact through the diffuse period.
//!
//! System matrices are time-invariant in this pass but are stored behind
//! the [`SystemMatrix`] accessor enum so time-varying support can be
//! added without an API break. All fallible routines return
//! [`SsmError`]; nothing in this crate panics on user input.
//!
//! Numerical conventions (tolerances, likelihood constants, filtered /
//! smoothed quantity definitions) match statsmodels' state-space
//! implementation with `use_exact_diffuse=True`, which this crate is
//! golden-tested against on the Nile data.

mod dense;
pub mod error;
pub mod filter;
pub mod model;
pub mod smoother;

pub use error::SsmError;
pub use filter::{filter_matrix, filter_univariate, FilterOutput, MatrixFilterOutput};
pub use model::{InitialState, Initialization, LinearGaussianSSM, SsmBuilder, SystemMatrix};
pub use smoother::{smooth_univariate, SmootherOutput};

// Re-export the shared linear-algebra layer (and, through it, the dense
// backend) so downstream crates see one faer version.
pub use tsecon_linalg;
