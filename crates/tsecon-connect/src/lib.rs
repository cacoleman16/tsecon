//! # tsecon-connect
//!
//! Variance-decomposition **connectedness** in the sense of Diebold and
//! Yilmaz — the Module 04 (multivariate) connectedness layer of the
//! `tsecon` time-series econometrics library (see ROADMAP). Given a
//! fitted reduced-form VAR from [`tsecon_var`], this crate measures how
//! much of each variable's forecast-error variance is explained by shocks
//! to the *other* variables, and aggregates those cross-variable shares
//! into total, directional, net, and pairwise connectedness indices.
//!
//! The construction, and the golden fixture `fixtures/connect.json`
//! (a VAR(2)-with-constant on macro data, horizon 10), follow:
//!
//! * [`generalized_fevd`] — the generalized forecast-error variance
//!   decomposition of Pesaran and Shin (1998), which (unlike the
//!   Cholesky FEVD) is invariant to the ordering of the variables,
//!   row-normalized so each variable's shares sum to one (the
//!   Diebold-Yilmaz normalization);
//! * [`ConnectednessTable`] — the Diebold and Yilmaz (2012, 2014)
//!   connectedness measures (total spillover index; directional "to
//!   others" and "from others"; net; net pairwise) with a [`Display`]
//!   that prints the standard spillover table with row/column margins;
//! * [`rolling_total_connectedness`] — the time-varying total index over
//!   a rolling estimation window.
//!
//! The generalized-FEVD core operates on bare MA(inf) `Psi` weights and a
//! residual covariance, so it can be exercised without an estimation
//! step; [`ConnectednessTable::from_var`] is the convenience path that
//! consumes a [`tsecon_var::VarResults`] directly (its
//! [`tsecon_var::VarResults::ma_rep`] weights and `sigma_u`).
//!
//! All fallible routines return [`ConnectError`]; nothing in this crate
//! panics on user input.
//!
//! ## References
//!
//! * Pesaran, H. H. and Shin, Y. (1998), "Generalized impulse response
//!   analysis in linear multivariate models", *Economics Letters* 58,
//!   17-29.
//! * Diebold, F. X. and Yilmaz, K. (2012), "Better to give than to
//!   receive: Predictive directional measurement of volatility
//!   spillovers", *International Journal of Forecasting* 28, 57-66.
//! * Diebold, F. X. and Yilmaz, K. (2014), "On the network topology of
//!   variance decompositions: Measuring the connectedness of financial
//!   firms", *Journal of Econometrics* 182, 119-134.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod error;
pub mod gfevd;
pub mod rolling;
pub mod table;

pub use error::ConnectError;
pub use gfevd::generalized_fevd;
pub use rolling::rolling_total_connectedness;
pub use table::ConnectednessTable;

// Re-export the reduced-form VAR layer (and, through it, the shared
// linear-algebra backend) so downstream callers see one faer version.
pub use tsecon_var;
