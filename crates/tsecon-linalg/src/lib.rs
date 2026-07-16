//! # tsecon-linalg
//!
//! Structured linear-algebra solvers for time series econometrics, built
//! on [`faer`] as the dense backend. This crate owns the algorithms that
//! exploit time-series structure (Toeplitz autocovariance matrices,
//! companion forms, stationary Lyapunov covariances) plus the shared
//! positive-definiteness hygiene utilities used by every covariance the
//! library emits.
//!
//! Contents:
//!
//! * [`levinson_durbin`] / [`levinson_durbin_from_series`] — the
//!   Levinson-Durbin recursion (AR coefficients at every order, partial
//!   autocorrelations, innovation variances), Brockwell-Davis conventions;
//! * [`toeplitz_solve`] — `O(n^2)` solve of a symmetric positive definite
//!   Toeplitz system with arbitrary right-hand side (Levinson recursion);
//! * [`solve_discrete_lyapunov`] — `X = A X A' + Q` via the doubling
//!   algorithm; the stationary-initialization workhorse for state-space
//!   models;
//! * [`companion_from_ar`] / [`companion_from_var`] / [`spectral_radius`]
//!   / [`is_stable`] / [`ar_psi_weights`] — companion-form utilities and
//!   the MA(infinity) / impulse-response expansion primitive;
//! * [`symmetrize`] / [`jittered_cholesky`] — positive-definiteness
//!   hygiene helpers.
//!
//! All fallible routines return [`LinalgError`]; nothing in this crate
//! panics on user input.

pub mod companion;
pub mod error;
pub mod hygiene;
pub mod levinson;
pub mod lyapunov;

pub use companion::{
    ar_psi_weights, companion_from_ar, companion_from_var, is_stable, spectral_radius,
};
pub use error::LinalgError;
pub use hygiene::{jittered_cholesky, symmetrize, JitteredCholesky};
pub use levinson::{
    autocovariances_biased, levinson_durbin, levinson_durbin_from_series, toeplitz_solve,
    LevinsonDurbin,
};
pub use lyapunov::solve_discrete_lyapunov;

// Re-export the dense backend so sibling crates and tests use one faer
// version through this crate.
pub use faer;
