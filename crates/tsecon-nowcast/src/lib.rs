//! # tsecon-nowcast
//!
//! Dynamic-factor-model **nowcasting** for the `tsecon` time-series
//! econometrics library: the Doz-Giannone-Reichlin (2011) two-step estimator
//! with Banbura-Modugno (2014) ragged-edge Kalman filtering. Nowcasting turns
//! a panel of mixed, partially-observed indicators into a real-time estimate
//! of a target series at the sample edge, using whatever data have arrived.
//!
//! ## The two-step estimator
//!
//! For a `T x N` panel with `r` common factors following a VAR(`p`):
//!
//! 1. **Standardize** the panel and extract `r` principal-component factors
//!    (the validated [`tsecon_favar::FactorModel`]).
//! 2. **Fit a factor VAR(`p`)** to the PC factors ([`tsecon_var`]) for the
//!    transition dynamics; read the idiosyncratic variances off the
//!    standardized reconstruction residuals.
//! 3. **Cast into state space** and run the Kalman **filter/smoother**
//!    ([`tsecon_ssm`]) to re-estimate the common factor optimally and obtain
//!    the Gaussian log-likelihood.
//!
//! The nowcast of a target series is its loadings dotted with the smoothed
//! factor at the edge, de-standardized to the series' level.
//!
//! ## Ragged edge
//!
//! Real-time panels are *jagged*: faster indicators are observed through the
//! current period while slower ones lag. The nowcast panel therefore carries
//! `NaN` in the last rows of the trailing series; the univariate Kalman filter
//! skips those missing elements in each measurement update, so the nowcast is
//! formed from exactly the data that have arrived. This is the whole point of
//! nowcasting, and it is handled by [`Nowcaster::nowcast_panel`].
//!
//! ## Validation: what is exact and what is not
//!
//! * **The Kalman / state-space step is reference-exact.** A single-factor
//!   DFM with an AR(`p`) factor and white-noise idiosyncratic errors is
//!   *exactly* statsmodels' `DynamicFactor(k_factors, factor_order=p,
//!   error_order=0)` state space (AR coefficients in the first rows of the
//!   companion transition, factor-innovation covariance `Q`, diagonal
//!   observation covariance, zero intercepts, stationary initialization).
//!   Given the same parameters and panel, [`smooth_fixed`] reproduces
//!   statsmodels' Kalman log-likelihood and smoothed states to ~`1e-8`
//!   (`fixtures/tsecon-nowcast.json`, `tests/golden.rs`). This isolates and
//!   validates the crate's Kalman step against an independent reference.
//! * **The two-step parameter estimates are the DGR estimator, *not* one-step
//!   MLE.** The DGR two-step estimator (PCA -> factor VAR -> one Kalman pass)
//!   is a different estimator from statsmodels' one-step Gaussian MLE, so their
//!   parameter estimates and smoothed factors do **not** coincide and are
//!   deliberately **not** tolerance-matched. The two-step estimator is
//!   validated structurally: on simulated data its smoothed factor tracks the
//!   true factor (`corr > 0.9`), the balanced nowcast is finite, and the
//!   ragged-edge nowcast moves in the economically expected direction.
//!
//! ## News / update decomposition
//!
//! When a newer data vintage arrives, the target nowcast revises. For fixed
//! parameters the Kalman smoother is a purely linear operator on the
//! observations, so the revision decomposes **exactly** as a weighted sum of
//! *news*: `new_nowcast - old_nowcast = Σ_j weight_j · news_j`, where
//! `news_j = actual_j - E_old[y_j]` is the surprise in each newly-revealed
//! observation and `weight_j = ∂nowcast/∂actual_j` is its Kalman weight
//! (Banbura-Modugno 2014). This adding-up identity is a *self-validating*
//! exact identity — asserted to ~`1e-10` — and the analytic Kalman weights are
//! cross-checked against an independent finite-difference reference to ~`1e-6`
//! (`fixtures/nowcast_news.json`). See [`Nowcaster::news_decomposition`] and
//! [`news::news_decomposition_at`], and the [`news`] module docs for the
//! derivation.
//!
//! All fallible routines return [`NowcastError`]; nothing in this crate panics
//! on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod error;
pub mod news;
pub mod statespace;
pub mod twostep;

pub use error::NowcastError;
pub use news::{news_decomposition_at, NewsContribution, NewsDecomposition};
pub use statespace::{smooth_fixed, DfmParams, DfmSmoothing};
pub use twostep::{NowcastResult, Nowcaster};

// Re-export the state-space engine (and, through it, the shared linear-algebra
// and dense backend) so downstream crates see one faer version and one
// `LinearGaussianSSM` type.
pub use tsecon_ssm;
