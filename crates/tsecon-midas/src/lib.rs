//! # tsecon-midas — MIDAS mixed-frequency regressions
//!
//! The first nowcasting slice of the tsecon library (ROADMAP §8,
//! "Nowcasting & mixed-frequency"): the MIDAS (MIxed-DAta Sampling) regression
//! family, which relates a low-frequency target (quarterly GDP) to many lags
//! of a higher-frequency predictor (monthly indicators) without pre-averaging
//! away the within-period information. This crate owns the MIDAS **weighting
//! machinery** — the reusable piece the wider nowcasting stack and volatility's
//! GARCH-MIDAS both consume — and the regression estimators built on it,
//! delegating every shared numeric to the library's single owners:
//!
//! - all OLS solves and standard errors (nonrobust / HC / HAC) to
//!   [`tsecon_hac`];
//! - the nonlinear-least-squares fit of weighted MIDAS to [`tsecon_optim`].
//!
//! ## What this crate provides
//!
//! - **Weight functions** ([`exp_almon_weights`], [`beta_weights`],
//!   [`almon_pdl_basis`] / [`almon_weights`]) — the exponential-Almon and
//!   two-parameter Beta profiles (each normalized to sum to one) and the
//!   linear Almon polynomial-distributed-lag basis, with their exact
//!   published formulas.
//! - **U-MIDAS** and **ADL-MIDAS** ([`umidas`], [`adl_midas`]) — unrestricted
//!   and autoregressive mixed-frequency regressions that are exactly OLS on a
//!   stacked design (Foroni, Marcellino & Schumacher 2015).
//! - **Weighted MIDAS** ([`weighted_midas`]) — the parsimonious
//!   Beta / exponential-Almon regression fit by nonlinear least squares.
//! - **The mixed-frequency design builder** ([`stack_high_freq_lags`]) — the
//!   ragged-edge, most-recent-first stacked high-frequency-lag matrix that
//!   feeds every estimator here.
//!
//! ```
//! use tsecon_midas::{stack_high_freq_lags, umidas, WeightScheme, weighted_midas};
//! use tsecon_hac::SeType;
//!
//! // 60 months of a high-frequency indicator, 20 quarters of a target,
//! // generated deterministically so the example is reproducible.
//! let mut state = 1u64;
//! let mut draw = || {
//!     state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
//!     (state >> 33) as f64 / (1u64 << 30) as f64 - 1.0 // in [-1, 1)
//! };
//! let hf: Vec<f64> = (0..60).map(|_| draw()).collect();
//! let low: Vec<f64> = (0..20).map(|_| draw()).collect();
//!
//! // Stack K = 4 monthly lags, most-recent-first, ratio = 3 (months/quarter).
//! let design = stack_high_freq_lags(&hf, &low, 3, 4).unwrap();
//!
//! // Unrestricted MIDAS by OLS with nonrobust standard errors.
//! let fit = umidas(&design.target, &design.columns, SeType::NonRobust).unwrap();
//! assert_eq!(fit.params.len(), 5); // intercept + 4 lag coefficients
//!
//! // Parsimonious weighted MIDAS by nonlinear least squares.
//! let w = weighted_midas(&design.target, &design.columns, WeightScheme::ExpAlmon, None)
//!     .unwrap();
//! let sum: f64 = w.weights.iter().sum();
//! assert!((sum - 1.0).abs() < 1e-12);
//! ```
//!
//! ## References
//!
//! - Almon, S. (1965). "The Distributed Lag Between Capital Appropriations and
//!   Expenditures." *Econometrica*.
//! - Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). "The MIDAS Touch:
//!   Mixed Data Sampling Regression Models." Working paper.
//! - Ghysels, E., Sinko, A., & Valkanov, R. (2007). "MIDAS Regressions:
//!   Further Results and New Directions." *Econometric Reviews*.
//! - Andreou, E., Ghysels, E., & Kourtellos, A. (2013). "Should Macroeconomic
//!   Forecasters Use Daily Financial Data and How?" *Journal of Business &
//!   Economic Statistics*.
//! - Foroni, C., Marcellino, M., & Schumacher, C. (2015). "Unrestricted Mixed
//!   Data Sampling (MIDAS): MIDAS Regressions with Unrestricted Lag
//!   Polynomials." *Journal of the Royal Statistical Society A*.
//! - Ghysels, E., Kvedaras, V., & Zemlys, V. (2016). "Mixed Frequency Data
//!   Sampling Regression Models: The R Package midasr." *Journal of
//!   Statistical Software*.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod design;
mod error;
mod umidas;
mod weighted;
mod weights;

pub use design::{stack_high_freq_lags, StackedDesign};
pub use error::MidasError;
pub use umidas::{adl_midas, umidas, MidasFit};
pub use weighted::{weighted_midas, WeightScheme, WeightedMidasFit};
pub use weights::{almon_pdl_basis, almon_weights, beta_weights, exp_almon_weights, BETA_EPS};

// Re-export the shared standard-error selector so callers configuring U-MIDAS
// / ADL-MIDAS inference do not need a separate `tsecon-hac` dependency.
pub use tsecon_hac::SeType;
