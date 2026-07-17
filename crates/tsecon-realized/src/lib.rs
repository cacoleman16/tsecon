//! # tsecon-realized — realized-volatility measures and the HAR-RV model
//!
//! Module 03 (volatility) realized-measure layer of tsecon. Given a vector
//! of intraday log-returns for a trading day it computes nonparametric
//! estimators of that day's quadratic variation and its continuous / jump
//! decomposition, and given a series of daily realized variances it fits
//! the Corsi (2009) HAR-RV forecasting regression.
//!
//! ## Realized measures (on an intraday return vector)
//!
//! - [`realized_variance`] — `RV = sum r_i^2`, the quadratic variation
//!   (Andersen, Bollerslev, Diebold & Labys 2001).
//! - [`bipower_variation`] — `BV = (pi/2) sum |r_i||r_{i-1}|`, the
//!   jump-robust continuous variation (Barndorff-Nielsen & Shephard 2004).
//! - [`realized_quarticity`] / [`tripower_quarticity`] — integrated
//!   quarticity, non-robust and jump-robust.
//! - [`jump_component`] — `max(RV - BV, 0)`, the jump contribution.
//! - [`parkinson`] / [`garman_klass`] — range-based variance from OHLC bars
//!   (Parkinson 1980; Garman & Klass 1980).
//! - [`bns_jump_ratio`] — the studentized `(RV-BV)/RV` ratio jump
//!   diagnostic (Barndorff-Nielsen & Shephard 2004; Huang & Tauchen 2005).
//!
//! ## HAR-RV (on a daily realized-variance series)
//!
//! [`har_rv`] regresses `RV_t` on a constant and its lagged daily / weekly
//! (5) / monthly (22) averages with HAC standard errors delegated to
//! [`tsecon_hac`]; [`HarVariant`] selects the level, log, or sqrt form and
//! [`HarConfig`] the burn-in, HAC lags, and small-sample correction.
//!
//! ```
//! use tsecon_realized::{realized_variance, bipower_variation, jump_component};
//!
//! let r = [0.5, -0.3, 0.8, -1.2, 0.1, 0.4, -0.6];
//! let rv = realized_variance(&r).unwrap();
//! let bv = bipower_variation(&r).unwrap();
//! assert!((rv - 2.95).abs() < 1e-12);
//! assert!(jump_component(&r).unwrap() >= 0.0);
//! assert!(rv >= 0.0 && bv >= 0.0);
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod error;
mod har;
mod jump;
mod measures;

pub use error::RealizedError;
pub use har::{har_rv, HarConfig, HarFit, HarVariant};
pub use jump::bns_jump_ratio;
pub use measures::{
    bipower_variation, garman_klass, jump_component, parkinson, realized_quarticity,
    realized_variance, tripower_quarticity,
};
