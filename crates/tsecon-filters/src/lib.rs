//! # tsecon-filters
//!
//! Trend-cycle decomposition filters for the `tsecon` time-series
//! econometrics library (diagnostics module; see ROADMAP §01 —
//! re-exported through `tsecon-diag`).
//!
//! Filters:
//!
//! * [`hp_filter`] — Hodrick-Prescott (1997) via an `O(n)` pentadiagonal
//!   `L D L'` solve, with [`ravn_uhlig_lambda`] frequency-rule defaults
//!   (Ravn-Uhlig 2002: 1600 quarterly, 6.25 annual, 129600 monthly) and
//!   the real-time one-sided variant [`hp_filter_one_sided`]
//!   (Stock-Watson 1999);
//! * [`bk_filter`] — Baxter-King (1999) symmetric truncated band-pass;
//!   loses `K` observations at each end and says so;
//! * [`cf_filter`] — Christiano-Fitzgerald (2003) asymmetric random-walk
//!   band-pass with optional drift adjustment; full sample retained;
//! * [`hamilton_filter`] — Hamilton (2018) regression filter (the
//!   recommended HP replacement), plus the
//!   [`hamilton_filter_random_walk`] special case
//!   (`cycle_t = y_t - y_{t-h}`).
//!
//! Every filter returns a [`Decomposition`] carrying explicit
//! [`Alignment`] metadata: filters that lose observations must say so —
//! silent misalignment is the deadliest bug class in applied macro (see
//! `docs/roadmap/00-architecture.md`).
//!
//! Accuracy: golden-value tests pin all four filters against
//! statsmodels 0.14.6 on the 100·log US real GDP series at `1e-8`
//! (`fixtures/filters.json`).
//!
//! ```
//! use tsecon_filters::{hp_filter, ravn_uhlig_lambda, Frequency};
//!
//! let y: Vec<f64> = (0..40).map(|t| t as f64 + (t as f64 * 0.7).sin()).collect();
//! let dec = hp_filter(&y, ravn_uhlig_lambda(Frequency::Quarterly)).unwrap();
//! let trend = dec.trend.unwrap();
//! assert_eq!(dec.alignment.first_index(), 0); // HP keeps the full sample
//! assert!((trend[5] + dec.cycle[5] - y[5]).abs() < 1e-12);
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod bandpass;
mod decomposition;
mod error;
mod hamilton;
mod hp;

pub use bandpass::{bk_filter, cf_filter};
pub use decomposition::{Alignment, Decomposition};
pub use error::FiltersError;
pub use hamilton::{
    hamilton_defaults, hamilton_filter, hamilton_filter_random_walk, HamiltonResult,
};
pub use hp::{hp_filter, hp_filter_one_sided, ravn_uhlig_lambda, Frequency};
