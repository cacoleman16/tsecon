//! # tsecon-survey — survey-expectations toolkit (roadmap E6)
//!
//! Tools for testing how professional forecasters form expectations, all built
//! on the library's single OLS + HAC engine ([`tsecon_hac`]) and its
//! chi-square/normal tails ([`tsecon_stats`]). Three capabilities:
//!
//! ## Module map
//!
//! - [`cg_regression`] / [`CgRegression`] and [`cg_series`]: the
//!   **Coibion-Gorodnichenko (2015)** information-rigidity regression of the
//!   mean forecast ERROR on the mean forecast REVISION. The slope `beta`
//!   measures information rigidity (`0` under full-information rational
//!   expectations, positive under sticky/noisy information) and maps to the
//!   implied degree of rigidity `beta / (1 + beta)`. Reports the intercept,
//!   slope, HAC standard errors, t-statistics, normal p-values, centered
//!   R-squared, and the implied rigidity. [`cg_series`] builds the aligned
//!   error/revision pair from a fixed-horizon mean-forecast series and the
//!   realized actual.
//! - [`disagreement`] / [`Disagreement`]: forecast **disagreement** — the
//!   per-period cross-sectional dispersion (standard deviation, quartiles, and
//!   inter-quartile range) of a (possibly ragged) forecaster panel, reproduced
//!   bit-for-bit against numpy `np.std` / `np.percentile`.
//! - [`efficiency_test`] / [`EfficiencyTest`]: the **Mincer-Zarnowitz**
//!   forecast-efficiency / rationality test in error-on-forecast form —
//!   regress the forecast error on a constant and the forecast (or any
//!   predetermined signals) and jointly test that all coefficients are zero
//!   via a HAC Wald statistic `W = b' V^{-1} b ~ chi-square(k)`.
//!
//! The lag-truncation of the Bartlett/Newey-West covariance is chosen with
//! [`HacBandwidth`] (an explicit `maxlags`, or the `floor(4 (n/100)^(2/9))`
//! rule of thumb).
//!
//! ## Validation
//!
//! The two regression estimators are pinned to an **independent** statsmodels
//! reference (`OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L,
//! "use_correction": ...}, use_t=False)`) and the disagreement measures to
//! numpy, both to ~1e-8 (`fixtures/tsecon-survey.json`, generated offline by
//! `fixtures/generate_survey_fixtures.py`). The two derived scalars
//! statsmodels does not report — the implied rigidity `beta/(1+beta)` and the
//! IQR `P75 - P25` — are documented closed forms.
//!
//! ```
//! use tsecon_survey::{cg_regression, efficiency_test, disagreement, HacBandwidth};
//!
//! // CG regression on aligned mean-forecast errors and revisions.
//! let errors = vec![0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.5, 0.1, -0.3, 0.2, 0.1, -0.1];
//! let revisions = vec![0.1, -0.2, 0.3, 0.1, 0.2, -0.1, 0.4, 0.0, -0.2, 0.1, 0.0, -0.1];
//! let cg = cg_regression(&errors, &revisions, HacBandwidth::Auto, true).unwrap();
//! assert!(cg.slope.is_finite() && (0.0..=1.0).contains(&cg.r_squared));
//!
//! // Disagreement of a small forecaster panel (two periods).
//! let panel = vec![vec![1.0, 1.5, 2.0, 2.5], vec![0.5, 1.0, 3.0]];
//! let d = disagreement(&panel, 0).unwrap();
//! assert_eq!(d.iqr.len(), 2);
//!
//! // Mincer-Zarnowitz efficiency test: error on the forecast.
//! let forecast = vec![1.0, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1.0, 0.6, 1.2, 1.0, 0.9];
//! let eff = efficiency_test(&errors, &[forecast], HacBandwidth::Auto, true).unwrap();
//! assert!(eff.wald >= 0.0 && (0.0..=1.0).contains(&eff.wald_pvalue));
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod cg;
mod common;
mod disagreement;
mod efficiency;
mod error;

pub use cg::{cg_regression, cg_series, CgRegression};
pub use common::HacBandwidth;
pub use disagreement::{disagreement, Disagreement};
pub use efficiency::{efficiency_test, EfficiencyTest};
pub use error::SurveyError;
