//! # tsecon-spectest — specification and diagnostic tests
//!
//! Classical model-specification and diagnostic tests, every one built on an
//! ordinary-least-squares regression fitted by the library's single OLS owner
//! [`tsecon_hac::ols`]. This is the roadmap E9 crate. Distribution tails come
//! from [`tsecon_stats`] (chi-square directly; the F tail via the regularized
//! incomplete beta function), and the CUSUM recursion factors its expanding
//! windows with `faer` through `tsecon-linalg`. This crate reimplements none
//! of them.
//!
//! ## The tests
//!
//! **White (1980) heteroskedasticity test.** Regress the squared OLS residuals
//! on the design's columns, their squares, and all pairwise cross-products;
//! the statistic is `n * R^2 ~ chi2(m - 1)`, with `m = k(k+1)/2` auxiliary
//! regressors. See [`white_test`] / [`HetTest`].
//!
//! **Breusch-Pagan (1979), Koenker (1981) studentized.** Regress the squared
//! residuals on the design; `LM = n * R^2 ~ chi2(k - 1)`. See
//! [`breusch_pagan_test`] / [`HetTest`].
//!
//! **Ramsey (1969) RESET.** Refit `y` on `[X, yhat^2, ..., yhat^max_power]`
//! and F-test the joint significance of the added powers. See [`reset_test`] /
//! [`FTest`].
//!
//! **Chow (1960) structural break at a known split.**
//! `F = [(SSR_pooled - SSR_1 - SSR_2)/k] / [(SSR_1 + SSR_2)/(n - 2k)]
//! ~ F(k, n - 2k)`. See [`chow_test`] / [`ChowTest`].
//!
//! **CUSUM parameter stability (Brown-Durbin-Evans 1975).** The standardized
//! cumulative sum of recursive residuals, with the documented `a = 0.948` 5%
//! boundary lines. See [`cusum_test`] / [`CusumTest`] / [`CUSUM_A_5PCT`].
//!
//! ## Module map
//!
//! - [`white_test`] / [`HetTest`] — White's test.
//! - [`breusch_pagan_test`] / [`HetTest`] — the Koenker-studentized
//!   Breusch-Pagan test.
//! - [`reset_test`] / [`FTest`] — the Ramsey RESET test.
//! - [`chow_test`] / [`ChowTest`] — the Chow known-split break test.
//! - [`cusum_test`] / [`CusumTest`] / [`CUSUM_A_5PCT`] — the CUSUM stability
//!   test.
//! - [`SpecTestError`] — the crate's error type ("errors that teach").
//!
//! ## Validation
//!
//! The golden fixture `fixtures/tsecon-spectest.json`
//! (`fixtures/generate_tsecon-spectest_fixtures.py`) is an INDEPENDENT
//! reference. The White, Breusch-Pagan, and RESET statistics and p-values are
//! pinned to statsmodels' `het_white`, `het_breuschpagan(robust=True)`, and
//! `linear_reset(power=3, use_f=True)`; the Chow statistic is assembled from
//! statsmodels OLS residual sums of squares with a `scipy.stats.f` p-value;
//! and the CUSUM path, bounds, and `sigma` are a documented-formula golden
//! evaluated with plain numpy (recursive residuals by refitting each expanding
//! window — a different code path from this crate's incremental recursion).
//! The reference match is verified to `~1e-8` in `tests/golden.rs`;
//! input-validation errors are covered in `tests/validation.rs`.
//!
//! ```
//! use tsecon_spectest::{breusch_pagan_test, white_test};
//!
//! // A short homoskedastic regression: const + one regressor.
//! let x1: Vec<f64> = (0..40).map(|t| (0.4 * t as f64).sin()).collect();
//! let cst = vec![1.0; 40];
//! let y: Vec<f64> = x1
//!     .iter()
//!     .enumerate()
//!     .map(|(t, v)| 0.5 + 0.8 * v + 0.1 * (t % 3) as f64)
//!     .collect();
//!
//! let white = white_test(&y, &[cst.clone(), x1.clone()]).unwrap();
//! let bp = breusch_pagan_test(&y, &[cst, x1]).unwrap();
//! assert!(white.pvalue.is_finite() && (0.0..=1.0).contains(&white.pvalue));
//! assert!(bp.pvalue.is_finite() && (0.0..=1.0).contains(&bp.pvalue));
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod breusch_pagan;
mod chow;
mod common;
mod cusum;
mod error;
mod reset;
mod results;
mod white;

pub use breusch_pagan::breusch_pagan_test;
pub use chow::{chow_test, ChowTest};
pub use cusum::{cusum_test, CusumTest, CUSUM_A_5PCT};
pub use error::SpecTestError;
pub use reset::reset_test;
pub use results::{FTest, HetTest};
pub use white::white_test;
