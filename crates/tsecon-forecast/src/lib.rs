//! # tsecon-forecast
//!
//! First slice of the forecasting-evaluation module for the `tsecon`
//! time-series econometrics library (see ROADMAP §09): point accuracy
//! measures, the Diebold-Mariano comparison test, and the benchmark zoo.
//!
//! * [`accuracy`] — the loss battery with guardrails: [`me`], [`mse`],
//!   [`rmse`], [`mae`], [`mdae`]; the percentage errors [`mape`] (hard
//!   error on zero actuals) and [`smape`] (M4 definition,
//!   `200|e|/(|y|+|yhat|)`, hard error on zero denominators; Goodwin &
//!   Lawton 1999); and the scaled errors [`mase`] and [`rmsse`]
//!   (Hyndman & Koehler 2006) normalized by the *training-sample*
//!   seasonal-naive MAE/MSE at a caller-chosen period.
//! * [`dm`] — the Diebold-Mariano (1995) test of equal predictive
//!   accuracy with squared, absolute, or custom loss, an h-step
//!   uniform-weight truncated long-run variance, and the Harvey,
//!   Leybourne & Newbold (1997) small-sample correction with t(n-1)
//!   p-values as the default: [`dm_test`] / [`dm_test_with_loss`].
//! * [`benchmarks`] — the mandatory baselines with analytic normal
//!   prediction intervals per Hyndman & Athanasopoulos (2021, §5.5):
//!   [`naive`], [`seasonal_naive`], [`drift`] (with the classic
//!   quadratic-in-horizon interval widening), and [`historical_mean`].
//! * [`theta`] — the Theta method (Assimakopoulos & Nikolopoulos 2000)
//!   via the Hyndman & Billah (2003) SES-with-drift equivalence, with
//!   classical multiplicative deseasonalization for seasonal data:
//!   [`theta_forecast`] / [`theta_forecast_with`].
//! * [`comparison`] — the [`ForecastComparison`] report: accuracy table
//!   plus pairwise DM tests plus a teaching interpretation string.
//!
//! Inputs follow the library-wide missing-data policy: NaN or infinite
//! values are a loud error, never silently skipped. Golden-value tests
//! pin the DM statistics to 1e-10 and the Theta forecast to 1e-6 relative
//! against statsmodels 0.14.6 fixtures.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod accuracy;
pub mod benchmarks;
pub mod comparison;
pub mod dm;
mod error;
pub mod theta;
mod validate;

pub use accuracy::{mae, mape, mase, mdae, me, mse, rmse, rmsse, smape};
pub use benchmarks::{drift, historical_mean, naive, seasonal_naive, BenchmarkForecast};
pub use comparison::{AccuracyRow, DmPair, ForecastComparison};
pub use dm::{dm_test, dm_test_with_loss, DmLoss, DmResult};
pub use error::ForecastError;
pub use theta::{theta_forecast, theta_forecast_with, ThetaForecast};
