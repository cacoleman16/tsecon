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
//!   plus pairwise DM tests plus a teaching interpretation string, and,
//!   when the models are declared nested, a Clark-West adjustment.
//! * [`backtest`] — the pseudo-out-of-sample [`Backtest`] engine over
//!   expanding and fixed-width rolling [`Window`] schemes, with a pinned
//!   `(origin, horizon, target)` alignment convention and an infrequent-refit
//!   contract; [`BacktestResult`] returns origin-aligned per-horizon
//!   forecasts, targets, errors, and an accuracy table.
//! * [`cw`] — the Clark-West (2007) adjusted-MSPE test [`cw_test`] for
//!   nested models, with a Bartlett long-run variance and a one-sided normal
//!   p-value.
//! * [`gw`] — the Giacomini-White (2006) equal-conditional-predictive-ability
//!   test: the unconditional [`gw_test`] (chi-squared, test function `h=1`)
//!   and the general q-dimensional Wald [`gw_test_conditional`].
//! * [`conformal`] — distribution-free conformal forecast intervals
//!   wrapping arbitrary point forecasters, with the backtest engine's
//!   leakage discipline: [`split_conformal`] (finite-sample-corrected
//!   residual-quantile calibration, symmetric and asymmetric),
//!   [`enbpi`] (Xu-Xie 2021 ensemble batch prediction intervals with
//!   bootstrap-aggregated leave-one-out residuals and a sliding window),
//!   and [`aci`] (Gibbs-Candès 2021 adaptive conformal inference,
//!   `alpha_{t+1} = alpha_t + gamma (alpha - err_t)`), plus the
//!   [`conformal_quantile`] primitive, the [`ar_forecast`] base learner,
//!   and online evaluators ([`split_conformal_online`], [`enbpi_online`]).
//! * [`var_backtest`] — the VaR backtest battery ([`var_backtest`] on
//!   returns + VaR forecasts with a documented sign convention,
//!   [`var_backtest_hits`] on a pre-computed 0/1 violation sequence):
//!   Kupiec (1995) unconditional coverage `LR_uc ~ chi^2(1)`,
//!   Christoffersen (1998) independence `LR_ind` and conditional coverage
//!   `LR_cc = LR_uc + LR_ind ~ chi^2(2)`, and the Engle-Manganelli (2004)
//!   dynamic quantile regression test `DQ ~ chi^2(k)`, with a teaching
//!   verdict string.
//!
//! Inputs follow the library-wide missing-data policy: NaN or infinite
//! values are a loud error, never silently skipped. All long-run variances
//! come from the single HAC engine (`tsecon-hac`), so identical settings
//! never yield different p-values across modules. Golden-value tests pin the
//! DM statistics to 1e-10 and the Theta forecast to 1e-6 relative against
//! statsmodels 0.14.6 fixtures, and the Clark-West / Giacomini-White
//! statistics against self-authored NumPy reference fixtures.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod accuracy;
pub mod backtest;
pub mod benchmarks;
pub mod comparison;
pub mod conformal;
pub mod cw;
pub mod dm;
mod error;
pub mod gw;
mod hac;
pub mod theta;
mod validate;
pub mod var_backtest;

pub use accuracy::{mae, mape, mase, mdae, me, mse, rmse, rmsse, smape};
pub use backtest::{Backtest, BacktestResult, Window};
pub use benchmarks::{drift, historical_mean, naive, seasonal_naive, BenchmarkForecast};
pub use comparison::{AccuracyRow, CwPair, DmPair, ForecastComparison};
pub use conformal::{
    aci, ar_forecast, conformal_quantile, enbpi, enbpi_online, split_conformal,
    split_conformal_online, AciForecast, AciOptions, ConformalOnline, EnbpiForecast, EnbpiOptions,
    SplitConformalForecast, SplitConformalOptions, SplitOnlineOptions,
};
pub use cw::{cw_test, CwResult};
pub use dm::{dm_test, dm_test_with_loss, DmLoss, DmResult};
pub use error::ForecastError;
pub use gw::{gw_test, gw_test_conditional, GwResult};
pub use theta::{theta_forecast, theta_forecast_with, ThetaForecast};
pub use var_backtest::{var_backtest, var_backtest_hits, VarBacktestResult};
