//! # tsecon-ets — innovations state-space exponential smoothing
//!
//! Roadmap Module 02 (Tier 1, "ETS state space (all error/trend/seasonal
//! combos) + AutoETS"): the exponential-smoothing family of Hyndman,
//! Koehler, Snyder & Grose (2002) and Hyndman, Koehler, Ord & Snyder
//! (2008), *Forecasting with Exponential Smoothing: The State Space
//! Approach* — the 30 models ETS(E, T, S) with additive (`A`) or
//! multiplicative (`M`) error, none / additive / multiplicative trend,
//! optionally damped (`Ad`, `Md`), and none / additive / multiplicative
//! seasonality. ETS(A,N,N) is simple exponential smoothing, ETS(A,A,N)
//! Holt's linear method, ETS(A,Ad,N) the damped trend, ETS(A,A,A) and
//! ETS(M,A,M) the additive and multiplicative Holt-Winters methods.
//!
//! ## The model
//!
//! Every member is an innovations state-space model: a single error
//! `e_t` drives the observation and every state. For ETS(A,Ad,A),
//!
//! ```text
//! y_t = l_{t-1} + phi b_{t-1} + s_{t-m} + e_t
//! l_t = l_{t-1} + phi b_{t-1} + alpha e_t
//! b_t = phi b_{t-1} + beta e_t
//! s_t = s_{t-m} + gamma e_t
//! ```
//!
//! and the other members replace `+` by `*` component by component
//! (Hyndman et al. 2008, Tables 2.2 and 2.3); [`filter`] gives the
//! recursion in R's `etscalc.c` arithmetic. The smoothing parameters are
//! Hyndman's `alpha`, `beta`, `gamma`, `phi` (not the `beta* = beta /
//! alpha`, `gamma* = gamma / (1 - alpha)` of the classical Holt-Winters
//! recursions), and the log-likelihood is the concentrated Gaussian one
//! of Ord, Koehler & Snyder (1997).
//!
//! ## What this crate provides
//!
//! - [`EtsSpec`] — one member of the taxonomy, with the inert-option
//!   refusals (`damped` without a trend, `seasonal_periods` without a
//!   seasonal) built into [`EtsSpec::new`].
//! - [`smooth`] / [`loglik`] — the recursion at fixed parameters and
//!   initial states: fitted values, residuals, state paths, likelihood.
//! - [`heuristic_initial_states`] — the Hyndman (2008, section 2.6.1)
//!   initialisation, exactly as statsmodels computes it.
//! - [`ets_fit`] — maximum likelihood over the smoothing parameters and
//!   (by default) the initial states; [`ets_at`] evaluates at fixed
//!   values.
//! - [`forecast`] — point forecasts, forecast variances and prediction
//!   intervals: the closed forms of Hyndman et al. (2008, Table 6.1) for
//!   the class-1 (linear, additive-error) models, seeded simulation for
//!   the others; [`forecast_from_state`], [`class1_forecast_variance`],
//!   [`simulate_paths`] expose the pieces.
//! - [`auto_ets`] — the information-criterion search over the taxonomy
//!   with R's admissibility restrictions.
//!
//! ## Validation (honest grade)
//!
//! Twenty of the thirty models — every one without a multiplicative
//! seasonal — are pinned at fixed parameters to statsmodels
//! `ETSModel.loglike` / `smooth` / `forecast` at 1e-10, the six class-1
//! forecast variances to statsmodels `get_prediction` and to the Table
//! 6.1 closed forms transcribed in the fixture generator, the heuristic
//! initialisation to `holtwinters.ExponentialSmoothing`, and the maximum
//! likelihood to statsmodels' fit within optimizer tolerance. For the ten
//! multiplicative-seasonal models statsmodels' Cython smoother updates the
//! seasonal with the *post-update* level and `gamma*` — the classical
//! Holt-Winters recursion, which coincides with the innovations form only
//! to first order in the error (its own `simulate` uses the innovations
//! form) — so those ten are pinned to an independent NumPy transcription
//! of the published recursion (and their simulator to statsmodels
//! `simulate` at given innovations); the measured statsmodels gap is
//! recorded, not hidden. The selection loop has no runnable third-party
//! reference (M3 parity is R-only) and is graded by seeded Monte-Carlo
//! recovery, as `auto_arima` was.
//!
//! ```
//! use tsecon_ets::{ets_fit, forecast, Component, ErrorType, EtsSpec, FitOptions};
//!
//! // A slowly drifting level with additive noise: Holt's linear method.
//! let y: Vec<f64> = (0..120)
//!     .map(|t| 10.0 + 0.1 * t as f64 + ((t * 7 % 11) as f64 - 5.0) * 0.2)
//!     .collect();
//! let spec = EtsSpec::new(ErrorType::Additive, Component::Additive, false, Component::None, None)
//!     .unwrap();
//! let fit = ets_fit(&spec, &y, &FitOptions::default()).unwrap();
//! let fc = forecast(&fit, 6, 0.95, 0, 0).unwrap();
//! assert_eq!(fc.mean.len(), 6);
//! assert!(fc.upper[5] - fc.lower[5] > fc.upper[0] - fc.lower[0]); // widening intervals
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod auto;
mod error;
pub mod filter;
mod fit;
mod forecast;
mod init;
mod spec;

pub use auto::{auto_ets, candidate_specs, AutoEtsOptions, AutoEtsResult, Candidate, Ic};
pub use error::EtsError;
pub use filter::{loglik, smooth, Smoothed};
pub use fit::{ets_at, ets_fit, EtsFit, FitOptions, Initialization, Optimizer};
pub use forecast::{
    class1_forecast_variance, forecast, forecast_from_state, simulate_paths, EtsForecast,
    IntervalMethod,
};
pub use init::{heuristic_initial_states, simple_initial_states};
pub use spec::{Component, ErrorType, EtsParams, EtsSpec, EtsStates};
