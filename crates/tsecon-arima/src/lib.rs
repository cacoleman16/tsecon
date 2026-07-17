//! # tsecon-arima
//!
//! ARMA/ARIMA estimation — the library's first full model class, built on
//! the shared linear-Gaussian state-space engine (`tsecon-ssm`) and the
//! optimization suite (`tsecon-optim`). Numeric conventions follow
//! statsmodels `SARIMAX`, and the golden fixture `fixtures/arima.json`
//! arbitrates:
//!
//! * [`ArimaSpec`] — the ARIMA(p, d, q) specification (optional constant;
//!   seasonal orders are `// TODO(phase0)` and slot into the same
//!   struct);
//! * [`arma_ssm`] — the Harvey (1989) / Jones (1980) canonical
//!   state-space form with state dimension `max(p, q + 1)`, stationary
//!   (discrete-Lyapunov) initialization, and the constant entering the
//!   state equation exactly as statsmodels `SARIMAX(trend='c')`;
//! * differencing (`d > 0`) is **simple differencing**: the data are
//!   differenced `d` times and the ARMA fits the differences, losing `d`
//!   observations (statsmodels `simple_differencing=True`); the levels
//!   state-space form is `// TODO(phase0)`;
//! * [`ArimaSpec::fit`] — exact Gaussian MLE: the Monahan (1984)
//!   stationarity transform for the AR block, its invertibility dual for
//!   the MA block, `exp` for `sigma2`, L-BFGS with central-difference
//!   gradients plus a Nelder-Mead polish/fallback, and Hannan-Rissanen
//!   (1982) starting values with a safe fallback;
//! * [`ArimaSpec::fit_css`] — conditional sum of squares, the fast
//!   alternative (equals exact MLE only asymptotically; documented on the
//!   method);
//! * [`ArimaSpec::loglike`] — the exact log-likelihood at fixed
//!   parameters (the golden-fixture entry point);
//! * [`ArimaResults`] — named parameters, log-likelihood, AIC/BIC with
//!   `sigma2` counted in `k` (statsmodels convention),
//!   [`ArimaResults::forecast`] via the state-space prediction recursion
//!   with exact re-cumulation to levels (correct cumulative variance)
//!   for `d > 0`, and [`ArimaResults::residuals`] — standardized
//!   one-step prediction errors from the Kalman filter.
//!
//! All fallible routines return [`ArimaError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod diff;
pub mod error;
mod estimate;
pub mod results;
pub mod spec;
pub mod ssm;

pub use error::ArimaError;
pub use results::{ArimaForecast, ArimaResults, EstimationMethod};
pub use spec::ArimaSpec;
pub use ssm::arma_ssm;
