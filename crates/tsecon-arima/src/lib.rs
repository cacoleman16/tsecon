//! # tsecon-arima
//!
//! ARMA/ARIMA estimation — the library's first full model class, built on
//! the shared linear-Gaussian state-space engine (`tsecon-ssm`) and the
//! optimization suite (`tsecon-optim`). Numeric conventions follow
//! statsmodels `SARIMAX`, and the golden fixture `fixtures/arima.json`
//! arbitrates:
//!
//! * [`ArimaSpec`] — the ARIMA(p, d, q) specification (optional
//!   constant), with multiplicative seasonal orders `(P, D, Q)_s` added
//!   through [`ArimaSpec::seasonal`]: the seasonal and regular lag
//!   polynomials are multiplied into a single dense
//!   ARMA(p + s*P, q + s*Q) that runs through the same engine, golden-
//!   pinned to statsmodels `SARIMAX(seasonal_order=...)` on the airline
//!   model (`fixtures/sarima.json`);
//! * [`arma_ssm`] — the Harvey (1989) / Jones (1980) canonical
//!   state-space form with state dimension `max(p, q + 1)`, stationary
//!   (discrete-Lyapunov) initialization, and the constant entering the
//!   state equation exactly as statsmodels `SARIMAX(trend='c')`;
//! * differencing (`d > 0`, `D > 0`) is **simple differencing**: the
//!   data are seasonally differenced `D` times and then differenced `d`
//!   times, and the ARMA fits the differences, losing `d + D*s`
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
//!   one-step prediction errors from the Kalman filter;
//! * [`ArimaResults::param_cov`] / [`ArimaResults::bse`] — the observed
//!   information (inverse negative numerical Hessian of the
//!   log-likelihood) and the parameter standard errors on its diagonal,
//!   matching statsmodels `fit(cov_type='approx').bse`. `sigma2` is
//!   differentiated multiplicatively, so the standard errors do not
//!   depend on the units of the series; the inversion refuses a
//!   numerically rank-deficient information matrix rather than returning
//!   the pseudo-inverse statsmodels would, and [`ParamCov::rcond`]
//!   reports how much conditioning was left;
//! * [`ArimaResults::forecast_with`] — forecasts that optionally add the
//!   estimated constant's delta-method contribution to the band width.
//!   [`ArimaResults::forecast`] deliberately keeps the
//!   parameters-treated-as-known convention that statsmodels
//!   `get_forecast` uses, so the parity gate keeps its meaning; the
//!   opt-in is where the honest drift term lives;
//! * [`auto_arima`] — Hyndman-Khandakar (2008) automatic order
//!   selection: `d` from successive KPSS tests (`tsecon-diag::ndiffs`),
//!   `D` from the seasonal-strength rule (`tsecon-diag::nsdiffs`), then
//!   the stepwise AICc search (AIC/BIC selectable; full-grid optional)
//!   with unit-circle admissibility guards, every candidate fit by
//!   [`ArimaSpec::fit`] and the full search trace returned. Graded by
//!   Monte-Carlo order recovery plus candidate-level statsmodels pins —
//!   the selection loop itself deliberately has no R/pmdarima parity
//!   gate (see the module docs of [`auto`]);
//! * [`bn_decomposition`] / [`bn_from_arma`] — the classic
//!   Beveridge-Nelson (1981) trend-cycle decomposition from an
//!   ARIMA(p, 1, q): closed-form long-run multiplier
//!   `psi(1) = theta(1)/phi(1)`, random-walk-with-drift trend
//!   `Delta tau_t = mu + psi(1) eps_t`, companion-form cycle (Morley
//!   2002), with exact finite-sample identities (see the module docs of
//!   [`bn`]).
//!
//! All fallible routines return [`ArimaError`]; nothing in this crate
//! panics on user input.

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod auto;
pub mod bn;
pub mod cov;
mod diff;
pub mod error;
mod estimate;
pub mod results;
pub mod spec;
pub mod ssm;

pub use auto::{
    auto_arima, AutoArimaCandidate, AutoArimaOptions, AutoArimaResult, CandidateStatus, SelectionIc,
};
pub use bn::{bn_decomposition, bn_from_arma, BnDecomposition};
pub use cov::ParamCov;
pub use error::ArimaError;
pub use results::{ArimaForecast, ArimaResults, EstimationMethod, ForecastOptions};
pub use spec::ArimaSpec;
pub use ssm::arma_ssm;
