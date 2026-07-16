//! The benchmark zoo: naive, seasonal naive, drift, and historical mean,
//! with analytic ("naive") normal prediction intervals.
//!
//! These are the mandatory baselines that anchor every skill score.
//! Point forecasts and interval standard errors follow Hyndman &
//! Athanasopoulos (2021, *Forecasting: Principles and Practice*, 3rd ed.,
//! §5.5). With `sigma_hat` the residual standard deviation (below) and
//! `T` the sample size, the h-step forecast standard errors are
//!
//! | method         | point forecast                        | `sigma_h`                              |
//! |----------------|---------------------------------------|----------------------------------------|
//! | mean           | `ybar`                                | `sigma_hat * sqrt(1 + 1/T)`            |
//! | naive          | `y_T`                                 | `sigma_hat * sqrt(h)`                  |
//! | seasonal naive | `y_{T+h-m(k+1)}`                      | `sigma_hat * sqrt(k + 1)`              |
//! | drift          | `y_T + h (y_T - y_1)/(T - 1)`         | `sigma_hat * sqrt(h (1 + h/(T - 1)))`  |
//!
//! where `m` is the seasonal period and `k = floor((h-1)/m)` is the number
//! of complete seasonal cycles in the forecast period before time `T + h`.
//! The drift multiplier grows like `h^2/(T-1)` inside the square root —
//! the classic widening that reflects the estimated slope's own sampling
//! error.
//!
//! `sigma_hat^2 = (1/(T - K - M)) * sum(residual_t^2)` where `K` is the
//! number of estimated parameters (0 for naive/seasonal naive, 1 for
//! drift and mean) and `M` the number of missing residuals (1, m, 1, 0
//! respectively) — the fpp3 convention. Residuals are the in-sample
//! one-step errors of each method.
//!
//! Intervals are `point ± z_{(1+level)/2} * sigma_h`, with the normal
//! quantile from `tsecon-stats` (Wichura AS241). They assume normal,
//! uncorrelated residuals — for the honest empirical alternative use the
//! bootstrap engine (TODO(phase0): wired in the backtesting slice).

use crate::error::ForecastError;
use crate::validate::{check_level, check_series, check_steps};
use tsecon_stats::special::inv_norm_cdf;

/// A benchmark point forecast with a naive normal prediction interval.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkForecast {
    /// Point forecasts for horizons `1..=steps`.
    pub mean: Vec<f64>,
    /// Lower interval bounds, `mean - z * sigma_h`.
    pub lower: Vec<f64>,
    /// Upper interval bounds, `mean + z * sigma_h`.
    pub upper: Vec<f64>,
    /// The h-step forecast standard errors `sigma_h`.
    pub sigma_h: Vec<f64>,
    /// The interval coverage level, e.g. `0.95`.
    pub level: f64,
}

/// Assemble a [`BenchmarkForecast`] from points and standard errors.
fn with_interval(
    mean: Vec<f64>,
    sigma_h: Vec<f64>,
    level: f64,
) -> Result<BenchmarkForecast, ForecastError> {
    let z = inv_norm_cdf(0.5 + 0.5 * level)?;
    let lower = mean
        .iter()
        .zip(sigma_h.iter())
        .map(|(&m, &s)| m - z * s)
        .collect();
    let upper = mean
        .iter()
        .zip(sigma_h.iter())
        .map(|(&m, &s)| m + z * s)
        .collect();
    Ok(BenchmarkForecast {
        mean,
        lower,
        upper,
        sigma_h,
        level,
    })
}

/// Naive (random walk) forecast: `yhat_{T+h} = y_T` for all `h`.
///
/// Residuals are the first differences `y_t - y_{t-1}` (T-1 of them, no
/// estimated parameters), and `sigma_h = sigma_hat * sqrt(h)` — the
/// random-walk variance grows linearly in the horizon. Reference:
/// Hyndman & Athanasopoulos (2021, §5.5); see also the Atkeson & Ohanian
/// (2001) case for random-walk benchmarks in macro forecasting.
///
/// # Errors
///
/// [`ForecastError::SeriesTooShort`] (`n < 2`),
/// [`ForecastError::NonFinite`], [`ForecastError::InvalidSteps`], or
/// [`ForecastError::InvalidLevel`].
pub fn naive(y: &[f64], steps: usize, level: f64) -> Result<BenchmarkForecast, ForecastError> {
    check_steps(steps)?;
    check_level(level)?;
    let n = check_series(y, 2, "naive benchmark")?;
    let last = y[n - 1];
    let sse: f64 = y.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let sigma = (sse / (n - 1) as f64).sqrt();
    let mean = vec![last; steps];
    let sigma_h = (1..=steps).map(|h| sigma * (h as f64).sqrt()).collect();
    with_interval(mean, sigma_h, level)
}

/// Seasonal naive forecast: `yhat_{T+h} = y_{T+h-m(k+1)}` with
/// `k = floor((h-1)/m)` — each horizon repeats the most recent observed
/// value of the same season.
///
/// Residuals are the seasonal differences `y_t - y_{t-m}` (T-m of them,
/// no estimated parameters), and `sigma_h = sigma_hat * sqrt(k + 1)`:
/// the standard error steps up once per completed seasonal cycle.
/// Reference: Hyndman & Athanasopoulos (2021, §5.5).
///
/// # Errors
///
/// [`ForecastError::InvalidPeriod`] (`period = 0` or `n < period + 1`),
/// plus the shared validation errors of [`naive`].
pub fn seasonal_naive(
    y: &[f64],
    period: usize,
    steps: usize,
    level: f64,
) -> Result<BenchmarkForecast, ForecastError> {
    const WHAT: &str = "seasonal-naive benchmark";
    check_steps(steps)?;
    check_level(level)?;
    if period == 0 || y.len() < period + 1 {
        return Err(ForecastError::InvalidPeriod {
            what: WHAT,
            period,
            n: y.len(),
            requirement:
                "1 <= period <= n - 1 (need at least one full season plus one observation)",
        });
    }
    let n = check_series(y, period + 1, WHAT)?;
    let sse: f64 = (period..n).map(|t| (y[t] - y[t - period]).powi(2)).sum();
    let sigma = (sse / (n - period) as f64).sqrt();
    let mut mean = Vec::with_capacity(steps);
    let mut sigma_h = Vec::with_capacity(steps);
    for h in 1..=steps {
        let k = (h - 1) / period;
        mean.push(y[n - period + ((h - 1) % period)]);
        sigma_h.push(sigma * ((k + 1) as f64).sqrt());
    }
    with_interval(mean, sigma_h, level)
}

/// Drift forecast: the line through the first and last observations,
/// extrapolated — `yhat_{T+h} = y_T + h * (y_T - y_1) / (T - 1)`.
///
/// Equivalent to a random walk with the drift estimated as the mean first
/// difference `(y_T - y_1)/(T - 1)`. Residuals are
/// `y_t - y_{t-1} - drift` (T-1 of them, one estimated parameter, so the
/// variance divides by T-2), and
/// `sigma_h = sigma_hat * sqrt(h * (1 + h/(T - 1)))` — the classic drift
/// widening: on top of the random-walk `sqrt(h)` term, the estimated
/// slope's sampling error compounds quadratically with the horizon.
/// Reference: Hyndman & Athanasopoulos (2021, §5.5).
///
/// # Errors
///
/// [`ForecastError::SeriesTooShort`] (`n < 3`), plus the shared
/// validation errors of [`naive`].
pub fn drift(y: &[f64], steps: usize, level: f64) -> Result<BenchmarkForecast, ForecastError> {
    check_steps(steps)?;
    check_level(level)?;
    let n = check_series(y, 3, "drift benchmark")?;
    let nf = n as f64;
    let slope = (y[n - 1] - y[0]) / (nf - 1.0);
    let sse: f64 = y.windows(2).map(|w| (w[1] - w[0] - slope).powi(2)).sum();
    let sigma = (sse / (nf - 2.0)).sqrt();
    let mut mean = Vec::with_capacity(steps);
    let mut sigma_h = Vec::with_capacity(steps);
    for h in 1..=steps {
        let hf = h as f64;
        mean.push(y[n - 1] + hf * slope);
        sigma_h.push(sigma * (hf * (1.0 + hf / (nf - 1.0))).sqrt());
    }
    with_interval(mean, sigma_h, level)
}

/// Historical-mean forecast: `yhat_{T+h} = ybar` for all `h`.
///
/// Residuals are the deviations `y_t - ybar` (T of them, one estimated
/// parameter, so the variance divides by T-1), and
/// `sigma_h = sigma_hat * sqrt(1 + 1/T)` — constant in the horizon, since
/// an i.i.d.-mean model has no dynamics; the `1/T` term is the sampling
/// error of the mean itself. Reference: Hyndman & Athanasopoulos (2021,
/// §5.5).
///
/// # Errors
///
/// [`ForecastError::SeriesTooShort`] (`n < 2`), plus the shared
/// validation errors of [`naive`].
pub fn historical_mean(
    y: &[f64],
    steps: usize,
    level: f64,
) -> Result<BenchmarkForecast, ForecastError> {
    check_steps(steps)?;
    check_level(level)?;
    let n = check_series(y, 2, "historical-mean benchmark")?;
    let nf = n as f64;
    let ybar = y.iter().sum::<f64>() / nf;
    let sse: f64 = y.iter().map(|&v| (v - ybar).powi(2)).sum();
    let sigma = (sse / (nf - 1.0)).sqrt();
    let s_h = sigma * (1.0 + 1.0 / nf).sqrt();
    with_interval(vec![ybar; steps], vec![s_h; steps], level)
}
