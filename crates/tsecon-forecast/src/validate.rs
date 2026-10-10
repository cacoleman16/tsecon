//! Shared input validation for the forecast-evaluation tools.

use crate::error::ForecastError;

/// Validate that every value in `y` is finite.
pub(crate) fn check_finite(y: &[f64], what: &'static str) -> Result<(), ForecastError> {
    for (index, &value) in y.iter().enumerate() {
        if !value.is_finite() {
            return Err(ForecastError::NonFinite { what, index, value });
        }
    }
    Ok(())
}

/// Validate a series: every value finite and at least `min_n` observations.
/// Returns the length on success.
pub(crate) fn check_series(
    y: &[f64],
    min_n: usize,
    what: &'static str,
) -> Result<usize, ForecastError> {
    check_finite(y, what)?;
    if y.len() < min_n {
        return Err(ForecastError::SeriesTooShort {
            what,
            n: y.len(),
            needed: min_n,
        });
    }
    Ok(y.len())
}

/// Validate an (actual, forecast) pair: equal nonzero lengths, all finite.
/// Returns the common length.
pub(crate) fn check_pair(
    actual: &[f64],
    forecast: &[f64],
    what: &'static str,
) -> Result<usize, ForecastError> {
    if actual.len() != forecast.len() {
        return Err(ForecastError::LengthMismatch {
            what,
            expected: actual.len(),
            actual: forecast.len(),
        });
    }
    if actual.is_empty() {
        return Err(ForecastError::SeriesTooShort {
            what,
            n: 0,
            needed: 1,
        });
    }
    check_finite(actual, what)?;
    check_finite(forecast, what)?;
    Ok(actual.len())
}

/// Validate a prediction-interval coverage level: strictly inside (0, 1).
pub(crate) fn check_level(level: f64) -> Result<(), ForecastError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(ForecastError::InvalidLevel { level });
    }
    Ok(())
}

/// Largest forecast horizon the point forecasters and conformal methods of
/// this crate accept (1,000,000 steps, the cap `arima_fit` and `ets_fit` put
/// on theirs) — a guard against absurd allocations, not a modelling limit:
/// every step is materialised, so a typo such as `2**31` used to request
/// 16 GiB and abort the process on the failed allocation.
pub const MAX_FORECAST_STEPS: usize = 1_000_000;

/// Validate a forecast step count: a positive integer no larger than
/// [`MAX_FORECAST_STEPS`]. `name` is the caller's keyword for it (`steps`,
/// `horizon`) and `what` the computation, for the refusal.
pub(crate) fn check_steps(
    steps: usize,
    name: &'static str,
    what: &'static str,
) -> Result<(), ForecastError> {
    if steps == 0 {
        return Err(ForecastError::InvalidSteps { steps });
    }
    if steps > MAX_FORECAST_STEPS {
        return Err(ForecastError::TooLarge {
            what,
            name,
            value: steps,
            requirement: format!(
                "every forecast step is materialised, so the horizon is capped at \
                 {MAX_FORECAST_STEPS} steps (as in arima_fit and ets_fit)"
            ),
            max: Some(MAX_FORECAST_STEPS),
        });
    }
    Ok(())
}
