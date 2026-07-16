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

/// Validate a forecast step count: must be a positive integer.
pub(crate) fn check_steps(steps: usize) -> Result<(), ForecastError> {
    if steps == 0 {
        return Err(ForecastError::InvalidSteps { steps });
    }
    Ok(())
}
