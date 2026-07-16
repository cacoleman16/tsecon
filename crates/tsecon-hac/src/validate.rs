//! Shared input validation for the estimators in this crate.

use crate::error::HacError;

/// Reject NaN/inf observations, reporting the first offender.
pub(crate) fn check_finite(x: &[f64], what: &'static str) -> Result<(), HacError> {
    for (index, &value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(HacError::NonFinite { what, index, value });
        }
    }
    Ok(())
}

/// Require at least `needed` observations.
pub(crate) fn check_min_len(x: &[f64], needed: usize, what: &'static str) -> Result<(), HacError> {
    if x.len() < needed {
        return Err(HacError::SeriesTooShort {
            what,
            n: x.len(),
            needed,
        });
    }
    Ok(())
}

/// Require a finite, non-negative bandwidth.
pub(crate) fn check_bandwidth(bandwidth: f64) -> Result<(), HacError> {
    if !bandwidth.is_finite() || bandwidth < 0.0 {
        return Err(HacError::InvalidBandwidth { value: bandwidth });
    }
    Ok(())
}
