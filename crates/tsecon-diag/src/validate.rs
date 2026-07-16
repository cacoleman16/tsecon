//! Shared input validation for the diagnostics.

use crate::error::DiagError;

/// Validate a series: every value finite, at least `min_n` observations.
/// Returns the length on success.
pub(crate) fn check_series(
    y: &[f64],
    min_n: usize,
    what: &'static str,
) -> Result<usize, DiagError> {
    for (index, &value) in y.iter().enumerate() {
        if !value.is_finite() {
            return Err(DiagError::NonFinite { index, value });
        }
    }
    if y.len() < min_n {
        return Err(DiagError::SeriesTooShort {
            what,
            n: y.len(),
            needed: min_n,
        });
    }
    Ok(y.len())
}
