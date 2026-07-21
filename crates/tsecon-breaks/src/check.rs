//! Shared input validation for the break estimators.

use crate::error::BreaksError;

/// Validate `(y, x, trim)` and return `(T, q, h)` with `h = ceil(trim * T)`.
///
/// Checks, in order: non-empty inputs, matching column lengths, finite
/// values, `trim` in `(0, 0.5)`, and `h >= q + 1` so every regime's OLS
/// leaves at least one residual degree of freedom.
pub(crate) fn validate(
    y: &[f64],
    x: &[Vec<f64>],
    trim: f64,
) -> Result<(usize, usize, usize), BreaksError> {
    if y.is_empty() {
        return Err(BreaksError::EmptyInput { what: "y" });
    }
    if x.is_empty() {
        return Err(BreaksError::NoRegressors);
    }
    let t = y.len();
    let q = x.len();
    for col in x {
        if col.len() != t {
            return Err(BreaksError::DimensionMismatch {
                what: "every column of x must match the length of y",
                expected: t,
                got: col.len(),
            });
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(BreaksError::NonFinite { what: "y" });
    }
    if x.iter().any(|col| col.iter().any(|v| !v.is_finite())) {
        return Err(BreaksError::NonFinite { what: "x" });
    }
    if !(trim > 0.0 && trim < 0.5) {
        return Err(BreaksError::InvalidArgument {
            what: "trim must be strictly between 0 and 0.5 (the fraction of the \
                   sample reserved as the minimal regime length; 0.15 is the \
                   standard choice)",
        });
    }
    let h = (trim * t as f64).ceil() as usize;
    if h < q + 1 {
        return Err(BreaksError::TrimTooSmall { h, q, t });
    }
    Ok((t, q, h))
}
