//! Input validation and the one-period predictive alignment shared by every
//! estimator in the crate.
//!
//! The predictive regression pairs a predictor observed at time `t` with the
//! response realised at `t + 1`. Given length-`n` series `r` and `x` we form
//! `N = n - 1` aligned pairs
//!
//! ```text
//! predictor  a_t := x[t]      for t = 0 .. N-1
//! target     b_t := r[t+1]    for t = 0 .. N-1
//! ```

use crate::error::PredRegError;

/// Reject NaN/inf entries in a named series.
pub(crate) fn check_finite(v: &[f64], what: &'static str) -> Result<(), PredRegError> {
    if v.iter().any(|z| !z.is_finite()) {
        return Err(PredRegError::NonFinite { what });
    }
    Ok(())
}

/// Validate a `(r, x)` pair and return the aligned `(a, b)` predictor/target
/// slices `a = x[..n-1]`, `b = r[1..]`, each of length `N = n - 1`.
pub(crate) fn align_pair<'a>(
    r: &'a [f64],
    x: &'a [f64],
) -> Result<(&'a [f64], &'a [f64]), PredRegError> {
    if x.is_empty() {
        return Err(PredRegError::EmptyInput {
            what: "predictor x",
        });
    }
    if r.is_empty() {
        return Err(PredRegError::EmptyInput { what: "response r" });
    }
    if r.len() != x.len() {
        return Err(PredRegError::DimensionMismatch {
            what: "response r and predictor x",
            expected: x.len(),
            got: r.len(),
        });
    }
    check_finite(x, "predictor x")?;
    check_finite(r, "response r")?;
    let n = x.len();
    if n < 3 {
        return Err(PredRegError::DegreesOfFreedom { n: n - 1, k: 2 });
    }
    Ok((&x[..n - 1], &r[1..]))
}

/// Sample mean of a non-empty slice.
pub(crate) fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
