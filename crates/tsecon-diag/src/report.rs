//! The [`DiagnosticReport`] bundle: one statistic, its p-value, and a
//! human-readable interpretation that teaches the next step.

use core::fmt;

use crate::error::DiagError;

/// A single diagnostic test summarized for humans.
///
/// Produced by the `report(alpha)` methods on
/// [`crate::PortmanteauResult`], [`crate::JarqueBeraResult`], and
/// [`crate::ArchLmResult`]. The `interpretation` string implements the
/// library's "errors that teach" pillar: it states the decision in plain
/// language and points at the next diagnostic or modeling step.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticReport {
    /// Short test name, e.g. `"Ljung-Box Q(10)"`.
    pub test: String,
    /// The test statistic.
    pub statistic: f64,
    /// The p-value of the statistic under the null.
    pub p_value: f64,
    /// Degrees of freedom of the reference chi-squared distribution.
    pub df: usize,
    /// The significance level the decision was taken at.
    pub alpha: f64,
    /// Whether the null is rejected at level `alpha` (`p_value < alpha`).
    pub reject: bool,
    /// Plain-language interpretation of the outcome, including what to
    /// check next.
    pub interpretation: String,
}

impl fmt::Display for DiagnosticReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: stat = {:.4}, df = {}, p = {:.4} [{}] — {}",
            self.test,
            self.statistic,
            self.df,
            self.p_value,
            if self.reject {
                "reject null"
            } else {
                "fail to reject"
            },
            self.interpretation
        )
    }
}

/// Validate a significance level: must lie strictly inside (0, 1).
pub(crate) fn check_alpha(alpha: f64) -> Result<(), DiagError> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(DiagError::InvalidAlpha { value: alpha });
    }
    Ok(())
}
