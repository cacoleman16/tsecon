//! Error types for `tsecon-diag`.
//!
//! Every fallible entry point in this crate returns `Result<_, DiagError>`;
//! nothing in the non-test code path panics. Error messages follow the
//! library's "errors that teach" pillar: they state what went wrong, why it
//! matters statistically, and what the caller can do about it.

use core::fmt;

use tsecon_stats::StatsError;

/// Errors produced by the diagnostic statistics in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagError {
    /// The series has too few observations for the requested computation.
    SeriesTooShort {
        /// Which diagnostic needed more data.
        what: &'static str,
        /// The number of observations supplied.
        n: usize,
        /// The minimum number of observations required.
        needed: usize,
    },
    /// The requested number of lags is outside the valid range for a series
    /// of this length.
    InvalidLags {
        /// Which diagnostic rejected the lag count.
        what: &'static str,
        /// The lag count that was supplied.
        nlags: usize,
        /// The number of observations supplied.
        n: usize,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The input contains a NaN or infinite value. Diagnostics never skip
    /// missing values silently; clean or impute the series first.
    NonFinite {
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The series is (numerically) constant, so its sample variance is zero
    /// and correlation-based diagnostics are undefined.
    ConstantSeries {
        /// Which diagnostic found the degenerate series.
        what: &'static str,
    },
    /// The regressor cross-product matrix of an internal OLS step is not
    /// positive definite (collinear or degenerate lag matrix).
    SingularDesign {
        /// Which diagnostic hit the singular design.
        what: &'static str,
    },
    /// A numerical invariant that holds in exact arithmetic (e.g. positive
    /// innovation variance in the Durbin-Levinson recursion) broke down.
    NumericalBreakdown {
        /// Which algorithm broke down.
        what: &'static str,
    },
    /// The significance level `alpha` passed to a report is outside (0, 1).
    InvalidAlpha {
        /// The offending value.
        value: f64,
    },
    /// An error propagated from the `tsecon-stats` special functions (e.g.
    /// the chi-squared survival function used for p-values).
    Stats(StatsError),
}

impl fmt::Display for DiagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagError::SeriesTooShort { what, n, needed } => write!(
                f,
                "{what}: series has {n} observations but needs at least {needed}; \
                 supply more data or reduce the requested lag order"
            ),
            DiagError::InvalidLags {
                what,
                nlags,
                n,
                requirement,
            } => write!(
                f,
                "{what}: nlags = {nlags} is invalid for a series of length {n}: \
                 requires {requirement}"
            ),
            DiagError::NonFinite { index, value } => write!(
                f,
                "input contains a non-finite value ({value}) at index {index}; \
                 diagnostics do not skip missing values silently — drop or impute \
                 NaN/inf observations before testing"
            ),
            DiagError::ConstantSeries { what } => write!(
                f,
                "{what}: the series is constant (zero sample variance), so \
                 correlation-based diagnostics are undefined; check that the \
                 right column was passed and that differencing did not remove \
                 all variation"
            ),
            DiagError::SingularDesign { what } => write!(
                f,
                "{what}: the lag regressor matrix is numerically singular \
                 (collinear lags); this usually means the series is (near-)\
                 deterministic or far too short for the requested lag order"
            ),
            DiagError::NumericalBreakdown { what } => write!(
                f,
                "{what}: numerical breakdown — an invariant that holds in exact \
                 arithmetic failed; this indicates a (near-)degenerate series"
            ),
            DiagError::InvalidAlpha { value } => write!(
                f,
                "significance level alpha = {value} is invalid: requires \
                 0 < alpha < 1 (conventional choices are 0.01, 0.05, 0.10)"
            ),
            DiagError::Stats(e) => write!(f, "special-function error: {e}"),
        }
    }
}

impl std::error::Error for DiagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DiagError::Stats(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StatsError> for DiagError {
    fn from(e: StatsError) -> Self {
        DiagError::Stats(e)
    }
}
