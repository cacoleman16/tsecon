//! Error type shared by the survey-expectations estimators.
//!
//! Every fallible public function in this crate returns
//! `Result<_, SurveyError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters, and what the caller can do.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_stats::StatsError;

/// Errors produced by the survey-expectations estimators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum SurveyError {
    /// A required series (or the panel) was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// Two series that must be the same length were not.
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The length that was expected.
        expected: usize,
        /// The length that was received.
        got: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A configuration scalar was outside its valid domain (e.g. a negative
    /// forecast horizon, or a degrees-of-freedom correction `ddof` that
    /// equals or exceeds the cross-section size for some period).
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// The forecast/actual series were too short to form even one aligned
    /// error/revision pair at the requested horizon.
    SeriesTooShort {
        /// Description of the length requirement that was violated.
        what: &'static str,
        /// The number of observations supplied.
        got: usize,
        /// The minimum number of observations required.
        need: usize,
    },
    /// The total (centered) sum of squares of the regression response was
    /// zero (a constant response), so the centered R-squared is undefined.
    ConstantResponse {
        /// Name of the offending response series.
        what: &'static str,
    },
    /// A matrix that must be invertible (the HAC parameter covariance used in
    /// the Wald quadratic form) was numerically singular.
    Singular {
        /// Which matrix the factorization rejected.
        what: &'static str,
    },
    /// An error propagated from the OLS / HAC layer (this crate never
    /// reimplements least squares).
    Hac(HacError),
    /// An error propagated from the chi-squared p-value evaluation.
    Stats(StatsError),
}

impl fmt::Display for SurveyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply a non-empty series (and, for a \
                 disagreement panel, at least one forecaster per period)"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected length {expected}, got {got})"
            ),
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; the survey \
                 estimators do not skip missing values silently — clean the \
                 data first"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::SeriesTooShort { what, got, need } => write!(
                f,
                "series too short: {what} (got {got} observations, need at \
                 least {need})"
            ),
            Self::ConstantResponse { what } => write!(
                f,
                "constant response: {what} has zero centered sum of squares, so \
                 the regression's R-squared is undefined; the response carries \
                 no variation to explain"
            ),
            Self::Singular { what } => write!(
                f,
                "{what}: matrix is numerically singular; the HAC parameter \
                 covariance is not positive definite (collinear regressors or a \
                 degenerate design)"
            ),
            Self::Hac(e) => write!(f, "OLS/HAC layer error: {e}"),
            Self::Stats(e) => write!(f, "chi-squared p-value error: {e}"),
        }
    }
}

impl std::error::Error for SurveyError {}

impl From<HacError> for SurveyError {
    fn from(e: HacError) -> Self {
        Self::Hac(e)
    }
}

impl From<StatsError> for SurveyError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
