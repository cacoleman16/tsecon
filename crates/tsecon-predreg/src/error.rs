//! Error type shared by the predictive-regression estimators.
//!
//! Every fallible public function in this crate returns
//! `Result<_, PredRegError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters, and what the caller can do.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_stats::StatsError;

/// Errors produced by the predictive-regression estimators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum PredRegError {
    /// A required series was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// The predictor and response series (or two predictor columns) have
    /// incompatible lengths.
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The length that was expected.
        expected: usize,
        /// The length that was received.
        got: usize,
    },
    /// Fewer usable observations than parameters: after the one-period lead
    /// alignment `N = n - 1`, the regression leaves no residual degrees of
    /// freedom, so variances are undefined.
    DegreesOfFreedom {
        /// The number of usable (aligned) observations `N`.
        n: usize,
        /// The number of regression parameters (predictors plus intercept).
        k: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A configuration scalar was outside its valid domain (e.g. an IVX
    /// exponent `alpha` not in `(0, 1)`).
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// A matrix that must be invertible (the IVX instrument-by-regressor
    /// cross-moment, or the instrument second-moment used in the Wald form)
    /// was numerically singular — collinear predictors or a degenerate,
    /// near-constant predictor path.
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

impl fmt::Display for PredRegError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply a predictor and response of length \
                 at least three (one observation is lost to the predictive lead)"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected length {expected}, got {got})"
            ),
            Self::DegreesOfFreedom { n, k } => write!(
                f,
                "N = {n} aligned observations with k = {k} parameters leaves no \
                 residual degrees of freedom (requires N > k); the predictive \
                 regression's variances are undefined — supply a longer series"
            ),
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; the predictive \
                 estimators do not skip missing values silently — clean the data \
                 first"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::Singular { what } => write!(
                f,
                "{what}: matrix is numerically singular; common causes are \
                 collinear predictors or a near-constant (degenerate) predictor \
                 path whose IVX instrument carries no variation"
            ),
            Self::Hac(e) => write!(f, "OLS/HAC layer error: {e}"),
            Self::Stats(e) => write!(f, "chi-squared p-value error: {e}"),
        }
    }
}

impl std::error::Error for PredRegError {}

impl From<HacError> for PredRegError {
    fn from(e: HacError) -> Self {
        Self::Hac(e)
    }
}

impl From<StatsError> for PredRegError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
