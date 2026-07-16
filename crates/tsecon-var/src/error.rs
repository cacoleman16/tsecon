//! Error type shared by the VAR estimation and analysis layer.

use core::fmt;

use tsecon_linalg::LinalgError;
use tsecon_stats::StatsError;

/// Errors returned by the VAR layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, VarError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum VarError {
    /// An error bubbled up from the structured linear-algebra layer
    /// (companion eigenvalues, Cholesky, ...).
    Linalg(LinalgError),
    /// An error bubbled up from the special-function layer (incomplete
    /// beta, inverse normal CDF, ...).
    Stats(StatsError),
    /// Two inputs (or an input and a model dimension) have incompatible
    /// sizes.
    Dimension {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// A scalar or structural argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// An input contained a NaN or infinity.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A matrix that must be symmetric positive definite (residual
    /// covariance, regressor cross-product, Wald middle matrix) failed
    /// its Cholesky factorization.
    NotPositiveDefinite {
        /// Name of the offending matrix.
        what: &'static str,
    },
    /// The sample is too short for the requested specification: OLS
    /// needs strictly more usable observations than regressors per
    /// equation.
    InsufficientObservations {
        /// Minimum number of usable observations required.
        needed: usize,
        /// Number of usable observations available.
        got: usize,
    },
}

impl fmt::Display for VarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Stats(e) => write!(f, "special-function failure: {e}"),
            Self::Dimension {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::NotPositiveDefinite { what } => {
                write!(f, "matrix is not positive definite: {what}")
            }
            Self::InsufficientObservations { needed, got } => write!(
                f,
                "insufficient observations: need at least {needed} usable rows, got {got}"
            ),
        }
    }
}

impl std::error::Error for VarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linalg(e) => Some(e),
            Self::Stats(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for VarError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<StatsError> for VarError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
