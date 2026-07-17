//! Error type shared by the connectedness layer.

use core::fmt;

use tsecon_linalg::LinalgError;
use tsecon_var::VarError;

/// Errors returned by the connectedness layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, ConnectError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectError {
    /// An error bubbled up from the reduced-form VAR layer (estimation,
    /// MA representation, Cholesky, ...).
    Var(VarError),
    /// An error bubbled up from the structured linear-algebra layer.
    Linalg(LinalgError),
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
    /// A quantity that must be strictly positive (a residual-covariance
    /// diagonal entry, a forecast-error variance) was non-positive, so a
    /// generalized-FEVD share could not be normalized.
    NotPositiveDefinite {
        /// Name of the offending quantity.
        what: &'static str,
    },
}

impl fmt::Display for ConnectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(e) => write!(f, "VAR layer failure: {e}"),
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
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
                write!(f, "quantity is not positive: {what}")
            }
        }
    }
}

impl std::error::Error for ConnectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Var(e) => Some(e),
            Self::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl From<VarError> for ConnectError {
    fn from(e: VarError) -> Self {
        Self::Var(e)
    }
}

impl From<LinalgError> for ConnectError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}
