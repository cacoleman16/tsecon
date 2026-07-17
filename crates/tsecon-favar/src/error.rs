//! Error type shared by the factor-model and FAVAR routines in this crate.

use core::fmt;

use tsecon_linalg::LinalgError;
use tsecon_var::VarError;

/// Errors returned by `tsecon-favar`.
///
/// Every fallible public function in this crate returns
/// `Result<_, FavarError>`; no code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum FavarError {
    /// An input matrix or slice was empty where a nonempty one is required.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// Two inputs have incompatible sizes (e.g. the policy series length
    /// does not match the number of panel observations).
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A panel column has (numerically) zero variance, so it cannot be
    /// standardized to unit standard deviation.
    ZeroVariance {
        /// Index of the constant column.
        column: usize,
    },
    /// A requested factor count (or candidate maximum) is not admissible
    /// for the panel dimensions.
    InvalidFactorCount {
        /// Description of the domain violation.
        what: &'static str,
        /// The requested value.
        requested: usize,
        /// The largest admissible value.
        max: usize,
    },
    /// A scalar or index argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// The `faer` singular value decomposition did not converge.
    SvdFailed,
    /// An error propagated from the structured linear-algebra layer.
    Linalg(LinalgError),
    /// An error propagated from the factor VAR (`tsecon-var`).
    Var(VarError),
}

impl fmt::Display for FavarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(f, "empty input: {what}"),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::ZeroVariance { column } => write!(
                f,
                "panel column {column} has zero variance and cannot be standardized"
            ),
            Self::InvalidFactorCount {
                what,
                requested,
                max,
            } => write!(
                f,
                "invalid factor count: {what} (requested {requested}, max {max})"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::SvdFailed => write!(f, "singular value decomposition failed to converge"),
            Self::Linalg(e) => write!(f, "linear-algebra error: {e}"),
            Self::Var(e) => write!(f, "factor VAR error: {e}"),
        }
    }
}

impl std::error::Error for FavarError {}

impl From<LinalgError> for FavarError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<VarError> for FavarError {
    fn from(e: VarError) -> Self {
        Self::Var(e)
    }
}
