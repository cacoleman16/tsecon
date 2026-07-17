//! Error type shared by the dynamic-factor nowcasting routines.

use core::fmt;

use tsecon_favar::FavarError;
use tsecon_linalg::LinalgError;
use tsecon_optim::OptimError;
use tsecon_ssm::SsmError;
use tsecon_var::VarError;

/// Errors returned by `tsecon-nowcast`.
///
/// Every fallible public function in this crate returns
/// `Result<_, NowcastError>`; no code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum NowcastError {
    /// An input matrix or slice was empty where a nonempty one is required.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// Two inputs have incompatible sizes.
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// An input contained a NaN or infinite entry where only finite values
    /// (or, for the panel, NaN-for-missing) are admissible.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A scalar or index argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// A series index passed to the nowcast query is out of range.
    SeriesOutOfRange {
        /// The requested series index.
        requested: usize,
        /// The number of series in the panel.
        n_series: usize,
    },
    /// An error propagated from the principal-component factor layer.
    Favar(FavarError),
    /// An error propagated from the factor VAR (`tsecon-var`).
    Var(VarError),
    /// An error propagated from the state-space engine (`tsecon-ssm`).
    Ssm(SsmError),
    /// An error propagated from the structured linear-algebra layer.
    Linalg(LinalgError),
    /// An error propagated from the numerical optimizer (`tsecon-optim`),
    /// used by the one-step MLE ([`crate::mle`]).
    Optim(OptimError),
}

impl fmt::Display for NowcastError {
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
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::SeriesOutOfRange {
                requested,
                n_series,
            } => write!(
                f,
                "series index {requested} out of range (panel has {n_series} series)"
            ),
            Self::Favar(e) => write!(f, "factor-model error: {e}"),
            Self::Var(e) => write!(f, "factor VAR error: {e}"),
            Self::Ssm(e) => write!(f, "state-space error: {e}"),
            Self::Linalg(e) => write!(f, "linear-algebra error: {e}"),
            Self::Optim(e) => write!(f, "optimizer error: {e}"),
        }
    }
}

impl std::error::Error for NowcastError {}

impl From<FavarError> for NowcastError {
    fn from(e: FavarError) -> Self {
        Self::Favar(e)
    }
}

impl From<VarError> for NowcastError {
    fn from(e: VarError) -> Self {
        Self::Var(e)
    }
}

impl From<SsmError> for NowcastError {
    fn from(e: SsmError) -> Self {
        Self::Ssm(e)
    }
}

impl From<LinalgError> for NowcastError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<OptimError> for NowcastError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}
