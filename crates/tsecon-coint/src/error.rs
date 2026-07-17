//! Error type shared by the cointegration and VECM layer.

use core::fmt;

use tsecon_diag::DiagError;
use tsecon_linalg::LinalgError;
use tsecon_stats::StatsError;

/// Errors returned by the cointegration / VECM layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, CointError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum CointError {
    /// An error bubbled up from the structured linear-algebra layer
    /// (Cholesky, eigenvalues, ...).
    Linalg(LinalgError),
    /// An error bubbled up from the special-function layer.
    Stats(StatsError),
    /// An error bubbled up from the diagnostics layer (the augmented
    /// Dickey-Fuller step of Engle-Granger).
    Diag(DiagError),
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
    /// A matrix that must be symmetric positive definite (a residual
    /// second-moment matrix `S_00`, `S_11`, ...) failed its Cholesky
    /// factorization: the auxiliary regressors are collinear or the
    /// sample is degenerate.
    NotPositiveDefinite {
        /// Name of the offending matrix.
        what: &'static str,
    },
    /// A square matrix that had to be inverted (a normalizing
    /// `beta[:r, :r]` block, an `S_00`, ...) was numerically singular.
    Singular {
        /// Name of the offending matrix.
        what: &'static str,
    },
    /// The requested cointegration rank is outside `0 ..= k` (`k` the
    /// number of series). Rank `0` is no cointegration (a VAR in
    /// differences); rank `k` is a stationary level VAR.
    InvalidRank {
        /// The rank that was requested.
        rank: usize,
        /// The number of series `k` (the maximum admissible rank).
        neqs: usize,
    },
    /// The sample is too short for the requested specification: the
    /// Johansen auxiliary regressions need strictly more usable rows than
    /// short-run regressors.
    InsufficientObservations {
        /// Minimum number of usable observations required.
        needed: usize,
        /// Number of usable observations available.
        got: usize,
    },
}

impl fmt::Display for CointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Stats(e) => write!(f, "special-function failure: {e}"),
            Self::Diag(e) => write!(f, "diagnostics failure: {e}"),
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
            Self::Singular { what } => write!(f, "matrix is numerically singular: {what}"),
            Self::InvalidRank { rank, neqs } => write!(
                f,
                "invalid cointegration rank {rank}: must satisfy 0 <= rank <= {neqs} \
                 (rank 0 is a VAR in differences, rank {neqs} a stationary level VAR)"
            ),
            Self::InsufficientObservations { needed, got } => write!(
                f,
                "insufficient observations: need at least {needed} usable rows, got {got}"
            ),
        }
    }
}

impl std::error::Error for CointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linalg(e) => Some(e),
            Self::Stats(e) => Some(e),
            Self::Diag(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for CointError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<StatsError> for CointError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}

impl From<DiagError> for CointError {
    fn from(e: DiagError) -> Self {
        Self::Diag(e)
    }
}
