//! Error type shared by the structural-identification layer.

use core::fmt;

use tsecon_bayes::BayesError;
use tsecon_linalg::LinalgError;
use tsecon_rng::RngError;
use tsecon_stats::StatsError;

/// Errors returned by the structural-identification layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, IdentError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum IdentError {
    /// An error bubbled up from the structured linear-algebra layer.
    Linalg(LinalgError),
    /// An error bubbled up from the special-function / distribution layer
    /// (the inverse normal CDF behind the Gaussian entries of the Haar
    /// draw).
    Stats(StatsError),
    /// An error bubbled up from the random-stream layer (substream
    /// spawning).
    Rng(RngError),
    /// An error bubbled up from the Bayesian foundations layer (the
    /// reduced-form posterior draw and its Cholesky impulse responses).
    Bayes(BayesError),
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
    /// A restriction referenced a variable, shock, or horizon outside the
    /// dimensions of the model or the sampled impulse-response horizon.
    RestrictionOutOfRange {
        /// Description of which index was out of range.
        what: &'static str,
        /// The offending index.
        index: usize,
        /// The exclusive upper bound the index must respect.
        bound: usize,
    },
    /// An input contained a NaN or infinity.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// An internal iterative algorithm exhausted its iteration budget.
    NoConvergence {
        /// Name of the algorithm that failed to converge.
        what: &'static str,
    },
}

impl fmt::Display for IdentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Stats(e) => write!(f, "special-function failure: {e}"),
            Self::Rng(e) => write!(f, "random-stream failure: {e}"),
            Self::Bayes(e) => write!(f, "Bayesian foundations failure: {e}"),
            Self::Dimension {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::RestrictionOutOfRange { what, index, bound } => write!(
                f,
                "restriction out of range: {what} index {index} is not below the bound {bound}"
            ),
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::NoConvergence { what } => {
                write!(f, "{what} failed to converge")
            }
        }
    }
}

impl std::error::Error for IdentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linalg(e) => Some(e),
            Self::Stats(e) => Some(e),
            Self::Rng(e) => Some(e),
            Self::Bayes(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for IdentError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<StatsError> for IdentError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}

impl From<RngError> for IdentError {
    fn from(e: RngError) -> Self {
        Self::Rng(e)
    }
}

impl From<BayesError> for IdentError {
    fn from(e: BayesError) -> Self {
        Self::Bayes(e)
    }
}
