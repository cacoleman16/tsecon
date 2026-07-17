//! Error type shared by the Bayesian foundations layer.

use core::fmt;

use tsecon_linalg::LinalgError;
use tsecon_rng::RngError;
use tsecon_ssm::SsmError;
use tsecon_stats::StatsError;

/// Errors returned by the Bayesian foundations layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, BayesError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum BayesError {
    /// An error bubbled up from the structured linear-algebra layer
    /// (Cholesky, companion form, ...).
    Linalg(LinalgError),
    /// An error bubbled up from the special-function / distribution layer
    /// (inverse normal CDF, chi-squared quantile, ...).
    Stats(StatsError),
    /// An error bubbled up from the state-space engine (Kalman filter,
    /// model validation, ...).
    Ssm(SsmError),
    /// An error bubbled up from the random-stream layer.
    Rng(RngError),
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
    /// The sample is too short for the requested specification.
    InsufficientObservations {
        /// Minimum number of usable observations required.
        needed: usize,
        /// Number of usable observations available.
        got: usize,
    },
    /// The exact-diffuse initialization had not collapsed after the first
    /// time period, so the stored filtered/predicted moments the backward
    /// sampler conditions on are not proper distributions. Use a `Known`
    /// or `Stationary` initialization, or provide enough identifying
    /// observations in the first period.
    DiffuseNotCollapsed {
        /// Length of the diffuse period reported by the filter.
        periods: usize,
    },
    /// An internal iterative algorithm exhausted its iteration budget.
    NoConvergence {
        /// Name of the algorithm that failed to converge.
        what: &'static str,
    },
}

impl fmt::Display for BayesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Stats(e) => write!(f, "special-function failure: {e}"),
            Self::Ssm(e) => write!(f, "state-space failure: {e}"),
            Self::Rng(e) => write!(f, "random-stream failure: {e}"),
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
            Self::InsufficientObservations { needed, got } => write!(
                f,
                "insufficient observations: need at least {needed} usable rows, got {got}"
            ),
            Self::DiffuseNotCollapsed { periods } => write!(
                f,
                "exact-diffuse initialization still active after {periods} periods: \
                 the backward sampler needs proper filtered moments (use Known or \
                 Stationary initialization, or more informative first-period data)"
            ),
            Self::NoConvergence { what } => {
                write!(f, "{what} failed to converge")
            }
        }
    }
}

impl std::error::Error for BayesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linalg(e) => Some(e),
            Self::Stats(e) => Some(e),
            Self::Ssm(e) => Some(e),
            Self::Rng(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for BayesError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<StatsError> for BayesError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}

impl From<SsmError> for BayesError {
    fn from(e: SsmError) -> Self {
        Self::Ssm(e)
    }
}

impl From<RngError> for BayesError {
    fn from(e: RngError) -> Self {
        Self::Rng(e)
    }
}
