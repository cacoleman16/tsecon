//! Error type shared by the ARIMA estimation and forecasting layer.

use core::fmt;

use tsecon_linalg::LinalgError;
use tsecon_optim::OptimError;
use tsecon_ssm::SsmError;
use tsecon_stats::StatsError;

/// Errors returned by the ARIMA layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, ArimaError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum ArimaError {
    /// An error bubbled up from the state-space engine (model validation,
    /// Kalman filtering, stationary initialization).
    Ssm(SsmError),
    /// An error bubbled up from the optimization suite (reparameterization
    /// domain violations, malformed optimizer inputs).
    Optim(OptimError),
    /// An error bubbled up from the structured linear-algebra layer
    /// (Levinson-Durbin, Cholesky hygiene).
    Linalg(LinalgError),
    /// An error bubbled up from the distribution layer (normal quantiles
    /// for forecast intervals).
    Stats(StatsError),
    /// A scalar or structural argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
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
    /// An input contained a NaN or infinity.
    ///
    /// NaN-coded missing values are not yet accepted by this crate's
    /// simple-differencing path even though the underlying filter supports
    /// them. `// TODO(phase0)`: missing-value support via the levels
    /// state-space form.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// The sample is too short for the requested specification (after
    /// differencing, estimation needs strictly more usable observations
    /// than free parameters).
    InsufficientObservations {
        /// Minimum number of usable observations required.
        needed: usize,
        /// Number of usable observations available.
        got: usize,
    },
    /// No optimization run produced a usable (finite) solution; the
    /// likelihood was non-finite at every point visited from every start.
    EstimationFailed {
        /// Description of the failure.
        what: &'static str,
    },
}

impl fmt::Display for ArimaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ssm(e) => write!(f, "state-space failure: {e}"),
            Self::Optim(e) => write!(f, "optimization failure: {e}"),
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Stats(e) => write!(f, "distribution failure: {e}"),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::Dimension {
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
            Self::InsufficientObservations { needed, got } => write!(
                f,
                "insufficient observations: need at least {needed} usable \
                 observations, got {got}"
            ),
            Self::EstimationFailed { what } => write!(f, "estimation failed: {what}"),
        }
    }
}

impl std::error::Error for ArimaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ssm(e) => Some(e),
            Self::Optim(e) => Some(e),
            Self::Linalg(e) => Some(e),
            Self::Stats(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SsmError> for ArimaError {
    fn from(e: SsmError) -> Self {
        Self::Ssm(e)
    }
}

impl From<OptimError> for ArimaError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}

impl From<LinalgError> for ArimaError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<StatsError> for ArimaError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
