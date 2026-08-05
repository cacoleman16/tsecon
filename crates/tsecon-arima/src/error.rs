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
        /// Zero-based index of the first offending entry, when the check
        /// scanned a user-supplied series.
        at: Option<usize>,
    },
    /// The sample is too short for the requested specification (after
    /// differencing, estimation needs strictly more usable observations
    /// than free parameters).
    InsufficientObservations {
        /// Minimum number of usable observations required.
        needed: usize,
        /// Number of usable observations available at the failing stage.
        got: usize,
        /// Number of observations in the series as it was supplied.
        nobs: usize,
        /// What those usable observations are needed for.
        what: &'static str,
    },
    /// No optimization run produced a usable (finite) solution; the
    /// likelihood was non-finite at every point visited from every start.
    EstimationFailed {
        /// Description of the failure.
        what: &'static str,
    },
    /// The parameter covariance (inverse observed information) could not
    /// be formed at the reported parameters. The point estimates,
    /// log-likelihood, and default forecasts are unaffected — only
    /// standard errors and the drift-uncertainty forecast term need it.
    CovarianceFailed {
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
            Self::InvalidArgument { what } => write!(f, "{what}"),
            Self::Dimension {
                what,
                expected,
                got,
            } => write!(f, "{what} (expected {expected}, got {got})"),
            Self::NonFinite {
                what,
                at: Some(index),
            } => write!(
                f,
                "non-finite value (NaN or inf) in {what} at index {index}: ARIMA has no \
                 missing-value handling on the differencing path, so one gap would spread \
                 through every difference — drop or impute it first \
                 (pandas: s.dropna() or s.interpolate())"
            ),
            Self::NonFinite { what, at: None } => write!(
                f,
                "non-finite value (NaN or inf) in {what}: check the inputs for missing \
                 values or magnitudes large enough to overflow"
            ),
            Self::InsufficientObservations {
                needed,
                got,
                nobs,
                what,
            } if got == nobs => write!(
                f,
                "ARIMA needs at least {needed} observations for {what}, but got {nobs}. \
                 Lower p, d, or q, or supply a longer series."
            ),
            Self::InsufficientObservations {
                needed,
                got,
                nobs,
                what,
            } => write!(
                f,
                "ARIMA needs at least {needed} usable observations for {what}, but only \
                 {got} of the {nobs} supplied survive differencing. Lower p, d, or q, or \
                 supply a longer series."
            ),
            Self::EstimationFailed { what } => write!(f, "ARIMA estimation failed: {what}"),
            Self::CovarianceFailed { what } => write!(
                f,
                "the ARIMA parameter covariance could not be formed: {what}. The fitted \
                 parameters, log-likelihood, and default forecasts are still valid; only \
                 standard errors and the drift-uncertainty forecast term need this."
            ),
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
