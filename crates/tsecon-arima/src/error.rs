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
        /// The remedy, naming the orders that set the requirement (the
        /// seasonal ones when they contribute), as a full sentence.
        remedy: &'static str,
    },
    /// The specification is too large to estimate within the crate's work
    /// and memory budgets ([`crate::spec::ARIMA_WORK_BUDGET`],
    /// [`crate::spec::ARIMA_BUDGET_VALUES`]; see
    /// [`ArimaSpec::estimation_cost`](crate::ArimaSpec::estimation_cost)),
    /// refused before anything is allocated.
    TooLarge {
        /// The state dimension `max(p + s*P, q + s*Q + 1)` of the exact
        /// likelihood's Kalman filter.
        state_dim: usize,
        /// The number of free parameters `k`.
        k_params: usize,
        /// The usable observations after differencing.
        n_eff: usize,
        /// Which budget (or both) the specification passes.
        exceeded: &'static str,
        /// The largest state dimension that fits both budgets at this
        /// sample size and parameter count (`0` when none does).
        max_state_dim: usize,
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
    /// Automatic order selection ([`crate::auto_arima`]) could not
    /// complete: a differencing advisor (`ndiffs`/`nsdiffs`) failed to
    /// run, or no candidate model produced an admissible fit. The
    /// message carries the underlying cause verbatim.
    Selection {
        /// Description of the failure, including the advisor's or the
        /// last candidate's own error text.
        what: String,
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
                remedy,
            } if got == nobs => write!(
                f,
                "ARIMA needs at least {needed} observations for {what}, but got {nobs}. \
                 {remedy}"
            ),
            Self::InsufficientObservations {
                needed,
                got,
                nobs,
                what,
                remedy,
            } => write!(
                f,
                "ARIMA needs at least {needed} usable observations for {what}, but only \
                 {got} of the {nobs} supplied survive differencing. {remedy}"
            ),
            Self::TooLarge {
                state_dim,
                k_params,
                n_eff,
                exceeded,
                max_state_dim,
            } => write!(
                f,
                "the ARIMA orders (p, q and seasonal=(P, D, Q, s)) are too large to \
                 estimate: the exact likelihood runs a Kalman filter on a state of dimension \
                 max(p + s*P, q + s*Q + 1) = {state_dim}, whose cost grows with the cube of \
                 that dimension, and with {k_params} free parameters on {n_eff} usable \
                 observations even the work every fit is guaranteed to do (one gradient and \
                 the observed-information Hessian behind bse, 2k(k + 2) + 1 likelihood \
                 evaluations) would pass {exceeded}, checked before anything is allocated. At \
                 this sample size and parameter count the state dimension can be at most \
                 {max_state_dim}: lower p or q, or the seasonal orders P, Q or the period s; a \
                 long seasonal period is usually better handled by seasonal adjustment (stl) \
                 than by seasonal ARMA lags."
            ),
            Self::EstimationFailed { what } => write!(f, "ARIMA estimation failed: {what}"),
            Self::CovarianceFailed { what } => write!(
                f,
                "the ARIMA parameter covariance could not be formed: {what}. The fitted \
                 parameters, log-likelihood, and default forecasts are still valid; only \
                 standard errors and the drift-uncertainty forecast term need this."
            ),
            Self::Selection { what } => write!(f, "auto_arima: {what}"),
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
