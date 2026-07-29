//! Error type for the univariate volatility-model crate.

use core::fmt;

use tsecon_optim::OptimError;

/// Errors returned by the univariate volatility models.
///
/// Every fallible public function in this crate returns
/// `Result<_, GarchError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum GarchError {
    /// The model specification itself is malformed (e.g. a GARCH order of
    /// zero symmetric ARCH terms, or more lags than observations).
    InvalidSpec {
        /// Description of the violated requirement.
        what: &'static str,
    },
    /// A parameter value is outside its admissible domain (negative ARCH
    /// coefficient, non-positive `omega`, persistence at or above one,
    /// `nu <= 2`, ...).
    InvalidParameter {
        /// Name of the offending parameter (or parameter group).
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// A parameter vector has the wrong length for the specification.
    DimensionMismatch {
        /// Description of the offending input.
        what: &'static str,
        /// The expected length.
        expected: usize,
        /// The actual length.
        actual: usize,
    },
    /// An input or an intermediate quantity contains NaN or infinity where
    /// finite values are required (data, parameters, or a conditional
    /// variance that left the representable range).
    NonFinite {
        /// Name of the offending quantity.
        what: &'static str,
        /// Zero-based index of the first offending entry, when the check
        /// scanned a user-supplied series.
        at: Option<usize>,
    },
    /// Too few observations for the requested model.
    InsufficientData {
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
        /// Longest lag in the volatility recursion.
        max_lag: usize,
        /// Number of free parameters in the specification.
        n_params: usize,
    },
    /// The numerical Hessian of the log-likelihood could not be inverted
    /// (flat or boundary optimum); standard errors are unavailable at this
    /// point.
    SingularHessian,
    /// The requested forecast has no analytic form in this release
    /// (EGARCH beyond one step requires simulation).
    UnsupportedForecast {
        /// Description of the unsupported request.
        what: &'static str,
    },
    /// An error bubbled up from the optimization layer.
    Optim(OptimError),
}

impl fmt::Display for GarchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec { what } => write!(f, "invalid model specification: {what}"),
            Self::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid parameter {name} = {value}: requires {requirement}"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {actual})"
            ),
            Self::NonFinite {
                what,
                at: Some(index),
            } => write!(
                f,
                "non-finite value (NaN or inf) in {what} at index {index}: the volatility \
                 recursion has no missing-value handling, so one gap would contaminate \
                 every later conditional variance — drop or impute it first \
                 (pandas: s.dropna()). Note that a return series built with .pct_change() \
                 starts with a NaN."
            ),
            Self::NonFinite { what, at: None } => write!(
                f,
                "non-finite value (NaN or inf) in {what}: the series is probably on a \
                 scale that overflows the recursion — GARCH is normally fitted to returns \
                 in percent (100 * log-differences), not to raw levels"
            ),
            Self::InsufficientData {
                needed,
                got,
                max_lag,
                n_params,
            } => write!(
                f,
                "this volatility model needs at least {needed} observations but got {got}: \
                 the recursion consumes {max_lag} presample observation(s) and its \
                 {n_params} parameters must then be estimated from what is left. Supply a \
                 longer series, or lower p/o/q."
            ),
            Self::SingularHessian => write!(
                f,
                "the numerical Hessian of the log-likelihood is singular, so standard \
                 errors are unavailable: the optimum sits on a boundary (persistence at \
                 1, or omega at 0), which usually means the series is too short or has \
                 no volatility clustering to identify"
            ),
            Self::UnsupportedForecast { what } => {
                write!(f, "unsupported forecast: {what}")
            }
            Self::Optim(e) => write!(f, "optimization failure: {e}"),
        }
    }
}

impl std::error::Error for GarchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Optim(e) => Some(e),
            _ => None,
        }
    }
}

impl From<OptimError> for GarchError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}
