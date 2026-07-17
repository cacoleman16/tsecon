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
    },
    /// Too few observations for the requested model.
    InsufficientData {
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
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
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::InsufficientData { needed, got } => write!(
                f,
                "insufficient data: {got} observations, at least {needed} required"
            ),
            Self::SingularHessian => write!(
                f,
                "numerical Hessian is singular (flat or boundary optimum); \
                 standard errors unavailable"
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
