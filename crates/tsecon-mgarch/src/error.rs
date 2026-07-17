//! Error type for the multivariate conditional-correlation GARCH crate.

use core::fmt;

use tsecon_garch::GarchError;
use tsecon_linalg::LinalgError;
use tsecon_optim::OptimError;

/// Errors returned by the multivariate GARCH models.
///
/// Every fallible public function in this crate returns
/// `Result<_, MgarchError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum MgarchError {
    /// Fewer than two series were supplied (a multivariate model needs at
    /// least a bivariate system).
    TooFewSeries {
        /// Number of series supplied.
        got: usize,
    },
    /// The supplied series do not all share a common length `T`.
    RaggedInput {
        /// Length of the first series (the reference length).
        expected: usize,
        /// Index of the first series whose length differs.
        series: usize,
        /// Length of that series.
        actual: usize,
    },
    /// Too few observations for the requested model.
    InsufficientData {
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
    },
    /// An input or an intermediate quantity contains NaN or infinity where
    /// finite values are required.
    NonFinite {
        /// Name of the offending quantity.
        what: &'static str,
    },
    /// A parameter value is outside its admissible domain (a negative DCC
    /// coefficient, or persistence `a + b` at or above one).
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// A forecast horizon of zero was requested.
    InvalidHorizon,
    /// The requested forecast has no analytic form in this release (DCC
    /// multi-step forecasts require simulation).
    UnsupportedForecast {
        /// Description of the unsupported request.
        what: &'static str,
    },
    /// A per-series univariate GARCH fit or evaluation failed.
    Univariate {
        /// Index of the offending series.
        series: usize,
        /// The underlying univariate error.
        source: GarchError,
    },
    /// A linear-algebra routine failed (a correlation matrix that could not
    /// be factorized even after the jitter ladder, a non-square matrix, ...).
    Linalg(LinalgError),
    /// The step-2 DCC optimization failed.
    Optim(OptimError),
}

impl fmt::Display for MgarchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSeries { got } => {
                write!(f, "a multivariate model needs at least 2 series, got {got}")
            }
            Self::RaggedInput {
                expected,
                series,
                actual,
            } => write!(
                f,
                "ragged input: series {series} has length {actual}, expected {expected}"
            ),
            Self::InsufficientData { needed, got } => write!(
                f,
                "insufficient data: {got} observations, at least {needed} required"
            ),
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid parameter {name} = {value}: requires {requirement}"
            ),
            Self::InvalidHorizon => write!(f, "forecast horizon must be at least 1"),
            Self::UnsupportedForecast { what } => {
                write!(f, "unsupported forecast: {what}")
            }
            Self::Univariate { series, source } => {
                write!(f, "univariate GARCH failure on series {series}: {source}")
            }
            Self::Linalg(e) => write!(f, "linear-algebra failure: {e}"),
            Self::Optim(e) => write!(f, "optimization failure: {e}"),
        }
    }
}

impl std::error::Error for MgarchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Univariate { source, .. } => Some(source),
            Self::Linalg(e) => Some(e),
            Self::Optim(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for MgarchError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<OptimError> for MgarchError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}
