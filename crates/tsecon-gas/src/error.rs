//! Error type for the score-driven (GAS/DCS) model crate.

use core::fmt;

use tsecon_optim::OptimError;

/// Errors returned by the score-driven models.
///
/// Every fallible public function in this crate returns
/// `Result<_, GasError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum GasError {
    /// A parameter value is outside its admissible domain (`omega <= 0`,
    /// `a < 0`, `b` not in `[0, 1)`, or `nu <= 2` for the Student-t model).
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The Student-t density requires a degrees-of-freedom parameter but
    /// none was supplied (or the Gaussian density was given one).
    DofMismatch {
        /// Description of the mismatch.
        what: &'static str,
    },
    /// An input or an intermediate quantity contains NaN or infinity where
    /// finite values are required (the data, the parameters, or a filtered
    /// variance that left the representable range).
    NonFinite {
        /// Name of the offending quantity.
        what: &'static str,
    },
    /// The series has no scale for a variance model to measure: the sample
    /// second moment `mean(y^2)` is zero.
    ///
    /// The log-likelihood is then unbounded above — with every `y_t = 0`
    /// the Gaussian log-density is `-0.5 (ln 2 pi + ln f_t)`, which grows
    /// without limit as the variance is driven to zero — so no
    /// maximum-likelihood estimate exists and any "fit" is an artifact of
    /// wherever the optimizer's floor happened to stop it.
    DegenerateSeries {
        /// What about the series leaves it without a scale.
        what: &'static str,
    },
    /// The series is constant, so the DCS *level* model has nothing to
    /// measure: the robust initial level equals every observation, every
    /// prediction error is exactly zero, and the log-likelihood is
    /// unbounded above as the innovation scale is driven to zero — no
    /// maximum-likelihood estimate exists.
    DegenerateLevel {
        /// What about the series leaves the level model degenerate.
        what: &'static str,
    },
    /// Too few observations for the requested operation.
    InsufficientData {
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
    },
    /// A forecast horizon of zero was requested.
    InvalidHorizon,
    /// An error bubbled up from the optimization layer.
    Optim(OptimError),
}

impl fmt::Display for GasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid parameter {name} = {value}: requires {requirement}"
            ),
            Self::DofMismatch { what } => write!(f, "degrees-of-freedom mismatch: {what}"),
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::DegenerateSeries { what } => write!(
                f,
                "degenerate series: {what}, so the sample second moment \
                 mean(y^2) is zero. A score-driven variance model has no \
                 maximum here — the log-likelihood is unbounded above, \
                 growing without limit as the variance is driven to zero, \
                 so no maximum-likelihood estimate exists. Check the input: \
                 an all-zero placeholder, a difference that is identically \
                 zero, or returns quoted in units so small that y^2 \
                 underflows"
            ),
            Self::DegenerateLevel { what } => write!(
                f,
                "degenerate series for a level model: {what}. Every one-step \
                 prediction error is then exactly zero, so the log-likelihood \
                 is unbounded above as the innovation scale is driven to zero \
                 and no maximum-likelihood estimate exists. Check the input: \
                 a constant placeholder column, or a series filled with a \
                 single repeated value"
            ),
            Self::InsufficientData { needed, got } => write!(
                f,
                "insufficient data: {got} observations, at least {needed} required"
            ),
            Self::InvalidHorizon => write!(f, "forecast horizon must be at least 1"),
            Self::Optim(e) => write!(f, "optimization failure: {e}"),
        }
    }
}

impl std::error::Error for GasError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Optim(e) => Some(e),
            _ => None,
        }
    }
}

impl From<OptimError> for GasError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}
