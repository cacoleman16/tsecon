//! Error type for the Markov-switching crate.

use core::fmt;

/// Errors returned by the Markov-switching models.
///
/// Every fallible public function in this crate returns
/// `Result<_, RegimeError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum RegimeError {
    /// The model specification itself is malformed (e.g. fewer than two
    /// regimes, or an expanded state space too large to materialize).
    InvalidSpec {
        /// Description of the violated requirement.
        what: &'static str,
    },
    /// A parameter value is outside its admissible domain (a non-positive
    /// variance, a transition probability outside `[0, 1]`, ...).
    InvalidParameter {
        /// Name of the offending parameter (or parameter group).
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// A transition matrix column does not sum to one (the columns of the
    /// column-stochastic transition matrix `P[i][j] = P(S_t = i | S_{t-1} =
    /// j)` must each be a probability distribution over the destination
    /// regime).
    NotStochastic {
        /// Index of the offending column (the conditioning regime `j`).
        column: usize,
        /// The column sum that should have been one.
        sum: f64,
    },
    /// A parameter or data container has the wrong length for the
    /// specification.
    DimensionMismatch {
        /// Description of the offending input.
        what: &'static str,
        /// The expected length.
        expected: usize,
        /// The actual length.
        actual: usize,
    },
    /// An input or an intermediate quantity contains NaN or infinity where
    /// finite values are required (data, parameters, or a mixture
    /// likelihood that underflowed to zero).
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
    /// A linear system in the estimation step (the stationary distribution
    /// solve, or an M-step normal-equation system) was singular.
    Singular {
        /// Description of the system that could not be solved.
        what: &'static str,
    },
}

impl fmt::Display for RegimeError {
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
            Self::NotStochastic { column, sum } => write!(
                f,
                "transition matrix column {column} sums to {sum}, not 1 \
                 (columns must be probability distributions)"
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
            Self::Singular { what } => {
                write!(f, "singular linear system: {what}")
            }
        }
    }
}

impl std::error::Error for RegimeError {}
