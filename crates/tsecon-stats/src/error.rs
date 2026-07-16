//! Error types for `tsecon-stats`.

use core::fmt;

/// Errors produced by the special functions and distribution methods in this
/// crate.
///
/// All fallible library entry points return `Result<_, StatsError>`; nothing
/// in the non-test code path panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatsError {
    /// A distribution parameter is outside its valid domain
    /// (e.g. `df <= 0` for a Student t distribution).
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// A function argument is outside the mathematical domain of the
    /// function (e.g. `p = 0` passed to an inverse CDF, or `x < 0` passed to
    /// the incomplete gamma function).
    Domain {
        /// Name of the offending argument.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// An iterative algorithm exhausted its iteration budget without
    /// converging. This indicates either an extreme parameter combination or
    /// a bug; it should not occur for the parameter ranges used in time
    /// series econometrics.
    NoConvergence {
        /// Name of the algorithm that failed to converge.
        what: &'static str,
        /// The iteration budget that was exhausted.
        iterations: u32,
    },
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatsError::InvalidParameter {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid parameter `{name}` = {value}: requires {requirement}"
            ),
            StatsError::Domain {
                name,
                value,
                requirement,
            } => write!(
                f,
                "argument `{name}` = {value} outside domain: requires {requirement}"
            ),
            StatsError::NoConvergence { what, iterations } => {
                write!(
                    f,
                    "{what} failed to converge within {iterations} iterations"
                )
            }
        }
    }
}

impl std::error::Error for StatsError {}
