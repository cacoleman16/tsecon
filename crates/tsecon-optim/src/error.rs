//! Error types for `tsecon-optim`.

use core::fmt;

/// Errors produced by the optimizers, line search, and transforms in this
/// crate.
///
/// All fallible library entry points return `Result<_, OptimError>`; nothing
/// in the non-test code path panics. Note that *failure to converge* is not
/// an error: optimizers report it through
/// [`OptimizeResult::converged`](crate::OptimizeResult) and
/// [`Termination`](crate::Termination) so the caller still receives the best
/// point found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimError {
    /// An option value is outside its valid domain (e.g. `c1 >= c2` for the
    /// strong-Wolfe line search, or a negative tolerance).
    InvalidOption {
        /// Name of the offending option.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// An input slice that must be non-empty is empty.
    EmptyInput {
        /// Name of the offending input.
        what: &'static str,
    },
    /// Two related inputs disagree in length (e.g. a gradient returned by
    /// [`ObjectiveFn::gradient`](crate::ObjectiveFn::gradient) whose length
    /// differs from `x`).
    DimensionMismatch {
        /// Name of the offending input.
        what: &'static str,
        /// The expected length.
        expected: usize,
        /// The actual length.
        actual: usize,
    },
    /// An input contains NaN or infinity where finite values are required
    /// (e.g. the starting point `x0`, or the objective/gradient evaluated at
    /// the starting point).
    NonFinite {
        /// Name of the offending input.
        what: &'static str,
    },
    /// A transform argument is outside the mathematical domain of the map
    /// (e.g. `theta <= 0` passed to [`Positive::inverse`](crate::Positive),
    /// or a non-increasing vector passed to
    /// [`Ordered::inverse`](crate::Ordered)).
    Domain {
        /// Name of the offending argument.
        name: &'static str,
        /// The invalid value that was supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// AR coefficients passed to
    /// [`StationaryAr::inverse`](crate::StationaryAr) do not lie in the
    /// stationarity region: the Monahan (1984) inverse recursion produced a
    /// partial autocorrelation with modulus `>= 1`.
    NotStationary {
        /// The lag order at which the recursion failed (1-based).
        order: usize,
        /// The offending partial autocorrelation.
        pacf: f64,
    },
}

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimError::InvalidOption {
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid option `{name}` = {value}: requires {requirement}"
            ),
            OptimError::EmptyInput { what } => {
                write!(f, "input `{what}` must not be empty")
            }
            OptimError::DimensionMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch in `{what}`: expected {expected}, got {actual}"
            ),
            OptimError::NonFinite { what } => {
                write!(f, "input `{what}` contains NaN or infinity")
            }
            OptimError::Domain {
                name,
                value,
                requirement,
            } => write!(
                f,
                "argument `{name}` = {value} outside domain: requires {requirement}"
            ),
            OptimError::NotStationary { order, pacf } => write!(
                f,
                "AR coefficients are not stationary: partial autocorrelation \
                 at lag {order} is {pacf} (modulus must be < 1)"
            ),
        }
    }
}

impl std::error::Error for OptimError {}
