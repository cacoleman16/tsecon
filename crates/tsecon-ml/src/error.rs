//! Error type shared by every routine in this crate.

use core::fmt;

/// Errors returned by the penalized-regression solvers and the
/// time-series cross-validation machinery in `tsecon-ml`.
///
/// Every fallible public function in this crate returns
/// `Result<_, MlError>`; no non-test code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum MlError {
    /// An input slice or matrix was empty where a nonempty one is required.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// Two inputs have incompatible sizes (e.g. `X` has `n` rows but `y`
    /// has a different length, or a scaler fitted on `p` columns is applied
    /// to a matrix with a different column count).
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A scalar (or configuration) argument was outside its valid domain,
    /// e.g. a negative penalty, an `l1_ratio` outside `[0, 1]`, a
    /// non-positive tolerance, a zero fold count, or a cross-validation
    /// window that cannot fit inside the sample.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// A dense decomposition (the thin SVD backing the ridge closed form
    /// and the ordinary-least-squares adaptive-LASSO weights) failed to
    /// converge.
    DecompositionFailed {
        /// Description of the computation that needed the decomposition.
        what: &'static str,
    },
    /// The coordinate-descent solver did not reach its coefficient-change
    /// tolerance within its iteration budget. The last iterate is discarded
    /// rather than returned silently as if converged.
    NoConvergence {
        /// Number of coordinate sweeps performed.
        iterations: usize,
        /// Largest absolute coefficient change in the final sweep.
        max_change: f64,
    },
    /// Too few observations for the requested fit. For the neural
    /// estimators the minimum counts the rows a temporal validation split
    /// or a reservoir washout removes before any training happens.
    InsufficientData {
        /// Minimum number of observations required.
        needed: usize,
        /// Number of observations supplied.
        got: usize,
    },
    /// A string option (an activation or solver name) was not one of the
    /// accepted values. The message lists them.
    UnknownChoice {
        /// Name of the option.
        what: &'static str,
        /// The value that was passed.
        got: String,
        /// The accepted values, rendered for the message.
        accepted: &'static str,
    },
    /// A configuration value was outside its domain, where the message
    /// needs the offending number (or the fix) spelled out — an empty or
    /// too-deep `hidden`, a `washout` that leaves no training rows, an
    /// argument passed explicitly where the chosen solver cannot use it.
    InvalidValue {
        /// Full description, naming the argument and the fix.
        what: String,
    },
    /// Neural training produced a non-finite loss (the weights blew up).
    Diverged {
        /// Zero-based index of the ensemble member that diverged.
        member: usize,
        /// Epoch (or L-BFGS iteration count) at which it was detected.
        epoch: usize,
    },
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(f, "empty input: {what}"),
            Self::DimensionMismatch {
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
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::DecompositionFailed { what } => {
                write!(f, "dense decomposition failed to converge in {what}")
            }
            Self::NoConvergence {
                iterations,
                max_change,
            } => write!(
                f,
                "coordinate descent did not converge after {iterations} sweeps \
                 (last max coefficient change {max_change:e})"
            ),
            Self::InsufficientData { needed, got } => write!(
                f,
                "insufficient data: {got} observations, at least {needed} required"
            ),
            Self::UnknownChoice {
                what,
                got,
                accepted,
            } => write!(f, "unknown {what} {got:?}; expected one of {accepted}"),
            Self::InvalidValue { what } => write!(f, "invalid argument: {what}"),
            Self::Diverged { member, epoch } => write!(
                f,
                "training diverged (non-finite loss) in ensemble member {member} at \
                 epoch {epoch}; lower learning_rate, raise alpha, or standardize the inputs"
            ),
        }
    }
}

impl std::error::Error for MlError {}
