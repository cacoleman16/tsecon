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
        }
    }
}

impl std::error::Error for MlError {}
