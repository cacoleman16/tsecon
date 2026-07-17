//! Error type shared by all solvers in this crate.

use core::fmt;

/// Errors returned by the structured solvers in `tsecon-linalg`.
///
/// Every fallible public function in this crate returns
/// `Result<_, LinalgError>`; no code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum LinalgError {
    /// An input slice or matrix was empty where a nonempty one is required.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// Two inputs (or an input and a requested order) have incompatible sizes.
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// A matrix argument is not square where a square one is required.
    NotSquare {
        /// Name of the offending argument.
        what: &'static str,
        /// Number of rows received.
        rows: usize,
        /// Number of columns received.
        cols: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A matrix (or autocovariance sequence) that must be symmetric positive
    /// definite turned out not to be, e.g. a Levinson-Durbin innovation
    /// variance hit zero or a Cholesky pivot went nonpositive.
    NotPositiveDefinite {
        /// Description of where positive definiteness failed.
        what: &'static str,
    },
    /// The transition matrix of a Lyapunov equation has spectral radius
    /// greater than or equal to one, so no stationary solution exists.
    Unstable {
        /// The computed spectral radius `max |lambda_i(A)|`.
        spectral_radius: f64,
    },
    /// An iterative algorithm failed to meet its convergence criterion
    /// within its iteration budget.
    NoConvergence {
        /// Number of iterations performed.
        iterations: usize,
        /// Last observed relative update / residual size.
        residual: f64,
    },
    /// The jittered Cholesky exhausted its bounded jitter ladder without
    /// producing a positive definite factorization.
    JitterExhausted {
        /// Number of factorization attempts made (including the clean one).
        attempts: usize,
        /// The largest jitter that was tried (absolute, already scaled).
        max_jitter: f64,
    },
    /// An eigenvalue decomposition inside `faer` did not converge.
    EigenFailed {
        /// Description of the computation that needed the eigenvalues.
        what: &'static str,
    },
    /// A scalar argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
}

impl fmt::Display for LinalgError {
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
            Self::NotSquare { what, rows, cols } => {
                write!(f, "matrix must be square: {what} is {rows}x{cols}")
            }
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::NotPositiveDefinite { what } => {
                write!(f, "not positive definite: {what}")
            }
            Self::Unstable { spectral_radius } => write!(
                f,
                "transition matrix is not stable: spectral radius {spectral_radius} >= 1; \
                 the discrete Lyapunov equation has no stationary solution"
            ),
            Self::NoConvergence {
                iterations,
                residual,
            } => write!(
                f,
                "no convergence after {iterations} iterations (last relative update {residual:e})"
            ),
            Self::JitterExhausted {
                attempts,
                max_jitter,
            } => write!(
                f,
                "Cholesky failed after {attempts} attempts with jitter up to {max_jitter:e}; \
                 matrix is too far from positive definite"
            ),
            Self::EigenFailed { what } => {
                write!(f, "eigenvalue decomposition failed to converge in {what}")
            }
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
        }
    }
}

impl std::error::Error for LinalgError {}
