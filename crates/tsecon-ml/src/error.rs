//! Error type shared by every routine in this crate.

use core::fmt;

use tsecon_hac::HacError;

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
    /// A domain violation whose message must carry runtime values (an
    /// offending column index, the value received, the accepted names).
    /// Displayed with the same `invalid argument:` prefix as
    /// [`MlError::InvalidArgument`].
    InvalidValue {
        /// Description of the violation, naming the argument and the fix.
        what: String,
    },
    /// A matrix that must be symmetric positive definite (the kernel ridge
    /// system `K + alpha I`) failed its Cholesky factorization.
    NotPositiveDefinite {
        /// Description of the matrix and what to change.
        what: String,
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
    /// Too few observations for the requested computation — a post-selection
    /// OLS refit (`post_lasso`, `pds_lasso`) needs strictly more rows than
    /// regressors, no residual degrees of freedom remain, or the
    /// kernel-regression local fits have too few rows once the
    /// cross-validation exclusion window is removed. The message
    /// follows the library-wide wording (`insufficient data: {got}
    /// observations, at least {needed} required`).
    InsufficientData {
        /// Number of observations supplied.
        got: usize,
        /// Minimum number of observations required.
        needed: usize,
        /// Which computation ran out of rows.
        what: &'static str,
    },
    /// A block length (block-bootstrap resampling or block permutation)
    /// outside `1..=n`.
    InvalidBlockLength {
        /// Name of the offending argument (`block_length`,
        /// `permutation_block`).
        what: &'static str,
        /// The block length received.
        block_length: usize,
        /// The sample size it must not exceed.
        n: usize,
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
    /// Neural training produced a non-finite loss (the weights blew up).
    Diverged {
        /// Zero-based index of the ensemble member that diverged.
        member: usize,
        /// Epoch (or L-BFGS iteration count) at which it was detected.
        epoch: usize,
    },
    /// An error raised by the shared HAC / OLS engine (`tsecon-hac`) while
    /// computing post-double-selection inference — a singular
    /// `[d, X_union]` design, a non-finite input it found first, or a
    /// covariance breakdown. Wrapped so callers see one error type.
    Hac(HacError),
}

impl From<HacError> for MlError {
    fn from(e: HacError) -> Self {
        Self::Hac(e)
    }
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
            Self::InvalidValue { what } => write!(f, "invalid argument: {what}"),
            Self::NotPositiveDefinite { what } => {
                write!(f, "not positive definite: {what}")
            }
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
            Self::InsufficientData { got, needed, what } => write!(
                f,
                "insufficient data: {got} observations, at least {needed} required ({what})"
            ),
            Self::Hac(e) => write!(f, "HAC/OLS engine: {e}"),
            Self::InvalidBlockLength {
                what,
                block_length,
                n,
            } => write!(
                f,
                "{what}={block_length} is outside 1..={n}: a block cannot be empty \
                 or longer than the {n}-row sample"
            ),
            Self::UnknownChoice {
                what,
                got,
                accepted,
            } => write!(f, "unknown {what} {got:?}; expected one of {accepted}"),
            Self::Diverged { member, epoch } => write!(
                f,
                "training diverged (non-finite loss) in ensemble member {member} at \
                 epoch {epoch}; lower learning_rate, raise alpha, or standardize the inputs"
            ),
        }
    }
}

impl std::error::Error for MlError {}
