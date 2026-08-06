//! Error type shared by the state-space engine.

use core::fmt;

use tsecon_linalg::LinalgError;

/// Errors returned by the state-space engine.
///
/// Every fallible public function in this crate returns
/// `Result<_, SsmError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum SsmError {
    /// An error bubbled up from the structured linear-algebra layer
    /// (Lyapunov solve, Cholesky hygiene, eigenvalues, ...).
    Linalg(LinalgError),
    /// Two inputs (or an input and a declared model dimension) have
    /// incompatible sizes.
    Dimension {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// A scalar or structural argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// An input contained an infinity, or a NaN outside the observation
    /// vector (NaN in `y` means "missing"; NaN anywhere else is an error).
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A covariance matrix (`H` or `Q`) failed the symmetric
    /// positive-semidefinite hygiene check.
    NotPsd {
        /// Name of the offending matrix.
        what: &'static str,
    },
    /// A required system matrix was not supplied to the builder.
    MissingMatrix {
        /// Name of the missing matrix.
        what: &'static str,
    },
    /// The univariate (sequential) filtering path requires a diagonal
    /// observation covariance `H`.
    ///
    /// The LDL' pre-whitening transform that lifts this restriction is
    /// `// TODO(phase0)`; until then, either diagonalize `H` yourself or
    /// use the matrix filter path.
    NonDiagonalH,
    /// The requested operation does not support exact-diffuse
    /// initialization (e.g. the matrix cross-check filter).
    DiffuseNotSupported {
        /// Name of the operation.
        what: &'static str,
    },
    /// Every observed element of `y` was numerically uninformative, so the
    /// effective sample is empty and the log-likelihood is an empty sum.
    ///
    /// The filter skips an element whose prediction variance `F` is below
    /// the relative rank tolerance — it cannot be told apart from
    /// cancellation noise. When *every* observed element is skipped the
    /// likelihood would come back as exactly `0.0`, which is not a small
    /// number but the absence of one; returning it as a success would let
    /// a caller optimize a constant and report the result as a fit.
    ///
    /// The cause is never the *scale* of the data or of a state: the rank
    /// test compares `F` against `|Z_i| |P| |Z_i|' + H_ii`, which carries
    /// the same units and the same direction, so rescaling `y` — or
    /// rescaling one state coordinate — cannot move this boundary. What
    /// does move it is the model:
    ///
    /// * the observation equation loads on nothing that varies — `Z` is
    ///   zero, or every state it does load on has zero variance from every
    ///   source (`H_ii = 0` with a zero `P_1` and a zero `R Q R'` in those
    ///   coordinates);
    /// * or `Z_i P Z_i' + H_ii` is a near-exact cancellation, which means
    ///   the model is deterministic along `Z_i` to within roundoff of the
    ///   magnitudes it was assembled from.
    NoInformation,
}

impl fmt::Display for SsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Dimension {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::NonFinite { what } => {
                write!(f, "non-finite value (NaN or infinity) in {what}")
            }
            Self::NotPsd { what } => write!(
                f,
                "{what} must be symmetric positive semidefinite \
                 (within the jitter-ladder tolerance)"
            ),
            Self::MissingMatrix { what } => {
                write!(f, "system matrix {what} was not supplied to the builder")
            }
            Self::NonDiagonalH => write!(
                f,
                "the univariate filtering path requires a diagonal observation \
                 covariance H; pre-whiten the observations or use the matrix filter"
            ),
            Self::DiffuseNotSupported { what } => write!(
                f,
                "{what} does not support exact-diffuse initialization; \
                 use the univariate filtering path"
            ),
            Self::NoInformation => write!(
                f,
                "every observed element of y was numerically uninformative: \
                 its prediction variance F = Z_i P Z_i' + H_ii was \
                 indistinguishable from zero next to |Z_i| |P| |Z_i|' + \
                 H_ii, the magnitude that same sum would have had without \
                 cancellation. The effective sample is therefore empty and \
                 the log-likelihood is an empty sum rather than a number. \
                 The test carries the units and the direction of F, so \
                 rescaling y will not change this, and neither will \
                 rescaling a state — look at the model along the row Z_i \
                 instead. Either it loads on nothing that varies (a zero \
                 row of Z, or zero H_ii together with zero initial and \
                 state variance in the states Z_i does load on — the \
                 degenerate model), or Z_i P Z_i' is a near-total \
                 cancellation, which says the model is deterministic along \
                 Z_i to within roundoff of the numbers it was assembled \
                 from. States in wildly different units make the second \
                 case easy to hit by accident"
            ),
        }
    }
}

impl std::error::Error for SsmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for SsmError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}
