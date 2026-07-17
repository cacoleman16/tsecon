//! Error type shared by the linear IV-GMM and nonlinear-GMM routines.
//!
//! Every fallible public function in this crate returns
//! `Result<_, GmmError>`; nothing in the non-test code path panics on user
//! input. Messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters, and what the caller can do.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_optim::OptimError;
use tsecon_stats::StatsError;

/// Errors produced by the GMM estimators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum GmmError {
    /// A required design/instrument matrix or slice was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// Two inputs have incompatible sizes (a column length that does not
    /// match the response, or a weight matrix of the wrong order).
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// The moment condition is under-identified: fewer instruments (moment
    /// conditions) than parameters, so the GMM criterion has no isolated
    /// minimum.
    UnderIdentified {
        /// Number of moment conditions (instruments) supplied.
        moments: usize,
        /// Number of parameters to estimate.
        params: usize,
    },
    /// Fewer observations than parameters: no residual degrees of freedom
    /// remain, so standard errors are undefined.
    DegreesOfFreedom {
        /// The number of observations supplied.
        n: usize,
        /// The number of parameters.
        k: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A matrix that must be symmetric positive definite was numerically
    /// singular or indefinite — collinear instruments, a rank-deficient
    /// projected design (weak instruments), or a degenerate moment
    /// covariance.
    SingularMatrix {
        /// Which matrix the factorization rejected.
        what: &'static str,
    },
    /// A supplied kernel bandwidth was negative, NaN, or infinite.
    InvalidBandwidth {
        /// The offending bandwidth.
        value: f64,
    },
    /// A scalar or configuration argument was outside its valid domain
    /// (e.g. a non-positive convergence tolerance or a zero iteration cap).
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// The user-supplied moment function returned a moment matrix whose shape
    /// changed between evaluations or did not match the declared dimensions.
    InconsistentMoments {
        /// Description of the inconsistency.
        what: &'static str,
    },
    /// An error propagated from the chi-squared p-value evaluation.
    Stats(StatsError),
    /// An error propagated from the derivative-free optimizer.
    Optim(OptimError),
    /// An error propagated from the HAC / robust weighting layer.
    Hac(HacError),
}

impl fmt::Display for GmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply at least one observation and one column"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            Self::UnderIdentified { moments, params } => write!(
                f,
                "under-identified GMM: {moments} moment conditions (instruments) \
                 for {params} parameters; GMM requires at least as many moments \
                 as parameters (moments >= params) — add instruments or drop \
                 regressors"
            ),
            Self::DegreesOfFreedom { n, k } => write!(
                f,
                "n = {n} observations with k = {k} parameters leaves no residual \
                 degrees of freedom (requires n > k); standard errors are undefined"
            ),
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; GMM estimators do \
                 not skip missing values silently — clean the data first"
            ),
            Self::SingularMatrix { what } => write!(
                f,
                "{what}: matrix is numerically singular or not positive definite; \
                 common causes are collinear instruments, weak instruments \
                 (a rank-deficient projected design), or a degenerate moment \
                 covariance"
            ),
            Self::InvalidBandwidth { value } => write!(
                f,
                "bandwidth = {value} is invalid: requires a finite value >= 0 \
                 (the lag-truncation parameter for the HAC weighting kernel)"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::InconsistentMoments { what } => write!(
                f,
                "inconsistent moment function output: {what}; the moment function \
                 must return an (n_obs x n_moments) matrix with the same shape at \
                 every parameter value"
            ),
            Self::Stats(e) => write!(f, "chi-squared p-value error: {e}"),
            Self::Optim(e) => write!(f, "optimizer error: {e}"),
            Self::Hac(e) => write!(f, "robust/HAC weighting error: {e}"),
        }
    }
}

impl std::error::Error for GmmError {}

impl From<StatsError> for GmmError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}

impl From<OptimError> for GmmError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}

impl From<HacError> for GmmError {
    fn from(e: HacError) -> Self {
        Self::Hac(e)
    }
}
