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
    /// HAC weighting was requested with a zero bandwidth — a silent no-op
    /// that returns the White estimator instead of anything
    /// serial-correlation robust. Rejected rather than honored: a caller who
    /// asks for HAC and gets White has the wrong standard errors and no way
    /// to notice.
    HacBandwidthNoOp {
        /// Name of the kernel that was requested (for the message).
        kernel: &'static str,
        /// The lag truncation the automatic rule would pick at this sample
        /// size, offered to the caller as a concrete alternative.
        suggested: usize,
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
            Self::HacBandwidthNoOp { kernel, suggested } => write!(
                f,
                "HAC weighting requested with bandwidth = 0, which is a silent \
                 no-op: at bandwidth 0 the {kernel} kernel puts zero weight on \
                 every lag j >= 1, so the moment covariance collapses to its \
                 lag-0 term (1/n) sum_t z_t z_t' u_t^2 — that is exactly the \
                 White (heteroskedasticity-robust) estimator, with no \
                 serial-correlation robustness at all. Pass an explicit \
                 positive bandwidth (the lag truncation), or ask for the \
                 automatic rule (GmmWeight::HacAuto), which at this sample \
                 size picks {suggested} lags via Newey-West (1994) \
                 floor(4*(n/100)^(2/9)). If the White estimator is what you \
                 want, request it by name with GmmWeight::Robust. Note that \
                 neither choice fixes HAC coverage: this library's \
                 interval-coverage audit measured iv_gmm(weight=\"hac\") at \
                 0.868 coverage against a nominal 0.95 under an AR(1) error \
                 (phi = 0.8, T = 250) with an explicit bandwidth of 10, and \
                 the automatic rule picks FEWER lags than that at the same \
                 sample size — it is a nonzero default, not a remedy"
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
