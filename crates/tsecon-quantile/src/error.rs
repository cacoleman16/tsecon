//! Error type shared by the quantile estimators.
//!
//! Every fallible public function in this crate returns
//! `Result<_, QuantileError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters, and what the caller can do.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_stats::StatsError;

/// Errors produced by the quantile-regression estimators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantileError {
    /// A required input series or column set was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// Two inputs that must share a length do not.
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The length that was expected.
        expected: usize,
        /// The length that was received.
        got: usize,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
        /// Index of the first offending entry.
        index: usize,
    },
    /// No quantile levels were supplied.
    NoTaus,
    /// A quantile level was outside the open interval `(0, 1)`.
    InvalidTau {
        /// The offending level.
        tau: f64,
    },
    /// Growth-at-risk requires strictly increasing quantile levels so the
    /// rearrangement and the "current risk read" are well defined.
    TausNotIncreasing {
        /// Index `i` at which `taus[i] <= taus[i-1]`.
        index: usize,
    },
    /// Fewer usable observations than parameters: the weighted least-squares
    /// steps (and the sandwich covariance) are undefined.
    DegreesOfFreedom {
        /// The number of usable observations.
        n: usize,
        /// The number of regression parameters.
        k: usize,
    },
    /// A horizon left too few observations after shifting and lagging.
    HorizonExhaustsSample {
        /// The offending horizon.
        horizon: usize,
        /// The series length.
        n: usize,
        /// Observations the design at this horizon would keep.
        nobs: usize,
        /// Parameters the design at this horizon carries.
        k: usize,
    },
    /// Growth-at-risk needs `horizon >= 1`: at `horizon = 0` the "h-ahead
    /// outcome" is the regressand itself.
    ZeroHorizon,
    /// The Powell-sandwich bandwidth could not be formed: either the
    /// Hall-Sheather offset pushed `tau ± h` outside `(0, 1)` (tau too
    /// extreme for this sample size), or the residual scale collapsed to
    /// zero (degenerate residuals), or the kernel density estimate at zero
    /// vanished (no residual mass near the fitted quantile).
    DegenerateBandwidth {
        /// The quantile level being fitted.
        tau: f64,
        /// The sample size.
        n: usize,
        /// Which quantity degenerated.
        what: &'static str,
    },
    /// An error propagated from the OLS layer (each IRLS step is a weighted
    /// least-squares solve delegated to `tsecon-hac`, the single OLS owner).
    Hac(HacError),
    /// An error propagated from the normal quantile evaluation.
    Stats(StatsError),
    /// The `X'X` bread of the sandwich covariance was numerically singular.
    Singular {
        /// Which matrix the factorization rejected.
        what: &'static str,
    },
}

impl fmt::Display for QuantileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply at least one observation per \
                 series and at least one design column"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected length {expected}, got \
                 {got}); every design column and condition series must align \
                 with the outcome observation for observation"
            ),
            Self::NonFinite { what, index } => write!(
                f,
                "non-finite value (NaN or infinity) in {what} at index \
                 {index}; the quantile estimators do not skip missing values \
                 silently — clean or drop the affected observations first"
            ),
            Self::NoTaus => write!(
                f,
                "no quantile levels supplied; pass at least one tau in the \
                 open interval (0, 1), e.g. [0.05, 0.5, 0.95]"
            ),
            Self::InvalidTau { tau } => write!(
                f,
                "invalid quantile level tau = {tau}; every tau must lie \
                 strictly inside (0, 1) — the 0th and 100th percentiles are \
                 the sample extremes, not regression quantiles"
            ),
            Self::TausNotIncreasing { index } => write!(
                f,
                "quantile levels must be strictly increasing for \
                 growth-at-risk, but taus[{index}] <= taus[{}]; sort the taus \
                 (and drop duplicates) so the rearranged quantile curve and \
                 the current risk read are well defined",
                index - 1
            ),
            Self::DegreesOfFreedom { n, k } => write!(
                f,
                "n = {n} observations with k = {k} parameters leaves no room \
                 to fit the quantile regression (requires n > k); supply a \
                 longer series or drop regressors"
            ),
            Self::HorizonExhaustsSample {
                horizon,
                n,
                nobs,
                k,
            } => write!(
                f,
                "horizon {horizon} leaves nobs = {nobs} usable observations \
                 from a series of length {n}, but the design carries k = {k} \
                 parameters (requires nobs > k); lower the maximum horizon, \
                 reduce the lag controls, or supply a longer series"
            ),
            Self::ZeroHorizon => write!(
                f,
                "growth-at-risk requires horizon >= 1: at horizon = 0 the \
                 'h-ahead outcome' is the regressand itself and the risk read \
                 is not a forecast; pass the number of periods ahead you want \
                 the conditional quantiles for"
            ),
            Self::DegenerateBandwidth { tau, n, what } => write!(
                f,
                "cannot form the Powell-sandwich bandwidth at tau = {tau} \
                 with n = {n}: {what}; use a less extreme tau, a longer \
                 sample, or check that the outcome is not (near-)constant"
            ),
            Self::Hac(e) => write!(f, "weighted least-squares step error: {e}"),
            Self::Stats(e) => write!(f, "normal quantile evaluation error: {e}"),
            Self::Singular { what } => write!(
                f,
                "{what}: matrix is numerically singular; common causes are \
                 collinear design columns or a duplicated condition series"
            ),
        }
    }
}

impl std::error::Error for QuantileError {}

impl From<HacError> for QuantileError {
    fn from(e: HacError) -> Self {
        Self::Hac(e)
    }
}

impl From<StatsError> for QuantileError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
