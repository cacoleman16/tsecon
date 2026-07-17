//! Error type shared by the long-memory estimators.
//!
//! Every fallible public function in this crate returns
//! `Result<_, LongMemoryError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they state
//! what went wrong, why it matters, and what the caller can do about it.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_optim::OptimError;
use tsecon_spectral::SpectralError;

/// Errors produced by the fractional-integration / long-memory estimators.
#[derive(Debug, Clone, PartialEq)]
pub enum LongMemoryError {
    /// A required series (or a request for zero weights) was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// An input contained a NaN or an infinite entry. The estimators do not
    /// skip missing values silently — clean the data first.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// A configuration scalar was outside its valid domain (e.g. the memory
    /// parameter `d` was non-finite).
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// The number of low-frequency ordinates `m` requested for the GPH or
    /// local-Whittle regression is outside the admissible range: it must be at
    /// least `min` (so the regression is identified with residual degrees of
    /// freedom) and at most `(n - 1) / 2` (so every ordinate is a strictly
    /// interior Fourier frequency below Nyquist, where the one-sided
    /// periodogram carries the same folding factor for every used bin).
    InvalidBandwidth {
        /// The requested bandwidth.
        m: usize,
        /// The series length.
        n: usize,
        /// The smallest admissible `m`.
        min: usize,
        /// The largest admissible `m` = `(n - 1) / 2`.
        max: usize,
    },
    /// A periodogram ordinate came out non-positive, so its logarithm (GPH) or
    /// its contribution to the local-Whittle objective is undefined. This
    /// signals a degenerate series (e.g. an exact spectral zero at a Fourier
    /// frequency); supply a longer or less degenerate series.
    NonPositivePeriodogram {
        /// The Fourier-ordinate index `j` (1-based) that vanished.
        j: usize,
    },
    /// The local-Whittle concentrated objective could not be minimized to a
    /// finite interior optimum. The best point found is not trustworthy.
    OptimizationFailed {
        /// Human-readable reason (the optimizer's termination).
        reason: &'static str,
    },
    /// An error propagated from the periodogram layer (this crate never
    /// reimplements a spectral estimator).
    Spectral(SpectralError),
    /// An error propagated from the OLS layer (this crate never reimplements
    /// least squares).
    Hac(HacError),
    /// An error propagated from the optimization layer (this crate never
    /// reimplements a minimizer).
    Optim(OptimError),
}

impl fmt::Display for LongMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => {
                write!(f, "empty input: {what}; supply at least one observation")
            }
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; the long-memory \
                 estimators do not skip missing values silently — clean the data \
                 first"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::InvalidBandwidth { m, n, min, max } => write!(
                f,
                "invalid bandwidth m = {m} for a series of length n = {n}: the \
                 number of low Fourier frequencies must satisfy {min} <= m <= {max} \
                 (= (n-1)/2) so the log-periodogram regression is identified and \
                 every ordinate is a strictly interior frequency below Nyquist"
            ),
            Self::NonPositivePeriodogram { j } => write!(
                f,
                "periodogram ordinate at Fourier frequency j = {j} is non-positive; \
                 its logarithm is undefined — the series is degenerate at that \
                 frequency, supply a longer or less degenerate series"
            ),
            Self::OptimizationFailed { reason } => write!(
                f,
                "local-Whittle minimization did not reach a trustworthy interior \
                 optimum ({reason}); the memory estimate cannot be reported"
            ),
            Self::Spectral(e) => write!(f, "periodogram layer error: {e}"),
            Self::Hac(e) => write!(f, "OLS layer error: {e}"),
            Self::Optim(e) => write!(f, "optimization layer error: {e}"),
        }
    }
}

impl std::error::Error for LongMemoryError {}

impl From<SpectralError> for LongMemoryError {
    fn from(e: SpectralError) -> Self {
        Self::Spectral(e)
    }
}

impl From<HacError> for LongMemoryError {
    fn from(e: HacError) -> Self {
        Self::Hac(e)
    }
}

impl From<OptimError> for LongMemoryError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}
