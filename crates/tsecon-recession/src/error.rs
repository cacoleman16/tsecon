//! Error type shared by the recession-probability estimators.
//!
//! Every fallible public function in this crate returns
//! `Result<_, RecessionError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they say
//! what went wrong, why it matters, and what the caller can do about it.

use core::fmt;

use tsecon_optim::OptimError;
use tsecon_stats::StatsError;

/// Errors produced by the static and dynamic probit/logit estimators.
#[derive(Debug, Clone, PartialEq)]
pub enum RecessionError {
    /// A required series was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// The design has no columns (no predictors, not even a constant).
    NoRegressors,
    /// The response `y` and a design column, or two design columns, have
    /// incompatible lengths.
    DimensionMismatch {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The length that was expected.
        expected: usize,
        /// The length that was received.
        got: usize,
    },
    /// The response contained a value other than `0.0` or `1.0`. A recession
    /// indicator is binary; this crate never silently rounds or thresholds.
    NonBinaryResponse {
        /// The observation index of the first offending value.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// The response is degenerate: every observation is `0`, or every
    /// observation is `1`. The likelihood is then maximized by pushing the
    /// intercept to +/- infinity, so no finite MLE exists.
    Degenerate {
        /// The number of ones in `y` (`0` or `n`).
        ones: usize,
        /// The sample size `n`.
        n: usize,
    },
    /// Fewer usable observations than parameters (`n <= k`): the model has no
    /// residual degrees of freedom and the covariance is undefined.
    DegreesOfFreedom {
        /// The sample size `n`.
        n: usize,
        /// The number of parameters `k`.
        k: usize,
    },
    /// The data are (quasi-)completely separated: some linear combination of
    /// the predictors perfectly predicts `y`, so the likelihood is maximized
    /// only in the limit as a coefficient diverges and no finite MLE exists.
    /// Detected when the fitted probabilities saturate to the observed `y`.
    Separation,
    /// The observed-information matrix (negative Hessian) was numerically
    /// singular or indefinite, so standard errors could not be formed — a
    /// symptom of collinear predictors or near-separation.
    SingularInformation,
    /// An error propagated from the optimizer.
    Optim(OptimError),
    /// An error propagated from the statistics layer (CDF / chi-square).
    Stats(StatsError),
}

impl fmt::Display for RecessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply a binary response and at least one \
                 predictor column (typically a constant plus the term spread)"
            ),
            Self::NoRegressors => write!(
                f,
                "no regressors: the design has zero columns; include at least a \
                 constant column (a column of ones) so the model has an intercept"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected length {expected}, got {got})"
            ),
            Self::NonBinaryResponse { index, value } => write!(
                f,
                "non-binary response: y[{index}] = {value} is neither 0 nor 1; a \
                 recession indicator must be coded in {{0, 1}} — this crate does not \
                 threshold or round for you"
            ),
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; the estimators do not \
                 skip missing values silently — clean the data first"
            ),
            Self::Degenerate { ones, n } => write!(
                f,
                "degenerate response: {ones} of {n} observations are 1 (all-zero or \
                 all-one y); the intercept-only likelihood is maximized at +/- \
                 infinity, so no finite MLE exists — you need both recession and \
                 non-recession observations"
            ),
            Self::DegreesOfFreedom { n, k } => write!(
                f,
                "n = {n} observations with k = {k} parameters leaves no residual \
                 degrees of freedom (requires n > k); the coefficient covariance is \
                 undefined — supply a longer series or fewer predictors"
            ),
            Self::Separation => write!(
                f,
                "(quasi-)complete separation: a linear combination of the predictors \
                 perfectly predicts the recession indicator, so the maximum-likelihood \
                 coefficients diverge and no finite MLE exists; drop the separating \
                 predictor, add observations, or use a penalized estimator"
            ),
            Self::SingularInformation => write!(
                f,
                "singular information matrix: the negative Hessian could not be \
                 inverted, so standard errors are undefined; the usual cause is \
                 collinear predictors or near-separation"
            ),
            Self::Optim(e) => write!(f, "optimizer error: {e}"),
            Self::Stats(e) => write!(f, "statistics-layer error: {e}"),
        }
    }
}

impl std::error::Error for RecessionError {}

impl From<OptimError> for RecessionError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}

impl From<StatsError> for RecessionError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
