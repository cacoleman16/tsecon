//! Error type shared by the EVT estimators.
//!
//! Every fallible public function in this crate returns
//! `Result<_, EvtError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters, and what the caller can do.

use core::fmt;

use tsecon_optim::OptimError;

/// Errors produced by the extreme-value estimators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum EvtError {
    /// A required input series was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// An input contained a NaN or infinite entry.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
        /// Index of the first offending entry.
        index: usize,
    },
    /// The threshold quantile was outside the open interval `(0, 1)`.
    InvalidQuantile {
        /// The offending quantile.
        q: f64,
    },
    /// Fewer exceedances over the threshold than the documented minimum
    /// ([`crate::MIN_EXCEEDANCES`]): the GPD MLE and its observed-information
    /// standard errors are meaningless on a handful of points.
    TooFewExceedances {
        /// Exceedances found (strictly above the threshold).
        n_exceed: usize,
        /// The documented minimum.
        min: usize,
        /// The threshold that was used.
        threshold: f64,
    },
    /// Fewer block maxima than the documented minimum
    /// ([`crate::MIN_MAXIMA`]): the three-parameter GEV MLE is not
    /// defensible on a handful of maxima.
    TooFewMaxima {
        /// Maxima available after blocking.
        n_maxima: usize,
        /// The documented minimum.
        min: usize,
    },
    /// `block_size` was zero or exceeded the series length.
    InvalidBlockSize {
        /// The offending block size.
        block_size: usize,
        /// The series length.
        n: usize,
    },
    /// A tail probability was outside the open interval `(0, 1)`.
    InvalidTailProb {
        /// The offending probability.
        p: f64,
    },
    /// A tail probability did not reach beyond the threshold: the POT
    /// quantile formula only extrapolates *outward* from the threshold,
    /// which requires `1 - p < n_exceed / n`.
    TailProbNotBeyondThreshold {
        /// The offending probability.
        p: f64,
        /// The empirical exceedance rate `n_exceed / n`.
        exceed_rate: f64,
    },
    /// A return period was not strictly greater than 1.
    InvalidReturnPeriod {
        /// The offending return period.
        t: f64,
    },
    /// A scale parameter passed to a log-likelihood evaluator was not
    /// strictly positive and finite.
    InvalidScale {
        /// The offending scale.
        scale: f64,
    },
    /// The exceedances (or maxima) are numerically constant, so the scale
    /// MLE degenerates to zero and the likelihood is unbounded.
    Degenerate {
        /// Which input degenerated.
        what: &'static str,
    },
    /// No admissible starting value gave a finite log-likelihood.
    NoAdmissibleStart {
        /// Which fit failed.
        what: &'static str,
    },
    /// An error propagated from the optimizer.
    Optim(OptimError),
}

impl fmt::Display for EvtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => {
                write!(f, "empty input: {what}; supply at least one observation")
            }
            Self::NonFinite { what, index } => write!(
                f,
                "non-finite value (NaN or infinity) in {what} at index \
                 {index}; the EVT estimators do not skip missing values \
                 silently — clean or drop the affected observations first"
            ),
            Self::InvalidQuantile { q } => write!(
                f,
                "invalid threshold quantile {q}; it must lie strictly inside \
                 (0, 1) — 0.90 (the top decile as exceedances) is the \
                 conventional POT default"
            ),
            Self::TooFewExceedances {
                n_exceed,
                min,
                threshold,
            } => write!(
                f,
                "only {n_exceed} observations exceed the threshold \
                 {threshold} but at least {min} are required; a GPD fitted \
                 to fewer points is noise dressed as a tail model — lower \
                 the threshold (or the threshold quantile), or supply a \
                 longer series"
            ),
            Self::TooFewMaxima { n_maxima, min } => write!(
                f,
                "only {n_maxima} block maxima but at least {min} are \
                 required; a three-parameter GEV fitted to fewer maxima is \
                 not defensible — use a smaller block size, supply a longer \
                 series, or consider the peaks-over-threshold route \
                 (gpd_fit), which uses the data more efficiently"
            ),
            Self::InvalidBlockSize { block_size, n } => write!(
                f,
                "invalid block_size {block_size} for a series of length \
                 {n}; it must satisfy 1 <= block_size <= n (a trailing \
                 partial block is dropped), and n / block_size must leave \
                 enough maxima to fit three GEV parameters"
            ),
            Self::InvalidTailProb { p } => write!(
                f,
                "invalid tail probability {p}; each entry of p_tail must \
                 lie strictly inside (0, 1), e.g. [0.99, 0.995, 0.999]"
            ),
            Self::TailProbNotBeyondThreshold { p, exceed_rate } => write!(
                f,
                "tail probability {p} does not reach beyond the threshold: \
                 the POT VaR/ES formulas extrapolate the fitted GPD tail \
                 outward from the threshold, which requires 1 - p < \
                 exceedance rate ({exceed_rate:.6}); request a more extreme \
                 p, or raise the threshold quantile above p"
            ),
            Self::InvalidReturnPeriod { t } => write!(
                f,
                "invalid return period {t}; each return period must be \
                 strictly greater than 1 block (T = 10 means the level \
                 exceeded once every 10 blocks on average, i.e. the \
                 1 - 1/10 GEV quantile)"
            ),
            Self::InvalidScale { scale } => write!(
                f,
                "invalid scale parameter {scale}; the GPD/GEV scale must be \
                 strictly positive and finite"
            ),
            Self::Degenerate { what } => write!(
                f,
                "{what} are numerically constant, so the scale MLE \
                 degenerates to zero and the likelihood is unbounded; a \
                 tail model needs dispersion — check for a constant or \
                 heavily discretized series"
            ),
            Self::NoAdmissibleStart { what } => write!(
                f,
                "no admissible starting value gives a finite \
                 log-likelihood for {what}; the input is degenerate or on \
                 a scale that overflows"
            ),
            Self::Optim(e) => write!(f, "optimizer error: {e}"),
        }
    }
}

impl std::error::Error for EvtError {}

impl From<OptimError> for EvtError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}
