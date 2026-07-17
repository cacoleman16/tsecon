//! Error types for `tsecon-midas`.
//!
//! Every fallible entry point in this crate returns `Result<_, MidasError>`;
//! nothing in the non-test code path panics. Errors that originate in the
//! shared HAC/OLS engine ([`tsecon_hac`]) or the shared optimizer
//! ([`tsecon_optim`]) are wrapped verbatim so the caller sees the underlying
//! statistical explanation, following the library's "errors that teach"
//! pillar.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_optim::OptimError;

/// Errors produced by the MIDAS weighting, design-assembly, and estimation
/// machinery in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum MidasError {
    /// The number of high-frequency lags `K` is too small for the requested
    /// construction (weight functions and the Beta lag support need `K >= 1`;
    /// the Beta `x_k = (k - 1)/(K - 1)` grid needs `K >= 2` to be
    /// non-degenerate).
    InvalidLagCount {
        /// Which construction rejected the lag count.
        what: &'static str,
        /// The lag count supplied.
        k: usize,
        /// The minimum lag count required.
        needed: usize,
    },
    /// A weight-function hyperparameter is outside its admissible domain (the
    /// Beta shape parameters must be strictly positive; every hyperparameter
    /// must be finite).
    InvalidWeightParam {
        /// Which weight function rejected the parameter.
        what: &'static str,
        /// Name of the offending hyperparameter.
        name: &'static str,
        /// The value supplied.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The requested Almon polynomial-distributed-lag degree is not usable:
    /// the `K x (degree + 1)` basis must have at least as many lags as basis
    /// columns (`degree + 1 <= K`) or the restricted design is rank-deficient.
    InvalidPolynomialDegree {
        /// The requested polynomial degree.
        degree: usize,
        /// The number of high-frequency lags available.
        k: usize,
    },
    /// The input contains a NaN or infinite value. MIDAS estimators never skip
    /// missing values silently; align the release calendar and drop or impute
    /// ragged-edge gaps before estimating.
    NonFinite {
        /// Which input the offending value was found in.
        what: &'static str,
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// A weight vector could not be normalized because its (unnormalized) mass
    /// underflowed to zero or overflowed to a non-finite total — the
    /// hyperparameters are so extreme that every lag weight is numerically
    /// zero or infinite.
    DegenerateWeights {
        /// Which weight function produced the degenerate mass.
        what: &'static str,
    },
    /// A high-frequency series is too short to build the requested number of
    /// stacked lags at the given frequency ratio for even one low-frequency
    /// period.
    SeriesTooShort {
        /// Which builder needed more data.
        what: &'static str,
        /// The number of high-frequency observations supplied.
        n: usize,
        /// The minimum number of high-frequency observations required.
        needed: usize,
    },
    /// Two related inputs disagree in length (e.g. the low-frequency target
    /// and a high-frequency block that should span a whole number of
    /// low-frequency periods).
    DimensionMismatch {
        /// What the mismatch is about.
        what: &'static str,
        /// The expected length.
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },
    /// An error surfaced by the shared HAC/OLS engine while fitting the
    /// (U-)MIDAS design (empty/collinear design, no residual degrees of
    /// freedom, ...); see [`tsecon_hac::HacError`].
    Ols(HacError),
    /// An error surfaced by the shared optimizer while fitting weighted MIDAS
    /// by nonlinear least squares; see [`tsecon_optim::OptimError`].
    Optim(OptimError),
}

impl fmt::Display for MidasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MidasError::InvalidLagCount { what, k, needed } => write!(
                f,
                "{what}: K = {k} high-frequency lags is too few; needs at least \
                 {needed} (the weight functions are defined over lags \
                 k = 1..=K)"
            ),
            MidasError::InvalidWeightParam {
                what,
                name,
                value,
                requirement,
            } => write!(
                f,
                "{what}: weight hyperparameter `{name}` = {value} is invalid; \
                 requires {requirement}"
            ),
            MidasError::InvalidPolynomialDegree { degree, k } => write!(
                f,
                "Almon PDL basis: degree {degree} needs degree + 1 = {} basis \
                 columns but only K = {k} high-frequency lags are available; \
                 the restricted design would be rank-deficient — lower the \
                 degree or add lags",
                degree + 1
            ),
            MidasError::NonFinite { what, index, value } => write!(
                f,
                "{what}: contains a non-finite value ({value}) at index {index}; \
                 MIDAS estimators do not skip missing values silently — align \
                 the calendar and drop or impute NaN/inf observations first"
            ),
            MidasError::DegenerateWeights { what } => write!(
                f,
                "{what}: the unnormalized weights underflowed to zero or \
                 overflowed to non-finite mass, so they cannot be normalized \
                 to sum one; the hyperparameters are numerically extreme"
            ),
            MidasError::SeriesTooShort { what, n, needed } => write!(
                f,
                "{what}: {n} high-frequency observations is too few; needs at \
                 least {needed} to build one low-frequency design row — supply \
                 more data or reduce K / the frequency ratio"
            ),
            MidasError::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "{what}: expected length {expected} but got {got}; the \
                 high-frequency block must span a whole number of \
                 low-frequency periods aligned with the target"
            ),
            MidasError::Ols(e) => write!(f, "U-MIDAS OLS fit failed: {e}"),
            MidasError::Optim(e) => write!(f, "weighted MIDAS NLS failed: {e}"),
        }
    }
}

impl std::error::Error for MidasError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MidasError::Ols(e) => Some(e),
            MidasError::Optim(e) => Some(e),
            _ => None,
        }
    }
}

impl From<HacError> for MidasError {
    fn from(e: HacError) -> Self {
        MidasError::Ols(e)
    }
}

impl From<OptimError> for MidasError {
    fn from(e: OptimError) -> Self {
        MidasError::Optim(e)
    }
}
