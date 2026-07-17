//! Error types for `tsecon-realized`.
//!
//! Every fallible entry point in this crate returns
//! `Result<_, RealizedError>`; nothing in the non-test code path panics.
//! Messages follow the library's "errors that teach" pillar: they state
//! what went wrong, why it matters, and what the caller can do about it.
//! HAR estimation delegates its linear algebra and standard errors to
//! `tsecon-hac`, so [`RealizedError::Hac`] transparently forwards any
//! [`tsecon_hac::HacError`] raised by that solve.

use core::fmt;

use tsecon_hac::HacError;

/// Errors produced by the realized-measure and HAR machinery in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum RealizedError {
    /// The intraday return / price series has too few observations for the
    /// requested measure (e.g. bipower variation needs at least two
    /// returns; tripower quarticity needs three).
    TooFewObservations {
        /// Which estimator needed more data.
        what: &'static str,
        /// The number of observations supplied.
        n: usize,
        /// The minimum number of observations required.
        needed: usize,
    },
    /// The input contains a NaN or infinite value. Realized measures never
    /// skip missing values silently; clean or impute the series first.
    NonFinite {
        /// Which input the offending value was found in.
        what: &'static str,
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// An OHLC bar is inconsistent: the high is below the low, or a price
    /// used inside a logarithm is non-positive.
    InvalidOhlc {
        /// Which range estimator rejected the bar.
        what: &'static str,
        /// Index of the offending bar.
        index: usize,
        /// A human-readable statement of the violated constraint.
        detail: &'static str,
    },
    /// The `start` burn-in index leaves no HAR estimation sample: the design
    /// needs `start >= 22` (so the monthly regressor `mean(RV[t-23..t-1])`
    /// is defined) and at least one target row `t = start+1 .. n-1`.
    InsufficientHarSample {
        /// The burn-in index supplied.
        start: usize,
        /// The length of the RV series supplied.
        n: usize,
    },
    /// A measure that must be strictly positive to studentize the jump test
    /// (realized variance or bipower variation) came out zero, so the
    /// ratio statistic is undefined; the series is (numerically) constant.
    DegenerateSeries {
        /// Which diagnostic found the degenerate series.
        what: &'static str,
    },
    /// The HAR solve delegated to `tsecon-hac` failed; the wrapped
    /// [`HacError`] carries the specifics (collinear design, no residual
    /// degrees of freedom, and so on).
    Hac(HacError),
}

impl From<HacError> for RealizedError {
    fn from(e: HacError) -> Self {
        RealizedError::Hac(e)
    }
}

impl fmt::Display for RealizedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RealizedError::TooFewObservations { what, n, needed } => write!(
                f,
                "{what}: series has {n} intraday observations but needs at \
                 least {needed}; supply a finer grid or a longer sample"
            ),
            RealizedError::NonFinite { what, index, value } => write!(
                f,
                "{what}: contains a non-finite value ({value}) at index \
                 {index}; realized measures do not skip missing values \
                 silently — drop or impute NaN/inf observations first"
            ),
            RealizedError::InvalidOhlc {
                what,
                index,
                detail,
            } => write!(
                f,
                "{what}: OHLC bar {index} is inconsistent ({detail}); check \
                 the bar ordering (open, high, low, close) and that prices \
                 are strictly positive"
            ),
            RealizedError::InsufficientHarSample { start, n } => write!(
                f,
                "HAR design: burn-in start = {start} with an RV series of \
                 length {n} leaves no estimation sample; the Corsi (2009) \
                 monthly regressor needs start >= 22 and at least one target \
                 row t in start+1 ..= n-1 (so n >= start + 2)"
            ),
            RealizedError::DegenerateSeries { what } => write!(
                f,
                "{what}: realized variance or bipower variation is \
                 (numerically) zero, so the studentized jump ratio is \
                 undefined; the series carries no variation"
            ),
            RealizedError::Hac(e) => write!(f, "HAR OLS/HAC solve failed: {e}"),
        }
    }
}

impl std::error::Error for RealizedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RealizedError::Hac(e) => Some(e),
            _ => None,
        }
    }
}
