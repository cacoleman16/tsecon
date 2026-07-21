//! Error type shared by the structural-break estimators.
//!
//! Every fallible public function in this crate returns
//! `Result<_, BreaksError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters, and what the caller can do.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_stats::StatsError;

/// Errors produced by the structural-break estimators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum BreaksError {
    /// A required input series was empty.
    EmptyInput {
        /// Name of the offending argument.
        what: &'static str,
    },
    /// The regressor matrix has zero columns.
    NoRegressors,
    /// The response and a regressor column have incompatible lengths.
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
    },
    /// A configuration scalar was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// The minimal segment length `h = ceil(trim * T)` leaves fewer
    /// observations per regime than regression parameters.
    TrimTooSmall {
        /// The minimal segment length implied by `trim`.
        h: usize,
        /// The number of regressors whose coefficients switch.
        q: usize,
        /// The sample size.
        t: usize,
    },
    /// The requested number of breaks cannot fit into the sample: every
    /// one of the `max_breaks + 1` regimes needs at least `h` observations.
    InfeasibleBreaks {
        /// The requested maximum number of breaks.
        max_breaks: usize,
        /// The minimal segment length `h = ceil(trim * T)`.
        h: usize,
        /// The sample size.
        t: usize,
    },
    /// The published critical-value / p-value tables cover at most 10
    /// switching regressors.
    UnsupportedQ {
        /// The number of regressors supplied.
        q: usize,
    },
    /// `bai_perron` selection uses the published Bai-Perron 5% critical
    /// values, tabulated only for trimming in {0.05, 0.10, 0.15, 0.20, 0.25}.
    UnsupportedTrim {
        /// The trimming fraction supplied (times 100, rounded).
        trim_pct: usize,
    },
    /// The sample is too short for the requested statistic.
    TooShort {
        /// The sample size.
        t: usize,
        /// The minimum sample size required.
        needed: usize,
        /// Which computation needed it.
        what: &'static str,
    },
    /// A segment's regressor cross-moment matrix was numerically singular:
    /// the columns of `x` are collinear over that stretch of observations.
    Singular {
        /// First observation (0-indexed) of the offending segment.
        start: usize,
        /// Last observation (0-indexed) of the offending segment.
        end: usize,
    },
    /// A candidate split produced a zero residual sum of squares, so the
    /// F statistic's denominator is degenerate.
    DegenerateFit {
        /// Which statistic hit the degenerate denominator.
        what: &'static str,
    },
    /// An error propagated from the OLS layer (this crate never
    /// reimplements the per-regime least squares it reports).
    Hac(HacError),
    /// An error propagated from the chi-square tail evaluation.
    Stats(StatsError),
}

impl fmt::Display for BreaksError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => write!(
                f,
                "empty input: {what}; supply a response and T x q regressor \
                 columns of equal positive length"
            ),
            Self::NoRegressors => write!(
                f,
                "no regressor columns: the pure-structural-change model needs \
                 at least one column whose coefficient can break (include the \
                 constant explicitly, e.g. a column of ones)"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected length {expected}, got {got})"
            ),
            Self::NonFinite { what } => write!(
                f,
                "non-finite value (NaN or infinity) in {what}; the break \
                 estimators do not skip missing values silently — clean the \
                 data first"
            ),
            Self::InvalidArgument { what } => write!(f, "invalid argument: {what}"),
            Self::TrimTooSmall { h, q, t } => write!(
                f,
                "trimming too small: the minimal segment length h = {h} must \
                 exceed the q = {q} regressors so every regime's OLS is \
                 identified with a residual; raise trim to at least \
                 {:.3} (= (q + 1) / T with T = {t})",
                (*q as f64 + 1.0) / *t as f64
            ),
            Self::InfeasibleBreaks { max_breaks, h, t } => write!(
                f,
                "max_breaks = {max_breaks} is infeasible: {} regimes of at \
                 least h = {h} observations need {} > T = {t}; lower \
                 max_breaks to at most {} or reduce trim",
                max_breaks + 1,
                (max_breaks + 1) * h,
                (t / h).saturating_sub(1)
            ),
            Self::UnsupportedQ { q } => write!(
                f,
                "q = {q} switching regressors exceeds the published tables \
                 (Bai-Perron critical values and Hansen p-value surfaces are \
                 tabulated for q = 1..10); reduce the design to at most 10 \
                 columns"
            ),
            Self::UnsupportedTrim { trim_pct } => write!(
                f,
                "trim = 0.{trim_pct:02} has no published Bai-Perron sequential \
                 critical values; use one of 0.05, 0.10, 0.15, 0.20, 0.25 \
                 (0.15 is the standard choice)"
            ),
            Self::TooShort { t, needed, what } => write!(
                f,
                "sample too short for {what}: T = {t} but at least {needed} \
                 observations are required at this trimming; supply a longer \
                 series or reduce trim"
            ),
            Self::Singular { start, end } => write!(
                f,
                "segment X'X over observations {start}..={end} is numerically \
                 singular: the regressor columns are collinear on that \
                 stretch (a dummy that is constant within candidate regimes \
                 is the usual cause); drop or merge the offending columns"
            ),
            Self::DegenerateFit { what } => write!(
                f,
                "{what}: a candidate segmentation fits the data exactly \
                 (zero residual sum of squares), so the F statistic is \
                 undefined; the response is deterministic given x on some \
                 admissible segment — this estimator needs noisy data"
            ),
            Self::Hac(e) => write!(f, "OLS layer error: {e}"),
            Self::Stats(e) => write!(f, "chi-square tail error: {e}"),
        }
    }
}

impl std::error::Error for BreaksError {}

impl From<HacError> for BreaksError {
    fn from(e: HacError) -> Self {
        Self::Hac(e)
    }
}

impl From<StatsError> for BreaksError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
