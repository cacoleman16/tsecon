//! Error type shared by the copula estimators.
//!
//! Every fallible public function in this crate returns
//! `Result<_, CopulaError>`; nothing outside `#[cfg(test)]` panics on user
//! input. Messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters, and what the caller can do.

use core::fmt;

use tsecon_optim::OptimError;

/// Errors produced by the copula estimators in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum CopulaError {
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
    /// The two margins have different lengths.
    LengthMismatch {
        /// Length of the first column.
        n1: usize,
        /// Length of the second column.
        n2: usize,
    },
    /// Fewer observations than the documented minimum
    /// ([`crate::MIN_OBS`]): a dependence parameter estimated from a
    /// handful of pairs is noise.
    TooFewObservations {
        /// Observations supplied.
        n: usize,
        /// The documented minimum.
        min: usize,
    },
    /// A pseudo-observation was outside the open interval `(0, 1)`.
    OutOfUnitInterval {
        /// Name of the offending column.
        what: &'static str,
        /// Index of the first offending entry.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// A column is (numerically) constant, so ranks — and Kendall's tau —
    /// are undefined.
    Degenerate {
        /// Which input degenerated.
        what: &'static str,
    },
    /// The empirical dependence is (numerically) perfectly monotone,
    /// putting every family's dependence parameter at its boundary.
    PerfectDependence {
        /// The empirical Kendall tau.
        tau: f64,
    },
    /// Clayton/Gumbel requested on data with non-positive Kendall tau.
    NegativeDependence {
        /// The family that was requested.
        family: &'static str,
        /// The empirical Kendall tau.
        tau: f64,
    },
    /// A Kendall tau outside the family's invertible range was passed to
    /// the tau-to-parameter map.
    TauOutOfRange {
        /// The family whose map was requested.
        family: &'static str,
        /// The offending tau.
        tau: f64,
        /// The invertible range, stated for the message.
        requirement: &'static str,
    },
    /// A copula parameter outside the family's domain was passed to an
    /// evaluator.
    InvalidParameter {
        /// The family.
        family: &'static str,
        /// Parameter name.
        name: &'static str,
        /// The offending value.
        value: f64,
        /// The domain, stated for the message.
        requirement: &'static str,
    },
    /// A parameter vector of the wrong length was passed to an evaluator.
    WrongParamCount {
        /// The family.
        family: &'static str,
        /// Parameters expected.
        expected: usize,
        /// Parameters received.
        got: usize,
    },
    /// An unknown family name was requested.
    UnknownFamily {
        /// The name that failed to parse.
        name: String,
    },
    /// An unknown fitting method was requested.
    UnknownMethod {
        /// The name that failed to parse.
        name: String,
    },
    /// `copula_select` was called with an empty family list.
    EmptyFamilies,
    /// `copula_select` was called with a family listed twice.
    DuplicateFamily {
        /// The duplicated family.
        family: &'static str,
    },
    /// Every family in a `copula_select` call was skipped.
    AllFamiliesSkipped {
        /// The per-family reasons, joined for the message.
        reasons: String,
    },
    /// No admissible starting value gave a finite log-likelihood.
    NoAdmissibleStart {
        /// Which fit failed.
        what: &'static str,
    },
    /// An error propagated from the optimizer.
    Optim(OptimError),
    /// An error propagated from a `tsecon-stats` special function
    /// (unreachable for validated inputs; surfaced rather than panicking).
    Stats(String),
}

impl fmt::Display for CopulaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput { what } => {
                write!(f, "empty input: {what}; supply at least one observation")
            }
            Self::NonFinite { what, index } => write!(
                f,
                "non-finite value (NaN or infinity) in {what} at index \
                 {index}; the copula estimators do not skip missing values \
                 silently — clean or drop the affected observations first"
            ),
            Self::LengthMismatch { n1, n2 } => write!(
                f,
                "the two margins have different lengths ({n1} vs {n2}); a \
                 copula is fitted to paired observations — align the series \
                 first"
            ),
            Self::TooFewObservations { n, min } => write!(
                f,
                "only {n} paired observations but at least {min} are \
                 required; a dependence parameter fitted to fewer pairs is \
                 noise dressed as a copula — supply a longer sample"
            ),
            Self::OutOfUnitInterval { what, index, value } => write!(
                f,
                "{what}[{index}] = {value} is outside the open interval \
                 (0, 1); copula_fit expects probability-scale \
                 pseudo-observations, not raw data — rank/PIT-transform \
                 first (pseudo_obs(x) does this in one call, and its \
                 rank/(n+1) scaling keeps every value strictly inside \
                 (0, 1), which the quantile transforms require)"
            ),
            Self::Degenerate { what } => write!(
                f,
                "{what} is numerically constant, so ranks — and Kendall's \
                 tau — are undefined; a copula needs variation in both \
                 margins — check for a constant or heavily discretized \
                 series"
            ),
            Self::PerfectDependence { tau } => write!(
                f,
                "the empirical Kendall tau is {tau:.6}, numerically a \
                 perfectly monotone relationship; every parametric copula's \
                 dependence parameter sits at its boundary there and no MLE \
                 exists — the two margins are (anti)comonotone, which is a \
                 functional relationship, not a stochastic dependence to \
                 model"
            ),
            Self::NegativeDependence { family, tau } => write!(
                f,
                "the {family} copula models positive dependence only, but \
                 the empirical Kendall tau is {tau:.4}; in this bivariate \
                 slice rotations (survival copulas) are not implemented — \
                 use the frank or gaussian family, which cover negative \
                 dependence"
            ),
            Self::TauOutOfRange {
                family,
                tau,
                requirement,
            } => write!(
                f,
                "Kendall tau {tau} is outside the invertible range for the \
                 {family} tau-to-parameter map ({requirement})"
            ),
            Self::InvalidParameter {
                family,
                name,
                value,
                requirement,
            } => write!(
                f,
                "invalid {family} copula parameter {name} = {value}; the \
                 domain is {requirement}"
            ),
            Self::WrongParamCount {
                family,
                expected,
                got,
            } => write!(
                f,
                "the {family} copula takes {expected} parameter(s), got \
                 {got}"
            ),
            Self::UnknownFamily { name } => write!(
                f,
                "unknown copula family {name:?}; expected \"gaussian\", \
                 \"t\", \"clayton\", \"gumbel\", or \"frank\""
            ),
            Self::UnknownMethod { name } => write!(
                f,
                "unknown fitting method {name:?}; expected \"mle\" \
                 (maximum likelihood) or \"tau\" (Kendall-tau inversion)"
            ),
            Self::EmptyFamilies => write!(
                f,
                "copula_select needs at least one family; pass e.g. \
                 [\"gaussian\", \"t\", \"clayton\", \"gumbel\", \"frank\"]"
            ),
            Self::DuplicateFamily { family } => write!(
                f,
                "family \"{family}\" is listed more than once in \
                 copula_select; each family is fitted once"
            ),
            Self::AllFamiliesSkipped { reasons } => write!(
                f,
                "every requested family was skipped, so there is nothing \
                 to rank: {reasons}"
            ),
            Self::NoAdmissibleStart { what } => write!(
                f,
                "no admissible starting value gives a finite \
                 log-likelihood for {what}; the input is degenerate or on \
                 a scale that overflows"
            ),
            Self::Optim(e) => write!(f, "optimizer error: {e}"),
            Self::Stats(e) => write!(f, "special-function error: {e}"),
        }
    }
}

impl std::error::Error for CopulaError {}

impl From<OptimError> for CopulaError {
    fn from(e: OptimError) -> Self {
        Self::Optim(e)
    }
}

impl From<tsecon_stats::StatsError> for CopulaError {
    fn from(e: tsecon_stats::StatsError) -> Self {
        Self::Stats(e.to_string())
    }
}
