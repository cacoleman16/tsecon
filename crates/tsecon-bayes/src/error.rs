//! Error type shared by the Bayesian foundations layer.

use core::fmt;

use tsecon_linalg::LinalgError;
use tsecon_rng::RngError;
use tsecon_ssm::SsmError;
use tsecon_stats::StatsError;

/// Errors returned by the Bayesian foundations layer.
///
/// Every fallible public function in this crate returns
/// `Result<_, BayesError>`; no library code path panics on user input.
#[derive(Debug, Clone, PartialEq)]
pub enum BayesError {
    /// An error bubbled up from the structured linear-algebra layer
    /// (Cholesky, companion form, ...).
    Linalg(LinalgError),
    /// An error bubbled up from the special-function / distribution layer
    /// (inverse normal CDF, chi-squared quantile, ...).
    Stats(StatsError),
    /// An error bubbled up from the state-space engine (Kalman filter,
    /// model validation, ...).
    Ssm(SsmError),
    /// An error bubbled up from the random-stream layer.
    Rng(RngError),
    /// Two inputs (or an input and a model dimension) have incompatible
    /// sizes.
    Dimension {
        /// Description of the constraint that was violated.
        what: &'static str,
        /// The size that was expected.
        expected: usize,
        /// The size that was received.
        got: usize,
    },
    /// A scalar or structural argument was outside its valid domain.
    InvalidArgument {
        /// Description of the domain violation.
        what: &'static str,
    },
    /// An input contained a NaN or infinity.
    NonFinite {
        /// Name of the offending argument.
        what: &'static str,
        /// Zero-based `(row, column)` of the first offending entry, when
        /// the check scanned a user-supplied array.
        at: Option<(usize, usize)>,
    },
    /// The sample is too short for the requested specification.
    InsufficientObservations {
        /// Minimum number of usable observations required.
        needed: usize,
        /// Number of usable observations available.
        got: usize,
        /// What those observations are needed for.
        what: &'static str,
    },
    /// The exact-diffuse initialization had not collapsed after the first
    /// time period, so the stored filtered/predicted moments the backward
    /// sampler conditions on are not proper distributions. Use a `Known`
    /// or `Stationary` initialization, or provide enough identifying
    /// observations in the first period.
    DiffuseNotCollapsed {
        /// Length of the diffuse period reported by the filter.
        periods: usize,
    },
    /// An internal iterative algorithm exhausted its iteration budget.
    NoConvergence {
        /// Name of the algorithm that failed to converge.
        what: &'static str,
    },
    /// A size argument (a horizon, a draw or chain count, a grid size) would
    /// push the call past its budget; refused, naming the argument, before
    /// anything is allocated (see [`crate::budget`]).
    TooLarge {
        /// The Python keyword the refusal names.
        arg: &'static str,
        /// The value passed.
        value: usize,
        /// The largest admissible value with every other argument as
        /// passed; `None` when no value of `arg` fits.
        max: Option<usize>,
        /// What the budget counts.
        what: &'static str,
        /// The budget, as quoted (e.g. [`crate::budget::MEMORY_BUDGET`]).
        budget: &'static str,
    },
    /// The univariate AR(`scale_ar`) scale regressions that calibrate the
    /// Minnesota prior need more rows than the sample has.
    ScaleArTooLong {
        /// The `scale_ar` lag order requested.
        scale_ar: usize,
        /// Rows the regressions need (`2 scale_ar + 2`).
        needed: usize,
        /// Rows the sample has.
        got: usize,
        /// The largest `scale_ar` the sample supports (`None` below four
        /// rows, where even `scale_ar = 1` does not fit).
        max: Option<usize>,
    },
}

impl fmt::Display for BayesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra failure: {e}"),
            Self::Stats(e) => write!(f, "special-function failure: {e}"),
            Self::Ssm(e) => write!(f, "state-space failure: {e}"),
            Self::Rng(e) => write!(f, "random-stream failure: {e}"),
            Self::Dimension {
                what,
                expected,
                got,
            } => write!(
                f,
                "dimension mismatch: {what} (expected {expected}, got {got})"
            ),
            Self::InvalidArgument { what } => write!(f, "{what}"),
            Self::NonFinite {
                what,
                at: Some((row, col)),
            } => write!(
                f,
                "non-finite value (NaN or inf) in {what} at row {row}, column {col}: the \
                 Bayesian samplers have no missing-value handling, so one gap would \
                 corrupt every draw — drop or impute those rows first \
                 (pandas: df.dropna())"
            ),
            Self::NonFinite { what, at: None } => write!(
                f,
                "non-finite value (NaN or inf) in {what}: check the inputs for missing \
                 values or magnitudes large enough to overflow"
            ),
            Self::InsufficientObservations { needed, got, what } => write!(
                f,
                "this model needs at least {needed} rows for {what}, but got {got}. \
                 Use fewer lags, drop a series, or supply a longer sample."
            ),
            Self::DiffuseNotCollapsed { periods } => write!(
                f,
                "exact-diffuse initialization still active after {periods} periods: \
                 the backward sampler needs proper filtered moments (use Known or \
                 Stationary initialization, or more informative first-period data)"
            ),
            Self::NoConvergence { what } => write!(
                f,
                "{what} failed to converge; the posterior is very poorly conditioned — \
                 rescale the data to comparable magnitudes, or tighten the prior"
            ),
            Self::TooLarge {
                arg,
                value,
                max,
                what,
                budget,
            } => {
                write!(
                    f,
                    "{arg} = {value} is too large: {what} would exceed {budget}, refused \
                     before anything is allocated; "
                )?;
                match max {
                    Some(m) => write!(
                        f,
                        "pass {arg} <= {m} (the largest value that fits with the other \
                         arguments as passed)"
                    ),
                    None => write!(
                        f,
                        "no value of {arg} fits with the other arguments as passed, so \
                         reduce those as well"
                    ),
                }
            }
            Self::ScaleArTooLong {
                scale_ar,
                needed,
                got,
                max,
            } => {
                write!(
                    f,
                    "scale_ar = {scale_ar}: the AR(scale_ar) regressions that calibrate the \
                     Minnesota prior's scales need at least {needed} rows (scale_ar presample \
                     rows plus scale_ar + 1 coefficients), but the sample has {got}; "
                )?;
                match max {
                    Some(m) => write!(
                        f,
                        "pass scale_ar <= {m} (4 = the library default, 1 = the \
                         Giannone-Lenza-Primiceri 2015 convention) or supply a longer sample"
                    ),
                    None => write!(
                        f,
                        "even scale_ar = 1 needs 4 rows, so supply a longer sample"
                    ),
                }
            }
        }
    }
}

impl std::error::Error for BayesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Linalg(e) => Some(e),
            Self::Stats(e) => Some(e),
            Self::Ssm(e) => Some(e),
            Self::Rng(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LinalgError> for BayesError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

impl From<StatsError> for BayesError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}

impl From<SsmError> for BayesError {
    fn from(e: SsmError) -> Self {
        Self::Ssm(e)
    }
}

impl From<RngError> for BayesError {
    fn from(e: RngError) -> Self {
        Self::Rng(e)
    }
}
