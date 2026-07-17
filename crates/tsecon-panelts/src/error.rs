//! Error type shared by the heterogeneous-panel estimators in this crate.
//!
//! Every fallible public function returns `Result<_, PanelTsError>`; no code
//! path panics on user input. Messages follow the library's "errors that
//! teach" convention: what went wrong, why it matters, and what to do.

use core::fmt;

use tsecon_hac::HacError;
use tsecon_stats::StatsError;

/// Errors returned by `tsecon-panelts`.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelTsError {
    /// The panel had fewer than two units, so the cross-unit sample covariance
    /// that the mean-group standard error is built from is undefined.
    TooFewUnits {
        /// The number of units supplied.
        n: usize,
    },
    /// A unit carried no regressors, so there is no slope vector to average.
    NoRegressors {
        /// Zero-based index of the offending unit.
        unit: usize,
    },
    /// Units disagree on the number of regressors `k`; the mean-group average
    /// is only defined when every unit shares the same slope dimension.
    InconsistentRegressors {
        /// Zero-based index of the offending unit.
        unit: usize,
        /// The `k` established by the first unit.
        expected: usize,
        /// The `k` this unit carried.
        got: usize,
    },
    /// Within a unit the response and a regressor column disagree in length.
    RaggedUnit {
        /// Zero-based index of the offending unit.
        unit: usize,
        /// Zero-based index of the offending regressor column.
        column: usize,
        /// The length of the unit's response vector.
        expected: usize,
        /// The length of the offending column.
        got: usize,
    },
    /// The common-correlated-effects estimator needs a *balanced* panel (every
    /// unit observed over the same time index) so that the per-period
    /// cross-section averages are well defined, but the units differ in length.
    UnbalancedPanel {
        /// Zero-based index of the offending unit.
        unit: usize,
        /// The number of time periods `T` established by the first unit.
        expected: usize,
        /// The number of time periods this unit carried.
        got: usize,
    },
    /// An error propagated from the per-unit OLS in `tsecon-hac` (e.g. a
    /// collinear augmented design, or too few periods for the regressors).
    Ols {
        /// Zero-based index of the unit whose OLS failed.
        unit: usize,
        /// The underlying `tsecon-hac` error.
        source: HacError,
    },
    /// An error propagated from the `tsecon-stats` distribution layer used for
    /// the mean-group p-values and confidence bands.
    Stats(StatsError),
}

impl fmt::Display for PanelTsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewUnits { n } => write!(
                f,
                "the panel has N = {n} unit(s); the mean-group estimator needs \
                 at least 2 units because its standard error is the cross-unit \
                 sample covariance of the per-unit slopes divided by N"
            ),
            Self::NoRegressors { unit } => write!(
                f,
                "unit {unit} has no regressors; supply at least one x column \
                 (the intercept is added internally and is never averaged)"
            ),
            Self::InconsistentRegressors {
                unit,
                expected,
                got,
            } => write!(
                f,
                "unit {unit} has {got} regressor(s) but the first unit had \
                 {expected}; every unit must share the same slope dimension k \
                 for the mean-group average to be defined"
            ),
            Self::RaggedUnit {
                unit,
                column,
                expected,
                got,
            } => write!(
                f,
                "unit {unit}: regressor column {column} has {got} observations \
                 but the response has {expected}; each unit's columns must be \
                 index-aligned with its own y"
            ),
            Self::UnbalancedPanel {
                unit,
                expected,
                got,
            } => write!(
                f,
                "unit {unit} spans {got} periods but the first unit spanned \
                 {expected}; the CCE estimator needs a balanced panel so the \
                 per-period cross-section averages line up across units"
            ),
            Self::Ols { unit, source } => {
                write!(f, "per-unit OLS failed for unit {unit}: {source}")
            }
            Self::Stats(e) => write!(f, "distribution error: {e}"),
        }
    }
}

impl std::error::Error for PanelTsError {}

impl From<StatsError> for PanelTsError {
    fn from(e: StatsError) -> Self {
        Self::Stats(e)
    }
}
