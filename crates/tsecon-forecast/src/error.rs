//! Error types for `tsecon-forecast`.
//!
//! Every fallible entry point in this crate returns
//! `Result<_, ForecastError>`; nothing in the non-test code path panics.
//! Error messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters statistically, and what the caller
//! can do about it.

use core::fmt;

use tsecon_stats::StatsError;

/// Errors produced by the forecast-evaluation tools in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum ForecastError {
    /// The series has too few observations for the requested computation.
    SeriesTooShort {
        /// Which computation needed more data.
        what: &'static str,
        /// The number of observations supplied.
        n: usize,
        /// The minimum number of observations required.
        needed: usize,
    },
    /// Two paired series (e.g. actuals and forecasts, or two forecast-error
    /// vectors) have different lengths.
    LengthMismatch {
        /// Which computation received the mismatched pair.
        what: &'static str,
        /// The length of the first (reference) series.
        expected: usize,
        /// The length of the second series.
        actual: usize,
    },
    /// The input contains a NaN or infinite value. Evaluation never skips
    /// missing values silently (a skipped period would silently change the
    /// evaluation sample); drop or impute them first.
    NonFinite {
        /// Which input contained the offending value.
        what: &'static str,
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// MAPE is undefined because an actual value is zero: the percentage
    /// error `100 e_t / y_t` divides by `y_t`.
    ZeroActualInMape {
        /// Index of the first zero actual.
        index: usize,
    },
    /// sMAPE is undefined because `|y_t| + |yhat_t| = 0` for some `t`
    /// (both actual and forecast are zero).
    ZeroDenominatorInSmape {
        /// Index of the first zero denominator.
        index: usize,
    },
    /// The MASE/RMSSE scaling denominator — the in-sample seasonal-naive
    /// MAE (or MSE) — is zero, so the scaled error is undefined.
    ZeroScaleDenominator {
        /// Which scaled measure hit the degenerate denominator.
        what: &'static str,
        /// The seasonal period used for the in-sample naive forecast.
        period: usize,
    },
    /// The seasonal period is invalid (zero, or too large for the series).
    InvalidPeriod {
        /// Which computation rejected the period.
        what: &'static str,
        /// The offending period.
        period: usize,
        /// The number of observations supplied.
        n: usize,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The number of forecast steps must be a positive integer.
    InvalidSteps {
        /// The offending step count.
        steps: usize,
    },
    /// The forecast horizon `h` passed to the Diebold-Mariano test is
    /// outside `1 <= h < n`: the long-run variance truncates the
    /// autocovariance sum at lag `h - 1`, which must exist in the sample.
    InvalidHorizon {
        /// The offending horizon.
        h: usize,
        /// The number of loss differentials supplied.
        n: usize,
    },
    /// The loss differential is degenerate (zero variance): the two
    /// forecasts have identical losses in every period, so there is no
    /// accuracy difference to test. This happens when the same forecast
    /// (or error vector) is compared with itself.
    DegenerateLossDifferential,
    /// The truncated uniform-weight long-run variance estimate of the mean
    /// loss differential is not positive, so the DM statistic is undefined.
    NonPositiveLongRunVariance {
        /// The offending variance estimate.
        value: f64,
    },
    /// The prediction-interval coverage level is outside (0, 1).
    InvalidLevel {
        /// The offending value.
        level: f64,
    },
    /// The significance level `alpha` passed to a comparison is outside
    /// (0, 1).
    InvalidAlpha {
        /// The offending value.
        value: f64,
    },
    /// The Theta-line parameter must satisfy `theta >= 1`.
    InvalidTheta {
        /// The offending value.
        theta: f64,
    },
    /// Two forecasts in a comparison share the same name, which would make
    /// the report ambiguous.
    DuplicateName {
        /// The repeated name.
        name: String,
    },
    /// A comparison needs at least one named forecast.
    EmptyComparison,
    /// An error propagated from the `tsecon-stats` distributions (e.g. the
    /// Student-t survival function used for DM p-values).
    Stats(StatsError),
}

impl fmt::Display for ForecastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForecastError::SeriesTooShort { what, n, needed } => write!(
                f,
                "{what}: series has {n} observations but needs at least \
                 {needed}; supply more data or reduce the seasonal period / \
                 horizon"
            ),
            ForecastError::LengthMismatch {
                what,
                expected,
                actual,
            } => write!(
                f,
                "{what}: paired series must be index-aligned and equally \
                 long, got lengths {expected} and {actual}; check that the \
                 forecast covers exactly the evaluation window"
            ),
            ForecastError::NonFinite { what, index, value } => write!(
                f,
                "{what}: input contains a non-finite value ({value}) at \
                 index {index}; forecast evaluation does not skip missing \
                 values silently — that would change the evaluation sample \
                 behind your back — drop or impute NaN/inf observations \
                 first"
            ),
            ForecastError::ZeroActualInMape { index } => write!(
                f,
                "MAPE is undefined: actual value at index {index} is zero \
                 and the percentage error 100*e_t/y_t divides by it. MAPE \
                 explodes near zero and penalizes over-forecasts \
                 asymmetrically (Goodwin & Lawton 1999); for data with \
                 zeros prefer a scaled error such as MASE or RMSSE \
                 (Hyndman & Koehler 2006)"
            ),
            ForecastError::ZeroDenominatorInSmape { index } => write!(
                f,
                "sMAPE is undefined: |actual| + |forecast| is zero at index \
                 {index}. Rather than silently returning inf that averages \
                 away, this is an error; for data with zeros prefer MASE or \
                 RMSSE (Hyndman & Koehler 2006)"
            ),
            ForecastError::ZeroScaleDenominator { what, period } => write!(
                f,
                "{what}: the in-sample seasonal-naive error at period \
                 {period} is exactly zero (the training series repeats \
                 every {period} observations), so the scaled error divides \
                 by zero; use a different period or an unscaled measure"
            ),
            ForecastError::InvalidPeriod {
                what,
                period,
                n,
                requirement,
            } => write!(
                f,
                "{what}: period = {period} is invalid for a series of \
                 length {n}: requires {requirement}"
            ),
            ForecastError::InvalidSteps { steps } => write!(
                f,
                "steps = {steps} is invalid: the forecast horizon must be a \
                 positive integer"
            ),
            ForecastError::InvalidHorizon { h, n } => write!(
                f,
                "Diebold-Mariano: forecast horizon h = {h} is invalid for \
                 {n} loss differentials: requires 1 <= h < n because the \
                 long-run variance sums autocovariances up to lag h - 1"
            ),
            ForecastError::DegenerateLossDifferential => write!(
                f,
                "Diebold-Mariano: the loss differential has zero variance — \
                 the two forecasts incur identical losses in every period, \
                 so equal predictive accuracy holds trivially and the DM \
                 statistic is 0/0. This usually means the same forecast was \
                 compared with itself; the test needs two genuinely \
                 different forecast streams"
            ),
            ForecastError::NonPositiveLongRunVariance { value } => write!(
                f,
                "Diebold-Mariano: the uniform-weight long-run variance \
                 estimate truncated at lag h-1 is not positive ({value}); \
                 this rectangular window is not guaranteed positive \
                 semi-definite. Reduce h, or use a HAC kernel estimate \
                 (Bartlett) from tsecon-hac for the variance"
            ),
            ForecastError::InvalidLevel { level } => write!(
                f,
                "prediction-interval level = {level} is invalid: requires \
                 0 < level < 1 (e.g. 0.95 for a 95% interval)"
            ),
            ForecastError::InvalidAlpha { value } => write!(
                f,
                "significance level alpha = {value} is invalid: requires \
                 0 < alpha < 1 (conventional choices are 0.01, 0.05, 0.10)"
            ),
            ForecastError::InvalidTheta { theta } => write!(
                f,
                "theta = {theta} is invalid: the Theta method requires \
                 theta >= 1, which puts non-negative weight (theta-1)/theta \
                 on the linear-trend line (theta = 2 is the classic \
                 Assimakopoulos-Nikolopoulos choice)"
            ),
            ForecastError::DuplicateName { name } => write!(
                f,
                "forecast comparison: the name {name:?} appears more than \
                 once; give each forecast a unique label so the accuracy \
                 table and DM pairs are unambiguous"
            ),
            ForecastError::EmptyComparison => write!(
                f,
                "forecast comparison: no forecasts supplied; pass at least \
                 one named forecast vector (two or more to get pairwise \
                 Diebold-Mariano tests)"
            ),
            ForecastError::Stats(e) => write!(f, "distribution error: {e}"),
        }
    }
}

impl std::error::Error for ForecastError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ForecastError::Stats(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StatsError> for ForecastError {
    fn from(e: StatsError) -> Self {
        ForecastError::Stats(e)
    }
}
