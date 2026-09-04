//! Error types for `tsecon-forecast`.
//!
//! Every fallible entry point in this crate returns
//! `Result<_, ForecastError>`; nothing in the non-test code path panics.
//! Error messages follow the library's "errors that teach" pillar: they
//! state what went wrong, why it matters statistically, and what the caller
//! can do about it.

use core::fmt;

use tsecon_bootstrap::BootstrapError;
use tsecon_hac::HacError;
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
    /// A declared nested `(small, large)` pair references a forecast label
    /// that was not supplied in the comparison's forecast list.
    UnknownForecastName {
        /// The label that could not be matched.
        name: String,
    },
    /// A backtest scheme parameter (minimum training size, rolling width,
    /// horizon, or refit cadence) violates its constraint.
    InvalidBacktestParam {
        /// Which parameter was rejected.
        what: &'static str,
        /// The offending value.
        value: usize,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The backtest scheme leaves no forecast origin in the sample: after
    /// reserving the training window and the `horizon` targets there is no
    /// origin `t` left with all horizons `1..=horizon` in-sample.
    NoBacktestOrigins {
        /// The series length.
        n: usize,
        /// The index of the first candidate origin (training window just
        /// filled).
        first_origin: usize,
        /// The maximum horizon reserved at the end of the sample.
        horizon: usize,
    },
    /// A forecaster closure returned the wrong number of forecasts for a
    /// backtest origin: the engine asked for `expected` horizons (`1..=h`)
    /// but got `actual`.
    ForecasterOutputLen {
        /// The origin index the closure was called for.
        origin: usize,
        /// The number of horizons requested.
        expected: usize,
        /// The number of forecasts returned.
        actual: usize,
    },
    /// The requested horizon is outside the range the backtest evaluated
    /// (`1..=horizon`).
    HorizonOutOfRange {
        /// The requested horizon.
        h: usize,
        /// The maximum horizon the backtest collected.
        max_h: usize,
    },
    /// The long-run-variance lag truncation is too large for the sample:
    /// `lags` must be strictly less than the number of observations, since
    /// the Bartlett sum needs an autocovariance at that lag.
    InvalidLrvLags {
        /// Which test rejected the lag count.
        what: &'static str,
        /// The offending lag truncation.
        lags: usize,
        /// The number of observations supplied.
        n: usize,
    },
    /// A Giacomini-White conditional test received no test functions, so the
    /// Wald form has dimension zero.
    EmptyTestFunctions,
    /// A pre-computed violation ("hit") series contains a value other than
    /// exactly 0 or 1. The VaR backtests are defined on an indicator
    /// sequence; anything else is almost certainly a raw return series
    /// passed without its VaR forecasts.
    InvalidHitValue {
        /// Index of the first offending observation.
        index: usize,
        /// The offending value.
        value: f64,
    },
    /// The hit sequence contains no violations at all, so the backtest
    /// battery degenerates: the independence and DQ statistics are
    /// undefined (their contingency cells / lagged-hit regressors are
    /// empty), and the Kupiec statistic collapses to its continuity limit
    /// `-2 n ln(1-alpha)`.
    NoViolations {
        /// The number of observations.
        n: usize,
        /// The VaR coverage level.
        alpha: f64,
    },
    /// Every observation in the hit sequence is a violation — the mirror
    /// degenerate case of [`ForecastError::NoViolations`], and almost
    /// always a sign-convention slip (see the `var_backtest` docs).
    AllViolations {
        /// The number of observations.
        n: usize,
        /// The VaR coverage level.
        alpha: f64,
    },
    /// The dynamic-quantile lag count is invalid for the sample: the DQ
    /// regression needs `dq_lags >= 1` and enough post-lag observations to
    /// identify its coefficients.
    InvalidDqLags {
        /// The offending lag count.
        lags: usize,
        /// The number of observations supplied.
        n: usize,
        /// The minimum series length for this lag count.
        needed: usize,
    },
    /// The dynamic-quantile design matrix `X'X` is singular, so the DQ
    /// statistic is undefined. With very few violations the lagged-hit
    /// columns are (numerically) constant and collinear with the intercept.
    SingularDqDesign {
        /// The number of DQ regressors (constant + lagged hits + VaR).
        k: usize,
        /// The number of violations in the full hit sequence.
        n_violations: usize,
    },
    /// The Giacomini-White conditional Wald covariance `Shat` is singular or
    /// indefinite (typically collinear or constant test functions), so its
    /// inverse — and the Wald statistic — is undefined.
    SingularWaldCovariance {
        /// The dimension of the (failed) `Shat`.
        q: usize,
    },
    /// The conformal calibration set is too small for the requested
    /// miscoverage: the finite-sample-corrected quantile index
    /// `ceil((m+1)(1-alpha))` exceeds the number of scores `m`, so no
    /// finite interval can carry the guarantee at this level.
    CalibrationTooSmall {
        /// Which conformal computation was refused.
        what: &'static str,
        /// The number of calibration scores supplied.
        n_calib: usize,
        /// The per-tail miscoverage the quantile was requested at.
        alpha: f64,
        /// The minimum number of scores that supports this level.
        needed: usize,
    },
    /// A conformal-method parameter violates its constraint (step size,
    /// lag order, ensemble size, evaluation-window size, ...).
    InvalidConformalParam {
        /// Which parameter was rejected.
        what: &'static str,
        /// The offending value.
        value: f64,
        /// Human-readable statement of the violated constraint.
        requirement: &'static str,
    },
    /// The lagged least-squares design of the AR base learner (used by
    /// EnbPI's bootstrap ensemble and the `"ar"` base) is singular —
    /// typically a constant series, or a bootstrap resample that collapsed
    /// onto collinear rows.
    SingularArDesign {
        /// The autoregressive order of the design.
        lags: usize,
        /// The number of design rows in the failed fit.
        n_rows: usize,
    },
    /// An error raised by the base point forecaster a conformal method
    /// wraps, carried through with its original message.
    BaseForecaster {
        /// The base forecaster's own error message.
        message: String,
    },
    /// An error propagated from the `tsecon-bootstrap` resampling engine
    /// (used for EnbPI's bootstrap index draws).
    Bootstrap(BootstrapError),
    /// An error propagated from the `tsecon-stats` distributions (e.g. the
    /// Student-t survival function used for DM p-values).
    Stats(StatsError),
    /// An error propagated from the `tsecon-hac` long-run-variance engine
    /// (used for the Clark-West and Giacomini-White variances).
    Hac(HacError),
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
                 every {period} observations — constant, for period 1), so \
                 the scaled error divides by zero. No parameter of this \
                 call selects an unscaled measure instead; the cure is a \
                 training sample that varies at that period — in a backtest \
                 the FIRST training window sets this scale, so lengthen or \
                 shift it (train=) past the constant/periodic stretch (or \
                 change insample_period); in a direct mase/rmsse call pass \
                 a non-constant insample, or compute the unscaled MAE/RMSE \
                 yourself"
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
            ForecastError::UnknownForecastName { name } => write!(
                f,
                "forecast comparison: the nested pair references {name:?}, \
                 which is not among the supplied forecast labels; declare \
                 nested (small, large) pairs using names that appear in the \
                 forecast list"
            ),
            ForecastError::InvalidBacktestParam {
                what,
                value,
                requirement,
            } => write!(
                f,
                "backtest: {what} = {value} is invalid: requires {requirement}"
            ),
            ForecastError::NoBacktestOrigins {
                n,
                first_origin,
                horizon,
            } => write!(
                f,
                "backtest: a series of length {n} leaves no forecast origin \
                 for this scheme — the first origin with its training window \
                 filled is index {first_origin}, but every origin also needs \
                 {horizon} in-sample target(s) ahead of it (origins run only \
                 up to index n - 1 - horizon). Supply a longer series, a \
                 smaller training window, or a shorter horizon"
            ),
            ForecastError::ForecasterOutputLen {
                origin,
                expected,
                actual,
            } => write!(
                f,
                "backtest: the forecaster closure returned {actual} forecasts \
                 at origin {origin} but the engine asked for {expected} \
                 (horizons 1..={expected}); a forecaster must return exactly \
                 the requested number of multi-step point forecasts"
            ),
            ForecastError::HorizonOutOfRange { h, max_h } => write!(
                f,
                "backtest result: horizon h = {h} was not evaluated; this \
                 backtest collected horizons 1..={max_h}"
            ),
            ForecastError::InvalidLrvLags { what, lags, n } => write!(
                f,
                "{what}: long-run-variance lag truncation {lags} is invalid \
                 for {n} observations: requires lags < n so the Bartlett sum \
                 has an autocovariance at every lag up to {lags}"
            ),
            ForecastError::EmptyTestFunctions => write!(
                f,
                "Giacomini-White conditional test: no test functions supplied; \
                 pass at least a constant (h_t = 1, which recovers the \
                 unconditional test) and typically also lagged loss \
                 differentials to test WHEN one forecast beats the other"
            ),
            ForecastError::InvalidHitValue { index, value } => write!(
                f,
                "VaR backtest: the hit series has value {value} at index \
                 {index}, but a pre-computed violation sequence must contain \
                 exactly 0 (no violation) and 1 (violation). If this is a \
                 return series, pass its VaR forecasts too so the violations \
                 can be computed (violation = return < VaR quantile)"
            ),
            ForecastError::NoViolations { n, alpha } => {
                let expected = alpha * *n as f64;
                let lr_limit = -2.0 * (*n as f64) * (1.0 - alpha).ln();
                write!(
                    f,
                    "VaR backtest: 0 violations in {n} observations where \
                     {expected:.1} were expected at alpha = {alpha}. With no \
                     violations the independence (Christoffersen) and DQ \
                     (Engle-Manganelli) statistics are undefined — their \
                     contingency cells and lagged-hit regressors are empty — \
                     and the Kupiec statistic degenerates to its continuity \
                     limit LR_uc = -2 n ln(1-alpha) = {lr_limit:.3} \
                     (chi-squared(1) 5% critical value 3.84; larger means \
                     zero violations is itself evidence the VaR is too \
                     conservative). Use a longer evaluation window or a \
                     larger alpha — and check the sign convention: a \
                     violation is return < VaR quantile, both on the return \
                     scale"
                )
            }
            ForecastError::AllViolations { n, alpha } => write!(
                f,
                "VaR backtest: every one of the {n} observations is a \
                 violation, where alpha = {alpha} predicts a {:.1}% violation \
                 rate. This is almost always a sign-convention slip: the \
                 convention here is returns and VaR quantiles on the same \
                 (return) scale, violation = return < VaR, so a 5% VaR \
                 forecast is typically a negative number. If you work in \
                 positive-loss space, negate both series before calling",
                alpha * 100.0
            ),
            ForecastError::InvalidDqLags { lags, n, needed } => write!(
                f,
                "VaR backtest: dq_lags = {lags} is invalid for {n} \
                 observations. The DQ test regresses hit_t - alpha on a \
                 constant, {lags} lagged hits, and the VaR forecast, so it \
                 needs dq_lags >= 1 and at least {needed} observations to \
                 identify the coefficients; reduce dq_lags (the \
                 Engle-Manganelli default is 4) or supply a longer series"
            ),
            ForecastError::SingularDqDesign { k, n_violations } => write!(
                f,
                "VaR backtest: the DQ design matrix X'X ({k}x{k}) is \
                 singular, so the dynamic-quantile statistic is undefined. \
                 With only {n_violations} violation(s) the lagged-hit \
                 regressors are (numerically) constant and collinear with \
                 the intercept; reduce dq_lags, use a longer evaluation \
                 window, or rely on the Kupiec/Christoffersen results, which \
                 remain valid"
            ),
            ForecastError::SingularWaldCovariance { q } => write!(
                f,
                "Giacomini-White conditional test: the {q}x{q} long-run \
                 covariance Shat of the instrumented loss differential is \
                 singular or indefinite, so the Wald statistic \
                 n*zbar'*Shat^-1*zbar is undefined. This usually means the \
                 test functions are collinear or constant across the \
                 evaluation window — drop redundant instruments"
            ),
            ForecastError::CalibrationTooSmall {
                what,
                n_calib,
                alpha,
                needed,
            } => write!(
                f,
                "{what}: {n_calib} calibration score(s) cannot support a \
                 finite-sample-corrected quantile at miscoverage alpha = \
                 {alpha}: the corrected index ceil((m+1)(1-alpha)) exceeds \
                 m, so the honest interval would be infinite. Supply at \
                 least {needed} calibration residuals (roughly (1-alpha)/\
                 alpha), enlarge alpha, or shrink the horizon"
            ),
            ForecastError::InvalidConformalParam {
                what,
                value,
                requirement,
            } => write!(
                f,
                "conformal: {what} = {value} is invalid: requires {requirement}"
            ),
            ForecastError::SingularArDesign { lags, n_rows } => write!(
                f,
                "AR base learner: the lagged least-squares design with \
                 {lags} lag(s) over {n_rows} row(s) is singular — the \
                 series is (numerically) constant or the regressors are \
                 collinear, so no AR fit exists. A constant series needs no \
                 interval; otherwise reduce lags or supply more data"
            ),
            ForecastError::BaseForecaster { message } => {
                write!(f, "conformal base forecaster failed: {message}")
            }
            ForecastError::Bootstrap(e) => write!(f, "bootstrap error: {e}"),
            ForecastError::Stats(e) => write!(f, "distribution error: {e}"),
            ForecastError::Hac(e) => write!(f, "long-run-variance error: {e}"),
        }
    }
}

impl std::error::Error for ForecastError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ForecastError::Stats(e) => Some(e),
            ForecastError::Hac(e) => Some(e),
            ForecastError::Bootstrap(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StatsError> for ForecastError {
    fn from(e: StatsError) -> Self {
        ForecastError::Stats(e)
    }
}

impl From<BootstrapError> for ForecastError {
    fn from(e: BootstrapError) -> Self {
        ForecastError::Bootstrap(e)
    }
}

impl From<HacError> for ForecastError {
    fn from(e: HacError) -> Self {
        ForecastError::Hac(e)
    }
}
