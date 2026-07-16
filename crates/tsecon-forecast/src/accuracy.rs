//! Point-forecast accuracy measures.
//!
//! All measures use the forecast-error convention `e_t = y_t - yhat_t`
//! (actual minus forecast) and validate their inputs up front: NaN or
//! infinite values are an error, never silently skipped — a skipped period
//! would change the evaluation sample behind the caller's back (the
//! library-wide missing-data policy).
//!
//! Scale-dependent measures ([`me`], [`mse`], [`rmse`], [`mae`], [`mdae`])
//! must not be averaged across series of different scales; for
//! cross-series comparison use the scaled errors [`mase`] and [`rmsse`]
//! (Hyndman & Koehler 2006), which are also the official M4/M5 metrics.
//! The percentage errors [`mape`] and [`smape`] are included for
//! compatibility with practice and guarded against their known
//! pathologies (Goodwin & Lawton 1999).

use crate::error::ForecastError;
use crate::validate::{check_pair, check_series};

/// Forecast errors `e_t = y_t - yhat_t` after validating the pair.
fn errors(actual: &[f64], forecast: &[f64], what: &'static str) -> Result<Vec<f64>, ForecastError> {
    check_pair(actual, forecast, what)?;
    Ok(actual
        .iter()
        .zip(forecast.iter())
        .map(|(&y, &f)| y - f)
        .collect())
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Mean error, `ME = mean(e_t)` with `e_t = y_t - yhat_t`.
///
/// A measure of bias, not accuracy: positive values mean the forecast
/// under-predicts on average. Reference: Hyndman & Koehler (2006).
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`], [`ForecastError::SeriesTooShort`]
/// (empty input), or [`ForecastError::NonFinite`].
pub fn me(actual: &[f64], forecast: &[f64]) -> Result<f64, ForecastError> {
    let e = errors(actual, forecast, "ME")?;
    Ok(mean(&e))
}

/// Mean squared error, `MSE = mean(e_t^2)`.
///
/// Scale-dependent; do not average across series of different scales —
/// use [`rmsse`] instead. Reference: Hyndman & Koehler (2006).
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`], [`ForecastError::SeriesTooShort`]
/// (empty input), or [`ForecastError::NonFinite`].
pub fn mse(actual: &[f64], forecast: &[f64]) -> Result<f64, ForecastError> {
    let e = errors(actual, forecast, "MSE")?;
    Ok(e.iter().map(|&v| v * v).sum::<f64>() / e.len() as f64)
}

/// Root mean squared error, `RMSE = sqrt(mean(e_t^2))`.
///
/// Scale-dependent; do not average RMSE across series of different scales
/// (the mean of square roots is not the square root of the mean) — use
/// [`rmsse`] for cross-series work. Reference: Hyndman & Koehler (2006).
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`], [`ForecastError::SeriesTooShort`]
/// (empty input), or [`ForecastError::NonFinite`].
pub fn rmse(actual: &[f64], forecast: &[f64]) -> Result<f64, ForecastError> {
    Ok(mse(actual, forecast)?.sqrt())
}

/// Mean absolute error, `MAE = mean(|e_t|)`.
///
/// Scale-dependent; optimal forecasts under MAE are conditional medians,
/// under MSE conditional means — do not mix the two when ranking models.
/// Reference: Hyndman & Koehler (2006).
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`], [`ForecastError::SeriesTooShort`]
/// (empty input), or [`ForecastError::NonFinite`].
pub fn mae(actual: &[f64], forecast: &[f64]) -> Result<f64, ForecastError> {
    let e = errors(actual, forecast, "MAE")?;
    Ok(e.iter().map(|&v| v.abs()).sum::<f64>() / e.len() as f64)
}

/// Median absolute error, `MdAE = median(|e_t|)`.
///
/// A robust companion to [`mae`]: insensitive to a few catastrophic
/// misses. The median of an even count is the midpoint of the two central
/// order statistics. Reference: Hyndman & Koehler (2006).
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`], [`ForecastError::SeriesTooShort`]
/// (empty input), or [`ForecastError::NonFinite`].
pub fn mdae(actual: &[f64], forecast: &[f64]) -> Result<f64, ForecastError> {
    let mut abs_e: Vec<f64> = errors(actual, forecast, "MdAE")?
        .iter()
        .map(|&v| v.abs())
        .collect();
    abs_e.sort_unstable_by(f64::total_cmp);
    let n = abs_e.len();
    Ok(if n % 2 == 1 {
        abs_e[n / 2]
    } else {
        0.5 * (abs_e[n / 2 - 1] + abs_e[n / 2])
    })
}

/// Mean absolute percentage error, `MAPE = 100 * mean(|e_t| / |y_t|)`.
///
/// Included for compatibility with practice; it explodes near zero and
/// penalizes over-forecasts more heavily than under-forecasts (Goodwin &
/// Lawton 1999). Any zero actual is therefore a hard error rather than a
/// silent `inf` that averages away.
///
/// # Errors
///
/// [`ForecastError::ZeroActualInMape`] if any `y_t == 0`, plus the shared
/// pair-validation errors.
pub fn mape(actual: &[f64], forecast: &[f64]) -> Result<f64, ForecastError> {
    check_pair(actual, forecast, "MAPE")?;
    let mut acc = 0.0;
    for (index, (&y, &f)) in actual.iter().zip(forecast.iter()).enumerate() {
        if y == 0.0 {
            return Err(ForecastError::ZeroActualInMape { index });
        }
        acc += ((y - f) / y).abs();
    }
    Ok(100.0 * acc / actual.len() as f64)
}

/// Symmetric mean absolute percentage error, M4 definition:
/// `sMAPE = mean(200 * |e_t| / (|y_t| + |yhat_t|))`.
///
/// This is the M4-competition variant (Makridakis, Spiliotis &
/// Assimakopoulos 2020), which divides by the sum of *absolute* values
/// and multiplies by 200, bounding the measure at 200. Some texts instead
/// divide by `(y_t + yhat_t)/2` without absolute values, which can go
/// negative; that variant is deliberately not implemented. Despite the
/// name, sMAPE is not actually symmetric in over-/under-forecasting
/// (Goodwin & Lawton 1999). A zero denominator (`y_t = yhat_t = 0`) is a
/// hard error rather than a silent `inf`.
///
/// # Errors
///
/// [`ForecastError::ZeroDenominatorInSmape`] if `|y_t| + |yhat_t| == 0`
/// for some `t`, plus the shared pair-validation errors.
pub fn smape(actual: &[f64], forecast: &[f64]) -> Result<f64, ForecastError> {
    check_pair(actual, forecast, "sMAPE")?;
    let mut acc = 0.0;
    for (index, (&y, &f)) in actual.iter().zip(forecast.iter()).enumerate() {
        let denom = y.abs() + f.abs();
        if denom == 0.0 {
            return Err(ForecastError::ZeroDenominatorInSmape { index });
        }
        acc += 200.0 * (y - f).abs() / denom;
    }
    Ok(acc / actual.len() as f64)
}

/// In-sample seasonal-naive scaling denominator:
/// `mean(|y_t - y_{t-m}|^p)` over `t = m..n-1` of the training sample,
/// with `p = 1` (MASE) or `p = 2` (RMSSE).
fn scale_denominator(
    insample: &[f64],
    period: usize,
    squared: bool,
    what: &'static str,
) -> Result<f64, ForecastError> {
    if period == 0 {
        return Err(ForecastError::InvalidPeriod {
            what,
            period,
            n: insample.len(),
            requirement: "period >= 1 (use 1 for non-seasonal data)",
        });
    }
    let n = check_series(insample, period + 1, what)?;
    let mut acc = 0.0;
    for t in period..n {
        let d = insample[t] - insample[t - period];
        acc += if squared { d * d } else { d.abs() };
    }
    let denom = acc / (n - period) as f64;
    if denom == 0.0 {
        return Err(ForecastError::ZeroScaleDenominator { what, period });
    }
    Ok(denom)
}

/// Mean absolute scaled error (Hyndman & Koehler 2006):
///
/// `MASE = mean(|e_t|) / mean_{t=m..T-1}(|y_t - y_{t-m}|)`
///
/// where the denominator is the in-sample MAE of the seasonal-naive
/// forecast at period `m` computed on the *training* sample `insample`
/// (never the evaluation sample — that would leak the test set into the
/// scale). `period = 1` gives the non-seasonal naive scaling. Values
/// below 1 mean the forecast beats the in-sample seasonal-naive
/// benchmark on average; the in-sample seasonal-naive forecast itself
/// scores exactly 1 by construction. MASE is the recommended default for
/// comparing accuracy across series of different scales and an official
/// M4 metric.
///
/// # Errors
///
/// [`ForecastError::ZeroScaleDenominator`] when the training series
/// repeats exactly every `period` observations (constant series for
/// `period = 1`), [`ForecastError::InvalidPeriod`] for `period = 0`, and
/// the shared validation errors.
pub fn mase(
    actual: &[f64],
    forecast: &[f64],
    insample: &[f64],
    period: usize,
) -> Result<f64, ForecastError> {
    let denom = scale_denominator(insample, period, false, "MASE")?;
    Ok(mae(actual, forecast)? / denom)
}

/// Root mean squared scaled error (the M5 metric; Hyndman & Koehler 2006
/// scaling applied to squared errors):
///
/// `RMSSE = sqrt( mean(e_t^2) / mean_{t=m..T-1}((y_t - y_{t-m})^2) )`
///
/// The denominator is the in-sample MSE of the seasonal-naive forecast at
/// period `m` on the *training* sample. Like [`mase`] it is scale-free
/// and safe to average across series; unlike MASE it lives on the
/// squared-error scale, so it ranks conditional-mean forecasts
/// consistently. Reference: Makridakis, Spiliotis & Assimakopoulos
/// (2022, M5 accuracy competition).
///
/// # Errors
///
/// Same as [`mase`].
pub fn rmsse(
    actual: &[f64],
    forecast: &[f64],
    insample: &[f64],
    period: usize,
) -> Result<f64, ForecastError> {
    let denom = scale_denominator(insample, period, true, "RMSSE")?;
    Ok((mse(actual, forecast)? / denom).sqrt())
}
