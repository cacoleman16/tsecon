//! The Theta method of Assimakopoulos & Nikolopoulos (2000), implemented
//! through the Hyndman & Billah (2003) SES-with-drift equivalence — the
//! M3-winning benchmark that remains shockingly hard to beat.
//!
//! # The model
//!
//! The original method decomposes the (deseasonalized) series into
//! theta-lines with modified curvature and averages the extrapolations of
//! the `theta = 0` line (a linear trend) and the `theta = 2` line
//! (extrapolated by simple exponential smoothing). Hyndman & Billah
//! (2003) showed the combination equals SES with a drift equal to half
//! the trend slope, which gives the closed form used here (and by
//! statsmodels, whose conventions this implementation reproduces):
//!
//! `Xhat_{T+h|T} = (theta-1)/theta * b0 * [h - 1 + 1/alpha - (1-alpha)^T / alpha] + Xtilde_{T+h|T}`
//!
//! for `h = 1..steps`, where
//!
//! * `b0` is the OLS slope of the deseasonalized series on the time trend
//!   `0, 1, ..., T-1`,
//! * `alpha` is the SES smoothing parameter estimated by minimizing the
//!   in-sample sum of squared one-step errors with the initial level
//!   fixed at the first observation (`l_0 = x_0`, the "known"
//!   initialization; for the innovations SES state space this
//!   least-squares fit and the concentrated Gaussian MLE coincide),
//! * `Xtilde_{T+h|T}` is the SES forecast, i.e. the terminal smoothed
//!   level (constant in `h`),
//! * `theta = 2` recovers the classic equal-weight combination
//!   (`trend weight (theta-1)/theta = 1/2`).
//!
//! # Seasonality
//!
//! For `period > 1` the series is first deseasonalized by classical
//! multiplicative decomposition (centered moving-average trend, per-phase
//! averages of the detrended series normalized to mean one), and the
//! forecasts are reseasonalized by the same factors. If the data are not
//! strictly positive (or any factor comes out non-positive), the
//! decomposition falls back to additive, mirroring statsmodels'
//! `method="auto"`. `period = 1` skips deseasonalization.
//!
//! Golden test: matches the statsmodels 0.14.6 `ThetaModel` 8-step
//! forecast on `realgdp` (`period=4`, `deseasonalize=True`,
//! `use_test=False`) to 1e-6 relative.
//!
//! TODO(phase0): prediction intervals (statsmodels draws `sigma2` from an
//! ARIMA(0,1,1)-with-drift fit) and the M4 90%-autocorrelation
//! seasonality pre-test arrive with the backtesting slice.

use crate::error::ForecastError;
use crate::validate::{check_series, check_steps};

/// Result of a Theta-method fit-and-forecast.
#[derive(Debug, Clone, PartialEq)]
pub struct ThetaForecast {
    /// Point forecasts for horizons `1..=steps` (reseasonalized).
    pub forecast: Vec<f64>,
    /// The estimated SES smoothing parameter.
    pub alpha: f64,
    /// The OLS slope of the deseasonalized series on `0..T-1`.
    pub b0: f64,
    /// The SES one-step forecast (terminal smoothed level) of the
    /// deseasonalized series.
    pub one_step: f64,
    /// The seasonal factors for phases `0..period` (relative to the start
    /// of the sample); empty when `period = 1`.
    pub seasonal: Vec<f64>,
    /// Whether the seasonal decomposition was multiplicative (`true`) or
    /// additive (`false`); meaningless when `seasonal` is empty.
    pub multiplicative: bool,
}

/// Classic Theta forecast (`theta = 2`, the Assimakopoulos &
/// Nikolopoulos 2000 choice). See [`theta_forecast_with`] and the
/// [module docs](self).
///
/// # Errors
///
/// Same as [`theta_forecast_with`].
pub fn theta_forecast(
    y: &[f64],
    period: usize,
    steps: usize,
) -> Result<ThetaForecast, ForecastError> {
    theta_forecast_with(y, period, steps, 2.0)
}

/// Theta forecast with a user-chosen theta-line parameter `theta >= 1`.
///
/// `theta = 1` collapses to a pure SES forecast (zero trend weight);
/// `theta -> inf` approaches the full trend adjustment `b0 * (...)`;
/// `theta = 2` is the classic method. See the [module docs](self) for
/// the formula and references (Assimakopoulos & Nikolopoulos 2000;
/// Hyndman & Billah 2003).
///
/// # Errors
///
/// [`ForecastError::InvalidTheta`] (`theta < 1` or non-finite),
/// [`ForecastError::InvalidSteps`], [`ForecastError::NonFinite`],
/// [`ForecastError::SeriesTooShort`] (`n < 4`, or `n < 2 * period` when
/// deseasonalizing — the decomposition needs two complete cycles), or
/// [`ForecastError::InvalidPeriod`] (`period = 0`).
pub fn theta_forecast_with(
    y: &[f64],
    period: usize,
    steps: usize,
    theta: f64,
) -> Result<ThetaForecast, ForecastError> {
    const WHAT: &str = "Theta method";
    check_steps(steps)?;
    if !theta.is_finite() || theta < 1.0 {
        return Err(ForecastError::InvalidTheta { theta });
    }
    if period == 0 {
        return Err(ForecastError::InvalidPeriod {
            what: WHAT,
            period,
            n: y.len(),
            requirement: "period >= 1 (use 1 for non-seasonal data)",
        });
    }
    let min_n = if period > 1 { (2 * period).max(4) } else { 4 };
    let n = check_series(y, min_n, WHAT)?;

    // 1. Deseasonalize (classical decomposition), statsmodels "auto":
    //    multiplicative when the data are strictly positive, else additive;
    //    fall back to additive if any multiplicative factor is <= 0.
    let (deseasonalized, seasonal, multiplicative): (Vec<f64>, Vec<f64>, bool) = if period > 1 {
        let mut multiplicative = y.iter().all(|&v| v > 0.0);
        let mut factors = classical_seasonal_factors(y, period, multiplicative);
        if multiplicative && factors.iter().any(|&s| s <= 0.0) {
            multiplicative = false;
            factors = classical_seasonal_factors(y, period, false);
        }
        let deseas: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(t, &v)| {
                if multiplicative {
                    v / factors[t % period]
                } else {
                    v - factors[t % period]
                }
            })
            .collect();
        (deseas, factors, multiplicative)
    } else {
        (y.to_vec(), Vec::new(), false)
    };

    // 2. b0: OLS slope of the deseasonalized series on the trend 0..n-1.
    let b0 = ols_trend_slope(&deseasonalized);

    // 3. alpha: SES least-squares fit with l0 fixed at the first
    //    observation (statsmodels initialization_method="known").
    let alpha = golden_section_min(|a| ses_sse(&deseasonalized, a), 1e-10, 1.0 - 1e-10);
    let one_step = ses_level(&deseasonalized, alpha);

    // 4. Combine: trend weight (theta-1)/theta on the theta=0 (trend)
    //    line adjustment, plus the SES forecast of the theta=2 line.
    let weight = (theta - 1.0) / theta;
    let nf = n as f64;
    let adj = 1.0 / alpha - (1.0 - alpha).powf(nf) / alpha;
    let mut forecast = Vec::with_capacity(steps);
    for j in 0..steps {
        let trend = b0 * (j as f64 + adj);
        let mut f = weight * trend + one_step;
        if !seasonal.is_empty() {
            let s = seasonal[(n + j) % period];
            f = if multiplicative { f * s } else { f + s };
        }
        forecast.push(f);
    }

    Ok(ThetaForecast {
        forecast,
        alpha,
        b0,
        one_step,
        seasonal,
        multiplicative,
    })
}

/// OLS slope of `y` regressed on an intercept and the trend `0..n-1`,
/// via the centered closed form `sum((t - tbar)(y - ybar)) / sum((t - tbar)^2)`.
fn ols_trend_slope(y: &[f64]) -> f64 {
    let n = y.len();
    let nf = n as f64;
    let tbar = (nf - 1.0) / 2.0;
    let ybar = y.iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut den = 0.0;
    for (t, &v) in y.iter().enumerate() {
        let dt = t as f64 - tbar;
        num += dt * (v - ybar);
        den += dt * dt;
    }
    num / den
}

/// Sum of squared one-step SES errors with `l_0 = y_0`:
/// `e_t = y_t - l_{t-1}`, `l_t = l_{t-1} + alpha * e_t`.
fn ses_sse(y: &[f64], alpha: f64) -> f64 {
    let mut level = y[0];
    let mut sse = 0.0;
    for &v in y {
        let e = v - level;
        sse += e * e;
        level += alpha * e;
    }
    sse
}

/// Terminal SES level (the one-step-ahead forecast) with `l_0 = y_0`.
fn ses_level(y: &[f64], alpha: f64) -> f64 {
    let mut level = y[0];
    for &v in y {
        level += alpha * (v - level);
    }
    level
}

/// Golden-section minimization of a unimodal scalar function on `[a, b]`.
///
/// The SES sum-of-squares is smooth and unimodal in practice; 120
/// iterations shrink the bracket below 1e-25, far past f64 resolution,
/// so the result is limited only by the objective's flatness near the
/// optimum (which is exactly what makes the forecast insensitive there).
fn golden_section_min<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_9; // (sqrt(5) - 1) / 2
    let mut c = b - INV_PHI * (b - a);
    let mut d = a + INV_PHI * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..120 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = f(d);
        }
        if (b - a).abs() < 1e-14 {
            break;
        }
    }
    0.5 * (a + b)
}

/// Seasonal factors by classical decomposition, mirroring statsmodels
/// `seasonal_decompose` (Brockwell & Davis 1991 §1.4 "small trends"
/// method): a centered moving-average trend (split end weights for even
/// periods), per-phase means of the detrended series over the interior
/// where the trend exists, normalized to mean one (multiplicative) or
/// mean zero (additive). Returns `period` factors aligned to phase
/// `t % period` of the sample.
///
/// Caller guarantees `y.len() >= 2 * period` and `period >= 2`.
fn classical_seasonal_factors(y: &[f64], period: usize, multiplicative: bool) -> Vec<f64> {
    let n = y.len();
    // Centered MA filter: odd period m -> m taps of 1/m; even period m ->
    // m+1 taps of [0.5, 1, ..., 1, 0.5]/m. Both have odd length.
    let (taps, half) = if period % 2 == 0 {
        (period + 1, period / 2)
    } else {
        (period, (period - 1) / 2)
    };
    let weight = 1.0 / period as f64;
    let end_weight = if period % 2 == 0 {
        0.5 * weight
    } else {
        weight
    };

    // Per-phase sums of the detrended series over t where the trend exists
    // (t = half ..= n - 1 - half).
    let mut sums = vec![0.0; period];
    let mut counts = vec![0usize; period];
    for t in half..(n - half) {
        let mut trend = 0.0;
        for j in 0..taps {
            let w = if j == 0 || j == taps - 1 {
                end_weight
            } else {
                weight
            };
            trend += w * y[t - half + j];
        }
        let detrended = if multiplicative {
            y[t] / trend
        } else {
            y[t] - trend
        };
        sums[t % period] += detrended;
        counts[t % period] += 1;
    }
    let mut factors: Vec<f64> = sums
        .iter()
        .zip(counts.iter())
        .map(|(&s, &c)| s / c as f64)
        .collect();
    let m = factors.iter().sum::<f64>() / period as f64;
    for s in factors.iter_mut() {
        if multiplicative {
            *s /= m;
        } else {
            *s -= m;
        }
    }
    factors
}
