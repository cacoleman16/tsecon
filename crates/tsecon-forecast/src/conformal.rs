//! Distribution-free conformal prediction intervals for point forecasters:
//! split conformal, EnbPI, and adaptive conformal inference (ACI).
//!
//! All three wrap an arbitrary point forecaster and calibrate interval
//! widths on *realized out-of-sample residuals*, never on in-sample fit —
//! the same leakage discipline as [`crate::backtest`], whose rectangular
//! `(origin, horizon, target)` grid and expanding training windows this
//! module reuses verbatim for its calibration residuals.
//!
//! # The three methods
//!
//! * **Split conformal** ([`split_conformal`]) — the baseline. Hold out the
//!   last `calib` forecast origins; at each, forecast `1..=horizon` steps
//!   ahead from an expanding training window and record the signed residual
//!   `y[t+h] - yhat`. The horizon-`h` interval around the forward forecast
//!   is the finite-sample-corrected empirical quantile of those residuals:
//!   symmetric mode uses the `ceil((m+1)(1-alpha))`-th smallest absolute
//!   residual (Vovk, Gammerman & Shafer 2005; Lei et al. 2018, JASA);
//!   asymmetric mode calibrates the two tails separately on the signed
//!   residuals at `alpha/2` each. Under *exchangeable* scores the coverage
//!   guarantee is `P(covered) >= 1 - alpha` in finite samples — exactly
//!   `k/(m+1)` for continuous symmetric scores — and the result reports
//!   that implied `finite_sample_level`. Forecast residuals are not
//!   exchangeable in general; the guarantee then holds approximately for
//!   stationary, well-fit series (see Chernozhukov, Wüthrich & Zhu 2018
//!   for exact-under-mixing variants), and the honest empirical grades
//!   live in this crate's Monte Carlo suites.
//! * **EnbPI** ([`enbpi`]) — the ensemble batch prediction intervals of
//!   Xu & Xie (ICML 2021, Algorithm 1; journal restatement in IEEE TPAMI
//!   2023). Implemented in the paper's regression formulation with an
//!   autoregressive lag design built internally: fit `n_boot` least-squares
//!   AR(`lags`) models on iid row-resamples of the lagged design, aggregate
//!   *leave-one-out* (models whose bootstrap sample excludes row `i`)
//!   predictions to get out-of-bag residuals, and take the interval offsets
//!   from the empirical residual quantiles — by default the paper's
//!   width-minimizing `beta` line search over `[q_beta, q_{1-alpha+beta}]`,
//!   or the symmetric absolute-residual quantile (the authors' released
//!   code) with `optimize_beta = false`. No exchangeability is assumed:
//!   the published guarantee is approximate marginal coverage under
//!   stationary, strongly mixing errors. Multi-step forecasts feed the
//!   ensemble's own point predictions back into the lag vector (the batch
//!   `s = horizon` mode of the paper, with the plug-in recursion the
//!   autoregressive design requires — stated, since the paper's features
//!   are exogenously known). [`enbpi_online`] is the paper's online mode:
//!   the ensemble is fit once, then the residual window slides forward by
//!   `batch` as test labels arrive.
//! * **ACI** ([`aci`]) — adaptive conformal inference (Gibbs & Candès,
//!   NeurIPS 2021): run split conformal online, but replace the fixed
//!   miscoverage `alpha` with the recursion
//!   `alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`, where `err_t`
//!   indicates whether the level-`alpha_t` interval missed. Coverage
//!   self-corrects under distribution shift: too many misses drive
//!   `alpha_t` down (wider intervals) until the realized error rate
//!   returns to `alpha`, with the long-run guarantee
//!   `|mean(err) - alpha| <= (max(alpha_1, 1-alpha_1) + gamma) / (gamma T)`
//!   holding for *any* data sequence. The default step size
//!   `gamma = 0.005` is the value used throughout the paper's experiments
//!   ("chosen because it gives relatively stable trajectories for alpha_t
//!   while still adapting to observed shifts"). When `alpha_t` drifts at
//!   or below zero the interval is infinite (`err = 0`); at or above one
//!   it is empty — represented as the degenerate point interval at the
//!   forecast, `err = 1` — exactly the paper's convention. Updates for
//!   horizon `h` apply with delay `h` (an `h`-step miss is only observable
//!   `h` periods later), so every stream stays strictly online.
//!
//! # Leakage discipline (shared with the backtest engine)
//!
//! Calibration residuals at origin `t` come from a forecaster that saw
//! `y[0..=t]` only; scores for the horizon-`h` interval are `h`-step-ahead
//! residuals, never one-step proxies; online methods only consume an error
//! indicator once its target has been realized. Any preprocessing that
//! could peek (scaling, seasonal adjustment, tuning) belongs inside the
//! forecaster closure, which is handed nothing but the training slice.
//!
//! # The finite-sample quantile correction
//!
//! [`conformal_quantile`] exposes the primitive: the corrected
//! `(1-alpha)` quantile of `m` scores is the `k`-th smallest with
//! `k = ceil((m+1)(1-alpha))`. If `k > m` — the calibration set is too
//! small for the requested level (`m < ceil((1-alpha)/alpha)` for the
//! symmetric interval at `alpha`) — this module refuses with an error that
//! says how many scores are needed rather than silently returning the
//! sample maximum.
//!
//! # References
//!
//! * Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random
//!   World*, Springer.
//! * Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018), "Distribution-Free
//!   Predictive Inference for Regression," *JASA* 113.
//! * Xu & Xie (2021), "Conformal Prediction Interval for Dynamic
//!   Time-Series," *ICML* (PMLR 139); journal version "Conformal Prediction
//!   for Time Series," *IEEE TPAMI* 45 (2023).
//! * Gibbs & Candès (2021), "Adaptive Conformal Inference Under
//!   Distribution Shift," *NeurIPS* 34.

use crate::backtest::{Backtest, Window};
use crate::error::ForecastError;
use crate::validate::{check_finite, check_series, check_steps};
use tsecon_bootstrap::{indices, BlockScheme};
use tsecon_rng::Stream;

// --------------------------------------------------------------------------
// The quantile primitive
// --------------------------------------------------------------------------

/// Integer ceiling of a positive float that is robust to the 1-ulp noise of
/// products like `(m+1) * 0.9` (which binary floating point renders as
/// `9.000000000000002` for `m = 9`): values within `1e-9` above an integer
/// are treated as that integer.
fn robust_ceil(x: f64) -> usize {
    let k = (x - 1e-9).ceil();
    if k < 1.0 {
        1
    } else {
        k as usize
    }
}

/// The finite-sample-corrected conformal quantile: the
/// `ceil((m+1)(1-alpha))`-th smallest of `m` calibration scores.
///
/// Under exchangeability of the `m` scores and the test score, a prediction
/// set of the form `{y : score(y) <= q}` with `q` from this function covers
/// with probability at least `1 - alpha` in finite samples — exactly
/// `ceil((m+1)(1-alpha)) / (m+1)` when scores are continuous (Vovk,
/// Gammerman & Shafer 2005). The `+1` is the correction that makes the
/// guarantee finite-sample rather than asymptotic; it is what widens the
/// interval when `m` is small.
///
/// # Errors
///
/// [`ForecastError::InvalidAlpha`] if `alpha` is outside `(0, 1)`;
/// [`ForecastError::NonFinite`] on NaN/inf scores;
/// [`ForecastError::CalibrationTooSmall`] if `ceil((m+1)(1-alpha)) > m`,
/// i.e. the calibration set cannot support the requested level (the error
/// names the minimum `m`).
///
/// # Example
///
/// ```
/// use tsecon_forecast::conformal_quantile;
/// // 19 scores at alpha = 0.1: k = ceil(20 * 0.9) = 18 -> the 18th smallest.
/// let scores: Vec<f64> = (1..=19).map(|v| v as f64).collect();
/// assert_eq!(conformal_quantile(&scores, 0.1).unwrap(), 18.0);
/// // 5 scores cannot support alpha = 0.1: refused, not silently the max.
/// assert!(conformal_quantile(&scores[..5], 0.1).is_err());
/// ```
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> Result<f64, ForecastError> {
    check_alpha(alpha)?;
    check_finite(scores, "conformal quantile scores")?;
    let m = scores.len();
    let k = robust_ceil((m as f64 + 1.0) * (1.0 - alpha));
    if k > m {
        return Err(ForecastError::CalibrationTooSmall {
            what: "conformal quantile",
            n_calib: m,
            alpha,
            needed: min_calib(alpha),
        });
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    Ok(sorted[k - 1])
}

/// The smallest calibration size `m` with `ceil((m+1)(1-alpha)) <= m`,
/// i.e. the minimum number of scores that supports a finite corrected
/// quantile at miscoverage `alpha` (9 for `alpha = 0.1`, 19 for `0.05`).
fn min_calib(alpha: f64) -> usize {
    let mut m = ((1.0 - alpha) / alpha).ceil().max(1.0) as usize;
    while robust_ceil((m as f64 + 1.0) * (1.0 - alpha)) > m {
        m += 1;
    }
    m
}

fn check_alpha(alpha: f64) -> Result<(), ForecastError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ForecastError::InvalidAlpha { value: alpha });
    }
    Ok(())
}

/// Plain (uncorrected) type-1 empirical quantile: the `ceil(p * m)`-th
/// smallest of a **sorted** slice, with `p = 0` giving the minimum. This is
/// the EnbPI convention (its guarantee is asymptotic, so it uses ordinary
/// empirical quantiles rather than the `+1`-corrected ones).
fn empirical_quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let m = sorted.len();
    if p <= 0.0 {
        return sorted[0];
    }
    let k = robust_ceil(p * m as f64).min(m);
    sorted[k - 1]
}

/// Sort a copy of `v` ascending (NaN-free by prior validation).
fn sorted_copy(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    s
}

// --------------------------------------------------------------------------
// Shared machinery: the calibration residual grid and interval offsets
// --------------------------------------------------------------------------

/// `(origins, residuals-per-horizon, forecasts-per-horizon)` from the grid.
type ResidualGrid = (Vec<usize>, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Per-horizon signed residual streams on the rectangular backtest grid:
/// `resid[h-1][i] = y[t_i + h] - yhat` for origins `t_i = t0 .. n-1-H`,
/// where every origin's forecaster saw `y[0..=t_i]` only. Also returns the
/// origin indices and the raw point forecasts (bit-identical to what the
/// forecaster produced, so downstream results never re-derive them).
fn residual_grid<F>(
    y: &[f64],
    horizon: usize,
    n_origins: usize,
    what: &'static str,
    forecaster: &mut F,
) -> Result<ResidualGrid, ForecastError>
where
    F: FnMut(&[f64], usize) -> Result<Vec<f64>, ForecastError>,
{
    let n = y.len();
    // p origins ending at n-1-H  =>  first training window has
    // min_train = n - horizon - p + 1 observations.
    let needed = horizon + n_origins;
    if n < needed + 1 {
        return Err(ForecastError::SeriesTooShort {
            what,
            n,
            needed: needed + 1,
        });
    }
    let min_train = n - horizon - n_origins + 1;
    let bt = Backtest::new(Window::Expanding { min_train }, horizon, 1)?;
    let res = bt.run(y, |train, h| forecaster(train, h))?;
    let mut resid = Vec::with_capacity(horizon);
    let mut fc = Vec::with_capacity(horizon);
    for h in 1..=horizon {
        resid.push(res.errors(h)?);
        fc.push(res.forecasts(h)?.to_vec());
    }
    Ok((res.origins().to_vec(), resid, fc))
}

/// The forward point forecast from the full sample, validated like a
/// backtest closure output (right length, finite).
fn forward_forecast<F>(
    y: &[f64],
    horizon: usize,
    forecaster: &mut F,
) -> Result<Vec<f64>, ForecastError>
where
    F: FnMut(&[f64], usize) -> Result<Vec<f64>, ForecastError>,
{
    let fc = forecaster(y, horizon)?;
    if fc.len() != horizon {
        return Err(ForecastError::ForecasterOutputLen {
            origin: y.len() - 1,
            expected: horizon,
            actual: fc.len(),
        });
    }
    check_finite(&fc, "conformal forward forecast")?;
    Ok(fc)
}

/// Interval offsets `(q_lower, q_upper)` from signed residuals with the
/// finite-sample correction. Symmetric: `(-q, +q)` with `q` the corrected
/// `(1-alpha)` quantile of the absolute residuals. Asymmetric: the two
/// tails calibrated separately on the signed residuals at `alpha/2` each —
/// upper offset the `ceil((m+1)(1-alpha/2))`-th smallest signed residual,
/// lower offset its mirror order statistic `m+1-k` from the bottom.
fn split_offsets(
    resid: &[f64],
    alpha: f64,
    symmetric: bool,
    what: &'static str,
) -> Result<(f64, f64), ForecastError> {
    let m = resid.len();
    if symmetric {
        let abs: Vec<f64> = resid.iter().map(|r| r.abs()).collect();
        let q = match conformal_quantile(&abs, alpha) {
            Ok(q) => q,
            Err(ForecastError::CalibrationTooSmall { alpha, needed, .. }) => {
                return Err(ForecastError::CalibrationTooSmall {
                    what,
                    n_calib: m,
                    alpha,
                    needed,
                })
            }
            Err(e) => return Err(e),
        };
        Ok((-q, q))
    } else {
        let k_up = robust_ceil((m as f64 + 1.0) * (1.0 - alpha / 2.0));
        if k_up > m {
            return Err(ForecastError::CalibrationTooSmall {
                what,
                n_calib: m,
                alpha: alpha / 2.0,
                needed: min_calib(alpha / 2.0),
            });
        }
        let sorted = sorted_copy(resid);
        let k_lo = m + 1 - k_up; // >= 1 because k_up <= m
        Ok((sorted[k_lo - 1], sorted[k_up - 1]))
    }
}

/// The exact coverage the corrected quantile targets under exchangeable
/// continuous scores: `k/(m+1)` (symmetric) or `(k_up - k_lo)/(m+1)`
/// (asymmetric) — always `>= 1 - alpha`.
fn finite_sample_level(m: usize, alpha: f64, symmetric: bool) -> f64 {
    if symmetric {
        let k = robust_ceil((m as f64 + 1.0) * (1.0 - alpha));
        k as f64 / (m as f64 + 1.0)
    } else {
        let k_up = robust_ceil((m as f64 + 1.0) * (1.0 - alpha / 2.0));
        let k_lo = m + 1 - k_up;
        (k_up - k_lo) as f64 / (m as f64 + 1.0)
    }
}

// --------------------------------------------------------------------------
// Split conformal
// --------------------------------------------------------------------------

/// Options for [`split_conformal`] / [`split_conformal_online`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitConformalOptions {
    /// Forecast horizon `H`; every horizon `1..=H` gets its own residual
    /// calibration.
    pub horizon: usize,
    /// Nominal miscoverage `alpha` (0.1 for a 90% interval).
    pub alpha: f64,
    /// Number of calibration residuals per horizon (`m`), taken from the
    /// most recent origins of the rectangular grid.
    pub calib: usize,
    /// `true`: symmetric absolute-residual intervals. `false`: asymmetric
    /// signed-residual intervals with the tails calibrated at `alpha/2`
    /// each (the interval may exclude the point forecast when the base is
    /// biased — that asymmetry is the mode's purpose).
    pub symmetric: bool,
}

impl Default for SplitConformalOptions {
    fn default() -> Self {
        SplitConformalOptions {
            horizon: 1,
            alpha: 0.1,
            calib: 50,
            symmetric: true,
        }
    }
}

/// Result of [`split_conformal`]: forward point forecasts with
/// residual-calibrated intervals plus the calibration diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitConformalForecast {
    /// Point forecasts for horizons `1..=horizon` from the full sample.
    pub mean: Vec<f64>,
    /// Lower interval bounds, `mean + q_lower`.
    pub lower: Vec<f64>,
    /// Upper interval bounds, `mean + q_upper`.
    pub upper: Vec<f64>,
    /// Nominal coverage level `1 - alpha`.
    pub level: f64,
    /// Nominal miscoverage `alpha`.
    pub alpha: f64,
    /// Per-horizon lower offsets (`-q` in symmetric mode).
    pub q_lower: Vec<f64>,
    /// Per-horizon upper offsets.
    pub q_upper: Vec<f64>,
    /// Per-horizon signed calibration residuals `y[t+h] - yhat` (one per
    /// calibration origin, ascending origin order) — the raw scores, so a
    /// caller can audit exactly what the quantiles were taken over.
    pub scores: Vec<Vec<f64>>,
    /// Number of calibration residuals per horizon.
    pub n_calib: usize,
    /// The coverage the corrected quantile targets *under exchangeable
    /// continuous scores*: `ceil((m+1)(1-alpha))/(m+1) >= 1 - alpha`.
    /// For dependent forecast residuals this is the design target, not a
    /// finite-sample theorem.
    pub finite_sample_level: f64,
    /// Whether symmetric (absolute-residual) calibration was used.
    pub symmetric: bool,
}

/// Split-conformal forecast intervals around an arbitrary point forecaster.
///
/// Calibrates on the last `opts.calib` origins of the rectangular backtest
/// grid (all `horizon` targets in-sample at every origin, expanding
/// training windows, refit at every origin) and applies the
/// finite-sample-corrected residual quantiles to the forward forecast from
/// the full sample. See the [module docs](self) for the method, the
/// guarantee and its limits.
///
/// The forecaster closure has the same contract as
/// [`Backtest::run`](crate::backtest::Backtest::run): called as
/// `forecaster(train, h)`, returns exactly `h` point forecasts for steps
/// `1..=h` from the end of `train`.
///
/// # Errors
///
/// [`ForecastError::InvalidSteps`] (`horizon = 0`),
/// [`ForecastError::InvalidAlpha`],
/// [`ForecastError::CalibrationTooSmall`] (`calib` cannot support `alpha`
/// at the requested mode), [`ForecastError::SeriesTooShort`],
/// [`ForecastError::NonFinite`], plus whatever the forecaster itself
/// returns.
///
/// # Example
///
/// ```
/// use tsecon_forecast::{split_conformal, SplitConformalOptions};
/// let y: Vec<f64> = (0..80).map(|t| (t as f64 * 0.7).sin() + 0.01 * t as f64).collect();
/// let opts = SplitConformalOptions { horizon: 2, alpha: 0.2, calib: 30, symmetric: true };
/// // A naive (last-value) base forecaster.
/// let r = split_conformal(&y, &opts, |train, h| {
///     Ok(vec![*train.last().unwrap(); h])
/// }).unwrap();
/// assert_eq!(r.mean.len(), 2);
/// assert!(r.lower[0] <= r.mean[0] && r.mean[0] <= r.upper[0]);
/// ```
pub fn split_conformal<F>(
    y: &[f64],
    opts: &SplitConformalOptions,
    mut forecaster: F,
) -> Result<SplitConformalForecast, ForecastError>
where
    F: FnMut(&[f64], usize) -> Result<Vec<f64>, ForecastError>,
{
    const WHAT: &str = "split conformal";
    check_steps(opts.horizon)?;
    check_alpha(opts.alpha)?;
    check_calib(opts.calib, opts.alpha, opts.symmetric, WHAT)?;
    let (_origins, resid, _fc) = residual_grid(y, opts.horizon, opts.calib, WHAT, &mut forecaster)?;
    let mean = forward_forecast(y, opts.horizon, &mut forecaster)?;

    let mut lower = Vec::with_capacity(opts.horizon);
    let mut upper = Vec::with_capacity(opts.horizon);
    let mut q_lower = Vec::with_capacity(opts.horizon);
    let mut q_upper = Vec::with_capacity(opts.horizon);
    for (h_idx, r) in resid.iter().enumerate() {
        let (lo, up) = split_offsets(r, opts.alpha, opts.symmetric, WHAT)?;
        lower.push(mean[h_idx] + lo);
        upper.push(mean[h_idx] + up);
        q_lower.push(lo);
        q_upper.push(up);
    }
    Ok(SplitConformalForecast {
        mean,
        lower,
        upper,
        level: 1.0 - opts.alpha,
        alpha: opts.alpha,
        q_lower,
        q_upper,
        scores: resid,
        n_calib: opts.calib,
        finite_sample_level: finite_sample_level(opts.calib, opts.alpha, opts.symmetric),
        symmetric: opts.symmetric,
    })
}

/// Validate a calibration size against the corrected quantile's minimum.
fn check_calib(
    calib: usize,
    alpha: f64,
    symmetric: bool,
    what: &'static str,
) -> Result<(), ForecastError> {
    let tail_alpha = if symmetric { alpha } else { alpha / 2.0 };
    let needed = min_calib(tail_alpha);
    if calib < needed {
        return Err(ForecastError::CalibrationTooSmall {
            what,
            n_calib: calib,
            alpha: tail_alpha,
            needed,
        });
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Online evaluation (split and ACI share this result shape)
// --------------------------------------------------------------------------

/// Per-origin online conformal evaluation over a held-out window: the
/// realized intervals, miss indicators, and coverage, per horizon.
///
/// Produced by [`split_conformal_online`], [`enbpi_online`], and (with
/// `alpha_trajectory` populated) by [`aci`]. All per-horizon vectors have
/// one entry per evaluation origin, ascending; `err[h-1][j] = true` means
/// the horizon-`h` interval formed at origin `origins[j]` missed its
/// realized target `y[origins[j] + h]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformalOnline {
    /// Maximum horizon evaluated.
    pub horizon: usize,
    /// Nominal miscoverage `alpha`.
    pub alpha: f64,
    /// Nominal coverage level `1 - alpha`.
    pub level: f64,
    /// The evaluation origin indices (the forecaster at origin `t` saw
    /// `y[0..=t]`).
    pub origins: Vec<usize>,
    /// Per-horizon point forecasts at each evaluation origin.
    pub mean: Vec<Vec<f64>>,
    /// Per-horizon lower interval bounds.
    pub lower: Vec<Vec<f64>>,
    /// Per-horizon upper interval bounds.
    pub upper: Vec<Vec<f64>>,
    /// Per-horizon miss indicators against the realized targets.
    pub err: Vec<Vec<bool>>,
    /// Per-horizon realized coverage `1 - mean(err)` over the window.
    pub realized_coverage: Vec<f64>,
    /// The per-horizon `alpha_t` trajectory (the level *used* at each
    /// origin) — populated by [`aci`] only, `None` for the fixed-level
    /// methods.
    pub alpha_trajectory: Option<Vec<Vec<f64>>>,
}

/// Options for [`split_conformal_online`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitOnlineOptions {
    /// Forecast horizon `H`.
    pub horizon: usize,
    /// Nominal miscoverage `alpha`.
    pub alpha: f64,
    /// Trailing calibration window size (residuals per horizon).
    pub calib: usize,
    /// Number of evaluation origins.
    pub n_eval: usize,
    /// Symmetric or asymmetric calibration (as in
    /// [`SplitConformalOptions::symmetric`]).
    pub symmetric: bool,
}

/// Rolling-recalibrated split conformal evaluated online: at each of the
/// last `n_eval` origins, calibrate on the trailing `calib` *realized*
/// `h`-step residuals (a residual is realized once its target has been
/// observed — for horizon `h` that excludes the `h - 1` most recent
/// origins, so early evaluation windows for long horizons hold up to
/// `h - 1` fewer scores) and score the interval against the realized
/// target. This is the honest way to measure what coverage split conformal
/// actually delivers on a given series — and the fixed-level baseline that
/// [`aci`] adapts.
///
/// # Errors
///
/// As [`split_conformal`], plus `n_eval = 0` is
/// [`ForecastError::InvalidConformalParam`]. `calib` must support the
/// level even in the smallest (horizon-`H` start-of-window) case, i.e.
/// `calib - horizon + 1` realized scores must suffice.
pub fn split_conformal_online<F>(
    y: &[f64],
    opts: &SplitOnlineOptions,
    mut forecaster: F,
) -> Result<ConformalOnline, ForecastError>
where
    F: FnMut(&[f64], usize) -> Result<Vec<f64>, ForecastError>,
{
    const WHAT: &str = "split conformal (online)";
    check_steps(opts.horizon)?;
    check_alpha(opts.alpha)?;
    if opts.n_eval == 0 {
        return Err(ForecastError::InvalidConformalParam {
            what: "n_eval",
            value: 0.0,
            requirement: "n_eval >= 1 evaluation origins",
        });
    }
    // The smallest trailing window (first eval origin, horizon H) holds
    // calib - (H - 1) realized residuals; require that to support alpha.
    if opts.calib < opts.horizon - 1 {
        return Err(ForecastError::InvalidConformalParam {
            what: "calib",
            value: opts.calib as f64,
            requirement: "calib >= horizon - 1 (trailing h-step residuals must exist)",
        });
    }
    check_calib(
        opts.calib - (opts.horizon - 1),
        opts.alpha,
        opts.symmetric,
        WHAT,
    )?;

    let p = opts.calib + opts.n_eval;
    let (origins, resid, fc) = residual_grid(y, opts.horizon, p, WHAT, &mut forecaster)?;

    let mut mean = vec![Vec::with_capacity(opts.n_eval); opts.horizon];
    let mut lower = vec![Vec::with_capacity(opts.n_eval); opts.horizon];
    let mut upper = vec![Vec::with_capacity(opts.n_eval); opts.horizon];
    let mut err = vec![Vec::with_capacity(opts.n_eval); opts.horizon];
    for h in 1..=opts.horizon {
        let r = &resid[h - 1];
        for j in 0..opts.n_eval {
            let e = opts.calib + j; // grid index of this eval origin
            let target = y[origins[e] + h];
            let point = fc[h - 1][e];
            // Trailing window of realized h-step residuals: grid indices
            // i <= e - h, at most `calib` of them.
            let hi = e - h; // e >= calib >= h - 1 guaranteed; e - h could be
                            // -1 only if calib = h - 1 and j = 0... guarded:
            let window_lo = (hi + 1).saturating_sub(opts.calib);
            let window = &r[window_lo..=hi];
            let (lo_off, up_off) = split_offsets(window, opts.alpha, opts.symmetric, WHAT)?;
            let lo = point + lo_off;
            let up = point + up_off;
            mean[h - 1].push(point);
            lower[h - 1].push(lo);
            upper[h - 1].push(up);
            err[h - 1].push(target < lo || target > up);
        }
    }
    let realized_coverage = err
        .iter()
        .map(|e| 1.0 - e.iter().filter(|&&m| m).count() as f64 / e.len() as f64)
        .collect();
    Ok(ConformalOnline {
        horizon: opts.horizon,
        alpha: opts.alpha,
        level: 1.0 - opts.alpha,
        origins: origins[opts.calib..].to_vec(),
        mean,
        lower,
        upper,
        err,
        realized_coverage,
        alpha_trajectory: None,
    })
}

// --------------------------------------------------------------------------
// ACI — adaptive conformal inference
// --------------------------------------------------------------------------

/// Options for [`aci`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AciOptions {
    /// Forecast horizon `H`; each horizon runs its own independent ACI
    /// stream (updates apply with delay `h`).
    pub horizon: usize,
    /// Nominal (target) miscoverage `alpha`; also the starting `alpha_1`.
    pub alpha: f64,
    /// Step size `gamma` of the update
    /// `alpha_{t+1} = alpha_t + gamma (alpha - err_t)`. The paper's
    /// experiments use `0.005`.
    pub gamma: f64,
    /// Trailing calibration window size (residuals per horizon).
    pub calib: usize,
    /// Number of online evaluation origins (the window the recursion runs
    /// over before the forward forecast).
    pub n_eval: usize,
}

impl Default for AciOptions {
    fn default() -> Self {
        AciOptions {
            horizon: 1,
            alpha: 0.1,
            gamma: 0.005,
            calib: 50,
            n_eval: 50,
        }
    }
}

/// Result of [`aci`]: the forward forecast at the adapted level plus the
/// full online trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct AciForecast {
    /// Forward point forecasts for horizons `1..=horizon` from the full
    /// sample.
    pub mean: Vec<f64>,
    /// Lower bounds at the adapted per-horizon levels `alpha_final`
    /// (`-inf` when the adapted interval is infinite).
    pub lower: Vec<f64>,
    /// Upper bounds at the adapted levels (`+inf` when infinite).
    pub upper: Vec<f64>,
    /// Nominal coverage level `1 - alpha` (the target the recursion
    /// steers toward — the *realized* level is in `online`).
    pub level: f64,
    /// Nominal miscoverage `alpha`.
    pub alpha: f64,
    /// The step size used.
    pub gamma: f64,
    /// Per-horizon `alpha_t` after all realized errors were absorbed —
    /// the level the forward intervals use. Can leave `(0, 1)`; see the
    /// [module docs](self) for the empty/infinite conventions.
    pub alpha_final: Vec<f64>,
    /// Trailing-window size used for every quantile.
    pub n_calib: usize,
    /// The online run: per-origin intervals, misses, realized coverage,
    /// and the `alpha_t` trajectory (`alpha_trajectory` is `Some`).
    pub online: ConformalOnline,
}

/// Adaptive conformal inference (Gibbs & Candès 2021) around an arbitrary
/// point forecaster: rolling split-conformal intervals whose miscoverage
/// level `alpha_t` is updated online by
/// `alpha_{t+1} = alpha_t + gamma (alpha - err_t)`.
///
/// Scores are absolute `h`-step residuals over a trailing window of
/// `opts.calib` realized residuals; the recursion runs over the last
/// `opts.n_eval` origins, and the forward interval uses the final adapted
/// level. When `alpha_t` makes the corrected quantile index exceed the
/// window (in particular whenever `alpha_t <= 0`) the interval is
/// **infinite** and `err_t = 0`; when `alpha_t >= 1` it is **empty**
/// (represented as the degenerate point interval at the forecast) and
/// `err_t = 1` — the paper's conventions. Horizon-`h` errors update the
/// stream only once realized, i.e. with delay `h`.
///
/// # Errors
///
/// As [`split_conformal_online`], plus a non-finite or negative `gamma` is
/// [`ForecastError::InvalidConformalParam`] (`gamma = 0` is allowed and
/// reduces ACI to rolling split conformal — useful as a control).
pub fn aci<F>(y: &[f64], opts: &AciOptions, mut forecaster: F) -> Result<AciForecast, ForecastError>
where
    F: FnMut(&[f64], usize) -> Result<Vec<f64>, ForecastError>,
{
    const WHAT: &str = "adaptive conformal inference";
    check_steps(opts.horizon)?;
    check_alpha(opts.alpha)?;
    if !opts.gamma.is_finite() || opts.gamma < 0.0 {
        return Err(ForecastError::InvalidConformalParam {
            what: "gamma",
            value: opts.gamma,
            requirement: "a finite step size gamma >= 0 (the paper's experiments use 0.005)",
        });
    }
    if opts.n_eval == 0 {
        return Err(ForecastError::InvalidConformalParam {
            what: "n_eval",
            value: 0.0,
            requirement:
                "n_eval >= 1 online origins (the alpha_t recursion needs a window to run over)",
        });
    }
    if opts.calib < opts.horizon - 1 {
        return Err(ForecastError::InvalidConformalParam {
            what: "calib",
            value: opts.calib as f64,
            requirement: "calib >= horizon - 1 (trailing h-step residuals must exist)",
        });
    }
    // The *starting* level alpha must be supported by the smallest window;
    // adapted levels below that produce infinite intervals by design.
    check_calib(opts.calib - (opts.horizon - 1), opts.alpha, true, WHAT)?;

    let p = opts.calib + opts.n_eval;
    let (origins, resid, fc) = residual_grid(y, opts.horizon, p, WHAT, &mut forecaster)?;

    let h_max = opts.horizon;
    let mut mean = vec![Vec::with_capacity(opts.n_eval); h_max];
    let mut lower = vec![Vec::with_capacity(opts.n_eval); h_max];
    let mut upper = vec![Vec::with_capacity(opts.n_eval); h_max];
    let mut err = vec![Vec::with_capacity(opts.n_eval); h_max];
    let mut trajectory = vec![Vec::with_capacity(opts.n_eval); h_max];
    let mut alpha_final = Vec::with_capacity(h_max);

    for h in 1..=h_max {
        let r = &resid[h - 1];
        let mut alpha_t = opts.alpha;
        let mut absorbed = 0usize; // how many err_j have been applied
        for j in 0..opts.n_eval {
            // Absorb every error realized by this origin: err_j' is known
            // once its target y[origin_j' + h] is observed, i.e. from
            // origin index j' + h onward => all j' <= j - h.
            while absorbed + h <= j {
                let e_j = err[h - 1][absorbed];
                alpha_t += opts.gamma * (opts.alpha - f64::from(u8::from(e_j)));
                absorbed += 1;
            }
            trajectory[h - 1].push(alpha_t);

            let e = opts.calib + j;
            let target = y[origins[e] + h];
            let point = fc[h - 1][e];
            let hi = e - h;
            let window_lo = (hi + 1).saturating_sub(opts.calib);
            let window = &r[window_lo..=hi];
            let (lo, up, miss) = aci_interval(window, alpha_t, point, target);
            mean[h - 1].push(point);
            lower[h - 1].push(lo);
            upper[h - 1].push(up);
            err[h - 1].push(miss);
        }
        // Absorb the rest (all errors are realized by the end of sample:
        // every eval target y[origin + h] is inside y by construction).
        while absorbed < opts.n_eval {
            let e_j = err[h - 1][absorbed];
            alpha_t += opts.gamma * (opts.alpha - f64::from(u8::from(e_j)));
            absorbed += 1;
        }
        alpha_final.push(alpha_t);
    }

    // Forward forecast at the adapted levels, on the trailing residuals.
    let fwd = forward_forecast(y, h_max, &mut forecaster)?;
    let mut f_lower = Vec::with_capacity(h_max);
    let mut f_upper = Vec::with_capacity(h_max);
    for h in 1..=h_max {
        let r = &resid[h - 1];
        let window_lo = r.len().saturating_sub(opts.calib);
        let window = &r[window_lo..];
        let (lo, up) = aci_bounds(window, alpha_final[h - 1], fwd[h - 1]);
        f_lower.push(lo);
        f_upper.push(up);
    }

    let realized_coverage = err
        .iter()
        .map(|e| 1.0 - e.iter().filter(|&&m| m).count() as f64 / e.len() as f64)
        .collect();
    Ok(AciForecast {
        mean: fwd,
        lower: f_lower,
        upper: f_upper,
        level: 1.0 - opts.alpha,
        alpha: opts.alpha,
        gamma: opts.gamma,
        alpha_final,
        n_calib: opts.calib,
        online: ConformalOnline {
            horizon: h_max,
            alpha: opts.alpha,
            level: 1.0 - opts.alpha,
            origins: origins[opts.calib..].to_vec(),
            mean,
            lower,
            upper,
            err,
            realized_coverage,
            alpha_trajectory: Some(trajectory),
        },
    })
}

/// The ACI interval at level `alpha_t` around `point`, plus the miss
/// indicator against `target`. Symmetric absolute-residual scores.
fn aci_interval(window: &[f64], alpha_t: f64, point: f64, target: f64) -> (f64, f64, bool) {
    let (lo, up) = aci_bounds(window, alpha_t, point);
    if up < lo {
        // Empty interval (alpha_t >= 1): degenerate point, always a miss
        // for continuous data.
        return (point, point, !(target == point));
    }
    (lo, up, target < lo || target > up)
}

/// Bounds only (used for the forward interval, where no target exists).
/// `alpha_t >= 1` yields the degenerate empty interval `(point, point)`
/// flagged by `upper < lower` upstream — here represented as the point.
fn aci_bounds(window: &[f64], alpha_t: f64, point: f64) -> (f64, f64) {
    let m = window.len();
    if alpha_t >= 1.0 {
        return (point, point);
    }
    let k = if alpha_t <= 0.0 {
        m + 1 // force the infinite branch
    } else {
        robust_ceil((m as f64 + 1.0) * (1.0 - alpha_t))
    };
    if k > m {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let abs: Vec<f64> = window.iter().map(|r| r.abs()).collect();
    let sorted = sorted_copy(&abs);
    let q = sorted[k - 1];
    (point - q, point + q)
}

// --------------------------------------------------------------------------
// EnbPI — ensemble batch prediction intervals (Xu & Xie 2021)
// --------------------------------------------------------------------------

/// Options for [`enbpi`] / [`enbpi_online`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnbpiOptions {
    /// Forecast horizon `H` (the batch `s = H` mode: all `H` steps share
    /// the current residual quantiles; lag values beyond the sample are
    /// plug-in ensemble predictions).
    pub horizon: usize,
    /// Nominal miscoverage `alpha`.
    pub alpha: f64,
    /// Autoregressive order of the internal lagged design (the base
    /// learner is least squares of `y_t` on `1, y_{t-1}, ..., y_{t-lags}`).
    pub lags: usize,
    /// Number of bootstrap models `B`.
    pub n_boot: usize,
    /// Seed for the bootstrap index draws (Philox substreams, one per
    /// model — bit-reproducible at any thread count).
    pub seed: u64,
    /// `true` (the ICML 2021 Algorithm 1 form): signed residuals with the
    /// width-minimizing `beta` line search over
    /// `[q_beta, q_{1-alpha+beta}]`. `false` (the authors' released code):
    /// symmetric absolute-residual quantile at `1 - alpha`.
    pub optimize_beta: bool,
    /// Grid points for the `beta` line search over `[0, alpha]`.
    pub n_beta: usize,
}

impl Default for EnbpiOptions {
    fn default() -> Self {
        EnbpiOptions {
            horizon: 1,
            alpha: 0.1,
            lags: 1,
            n_boot: 25,
            seed: 0,
            optimize_beta: true,
            n_beta: 21,
        }
    }
}

/// Result of [`enbpi`].
#[derive(Debug, Clone, PartialEq)]
pub struct EnbpiForecast {
    /// Point forecasts for horizons `1..=horizon`: the leave-one-out
    /// aggregated ensemble prediction (Algorithm 1, line 12), recursed for
    /// multi-step.
    pub mean: Vec<f64>,
    /// Lower bounds `mean + q_beta` (or `mean - q_{1-alpha}(|resid|)`).
    pub lower: Vec<f64>,
    /// Upper bounds `mean + q_{1-alpha+beta}` (or `+ q_{1-alpha}(|resid|)`).
    pub upper: Vec<f64>,
    /// Nominal coverage level `1 - alpha`.
    pub level: f64,
    /// Nominal miscoverage `alpha`.
    pub alpha: f64,
    /// The width-minimizing `beta` chosen by the line search (`None` when
    /// `optimize_beta = false`).
    pub beta: Option<f64>,
    /// The leave-one-out (out-of-bag) residuals in time order — the
    /// calibration scores the quantiles were taken over.
    pub residuals: Vec<f64>,
    /// Number of residuals (`n - lags` minus any rows every bootstrap
    /// sample happened to contain).
    pub n_calib: usize,
    /// Rows excluded because every bootstrap sample contained them (no
    /// leave-one-out model exists; vanishingly rare for `n_boot >= 20`).
    pub n_excluded: usize,
    /// The autoregressive order used.
    pub lags: usize,
    /// The ensemble size used.
    pub n_boot: usize,
}

/// A fitted EnbPI ensemble: per-model coefficients plus the row-membership
/// structure needed for leave-one-out aggregation.
struct EnbpiEnsemble {
    /// Per-model OLS coefficients `[intercept, phi_1, ..., phi_lags]`.
    coef: Vec<Vec<f64>>,
    /// `in_bag[b][i]`: did model `b`'s bootstrap sample contain row `i`?
    in_bag: Vec<Vec<bool>>,
    /// Rows with at least one leave-one-out model, in time order.
    loo_rows: Vec<usize>,
}

impl EnbpiEnsemble {
    /// Predict with model `b` at a lag vector (most recent value last).
    fn predict_one(&self, b: usize, lag_vec: &[f64]) -> f64 {
        let c = &self.coef[b];
        let mut v = c[0];
        for (l, &x) in lag_vec.iter().rev().enumerate() {
            v += c[1 + l] * x;
        }
        v
    }

    /// The Algorithm-1 line-12 test-point predictor: the average over
    /// training rows `i` of the leave-one-out aggregations
    /// `phi({fhat^b(x): i not in S_b})` (`phi` = mean).
    fn loo_center(&self, lag_vec: &[f64]) -> f64 {
        let preds: Vec<f64> = (0..self.coef.len())
            .map(|b| self.predict_one(b, lag_vec))
            .collect();
        let mut total = 0.0;
        for &i in &self.loo_rows {
            let mut s = 0.0;
            let mut c = 0usize;
            for (b, p) in preds.iter().enumerate() {
                if !self.in_bag[b][i] {
                    s += p;
                    c += 1;
                }
            }
            total += s / c as f64; // c >= 1 by loo_rows construction
        }
        total / self.loo_rows.len() as f64
    }
}

/// Fit the bootstrap ensemble and compute leave-one-out residuals on
/// `y[..t_end]` (rows `lags..t_end`). Returns the ensemble and the
/// time-ordered LOO residuals.
fn fit_enbpi_ensemble(
    y: &[f64],
    t_end: usize,
    opts: &EnbpiOptions,
) -> Result<(EnbpiEnsemble, Vec<f64>, usize), ForecastError> {
    let lags = opts.lags;
    let t_rows = t_end - lags; // rows lags..t_end
    let row_index = |i: usize| i + lags; // design row i targets y[i + lags]

    // Bootstrap index draws: one Philox substream per model, so results are
    // reproducible bit-for-bit regardless of evaluation order.
    let mut streams = Stream::substreams(opts.seed, opts.n_boot)
        .map_err(tsecon_bootstrap::BootstrapError::from)?;
    let mut coef = Vec::with_capacity(opts.n_boot);
    let mut in_bag = vec![vec![false; t_rows]; opts.n_boot];
    for (b, stream) in streams.iter_mut().enumerate() {
        let sample = indices(BlockScheme::Iid, t_rows, stream)?;
        for &i in &sample {
            in_bag[b][i] = true;
        }
        coef.push(fit_ar_ols_rows(
            y,
            lags,
            &sample.iter().map(|&i| row_index(i)).collect::<Vec<_>>(),
        )?);
    }

    // Leave-one-out residuals in time order.
    let ens = EnbpiEnsemble {
        coef,
        in_bag,
        loo_rows: Vec::new(),
    };
    let mut loo_rows = Vec::with_capacity(t_rows);
    let mut residuals = Vec::with_capacity(t_rows);
    let mut excluded = 0usize;
    for i in 0..t_rows {
        let t = row_index(i);
        let lag_vec = &y[t - lags..t];
        let mut s = 0.0;
        let mut c = 0usize;
        for b in 0..opts.n_boot {
            if !ens.in_bag[b][i] {
                s += ens.predict_one(b, lag_vec);
                c += 1;
            }
        }
        if c == 0 {
            excluded += 1;
            continue;
        }
        loo_rows.push(i);
        residuals.push(y[t] - s / c as f64);
    }
    if loo_rows.is_empty() {
        // Only possible in pathological tiny-B cases; refuse loudly.
        return Err(ForecastError::InvalidConformalParam {
            what: "n_boot",
            value: opts.n_boot as f64,
            requirement: "an ensemble large enough that some bootstrap sample excludes some row \
                 (n_boot >= 20 makes exclusion failures vanishingly rare)",
        });
    }
    let ens = EnbpiEnsemble { loo_rows, ..ens };
    Ok((ens, residuals, excluded))
}

/// EnbPI interval offsets from a residual window: the `beta` line search
/// over `[q_beta, q_{1-alpha+beta}]` (signed residuals, Algorithm 1) or
/// the symmetric absolute quantile. Returns `(lo, up, beta)`.
fn enbpi_offsets(window: &[f64], opts: &EnbpiOptions) -> (f64, f64, Option<f64>) {
    if opts.optimize_beta {
        let sorted = sorted_copy(window);
        let grid = opts.n_beta.max(2);
        let mut best = (f64::INFINITY, 0.0, 0.0, 0.0);
        for g in 0..grid {
            let beta = opts.alpha * g as f64 / (grid - 1) as f64;
            let lo = empirical_quantile_sorted(&sorted, beta);
            let up = empirical_quantile_sorted(&sorted, 1.0 - opts.alpha + beta);
            let width = up - lo;
            if width < best.0 {
                best = (width, lo, up, beta);
            }
        }
        (best.1, best.2, Some(best.3))
    } else {
        let abs: Vec<f64> = window.iter().map(|r| r.abs()).collect();
        let sorted = sorted_copy(&abs);
        let q = empirical_quantile_sorted(&sorted, 1.0 - opts.alpha);
        (-q, q, None)
    }
}

fn check_enbpi(y: &[f64], opts: &EnbpiOptions, reserve: usize) -> Result<(), ForecastError> {
    const WHAT: &str = "EnbPI";
    check_steps(opts.horizon)?;
    check_alpha(opts.alpha)?;
    if opts.lags == 0 {
        return Err(ForecastError::InvalidConformalParam {
            what: "lags",
            value: 0.0,
            requirement: "lags >= 1 (the autoregressive design needs at least one lag)",
        });
    }
    if opts.n_boot < 2 {
        return Err(ForecastError::InvalidConformalParam {
            what: "n_boot",
            value: opts.n_boot as f64,
            requirement: "n_boot >= 2 bootstrap models (the paper's regime is a few dozen)",
        });
    }
    if opts.optimize_beta && opts.n_beta < 2 {
        return Err(ForecastError::InvalidConformalParam {
            what: "n_beta",
            value: opts.n_beta as f64,
            requirement: "n_beta >= 2 grid points over [0, alpha] (or set optimize_beta = false)",
        });
    }
    // Training rows after reserving `reserve` evaluation points: need at
    // least lags + 2 rows to identify the lags + 1 OLS coefficients with a
    // residual degree of freedom.
    let needed = 2 * opts.lags + 2 + reserve;
    check_series(y, needed, WHAT)?;
    Ok(())
}

/// EnbPI forecast intervals (Xu & Xie 2021, Algorithm 1) from the end of
/// the sample: bootstrap-ensemble AR(`lags`) least squares with
/// leave-one-out residual calibration. See the [module docs](self) for the
/// algorithm, the version implemented, and the honest caveats (plug-in
/// recursion for multi-step; batch-shared quantiles).
///
/// # Errors
///
/// [`ForecastError::InvalidSteps`], [`ForecastError::InvalidAlpha`],
/// [`ForecastError::InvalidConformalParam`] (`lags = 0`, `n_boot < 2`,
/// degenerate `n_beta`), [`ForecastError::SeriesTooShort`],
/// [`ForecastError::NonFinite`], or
/// [`ForecastError::SingularArDesign`] if a bootstrap sample's lagged
/// design is collinear (e.g. a constant series).
pub fn enbpi(y: &[f64], opts: &EnbpiOptions) -> Result<EnbpiForecast, ForecastError> {
    check_enbpi(y, opts, 0)?;
    let n = y.len();
    let (ens, residuals, excluded) = fit_enbpi_ensemble(y, n, opts)?;
    let (lo, up, beta) = enbpi_offsets(&residuals, opts);

    // Multi-step recursion: lag values beyond the sample are the
    // ensemble's own point predictions (batch s = horizon).
    let mut path: Vec<f64> = y[n - opts.lags..].to_vec();
    let mut mean = Vec::with_capacity(opts.horizon);
    for _ in 0..opts.horizon {
        let center = ens.loo_center(&path[path.len() - opts.lags..]);
        mean.push(center);
        path.push(center);
    }
    let lower = mean.iter().map(|m| m + lo).collect();
    let upper = mean.iter().map(|m| m + up).collect();
    Ok(EnbpiForecast {
        mean,
        lower,
        upper,
        level: 1.0 - opts.alpha,
        alpha: opts.alpha,
        beta,
        n_calib: residuals.len(),
        residuals,
        n_excluded: excluded,
        lags: opts.lags,
        n_boot: opts.n_boot,
    })
}

/// EnbPI evaluated online (the published algorithm's operating mode): fit
/// the ensemble once on `y[..n - n_eval]`, then walk the last `n_eval`
/// points in batches of `batch`, forming each interval from the *current*
/// sliding residual window, revealing the batch's labels, appending their
/// residuals and dropping the oldest (window length stays fixed). Within a
/// batch, unrevealed lag values are plug-in ensemble predictions; with
/// `batch = 1` every forecast uses fully realized lags.
///
/// # Errors
///
/// As [`enbpi`], plus `n_eval = 0` or `batch = 0` are
/// [`ForecastError::InvalidConformalParam`], and `n_eval` must leave
/// enough training rows.
pub fn enbpi_online(
    y: &[f64],
    opts: &EnbpiOptions,
    n_eval: usize,
    batch: usize,
) -> Result<ConformalOnline, ForecastError> {
    if n_eval == 0 {
        return Err(ForecastError::InvalidConformalParam {
            what: "n_eval",
            value: 0.0,
            requirement: "n_eval >= 1 evaluation points",
        });
    }
    if batch == 0 {
        return Err(ForecastError::InvalidConformalParam {
            what: "batch",
            value: 0.0,
            requirement: "batch >= 1 (the paper's s; labels are revealed once per batch)",
        });
    }
    check_enbpi(y, opts, n_eval)?;
    let n = y.len();
    let t_train = n - n_eval;
    let (ens, mut window, _excluded) = fit_enbpi_ensemble(y, t_train, opts)?;
    let win_len = window.len();

    let mut mean = Vec::with_capacity(n_eval);
    let mut lower = Vec::with_capacity(n_eval);
    let mut upper = Vec::with_capacity(n_eval);
    let mut err = Vec::with_capacity(n_eval);
    let mut origins = Vec::with_capacity(n_eval);

    let mut t = t_train;
    while t < n {
        let this_batch = batch.min(n - t);
        let (lo, up, _beta) = enbpi_offsets(&window, opts);
        // Predict the whole batch with plug-in recursion past `t`.
        let mut plugged: Vec<f64> = Vec::with_capacity(this_batch);
        for s in 0..this_batch {
            let mut lag_vec = Vec::with_capacity(opts.lags);
            for l in (1..=opts.lags).rev() {
                let idx = t + s - l;
                if idx < t {
                    lag_vec.push(y[idx]);
                } else {
                    lag_vec.push(plugged[idx - t]);
                }
            }
            let center = ens.loo_center(&lag_vec);
            plugged.push(center);
            let target = y[t + s];
            origins.push(t + s - 1); // forecaster "saw" y[0..=t+s-1] lags
            mean.push(center);
            lower.push(center + lo);
            upper.push(center + up);
            err.push(target < center + lo || target > center + up);
        }
        // Reveal the batch: slide the residual window.
        for (s, &p) in plugged.iter().enumerate() {
            window.push(y[t + s] - p);
        }
        let excess = window.len() - win_len;
        window.drain(0..excess);
        t += this_batch;
    }

    let missed = err.iter().filter(|&&m| m).count();
    let realized = 1.0 - missed as f64 / err.len() as f64;
    Ok(ConformalOnline {
        horizon: 1,
        alpha: opts.alpha,
        level: 1.0 - opts.alpha,
        origins,
        mean: vec![mean],
        lower: vec![lower],
        upper: vec![upper],
        err: vec![err],
        realized_coverage: vec![realized],
        alpha_trajectory: None,
    })
}

// --------------------------------------------------------------------------
// The AR(p) least-squares base learner (shared by EnbPI and base = "ar")
// --------------------------------------------------------------------------

/// OLS of `y[t]` on `[1, y[t-1], ..., y[t-lags]]` over the given target
/// rows `t`. Returns `[intercept, phi_1, ..., phi_lags]`.
#[allow(clippy::needless_range_loop)] // triangular index arithmetic is the point
fn fit_ar_ols_rows(y: &[f64], lags: usize, rows: &[usize]) -> Result<Vec<f64>, ForecastError> {
    let k = lags + 1;
    let mut xtx = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    let mut xrow = vec![0.0f64; k];
    for &t in rows {
        xrow[0] = 1.0;
        for l in 1..=lags {
            xrow[l] = y[t - l];
        }
        for a in 0..k {
            xty[a] += xrow[a] * y[t];
            for b in a..k {
                xtx[a][b] += xrow[a] * xrow[b];
            }
        }
    }
    for a in 0..k {
        for b in 0..a {
            xtx[a][b] = xtx[b][a];
        }
    }
    solve_spd(&mut xtx, &mut xty).ok_or(ForecastError::SingularArDesign {
        lags,
        n_rows: rows.len(),
    })
}

/// Solve the symmetric positive-definite system in place by Cholesky;
/// `None` when the matrix is (numerically) not positive definite.
#[allow(clippy::needless_range_loop)] // triangular index arithmetic is the point
fn solve_spd(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let k = b.len();
    // Scale-aware tolerance from the diagonal.
    let mut max_diag = 0.0f64;
    for (i, row) in a.iter().enumerate() {
        max_diag = max_diag.max(row[i].abs());
    }
    let tol = max_diag * 1e-12 + f64::MIN_POSITIVE;
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i][j];
            for l in 0..j {
                s -= a[i][l] * a[j][l];
            }
            if i == j {
                if s <= tol {
                    return None;
                }
                a[i][i] = s.sqrt();
            } else {
                a[i][j] = s / a[j][j];
            }
        }
    }
    // Forward then back substitution.
    for i in 0..k {
        let mut s = b[i];
        for l in 0..i {
            s -= a[i][l] * b[l];
        }
        b[i] = s / a[i][i];
    }
    for i in (0..k).rev() {
        let mut s = b[i];
        for l in i + 1..k {
            s -= a[l][i] * b[l];
        }
        b[i] = s / a[i][i];
    }
    Some(b.to_vec())
}

/// Least-squares AR(`lags`) point forecast: fit
/// `y_t = c + phi_1 y_{t-1} + ... + phi_lags y_{t-lags}` by OLS on the
/// full training slice and iterate `steps` ahead, feeding forecasts back
/// into the lag vector. This is the conformal module's `"ar"` base learner
/// — the same least-squares fit EnbPI's bootstrap ensemble uses — exposed
/// so split conformal and ACI can wrap an identical base for
/// apples-to-apples comparisons.
///
/// # Errors
///
/// [`ForecastError::InvalidSteps`], [`ForecastError::NonFinite`],
/// [`ForecastError::InvalidConformalParam`] (`lags = 0`),
/// [`ForecastError::SeriesTooShort`] (needs `2 * lags + 2` observations),
/// or [`ForecastError::SingularArDesign`] (collinear design, e.g. a
/// constant series).
pub fn ar_forecast(y: &[f64], lags: usize, steps: usize) -> Result<Vec<f64>, ForecastError> {
    check_steps(steps)?;
    if lags == 0 {
        return Err(ForecastError::InvalidConformalParam {
            what: "lags",
            value: 0.0,
            requirement:
                "lags >= 1 (an AR(0) base would be the historical mean; use base \"mean\")",
        });
    }
    // lags + 2 design rows for lags + 1 coefficients plus one residual df.
    check_series(y, 2 * lags + 2, "AR least-squares forecast")?;
    let n = y.len();
    let rows: Vec<usize> = (lags..n).collect();
    let coef = fit_ar_ols_rows(y, lags, &rows)?;
    let mut path: Vec<f64> = y[n - lags..].to_vec();
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let mut v = coef[0];
        for l in 1..=lags {
            v += coef[l] * path[path.len() - l];
        }
        out.push(v);
        path.push(v);
    }
    Ok(out)
}
