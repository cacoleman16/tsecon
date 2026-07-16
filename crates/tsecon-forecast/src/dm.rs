//! The Diebold-Mariano test of equal predictive accuracy with the
//! Harvey-Leybourne-Newbold small-sample correction.
//!
//! Given two h-step forecast-error vectors `e1`, `e2` and a loss function
//! `g`, the test works on the loss differential `d_t = g(e1_t) - g(e2_t)`
//! and asks whether `E[d_t] = 0` (Diebold & Mariano 1995). Under the null,
//!
//! `DM = dbar / sqrt(Vhat(dbar))`,
//!
//! where `Vhat(dbar)` is a long-run variance of `dbar` that accounts for
//! the MA(h-1) serial correlation an optimal h-step forecast error
//! inherits. Following the original recipe, this crate uses the
//! uniform-weight (rectangular) truncated autocovariance sum
//!
//! `Vhat(dbar) = [gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k] / n`,
//! `gamma_k = (1/(n-k)) * sum_{t=k}^{n-1} (d_t - dbar)(d_{t-k} - dbar)`,
//!
//! i.e. each autocovariance is averaged over its `n - k` available
//! products (the convention pinned by the golden fixture). The rectangular
//! window is not positive semi-definite, so the estimate can go negative
//! for large `h`; that surfaces as
//! [`ForecastError::NonPositiveLongRunVariance`] rather than a NaN
//! statistic.
//!
//! The Harvey, Leybourne & Newbold (1997) correction rescales the
//! statistic for small samples,
//!
//! `HLN = DM * sqrt( (n + 1 - 2h + h(h-1)/n) / n )`,
//!
//! and compares it against a Student-t distribution with `n - 1` degrees
//! of freedom instead of the standard normal. The reported
//! [`DmResult::p_value`] is the two-sided HLN t(n-1) p-value — the
//! recommended default, not an option.
//!
//! Scope note (Diebold 2015): the DM test compares *forecasts*, not
//! models. For nested models evaluated under recursive schemes the DM
//! statistic is degenerate; use a Clark-West style adjustment instead
//! (TODO(phase0): Clark-West and Giacomini-White live in a later slice of
//! the evaluation module).

use crate::error::ForecastError;
use crate::validate::check_finite;
use tsecon_stats::{ContinuousDist, StudentT};

/// Built-in loss functions for the Diebold-Mariano loss differential.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmLoss {
    /// Squared-error loss `g(e) = e^2` (tests equal MSE).
    Squared,
    /// Absolute-error loss `g(e) = |e|` (tests equal MAE).
    Absolute,
}

/// Result of a Diebold-Mariano test.
#[derive(Debug, Clone, PartialEq)]
pub struct DmResult {
    /// Number of loss differentials (evaluation periods).
    pub n: usize,
    /// Forecast horizon `h`; the long-run variance truncates the
    /// autocovariance sum at lag `h - 1`.
    pub h: usize,
    /// Mean loss differential `dbar = mean(g(e1_t) - g(e2_t))`. Positive
    /// values mean forecast 1 incurs the larger loss.
    pub mean_loss_diff: f64,
    /// The estimated long-run variance of `dbar` (uniform weights,
    /// truncated at lag `h - 1`).
    pub var_mean_loss_diff: f64,
    /// The uncorrected Diebold-Mariano statistic (asymptotically N(0,1)
    /// under the null).
    pub dm_stat: f64,
    /// The Harvey-Leybourne-Newbold small-sample corrected statistic.
    pub hln_stat: f64,
    /// Two-sided p-value of [`DmResult::hln_stat`] under Student-t with
    /// `n - 1` degrees of freedom (the HLN recommendation).
    pub p_value: f64,
}

/// Diebold-Mariano test with a built-in loss.
///
/// `e1` and `e2` are the h-step forecast-*error* vectors (actual minus
/// forecast) of the two competing forecasts over the same evaluation
/// window; `h` is the forecast horizon used for the variance truncation
/// (`h = 1` for one-step forecasts). See the [module docs](self) for the
/// formulas and references (Diebold & Mariano 1995; Harvey, Leybourne &
/// Newbold 1997).
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`], [`ForecastError::SeriesTooShort`],
/// [`ForecastError::NonFinite`], [`ForecastError::InvalidHorizon`],
/// [`ForecastError::DegenerateLossDifferential`] (identical losses, e.g.
/// a forecast compared with itself), or
/// [`ForecastError::NonPositiveLongRunVariance`].
pub fn dm_test(e1: &[f64], e2: &[f64], h: usize, loss: DmLoss) -> Result<DmResult, ForecastError> {
    match loss {
        DmLoss::Squared => dm_test_with_loss(e1, e2, h, |e| e * e),
        DmLoss::Absolute => dm_test_with_loss(e1, e2, h, f64::abs),
    }
}

/// Diebold-Mariano test with a custom loss function `g(e)`.
///
/// The loss differential is `d_t = g(e1_t) - g(e2_t)`; everything else is
/// as in [`dm_test`]. The closure must map finite errors to finite losses
/// — a non-finite loss is reported as [`ForecastError::NonFinite`] on the
/// loss differential.
///
/// # Errors
///
/// Same as [`dm_test`].
pub fn dm_test_with_loss<F>(
    e1: &[f64],
    e2: &[f64],
    h: usize,
    loss: F,
) -> Result<DmResult, ForecastError>
where
    F: Fn(f64) -> f64,
{
    const WHAT: &str = "Diebold-Mariano";
    if e1.len() != e2.len() {
        return Err(ForecastError::LengthMismatch {
            what: WHAT,
            expected: e1.len(),
            actual: e2.len(),
        });
    }
    let n = e1.len();
    if n < 2 {
        return Err(ForecastError::SeriesTooShort {
            what: WHAT,
            n,
            needed: 2,
        });
    }
    check_finite(e1, WHAT)?;
    check_finite(e2, WHAT)?;
    if h == 0 || h >= n {
        return Err(ForecastError::InvalidHorizon { h, n });
    }

    let d: Vec<f64> = e1
        .iter()
        .zip(e2.iter())
        .map(|(&a, &b)| loss(a) - loss(b))
        .collect();
    check_finite(&d, "Diebold-Mariano loss differential")?;

    let nf = n as f64;
    let dbar = d.iter().sum::<f64>() / nf;

    // gamma_k averaged over the n - k available products (the fixture's
    // np.mean convention), k = 0 .. h-1.
    let mut lrv_sum = 0.0;
    let mut gamma0 = 0.0;
    for k in 0..h {
        let mut acc = 0.0;
        for t in k..n {
            acc += (d[t] - dbar) * (d[t - k] - dbar);
        }
        let gamma_k = acc / (n - k) as f64;
        if k == 0 {
            gamma0 = gamma_k;
            lrv_sum += gamma_k;
        } else {
            lrv_sum += 2.0 * gamma_k;
        }
    }
    if gamma0 <= 0.0 {
        return Err(ForecastError::DegenerateLossDifferential);
    }
    let var_dbar = lrv_sum / nf;
    if var_dbar <= 0.0 {
        return Err(ForecastError::NonPositiveLongRunVariance { value: var_dbar });
    }

    let dm_stat = dbar / var_dbar.sqrt();
    let hf = h as f64;
    let hln_factor = ((nf + 1.0 - 2.0 * hf + hf * (hf - 1.0) / nf) / nf).sqrt();
    let hln_stat = dm_stat * hln_factor;
    let t = StudentT::new(nf - 1.0)?;
    let p_value = 2.0 * t.sf(hln_stat.abs());

    Ok(DmResult {
        n,
        h,
        mean_loss_diff: dbar,
        var_mean_loss_diff: var_dbar,
        dm_stat,
        hln_stat,
        p_value,
    })
}
