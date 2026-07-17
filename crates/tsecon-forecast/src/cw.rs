//! The Clark-West (2007) test of equal predictive accuracy for **nested**
//! models.
//!
//! When a large (unrestricted) model nests a small (restricted) one, the
//! Diebold-Mariano statistic is degenerate: under the null that the extra
//! regressors are useless, the large model's out-of-sample forecasts are the
//! small model's plus pure estimation noise, so its expected squared error is
//! *larger*, biasing the naive MSPE comparison against the (true) large
//! model. Clark & West (2006, 2007) correct the loss differential by exactly
//! the noise term. With small-model errors `e1_t = y_t - yhat1_t` and
//! large-model errors `e2_t = y_t - yhat2_t`, the adjusted differential is
//!
//! ```text
//! f_t = e1_t^2 - e2_t^2 + (yhat1_t - yhat2_t)^2
//! ```
//!
//! (the `(yhat1 - yhat2)^2` term adds back the estimation-noise penalty the
//! large model incurred). The statistic regresses `f_t` on a constant and
//! reports the HAC t-ratio
//!
//! ```text
//! CW = mean(f) / sqrt( LRV_Bartlett(f, L) / n ),
//! ```
//!
//! where `LRV_Bartlett(f, L)` is the Newey-West (1987) Bartlett long-run
//! variance of the demeaned `f_t` at lag truncation `L` (see [`crate::hac`]).
//! The test is **one-sided by construction** — a large model that genuinely
//! improves on the small one pushes `mean(f)` positive — so this function
//! reports the one-sided upper-tail standard-normal p-value
//! `P(Z > CW)`; reject "the small model is (weakly) at least as good" for
//! large positive `CW`. Clark & West note the standard-normal critical values
//! are mildly conservative (the test is slightly undersized).
//!
//! Scope (Clark & West 2007; Diebold 2015): use this instead of
//! Diebold-Mariano precisely when the models are nested and estimated
//! recursively/rolling. For the *conditional* predictive-ability question and
//! fixed-width rolling windows, see [`crate::gw`].

use crate::error::ForecastError;
use crate::hac::bartlett_lrv;
use crate::validate::check_finite;
use tsecon_stats::{ContinuousDist, StdNormal};

/// Result of a Clark-West (2007) nested-model predictive-ability test.
#[derive(Debug, Clone, PartialEq)]
pub struct CwResult {
    /// Number of out-of-sample periods (length of the differential series).
    pub n: usize,
    /// The Bartlett long-run-variance lag truncation `L` used.
    pub lrv_lags: usize,
    /// Mean adjusted differential `mean(f_t)`; positive values favour the
    /// larger model.
    pub mean_adj_diff: f64,
    /// The Bartlett long-run variance of the demeaned `f_t` at lag `L`.
    pub long_run_var: f64,
    /// The Clark-West statistic `mean(f) / sqrt(LRV / n)` (asymptotically
    /// standard normal under the null).
    pub cw_stat: f64,
    /// One-sided upper-tail standard-normal p-value `P(Z > cw_stat)`.
    pub p_value: f64,
}

/// The Clark-West (2007) adjusted-MSPE test for two nested forecasts.
///
/// * `e_small` — forecast errors `y_t - yhat1_t` of the **restricted**
///   (small) model.
/// * `e_large` — forecast errors `y_t - yhat2_t` of the **unrestricted**
///   (large) model that nests the small one.
/// * `yhat_small`, `yhat_large` — the corresponding point forecasts, used
///   for the estimation-noise correction `(yhat1 - yhat2)^2`.
/// * `lrv_lags` — the Bartlett long-run-variance lag truncation `L`
///   (`0` reduces the variance to the plain sample variance of `f_t`; for
///   `h`-step forecasts `L >= h - 1` is the usual floor).
///
/// All four series must be index-aligned and the same length. See the
/// [module docs](self) for the formula and references (Clark & West 2007).
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`] if the four series differ in length,
/// [`ForecastError::SeriesTooShort`] for fewer than 2 periods,
/// [`ForecastError::NonFinite`] on NaN/infinite inputs,
/// [`ForecastError::InvalidLrvLags`] if `lrv_lags >= n`, and a wrapped
/// [`ForecastError::Hac`] / [`ForecastError::Stats`] from the variance and
/// normal-tail computations.
pub fn cw_test(
    e_small: &[f64],
    e_large: &[f64],
    yhat_small: &[f64],
    yhat_large: &[f64],
    lrv_lags: usize,
) -> Result<CwResult, ForecastError> {
    const WHAT: &str = "Clark-West";
    let n = e_small.len();
    for other in [e_large.len(), yhat_small.len(), yhat_large.len()] {
        if other != n {
            return Err(ForecastError::LengthMismatch {
                what: WHAT,
                expected: n,
                actual: other,
            });
        }
    }
    if n < 2 {
        return Err(ForecastError::SeriesTooShort {
            what: WHAT,
            n,
            needed: 2,
        });
    }
    check_finite(e_small, "Clark-West e_small")?;
    check_finite(e_large, "Clark-West e_large")?;
    check_finite(yhat_small, "Clark-West yhat_small")?;
    check_finite(yhat_large, "Clark-West yhat_large")?;

    // Adjusted differential f_t = e1^2 - e2^2 + (yhat1 - yhat2)^2.
    let f: Vec<f64> = (0..n)
        .map(|t| {
            let df = yhat_small[t] - yhat_large[t];
            e_small[t] * e_small[t] - e_large[t] * e_large[t] + df * df
        })
        .collect();

    let mean_adj_diff = f.iter().sum::<f64>() / n as f64;
    let long_run_var = bartlett_lrv(&f, lrv_lags, WHAT)?;
    // The Bartlett long-run variance is non-negative; it is zero only when the
    // adjusted differential is constant (e.g. the two forecasts are identical,
    // so f_t == 0), which leaves the standardized statistic 0/0.
    if long_run_var <= 0.0 {
        return Err(ForecastError::SingularWaldCovariance { q: 1 });
    }
    let cw_stat = mean_adj_diff / (long_run_var / n as f64).sqrt();
    let p_value = StdNormal.sf(cw_stat);

    Ok(CwResult {
        n,
        lrv_lags,
        mean_adj_diff,
        long_run_var,
        cw_stat,
        p_value,
    })
}
