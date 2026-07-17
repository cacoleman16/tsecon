//! The Giacomini-White (2006) test of equal *conditional* predictive
//! ability.
//!
//! Diebold-Mariano and Clark-West ask whether one set of forecasts is more
//! accurate *on average*. Giacomini & White (2006) instead test equal
//! predictive ability of the two forecasting **methods** (model + estimation
//! window + everything) conditional on information available when the
//! forecast is made — which can reveal *when* each method wins, and, unlike
//! West-style tests, is valid for nested models as long as estimation is done
//! on a **fixed-width rolling window** (so estimation error does not vanish).
//!
//! With loss differential `dL_t = L1_t - L2_t` and a length-`q` vector of
//! test functions `h_t` measurable at the forecast origin, the moment tested
//! is `E[h_t dL_{t+tau}] = 0`. Let `z_t = h_t * dL_t` (the `q`-vector scaled
//! by the scalar differential), `zbar = mean_t z_t`, and `Shat` the Bartlett
//! HAC long-run covariance of the demeaned `z_t`. The statistic is the Wald
//! form
//!
//! ```text
//! GW = n * zbar' Shat^{-1} zbar  ~  chi^2(q)  under the null.
//! ```
//!
//! The **unconditional** case `h_t = 1` (`q = 1`) recovers a Diebold-Mariano
//! test with a Bartlett variance and a `chi^2(1)` (equivalently squared
//! standard-normal) reference — [`gw_test`]. The general `q`-dimensional form
//! is [`gw_test_conditional`]; passing a single constant test function makes
//! it agree with [`gw_test`] to machine precision (a pinned property test).
//!
//! Scope (Giacomini & White 2006): the fixed-width rolling-window requirement
//! is what keeps the limiting distribution non-degenerate for nested models.
//! Under a recursive (expanding) scheme estimation error vanishes and this
//! test's asymptotics do not apply — route nested recursive comparisons to
//! [`crate::cw`] instead. The variance here is a plain Bartlett HAC; the
//! `chi^2` critical values ignore estimation of `Shat`.

use crate::error::ForecastError;
use crate::hac::{bartlett_hac_matrix, bartlett_lrv, wald_statistic};
use crate::validate::check_finite;
use tsecon_stats::chi2_sf;

/// Result of a Giacomini-White (2006) equal-conditional-predictive-ability
/// test (unconditional or conditional).
#[derive(Debug, Clone, PartialEq)]
pub struct GwResult {
    /// Number of out-of-sample periods.
    pub n: usize,
    /// Degrees of freedom of the Wald statistic — the number of test
    /// functions `q` (`1` for the unconditional test).
    pub df: usize,
    /// The Bartlett long-run-variance lag truncation `L` used.
    pub lrv_lags: usize,
    /// The Wald statistic `n * zbar' Shat^{-1} zbar`.
    pub gw_stat: f64,
    /// Upper-tail `chi^2(df)` p-value `P(chi^2_df > gw_stat)`.
    pub p_value: f64,
}

/// The Giacomini-White (2006) **unconditional** equal-predictive-ability
/// test (test function `h_t = 1`).
///
/// * `loss1`, `loss2` — the realized out-of-sample loss series of the two
///   forecasting methods (e.g. squared errors), index-aligned and equally
///   long.
/// * `lrv_lags` — the Bartlett long-run-variance lag truncation `L`.
///
/// Computes `dL_t = loss1_t - loss2_t`, then
/// `GW = n * mean(dL)^2 / LRV_Bartlett(dL, L)`, referred to `chi^2(1)`.
/// This is the `q = 1` special case of [`gw_test_conditional`].
///
/// # Errors
///
/// [`ForecastError::LengthMismatch`], [`ForecastError::SeriesTooShort`]
/// (fewer than 2 periods), [`ForecastError::NonFinite`],
/// [`ForecastError::InvalidLrvLags`], and wrapped
/// [`ForecastError::Hac`] / [`ForecastError::Stats`].
pub fn gw_test(loss1: &[f64], loss2: &[f64], lrv_lags: usize) -> Result<GwResult, ForecastError> {
    const WHAT: &str = "Giacomini-White";
    let dl = loss_differential(loss1, loss2, WHAT)?;
    let n = dl.len();
    let dbar = dl.iter().sum::<f64>() / n as f64;
    let lrv = bartlett_lrv(&dl, lrv_lags, WHAT)?;
    if lrv <= 0.0 {
        return Err(ForecastError::SingularWaldCovariance { q: 1 });
    }
    // Compute the Wald form exactly as the q == 1 Cholesky path in
    // `gw_test_conditional` does (`n * (dbar / sqrt(lrv))^2`), so the
    // unconditional test and the conditional test with a single constant
    // instrument agree bit-for-bit.
    let w = dbar / lrv.sqrt();
    let gw_stat = n as f64 * (w * w);
    let p_value = chi2_sf(gw_stat, 1.0)?;
    Ok(GwResult {
        n,
        df: 1,
        lrv_lags,
        gw_stat,
        p_value,
    })
}

/// The Giacomini-White (2006) **conditional** predictive-ability test with
/// user-supplied test functions.
///
/// * `loss1`, `loss2` — the two out-of-sample loss series.
/// * `test_functions` — one length-`q` row `h_t` per period (an `n`-by-`q`
///   matrix, row-major), each measurable at the forecast origin. The
///   canonical choice is `h_t = (1, dL_{t-1}, ...)`; a single constant column
///   (`h_t = [1.0]`) recovers [`gw_test`]. All rows must have the same width
///   `q >= 1`.
/// * `lrv_lags` — the Bartlett long-run-variance lag truncation `L` for the
///   `q`-by-`q` covariance `Shat`.
///
/// Forms `z_t = h_t * dL_t`, `zbar = mean_t z_t`, the Bartlett HAC covariance
/// `Shat` of the demeaned `z_t`, and the Wald statistic
/// `GW = n * zbar' Shat^{-1} zbar ~ chi^2(q)`.
///
/// # Errors
///
/// [`ForecastError::EmptyTestFunctions`] if no test functions are given,
/// [`ForecastError::LengthMismatch`] if `test_functions` has a different
/// number of rows than the loss series or a row whose width differs from the
/// first, plus the [`gw_test`] error conditions and
/// [`ForecastError::SingularWaldCovariance`] when `Shat` is not positive
/// definite.
pub fn gw_test_conditional(
    loss1: &[f64],
    loss2: &[f64],
    test_functions: &[Vec<f64>],
    lrv_lags: usize,
) -> Result<GwResult, ForecastError> {
    const WHAT: &str = "Giacomini-White conditional";
    let dl = loss_differential(loss1, loss2, WHAT)?;
    let n = dl.len();
    if test_functions.is_empty() {
        return Err(ForecastError::EmptyTestFunctions);
    }
    if test_functions.len() != n {
        return Err(ForecastError::LengthMismatch {
            what: "Giacomini-White test functions",
            expected: n,
            actual: test_functions.len(),
        });
    }
    let q = test_functions[0].len();
    if q == 0 {
        return Err(ForecastError::EmptyTestFunctions);
    }
    for row in test_functions {
        if row.len() != q {
            return Err(ForecastError::LengthMismatch {
                what: "Giacomini-White test-function row width",
                expected: q,
                actual: row.len(),
            });
        }
        check_finite(row, "Giacomini-White test functions")?;
    }

    // z_t = h_t * dL_t (q-vector scaled by the scalar differential).
    let z: Vec<Vec<f64>> = test_functions
        .iter()
        .zip(dl.iter())
        .map(|(h, &d)| h.iter().map(|&hv| hv * d).collect())
        .collect();

    let mut zbar = vec![0.0; q];
    for row in &z {
        for (zb, &v) in zbar.iter_mut().zip(row.iter()) {
            *zb += v;
        }
    }
    for zb in &mut zbar {
        *zb /= n as f64;
    }

    let shat = bartlett_hac_matrix(&z, q, lrv_lags, WHAT)?;
    let gw_stat = wald_statistic(&shat, &zbar, n)?;
    let p_value = chi2_sf(gw_stat, q as f64)?;
    Ok(GwResult {
        n,
        df: q,
        lrv_lags,
        gw_stat,
        p_value,
    })
}

/// Validate and difference two loss series: `dL_t = loss1_t - loss2_t`.
fn loss_differential(
    loss1: &[f64],
    loss2: &[f64],
    what: &'static str,
) -> Result<Vec<f64>, ForecastError> {
    if loss1.len() != loss2.len() {
        return Err(ForecastError::LengthMismatch {
            what,
            expected: loss1.len(),
            actual: loss2.len(),
        });
    }
    let n = loss1.len();
    if n < 2 {
        return Err(ForecastError::SeriesTooShort { what, n, needed: 2 });
    }
    check_finite(loss1, what)?;
    check_finite(loss2, what)?;
    Ok(loss1
        .iter()
        .zip(loss2.iter())
        .map(|(&a, &b)| a - b)
        .collect())
}
