//! Realized-volatility measures on a vector of intraday log-returns, plus
//! two range-based estimators that consume OHLC bars.
//!
//! Fix a trading day partitioned into `n` intraday intervals and write
//! `r_i` for the `i`-th log-return. Under a continuous semimartingale with
//! spot volatility `sigma(s)` and jumps, the object of interest is the
//! quadratic variation `QV = IV + sum (jumps)^2`, where the integrated
//! variance `IV = int sigma^2(s) ds` is the continuous part. The estimators
//! below target these quantities:
//!
//! * [`realized_variance`] estimates the full quadratic variation `QV`.
//! * [`bipower_variation`] estimates the continuous part `IV` alone and is
//!   robust to (finitely many) jumps.
//! * [`jump_component`] is the (floored) difference `RV - BV`, an estimate
//!   of the jump contribution to `QV`.
//! * [`realized_quarticity`] / [`tripower_quarticity`] estimate the
//!   integrated quarticity `int sigma^4(s) ds`, the asymptotic-variance
//!   nuisance parameter needed to studentize jump tests (tripower being the
//!   jump-robust version).
//! * [`parkinson`] and [`garman_klass`] estimate the same daily variance
//!   from the extremes of OHLC bars rather than from a return grid.

use core::f64::consts::PI;

use crate::error::RealizedError;
use tsecon_stats::special::ln_gamma;

/// `mu_1^{-2} = pi/2`, the scaling that makes bipower variation consistent
/// for integrated variance, where `mu_1 = E|Z| = sqrt(2/pi)` for a
/// standard normal `Z` (Barndorff-Nielsen & Shephard 2004).
const MU1_INV_SQ: f64 = PI / 2.0;

fn check_finite(x: &[f64], what: &'static str) -> Result<(), RealizedError> {
    for (index, &value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(RealizedError::NonFinite { what, index, value });
        }
    }
    Ok(())
}

/// Realized variance `RV = sum_{i=1}^n r_i^2`.
///
/// The canonical nonparametric estimator of the quadratic variation of the
/// log-price over the sampling window; consistent for `QV = IV + sum J^2`
/// as the grid refines (Andersen, Bollerslev, Diebold & Labys 2001).
///
/// # Errors
///
/// [`RealizedError::TooFewObservations`] on an empty series and
/// [`RealizedError::NonFinite`] on NaN/inf input.
pub fn realized_variance(r: &[f64]) -> Result<f64, RealizedError> {
    if r.is_empty() {
        return Err(RealizedError::TooFewObservations {
            what: "realized variance",
            n: 0,
            needed: 1,
        });
    }
    check_finite(r, "realized variance")?;
    Ok(r.iter().map(|&x| x * x).sum())
}

/// Bipower variation `BV = (pi/2) sum_{i=2}^n |r_i| |r_{i-1}|`.
///
/// Barndorff-Nielsen & Shephard (2004): products of adjacent absolute
/// returns annihilate the contribution of isolated jumps in the limit, so
/// `BV` consistently estimates the continuous integrated variance `IV`
/// alone, whereas [`realized_variance`] captures the full `QV`.
///
/// # Errors
///
/// [`RealizedError::TooFewObservations`] with fewer than two returns and
/// [`RealizedError::NonFinite`] on NaN/inf input.
pub fn bipower_variation(r: &[f64]) -> Result<f64, RealizedError> {
    if r.len() < 2 {
        return Err(RealizedError::TooFewObservations {
            what: "bipower variation",
            n: r.len(),
            needed: 2,
        });
    }
    check_finite(r, "bipower variation")?;
    let s: f64 = r.windows(2).map(|w| w[0].abs() * w[1].abs()).sum();
    Ok(MU1_INV_SQ * s)
}

/// Realized quarticity `RQ = (n/3) sum_{i=1}^n r_i^4`.
///
/// Consistent for the integrated quarticity `int sigma^4(s) ds`
/// (Barndorff-Nielsen & Shephard 2002), the scale factor in the
/// asymptotic variance of realized variance. Not jump-robust — for a
/// jump-robust quarticity use [`tripower_quarticity`].
///
/// # Errors
///
/// [`RealizedError::TooFewObservations`] on an empty series and
/// [`RealizedError::NonFinite`] on NaN/inf input.
pub fn realized_quarticity(r: &[f64]) -> Result<f64, RealizedError> {
    if r.is_empty() {
        return Err(RealizedError::TooFewObservations {
            what: "realized quarticity",
            n: 0,
            needed: 1,
        });
    }
    check_finite(r, "realized quarticity")?;
    let n = r.len() as f64;
    let s: f64 = r.iter().map(|&x| x * x * x * x).sum();
    Ok(n / 3.0 * s)
}

/// `mu_{4/3}^{-3}`, the scaling for tripower quarticity, where
/// `mu_{4/3} = 2^{2/3} Gamma(7/6) / Gamma(1/2) = E|Z|^{4/3}` for a standard
/// normal `Z` (Barndorff-Nielsen & Shephard 2004). Computed from
/// `tsecon_stats::special::ln_gamma` rather than a hard-coded literal so
/// the constant is self-documenting.
fn mu_four_thirds_inv_cubed() -> f64 {
    let mu = 2.0_f64.powf(2.0 / 3.0) * (ln_gamma(7.0 / 6.0) - ln_gamma(0.5)).exp();
    mu.powi(-3)
}

/// Tripower quarticity
/// `TQ = n mu_{4/3}^{-3} sum_{i=3}^n |r_i|^{4/3} |r_{i-1}|^{4/3} |r_{i-2}|^{4/3}`.
///
/// The jump-robust estimator of the integrated quarticity
/// `int sigma^4(s) ds` (Barndorff-Nielsen & Shephard 2004); used to
/// studentize the ratio jump test in [`crate::bns_jump_ratio`], since the
/// non-robust [`realized_quarticity`] is itself inflated by jumps.
///
/// # Errors
///
/// [`RealizedError::TooFewObservations`] with fewer than three returns and
/// [`RealizedError::NonFinite`] on NaN/inf input.
pub fn tripower_quarticity(r: &[f64]) -> Result<f64, RealizedError> {
    if r.len() < 3 {
        return Err(RealizedError::TooFewObservations {
            what: "tripower quarticity",
            n: r.len(),
            needed: 3,
        });
    }
    check_finite(r, "tripower quarticity")?;
    let n = r.len() as f64;
    let p = 4.0 / 3.0;
    let s: f64 = r
        .windows(3)
        .map(|w| w[0].abs().powf(p) * w[1].abs().powf(p) * w[2].abs().powf(p))
        .sum();
    Ok(n * mu_four_thirds_inv_cubed() * s)
}

/// Jump component `J = max(RV - BV, 0)`.
///
/// The realized-variance minus bipower-variation difference estimates the
/// contribution of jumps to the quadratic variation (Barndorff-Nielsen &
/// Shephard 2004); it is floored at zero because sampling noise can make
/// the raw difference negative even under a purely continuous path.
///
/// # Errors
///
/// Propagates [`realized_variance`] and [`bipower_variation`] errors
/// (needs at least two returns).
pub fn jump_component(r: &[f64]) -> Result<f64, RealizedError> {
    let rv = realized_variance(r)?;
    let bv = bipower_variation(r)?;
    Ok((rv - bv).max(0.0))
}

/// Parkinson (1980) range variance
/// `P = (1/(4 ln 2)) sum_i (ln(H_i / L_i))^2` over OHLC bars.
///
/// Uses each bar's high-low range, which is far more efficient than a
/// close-to-close estimator for a driftless geometric Brownian motion. The
/// summed form here estimates the variance accumulated across the supplied
/// bars; divide by the number of bars for the per-bar average variance.
///
/// # Errors
///
/// [`RealizedError::TooFewObservations`] on empty input,
/// [`RealizedError::NonFinite`] on NaN/inf input, and
/// [`RealizedError::InvalidOhlc`] if any `high < low` or a price is
/// non-positive.
pub fn parkinson(high: &[f64], low: &[f64]) -> Result<f64, RealizedError> {
    let n = high.len();
    if n == 0 {
        return Err(RealizedError::TooFewObservations {
            what: "Parkinson range variance",
            n: 0,
            needed: 1,
        });
    }
    if low.len() != n {
        return Err(RealizedError::InvalidOhlc {
            what: "Parkinson range variance",
            index: 0,
            detail: "high and low series must have equal length",
        });
    }
    check_finite(high, "Parkinson high")?;
    check_finite(low, "Parkinson low")?;
    let scale = 1.0 / (4.0 * 2.0_f64.ln());
    let mut acc = 0.0;
    for (index, (&h, &l)) in high.iter().zip(low.iter()).enumerate() {
        check_ohlc_positive(h, l, index, "Parkinson range variance")?;
        if h < l {
            return Err(RealizedError::InvalidOhlc {
                what: "Parkinson range variance",
                index,
                detail: "high is below low",
            });
        }
        let ln_hl = (h / l).ln();
        acc += ln_hl * ln_hl;
    }
    Ok(scale * acc)
}

/// Garman-Klass (1980) range variance
/// `GK = sum_i [ 0.5 (ln(H_i/L_i))^2 - (2 ln 2 - 1)(ln(C_i/O_i))^2 ]`.
///
/// Combines the high-low range with the open-close move for an estimator
/// roughly eight times as efficient as close-to-close under driftless
/// geometric Brownian motion. As with [`parkinson`], the summed form
/// estimates the variance accumulated across the supplied bars.
///
/// # Errors
///
/// [`RealizedError::TooFewObservations`] on empty input,
/// [`RealizedError::NonFinite`] on NaN/inf input, and
/// [`RealizedError::InvalidOhlc`] on mismatched lengths, `high < low`, or a
/// non-positive price.
pub fn garman_klass(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<f64, RealizedError> {
    let n = open.len();
    if n == 0 {
        return Err(RealizedError::TooFewObservations {
            what: "Garman-Klass range variance",
            n: 0,
            needed: 1,
        });
    }
    if high.len() != n || low.len() != n || close.len() != n {
        return Err(RealizedError::InvalidOhlc {
            what: "Garman-Klass range variance",
            index: 0,
            detail: "open, high, low, close series must have equal length",
        });
    }
    check_finite(open, "Garman-Klass open")?;
    check_finite(high, "Garman-Klass high")?;
    check_finite(low, "Garman-Klass low")?;
    check_finite(close, "Garman-Klass close")?;
    let c2 = 2.0 * 2.0_f64.ln() - 1.0;
    let mut acc = 0.0;
    for index in 0..n {
        let (o, h, l, c) = (open[index], high[index], low[index], close[index]);
        check_ohlc_positive(h, l, index, "Garman-Klass range variance")?;
        if o <= 0.0 || c <= 0.0 {
            return Err(RealizedError::InvalidOhlc {
                what: "Garman-Klass range variance",
                index,
                detail: "open and close must be strictly positive",
            });
        }
        if h < l {
            return Err(RealizedError::InvalidOhlc {
                what: "Garman-Klass range variance",
                index,
                detail: "high is below low",
            });
        }
        let ln_hl = (h / l).ln();
        let ln_co = (c / o).ln();
        acc += 0.5 * ln_hl * ln_hl - c2 * ln_co * ln_co;
    }
    Ok(acc)
}

/// Reject a bar whose high or low price is non-positive (both feed a
/// logarithm downstream).
fn check_ohlc_positive(
    high: f64,
    low: f64,
    index: usize,
    what: &'static str,
) -> Result<(), RealizedError> {
    if high <= 0.0 || low <= 0.0 {
        return Err(RealizedError::InvalidOhlc {
            what,
            index,
            detail: "high and low must be strictly positive",
        });
    }
    Ok(())
}
