//! Band-pass filters: Baxter-King (symmetric, truncated) and
//! Christiano-Fitzgerald (asymmetric, random-walk optimal).
//!
//! Both isolate fluctuations with periods between `low` and `high`
//! observations (business-cycle frequencies: 6-32 quarters). The ideal
//! band-pass weights are
//!
//! ```text
//! b_0 = (omega_2 - omega_1) / pi,
//! b_j = (sin(omega_2 j) - sin(omega_1 j)) / (pi j),   j >= 1,
//! omega_1 = 2 pi / high,   omega_2 = 2 pi / low.
//! ```
//!
//! Baxter-King truncates them symmetrically at `K` lags;
//! Christiano-Fitzgerald keeps the full-sample asymmetric projection
//! that is optimal when the series is a random walk.

use core::f64::consts::PI;

use crate::decomposition::{Alignment, Decomposition};
use crate::error::{check_finite, FiltersError};

/// Validate a band-pass frequency band: `2 <= low < high` (periods
/// shorter than 2 observations are above the Nyquist frequency).
fn check_band(low: f64, high: f64) -> Result<(), FiltersError> {
    if !low.is_finite() || low < 2.0 {
        return Err(FiltersError::InvalidParameter {
            name: "low",
            value: low,
            requirement: "a finite period >= 2 observations (Nyquist)",
        });
    }
    if !high.is_finite() || high <= low {
        return Err(FiltersError::InvalidParameter {
            name: "high",
            value: high,
            requirement: "a finite period > low",
        });
    }
    Ok(())
}

/// Ideal band-pass weight `b_j` for the band `(2pi/high, 2pi/low)`.
fn ideal_weight(j: usize, omega_1: f64, omega_2: f64) -> f64 {
    if j == 0 {
        (omega_2 - omega_1) / PI
    } else {
        let jf = j as f64;
        ((omega_2 * jf).sin() - (omega_1 * jf).sin()) / (PI * jf)
    }
}

/// Baxter-King symmetric band-pass filter.
///
/// Truncates the ideal band-pass weights at `+/- K` lags and demeans
/// them so they sum to zero (which removes unit roots and up to
/// quadratic deterministic trends):
///
/// ```text
/// cycle_t = sum_{j=-K}^{K} (b_|j| - bbar) y_{t+j},
/// bbar    = sum_{j=-K}^{K} b_|j| / (2K + 1),
/// ```
///
/// for `t = K, ..., n-K-1`. The moving average is two-sided, so `K`
/// observations are lost at **each** end: the returned cycle has
/// `n - 2K` elements and `alignment.lost_start = alignment.lost_end = K`
/// (matching statsmodels `bkfilter`, which trims the convolution to its
/// valid range). No trend component is defined — `trend` is `None`.
///
/// Defaults for quarterly data are `low = 6`, `high = 32`, `K = 12`
/// (Baxter & King 1999).
///
/// Reference: Baxter & King (1999), "Measuring Business Cycles:
/// Approximate Band-Pass Filters for Economic Time Series", *Review of
/// Economics and Statistics* 81(4).
pub fn bk_filter(y: &[f64], low: f64, high: f64, k: usize) -> Result<Decomposition, FiltersError> {
    check_band(low, high)?;
    if k == 0 {
        return Err(FiltersError::InvalidParameter {
            name: "k",
            value: 0.0,
            requirement: "a truncation lag >= 1",
        });
    }
    let n = y.len();
    let needed = 2 * k + 1;
    if n < needed {
        return Err(FiltersError::SeriesTooShort {
            filter: "bk_filter",
            needed,
            got: n,
        });
    }
    check_finite(y)?;

    let omega_1 = 2.0 * PI / high;
    let omega_2 = 2.0 * PI / low;
    // Symmetric weights b[k + j] = b[k - j], demeaned to sum to zero.
    let mut b = vec![0.0_f64; 2 * k + 1];
    b[k] = ideal_weight(0, omega_1, omega_2);
    for j in 1..=k {
        let w = ideal_weight(j, omega_1, omega_2);
        b[k + j] = w;
        b[k - j] = w;
    }
    let mean = b.iter().sum::<f64>() / b.len() as f64;
    for w in &mut b {
        *w -= mean;
    }

    let mut cycle = Vec::with_capacity(n - 2 * k);
    for t in k..(n - k) {
        let mut s = 0.0;
        for (j, w) in b.iter().enumerate() {
            s += w * y[t + j - k];
        }
        cycle.push(s);
    }
    Ok(Decomposition {
        trend: None,
        cycle,
        alignment: Alignment {
            lost_start: k,
            lost_end: k,
            input_len: n,
        },
    })
}

/// Christiano-Fitzgerald asymmetric (random-walk optimal) band-pass
/// filter.
///
/// Uses all `n` observations at every date with time-varying weights, so
/// no observations are lost (alignment is full sample), at the price of
/// end-point revisions as data accrue. For each `t` (0-indexed),
///
/// ```text
/// cycle_t = b_0 x_t + sum_{j=1}^{n-t-2} b_j x_{t+j} + B_t x_{n-1}
///                   + sum_{j=1}^{t-1}   b_j x_{t-j} + A_t x_0,
/// B_t = -b_0/2 - sum_{j=1}^{n-t-2} b_j,
/// A_t = -b_0/2 - sum_{j=1}^{t-1}   b_j,
/// ```
///
/// where the endpoint weights `A_t`, `B_t` make each row of the filter
/// sum to zero — the optimal-approximation solution when the series is a
/// random walk (Christiano & Fitzgerald 2003, eq. (18); the "RW" filter
/// of their taxonomy — the stationary/I(0) variants of their section 4
/// are deferred, see TODO in the source).
///
/// With `drift = true` a linear drift line through the endpoints is
/// removed first, `x_t = y_t - t (y_{n-1} - y_0)/(n - 1)`, matching
/// statsmodels `cffilter(drift=True)`. **Convention warning** (inherited
/// from statsmodels, pinned by the golden fixture): the returned trend
/// is `x - cycle`, i.e. the trend of the *drift-adjusted* series —
/// `trend + cycle` reconstructs the drift-adjusted series, not the raw
/// input. With `drift = false`, `trend + cycle == y` over the full
/// sample.
///
/// Cost is `O(n^2)` (time-varying weights); fine for macro sample sizes.
///
/// Reference: Christiano & Fitzgerald (2003), "The Band Pass Filter",
/// *International Economic Review* 44(2).
// TODO(phase0): stationary/I(0) and symmetric variants of the CF filter
// (Christiano-Fitzgerald 2003, section 4) — only the random-walk-optimal
// asymmetric filter is implemented, matching statsmodels `cffilter`.
pub fn cf_filter(
    y: &[f64],
    low: f64,
    high: f64,
    drift: bool,
) -> Result<Decomposition, FiltersError> {
    check_band(low, high)?;
    let n = y.len();
    if n < 3 {
        return Err(FiltersError::SeriesTooShort {
            filter: "cf_filter",
            needed: 3,
            got: n,
        });
    }
    check_finite(y)?;

    // Drift adjustment: remove the straight line through the endpoints.
    let x: Vec<f64> = if drift {
        let slope = (y[n - 1] - y[0]) / (n as f64 - 1.0);
        y.iter()
            .enumerate()
            .map(|(t, v)| v - t as f64 * slope)
            .collect()
    } else {
        y.to_vec()
    };

    let omega_1 = 2.0 * PI / high;
    let omega_2 = 2.0 * PI / low;
    let b: Vec<f64> = (0..=n).map(|j| ideal_weight(j, omega_1, omega_2)).collect();
    // Prefix sums: bsum[m] = b_1 + ... + b_m.
    let mut bsum = vec![0.0_f64; n + 1];
    for m in 1..=n {
        bsum[m] = bsum[m - 1] + b[m];
    }

    let mut cycle = Vec::with_capacity(n);
    for t in 0..n {
        let m_fwd = n.saturating_sub(t + 2); // forward terms j = 1..=m_fwd
        let m_bwd = t.saturating_sub(1); // backward terms j = 1..=m_bwd
        let b_end = -0.5 * b[0] - bsum[m_fwd];
        let a_end = -0.5 * b[0] - bsum[m_bwd];
        let mut s = b[0] * x[t] + b_end * x[n - 1] + a_end * x[0];
        for j in 1..=m_fwd {
            s += b[j] * x[t + j];
        }
        for j in 1..=m_bwd {
            s += b[j] * x[t - j];
        }
        cycle.push(s);
    }
    let trend: Vec<f64> = x.iter().zip(&cycle).map(|(xi, ci)| xi - ci).collect();
    Ok(Decomposition {
        trend: Some(trend),
        cycle,
        alignment: Alignment::full(n),
    })
}
