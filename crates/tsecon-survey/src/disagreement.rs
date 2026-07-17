//! Forecast disagreement: the cross-sectional dispersion of individual
//! forecasts, period by period.
//!
//! Disagreement across forecasters is the standard empirical proxy for
//! subjective uncertainty in the survey-expectations literature (e.g.
//! Mankiw-Reis-Wolfers 2004, Coibion-Gorodnichenko 2012). Given a panel of
//! individual forecasts — a cross-section of forecasters for each time period —
//! this module reports, for every period, the dispersion of the cross-section:
//!
//! * the standard deviation (numpy `np.std(ddof=...)`; population `ddof = 0`
//!   by default, sample `ddof = 1` optionally), and
//! * the inter-quartile range `IQR = P75 - P25` together with the three
//!   quartiles, where each percentile uses numpy's default *linear*
//!   interpolation (`np.percentile(..., method="linear")`).
//!
//! The panel is a slice of per-period cross-sections and may be **ragged**
//! (different numbers of forecasters in different periods). Nothing here is
//! estimated with error — these are exact descriptive statistics reproduced
//! bit-for-bit against numpy.

use crate::common::check_finite;
use crate::error::SurveyError;

/// Per-period cross-sectional disagreement of a forecaster panel. Every vector
/// is indexed by period (same order as the input panel).
#[derive(Debug, Clone, PartialEq)]
pub struct Disagreement {
    /// Cross-sectional standard deviation, one per period (numpy `np.std`
    /// with the requested `ddof`).
    pub std: Vec<f64>,
    /// 25th percentile (first quartile), one per period.
    pub p25: Vec<f64>,
    /// 50th percentile (median), one per period.
    pub p50: Vec<f64>,
    /// 75th percentile (third quartile), one per period.
    pub p75: Vec<f64>,
    /// Inter-quartile range `p75 - p25`, one per period.
    pub iqr: Vec<f64>,
    /// Cross-section size (number of forecasters) in each period.
    pub counts: Vec<usize>,
    /// The degrees-of-freedom correction used for the standard deviation.
    pub ddof: usize,
}

/// Compute per-period forecast disagreement for a (possibly ragged) forecaster
/// panel.
///
/// `panel[t]` is the cross-section of individual forecasts in period `t`.
/// `ddof` is the numpy delta-degrees-of-freedom for the standard deviation
/// (`0` = population, the numpy default; `1` = sample). The variance divisor
/// is `count - ddof`.
///
/// # Errors
///
/// [`SurveyError::EmptyInput`] if the panel is empty or any period has no
/// forecasters; [`SurveyError::NonFinite`] on NaN/inf; and
/// [`SurveyError::InvalidArgument`] if `ddof >= count` for some period (a
/// non-positive variance divisor).
pub fn disagreement(panel: &[Vec<f64>], ddof: usize) -> Result<Disagreement, SurveyError> {
    if panel.is_empty() {
        return Err(SurveyError::EmptyInput {
            what: "forecaster panel",
        });
    }
    let periods = panel.len();
    let mut std = Vec::with_capacity(periods);
    let mut p25 = Vec::with_capacity(periods);
    let mut p50 = Vec::with_capacity(periods);
    let mut p75 = Vec::with_capacity(periods);
    let mut iqr = Vec::with_capacity(periods);
    let mut counts = Vec::with_capacity(periods);

    for cross_section in panel {
        if cross_section.is_empty() {
            return Err(SurveyError::EmptyInput {
                what: "a period's forecaster cross-section",
            });
        }
        check_finite(cross_section, "forecaster panel")?;
        let m = cross_section.len();
        if ddof >= m {
            return Err(SurveyError::InvalidArgument {
                what: "ddof must be smaller than every period's cross-section \
                       size (the standard-deviation divisor count - ddof would \
                       otherwise be non-positive)",
            });
        }

        std.push(cross_section_std(cross_section, ddof));
        let q25 = percentile_linear(cross_section, 25.0);
        let q50 = percentile_linear(cross_section, 50.0);
        let q75 = percentile_linear(cross_section, 75.0);
        p25.push(q25);
        p50.push(q50);
        p75.push(q75);
        iqr.push(q75 - q25);
        counts.push(m);
    }

    Ok(Disagreement {
        std,
        p25,
        p50,
        p75,
        iqr,
        counts,
        ddof,
    })
}

/// numpy `np.std(x, ddof=ddof)`: `sqrt( sum_i (x_i - xbar)^2 / (m - ddof) )`.
/// The caller guarantees `m > ddof`.
fn cross_section_std(x: &[f64], ddof: usize) -> f64 {
    let m = x.len() as f64;
    let mean = x.iter().sum::<f64>() / m;
    let ss: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    (ss / (m - ddof as f64)).sqrt()
}

/// numpy `np.percentile(x, q, method="linear")` — the default linear
/// interpolation of the empirical CDF.
///
/// For a length-`m` sample sorted ascending, the (0-based) virtual index of
/// quantile `q in [0, 100]` is `h = (m - 1) * q / 100`; the result linearly
/// interpolates between the order statistics bracketing `h`:
/// `x_sorted[floor(h)] + (h - floor(h)) * (x_sorted[ceil(h)] - x_sorted[floor(h)])`.
/// The caller guarantees `x` is non-empty and finite.
fn percentile_linear(x: &[f64], q: f64) -> f64 {
    let m = x.len();
    let mut sorted = x.to_vec();
    // Total order on finite f64 (no NaNs by the finiteness check upstream).
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    if m == 1 {
        return sorted[0];
    }
    let h = (m as f64 - 1.0) * q / 100.0;
    let lo = h.floor();
    let lo_idx = lo as usize;
    let frac = h - lo;
    if lo_idx + 1 >= m {
        // h lands exactly on the last order statistic (q = 100).
        return sorted[m - 1];
    }
    sorted[lo_idx] + frac * (sorted[lo_idx + 1] - sorted[lo_idx])
}
