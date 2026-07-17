//! Shared frequency-domain plumbing for the semiparametric memory estimators.
//!
//! Both the GPH log-periodogram regression and the Robinson local-Whittle
//! estimator operate on the raw periodogram evaluated at the lowest `m`
//! Fourier frequencies `lambda_j = 2*pi*j/n`, `j = 1, ..., m`. This module
//! owns the one place that calls [`tsecon_spectral::periodogram`] (the
//! library's single periodogram owner) and slices out those ordinates, plus
//! the bandwidth validation both estimators share.

use tsecon_spectral::{periodogram, Detrend, Scaling, Window};

use crate::error::LongMemoryError;

/// The textbook automatic bandwidth `m = floor(sqrt(n))` for the
/// semiparametric memory estimators.
///
/// This is the common rule of thumb (`m ~ n^{1/2}`) for both GPH and local
/// Whittle. It is only a default; pass an explicit `m` to
/// [`crate::gph`] / [`crate::local_whittle`] to trade bias against variance.
///
/// # Example
/// ```
/// use tsecon_longmemory::default_bandwidth;
/// assert_eq!(default_bandwidth(400), 20);
/// ```
pub fn default_bandwidth(n: usize) -> usize {
    (n as f64).sqrt().floor() as usize
}

/// Validate the bandwidth `m` against the series length `n`, requiring
/// `min <= m <= (n - 1) / 2`.
///
/// The lower bound keeps the regression/objective identified; the upper bound
/// keeps every ordinate a strictly interior Fourier frequency (below Nyquist),
/// so the one-sided periodogram carries the same folding factor for every bin
/// used — which is what makes both estimators invariant to the periodogram's
/// overall normalization.
pub fn check_bandwidth(m: usize, n: usize, min: usize) -> Result<usize, LongMemoryError> {
    let max = if n >= 1 { (n - 1) / 2 } else { 0 };
    if m < min || m > max {
        return Err(LongMemoryError::InvalidBandwidth { m, n, min, max });
    }
    Ok(max)
}

/// The lowest `m` Fourier frequencies and their raw-periodogram ordinates.
///
/// Returns `(lambdas, i_j)` where `lambdas[j-1] = 2*pi*j/n` and `i_j[j-1]` is
/// the periodogram at that frequency, for `j = 1, ..., m`. The periodogram is
/// the single-taper boxcar density estimate from [`tsecon_spectral`]; its
/// overall scaling is immaterial to both memory estimators (GPH: absorbed by
/// the regression intercept; local Whittle: an additive constant in the
/// concentrated objective, so the minimizer is unchanged).
///
/// # Errors
///
/// Propagates [`LongMemoryError::Spectral`] from the periodogram (including on
/// non-finite input) and returns [`LongMemoryError::NonPositivePeriodogram`] if
/// an ordinate vanishes.
pub fn low_frequency_periodogram(
    x: &[f64],
    m: usize,
) -> Result<(Vec<f64>, Vec<f64>), LongMemoryError> {
    let n = x.len();
    let spectrum = periodogram(x, 1.0, Window::Boxcar, Scaling::Density, Detrend::None)?;
    // freqs[j] = j / n (cycles per unit time); the angular Fourier frequency is
    // lambda_j = 2*pi*freqs[j] = 2*pi*j/n.
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut lambdas = Vec::with_capacity(m);
    let mut i_j = Vec::with_capacity(m);
    for j in 1..=m {
        let ordinate = spectrum.psd[j];
        if ordinate <= 0.0 || !ordinate.is_finite() {
            return Err(LongMemoryError::NonPositivePeriodogram { j });
        }
        lambdas.push(two_pi * (j as f64) / (n as f64));
        i_j.push(ordinate);
    }
    Ok((lambdas, i_j))
}
