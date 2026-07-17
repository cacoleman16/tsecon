//! The raw periodogram: a single-segment spectral density estimate.

use crate::error::{check_finite, SpectralError};
use crate::helpers::{rfftfreq, segment_psd, RealFft};
use crate::window::{Detrend, Scaling, Window};

/// A one-sided power spectral density estimate on a real frequency grid.
///
/// `freqs[i]` (cycles per unit time) carries density `psd[i]`. The grid runs
/// from `0` (DC) to `fs/2` (Nyquist); with density scaling the trapezoidal
/// integral of `psd` over `freqs` recovers the signal variance.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerSpectrum {
    /// One-sided frequency grid `rfftfreq(n, 1/fs)`, length `n/2 + 1`.
    pub freqs: Vec<f64>,
    /// Power spectral density at each frequency (nonnegative).
    pub psd: Vec<f64>,
}

/// Raw periodogram of `x` via a single FFT (Schuster 1898; Percival &
/// Walden 1993 §6).
///
/// The segment is detrended, tapered by `window`, transformed, and scaled.
/// For the one-sided spectrum every bin except DC and — when `n` is even —
/// Nyquist is doubled to account for the folded negative frequencies. With
/// `window = Boxcar`, `scaling = Density`, `detrend = None` the estimate is
///
/// ```text
///   psd[i] = |X[i]|^2 / (fs * n),   doubled for 0 < i < n/2,
/// ```
///
/// where `X = rfft(x)`; this matches
/// `scipy.signal.periodogram(x, fs, window='boxcar',
/// scaling='density', detrend=False)` to floating-point precision.
///
/// # Errors
/// Returns [`SpectralError::NonFiniteInput`] if `x` holds a NaN/infinity,
/// or [`SpectralError::InvalidParameter`] if `fs <= 0` or `x` is empty.
///
/// # Example
/// ```
/// use tsecon_spectral::{periodogram, Detrend, Scaling, Window};
/// let x: Vec<f64> = (0..64).map(|t| (0.2 * t as f64).sin()).collect();
/// let s = periodogram(&x, 1.0, Window::Boxcar, Scaling::Density, Detrend::None).unwrap();
/// assert_eq!(s.freqs.len(), x.len() / 2 + 1);
/// assert!(s.psd.iter().all(|&p| p >= 0.0));
/// ```
pub fn periodogram(
    x: &[f64],
    fs: f64,
    window: Window,
    scaling: Scaling,
    detrend: Detrend,
) -> Result<PowerSpectrum, SpectralError> {
    check_finite(x)?;
    let n = x.len();
    if n == 0 {
        return Err(SpectralError::InvalidParameter {
            name: "x.len()",
            value: 0.0,
            requirement: "at least 1 observation",
        });
    }
    if fs.is_nan() || fs <= 0.0 {
        return Err(SpectralError::InvalidParameter {
            name: "fs",
            value: fs,
            requirement: "a positive sampling frequency",
        });
    }

    let detrended = detrend.apply(x);
    let win = window.values(n);
    let scale = scaling.factor(fs, &win);
    let windowed: Vec<f64> = detrended
        .iter()
        .zip(win.iter())
        .map(|(v, w)| v * w)
        .collect();

    let spectrum = RealFft::new(n).transform(&windowed);
    let psd = segment_psd(&spectrum, scale, n);
    let freqs = rfftfreq(n, fs);
    Ok(PowerSpectrum { freqs, psd })
}
