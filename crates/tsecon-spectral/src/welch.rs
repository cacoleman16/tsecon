//! Welch's averaged-periodogram PSD and the magnitude-squared coherence
//! built on the same segmenting machinery.

use rustfft::num_complex::Complex;

use crate::error::{check_finite, SpectralError};
use crate::helpers::{rfftfreq, segment_csd, segment_psd, RealFft};
use crate::periodogram::PowerSpectrum;
use crate::window::{Detrend, Scaling, Window};

/// A one-sided magnitude-squared coherence estimate on a real frequency grid.
///
/// `coherence[i]` lies in `[0, 1]` and measures the fraction of the variance
/// of one series at frequency `freqs[i]` that is linearly predictable from
/// the other. It is identically `1` for a single segment, which is why
/// coherence is only meaningful once Welch averaging (`>= 2` segments) is in
/// play (Priestley 1981).
#[derive(Debug, Clone, PartialEq)]
pub struct Coherence {
    /// One-sided frequency grid `rfftfreq(nperseg, 1/fs)`.
    pub freqs: Vec<f64>,
    /// Magnitude-squared coherence at each frequency, in `[0, 1]`.
    pub coherence: Vec<f64>,
}

/// Validate the Welch segmentation parameters and enumerate the segment
/// start indices (the hop between them is `nperseg - noverlap`).
fn frame(n: usize, nperseg: usize, noverlap: Option<usize>) -> Result<Vec<usize>, SpectralError> {
    if nperseg == 0 {
        return Err(SpectralError::InvalidParameter {
            name: "nperseg",
            value: 0.0,
            requirement: "a positive segment length",
        });
    }
    if nperseg > n {
        return Err(SpectralError::SegmentTooLong { nperseg, n });
    }
    // SciPy default: 50% overlap.
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    if noverlap >= nperseg {
        return Err(SpectralError::InvalidParameter {
            name: "noverlap",
            value: noverlap as f64,
            requirement: "noverlap < nperseg",
        });
    }
    let step = nperseg - noverlap;
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start + nperseg <= n {
        starts.push(start);
        start += step;
    }
    Ok(starts)
}

/// Windowed, detrended FFT of the segment beginning at `start`.
fn segment_spectrum(
    x: &[f64],
    start: usize,
    nperseg: usize,
    detrend: Detrend,
    win: &[f64],
    fft: &RealFft,
) -> Vec<Complex<f64>> {
    let detrended = detrend.apply(&x[start..start + nperseg]);
    let windowed: Vec<f64> = detrended
        .iter()
        .zip(win.iter())
        .map(|(v, w)| v * w)
        .collect();
    fft.transform(&windowed)
}

/// Welch's method: the averaged modified periodogram (Welch 1967).
///
/// `x` is split into segments of length `nperseg` overlapping by `noverlap`
/// (default `nperseg / 2`, i.e. 50%). Each segment is detrended, tapered by
/// `window`, and turned into a modified periodogram with the
/// `1 / (fs * sum(w^2))` (density) normalisation; the segment periodograms
/// are then averaged. With `window = Hann`, `noverlap = None`,
/// `scaling = Density`, `detrend = None` this matches
/// `scipy.signal.welch(x, fs, nperseg=nperseg, detrend=False)` to
/// floating-point precision.
///
/// # Errors
/// [`SpectralError::NonFiniteInput`] for a non-finite sample,
/// [`SpectralError::SegmentTooLong`] if `nperseg > x.len()`, and
/// [`SpectralError::InvalidParameter`] for `fs <= 0`, `nperseg == 0`, or
/// `noverlap >= nperseg`.
pub fn welch(
    x: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
    window: Window,
    scaling: Scaling,
    detrend: Detrend,
) -> Result<PowerSpectrum, SpectralError> {
    check_finite(x)?;
    if fs.is_nan() || fs <= 0.0 {
        return Err(SpectralError::InvalidParameter {
            name: "fs",
            value: fs,
            requirement: "a positive sampling frequency",
        });
    }
    let starts = frame(x.len(), nperseg, noverlap)?;
    let win = window.values(nperseg);
    let scale = scaling.factor(fs, &win);
    let fft = RealFft::new(nperseg);

    let nbins = nperseg / 2 + 1;
    let mut acc = vec![0.0f64; nbins];
    for &start in &starts {
        let spectrum = segment_spectrum(x, start, nperseg, detrend, &win, &fft);
        let psd = segment_psd(&spectrum, scale, nperseg);
        for (a, p) in acc.iter_mut().zip(psd.iter()) {
            *a += p;
        }
    }
    let nseg = starts.len() as f64;
    for a in acc.iter_mut() {
        *a /= nseg;
    }
    let freqs = rfftfreq(nperseg, fs);
    Ok(PowerSpectrum { freqs, psd: acc })
}

/// Magnitude-squared coherence `|Pxy|^2 / (Pxx * Pyy)` between `x` and `y`
/// (Welch cross-spectral machinery; Priestley 1981; Percival & Walden 1993).
///
/// `Pxx` and `Pyy` are the Welch auto-spectra and `Pxy` the Welch
/// cross-spectral density — the segment cross-periodograms
/// `conj(Xx) * Xy`, averaged with the same windowing and overlap as
/// [`welch`]. The result is invariant to the scaling convention, so density
/// scaling is used throughout. Matches
/// `scipy.signal.coherence(x, y, fs, nperseg=nperseg, detrend=False)` to
/// floating-point precision.
///
/// A bin whose averaged auto-power is zero (a degenerate all-constant band)
/// has undefined coherence; it is reported as `0.0` rather than `NaN`.
///
/// # Errors
/// [`SpectralError::LengthMismatch`] if `x` and `y` differ in length, plus
/// every error condition of [`welch`].
pub fn coherence(
    x: &[f64],
    y: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
    window: Window,
    detrend: Detrend,
) -> Result<Coherence, SpectralError> {
    check_finite(x)?;
    check_finite(y)?;
    if x.len() != y.len() {
        return Err(SpectralError::LengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    if fs.is_nan() || fs <= 0.0 {
        return Err(SpectralError::InvalidParameter {
            name: "fs",
            value: fs,
            requirement: "a positive sampling frequency",
        });
    }
    let starts = frame(x.len(), nperseg, noverlap)?;
    // Coherence is scale-invariant; density scaling fixes the convention.
    let win = window.values(nperseg);
    let scale = Scaling::Density.factor(fs, &win);
    let fft = RealFft::new(nperseg);

    let nbins = nperseg / 2 + 1;
    let mut pxx = vec![0.0f64; nbins];
    let mut pyy = vec![0.0f64; nbins];
    let mut pxy = vec![Complex::new(0.0, 0.0); nbins];
    for &start in &starts {
        let sx = segment_spectrum(x, start, nperseg, detrend, &win, &fft);
        let sy = segment_spectrum(y, start, nperseg, detrend, &win, &fft);
        let ppx = segment_psd(&sx, scale, nperseg);
        let ppy = segment_psd(&sy, scale, nperseg);
        let pcxy = segment_csd(&sx, &sy, scale, nperseg);
        for i in 0..nbins {
            pxx[i] += ppx[i];
            pyy[i] += ppy[i];
            pxy[i] += pcxy[i];
        }
    }
    // The 1/nseg averaging factor cancels in the ratio, so it is omitted.
    let coh: Vec<f64> = (0..nbins)
        .map(|i| {
            let denom = pxx[i] * pyy[i];
            if denom > 0.0 {
                (pxy[i].norm_sqr() / denom).min(1.0)
            } else {
                0.0
            }
        })
        .collect();
    let freqs = rfftfreq(nperseg, fs);
    Ok(Coherence {
        freqs,
        coherence: coh,
    })
}
