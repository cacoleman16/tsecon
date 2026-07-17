//! Tapering windows, detrending, and PSD scaling conventions.
//!
//! These small option enums mirror `scipy.signal` so that a user who knows
//! the SciPy call can translate it one-to-one, and so that the crate's
//! golden tests can pin every convention that differs across
//! implementations.

/// Tapering window applied to each segment before the FFT.
///
/// A window `w[k]`, `k = 0..n`, trades spectral leakage against resolution.
/// The two supported here are the ones [`crate::periodogram`] and
/// [`crate::welch`] default to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// The rectangular (`boxcar`) window: `w[k] = 1`. No tapering; this is
    /// the raw periodogram window and SciPy's `periodogram` default.
    Boxcar,
    /// The periodic Hann window (SciPy `hann(n, sym=False)`):
    /// `w[k] = 0.5 - 0.5 cos(2 pi k / n)`, `k = 0..n`.
    ///
    /// "Periodic" (a.k.a. DFT-even) means the length-`n` window is the
    /// length-`n+1` symmetric Hann window with its last point dropped; this
    /// is the correct convention for spectral estimation and SciPy's
    /// `welch` default. Blackman-Harris (1978); see Percival & Walden (1993).
    Hann,
}

impl Window {
    /// Materialise the length-`n` window coefficients.
    pub fn values(&self, n: usize) -> Vec<f64> {
        match self {
            Window::Boxcar => vec![1.0; n],
            Window::Hann => (0..n)
                .map(|k| {
                    // Periodic Hann: divide by n, not n-1.
                    0.5 - 0.5 * (2.0 * core::f64::consts::PI * k as f64 / n as f64).cos()
                })
                .collect(),
        }
    }
}

/// Deterministic detrending applied to each segment before windowing,
/// mirroring the `detrend` argument of `scipy.signal.periodogram`/`welch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detrend {
    /// Leave the segment untouched (`detrend=False`).
    None,
    /// Subtract the segment mean (`detrend='constant'`).
    Constant,
    /// Subtract the least-squares straight-line fit (`detrend='linear'`).
    Linear,
}

impl Detrend {
    /// Return a detrended copy of `seg`.
    pub(crate) fn apply(&self, seg: &[f64]) -> Vec<f64> {
        match self {
            Detrend::None => seg.to_vec(),
            Detrend::Constant => {
                let n = seg.len();
                if n == 0 {
                    return Vec::new();
                }
                let mean = seg.iter().sum::<f64>() / n as f64;
                seg.iter().map(|v| v - mean).collect()
            }
            Detrend::Linear => {
                let n = seg.len();
                if n < 2 {
                    // A single point has no slope; fall back to demeaning.
                    return Detrend::Constant.apply(seg);
                }
                // Ordinary least squares of seg on t = 0..n. Closed form with
                // centred abscissae keeps the normal equations diagonal.
                let nf = n as f64;
                let t_mean = (nf - 1.0) / 2.0;
                let y_mean = seg.iter().sum::<f64>() / nf;
                let mut sxy = 0.0;
                let mut sxx = 0.0;
                for (k, &y) in seg.iter().enumerate() {
                    let dt = k as f64 - t_mean;
                    sxy += dt * (y - y_mean);
                    sxx += dt * dt;
                }
                let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
                let intercept = y_mean - slope * t_mean;
                seg.iter()
                    .enumerate()
                    .map(|(k, &y)| y - (intercept + slope * k as f64))
                    .collect()
            }
        }
    }
}

/// Normalisation of the power spectral density, mirroring the `scaling`
/// argument of `scipy.signal.periodogram`/`welch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scaling {
    /// Power-spectral-*density* scaling (`scaling='density'`): units of
    /// power per unit frequency, so the spectrum integrates to the signal
    /// variance. The per-segment normalisation is `1 / (fs * sum(w^2))`.
    Density,
    /// Power-*spectrum* scaling (`scaling='spectrum'`): units of power, so
    /// peaks read directly as the power of a sinusoid. The per-segment
    /// normalisation is `1 / (sum(w))^2`.
    Spectrum,
}

impl Scaling {
    /// Per-segment scale factor given the sampling frequency and window.
    pub(crate) fn factor(&self, fs: f64, window: &[f64]) -> f64 {
        match self {
            Scaling::Density => {
                let s2: f64 = window.iter().map(|w| w * w).sum();
                1.0 / (fs * s2)
            }
            Scaling::Spectrum => {
                let s1: f64 = window.iter().sum();
                1.0 / (s1 * s1)
            }
        }
    }
}
