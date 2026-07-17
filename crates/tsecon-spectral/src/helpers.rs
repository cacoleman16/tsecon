//! Internal FFT and framing machinery shared by the estimators.
//!
//! Nothing here is public; the module exists so that [`crate::periodogram`],
//! [`crate::welch`], and [`crate::coherence`] share one definition of the
//! real FFT, the one-sided folding rule, and the frequency grid.

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// A cached forward FFT plan that turns a real length-`n` segment into its
/// one-sided complex spectrum (bins `0..=n/2`).
///
/// Planning is amortised across the many segments of a Welch average, which
/// is why the plan is stored rather than rebuilt per call.
pub(crate) struct RealFft {
    fft: Arc<dyn Fft<f64>>,
    n: usize,
}

impl RealFft {
    /// Plan a forward FFT of length `n`.
    pub(crate) fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        RealFft { fft, n }
    }

    /// Forward-transform a real segment and return the one-sided spectrum
    /// `X[0..=n/2]`. The segment length must equal the planned `n`.
    pub(crate) fn transform(&self, signal: &[f64]) -> Vec<Complex<f64>> {
        let mut buf: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
        self.fft.process(&mut buf);
        buf.truncate(self.n / 2 + 1);
        buf
    }
}

/// The one-sided frequency grid `rfftfreq(n, 1/fs)`:
/// `f[i] = i * fs / n`, `i = 0..=n/2`.
pub(crate) fn rfftfreq(n: usize, fs: f64) -> Vec<f64> {
    (0..n / 2 + 1).map(|i| i as f64 * fs / n as f64).collect()
}

/// True at every one-sided bin that must be doubled to stand in for its
/// negative-frequency mirror: everything except DC (`i == 0`) and, when `n`
/// is even, the Nyquist bin (`i == n/2`).
fn is_doubled(i: usize, one_sided_len: usize, n: usize) -> bool {
    if i == 0 {
        return false;
    }
    if n % 2 == 0 && i == one_sided_len - 1 {
        return false;
    }
    true
}

/// Modified periodogram of one already windowed+detrended real segment:
/// `|X[i]|^2 * scale`, with the interior bins doubled for the one-sided
/// spectrum.
pub(crate) fn segment_psd(spectrum: &[Complex<f64>], scale: f64, n: usize) -> Vec<f64> {
    let len = spectrum.len();
    spectrum
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let base = c.norm_sqr() * scale;
            if is_doubled(i, len, n) {
                2.0 * base
            } else {
                base
            }
        })
        .collect()
}

/// Cross-periodogram of one segment pair: `conj(Xx[i]) * Xy[i] * scale`,
/// interior bins doubled exactly as [`segment_psd`] doubles the auto-spectra
/// (the factor cancels in a coherence, but is applied for consistency with
/// the cross-spectral density itself).
pub(crate) fn segment_csd(
    spectrum_x: &[Complex<f64>],
    spectrum_y: &[Complex<f64>],
    scale: f64,
    n: usize,
) -> Vec<Complex<f64>> {
    let len = spectrum_x.len();
    spectrum_x
        .iter()
        .zip(spectrum_y.iter())
        .enumerate()
        .map(|(i, (xx, xy))| {
            let base = xx.conj() * xy * scale;
            if is_doubled(i, len, n) {
                base * 2.0
            } else {
                base
            }
        })
        .collect()
}
