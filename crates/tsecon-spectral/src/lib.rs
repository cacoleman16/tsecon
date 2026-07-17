//! # tsecon-spectral
//!
//! Frequency-domain analysis for the `tsecon` time-series econometrics
//! library — the spectral rows of the diagnostics/exploration module
//! (ROADMAP §01, "Spectral analysis"; re-exported through `tsecon-diag`).
//!
//! Estimators:
//!
//! * [`periodogram`] — the raw periodogram via one FFT (Schuster 1898),
//!   with selectable [`Window`], [`Scaling`], and [`Detrend`];
//! * [`welch`] — Welch's (1967) averaged modified periodogram: overlapping
//!   windowed segments averaged into a lower-variance PSD;
//! * [`coherence`] — magnitude-squared coherence
//!   `|Pxy|^2 / (Pxx Pyy)` from the Welch cross-spectral density, the
//!   frequency-domain measure of comovement (Priestley 1981);
//! * [`PeriodBand`] and [`frequency_to_period`] — the period<->frequency
//!   bridge that lets a macro user read a spectrum in business-cycle terms
//!   ("6-32 quarters carry X% of the variance").
//!
//! ## Conventions
//!
//! Every normalisation follows `scipy.signal` so results are directly
//! comparable, and the golden tests pin `periodogram`, `welch`, and
//! `coherence` against SciPy 1.18.0 (`fixtures/spectral.json`) to `1e-8`
//! relative. One-sided spectra double every bin except DC and (for even `n`)
//! Nyquist; density scaling makes the spectrum integrate to the variance.
//!
//! References: Welch (1967); Percival & Walden, *Spectral Analysis for
//! Physical Applications* (1993); Priestley, *Spectral Analysis and Time
//! Series* (1981).
//!
//! ```
//! use tsecon_spectral::{periodogram, PeriodBand, Detrend, Scaling, Window};
//!
//! // A cycle every 8 samples on top of noise.
//! let x: Vec<f64> = (0..256)
//!     .map(|t| (2.0 * std::f64::consts::PI * t as f64 / 8.0).sin())
//!     .collect();
//! let s = periodogram(&x, 1.0, Window::Boxcar, Scaling::Density, Detrend::None).unwrap();
//!
//! // The 8-sample cycle is frequency 1/8 = 0.125; it dominates the spectrum.
//! let peak = s
//!     .freqs
//!     .iter()
//!     .zip(&s.psd)
//!     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
//!     .unwrap()
//!     .0;
//! assert!((peak - 0.125).abs() < 1e-9);
//!
//! // Read it as a period band: periods of 6-10 samples carry the variance.
//! let share = PeriodBand::new(6.0, 10.0).variance_share(&s);
//! assert!(share > 0.9);
//! ```

#![warn(missing_docs)]
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod business_cycle;
mod error;
mod helpers;
mod periodogram;
mod welch;
mod window;

pub use business_cycle::{frequency_to_period, period_to_frequency, PeriodBand};
pub use error::SpectralError;
pub use periodogram::{periodogram, PowerSpectrum};
pub use welch::{coherence, welch, Coherence};
pub use window::{Detrend, Scaling, Window};
