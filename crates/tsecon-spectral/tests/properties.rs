//! Property tests: PSD nonnegativity, coherence in [0,1], the sinusoid peak,
//! Parseval's variance identity, the period<->frequency bridge, detrending,
//! and the error paths.

use std::f64::consts::PI;

use tsecon_spectral::{
    coherence, frequency_to_period, period_to_frequency, periodogram, welch, Detrend, PeriodBand,
    Scaling, SpectralError, Window,
};

/// Deterministic pseudo-noise via a fixed LCG (no RNG dependency).
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn sinusoid(n: usize, freq: f64) -> Vec<f64> {
    (0..n).map(|t| (2.0 * PI * freq * t as f64).sin()).collect()
}

#[test]
fn periodogram_psd_is_nonnegative() {
    let x = noise(200, 0x1234_5678);
    for window in [Window::Boxcar, Window::Hann] {
        for scaling in [Scaling::Density, Scaling::Spectrum] {
            let s = periodogram(&x, 1.0, window, scaling, Detrend::Constant).unwrap();
            assert!(s.psd.iter().all(|&p| p >= 0.0), "psd has a negative bin");
        }
    }
}

#[test]
fn welch_psd_is_nonnegative() {
    let x = noise(400, 0xABCD);
    let s = welch(
        &x,
        1.0,
        64,
        None,
        Window::Hann,
        Scaling::Density,
        Detrend::None,
    )
    .unwrap();
    assert!(s.psd.iter().all(|&p| p >= 0.0));
}

#[test]
fn coherence_lies_in_unit_interval() {
    // Build y as a filtered, noisy copy of x so coherence varies across
    // frequency but is never outside [0, 1].
    let x = noise(600, 0x5EED);
    let e = noise(600, 0xF00D);
    let y: Vec<f64> = (0..x.len())
        .map(|i| {
            let lag = if i > 0 { x[i - 1] } else { 0.0 };
            0.6 * x[i] + 0.3 * lag + 0.5 * e[i]
        })
        .collect();
    let c = coherence(&x, &y, 1.0, 128, None, Window::Hann, Detrend::None).unwrap();
    for (i, &v) in c.coherence.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "coherence[{i}] = {v} escaped [0, 1]"
        );
    }
}

#[test]
fn periodogram_of_a_sinusoid_peaks_at_its_frequency() {
    // A pure tone at f0 = 0.1875 = 48/256 lands exactly on a Fourier bin.
    let f0 = 0.1875;
    let x = sinusoid(256, f0);
    let s = periodogram(&x, 1.0, Window::Boxcar, Scaling::Density, Detrend::None).unwrap();
    let peak_freq = s
        .freqs
        .iter()
        .zip(&s.psd)
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!(
        (peak_freq - f0).abs() < 1e-12,
        "peak at {peak_freq}, want {f0}"
    );
}

#[test]
fn periodogram_satisfies_parseval() {
    // One-sided density periodogram (boxcar, no detrend): the Riemann sum
    // sum(psd) * df with df = fs/n integrates to the mean square of x.
    let x = noise(512, 0x00FF_1CE0);
    let fs = 1.0;
    let n = x.len();
    let s = periodogram(&x, fs, Window::Boxcar, Scaling::Density, Detrend::None).unwrap();
    let df = fs / n as f64;
    let integral: f64 = s.psd.iter().sum::<f64>() * df;
    let mean_square = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    assert!(
        (integral - mean_square).abs() <= 1e-10 * mean_square.max(1.0),
        "Parseval: integral {integral}, mean square {mean_square}"
    );
}

#[test]
fn period_frequency_bridge_roundtrips() {
    for &p in &[2.0, 6.0, 8.0, 32.0, 96.0] {
        let f = period_to_frequency(p);
        assert!((frequency_to_period(f) - p).abs() < 1e-12);
    }
    assert_eq!(frequency_to_period(0.0), f64::INFINITY);
    assert_eq!(period_to_frequency(f64::INFINITY), 0.0);
    // A period of 8 samples is frequency 0.125.
    assert!((period_to_frequency(8.0) - 0.125).abs() < 1e-15);
}

#[test]
fn business_cycle_band_captures_its_tone() {
    // An 8-sample cycle sits inside the 6-10 period band and should carry
    // almost all the variance; the band share is a valid fraction.
    let x = sinusoid(256, 0.125);
    let s = periodogram(&x, 1.0, Window::Boxcar, Scaling::Density, Detrend::None).unwrap();
    let band = PeriodBand::new(6.0, 10.0);
    let share = band.variance_share(&s);
    assert!((0.0..=1.0).contains(&share));
    assert!(
        share > 0.9,
        "expected the 8-period tone in band, got {share}"
    );
    // The tone frequency 0.125 is inside the band; DC (freq 0) is not.
    assert!(band.contains_frequency(0.125));
    assert!(!band.contains_frequency(0.0));
    // Quarterly business-cycle band edges: 1/32 .. 1/6.
    let bc = PeriodBand::business_cycle_quarterly();
    assert!((bc.low_frequency() - 1.0 / 32.0).abs() < 1e-15);
    assert!((bc.high_frequency() - 1.0 / 6.0).abs() < 1e-15);
}

#[test]
fn constant_detrend_zeroes_the_dc_bin() {
    let x: Vec<f64> = noise(128, 0x99).iter().map(|v| v + 10.0).collect();
    let s = periodogram(&x, 1.0, Window::Boxcar, Scaling::Density, Detrend::Constant).unwrap();
    assert!(s.psd[0] < 1e-18, "DC bin not removed: {}", s.psd[0]);
}

#[test]
fn linear_detrend_annihilates_a_straight_line() {
    let x: Vec<f64> = (0..128).map(|t| 3.0 + 0.5 * t as f64).collect();
    let s = periodogram(&x, 1.0, Window::Boxcar, Scaling::Density, Detrend::Linear).unwrap();
    assert!(
        s.psd.iter().all(|&p| p < 1e-16),
        "linear detrend left spectral mass on a line"
    );
}

#[test]
fn rejects_non_finite_input() {
    let mut x = noise(64, 1);
    x[10] = f64::NAN;
    assert_eq!(
        periodogram(&x, 1.0, Window::Boxcar, Scaling::Density, Detrend::None),
        Err(SpectralError::NonFiniteInput { index: 10 })
    );
}

#[test]
fn rejects_bad_parameters() {
    let x = noise(64, 2);
    assert!(matches!(
        periodogram(&x, 0.0, Window::Boxcar, Scaling::Density, Detrend::None),
        Err(SpectralError::InvalidParameter { name: "fs", .. })
    ));
    assert!(matches!(
        periodogram(&[], 1.0, Window::Boxcar, Scaling::Density, Detrend::None),
        Err(SpectralError::InvalidParameter { .. })
    ));
    assert_eq!(
        welch(
            &x,
            1.0,
            128,
            None,
            Window::Hann,
            Scaling::Density,
            Detrend::None
        ),
        Err(SpectralError::SegmentTooLong {
            nperseg: 128,
            n: 64
        })
    );
    assert!(matches!(
        welch(
            &x,
            1.0,
            32,
            Some(32),
            Window::Hann,
            Scaling::Density,
            Detrend::None
        ),
        Err(SpectralError::InvalidParameter {
            name: "noverlap",
            ..
        })
    ));
    assert!(matches!(
        welch(
            &x,
            1.0,
            0,
            None,
            Window::Hann,
            Scaling::Density,
            Detrend::None
        ),
        Err(SpectralError::InvalidParameter {
            name: "nperseg",
            ..
        })
    ));
}

#[test]
fn coherence_rejects_length_mismatch() {
    let x = noise(200, 3);
    let y = noise(180, 4);
    assert_eq!(
        coherence(&x, &y, 1.0, 64, None, Window::Hann, Detrend::None),
        Err(SpectralError::LengthMismatch {
            x_len: 200,
            y_len: 180
        })
    );
}
