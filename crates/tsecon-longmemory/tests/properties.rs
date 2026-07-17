//! Monte-Carlo property tests — the statistical validation of the crate.
//!
//! The golden fixtures pin the *algebra*; these seeded simulations establish
//! that the algebra recovers a true memory parameter. On simulated
//! ARFIMA(0, d, 0) series `x = (1 - L)^{-d} e`, `e ~ N(0, 1)`, with known
//! `d in {0.2, 0.4}`, both the GPH log-periodogram estimator and the Robinson
//! (1995) local-Whittle estimator recover `d` within Monte-Carlo bands.
//!
//! All randomness is the library's seeded Philox stream (`tsecon_rng`), so the
//! numbers are reproducible run to run. Both estimators are consistent but
//! finite-sample biased; the acceptance band (0.06) is chosen to be many
//! standard-errors-of-the-mean wide at `reps = 300`, `n = 2048`.

use tsecon_longmemory::{default_bandwidth, frac_integrate, gph, local_whittle};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

/// Standard normal via the inverse-CDF of a Philox uniform.
fn gaussian(s: &mut Stream) -> f64 {
    let u = s.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Simulate ARFIMA(0, d, 0): fractionally integrate i.i.d. N(0,1) noise.
fn simulate_arfima(s: &mut Stream, n: usize, d: f64) -> Vec<f64> {
    let e: Vec<f64> = (0..n).map(|_| gaussian(s)).collect();
    frac_integrate(&e, d).expect("frac_integrate on finite noise")
}

const REPS: usize = 300;
const N: usize = 2048;
const BAND: f64 = 0.06;

#[test]
fn gph_recovers_known_memory_parameter() {
    let m = default_bandwidth(N);
    for &d_true in &[0.2_f64, 0.4_f64] {
        let mut s = Stream::new(0x6D_A0 ^ ((d_true * 1000.0) as u64));
        let mut sum = 0.0_f64;
        for _ in 0..REPS {
            let x = simulate_arfima(&mut s, N, d_true);
            sum += gph(&x, m).expect("gph").d;
        }
        let mean = sum / REPS as f64;
        assert!(
            (mean - d_true).abs() < BAND,
            "GPH mean d_hat = {mean:.4} not within {BAND} of true d = {d_true} (m = {m})"
        );
    }
}

#[test]
fn local_whittle_recovers_known_memory_parameter() {
    let m = default_bandwidth(N);
    for &d_true in &[0.2_f64, 0.4_f64] {
        let mut s = Stream::new(0x777 ^ ((d_true * 1000.0) as u64));
        let mut sum = 0.0_f64;
        for _ in 0..REPS {
            let x = simulate_arfima(&mut s, N, d_true);
            sum += local_whittle(&x, m).expect("local_whittle").d;
        }
        let mean = sum / REPS as f64;
        assert!(
            (mean - d_true).abs() < BAND,
            "local-Whittle mean d_hat = {mean:.4} not within {BAND} of true d = {d_true} (m = {m})"
        );
    }
}

/// The estimators run on the same series and, being consistent for the same
/// `d`, land near each other on a long realisation.
#[test]
fn gph_and_local_whittle_agree_on_a_long_series() {
    let mut s = Stream::new(0xA11CE);
    let d_true = 0.3;
    let x = simulate_arfima(&mut s, 4096, d_true);
    let m = default_bandwidth(x.len());
    let g = gph(&x, m).expect("gph").d;
    let w = local_whittle(&x, m).expect("lw").d;
    assert!(
        (g - w).abs() < 0.15,
        "GPH d = {g:.4} and local-Whittle d = {w:.4} disagree on a long series"
    );
}
