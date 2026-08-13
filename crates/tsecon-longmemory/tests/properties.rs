//! Monte-Carlo property tests — the statistical validation of the crate.
//!
//! The golden fixtures pin the *algebra*; these seeded simulations establish
//! that the algebra recovers a true memory parameter. On simulated
//! ARFIMA(0, d, 0) series `x = (1 - L)^{-d} e`, `e ~ N(0, 1)`, with known
//! `d in {0.2, 0.4}`, both the GPH log-periodogram estimator and the Robinson
//! (1995) local-Whittle estimator recover `d` within Monte-Carlo bands.
//!
//! The same simulations also **calibrate the reported standard errors**: a
//! reported `se` earns its name only if it tracks the realised sampling
//! dispersion of `d_hat`. That check is run across a *sweep of bandwidths*,
//! because the SE bug it exists to catch — reporting the large-`m` limit
//! `pi/sqrt(24 m)` or `1/(2 sqrt(m))` in place of the exact expression — is
//! invisible at one bandwidth and grows as `m` shrinks (at `m = floor(sqrt n)`
//! the limits are 18-30% too narrow, so nominal-95% intervals cover 84-90%).
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

/// Sample standard deviation and nominal-95% coverage of a batch of estimates.
fn sd_and_coverage(estimates: &[f64], d_true: f64, se: f64) -> (f64, f64) {
    let r = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / r;
    let sd = (estimates
        .iter()
        .map(|&e| (e - mean) * (e - mean))
        .sum::<f64>()
        / (r - 1.0))
        .sqrt();
    let hit = estimates
        .iter()
        .filter(|&&e| (e - d_true).abs() <= 1.959_964 * se)
        .count() as f64;
    (sd, hit / r)
}

/// **The SE calibration test.** The reported `se` must equal the realised
/// sampling standard deviation of `d_hat`, and its nominal-95% interval must
/// actually cover 95%.
///
/// Run over a **bandwidth sweep** (`n = 256, m = 16` and `n = 512, m = 22`,
/// both the library's own `default_bandwidth`), because the failure this
/// guards against is bandwidth-dependent and a single-bandwidth test cannot
/// see it. Reporting the textbook large-`m` constants instead —
/// `pi/sqrt(24 m)` and `1/(2 sqrt m)` — puts the ratio at 0.70-0.82 across
/// these cells, far outside the band asserted here, and drops nominal-95%
/// coverage to 0.84-0.90.
#[test]
fn reported_se_tracks_the_realised_sampling_dispersion() {
    // Fixed seeds, so these are deterministic numbers, not a coin flip. The
    // band is wide enough for Monte-Carlo noise (~1/sqrt(2*reps) ~ 2% on the
    // ratio) and far tighter than the 20-30% error it is there to catch.
    const CAL_REPS: usize = 1000;
    const RATIO_LO: f64 = 0.85;
    const RATIO_HI: f64 = 1.15;

    for &n in &[256_usize, 512] {
        let m = default_bandwidth(n);
        for &d_true in &[0.0_f64, 0.2, 0.4] {
            let mut s = Stream::new(0x5E_0000 ^ (n as u64) ^ ((d_true * 1000.0) as u64));
            let mut g_hat = Vec::with_capacity(CAL_REPS);
            let mut w_hat = Vec::with_capacity(CAL_REPS);
            let mut g_se = f64::NAN;
            let mut w_se = f64::NAN;
            for _ in 0..CAL_REPS {
                let x = simulate_arfima(&mut s, n, d_true);
                let g = gph(&x, m).expect("gph");
                let w = local_whittle(&x, m).expect("local_whittle");
                // The SEs depend only on (n, m), so they are constant across
                // replications; assert that rather than assume it.
                if g_se.is_nan() {
                    g_se = g.se;
                    w_se = w.se;
                }
                assert_eq!(g.se, g_se, "gph.se is not a function of (n, m) alone");
                assert_eq!(w.se, w_se, "whittle.se is not a function of (n, m) alone");
                g_hat.push(g.d);
                w_hat.push(w.d);
            }

            for (name, hat, se) in [("GPH", &g_hat, g_se), ("local-Whittle", &w_hat, w_se)] {
                let (sd, cov) = sd_and_coverage(hat, d_true, se);
                let ratio = se / sd;
                println!(
                    "{name:13} n={n:5} m={m:3} d={d_true:.1}: se={se:.4} sd={sd:.4} \
                     ratio={ratio:.3} cov95={cov:.3}"
                );
                assert!(
                    (RATIO_LO..=RATIO_HI).contains(&ratio),
                    "{name} n={n} m={m} d={d_true}: reported se = {se:.4} but the realised \
                     sampling sd is {sd:.4} (ratio {ratio:.3}, want {RATIO_LO}..{RATIO_HI})"
                );
                // Coverage floor is 0.89, not 0.94: local Whittle retains a
                // genuine ~5-9% narrowness at these bandwidths (its ratio runs
                // 0.91-0.94, cover 0.91-0.93) because the low-frequency
                // periodogram ordinates are not exactly i.i.d. Exp(1) at small
                // m. That residual is documented, not papered over. The
                // textbook constants sit at 0.84-0.90 and still fail here.
                assert!(
                    (0.89..=0.98).contains(&cov),
                    "{name} n={n} m={m} d={d_true}: nominal-95% intervals from se = {se:.4} \
                     covered {cov:.3} of {CAL_REPS} replications"
                );
            }
        }
    }
}

/// The reported SEs carry no scale of their own: rescaling the data multiplies
/// the periodogram by a constant, which both estimators absorb, so `d` and
/// every SE must be bit-stable across many orders of magnitude. (A tolerance,
/// ridge or floor compared against a quantity in units of `y^2` would show up
/// here.)
#[test]
fn estimates_and_ses_are_invariant_to_the_scale_of_the_data() {
    let mut s = Stream::new(0xCA1B);
    let base = simulate_arfima(&mut s, 1024, 0.3);
    let m = default_bandwidth(base.len());
    let g0 = gph(&base, m).expect("gph");
    let w0 = local_whittle(&base, m).expect("local_whittle");
    for &c in &[1e-8_f64, 1e-3, 1e3, 1e8] {
        let scaled: Vec<f64> = base.iter().map(|&v| v * c).collect();
        let g = gph(&scaled, m).expect("gph on scaled");
        let w = local_whittle(&scaled, m).expect("local_whittle on scaled");
        assert!(
            (g.d - g0.d).abs() < 1e-10,
            "GPH d moved from {:.12} to {:.12} under scaling by {c:e}",
            g0.d,
            g.d
        );
        // Local Whittle goes through a derivative-free minimizer, so `d` is
        // reproduced only to the simplex's termination granularity (~1.5e-8,
        // i.e. 2^-26 on a 0.1 initial step). That granularity is flat in `c`:
        // measured over 24 orders of magnitude of rescaling it never grows,
        // which is the point — a scale-carrying tolerance would show drift.
        assert!(
            (w.d - w0.d).abs() < 1e-6,
            "local-Whittle d moved from {:.12} to {:.12} under scaling by {c:e}",
            w0.d,
            w.d
        );
        assert_eq!(g.se, g0.se, "gph.se changed under scaling by {c:e}");
        assert_eq!(
            g.se_asymptotic, g0.se_asymptotic,
            "gph.se_asymptotic changed under scaling by {c:e}"
        );
        assert_eq!(w.se, w0.se, "whittle.se changed under scaling by {c:e}");
        assert_eq!(
            w.se_asymptotic, w0.se_asymptotic,
            "whittle.se_asymptotic changed under scaling by {c:e}"
        );
        assert!(
            (g.se_regression - g0.se_regression).abs() < 1e-10,
            "gph.se_regression moved under scaling by {c:e}"
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
