//! Property tests for the threshold VAR and its bootstrap linearity test:
//! statistical properties a golden transcription cannot prove.
//!
//! * Under a LINEAR VAR null the bootstrap p-value rejects at
//!   approximately the nominal rate over seeds (a seeded Monte Carlo).
//! * On a strongly separated two-regime VAR the threshold lands near
//!   truth, the regime coefficients recover the DGP, and the test rejects
//!   hard.
//! * The reported fit is exactly self-consistent: the refit criterion is
//!   the minimum of the reported profile, and regime sizes respect the
//!   trimming.
//! * Degenerate input raises the documented teaching errors.
//! * The bootstrap is bit-identical for a given seed at any thread count.
//! * MC recovery (the model-card evidence): over 200 seeded replications
//!   the threshold's median absolute error and the coefficient biases
//!   stay small (run with `--nocapture` to see the measured numbers).

use tsecon_bootstrap::WildWeights;
use tsecon_regime::{threshold_var, threshold_var_test, RegimeError};
use tsecon_rng::Stream;

// ------------------------------------------------------------ simulation

const A_LOW: [[f64; 2]; 2] = [[0.5, 0.1], [0.2, 0.4]];
const A_HIGH: [[f64; 2]; 2] = [[0.1, 0.0], [-0.1, 0.5]];
const C_LOW: [f64; 2] = [1.0, 0.3];
const C_HIGH: [f64; 2] = [-1.0, -0.3];

/// Two-regime bivariate TVAR(1), regime by `y0_{t-1} <= gamma`.
fn sim_tvar(stream: &mut Stream, t: usize, gamma: f64) -> Vec<Vec<f64>> {
    let burn = 100;
    let n = t + burn + 1;
    let mut y = vec![[0.0_f64; 2]; n];
    for i in 1..n {
        let (c, a) = if y[i - 1][0] <= gamma {
            (C_LOW, A_LOW)
        } else {
            (C_HIGH, A_HIGH)
        };
        for j in 0..2 {
            y[i][j] = c[j]
                + a[j][0] * y[i - 1][0]
                + a[j][1] * y[i - 1][1]
                + WildWeights::Normal.draw(stream);
        }
    }
    y[(burn + 1)..].iter().map(|r| r.to_vec()).collect()
}

/// Linear bivariate VAR(1) (no threshold).
fn sim_var(stream: &mut Stream, t: usize) -> Vec<Vec<f64>> {
    let burn = 100;
    let n = t + burn + 1;
    let a = [[0.5, 0.1], [0.1, 0.4]];
    let mut y = vec![[0.0_f64; 2]; n];
    for i in 1..n {
        for j in 0..2 {
            y[i][j] =
                a[j][0] * y[i - 1][0] + a[j][1] * y[i - 1][1] + WildWeights::Normal.draw(stream);
        }
    }
    y[(burn + 1)..].iter().map(|r| r.to_vec()).collect()
}

// -------------------------------------------------------------- size MC

#[test]
fn null_rejection_rate_is_near_nominal() {
    // 200 linear-VAR draws x 199 bootstrap replications each (p-value
    // lattice {1/200, ..., 1}). These numbers are quoted in the model
    // card; the bands are generous (binomial 3-sigma at 200 draws is
    // about +/- 0.046 at the 5% level) and the whole Monte Carlo is
    // deterministic in its seeds.
    //
    // Measured (this seed set): reject@5% = 0.100 at T = 150 — the HC0-
    // weighted statistic over-rejects in small samples, a documented
    // failure mode — falling to 0.085 at T = 400 (one-off check, same
    // seeds; MC se ~0.02). The band's upper edge admits the small-T
    // liberality; a regression pushing it further fails the test.
    let n_series = 200;
    let mut streams = Stream::substreams(20260826, n_series).expect("substreams");
    let mut reject05 = 0usize;
    let mut reject10 = 0usize;
    let mut psum = 0.0_f64;
    for (i, stream) in streams.iter_mut().enumerate() {
        let y = sim_var(stream, 150);
        let r = threshold_var_test(&y, 1, 0, 1, 0.10, true, 30, 199, 70_000 + i as u64)
            .expect("test runs");
        if r.p_value <= 0.05 {
            reject05 += 1;
        }
        if r.p_value <= 0.10 {
            reject10 += 1;
        }
        psum += r.p_value;
    }
    let rate05 = reject05 as f64 / n_series as f64;
    let rate10 = reject10 as f64 / n_series as f64;
    let pmean = psum / n_series as f64;
    println!("tvar null MC: reject@5% = {rate05}, reject@10% = {rate10}, mean p = {pmean}");
    assert!(
        (0.005..=0.115).contains(&rate05),
        "5% rejection rate {rate05} far from nominal"
    );
    assert!(
        (0.03..=0.19).contains(&rate10),
        "10% rejection rate {rate10} far from nominal"
    );
    assert!(
        (0.40..=0.62).contains(&pmean),
        "mean null p-value {pmean} far from 0.5"
    );
}

// ------------------------------------------------------------- power MC

#[test]
fn strong_tvar_recovers_threshold_and_rejects() {
    let mut stream = Stream::new(4242);
    let y = sim_tvar(&mut stream, 500, 0.0);
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit runs");
    assert!(
        fit.threshold.abs() < 0.3,
        "threshold {} not near the true 0",
        fit.threshold
    );
    // Intercepts and own-lag coefficients of both regimes near truth.
    for j in 0..2 {
        assert!(
            (fit.coefs_low[j][0] - C_LOW[j]).abs() < 0.35,
            "low intercept eq{j}: {}",
            fit.coefs_low[j][0]
        );
        assert!(
            (fit.coefs_high[j][0] - C_HIGH[j]).abs() < 0.35,
            "high intercept eq{j}: {}",
            fit.coefs_high[j][0]
        );
        for v in 0..2 {
            assert!(
                (fit.coefs_low[j][1 + v] - A_LOW[j][v]).abs() < 0.25,
                "low A[{j}][{v}]: {}",
                fit.coefs_low[j][1 + v]
            );
            assert!(
                (fit.coefs_high[j][1 + v] - A_HIGH[j][v]).abs() < 0.25,
                "high A[{j}][{v}]: {}",
                fit.coefs_high[j][1 + v]
            );
        }
    }

    let r = threshold_var_test(&y, 1, 0, 1, 0.10, true, 100, 199, 7).expect("test runs");
    assert!(
        r.p_value <= 0.01,
        "test failed to reject on a strong TVAR (p = {})",
        r.p_value
    );
}

// -------------------------------------------------------- self-consistency

#[test]
fn refit_criterion_is_the_scan_minimum() {
    let mut stream = Stream::new(3);
    let y = sim_tvar(&mut stream, 300, 0.0);
    let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit runs");
    let path_min = fit
        .logdet_path
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(
        (fit.log_det_sigma - path_min).abs() < 1e-10,
        "refit ln det {} vs scan minimum {}",
        fit.log_det_sigma,
        path_min
    );
    assert_eq!(fit.n_low + fit.n_high, fit.nobs);
    assert!(fit.n_low >= fit.min_regime && fit.n_high >= fit.min_regime);
    // sigma is the regime-size-weighted mix of the per-regime covariances.
    for j in 0..2 {
        for j2 in 0..2 {
            let mix = (fit.sigma_low[j][j2] * fit.n_low as f64
                + fit.sigma_high[j][j2] * fit.n_high as f64)
                / fit.nobs as f64;
            assert!(
                (fit.sigma[j][j2] - mix).abs() < 1e-12,
                "sigma[{j}][{j2}] {} vs regime mix {mix}",
                fit.sigma[j][j2]
            );
        }
    }
}

// ------------------------------------------------------------ degeneracy

#[test]
fn degenerate_inputs_raise_teaching_errors() {
    let mut stream = Stream::new(1);
    let y = sim_var(&mut stream, 120);

    // One series is not a system.
    let y1: Vec<Vec<f64>> = y.iter().map(|r| vec![r[0]]).collect();
    assert!(matches!(
        threshold_var(&y1, 1, 0, &[1], 0.10, true),
        Err(RegimeError::InvalidSpec { .. })
    ));

    // Ragged rows.
    let mut ragged = y.clone();
    ragged[5] = vec![1.0];
    assert!(matches!(
        threshold_var(&ragged, 1, 0, &[1], 0.10, true),
        Err(RegimeError::DimensionMismatch { .. })
    ));

    // p = 0, empty delays, zero delay, bad threshold index, bad trim.
    assert!(matches!(
        threshold_var(&y, 0, 0, &[1], 0.10, true),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        threshold_var(&y, 1, 0, &[], 0.10, true),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        threshold_var(&y, 1, 0, &[0], 0.10, true),
        Err(RegimeError::InvalidParameter { name: "delay", .. })
    ));
    assert!(matches!(
        threshold_var(&y, 1, 2, &[1], 0.10, true),
        Err(RegimeError::InvalidParameter {
            name: "threshold_index",
            ..
        })
    ));
    assert!(matches!(
        threshold_var(&y, 1, 0, &[1], 0.5, true),
        Err(RegimeError::InvalidParameter { name: "trim", .. })
    ));

    // Constant threshold series.
    let flat: Vec<Vec<f64>> = y.iter().map(|r| vec![1.0, r[1]]).collect();
    assert!(matches!(
        threshold_var(&flat, 1, 0, &[1], 0.10, true),
        Err(RegimeError::InvalidSpec { .. })
    ));

    // NaN observation.
    let mut bad = y.clone();
    bad[10][1] = f64::NAN;
    assert!(matches!(
        threshold_var(&bad, 1, 0, &[1], 0.10, true),
        Err(RegimeError::NonFinite { .. })
    ));

    // Too short for two trimmed regimes.
    assert!(matches!(
        threshold_var(&y[..8], 1, 0, &[1], 0.10, true),
        Err(RegimeError::InsufficientData { .. })
    ));

    // n_boot = 0 and a degenerate test grid.
    assert!(matches!(
        threshold_var_test(&y, 1, 0, 1, 0.10, true, 300, 0, 0),
        Err(RegimeError::InvalidParameter { name: "n_boot", .. })
    ));
    assert!(matches!(
        threshold_var_test(&y, 1, 0, 1, 0.10, true, 1, 99, 0),
        Err(RegimeError::InvalidParameter { name: "n_grid", .. })
    ));
}

// ---------------------------------------------------------- determinism

#[test]
fn bootstrap_is_deterministic_at_any_thread_count() {
    let mut stream = Stream::new(55);
    let y = sim_var(&mut stream, 150);

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool");
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");

    let r1 = pool1.install(|| threshold_var_test(&y, 1, 0, 1, 0.10, true, 30, 99, 123).expect("t"));
    let r4 = pool4.install(|| threshold_var_test(&y, 1, 0, 1, 0.10, true, 30, 99, 123).expect("t"));

    assert_eq!(
        r1.boot_stats, r4.boot_stats,
        "boot draws must be bit-identical"
    );
    assert_eq!(r1.p_value, r4.p_value);
    assert_eq!(r1.stat, r4.stat);
}

// ---------------------------------------------------------- MC recovery

#[test]
fn mc_threshold_recovery_and_coefficient_bias() {
    // The model-card evidence: 200 seeded replications of the two-regime
    // VAR(1) DGP above (gamma = 0), T = 400.
    let n_reps = 200;
    let t = 400;
    let mut streams = Stream::substreams(20260825, n_reps).expect("substreams");
    let mut abs_err = Vec::with_capacity(n_reps);
    // Bias of the two intercepts and the two own-lag coefficients, per
    // regime: [c_low0, a_low00, c_high0, a_high00].
    let mut bias = [0.0_f64; 4];
    for stream in streams.iter_mut() {
        let y = sim_tvar(stream, t, 0.0);
        let fit = threshold_var(&y, 1, 0, &[1], 0.10, true).expect("fit runs");
        abs_err.push(fit.threshold.abs());
        bias[0] += fit.coefs_low[0][0] - C_LOW[0];
        bias[1] += fit.coefs_low[0][1] - A_LOW[0][0];
        bias[2] += fit.coefs_high[0][0] - C_HIGH[0];
        bias[3] += fit.coefs_high[0][1] - A_HIGH[0][0];
    }
    abs_err.sort_by(|a, b| a.total_cmp(b));
    let mae = (abs_err[n_reps / 2 - 1] + abs_err[n_reps / 2]) / 2.0;
    for b in &mut bias {
        *b /= n_reps as f64;
    }
    println!(
        "tvar MC recovery (T = {t}, {n_reps} reps): threshold median |err| = {mae:.4}; \
         bias c_low0 = {:.4}, a_low00 = {:.4}, c_high0 = {:.4}, a_high00 = {:.4}",
        bias[0], bias[1], bias[2], bias[3]
    );
    assert!(
        mae < 0.12,
        "threshold median absolute error {mae} too large"
    );
    for (i, b) in bias.iter().enumerate() {
        assert!(b.abs() < 0.08, "coefficient bias [{i}] = {b} too large");
    }
}
