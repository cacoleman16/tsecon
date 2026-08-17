//! Property tests for the SETAR fit and the Hansen bootstrap linearity
//! test: statistical properties a golden transcription cannot prove.
//!
//! * Under a LINEAR null the bootstrap p-value rejects at approximately the
//!   nominal rate over seeds (a small seeded Monte Carlo).
//! * On a strongly separated SETAR the threshold lands near truth and the
//!   test rejects hard.
//! * Nested case: on linear data both regime fits track the plain AR OLS
//!   fit, and the reported per-regime OLS is exactly self-consistent with a
//!   direct OLS on the identified split.
//! * Scale/location equivariance of the threshold and coefficients; the
//!   sup-F statistic is scale-free.
//! * Degenerate input raises the documented errors.
//! * The bootstrap is bit-identical for a given seed at any thread count.
//! * MC recovery (the model-card evidence): over 200 seeded replications,
//!   the threshold's median absolute error and per-coefficient bias stay
//!   small (run with `--nocapture` to see the measured numbers).

use tsecon_bootstrap::WildWeights;
use tsecon_regime::{setar, setar_test, RegimeError};
use tsecon_rng::Stream;

// ------------------------------------------------------------ simulation

/// Two-regime SETAR(1) with delay 1: `y_t = c_j + phi_j y_{t-1} + e_t`,
/// regime by `y_{t-1} <= gamma`, standard normal innovations, 100 burn-in.
fn sim_setar1(
    stream: &mut Stream,
    t: usize,
    low: [f64; 2],
    high: [f64; 2],
    gamma: f64,
) -> Vec<f64> {
    let burn = 100;
    let mut y = vec![0.0_f64; t + burn + 1];
    for i in 1..y.len() {
        let c = if y[i - 1] <= gamma { low } else { high };
        y[i] = c[0] + c[1] * y[i - 1] + WildWeights::Normal.draw(stream);
    }
    y[(burn + 1)..].to_vec()
}

/// Linear AR(1) `y_t = phi y_{t-1} + e_t`, standard normal innovations.
fn sim_ar1(stream: &mut Stream, t: usize, phi: f64) -> Vec<f64> {
    let burn = 100;
    let mut y = vec![0.0_f64; t + burn + 1];
    for i in 1..y.len() {
        y[i] = phi * y[i - 1] + WildWeights::Normal.draw(stream);
    }
    y[(burn + 1)..].to_vec()
}

/// Direct OLS of `y` on `[1, y_{t-1}]` by 2x2 normal equations, returning
/// `[c, phi]` — an independent check implementation for the tests.
fn direct_ar1_ols(rows_x: &[f64], rows_y: &[f64]) -> [f64; 2] {
    let n = rows_y.len() as f64;
    let sx: f64 = rows_x.iter().sum();
    let sy: f64 = rows_y.iter().sum();
    let sxx: f64 = rows_x.iter().map(|&v| v * v).sum();
    let sxy: f64 = rows_x.iter().zip(rows_y).map(|(&a, &b)| a * b).sum();
    let det = n * sxx - sx * sx;
    [(sy * sxx - sx * sxy) / det, (n * sxy - sx * sy) / det]
}

// -------------------------------------------------------------- size MC

#[test]
fn null_rejection_rate_is_near_nominal() {
    // 200 linear-AR series x 199 bootstrap draws. With B = 199 the p-value
    // lattice is {1/200, ..., 200/200}, so exact nominal-level rejection is
    // possible; the bands below are generous (binomial 3-sigma around the
    // nominal rate is about +/- 0.05 at the 5% level with 200 draws).
    let n_series = 200;
    let mut streams = Stream::substreams(20260817, n_series).expect("substreams");
    let mut reject05 = 0usize;
    let mut reject10 = 0usize;
    let mut psum = 0.0_f64;
    for (i, stream) in streams.iter_mut().enumerate() {
        let y = sim_ar1(stream, 100, 0.5);
        let r = setar_test(&y, 1, 1, 0.15, true, 199, 90_000 + i as u64).expect("test runs");
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
    println!("null MC: reject@5% = {rate05}, reject@10% = {rate10}, mean p = {pmean}");
    assert!(
        (0.005..=0.115).contains(&rate05),
        "5% rejection rate {rate05} far from nominal"
    );
    assert!(
        (0.03..=0.19).contains(&rate10),
        "10% rejection rate {rate10} far from nominal"
    );
    // A roughly uniform p-value has mean ~0.5.
    assert!(
        (0.40..=0.60).contains(&pmean),
        "mean null p-value {pmean} far from 0.5"
    );
}

// ------------------------------------------------------------- power MC

#[test]
fn strong_setar_recovers_threshold_and_rejects() {
    let mut stream = Stream::new(4242);
    let y = sim_setar1(&mut stream, 500, [1.0, 0.6], [-1.0, 0.2], 0.0);
    let fit = setar(&y, 1, &[1], 0.15, true).expect("fit runs");
    assert!(
        fit.threshold.abs() < 0.25,
        "threshold {} not near the true 0",
        fit.threshold
    );
    assert!(
        (fit.coefs_low[0] - 1.0).abs() < 0.3,
        "c_low {}",
        fit.coefs_low[0]
    );
    assert!(
        (fit.coefs_low[1] - 0.6).abs() < 0.2,
        "phi_low {}",
        fit.coefs_low[1]
    );
    assert!(
        (fit.coefs_high[0] + 1.0).abs() < 0.3,
        "c_high {}",
        fit.coefs_high[0]
    );
    assert!(
        (fit.coefs_high[1] - 0.2).abs() < 0.2,
        "phi_high {}",
        fit.coefs_high[1]
    );

    let r = setar_test(&y, 1, 1, 0.15, true, 199, 7).expect("test runs");
    assert!(
        r.p_value <= 0.01,
        "test failed to reject on a strong SETAR (p = {})",
        r.p_value
    );
}

// ------------------------------------------------------------ nested case

#[test]
fn linear_data_regime_fits_track_the_pooled_ols() {
    // With no true threshold the two regime fits both estimate the SAME
    // linear AR, so on a long sample each regime's coefficients approach
    // the pooled OLS fit (and, exactly, SSR_setar <= SSR_linear by
    // construction — the split fit nests the pooled one).
    let mut stream = Stream::new(99);
    let y = sim_ar1(&mut stream, 3000, 0.5);
    let fit = setar(&y, 1, &[1], 0.15, true).expect("fit runs");

    let x: Vec<f64> = y[..y.len() - 1].to_vec();
    let resp: Vec<f64> = y[1..].to_vec();
    let pooled = direct_ar1_ols(&x, &resp);
    for (j, &pj) in pooled.iter().enumerate() {
        assert!(
            (fit.coefs_low[j] - pj).abs() < 0.12,
            "low regime coef {j}: {} vs pooled {pj}",
            fit.coefs_low[j]
        );
        assert!(
            (fit.coefs_high[j] - pj).abs() < 0.12,
            "high regime coef {j}: {} vs pooled {pj}",
            fit.coefs_high[j]
        );
    }

    let r = setar_test(&y, 1, 1, 0.15, true, 99, 5).expect("test runs");
    assert!(
        r.ssr_setar <= r.ssr_linear,
        "split SSR must nest the pooled SSR"
    );
}

#[test]
fn reported_regime_fits_are_exactly_direct_ols_on_the_split() {
    // Self-consistency at fixture precision: recompute each regime's OLS
    // directly on the split the estimator reports.
    let mut stream = Stream::new(3);
    let y = sim_setar1(&mut stream, 300, [1.0, 0.6], [-1.0, 0.2], 0.0);
    let fit = setar(&y, 1, &[1], 0.15, true).expect("fit runs");

    let n = y.len();
    let mut lo_x = Vec::new();
    let mut lo_y = Vec::new();
    let mut hi_x = Vec::new();
    let mut hi_y = Vec::new();
    for t in 1..n {
        if y[t - 1] <= fit.threshold {
            lo_x.push(y[t - 1]);
            lo_y.push(y[t]);
        } else {
            hi_x.push(y[t - 1]);
            hi_y.push(y[t]);
        }
    }
    assert_eq!(lo_y.len(), fit.n_low);
    assert_eq!(hi_y.len(), fit.n_high);
    let bl = direct_ar1_ols(&lo_x, &lo_y);
    let bh = direct_ar1_ols(&hi_x, &hi_y);
    for j in 0..2 {
        assert!(
            (fit.coefs_low[j] - bl[j]).abs() < 1e-8,
            "low coef {j}: {} vs direct {}",
            fit.coefs_low[j],
            bl[j]
        );
        assert!(
            (fit.coefs_high[j] - bh[j]).abs() < 1e-8,
            "high coef {j}: {} vs direct {}",
            fit.coefs_high[j],
            bh[j]
        );
    }
    // The reported SSR is the minimum of the reported profile.
    let path_min = fit.ssr_path.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        ((fit.ssr - path_min) / path_min).abs() < 1e-10,
        "refit SSR {} vs scan minimum {}",
        fit.ssr,
        path_min
    );
}

// --------------------------------------------------------- equivariance

#[test]
fn scale_and_location_equivariance() {
    let mut stream = Stream::new(17);
    let y = sim_setar1(&mut stream, 400, [1.0, 0.6], [-1.0, 0.2], 0.0);
    let (a, b) = (2.5_f64, 3.0_f64);
    let yt: Vec<f64> = y.iter().map(|&v| a * v + b).collect();

    let f0 = setar(&y, 1, &[1], 0.15, true).expect("fit runs");
    let f1 = setar(&yt, 1, &[1], 0.15, true).expect("fit runs");

    let rel = |x: f64, e: f64| ((x - e) / e).abs();
    // gamma' = a gamma + b (order statistics are affine-equivariant).
    assert!(
        rel(f1.threshold, a * f0.threshold + b) < 1e-8,
        "threshold {} vs {}",
        f1.threshold,
        a * f0.threshold + b
    );
    assert_eq!(f1.n_low, f0.n_low);
    assert_eq!(f1.n_high, f0.n_high);
    // Slopes are invariant, intercepts map as c' = a c + b (1 - phi).
    for (c1, c0) in [
        (&f1.coefs_low, &f0.coefs_low),
        (&f1.coefs_high, &f0.coefs_high),
    ] {
        assert!(rel(c1[1], c0[1]) < 1e-8, "slope {} vs {}", c1[1], c0[1]);
        let expect_c = a * c0[0] + b * (1.0 - c0[1]);
        assert!(
            rel(c1[0], expect_c) < 1e-8,
            "const {} vs {}",
            c1[0],
            expect_c
        );
    }
    // Slope SEs invariant; SSR and variances scale by a^2.
    assert!(rel(f1.se_low[1], f0.se_low[1]) < 1e-8);
    assert!(rel(f1.se_high[1], f0.se_high[1]) < 1e-8);
    assert!(rel(f1.ssr, a * a * f0.ssr) < 1e-8);
    assert!(rel(f1.sigma2, a * a * f0.sigma2) < 1e-8);
    assert!(rel(f1.sigma2_low, a * a * f0.sigma2_low) < 1e-8);
    assert!(rel(f1.sigma2_high, a * a * f0.sigma2_high) < 1e-8);

    // The sup-F statistic is a ratio of SSRs: scale- and location-free.
    let t0 = setar_test(&y, 1, 1, 0.15, true, 99, 11).expect("test runs");
    let t1 = setar_test(&yt, 1, 1, 0.15, true, 99, 11).expect("test runs");
    assert!(
        rel(t1.stat, t0.stat) < 1e-8,
        "supF {} vs {}",
        t1.stat,
        t0.stat
    );
    // Same seed, scale-free statistic: the bootstrap p-value moves at most
    // one lattice step (floating-point could flip an exact-tie comparison).
    assert!(
        (t1.p_value - t0.p_value).abs() <= 1.0 / 100.0 + 1e-12,
        "p {} vs {}",
        t1.p_value,
        t0.p_value
    );
}

// ------------------------------------------------------------ degeneracy

#[test]
fn degenerate_inputs_raise_teaching_errors() {
    let mut stream = Stream::new(1);
    let y = sim_ar1(&mut stream, 100, 0.4);

    // Constant series.
    let c = vec![1.0; 100];
    assert!(matches!(
        setar(&c, 1, &[1], 0.15, true),
        Err(RegimeError::InvalidSpec { .. })
    ));

    // Too few observations for two trimmed regimes.
    assert!(matches!(
        setar(&y[..4], 1, &[1], 0.15, true),
        Err(RegimeError::InsufficientData { .. })
    ));

    // Empty series.
    assert!(matches!(
        setar(&[], 1, &[1], 0.15, true),
        Err(RegimeError::InsufficientData { .. })
    ));

    // trim outside (0, 0.5).
    assert!(matches!(
        setar(&y, 1, &[1], 0.5, true),
        Err(RegimeError::InvalidParameter { name: "trim", .. })
    ));
    assert!(matches!(
        setar(&y, 1, &[1], 0.0, true),
        Err(RegimeError::InvalidParameter { name: "trim", .. })
    ));

    // delay 0, p 0, empty delay list.
    assert!(matches!(
        setar(&y, 1, &[0], 0.15, true),
        Err(RegimeError::InvalidParameter { name: "delay", .. })
    ));
    assert!(matches!(
        setar(&y, 0, &[1], 0.15, true),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        setar(&y, 1, &[], 0.15, true),
        Err(RegimeError::InvalidSpec { .. })
    ));

    // NaN observation.
    let mut bad = y.clone();
    bad[10] = f64::NAN;
    assert!(matches!(
        setar(&bad, 1, &[1], 0.15, true),
        Err(RegimeError::NonFinite { .. })
    ));

    // n_boot = 0.
    assert!(matches!(
        setar_test(&y, 1, 1, 0.15, true, 0, 0),
        Err(RegimeError::InvalidParameter { name: "n_boot", .. })
    ));
}

// ---------------------------------------------------------- determinism

#[test]
fn bootstrap_is_deterministic_at_any_thread_count() {
    let mut stream = Stream::new(55);
    let y = sim_ar1(&mut stream, 150, 0.3);

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool");
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");

    let r1 = pool1.install(|| setar_test(&y, 1, 1, 0.15, true, 199, 123).expect("test"));
    let r4 = pool4.install(|| setar_test(&y, 1, 1, 0.15, true, 199, 123).expect("test"));

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
    // DGP y_t = (1.0 + 0.6 y_{t-1}) 1{y_{t-1} <= 0} +
    //           (-1.0 + 0.2 y_{t-1}) 1{y_{t-1} > 0} + e_t, T = 400.
    let n_reps = 200;
    let t = 400;
    let truth_low = [1.0, 0.6];
    let truth_high = [-1.0, 0.2];
    let mut streams = Stream::substreams(20260816, n_reps).expect("substreams");
    let mut abs_err = Vec::with_capacity(n_reps);
    let mut bias = [0.0_f64; 4]; // c_low, phi_low, c_high, phi_high
    for stream in streams.iter_mut() {
        let y = sim_setar1(stream, t, truth_low, truth_high, 0.0);
        let fit = setar(&y, 1, &[1], 0.15, true).expect("fit runs");
        abs_err.push(fit.threshold.abs());
        bias[0] += fit.coefs_low[0] - truth_low[0];
        bias[1] += fit.coefs_low[1] - truth_low[1];
        bias[2] += fit.coefs_high[0] - truth_high[0];
        bias[3] += fit.coefs_high[1] - truth_high[1];
    }
    abs_err.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let mae = (abs_err[n_reps / 2 - 1] + abs_err[n_reps / 2]) / 2.0;
    for b in &mut bias {
        *b /= n_reps as f64;
    }
    println!(
        "MC recovery (T = {t}, {n_reps} reps): threshold median |err| = {mae:.4}; \
         bias c_low = {:.4}, phi_low = {:.4}, c_high = {:.4}, phi_high = {:.4}",
        bias[0], bias[1], bias[2], bias[3]
    );
    assert!(
        mae < 0.10,
        "threshold median absolute error {mae} too large"
    );
    for (i, b) in bias.iter().enumerate() {
        assert!(b.abs() < 0.06, "coefficient bias [{i}] = {b} too large");
    }
}
