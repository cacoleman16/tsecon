//! Property tests for the Hansen (1997/2000) threshold confidence set:
//! the statistical and structural claims a golden transcription cannot
//! make.
//!
//! * (a) The set always contains `gamma_hat`, where `LR_n` is exactly 0.
//! * (b) Sets are nested in the level: 80% within 90% within 95% within
//!   99%, pointwise on the grid and as hulls.
//! * (c) **Measured coverage** (the model-card evidence): over 500 seeded
//!   replications per cell, the fraction of 90% and 95% sets that contain
//!   the true threshold at n = 100, 250, 500 — on a threshold-regression
//!   design after Hansen (2000, §5) through `threshold_regression_ci`, and
//!   on a SETAR(2) through `setar_threshold_ci`; plus the
//!   heteroskedasticity correction's identification rate, noise, and
//!   effect on a design whose error variance is keyed to the threshold
//!   variable. Run with `--nocapture`
//!   to see the numbers. Hansen's theory: the set is asymptotically exact
//!   under the shrinking-effect frame and conservative under a fixed
//!   effect; the assertions below are bands around what was measured,
//!   with the small-sample under-coverage stated where it occurs.
//! * (d) Invariance to the scale and location of `y` where the
//!   construction implies it: the LR profile, the set membership, `eta^2`
//!   and the p-value are affine-invariant; interval endpoints map
//!   affinely; slope-coefficient unions are invariant.
//! * The general `threshold_regression_ci` fed the SETAR design by hand
//!   reproduces `setar_threshold_ci` bit for bit.
//! * Degenerate input raises the documented errors; results are
//!   deterministic (no randomness anywhere in the construction).

use tsecon_bootstrap::WildWeights;
use tsecon_regime::{
    hansen_lr_critical_value, setar, setar_threshold_ci, threshold_regression_ci, RegimeError,
    ThresholdCiOptions,
};
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

/// Two-regime SETAR(2) with delay 1: `y_t = c_j + a_j y_{t-1} + b_j y_{t-2}
/// + e_t`, regime by `y_{t-1} <= gamma`, standard normal innovations.
fn sim_setar2(
    stream: &mut Stream,
    t: usize,
    low: [f64; 3],
    high: [f64; 3],
    gamma: f64,
) -> Vec<f64> {
    let burn = 100;
    let mut y = vec![0.0_f64; t + burn + 2];
    for i in 2..y.len() {
        let c = if y[i - 1] <= gamma { low } else { high };
        y[i] = c[0] + c[1] * y[i - 1] + c[2] * y[i - 2] + WildWeights::Normal.draw(stream);
    }
    y[(burn + 2)..].to_vec()
}

/// A threshold-regression design after Hansen (2000, §5):
/// `y_i = theta_1' x_i 1{q_i <= 2} + theta_2' x_i 1{q_i > 2} + e_i` with
/// `x_i = (1, z_i)'`, `z_i ~ N(0, 1)`, `q_i ~ N(2, 1)` (so the true
/// threshold `gamma_0 = 2` is the median of the threshold variable),
/// `theta_1 = (0, 0)`, `theta_2 = (delta, delta)`, and `e_i ~ N(0, 1)`
/// — or, with `het`, `e_i = eps_i exp((q_i - 2) / 2)`, whose variance
/// `exp(q_i - 2)` rises with the threshold variable (it is 1 at
/// `gamma_0` against a pooled `e^{1/2}`), so Hansen's `eta^2` differs
/// from one.
fn sim_hansen_design(
    stream: &mut Stream,
    n: usize,
    delta: f64,
    het: bool,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let mut y = Vec::with_capacity(n);
    let mut ones = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);
    for _ in 0..n {
        let zi = WildWeights::Normal.draw(stream);
        let qi = 2.0 + WildWeights::Normal.draw(stream);
        let mut ei = WildWeights::Normal.draw(stream);
        if het {
            ei *= ((qi - 2.0) / 2.0).exp();
        }
        let mean = if qi <= 2.0 { 0.0 } else { delta * (1.0 + zi) };
        y.push(mean + ei);
        ones.push(1.0);
        z.push(zi);
        q.push(qi);
    }
    (y, vec![ones, z], q)
}

fn opts(level: f64) -> ThresholdCiOptions {
    ThresholdCiOptions {
        level,
        ..ThresholdCiOptions::default()
    }
}

// ----------------------------------------------------- (a) contains gamma^

#[test]
fn set_contains_the_estimate_where_lr_is_exactly_zero() {
    let mut streams = Stream::substreams(20260901, 20).expect("substreams");
    for stream in streams.iter_mut() {
        let y = sim_setar1(stream, 200, [1.0, 0.6], [-1.0, 0.2], 0.0);
        for level in [0.5, 0.8, 0.95] {
            let r = setar_threshold_ci(&y, 1, &[1], 0.15, true, &opts(level)).expect("runs");
            let best = r
                .thresholds
                .iter()
                .position(|&g| g == r.threshold)
                .expect("estimate on the grid");
            assert_eq!(r.lr_stat[best], 0.0, "LR at gamma^ is exactly zero");
            assert!(
                r.lr_stat.iter().all(|&v| v >= 0.0),
                "LR profile is nonnegative"
            );
            assert!(r.in_set[best], "gamma^ is in the set");
            assert!(r.n_in_set >= 1);
            assert!(r.ci_low <= r.threshold && r.threshold <= r.ci_high);
            assert!(
                r.intervals
                    .iter()
                    .any(|&(lo, hi)| lo <= r.threshold && r.threshold <= hi),
                "some interval contains gamma^"
            );
            assert_eq!(r.is_connected, r.intervals.len() == 1);
            // Intervals are ascending, disjoint, and sit on the grid.
            for w in r.intervals.windows(2) {
                assert!(w[0].1 < w[1].0, "intervals ascending and disjoint");
            }
            for &(lo, hi) in &r.intervals {
                assert!(r.thresholds.contains(&lo) && r.thresholds.contains(&hi));
            }
            assert_eq!(r.n_in_set, r.in_set.iter().filter(|&&b| b).count());
        }
    }
}

// ------------------------------------------------------------ (b) nesting

#[test]
fn sets_are_nested_in_the_level() {
    let mut streams = Stream::substreams(20260902, 12).expect("substreams");
    for (i, stream) in streams.iter_mut().enumerate() {
        // Alternate a sharp SETAR and a linear AR (wide, ragged sets).
        let y = if i % 2 == 0 {
            sim_setar1(stream, 150, [1.0, 0.6], [-1.0, 0.2], 0.0)
        } else {
            sim_setar1(stream, 150, [0.0, 0.5], [0.0, 0.5], 0.0)
        };
        let mut prev: Option<tsecon_regime::ThresholdCi> = None;
        for level in [0.80, 0.90, 0.95, 0.99] {
            let r = setar_threshold_ci(&y, 1, &[1], 0.15, true, &opts(level)).expect("runs");
            if let Some(p) = &prev {
                assert!(r.lr_crit > p.lr_crit, "critical value grows with the level");
                assert_eq!(
                    r.lr_stat, p.lr_stat,
                    "the profile does not depend on the level"
                );
                for (a, b) in p.in_set.iter().zip(&r.in_set) {
                    assert!(
                        !a || *b,
                        "a lower-level member stays in the higher-level set"
                    );
                }
                assert!(r.n_in_set >= p.n_in_set);
                assert!(r.ci_low <= p.ci_low && r.ci_high >= p.ci_high, "hulls nest");
            }
            prev = Some(r);
        }
    }
}

// ----------------------------------------------------------- (c) coverage

/// Coverage of the true threshold at the 90% and 95% levels from one
/// 95%-level call per replication: `gamma_0` is covered at level `L` iff
/// `LR_n(gamma_0) / eta^2 <= c(L)`. A `gamma_0` outside the trimmed grid
/// (the set cannot contain it) counts as not covered.
struct Coverage {
    c90: f64,
    c95: f64,
}

fn coverage_from(results: &[Option<(f64, f64)>]) -> Coverage {
    let c90 = hansen_lr_critical_value(0.90).expect("level");
    let c95 = hansen_lr_critical_value(0.95).expect("level");
    let n = results.len() as f64;
    let count = |c: f64| {
        results
            .iter()
            .filter(|r| matches!(r, Some((lr0, eta2)) if lr0 / eta2 <= c))
            .count() as f64
            / n
    };
    Coverage {
        c90: count(c90),
        c95: count(c95),
    }
}

#[test]
fn coverage_on_a_hansen_2000_style_threshold_regression() {
    let n_reps = 500;
    let mut table = Vec::new();
    for &delta in &[0.5_f64, 1.0] {
        for &n in &[100_usize, 250, 500] {
            let mut streams = Stream::substreams(20260903 + n as u64, n_reps).expect("substreams");
            let mut results = Vec::with_capacity(n_reps);
            for stream in streams.iter_mut() {
                let (y, x, q) = sim_hansen_design(stream, n, delta, false);
                let o = ThresholdCiOptions {
                    level: 0.95,
                    null_threshold: Some(2.0),
                    ..ThresholdCiOptions::default()
                };
                results.push(
                    threshold_regression_ci(&y, &x, &q, 0.15, &o)
                        .ok()
                        .map(|r| (r.lr_at_null.expect("null"), r.eta2)),
                );
            }
            let cov = coverage_from(&results);
            println!(
                "Hansen-design coverage (delta = {delta}, n = {n}, {n_reps} reps): \
                 90% set {:.3}, 95% set {:.3}",
                cov.c90, cov.c95
            );
            table.push((delta, n, cov));
        }
    }
    // Bands: binomial 3-sigma at 500 reps is about +/- 0.04 at 90% and
    // +/- 0.03 at 95%. The construction is asymptotically conservative for
    // a fixed effect, so the measured rates sit at or above nominal from
    // n = 250 on; at n = 100 with the smaller effect they may dip below.
    for (delta, n, cov) in &table {
        if *n >= 250 {
            assert!(
                cov.c90 >= 0.86,
                "90% coverage {} at delta {delta}, n {n} far below nominal",
                cov.c90
            );
            assert!(
                cov.c95 >= 0.92,
                "95% coverage {} at delta {delta}, n {n} far below nominal",
                cov.c95
            );
        } else {
            assert!(
                cov.c90 >= 0.80,
                "90% coverage {} at n = 100 collapsed",
                cov.c90
            );
            assert!(
                cov.c95 >= 0.86,
                "95% coverage {} at n = 100 collapsed",
                cov.c95
            );
        }
        assert!(
            cov.c95 >= cov.c90,
            "95% set covers at least as often as the 90% set"
        );
    }
}

#[test]
fn coverage_on_a_setar2_dgp() {
    // y_t = (1.0 + 0.5 y_{t-1} + 0.2 y_{t-2}) 1{y_{t-1} <= 0} +
    //       (-1.0 + 0.3 y_{t-1} - 0.2 y_{t-2}) 1{y_{t-1} > 0} + e_t.
    let n_reps = 500;
    let low = [1.0, 0.5, 0.2];
    let high = [-1.0, 0.3, -0.2];
    let mut table = Vec::new();
    for &n in &[100_usize, 250, 500] {
        let mut streams = Stream::substreams(20260904 + n as u64, n_reps).expect("substreams");
        let mut results = Vec::with_capacity(n_reps);
        for stream in streams.iter_mut() {
            let y = sim_setar2(stream, n, low, high, 0.0);
            let o = ThresholdCiOptions {
                level: 0.95,
                null_threshold: Some(0.0),
                ..ThresholdCiOptions::default()
            };
            results.push(
                setar_threshold_ci(&y, 2, &[1], 0.15, true, &o)
                    .ok()
                    .map(|r| (r.lr_at_null.expect("null"), r.eta2)),
            );
        }
        let cov = coverage_from(&results);
        println!(
            "SETAR(2) coverage (n = {n}, {n_reps} reps): 90% set {:.3}, 95% set {:.3}",
            cov.c90, cov.c95
        );
        table.push((n, cov));
    }
    for (n, cov) in &table {
        if *n >= 250 {
            assert!(
                cov.c90 >= 0.86,
                "90% coverage {} at n {n} far below nominal",
                cov.c90
            );
            assert!(
                cov.c95 >= 0.92,
                "95% coverage {} at n {n} far below nominal",
                cov.c95
            );
        } else {
            assert!(
                cov.c90 >= 0.80,
                "90% coverage {} at n = 100 collapsed",
                cov.c90
            );
            assert!(
                cov.c95 >= 0.86,
                "95% coverage {} at n = 100 collapsed",
                cov.c95
            );
        }
        assert!(cov.c95 >= cov.c90);
    }
}

/// Coverage conditional on the correction being identified (a `None`
/// result is the documented refusal), with the identified count.
fn coverage_conditional(results: &[Option<(f64, f64)>]) -> (Coverage, usize) {
    let identified: Vec<Option<(f64, f64)>> =
        results.iter().filter(|r| r.is_some()).cloned().collect();
    (coverage_from(&identified), identified.len())
}

#[test]
fn heteroskedasticity_correction_rescales_the_set_and_is_noisy() {
    // Error variance keyed to the threshold variable (exp(q - 2), rising
    // in q): at gamma_0 it is 1 against a pooled sigma^2 = e^{1/2}, so the
    // true eta^2 = e^{-1/2} = 0.607 and the uncorrected set — built on the
    // pooled scale — is too WIDE. Hansen's eta^2 rescales it; measured
    // here: how often the quadratic-regression estimate is identified,
    // how noisy it is, and what the corrected set then covers.
    let n_reps = 500;
    let n = 250;
    let delta = 1.0;
    let mut plain = Vec::with_capacity(n_reps);
    let mut robust = Vec::with_capacity(n_reps);
    let mut eta2s = Vec::new();
    let mut streams = Stream::substreams(20260905, n_reps).expect("substreams");
    for stream in streams.iter_mut() {
        let (y, x, q) = sim_hansen_design(stream, n, delta, true);
        let o = ThresholdCiOptions {
            level: 0.95,
            null_threshold: Some(2.0),
            ..ThresholdCiOptions::default()
        };
        plain.push(
            threshold_regression_ci(&y, &x, &q, 0.15, &o)
                .ok()
                .map(|r| (r.lr_at_null.expect("null"), r.eta2)),
        );
        let oh = ThresholdCiOptions {
            het_robust: true,
            ..o
        };
        let rh = threshold_regression_ci(&y, &x, &q, 0.15, &oh).ok();
        if let Some(r) = &rh {
            eta2s.push(r.eta2);
        }
        robust.push(rh.map(|r| (r.lr_at_null.expect("null"), r.eta2)));
    }
    let cp = coverage_from(&plain);
    let (cr_cond, n_ident) = coverage_conditional(&robust);
    let cr_all = coverage_from(&robust);
    eta2s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let mean_eta2 = eta2s.iter().sum::<f64>() / eta2s.len() as f64;
    let q10 = eta2s[eta2s.len() / 10];
    let q50 = eta2s[eta2s.len() / 2];
    let q90 = eta2s[eta2s.len() * 9 / 10];
    println!(
        "heteroskedastic design (n = {n}, delta = {delta}, {n_reps} reps, true eta^2 = \
         {:.3}): plain 90%/95% = {:.3}/{:.3}; het_robust identified in {n_ident} reps, \
         eta^2 mean {mean_eta2:.3} (deciles {q10:.3} / {q50:.3} / {q90:.3}); robust \
         90%/95% = {:.3}/{:.3} conditional on identification, {:.3}/{:.3} counting \
         refusals as misses",
        (-0.5_f64).exp(),
        cp.c90,
        cp.c95,
        cr_cond.c90,
        cr_cond.c95,
        cr_all.c90,
        cr_all.c95
    );
    // The plain set over-covers (its scale is the pooled variance).
    assert!(
        cp.c90 >= 0.95 && cp.c95 >= 0.97,
        "plain set should over-cover here"
    );
    // The correction is identified in the large majority of replications
    // and centered near the truth, but noisy.
    assert!(
        n_ident >= n_reps * 9 / 10,
        "eta^2 identified in {n_ident} of {n_reps}"
    );
    assert!(
        (mean_eta2 - (-0.5_f64).exp()).abs() < 0.15,
        "mean eta^2 {mean_eta2}"
    );
    assert!(q50 < 1.0, "median eta^2 below one on this design");
    // The corrected set is narrower (never covers more often) and, given
    // identification, sits near nominal — the measured 0.92 / 0.94 is a
    // few points under, the price of the noisy scale estimate.
    assert!(cr_cond.c95 <= cp.c95 + 1e-12);
    assert!(
        cr_cond.c90 >= 0.86,
        "robust 90% coverage {} collapsed",
        cr_cond.c90
    );
    assert!(
        cr_cond.c95 >= 0.90,
        "robust 95% coverage {} collapsed",
        cr_cond.c95
    );
}

// ---------------------------------------------------------- (d) invariance

#[test]
fn affine_transformations_of_y_leave_the_construction_invariant() {
    let mut stream = Stream::new(20260906);
    let y = sim_setar1(&mut stream, 300, [1.0, 0.6], [-1.0, 0.2], 0.0);
    let (a, b) = (2.5_f64, 3.0_f64);
    let yt: Vec<f64> = y.iter().map(|&v| a * v + b).collect();
    let o = ThresholdCiOptions {
        level: 0.90,
        het_robust: true,
        slope_level: Some(0.95),
        slope_region_level: 0.80,
        null_threshold: Some(0.05),
    };
    let ot = ThresholdCiOptions {
        null_threshold: Some(a * 0.05 + b),
        ..o.clone()
    };
    let r0 = setar_threshold_ci(&y, 1, &[1], 0.15, true, &o).expect("runs");
    let r1 = setar_threshold_ci(&yt, 1, &[1], 0.15, true, &ot).expect("runs");
    let rel = |x: f64, e: f64| {
        if e == 0.0 {
            x.abs()
        } else {
            ((x - e) / e).abs()
        }
    };

    assert_eq!(r0.thresholds.len(), r1.thresholds.len());
    for (g0, g1) in r0.thresholds.iter().zip(&r1.thresholds) {
        assert!(rel(*g1, a * g0 + b) < 1e-10, "grid maps affinely");
    }
    for (l0, l1) in r0.lr_stat.iter().zip(&r1.lr_stat) {
        assert!(
            rel(*l1, *l0) < 1e-8,
            "LR profile is affine-invariant: {l0} vs {l1}"
        );
    }
    assert_eq!(r0.in_set, r1.in_set, "set membership is affine-invariant");
    assert_eq!(r0.intervals.len(), r1.intervals.len());
    for (i0, i1) in r0.intervals.iter().zip(&r1.intervals) {
        assert!(rel(i1.0, a * i0.0 + b) < 1e-10);
        assert!(rel(i1.1, a * i0.1 + b) < 1e-10);
    }
    assert!(rel(r1.threshold, a * r0.threshold + b) < 1e-10);
    assert!(
        rel(r1.eta2, r0.eta2) < 1e-8,
        "eta^2 is affine-invariant: {} vs {}",
        r0.eta2,
        r1.eta2
    );
    assert!(rel(r1.lr_at_null.unwrap(), r0.lr_at_null.unwrap()) < 1e-8);
    assert!(
        rel(
            r1.pvalue_at_threshold.unwrap(),
            r0.pvalue_at_threshold.unwrap()
        ) < 1e-8
    );
    let s0 = r0.slope.as_ref().unwrap();
    let s1 = r1.slope.as_ref().unwrap();
    assert_eq!(s0.n_region, s1.n_region);
    // Lag-coefficient (slope) unions are invariant; the constant's map
    // depends on gamma through the slope, so only the slope is checked.
    assert!(rel(s1.low_lower[1], s0.low_lower[1]) < 1e-8);
    assert!(rel(s1.low_upper[1], s0.low_upper[1]) < 1e-8);
    assert!(rel(s1.high_lower[1], s0.high_lower[1]) < 1e-8);
    assert!(rel(s1.high_upper[1], s0.high_upper[1]) < 1e-8);

    // Pure scale (b = 0): the constant's union scales by a.
    let ys: Vec<f64> = y.iter().map(|&v| a * v).collect();
    let os = ThresholdCiOptions {
        null_threshold: Some(a * 0.05),
        ..o
    };
    let r2 = setar_threshold_ci(&ys, 1, &[1], 0.15, true, &os).expect("runs");
    let s2 = r2.slope.as_ref().unwrap();
    assert!(rel(s2.low_lower[0], a * s0.low_lower[0]) < 1e-8);
    assert!(rel(s2.low_upper[0], a * s0.low_upper[0]) < 1e-8);
    assert!(rel(s2.high_lower[0], a * s0.high_lower[0]) < 1e-8);
    assert!(rel(s2.high_upper[0], a * s0.high_upper[0]) < 1e-8);
}

// ------------------------------------------------- generic == SETAR wrapper

#[test]
fn general_threshold_regression_on_the_setar_design_matches_the_wrapper_bit_for_bit() {
    let mut stream = Stream::new(20260907);
    let y = sim_setar1(&mut stream, 250, [1.0, 0.6], [-1.0, 0.2], 0.0);
    let o = ThresholdCiOptions {
        level: 0.90,
        het_robust: true,
        slope_level: Some(0.90),
        slope_region_level: 0.80,
        null_threshold: Some(0.1),
    };
    let rs = setar_threshold_ci(&y, 1, &[1], 0.15, true, &o).expect("runs");
    // The SETAR(1) design by hand: response y_t, columns [1, y_{t-1}],
    // threshold variable y_{t-1}, t = 1..T-1.
    let resp: Vec<f64> = y[1..].to_vec();
    let lag: Vec<f64> = y[..y.len() - 1].to_vec();
    let x = vec![vec![1.0; resp.len()], lag.clone()];
    let rg = threshold_regression_ci(&resp, &x, &lag, 0.15, &o).expect("runs");
    assert_eq!(rg.delay, None);
    assert_eq!(rs.delay, Some(1));
    assert_eq!(rs.threshold.to_bits(), rg.threshold.to_bits());
    assert_eq!(rs.thresholds, rg.thresholds);
    assert_eq!(rs.ssr_path, rg.ssr_path);
    assert_eq!(rs.lr_stat, rg.lr_stat);
    assert_eq!(rs.in_set, rg.in_set);
    assert_eq!(rs.intervals, rg.intervals);
    assert_eq!(rs.eta2.to_bits(), rg.eta2.to_bits());
    assert_eq!(rs.pvalue_at_threshold, rg.pvalue_at_threshold);
    assert_eq!(rs.slope, rg.slope);
    assert_eq!(rs.nobs, rg.nobs);
    assert_eq!(rs.k, rg.k);

    // And both sit on the plain fit.
    let fit = setar(&y, 1, &[1], 0.15, true).expect("fit");
    assert_eq!(fit.thresholds, rg.thresholds);
    assert_eq!(fit.ssr_path, rg.ssr_path);
}

// ----------------------------------------------------------- determinism

#[test]
fn construction_is_deterministic() {
    let mut stream = Stream::new(20260908);
    let y = sim_setar1(&mut stream, 200, [1.0, 0.6], [-1.0, 0.2], 0.0);
    let o = ThresholdCiOptions {
        het_robust: true,
        slope_level: Some(0.95),
        null_threshold: Some(0.0),
        ..ThresholdCiOptions::default()
    };
    let r1 = setar_threshold_ci(&y, 1, &[1, 2], 0.15, true, &o).expect("runs");
    let r2 = setar_threshold_ci(&y, 1, &[1, 2], 0.15, true, &o).expect("runs");
    assert_eq!(r1, r2);
}

// ------------------------------------------------------------ degeneracy

#[test]
fn degenerate_inputs_raise_teaching_errors() {
    let mut stream = Stream::new(20260909);
    let y = sim_setar1(&mut stream, 120, [1.0, 0.6], [-1.0, 0.2], 0.0);

    // Levels outside (0, 1), including NaN and the endpoints.
    for bad in [0.0, 1.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            setar_threshold_ci(&y, 1, &[1], 0.15, true, &opts(bad)),
            Err(RegimeError::InvalidParameter { name: "level", .. })
        ));
        assert!(matches!(
            hansen_lr_critical_value(bad),
            Err(RegimeError::InvalidParameter { name: "level", .. })
        ));
        let o = ThresholdCiOptions {
            slope_level: Some(bad),
            ..ThresholdCiOptions::default()
        };
        assert!(matches!(
            setar_threshold_ci(&y, 1, &[1], 0.15, true, &o),
            Err(RegimeError::InvalidParameter {
                name: "slope_level",
                ..
            })
        ));
        let o = ThresholdCiOptions {
            slope_level: Some(0.95),
            slope_region_level: bad,
            ..ThresholdCiOptions::default()
        };
        assert!(matches!(
            setar_threshold_ci(&y, 1, &[1], 0.15, true, &o),
            Err(RegimeError::InvalidParameter {
                name: "slope_region_level",
                ..
            })
        ));
    }
    // A bad region level is inert (and unchecked) without a slope level:
    // the Rust struct always carries a value; the Python surface refuses
    // it when passed explicitly.
    let o = ThresholdCiOptions {
        slope_level: None,
        slope_region_level: 7.0,
        ..ThresholdCiOptions::default()
    };
    assert!(setar_threshold_ci(&y, 1, &[1], 0.15, true, &o).is_ok());

    // Null threshold: NaN, and outside the trimmed grid.
    let o = ThresholdCiOptions {
        null_threshold: Some(f64::NAN),
        ..ThresholdCiOptions::default()
    };
    assert!(matches!(
        setar_threshold_ci(&y, 1, &[1], 0.15, true, &o),
        Err(RegimeError::NonFinite { .. })
    ));
    let fit = setar(&y, 1, &[1], 0.15, true).expect("fit");
    for g0 in [
        fit.thresholds[0] - 1.0,
        fit.thresholds[fit.thresholds.len() - 1] + 1.0,
    ] {
        let o = ThresholdCiOptions {
            null_threshold: Some(g0),
            ..ThresholdCiOptions::default()
        };
        assert!(matches!(
            setar_threshold_ci(&y, 1, &[1], 0.15, true, &o),
            Err(RegimeError::InvalidParameter {
                name: "null_threshold",
                ..
            })
        ));
    }
    // The grid endpoints themselves are admissible, and a null value
    // between two candidates uses the lower one (the step function).
    let g = &fit.thresholds;
    for g0 in [g[0], g[g.len() - 1], 0.5 * (g[3] + g[4])] {
        let o = ThresholdCiOptions {
            null_threshold: Some(g0),
            ..ThresholdCiOptions::default()
        };
        let r = setar_threshold_ci(&y, 1, &[1], 0.15, true, &o).expect("runs");
        let used = r.null_threshold_used.unwrap();
        assert!(used <= g0 && g.contains(&used));
        let p = r.pvalue_at_threshold.unwrap();
        assert!((0.0..=1.0).contains(&p));
    }

    // The p-value function refuses NaN and negative statistics.
    assert!(matches!(
        tsecon_regime::hansen_lr_pvalue(f64::NAN),
        Err(RegimeError::NonFinite { .. })
    ));
    assert!(matches!(
        tsecon_regime::hansen_lr_pvalue(-1.0),
        Err(RegimeError::InvalidParameter { name: "lr", .. })
    ));
    assert_eq!(tsecon_regime::hansen_lr_pvalue(0.0).unwrap(), 1.0);

    // The SETAR input errors pass through unchanged.
    assert!(matches!(
        setar_threshold_ci(&y, 0, &[1], 0.15, true, &opts(0.95)),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        setar_threshold_ci(&y, 1, &[], 0.15, true, &opts(0.95)),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        setar_threshold_ci(&y, 1, &[1], 0.5, true, &opts(0.95)),
        Err(RegimeError::InvalidParameter { name: "trim", .. })
    ));
    assert!(matches!(
        setar_threshold_ci(&y[..4], 1, &[1], 0.15, true, &opts(0.95)),
        Err(RegimeError::InsufficientData { .. })
    ));

    // The general regression's own checks.
    let n = 60;
    let resp: Vec<f64> = y[1..=n].to_vec();
    let lag: Vec<f64> = y[..n].to_vec();
    let x = vec![vec![1.0; n], lag.clone()];
    assert!(matches!(
        threshold_regression_ci(&resp, &[], &lag, 0.15, &opts(0.95)),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        threshold_regression_ci(&resp, &x, &lag[..n - 1], 0.15, &opts(0.95)),
        Err(RegimeError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        threshold_regression_ci(&resp, &[vec![1.0; n - 1]], &lag, 0.15, &opts(0.95)),
        Err(RegimeError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        threshold_regression_ci(&resp, &x, &vec![1.0; n], 0.15, &opts(0.95)),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        threshold_regression_ci(&resp, &x, &lag, 0.0, &opts(0.95)),
        Err(RegimeError::InvalidParameter { name: "trim", .. })
    ));
    let mut bad_q = lag.clone();
    bad_q[3] = f64::NAN;
    assert!(matches!(
        threshold_regression_ci(&resp, &x, &bad_q, 0.15, &opts(0.95)),
        Err(RegimeError::NonFinite { .. })
    ));
    // n = 5 cannot hold two regimes of k + 1 = 3 (n = 8 can, and runs).
    assert!(matches!(
        threshold_regression_ci(
            &resp[..5],
            &[vec![1.0; 5], lag[..5].to_vec()],
            &lag[..5],
            0.15,
            &opts(0.95)
        ),
        Err(RegimeError::InsufficientData { .. })
    ));
    assert!(threshold_regression_ci(
        &resp[..8],
        &[vec![1.0; 8], lag[..8].to_vec()],
        &lag[..8],
        0.15,
        &opts(0.95)
    )
    .is_ok());
}
