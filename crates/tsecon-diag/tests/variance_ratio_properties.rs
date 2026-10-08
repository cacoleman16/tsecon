//! Property, invariance, and Monte-Carlo tests for the Lo-MacKinlay
//! variance-ratio test and the Chow-Denning joint test.
//!
//! The golden file pins the numbers against arch; these seeded tests carry
//! the statistical and contractual claims:
//!
//! * invariances — affine maps `y -> a + b y` (both trends, `b < 0`
//!   included; bit-identical for power-of-two `b`), adding a linear drift
//!   under trend "c", the non-overlapping truncation (`n_dropped =
//!   (n - 1) mod q`, identical to calling on the pre-truncated series);
//! * the closed-form homoskedastic variance, exactly;
//! * the inert-keyword contract: `debiased`/`robust` with `overlap = false`
//!   are refused in the core too, so a direct Rust caller cannot reach the
//!   silently-ignored class;
//! * Chow-Denning identities — `m = 1` is the single test bit-for-bit, the
//!   critical values are then the two-sided normal quantiles, permutation
//!   invariance, `p_cd` in `[min p_i, 1]`, and the p-value / critical-value
//!   duality;
//! * Monte-Carlo size (iid and GARCH increments, robust vs homoskedastic)
//!   and power (AR(0.3) increments) — deterministic seeds, so pass/fail is
//!   fixed once;
//! * the refusal surface (errors that teach, never garbage numbers).

use tsecon_diag::{chow_denning, variance_ratio, DiagError, VarianceRatioSpec, VrSeType, VrTrend};
use tsecon_stats::simultaneous::pointwise_critical_value;

/// Deterministic LCG (Knuth MMIX), as in the other diag property suites,
/// with Box-Muller normals (exact Gaussian tails matter for size checks).
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn cumsum0(e: &[f64]) -> Vec<f64> {
    let mut y = Vec::with_capacity(e.len() + 1);
    let mut acc = 0.0;
    y.push(acc);
    for &v in e {
        acc += v;
        y.push(acc);
    }
    y
}

/// Random walk with iid N(0, 1) increments (n levels).
fn walk(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let e: Vec<f64> = (0..n - 1).map(|_| rng.normal()).collect();
    cumsum0(&e)
}

/// Random walk with GARCH(1, 1) increments (omega 0.05, alpha, beta).
fn garch_walk(rng: &mut Lcg, n: usize, alpha: f64, beta: f64) -> Vec<f64> {
    let omega = 0.05;
    let mut s2 = omega / (1.0 - alpha - beta);
    let e: Vec<f64> = (0..n - 1)
        .map(|_| {
            let v = s2.sqrt() * rng.normal();
            s2 = omega + alpha * v * v + beta * s2;
            v
        })
        .collect();
    cumsum0(&e)
}

/// Integrated AR(1): increments u_t = phi u_{t-1} + e_t.
fn ar_walk(rng: &mut Lcg, n: usize, phi: f64) -> Vec<f64> {
    let mut prev = 0.0;
    let e: Vec<f64> = (0..n - 1)
        .map(|_| {
            prev = phi * prev + rng.normal();
            prev
        })
        .collect();
    cumsum0(&e)
}

fn stat_close(a: f64, e: f64, ctx: &str) {
    let tol = 1e-12 + 1e-10 * e.abs();
    assert!((a - e).abs() <= tol, "{ctx}: {a} vs {e}");
}

fn all_specs() -> Vec<VarianceRatioSpec> {
    let mut out = Vec::new();
    for trend in [VrTrend::None, VrTrend::Drift] {
        for (debiased, robust) in [(true, true), (true, false), (false, true), (false, false)] {
            out.push(VarianceRatioSpec {
                trend,
                overlap: true,
                debiased,
                robust,
            });
        }
        out.push(VarianceRatioSpec::non_overlapping(trend));
    }
    out
}

// ------------------------------------------------------------- invariances

#[test]
fn affine_maps_leave_the_test_unchanged() {
    let mut rng = Lcg(11);
    let y = ar_walk(&mut rng, 301, 0.2);
    for spec in all_specs() {
        for q in [2, 3, 7, 12] {
            let base = variance_ratio(&y, q, spec).unwrap();
            for (a, b) in [(5.0, 3.0), (-2.5, -0.7), (1e3, 1e-3), (0.0, -1.0)] {
                let z: Vec<f64> = y.iter().map(|&v| a + b * v).collect();
                let r = variance_ratio(&z, q, spec).unwrap();
                let ctx = format!("{spec:?} q={q} a={a} b={b}");
                assert!(((r.vr - base.vr) / base.vr).abs() <= 1e-10, "{ctx} vr");
                stat_close(r.stat, base.stat, &ctx);
                assert_eq!(r.nobs, base.nobs);
            }
            // A power-of-two scale (no shift) is exact: the core rescales by
            // a power of two internally, so the arithmetic is identical.
            for b in [0.25, 8.0, -2.0, 2f64.powi(600), 2f64.powi(-600)] {
                let z: Vec<f64> = y.iter().map(|&v| b * v).collect();
                let r = variance_ratio(&z, q, spec).unwrap();
                assert_eq!(r.vr, base.vr, "{spec:?} q={q} b={b}");
                assert_eq!(r.stat, base.stat, "{spec:?} q={q} b={b}");
            }
        }
    }
}

#[test]
fn extreme_magnitudes_do_not_overflow() {
    // Levels near 1e300: the squared q-sums would overflow without the
    // internal power-of-two rescaling.
    let mut rng = Lcg(12);
    let y = walk(&mut rng, 200);
    let base = variance_ratio(&y, 4, VarianceRatioSpec::default()).unwrap();
    let big: Vec<f64> = y.iter().map(|&v| v * 1e300).collect();
    let r = variance_ratio(&big, 4, VarianceRatioSpec::default()).unwrap();
    assert!(r.vr.is_finite() && r.stat.is_finite());
    assert!(((r.vr - base.vr) / base.vr).abs() <= 1e-12);
    stat_close(r.stat, base.stat, "1e300");
    let tiny: Vec<f64> = y.iter().map(|&v| v * 1e-300).collect();
    let r = variance_ratio(&tiny, 4, VarianceRatioSpec::default()).unwrap();
    assert!(((r.vr - base.vr) / base.vr).abs() <= 1e-12);
}

#[test]
fn drift_is_removed_under_trend_c() {
    let mut rng = Lcg(13);
    let y = walk(&mut rng, 257);
    for spec in all_specs()
        .into_iter()
        .filter(|s| s.trend == VrTrend::Drift)
    {
        for q in [2, 5, 9] {
            let base = variance_ratio(&y, q, spec).unwrap();
            let z: Vec<f64> = y
                .iter()
                .enumerate()
                .map(|(t, &v)| v + 0.37 * t as f64)
                .collect();
            let r = variance_ratio(&z, q, spec).unwrap();
            assert!(
                ((r.vr - base.vr) / base.vr).abs() <= 1e-10,
                "{spec:?} q={q}"
            );
            stat_close(r.stat, base.stat, &format!("{spec:?} q={q}"));
        }
    }
    // ... and NOT under trend "n", where a drift is a departure from the
    // null that inflates the long-horizon variance.
    let spec = VarianceRatioSpec {
        trend: VrTrend::None,
        ..VarianceRatioSpec::default()
    };
    let z: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(t, &v)| v + 0.5 * t as f64)
        .collect();
    assert!(variance_ratio(&z, 4, spec).unwrap().vr > variance_ratio(&y, 4, spec).unwrap().vr);
}

#[test]
fn homoskedastic_variance_is_the_closed_form() {
    let mut rng = Lcg(14);
    let y = walk(&mut rng, 400);
    for q in 2..=30usize {
        for debiased in [true, false] {
            let spec = VarianceRatioSpec {
                debiased,
                robust: false,
                ..VarianceRatioSpec::default()
            };
            let r = variance_ratio(&y, q, spec).unwrap();
            let qf = q as f64;
            assert_eq!(r.se_type, VrSeType::Homoskedastic);
            assert_eq!(
                r.stat_variance,
                (2.0 * (2.0 * qf - 1.0) * (qf - 1.0)) / (3.0 * qf),
                "q={q}"
            );
        }
        let r = variance_ratio(&y, q, VarianceRatioSpec::non_overlapping(VrTrend::Drift)).unwrap();
        assert_eq!(r.stat_variance, 2.0 * (q as f64 - 1.0));
        assert_eq!(r.se_type, VrSeType::NonOverlapping);
        assert!(!r.debiased);
    }
}

#[test]
fn non_overlap_truncation_drops_the_trailing_remainder() {
    let mut rng = Lcg(15);
    let y = walk(&mut rng, 503);
    for trend in [VrTrend::None, VrTrend::Drift] {
        let spec = VarianceRatioSpec::non_overlapping(trend);
        for q in [2, 3, 4, 5, 7, 8, 16, 50] {
            let r = variance_ratio(&y, q, spec).unwrap();
            let dropped = (y.len() - 1) % q;
            assert_eq!(r.n_dropped, dropped, "q={q}");
            assert_eq!(r.nobs, y.len() - 1 - dropped);
            assert_eq!(r.nobs % q, 0);
            let pre = variance_ratio(&y[..y.len() - dropped], q, spec).unwrap();
            assert_eq!(pre.n_dropped, 0);
            assert_eq!((pre.vr, pre.stat, pre.p_value), (r.vr, r.stat, r.p_value));
        }
        // The overlapping estimator never truncates.
        let r = variance_ratio(&y, 4, VarianceRatioSpec::default()).unwrap();
        assert_eq!((r.n_dropped, r.nobs), (0, y.len() - 1));
    }
}

// ------------------------------------------------------ inert-keyword contract

#[test]
fn debiased_or_robust_without_overlap_is_refused_in_the_core() {
    let mut rng = Lcg(16);
    let y = walk(&mut rng, 100);
    for trend in [VrTrend::None, VrTrend::Drift] {
        for (debiased, robust, word) in [
            (true, false, "debiased"),
            (false, true, "robust"),
            (true, true, "debiased"),
        ] {
            let spec = VarianceRatioSpec {
                trend,
                overlap: false,
                debiased,
                robust,
            };
            match variance_ratio(&y, 4, spec).unwrap_err() {
                DiagError::InvalidArgument { what } => {
                    assert!(what.contains(word), "{what}");
                    assert!(what.contains("overlap"), "{what}");
                }
                other => panic!("expected InvalidArgument, got {other:?}"),
            }
        }
        // The only valid non-overlapping spec works.
        assert!(variance_ratio(&y, 4, VarianceRatioSpec::non_overlapping(trend)).is_ok());
    }
    // Under overlap the flags are live: they change the answer.
    let a = variance_ratio(&y, 4, VarianceRatioSpec::default()).unwrap();
    let b = variance_ratio(
        &y,
        4,
        VarianceRatioSpec {
            debiased: false,
            ..VarianceRatioSpec::default()
        },
    )
    .unwrap();
    let c = variance_ratio(
        &y,
        4,
        VarianceRatioSpec {
            robust: false,
            ..VarianceRatioSpec::default()
        },
    )
    .unwrap();
    assert_ne!(a.vr, b.vr);
    assert_eq!(a.vr, c.vr);
    assert_ne!(a.stat_variance, c.stat_variance);
}

// ------------------------------------------------------ Chow-Denning identities

#[test]
fn chow_denning_with_one_horizon_is_the_single_test() {
    let mut rng = Lcg(17);
    let y = ar_walk(&mut rng, 300, 0.15);
    for spec in all_specs() {
        for q in [2, 3, 6, 10] {
            let single = variance_ratio(&y, q, spec).unwrap();
            let cd = chow_denning(&y, &[q], spec).unwrap();
            assert_eq!(cd.individual[0], single);
            assert_eq!(cd.stat, single.stat.abs());
            assert_eq!(cd.lag, q);
            assert_eq!(cd.p_value, single.p_value, "bit-exact at m = 1");
            assert_eq!(cd.crit.pct1, pointwise_critical_value(0.01).unwrap());
            assert_eq!(cd.crit.pct5, pointwise_critical_value(0.05).unwrap());
            assert_eq!(cd.crit.pct10, pointwise_critical_value(0.10).unwrap());
        }
    }
}

#[test]
fn chow_denning_horizons_are_bit_identical_to_the_single_calls() {
    // With overlap the joint test shares one set of increments (and the
    // robust lag products up to the longest horizon) across horizons; each
    // per-horizon result must still be exactly the single-horizon call.
    let mut rng = Lcg(20);
    let y = garch_walk(&mut rng, 600, 0.12, 0.8);
    let lags = [16usize, 2, 7, 3, 40, 5];
    for spec in all_specs() {
        let cd = chow_denning(&y, &lags, spec).unwrap();
        for (r, &q) in cd.individual.iter().zip(&lags) {
            assert_eq!(*r, variance_ratio(&y, q, spec).unwrap(), "{spec:?} q={q}");
        }
    }
}

#[test]
fn chow_denning_checks_every_horizon_before_any_data_work() {
    let mut rng = Lcg(21);
    let y = walk(&mut rng, 4001);
    // A valid long horizon followed by one the sample cannot hold: refused
    // by the length rule, naming the bad horizon, without first paying the
    // O(n q) robust variance of the valid one.
    match chow_denning(&y, &[3999, 2, 4000], VarianceRatioSpec::default()).unwrap_err() {
        DiagError::InvalidLags { nlags, n, .. } => assert_eq!((nlags, n), (4000, 4001)),
        other => panic!("{other:?}"),
    }
    // Argument errors of any horizon come before a data error.
    let constant = vec![1.5; 50];
    assert!(matches!(
        chow_denning(&constant, &[2, 1], VarianceRatioSpec::default()).unwrap_err(),
        DiagError::InvalidLags { nlags: 1, .. }
    ));
    assert!(matches!(
        chow_denning(&constant, &[2, 3], VarianceRatioSpec::default()).unwrap_err(),
        DiagError::InvalidArgument { .. }
    ));
}

#[test]
fn chow_denning_is_permutation_invariant_and_bounded() {
    let mut rng = Lcg(18);
    for rep in 0..20 {
        let phi = [0.0, 0.1, -0.1, 0.25][rep % 4];
        let y = ar_walk(&mut rng, 400, phi);
        for spec in [
            VarianceRatioSpec::default(),
            VarianceRatioSpec::non_overlapping(VrTrend::Drift),
        ] {
            let lags = [2usize, 4, 8, 16, 5];
            let base = chow_denning(&y, &lags, spec).unwrap();
            let pmin = base
                .individual
                .iter()
                .map(|r| r.p_value)
                .fold(f64::INFINITY, f64::min);
            assert!(base.p_value >= pmin && base.p_value <= 1.0);
            // Šidák critical values grow with m and exceed the pointwise one.
            assert!(base.crit.pct5 > pointwise_critical_value(0.05).unwrap());
            assert!(base.crit.pct1 > base.crit.pct5 && base.crit.pct5 > base.crit.pct10);
            for perm in [[4usize, 3, 2, 1, 0], [1, 3, 0, 4, 2], [2, 0, 4, 1, 3]] {
                let l: Vec<usize> = perm.iter().map(|&i| lags[i]).collect();
                let r = chow_denning(&y, &l, spec).unwrap();
                assert_eq!(r.stat, base.stat);
                assert_eq!(r.p_value, base.p_value);
                // No exact ties on continuous data, so the argmax horizon is
                // the same whatever the order.
                assert_eq!(r.lag, base.lag);
                for (i, &k) in perm.iter().enumerate() {
                    assert_eq!(r.individual[i], base.individual[k]);
                }
            }
            for (alpha, c) in [
                (0.01, base.crit.pct1),
                (0.05, base.crit.pct5),
                (0.10, base.crit.pct10),
            ] {
                assert_eq!(base.p_value < alpha, base.stat > c, "duality at {alpha}");
            }
        }
    }
}

#[test]
fn chow_denning_refuses_empty_and_repeated_horizons() {
    let mut rng = Lcg(19);
    let y = walk(&mut rng, 100);
    for lags in [vec![], vec![2, 4, 2], vec![3, 3]] {
        match chow_denning(&y, &lags, VarianceRatioSpec::default()).unwrap_err() {
            DiagError::InvalidArgument { what } => assert!(what.contains("lags"), "{what}"),
            other => panic!("{lags:?}: expected InvalidArgument, got {other:?}"),
        }
    }
    // A bad horizon inside a valid set is the single test's refusal.
    assert!(matches!(
        chow_denning(&y, &[2, 1], VarianceRatioSpec::default()).unwrap_err(),
        DiagError::InvalidLags { nlags: 1, .. }
    ));
}

// ------------------------------------------------------------- Monte Carlo

fn rejection_rates(
    mut draw: impl FnMut(&mut Lcg) -> Vec<f64>,
    seed: u64,
    reps: usize,
    qs: &[usize],
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut rng = Lcg(seed);
    let robust = VarianceRatioSpec::default();
    let homo = VarianceRatioSpec {
        robust: false,
        ..VarianceRatioSpec::default()
    };
    let mut rej_r = vec![0usize; qs.len()];
    let mut rej_h = vec![0usize; qs.len()];
    let mut rej_cd = 0usize;
    for _ in 0..reps {
        let y = draw(&mut rng);
        for (i, &q) in qs.iter().enumerate() {
            if variance_ratio(&y, q, robust).unwrap().p_value < 0.05 {
                rej_r[i] += 1;
            }
            if variance_ratio(&y, q, homo).unwrap().p_value < 0.05 {
                rej_h[i] += 1;
            }
        }
        if chow_denning(&y, qs, robust).unwrap().p_value < 0.05 {
            rej_cd += 1;
        }
    }
    let f = |c: usize| c as f64 / reps as f64;
    (
        rej_r.into_iter().map(f).collect(),
        rej_h.into_iter().map(f).collect(),
        f(rej_cd),
    )
}

#[test]
fn size_under_iid_increments() {
    let qs = [2, 4, 8];
    let (rob, homo, cd) = rejection_rates(|r| walk(r, 500), 2024, 400, &qs);
    for (i, &q) in qs.iter().enumerate() {
        assert!((0.025..=0.08).contains(&rob[i]), "robust q={q}: {}", rob[i]);
        assert!((0.025..=0.08).contains(&homo[i]), "homo q={q}: {}", homo[i]);
    }
    // Šidák across positively correlated horizons is conservative: an upper
    // bound only.
    assert!(cd <= 0.07, "Chow-Denning size {cd}");
}

#[test]
fn robust_statistic_keeps_its_size_under_garch() {
    let qs = [2, 4, 8];
    let (rob, homo, cd) = rejection_rates(|r| garch_walk(r, 500, 0.15, 0.80), 77, 400, &qs);
    for (i, &q) in qs.iter().enumerate() {
        assert!((0.025..=0.08).contains(&rob[i]), "robust q={q}: {}", rob[i]);
        assert!(
            homo[i] > rob[i],
            "homoskedastic should over-reject under ARCH: q={q} homo {} vs robust {}",
            homo[i],
            rob[i]
        );
    }
    assert!(cd <= 0.07, "Chow-Denning size {cd}");
}

#[test]
fn power_against_momentum_increments() {
    let mut rng = Lcg(31);
    let y = ar_walk(&mut rng, 400, 0.3);
    let r = variance_ratio(&y, 2, VarianceRatioSpec::default()).unwrap();
    assert!(r.stat > 0.0 && r.vr > 1.0, "{r:?}");
    assert!(r.p_value < 0.01, "{r:?}");
    // Mean reversion pushes the ratio below one.
    let y = ar_walk(&mut rng, 400, -0.3);
    let r = variance_ratio(&y, 2, VarianceRatioSpec::default()).unwrap();
    assert!(r.stat < 0.0 && r.vr < 1.0 && r.p_value < 0.01, "{r:?}");
}

// ---------------------------------------------------------------- refusals

#[test]
fn refusals_teach_and_never_return_garbage() {
    let mut rng = Lcg(41);
    let y = walk(&mut rng, 60);
    let spec = VarianceRatioSpec::default();
    for q in [0, 1] {
        match variance_ratio(&y, q, spec).unwrap_err() {
            DiagError::InvalidLags { requirement, .. } => {
                assert!(requirement.contains("lags >= 2"), "{requirement}")
            }
            other => panic!("q={q}: {other:?}"),
        }
    }
    // An absurd horizon is refused before any O(n q) work.
    assert!(matches!(
        variance_ratio(&y, usize::MAX, spec).unwrap_err(),
        DiagError::InvalidLags { .. }
    ));
    assert!(matches!(
        variance_ratio(
            &y,
            usize::MAX,
            VarianceRatioSpec::non_overlapping(VrTrend::Drift)
        )
        .unwrap_err(),
        DiagError::InvalidLags { .. }
    ));
    // One q-sum (both trends, both estimators) is refused; two are accepted.
    for trend in [VrTrend::None, VrTrend::Drift] {
        for q in [2usize, 3, 4, 9] {
            let ov = VarianceRatioSpec {
                trend,
                debiased: false,
                ..VarianceRatioSpec::default()
            };
            assert!(matches!(
                variance_ratio(&y[..q + 1], q, ov).unwrap_err(),
                DiagError::InvalidLags { .. }
            ));
            assert!(variance_ratio(&y[..q + 2], q, ov).is_ok());
            assert!(variance_ratio(
                &y[..q + 2],
                q,
                VarianceRatioSpec {
                    trend,
                    ..VarianceRatioSpec::default()
                }
            )
            .is_ok());
            let no = VarianceRatioSpec::non_overlapping(trend);
            assert!(matches!(
                variance_ratio(&y[..2 * q], q, no).unwrap_err(),
                DiagError::InvalidLags { .. }
            ));
            assert!(variance_ratio(&y[..2 * q + 1], q, no).is_ok());
        }
    }
    // Non-finite input.
    let mut bad = y.clone();
    bad[7] = f64::NAN;
    assert!(matches!(
        variance_ratio(&bad, 2, spec).unwrap_err(),
        DiagError::NonFinite { index: 7, .. }
    ));
    bad[7] = f64::INFINITY;
    assert!(matches!(
        variance_ratio(&bad, 2, spec).unwrap_err(),
        DiagError::NonFinite { index: 7, .. }
    ));
    assert!(matches!(
        variance_ratio(&[], 2, spec).unwrap_err(),
        DiagError::SeriesTooShort { .. }
    ));
    assert!(matches!(
        variance_ratio(&[1.0], 2, spec).unwrap_err(),
        DiagError::SeriesTooShort { .. }
    ));
    // Zero variance about the drift: constant levels (both trends), and an
    // exact or rounding-level linear trend under drift.
    let constant = vec![3.5; 50];
    let zeros = vec![0.0; 50];
    let line: Vec<f64> = (0..50).map(|t| t as f64).collect();
    let line_frac: Vec<f64> = (0..500).map(|t| 0.1 * t as f64 + 7.0).collect();
    for (yy, trend) in [
        (&constant, VrTrend::Drift),
        (&constant, VrTrend::None),
        (&zeros, VrTrend::None),
        (&line, VrTrend::Drift),
        (&line_frac, VrTrend::Drift),
    ] {
        for s in [
            VarianceRatioSpec {
                trend,
                ..VarianceRatioSpec::default()
            },
            VarianceRatioSpec::non_overlapping(trend),
        ] {
            match variance_ratio(yy, 2, s).unwrap_err() {
                DiagError::InvalidArgument { what } => {
                    assert!(what.contains("zero variance"), "{what}")
                }
                other => panic!("{s:?}: {other:?}"),
            }
        }
    }
    // A linear trend under trend "n" is a legitimate (non-null) input.
    let r = variance_ratio(
        &line,
        2,
        VarianceRatioSpec {
            trend: VrTrend::None,
            robust: false,
            ..VarianceRatioSpec::default()
        },
    )
    .unwrap();
    assert!(r.vr.is_finite() && r.vr > 1.0);
}

#[test]
fn robust_variance_breakdown_is_refused() {
    // Increments that are nonzero only every third period (trend "n"): no
    // two squared increments lie within q - 1 = 1 period, so theta = 0.
    let mut y = vec![0.0];
    for t in 0..60 {
        let step = if t % 3 == 0 { 1.0 } else { 0.0 };
        y.push(y[t] + step);
    }
    let spec = VarianceRatioSpec {
        trend: VrTrend::None,
        ..VarianceRatioSpec::default()
    };
    match variance_ratio(&y, 2, spec).unwrap_err() {
        DiagError::NumericalBreakdown { what } => assert!(what.contains("theta"), "{what}"),
        other => panic!("{other:?}"),
    }
    // The homoskedastic statistic is still defined there.
    assert!(variance_ratio(
        &y,
        2,
        VarianceRatioSpec {
            robust: false,
            ..spec
        }
    )
    .is_ok());
}
