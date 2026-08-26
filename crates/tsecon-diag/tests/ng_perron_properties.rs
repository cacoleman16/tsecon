//! Property, invariant, and Monte-Carlo tests for the Ng-Perron (2001)
//! M unit-root tests. No runnable independent implementation of the M
//! tests exists anywhere (statsmodels 0.14.6 and arch 8.0.0 do not ship
//! them, and no CRAN package does — checked against the CRAN mirror:
//! `urca::ur.ers` stops at DF-GLS and the ERS P-test, `bootUR`/`CADFtest`
//! borrow only the MAIC idea), so per the validation policy these seeded
//! property and Monte-Carlo tests carry the statistical claim:
//!
//! * the exact algebraic identity `MZt = MZa * MSB` to machine precision;
//! * invariance to scaling and to the deterministics the test removes;
//! * Monte-Carlo *size* at the transcribed Table 1 asymptotic critical
//!   values on null (random-walk) data — lag-0 at T = 1000 where the
//!   asymptotic table should be nearly exact, and the full MAIC pipeline
//!   at T = 250;
//! * Monte-Carlo *power ordering* — a stationary AR(1) rejects far more
//!   often than the null does — and the MAIC mechanism check that a
//!   negative MA root lengthens the chosen lag (the paper's motivation);
//! * the error surface (errors that teach, never garbage numbers).

use tsecon_diag::{ng_perron, ng_perron_crit, DfglsTrend, DiagError, NgPerronLagSelection};

/// Deterministic LCG (Knuth MMIX), same generator as the dfgls property
/// suite, so results are reproducible across platforms with no system RNG.
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately standard-normal increment (Irwin-Hall, 12 uniforms).
    fn normalish(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// A deterministic, reproducible random walk (the unit-root null).
fn walk(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut acc = 0.0;
    (0..n)
        .map(|_| {
            acc += rng.normalish();
            acc
        })
        .collect()
}

/// Deterministic stationary pseudo-noise around zero.
fn stationary(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| rng.normalish()).collect()
}

/// A stationary AR(1) with coefficient `rho`.
fn ar1(n: usize, rho: f64, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut y = Vec::with_capacity(n);
    let mut prev = 0.0;
    for _ in 0..n {
        prev = rho * prev + rng.normalish();
        y.push(prev);
    }
    y
}

/// A random walk whose increments are MA(1) with parameter `theta`
/// (`theta` near -1 is the classic size-distortion design of the paper).
fn walk_ma(n: usize, theta: f64, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut y = Vec::with_capacity(n);
    let mut acc = 0.0;
    let mut e_prev = rng.normalish();
    for _ in 0..n {
        let e = rng.normalish();
        acc += e + theta * e_prev;
        e_prev = e;
        y.push(acc);
    }
    y
}

const MAIC: NgPerronLagSelection = NgPerronLagSelection::Maic(None);
const BOTH: [DfglsTrend; 2] = [DfglsTrend::Constant, DfglsTrend::ConstantTrend];

// ------------------------------------------------------------- invariants

#[test]
fn mzt_equals_mza_times_msb_to_machine_precision() {
    // The exact algebraic identity of the paper. The implementation
    // computes MZt from its own formula, so this is a genuine invariant
    // check, not `x == x`; the only floating-point slack is
    // sqrt(a)*sqrt(b) vs sqrt(a*b) — a couple of ulps.
    for seed in [3_u64, 21, 47, 90] {
        for trend in BOTH {
            for lags in [
                MAIC,
                NgPerronLagSelection::Fixed(0),
                NgPerronLagSelection::Fixed(4),
            ] {
                for y in [walk(200, seed), stationary(200, seed), ar1(200, 0.8, seed)] {
                    let r = ng_perron(&y, trend, lags).unwrap();
                    let rel = ((r.mzt - r.mza * r.msb) / r.mzt).abs();
                    assert!(
                        rel <= 1e-13,
                        "MZt != MZa*MSB: {} vs {} (rel {rel:e}, seed {seed}, {trend:?})",
                        r.mzt,
                        r.mza * r.msb,
                    );
                }
            }
        }
    }
}

#[test]
fn statistics_are_scale_invariant() {
    // All four statistics are ratios of quantities quadratic in y, and
    // GLS detrending is linear, so rescaling y changes nothing (nor the
    // MAIC-selected lag).
    let y = walk(150, 21);
    for trend in BOTH {
        let base = ng_perron(&y, trend, MAIC).unwrap();
        for scale in [1e-6, 3.0, 1e6] {
            let ys: Vec<f64> = y.iter().map(|&v| v * scale).collect();
            let s = ng_perron(&ys, trend, MAIC).unwrap();
            assert_eq!(s.used_lag, base.used_lag, "lag under scale {scale}");
            for (a, b, what) in [
                (s.mza, base.mza, "mza"),
                (s.mzt, base.mzt, "mzt"),
                (s.msb, base.msb, "msb"),
                (s.mpt, base.mpt, "mpt"),
            ] {
                let rel = ((a - b) / b).abs();
                assert!(
                    rel < 1e-9,
                    "{what} under scale {scale}: {a} vs {b} (rel {rel:e})"
                );
            }
        }
    }
}

#[test]
fn statistics_are_invariant_to_the_deterministics_they_remove() {
    let y = walk(150, 37);
    for trend in BOTH {
        let base = ng_perron(&y, trend, MAIC).unwrap();
        let shifted: Vec<f64> = y.iter().map(|&v| v + 250.0).collect();
        let s = ng_perron(&shifted, trend, MAIC).unwrap();
        assert_eq!(s.used_lag, base.used_lag);
        assert!(((s.mza - base.mza) / base.mza).abs() < 1e-8, "shift mza");
        assert!(((s.mpt - base.mpt) / base.mpt).abs() < 1e-8, "shift mpt");
    }
    // A linear trend is absorbed exactly in the "ct" case only.
    let base = ng_perron(&y, DfglsTrend::ConstantTrend, MAIC).unwrap();
    let trended: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(t, &v)| v + 5.0 + 0.4 * t as f64)
        .collect();
    let s = ng_perron(&trended, DfglsTrend::ConstantTrend, MAIC).unwrap();
    assert_eq!(s.used_lag, base.used_lag);
    assert!(((s.mza - base.mza) / base.mza).abs() < 1e-8, "trend mza");
    assert!(((s.mzt - base.mzt) / base.mzt).abs() < 1e-8, "trend mzt");
}

#[test]
fn fixed_lag_at_the_selected_lag_reproduces_the_auto_statistics() {
    let y = walk(180, 47);
    for trend in BOTH {
        let auto = ng_perron(&y, trend, MAIC).unwrap();
        let fixed = ng_perron(&y, trend, NgPerronLagSelection::Fixed(auto.used_lag)).unwrap();
        assert_eq!(fixed, auto);
    }
}

#[test]
fn nobs_is_n_minus_one_minus_lag_and_crit_is_the_table() {
    let y = walk(120, 71);
    let r = ng_perron(&y, DfglsTrend::Constant, NgPerronLagSelection::Fixed(4)).unwrap();
    assert_eq!(r.used_lag, 4);
    assert_eq!(r.nobs, 120 - 1 - 4);
    assert_eq!(r.crit, ng_perron_crit(DfglsTrend::Constant));
    // Spot-check the transcribed Table 1 numbers (Ng-Perron 2001).
    assert_eq!(r.crit.mza.pct5, -8.1);
    assert_eq!(r.crit.mzt.pct5, -1.98);
    assert_eq!(r.crit.msb.pct5, 0.233);
    assert_eq!(r.crit.mpt.pct5, 3.17);
    let ct = ng_perron_crit(DfglsTrend::ConstantTrend);
    assert_eq!(ct.mza.pct5, -17.3);
    assert_eq!(ct.mzt.pct5, -2.91);
    assert_eq!(ct.msb.pct5, 0.168);
    assert_eq!(ct.mpt.pct5, 5.48);
}

// ------------------------------------------------------- statistical sanity

#[test]
fn rejects_stationary_alternatives_and_not_a_random_walk() {
    // A persistent stationary AR(0.7) — the regime the M tests are built
    // for — rejects on all four statistics at the 5% level under MAIC.
    let ar = ng_perron(&ar1(250, 0.7, 61), DfglsTrend::Constant, MAIC).unwrap();
    let cv = ar.crit;
    assert!(ar.mza < cv.mza.pct5, "MZa should reject: {}", ar.mza);
    assert!(ar.mzt < cv.mzt.pct5, "MZt should reject: {}", ar.mzt);
    assert!(ar.msb < cv.msb.pct5, "MSB should reject: {}", ar.msb);
    assert!(ar.mpt < cv.mpt.pct5, "MPT should reject: {}", ar.mpt);

    // Pure i.i.d. noise at a fixed lag rejects overwhelmingly.
    let noise = stationary(250, 61);
    let r = ng_perron(&noise, DfglsTrend::Constant, NgPerronLagSelection::Fixed(0)).unwrap();
    assert!(r.mza < cv.mza.pct1, "MZa lag-0 should reject: {}", r.mza);
    assert!(r.mzt < cv.mzt.pct1, "MZt lag-0 should reject: {}", r.mzt);
    assert!(r.msb < cv.msb.pct1, "MSB lag-0 should reject: {}", r.msb);
    assert!(r.mpt < cv.mpt.pct1, "MPT lag-0 should reject: {}", r.mpt);

    let rw = walk(200, 11);
    for trend in BOTH {
        let r = ng_perron(&rw, trend, MAIC).unwrap();
        let cv = r.crit;
        assert!(
            r.mza > cv.mza.pct10,
            "MZa should not reject ({trend:?}): {}",
            r.mza
        );
        assert!(
            r.mzt > cv.mzt.pct10,
            "MZt should not reject ({trend:?}): {}",
            r.mzt
        );
        assert!(
            r.msb > cv.msb.pct10,
            "MSB should not reject ({trend:?}): {}",
            r.msb
        );
        assert!(
            r.mpt > cv.mpt.pct10,
            "MPT should not reject ({trend:?}): {}",
            r.mpt
        );
    }
}

/// The documented Perron-Qu (2007) caveat, pinned so it stays documented:
/// on data *far* from the null (pure i.i.d. noise), MAIC drives the lag to
/// its maximum and the AR spectral density collapses, so the M tests lose
/// power exactly where a unit-root test is least needed. On this seed the
/// selected lag is the Schwert cap and MZa fails to reject at 5% even
/// though the lag-0 statistic is -124.9. The model card documents the
/// remedy (a fixed or capped lag when the series is obviously stationary).
#[test]
fn maic_power_reversal_on_far_from_null_data_is_real_and_documented() {
    let noise = stationary(250, 61);
    let r = ng_perron(&noise, DfglsTrend::Constant, MAIC).unwrap();
    assert_eq!(r.used_lag, 16, "Schwert cap at T = 250");
    assert!(
        r.mza > r.crit.mza.pct5,
        "expected the documented power reversal, got MZa = {}",
        r.mza
    );
    let lag0 = ng_perron(&noise, DfglsTrend::Constant, NgPerronLagSelection::Fixed(0)).unwrap();
    assert!(
        lag0.mza < -100.0,
        "lag-0 MZa should be decisive: {}",
        lag0.mza
    );
}

/// Count of rejections (below the critical value) at the 1% and 5% level
/// for each of the four statistics.
#[derive(Default, Debug)]
struct Rejections {
    n: usize,
    at5: [usize; 4],
    at1: [usize; 4],
}

impl Rejections {
    fn tally(&mut self, r: &tsecon_diag::NgPerronResult) {
        let cv = &r.crit;
        let stats = [r.mza, r.mzt, r.msb, r.mpt];
        let cv5 = [cv.mza.pct5, cv.mzt.pct5, cv.msb.pct5, cv.mpt.pct5];
        let cv1 = [cv.mza.pct1, cv.mzt.pct1, cv.msb.pct1, cv.mpt.pct1];
        for i in 0..4 {
            if stats[i] < cv5[i] {
                self.at5[i] += 1;
            }
            if stats[i] < cv1[i] {
                self.at1[i] += 1;
            }
        }
        self.n += 1;
    }

    fn rate5(&self, i: usize) -> f64 {
        self.at5[i] as f64 / self.n as f64
    }

    fn rate1(&self, i: usize) -> f64 {
        self.at1[i] as f64 / self.n as f64
    }
}

const STAT_NAMES: [&str; 4] = ["MZa", "MZt", "MSB", "MPT"];

/// Monte-Carlo size at the asymptotic critical values, lag fixed at 0,
/// T = 1000, i.i.d. Gaussian-ish increments: the asymptotic Table 1
/// should be close to exact, so measured size at the 5% (1%) critical
/// value must sit near 0.05 (0.01). Reps and measured rates are quoted in
/// the validation matrix; the MC standard error at 2000 reps is ~0.005.
#[test]
fn mc_size_lag0_large_t_is_near_nominal() {
    const REPS: u64 = 2000;
    const T: usize = 1000;
    for trend in BOTH {
        let mut tally = Rejections::default();
        for rep in 0..REPS {
            let y = walk(T, 1_000_000 + rep);
            let r = ng_perron(&y, trend, NgPerronLagSelection::Fixed(0)).unwrap();
            tally.tally(&r);
        }
        for (i, name) in STAT_NAMES.iter().enumerate() {
            let (s5, s1) = (tally.rate5(i), tally.rate1(i));
            eprintln!("size lag0 T={T} {trend:?} {name}: 5% -> {s5:.4}, 1% -> {s1:.4}");
            assert!(
                (0.025..=0.075).contains(&s5),
                "{trend:?} {name} 5% size {s5} outside [0.025, 0.075]"
            );
            assert!(
                (0.002..=0.025).contains(&s1),
                "{trend:?} {name} 1% size {s1} outside [0.002, 0.025]"
            );
        }
    }
}

/// Monte-Carlo size of the full pipeline (MAIC lag selection, default
/// Schwert cap) at T = 250 on null data. The finite-sample size of the
/// MAIC-selected M tests is known to sit at or slightly below nominal for
/// i.i.d. errors (Ng-Perron 2001, Table 2), so the acceptance band is
/// one-sided-tight above and generous below.
#[test]
fn mc_size_maic_moderate_t_is_controlled() {
    const REPS: u64 = 600;
    const T: usize = 250;
    for trend in BOTH {
        let mut tally = Rejections::default();
        for rep in 0..REPS {
            let y = walk(T, 5_000_000 + rep);
            let r = ng_perron(&y, trend, MAIC).unwrap();
            tally.tally(&r);
        }
        for (i, name) in STAT_NAMES.iter().enumerate() {
            let s5 = tally.rate5(i);
            eprintln!(
                "size MAIC T={T} {trend:?} {name}: 5% -> {s5:.4}, 1% -> {:.4}",
                tally.rate1(i)
            );
            assert!(
                (0.005..=0.09).contains(&s5),
                "{trend:?} {name} 5% size {s5} outside [0.005, 0.09]"
            );
        }
    }
}

/// Power ordering: a stationary AR(0.8) at T = 250 must reject far more
/// often than the measured null rate — the point of a unit-root test.
#[test]
fn mc_power_stationary_alternative_rejects_more() {
    const REPS: u64 = 300;
    const T: usize = 250;
    for trend in BOTH {
        let mut null_tally = Rejections::default();
        let mut alt_tally = Rejections::default();
        for rep in 0..REPS {
            let rw = walk(T, 9_000_000 + rep);
            let ar = ar1(T, 0.8, 12_000_000 + rep);
            null_tally.tally(&ng_perron(&rw, trend, MAIC).unwrap());
            alt_tally.tally(&ng_perron(&ar, trend, MAIC).unwrap());
        }
        for (i, name) in STAT_NAMES.iter().enumerate() {
            let (size, power) = (null_tally.rate5(i), alt_tally.rate5(i));
            eprintln!("power T={T} {trend:?} {name}: null {size:.4} vs AR(0.8) {power:.4}");
            assert!(
                power > 0.5 && power > size + 0.3,
                "{trend:?} {name}: power {power} not clearly above size {size}"
            );
        }
    }
}

/// The MAIC mechanism (the reason the tests exist): under a unit root with
/// a large negative MA component in the increments, MAIC must select
/// substantially longer lags than under i.i.d. increments — that is
/// exactly how it protects the size where AIC/BIC collapse.
#[test]
fn maic_lengthens_the_lag_under_a_negative_ma_root() {
    const REPS: u64 = 120;
    const T: usize = 250;
    let mut iid_lags = 0usize;
    let mut ma_lags = 0usize;
    for rep in 0..REPS {
        let iid = walk(T, 21_000_000 + rep);
        let ma = walk_ma(T, -0.8, 23_000_000 + rep);
        iid_lags += ng_perron(&iid, DfglsTrend::Constant, MAIC)
            .unwrap()
            .used_lag;
        ma_lags += ng_perron(&ma, DfglsTrend::Constant, MAIC).unwrap().used_lag;
    }
    let (iid_mean, ma_mean) = (iid_lags as f64 / REPS as f64, ma_lags as f64 / REPS as f64);
    eprintln!("MAIC mean lag: iid {iid_mean:.3} vs MA(-0.8) {ma_mean:.3}");
    assert!(
        ma_mean > iid_mean + 1.0,
        "MAIC did not lengthen lags under MA(-0.8): {ma_mean} vs {iid_mean}"
    );
}

// ------------------------------------------------------------ error surface

#[test]
fn error_surface() {
    // Empty.
    assert!(matches!(
        ng_perron(&[], DfglsTrend::Constant, MAIC),
        Err(DiagError::SeriesTooShort { .. })
    ));
    // Constant series.
    let c = vec![3.0; 50];
    assert!(matches!(
        ng_perron(&c, DfglsTrend::Constant, MAIC),
        Err(DiagError::ConstantSeries { .. })
    ));
    // Non-finite.
    let mut bad = walk(50, 5);
    bad[10] = f64::NAN;
    assert!(matches!(
        ng_perron(&bad, DfglsTrend::Constant, MAIC),
        Err(DiagError::NonFinite { .. })
    ));
    // Too short for the trend spec.
    let short = walk(4, 5);
    assert!(matches!(
        ng_perron(&short, DfglsTrend::ConstantTrend, MAIC),
        Err(DiagError::SeriesTooShort { .. })
    ));
    // Fixed lag too large for the sample.
    let y = walk(20, 5);
    assert!(matches!(
        ng_perron(&y, DfglsTrend::Constant, NgPerronLagSelection::Fixed(10)),
        Err(DiagError::SeriesTooShort { .. })
    ));
    // A user max_lag too large for the sample.
    assert!(matches!(
        ng_perron(
            &y,
            DfglsTrend::Constant,
            NgPerronLagSelection::Maic(Some(12))
        ),
        Err(DiagError::SeriesTooShort { .. })
    ));
    // An exact linear trend: the GLS fit is exact and the regression
    // degenerates — a teaching error, not a garbage number.
    let det: Vec<f64> = (0..60).map(|t| 2.0 + 0.5 * t as f64).collect();
    assert!(ng_perron(&det, DfglsTrend::ConstantTrend, MAIC).is_err());
}
