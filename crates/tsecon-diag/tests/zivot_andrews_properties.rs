//! Property tests for the Zivot-Andrews one-break unit-root test:
//! invariances that hold in exact arithmetic, break localization on an
//! engineered shift, trim-window guarantees, and teaching errors on
//! degenerate input.

use tsecon_diag::{za_crit, za_p, zivot_andrews, DiagError, ZaLagSelection, ZaRegression};

/// Deterministic LCG noise so the tests need no RNG dependency.
fn noise(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Two uniform draws -> one rough normal via sum of 4 uniforms.
            let mut s = 0.0;
            for _ in 0..4 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s += (state >> 11) as f64 / (1u64 << 53) as f64;
            }
            s - 2.0
        })
        .collect()
}

fn random_walk(seed: u64, n: usize) -> Vec<f64> {
    let e = noise(seed, n);
    let mut y = Vec::with_capacity(n);
    let mut acc = 0.0;
    for v in e {
        acc += v;
        y.push(acc);
    }
    y
}

const REGS: [ZaRegression; 3] = [
    ZaRegression::Constant,
    ZaRegression::Trend,
    ZaRegression::ConstantTrend,
];

// ------------------------------------------------------------ invariances

/// The minimum-t statistic (and the chosen break) is invariant to affine
/// transforms a + b*y with b > 0: the regression contains a constant and
/// the t-ratio is scale-free. (Exact in exact arithmetic; 1e-8 leaves room
/// for the floating-point renormalization.)
#[test]
fn location_scale_invariance() {
    let y = random_walk(3, 180);
    let shifted: Vec<f64> = y.iter().map(|&v| 1000.0 + 5.0 * v).collect();
    for reg in REGS {
        for sel in [ZaLagSelection::Fixed(2), ZaLagSelection::Aic(None)] {
            let a = zivot_andrews(&y, reg, 0.15, sel).expect("base runs");
            let b = zivot_andrews(&shifted, reg, 0.15, sel).expect("affine runs");
            let rel = ((a.statistic - b.statistic) / a.statistic).abs();
            assert!(
                rel < 1e-8,
                "{reg:?}/{sel:?}: stat {} vs {} (rel {rel:e})",
                a.statistic,
                b.statistic
            );
            assert_eq!(a.break_index, b.break_index, "{reg:?}/{sel:?} break moved");
            assert_eq!(a.used_lag, b.used_lag, "{reg:?}/{sel:?} lag moved");
        }
    }
}

/// A sign flip is an affine transform too (b < 0): the break dummies enter
/// symmetrically, so the statistic is again unchanged.
#[test]
fn sign_flip_invariance() {
    let y = random_walk(9, 150);
    let flipped: Vec<f64> = y.iter().map(|&v| -v).collect();
    for reg in REGS {
        let a = zivot_andrews(&y, reg, 0.15, ZaLagSelection::Fixed(1)).expect("base");
        let b = zivot_andrews(&flipped, reg, 0.15, ZaLagSelection::Fixed(1)).expect("flip");
        let rel = ((a.statistic - b.statistic) / a.statistic).abs();
        assert!(rel < 1e-8, "{reg:?}: {} vs {}", a.statistic, b.statistic);
        assert_eq!(a.break_index, b.break_index);
    }
}

// ------------------------------------------------- break localization

/// A large engineered level shift in an otherwise stationary series must
/// be (a) detected — the "c" statistic rejects far beyond the 1% critical
/// value — and (b) localized: the chosen break index lands within one
/// observation of the true last pre-break index.
#[test]
fn engineered_level_shift_is_localized() {
    let true_last_pre_break = 99_usize; // shift starts at index 100
    for seed in [1, 2, 5] {
        let mut y = noise(seed, 200);
        for v in y.iter_mut().skip(true_last_pre_break + 1) {
            *v += 12.0;
        }
        let r = zivot_andrews(&y, ZaRegression::Constant, 0.15, ZaLagSelection::Aic(None))
            .expect("runs");
        assert!(
            r.statistic < r.crit.pct1,
            "seed {seed}: stat {} does not reject (1% cv {})",
            r.statistic,
            r.crit.pct1
        );
        assert!(
            r.break_index.abs_diff(true_last_pre_break) <= 1,
            "seed {seed}: break at {} not near {true_last_pre_break}",
            r.break_index
        );
        assert!(r.p_value <= 0.01, "seed {seed}: p {} too large", r.p_value);
    }
}

// ------------------------------------------------------ trim guarantees

/// The reported break index always lies inside the trimmed window
/// `[trimcnt, n - trimcnt - 1]`, for every regression and several trims.
#[test]
fn break_index_respects_trim_bounds() {
    for seed in [4, 8, 15] {
        let y = random_walk(seed, 120);
        let n = y.len();
        for reg in REGS {
            for trim in [0.05, 0.15, 0.25, 1.0 / 3.0] {
                let trimcnt = (n as f64 * trim) as usize;
                let r = match zivot_andrews(&y, reg, trim, ZaLagSelection::Fixed(1)) {
                    Ok(r) => r,
                    // Tiny trims can make the lag guard fire; that is the
                    // documented contract, not a property failure.
                    Err(DiagError::InvalidLags { .. }) => continue,
                    Err(e) => panic!("unexpected error: {e}"),
                };
                assert!(
                    r.break_index >= trimcnt && r.break_index < n - trimcnt,
                    "seed {seed} {reg:?} trim {trim}: break {} outside [{}, {}]",
                    r.break_index,
                    trimcnt,
                    n - trimcnt - 1
                );
                assert_eq!(r.trim, trim, "trim echoed");
                assert_eq!(r.nobs, n, "nobs echoed");
            }
        }
    }
}

/// Out-of-range trims are refused with the teaching error.
#[test]
fn invalid_trim_is_refused() {
    let y = random_walk(1, 100);
    for trim in [-0.01, 0.34, 0.5, 1.0, f64::NAN] {
        match zivot_andrews(&y, ZaRegression::Constant, trim, ZaLagSelection::Fixed(1)) {
            Err(DiagError::InvalidTrim { value }) => {
                assert!(value.is_nan() && trim.is_nan() || value == trim);
            }
            other => panic!("trim {trim}: expected InvalidTrim, got {other:?}"),
        }
    }
}

/// A lag order too large for the trim window (the break dummy would lose
/// its pre-break regime) is refused with an explanation, not a rank panic.
#[test]
fn lag_exceeding_trim_window_is_refused() {
    let y = random_walk(2, 100);
    // trimcnt = 15 -> lags must be <= 14.
    let r = zivot_andrews(&y, ZaRegression::Constant, 0.15, ZaLagSelection::Fixed(15));
    match r {
        Err(DiagError::InvalidLags { nlags, .. }) => assert_eq!(nlags, 15),
        other => panic!("expected InvalidLags, got {other:?}"),
    }
    assert!(zivot_andrews(&y, ZaRegression::Constant, 0.15, ZaLagSelection::Fixed(14)).is_ok());
}

// ------------------------------------------------------ degenerate input

#[test]
fn degenerate_input_raises() {
    // Empty and too-short series.
    match zivot_andrews(&[], ZaRegression::Constant, 0.15, ZaLagSelection::Fixed(0)) {
        Err(DiagError::SeriesTooShort { .. }) => {}
        other => panic!("empty: expected SeriesTooShort, got {other:?}"),
    }
    let short: Vec<f64> = (0..6).map(|i| i as f64).collect();
    match zivot_andrews(
        &short,
        ZaRegression::Constant,
        0.15,
        ZaLagSelection::Fixed(0),
    ) {
        Err(DiagError::SeriesTooShort { .. }) => {}
        other => panic!("short: expected SeriesTooShort, got {other:?}"),
    }

    // Constant series.
    let flat = vec![2.5; 100];
    match zivot_andrews(
        &flat,
        ZaRegression::Constant,
        0.15,
        ZaLagSelection::Fixed(0),
    ) {
        Err(DiagError::ConstantSeries { .. }) => {}
        other => panic!("constant: expected ConstantSeries, got {other:?}"),
    }

    // Non-finite observations.
    let mut y = random_walk(6, 100);
    y[40] = f64::NAN;
    match zivot_andrews(&y, ZaRegression::Constant, 0.15, ZaLagSelection::Fixed(0)) {
        Err(DiagError::NonFinite { index, .. }) => assert_eq!(index, 40),
        other => panic!("nan: expected NonFinite, got {other:?}"),
    }
}

// ------------------------------------------------- p-value / crit sanity

/// The p-value map is monotone in the statistic, clamped to the table
/// range, and the critical values are ordered and are the table knots.
#[test]
fn p_map_monotone_and_crit_ordered() {
    for reg in REGS {
        // The clamps are the table ends mapped through the same /100.0 the
        // reference applies (0.001% and 99.9%; the latter is
        // 0.999000...0001 in binary).
        let (p_lo, p_hi) = (za_p(f64::NEG_INFINITY, reg), za_p(f64::INFINITY, reg));
        assert_eq!(p_lo, 0.001 / 100.0, "{reg:?} left clamp");
        assert_eq!(p_hi, 99.9 / 100.0, "{reg:?} right clamp");
        let mut last = -1.0;
        for i in 0..=100 {
            let stat = -8.0 + 8.5 * (i as f64) / 100.0;
            let p = za_p(stat, reg);
            assert!((p_lo..=p_hi).contains(&p), "{reg:?}: p {p} out of range");
            assert!(p >= last, "{reg:?}: p not monotone at stat {stat}");
            last = p;
        }
        let cv = za_crit(reg);
        assert!(
            cv.pct1 < cv.pct5 && cv.pct5 < cv.pct10,
            "{reg:?}: crit not ordered: {cv:?}"
        );
    }
}

/// The result is consistent with its own p-value map: re-mapping the
/// reported statistic reproduces the reported p-value bit-for-bit.
#[test]
fn result_consistent_with_p_map() {
    let y = random_walk(12, 160);
    for reg in REGS {
        let r = zivot_andrews(&y, reg, 0.15, ZaLagSelection::Aic(None)).expect("runs");
        assert_eq!(r.p_value, za_p(r.statistic, reg));
        assert_eq!(r.crit, za_crit(reg));
    }
}
