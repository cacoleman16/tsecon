//! Property tests for STL decomposition: exact reconstruction, the
//! near-zero per-cycle seasonal mean the low-pass step enforces, the
//! nested-case equivalences (`robust = false` == explicit
//! `inner_iter = 5, outer_iter = 0`; `robust = true` == 2/15), exact scale
//! equivariance under a power-of-two rescale, shift behaviour (the trend
//! absorbs a level shift; the seasonal ignores it), and the errors the
//! degenerate inputs must raise.

// Elementwise assertions over parallel component arrays read most clearly
// as index loops.
#![allow(clippy::needless_range_loop)]

use tsecon_filters::{seasonal_strength, stl, FiltersError, StlParams};

/// A deterministic monthly test series: trend + stable seasonal + a small
/// deterministic "noise" wobble (no RNG dependency).
fn monthly(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let tf = t as f64;
            10.0 + 0.05 * tf
                + 2.0 * (2.0 * std::f64::consts::PI * tf / 12.0).sin()
                + 0.3 * (tf * 0.7134).sin() * (tf * 0.2718).cos()
        })
        .collect()
}

#[test]
fn components_reconstruct_y() {
    let y = monthly(144);
    let r = stl(&y, 12, &StlParams::default()).expect("stl runs");
    for i in 0..y.len() {
        let recon = r.seasonal[i] + r.trend[i] + r.resid[i];
        assert!(
            (recon - y[i]).abs() <= 1e-10 * y[i].abs().max(1.0),
            "reconstruction failed at {i}: {recon} vs {}",
            y[i]
        );
    }
}

#[test]
fn seasonal_averages_near_zero_over_each_cycle() {
    let y = monthly(144);
    let r = stl(&y, 12, &StlParams::default()).expect("stl runs");
    let sd = {
        let m = r.seasonal.iter().sum::<f64>() / 144.0;
        (r.seasonal.iter().map(|s| (s - m) * (s - m)).sum::<f64>() / 143.0).sqrt()
    };
    for c in 0..12 {
        let mean: f64 = r.seasonal[c * 12..(c + 1) * 12].iter().sum::<f64>() / 12.0;
        assert!(
            mean.abs() <= 0.1 * sd,
            "cycle {c}: seasonal mean {mean} exceeds 10% of the seasonal sd {sd}"
        );
    }
}

#[test]
fn robust_false_equals_explicit_inner5_outer0() {
    let y = monthly(120);
    let a = stl(&y, 12, &StlParams::default()).expect("default runs");
    let b = stl(
        &y,
        12,
        &StlParams {
            inner_iter: Some(5),
            outer_iter: Some(0),
            ..Default::default()
        },
    )
    .expect("explicit runs");
    assert_eq!(a.seasonal, b.seasonal, "seasonal must match bitwise");
    assert_eq!(a.trend, b.trend, "trend must match bitwise");
    assert_eq!(a.resid, b.resid, "resid must match bitwise");
    assert!(
        a.weights.iter().all(|&w| w == 1.0),
        "outer = 0 => weights 1"
    );
    assert_eq!(a.config.inner_iter, 5);
    assert_eq!(a.config.outer_iter, 0);
}

#[test]
fn robust_true_equals_explicit_inner2_outer15() {
    let mut y = monthly(120);
    y[40] += 9.0; // give the robustness loop an outlier to work on
    let a = stl(
        &y,
        12,
        &StlParams {
            robust: true,
            ..Default::default()
        },
    )
    .expect("robust runs");
    let b = stl(
        &y,
        12,
        &StlParams {
            inner_iter: Some(2),
            outer_iter: Some(15),
            ..Default::default()
        },
    )
    .expect("explicit runs");
    assert_eq!(a.seasonal, b.seasonal, "seasonal must match bitwise");
    assert_eq!(a.trend, b.trend, "trend must match bitwise");
    assert_eq!(a.weights, b.weights, "weights must match bitwise");
    // The planted outlier must be visibly downweighted.
    assert!(
        a.weights[40] < 0.5,
        "outlier weight {} not downweighted",
        a.weights[40]
    );
    assert!(a.weights.iter().all(|&w| (0.0..=1.0).contains(&w)));
}

#[test]
fn scale_equivariance() {
    let mut y = monthly(120);
    y[33] -= 7.0;
    let params = StlParams {
        robust: true,
        ..Default::default()
    };
    let base = stl(&y, 12, &params).expect("base runs");

    // A power-of-two rescale commutes with every IEEE operation in the
    // algorithm (the loess weights and robustness weights are scale-free),
    // so the components must double BITWISE.
    let y2: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();
    let r2 = stl(&y2, 12, &params).expect("scaled runs");
    for i in 0..y.len() {
        assert_eq!(r2.seasonal[i], 2.0 * base.seasonal[i], "seasonal[{i}]");
        assert_eq!(r2.trend[i], 2.0 * base.trend[i], "trend[{i}]");
        assert_eq!(r2.weights[i], base.weights[i], "weights[{i}]");
    }

    // A general rescale holds to rounding error.
    let c = 3.7;
    let yc: Vec<f64> = y.iter().map(|v| c * v).collect();
    let rc = stl(&yc, 12, &params).expect("scaled runs");
    for i in 0..y.len() {
        assert!(
            (rc.seasonal[i] - c * base.seasonal[i]).abs() <= 1e-9,
            "seasonal[{i}]: {} vs {}",
            rc.seasonal[i],
            c * base.seasonal[i]
        );
        assert!(
            (rc.trend[i] - c * base.trend[i]).abs() <= 1e-9 * base.trend[i].abs().max(1.0),
            "trend[{i}]"
        );
    }
}

#[test]
fn level_shift_goes_to_the_trend() {
    let y = monthly(120);
    let base = stl(&y, 12, &StlParams::default()).expect("base runs");
    let shifted: Vec<f64> = y.iter().map(|v| v + 250.0).collect();
    let r = stl(&shifted, 12, &StlParams::default()).expect("shifted runs");
    for i in 0..y.len() {
        assert!(
            (r.seasonal[i] - base.seasonal[i]).abs() <= 1e-7,
            "seasonal[{i}] moved under a level shift"
        );
        assert!(
            (r.trend[i] - (base.trend[i] + 250.0)).abs() <= 1e-7,
            "trend[{i}] did not absorb the shift"
        );
    }
}

#[test]
fn constant_series_decomposes_to_flat_trend() {
    let y = vec![5.0; 60];
    let r = stl(&y, 12, &StlParams::default()).expect("constant runs");
    for i in 0..60 {
        assert!(r.seasonal[i].abs() <= 1e-10, "seasonal[{i}] not ~0");
        assert!((r.trend[i] - 5.0).abs() <= 1e-10, "trend[{i}] not ~5");
    }
    // cmad = 0 short-circuit: an EXACTLY fit series (all zeros — no
    // rounding residue at all) keeps every robustness weight at 1, matching
    // statsmodels. (A constant nonzero series leaves ~1e-15 rounding
    // residue, so its weights are a bisquare of noise there too.)
    let zeros = vec![0.0; 60];
    let rr = stl(
        &zeros,
        12,
        &StlParams {
            robust: true,
            ..Default::default()
        },
    )
    .expect("robust zeros runs");
    assert!(rr.weights.iter().all(|&w| w == 1.0));
}

// ------------------------------------------------------ degenerate inputs

#[test]
fn period_below_two_raises() {
    let y = monthly(48);
    for period in [0usize, 1] {
        match stl(&y, period, &StlParams::default()) {
            Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, "period"),
            other => panic!("period = {period}: expected InvalidParameter, got {other:?}"),
        }
    }
}

#[test]
fn too_short_series_raises() {
    let y = monthly(23); // < 2 * 12
    match stl(&y, 12, &StlParams::default()) {
        Err(FiltersError::SeriesTooShort { needed, got, .. }) => {
            assert_eq!(needed, 24);
            assert_eq!(got, 23);
        }
        other => panic!("expected SeriesTooShort, got {other:?}"),
    }
    // Empty input trips the same guard.
    assert!(matches!(
        stl(&[], 12, &StlParams::default()),
        Err(FiltersError::SeriesTooShort { .. })
    ));
}

#[test]
fn invalid_windows_degrees_jumps_raise() {
    let y = monthly(72);
    let check = |params: StlParams, expect: &str| match stl(&y, 12, &params) {
        Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, expect),
        other => panic!("{expect}: expected InvalidParameter, got {other:?}"),
    };
    check(
        StlParams {
            seasonal: 4,
            ..Default::default()
        },
        "seasonal",
    ); // even
    check(
        StlParams {
            seasonal: 1,
            ..Default::default()
        },
        "seasonal",
    ); // < 3
    check(
        StlParams {
            trend: Some(11),
            ..Default::default()
        },
        "trend",
    ); // <= period
    check(
        StlParams {
            trend: Some(14),
            ..Default::default()
        },
        "trend",
    ); // even
    check(
        StlParams {
            low_pass: Some(9),
            ..Default::default()
        },
        "low_pass",
    ); // <= period
    check(
        StlParams {
            seasonal_deg: 2,
            ..Default::default()
        },
        "seasonal_deg",
    );
    check(
        StlParams {
            trend_deg: 2,
            ..Default::default()
        },
        "trend_deg",
    );
    check(
        StlParams {
            low_pass_deg: 2,
            ..Default::default()
        },
        "low_pass_deg",
    );
    check(
        StlParams {
            seasonal_jump: 0,
            ..Default::default()
        },
        "seasonal_jump",
    );
    check(
        StlParams {
            trend_jump: 0,
            ..Default::default()
        },
        "trend_jump",
    );
    check(
        StlParams {
            low_pass_jump: 0,
            ..Default::default()
        },
        "low_pass_jump",
    );
    check(
        StlParams {
            inner_iter: Some(0),
            ..Default::default()
        },
        "inner_iter",
    );
}

#[test]
fn non_finite_input_raises() {
    let mut y = monthly(48);
    y[10] = f64::NAN;
    assert!(matches!(
        stl(&y, 12, &StlParams::default()),
        Err(FiltersError::NonFiniteInput { index: 10 })
    ));
}

// ------------------------------------------------------- strength measures

#[test]
fn strength_bounds_and_orderings() {
    // Strong deterministic seasonality: seasonal strength near 1.
    let y = monthly(144);
    let s = seasonal_strength(&y, 12).expect("strength runs");
    assert!(
        s.seasonal_strength > 0.9,
        "clean sine seasonal strength {} not near 1",
        s.seasonal_strength
    );
    assert!((0.0..=1.0).contains(&s.seasonal_strength));
    assert!((0.0..=1.0).contains(&s.trend_strength));
    assert!(s.trend_strength > 0.9, "strong trend not detected");

    // Strength is scale-invariant (variance ratios).
    let y2: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();
    let s2 = seasonal_strength(&y2, 12).expect("strength runs");
    assert_eq!(s.seasonal_strength, s2.seasonal_strength);
    assert_eq!(s.trend_strength, s2.trend_strength);

    // Errors propagate unchanged.
    assert!(matches!(
        seasonal_strength(&y[..20], 12),
        Err(FiltersError::SeriesTooShort { .. })
    ));
}
