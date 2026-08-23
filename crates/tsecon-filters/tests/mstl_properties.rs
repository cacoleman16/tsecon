//! Property tests for MSTL: exact reconstruction from all components,
//! period sorting (input order must not matter), the period >= n/2 drop
//! rule with honest reporting, the single-period degeneration to plain
//! STL, forwarding of the STL knobs, the per-period strength measures,
//! and the errors the degenerate inputs must raise.

// Elementwise assertions over parallel component arrays read most clearly
// as index loops.
#![allow(clippy::needless_range_loop)]

use tsecon_filters::{
    mstl, mstl_seasonal_strengths, stl, FiltersError, MstlParams, StlParams,
};

/// A deterministic two-seasonal test series (periods 6 and 21, n = 252 —
/// no RNG dependency): trend + two stable seasonals + a small
/// deterministic wobble.
fn two_seasonal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let tf = t as f64;
            8.0 + 0.03 * tf
                + 1.5 * (2.0 * std::f64::consts::PI * tf / 6.0).sin()
                + 2.0 * (2.0 * std::f64::consts::PI * tf / 21.0).sin()
                + 0.2 * (tf * 0.7134).sin() * (tf * 0.2718).cos()
        })
        .collect()
}

#[test]
fn components_reconstruct_y() {
    let y = two_seasonal(252);
    let r = mstl(&y, &[6, 21], &MstlParams::default()).expect("mstl runs");
    assert_eq!(r.seasonal.len(), 2);
    assert_eq!(r.iterate, 2, "two periods keep the default iterate");
    for i in 0..y.len() {
        let recon = r.seasonal[0][i] + r.seasonal[1][i] + r.trend[i] + r.resid[i];
        assert!(
            (recon - y[i]).abs() <= 1e-10 * y[i].abs().max(1.0),
            "reconstruction failed at {i}: {recon} vs {}",
            y[i]
        );
    }
}

#[test]
fn period_order_does_not_matter() {
    let y = two_seasonal(252);
    let a = mstl(&y, &[6, 21], &MstlParams::default()).expect("sorted runs");
    let b = mstl(&y, &[21, 6], &MstlParams::default()).expect("unsorted runs");
    assert_eq!(a.periods, vec![6, 21], "periods sorted ascending");
    assert_eq!(b.periods, vec![6, 21], "input order normalized away");
    assert_eq!(a.windows, vec![11, 15], "default windows follow sorted order");
    assert_eq!(a.seasonal, b.seasonal, "components must match bitwise");
    assert_eq!(a.trend, b.trend);
    assert_eq!(a.resid, b.resid);

    // Explicit windows travel WITH their period through the sort.
    let c = mstl(
        &y,
        &[21, 6],
        &MstlParams {
            windows: Some(vec![15, 11]),
            ..Default::default()
        },
    )
    .expect("paired runs");
    assert_eq!(c.windows, vec![11, 15]);
    assert_eq!(c.seasonal, a.seasonal, "paired (period, window) sort matches");
}

#[test]
fn long_periods_are_dropped_and_reported() {
    let y = two_seasonal(252);
    // 126 >= 252/2 is dropped; 6 stays. The result must equal a plain
    // [6]-only run (which forces iterate = 1) bitwise.
    let r = mstl(&y, &[6, 126], &MstlParams::default()).expect("mstl runs");
    assert_eq!(r.periods, vec![6]);
    assert_eq!(r.windows, vec![11]);
    assert_eq!(r.dropped_periods, vec![126]);
    assert_eq!(r.iterate, 1, "one surviving period forces one round");
    let only = mstl(&y, &[6], &MstlParams::default()).expect("single runs");
    assert_eq!(r.seasonal, only.seasonal);
    assert_eq!(r.trend, only.trend);
    // Boundary: period = n/2 exactly is dropped (the statsmodels rule is
    // period >= n/2), while n/2 - 1 survives.
    let r = mstl(&y[..48], &[24], &MstlParams::default());
    assert!(matches!(r, Err(FiltersError::SeriesTooShort { .. })));
    assert!(mstl(&y[..50], &[24], &MstlParams::default()).is_ok());
}

#[test]
fn single_period_equals_stl_with_window_11() {
    let y = two_seasonal(144);
    let m = mstl(&y, &[6], &MstlParams::default()).expect("mstl runs");
    let s = stl(
        &y,
        6,
        &StlParams {
            seasonal: 11,
            ..Default::default()
        },
    )
    .expect("stl runs");
    assert_eq!(m.seasonal[0], s.seasonal, "seasonal must match bitwise");
    assert_eq!(m.trend, s.trend, "trend must match bitwise");
    assert_eq!(m.resid, s.resid, "resid must match bitwise");
}

#[test]
fn stl_knobs_are_forwarded() {
    let mut y = two_seasonal(252);
    y[40] += 9.0;
    // robust=true must change the answer and produce a downweighted point;
    // the explicit inner/outer equivalent must reproduce it bitwise (the
    // same nesting STL itself guarantees).
    let base = mstl(&y, &[6, 21], &MstlParams::default()).expect("base runs");
    let robust = mstl(
        &y,
        &[6, 21],
        &MstlParams {
            stl: StlParams {
                robust: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("robust runs");
    let explicit = mstl(
        &y,
        &[6, 21],
        &MstlParams {
            stl: StlParams {
                inner_iter: Some(2),
                outer_iter: Some(15),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("explicit runs");
    assert_eq!(robust.seasonal, explicit.seasonal, "robust == inner 2/outer 15");
    assert_eq!(robust.weights, explicit.weights);
    assert!(
        robust.weights[40] < 0.5,
        "outlier weight {} not downweighted",
        robust.weights[40]
    );
    assert!(base.weights.iter().all(|&w| w == 1.0), "non-robust weights are 1");
    assert_ne!(robust.trend, base.trend, "robustness must change the fit");

    // The ignored `stl.seasonal` field really is ignored (windows rule):
    let odd_seasonal = mstl(
        &y,
        &[6, 21],
        &MstlParams {
            stl: StlParams {
                seasonal: 35,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("runs");
    assert_eq!(odd_seasonal.windows, vec![11, 15]);
    assert_eq!(odd_seasonal.seasonal, base.seasonal, "stl.seasonal is ignored");
}

#[test]
fn constant_series_decomposes_flat() {
    let y = vec![5.0; 252];
    let r = mstl(&y, &[6, 21], &MstlParams::default()).expect("constant runs");
    for k in 0..2 {
        for i in 0..y.len() {
            assert!(
                r.seasonal[k][i].abs() <= 1e-10,
                "seasonal[{k}][{i}] not ~0"
            );
        }
    }
    for i in 0..y.len() {
        assert!((r.trend[i] - 5.0).abs() <= 1e-10, "trend[{i}] not ~5");
    }
}

// ------------------------------------------------------ degenerate inputs

#[test]
fn empty_and_duplicate_periods_raise() {
    let y = two_seasonal(252);
    match mstl(&y, &[], &MstlParams::default()) {
        Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, "periods"),
        other => panic!("empty periods: expected InvalidParameter, got {other:?}"),
    }
    match mstl(&y, &[6, 21, 6], &MstlParams::default()) {
        Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, "periods"),
        other => panic!("duplicate periods: expected InvalidParameter, got {other:?}"),
    }
    // A period below 2 that survives the drop rule is rejected by name.
    match mstl(&y, &[1, 21], &MstlParams::default()) {
        Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, "periods"),
        other => panic!("period 1: expected InvalidParameter, got {other:?}"),
    }
}

#[test]
fn all_periods_dropped_raises_series_too_short() {
    let y = two_seasonal(40);
    match mstl(&y, &[20, 30], &MstlParams::default()) {
        Err(FiltersError::SeriesTooShort { filter, needed, got, .. }) => {
            assert_eq!(filter, "mstl");
            assert_eq!(needed, 41, "2 * min(period) + 1");
            assert_eq!(got, 40);
        }
        other => panic!("expected SeriesTooShort, got {other:?}"),
    }
    // Empty input trips the same guard (every period >= 0/2 = 0).
    assert!(matches!(
        mstl(&[], &[12], &MstlParams::default()),
        Err(FiltersError::SeriesTooShort { .. })
    ));
}

#[test]
fn bad_windows_and_iterate_raise() {
    let y = two_seasonal(252);
    // Length mismatch.
    match mstl(
        &y,
        &[6, 21],
        &MstlParams {
            windows: Some(vec![11]),
            ..Default::default()
        },
    ) {
        Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, "windows"),
        other => panic!("window length: expected InvalidParameter, got {other:?}"),
    }
    // Even / too-small windows.
    for bad in [vec![10, 15], vec![1, 15]] {
        match mstl(
            &y,
            &[6, 21],
            &MstlParams {
                windows: Some(bad),
                ..Default::default()
            },
        ) {
            Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, "windows"),
            other => panic!("bad window: expected InvalidParameter, got {other:?}"),
        }
    }
    // iterate = 0 (statsmodels would crash with NameError).
    match mstl(
        &y,
        &[6, 21],
        &MstlParams {
            iterate: 0,
            ..Default::default()
        },
    ) {
        Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, "iterate"),
        other => panic!("iterate 0: expected InvalidParameter, got {other:?}"),
    }
    // Forwarded STL knobs keep their own validation (named after the knob).
    match mstl(
        &y,
        &[6, 21],
        &MstlParams {
            stl: StlParams {
                trend: Some(11), // <= period 21 once the second pass runs
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Err(FiltersError::InvalidParameter { name, .. }) => assert_eq!(name, "trend"),
        other => panic!("bad trend window: expected InvalidParameter, got {other:?}"),
    }
}

#[test]
fn non_finite_input_raises() {
    let mut y = two_seasonal(252);
    y[10] = f64::NAN;
    assert!(matches!(
        mstl(&y, &[6, 21], &MstlParams::default()),
        Err(FiltersError::NonFiniteInput { index: 10 })
    ));
}

// ------------------------------------------------------- strength measures

#[test]
fn per_period_strengths_are_sane() {
    let y = two_seasonal(252);
    let r = mstl(&y, &[6, 21], &MstlParams::default()).expect("mstl runs");
    let s = mstl_seasonal_strengths(&r);
    assert_eq!(s.len(), 2, "one strength per period");
    for (k, &v) in s.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "strength[{k}] = {v} outside [0, 1]"
        );
        assert!(
            v > 0.9,
            "clean deterministic seasonal {k} should be strong, got {v}"
        );
    }
    // The formula is the same guarded one as strength_from_components: on
    // a single-period fit the two agree exactly.
    let m1 = mstl(&y, &[6], &MstlParams::default()).expect("single runs");
    let s1 = mstl_seasonal_strengths(&m1);
    let stl_fit = stl(
        &y,
        6,
        &StlParams {
            seasonal: 11,
            ..Default::default()
        },
    )
    .expect("stl runs");
    let reference = tsecon_filters::strength_from_components(&stl_fit);
    assert_eq!(s1.len(), 1);
    assert_eq!(s1[0], reference.seasonal_strength, "same formula, same bits");
}
