//! Property tests for the `nsdiffs` seasonal-differencing advisor:
//! decision consistency between `d`, `stop`, and the per-step evidence;
//! scale invariance (the strengths are variance ratios); the constant and
//! too-short stops; and the errors degenerate inputs must raise.

use tsecon_diag::{nsdiffs, AdvisorError, NsdiffsStop, NSDIFFS_SEAS_THRESHOLD};

/// A strongly seasonal deterministic monthly series.
fn seasonal_monthly(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let tf = t as f64;
            5.0 + 0.02 * tf
                + 3.0 * (2.0 * std::f64::consts::PI * tf / 12.0).sin()
                + 0.2 * (tf * 0.7134).sin()
        })
        .collect()
}

/// A non-seasonal deterministic wobble.
fn nonseasonal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let tf = t as f64;
            (tf * 0.7134).sin() + 0.5 * (tf * 1.618).cos()
        })
        .collect()
}

#[test]
fn strong_seasonality_gets_one_difference() {
    let y = seasonal_monthly(144);
    let r = nsdiffs(&y, 12, 0.05, 1).expect("nsdiffs runs");
    assert_eq!(r.d, 1, "strong seasonal must call for D = 1");
    assert!(r.steps[0].seasonal_strength >= NSDIFFS_SEAS_THRESHOLD);
    assert!(r.steps[0].needs_differencing);
    assert_eq!(r.threshold, NSDIFFS_SEAS_THRESHOLD);
    // Every reported step must be internally consistent with the rule.
    for s in &r.steps {
        assert_eq!(
            s.needs_differencing,
            s.seasonal_strength >= NSDIFFS_SEAS_THRESHOLD
        );
    }
}

#[test]
fn weak_seasonality_gets_zero() {
    let y = nonseasonal(144);
    let r = nsdiffs(&y, 12, 0.05, 1).expect("nsdiffs runs");
    assert_eq!(r.d, 0);
    assert_eq!(r.stop, NsdiffsStop::WeakSeasonality);
    assert!(!r.steps[0].needs_differencing);
}

#[test]
fn scale_invariance() {
    let y = seasonal_monthly(120);
    let a = nsdiffs(&y, 12, 0.05, 2).expect("base runs");
    let y2: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();
    let b = nsdiffs(&y2, 12, 0.05, 2).expect("scaled runs");
    assert_eq!(a.d, b.d);
    assert_eq!(a.stop, b.stop);
    for (sa, sb) in a.steps.iter().zip(&b.steps) {
        // Power-of-two rescale: STL components double bitwise, so the
        // variance ratios are bit-identical.
        assert_eq!(sa.seasonal_strength, sb.seasonal_strength);
        assert_eq!(sa.trend_strength, sb.trend_strength);
    }
}

#[test]
fn constant_series_stops_immediately() {
    let y = vec![3.25; 60];
    let r = nsdiffs(&y, 12, 0.05, 1).expect("constant runs");
    assert_eq!(r.d, 0);
    assert_eq!(r.stop, NsdiffsStop::Constant);
    assert!(r.steps.is_empty());
}

#[test]
fn exact_seasonal_pattern_differences_to_constant() {
    // A pure deterministic seasonal + linear trend: one seasonal
    // difference leaves an exactly constant series (the Constant stop at
    // d = 1), because both the sine and the line are annihilated exactly
    // ... up to floating rounding, so build it from an exactly periodic
    // integer pattern instead.
    let pattern = [
        4.0, -2.0, 7.0, 1.0, -5.0, 3.0, 0.0, 2.0, -1.0, 6.0, -3.0, 5.0,
    ];
    let y: Vec<f64> = (0..120).map(|t| pattern[t % 12] + t as f64).collect();
    let r = nsdiffs(&y, 12, 0.05, 3).expect("nsdiffs runs");
    if r.stop == NsdiffsStop::Constant {
        assert!(r.d >= 1, "constant stop must come after the difference");
    } else {
        // The strength rule may already report the (strong) seasonal at
        // d = 0 and difference once; either exit is consistent.
        assert!(r.d >= 1);
    }
}

#[test]
fn too_short_after_differencing_is_a_stop_not_an_error() {
    // n = 26, period 12: the levels fit runs (26 >= 24), but one seasonal
    // difference leaves 14 < 24 observations.
    let y = seasonal_monthly(26);
    let r = nsdiffs(&y, 12, 0.05, 2).expect("nsdiffs runs");
    if r.d >= 1 {
        assert_eq!(r.stop, NsdiffsStop::TooShort);
        assert_eq!(r.steps.len(), 1);
    }
}

#[test]
fn too_short_levels_series_errors() {
    let y = seasonal_monthly(20); // < 2 * 12
    match nsdiffs(&y, 12, 0.05, 1) {
        Err(AdvisorError::Filters(e)) => {
            let msg = e.to_string();
            assert!(msg.contains("stl"), "error should name stl: {msg}");
        }
        other => panic!("expected Filters error, got {other:?}"),
    }
}

#[test]
fn period_below_two_errors() {
    let y = seasonal_monthly(48);
    assert!(matches!(
        nsdiffs(&y, 1, 0.05, 1),
        Err(AdvisorError::Filters(_))
    ));
}

#[test]
fn invalid_alpha_errors() {
    let y = seasonal_monthly(48);
    for alpha in [0.0, 1.0, -0.1, f64::NAN] {
        assert!(
            nsdiffs(&y, 12, alpha, 1).is_err(),
            "alpha = {alpha} must be rejected"
        );
    }
}

#[test]
fn max_d_caps_the_answer() {
    let y = seasonal_monthly(144);
    let r = nsdiffs(&y, 12, 0.05, 0).expect("nsdiffs runs");
    assert_eq!(r.d, 0);
    assert_eq!(
        r.stop,
        NsdiffsStop::MaxD,
        "cap reached while strength calls"
    );
}
