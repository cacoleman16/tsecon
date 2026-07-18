//! Input-validation and range-estimator tests: every guard in
//! `measures.rs` returns a typed [`RealizedError`] rather than panicking or
//! -- worse -- silently returning a finite but meaningless number.
//!
//! Coverage motivation: before this file, `parkinson` and `garman_klass`
//! had **zero** Rust-side coverage (their only exercise was through the
//! Python `realized_range` wrapper), and none of the `TooFewObservations`
//! / `NonFinite` / `InvalidOhlc` guards in the crate were asserted on from
//! Rust at all. Each of those guards protects a silent-wrong-answer path:
//! a non-positive price feeds `ln` and yields NaN, and an inverted bar
//! (`high < low`) yields a perfectly finite, perfectly wrong variance,
//! because `(ln(H/L))^2` is squared and so loses the sign of the error.

use tsecon_realized::{
    bipower_variation, garman_klass, jump_component, parkinson, realized_quarticity,
    realized_variance, tripower_quarticity, RealizedError,
};

/// A well-formed two-bar OHLC block used as the base for the guard tests.
fn bars() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let open = vec![100.0, 101.0];
    let high = vec![102.0, 103.5];
    let low = vec![99.0, 100.5];
    let close = vec![101.0, 102.0];
    (open, high, low, close)
}

// ---------------------------------------------------------------------
// Closed-form values for the two range estimators.
// ---------------------------------------------------------------------

/// Parkinson (1980) is `(1/(4 ln 2)) sum (ln(H/L))^2`. Pinning it against
/// the formula written out by hand catches a wrong scale constant, which
/// is otherwise invisible: the estimator would still return a positive,
/// plausible-looking variance.
#[test]
fn parkinson_matches_closed_form() {
    let (_, high, low, _) = bars();
    let expected = (1.0 / (4.0 * 2.0_f64.ln()))
        * ((102.0_f64 / 99.0).ln().powi(2) + (103.5_f64 / 100.5).ln().powi(2));
    let got = parkinson(&high, &low).expect("well-formed bars");
    assert!(
        (got - expected).abs() <= 1e-15,
        "parkinson: got {got}, expected {expected}"
    );
}

/// Garman-Klass (1980) is
/// `sum [0.5 (ln(H/L))^2 - (2 ln 2 - 1)(ln(C/O))^2]`.
#[test]
fn garman_klass_matches_closed_form() {
    let (open, high, low, close) = bars();
    let c2 = 2.0 * 2.0_f64.ln() - 1.0;
    let mut expected = 0.0;
    for i in 0..2 {
        let ln_hl = (high[i] / low[i]).ln();
        let ln_co = (close[i] / open[i]).ln();
        expected += 0.5 * ln_hl * ln_hl - c2 * ln_co * ln_co;
    }
    let got = garman_klass(&open, &high, &low, &close).expect("well-formed bars");
    assert!(
        (got - expected).abs() <= 1e-15,
        "garman_klass: got {got}, expected {expected}"
    );
}

/// Both range estimators are nonnegative-by-construction on a bar with no
/// open-to-close move, and Garman-Klass then reduces to exactly half the
/// squared log range.
#[test]
fn garman_klass_reduces_to_half_squared_range_without_drift() {
    let open = vec![100.0];
    let close = vec![100.0];
    let high = vec![101.0];
    let low = vec![99.0];
    let got = garman_klass(&open, &high, &low, &close).expect("well-formed bar");
    let expected = 0.5 * (101.0_f64 / 99.0).ln().powi(2);
    assert!(
        (got - expected).abs() <= 1e-15,
        "got {got}, want {expected}"
    );
}

// ---------------------------------------------------------------------
// Parkinson guards.
// ---------------------------------------------------------------------

#[test]
fn parkinson_rejects_empty_input() {
    let err = parkinson(&[], &[]).unwrap_err();
    assert!(matches!(
        err,
        RealizedError::TooFewObservations {
            needed: 1,
            n: 0,
            ..
        }
    ));
}

#[test]
fn parkinson_rejects_length_mismatch() {
    let err = parkinson(&[102.0, 103.0], &[99.0]).unwrap_err();
    assert!(matches!(err, RealizedError::InvalidOhlc { .. }));
}

/// An inverted bar must be rejected, not squared away. Without the guard
/// `(ln(H/L))^2` is finite and positive, so the caller would receive a
/// wrong variance with no indication anything was amiss.
#[test]
fn parkinson_rejects_inverted_bar() {
    let high = vec![102.0, 99.0];
    let low = vec![99.0, 103.0];
    let err = parkinson(&high, &low).unwrap_err();
    match err {
        RealizedError::InvalidOhlc { index, detail, .. } => {
            assert_eq!(index, 1, "the second bar is the inverted one");
            assert!(detail.contains("below"), "detail was {detail:?}");
        }
        other => panic!("expected InvalidOhlc, got {other:?}"),
    }
}

/// A non-positive price would go into `ln` and produce NaN/-inf silently.
#[test]
fn parkinson_rejects_non_positive_price() {
    let err = parkinson(&[102.0, 100.0], &[99.0, 0.0]).unwrap_err();
    match err {
        RealizedError::InvalidOhlc { index, .. } => assert_eq!(index, 1),
        other => panic!("expected InvalidOhlc, got {other:?}"),
    }
    let err = parkinson(&[-1.0], &[-2.0]).unwrap_err();
    assert!(matches!(err, RealizedError::InvalidOhlc { index: 0, .. }));
}

#[test]
fn parkinson_rejects_non_finite() {
    let err = parkinson(&[102.0, f64::NAN], &[99.0, 100.0]).unwrap_err();
    assert!(matches!(err, RealizedError::NonFinite { index: 1, .. }));
    let err = parkinson(&[102.0, 103.0], &[99.0, f64::INFINITY]).unwrap_err();
    assert!(matches!(err, RealizedError::NonFinite { index: 1, .. }));
}

// ---------------------------------------------------------------------
// Garman-Klass guards.
// ---------------------------------------------------------------------

#[test]
fn garman_klass_rejects_empty_input() {
    let err = garman_klass(&[], &[], &[], &[]).unwrap_err();
    assert!(matches!(
        err,
        RealizedError::TooFewObservations {
            needed: 1,
            n: 0,
            ..
        }
    ));
}

/// Each of the three "other" series must be checked against `open.len()`;
/// a mismatch in any one of them would otherwise index out of bounds or
/// silently truncate.
#[test]
fn garman_klass_rejects_length_mismatch_in_each_series() {
    let (open, high, low, close) = bars();
    for (o, h, l, c) in [
        (&open[..], &high[..1], &low[..], &close[..]),
        (&open[..], &high[..], &low[..1], &close[..]),
        (&open[..], &high[..], &low[..], &close[..1]),
    ] {
        let err = garman_klass(o, h, l, c).unwrap_err();
        assert!(
            matches!(err, RealizedError::InvalidOhlc { .. }),
            "expected InvalidOhlc, got {err:?}"
        );
    }
}

#[test]
fn garman_klass_rejects_inverted_bar() {
    let (open, _, _, close) = bars();
    let high = vec![102.0, 100.0];
    let low = vec![99.0, 101.0];
    let err = garman_klass(&open, &high, &low, &close).unwrap_err();
    match err {
        RealizedError::InvalidOhlc { index, detail, .. } => {
            assert_eq!(index, 1);
            assert!(detail.contains("below"), "detail was {detail:?}");
        }
        other => panic!("expected InvalidOhlc, got {other:?}"),
    }
}

/// `open`/`close` get their own positivity guard (separate from the
/// shared high/low one), because they feed the `ln(C/O)` term.
#[test]
fn garman_klass_rejects_non_positive_open_or_close() {
    let (_, high, low, _) = bars();
    let err = garman_klass(&[100.0, 0.0], &high, &low, &[101.0, 102.0]).unwrap_err();
    match err {
        RealizedError::InvalidOhlc { index, detail, .. } => {
            assert_eq!(index, 1);
            assert!(detail.contains("open and close"), "detail was {detail:?}");
        }
        other => panic!("expected InvalidOhlc, got {other:?}"),
    }
    let err = garman_klass(&[100.0, 101.0], &high, &low, &[101.0, -1.0]).unwrap_err();
    assert!(matches!(err, RealizedError::InvalidOhlc { index: 1, .. }));
}

#[test]
fn garman_klass_rejects_non_positive_high_or_low() {
    let err = garman_klass(&[100.0], &[0.0], &[0.0], &[101.0]).unwrap_err();
    match err {
        RealizedError::InvalidOhlc { index, detail, .. } => {
            assert_eq!(index, 0);
            assert!(detail.contains("high and low"), "detail was {detail:?}");
        }
        other => panic!("expected InvalidOhlc, got {other:?}"),
    }
}

#[test]
fn garman_klass_rejects_non_finite_in_each_series() {
    let (open, high, low, close) = bars();
    let nan = [f64::NAN, 1.0];
    for (o, h, l, c) in [
        (&nan[..], &high[..], &low[..], &close[..]),
        (&open[..], &nan[..], &low[..], &close[..]),
        (&open[..], &high[..], &nan[..], &close[..]),
        (&open[..], &high[..], &low[..], &nan[..]),
    ] {
        let err = garman_klass(o, h, l, c).unwrap_err();
        assert!(
            matches!(err, RealizedError::NonFinite { index: 0, .. }),
            "expected NonFinite, got {err:?}"
        );
    }
}

// ---------------------------------------------------------------------
// Return-based measure guards.
// ---------------------------------------------------------------------

/// Each measure states its own minimum sample; an off-by-one here would
/// let a window estimator run on a sample too short to identify it.
#[test]
fn return_measures_enforce_their_minimum_sample() {
    assert!(matches!(
        realized_variance(&[]).unwrap_err(),
        RealizedError::TooFewObservations {
            needed: 1,
            n: 0,
            ..
        }
    ));
    assert!(matches!(
        realized_quarticity(&[]).unwrap_err(),
        RealizedError::TooFewObservations {
            needed: 1,
            n: 0,
            ..
        }
    ));
    assert!(matches!(
        bipower_variation(&[0.01]).unwrap_err(),
        RealizedError::TooFewObservations {
            needed: 2,
            n: 1,
            ..
        }
    ));
    assert!(matches!(
        tripower_quarticity(&[0.01, -0.02]).unwrap_err(),
        RealizedError::TooFewObservations {
            needed: 3,
            n: 2,
            ..
        }
    ));
    // ... and each accepts exactly its stated minimum.
    assert!(realized_variance(&[0.01]).is_ok());
    assert!(realized_quarticity(&[0.01]).is_ok());
    assert!(bipower_variation(&[0.01, -0.02]).is_ok());
    assert!(tripower_quarticity(&[0.01, -0.02, 0.03]).is_ok());
}

/// Realized measures must never skip a NaN silently: a dropped
/// observation biases the variance downward without any error surfacing.
#[test]
fn return_measures_reject_non_finite() {
    let bad = [0.01, f64::NAN, -0.02];
    for err in [
        realized_variance(&bad).unwrap_err(),
        bipower_variation(&bad).unwrap_err(),
        realized_quarticity(&bad).unwrap_err(),
        tripower_quarticity(&bad).unwrap_err(),
        jump_component(&bad).unwrap_err(),
    ] {
        match err {
            RealizedError::NonFinite { index, value, .. } => {
                assert_eq!(index, 1, "the NaN is the second observation");
                assert!(value.is_nan(), "offending value should be reported back");
            }
            other => panic!("expected NonFinite, got {other:?}"),
        }
    }
    assert!(matches!(
        realized_variance(&[0.01, f64::INFINITY]).unwrap_err(),
        RealizedError::NonFinite { index: 1, .. }
    ));
}

/// `jump_component` delegates to both `realized_variance` and
/// `bipower_variation`, so it inherits the stricter (two-observation)
/// minimum rather than the one-observation minimum of RV alone.
#[test]
fn jump_component_inherits_the_stricter_minimum() {
    assert!(matches!(
        jump_component(&[]).unwrap_err(),
        RealizedError::TooFewObservations { needed: 1, .. }
    ));
    assert!(matches!(
        jump_component(&[0.01]).unwrap_err(),
        RealizedError::TooFewObservations {
            needed: 2,
            n: 1,
            ..
        }
    ));
}

/// The jump component is floored at zero: on a smooth path where
/// `BV > RV` the raw difference is negative and must not leak through.
#[test]
fn jump_component_is_floored_at_zero() {
    // Alternating equal-magnitude returns make BV = (pi/2) * RV > RV.
    let r = [0.01, -0.01, 0.01, -0.01, 0.01];
    let rv = realized_variance(&r).expect("ok");
    let bv = bipower_variation(&r).expect("ok");
    assert!(bv > rv, "construction check: BV {bv} should exceed RV {rv}");
    assert_eq!(jump_component(&r).expect("ok"), 0.0);
}
