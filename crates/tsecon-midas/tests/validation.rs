//! Validation and equivalence tests for the two OLS-based MIDAS
//! estimators.
//!
//! Coverage motivation: `adl_midas` -- a documented, publicly exported
//! estimator -- had **zero** Rust-side coverage. Nothing pinned its
//! parameter ordering (`[c, rho_1..rho_P, b_1..b_K]`), which is the kind
//! of mistake that returns a full set of plausible numbers with the
//! autoregressive and high-frequency blocks transposed. The
//! `InvalidLagCount` guards on both estimators and the constant-target
//! `rsquared = NaN` path were likewise unasserted.

use tsecon_midas::{adl_midas, umidas, MidasError, SeType};

/// A deterministic design: `y` is an exact linear function of one own-lag
/// and two high-frequency columns, so OLS must recover the coefficients
/// to machine precision.
fn exact_design() -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = 24;
    let x1: Vec<f64> = (0..n).map(|t| (0.31 * t as f64).sin()).collect();
    let x2: Vec<f64> = (0..n).map(|t| (0.17 * t as f64).cos()).collect();
    let ylag: Vec<f64> = (0..n).map(|t| 0.05 * (t % 7) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|t| 1.5 + 0.4 * ylag[t] - 0.7 * x1[t] + 0.25 * x2[t])
        .collect();
    (y, vec![ylag], vec![x1, x2])
}

/// ADL-MIDAS must lay its parameters out as `[c, rho_1..rho_P,
/// b_1..b_K]`. On a design with an exact linear DGP the coefficients are
/// identified exactly, so a swapped block shows up immediately.
#[test]
fn adl_midas_recovers_exact_coefficients_in_documented_order() {
    let (y, y_lags, hf_lags) = exact_design();
    let fit = adl_midas(&y, &y_lags, &hf_lags, SeType::NonRobust).expect("well-posed design");
    assert_eq!(fit.params.len(), 4, "c + 1 own-lag + 2 hf lags");
    let expected = [1.5, 0.4, -0.7, 0.25];
    for (i, (&got, &want)) in fit.params.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-9,
            "param[{i}]: got {got}, want {want}"
        );
    }
    assert!(
        (fit.rsquared - 1.0).abs() <= 1e-12,
        "an exact linear DGP must give R^2 = 1, got {}",
        fit.rsquared
    );
}

/// With no own-lags, ADL-MIDAS is exactly U-MIDAS. This pins the design
/// assembly of the two entry points against each other: if `adl_midas`
/// pushed its blocks in the wrong order or dropped the intercept, the two
/// would diverge.
#[test]
fn adl_midas_without_own_lags_equals_umidas() {
    let (y, _, hf_lags) = exact_design();
    let adl = adl_midas(&y, &[], &hf_lags, SeType::NonRobust).expect("ok");
    let u = umidas(&y, &hf_lags, SeType::NonRobust).expect("ok");
    assert_eq!(adl.params.len(), u.params.len());
    for (i, (&a, &b)) in adl.params.iter().zip(u.params.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-12, "param[{i}]: {a} vs {b}");
    }
    for (i, (&a, &b)) in adl.bse.iter().zip(u.bse.iter()).enumerate() {
        assert!((a - b).abs() <= 1e-12, "bse[{i}]: {a} vs {b}");
    }
}

/// ADL-MIDAS with only own-lags is a pure autoregression -- a legitimate
/// configuration that must not trip the "no regressors" guard.
#[test]
fn adl_midas_accepts_own_lags_only() {
    let (y, y_lags, _) = exact_design();
    let fit = adl_midas(&y, &y_lags, &[], SeType::NonRobust).expect("AR-only is well posed");
    assert_eq!(fit.params.len(), 2, "intercept + one own-lag");
}

/// Both estimators must refuse a design with no regressors at all rather
/// than silently fitting an intercept-only model and reporting it as a
/// MIDAS regression.
#[test]
fn empty_lag_sets_are_rejected() {
    let (y, _, _) = exact_design();
    assert!(matches!(
        umidas(&y, &[], SeType::NonRobust).unwrap_err(),
        MidasError::InvalidLagCount {
            k: 0,
            needed: 1,
            ..
        }
    ));
    assert!(matches!(
        adl_midas(&y, &[], &[], SeType::NonRobust).unwrap_err(),
        MidasError::InvalidLagCount {
            k: 0,
            needed: 1,
            ..
        }
    ));
}

/// A numerically constant target has zero total sum of squares, so the
/// centered R^2 is undefined. It must come back as NaN rather than as a
/// division-by-zero infinity or a spurious 1.0 that would advertise a
/// perfect fit.
#[test]
fn constant_target_gives_nan_rsquared_not_a_spurious_perfect_fit() {
    let (_, _, hf_lags) = exact_design();
    let y = vec![3.0; 24];
    let fit = umidas(&y, &hf_lags, SeType::NonRobust).expect("design is full rank");
    assert!(
        fit.rsquared.is_nan(),
        "constant target must give NaN R^2, got {}",
        fit.rsquared
    );
    let fit = adl_midas(&y, &[], &hf_lags, SeType::NonRobust).expect("ok");
    assert!(fit.rsquared.is_nan(), "got {}", fit.rsquared);
}

/// A column whose length does not match `y` must be rejected by the
/// shared OLS engine rather than silently truncating the sample.
#[test]
fn mismatched_column_length_is_rejected() {
    let (y, _, hf_lags) = exact_design();
    let short = vec![hf_lags[0][..10].to_vec()];
    assert!(umidas(&y, &short, SeType::NonRobust).is_err());
    assert!(adl_midas(&y, &[], &short, SeType::NonRobust).is_err());
}

/// A perfectly collinear design has no unique OLS solution; the estimator
/// must surface that instead of returning an arbitrary point on the
/// solution manifold.
#[test]
fn collinear_design_is_rejected() {
    let (y, _, hf_lags) = exact_design();
    let duplicated = vec![hf_lags[0].clone(), hf_lags[0].clone()];
    assert!(
        umidas(&y, &duplicated, SeType::NonRobust).is_err(),
        "duplicate regressor columns must not solve"
    );
}
