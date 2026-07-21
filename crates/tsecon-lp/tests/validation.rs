//! Input-validation tests ("errors that teach"): every guard on the
//! smooth-LP entry point returns a typed [`LpError`] whose Display names
//! what happened and what to try, never a panic.

use tsecon_lp::{smooth_lp, LpError, SmoothLpSpec};

/// A small well-formed pair of series.
fn series(n: usize) -> (Vec<f64>, Vec<f64>) {
    let e: Vec<f64> = (0..n)
        .map(|t| ((t * 37 + 11) % 17) as f64 / 8.5 - 1.0)
        .collect();
    let mut y = vec![0.0; n];
    for t in 1..n {
        y[t] = 0.6 * y[t - 1] + e[t] + 0.1 * ((t % 7) as f64 - 3.0);
    }
    (y, e)
}

#[test]
fn length_mismatch_rejected() {
    let (y, mut e) = series(80);
    e.pop();
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(6, 2)).unwrap_err();
    assert!(matches!(err, LpError::LengthMismatch { .. }));
    assert!(err.to_string().contains("index-aligned"));
}

#[test]
fn non_finite_rejected() {
    let (mut y, e) = series(80);
    y[10] = f64::NAN;
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(6, 2)).unwrap_err();
    assert!(matches!(err, LpError::NonFinite { .. }));
    assert!(
        err.to_string().contains("drop or"),
        "teaches the fix: {err}"
    );

    let (y2, mut e2) = series(80);
    e2[3] = f64::INFINITY;
    let err2 = smooth_lp(&y2, &e2, &SmoothLpSpec::new(6, 2)).unwrap_err();
    assert!(matches!(err2, LpError::NonFinite { .. }));
}

#[test]
fn short_series_and_long_horizon_rejected() {
    let (y, e) = series(4);
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(2, 6)).unwrap_err();
    assert!(matches!(err, LpError::SeriesTooShort { .. }));

    let (y, e) = series(20);
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(18, 2)).unwrap_err();
    assert!(matches!(err, LpError::HorizonTooLong { .. }));
    assert!(
        err.to_string().contains("shorten the maximum horizon"),
        "teaches the fix: {err}"
    );
}

#[test]
fn spline_config_violations_rejected_with_the_constraint_named() {
    let (y, e) = series(120);

    // degree 0 is refused.
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_degree(0)).unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
    assert!(err.to_string().contains("degree >= 1"), "{err}");

    // More basis functions than horizon points.
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_n_basis(12)).unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
    assert!(err.to_string().contains("n_basis <= horizons + 1"), "{err}");

    // Too few basis functions for the degree.
    let err = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2).with_degree(3).with_n_basis(3),
    )
    .unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
    assert!(err.to_string().contains("n_basis >= degree + 1"), "{err}");

    // Penalty order out of range.
    let err = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2)
            .with_n_basis(4)
            .with_penalty_order(4),
    )
    .unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
    assert!(err.to_string().contains("penalty_order < n_basis"), "{err}");

    // Horizons too short for the default cubic interpolating basis:
    // H = 2 gives K = 3 < degree + 1 = 4.
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(2, 2)).unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
}

#[test]
fn bad_lambda_rejected() {
    let (y, e) = series(120);
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_lambda(bad)).unwrap_err();
        assert!(matches!(err, LpError::InvalidLambda { .. }), "lambda {bad}");
        assert!(err.to_string().contains("cross-validation"), "{err}");
    }

    // A bad value hiding inside a CV grid is caught too.
    let err = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2).with_cv(Some(vec![1.0, -3.0]), 4),
    )
    .unwrap_err();
    assert!(matches!(err, LpError::InvalidLambda { .. }));
}

#[test]
fn empty_grid_rejected() {
    let (y, e) = series(120);
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_cv(Some(vec![]), 4)).unwrap_err();
    assert!(matches!(err, LpError::EmptyLambdaGrid));
    assert!(err.to_string().contains("default"), "{err}");
}

#[test]
fn infeasible_cv_folds_rejected() {
    let (y, e) = series(120);

    // One fold is not cross-validation.
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_cv(Some(vec![1.0]), 1)).unwrap_err();
    assert!(matches!(err, LpError::CvConfig { .. }));
    assert!(err.to_string().contains("n_folds"), "{err}");

    // More folds than usable base periods.
    let err = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2).with_cv(Some(vec![1.0]), 500),
    )
    .unwrap_err();
    assert!(matches!(err, LpError::CvConfig { .. }));
}

#[test]
fn constant_shock_is_a_singular_design_that_teaches() {
    // A constant shock makes the spline block collinear with the intercepts.
    let (y, _) = series(120);
    let e = vec![1.0; 120];
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(6, 2).with_lambda(1.0)).unwrap_err();
    assert!(matches!(err, LpError::Hac(_)), "got {err:?}");
    assert!(err.to_string().contains("collinear"), "{err}");
}
