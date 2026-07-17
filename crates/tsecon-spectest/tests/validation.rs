//! Input-validation error tests: every guard in the crate returns a typed
//! [`SpecTestError`] rather than panicking.

use tsecon_spectest::{
    breusch_pagan_test, chow_test, cusum_test, reset_test, white_test, SpecTestError,
};

/// A small well-formed design: const + two regressors, `n = 20`.
fn design() -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = 20;
    let cst = vec![1.0; n];
    let x1: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
    let x2: Vec<f64> = (0..n).map(|t| (0.17 * t as f64).cos()).collect();
    let y: Vec<f64> = (0..n)
        .map(|t| 0.5 + 0.4 * x1[t] - 0.2 * x2[t] + 0.05 * (t % 5) as f64)
        .collect();
    (y, vec![cst, x1, x2])
}

#[test]
fn empty_response_rejected() {
    let (_, x) = design();
    let err = white_test(&[], &x).unwrap_err();
    assert!(matches!(err, SpecTestError::EmptyInput { .. }));
}

#[test]
fn no_regressors_rejected() {
    let (y, _) = design();
    let err = white_test(&y, &[]).unwrap_err();
    assert!(matches!(err, SpecTestError::NoRegressors));
}

#[test]
fn dimension_mismatch_rejected() {
    let (y, mut x) = design();
    x[1].push(0.0); // one column too long
    let err = breusch_pagan_test(&y, &x).unwrap_err();
    assert!(matches!(err, SpecTestError::DimensionMismatch { .. }));
}

#[test]
fn non_finite_rejected() {
    let (mut y, x) = design();
    y[3] = f64::NAN;
    let err = reset_test(&y, &x, 3).unwrap_err();
    assert!(matches!(err, SpecTestError::NonFinite { .. }));

    let (y2, mut x2) = design();
    x2[2][1] = f64::INFINITY;
    let err2 = cusum_test(&y2, &x2).unwrap_err();
    assert!(matches!(err2, SpecTestError::NonFinite { .. }));
}

#[test]
fn missing_constant_rejected_by_het_tests() {
    // A design with no constant column: White / Breusch-Pagan need one.
    let n = 20;
    let x1: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
    let x2: Vec<f64> = (0..n).map(|t| 1.0 + (0.17 * t as f64).cos()).collect();
    let y: Vec<f64> = (0..n).map(|t| 0.4 * x1[t] - 0.2 * x2[t]).collect();
    let x = vec![x1, x2];

    assert!(matches!(
        white_test(&y, &x).unwrap_err(),
        SpecTestError::MissingConstant { .. }
    ));
    assert!(matches!(
        breusch_pagan_test(&y, &x).unwrap_err(),
        SpecTestError::MissingConstant { .. }
    ));
}

#[test]
fn reset_invalid_power_rejected() {
    let (y, x) = design();
    assert!(matches!(
        reset_test(&y, &x, 1).unwrap_err(),
        SpecTestError::InvalidPower { max_power: 1 }
    ));
}

#[test]
fn chow_invalid_split_rejected() {
    let (y, x) = design(); // n = 20, k = 3
                           // split too small: first regime not estimable.
    assert!(matches!(
        chow_test(&y, &x, 2).unwrap_err(),
        SpecTestError::InvalidSplit { .. }
    ));
    // split too large: second regime not estimable.
    assert!(matches!(
        chow_test(&y, &x, 18).unwrap_err(),
        SpecTestError::InvalidSplit { .. }
    ));
    // A valid split in the middle succeeds.
    assert!(chow_test(&y, &x, 10).is_ok());
}

#[test]
fn degrees_of_freedom_rejected() {
    // n = 3, k = 3 -> n <= k, no residual dof.
    let cst = vec![1.0; 3];
    let x1 = vec![0.0, 1.0, 2.0];
    let x2 = vec![0.0, 1.0, 4.0];
    let y = vec![1.0, 2.0, 1.5];
    let x = vec![cst, x1, x2];
    assert!(matches!(
        cusum_test(&y, &x).unwrap_err(),
        SpecTestError::DegreesOfFreedom { .. }
    ));
    assert!(matches!(
        white_test(&y, &x).unwrap_err(),
        SpecTestError::DegreesOfFreedom { .. }
    ));
}

#[test]
fn singular_design_rejected() {
    // Two identical regressors (besides the constant) -> collinear design.
    let n = 20;
    let cst = vec![1.0; n];
    let x1: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
    let x2 = x1.clone();
    let y: Vec<f64> = (0..n).map(|t| 0.5 + 0.4 * x1[t]).collect();
    let x = vec![cst, x1, x2];
    assert!(matches!(
        reset_test(&y, &x, 3).unwrap_err(),
        SpecTestError::SingularDesign { .. }
    ));
}
