//! Input-validation error tests: every guard in the crate returns a typed
//! [`SpecTestError`] rather than panicking.

use tsecon_spectest::{
    breusch_pagan_test, chow_test, cusum_test, reset_test, white_test, SpecTestError,
    BG_MAX_AUX_CELLS, BG_WORK_BUDGET,
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

/// Audit round 16, A9: `max_power = 2**31` used to build fitted-value power
/// columns one at a time until the allocator gave up (SIGABRT after about
/// 50 s) — the degrees-of-freedom check ran only after the loop. It is now
/// refused at once, naming max_power and the largest that fits.
#[test]
fn reset_power_past_the_sample_is_refused_before_any_column() {
    let (y, x) = design(); // n = 20, k = 3: RESET needs max_power <= n - k = 17
    for max_power in [18usize, 5000, 1 << 31, 1 << 40, usize::MAX] {
        let err = reset_test(&y, &x, max_power).unwrap_err();
        assert_eq!(
            err,
            SpecTestError::ResetPowerTooLarge {
                max_power,
                n: 20,
                k: 3,
                max: Some(17),
                over_budget: false,
            }
        );
        let msg = err.to_string();
        assert!(
            msg.starts_with(&format!("max_power = {max_power} adds")),
            "{msg}"
        );
        assert!(msg.ends_with("; pass max_power <= 17"), "{msg}");
    }
    // The largest admissible power is accepted by the size check (17 powers
    // of a well-scaled fit are numerically collinear, so the fit may still
    // be refused as singular — but not as too large).
    match reset_test(&y, &x, 17) {
        Ok(_) | Err(SpecTestError::SingularDesign { .. }) => {}
        Err(e) => panic!("max_power = 17 refused as {e:?}"),
    }
    // With n = k + 1 not even max_power = 2 fits.
    let (y4, x4) = (
        y[..4].to_vec(),
        x.iter().map(|c| c[..4].to_vec()).collect::<Vec<_>>(),
    );
    let err = reset_test(&y4, &x4, 2).unwrap_err();
    assert!(matches!(
        err,
        SpecTestError::ResetPowerTooLarge { max: None, .. }
    ));
    assert!(err.to_string().contains("even max_power = 2 does not fit"));
}

#[test]
fn reset_power_past_the_budget_is_refused_by_name() {
    // n = 1e5 and max_power = 30000 leaves residual degrees of freedom but
    // the augmented design is 3e9 entries (24 GB): refused by the
    // Breusch-Godfrey auxiliary-regression budgets, quoting the largest
    // power within both.
    let n = 100_000;
    let x = vec![
        vec![1.0; n],
        (0..n).map(|t| (0.01 * t as f64).sin()).collect(),
    ];
    let y: Vec<f64> = (0..n)
        .map(|t| x[1][t] + 0.1 * ((t * 7919) % 13) as f64)
        .collect();
    let err = reset_test(&y, &x, 30_000).unwrap_err();
    let SpecTestError::ResetPowerTooLarge {
        max: Some(m),
        over_budget: true,
        ..
    } = err
    else {
        panic!("expected a budget refusal, got {err:?}");
    };
    let fits = |p: u128| {
        let n = n as u128;
        n * p <= BG_MAX_AUX_CELLS as u128 && n * p * p / 2 + 6 * p * p * p <= BG_WORK_BUDGET
    };
    // p = k + max_power - 1 regressors at the quoted maximum fit; one more does not.
    assert!(fits(2 + m as u128 - 1) && !fits(2 + m as u128), "max {m}");
    let msg = err.to_string();
    assert!(msg.contains("past both the memory budget"), "{msg}");
    assert!(msg.ends_with(&format!("; pass max_power <= {m}")), "{msg}");
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
