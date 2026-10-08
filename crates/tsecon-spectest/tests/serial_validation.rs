//! Input-validation tests for the serial-correlation tests: every malformed
//! input is a typed [`SpecTestError`], never a panic.

use tsecon_spectest::{
    breusch_godfrey_test, durbin_watson_statistic, durbin_watson_test, DwAlternative,
    SpecTestError, DW_EXACT_MAX_NOBS,
};

/// const + one regressor, `n` observations, non-degenerate residuals.
fn design(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let x1: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|t| 0.5 + 0.4 * x1[t] + 0.1 * ((t * t) % 7) as f64)
        .collect();
    (y, vec![vec![1.0; n], x1])
}

const POS: DwAlternative = DwAlternative::Positive;

#[test]
fn empty_and_non_finite_inputs_are_refused() {
    let (y, x) = design(20);
    assert!(matches!(
        breusch_godfrey_test(&[], &x, None).unwrap_err(),
        SpecTestError::EmptyInput { .. }
    ));
    assert!(matches!(
        durbin_watson_test(&[], &x, POS).unwrap_err(),
        SpecTestError::EmptyInput { .. }
    ));
    assert!(matches!(
        breusch_godfrey_test(&y, &[], Some(1)).unwrap_err(),
        SpecTestError::NoRegressors
    ));
    let mut bad = y.clone();
    bad[4] = f64::NAN;
    assert!(matches!(
        breusch_godfrey_test(&bad, &x, None).unwrap_err(),
        SpecTestError::NonFinite { .. }
    ));
    let mut xb = x.clone();
    xb[1][2] = f64::INFINITY;
    assert!(matches!(
        durbin_watson_test(&y, &xb, POS).unwrap_err(),
        SpecTestError::NonFinite { .. }
    ));
    let mut short = x;
    short[1].pop();
    assert!(matches!(
        breusch_godfrey_test(&y, &short, None).unwrap_err(),
        SpecTestError::DimensionMismatch { .. }
    ));
    assert!(durbin_watson_statistic(&[1.0]).is_err());
    assert!(durbin_watson_statistic(&[0.0, 0.0]).is_err());
    assert!(durbin_watson_statistic(&[1.0, f64::NAN]).is_err());
}

#[test]
fn breusch_godfrey_order_errors() {
    let (y, x) = design(20);
    // Explicit zero.
    assert!(matches!(
        breusch_godfrey_test(&y, &x, Some(0)).unwrap_err(),
        SpecTestError::InvalidLags {
            nlags: 0,
            defaulted: false,
            ..
        }
    ));
    // Default of zero on a sample shorter than five.
    let (y4, x4) = design(4);
    assert!(matches!(
        breusch_godfrey_test(&y4, &x4, None).unwrap_err(),
        SpecTestError::InvalidLags {
            defaulted: true,
            ..
        }
    ));
    // Too long for the sample, and absurdly long: refused before any
    // allocation, no overflow.
    for p in [18usize, 1 << 40, usize::MAX] {
        assert!(matches!(
            breusch_godfrey_test(&y, &x, Some(p)).unwrap_err(),
            SpecTestError::DegreesOfFreedom { .. }
        ));
    }
}

#[test]
fn degenerate_and_singular_inputs_are_refused() {
    let n = 20;
    let x1: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
    // y exactly in the column space: zero residuals.
    let y: Vec<f64> = x1.iter().map(|v| 1.0 + 2.0 * v).collect();
    let x = vec![vec![1.0; n], x1.clone()];
    assert!(matches!(
        breusch_godfrey_test(&y, &x, None).unwrap_err(),
        SpecTestError::DegenerateResponse { .. }
    ));
    assert!(matches!(
        durbin_watson_test(&y, &x, POS).unwrap_err(),
        SpecTestError::DegenerateResponse { .. }
    ));
    // A design without a constant whose residuals are a nonzero constant
    // (y = 3 + 0.5 z with z orthogonal to the constant): the auxiliary
    // regression adds an intercept, so the centered R^2 is 0/0.
    let z: Vec<f64> = (0..n)
        .map(|t| if t % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let yc: Vec<f64> = z.iter().map(|v| 3.0 + 0.5 * v).collect();
    assert!(matches!(
        breusch_godfrey_test(&yc, std::slice::from_ref(&z), Some(1)).unwrap_err(),
        SpecTestError::DegenerateResponse { .. }
    ));
    // Collinear design.
    let xs = vec![vec![1.0; n], x1.clone(), x1];
    let (y2, _) = design(n);
    assert!(matches!(
        breusch_godfrey_test(&y2, &xs, Some(2)).unwrap_err(),
        SpecTestError::SingularDesign { .. }
    ));
    assert!(matches!(
        durbin_watson_test(&y2, &xs, POS).unwrap_err(),
        SpecTestError::SingularDesign { .. }
    ));
}

#[test]
fn durbin_watson_sample_size_limits() {
    // n = k + 1: the statistic would be a constant.
    let (y, x) = design(3);
    assert!(matches!(
        durbin_watson_test(&y, &x, POS).unwrap_err(),
        SpecTestError::DegreesOfFreedom { .. }
    ));
    // One observation.
    assert!(matches!(
        durbin_watson_test(&[1.0], &[vec![1.0]], POS).unwrap_err(),
        SpecTestError::DegreesOfFreedom { .. }
    ));
    // Beyond the exact-computation cap: refused before any n x n allocation.
    let (y, x) = design(DW_EXACT_MAX_NOBS + 1);
    assert!(matches!(
        durbin_watson_test(&y, &x, POS).unwrap_err(),
        SpecTestError::SampleTooLong { .. }
    ));
    // The smallest admissible sample works.
    let (y, x) = design(4);
    let r = durbin_watson_test(&y, &x, POS).expect("n = k + 2");
    assert!((0.0..=1.0).contains(&r.pvalue));
}

#[test]
fn error_messages_teach() {
    let (y, x) = design(20);
    let msg = breusch_godfrey_test(&y, &x, Some(0))
        .unwrap_err()
        .to_string();
    assert!(msg.contains("nlags"), "{msg}");
    let (y4, x4) = design(4);
    let msg = breusch_godfrey_test(&y4, &x4, None)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("min(10, n // 5)"), "{msg}");
    let (yl, xl) = design(DW_EXACT_MAX_NOBS + 1);
    let msg = durbin_watson_test(&yl, &xl, POS).unwrap_err().to_string();
    assert!(msg.contains("Breusch-Godfrey"), "{msg}");
}
