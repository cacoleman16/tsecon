//! Structural-invariant and error-path tests for tsecon-survey.
//!
//! These complement the independent-reference golden (`golden.rs`): they pin
//! the documented closed forms (`implied_rigidity = beta/(1+beta)`,
//! `IQR = P75 - P25`), a few hand-computed numpy percentile/std values, and
//! every user-input error path.

use tsecon_survey::{
    cg_regression, cg_series, disagreement, efficiency_test, HacBandwidth, SurveyError,
};

fn approx(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() < tol, "{what}: {a} vs {b}");
}

// -------- CG regression --------------------------------------------------

#[test]
fn implied_rigidity_is_slope_over_one_plus_slope() {
    // Errors that load positively on the revision => positive slope.
    let revisions: Vec<f64> = (0..40).map(|t| ((t as f64) * 0.7).sin()).collect();
    let errors: Vec<f64> = revisions
        .iter()
        .enumerate()
        .map(|(t, r)| 0.05 + 0.8 * r + 0.1 * ((t as f64) * 0.3).cos())
        .collect();
    let fit = cg_regression(&errors, &revisions, HacBandwidth::Auto, true).unwrap();
    approx(
        fit.implied_rigidity,
        fit.slope / (1.0 + fit.slope),
        1e-12,
        "implied_rigidity",
    );
    assert!((0.0..=1.0).contains(&fit.r_squared));
}

#[test]
fn cg_series_alignment_hand_example() {
    // n = 6, h = 1. Usable t = 1..=4 (t-1>=0, t+1<=5).
    let f = vec![10.0, 11.0, 13.0, 12.0, 15.0, 16.0];
    let y = vec![0.0, 100.0, 200.0, 300.0, 400.0, 500.0];
    let (errors, revisions) = cg_series(&f, &y, 1).unwrap();
    // t = 1: err = y[2]-f[1] = 200-11 = 189; rev = f[1]-f[0] = 1.
    // t = 2: err = y[3]-f[2] = 300-13 = 287; rev = f[2]-f[1] = 2.
    // t = 3: err = y[4]-f[3] = 400-12 = 388; rev = f[3]-f[2] = -1.
    // t = 4: err = y[5]-f[4] = 500-15 = 485; rev = f[4]-f[3] = 3.
    assert_eq!(errors, vec![189.0, 287.0, 388.0, 485.0]);
    assert_eq!(revisions, vec![1.0, 2.0, -1.0, 3.0]);
}

#[test]
fn cg_series_too_short_errs() {
    // n = 2, h = 1 needs n >= h + 2 = 3.
    let e = cg_series(&[1.0, 2.0], &[1.0, 2.0], 1).unwrap_err();
    assert!(matches!(e, SurveyError::SeriesTooShort { .. }));
}

#[test]
fn cg_dimension_and_finiteness_errors() {
    assert!(matches!(
        cg_regression(&[1.0, 2.0], &[1.0], HacBandwidth::Auto, true).unwrap_err(),
        SurveyError::DimensionMismatch { .. }
    ));
    assert!(matches!(
        cg_regression(&[], &[], HacBandwidth::Auto, true).unwrap_err(),
        SurveyError::EmptyInput { .. }
    ));
    let bad = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let ok = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(matches!(
        cg_regression(&bad, &ok, HacBandwidth::Auto, true).unwrap_err(),
        SurveyError::NonFinite { .. }
    ));
}

// -------- Disagreement ---------------------------------------------------

#[test]
fn disagreement_hand_computed_numpy_values() {
    // np.std([10,20,30,40], ddof=0) = sqrt(125) = 11.180339887498949
    // np.percentile linear: p25=17.5, p50=25, p75=32.5, IQR=15.
    let panel = vec![vec![10.0, 20.0, 30.0, 40.0]];
    let d = disagreement(&panel, 0).unwrap();
    approx(d.std[0], 125.0_f64.sqrt(), 1e-12, "std");
    approx(d.p25[0], 17.5, 1e-12, "p25");
    approx(d.p50[0], 25.0, 1e-12, "p50");
    approx(d.p75[0], 32.5, 1e-12, "p75");
    approx(d.iqr[0], 15.0, 1e-12, "iqr");
    assert_eq!(d.counts[0], 4);
    // Sample std uses divisor 3: sqrt(500/3).
    let ds = disagreement(&panel, 1).unwrap();
    approx(ds.std[0], (500.0_f64 / 3.0).sqrt(), 1e-12, "sample std");
}

#[test]
fn disagreement_iqr_is_p75_minus_p25_ragged() {
    let panel = vec![
        vec![1.0, 2.0, 3.0],
        vec![5.0, 5.0, 5.0, 5.0, 5.0],
        vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    ];
    let d = disagreement(&panel, 0).unwrap();
    for t in 0..panel.len() {
        approx(d.iqr[t], d.p75[t] - d.p25[t], 1e-12, "iqr==p75-p25");
        assert!(d.std[t] >= 0.0);
    }
    // A constant cross-section has zero dispersion.
    approx(d.std[1], 0.0, 1e-15, "constant std");
    approx(d.iqr[1], 0.0, 1e-15, "constant iqr");
}

#[test]
fn disagreement_single_forecaster_is_degenerate() {
    let panel = vec![vec![7.5]];
    let d = disagreement(&panel, 0).unwrap();
    approx(d.std[0], 0.0, 1e-15, "std");
    approx(d.p25[0], 7.5, 1e-15, "p25");
    approx(d.iqr[0], 0.0, 1e-15, "iqr");
}

#[test]
fn disagreement_error_paths() {
    assert!(matches!(
        disagreement(&[], 0).unwrap_err(),
        SurveyError::EmptyInput { .. }
    ));
    assert!(matches!(
        disagreement(&[vec![1.0, 2.0], vec![]], 0).unwrap_err(),
        SurveyError::EmptyInput { .. }
    ));
    // ddof >= cross-section size => invalid divisor.
    assert!(matches!(
        disagreement(&[vec![1.0, 2.0]], 2).unwrap_err(),
        SurveyError::InvalidArgument { .. }
    ));
    assert!(matches!(
        disagreement(&[vec![1.0, f64::INFINITY]], 0).unwrap_err(),
        SurveyError::NonFinite { .. }
    ));
}

// -------- Efficiency / Mincer-Zarnowitz -----------------------------------

#[test]
fn efficiency_wald_is_nonnegative_and_pvalue_in_unit_interval() {
    let forecast: Vec<f64> = (0..80).map(|t| ((t as f64) * 0.2).sin()).collect();
    // Error mildly predictable from the forecast => efficiency should be
    // rejectable, but here we only assert the statistic is well-formed.
    let errors: Vec<f64> = forecast
        .iter()
        .enumerate()
        .map(|(t, f)| 0.1 + 0.2 * f + 0.3 * ((t as f64) * 0.5).cos())
        .collect();
    let fit = efficiency_test(&errors, &[forecast], HacBandwidth::Auto, true).unwrap();
    assert!(fit.wald >= 0.0, "wald >= 0");
    assert!((0.0..=1.0).contains(&fit.wald_pvalue), "pvalue in [0,1]");
    assert_eq!(fit.wald_df, 2);
    assert_eq!(fit.params.len(), 2);
}

#[test]
fn efficiency_needs_a_regressor() {
    assert!(matches!(
        efficiency_test(&[1.0, 2.0, 3.0], &[], HacBandwidth::Auto, true).unwrap_err(),
        SurveyError::EmptyInput { .. }
    ));
}

#[test]
fn efficiency_collinear_regressors_error() {
    let errors: Vec<f64> = (0..30).map(|t| (t as f64).sin()).collect();
    let x1: Vec<f64> = (0..30).map(|t| t as f64).collect();
    // x2 = 2 * x1 => perfectly collinear design (with the intercept it is the
    // OLS normal equations that are singular).
    let x2: Vec<f64> = x1.iter().map(|v| 2.0 * v).collect();
    let e = efficiency_test(&errors, &[x1, x2], HacBandwidth::Auto, true).unwrap_err();
    assert!(matches!(
        e,
        SurveyError::Hac(_) | SurveyError::Singular { .. }
    ));
}
