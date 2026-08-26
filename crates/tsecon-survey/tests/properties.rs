//! Structural-invariant and error-path tests for tsecon-survey.
//!
//! These complement the independent-reference golden (`golden.rs`): they pin
//! the documented closed forms (`implied_rigidity = beta/(1+beta)`,
//! `IQR = P75 - P25`), a few hand-computed numpy percentile/std values, and
//! every user-input error path.

use tsecon_survey::{
    cg_regression, cg_series, cg_series_fixed_event, disagreement, efficiency_test, HacBandwidth,
    SurveyError,
};

fn approx(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() < tol, "{what}: {a} vs {b}");
}

/// Deterministic pseudo-random uniforms in (-0.5, 0.5) via a 64-bit LCG
/// (Knuth MMIX constants) — no RNG dependency needed at this quality.
fn lcg_series(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
    }
    out
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
fn cg_series_fixed_event_alignment_hand_example() {
    // n = 6, h = 1. Usable t = 1..=4 (t-1>=0, t+1<=5).
    // f[t] = F_t x_{t+1} (1-step), g[t] = F_t x_{t+2} (2-step), so the CG
    // fixed-event revision at t is f[t] - g[t-1]: both forecasts of x_{t+1}.
    let f = vec![10.0, 11.0, 13.0, 12.0, 15.0, 16.0];
    let g = vec![9.0, 10.5, 12.0, 13.0, 14.0, 15.0];
    let y = vec![0.0, 100.0, 200.0, 300.0, 400.0, 500.0];
    let (errors, revisions) = cg_series_fixed_event(&f, &g, &y, 1).unwrap();
    // Errors are the same as cg_series: y[t+1] - f[t].
    assert_eq!(errors, vec![189.0, 287.0, 388.0, 485.0]);
    // t = 1: rev = f[1]-g[0] = 11-9   = 2.
    // t = 2: rev = f[2]-g[1] = 13-10.5 = 2.5.
    // t = 3: rev = f[3]-g[2] = 12-12  = 0.
    // t = 4: rev = f[4]-g[3] = 15-13  = 2.
    assert_eq!(revisions, vec![2.0, 2.5, 0.0, 2.0]);
}

#[test]
fn cg_series_fixed_event_collapses_to_fixed_horizon_for_horizon_free_forecasts() {
    // When the (h+1)-step forecast equals the h-step forecast at every t —
    // a random-walk/martingale target, whose forecast does not depend on the
    // horizon — the fixed-event revision F_t x_{t+h} - F_{t-1} x_{t+h}
    // equals the fixed-horizon difference F_t x_{t+h} - F_{t-1} x_{t+h-1},
    // and the two builders agree exactly.
    let f = vec![10.0, 11.0, 13.0, 12.0, 15.0, 16.0, 14.0];
    let y = vec![9.0, 12.0, 12.5, 13.0, 14.0, 15.5, 15.0];
    let fe = cg_series_fixed_event(&f, &f, &y, 2).unwrap();
    let fh = cg_series(&f, &y, 2).unwrap();
    assert_eq!(fe, fh);
}

#[test]
fn cg_series_fixed_event_error_paths() {
    let ok = vec![1.0, 2.0, 3.0, 4.0];
    assert!(matches!(
        cg_series_fixed_event(&[], &[], &[], 1).unwrap_err(),
        SurveyError::EmptyInput { .. }
    ));
    assert!(matches!(
        cg_series_fixed_event(&ok, &ok[..3], &ok, 1).unwrap_err(),
        SurveyError::DimensionMismatch { .. }
    ));
    assert!(matches!(
        cg_series_fixed_event(&ok, &ok, &ok[..3], 1).unwrap_err(),
        SurveyError::DimensionMismatch { .. }
    ));
    // n = 3, h = 2 needs n >= h + 2 = 4.
    assert!(matches!(
        cg_series_fixed_event(&ok[..3], &ok[..3], &ok[..3], 2).unwrap_err(),
        SurveyError::SeriesTooShort { .. }
    ));
    let bad = vec![1.0, f64::NAN, 3.0, 4.0];
    assert!(matches!(
        cg_series_fixed_event(&ok, &bad, &ok, 1).unwrap_err(),
        SurveyError::NonFinite { .. }
    ));
}

/// The round-9 sweep's defect, pinned as a permanent regression test: on an
/// EXACT Mankiw-Reis sticky-information DGP the fixed-event revision
/// (the Coibion-Gorodnichenko 2015 construction) identifies the rigidity
/// `lambda`, while the fixed-horizon single-series proxy does not.
///
/// DGP (closed-form consensus forecasts, no approximation):
///   x_t = rho x_{t-1} + eps_t (iid),  updating hazard 1 - lambda.
///   Consensus: F_t x_{t+k} = (1-lambda) sum_j lambda^j E_{t-j} x_{t+k}
///            = rho^k S_t,  with  S_t = (1-lambda) x_t + lambda rho S_{t-1}.
/// CG identity: x_{t+h} - F_t x_{t+h}
///            = (lambda/(1-lambda)) (F_t x_{t+h} - F_{t-1} x_{t+h}) + v_t,
/// where v_t = x_{t+h} - E_t x_{t+h} is orthogonal to the time-t revision —
/// so the FIXED-EVENT regression slope converges to lambda/(1-lambda)
/// exactly. The FIXED-HORIZON revision F_t x_{t+h} - F_{t-1} x_{t+h-1}
/// differences forecasts of different targets and has a different plim.
#[test]
fn sticky_information_dgp_fixed_event_recovers_lambda_fixed_horizon_does_not() {
    let lambda = 0.5; // beta = lambda/(1-lambda) = 1, implied rigidity 0.5
    let rho = 0.5;
    let h = 3usize;
    let t_len = 400_000usize;
    let burn = 1_000usize;
    let n = t_len + burn;

    // AR(1) fundamentals driven by iid LCG uniforms (mean 0; the CG identity
    // needs only serially-uncorrelated innovations, not normality).
    let e = lcg_series(n, 20260826);
    let mut x = vec![0.0f64; n];
    for t in 1..n {
        x[t] = rho * x[t - 1] + e[t];
    }
    // Closed-form consensus recursion S_t = (1-lambda) x_t + lambda rho S_{t-1}.
    let mut s = vec![0.0f64; n];
    s[0] = (1.0 - lambda) * x[0];
    for t in 1..n {
        s[t] = (1.0 - lambda) * x[t] + lambda * rho * s[t - 1];
    }
    let rho_h = rho.powi(h as i32);
    let forecast_h: Vec<f64> = s[burn..].iter().map(|&v| rho_h * v).collect();
    let forecast_h1: Vec<f64> = s[burn..].iter().map(|&v| rho_h * rho * v).collect();
    let actual: Vec<f64> = x[burn..].to_vec();

    let (err_fe, rev_fe) = cg_series_fixed_event(&forecast_h, &forecast_h1, &actual, h).unwrap();
    let (err_fh, rev_fh) = cg_series(&forecast_h, &actual, h).unwrap();
    let fe = cg_regression(&err_fe, &rev_fe, HacBandwidth::Lags(5), true).unwrap();
    let fh = cg_regression(&err_fh, &rev_fh, HacBandwidth::Lags(5), true).unwrap();

    // beta_true = lambda/(1-lambda) = 1.0.
    let beta_true = lambda / (1.0 - lambda);
    // Fixed-event: recovers beta (and hence lambda) within MC tolerance
    // (T = 400k; measured slope 1.012899, implied rigidity 0.503204).
    approx(
        fe.slope,
        beta_true,
        0.02,
        "fixed-event slope vs lambda/(1-lambda)",
    );
    approx(
        fe.implied_rigidity,
        lambda,
        0.01,
        "fixed-event implied rigidity vs lambda",
    );
    // Fixed-horizon: measurably biased on the SAME draws. In this AR(1)
    // sticky-information DGP the fixed-horizon slope has the closed-form plim
    // beta * (1 + rho) / 2 = 0.75 (it converges to beta only as rho -> 1, the
    // random-walk case where the two revisions coincide). Measured slope
    // 0.764665, implied "rigidity" 0.433320 vs the true lambda 0.5.
    let fh_plim = beta_true * (1.0 + rho) / 2.0; // = 0.75
    approx(fh.slope, fh_plim, 0.02, "fixed-horizon slope vs its plim");
    assert!(
        (fh.slope - beta_true).abs() > 0.2,
        "fixed-horizon slope {} unexpectedly close to the CG estimand {}",
        fh.slope,
        beta_true
    );
    assert!(
        (fh.implied_rigidity - lambda).abs() > 0.05,
        "fixed-horizon implied rigidity {} unexpectedly close to lambda {}",
        fh.implied_rigidity,
        lambda
    );
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
