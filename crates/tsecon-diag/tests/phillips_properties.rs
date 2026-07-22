//! Property / contract tests for the Phillips-Perron and Phillips-Ouliaris
//! tests that do not need an external golden: default bandwidth rules, the
//! statistic/critical-value availability contract, and the error surface.

use tsecon_diag::{
    phillips_ouliaris, phillips_perron, AdfRegression, DiagError, PoTestType, PoTrend, PpTestType,
};

/// A deterministic, reproducible "random walk" (no system RNG): increments
/// from a fixed low-discrepancy-ish recurrence so the tests are stable.
fn walk(n: usize, seed: f64) -> Vec<f64> {
    let mut y = Vec::with_capacity(n);
    let mut acc = 0.0;
    let mut x = seed;
    for _ in 0..n {
        // A simple chaotic map for reproducible pseudo-noise in (-1, 1).
        x = (1.3 * x + 0.7).sin();
        acc += x;
        y.push(acc);
    }
    y
}

/// A deterministic stationary series around zero.
fn stationary(n: usize, seed: f64) -> Vec<f64> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = (1.7 * x + 0.3).sin();
            x
        })
        .collect()
}

#[test]
fn pp_default_bandwidth_is_schwert_rule_on_full_length() {
    let n = 200usize;
    let y = walk(n, 0.11);
    let expected = (12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as usize;
    let res = phillips_perron(&y, AdfRegression::Constant, PpTestType::Tau, None).unwrap();
    assert_eq!(res.lags, expected);
    assert_eq!(res.nobs, n - 1);
}

#[test]
fn pp_reports_both_statistics_regardless_of_selection() {
    let y = walk(150, 0.31);
    let tau = phillips_perron(&y, AdfRegression::ConstantTrend, PpTestType::Tau, Some(6)).unwrap();
    let rho = phillips_perron(&y, AdfRegression::ConstantTrend, PpTestType::Rho, Some(6)).unwrap();
    // Both selections expose the same underlying ztau / zalpha.
    assert_eq!(tau.ztau, rho.ztau);
    assert_eq!(tau.zalpha, rho.zalpha);
    // `stat` tracks the selected test type.
    assert_eq!(tau.stat, tau.ztau);
    assert_eq!(rho.stat, rho.zalpha);
    assert!(tau.p_value.is_finite() && rho.p_value.is_finite());
}

#[test]
fn pp_error_surface() {
    // Constant series.
    let c = vec![3.0; 40];
    assert!(matches!(
        phillips_perron(&c, AdfRegression::Constant, PpTestType::Tau, None),
        Err(DiagError::ConstantSeries { .. })
    ));
    // Non-finite.
    let mut bad = walk(40, 0.5);
    bad[10] = f64::NAN;
    assert!(matches!(
        phillips_perron(&bad, AdfRegression::Constant, PpTestType::Tau, None),
        Err(DiagError::NonFinite { .. })
    ));
    // Too short for the constant-trend spec (needs 2*(2+1) = 6 obs).
    let short = walk(5, 0.5);
    assert!(matches!(
        phillips_perron(&short, AdfRegression::ConstantTrend, PpTestType::Tau, None),
        Err(DiagError::SeriesTooShort { .. })
    ));
    // Bandwidth larger than nobs = n - 1.
    let y = walk(30, 0.5);
    assert!(matches!(
        phillips_perron(&y, AdfRegression::Constant, PpTestType::Tau, Some(30)),
        Err(DiagError::InvalidLags { .. })
    ));
}

#[test]
fn po_default_bandwidth_is_newey_west_maxlags_on_t_minus_1() {
    let t = 200usize;
    let x0 = walk(t, 0.2);
    let x = vec![x0];
    let y: Vec<f64> = walk(t, 0.9);
    let expected = (4.0 * ((t - 1) as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;
    let res = phillips_ouliaris(&y, &x, PoTrend::Constant, PoTestType::Zt, None).unwrap();
    assert_eq!(res.lags, expected);
    assert_eq!(res.nobs, t);
    assert_eq!(res.n_vars, 2);
}

#[test]
fn po_za_has_no_pvalue_or_crit() {
    let t = 120usize;
    let x = vec![walk(t, 0.2), walk(t, 0.4)];
    let y = walk(t, 0.8);
    let res = phillips_ouliaris(&y, &x, PoTrend::Constant, PoTestType::Za, Some(5)).unwrap();
    assert!(res.p_value.is_nan());
    assert!(res.crit.is_none());
    assert_eq!(res.n_vars, 3);
}

#[test]
fn po_crit_availability_contract() {
    let t = 120usize;
    // No-constant trend with N > 1: p-value available, crit unavailable.
    let x = vec![walk(t, 0.2)];
    let y = walk(t, 0.8);
    let res = phillips_ouliaris(&y, &x, PoTrend::None, PoTestType::Zt, Some(5)).unwrap();
    assert!(res.p_value.is_finite());
    assert!(res.crit.is_none());

    // N = 7 (six regressors): p-value NaN (N > 6), but crit available (N <= 12).
    let x6: Vec<Vec<f64>> = (0..6).map(|j| walk(t, 0.1 * (j + 1) as f64)).collect();
    let res7 = phillips_ouliaris(&y, &x6, PoTrend::Constant, PoTestType::Zt, Some(5)).unwrap();
    assert_eq!(res7.n_vars, 7);
    assert!(res7.p_value.is_nan());
    assert!(res7.crit.is_some());
}

#[test]
fn po_rejects_zero_regressors() {
    let y = walk(50, 0.3);
    let x: Vec<Vec<f64>> = Vec::new();
    assert!(phillips_ouliaris(&y, &x, PoTrend::Constant, PoTestType::Zt, Some(4)).is_err());
}

#[test]
fn po_separates_cointegrated_from_spurious() {
    let t = 200usize;
    let x0 = walk(t, 0.25);
    // Cointegrated: y is a stationary deviation from 2*x0.
    let noise = stationary(t, 0.6);
    let y_co: Vec<f64> = x0
        .iter()
        .zip(&noise)
        .map(|(&a, &e)| 2.0 * a + 0.3 * e)
        .collect();
    let co = phillips_ouliaris(
        &y_co,
        std::slice::from_ref(&x0),
        PoTrend::Constant,
        PoTestType::Zt,
        Some(5),
    )
    .unwrap();
    // Not cointegrated: y is an independent walk.
    let y_no = walk(t, 0.95);
    let no = phillips_ouliaris(
        &y_no,
        std::slice::from_ref(&x0),
        PoTrend::Constant,
        PoTestType::Zt,
        Some(5),
    )
    .unwrap();
    // Cointegration rejects (very negative Zt, small p-value); the spurious
    // pair does not reject nearly as strongly.
    assert!(
        co.stat < no.stat,
        "co.stat {} !< no.stat {}",
        co.stat,
        no.stat
    );
    assert!(
        co.p_value < 0.10,
        "cointegrated p-value {} not < 0.10",
        co.p_value
    );
}
