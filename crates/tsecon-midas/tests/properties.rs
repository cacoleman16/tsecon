//! Property and validation tests for the MIDAS estimators.
//!
//! * weight functions are normalized and shaped as documented;
//! * the mixed-frequency design builder aligns periods exactly;
//! * U-MIDAS reproduces plain OLS;
//! * U-MIDAS is the free-lag limit of weighted MIDAS (weaker restriction =>
//!   no worse fit);
//! * weighted MIDAS recovers the weights of a seeded Beta-weight DGP.

use tsecon_midas::{
    almon_pdl_basis, almon_weights, beta_weights, exp_almon_weights, stack_high_freq_lags, umidas,
    weighted_midas, SeType, WeightScheme,
};
use tsecon_rng::Stream;

// ---------------------------------------------------------------------------
// Weight-function properties.
// ---------------------------------------------------------------------------

#[test]
fn weights_sum_to_one() {
    for &(t1, t2) in &[(0.5, -0.2), (0.0, -0.05), (-0.3, -0.1), (0.2, 0.0)] {
        for k in 2..=12 {
            let w = exp_almon_weights(t1, t2, k).expect("exp-Almon");
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "exp-Almon sum k={k}: {s}");
            assert!(w.iter().all(|&x| x >= 0.0), "exp-Almon nonneg");
        }
    }
    for &(a, b) in &[(2.0, 3.0), (1.0, 5.0), (3.0, 1.5), (0.7, 0.9)] {
        for k in 2..=12 {
            let w = beta_weights(a, b, k).expect("Beta");
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "Beta sum k={k}: {s}");
            assert!(w.iter().all(|&x| x >= 0.0), "Beta nonneg");
        }
    }
    // Almon PDL polynomial profile normalizes too (positive-valued profile).
    let w = almon_weights(&[1.0, -0.05], 8).expect("Almon weights");
    let s: f64 = w.iter().sum();
    assert!((s - 1.0).abs() < 1e-12, "Almon-weights sum: {s}");
}

#[test]
fn exp_almon_theta2_negative_decays() {
    // theta1 = 0, theta2 < 0 => strictly decreasing weights.
    let w = exp_almon_weights(0.0, -0.1, 10).expect("exp-Almon");
    for k in 1..w.len() {
        assert!(w[k] < w[k - 1], "not decaying at k={k}: {:?}", w);
    }
}

#[test]
fn almon_pdl_basis_shape_and_values() {
    // K x (degree + 1) basis with Q[k-1][j] = k^j.
    let basis = almon_pdl_basis(5, 2).expect("basis");
    assert_eq!(basis.len(), 3, "degree + 1 columns");
    assert!(basis.iter().all(|c| c.len() == 5), "K rows");
    // Constant column is all ones.
    assert!(basis[0].iter().all(|&v| v == 1.0));
    // Linear column is [1, 2, 3, 4, 5].
    assert_eq!(basis[1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    // Quadratic column is [1, 4, 9, 16, 25].
    assert_eq!(basis[2], vec![1.0, 4.0, 9.0, 16.0, 25.0]);
    // Rank guard: degree + 1 > K rejected.
    assert!(almon_pdl_basis(3, 3).is_err());
}

// ---------------------------------------------------------------------------
// Design builder alignment.
// ---------------------------------------------------------------------------

#[test]
fn design_builder_aligns_known_example() {
    // hf indices carry their own value so alignment is self-documenting.
    let hf: Vec<f64> = (10..22).map(|i| i as f64).collect(); // len 12
    let low = vec![100.0, 101.0, 102.0, 103.0];
    let d = stack_high_freq_lags(&hf, &low, 3, 4).expect("stack");

    assert_eq!(d.first_low_period, 1, "warm-up = ceil(4/3) - 1");
    assert_eq!(d.nobs, 3);
    assert_eq!(d.target, vec![101.0, 102.0, 103.0]);
    // Most-recent-first: column 0 is the last month of each quarter.
    assert_eq!(d.columns[0], vec![15.0, 18.0, 21.0]);
    assert_eq!(d.columns[1], vec![14.0, 17.0, 20.0]);
    assert_eq!(d.columns[2], vec![13.0, 16.0, 19.0]);
    assert_eq!(d.columns[3], vec![12.0, 15.0, 18.0]);
    // Column c + ratio equals column c lagged one low-frequency period.
    for t in 1..d.nobs {
        assert_eq!(d.columns[3][t], d.columns[0][t - 1]);
    }
}

#[test]
fn design_builder_rejects_short_and_bad_input() {
    // Too few HF obs for coverage.
    assert!(stack_high_freq_lags(&[1.0, 2.0], &[1.0, 2.0], 3, 2).is_err());
    // No usable row (K needs more history than the series has periods).
    assert!(
        stack_high_freq_lags(&(0..3).map(|i| i as f64).collect::<Vec<_>>(), &[1.0], 3, 4).is_err()
    );
    // Non-finite input.
    assert!(stack_high_freq_lags(&[1.0, f64::NAN, 3.0, 4.0], &[1.0], 3, 2).is_err());
    // Zero ratio / zero lags.
    assert!(stack_high_freq_lags(&[1.0, 2.0, 3.0], &[1.0], 0, 2).is_err());
    assert!(stack_high_freq_lags(&[1.0, 2.0, 3.0], &[1.0], 3, 0).is_err());
}

// ---------------------------------------------------------------------------
// U-MIDAS = OLS.
// ---------------------------------------------------------------------------

#[test]
fn umidas_equals_closed_form_ols() {
    // Single high-frequency regressor + const: closed-form OLS.
    let x = vec![1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
    let y = vec![2.1, 3.9, 6.2, 9.8, 16.1, 25.9, 42.2, 68.0];
    let n = x.len() as f64;
    let xbar = x.iter().sum::<f64>() / n;
    let ybar = y.iter().sum::<f64>() / n;
    let sxy: f64 = x
        .iter()
        .zip(&y)
        .map(|(xi, yi)| (xi - xbar) * (yi - ybar))
        .sum();
    let sxx: f64 = x.iter().map(|xi| (xi - xbar).powi(2)).sum();
    let slope = sxy / sxx;
    let intercept = ybar - slope * xbar;

    let fit = umidas(&y, std::slice::from_ref(&x), SeType::NonRobust).expect("umidas");
    assert!((fit.params[0] - intercept).abs() < 1e-10, "intercept");
    assert!((fit.params[1] - slope).abs() < 1e-10, "slope");

    // Centered R^2 matches 1 - RSS/TSS computed independently.
    let rss: f64 = fit.residuals.iter().map(|u| u * u).sum();
    let tss: f64 = y.iter().map(|yi| (yi - ybar).powi(2)).sum();
    assert!((fit.rsquared - (1.0 - rss / tss)).abs() < 1e-12, "rsquared");
}

// ---------------------------------------------------------------------------
// Weighted MIDAS.
// ---------------------------------------------------------------------------

/// Box-Muller standard normal from a seeded uniform stream.
fn normal(stream: &mut Stream) -> f64 {
    let mut u1 = stream.uniform_f64();
    if u1 < 1e-300 {
        u1 = 1e-300;
    }
    let u2 = stream.uniform_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Simulate a Beta-weight DGP and recover the weights by NLS. Loose bounds:
/// NLS on 119 noisy observations recovers the weight *shape*, not the exact
/// hyperparameters (the Beta map is flat in some directions).
#[test]
fn weighted_midas_recovers_beta_dgp() {
    let mut stream = Stream::new(20260717);
    let ratio = 3usize;
    let k = 6usize;
    let n_low = 120usize;
    let n_hf = n_low * ratio;

    let hf: Vec<f64> = (0..n_hf).map(|_| normal(&mut stream)).collect();
    let low_dummy = vec![0.0; n_low];
    let design = stack_high_freq_lags(&hf, &low_dummy, ratio, k).expect("stack");
    let cols = design.columns;
    let nobs = design.nobs;

    let (t1_true, t2_true) = (2.0, 4.0);
    let (alpha_true, beta_true) = (1.0, 3.0);
    let sigma = 0.1;
    let w_true = beta_weights(t1_true, t2_true, k).expect("true weights");

    let y: Vec<f64> = (0..nobs)
        .map(|t| {
            let agg: f64 = cols.iter().zip(&w_true).map(|(c, &w)| w * c[t]).sum();
            alpha_true + beta_true * agg + sigma * normal(&mut stream)
        })
        .collect();

    let fit = weighted_midas(&y, &cols, WeightScheme::Beta, None).expect("weighted MIDAS");

    // Weight shape recovered.
    let max_w_err = fit
        .weights
        .iter()
        .zip(&w_true)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_w_err < 0.03,
        "beta weight recovery: max err {max_w_err}, got {:?} vs {:?}",
        fit.weights,
        w_true
    );
    // Aggregate slope and intercept recovered within a loose band.
    assert!((fit.slope - beta_true).abs() < 0.3, "slope {}", fit.slope);
    assert!(
        (fit.intercept - alpha_true).abs() < 0.3,
        "intercept {}",
        fit.intercept
    );
    assert!(fit.rsquared > 0.9, "rsquared {}", fit.rsquared);
}

/// U-MIDAS is the free-lag limit: with all K lag coefficients unrestricted its
/// residual sum of squares cannot exceed that of any weight-restricted fit on
/// the same design. Checked on the real fixture design.
#[test]
fn umidas_is_free_lag_limit_of_weighted() {
    let path = format!("{}/../../fixtures/midas.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture readable");
    let fx: serde_json::Value = serde_json::from_str(&text).expect("json");
    let y: Vec<f64> = fx["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let cols: Vec<Vec<f64>> = fx["X_stacked"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            c.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();

    let u = umidas(&y, &cols, SeType::NonRobust).expect("umidas");
    let rss_umidas: f64 = u.residuals.iter().map(|e| e * e).sum();

    for scheme in [WeightScheme::ExpAlmon, WeightScheme::Beta] {
        let w = weighted_midas(&y, &cols, scheme, None).expect("weighted");
        assert!(
            rss_umidas <= w.ssr + 1e-6,
            "U-MIDAS RSS {rss_umidas} exceeds weighted {scheme:?} SSR {}",
            w.ssr
        );
    }
}
