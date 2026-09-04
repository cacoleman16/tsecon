//! Golden-value tests against `fixtures/kernel.json`.
//!
//! * `kernel_ridge`: scikit-learn 1.9.0 `KernelRidge` — `dual_coef_`,
//!   `predict(X)`, `predict(X_test)` for all four kernels, asserted at
//!   1e-8 (independent package; the harness prints the achieved figure).
//! * `kernel_regression`: statsmodels 0.15.0 `KernelReg` at fixed
//!   bandwidths — `fit()` at the training rows and at `x_test` for both
//!   `reg_type`s and `k = 1, 2`, asserted at 1e-8, plus the leave-one-out
//!   criterion `cv_loo(bw, func)` at 1e-10 (independent package).
//! * the leave-block-out criterion and the effective degrees of freedom:
//!   documented-formula NumPy transcriptions in the generator (no package
//!   computes them), asserted at 1e-10 — a transcription grade, stated as
//!   such in the model card.
//! * statsmodels' own `bw="cv_ls"` optimum is used only as a property
//!   target: the criterion our search reaches is no worse than fmin's.

mod common;

use common::{as_f64_vec, as_mat, assert_slice_close, load_fixture};
use serde_json::Value;
use tsecon_ml::{
    cv_criterion, kernel_regression, kernel_ridge, BandwidthSpec, KernelRegressionOptions,
    KernelRidgeOptions, KernelType, RegressionKernel, RegressionKind,
};

fn kind_of(reg_type: &str) -> RegressionKind {
    match reg_type {
        "lc" => RegressionKind::NadarayaWatson,
        "ll" => RegressionKind::LocalLinear,
        other => panic!("fixture reg_type {other}"),
    }
}

fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) -> f64 {
    let d = (actual - expected).abs();
    assert!(
        d <= tol,
        "{what}: {actual} vs {expected} (diff {d:e}, tol {tol:e})"
    );
    d
}

/// scikit-learn `KernelRidge`: dual coefficients, fitted values, and test
/// predictions for rbf / laplacian / polynomial / linear at 1e-8.
#[test]
fn golden_kernel_ridge_matches_sklearn() {
    let fx = load_fixture("kernel.json");
    let krr = &fx["kernel_ridge"];
    let x = as_mat(&krr["X"]);
    let xt = as_mat(&krr["X_test"]);
    let y = as_f64_vec(&krr["y"]);

    let mut worst = 0.0f64;
    let mut n_cases = 0;
    for case in krr["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let p = &case["params"];
        let opts = KernelRidgeOptions {
            alpha: p["alpha"].as_f64().unwrap(),
            kernel: KernelType::parse(p["kernel"].as_str().unwrap()).unwrap(),
            gamma: p["gamma"].as_f64(),
            degree: p["degree"].as_f64().unwrap(),
            coef0: p["coef0"].as_f64().unwrap(),
            rff_features: None,
            seed: 0,
        };
        let fit = kernel_ridge(x.as_ref(), &y, Some(xt.as_ref()), &opts).unwrap();
        match case["gamma_resolved"].as_f64() {
            Some(g) => assert_close(fit.gamma.unwrap(), g, 1e-15, "gamma"),
            None => {
                assert!(fit.gamma.is_none(), "{name}: linear kernel reports a gamma");
                0.0
            }
        };
        let d1 = assert_slice_close(
            fit.dual_coef.as_ref().unwrap(),
            &as_f64_vec(&case["dual_coef"]),
            1e-8,
            &format!("{name} dual_coef"),
        );
        let d2 = assert_slice_close(
            &fit.fitted,
            &as_f64_vec(&case["fitted"]),
            1e-8,
            &format!("{name} fitted"),
        );
        let d3 = assert_slice_close(
            fit.predicted.as_ref().unwrap(),
            &as_f64_vec(&case["predicted"]),
            1e-8,
            &format!("{name} predicted"),
        );
        assert!(fit.coef.is_none() && fit.n_rff_features.is_none());
        worst = worst.max(d1).max(d2).max(d3);
        n_cases += 1;
    }
    assert_eq!(n_cases, 8, "fixture should carry two cases per kernel");
    println!("kernel_ridge vs sklearn achieved max abs error: {worst:e}");
    assert!(worst < 1e-8);
}

/// statsmodels `KernelReg.fit()` at fixed bandwidths (1e-8) and
/// `cv_loo` (1e-10); the block-CV and effective-df transcriptions (1e-10).
#[test]
fn golden_kernel_regression_matches_statsmodels_at_fixed_bandwidths() {
    let fx = load_fixture("kernel.json");
    let kr = &fx["kernel_regression"];

    let mut worst_fit = 0.0f64;
    let mut worst_cv = 0.0f64;
    let mut worst_block = 0.0f64;
    let mut worst_edf = 0.0f64;
    let mut n_cases = 0;
    for case in kr["cases"].as_array().unwrap() {
        let sid = case["series"].as_str().unwrap();
        let s = &kr["series"][sid];
        let x = as_mat(&s["x"]);
        let xt = as_mat(&s["x_test"]);
        let y = as_f64_vec(&s["y"]);
        let reg_type = case["reg_type"].as_str().unwrap();
        let kind = kind_of(reg_type);
        let bw = as_f64_vec(&case["bw"]);
        let label = format!("{sid} {reg_type} bw={bw:?}");

        let opts = KernelRegressionOptions {
            kind,
            kernel: RegressionKernel::Gaussian,
            bandwidth: BandwidthSpec::Fixed(bw.clone()),
        };
        let fit = kernel_regression(x.as_ref(), &y, Some(xt.as_ref()), &opts).unwrap();
        assert_eq!(fit.bandwidth, bw);
        assert_eq!(fit.bandwidth_method, "fixed");
        assert_eq!(fit.block, None);
        assert!(!fit.bandwidth_at_boundary);
        assert_eq!(fit.n_criterion_evaluations, 0);

        let d1 = assert_slice_close(
            &fit.fitted,
            &as_f64_vec(&case["fitted"]),
            1e-8,
            &format!("{label} fitted"),
        );
        let d2 = assert_slice_close(
            fit.predicted.as_ref().unwrap(),
            &as_f64_vec(&case["predicted"]),
            1e-8,
            &format!("{label} predicted"),
        );
        worst_fit = worst_fit.max(d1).max(d2);

        // The LOO criterion is reported under "fixed" and is statsmodels'
        // cv_loo exactly; the standalone evaluator agrees bit for bit.
        let cv_loo = case["cv_loo"].as_f64().unwrap();
        let d3 = assert_close(fit.cv_criterion, cv_loo, 1e-10, &format!("{label} cv_loo"));
        let standalone = cv_criterion(x.as_ref(), &y, &bw, kind, 0).unwrap();
        assert_eq!(
            standalone, fit.cv_criterion,
            "{label}: cv_criterion paths disagree"
        );
        worst_cv = worst_cv.max(d3);

        // Transcription legs.
        for (l, v) in case["block_cv"].as_object().unwrap() {
            let l: usize = l.parse().unwrap();
            let got = cv_criterion(x.as_ref(), &y, &bw, kind, l).unwrap();
            let d = assert_close(
                got,
                v.as_f64().unwrap(),
                1e-10,
                &format!("{label} block_cv l={l}"),
            );
            worst_block = worst_block.max(d);
        }
        let d4 = assert_close(
            fit.effective_df,
            case["effective_df"].as_f64().unwrap(),
            1e-10,
            &format!("{label} effective_df"),
        );
        worst_edf = worst_edf.max(d4);
        n_cases += 1;
    }
    assert_eq!(n_cases, 12);
    println!("kernel_regression vs statsmodels fit achieved max abs error: {worst_fit:e}");
    println!("kernel_regression vs statsmodels cv_loo achieved max abs error: {worst_cv:e}");
    println!("kernel_regression block_cv transcription achieved max abs error: {worst_block:e}");
    println!("kernel_regression effective_df transcription achieved max abs error: {worst_edf:e}");
    assert!(worst_fit < 1e-8 && worst_cv < 1e-10 && worst_block < 1e-10 && worst_edf < 1e-10);
}

/// The bandwidth search reaches a criterion no worse than statsmodels'
/// Nelder-Mead optimum (`bw="cv_ls"`), and the criterion at statsmodels'
/// own optimum is reproduced at 1e-10 (another fixed-bandwidth pin).
#[test]
fn selected_bandwidth_is_no_worse_than_statsmodels_fmin_optimum() {
    let fx = load_fixture("kernel.json");
    let kr = &fx["kernel_regression"];
    let mut n = 0;
    for sel in kr["cv_ls_selections"].as_array().unwrap() {
        let sid = sel["series"].as_str().unwrap();
        let s = &kr["series"][sid];
        let x = as_mat(&s["x"]);
        let y = as_f64_vec(&s["y"]);
        let reg_type = sel["reg_type"].as_str().unwrap();
        let kind = kind_of(reg_type);
        let bw_sm = as_f64_vec(&sel["bw_cv_ls"]);
        let cv_sm = sel["cv_loo_at_bw_cv_ls"].as_f64().unwrap();
        let label = format!("{sid} {reg_type}");

        // Pin: our criterion at statsmodels' bandwidth.
        let at_sm = cv_criterion(x.as_ref(), &y, &bw_sm, kind, 0).unwrap();
        assert_close(
            at_sm,
            cv_sm,
            1e-10,
            &format!("{label} cv at statsmodels bw"),
        );

        // Property: our search is at least as good.
        let opts = KernelRegressionOptions {
            kind,
            kernel: RegressionKernel::Gaussian,
            bandwidth: BandwidthSpec::LooCv,
        };
        let fit = kernel_regression(x.as_ref(), &y, None, &opts).unwrap();
        println!(
            "{label}: tsecon bw={:?} cv={:.10} | statsmodels fmin bw={:?} cv={:.10} | evals={}",
            fit.bandwidth, fit.cv_criterion, bw_sm, cv_sm, fit.n_criterion_evaluations
        );
        assert!(
            fit.cv_criterion <= cv_sm * (1.0 + 1e-9),
            "{label}: search criterion {} worse than statsmodels' {}",
            fit.cv_criterion,
            cv_sm
        );
        assert!(
            !fit.bandwidth_at_boundary,
            "{label}: interior optimum flagged"
        );
        n += 1;
    }
    assert_eq!(n, 4);
}

/// The fixture records the reference versions this test claims.
#[test]
fn fixture_meta_names_the_references() {
    let fx = load_fixture("kernel.json");
    let meta: &Value = &fx["_meta"];
    assert_eq!(meta["scikit_learn"].as_str().unwrap(), "1.9.0");
    assert_eq!(meta["statsmodels"].as_str().unwrap(), "0.15.0");
    // The transcription legs were cross-checked against statsmodels where
    // they overlap, at generation time.
    let checks = &meta["transcription_checks"];
    assert!(checks["fit_vs_transcription"].as_f64().unwrap() < 1e-12);
    assert!(checks["cv_loo_vs_transcription_l0"].as_f64().unwrap() < 1e-12);
}
