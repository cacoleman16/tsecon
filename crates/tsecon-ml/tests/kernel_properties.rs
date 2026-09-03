//! Property / invariant tests for the kernel-methods slice, beyond the
//! goldens in `kernel_golden.rs`:
//!
//! * random Fourier features: same seed bit-identical, different seeds
//!   differ, and the approximation error against the exact kernel ridge
//!   fit falls as the feature count grows over three sizes;
//! * exact kernel ridge: the dual system is solved (residual check), the
//!   linear kernel reproduces primal ridge, `x_test == x` reproduces the
//!   fitted values;
//! * kernel regression: `tr(S)` runs between its documented limits
//!   (Nadaraya-Watson `1..n`, local linear `k+1..n`) and the large-
//!   bandwidth limits are the global constant / OLS fits; the selected
//!   bandwidth is a local minimum of its criterion; the boundary flag
//!   fires on a signal-free target and stays quiet on a clear one;
//! * dependence: with AR(1) errors, leave-block-out CV selects a wider
//!   bandwidth than leave-one-out on most seeds (the undersmoothing that
//!   motivates the block criterion), measured over a seeded Monte Carlo;
//! * every teaching error names its argument and fix, and no input panics.

mod common;

use common::{mat_from_cols, Lcg};
use tsecon_ml::faer::Mat;
use tsecon_ml::{
    cv_criterion, kernel_regression, kernel_ridge, ridge, BandwidthSpec, KernelRegressionOptions,
    KernelRidgeOptions, KernelType, MlError, RegressionKernel, RegressionKind,
};

fn nonlinear_design(seed: u64, n: usize, p: usize, noise: f64) -> (Mat<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let cols: Vec<Vec<f64>> = (0..p)
        .map(|_| (0..n).map(|_| rng.normal()).collect())
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let mut f = (1.5 * cols[0][i]).sin();
            if p > 1 {
                f += 0.5 * cols[1][i] * cols[1][i];
            }
            if p > 2 {
                f -= cols[2][i];
            }
            f + noise * rng.normal()
        })
        .collect();
    (mat_from_cols(&cols), y)
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

// ------------------------------------------------------------ kernel ridge

#[test]
fn rff_is_deterministic_in_the_seed_and_differs_across_seeds() {
    let (x, y) = nonlinear_design(3, 80, 2, 0.3);
    let opts = |seed| KernelRidgeOptions {
        alpha: 0.5,
        rff_features: Some(64),
        seed,
        ..Default::default()
    };
    let a = kernel_ridge(x.as_ref(), &y, None, &opts(7)).unwrap();
    let b = kernel_ridge(x.as_ref(), &y, None, &opts(7)).unwrap();
    let c = kernel_ridge(x.as_ref(), &y, None, &opts(8)).unwrap();
    assert_eq!(a, b, "same seed must be bit-identical");
    assert_ne!(a.coef, c.coef, "different seeds must differ");
    assert_eq!(a.n_rff_features, Some(64));
    assert!(a.dual_coef.is_none());
    assert_eq!(a.coef.as_ref().unwrap().len(), 64);
    assert_eq!(a.gamma, Some(0.5)); // 1 / n_features
}

#[test]
fn rff_error_against_exact_krr_falls_with_the_feature_count() {
    let (x, y) = nonlinear_design(11, 150, 2, 0.3);
    let (xt, _) = nonlinear_design(12, 40, 2, 0.3);
    let exact = kernel_ridge(
        x.as_ref(),
        &y,
        Some(xt.as_ref()),
        &KernelRidgeOptions {
            alpha: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    let mut errs = Vec::new();
    for d in [20usize, 200, 2000] {
        let rff = kernel_ridge(
            x.as_ref(),
            &y,
            Some(xt.as_ref()),
            &KernelRidgeOptions {
                alpha: 0.5,
                rff_features: Some(d),
                seed: 20260903,
                ..Default::default()
            },
        )
        .unwrap();
        let e_fit = rmse(&rff.fitted, &exact.fitted);
        let e_pred = rmse(
            rff.predicted.as_ref().unwrap(),
            exact.predicted.as_ref().unwrap(),
        );
        println!("D={d}: rmse(fitted)={e_fit:.5} rmse(predicted)={e_pred:.5}");
        errs.push((e_fit, e_pred));
    }
    assert!(
        errs[0].0 > errs[1].0 && errs[1].0 > errs[2].0,
        "fitted error not decreasing: {errs:?}"
    );
    assert!(
        errs[0].1 > errs[1].1 && errs[1].1 > errs[2].1,
        "predicted error not decreasing: {errs:?}"
    );
    // At D = 2000 the approximation is close in the units of y (sd ~ 1).
    assert!(errs[2].0 < 0.05 && errs[2].1 < 0.1, "{errs:?}");
}

#[test]
fn exact_krr_solves_the_dual_system_and_reproduces_itself_on_x_test() {
    let (x, y) = nonlinear_design(5, 60, 3, 0.2);
    for kernel in [
        KernelType::Rbf,
        KernelType::Laplacian,
        KernelType::Polynomial,
        KernelType::Linear,
    ] {
        let opts = KernelRidgeOptions {
            alpha: 0.7,
            kernel,
            ..Default::default()
        };
        let fit = kernel_ridge(x.as_ref(), &y, Some(x.as_ref()), &opts).unwrap();
        let a = fit.dual_coef.as_ref().unwrap();
        // (K + alpha I) a = y  <=>  fitted + alpha a = y.
        for i in 0..60 {
            let r = fit.fitted[i] + 0.7 * a[i] - y[i];
            assert!(r.abs() < 1e-9, "{kernel:?}: dual residual {r:e} at {i}");
        }
        let pred = fit.predicted.as_ref().unwrap();
        for (p, f) in pred.iter().zip(&fit.fitted) {
            assert!((p - f).abs() < 1e-12);
        }
        assert_eq!(fit.kernel, kernel);
    }
}

#[test]
fn linear_kernel_ridge_equals_primal_ridge_for_n_gt_p() {
    let (x, y) = nonlinear_design(9, 50, 3, 0.2);
    let fit = kernel_ridge(
        x.as_ref(),
        &y,
        None,
        &KernelRidgeOptions {
            alpha: 2.0,
            kernel: KernelType::Linear,
            ..Default::default()
        },
    )
    .unwrap();
    let beta = ridge(x.as_ref(), &y, 2.0).unwrap();
    for i in 0..50 {
        let primal: f64 = (0..3).map(|j| x[(i, j)] * beta[j]).sum();
        assert!((fit.fitted[i] - primal).abs() < 1e-10);
    }
}

#[test]
fn kernel_ridge_teaching_errors_name_the_argument() {
    let (x, y) = nonlinear_design(1, 30, 2, 0.2);
    let base = KernelRidgeOptions::default();
    let msg = |o: &KernelRidgeOptions| {
        kernel_ridge(x.as_ref(), &y, None, o)
            .unwrap_err()
            .to_string()
    };

    assert!(msg(&KernelRidgeOptions {
        alpha: -1.0,
        ..base.clone()
    })
    .contains("alpha=-1"));
    assert!(msg(&KernelRidgeOptions {
        gamma: Some(0.0),
        ..base.clone()
    })
    .contains("gamma=0"));
    let m = msg(&KernelRidgeOptions {
        kernel: KernelType::Linear,
        gamma: Some(0.3),
        ..base.clone()
    });
    assert!(m.contains("gamma=0.3") && m.contains("linear"), "{m}");
    let m = msg(&KernelRidgeOptions {
        kernel: KernelType::Polynomial,
        rff_features: Some(16),
        ..base.clone()
    });
    assert!(m.contains("rff_features=16") && m.contains("rbf"), "{m}");
    let m = msg(&KernelRidgeOptions {
        seed: 5,
        ..base.clone()
    });
    assert!(m.contains("seed=5") && m.contains("rff_features"), "{m}");
    let m = msg(&KernelRidgeOptions {
        degree: 4.0,
        ..base.clone()
    });
    assert!(m.contains("degree=4") && m.contains("polynomial"), "{m}");
    assert!(KernelType::parse("cosine")
        .unwrap_err()
        .to_string()
        .contains("\"laplacian\""));

    // Non-finite inputs name the array.
    let mut xn = x.clone();
    xn[(3, 1)] = f64::NAN;
    assert_eq!(
        kernel_ridge(xn.as_ref(), &y, None, &base).unwrap_err(),
        MlError::NonFinite { what: "x" }
    );
    let mut yn = y.clone();
    yn[0] = f64::INFINITY;
    assert_eq!(
        kernel_ridge(x.as_ref(), &yn, None, &base).unwrap_err(),
        MlError::NonFinite { what: "y" }
    );
    assert_eq!(
        kernel_ridge(x.as_ref(), &y, Some(xn.as_ref()), &base).unwrap_err(),
        MlError::NonFinite { what: "x_test" }
    );
    // Conformability of x_test.
    let bad = Mat::<f64>::zeros(4, 3);
    assert!(matches!(
        kernel_ridge(x.as_ref(), &y, Some(bad.as_ref()), &base).unwrap_err(),
        MlError::DimensionMismatch {
            expected: 2,
            got: 3,
            ..
        }
    ));
}

#[test]
fn krr_refuses_a_singular_system_instead_of_falling_back() {
    // Duplicate rows with alpha = 0: K + 0 I is singular for every kernel.
    let x = Mat::from_fn(6, 1, |i, _| (i / 2) as f64);
    let y = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let e = kernel_ridge(
        x.as_ref(),
        &y,
        None,
        &KernelRidgeOptions {
            alpha: 0.0,
            ..Default::default()
        },
    )
    .unwrap_err();
    let m = e.to_string();
    assert!(matches!(e, MlError::NotPositiveDefinite { .. }));
    assert!(m.contains("alpha=0") && m.contains("Increase alpha"), "{m}");
    // A positive penalty makes the same problem well posed.
    assert!(kernel_ridge(
        x.as_ref(),
        &y,
        None,
        &KernelRidgeOptions {
            alpha: 0.1,
            ..Default::default()
        }
    )
    .is_ok());
}

// ------------------------------------------------------- kernel regression

fn kreg(kind: RegressionKind, bandwidth: BandwidthSpec) -> KernelRegressionOptions {
    KernelRegressionOptions {
        kind,
        kernel: RegressionKernel::Gaussian,
        bandwidth,
    }
}

#[test]
fn effective_df_runs_between_its_documented_limits() {
    let (x, y) = nonlinear_design(2, 80, 2, 0.3);
    let n = 80.0;
    for (kind, lo) in [
        (RegressionKind::NadarayaWatson, 1.0),
        (RegressionKind::LocalLinear, 3.0),
    ] {
        let wide = kernel_regression(
            x.as_ref(),
            &y,
            None,
            &kreg(kind, BandwidthSpec::Fixed(vec![1e4, 1e4])),
        )
        .unwrap();
        assert!(
            (wide.effective_df - lo).abs() < 1e-6,
            "{kind:?}: {}",
            wide.effective_df
        );
        let narrow = kernel_regression(
            x.as_ref(),
            &y,
            None,
            &kreg(kind, BandwidthSpec::Fixed(vec![1e-3, 1e-3])),
        )
        .unwrap();
        assert!(
            (narrow.effective_df - n).abs() < 1e-6,
            "{kind:?}: {}",
            narrow.effective_df
        );
        let mid = kernel_regression(
            x.as_ref(),
            &y,
            None,
            &kreg(kind, BandwidthSpec::Fixed(vec![0.5, 0.5])),
        )
        .unwrap();
        assert!(mid.effective_df > lo && mid.effective_df < n);
    }
}

#[test]
fn wide_bandwidth_limits_are_the_global_constant_and_ols_fits() {
    let (x, y) = nonlinear_design(4, 70, 2, 0.3);
    let mean = y.iter().sum::<f64>() / 70.0;
    let nw = kernel_regression(
        x.as_ref(),
        &y,
        None,
        &kreg(
            RegressionKind::NadarayaWatson,
            BandwidthSpec::Fixed(vec![1e5, 1e5]),
        ),
    )
    .unwrap();
    for v in &nw.fitted {
        assert!((v - mean).abs() < 1e-6);
    }
    // Local linear -> OLS of y on [1, x].
    let z = Mat::from_fn(70, 3, |i, j| if j == 0 { 1.0 } else { x[(i, j - 1)] });
    let beta = ridge(z.as_ref(), &y, 0.0).unwrap();
    let ll = kernel_regression(
        x.as_ref(),
        &y,
        None,
        &kreg(
            RegressionKind::LocalLinear,
            BandwidthSpec::Fixed(vec![1e5, 1e5]),
        ),
    )
    .unwrap();
    for i in 0..70 {
        let ols = beta[0] + beta[1] * x[(i, 0)] + beta[2] * x[(i, 1)];
        assert!(
            (ll.fitted[i] - ols).abs() < 1e-6,
            "{i}: {} vs {ols}",
            ll.fitted[i]
        );
    }
}

#[test]
fn selected_bandwidth_is_a_local_minimum_of_its_criterion() {
    let (x, y) = nonlinear_design(6, 120, 1, 0.3);
    for (spec, half) in [
        (BandwidthSpec::LooCv, 0usize),
        (BandwidthSpec::BlockCv { block: Some(3) }, 3),
    ] {
        for kind in [RegressionKind::NadarayaWatson, RegressionKind::LocalLinear] {
            let fit = kernel_regression(x.as_ref(), &y, None, &kreg(kind, spec.clone())).unwrap();
            let h = fit.bandwidth[0];
            let at = cv_criterion(x.as_ref(), &y, &[h], kind, half).unwrap();
            assert_eq!(at, fit.cv_criterion);
            let up = cv_criterion(x.as_ref(), &y, &[h * 1.05], kind, half).unwrap();
            let dn = cv_criterion(x.as_ref(), &y, &[h / 1.05], kind, half).unwrap();
            assert!(at <= up && at <= dn, "{kind:?} {spec:?}: {dn} {at} {up}");
            assert_eq!(fit.bandwidth_method, spec.method_name());
            assert_eq!(fit.block, if half == 0 { None } else { Some(half) });
            assert!(fit.n_criterion_evaluations >= 21);
            assert!(!fit.bandwidth_at_boundary);
        }
    }
}

#[test]
fn block_cv_default_block_is_cube_root_of_n_and_criteria_nest() {
    let (x, y) = nonlinear_design(8, 100, 1, 0.3);
    let fit = kernel_regression(
        x.as_ref(),
        &y,
        None,
        &kreg(
            RegressionKind::LocalLinear,
            BandwidthSpec::BlockCv { block: None },
        ),
    )
    .unwrap();
    assert_eq!(fit.block, Some(5)); // ceil(100^(1/3)) = ceil(4.64) = 5
                                    // block = 0 in the standalone evaluator is the LOO criterion.
    let loo = kernel_regression(
        x.as_ref(),
        &y,
        None,
        &kreg(
            RegressionKind::LocalLinear,
            BandwidthSpec::Fixed(fit.bandwidth.clone()),
        ),
    )
    .unwrap();
    assert_eq!(
        loo.cv_criterion,
        cv_criterion(
            x.as_ref(),
            &y,
            &fit.bandwidth,
            RegressionKind::LocalLinear,
            0
        )
        .unwrap()
    );
}

#[test]
fn boundary_flag_fires_on_a_signal_free_target_and_not_on_a_clear_one() {
    let mut rng = Lcg::new(21);
    let n = 120;
    let xs: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let noise: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let x = mat_from_cols(std::slice::from_ref(&xs));
    // No signal: the criterion keeps falling toward the global fit.
    let flat = kernel_regression(
        x.as_ref(),
        &noise,
        None,
        &kreg(RegressionKind::LocalLinear, BandwidthSpec::LooCv),
    )
    .unwrap();
    assert!(flat.bandwidth_at_boundary, "bw={:?}", flat.bandwidth);
    // A clear signal: interior optimum.
    let y: Vec<f64> = (0..n)
        .map(|i| (2.0 * xs[i]).sin() + 0.2 * noise[i])
        .collect();
    let clear = kernel_regression(
        x.as_ref(),
        &y,
        None,
        &kreg(RegressionKind::LocalLinear, BandwidthSpec::LooCv),
    )
    .unwrap();
    assert!(!clear.bandwidth_at_boundary, "bw={:?}", clear.bandwidth);
}

/// Serially correlated errors make leave-one-out undersmooth; the block
/// criterion picks a wider bandwidth on most seeds.
#[test]
fn block_cv_selects_wider_bandwidths_than_loo_under_ar1_errors() {
    let n = 200;
    let rho = 0.9;
    let mut wider = 0;
    let n_seeds = 10;
    for seed in 0..n_seeds {
        let mut rng = Lcg::new(100 + seed);
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 6.0 - 3.0).collect();
        let mut e = vec![0.0; n];
        for t in 1..n {
            e[t] = rho * e[t - 1] + 0.5 * rng.normal();
        }
        let y: Vec<f64> = (0..n).map(|i| (xs[i]).sin() + e[i]).collect();
        let x = mat_from_cols(&[xs]);
        let loo = kernel_regression(
            x.as_ref(),
            &y,
            None,
            &kreg(RegressionKind::LocalLinear, BandwidthSpec::LooCv),
        )
        .unwrap();
        let block = kernel_regression(
            x.as_ref(),
            &y,
            None,
            &kreg(
                RegressionKind::LocalLinear,
                BandwidthSpec::BlockCv { block: Some(10) },
            ),
        )
        .unwrap();
        println!(
            "seed {seed}: loo h={:.4} block h={:.4}",
            loo.bandwidth[0], block.bandwidth[0]
        );
        if block.bandwidth[0] > loo.bandwidth[0] {
            wider += 1;
        }
    }
    println!("block-CV wider than LOO on {wider}/{n_seeds} seeds");
    assert!(wider >= 8, "block-CV wider on only {wider}/{n_seeds} seeds");
}

#[test]
fn predictions_at_the_training_rows_reproduce_the_fitted_values() {
    let (x, y) = nonlinear_design(13, 60, 3, 0.3);
    for kind in [RegressionKind::NadarayaWatson, RegressionKind::LocalLinear] {
        let fit = kernel_regression(
            x.as_ref(),
            &y,
            Some(x.as_ref()),
            &kreg(kind, BandwidthSpec::Fixed(vec![0.6, 0.6, 0.6])),
        )
        .unwrap();
        let pred = fit.predicted.as_ref().unwrap();
        assert_eq!(pred, &fit.fitted);
    }
}

#[test]
fn kernel_regression_teaching_errors_name_the_argument_and_fix() {
    let (x, y) = nonlinear_design(17, 40, 2, 0.3);
    let ll = RegressionKind::LocalLinear;
    let msg = |o: &KernelRegressionOptions| {
        kernel_regression(x.as_ref(), &y, None, o)
            .unwrap_err()
            .to_string()
    };

    let m = msg(&kreg(ll, BandwidthSpec::Fixed(vec![0.5, -0.1])));
    assert!(
        m.contains("bandwidth[1]=-0.1") && m.contains("positive"),
        "{m}"
    );
    let m = msg(&kreg(ll, BandwidthSpec::Fixed(vec![0.5])));
    assert!(
        m.contains("dimension mismatch") && m.contains("expected 2, got 1"),
        "{m}"
    );
    let m = msg(&kreg(ll, BandwidthSpec::BlockCv { block: Some(0) }));
    assert!(m.contains("block=0") && m.contains("loo_cv"), "{m}");
    let m = RegressionKind::parse("loess").unwrap_err().to_string();
    assert!(
        m.contains("\"local_linear\"") && m.contains("\"nadaraya_watson\""),
        "{m}"
    );
    let m = RegressionKernel::parse("tricube").unwrap_err().to_string();
    assert!(m.contains("\"gaussian\""), "{m}");

    // Too many columns.
    let wide = Mat::<f64>::zeros(40, 4);
    let m = kernel_regression(wide.as_ref(), &y, None, &kreg(ll, BandwidthSpec::LooCv))
        .unwrap_err()
        .to_string();
    assert!(m.contains("4 columns") && m.contains("kernel_ridge"), "{m}");

    // A constant column under a CV method.
    let cols = vec![(0..40).map(|i| (i as f64).sin()).collect(), vec![2.0; 40]];
    let xc = mat_from_cols(&cols);
    let m = kernel_regression(xc.as_ref(), &y, None, &kreg(ll, BandwidthSpec::LooCv))
        .unwrap_err()
        .to_string();
    assert!(m.contains("column 1 of x is constant"), "{m}");
    // ...but it is fine at a fixed bandwidth.
    assert!(kernel_regression(
        xc.as_ref(),
        &y,
        None,
        &kreg(ll, BandwidthSpec::Fixed(vec![0.5, 1.0]))
    )
    .is_ok());

    // Non-finite inputs name the array.
    let mut xn = x.clone();
    xn[(2, 0)] = f64::NAN;
    assert_eq!(
        kernel_regression(xn.as_ref(), &y, None, &kreg(ll, BandwidthSpec::LooCv)).unwrap_err(),
        MlError::NonFinite { what: "x" }
    );
    assert_eq!(
        kernel_regression(
            x.as_ref(),
            &y,
            Some(xn.as_ref()),
            &kreg(ll, BandwidthSpec::LooCv)
        )
        .unwrap_err(),
        MlError::NonFinite { what: "x_test" }
    );

    // Insufficiency uses the house wording with the exact minimum.
    let small = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
    let e = kernel_regression(
        small.as_ref(),
        &y[..3],
        None,
        &kreg(ll, BandwidthSpec::LooCv),
    )
    .unwrap_err();
    assert_eq!(e, MlError::InsufficientData { needed: 4, got: 3 });
    assert_eq!(
        e.to_string(),
        "insufficient data: 3 observations, at least 4 required"
    );
    // Empty input never panics.
    let empty = Mat::<f64>::zeros(0, 1);
    assert!(kernel_regression(empty.as_ref(), &[], None, &kreg(ll, BandwidthSpec::LooCv)).is_err());
    assert!(kernel_ridge(empty.as_ref(), &[], None, &KernelRidgeOptions::default()).is_err());
}
