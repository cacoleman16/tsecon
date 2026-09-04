//! Property / Monte-Carlo tests for the neural estimators, beyond the
//! golden fixture (which pins the mechanics at fixed weights):
//!
//! * `mlp_regression` recovers a known nonlinear AR(1) map out of sample
//!   (R^2 above a documented bar, both solvers);
//! * the seed ensemble beats the median single member out of sample;
//! * early stopping fires on an easy problem and cannot fire at
//!   `max_epochs = 1`;
//! * the scaler is fit on the training rows only — perturbing the
//!   validation rows leaves it bit-identical (leakage safety);
//! * the seed contract (same seed bit-identical, different seeds differ)
//!   for both estimators;
//! * the sentinel refusals under `solver = Lbfgs` and the teaching errors;
//! * `echo_state_network` reaches a documented NARMA-10 NRMSE out of
//!   sample, hits its target spectral radius, and continues the state
//!   recursion across `x_test` exactly as one long run would.
//!
//! Wall-clock budgets are measured on the release wheel in
//! `bindings/python/tests/test_neural.py`, not here (this binary is
//! unoptimized).

mod common;

use common::{mat_from_cols, Lcg};
use tsecon_ml::faer::Mat;
use tsecon_ml::{
    echo_state_network, esn_states, esn_states_from, mlp_regression, Activation, EsnOptions,
    MlError, MlpFit, MlpOptions, Solver,
};

/// `y_t = sin(2 y_{t-1}) + sigma * e_t`, returned as (lag matrix, target).
fn sin_ar1(seed: u64, n: usize, sigma: f64) -> (Mat<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let mut y = vec![0.0f64; n + 101];
    for t in 1..y.len() {
        y[t] = (2.0 * y[t - 1]).sin() + sigma * rng.normal();
    }
    let y = &y[100..];
    let x = mat_from_cols(&[y[..n].to_vec()]);
    (x, y[1..=n].to_vec())
}

/// Out-of-sample R^2 in the Campbell-Thompson (2008) sense: the benchmark
/// is the TRAINING mean, so a test window that happens to sit in one
/// basin of the bistable map (sin(2y) has attracting fixed points near
/// +-0.95 that the noise flips between) is still scored against a
/// forecast a practitioner could have made.
fn r2(pred: &[f64], truth: &[f64], train_mean: f64) -> f64 {
    let sst: f64 = truth
        .iter()
        .map(|v| (v - train_mean) * (v - train_mean))
        .sum();
    let sse: f64 = pred.iter().zip(truth).map(|(p, t)| (p - t) * (p - t)).sum();
    1.0 - sse / sst
}

fn mse(pred: &[f64], truth: &[f64]) -> f64 {
    pred.iter()
        .zip(truth)
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f64>()
        / truth.len() as f64
}

fn head(x: &Mat<f64>, n: usize) -> Mat<f64> {
    Mat::from_fn(n, x.ncols(), |i, j| x[(i, j)])
}

fn tail(x: &Mat<f64>, from: usize) -> Mat<f64> {
    Mat::from_fn(x.nrows() - from, x.ncols(), |i, j| x[(from + i, j)])
}

/// Documented bars for the nonlinear AR(1) recovery (`sigma = 0.3`, 600
/// training / 100 test rows, Campbell-Thompson R^2). On the release wheel
/// across six data seeds the oracle map `sin(2 y_{t-1})` scores 0.78-0.90,
/// mini-batch Adam and L-BFGS 0.76-0.90, the all-defaults call 0.59-0.82,
/// and a linear AR(1) 0.46-0.76.
const SIN_AR1_SIGMA: f64 = 0.3;
const SIN_AR1_R2_BAR_ADAM_MINIBATCH: f64 = 0.6;
const SIN_AR1_R2_BAR_LBFGS: f64 = 0.6;

/// (fit, test targets, training mean)
fn fit_sin_ar1(opts: &MlpOptions) -> (MlpFit, Vec<f64>, f64) {
    let (x, y) = sin_ar1(7, 700, SIN_AR1_SIGMA);
    let n_train = 600;
    let fit = mlp_regression(
        head(&x, n_train).as_ref(),
        &y[..n_train],
        Some(tail(&x, n_train).as_ref()),
        opts,
    )
    .unwrap();
    let train_mean = y[..n_train].iter().sum::<f64>() / n_train as f64;
    (fit, y[n_train..].to_vec(), train_mean)
}

/// Out-of-sample R^2 of a linear AR(1) fit on the same split.
fn linear_r2() -> f64 {
    let (x, y) = sin_ar1(7, 700, SIN_AR1_SIGMA);
    let xs: Vec<f64> = (0..600).map(|i| x[(i, 0)]).collect();
    let mx = xs.iter().sum::<f64>() / 600.0;
    let my = y[..600].iter().sum::<f64>() / 600.0;
    let sxy: f64 = xs
        .iter()
        .zip(&y[..600])
        .map(|(a, b)| (a - mx) * (b - my))
        .sum();
    let sxx: f64 = xs.iter().map(|a| (a - mx) * (a - mx)).sum();
    let slope = sxy / sxx;
    let lin: Vec<f64> = (600..700).map(|i| my + slope * (x[(i, 0)] - mx)).collect();
    r2(&lin, &y[600..], my)
}

/// Out-of-sample recovery of `y_t = sin(2 y_{t-1}) + e_t` by the default
/// architecture with seeded mini-batch Adam, against the linear AR(1).
#[test]
fn mlp_recovers_nonlinear_ar1_adam() {
    let opts = MlpOptions {
        batch_size: Some(32),
        learning_rate: Some(1e-2),
        max_epochs: 200,
        ..MlpOptions::default()
    };
    let (fit, truth, train_mean) = fit_sin_ar1(&opts);
    let pred = fit.predicted.as_ref().unwrap();
    let score = r2(pred, &truth, train_mean);
    let lin = linear_r2();
    println!(
        "sin-AR(1) out-of-sample R^2 (adam, batch 32, lr 1e-2): {score:.4}; linear AR(1): {lin:.4}"
    );
    assert!(score > SIN_AR1_R2_BAR_ADAM_MINIBATCH, "R^2 {score}");
    assert!(score > lin + 0.1, "R^2 {score} vs linear {lin}");
}

/// The same recovery through `solver = Lbfgs`.
#[test]
fn mlp_recovers_nonlinear_ar1_lbfgs() {
    let opts = MlpOptions {
        solver: Solver::Lbfgs,
        max_epochs: 300,
        ..MlpOptions::default()
    };
    let (fit, truth, train_mean) = fit_sin_ar1(&opts);
    let pred = fit.predicted.as_ref().unwrap();
    let score = r2(pred, &truth, train_mean);
    println!("sin-AR(1) out-of-sample R^2 (lbfgs): {score:.4}");
    assert!(score > SIN_AR1_R2_BAR_LBFGS, "R^2 {score}");
    for (m, path) in fit.train_loss_path.iter().enumerate() {
        assert_eq!(
            path.len(),
            2,
            "lbfgs paths hold the initial and final objective"
        );
        assert!(
            path[1] <= path[0],
            "member {m} did not decrease the objective"
        );
        assert_eq!(fit.validation_loss_path[m].len(), 2);
    }
}

/// The default call (full-batch Adam, 500 epochs, 5 seeds) on the same DGP:
/// its R^2 is printed for the model card; the assertion is that it beats
/// the linear AR(1), since the spec fixes these defaults.
#[test]
fn mlp_default_call_on_nonlinear_ar1_is_reported() {
    let (fit, truth, train_mean) = fit_sin_ar1(&MlpOptions::default());
    let pred = fit.predicted.as_ref().unwrap();
    let score = r2(pred, &truth, train_mean);
    let lin = linear_r2();
    println!(
        "sin-AR(1) out-of-sample R^2 (defaults): {score:.4} vs linear {lin:.4}; best epochs {:?}; \
         converged {:?}",
        fit.best_epoch, fit.converged
    );
    assert!(score > lin, "defaults {score} vs linear {lin}");
}

/// The seed ensemble versus its members, out of sample, on a DGP where
/// members genuinely differ: `y_t = sin(2 y_{t-1}) + 0.7 e_t`, 60 training
/// rows, a (32, 16) net with `alpha = 1e-6` run to L-BFGS convergence with
/// no validation split — every member overfits the noise its own way.
/// Over 10 replications (data seeds) with 9 members each the ensemble mean
///
/// * beats the *mean* member MSE in 10/10 (Jensen's inequality: squared
///   error is convex, so this is a theorem, not luck), and
/// * beats the *median* member in a majority — measured 7/10 on these
///   draws (10/10 on the NumPy draws of `test_neural.py`); asserted at
///   >= 6/10.
///
/// The claim is therefore a majority-of-replications one: averaging does
/// not dominate the median member replication by replication, and one
/// wild member (a bad extrapolation) can drag the mean below the median in
/// a given replication — the pooled ensemble/median MSE ratio is printed,
/// not asserted, for that reason.
#[test]
fn mlp_ensemble_beats_median_member_out_of_sample() {
    let n_train = 60;
    let opts = MlpOptions {
        hidden: vec![32, 16],
        solver: Solver::Lbfgs,
        alpha: 1e-6,
        validation_fraction: 0.0,
        max_epochs: 500,
        n_seeds: 9,
        ..MlpOptions::default()
    };
    let mut wins_median = 0usize;
    let mut wins_mean = 0usize;
    let mut pooled_ens = 0.0;
    let mut pooled_median = 0.0;
    let mut ratios = Vec::new();
    for seed in 1..=10u64 {
        let (x, y) = sin_ar1(seed, n_train + 200, 0.7);
        let fit = mlp_regression(
            head(&x, n_train).as_ref(),
            &y[..n_train],
            Some(tail(&x, n_train).as_ref()),
            &opts,
        )
        .unwrap();
        let truth = &y[n_train..];
        let ens = mse(fit.predicted.as_ref().unwrap(), truth);
        let mut members: Vec<f64> = fit
            .member_predictions
            .as_ref()
            .unwrap()
            .iter()
            .map(|p| mse(p, truth))
            .collect();
        let mean = members.iter().sum::<f64>() / members.len() as f64;
        members.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = members[members.len() / 2];
        wins_median += usize::from(ens < median);
        wins_mean += usize::from(ens < mean);
        pooled_ens += ens;
        pooled_median += median;
        ratios.push(ens / median);
    }
    println!(
        "ensemble vs members over 10 replications: beats median {wins_median}/10, beats mean \
         {wins_mean}/10, pooled ensemble/median MSE ratio {:.3}, per-replication ratios {ratios:?}",
        pooled_ens / pooled_median
    );
    assert_eq!(
        wins_mean, 10,
        "Jensen: the ensemble must beat the mean member every time"
    );
    assert!(
        wins_median >= 6,
        "ensemble beat the median member in only {wins_median}/10"
    );
}

/// A linear target with tiny noise: early stopping fires for every member
/// well before the epoch budget (`converged = true`); with
/// `max_epochs = 1` it cannot (`converged = false`, paths of length 1).
#[test]
fn mlp_early_stopping_fires_on_easy_problem_and_not_at_one_epoch() {
    let mut rng = Lcg::new(11);
    let n = 300;
    let x = mat_from_cols(&[
        (0..n).map(|_| rng.normal()).collect(),
        (0..n).map(|_| rng.normal()).collect(),
    ]);
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * x[(i, 0)] - x[(i, 1)] + 0.01 * rng.normal())
        .collect();
    let opts = MlpOptions {
        batch_size: Some(32),
        learning_rate: Some(1e-2),
        patience: Some(10),
        max_epochs: 500,
        ..MlpOptions::default()
    };
    let fit = mlp_regression(x.as_ref(), &y, None, &opts).unwrap();
    println!(
        "easy problem: best epochs {:?}, converged {:?}",
        fit.best_epoch, fit.converged
    );
    assert!(fit.converged.iter().all(|&c| c));
    for (m, path) in fit.train_loss_path.iter().enumerate() {
        assert!(path.len() < 500, "member {m} ran the full budget");
        assert!(fit.best_epoch[m] >= 1 && fit.best_epoch[m] <= path.len());
        assert_eq!(fit.validation_loss_path[m].len(), path.len());
    }
    let my = y.iter().sum::<f64>() / y.len() as f64;
    assert!(r2(&fit.fitted, &y, my) > 0.99);

    let one = mlp_regression(
        x.as_ref(),
        &y,
        None,
        &MlpOptions {
            max_epochs: 1,
            ..opts.clone()
        },
    )
    .unwrap();
    assert!(one.converged.iter().all(|&c| !c));
    assert!(one.train_loss_path.iter().all(|p| p.len() == 1));
    assert!(one.best_epoch.iter().all(|&e| e == 1));
}

/// Same seed: bit-identical output. Different seed: different output.
/// Explicit Adam defaults: bit-identical to the `None` sentinels.
#[test]
fn mlp_seed_contract_and_sentinel_defaults() {
    let (x, y) = sin_ar1(3, 200, 0.3);
    let base = MlpOptions {
        max_epochs: 50,
        n_seeds: 3,
        ..MlpOptions::default()
    };
    let a = mlp_regression(x.as_ref(), &y, Some(x.as_ref()), &base).unwrap();
    let b = mlp_regression(x.as_ref(), &y, Some(x.as_ref()), &base).unwrap();
    assert_eq!(a, b, "same seed must be bit-identical");
    let c = mlp_regression(
        x.as_ref(),
        &y,
        Some(x.as_ref()),
        &MlpOptions {
            seed: 1,
            ..base.clone()
        },
    )
    .unwrap();
    assert_ne!(a.fitted, c.fitted, "different seeds must differ");
    let explicit = mlp_regression(
        x.as_ref(),
        &y,
        Some(x.as_ref()),
        &MlpOptions {
            learning_rate: Some(1e-3),
            patience: Some(20),
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(
        a, explicit,
        "explicit Adam defaults must equal the sentinels"
    );
    assert_eq!(a.n_parameters, 16 + 16 + 16 + 1); // W1 (1x16), b1, W2 (16x1), b2
    assert_eq!(a.weights.len(), 3);
    assert_eq!(a.member_predictions.as_ref().unwrap().len(), 3);
}

/// Leakage safety: the scaler depends on the training rows only, so
/// perturbing the validation rows (the last 20%) leaves `x_mean`,
/// `x_scale`, `y_mean`, `y_scale` bit-identical.
#[test]
fn mlp_scaler_is_fit_on_training_rows_only() {
    let (x, y) = sin_ar1(5, 250, 0.3);
    let opts = MlpOptions {
        max_epochs: 20,
        n_seeds: 2,
        ..MlpOptions::default()
    };
    let base = mlp_regression(x.as_ref(), &y, None, &opts).unwrap();
    assert_eq!(base.n_validation, 50);
    assert_eq!(base.n_train, 200);
    let mut x2 = x.clone();
    let mut y2 = y.clone();
    for i in 200..250 {
        x2[(i, 0)] += 100.0 * (i as f64);
        y2[i] -= 50.0;
    }
    let pert = mlp_regression(x2.as_ref(), &y2, None, &opts).unwrap();
    assert_eq!(base.x_mean, pert.x_mean);
    assert_eq!(base.x_scale, pert.x_scale);
    assert_eq!(base.y_mean, pert.y_mean);
    assert_eq!(base.y_scale, pert.y_scale);
    // ...whereas perturbing a TRAINING row does move the scaler.
    let mut x3 = x.clone();
    x3[(0, 0)] += 100.0;
    let moved = mlp_regression(x3.as_ref(), &y, None, &opts).unwrap();
    assert_ne!(base.x_mean, moved.x_mean);
    // Without standardization the reported scaler is the identity.
    let raw = mlp_regression(
        x.as_ref(),
        &y,
        None,
        &MlpOptions {
            standardize: false,
            ..opts.clone()
        },
    )
    .unwrap();
    assert_eq!(raw.x_mean, vec![0.0]);
    assert_eq!(raw.x_scale, vec![1.0]);
    assert_eq!((raw.y_mean, raw.y_scale), (0.0, 1.0));
}

/// Epoch-wise arguments are refused under L-BFGS (sentinel convention),
/// naming the argument; the teaching errors name the array, the accepted
/// choices, the hidden-layer limit, and the insufficiency count.
#[test]
fn mlp_sentinels_and_teaching_errors() {
    let (x, y) = sin_ar1(9, 60, 0.3);
    let lb = MlpOptions {
        solver: Solver::Lbfgs,
        max_epochs: 5,
        n_seeds: 1,
        ..MlpOptions::default()
    };
    for (opts, name) in [
        (
            MlpOptions {
                learning_rate: Some(0.01),
                ..lb.clone()
            },
            "learning_rate=0.01",
        ),
        (
            MlpOptions {
                batch_size: Some(8),
                ..lb.clone()
            },
            "batch_size=8",
        ),
        (
            MlpOptions {
                patience: Some(3),
                ..lb.clone()
            },
            "patience=3",
        ),
    ] {
        match mlp_regression(x.as_ref(), &y, None, &opts) {
            Err(MlError::InvalidValue { what }) => {
                assert!(what.contains(name) && what.contains("lbfgs"), "{what}")
            }
            other => panic!("expected refusal for {name}, got {other:?}"),
        }
    }
    assert!(mlp_regression(x.as_ref(), &y, None, &lb).is_ok());

    let quick = MlpOptions {
        max_epochs: 2,
        n_seeds: 1,
        ..MlpOptions::default()
    };
    let mut xn = x.clone();
    xn[(3, 0)] = f64::NAN;
    assert_eq!(
        mlp_regression(xn.as_ref(), &y, None, &quick),
        Err(MlError::NonFinite { what: "x" })
    );
    let mut yn = y.clone();
    yn[3] = f64::INFINITY;
    assert_eq!(
        mlp_regression(x.as_ref(), &yn, None, &quick),
        Err(MlError::NonFinite { what: "y" })
    );
    assert_eq!(
        mlp_regression(x.as_ref(), &y, Some(xn.as_ref()), &quick),
        Err(MlError::NonFinite { what: "x_test" })
    );
    // Insufficiency counts the validation split: with 20% held out the
    // smallest feasible sample is 5 (1 validation row, 4 training rows).
    let err = mlp_regression(head(&x, 4).as_ref(), &y[..4], None, &quick).unwrap_err();
    assert_eq!(
        err,
        MlError::InsufficientData {
            needed: 5,
            got: 4,
            what: "mlp_regression"
        }
    );
    assert_eq!(
        err.to_string(),
        "insufficient data: 4 observations, at least 5 required (mlp_regression)"
    );
    assert!(mlp_regression(head(&x, 5).as_ref(), &y[..5], None, &quick).is_ok());

    match Activation::parse("swish") {
        Err(MlError::UnknownChoice {
            what,
            got,
            accepted,
        }) => {
            assert_eq!((what, got.as_str()), ("activation", "swish"));
            assert!(accepted.contains("tanh") && accepted.contains("logistic"));
        }
        other => panic!("{other:?}"),
    }
    assert!(Solver::parse("sgd")
        .unwrap_err()
        .to_string()
        .contains("\"adam\", \"lbfgs\""));
    for hidden in [vec![], vec![4, 4, 4], vec![4, 0]] {
        match mlp_regression(
            x.as_ref(),
            &y,
            None,
            &MlpOptions {
                hidden: hidden.clone(),
                ..quick.clone()
            },
        ) {
            Err(MlError::InvalidValue { what }) => assert!(what.contains("hidden"), "{what}"),
            other => panic!("hidden={hidden:?}: {other:?}"),
        }
    }
    match mlp_regression(
        x.as_ref(),
        &y,
        None,
        &MlpOptions {
            batch_size: Some(1000),
            ..quick.clone()
        },
    ) {
        Err(MlError::InvalidValue { what }) => assert!(what.contains("batch_size=1000")),
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------------------- ESN

/// NARMA-10 (Atiya & Parlos 2000): `y_{t+1} = 0.3 y_t + 0.05 y_t
/// sum_{i=0}^{9} y_{t-i} + 1.5 u_{t-9} u_t + 0.1`, `u ~ U(0, 0.5)`.
fn narma10(seed: u64, n: usize) -> (Mat<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let u: Vec<f64> = (0..n).map(|_| 0.5 * rng.uniform()).collect();
    let mut y = vec![0.0; n];
    for t in 9..n - 1 {
        let s: f64 = y[t - 9..=t].iter().sum();
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * s + 1.5 * u[t - 9] * u[t] + 0.1;
    }
    (mat_from_cols(&[u]), y)
}

fn nrmse(pred: &[f64], truth: &[f64]) -> f64 {
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let var = truth.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / truth.len() as f64;
    (mse(pred, truth) / var).sqrt()
}

/// Documented NARMA-10 bar: the default reservoir (N = 200, radius 0.9,
/// leak 1, connectivity 0.1, washout 50, ridge 1e-6) with
/// `input_scaling = 0.3` — the one knob NARMA-10's `u in [0, 0.5]` needs,
/// since `input_scaling = 1` drives the tanh units toward saturation —
/// on 1000 training and 200 continuation rows, averaged over four data
/// seeds. Measured 0.317 (per seed 0.26-0.41); the all-defaults call
/// averages 0.46 over the first two seeds (0.36-0.56 over four) and is
/// printed for the model card. The N = 400 / 2000-row regime (measured
/// 0.19) is checked in `test_neural.py` on the release wheel, where it is
/// cheap.
const NARMA10_NRMSE_BAR: f64 = 0.4;
const NARMA10_INPUT_SCALING: f64 = 0.3;

#[test]
fn esn_narma10_out_of_sample_nrmse_below_bar() {
    let n_train = 1000;
    let documented = EsnOptions {
        input_scaling: NARMA10_INPUT_SCALING,
        ..EsnOptions::default()
    };
    let mut sum_doc = 0.0;
    let mut sum_def = 0.0;
    let mut per_seed = Vec::new();
    for seed in 1..=4u64 {
        let (x, y) = narma10(seed, 1200);
        let fit = echo_state_network(
            head(&x, n_train).as_ref(),
            &y[..n_train],
            Some(tail(&x, n_train).as_ref()),
            &documented,
        )
        .unwrap();
        let oos = nrmse(fit.predicted.as_ref().unwrap(), &y[n_train..]);
        let ins = nrmse(&fit.fitted, &y[fit.n_washout..n_train]);
        per_seed.push((oos * 1e4).round() / 1e4);
        sum_doc += oos;
        assert_eq!(fit.fitted.len(), n_train - 50);
        assert_eq!(fit.readout.len(), 1 + 1 + 200);
        assert_eq!(
            (fit.reservoir_size, fit.n_washout, fit.n_train),
            (200, 50, 950)
        );
        assert!((fit.spectral_radius_achieved - 0.9).abs() < 1e-6);
        assert!(
            ins < oos + 0.1,
            "in-sample {ins} should not exceed out-of-sample {oos} by much"
        );
        if seed <= 2 {
            let def = echo_state_network(
                head(&x, n_train).as_ref(),
                &y[..n_train],
                Some(tail(&x, n_train).as_ref()),
                &EsnOptions::default(),
            )
            .unwrap();
            sum_def += nrmse(def.predicted.as_ref().unwrap(), &y[n_train..]);
        }
    }
    let avg_doc = sum_doc / 4.0;
    let avg_def = sum_def / 2.0;
    println!(
        "NARMA-10 out-of-sample NRMSE, input_scaling 0.3: mean {avg_doc:.4} over seeds {per_seed:?}; \
         all defaults (input_scaling 1): mean {avg_def:.4}"
    );
    assert!(avg_doc < NARMA10_NRMSE_BAR, "mean NRMSE {avg_doc}");
}

/// Same seed bit-identical; different seed differs; the achieved radius
/// tracks the requested one across targets (including above 1, accepted).
#[test]
fn esn_seed_contract_and_spectral_radius_targets() {
    let (x, y) = narma10(4, 300);
    let opts = EsnOptions {
        reservoir_size: 50,
        ..EsnOptions::default()
    };
    let a = echo_state_network(x.as_ref(), &y, Some(x.as_ref()), &opts).unwrap();
    let b = echo_state_network(x.as_ref(), &y, Some(x.as_ref()), &opts).unwrap();
    assert_eq!(a, b);
    let c = echo_state_network(
        x.as_ref(),
        &y,
        Some(x.as_ref()),
        &EsnOptions {
            seed: 9,
            ..opts.clone()
        },
    )
    .unwrap();
    assert_ne!(a.readout, c.readout);
    for target in [0.5, 0.9, 1.25] {
        let f = echo_state_network(
            x.as_ref(),
            &y,
            None,
            &EsnOptions {
                spectral_radius: target,
                ..opts.clone()
            },
        )
        .unwrap();
        assert!(
            (f.spectral_radius_achieved - target).abs() < 1e-6,
            "{target}"
        );
    }
}

/// `x_test` is the continuation of `x`: the states it produces equal the
/// tail of one long run on the concatenated inputs (to 1e-14).
#[test]
fn esn_prediction_continues_the_state_recursion() {
    let mut rng = Lcg::new(6);
    let n_units = 8;
    let w = Mat::from_fn(n_units, n_units, |_, _| {
        if rng.uniform() < 0.4 {
            0.3 * rng.normal()
        } else {
            0.0
        }
    });
    let w_in = Mat::from_fn(n_units, 2, |_, _| rng.normal());
    let x = Mat::from_fn(60, 2, |_, _| rng.normal());
    let long = esn_states(w_in.as_ref(), w.as_ref(), x.as_ref(), 0.6).unwrap();
    let first = esn_states(w_in.as_ref(), w.as_ref(), head(&x, 40).as_ref(), 0.6).unwrap();
    let last: Vec<f64> = (0..n_units).map(|i| first[(39, i)]).collect();
    let cont =
        esn_states_from(w_in.as_ref(), w.as_ref(), tail(&x, 40).as_ref(), 0.6, &last).unwrap();
    for t in 0..20 {
        for i in 0..n_units {
            assert!((cont[(t, i)] - long[(40 + t, i)]).abs() < 1e-14);
        }
    }
}

/// Teaching errors: `washout >= n` names the fix, too few surviving rows
/// reports the insufficiency count (washout included), NaN names the
/// array, and the scalar domains are enforced.
#[test]
fn esn_teaching_errors() {
    let (x, y) = narma10(8, 40);
    let small = EsnOptions {
        reservoir_size: 10,
        washout: 5,
        ..EsnOptions::default()
    };
    assert!(echo_state_network(x.as_ref(), &y, None, &small).is_ok());
    match echo_state_network(
        x.as_ref(),
        &y,
        None,
        &EsnOptions {
            washout: 40,
            ..small.clone()
        },
    ) {
        Err(MlError::InvalidValue { what }) => {
            assert!(
                what.contains("washout=40") && what.contains("n=40"),
                "{what}"
            )
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        echo_state_network(
            x.as_ref(),
            &y,
            None,
            &EsnOptions {
                washout: 39,
                ..small.clone()
            }
        ),
        Err(MlError::InsufficientData {
            needed: 41,
            got: 40,
            what: "echo_state_network"
        })
    );
    let mut xn = x.clone();
    xn[(2, 0)] = f64::NAN;
    assert_eq!(
        echo_state_network(xn.as_ref(), &y, None, &small),
        Err(MlError::NonFinite { what: "x" })
    );
    assert_eq!(
        echo_state_network(x.as_ref(), &y, Some(xn.as_ref()), &small),
        Err(MlError::NonFinite { what: "x_test" })
    );
    let mut yn = y.clone();
    yn[0] = f64::NAN;
    assert_eq!(
        echo_state_network(x.as_ref(), &yn, None, &small),
        Err(MlError::NonFinite { what: "y" })
    );
    for (opts, key) in [
        (
            EsnOptions {
                leak_rate: 0.0,
                ..small.clone()
            },
            "leak_rate",
        ),
        (
            EsnOptions {
                leak_rate: 1.5,
                ..small.clone()
            },
            "leak_rate",
        ),
        (
            EsnOptions {
                sparsity: 0.0,
                ..small.clone()
            },
            "sparsity",
        ),
        (
            EsnOptions {
                reservoir_size: 0,
                ..small.clone()
            },
            "reservoir_size",
        ),
        (
            EsnOptions {
                spectral_radius: 0.0,
                ..small.clone()
            },
            "spectral_radius",
        ),
        (
            EsnOptions {
                input_scaling: -1.0,
                ..small.clone()
            },
            "input_scaling",
        ),
        (
            EsnOptions {
                ridge_alpha: -1.0,
                ..small.clone()
            },
            "ridge_alpha",
        ),
    ] {
        let err = echo_state_network(x.as_ref(), &y, None, &opts).unwrap_err();
        assert!(err.to_string().contains(key), "{key}: {err}");
    }
}
