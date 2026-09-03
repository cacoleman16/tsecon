//! Golden-value tests against `fixtures/neural.json`.
//!
//! MLP — independent package (scikit-learn 1.9.0 `MLPRegressor`). The
//! fixture stores sklearn's *fitted* weights per (architecture,
//! activation) case, so nothing here reproduces an optimizer trajectory;
//! the pins are on the mechanics evaluated at those weights:
//!
//! * forward pass = sklearn `predict` on held-out rows (1e-12);
//! * objective at the fitted weights = the sklearn-convention loss
//!   recomputed in the generator and read back from `est.loss_` (1e-10);
//! * analytic gradient = sklearn's own `_backprop` at random and at fitted
//!   weights (1e-10), and = a central finite difference of our own loss on
//!   the smooth activations (1e-6 relative);
//! * gradient inf-norm at sklearn's converged weights = the norm the
//!   generator measured there (1e-8), below the stationarity bar.
//!
//! ESN — the state path is pinned against a NumPy transcription of the
//! leaky-integrator recursion that `reservoirpy` 0.4.2 reproduced exactly
//! at generation time (`_meta.esn.reservoirpy`), the readout against the
//! closed-form ridge cross-checked there against scikit-learn `Ridge`, and
//! the spectral radius against `numpy.linalg.eigvals`.

mod common;

use common::{as_f64_vec, as_mat, assert_slice_close, load_fixture};
use serde_json::Value;
use tsecon_ml::faer::Mat;
use tsecon_ml::{
    esn_readout, esn_states, mlp_forward, mlp_loss, mlp_loss_gradient, scale_to_spectral_radius,
    spectral_radius, Activation, MlpWeights,
};
use tsecon_optim::{central_difference_gradient, FnObjective};

fn weights_from(coefs: &Value, intercepts: &Value) -> MlpWeights {
    MlpWeights {
        coefs: coefs.as_array().unwrap().iter().map(as_mat).collect(),
        intercepts: intercepts
            .as_array()
            .unwrap()
            .iter()
            .map(as_f64_vec)
            .collect(),
    }
}

fn mlp_inputs(fx: &Value) -> (Mat<f64>, Vec<f64>, Mat<f64>) {
    (
        as_mat(&fx["mlp"]["x_train"]),
        as_f64_vec(&fx["mlp"]["y_train"]),
        as_mat(&fx["mlp"]["x_test"]),
    )
}

fn case_activation(case: &Value) -> Activation {
    Activation::parse(case["activation"].as_str().unwrap()).unwrap()
}

/// (a) The forward pass reproduces scikit-learn `predict` at its fitted
/// weights on the held-out rows, to 1e-12.
#[test]
fn golden_mlp_forward_matches_sklearn_predict() {
    let fx = load_fixture("neural.json");
    let (_x, _y, x_test) = mlp_inputs(&fx);
    let mut worst = 0.0f64;
    for case in fx["mlp"]["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let w = weights_from(&case["coefs"], &case["intercepts"]);
        let pred = mlp_forward(&w, case_activation(case), x_test.as_ref()).unwrap();
        let d = assert_slice_close(&pred, &as_f64_vec(&case["predict_test"]), 1e-12, name);
        worst = worst.max(d);
    }
    println!("mlp forward achieved max abs error: {worst:e}");
    assert!(worst < 1e-12);
}

/// (b) The objective at sklearn's fitted weights equals the
/// scikit-learn-convention loss (formula, `_backprop`, and `est.loss_`
/// agree in the fixture) to 1e-10; the same at random weights.
#[test]
fn golden_mlp_loss_matches_sklearn_objective() {
    let fx = load_fixture("neural.json");
    let (x, y, _) = mlp_inputs(&fx);
    let mut worst = 0.0f64;
    for case in fx["mlp"]["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let act = case_activation(case);
        let alpha = case["alpha"].as_f64().unwrap();
        let w = weights_from(&case["coefs"], &case["intercepts"]);
        let loss = mlp_loss(&w, act, x.as_ref(), &y, alpha).unwrap();
        let expected = case["loss_fitted"].as_f64().unwrap();
        let attr = case["loss_attr"].as_f64().unwrap();
        assert!(
            (expected - attr).abs() < 1e-12,
            "{name}: fixture loss_ mismatch"
        );
        let d = (loss - expected).abs();
        assert!(d < 1e-10, "{name}: loss {loss} vs {expected} (diff {d:e})");
        worst = worst.max(d);

        let rw = &case["random_weights"];
        let w_r = weights_from(&rw["coefs"], &rw["intercepts"]);
        let loss_r = mlp_loss(&w_r, act, x.as_ref(), &y, alpha).unwrap();
        let d = (loss_r - rw["loss"].as_f64().unwrap()).abs();
        assert!(d < 1e-10, "{name}: random-weight loss diff {d:e}");
        worst = worst.max(d);
    }
    println!("mlp loss achieved max abs error: {worst:e}");
}

fn assert_grad_close(
    got: &MlpWeights,
    coefs: &Value,
    intercepts: &Value,
    tol: f64,
    name: &str,
) -> f64 {
    let expected = weights_from(coefs, intercepts);
    let mut worst = 0.0f64;
    for (l, (g, e)) in got.coefs.iter().zip(&expected.coefs).enumerate() {
        assert_eq!(
            (g.nrows(), g.ncols()),
            (e.nrows(), e.ncols()),
            "{name}: coef shape"
        );
        for i in 0..g.nrows() {
            for j in 0..g.ncols() {
                let d = (g[(i, j)] - e[(i, j)]).abs();
                assert!(
                    d <= tol,
                    "{name}: dW[{l}][{i},{j}] {} vs {} (diff {d:e})",
                    g[(i, j)],
                    e[(i, j)]
                );
                worst = worst.max(d);
            }
        }
    }
    for (l, (g, e)) in got.intercepts.iter().zip(&expected.intercepts).enumerate() {
        let d = assert_slice_close(g, e, tol, &format!("{name}: db[{l}]"));
        worst = worst.max(d);
    }
    worst
}

/// (c)/(d) The analytic gradient equals scikit-learn's own `_backprop`
/// gradient at Glorot-scale random weights and at the fitted weights, to
/// 1e-10 — including the relu case, whose kinks make finite differences
/// the wrong yardstick.
#[test]
fn golden_mlp_gradient_matches_sklearn_backprop() {
    let fx = load_fixture("neural.json");
    let (x, y, _) = mlp_inputs(&fx);
    let mut worst = 0.0f64;
    for case in fx["mlp"]["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let act = case_activation(case);
        let alpha = case["alpha"].as_f64().unwrap();
        let rw = &case["random_weights"];
        let w_r = weights_from(&rw["coefs"], &rw["intercepts"]);
        let (loss_r, g_r) = mlp_loss_gradient(&w_r, act, x.as_ref(), &y, alpha).unwrap();
        assert!((loss_r - rw["loss"].as_f64().unwrap()).abs() < 1e-10);
        worst = worst.max(assert_grad_close(
            &g_r,
            &rw["grad_coefs"],
            &rw["grad_intercepts"],
            1e-10,
            &format!("{name} random"),
        ));
        let w = weights_from(&case["coefs"], &case["intercepts"]);
        let (_, g) = mlp_loss_gradient(&w, act, x.as_ref(), &y, alpha).unwrap();
        worst = worst.max(assert_grad_close(
            &g,
            &case["grad_fitted"]["coefs"],
            &case["grad_fitted"]["intercepts"],
            1e-10,
            &format!("{name} fitted"),
        ));
    }
    println!("mlp gradient vs sklearn backprop achieved max abs error: {worst:e}");
}

/// (c) On the smooth activations the analytic gradient equals a central
/// finite difference of our own loss at random weights, to 1e-6 relative
/// (absolute floor 1e-9 for entries near zero).
#[test]
fn golden_mlp_gradient_matches_central_difference() {
    let fx = load_fixture("neural.json");
    let (x, y, _) = mlp_inputs(&fx);
    let mut worst_rel = 0.0f64;
    for case in fx["mlp"]["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let act = case_activation(case);
        if act == Activation::Relu {
            continue; // piecewise linear: pinned against sklearn's backprop instead
        }
        let alpha = case["alpha"].as_f64().unwrap();
        let rw = &case["random_weights"];
        let w_r = weights_from(&rw["coefs"], &rw["intercepts"]);
        let units = w_r.layer_units().unwrap();
        let theta = w_r.to_flat();
        let (_, g) = mlp_loss_gradient(&w_r, act, x.as_ref(), &y, alpha).unwrap();
        let analytic = g.to_flat();
        let mut f = FnObjective::new(|th: &[f64]| {
            let w = MlpWeights::from_flat(&units, th).unwrap();
            mlp_loss(&w, act, x.as_ref(), &y, alpha).unwrap()
        });
        let fd = central_difference_gradient(&mut f, &theta);
        for (k, (a, d)) in analytic.iter().zip(&fd).enumerate() {
            let tol = 1e-6 * a.abs().max(1e-3);
            let diff = (a - d).abs();
            assert!(
                diff <= tol,
                "{name}: theta[{k}] analytic {a} vs fd {d} (diff {diff:e})"
            );
            worst_rel = worst_rel.max(diff / a.abs().max(1e-3));
        }
    }
    println!("mlp gradient vs central difference achieved max relative error: {worst_rel:e}");
}

/// (d) At scikit-learn's converged L-BFGS weights our gradient's inf-norm
/// reproduces the norm the generator measured there (1e-8) and sits below
/// the fixture's stationarity bar.
#[test]
fn golden_mlp_gradient_norm_at_sklearn_solution() {
    let fx = load_fixture("neural.json");
    let (x, y, _) = mlp_inputs(&fx);
    let bar = fx["_meta"]["mlp"]["stationary_bar_inf_norm"]
        .as_f64()
        .unwrap();
    let mut worst = 0.0f64;
    for case in fx["mlp"]["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let act = case_activation(case);
        let alpha = case["alpha"].as_f64().unwrap();
        let w = weights_from(&case["coefs"], &case["intercepts"]);
        let (_, g) = mlp_loss_gradient(&w, act, x.as_ref(), &y, alpha).unwrap();
        let inf = g.to_flat().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let two = g.to_flat().iter().map(|v| v * v).sum::<f64>().sqrt();
        let e_inf = case["grad_norm_inf_fitted"].as_f64().unwrap();
        let e_two = case["grad_norm_2_fitted"].as_f64().unwrap();
        assert!(
            (inf - e_inf).abs() < 1e-8,
            "{name}: |g|_inf {inf} vs {e_inf}"
        );
        assert!((two - e_two).abs() < 1e-8, "{name}: |g|_2 {two} vs {e_two}");
        assert!(
            inf < bar,
            "{name}: sklearn solution not stationary ({inf} >= {bar})"
        );
        worst = worst.max((inf - e_inf).abs());
        println!("{name}: |grad|_inf at sklearn solution {inf:.3e} (bar {bar:e})");
    }
    println!("mlp gradient norm achieved max abs error: {worst:e}");
}

/// The state recursion reproduces the NumPy transcription (which
/// reservoirpy 0.4.2 matched bit for bit at generation time) to 1e-12.
#[test]
fn golden_esn_states_match_transcription() {
    let fx = load_fixture("neural.json");
    let tr = &fx["esn"]["transcription"];
    let w = as_mat(&tr["w"]);
    let w_in = as_mat(&tr["w_in"]);
    let u = as_mat(&tr["u"]);
    let leak = tr["leak_rate"].as_f64().unwrap();
    let states = esn_states(w_in.as_ref(), w.as_ref(), u.as_ref(), leak).unwrap();
    let expected = as_mat(&tr["states"]);
    assert_eq!(
        (states.nrows(), states.ncols()),
        (expected.nrows(), expected.ncols())
    );
    let mut worst = 0.0f64;
    for t in 0..states.nrows() {
        for i in 0..states.ncols() {
            let d = (states[(t, i)] - expected[(t, i)]).abs();
            assert!(
                d < 1e-12,
                "state[{t},{i}] {} vs {}",
                states[(t, i)],
                expected[(t, i)]
            );
            worst = worst.max(d);
        }
    }
    let meta = &fx["_meta"]["esn"]["reservoirpy"];
    println!(
        "esn states achieved max abs error: {worst:e}; reservoirpy installed at generation: {} \
         (state-path diff {:?})",
        meta["installed"], meta["max_abs_state_diff"]
    );
}

/// The ridge readout on `[1, u, s]` after the washout reproduces the
/// closed form `(Z'Z + alpha I)^{-1} Z'y` (cross-checked in the generator
/// against scikit-learn `Ridge`) to 1e-10, and so do the fitted values.
#[test]
fn golden_esn_readout_matches_closed_form() {
    let fx = load_fixture("neural.json");
    let tr = &fx["esn"]["transcription"];
    let w = as_mat(&tr["w"]);
    let w_in = as_mat(&tr["w_in"]);
    let u = as_mat(&tr["u"]);
    let y = as_f64_vec(&tr["y"]);
    let leak = tr["leak_rate"].as_f64().unwrap();
    let washout = tr["washout"].as_u64().unwrap() as usize;
    let alpha = tr["ridge_alpha"].as_f64().unwrap();
    let states = esn_states(w_in.as_ref(), w.as_ref(), u.as_ref(), leak).unwrap();
    let b = esn_readout(states.as_ref(), u.as_ref(), &y, washout, alpha).unwrap();
    let d_b = assert_slice_close(&b, &as_f64_vec(&tr["readout"]), 1e-10, "readout");
    // fitted = Z b on the post-washout rows
    let p = u.ncols();
    let fitted: Vec<f64> = (washout..u.nrows())
        .map(|t| {
            let mut v = b[0];
            for j in 0..p {
                v += b[1 + j] * u[(t, j)];
            }
            for i in 0..states.ncols() {
                v += b[1 + p + i] * states[(t, i)];
            }
            v
        })
        .collect();
    let d_f = assert_slice_close(&fitted, &as_f64_vec(&tr["fitted"]), 1e-10, "fitted");
    let ridge_gap = tr["readout_sklearn_ridge_max_abs_diff"].as_f64().unwrap();
    assert!(ridge_gap < 1e-10);
    println!(
        "esn readout achieved max abs error: {d_b:e} (fitted {d_f:e}); generator's sklearn Ridge \
         cross-check gap {ridge_gap:e}"
    );
}

/// The spectral radius reproduces `numpy.linalg.eigvals` on the fixture
/// matrix, and rescaling to the target lands within 1e-6 of it.
#[test]
fn golden_esn_spectral_radius_matches_numpy_eigvals() {
    let fx = load_fixture("neural.json");
    let sp = &fx["esn"]["spectral"];
    let w = as_mat(&sp["w"]);
    let expected = sp["radius_numpy"].as_f64().unwrap();
    let rho = spectral_radius(w.as_ref()).unwrap();
    let rel = (rho - expected).abs() / expected;
    assert!(rel < 1e-6, "radius {rho} vs numpy {expected} (rel {rel:e})");
    let target = sp["target"].as_f64().unwrap();
    let (scaled, achieved) = scale_to_spectral_radius(w.as_ref(), target).unwrap();
    assert!(
        (achieved - target).abs() < 1e-6,
        "achieved {achieved} vs target {target}"
    );
    // The scaling is the exact ratio target / rho, entry for entry.
    let f = target / expected;
    let mut worst_entry = 0.0f64;
    for i in 0..w.nrows() {
        for j in 0..w.ncols() {
            worst_entry = worst_entry.max((scaled[(i, j)] - w[(i, j)] * f).abs());
        }
    }
    assert!(worst_entry < 1e-12);
    // The transcription reservoir was itself scaled to 0.8 by numpy eigvals.
    let tr = &fx["esn"]["transcription"];
    let rho_tr = spectral_radius(as_mat(&tr["w"]).as_ref()).unwrap();
    assert!((rho_tr - tr["spectral_radius"].as_f64().unwrap()).abs() < 1e-10);
    println!(
        "esn spectral radius achieved relative error: {rel:e}; scaled-to-target error {:e}",
        (achieved - target).abs()
    );
}

/// The fixture states its grades: the MLP leg is an independent-package
/// golden against the pinned scikit-learn version; the ESN leg records
/// whether reservoirpy pinned the state path.
#[test]
fn fixture_meta_states_the_grades() {
    let fx = load_fixture("neural.json");
    let meta = &fx["_meta"];
    assert_eq!(meta["sklearn"].as_str().unwrap(), "1.9.0");
    assert!(meta["mlp"]["grade"]
        .as_str()
        .unwrap()
        .contains("independent package"));
    assert!(meta["mlp"]["objective_note"]
        .as_str()
        .unwrap()
        .contains("(1/(2n))"));
    let rp = &meta["esn"]["reservoirpy"];
    assert!(rp["installed"].is_boolean());
    if rp["installed"].as_bool().unwrap() {
        assert!(rp["max_abs_state_diff"].as_f64().unwrap() < 1e-12);
        assert!(meta["esn"]["grade"]
            .as_str()
            .unwrap()
            .contains("third-party"));
    } else {
        assert!(meta["esn"]["grade"]
            .as_str()
            .unwrap()
            .contains("transcription"));
    }
}
