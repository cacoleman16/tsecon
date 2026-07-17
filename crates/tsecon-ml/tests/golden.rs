//! Golden-value tests against `fixtures/ml.json` (coefficient vectors from
//! scikit-learn 1.9.0 `Ridge`, `Lasso`, and `ElasticNet`, run to
//! `tol = 1e-12`). Every case must match to 1e-6 absolute; the harness
//! also prints the achieved tolerance, which for the coordinate-descent
//! solvers is far tighter.

mod common;

use common::{as_f64_vec, as_mat, assert_slice_close, load_fixture};
use tsecon_ml::{elastic_net, lasso, ridge, CoordDescentOptions};

/// scikit-learn's `Ridge` (objective `||y - Xb||^2 + alpha*||b||^2`, no
/// `1/n`) matches our SVD closed form to 1e-6; achieved is ~1e-9.
#[test]
fn golden_ridge() {
    let fx = load_fixture("ml.json");
    let x = as_mat(&fx["X_standardized"]);
    let y = as_f64_vec(&fx["y_centered"]);

    let mut worst = 0.0f64;
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if !name.starts_with("ridge") {
            continue;
        }
        let alpha = case["params"]["alpha"].as_f64().unwrap();
        let expected = as_f64_vec(&case["coef"]);
        let beta = ridge(x.as_ref(), &y, alpha).unwrap();
        let d = assert_slice_close(&beta, &expected, 1e-6, name);
        worst = worst.max(d);
    }
    println!("ridge achieved max abs error: {worst:e}");
    assert!(worst < 1e-6, "ridge worst error {worst:e} exceeds 1e-6");
}

/// scikit-learn's `Lasso` (objective `(1/(2n))||y - Xb||^2 + alpha*||b||_1`)
/// matches our coordinate descent to 1e-6; achieved is ~1e-9 or better.
#[test]
fn golden_lasso() {
    let fx = load_fixture("ml.json");
    let x = as_mat(&fx["X_standardized"]);
    let y = as_f64_vec(&fx["y_centered"]);
    let opts = CoordDescentOptions {
        tol: 1e-12,
        max_iter: 100_000,
    };

    let mut worst = 0.0f64;
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if !name.starts_with("lasso") {
            continue;
        }
        let alpha = case["params"]["alpha"].as_f64().unwrap();
        let expected = as_f64_vec(&case["coef"]);
        let fit = lasso(x.as_ref(), &y, alpha, opts).unwrap();
        let d = assert_slice_close(&fit.coef, &expected, 1e-6, name);
        worst = worst.max(d);
    }
    println!("lasso achieved max abs error: {worst:e}");
    assert!(worst < 1e-6, "lasso worst error {worst:e} exceeds 1e-6");
}

/// scikit-learn's `ElasticNet` (full objective with `l1_ratio`) matches our
/// coordinate descent to 1e-6; achieved is ~1e-9 or better.
#[test]
fn golden_elastic_net() {
    let fx = load_fixture("ml.json");
    let x = as_mat(&fx["X_standardized"]);
    let y = as_f64_vec(&fx["y_centered"]);
    let opts = CoordDescentOptions {
        tol: 1e-12,
        max_iter: 100_000,
    };

    let mut worst = 0.0f64;
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if !name.starts_with("enet") {
            continue;
        }
        let alpha = case["params"]["alpha"].as_f64().unwrap();
        let l1_ratio = case["params"]["l1_ratio"].as_f64().unwrap();
        let expected = as_f64_vec(&case["coef"]);
        let fit = elastic_net(x.as_ref(), &y, alpha, l1_ratio, opts).unwrap();
        let d = assert_slice_close(&fit.coef, &expected, 1e-6, name);
        worst = worst.max(d);
    }
    println!("elastic net achieved max abs error: {worst:e}");
    assert!(
        worst < 1e-6,
        "elastic net worst error {worst:e} exceeds 1e-6"
    );
}
