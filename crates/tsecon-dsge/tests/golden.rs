//! Documented-closed-form golden tests for the Blanchard-Kahn solver
//! (validation target (b)).
//!
//! `fixtures/tsecon-dsge.json` is produced by
//! `fixtures/generate_tsecon-dsge_fixtures.py`, whose docstring writes out the
//! closed-form solution of two textbook forward-looking models:
//!
//!   * the Cagan / asset-price model `p_t = a E_t[p_{t+1}] + u_t` with an AR(1)
//!     fundamental, whose fundamental solution `p_t = u_t / (1 - a rho)` gives
//!     `G = 1/(1 - a rho)`, `P = rho`, `Q = sigma`;
//!   * a two-shock variant `p_t = a E_t[p_{t+1}] + u1_t + u2_t` with
//!     `G = [1/(1 - a rho1), 1/(1 - a rho2)]`, diagonal `P` and `Q`.
//!
//! The generator types those analytic matrices straight from the derivation and
//! never calls this Rust solver, and it independently re-derives the
//! eigenvalues via `numpy.linalg.eigvals` (a separate code path). Reproducing
//! the matrices and eigenvalues to ~1e-8 is therefore a genuine, non-circular
//! check.

use serde_json::Value;
use tsecon_dsge::{solve, LinearReModel};

/// Closed-form match tolerance.
const TOL: f64 = 1e-8;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-dsge.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .expect("matrix array")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("row array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect()
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn close(actual: f64, expected: f64, what: &str) {
    let err = (actual - expected).abs();
    assert!(
        err < TOL,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

fn check_matrix(g: &tsecon_linalg::faer::Mat<f64>, expected: &[Vec<f64>], what: &str) {
    assert_eq!(g.nrows(), expected.len(), "{what} row count");
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(g.ncols(), row.len(), "{what} col count");
        for (j, &e) in row.iter().enumerate() {
            close(g[(i, j)], e, &format!("{what}[{i},{j}]"));
        }
    }
}

fn check_block(fx: &Value) {
    let a = rows(&fx["A"]);
    let b = rows(&fx["B"]);
    let c = rows(&fx["C"]);
    let n_pre = fx["n_predetermined"].as_u64().expect("n_predetermined") as usize;
    let model = LinearReModel::new(&a, &b, &c, n_pre).expect("valid model");
    let sol = solve(&model).expect("unique solution");

    assert!(sol.verdict().is_unique(), "expected a unique BK solution");
    check_matrix(sol.g(), &rows(&fx["G"]), "G");
    check_matrix(sol.p(), &rows(&fx["P"]), "P");
    check_matrix(sol.q(), &rows(&fx["Q"]), "Q");

    // Eigenvalues (real in both golden models), compared as sorted moduli and
    // sorted real parts.
    let mut mods: Vec<f64> = sol.eigenvalues().iter().map(|z| z.re.hypot(z.im)).collect();
    mods.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
    for (got, exp) in mods.iter().zip(f64s(&fx["eigenvalues_sorted_abs"])) {
        close(*got, exp, "eigenvalue modulus");
    }
    let mut reals: Vec<f64> = sol.eigenvalues().iter().map(|z| z.re).collect();
    reals.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
    for (got, exp) in reals.iter().zip(f64s(&fx["eigenvalues_real_sorted"])) {
        close(*got, exp, "eigenvalue real part");
    }
}

#[test]
fn cagan_matches_closed_form() {
    let fx = load();
    check_block(&fx["cagan"]);
}

#[test]
fn two_shock_matches_closed_form() {
    let fx = load();
    check_block(&fx["two_shock"]);
}

/// The Cagan policy rule reproduces the fundamental solution
/// `p_t = u_t / (1 - a rho)` exactly, with a = 0.5, rho = 0.6.
#[test]
fn cagan_fundamental_value() {
    let fx = load();
    let block = &fx["cagan"];
    let a = rows(&block["A"]);
    let b = rows(&block["B"]);
    let c = rows(&block["C"]);
    let model = LinearReModel::new(&a, &b, &c, 1).expect("model");
    let sol = solve(&model).expect("solution");
    // 1 / (1 - a rho) = 1 / (1 - 0.5 * 0.6) = 1 / 0.7.
    close(sol.g()[(0, 0)], 1.0 / 0.7, "G = 1/(1 - a rho)");
}
