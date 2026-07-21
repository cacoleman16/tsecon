//! Golden-value tests for smooth local projections against
//! `fixtures/smoothlp.json` (`fixtures/generate_smoothlp_fixtures.py`).
//!
//! Every pinned quantity was produced by an independent path: the basis by
//! `scipy.interpolate.BSpline.design_matrix` on the same uniform knot
//! vector, the per-horizon anchors by statsmodels OLS/HAC, and the stacked
//! penalized estimator (`theta`, IRF, sandwich SEs) plus the
//! leave-h-block-out CV scores by plain-NumPy normal equations transcribing
//! the closed form documented in the generator. Pins:
//!
//! * basis matrix: `1e-10` absolute (de Boor vs scipy);
//! * `theta` / `irf` / `se` at each fixture `lambda`: `1e-8` relative;
//! * `lambda = 0` IRF vs *statsmodels* per-horizon OLS beta: `1e-8`;
//! * `irf_raw` / `se_raw` vs statsmodels HAC (`maxlags = h + p`,
//!   `use_correction=True`): `1e-10` / `1e-8`;
//! * CV scores: `1e-8` relative; chosen `lambda`: exact grid value.

use serde_json::Value;
use tsecon_lp::{smooth_lp, SmoothLpSpec};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/smoothlp.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn usize_of(v: &Value) -> usize {
    v.as_u64().expect("unsigned integer") as usize
}

fn assert_close(actual: f64, expected: f64, atol: f64, rtol: f64, ctx: &str) {
    let tol = atol + rtol * expected.abs();
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > tol {tol:e}"
    );
}

/// Build the crate spec for a fixture case at a fixed lambda.
fn case_spec(case: &Value, lambda: f64) -> SmoothLpSpec {
    SmoothLpSpec::new(
        usize_of(&case["horizons"]),
        usize_of(&case["n_lag_controls"]),
    )
    .with_degree(usize_of(&case["degree"]))
    .with_n_basis(usize_of(&case["n_basis"]))
    .with_penalty_order(usize_of(&case["penalty_order"]))
    .with_lambda(lambda)
}

fn check_case(case: &Value, name: &str) {
    let y = f64s(&case["y"]);
    let e = f64s(&case["e"]);
    let hmax = usize_of(&case["horizons"]);
    let knots_fx = f64s(&case["knots"]);
    let basis_fx: Vec<Vec<f64>> = case["basis"]
        .as_array()
        .expect("basis rows")
        .iter()
        .map(f64s)
        .collect();

    for entry in case["smooth"].as_array().expect("smooth entries") {
        let lambda = entry["lambda"].as_f64().expect("lambda");
        let spec = case_spec(case, lambda);
        let res = smooth_lp(&y, &e, &spec).expect("smooth_lp");
        let ctx = |what: &str, i: usize| format!("{name} lambda={lambda} {what}[{i}]");

        // Basis and knots: de Boor vs scipy design_matrix.
        assert_eq!(res.knots.len(), knots_fx.len(), "{name}: knot count");
        for (i, (&a, &b)) in res.knots.iter().zip(&knots_fx).enumerate() {
            assert_close(a, b, 1e-12, 0.0, &ctx("knots", i));
        }
        for (h, row_fx) in basis_fx.iter().enumerate() {
            for (k, &b) in row_fx.iter().enumerate() {
                assert_close(
                    res.basis[h][k],
                    b,
                    1e-10,
                    0.0,
                    &format!("{name} basis[{h}][{k}]"),
                );
            }
        }

        let theta_fx = f64s(&entry["theta"]);
        let irf_fx = f64s(&entry["irf"]);
        let se_fx = f64s(&entry["se"]);
        assert_eq!(res.horizons, (0..=hmax).collect::<Vec<_>>());
        for (i, &t_fx) in theta_fx.iter().enumerate() {
            assert_close(res.theta[i], t_fx, 1e-10, 1e-8, &ctx("theta", i));
        }
        for h in 0..=hmax {
            assert_close(res.irf[h], irf_fx[h], 1e-10, 1e-8, &ctx("irf", h));
            assert_close(res.se[h], se_fx[h], 1e-10, 1e-8, &ctx("se", h));
        }
        assert_eq!(res.lambda, lambda, "{name}: lambda_used echoes the input");
    }
}

#[test]
fn case_a_matches_numpy_reference() {
    let fx = load();
    check_case(&fx["case_a"], "case_a");
}

#[test]
fn case_b_matches_numpy_reference() {
    // Non-default geometry: degree 2, K < H + 1, first-difference penalty.
    let fx = load();
    check_case(&fx["case_b"], "case_b");
}

#[test]
fn lambda_zero_matches_statsmodels_per_horizon_ols() {
    // With the interpolating basis (K = H + 1), lambda = 0 must reproduce
    // the per-horizon statsmodels OLS betas — a fully independent anchor.
    let fx = load();
    let case = &fx["case_a"];
    let y = f64s(&case["y"]);
    let e = f64s(&case["e"]);
    let res = smooth_lp(&y, &e, &case_spec(case, 0.0)).expect("smooth_lp lambda=0");

    for (h, entry) in case["perh"].as_array().expect("perh").iter().enumerate() {
        let beta = entry["beta"].as_f64().expect("beta");
        assert_close(
            res.irf[h],
            beta,
            1e-8,
            0.0,
            &format!("lambda=0 irf vs statsmodels beta h={h}"),
        );
        assert_eq!(
            res.nobs_per_h[h],
            usize_of(&entry["nobs"]),
            "h={h}: nobs matches the statsmodels sample"
        );
    }
}

#[test]
fn raw_lp_path_matches_statsmodels_hac() {
    // irf_raw / se_raw must be exactly the per-horizon HAC LP pinned in the
    // fixture (statsmodels maxlags = h + p, use_correction=True).
    let fx = load();
    let case = &fx["case_a"];
    let y = f64s(&case["y"]);
    let e = f64s(&case["e"]);
    let res = smooth_lp(&y, &e, &case_spec(case, 50.0)).expect("smooth_lp");

    for (h, entry) in case["perh"].as_array().expect("perh").iter().enumerate() {
        assert_close(
            res.irf_raw[h],
            entry["beta"].as_f64().expect("beta"),
            0.0,
            1e-10,
            &format!("irf_raw h={h}"),
        );
        assert_close(
            res.se_raw[h],
            entry["se_hac"].as_f64().expect("se_hac"),
            1e-8,
            0.0,
            &format!("se_raw h={h}"),
        );
    }
}

#[test]
fn cv_scores_and_choice_match_numpy_reference() {
    let fx = load();
    let case = &fx["case_a"];
    let y = f64s(&case["y"]);
    let e = f64s(&case["e"]);
    let cv = &case["cv"];
    let grid = f64s(&cv["grid"]);
    let n_folds = usize_of(&cv["n_folds"]);
    assert_eq!(
        usize_of(&cv["buffer"]),
        usize_of(&case["horizons"]) + usize_of(&case["n_lag_controls"]),
        "fixture buffer follows the horizons + n_lag_controls rule"
    );

    let spec = SmoothLpSpec::new(
        usize_of(&case["horizons"]),
        usize_of(&case["n_lag_controls"]),
    )
    .with_degree(usize_of(&case["degree"]))
    .with_n_basis(usize_of(&case["n_basis"]))
    .with_penalty_order(usize_of(&case["penalty_order"]))
    .with_cv(Some(grid.clone()), n_folds);
    let res = smooth_lp(&y, &e, &spec).expect("smooth_lp CV");

    let scores_fx = f64s(&cv["scores"]);
    assert_eq!(res.cv_grid, grid, "grid echoed");
    assert_eq!(res.cv_scores.len(), scores_fx.len());
    for (i, (&a, &b)) in res.cv_scores.iter().zip(&scores_fx).enumerate() {
        assert_close(a, b, 0.0, 1e-8, &format!("cv score[{i}]"));
    }
    let chosen = cv["lambda_chosen"].as_f64().expect("lambda_chosen");
    assert_eq!(res.lambda, chosen, "CV picks the fixture's lambda");
}
