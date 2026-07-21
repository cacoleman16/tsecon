//! Independent-reference golden tests for the functional-shock estimators.
//!
//! `fixtures/tsecon-funcshock.json` is produced by
//! `fixtures/generate_tsecon-funcshock_fixtures.py`:
//!
//! * functional PCA is pinned to `numpy.linalg.eigh` of `cov = Xc'Xc/T`
//!   (identical sign convention: largest-|.| entry positive, first index on
//!   ties) at `1e-10`;
//! * the functional local projection to statsmodels
//!   `OLS(...).fit(cov_type="HAC", maxlags=h+p, use_correction=True)` per
//!   horizon (betas, JOINT K x K covariance, SEs) at `1e-8`;
//! * the scenario response to the numpy closed form `w'beta_h`,
//!   `sqrt(w' Cov_h w)` at `1e-8`;
//! * the FVAR scenario to statsmodels `VAR([scores, y]).fit(lags,
//!   trend="c")` + `orth_ma_rep` + scipy triangular solve at `1e-8`.
//!
//! Each reference reaches its numbers independently of this crate, so
//! reproducing them is a genuine cross-implementation check.

use serde_json::Value;
use tsecon_funcshock::{flp, flp_scenario, functional_pca, fvar_scenario, scenario_weights};

const TOL_FPCA: f64 = 1e-10;
const TOL: f64 = 1e-8;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-funcshock.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn u(v: &Value) -> usize {
    v.as_u64().expect("uint") as usize
}

fn close_at(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e} rel={rel:.3e}"
    );
}

fn close_slice(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close_at(*a, *e, tol, &format!("{what}[{i}]"));
    }
}

#[test]
fn functional_pca_matches_numpy_eigh() {
    let fx = load();
    let cases = fx["fpca"].as_array().expect("array");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let curves = rows(&case["curves"]);
        let k = u(&case["n_factors"]);

        let r = functional_pca(&curves, k).expect("fpca ok");

        close_slice(
            &r.mean_curve,
            &f64s(&case["mean_curve"]),
            TOL_FPCA,
            &format!("fpca[{name}] mean_curve"),
        );
        // Fixture stores ALL M eigenvalues descending; we keep the leading K.
        let all_evals = f64s(&case["eigenvalues"]);
        close_slice(
            &r.eigenvalues,
            &all_evals[..k],
            TOL_FPCA,
            &format!("fpca[{name}] eigenvalues"),
        );
        close_at(
            r.total_variance,
            all_evals.iter().sum::<f64>(),
            TOL_FPCA,
            &format!("fpca[{name}] total_variance == sum of all eigenvalues"),
        );
        close_slice(
            &r.explained,
            &f64s(&case["explained"]),
            TOL_FPCA,
            &format!("fpca[{name}] explained"),
        );
        let phi_ref = rows(&case["eigenfunctions"]); // K rows x M
        assert_eq!(r.eigenfunctions.len(), phi_ref.len());
        for (j, (got, want)) in r.eigenfunctions.iter().zip(phi_ref.iter()).enumerate() {
            close_slice(got, want, TOL_FPCA, &format!("fpca[{name}] phi[{j}]"));
        }
        let scores_ref = rows(&case["scores"]); // T rows x K
        assert_eq!(r.scores.len(), scores_ref.len());
        for (t, (got, want)) in r.scores.iter().zip(scores_ref.iter()).enumerate() {
            close_slice(got, want, TOL_FPCA, &format!("fpca[{name}] scores[{t}]"));
        }
    }
}

#[test]
fn flp_matches_statsmodels_hac() {
    let fx = load();
    let p = &fx["pipeline"];
    let curves = rows(&p["curves"]);
    let y = f64s(&p["y"]);
    let k = u(&p["n_factors"]);
    let block = &p["flp"];
    let horizons = u(&block["horizons"]);
    let n_lag_controls = u(&block["n_lag_controls"]);

    let fpca = functional_pca(&curves, k).expect("fpca ok");
    let fit = flp(&y, &fpca.scores, horizons, n_lag_controls, None).expect("flp ok");

    let betas_ref = rows(&block["betas"]);
    let se_ref = rows(&block["se"]);
    let nobs_ref: Vec<usize> = block["nobs"]
        .as_array()
        .expect("array")
        .iter()
        .map(u)
        .collect();
    assert_eq!(fit.betas.len(), horizons + 1);
    assert_eq!(fit.n_factors, k);
    for h in 0..=horizons {
        assert_eq!(fit.nobs[h], nobs_ref[h], "flp nobs[{h}]");
        close_slice(&fit.betas[h], &betas_ref[h], TOL, &format!("flp beta[{h}]"));
        close_slice(&fit.se[h], &se_ref[h], TOL, &format!("flp se[{h}]"));
        // The JOINT K x K covariance, off-diagonals included.
        let cov_ref = rows(&block["covs"][h]); // K x K nested
        for (i, cov_row) in cov_ref.iter().enumerate() {
            for (j, want) in cov_row.iter().enumerate() {
                close_at(
                    fit.covs[h][i * k + j],
                    *want,
                    TOL,
                    &format!("flp cov[{h}][{i},{j}]"),
                );
            }
        }
    }
}

#[test]
fn scenario_response_matches_numpy_closed_form() {
    let fx = load();
    let p = &fx["pipeline"];
    let curves = rows(&p["curves"]);
    let y = f64s(&p["y"]);
    let k = u(&p["n_factors"]);
    let block = &p["flp"];
    let scen = &p["scenario"];
    let delta = f64s(&scen["delta"]);

    let fpca = functional_pca(&curves, k).expect("fpca ok");
    let fit = flp(
        &y,
        &fpca.scores,
        u(&block["horizons"]),
        u(&block["n_lag_controls"]),
        None,
    )
    .expect("flp ok");

    let w = scenario_weights(&fpca.eigenfunctions, &delta).expect("weights ok");
    close_slice(&w, &f64s(&scen["weights"]), TOL, "scenario weights");

    let irf = flp_scenario(&fpca, &fit, &delta).expect("scenario ok");
    close_slice(
        &irf.response,
        &f64s(&scen["response"]),
        TOL,
        "scenario response",
    );
    close_slice(&irf.se, &f64s(&scen["se"]), TOL, "scenario se");
}

#[test]
fn fvar_scenario_matches_statsmodels_var() {
    let fx = load();
    let p = &fx["pipeline"];
    let curves = rows(&p["curves"]);
    let y = f64s(&p["y"]);
    let k = u(&p["n_factors"]);
    let scen = &p["scenario"];
    let block = &p["fvar"];
    let lags = u(&block["lags"]);
    let horizon = u(&block["horizon"]);

    let fpca = functional_pca(&curves, k).expect("fpca ok");
    let w = scenario_weights(&fpca.eigenfunctions, &f64s(&scen["delta"])).expect("weights ok");

    let r = fvar_scenario(&fpca.scores, &y, &w, lags, horizon).expect("fvar ok");

    close_slice(
        &r.response_outcome,
        &f64s(&block["response_outcome"]),
        TOL,
        "fvar response_outcome",
    );
    let responses_ref = rows(&block["responses"]); // (H+1) x (K+1)
    assert_eq!(r.responses.len(), responses_ref.len());
    for (h, (got, want)) in r.responses.iter().zip(responses_ref.iter()).enumerate() {
        close_slice(got, want, TOL, &format!("fvar responses[{h}]"));
    }
    close_at(
        r.implied_outcome_innovation,
        responses_ref[0][k],
        TOL,
        "fvar implied_outcome_innovation",
    );
}
