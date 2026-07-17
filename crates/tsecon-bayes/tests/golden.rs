//! Golden-value tests against the workspace fixtures:
//!
//! * `bvar_niw.json` — self-authored closed-form NIW conjugate updating
//!   (see `fixtures/generate_bayes_fixtures.py` for the exact equations);
//! * `convergence.json` — ArviZ-pinned rank-normalized split R-hat and
//!   bulk/tail ESS on "good" (well-mixed) and "bad" (sticky, shifted)
//!   chain sets.

mod common;

use common::{as_vec, assert_mat_close, assert_rel_close, load_fixture};
use tsecon_bayes::{ess_bulk, ess_tail, rhat_rank, MinnesotaNiwPrior};

/// Builds the fixture's prior: p = 2, lambda0 = 100, lambda1 = 0.2,
/// lambda3 = 1, delta = 0 (growth-rate data).
fn fixture_prior() -> (serde_json::Value, MinnesotaNiwPrior) {
    let fx = load_fixture("bvar_niw.json");
    let data = common::as_mat(&fx["data"]);
    let spec = &fx["spec"];
    let prior = MinnesotaNiwPrior::new(
        data.as_ref(),
        spec["p"].as_u64().unwrap() as usize,
        spec["lambda0"].as_f64().unwrap(),
        spec["lambda1"].as_f64().unwrap(),
        spec["lambda3"].as_f64().unwrap(),
        0.0,
    )
    .unwrap();
    (fx, prior)
}

#[test]
fn minnesota_prior_matches_fixture() {
    let (fx, prior) = fixture_prior();

    // AR(4) residual-variance scales.
    let sig2 = as_vec(&fx["spec"]["ar_resid_var_lag4"]);
    assert_eq!(prior.s0_diag().len(), sig2.len());
    for (j, (&a, &e)) in prior.s0_diag().iter().zip(&sig2).enumerate() {
        assert_rel_close(a, e, 1e-10, &format!("s0_diag[{j}]"));
    }

    // Omega0 diagonal: intercept, then lag-1 block, then lag-2 block.
    let omega = as_vec(&fx["prior"]["omega0_diag"]);
    assert_eq!(prior.omega0_diag().len(), omega.len());
    for (j, (&a, &e)) in prior.omega0_diag().iter().zip(&omega).enumerate() {
        assert_rel_close(a, e, 1e-10, &format!("omega0_diag[{j}]"));
    }

    assert_rel_close(prior.v0(), fx["spec"]["v0"].as_f64().unwrap(), 0.0, "v0");

    // Own-lag prior mean: delta = 0 here, so B0 = 0.
    let b0 = prior.b0();
    for j in 0..b0.ncols() {
        for i in 0..b0.nrows() {
            assert_eq!(b0[(i, j)], 0.0, "B0[({i},{j})]");
        }
    }
}

#[test]
fn minnesota_prior_delta_sets_own_first_lag() {
    let fx = load_fixture("bvar_niw.json");
    let data = common::as_mat(&fx["data"]);
    let prior = MinnesotaNiwPrior::new(data.as_ref(), 2, 100.0, 0.2, 1.0, 1.0).unwrap();
    let b0 = prior.b0();
    let n = prior.n_vars();
    for j in 0..n {
        for i in 0..b0.nrows() {
            let expected = if i == 1 + j { 1.0 } else { 0.0 };
            assert_eq!(b0[(i, j)], expected, "B0[({i},{j})] with delta = 1");
        }
    }
}

#[test]
fn niw_posterior_matches_fixture() {
    let (fx, prior) = fixture_prior();
    let data = common::as_mat(&fx["data"]);
    let post = prior.posterior(data.as_ref()).unwrap();
    let px = &fx["posterior"];

    assert_mat_close(&post.b_bar().to_owned(), &px["b_bar"], 1e-9, "b_bar");
    assert_mat_close(
        &post.omega_bar().to_owned(),
        &px["omega_bar"],
        1e-9,
        "omega_bar",
    );
    assert_mat_close(&post.s_bar().to_owned(), &px["s_bar"], 1e-9, "s_bar");
    assert_rel_close(post.v_bar(), px["v_bar"].as_f64().unwrap(), 0.0, "v_bar");
    assert_mat_close(
        &post.sigma_posterior_mean().unwrap(),
        &px["sigma_posterior_mean"],
        1e-9,
        "sigma_posterior_mean",
    );
    assert_rel_close(
        post.log_marginal_likelihood(),
        px["log_marginal_likelihood"].as_f64().unwrap(),
        1e-9,
        "log_marginal_likelihood",
    );
}

#[test]
fn convergence_diagnostics_match_arviz() {
    let fx = load_fixture("convergence.json");
    for set in ["good", "bad"] {
        let chains = common::as_mat(&fx[set]["chains"]);
        let rhat = rhat_rank(chains.as_ref()).unwrap();
        let bulk = ess_bulk(chains.as_ref()).unwrap();
        let tail = ess_tail(chains.as_ref()).unwrap();
        assert_rel_close(
            rhat,
            fx[set]["rhat_rank"].as_f64().unwrap(),
            1e-9,
            &format!("{set}: rhat_rank"),
        );
        assert_rel_close(
            bulk,
            fx[set]["ess_bulk"].as_f64().unwrap(),
            1e-9,
            &format!("{set}: ess_bulk"),
        );
        assert_rel_close(
            tail,
            fx[set]["ess_tail"].as_f64().unwrap(),
            1e-9,
            &format!("{set}: ess_tail"),
        );
    }
}
