//! Behavioral tests for the SSVS-BVAR (George, Sun & Ni 2008) public API
//! against `fixtures/ssvs.json`.
//!
//! The headline is a Monte Carlo recovery test on a stable sparse VAR(2)
//! DGP: the fixture stores the lag matrices, `Sigma`'s Cholesky, and the
//! true-nonzero / true-zero coefficient masks; the data are simulated here
//! from a `tsecon_rng::Stream` (the crate's property-test convention), and
//! `bvar_ssvs` must drive the posterior inclusion probabilities near 1 on
//! the true-nonzeros and near 0 on the true-zeros. The remaining tests pin
//! reproducibility, output shapes, the covariance-selection path, the
//! multi-chain diagnostics, and the input guardrails. The closed-form
//! conditional-moment anchors and the block-1 draw-kernel MC live in the
//! crate's unit tests (`src/ssvs.rs`), where the deterministic pieces are
//! reachable.
//!
//! Monte Carlo thresholds are deterministic at the fixed seeds.

mod common;

use common::load_fixture;
use tsecon_bayes::ssvs::{bvar_ssvs, SsvsConfig};
use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;

/// Inverse-CDF standard normal for simulating the DGP (AS241 via
/// tsecon-stats; exact-zero uniforms rejected) — matches the crate's other
/// property tests.
fn std_normal(stream: &mut Stream) -> f64 {
    let mut u = stream.uniform_f64();
    while u == 0.0 {
        u = stream.uniform_f64();
    }
    tsecon_stats::special::inv_norm_cdf(u).unwrap()
}

/// Simulates the fixture's sparse VAR(`p`): `y_t = sum_l A_l y_{t-l} + L e_t`
/// with `e_t` iid standard normal and `L` the stored lower Cholesky of
/// `Sigma`. Returns the last `t_keep` rows after a burn-in.
fn simulate(fx: &serde_json::Value) -> Mat<f64> {
    let mc = &fx["mc_recovery"];
    let n = mc["n"].as_u64().unwrap() as usize;
    let t_keep = mc["T"].as_u64().unwrap() as usize;
    let burn = mc["burn"].as_u64().unwrap() as usize;
    let seed = mc["sim_seed"].as_u64().unwrap();
    let a_lags: Vec<Mat<f64>> = mc["A_lags"]
        .as_array()
        .unwrap()
        .iter()
        .map(common::as_mat)
        .collect();
    let l = common::as_mat(&mc["sigma_chol"]);

    let total = t_keep + burn;
    let mut y = Mat::<f64>::zeros(total, n);
    let mut stream = Stream::new(seed);
    for t in 0..total {
        // e_t = L z, z ~ N(0, I).
        let mut z = vec![0.0; n];
        for zi in z.iter_mut() {
            *zi = std_normal(&mut stream);
        }
        for i in 0..n {
            let mut val = 0.0;
            for j in 0..=i {
                val += l[(i, j)] * z[j];
            }
            // Autoregressive part (zero for t < p, since y starts at 0).
            for (lag, a_l) in a_lags.iter().enumerate() {
                let l1 = lag + 1;
                if t >= l1 {
                    for j in 0..n {
                        val += a_l[(i, j)] * y[(t - l1, j)];
                    }
                }
            }
            y[(t, i)] = val;
        }
    }
    Mat::from_fn(t_keep, n, |i, j| y[(burn + i, j)])
}

fn recovery_config() -> SsvsConfig {
    SsvsConfig {
        lags: 2,
        n_draws: 8_000,
        burn: 2_000,
        // A wide slab and a mild parsimony prior sharpen the Occam penalty
        // on the true zeros (whose se-scaled inclusion is governed by the
        // prior, not the sample size); the tight spike is the GSN default.
        c0: 0.1,
        c1: 20.0,
        prior_inclusion: 0.35,
        horizon: 8,
        thin: 4,
        ..SsvsConfig::default()
    }
}

// -------------------------------------------------------------------------
// Headline: Monte Carlo recovery on the sparse-true VAR.
// -------------------------------------------------------------------------

/// Posterior inclusion probabilities separate the true-nonzero lag
/// coefficients (>= 0.85) from the true-zeros (<= 0.25), the coefficient
/// means recover the nonzeros in sign and magnitude, and the never-searched
/// intercept row is pinned to 1.0.
#[test]
fn mc_recovery_separates_true_support() {
    let fx = load_fixture("ssvs.json");
    let data = simulate(&fx);
    let mc = &fx["mc_recovery"];
    let n = mc["n"].as_u64().unwrap() as usize;
    let p = mc["p"].as_u64().unwrap() as usize;
    let k = 1 + n * p;
    let true_coef = common::as_mat(&mc["true_coef"]);
    let mask = &mc["true_nonzero_mask"];

    let res = bvar_ssvs(data.as_ref(), &recovery_config(), 12_345).unwrap();

    assert_eq!(res.inclusion_prob.nrows(), k);
    assert_eq!(res.inclusion_prob.ncols(), n);

    // Intercept row is never searched: inclusion probability exactly 1.
    for c in 0..n {
        assert_eq!(
            res.inclusion_prob[(0, c)],
            1.0,
            "intercept ({c}) not pinned"
        );
    }

    let mut n_nonzero = 0usize;
    let mut n_zero = 0usize;
    for r in 1..k {
        for c in 0..n {
            let is_nonzero = mask[r].as_array().unwrap()[c].as_i64().unwrap() == 1;
            let ip = res.inclusion_prob[(r, c)];
            assert!((0.0..=1.0).contains(&ip), "inclusion prob out of range");
            if is_nonzero {
                n_nonzero += 1;
                assert!(
                    ip >= 0.85,
                    "true-nonzero coef ({r},{c}) has low inclusion prob {ip}"
                );
                // Sign and magnitude recovery.
                let truth = true_coef[(r, c)];
                let est = res.coef_mean[(r, c)];
                assert!(
                    est.signum() == truth.signum(),
                    "coef ({r},{c}) sign wrong: {est} vs {truth}"
                );
                assert!(
                    (est - truth).abs() <= 0.15,
                    "coef ({r},{c}) off: {est} vs {truth}"
                );
            } else {
                n_zero += 1;
                assert!(
                    ip <= 0.25,
                    "true-zero coef ({r},{c}) has high inclusion prob {ip}"
                );
            }
        }
    }
    assert_eq!(n_nonzero, 6, "fixture should have 6 true-nonzeros");
    assert!(n_zero == (k - 1) * n - 6);

    // Model size sits near the true support (6), well below the full 18.
    assert!(
        res.mean_model_size > 4.0 && res.mean_model_size < 10.0,
        "mean model size {} implausible",
        res.mean_model_size
    );
    // Sigma mean is finite and roughly the identity-plus-correlation DGP.
    for c in 0..n {
        assert!(res.sigma_mean[(c, c)] > 0.5 && res.sigma_mean[(c, c)] < 1.7);
    }
    assert!(res.log_marginal_likelihood_median.is_finite());
}

// -------------------------------------------------------------------------
// Output shapes and IRF-draw container.
// -------------------------------------------------------------------------

/// The IRF-draw container is `[Kt][horizon+1]` with `n x n` entries,
/// `Kt = ceil((n_draws - burn)/thin)`, and `Theta_0 Theta_0' = Sigma` for a
/// drawn covariance factor.
#[test]
fn irf_draw_container_shape_and_impact() {
    let fx = load_fixture("ssvs.json");
    let data = simulate(&fx);
    let n = fx["mc_recovery"]["n"].as_u64().unwrap() as usize;
    let cfg = SsvsConfig {
        lags: 2,
        n_draws: 1_200,
        burn: 200,
        horizon: 6,
        thin: 5,
        ..SsvsConfig::default()
    };
    let res = bvar_ssvs(data.as_ref(), &cfg, 7).unwrap();

    let kt = (cfg.n_draws - cfg.burn).div_ceil(cfg.thin);
    assert_eq!(res.irf_draws.len(), kt);
    assert_eq!(res.n_draws_kept, kt);
    for draw in &res.irf_draws {
        assert_eq!(draw.len(), cfg.horizon + 1);
        for h in draw {
            assert_eq!(h.nrows(), n);
            assert_eq!(h.ncols(), n);
        }
    }
    // Theta_0 Theta_0' reconstructs the impact covariance (a Cholesky of the
    // draw's Sigma), so it is symmetric with a positive diagonal.
    let theta0 = &res.irf_draws[0][0];
    let recon = theta0.as_ref() * theta0.transpose();
    for c in 0..n {
        assert!(recon[(c, c)] > 0.0);
        for r in 0..n {
            assert!((recon[(r, c)] - recon[(c, r)]).abs() < 1e-9);
        }
    }
}

// -------------------------------------------------------------------------
// Reproducibility: fixed seed => bitwise-identical output.
// -------------------------------------------------------------------------

#[test]
fn fixed_seed_is_bitwise_reproducible() {
    let fx = load_fixture("ssvs.json");
    let data = simulate(&fx);
    let cfg = SsvsConfig {
        lags: 2,
        n_draws: 800,
        burn: 200,
        horizon: 4,
        thin: 3,
        ..SsvsConfig::default()
    };
    let a = bvar_ssvs(data.as_ref(), &cfg, 999).unwrap();
    let b = bvar_ssvs(data.as_ref(), &cfg, 999).unwrap();

    let (k, n) = (a.inclusion_prob.nrows(), a.inclusion_prob.ncols());
    for r in 0..k {
        for c in 0..n {
            assert_eq!(a.inclusion_prob[(r, c)], b.inclusion_prob[(r, c)]);
            assert_eq!(a.coef_mean[(r, c)], b.coef_mean[(r, c)]);
        }
    }
    for r in 0..n {
        for c in 0..n {
            assert_eq!(a.sigma_mean[(r, c)], b.sigma_mean[(r, c)]);
        }
    }
    assert_eq!(a.irf_draws.len(), b.irf_draws.len());
    for (da, db) in a.irf_draws.iter().zip(&b.irf_draws) {
        for (ha, hb) in da.iter().zip(db) {
            for c in 0..n {
                for r in 0..n {
                    assert_eq!(ha[(r, c)], hb[(r, c)]);
                }
            }
        }
    }
    assert_eq!(
        a.log_marginal_likelihood_median,
        b.log_marginal_likelihood_median
    );
}

// -------------------------------------------------------------------------
// Covariance selection (ssvs_cov) and multi-chain diagnostics.
// -------------------------------------------------------------------------

/// With `ssvs_cov = true` the off-diagonal precision inclusion probabilities
/// are returned, strictly-upper populated in `[0, 1]`, lower/diagonal zero.
#[test]
fn ssvs_cov_returns_precision_inclusion() {
    let fx = load_fixture("ssvs.json");
    let data = simulate(&fx);
    let n = fx["mc_recovery"]["n"].as_u64().unwrap() as usize;
    let cfg = SsvsConfig {
        lags: 2,
        n_draws: 1_500,
        burn: 300,
        horizon: 4,
        ssvs_cov: true,
        ..SsvsConfig::default()
    };
    let res = bvar_ssvs(data.as_ref(), &cfg, 4).unwrap();
    let cov = res.inclusion_prob_cov.expect("cov inclusion present");
    assert_eq!(cov.nrows(), n);
    for c in 0..n {
        for r in 0..n {
            if r < c {
                assert!((0.0..=1.0).contains(&cov[(r, c)]));
            } else {
                assert_eq!(cov[(r, c)], 0.0, "lower/diag must be zero at ({r},{c})");
            }
        }
    }
    // Without ssvs_cov the field is omitted.
    let plain = bvar_ssvs(data.as_ref(), &recovery_config(), 4).unwrap();
    assert!(plain.inclusion_prob_cov.is_none());
}

/// Multiple chains yield the model-size R-hat and bulk ESS; on a
/// well-mixing sampler R-hat is near 1.
#[test]
fn multi_chain_reports_convergence() {
    let fx = load_fixture("ssvs.json");
    let data = simulate(&fx);
    let cfg = SsvsConfig {
        lags: 2,
        n_draws: 2_000,
        burn: 500,
        horizon: 4,
        thin: 2,
        n_chains: 3,
        ..SsvsConfig::default()
    };
    let res = bvar_ssvs(data.as_ref(), &cfg, 555).unwrap();
    let rhat = res.rhat.expect("rhat present with >1 chain");
    let ess = res.ess_bulk.expect("ess present with >1 chain");
    assert!(rhat.is_finite() && rhat < 1.3, "rhat {rhat} too large");
    assert!(ess.is_finite() && ess > 0.0, "ess {ess} invalid");
    // Kept draws pool over the 3 chains.
    let kt_per_chain = (cfg.n_draws - cfg.burn).div_ceil(cfg.thin);
    assert_eq!(res.n_draws_kept, cfg.n_chains * kt_per_chain);
    // Single-chain runs report no diagnostics.
    let single = bvar_ssvs(data.as_ref(), &recovery_config(), 555).unwrap();
    assert!(single.rhat.is_none() && single.ess_bulk.is_none());
}

// -------------------------------------------------------------------------
// Input guardrails.
// -------------------------------------------------------------------------

#[test]
fn errors_on_invalid_config_and_data() {
    let fx = load_fixture("ssvs.json");
    let data = simulate(&fx);
    let base = recovery_config();

    // Zero lags.
    assert!(bvar_ssvs(data.as_ref(), &SsvsConfig { lags: 0, ..base }, 0).is_err());
    // burn >= n_draws.
    assert!(bvar_ssvs(
        data.as_ref(),
        &SsvsConfig {
            burn: 8_000,
            n_draws: 8_000,
            ..base
        },
        0
    )
    .is_err());
    // Non-positive scale factor.
    assert!(bvar_ssvs(data.as_ref(), &SsvsConfig { c0: 0.0, ..base }, 0).is_err());
    // Prior probability out of range.
    assert!(bvar_ssvs(
        data.as_ref(),
        &SsvsConfig {
            prior_inclusion: 1.5,
            ..base
        },
        0
    )
    .is_err());
    // thin == 0 and n_chains == 0.
    assert!(bvar_ssvs(data.as_ref(), &SsvsConfig { thin: 0, ..base }, 0).is_err());
    assert!(bvar_ssvs(
        data.as_ref(),
        &SsvsConfig {
            n_chains: 0,
            ..base
        },
        0
    )
    .is_err());

    // Insufficient observations: T <= k + p.
    let tiny = Mat::<f64>::from_fn(8, 3, |i, j| (i + j) as f64);
    assert!(bvar_ssvs(tiny.as_ref(), &recovery_config(), 0).is_err());

    // Non-finite data.
    let mut bad = data.clone();
    bad[(0, 0)] = f64::NAN;
    assert!(bvar_ssvs(bad.as_ref(), &recovery_config(), 0).is_err());
}
