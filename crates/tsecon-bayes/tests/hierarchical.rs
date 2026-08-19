//! Golden and property tests for the hierarchical (GLP 2015 empirical-Bayes /
//! ML-II) Minnesota tightness selection, against
//! `fixtures/bvar_hierarchical.json` — an independent NumPy/SciPy
//! re-implementation of the same closed-form marginal likelihood, maximized
//! by `scipy.optimize.minimize_scalar` (see
//! `fixtures/generate_bvar_hierarchical_fixtures.py`).

mod common;

use common::{as_vec, assert_mat_close, assert_rel_close, load_fixture};
use tsecon_bayes::hierarchical::{bvar_hierarchical, HierarchicalConfig, Hyperprior};

/// The main-dataset config: p = 2, lambda0 = 100, lambda3 = 1, delta = 0,
/// the default box [1e-4, 10] and 25-point grid.
fn main_config() -> HierarchicalConfig {
    HierarchicalConfig::default()
}

fn main_data() -> (serde_json::Value, tsecon_linalg::faer::Mat<f64>) {
    let fx = load_fixture("bvar_hierarchical.json");
    let data = common::as_mat(&fx["data"]);
    (fx, data)
}

/// The fixed-lambda battery used for the optimality certificate.
const FIXED_BATTERY: [f64; 7] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];

fn log_ml_at(data: &tsecon_linalg::faer::Mat<f64>, p: usize, lambda1: f64) -> f64 {
    let prior =
        tsecon_bayes::MinnesotaNiwPrior::new(data.as_ref(), p, 100.0, lambda1, 1.0, 0.0).unwrap();
    prior
        .posterior(data.as_ref())
        .unwrap()
        .log_marginal_likelihood()
}

// -------------------------------------------------------------------------
// Golden checks against the NumPy/SciPy reference.
// -------------------------------------------------------------------------

/// The returned grid `lambda1` values reproduce the reference log-spaced
/// grid (near bit-for-bit — same natural-log interpolation).
#[test]
fn grid_lambda1_matches_reference() {
    let (fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    let ref_grid = as_vec(&fx["grid_lambda1_25"]);
    assert_eq!(fit.grid_lambda1.len(), ref_grid.len());
    for (i, (&a, &e)) in fit.grid_lambda1.iter().zip(&ref_grid).enumerate() {
        assert_rel_close(a, e, 1e-12, &format!("grid_lambda1[{i}]"));
    }
}

/// The returned marginal-likelihood profile matches the reference at every
/// grid point — a near-closed-form check that each returned ML is the
/// shipped normalizer, independent of the optimizer. This also pins the
/// hyperparameter-dependent normalizers (`-(n/2) ln|Omega0|`,
/// `+(n/2) ln|Obar|`): dropping them would move every profile value.
#[test]
fn grid_log_ml_matches_reference() {
    let (fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    let ref_ml = as_vec(&fx["grid_log_ml_25"]);
    assert_eq!(fit.grid_log_ml.len(), ref_ml.len());
    for (i, (&a, &e)) in fit.grid_log_ml.iter().zip(&ref_ml).enumerate() {
        assert_rel_close(a, e, 1e-9, &format!("grid_log_ml[{i}]"));
    }
}

/// The strongest anchor: the marginal likelihood *at the optimum* matches
/// the reference maximum. Both evaluate the identical closed form and both
/// argmaxes land in the same flat peak, so the ML value is insensitive to
/// residual argmax error.
#[test]
fn optimum_log_ml_matches_reference() {
    let (fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    assert_rel_close(
        fit.log_ml,
        fx["log_ml_star"].as_f64().unwrap(),
        1e-8,
        "log_ml at optimum",
    );
}

/// The recovered `lambda1` matches the reference argmax (loose on purpose —
/// the GLP marginal likelihood is flat near its peak, so the defensible
/// quantity is the ML value, graded above; the argmax is secondary).
#[test]
fn optimum_lambda1_matches_reference() {
    let (fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    let star = fx["lambda1_star"].as_f64().unwrap();
    let abs_ok = (fit.lambda1 - star).abs() <= 2e-2;
    let rel_ok = (fit.lambda1 - star).abs() <= 0.05 * star.abs();
    assert!(
        abs_ok || rel_ok,
        "lambda1_opt {} vs reference {star} (abs diff {:e})",
        fit.lambda1,
        (fit.lambda1 - star).abs()
    );
}

/// The conjugate posterior refitted at the optimum reproduces the reference
/// coefficient and covariance means (the drop-in richer `bvar_fit`).
#[test]
fn optimum_posterior_means_match_reference() {
    let (fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    assert_mat_close(
        &fit.posterior.b_bar().to_owned(),
        &fx["b_bar_star"],
        1e-4,
        "b_bar at optimum",
    );
    assert_mat_close(
        &fit.posterior.sigma_posterior_mean().unwrap(),
        &fx["sigma_mean_star"],
        1e-4,
        "sigma_posterior_mean at optimum",
    );
}

/// The fixed-lambda reference ML equals the closed form at `lambda1_init` —
/// no optimizer involved.
#[test]
fn fixed_lambda_log_ml_matches_reference() {
    let (fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    assert_rel_close(
        fit.lambda1_fixed_log_ml,
        fx["lambda1_init_log_ml"].as_f64().unwrap(),
        1e-9,
        "lambda1_fixed_log_ml",
    );
}

// -------------------------------------------------------------------------
// Optimality certificate (no fixture): ML-II must dominate any fixed lambda.
// -------------------------------------------------------------------------

/// With `hyperprior = None` the selected ML dominates every fixed-lambda ML
/// in the battery and the `lambda1_init` reference, up to optimizer
/// tolerance — the optimality certificate implied by ML-II selection.
#[test]
fn optimum_dominates_fixed_battery() {
    let (_fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    assert!(
        fit.log_ml >= fit.lambda1_fixed_log_ml - 1e-6,
        "optimum {} must dominate fixed lambda1_init {}",
        fit.log_ml,
        fit.lambda1_fixed_log_ml
    );
    for &l in &FIXED_BATTERY {
        let ml = log_ml_at(&data, 2, l);
        assert!(
            fit.log_ml >= ml - 1e-6,
            "optimum {} must dominate ml({l}) = {ml}",
            fit.log_ml
        );
    }
}

/// The optimum is a finite interior point of the search box, and the polish
/// converged.
#[test]
fn optimum_is_interior_and_converged() {
    let (_fx, data) = main_data();
    let cfg = main_config();
    let fit = bvar_hierarchical(data.as_ref(), 2, &cfg).unwrap();
    assert!(fit.lambda1.is_finite());
    assert!(
        cfg.lambda1_lo < fit.lambda1 && fit.lambda1 < cfg.lambda1_hi,
        "lambda1_opt {} not interior to ({}, {})",
        fit.lambda1,
        cfg.lambda1_lo,
        cfg.lambda1_hi
    );
    assert!(fit.converged, "polish did not converge");
    assert_eq!(
        fit.lambda3, cfg.lambda3,
        "lambda3 unchanged when not optimized"
    );
}

/// The marginal-likelihood profile has a strict interior maximum — it rises
/// then falls. A monotone profile is exactly what would appear if the
/// lambda-dependent prior/posterior normalizers were dropped, so this is a
/// focused regression pin on the proper marginal likelihood.
#[test]
fn profile_has_interior_maximum() {
    let (_fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    let argmax = fit
        .grid_log_ml
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert!(
        argmax > 0 && argmax < fit.grid_log_ml.len() - 1,
        "grid argmax at index {argmax} is on the boundary, not interior"
    );
}

/// Under `hyperprior = None`, `log_posterior == log_ml` exactly.
#[test]
fn log_posterior_equals_log_ml_without_hyperprior() {
    let (_fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    assert_eq!(fit.log_posterior, fit.log_ml);
}

/// Disabling the grid pre-scan still recovers the same optimum (the polish
/// seeds from `lambda1_init` instead), and returns empty grid arrays.
#[test]
fn grid_disabled_still_optimizes() {
    let (fx, data) = main_data();
    let cfg = HierarchicalConfig {
        n_grid: 0,
        ..main_config()
    };
    let fit = bvar_hierarchical(data.as_ref(), 2, &cfg).unwrap();
    assert!(fit.grid_lambda1.is_empty());
    assert!(fit.grid_log_ml.is_empty());
    assert_rel_close(
        fit.log_ml,
        fx["log_ml_star"].as_f64().unwrap(),
        1e-8,
        "log_ml at optimum (no grid)",
    );
}

/// `n_evals` accounts for the grid pre-scan plus a nonzero polish budget.
#[test]
fn n_evals_counts_grid_plus_polish() {
    let (_fx, data) = main_data();
    let cfg = main_config();
    let fit = bvar_hierarchical(data.as_ref(), 2, &cfg).unwrap();
    assert!(
        fit.n_evals > cfg.n_grid,
        "n_evals {} should exceed the {} grid evals",
        fit.n_evals,
        cfg.n_grid
    );
}

// -------------------------------------------------------------------------
// Simulated-data recovery: a stationary VAR(1), spectral radius 0.6.
// -------------------------------------------------------------------------

/// On simulated stationary VAR(1) data the optimum is a finite interior
/// point whose ML dominates the fixed battery, and it lands near the
/// reference argmax.
#[test]
fn simulated_var1_recovers_interior_dominant_optimum() {
    let fx = load_fixture("bvar_hierarchical.json");
    let sim = common::as_mat(&fx["sim_var1"]["data"]);
    let cfg = main_config();
    let fit = bvar_hierarchical(sim.as_ref(), 1, &cfg).unwrap();

    assert!(fit.lambda1.is_finite());
    assert!(
        cfg.lambda1_lo < fit.lambda1 && fit.lambda1 < cfg.lambda1_hi,
        "sim lambda1_opt {} not interior",
        fit.lambda1
    );
    // Dominates the fixed battery (ML-II optimality on independent data).
    for &l in &FIXED_BATTERY {
        let prior =
            tsecon_bayes::MinnesotaNiwPrior::new(sim.as_ref(), 1, 100.0, l, 1.0, 0.0).unwrap();
        let ml = prior
            .posterior(sim.as_ref())
            .unwrap()
            .log_marginal_likelihood();
        assert!(
            fit.log_ml >= ml - 1e-6,
            "sim optimum {} must dominate ml({l}) = {ml}",
            fit.log_ml
        );
    }
    // Near the reference argmax (loose — flat peak).
    let star = fx["sim_var1"]["lambda1_star"].as_f64().unwrap();
    assert!(
        (fit.lambda1 - star).abs() <= 2e-2 || (fit.lambda1 - star).abs() <= 0.05 * star,
        "sim lambda1_opt {} vs reference {star}",
        fit.lambda1
    );
}

// -------------------------------------------------------------------------
// scale_ar: the GLP residual-scale convention (AR(1) scale regressions)
// against the independent NumPy oracle, and the default path's bit-identity.
// -------------------------------------------------------------------------

/// The main-dataset config under the GLP scale convention (only the
/// sigma_j^2 rule changes: AR(1) instead of AR(4) residual variances).
fn ar1_config() -> HierarchicalConfig {
    HierarchicalConfig {
        scale_ar: 1,
        ..HierarchicalConfig::default()
    }
}

/// Under `scale_ar = 1` the prior's S0 diagonal is the AR(1) OLS residual
/// variances the independent NumPy generator pinned.
#[test]
fn scale_ar1_prior_scales_match_reference() {
    let (fx, data) = main_data();
    let prior =
        tsecon_bayes::MinnesotaNiwPrior::with_scale_ar(data.as_ref(), 2, 100.0, 0.2, 1.0, 0.0, 1)
            .unwrap();
    assert_eq!(prior.scale_ar(), 1);
    let sig2 = as_vec(&fx["scale_ar1"]["ar_resid_var_lag1"]);
    assert_eq!(prior.s0_diag().len(), sig2.len());
    for (j, (&a, &e)) in prior.s0_diag().iter().zip(&sig2).enumerate() {
        assert_rel_close(a, e, 1e-10, &format!("AR(1) s0_diag[{j}]"));
    }
}

/// The scale_ar = 1 marginal-likelihood profile matches the independent
/// NumPy oracle at every grid point (same grid, same closed form, AR(1)
/// scales), at the same tolerance the AR(4) profile is pinned at.
#[test]
fn scale_ar1_grid_log_ml_matches_reference() {
    let (fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &ar1_config()).unwrap();
    let ref_ml = as_vec(&fx["scale_ar1"]["grid_log_ml_25"]);
    assert_eq!(fit.grid_log_ml.len(), ref_ml.len());
    for (i, (&a, &e)) in fit.grid_log_ml.iter().zip(&ref_ml).enumerate() {
        assert_rel_close(a, e, 1e-9, &format!("AR(1) grid_log_ml[{i}]"));
    }
}

/// The scale_ar = 1 selected tightness and its marginal likelihood match
/// the oracle's `scipy.optimize` maximum (same tolerances as the AR(4)
/// golden: 1e-8 on the ML value, loose on the flat-peak argmax), and the
/// fixed-lambda reference reproduces the closed form exactly.
#[test]
fn scale_ar1_optimum_matches_reference() {
    let (fx, data) = main_data();
    let fit = bvar_hierarchical(data.as_ref(), 2, &ar1_config()).unwrap();
    let blk = &fx["scale_ar1"];
    assert_rel_close(
        fit.log_ml,
        blk["log_ml_star"].as_f64().unwrap(),
        1e-8,
        "AR(1) log_ml at optimum",
    );
    let star = blk["lambda1_star"].as_f64().unwrap();
    assert!(
        (fit.lambda1 - star).abs() <= 2e-2 || (fit.lambda1 - star).abs() <= 0.05 * star.abs(),
        "AR(1) lambda1_opt {} vs reference {star}",
        fit.lambda1
    );
    assert_rel_close(
        fit.lambda1_fixed_log_ml,
        blk["lambda1_init_log_ml"].as_f64().unwrap(),
        1e-9,
        "AR(1) lambda1_fixed_log_ml",
    );
    assert!(fit.converged, "AR(1) polish did not converge");
}

/// The default path must not move: `scale_ar = 4` passed explicitly is
/// bit-identical to the default at both the prior level (every S0/Omega0
/// entry, the posterior log-ML) and the hierarchical level (selected
/// lambda1, profile, refit) — so the existing AR(4) fixture pins carry
/// over to the explicit spelling unchanged.
#[test]
fn scale_ar4_is_bit_identical_to_default() {
    let (_fx, data) = main_data();

    let by_default =
        tsecon_bayes::MinnesotaNiwPrior::new(data.as_ref(), 2, 100.0, 0.2, 1.0, 0.0).unwrap();
    let explicit =
        tsecon_bayes::MinnesotaNiwPrior::with_scale_ar(data.as_ref(), 2, 100.0, 0.2, 1.0, 0.0, 4)
            .unwrap();
    assert_eq!(by_default.scale_ar(), 4);
    assert_eq!(explicit.scale_ar(), 4);
    assert_eq!(by_default.s0_diag(), explicit.s0_diag());
    assert_eq!(by_default.omega0_diag(), explicit.omega0_diag());
    assert_eq!(
        by_default
            .posterior(data.as_ref())
            .unwrap()
            .log_marginal_likelihood(),
        explicit
            .posterior(data.as_ref())
            .unwrap()
            .log_marginal_likelihood()
    );

    let fit_default = bvar_hierarchical(data.as_ref(), 2, &main_config()).unwrap();
    let cfg_explicit = HierarchicalConfig {
        scale_ar: 4,
        ..main_config()
    };
    let fit_explicit = bvar_hierarchical(data.as_ref(), 2, &cfg_explicit).unwrap();
    assert_eq!(fit_default.lambda1, fit_explicit.lambda1);
    assert_eq!(fit_default.log_ml, fit_explicit.log_ml);
    assert_eq!(fit_default.grid_log_ml, fit_explicit.grid_log_ml);
    assert_eq!(
        fit_default.lambda1_fixed_log_ml,
        fit_explicit.lambda1_fixed_log_ml
    );
}

/// The two conventions genuinely differ on this data (the option is not a
/// no-op): AR(1) and AR(4) scales produce different S0 diagonals and a
/// different selected evidence.
#[test]
fn scale_ar1_differs_from_scale_ar4() {
    let (_fx, data) = main_data();
    let p4 = tsecon_bayes::MinnesotaNiwPrior::new(data.as_ref(), 2, 100.0, 0.2, 1.0, 0.0).unwrap();
    let p1 =
        tsecon_bayes::MinnesotaNiwPrior::with_scale_ar(data.as_ref(), 2, 100.0, 0.2, 1.0, 0.0, 1)
            .unwrap();
    for (j, (&a4, &a1)) in p4.s0_diag().iter().zip(p1.s0_diag()).enumerate() {
        assert!(
            (a4 - a1).abs() > 1e-12,
            "s0_diag[{j}] identical across conventions: {a4}"
        );
    }
}

/// `scale_ar = 0` is rejected with a teaching error, at both entry points.
#[test]
fn scale_ar0_errors() {
    let (_fx, data) = main_data();
    assert!(tsecon_bayes::MinnesotaNiwPrior::with_scale_ar(
        data.as_ref(),
        2,
        100.0,
        0.2,
        1.0,
        0.0,
        0
    )
    .is_err());
    let cfg = HierarchicalConfig {
        scale_ar: 0,
        ..main_config()
    };
    assert!(bvar_hierarchical(data.as_ref(), 2, &cfg).is_err());
}

// -------------------------------------------------------------------------
// GLP Gamma hyperprior: MAP-II pulls the tightness toward the mode.
// -------------------------------------------------------------------------

/// With the GLP Gamma hyperprior (mode 0.2), MAP-II selection pulls the
/// tightness toward 0.2 relative to pure ML-II, on data whose ML peak
/// (~0.385 for the simulated VAR(1)) is clearly away from the mode. The
/// reported log posterior differs from the log marginal likelihood by
/// exactly the log hyperprior density.
#[test]
fn glp_hyperprior_pulls_toward_mode() {
    let fx = load_fixture("bvar_hierarchical.json");
    let sim = common::as_mat(&fx["sim_var1"]["data"]);

    let none = bvar_hierarchical(sim.as_ref(), 1, &main_config()).unwrap();
    let glp_cfg = HierarchicalConfig {
        hyperprior: Hyperprior::Glp,
        ..main_config()
    };
    let glp = bvar_hierarchical(sim.as_ref(), 1, &glp_cfg).unwrap();

    assert!(glp.converged, "GLP polish did not converge");
    // Peak is above the mode here, so MAP is pulled down toward 0.2.
    assert!(
        none.lambda1 > 0.2,
        "precondition: ML peak {} should exceed the mode 0.2",
        none.lambda1
    );
    assert!(
        (glp.lambda1 - 0.2).abs() + 1e-3 < (none.lambda1 - 0.2).abs(),
        "GLP MAP {} should be pulled toward 0.2 relative to ML-II {}",
        glp.lambda1,
        none.lambda1
    );
    assert!(
        glp.lambda1 < none.lambda1,
        "GLP MAP {} should sit below the ML-II peak {}",
        glp.lambda1,
        none.lambda1
    );

    // log_posterior != log_ml, and the gap is exactly the log hyperprior.
    assert!(
        (glp.log_posterior - glp.log_ml).abs() > 1e-6,
        "GLP log_posterior should differ from log_ml"
    );
    let a = (9.0 + 17.0_f64.sqrt()) / 8.0;
    let s = 0.2 / (a - 1.0);
    let log_prior = (a - 1.0) * glp.lambda1.ln()
        - glp.lambda1 / s
        - a * s.ln()
        - tsecon_stats::special::ln_gamma(a);
    assert_rel_close(
        glp.log_posterior - glp.log_ml,
        log_prior,
        1e-9,
        "log_posterior - log_ml == log hyperprior",
    );
}

// -------------------------------------------------------------------------
// Error handling (mirrors bvar_fit).
// -------------------------------------------------------------------------

#[test]
fn errors_on_zero_lags() {
    let (_fx, data) = main_data();
    assert!(bvar_hierarchical(data.as_ref(), 0, &main_config()).is_err());
}

#[test]
fn errors_on_inverted_bounds() {
    let (_fx, data) = main_data();
    let cfg = HierarchicalConfig {
        lambda1_lo: 5.0,
        lambda1_hi: 1.0,
        ..main_config()
    };
    assert!(bvar_hierarchical(data.as_ref(), 2, &cfg).is_err());
}

#[test]
fn errors_on_nonpositive_lower_bound() {
    let (_fx, data) = main_data();
    let cfg = HierarchicalConfig {
        lambda1_lo: 0.0,
        ..main_config()
    };
    assert!(bvar_hierarchical(data.as_ref(), 2, &cfg).is_err());
}

#[test]
fn errors_on_nonfinite_data() {
    let (_fx, data) = main_data();
    let mut bad = data.clone();
    bad[(0, 0)] = f64::NAN;
    assert!(bvar_hierarchical(bad.as_ref(), 2, &main_config()).is_err());
}
