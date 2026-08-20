//! Golden-value tests for the DCS robust local level against
//! `fixtures/tsecon-dcs.json`.
//!
//! Two legs, graded honestly (see the fixture generator's docstring):
//!
//! 1. **Gaussian limit — independent package, pinned THROUGH the
//!    steady-state mapping.** statsmodels `UnobservedComponents(y,
//!    'llevel')` is fitted by its own MLE in its own parameterization
//!    (`sigma2_eps`, `sigma2_eta`); the generator maps those variances to
//!    the DCS parameters via the exact steady-state algebra
//!    (`q = sigma2_eta/sigma2_eps`, `p = (q + sqrt(q^2+4q))/2`,
//!    `kappa = p/(1+p)`, `scale^2 = sigma2_eps (1+p)`) and re-runs
//!    *statsmodels itself* at the mapped parameters with known
//!    steady-state initialization, where its Kalman gain is constant and
//!    equal to `kappa`. The DCS-Gaussian filter must reproduce that
//!    statsmodels run: log-likelihood to 1e-8, level path to 1e-6, on two
//!    seeded local-level series and the Nile.
//!
//! 2. **Fitted parameters — one criterion, two optimizers.** `dcs_mle`
//!    in the fixture maximizes the *identical* criterion (exact
//!    conditional likelihood given the ten-point-median `mu_1`) with
//!    scipy L-BFGS-B plus a high-precision Nelder-Mead polish; the Rust
//!    fit (tsecon-optim Nelder-Mead multistart) must land on the same
//!    interior optimum to 1e-4 in the parameters and 1e-8 in the
//!    log-likelihood.
//!
//! The Student-t and Laplace filters have no runnable third-party
//! reference (DCS reference code is R/Matlab), so their fixed-parameter
//! goldens are documented-formula values (the recursion applied literally
//! in NumPy, with the t observation density cross-checked against
//! scipy.stats.t in the generator) pinned at 1e-10; the *statistical*
//! claims for those densities are carried by the Monte-Carlo property
//! tests in `dcs_properties.rs`.

use serde_json::Value;
use tsecon_gas::{DcsDensity, DcsModel, DcsParams};

fn load_fixture() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-dcs.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("read fixture");
    serde_json::from_str(&text).expect("parse fixture")
}

fn as_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol * scale,
        "{what}: {actual} vs {expected} (diff {:e}, tol {tol:e})",
        (actual - expected).abs()
    );
}

/// Leg 1: the DCS-Gaussian filter at the mapped parameters reproduces the
/// statsmodels steady-state Kalman filter — log-likelihood 1e-8, level
/// path 1e-6, out-of-sample prediction 1e-8 — on all three series.
#[test]
fn gaussian_filter_matches_statsmodels_steady_state_kalman() {
    let fx = load_fixture();
    for case in fx["gaussian_ss"].as_array().expect("cases") {
        let name = case["series"].as_str().expect("name");
        let y = as_f64_vec(&case["y"]);
        let kappa = case["map"]["kappa"].as_f64().expect("kappa");
        let scale = case["map"]["scale"].as_f64().expect("scale");

        let model = DcsModel::new(&y, DcsDensity::Gaussian).expect("model");
        // The robust initialization must agree with the generator's
        // median(y[..10]) exactly — the whole path hangs off it.
        assert_close(
            model.mu_init(),
            case["mu0"].as_f64().expect("mu0"),
            1e-12,
            &format!("{name} mu0"),
        );
        let out = model
            .filter(&DcsParams::gaussian(kappa, scale))
            .expect("filter");

        assert_close(
            out.loglik,
            case["ss_filter"]["loglik"].as_f64().expect("loglik"),
            1e-8,
            &format!("{name} steady-state loglik"),
        );
        let exp_level = as_f64_vec(&case["ss_filter"]["level"]);
        assert_eq!(out.level.len(), exp_level.len());
        for (t, (&a, &e)) in out.level.iter().zip(&exp_level).enumerate() {
            assert_close(a, e, 1e-6, &format!("{name} level[{t}]"));
        }
        assert_close(
            out.next_level,
            case["ss_filter"]["next_level"].as_f64().expect("next"),
            1e-8,
            &format!("{name} next_level"),
        );
    }
}

/// Leg 2: the Rust MLE (tsecon-optim Nelder-Mead multistart) lands on the
/// scipy MLE of the identical criterion — parameters to 1e-4,
/// log-likelihood to 1e-8 — and certifies convergence on these well-posed
/// fits.
#[test]
fn gaussian_fit_matches_scipy_mle_across_optimizers() {
    let fx = load_fixture();
    for case in fx["gaussian_ss"].as_array().expect("cases") {
        let name = case["series"].as_str().expect("name");
        let y = as_f64_vec(&case["y"]);
        let mle = &case["dcs_mle"];

        let model = DcsModel::new(&y, DcsDensity::Gaussian).expect("model");
        let res = model.fit().expect("fit");

        assert_close(
            res.params.kappa,
            mle["kappa"].as_f64().expect("kappa"),
            1e-4,
            &format!("{name} fitted kappa"),
        );
        assert_close(
            res.params.scale,
            mle["scale"].as_f64().expect("scale"),
            1e-4,
            &format!("{name} fitted scale"),
        );
        assert_close(
            res.loglik,
            mle["loglik"].as_f64().expect("loglik"),
            1e-8,
            &format!("{name} fitted loglik"),
        );
        assert!(
            res.converged,
            "{name}: a well-posed Gaussian DCS fit reported converged = false \
             (iterations {}, fevals {})",
            res.iterations, res.fevals
        );
        // Observed-information SEs exist at this interior optimum.
        assert!(
            res.se.kappa.is_finite() && res.se.kappa > 0.0,
            "{name}: se(kappa) = {}",
            res.se.kappa
        );
        assert!(
            res.se.scale.is_finite() && res.se.scale > 0.0,
            "{name}: se(scale) = {}",
            res.se.scale
        );
        assert!(res.se.nu.is_nan(), "{name}: Gaussian fit reported se(nu)");
        // Information criteria are consistent with the reported loglik.
        assert_close(
            res.aic(),
            2.0 * 2.0 - 2.0 * res.loglik,
            1e-12,
            &format!("{name} aic"),
        );
        assert_close(
            res.bic(),
            2.0 * (res.n_obs as f64).ln() - 2.0 * res.loglik,
            1e-12,
            &format!("{name} bic"),
        );
    }
}

/// Documented-formula goldens: the Student-t, Laplace (exact hard sign),
/// and Gaussian filters at fixed parameters reproduce the NumPy recursion
/// to 1e-10 on a contaminated series.
#[test]
fn fixed_parameter_filters_match_documented_recursion() {
    let fx = load_fixture();
    let case = &fx["filter_golden"];
    let y = as_f64_vec(&case["y"]);

    for (key, density) in [
        ("student_t", DcsDensity::StudentT),
        ("laplace", DcsDensity::Laplace),
        ("gaussian", DcsDensity::Gaussian),
    ] {
        let block = &case[key];
        let p = &block["params"];
        let params = match density {
            DcsDensity::StudentT => DcsParams::student_t(
                p["kappa"].as_f64().expect("kappa"),
                p["scale"].as_f64().expect("scale"),
                p["nu"].as_f64().expect("nu"),
            ),
            DcsDensity::Laplace => DcsParams::laplace(
                p["kappa"].as_f64().expect("kappa"),
                p["scale"].as_f64().expect("scale"),
            ),
            DcsDensity::Gaussian => DcsParams::gaussian(
                p["kappa"].as_f64().expect("kappa"),
                p["scale"].as_f64().expect("scale"),
            ),
        };
        let model = DcsModel::new(&y, density).expect("model");
        let out = model.filter(&params).expect("filter");

        let exp_level = as_f64_vec(&block["level"]);
        assert_eq!(out.level.len(), exp_level.len());
        for (t, (&a, &e)) in out.level.iter().zip(&exp_level).enumerate() {
            assert_close(a, e, 1e-10, &format!("{key} level[{t}]"));
        }
        assert_close(
            out.loglik,
            block["loglik"].as_f64().expect("loglik"),
            1e-10,
            &format!("{key} loglik"),
        );
        assert_close(
            out.next_level,
            block["next_level"].as_f64().expect("next"),
            1e-10,
            &format!("{key} next_level"),
        );
    }
}

/// Regression: the Student-t fit on clean Gaussian data rides `nu` up the
/// Gaussian ridge — and both things that went wrong out there stay fixed.
///
/// 1. **The log-likelihood is a log-likelihood.** The t family's
///    likelihood supremum on any data is bounded by (and approaches) the
///    Gaussian fit's at `nu -> inf`, yet the literal
///    `ln Γ((nu+1)/2) - ln Γ(nu/2)` constant used to cancel so badly at
///    `nu ~ 1e15` that this fit reported `loglik = +54230` on a series
///    whose Gaussian log-likelihood is `-744` — and the optimizer climbed
///    that noise. Now the ridge is numerically clean: the t fit's loglik
///    must sit within a whisker of the Gaussian fit's, never wildly above.
/// 2. **The flag does not depend on the platform's libm.** Whether the
///    simplex happens to collapse on the flat ridge differed between
///    Linux and Windows MSVC (caught by CI on the identical fixture);
///    past `NU_GAUSSIAN_RIDGE` the certificate is forced `false`
///    deterministically.
#[test]
fn student_t_fit_on_gaussian_data_reports_ridge_not_garbage() {
    let fx = load_fixture();
    for case in fx["gaussian_ss"].as_array().expect("cases").iter().take(2) {
        let name = case["series"].as_str().expect("name");
        let y = as_f64_vec(&case["y"]);

        let g = DcsModel::new(&y, DcsDensity::Gaussian)
            .expect("model")
            .fit()
            .expect("gaussian fit");
        let t = DcsModel::new(&y, DcsDensity::StudentT)
            .expect("model")
            .fit()
            .expect("t fit");

        assert!(
            t.params.nu > tsecon_gas::kernel::NU_GAUSSIAN_RIDGE,
            "{name}: expected nu to diverge on Gaussian data, got {}",
            t.params.nu
        );
        assert!(
            !t.converged,
            "{name}: certified an optimum on the nu ridge (nu = {:e})",
            t.params.nu
        );
        // The nesting bound: |t loglik - gaussian loglik| is O(n/nu) plus
        // finite-termination slop in (kappa, scale) — small either way,
        // and categorically not tens of thousands.
        assert!(
            (t.loglik - g.loglik).abs() < 0.5,
            "{name}: t fit loglik {} vs gaussian {} — the ridge is not \
             numerically clean",
            t.loglik,
            g.loglik
        );
    }
}
