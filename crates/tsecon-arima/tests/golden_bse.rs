//! Golden-value tests for the ARIMA parameter covariance against
//! statsmodels SARIMAX (`fixtures/arima_bse.json`, produced by
//! `fixtures/generate_arima_bse_fixtures.py`).
//!
//! The reference is `SARIMAX(..., simple_differencing=True)
//! .fit(cov_type='approx').bse` — the square root of the diagonal of
//! `pinv(-H)` for the Hessian of the *total* log-likelihood. This crate
//! computes the same object by four-point central differences; the
//! reference differentiates by complex step. The gap between the two is
//! therefore our finite-difference truncation and nothing else, which is
//! why the tolerances here are 5e-6 / 5e-5 relative rather than the
//! 1e-8 the crate's closed-form goldens hold.
//!
//! "The same object" holds on these six cases and stops holding when the
//! information matrix is ill conditioned: `pinv` truncates small singular
//! values and returns a number, while this crate refuses below an
//! equilibrated `rcond` of 1e-6 (see `crate::cov`). Every case here is
//! comfortably inside that — the tightest, `nile_arma11c`, sits at
//! `rcond = 5.1e-4`, which `cov_accuracy::rank_guard_margins` pins.
//!
//! What this file cannot do is catch a defect the two implementations
//! share, or say anything outside the fixtures' range: every case has
//! `sigma2` in `[0.94, 2e4]`, and the `sigma2` standard error used to be
//! 4.6% wrong at `sigma2 = 1e-4` with no fixture anywhere near it. The
//! closed-form sweeps in `tests/cov_accuracy.rs` are the complement.
//!
//! Every case is evaluated through [`ArimaSpec::at_params`] at the
//! fixture's *recorded* parameters. Standard errors are a local
//! curvature quantity, so comparing them across two optimizers'
//! stopping points would confound the Hessian method with the
//! optimizers' disagreement — and on `nile_arma11c` that disagreement is
//! real and documented in `golden.rs`.

mod common;

use common::{as_vec, assert_rel_close, load_fixture};
use tsecon_arima::ArimaSpec;

/// Gate for the well-conditioned cases. Measured worst-case *pure*
/// relative errors (no absolute floor — these standard errors are all
/// below 1, so a floored comparison would silently be an absolute one):
///
/// ```text
/// rw_drift_010c_T60    3.6e-8
/// arma11c_T300         2.5e-7
/// arma11_noconst_T300  3.5e-7
/// ar2c_T300            4.8e-7
/// arima111c_T300       2.5e-7
/// ```
///
/// Gated at 5e-6 — one order of magnitude of headroom for platform libm
/// differences, and nowhere near the 1e-8 the crate's closed-form
/// goldens hold, because a real finite-difference Hessian cannot deliver
/// that against a complex-step reference.
const TOL: f64 = 5e-6;

/// `nile_arma11c` alone: sigma2 ~ 2e4 against phi ~ 1 (four orders of
/// magnitude of parameter scale), an evaluation point that is *not* a
/// stationary point of the likelihood, and the worst conditioning in the
/// set (`rcond = 5.1e-4`). Measured worst case 3.5e-6; gated at 5e-5.
const TOL_NILE: f64 = 5e-5;

/// Pure relative closeness, `|a - e| <= tol * |e|`.
///
/// Deliberately *not* `common::assert_rel_close`, whose `max(|e|, 1)`
/// scale would turn every comparison in this file into an absolute one:
/// standard errors here run from 0.055 to 3034, and a shared absolute
/// tolerance would be meaninglessly loose at one end and impossible at
/// the other.
fn assert_pure_rel(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        expected != 0.0 && (actual - expected).abs() <= tol * expected.abs(),
        "{what}: {actual} vs {expected} (rel diff {:e}, tol {tol:e})",
        (actual - expected).abs() / expected.abs()
    );
}

fn nile() -> Vec<f64> {
    as_vec(&load_fixture("diagnostics.json")["nile"])
}

/// Runs one fixture case: build the spec, evaluate at the recorded
/// parameters, and compare `bse()` entry by entry.
fn check_case(name: &str, y: &[f64], tol: f64) {
    let fx = load_fixture("arima_bse.json");
    let block = &fx["cases"][name];
    let order = as_vec(&block["order"]);
    let (p, d, q) = (order[0] as usize, order[1] as usize, order[2] as usize);
    let constant = block["trend"].as_str().expect("trend") == "c";
    let params = as_vec(&block["params"]);
    let bse_exp = as_vec(&block["bse_approx"]);

    let spec = ArimaSpec::new(p, d, q).unwrap().with_constant(constant);
    let res = spec.at_params(y, &params).unwrap();

    // The reference agrees on the sample size behind the likelihood, so
    // a mismatch here would mean the two are differentiating different
    // functions and every bse comparison below would be meaningless.
    assert_eq!(
        res.nobs,
        block["nobs"].as_u64().expect("nobs") as usize,
        "{name}: nobs"
    );
    assert_rel_close(
        res.loglik,
        block["loglike"].as_f64().expect("loglike"),
        1e-8,
        &format!("{name}: loglik at fixture params"),
    );

    let bse = res.bse().unwrap();
    assert_eq!(bse.len(), bse_exp.len(), "{name}: bse length");
    for (i, (&got, &want)) in bse.iter().zip(&bse_exp).enumerate() {
        assert_pure_rel(got, want, tol, &format!("{name}: bse[{i}]"));
    }

    // `bse` is exactly the diagonal of `param_cov`, and the covariance
    // is symmetric.
    let pc = res.param_cov().unwrap();
    assert_eq!(pc.k(), bse.len(), "{name}: cov dimension");
    for (i, &se_i) in bse.iter().enumerate() {
        assert_eq!(
            pc.get(i, i).unwrap().sqrt().to_bits(),
            se_i.to_bits(),
            "{name}: se[{i}] is not sqrt(cov[{i}][{i}])"
        );
        for j in 0..i {
            assert_eq!(pc.get(i, j), pc.get(j, i), "{name}: cov asymmetric");
        }
    }
}

/// Loads a case's stored series (all cases except `nile_arma11c`, which
/// reuses the Nile from `diagnostics.json` rather than duplicating it).
fn stored_series(name: &str) -> Vec<f64> {
    as_vec(&load_fixture("arima_bse.json")["cases"][name]["y"])
}

/// ARIMA(0,1,0) + constant on a random walk with drift, T = 60 — the
/// interval-coverage audit's own case. Both standard errors have exact
/// closed forms here (`sqrt(sigma2/n)` and `sqrt(2 sigma2^2 / n)`), and
/// the fixture generator asserts statsmodels reproduces them, so this
/// case pins the whole chain against arithmetic rather than against
/// another implementation.
#[test]
fn golden_bse_rw_drift_010c() {
    let name = "rw_drift_010c_T60";
    check_case(name, &stored_series(name), TOL);

    // Independently of statsmodels: the closed form the fixture recorded.
    let fx = load_fixture("arima_bse.json");
    let block = &fx["cases"][name];
    let closed = as_vec(&block["closed_form_bse"]);
    let spec = ArimaSpec::new(0, 1, 0).unwrap().with_constant(true);
    let res = spec
        .at_params(&stored_series(name), &as_vec(&block["params"]))
        .unwrap();
    let bse = res.bse().unwrap();
    assert_pure_rel(bse[0], closed[0], 1e-6, "se(const) = sqrt(sigma2/n)");
    assert_pure_rel(bse[1], closed[1], 1e-6, "se(sigma2) = sqrt(2 sigma2^2/n)");
}

/// ARMA(1,1) + constant, T = 300: the workhorse case, all four
/// parameters on comparable scales.
#[test]
fn golden_bse_arma11c() {
    let name = "arma11c_T300";
    check_case(name, &stored_series(name), TOL);
}

/// The same series demeaned and fit without a constant: `k = 3` and no
/// leading `const` slot, so this catches an off-by-one in the parameter
/// packing that the constant cases would hide.
#[test]
fn golden_bse_arma11_no_constant() {
    let name = "arma11_noconst_T300";
    check_case(name, &stored_series(name), TOL);
}

/// AR(2) + constant: two AR coefficients, exercising a genuinely
/// non-diagonal AR block of the information matrix.
#[test]
fn golden_bse_ar2c() {
    let name = "ar2c_T300";
    check_case(name, &stored_series(name), TOL);
}

/// ARIMA(1,1,1) + constant: the `d > 0` path, where the Hessian must be
/// taken on the differenced sample (one observation lost) exactly as
/// statsmodels `simple_differencing=True` does.
#[test]
fn golden_bse_arima111c() {
    let name = "arima111c_T300";
    check_case(name, &stored_series(name), TOL);
}

/// Nile discharge, ARMA(1,1) + constant: real data with `sigma2 ~ 2e4`
/// beside `phi ~ 1`, evaluated at statsmodels' non-stationary stopping
/// point. The loosest agreement of the set, and the honest bound on what
/// a relative-step numerical Hessian delivers across four orders of
/// parameter magnitude.
#[test]
fn golden_bse_nile_arma11c() {
    check_case("nile_arma11c", &nile(), TOL_NILE);
}
