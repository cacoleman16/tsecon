//! Property tests: maximum-likelihood parameter recovery on a longer
//! simulated GAS series (`sim_gaussian`, `sim_student_t` in the fixture),
//! and the sanity check that the log-likelihood at the MLE exceeds the
//! likelihood at a perturbed starting point.

use serde_json::Value;
use tsecon_gas::{Density, GasModel, GasParams};

fn load_fixture() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-gas.json",
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

/// On 4000 observations simulated from a Gaussian GAS(1,1), ML recovers the
/// true `(omega, a, b)` within Monte-Carlo tolerance, and the likelihood at
/// the MLE strictly exceeds the likelihood at the (mis-specified) starting
/// values used to seed a perturbation.
#[test]
fn ml_recovers_gaussian() {
    let fx = load_fixture();
    let case = &fx["sim_gaussian"];
    let y = as_f64_vec(&case["y"]);
    let tp = &case["true_params"];
    let (omega, a, b) = (
        tp["omega"].as_f64().unwrap(),
        tp["a"].as_f64().unwrap(),
        tp["b"].as_f64().unwrap(),
    );

    let model = GasModel::new(&y, Density::Gaussian).unwrap();
    let res = model.fit().unwrap();

    // The estimate is a genuine maximizer: its log-likelihood is at least as
    // large as the log-likelihood evaluated at the true parameters.
    let ll_truth = model.loglike(&GasParams::gaussian(omega, a, b)).unwrap();
    assert!(
        res.loglik >= ll_truth - 1e-6,
        "MLE loglik {} below truth {}",
        res.loglik,
        ll_truth
    );

    // Recovery within Monte-Carlo tolerance (single 4000-obs path). The
    // score loading a and the persistence b are only weakly separated, so
    // the bands are generous; the well-identified unconditional variance is
    // checked tightly below.
    assert!(
        (res.params.b - b).abs() < 0.08,
        "b: {} vs {}",
        res.params.b,
        b
    );
    assert!(
        (res.params.a - a).abs() < 0.06,
        "a: {} vs {}",
        res.params.a,
        a
    );
    // omega is the least-identified; check the implied unconditional
    // variance omega/(1-b) instead, which is pinned by the data.
    let uncond_hat = res.params.omega / (1.0 - res.params.b);
    let uncond_true = omega / (1.0 - b);
    assert!(
        (uncond_hat - uncond_true).abs() < 0.15 * uncond_true,
        "uncond var: {uncond_hat} vs {uncond_true}"
    );

    // Likelihood at the MLE beats the likelihood at a perturbed point.
    let perturbed = GasParams::gaussian(omega * 2.0, a * 0.3, (b - 0.1).max(0.0));
    let ll_perturbed = model.loglike(&perturbed).unwrap();
    assert!(
        res.loglik > ll_perturbed,
        "MLE loglik {} not above perturbed {}",
        res.loglik,
        ll_perturbed
    );

    // Basic invariants on the returned diagnostics.
    assert_eq!(res.variance.len(), y.len());
    assert_eq!(res.std_resid.len(), y.len());
    assert!(res.variance.iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(res.aic().is_finite() && res.bic().is_finite());
    assert!(res.forecast(5).unwrap().iter().all(|v| *v > 0.0));
}

/// On 4000 observations simulated from a Student-t GAS(1,1), ML recovers the
/// true `(a, b, nu)` within Monte-Carlo tolerance, and the likelihood at the
/// MLE exceeds the likelihood at a perturbed point.
#[test]
fn ml_recovers_student_t() {
    let fx = load_fixture();
    let case = &fx["sim_student_t"];
    let y = as_f64_vec(&case["y"]);
    let tp = &case["true_params"];
    let (omega, a, b, nu) = (
        tp["omega"].as_f64().unwrap(),
        tp["a"].as_f64().unwrap(),
        tp["b"].as_f64().unwrap(),
        tp["nu"].as_f64().unwrap(),
    );

    let model = GasModel::new(&y, Density::StudentT).unwrap();
    let res = model.fit().unwrap();

    // The estimate is a genuine maximizer: on this particular 4000-obs path
    // the MLE (b ~ 0.84, a ~ 0.094) attains a strictly higher likelihood
    // than the data-generating parameters — the weakly separated a/b pair
    // trades off while the unconditional variance omega/(1-b) is pinned.
    let ll_truth = model
        .loglike(&GasParams::student_t(omega, a, b, nu))
        .unwrap();
    assert!(
        res.loglik >= ll_truth - 1e-6,
        "MLE loglik {} below truth {}",
        res.loglik,
        ll_truth
    );

    assert!(
        (res.params.b - b).abs() < 0.10,
        "b: {} vs {}",
        res.params.b,
        b
    );
    assert!(
        (res.params.a - a).abs() < 0.06,
        "a: {} vs {}",
        res.params.a,
        a
    );
    let uncond_hat = res.params.omega / (1.0 - res.params.b);
    let uncond_true = omega / (1.0 - b);
    assert!(
        (uncond_hat - uncond_true).abs() < 0.15 * uncond_true,
        "uncond var: {uncond_hat} vs {uncond_true}"
    );
    // Degrees of freedom are hard to pin precisely; a generous band still
    // rules out the Gaussian limit (large nu) and near-Cauchy (nu near 2).
    assert!(
        res.params.nu > 3.5 && res.params.nu < 12.0,
        "nu: {} (true {})",
        res.params.nu,
        nu
    );

    let perturbed = GasParams::student_t(omega * 2.0, a * 0.3, (b - 0.1).max(0.0), 15.0);
    let ll_perturbed = model.loglike(&perturbed).unwrap();
    assert!(
        res.loglik > ll_perturbed,
        "MLE loglik {} not above perturbed {}",
        res.loglik,
        ll_perturbed
    );
}
