//! AFNS tests against `fixtures/afns.json`, plus AFNS properties.
//!
//! The fixture is a DOCUMENTED-FORMULA golden: `fixtures/generate_afns_fixtures.py`
//! transcribes the Christensen-Diebold-Rudebusch (2011) independent-factor
//! yield-adjustment closed form `A(tau)/tau` directly into NumPy and evaluates
//! it on a grid of maturities / lambda / sigma. Nothing in the generator calls
//! the Rust crate, so the golden is non-circular. The crate is expected to
//! reproduce the signed adjustment `-A(tau)/tau` to ~1e-10.

use serde_json::Value;
use tsecon_termstructure::{afns_yield_adjustment, fit_afns, fit_nelson_siegel};

fn load() -> Value {
    let path = format!("{}/../../fixtures/afns.json", env!("CARGO_MANIFEST_DIR"));
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

fn sigma3(v: &Value) -> [f64; 3] {
    let s = f64s(v);
    assert_eq!(s.len(), 3, "sigma_diag has three entries");
    [s[0], s[1], s[2]]
}

fn assert_close(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > atol {atol:e}"
    );
}

#[test]
fn afns_adjustment_matches_cdr_closed_form() {
    // Reference-matched: the signed adjustment -A(tau)/tau reproduces the
    // documented CDR (2011) closed form (computed independently in NumPy) to
    // 1e-10 on every case in the grid.
    let fx = load();
    let cases = fx["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "fixture has cases");

    for (c, case) in cases.iter().enumerate() {
        let maturities = f64s(&case["maturities"]);
        let lambda = case["lambda"].as_f64().expect("lambda");
        let sigma = sigma3(&case["sigma_diag"]);
        let expected = f64s(&case["adjustment"]); // signed -A(tau)/tau

        let got = afns_yield_adjustment(&maturities, lambda, sigma).expect("adjustment");
        assert_eq!(got.len(), expected.len(), "case {c} length");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_close(g, e, 1e-10, &format!("case {c} adjustment[{i}]"));
        }
    }
}

#[test]
fn afns_adjustment_is_nonpositive_and_grows_with_maturity() {
    // Property: A(tau)/tau >= 0, so the signed adjustment is <= 0, and (via the
    // sigma_11^2 tau^2/6 term) its magnitude grows monotonically with maturity.
    let fx = load();
    let cases = fx["cases"].as_array().expect("cases array");

    for (c, case) in cases.iter().enumerate() {
        let maturities = f64s(&case["maturities"]);
        let lambda = case["lambda"].as_f64().expect("lambda");
        let sigma = sigma3(&case["sigma_diag"]);
        let adj = afns_yield_adjustment(&maturities, lambda, sigma).expect("adjustment");

        for (i, &a) in adj.iter().enumerate() {
            assert!(a <= 0.0, "case {c} adjustment[{i}] = {a} should be <= 0");
        }
        // Magnitude non-decreasing in maturity (maturities are sorted ascending).
        for i in 1..adj.len() {
            assert!(
                adj[i].abs() >= adj[i - 1].abs() - 1e-15,
                "case {c}: |adjustment| not growing at {i}: {} < {}",
                adj[i].abs(),
                adj[i - 1].abs()
            );
        }
    }
}

#[test]
fn afns_nests_nelson_siegel_as_sigma_goes_to_zero() {
    // Property: with sigma = 0 the adjustment is identically zero and the AFNS
    // fit coincides with the plain Nelson-Siegel fit.
    let maturities = [0.25, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0];
    let yields = [4.10, 3.99, 4.02, 4.09, 4.25, 4.31, 4.43, 4.50];
    let lambda = 0.5;

    let adj = afns_yield_adjustment(&maturities, lambda, [0.0, 0.0, 0.0]).expect("adjustment");
    assert!(
        adj.iter().all(|&a| a == 0.0),
        "zero-sigma adjustment is zero"
    );

    let ns = fit_nelson_siegel(&maturities, &yields, lambda).expect("ns");
    let afns = fit_afns(&maturities, &yields, lambda, [0.0, 0.0, 0.0]).expect("afns");

    assert_eq!(afns.factors(), ns.factors, "factors coincide at sigma=0");
    let ns_fitted = ns.fitted(&maturities).expect("ns fitted");
    for (i, (&a, &n)) in afns.fitted.iter().zip(ns_fitted.iter()).enumerate() {
        assert_close(a, n, 1e-12, &format!("fitted[{i}] AFNS vs NS at sigma=0"));
    }
}

#[test]
fn afns_fit_reconstructs_curve_and_exposes_pieces() {
    // The arbitrage-free fitted curve = NS fit + adjustment reconstructs the
    // observed yields, and the pieces (factors, sigma, adjustment) are exposed.
    let maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
    let yields = [4.10, 3.99, 3.98, 4.02, 4.09, 4.25, 4.31, 4.43];
    let lambda = 0.5;
    let sigma = [0.01, 0.008, 0.012];

    let afns = fit_afns(&maturities, &yields, lambda, sigma).expect("afns");

    // Exposed pieces.
    assert_eq!(afns.sigma_diag, sigma);
    assert_eq!(afns.adjustment.len(), maturities.len());
    assert_eq!(afns.fitted.len(), maturities.len());
    assert_eq!(afns.factors(), afns.ns.factors);

    // fitted == ns.fitted + adjustment, exactly by construction.
    let ns_fitted = afns.ns.fitted(&maturities).expect("ns fitted");
    for (i, ((&f, &nf), &a)) in afns
        .fitted
        .iter()
        .zip(ns_fitted.iter())
        .zip(afns.adjustment.iter())
        .enumerate()
    {
        assert_close(f, nf + a, 1e-12, &format!("fitted decomposition[{i}]"));
    }

    // The arbitrage-free curve reconstructs the observed yields well.
    for (i, (&f, &y)) in afns.fitted.iter().zip(yields.iter()).enumerate() {
        assert!((f - y).abs() < 0.25, "maturity {i}: fitted {f} vs {y}");
    }

    // yield_at agrees with the on-grid fitted value.
    for (i, &t) in maturities.iter().enumerate() {
        let y_at = afns.yield_at(t).expect("yield_at");
        assert_close(y_at, afns.fitted[i], 1e-10, &format!("yield_at[{i}]"));
    }
}

#[test]
fn afns_rejects_negative_sigma() {
    let maturities = [1.0, 2.0, 3.0, 5.0, 7.0];
    assert!(afns_yield_adjustment(&maturities, 0.5, [-0.01, 0.0, 0.0]).is_err());
    let yields = [4.0, 4.1, 4.2, 4.3, 4.4];
    assert!(fit_afns(&maturities, &yields, 0.5, [0.0, f64::NAN, 0.0]).is_err());
}
