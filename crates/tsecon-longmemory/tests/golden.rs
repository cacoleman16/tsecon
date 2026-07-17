//! Documented-formula golden tests.
//!
//! `fixtures/longmemory.json` is produced by
//! `fixtures/generate_longmemory_fixtures.py`, which computes every published
//! quantity by literally writing the closed form in NumPy (no call to this
//! crate). Matching it proves the Rust reproduces the documented algebra:
//!
//! * fractional differencing / integration weights, filter, and exact
//!   round-trip inverse — to ~1e-12;
//! * the GPH log-periodogram regression `d`, its asymptotic SE
//!   `pi/sqrt(24 m)`, and the OLS nonrobust slope SE — to ~1e-8;
//! * the Robinson (1995) local-Whittle minimizer `d` and its SE
//!   `1/(2 sqrt(m))` — to ~1e-6.
//!
//! These pin the algebra only; the statistical recovery of a true `d` is what
//! `properties.rs` establishes by Monte-Carlo.

use serde_json::Value;
use tsecon_longmemory::{frac_diff, frac_diff_weights, frac_integrate, gph, local_whittle};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/longmemory.json",
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

fn g(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

fn close_vec(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close(*a, *e, tol, &format!("{what}[{i}]"));
    }
}

#[test]
fn fracdiff_weights_filter_and_inverse_match_documented_formula() {
    let fx = load();
    let cases = fx["fracdiff"]["cases"].as_array().expect("cases array");
    for case in cases {
        let d = g(&case["d"]);
        let nw = case["n_weights"].as_u64().expect("n_weights") as usize;
        let x = f64s(&case["x"]);

        let w = frac_diff_weights(d, nw).expect("weights");
        close_vec(
            &w,
            &f64s(&case["weights"]),
            1e-12,
            &format!("weights(d={d})"),
        );

        let fd = frac_diff(&x, d).expect("frac_diff");
        close_vec(
            &fd,
            &f64s(&case["frac_diff"]),
            1e-12,
            &format!("frac_diff(d={d})"),
        );

        let fi = frac_integrate(&x, d).expect("frac_integrate");
        close_vec(
            &fi,
            &f64s(&case["frac_integrate"]),
            1e-12,
            &format!("frac_integrate(d={d})"),
        );

        // The documented exact-inverse property, also pinned in the fixture.
        let rt = frac_integrate(&frac_diff(&x, d).expect("fd"), d).expect("fi");
        close_vec(
            &rt,
            &f64s(&case["roundtrip"]),
            1e-12,
            &format!("roundtrip(d={d})"),
        );
        // ...and it recovers the original series.
        close_vec(&rt, &x, 1e-10, &format!("roundtrip==x(d={d})"));
    }
}

#[test]
fn gph_matches_documented_regression() {
    let fx = load();
    let s = &fx["semiparametric"];
    let x = f64s(&s["x"]);
    let m = s["m"].as_u64().expect("m") as usize;
    let fit = gph(&x, m).expect("gph");
    let e = &s["gph"];
    close(fit.d, g(&e["d"]), 1e-8, "gph.d");
    close(fit.se, g(&e["se"]), 1e-12, "gph.se (pi/sqrt(24m))");
    close(
        fit.se_regression,
        g(&e["se_regression"]),
        1e-8,
        "gph.se_regression",
    );
    // The intercept absorbs the periodogram's overall normalization, which
    // differs between the raw NumPy |rfft|^2 and tsecon-spectral's density
    // scaling, so it is intentionally NOT golden-matched. d and both SEs are
    // invariant to that constant and ARE matched above.
    assert!(fit.intercept.is_finite());
    assert_eq!(fit.m, m);
}

#[test]
fn local_whittle_matches_documented_minimizer() {
    let fx = load();
    let s = &fx["semiparametric"];
    let x = f64s(&s["x"]);
    let m = s["m"].as_u64().expect("m") as usize;
    let fit = local_whittle(&x, m).expect("local_whittle");
    let e = &s["whittle"];
    close(fit.d, g(&e["d"]), 1e-6, "whittle.d");
    close(fit.se, g(&e["se"]), 1e-12, "whittle.se (1/(2 sqrt m))");
    assert_eq!(fit.m, m);
}
