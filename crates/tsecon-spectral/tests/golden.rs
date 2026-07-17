//! Golden-value tests against `fixtures/spectral.json` (generated with
//! SciPy 1.18.0 / NumPy 2.5.1 on a 512-sample bivariate series; see the
//! `_meta` block in the fixture for the exact `scipy.signal` calls).
//!
//! Tolerance: 1e-8, absolute-or-relative — `|a - e| <= tol * max(1, |e|)`.
//! The conventions pinned here are exactly the ones that differ across
//! implementations: the DC/Nyquist non-doubling, the density normalisation,
//! the periodic-Hann window, and the 50%-overlap segment averaging.

use serde_json::Value;
use tsecon_spectral::{coherence, periodogram, welch, Detrend, Scaling, Window};

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/spectral.json");
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

/// Absolute-or-relative comparison: |a - e| <= tol * max(1, |e|).
fn assert_close(actual: f64, expected: f64, tol: f64, ctx: &str) {
    let err = (actual - expected).abs() / expected.abs().max(1.0);
    assert!(
        err <= tol,
        "{ctx}: actual {actual}, expected {expected}, err {err:e} > {tol:e}"
    );
}

fn assert_all_close(actual: &[f64], expected: &[f64], tol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_close(a, e, tol, &format!("{ctx}[{i}]"));
    }
}

#[test]
fn periodogram_boxcar_density_golden() {
    let fx = fixture();
    let x = f64s(&fx["x"]);
    let s = periodogram(&x, 1.0, Window::Boxcar, Scaling::Density, Detrend::None)
        .expect("periodogram succeeds");
    assert_all_close(
        &s.freqs,
        &f64s(&fx["periodogram"]["freqs"]),
        1e-8,
        "periodogram freqs",
    );
    assert_all_close(
        &s.psd,
        &f64s(&fx["periodogram"]["psd"]),
        1e-8,
        "periodogram psd",
    );
}

#[test]
fn welch_nperseg128_golden() {
    let fx = fixture();
    let x = f64s(&fx["x"]);
    // SciPy defaults: window='hann', noverlap=nperseg/2, scaling='density'.
    let s = welch(
        &x,
        1.0,
        128,
        None,
        Window::Hann,
        Scaling::Density,
        Detrend::None,
    )
    .expect("welch succeeds");
    assert_all_close(
        &s.freqs,
        &f64s(&fx["welch_nperseg128"]["freqs"]),
        1e-8,
        "welch freqs",
    );
    assert_all_close(
        &s.psd,
        &f64s(&fx["welch_nperseg128"]["psd"]),
        1e-8,
        "welch psd",
    );
}

#[test]
fn coherence_nperseg128_golden() {
    let fx = fixture();
    let x = f64s(&fx["x"]);
    let y = f64s(&fx["y"]);
    let c =
        coherence(&x, &y, 1.0, 128, None, Window::Hann, Detrend::None).expect("coherence succeeds");
    assert_all_close(
        &c.freqs,
        &f64s(&fx["coherence_nperseg128"]["freqs"]),
        1e-8,
        "coherence freqs",
    );
    assert_all_close(
        &c.coherence,
        &f64s(&fx["coherence_nperseg128"]["coherence"]),
        1e-8,
        "coherence",
    );
}
