//! Shared helpers for the integration tests: fixture loading, tolerance
//! assertions, and a tiny seeded generator (tests must not depend on
//! tsecon-rng).
#![allow(dead_code)]

use serde_json::Value;

/// Loads a JSON fixture from the workspace-level `fixtures/` directory.
pub fn load_fixture(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse fixture {path}: {e}"))
}

/// Extracts a `Vec<f64>` from a JSON array of numbers.
pub fn as_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| x.as_f64().expect("expected number"))
        .collect()
}

/// Asserts `|a - e| <= tol * max(1, |e|)` (relative with an absolute floor
/// of `tol` near zero).
pub fn assert_rel_close(actual: f64, expected: f64, tol: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol * scale,
        "{what}: {actual} vs {expected} (rel diff {:e}, tol {tol:e})",
        (actual - expected).abs() / scale
    );
}

/// Asserts `|a - e| <= tol` absolutely.
pub fn assert_abs_close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: {actual} vs {expected} (abs diff {:e}, tol {tol:e})",
        (actual - expected).abs()
    );
}

/// A tiny SplitMix64 generator for reproducible test randomness.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on (0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal by Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
