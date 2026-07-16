//! Shared helpers for the integration tests: fixture loading (with
//! NaN-for-missing handling) and a tiny seeded LCG (tests must not depend
//! on tsecon-rng).
#![allow(dead_code)]

use serde_json::Value;
use tsecon_linalg::faer::Mat;

/// Loads a JSON fixture from the workspace-level `fixtures/` directory.
///
/// The Python generator writes missing values as bare `NaN` tokens
/// (`json.dumps` default), which strict JSON parsers reject; they are
/// rewritten to `null` before parsing and mapped back to `f64::NAN` by
/// [`as_f64_vec`]. No fixture string contains the letters "NaN".
pub fn load_fixture(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {path}: {e}"));
    let text = text.replace("NaN", "null");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse fixture {path}: {e}"))
}

/// Extracts a `Vec<f64>` from a JSON array of numbers; `null` becomes NaN
/// (missing observation).
pub fn as_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| {
            if x.is_null() {
                f64::NAN
            } else {
                x.as_f64().expect("expected number or null")
            }
        })
        .collect()
}

/// A slice as an `n x 1` observation matrix.
pub fn col(y: &[f64]) -> Mat<f64> {
    Mat::from_fn(y.len(), 1, |i, _| y[i])
}

/// Asserts `|a - e| <= tol * max(1, |e|)` (relative with an absolute
/// floor of `tol` near zero).
pub fn assert_rel_close(actual: f64, expected: f64, tol: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol * scale,
        "{what}: {actual} vs {expected} (rel diff {:e}, tol {tol:e})",
        (actual - expected).abs() / scale
    );
}

/// Elementwise [`assert_rel_close`] over slices.
pub fn assert_slice_rel_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_rel_close(*a, *e, tol, &format!("{what}[{i}]"));
    }
}

/// Minimal 64-bit LCG (Knuth MMIX constants) for seeded test randomness.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in (-1, 1).
    pub fn symmetric(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }
}
