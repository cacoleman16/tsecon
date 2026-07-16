//! Shared helpers for the integration tests: fixture loading, matrix
//! conversion, closeness assertions, and a tiny seeded LCG (tests must
//! not depend on tsecon-rng).
#![allow(dead_code)]

use serde_json::Value;
use tsecon_linalg::faer::Mat;

/// Loads a JSON fixture from the workspace-level `fixtures/` directory.
pub fn load_fixture(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse fixture {path}: {e}"))
}

/// A JSON array-of-arrays (rows) as a faer matrix.
pub fn as_mat(v: &Value) -> Mat<f64> {
    let rows = v.as_array().expect("expected JSON array of rows");
    let ncols = rows[0].as_array().expect("expected JSON row").len();
    Mat::from_fn(rows.len(), ncols, |i, j| {
        rows[i].as_array().expect("expected JSON row")[j]
            .as_f64()
            .expect("expected number")
    })
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

/// Elementwise [`assert_rel_close`] of a faer matrix against a JSON
/// array-of-arrays.
pub fn assert_mat_close(actual: &Mat<f64>, expected: &Value, tol: f64, what: &str) {
    let e = as_mat(expected);
    assert_eq!(actual.nrows(), e.nrows(), "{what}: row count");
    assert_eq!(actual.ncols(), e.ncols(), "{what}: column count");
    for i in 0..e.nrows() {
        for j in 0..e.ncols() {
            assert_rel_close(
                actual[(i, j)],
                e[(i, j)],
                tol,
                &format!("{what}[({i},{j})]"),
            );
        }
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

    /// Standard normal via Box-Muller (uniforms bounded away from 0).
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}
