//! Shared helpers for the integration tests: fixture loading and a tiny
//! seeded LCG (tests must not depend on tsecon-rng).
//!
//! Each integration-test binary compiles its own copy of this module and
//! uses a different subset of the helpers.
#![allow(dead_code)]

use faer::Mat;
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

/// Extracts a faer matrix from a JSON array of row arrays.
pub fn as_mat(v: &Value) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .expect("expected JSON array of rows")
        .iter()
        .map(as_f64_vec)
        .collect();
    let nrows = rows.len();
    let ncols = rows[0].len();
    Mat::from_fn(nrows, ncols, |i, j| rows[i][j])
}

/// Asserts `|a - b| <= tol` elementwise on slices.
pub fn assert_slice_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{what}[{i}]: {a} vs {e} (diff {:e}, tol {tol:e})",
            (a - e).abs()
        );
    }
}

/// Asserts `|a - b| <= tol` elementwise on matrices.
pub fn assert_mat_close(actual: &Mat<f64>, expected: &Mat<f64>, tol: f64, what: &str) {
    assert_eq!(actual.nrows(), expected.nrows(), "{what}: nrows mismatch");
    assert_eq!(actual.ncols(), expected.ncols(), "{what}: ncols mismatch");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let (a, e) = (actual[(i, j)], expected[(i, j)]);
            assert!(
                (a - e).abs() <= tol,
                "{what}[({i},{j})]: {a} vs {e} (diff {:e}, tol {tol:e})",
                (a - e).abs()
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

    /// Uniform in (-1, 1).
    pub fn symmetric(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }
}
