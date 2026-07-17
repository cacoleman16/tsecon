//! Shared helpers for the integration tests: fixture loading (exact-double
//! via serde_json's `float_roundtrip`), matrix/vector conversion, and
//! closeness assertions.
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

/// A JSON array-of-arrays (rows) as a faer matrix, mapping JSON `null` to
/// `NaN` (the ragged-edge missing marker the panel routines expect).
pub fn as_mat_nan(v: &Value) -> Mat<f64> {
    let rows = v.as_array().expect("expected JSON array of rows");
    let ncols = rows[0].as_array().expect("expected JSON row").len();
    Mat::from_fn(rows.len(), ncols, |i, j| {
        let cell = &rows[i].as_array().expect("expected JSON row")[j];
        if cell.is_null() {
            f64::NAN
        } else {
            cell.as_f64().expect("expected number or null")
        }
    })
}

/// A JSON array of numbers as a `Vec<f64>`.
pub fn as_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| x.as_f64().expect("expected number"))
        .collect()
}

/// Asserts `|a - e| <= tol * max(1, |e|)`.
pub fn assert_rel_close(actual: f64, expected: f64, tol: f64, what: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol * scale,
        "{what}: {actual} vs {expected} (rel diff {:e}, tol {tol:e})",
        (actual - expected).abs() / scale
    );
}

/// Pearson correlation of two equal-length series.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa.sqrt() * sbb.sqrt())
}
