//! Shared helpers for the integration tests: fixture loading, matrix
//! conversion, and closeness assertions.
#![allow(dead_code)]

use serde_json::Value;
use tsecon_linalg::faer::Mat;

/// Loads a JSON fixture from the workspace-level `fixtures/` directory.
///
/// The Python generator writes missing values as bare `NaN` tokens
/// (`json.dumps` default), which strict JSON parsers reject; they are
/// rewritten to `null` before parsing and mapped back to `f64::NAN` by
/// [`as_vec`]. No fixture string contains the letters "NaN".
pub fn load_fixture(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {path}: {e}"));
    let text = text.replace("NaN", "null");
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

/// A JSON array of numbers as a Vec; `null` becomes NaN (missing
/// observation).
pub fn as_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| {
            if x.is_null() {
                f64::NAN
            } else {
                x.as_f64().expect("expected number")
            }
        })
        .collect()
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

/// Mean of a slice.
pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample variance (ddof = 1).
pub fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() as f64 - 1.0)
}
