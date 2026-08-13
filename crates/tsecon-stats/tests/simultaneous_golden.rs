//! Golden-value tests for `tsecon_stats::simultaneous` against
//! `fixtures/simultaneous.json` (generated with NumPy 1.26.4 / SciPy 1.17.1;
//! see `fixtures/generate_simultaneous_fixtures.py`).
//!
//! Tolerances:
//!   * closed forms (pointwise / Bonferroni / Sidak) — 1e-12 relative against
//!     `scipy.stats.norm.ppf` of the documented per-cell level;
//!   * sup-t from draws — 1e-14 relative against NumPy's max-then-quantile on
//!     the identical stored draws matrix (worst measured: 1.8e-16);
//!   * sup-t from a covariance — 1e-10 absolute (worst measured: 4.2e-14).
//!     Both sides consume the same SplitMix64 uniform stream, but the Rust
//!     routine shifts each uniform by half a 2^-53 grid cell before inverting
//!     (so an exact-zero draw cannot become -inf); that shift is what the
//!     absolute rather than relative tolerance leaves room for.

use serde_json::Value;
use tsecon_stats::simultaneous::{
    bonferroni_critical_value, pointwise_critical_value, sidak_critical_value, sup_t_from_cov,
    sup_t_from_draws,
};

// ---------------------------------------------------------------------------
// SplitMix64 — must stay bit-identical to `splitmix64_uniforms` in
// fixtures/generate_simultaneous_fixtures.py.
// ---------------------------------------------------------------------------

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn uniforms(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = Self::new(seed);
        (0..n).map(|_| rng.uniform()).collect()
    }
}

// ---------------------------------------------------------------------------
// Fixture plumbing
// ---------------------------------------------------------------------------

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/simultaneous.json"
    );
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

fn assert_rel(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel <= rtol,
        "{ctx}: actual {actual}, expected {expected}, rel err {rel:e} > {rtol:e}"
    );
}

fn assert_abs(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let d = (actual - expected).abs();
    assert!(
        d <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {d:e} > {atol:e}"
    );
}

// ---------------------------------------------------------------------------
// 1. Closed forms
// ---------------------------------------------------------------------------

#[test]
fn closed_form_critical_values_match_scipy() {
    let fx = fixture();
    let rows = fx["closed_form"].as_array().expect("closed_form array");
    assert!(
        rows.len() >= 32,
        "fixture should cover a grid of (alpha, k)"
    );
    for row in rows {
        let alpha = row["alpha"].as_f64().unwrap();
        let k = row["k"].as_u64().unwrap() as usize;
        let ctx = format!("alpha={alpha}, k={k}");

        assert_rel(
            pointwise_critical_value(alpha).unwrap(),
            row["pointwise"].as_f64().unwrap(),
            1e-12,
            &format!("pointwise {ctx}"),
        );
        assert_rel(
            bonferroni_critical_value(alpha, k).unwrap(),
            row["bonferroni"].as_f64().unwrap(),
            1e-12,
            &format!("bonferroni {ctx}"),
        );
        assert_rel(
            sidak_critical_value(alpha, k).unwrap(),
            row["sidak"].as_f64().unwrap(),
            1e-12,
            &format!("sidak {ctx}"),
        );
    }
}

/// The two closed forms collapse to the pointwise value at K = 1 *exactly*,
/// not merely to within a tolerance. This is the cheapest correctness tell in
/// the module.
#[test]
fn closed_forms_collapse_exactly_at_k_equals_one() {
    for &alpha in &[0.5, 0.32, 0.10, 0.05, 0.01, 0.001] {
        let z = pointwise_critical_value(alpha).unwrap();
        assert_eq!(
            bonferroni_critical_value(alpha, 1).unwrap(),
            z,
            "bonferroni at k=1, alpha={alpha}"
        );
        assert_eq!(
            sidak_critical_value(alpha, 1).unwrap(),
            z,
            "sidak at k=1, alpha={alpha}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. sup-t from draws
// ---------------------------------------------------------------------------

#[test]
fn sup_t_from_draws_matches_numpy() {
    let fx = fixture();
    let b = &fx["sup_t_draws"];
    let k = b["k"].as_u64().unwrap() as usize;
    let n_draws = b["n_draws"].as_u64().unwrap() as usize;
    let theta_hat = f64s(&b["theta_hat"]);
    let se = f64s(&b["se"]);
    let draws = f64s(&b["draws_row_major"]);
    let alphas = f64s(&b["alphas"]);
    let expected = f64s(&b["critical_value"]);
    assert_eq!(draws.len(), n_draws * k);

    for (i, &alpha) in alphas.iter().enumerate() {
        let c = sup_t_from_draws(&draws, n_draws, &theta_hat, &se, alpha).unwrap();
        assert_rel(
            c,
            expected[i],
            1e-14,
            &format!("sup_t_from_draws alpha={alpha}"),
        );
        // The floor never bound on this design: the golden is the raw quantile.
        assert!(
            c > pointwise_critical_value(alpha).unwrap(),
            "alpha={alpha}: sup-t {c} should exceed the pointwise value"
        );
    }
}

/// A cell pinned by a normalization has `se == 0`. It must drop out of the
/// maximum instead of producing `0/0` or `inf`, and the answer must equal what
/// NumPy gets when that column is masked out.
#[test]
fn sup_t_from_draws_handles_a_pinned_cell() {
    let fx = fixture();
    let b = &fx["sup_t_draws"];
    let k = b["k"].as_u64().unwrap() as usize;
    let n_draws = b["n_draws"].as_u64().unwrap() as usize;
    let theta_hat = f64s(&b["theta_hat"]);
    let se_pinned = f64s(&b["se_pinned"]);
    let draws_pinned = f64s(&b["draws_pinned_row_major"]);
    let alphas = f64s(&b["alphas"]);
    let expected = f64s(&b["critical_value_pinned"]);
    assert_eq!(se_pinned[6], 0.0, "fixture pins cell 6");
    assert_eq!(draws_pinned.len(), n_draws * k);

    for (i, &alpha) in alphas.iter().enumerate() {
        let c = sup_t_from_draws(&draws_pinned, n_draws, &theta_hat, &se_pinned, alpha).unwrap();
        assert!(c.is_finite(), "alpha={alpha}: pinned cell produced {c}");
        assert_rel(c, expected[i], 1e-14, &format!("pinned alpha={alpha}"));
    }
}

// ---------------------------------------------------------------------------
// 3. sup-t from a covariance matrix
// ---------------------------------------------------------------------------

#[test]
fn splitmix64_stream_matches_the_generator() {
    let fx = fixture();
    for case in fx["sup_t_cov"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let seed = case["seed"].as_u64().unwrap();
        let k = case["k"].as_u64().unwrap() as usize;
        let n_sim = case["n_sim"].as_u64().unwrap() as usize;
        let head = f64s(&case["uniform_head"]);
        let mean = case["uniform_mean"].as_f64().unwrap();

        let u = SplitMix64::uniforms(seed, k * n_sim);
        // 1e-15 relative, not bit equality: this crate's `serde_json` is
        // configured without `float_roundtrip`, so the JSON parse can land one
        // ulp off. A genuine generator divergence is an O(1) mismatch.
        for (i, &h) in head.iter().enumerate() {
            assert_rel(
                u[i],
                h,
                1e-15,
                &format!("{name}: uniform[{i}] vs the generator"),
            );
        }
        let got_mean = u.iter().sum::<f64>() / u.len() as f64;
        assert_abs(got_mean, mean, 1e-12, &format!("{name} uniform mean"));
    }
}

#[test]
fn sup_t_from_cov_matches_numpy() {
    let fx = fixture();
    for case in fx["sup_t_cov"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let k = case["k"].as_u64().unwrap() as usize;
        let n_sim = case["n_sim"].as_u64().unwrap() as usize;
        let seed = case["seed"].as_u64().unwrap();
        let sigma = f64s(&case["sigma_row_major"]);
        let alphas = f64s(&case["alphas"]);
        let expected = f64s(&case["critical_value"]);

        let u = SplitMix64::uniforms(seed, k * n_sim);
        for (i, &alpha) in alphas.iter().enumerate() {
            let c = sup_t_from_cov(&sigma, k, alpha, &u).unwrap();
            assert_abs(c, expected[i], 1e-10, &format!("{name} alpha={alpha}"));
        }
    }
}

/// The `single_cell` case is the K = 1 collapse for the simulated route: with
/// 100k simulations the sup-t critical value should sit on the pointwise `z`
/// to within Monte Carlo error, and (because of the floor) never below it.
#[test]
fn sup_t_from_cov_collapses_to_pointwise_at_k_equals_one() {
    let fx = fixture();
    let case = fx["sup_t_cov"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "single_cell")
        .expect("single_cell case present");
    let n_sim = case["n_sim"].as_u64().unwrap() as usize;
    let seed = case["seed"].as_u64().unwrap();
    let sigma = f64s(&case["sigma_row_major"]);
    let u = SplitMix64::uniforms(seed, n_sim);

    for &alpha in &[0.32, 0.10, 0.05] {
        let z = pointwise_critical_value(alpha).unwrap();
        let c = sup_t_from_cov(&sigma, 1, alpha, &u).unwrap();
        assert!(c >= z, "alpha={alpha}: floor violated, {c} < {z}");
        assert!(
            c - z < 0.05,
            "alpha={alpha}: k=1 sup-t {c} should collapse onto pointwise {z}"
        );
    }
}
