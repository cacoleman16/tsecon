//! Shared helpers for the integration tests: fixture loading, closeness
//! assertions, a tiny seeded LCG (tests must not depend on tsecon-rng),
//! and an ARMA simulation helper.
#![allow(dead_code)]

use serde_json::Value;

/// Loads a JSON fixture from the workspace-level `fixtures/` directory.
pub fn load_fixture(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse fixture {path}: {e}"))
}

/// A JSON array of numbers as a `Vec<f64>`.
pub fn as_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| x.as_f64().expect("expected number"))
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
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Simulates `n` observations of a stationary/invertible ARMA(p, q) with
/// intercept, `y_t = c + sum phi_i y_{t-i} + e_t + sum theta_j e_{t-j}`,
/// `e ~ N(0, sigma2)`, discarding a 500-observation burn-in.
pub fn simulate_arma(
    rng: &mut Lcg,
    n: usize,
    constant: f64,
    ar: &[f64],
    ma: &[f64],
    sigma2: f64,
) -> Vec<f64> {
    let burn = 500;
    let total = n + burn;
    let sd = sigma2.sqrt();
    let e: Vec<f64> = (0..total).map(|_| sd * rng.gaussian()).collect();
    let mut y = vec![0.0; total];
    for t in 0..total {
        let mut v = constant + e[t];
        for (i, phi) in ar.iter().enumerate() {
            if t > i {
                v += phi * y[t - 1 - i];
            }
        }
        for (j, theta) in ma.iter().enumerate() {
            if t > j {
                v += theta * e[t - 1 - j];
            }
        }
        y[t] = v;
    }
    y.split_off(burn)
}

/// Cumulative sum applied `d` times, starting each level from zero:
/// turns an ARMA sample into an ARIMA(p, d, q) sample.
pub fn integrate(x: &[f64], d: usize) -> Vec<f64> {
    let mut y = x.to_vec();
    for _ in 0..d {
        let mut acc = 0.0;
        for v in &mut y {
            acc += *v;
            *v = acc;
        }
    }
    y
}
