//! Golden and property tests for the zero + sign restricted SVAR
//! (`zero_sign_svar`, Rubio-Ramirez-Waggoner-Zha 2010).
//!
//! `fixtures/zero_sign_svar.json` is produced by
//! `fixtures/generate_zero_sign_svar_fixtures.py`, a NumPy-only reference
//! (never imports tsecon) that transcribes `Theta_h = Psi_h chol_lower(Sigma)`
//! from the pure companion-power MA recursion. The PRIMARY golden is the
//! recursive/Cholesky recovery: with strict-upper-triangle impact zeros, no
//! sign restrictions, and positive-diagonal normalization, the RWZ column
//! recursion is one-dimensional at every step, the ARW weight is one, and the
//! whole path collapses to the Cholesky IRF DETERMINISTICALLY. Reproducing
//! the fixture `theta[h]` to 1e-10 (independent of the seed) validates both
//! `cholesky_irf` and the null-space recursion at once.

use serde_json::Value;
use tsecon_bayes::cholesky_irf;
use tsecon_ident::zero::{zero_constrained_rotation, ZeroRestriction, ZeroRestrictionSet};
use tsecon_ident::IdentError;
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::Stream;

const TOL: f64 = 1e-10;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/zero_sign_svar.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn mat_from(v: &Value) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .expect("2-D array")
        .iter()
        .map(|r| {
            r.as_array()
                .expect("row array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect();
    let nr = rows.len();
    let nc = rows[0].len();
    Mat::from_fn(nr, nc, |i, j| rows[i][j])
}

/// Positive-diagonal orientation on the impact matrix.
fn positive_diagonal(irf: &mut [Mat<f64>], n: usize) {
    for j in 0..n {
        if irf[0][(j, j)] < 0.0 {
            for m in irf.iter_mut() {
                for i in 0..n {
                    m[(i, j)] *= -1.0;
                }
            }
        }
    }
}

fn structural_irf(base: &[Mat<f64>], q: MatRef<'_, f64>) -> Vec<Mat<f64>> {
    base.iter().map(|m| m.as_ref() * q).collect()
}

fn strict_upper_impact_zeros(n: usize, horizon: usize) -> ZeroRestrictionSet {
    let mut rs = Vec::new();
    for j in 0..n {
        for i in 0..j {
            rs.push(ZeroRestriction::at(i, j, 0));
        }
    }
    ZeroRestrictionSet::new(rs, n, horizon).expect("recursive zeros valid")
}

/// Runs the recursive-recovery golden for one fixture case at one seed.
fn check_recursive_case(case: &Value, seed: u64) -> Result<(), IdentError> {
    let b = mat_from(&case["b"]);
    let sigma = mat_from(&case["sigma"]);
    let p = case["p"].as_u64().expect("p") as usize;
    let horizon = case["horizon"].as_u64().expect("horizon") as usize;
    let n = sigma.nrows();

    let base = cholesky_irf(b.as_ref(), sigma.as_ref(), p, horizon)?;
    let zeros = strict_upper_impact_zeros(n, horizon);

    let mut streams = Stream::substreams(seed, 1)?;
    let (q, w) = zero_constrained_rotation(&base, &zeros, &mut streams[0])?;

    // Impact-only pattern => weight is exactly one.
    assert_eq!(w, 1.0, "impact-only weight must be 1");

    // Q orthogonal to 1e-12.
    for a in 0..n {
        for c in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                s += q[(i, a)] * q[(i, c)];
            }
            let target = if a == c { 1.0 } else { 0.0 };
            assert!(
                (s - target).abs() < 1e-12,
                "Q not orthogonal: (Q'Q)[{a}][{c}] = {s}"
            );
        }
    }

    let mut theta = structural_irf(&base, q.as_ref());
    positive_diagonal(&mut theta, n);

    // Strict-upper-triangle impact zeros hold exactly.
    for i in 0..n {
        for j in 0..n {
            if i < j {
                assert!(
                    theta[0][(i, j)].abs() < 1e-12,
                    "impact zero violated at ({i},{j}): {}",
                    theta[0][(i, j)]
                );
            }
        }
    }

    // Deterministic match to the NumPy Theta_h = Psi_h L to 1e-10.
    let expected = case["theta"].as_array().expect("theta array");
    for (h, theta_h) in theta.iter().enumerate() {
        let exp_h = mat_from(&expected[h]);
        for i in 0..n {
            for j in 0..n {
                let a = theta_h[(i, j)];
                let e = exp_h[(i, j)];
                assert!(
                    (a - e).abs() < TOL,
                    "theta[{h}][{i}][{j}]: actual={a:.15e} expected={e:.15e} abs_err={:.3e}",
                    (a - e).abs()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn recursive_recovery_p1() -> Result<(), IdentError> {
    let fx = load();
    check_recursive_case(&fx["recursive_3var_p1"], 0)
}

#[test]
fn recursive_recovery_p2() -> Result<(), IdentError> {
    let fx = load();
    check_recursive_case(&fx["recursive_3var_p2"], 0)
}

#[test]
fn recursive_recovery_is_seed_independent() -> Result<(), IdentError> {
    // The recursive/Cholesky collapse is deterministic: any seed reproduces
    // the same structural IRF (the Gaussian only picks the sign that
    // positive-diagonal normalization then fixes).
    let fx = load();
    for seed in [0u64, 1, 42, 20260722, u64::MAX / 3] {
        check_recursive_case(&fx["recursive_3var_p1"], seed)?;
    }
    Ok(())
}

#[test]
fn impact_matrix_equals_cholesky_factor() -> Result<(), IdentError> {
    // theta[0] must equal the stored lower Cholesky factor L.
    let fx = load();
    let case = &fx["recursive_3var_p1"];
    let b = mat_from(&case["b"]);
    let sigma = mat_from(&case["sigma"]);
    let p = case["p"].as_u64().expect("p") as usize;
    let horizon = case["horizon"].as_u64().expect("horizon") as usize;
    let n = sigma.nrows();
    let base = cholesky_irf(b.as_ref(), sigma.as_ref(), p, horizon)?;
    let zeros = strict_upper_impact_zeros(n, horizon);
    let mut streams = Stream::substreams(7, 1)?;
    let (q, _) = zero_constrained_rotation(&base, &zeros, &mut streams[0])?;
    let mut theta = structural_irf(&base, q.as_ref());
    positive_diagonal(&mut theta, n);
    let chol = mat_from(&case["chol"]);
    for i in 0..n {
        for j in 0..n {
            assert!((theta[0][(i, j)] - chol[(i, j)]).abs() < TOL);
        }
    }
    Ok(())
}
