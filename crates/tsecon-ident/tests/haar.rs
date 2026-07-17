//! Statistical validation of the Haar-uniform rotation kernel: column
//! orthonormality, entry mean, determinant-sign balance, and the first-entry
//! marginal moments predicted by the uniform-on-`O(m)` measure
//! (Mezzadri 2007).

use tsecon_ident::{haar_rotation, IdentError};
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_rng::Stream;

/// Determinant by Gaussian elimination with partial pivoting (small matrices
/// only; sign-tracked pivoting is enough for the ±1 test).
fn det(a: MatRef<'_, f64>) -> f64 {
    let n = a.nrows();
    let mut m = a.to_owned();
    let mut d = 1.0;
    for k in 0..n {
        let mut piv = k;
        let mut best = m[(k, k)].abs();
        for i in (k + 1)..n {
            if m[(i, k)].abs() > best {
                best = m[(i, k)].abs();
                piv = i;
            }
        }
        if best == 0.0 {
            return 0.0;
        }
        if piv != k {
            for j in 0..n {
                let t = m[(k, j)];
                m[(k, j)] = m[(piv, j)];
                m[(piv, j)] = t;
            }
            d = -d;
        }
        d *= m[(k, k)];
        for i in (k + 1)..n {
            let f = m[(i, k)] / m[(k, k)];
            for j in k..n {
                m[(i, j)] -= f * m[(k, j)];
            }
        }
    }
    d
}

#[test]
fn zero_order_is_rejected() {
    let mut s = Stream::new(1);
    assert!(matches!(
        haar_rotation(0, &mut s),
        Err(IdentError::InvalidArgument { .. })
    ));
}

#[test]
fn columns_are_orthonormal_to_machine_precision() {
    let mut s = Stream::new(20240717);
    for m in 1..=6 {
        for _ in 0..50 {
            let q = haar_rotation(m, &mut s).expect("haar draw");
            for a in 0..m {
                for b in 0..m {
                    let mut dot = 0.0;
                    for i in 0..m {
                        dot += q[(i, a)] * q[(i, b)];
                    }
                    let target = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (dot - target).abs() < 1e-12,
                        "column ({a},{b}) inner product {dot} off target {target}"
                    );
                }
            }
        }
    }
}

#[test]
fn entry_mean_is_near_zero() {
    let mut s = Stream::new(7);
    let m = 4;
    let n_draws = 4000;
    let mut sum = 0.0;
    for _ in 0..n_draws {
        let q = haar_rotation(m, &mut s).expect("haar draw");
        for j in 0..m {
            for i in 0..m {
                sum += q[(i, j)];
            }
        }
    }
    let mean = sum / (n_draws * m * m) as f64;
    // SE ~ (1/sqrt(m)) / sqrt(n_draws * m^2) ~ 0.002; 0.02 is ~10 SE.
    assert!(mean.abs() < 0.02, "entry mean {mean} not near zero");
}

#[test]
fn determinant_sign_is_balanced() {
    let mut s = Stream::new(99);
    let m = 3;
    let n_draws = 4000;
    let mut positives = 0usize;
    for _ in 0..n_draws {
        let q = haar_rotation(m, &mut s).expect("haar draw");
        let d = det(q.as_ref());
        assert!((d.abs() - 1.0).abs() < 1e-10, "determinant {d} not +/-1");
        if d > 0.0 {
            positives += 1;
        }
    }
    let frac = positives as f64 / n_draws as f64;
    // SE ~ 0.5/sqrt(4000) ~ 0.008; 0.04 is ~5 SE.
    assert!(
        (frac - 0.5).abs() < 0.04,
        "fraction with det +1 is {frac}, not near 1/2"
    );
}

#[test]
fn first_entry_marginal_moments_match_theory() {
    // The first column of a Haar orthogonal matrix is uniform on the unit
    // sphere S^{m-1}, so any single entry has mean 0 and variance 1/m.
    let mut s = Stream::new(2024);
    let m = 4;
    let n_draws = 20000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..n_draws {
        let q = haar_rotation(m, &mut s).expect("haar draw");
        let x = q[(0, 0)];
        sum += x;
        sum_sq += x * x;
    }
    let mean = sum / n_draws as f64;
    let var = sum_sq / n_draws as f64 - mean * mean;
    assert!(mean.abs() < 0.02, "Q[0,0] mean {mean} not near 0");
    assert!(
        (var - 1.0 / m as f64).abs() < 0.02,
        "Q[0,0] variance {var} not near 1/m = {}",
        1.0 / m as f64
    );
}

#[test]
fn draw_is_deterministic_for_a_seed() {
    let mut a = Stream::new(555);
    let mut b = Stream::new(555);
    let qa = haar_rotation(5, &mut a).expect("haar draw");
    let qb = haar_rotation(5, &mut b).expect("haar draw");
    for j in 0..5 {
        for i in 0..5 {
            assert_eq!(qa[(i, j)].to_bits(), qb[(i, j)].to_bits());
        }
    }
}

#[test]
fn one_by_one_is_plus_or_minus_one() {
    let mut s = Stream::new(3);
    for _ in 0..100 {
        let q: Mat<f64> = haar_rotation(1, &mut s).expect("haar draw");
        assert!((q[(0, 0)].abs() - 1.0).abs() < 1e-15);
    }
}
