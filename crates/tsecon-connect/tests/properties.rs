//! Property and invariant tests: generalized-FEVD rows sum to one, total
//! connectedness stays in [0, 100], a (near-)diagonal system is nearly
//! unconnected, the to/from/net accounting identities hold, and rolling
//! connectedness runs and stays bounded.

mod common;

use common::Lcg;
use tsecon_connect::{generalized_fevd, rolling_total_connectedness, ConnectednessTable};
use tsecon_linalg::faer::Mat;
use tsecon_var::{ma_rep, Trend, VarSpec};

/// A diagonal `k x k` matrix from a slice of diagonal entries.
fn diag(d: &[f64]) -> Mat<f64> {
    let k = d.len();
    Mat::from_fn(k, k, |i, j| if i == j { d[i] } else { 0.0 })
}

/// Simulates `n` observations of a VAR(1) `y_t = A y_{t-1} + u_t` with
/// `u_t` iid N(0, diag(s)^2), returned as an `n x k` matrix.
fn simulate_var1(a: &Mat<f64>, s: &[f64], n: usize, seed: u64) -> Mat<f64> {
    let k = a.nrows();
    let mut rng = Lcg::new(seed);
    let mut y = vec![0.0_f64; k];
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut next = vec![0.0_f64; k];
        for i in 0..k {
            let mut acc = s[i] * rng.gaussian();
            for j in 0..k {
                acc += a[(i, j)] * y[j];
            }
            next[i] = acc;
        }
        rows.push(next.clone());
        y = next;
    }
    Mat::from_fn(n, k, |t, i| rows[t][i])
}

/// Every row of the row-normalized generalized FEVD sums to one.
#[test]
fn gfevd_rows_sum_to_one() {
    let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => 0.5,
        (0, 1) => 0.2,
        (1, 1) => 0.3,
        (1, 2) => -0.15,
        (2, 0) => 0.1,
        (2, 2) => 0.4,
        _ => 0.0,
    });
    let psi = ma_rep(&[a], 10).unwrap();
    let sigma = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => 1.0,
        (1, 1) => 2.0,
        (2, 2) => 0.5,
        (0, 1) | (1, 0) => 0.3,
        _ => 0.0,
    });
    let theta = generalized_fevd(&psi, sigma.as_ref()).unwrap();
    for i in 0..3 {
        let s: f64 = (0..3).map(|j| theta[(i, j)]).sum();
        assert!((s - 1.0).abs() < 1e-12, "row {i} sums to {s}");
    }
}

/// The total connectedness index lies in [0, 100], and the to/from/net
/// accounting identities hold: `net = to - from`, `sum net = 0`, and each
/// row of the net-pairwise matrix sums to that variable's net.
#[test]
fn connectedness_identities() {
    let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => 0.4,
        (0, 2) => 0.25,
        (1, 0) => 0.2,
        (1, 1) => 0.3,
        (2, 1) => 0.15,
        (2, 2) => 0.5,
        _ => 0.0,
    });
    let psi = ma_rep(&[a], 12).unwrap();
    let sigma = diag(&[1.0, 1.5, 0.8]);
    let theta = generalized_fevd(&psi, sigma.as_ref()).unwrap();
    let t = ConnectednessTable::from_gfevd(theta.as_ref()).unwrap();

    assert!(t.total >= 0.0 && t.total <= 100.0, "total = {}", t.total);

    let mut net_sum = 0.0;
    for i in 0..t.k {
        let net = t.to_others[i] - t.from_others[i];
        assert!((net - t.net[i]).abs() < 1e-12, "net[{i}] identity");
        net_sum += t.net[i];
        let row: f64 = (0..t.k).map(|j| t.pairwise_net[(i, j)]).sum();
        assert!(
            (row - t.net[i]).abs() < 1e-10,
            "pairwise row {i} sums to {row}, net {}",
            t.net[i]
        );
    }
    assert!(net_sum.abs() < 1e-10, "net sums to {net_sum}");

    // Total is the shared off-diagonal mass: average from == average to.
    let avg_to: f64 = t.to_others.iter().sum();
    let avg_from: f64 = t.from_others.iter().sum();
    assert!((avg_to - t.total).abs() < 1e-10);
    assert!((avg_from - t.total).abs() < 1e-10);
}

/// A (near-)diagonal VAR with diagonal innovations has near-zero
/// connectedness: each variable's variance is explained almost entirely
/// by its own shock.
#[test]
fn near_diagonal_var_is_unconnected() {
    let a = diag(&[0.5, -0.3, 0.4]);
    let psi = ma_rep(&[a], 20).unwrap();
    let sigma = diag(&[1.0, 2.0, 0.5]);
    let theta = generalized_fevd(&psi, sigma.as_ref()).unwrap();
    // theta is the identity to roundoff.
    for i in 0..3 {
        for j in 0..3 {
            let target = f64::from(u8::from(i == j));
            assert!(
                (theta[(i, j)] - target).abs() < 1e-12,
                "theta[{i},{j}] = {}",
                theta[(i, j)]
            );
        }
    }
    let t = ConnectednessTable::from_gfevd(theta.as_ref()).unwrap();
    assert!(t.total < 1e-9, "diagonal total connectedness = {}", t.total);
}

/// Rolling total connectedness runs on simulated data, returns one value
/// per window, and every value stays inside [0, 100].
#[test]
fn rolling_runs_and_stays_bounded() {
    let a = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => 0.4,
        (0, 1) => 0.2,
        (1, 1) => 0.3,
        (2, 0) => 0.15,
        (2, 2) => 0.35,
        _ => 0.0,
    });
    let data = simulate_var1(&a, &[1.0, 1.2, 0.9], 300, 20260717);
    let spec = VarSpec::new(2, Trend::Constant).unwrap();
    let window = 100;
    let series = rolling_total_connectedness(data.as_ref(), window, spec, 10).unwrap();
    assert_eq!(series.len(), 300 - window + 1);
    for (w, &c) in series.iter().enumerate() {
        assert!(
            c.is_finite() && (0.0..=100.0).contains(&c),
            "window {w} total = {c}"
        );
    }
}

/// Error paths: empty MA weights, a non-square sigma, and a mismatched
/// label list are all rejected without panicking.
#[test]
fn error_paths() {
    let sigma = diag(&[1.0, 1.0]);
    assert!(generalized_fevd(&[], sigma.as_ref()).is_err());

    let psi = vec![Mat::<f64>::from_fn(2, 2, |i, j| {
        f64::from(u8::from(i == j))
    })];
    let nonsquare = Mat::<f64>::zeros(2, 3);
    assert!(generalized_fevd(&psi, nonsquare.as_ref()).is_err());

    let theta = generalized_fevd(&psi, sigma.as_ref()).unwrap();
    let t = ConnectednessTable::from_gfevd(theta.as_ref()).unwrap();
    assert!(t.with_labels(vec!["only-one".into()]).is_err());
}
