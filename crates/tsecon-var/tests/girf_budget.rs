//! The GIRF engine's memory budget (audit round 13, the S3 class): a
//! request whose draw buffers or per-history results would exceed
//! `MEMORY_BUDGET_BYTES` is refused up front as `VarError::MemoryBudget`
//! — quickly, naming the counts — instead of being handed to the
//! allocator (which aborted the process at 2^31 values), and a request
//! under the budget runs as before.

use std::time::Instant;

use tsecon_var::{girf, GirfOptions, GirfShock, LinearVarModel, VarError, MEMORY_BUDGET_BYTES};

fn model() -> LinearVarModel {
    let a1 = vec![
        vec![0.5, 0.1, 0.0],
        vec![0.0, 0.4, 0.1],
        vec![0.1, 0.0, 0.3],
    ];
    let sigma = vec![
        vec![1.0, 0.2, 0.0],
        vec![0.2, 1.0, 0.1],
        vec![0.0, 0.1, 1.0],
    ];
    LinearVarModel::new(&[a1], vec![0.0; 3], sigma).expect("model")
}

fn opts(horizon: usize, n_draws: usize) -> GirfOptions {
    GirfOptions {
        shock: GirfShock::Orthogonal { var: 0, size: 1.0 },
        horizon,
        n_draws,
        seed: 3,
        antithetic: false,
        bands: (0.16, 0.84),
    }
}

#[test]
fn a_request_beyond_the_budget_is_refused_quickly_naming_the_counts() {
    let m = model();
    let hist = vec![vec![0.1, -0.2, 0.3]];
    let t0 = Instant::now();
    let err = girf(&m, &hist, &opts(10, 1 << 31)).unwrap_err();
    assert!(t0.elapsed().as_millis() < 500, "refusal must not allocate");
    match err {
        VarError::MemoryBudget { bytes, budget, .. } => {
            assert_eq!(budget, MEMORY_BUDGET_BYTES);
            assert!(bytes > MEMORY_BUDGET_BYTES as u128);
        }
        other => panic!("expected MemoryBudget, got {other:?}"),
    }
    let text = err.to_string();
    for name in ["n_draws", "horizon", "histories", "budget", "GiB"] {
        assert!(text.contains(name), "{text}");
    }
    // A long horizon with many draws is the same class.
    let err = girf(&m, &hist, &opts(500_000, 4_000)).unwrap_err();
    assert!(matches!(err, VarError::MemoryBudget { .. }), "{err}");
    // Many histories times a long horizon (the per-history results) too.
    let many: Vec<Vec<f64>> = (0..200_000).map(|i| vec![i as f64 * 1e-6; 3]).collect();
    let err = girf(&m, &many, &opts(200_000, 2)).unwrap_err();
    assert!(matches!(err, VarError::MemoryBudget { .. }), "{err}");
}

#[test]
fn a_request_under_the_budget_runs_and_the_budget_is_two_gib() {
    assert_eq!(MEMORY_BUDGET_BYTES, 2 << 30);
    let m = model();
    let hist = vec![vec![0.1, -0.2, 0.3], vec![0.0, 0.5, -0.1]];
    let g = girf(&m, &hist, &opts(12, 64)).expect("small run");
    assert_eq!(g.girf.len(), 13);
    assert_eq!(g.n_histories, 2);
    // The bytes estimate is monotone in every count, so a run that fits is
    // reported as fitting whatever the thread count.
    let g2 = girf(&m, &hist, &opts(12, 2)).expect("smaller run");
    assert_eq!(g2.n_draws, 2);
}
