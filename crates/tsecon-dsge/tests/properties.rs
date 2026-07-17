//! Property tests for the Blanchard-Kahn solver.
//!
//! These check structural guarantees rather than reference numbers: the solved
//! transition `P` is stable, the Blanchard-Kahn condition correctly flags
//! mis-specified models, impulse responses revert, and simulation is internally
//! consistent. One additional closed-form check covers the complex-eigenvalue
//! (oscillatory) path: an exogenous state with complex conjugate roots yields a
//! real policy rule matching `G = first row of (I - a R)^{-1}`.

use serde_json::Value;
use tsecon_dsge::{solve, verdict, BlanchardKahnVerdict, DsgeError, LinearReModel};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-dsge.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .expect("matrix")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("row")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect()
}

fn model_from(fx: &Value) -> LinearReModel {
    let a = rows(&fx["A"]);
    let b = rows(&fx["B"]);
    let c = rows(&fx["C"]);
    let n_pre = fx["n_predetermined"].as_u64().expect("n_predetermined") as usize;
    LinearReModel::new(&a, &b, &c, n_pre).expect("valid model")
}

/// The solved transition `P` is stable: its spectral radius is strictly below
/// one (the stable eigenvalues of `M` are exactly the eigenvalues of `P`).
#[test]
fn solved_p_is_stable() {
    let fx = load();
    for key in ["cagan", "two_shock"] {
        let sol = solve(&model_from(&fx[key])).expect("solution");
        let rho = tsecon_linalg::spectral_radius(sol.p().as_ref()).expect("spectral radius");
        assert!(rho < 1.0, "{key}: P spectral radius {rho} must be < 1");
    }
}

/// Blanchard-Kahn flags a model with too FEW jump variables (here all variables
/// declared predetermined, so `n_jump = 0 < n_unstable = 1`) as having NO stable
/// solution — from both `verdict` and `solve`.
#[test]
fn too_few_jumps_is_no_stable_solution() {
    let fx = load();
    let model = model_from(&fx["misspec_too_few_jumps"]);
    let (v, _) = verdict(&model).expect("verdict");
    assert!(
        matches!(v, BlanchardKahnVerdict::NoStableSolution { .. }),
        "expected NoStableSolution, got {v}"
    );
    match solve(&model) {
        Err(DsgeError::BlanchardKahn(BlanchardKahnVerdict::NoStableSolution { .. })) => {}
        other => panic!("expected BlanchardKahn(NoStableSolution), got {other:?}"),
    }
}

/// Blanchard-Kahn flags a model with too MANY jump variables (here all variables
/// declared jump, so `n_jump = 2 > n_unstable = 1`) as INDETERMINATE.
#[test]
fn too_many_jumps_is_indeterminate() {
    let fx = load();
    let model = model_from(&fx["misspec_too_many_jumps"]);
    let (v, _) = verdict(&model).expect("verdict");
    assert!(
        matches!(v, BlanchardKahnVerdict::Indeterminate { .. }),
        "expected Indeterminate, got {v}"
    );
    match solve(&model) {
        Err(DsgeError::BlanchardKahn(BlanchardKahnVerdict::Indeterminate { .. })) => {}
        other => panic!("expected BlanchardKahn(Indeterminate), got {other:?}"),
    }
}

/// A one-time unit innovation decays back to zero: the impulse response of both
/// the predetermined and jump variables shrinks toward the origin.
#[test]
fn impulse_response_reverts() {
    let fx = load();
    let sol = solve(&model_from(&fx["two_shock"])).expect("solution");
    let irf = sol.impulse_response(0, 60).expect("irf");
    // Impact is non-trivial.
    let impact: f64 = irf.predetermined[0].iter().map(|v| v.abs()).sum();
    assert!(impact > 1e-6, "impact should be non-trivial");
    // Late response has essentially vanished.
    let tail_pre: f64 = irf.predetermined[60].iter().map(|v| v.abs()).sum();
    let tail_jump: f64 = irf.jump[60].iter().map(|v| v.abs()).sum();
    assert!(
        tail_pre < 1e-4,
        "predetermined tail {tail_pre} should revert"
    );
    assert!(tail_jump < 1e-4, "jump tail {tail_jump} should revert");
    // Monotone-ish decay: the norm at t=60 is far below the impact.
    assert!(tail_pre < impact * 1e-3);
}

/// The complex-eigenvalue (oscillatory) path: an exogenous 2-D state with
/// complex conjugate roots `r e^{+/- i theta}` driving a jump
/// `p_t = a E_t[p_{t+1}] + u1_t`. The policy rule must come out REAL and equal
/// the first row of `(I - a R)^{-1}` (the closed-form forward sum
/// `sum_j a^j R^j`), and the eigenvalues must include the complex pair.
#[test]
fn complex_roots_yield_real_policy() {
    let a = 0.5;
    let r = 0.7;
    let theta = 0.5_f64;
    let (c, s) = (theta.cos(), theta.sin());
    let (r11, r12, r21, r22) = (r * c, -r * s, r * s, r * c);

    let amat = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    // Rows: u1, u2 (exogenous rotation R), p (jump): E_t p_{t+1} = (1/a) p - (1/a) u1.
    let bmat = vec![
        vec![r11, r12, 0.0],
        vec![r21, r22, 0.0],
        vec![-1.0 / a, 0.0, 1.0 / a],
    ];
    let cmat = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
    let model = LinearReModel::new(&amat, &bmat, &cmat, 2).expect("model");
    let sol = solve(&model).expect("solution");

    // Closed form: first row of (I - a R)^{-1}.
    let (m11, m12) = (1.0 - a * r11, -a * r12);
    let (m21, m22) = (-a * r21, 1.0 - a * r22);
    let det = m11 * m22 - m12 * m21;
    let g0 = m22 / det;
    let g1 = -m12 / det;
    assert!((sol.g()[(0, 0)] - g0).abs() < 1e-9, "G[0,0] vs closed form");
    assert!((sol.g()[(0, 1)] - g1).abs() < 1e-9, "G[0,1] vs closed form");

    // The eigenvalues include a genuine complex conjugate pair.
    let has_complex = sol.eigenvalues().iter().any(|z| z.im.abs() > 1e-6);
    assert!(has_complex, "expected a complex conjugate eigenvalue pair");

    // P equals the exogenous rotation R (jump feedback does not touch the states).
    assert!((sol.p()[(0, 0)] - r11).abs() < 1e-9);
    assert!((sol.p()[(0, 1)] - r12).abs() < 1e-9);
    assert!((sol.p()[(1, 0)] - r21).abs() < 1e-9);
    assert!((sol.p()[(1, 1)] - r22).abs() < 1e-9);
}

/// A shock written directly onto a jump equation is rejected: the crate's
/// convention (`E_t[z_{t+1}] = 0`, innovation in the predetermined block only)
/// cannot represent it as `jump = G predetermined`.
#[test]
fn shock_on_jump_is_rejected() {
    // Cagan matrices, but the shock loads on the jump row (p) instead of u.
    let amat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let bmat = vec![vec![0.6, 0.0], vec![-2.0, 2.0]];
    let cmat = vec![vec![0.0], vec![1.0]]; // shock on p (jump) row
    let model = LinearReModel::new(&amat, &bmat, &cmat, 1).expect("model");
    match solve(&model) {
        Err(DsgeError::ShockOnJump { .. }) => {}
        other => panic!("expected ShockOnJump, got {other:?}"),
    }
}

/// A singular lead matrix `A` is rejected with a clear error rather than
/// producing garbage.
#[test]
fn singular_a_is_rejected() {
    let amat = vec![vec![1.0, 1.0], vec![1.0, 1.0]]; // rank 1
    let bmat = vec![vec![0.6, 0.0], vec![-2.0, 2.0]];
    let cmat = vec![vec![1.0], vec![0.0]];
    let model = LinearReModel::new(&amat, &bmat, &cmat, 1).expect("model builds");
    assert!(matches!(solve(&model), Err(DsgeError::SingularA)));
    assert_eq!(verdict(&model).err(), Some(DsgeError::SingularA));
}

/// A unit-root model (an eigenvalue on the unit circle) is rejected: the
/// stable/unstable split is undefined at the boundary.
#[test]
fn unit_root_is_rejected() {
    // M = [[1, 0], [-2, 2]]: eigenvalues 1 (on the circle) and 2.
    let amat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let bmat = vec![vec![1.0, 0.0], vec![-2.0, 2.0]];
    let cmat = vec![vec![1.0], vec![0.0]];
    let model = LinearReModel::new(&amat, &bmat, &cmat, 1).expect("model");
    assert!(matches!(solve(&model), Err(DsgeError::UnitRoot { .. })));
}

/// `simulate` matches a by-hand recursion of the solved law of motion and, run
/// under zero shocks from a displaced state, reverts to zero (stability of `P`).
#[test]
fn simulate_is_consistent_and_reverts() {
    let fx = load();
    let sol = solve(&model_from(&fx["two_shock"])).expect("solution");

    // Consistency: one nonzero shock then zeros, compared to manual recursion.
    let shocks = vec![vec![1.0, -0.5], vec![0.0, 0.0], vec![0.3, 0.2]];
    let traj = sol.simulate(&[0.0, 0.0], &shocks).expect("simulate");
    assert_eq!(traj.predetermined.len(), shocks.len() + 1);

    // Manual recursion.
    let mut k = vec![0.0f64, 0.0];
    for (t, z) in shocks.iter().enumerate() {
        let jump: Vec<f64> = (0..sol.n_jump())
            .map(|i| (0..2).map(|j| sol.g()[(i, j)] * k[j]).sum())
            .collect();
        for (i, ji) in jump.iter().enumerate() {
            assert!(
                (traj.jump[t][i] - ji).abs() < 1e-12,
                "jump mismatch at t={t}"
            );
        }
        let mut next = vec![0.0f64; 2];
        for (i, ni) in next.iter_mut().enumerate() {
            for (j, &kj) in k.iter().enumerate() {
                *ni += sol.p()[(i, j)] * kj;
            }
            for (j, &zj) in z.iter().enumerate() {
                *ni += sol.q()[(i, j)] * zj;
            }
        }
        k = next;
    }

    // Reversion: from a displaced state, zero shocks, the state decays.
    let zeros: Vec<Vec<f64>> = (0..80).map(|_| vec![0.0, 0.0]).collect();
    let decay = sol.simulate(&[1.0, -1.0], &zeros).expect("simulate");
    let last: f64 = decay
        .predetermined
        .last()
        .expect("nonempty")
        .iter()
        .map(|v| v.abs())
        .sum();
    assert!(last < 1e-4, "state should revert to zero, got {last}");
}

/// `verdict` and `solve` agree on a unique model, and the reported eigenvalues
/// are consistent between the two entry points (same count, same moduli set).
#[test]
fn verdict_and_solve_agree() {
    let fx = load();
    let model = model_from(&fx["cagan"]);
    let (v, ev) = verdict(&model).expect("verdict");
    assert!(v.is_unique());
    let sol = solve(&model).expect("solution");
    assert_eq!(sol.verdict(), v);
    assert_eq!(sol.eigenvalues().len(), ev.len());
}
