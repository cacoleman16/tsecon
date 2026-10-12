//! Golden tests for the historical decomposition (`decompose`, Kilian &
//! Lütkepohl 2017, ch. 4) and its structural-shock / baseline core.
//!
//! `fixtures/historical_decomposition_chol.json` is produced by
//! `fixtures/generate_historical_decomposition_fixtures.py`, a NumPy-only
//! reference (never imports tsecon) that fits a fixed VAR(2) by OLS on a
//! seeded dataset, Cholesky-identifies (Q = I), and computes the structural
//! shocks `E`, the structural IRF `Theta_s`, the decomposition tensor
//! `HD[t][i][j]`, and the deterministic/initial-condition baseline from an
//! INDEPENDENT implementation. The Rust core (faer residuals +
//! forward-substitution orthogonalization + companion-power MA) reproducing
//! these to rtol=1e-8/atol=1e-10 — and the adding-up residual staying below
//! 1e-9 — validates the whole HD stack the narrative sampler rests on.
//!
//! `fixtures/historical_decomposition_impact.json` (from
//! `fixtures/generate_historical_decomposition_impact_fixtures.py`, statsmodels
//! VAR reduced form + a NumPy closed form, never tsecon) pins the
//! arbitrary-impact entry point `decompose_with_impact`: the Blanchard-Quah
//! VAR(8) on the replication page's data, and a seeded 3-variable VAR(2) with
//! a random non-orthogonal impact matrix that is not covariance-consistent.

use serde_json::Value;
use tsecon_ident::histdecomp::{decompose, decompose_with_impact};
use tsecon_ident::IdentError;
use tsecon_linalg::faer::{Mat, MatRef};

const RTOL: f64 = 1e-8;
const ATOL: f64 = 1e-10;

fn load() -> Value {
    load_named("historical_decomposition_chol.json")
}

fn load_named(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
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

fn mats_from(v: &Value) -> Vec<Mat<f64>> {
    v.as_array()
        .expect("array of matrices")
        .iter()
        .map(mat_from)
        .collect()
}

fn assert_close(got: f64, want: f64, ctx: &str) {
    let tol = ATOL + RTOL * want.abs();
    assert!(
        (got - want).abs() <= tol,
        "{ctx}: got {got}, want {want} (|diff|={:.3e} > tol={:.3e})",
        (got - want).abs(),
        tol
    );
}

fn assert_mat_close(got: MatRef<'_, f64>, want: MatRef<'_, f64>, ctx: &str) {
    assert_eq!(got.nrows(), want.nrows(), "{ctx}: row count");
    assert_eq!(got.ncols(), want.ncols(), "{ctx}: col count");
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            assert_close(got[(i, j)], want[(i, j)], &format!("{ctx}[{i},{j}]"));
        }
    }
}

fn run_case(name: &str, case: &Value) -> Result<(), IdentError> {
    let p = case["p"].as_u64().expect("p") as usize;
    let n = case["n"].as_u64().expect("n") as usize;
    let horizon = case["horizon"].as_u64().expect("horizon") as usize;
    let data = mat_from(&case["data"]);
    let b = mat_from(&case["b"]);
    let sigma = mat_from(&case["sigma"]);
    let eye = Mat::<f64>::identity(n, n);

    let hd = decompose(
        data.as_ref(),
        b.as_ref(),
        sigma.as_ref(),
        eye.as_ref(),
        p,
        horizon,
    )?;

    // Structural shocks E.
    assert_mat_close(
        hd.shocks(),
        mat_from(&case["shocks"]).as_ref(),
        &format!("{name} shocks"),
    );

    // Structural IRF Theta_s (Q = I => Theta_s = Psi_s P, the Cholesky IRF).
    let theta_want = mats_from(&case["theta"]);
    assert_eq!(hd.theta().len(), theta_want.len(), "{name} theta length");
    for (s, want) in theta_want.iter().enumerate() {
        assert_mat_close(
            hd.theta()[s].as_ref(),
            want.as_ref(),
            &format!("{name} theta[{s}]"),
        );
    }

    // Decomposition tensor HD[t][i][j].
    let hd_want = mats_from(&case["hd"]);
    assert_eq!(hd.hd().len(), hd_want.len(), "{name} hd length");
    for (t, want) in hd_want.iter().enumerate() {
        assert_mat_close(
            hd.hd()[t].as_ref(),
            want.as_ref(),
            &format!("{name} hd[{t}]"),
        );
    }

    // Baseline path.
    assert_mat_close(
        hd.baseline(),
        mat_from(&case["baseline"]).as_ref(),
        &format!("{name} baseline"),
    );

    // Adding-up identity: max |y - baseline - sum_j HD| below 1e-9.
    let resid = hd.adding_up_residual(data.as_ref(), p);
    assert!(
        resid < 1e-9,
        "{name} adding-up residual {resid} exceeds 1e-9"
    );

    Ok(())
}

#[test]
fn hd_4var_p2_golden() -> Result<(), IdentError> {
    let fx = load();
    run_case("hd_4var_p2", &fx["hd_4var_p2"])
}

#[test]
fn hd_3var_p1_golden() -> Result<(), IdentError> {
    let fx = load();
    run_case("hd_3var_p1", &fx["hd_3var_p1"])
}

#[test]
fn episode_contribution_is_the_window_difference() -> Result<(), IdentError> {
    // Independent of the golden numbers: C_k over [t1,t2] must equal
    // HD[t2] - HD[t1-1] cell-by-cell.
    let fx = load();
    let case = &fx["hd_3var_p1"];
    let p = case["p"].as_u64().unwrap() as usize;
    let n = case["n"].as_u64().unwrap() as usize;
    let horizon = case["horizon"].as_u64().unwrap() as usize;
    let data = mat_from(&case["data"]);
    let b = mat_from(&case["b"]);
    let sigma = mat_from(&case["sigma"]);
    let eye = Mat::<f64>::identity(n, n);
    let hd = decompose(
        data.as_ref(),
        b.as_ref(),
        sigma.as_ref(),
        eye.as_ref(),
        p,
        horizon,
    )?;
    let (t1, t2) = (4usize, 12usize);
    for i in 0..n {
        for k in 0..n {
            let c = hd.episode_contribution(i, k, t1, t2)?;
            let expect = hd.hd()[t2][(i, k)] - hd.hd()[t1 - 1][(i, k)];
            assert!((c - expect).abs() < 1e-12, "episode ({i},{k})");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// decompose_with_impact: the arbitrary-impact entry point
// ---------------------------------------------------------------------------

fn run_impact_case(name: &str, case: &Value) -> Result<(), IdentError> {
    let p = case["p"].as_u64().expect("p") as usize;
    let t_eff = case["t_eff"].as_u64().expect("t_eff") as usize;
    let data = mat_from(&case["data"]);
    let b = mat_from(&case["b"]);
    let a0 = mat_from(&case["impact"]);
    let hd = decompose_with_impact(data.as_ref(), b.as_ref(), a0.as_ref(), p, t_eff - 1)?;
    assert_eq!(hd.t_eff(), t_eff);
    // Theta is built to T_eff - 1 exactly, whatever the horizon.
    assert_eq!(hd.theta().len(), t_eff, "{name} theta length");
    assert_mat_close(
        hd.shocks(),
        mat_from(&case["shocks"]).as_ref(),
        &format!("{name} shocks"),
    );
    for (s, want) in mats_from(&case["theta_head"]).iter().enumerate() {
        assert_mat_close(
            hd.theta()[s].as_ref(),
            want.as_ref(),
            &format!("{name} theta[{s}]"),
        );
    }
    let hd_want = mats_from(&case["hd"]);
    assert_eq!(hd.hd().len(), hd_want.len(), "{name} hd length");
    for (t, want) in hd_want.iter().enumerate() {
        assert_mat_close(
            hd.hd()[t].as_ref(),
            want.as_ref(),
            &format!("{name} hd[{t}]"),
        );
    }
    assert_mat_close(
        hd.baseline(),
        mat_from(&case["baseline"]).as_ref(),
        &format!("{name} baseline"),
    );
    assert_mat_close(
        hd.shock_covariance().as_ref(),
        mat_from(&case["shock_covariance"]).as_ref(),
        &format!("{name} shock_covariance"),
    );
    let resid = hd.adding_up_residual(data.as_ref(), p);
    assert!(resid < 1e-12, "{name} adding-up residual {resid}");
    Ok(())
}

#[test]
fn hd_impact_bq_var8_golden() -> Result<(), IdentError> {
    let fx = load_named("historical_decomposition_impact.json");
    let case = &fx["bq_var8"];
    run_impact_case("bq_var8", case)?;
    // The seven NBER recession rows of the replication page.
    let p = case["p"].as_u64().unwrap() as usize;
    let t_eff = case["t_eff"].as_u64().unwrap() as usize;
    let data = mat_from(&case["data"]);
    let hd = decompose_with_impact(
        data.as_ref(),
        mat_from(&case["b"]).as_ref(),
        mat_from(&case["impact"]).as_ref(),
        p,
        t_eff - 1,
    )?;
    let rows = case["recessions"].as_array().unwrap();
    assert_eq!(rows.len(), 7);
    for r in rows {
        let a = r["peak_t"].as_u64().unwrap() as usize;
        let z = r["trough_t"].as_u64().unwrap() as usize;
        let actual =
            (data[(p + z, 1)] - hd.baseline()[(z, 1)]) - (data[(p + a, 1)] - hd.baseline()[(a, 1)]);
        assert_close(actual, r["actual"].as_f64().unwrap(), "recession actual");
        assert_close(
            hd.hd()[z][(1, 1)] - hd.hd()[a][(1, 1)],
            r["demand"].as_f64().unwrap(),
            "recession demand",
        );
        assert_close(
            hd.hd()[z][(1, 0)] - hd.hd()[a][(1, 0)],
            r["supply"].as_f64().unwrap(),
            "recession supply",
        );
    }
    // A long-run impact built from the df-adjusted Sigma orthogonalizes the
    // residuals to (T - m)/T * I exactly.
    let m = 1 + 2 * p;
    let want = (t_eff - m) as f64 / t_eff as f64;
    let sc = hd.shock_covariance();
    assert!((sc[(0, 0)] - want).abs() < 1e-10 && (sc[(1, 1)] - want).abs() < 1e-10);
    assert!(sc[(0, 1)].abs() < 1e-10);
    Ok(())
}

#[test]
fn hd_impact_random_nonorthogonal_golden() -> Result<(), IdentError> {
    let fx = load_named("historical_decomposition_impact.json");
    run_impact_case("random_3var_p2", &fx["random_3var_p2"])
}

/// Lower Cholesky factor of a small SPD matrix, transcribed (no library call)
/// so the reduction test does not lean on the code under test.
fn chol_lower(a: MatRef<'_, f64>) -> Mat<f64> {
    let n = a.nrows();
    let mut l = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        let mut d = a[(j, j)];
        for k in 0..j {
            d -= l[(j, k)] * l[(j, k)];
        }
        l[(j, j)] = d.sqrt();
        for i in (j + 1)..n {
            let mut v = a[(i, j)];
            for k in 0..j {
                v -= l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = v / l[(j, j)];
        }
    }
    l
}

fn assert_hd_equal(
    a: &tsecon_ident::HistoricalDecomposition,
    b: &tsecon_ident::HistoricalDecomposition,
    tol: f64,
    ctx: &str,
) {
    assert_eq!(a.t_eff(), b.t_eff());
    for t in 0..a.t_eff() {
        for i in 0..a.n_vars() {
            for j in 0..a.n_vars() {
                let d = (a.hd()[t][(i, j)] - b.hd()[t][(i, j)]).abs();
                assert!(d < tol, "{ctx}: hd[{t}][{i}][{j}] differs by {d:.3e}");
            }
            let d = (a.baseline()[(t, i)] - b.baseline()[(t, i)]).abs();
            assert!(d < tol, "{ctx}: baseline[{t}][{i}] differs by {d:.3e}");
            let d = (a.shocks()[(t, i)] - b.shocks()[(t, i)]).abs();
            assert!(d < tol, "{ctx}: shocks[{t}][{i}] differs by {d:.3e}");
        }
    }
}

#[test]
fn impact_chol_sigma_reduces_to_decompose_q_identity() -> Result<(), IdentError> {
    let fx = load();
    for name in ["hd_4var_p2", "hd_3var_p1"] {
        let case = &fx[name];
        let p = case["p"].as_u64().unwrap() as usize;
        let n = case["n"].as_u64().unwrap() as usize;
        let horizon = case["horizon"].as_u64().unwrap() as usize;
        let data = mat_from(&case["data"]);
        let b = mat_from(&case["b"]);
        let sigma = mat_from(&case["sigma"]);
        let eye = Mat::<f64>::identity(n, n);
        let chol = decompose(
            data.as_ref(),
            b.as_ref(),
            sigma.as_ref(),
            eye.as_ref(),
            p,
            horizon,
        )?;
        let pc = chol_lower(sigma.as_ref());
        let imp = decompose_with_impact(data.as_ref(), b.as_ref(), pc.as_ref(), p, horizon)?;
        assert_hd_equal(&chol, &imp, 1e-12, name);
        // E'E/T = I for the MLE Cholesky factor.
        let sc = imp.shock_covariance();
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((sc[(i, j)] - want).abs() < 1e-10, "{name} shock_covariance");
            }
        }
        // A rotation: impact = P Q equals decompose(q = Q).
        let (s, c) = 0.7f64.sin_cos();
        let q = Mat::from_fn(n, n, |i, j| match (i, j) {
            (0, 0) => c,
            (0, 1) => -s,
            (1, 0) => s,
            (1, 1) => c,
            (a, b) if a == b => 1.0,
            _ => 0.0,
        });
        let rot = decompose(
            data.as_ref(),
            b.as_ref(),
            sigma.as_ref(),
            q.as_ref(),
            p,
            horizon,
        )?;
        let pq = pc.as_ref() * q.as_ref();
        let imp_q = decompose_with_impact(data.as_ref(), b.as_ref(), pq.as_ref(), p, horizon)?;
        assert_hd_equal(&rot, &imp_q, 1e-12, &format!("{name} rotated"));
    }
    Ok(())
}

/// A deterministic pseudo-random draw in (-0.5, 0.5) (xorshift; no tsecon RNG).
fn xorshift(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

#[test]
fn adding_up_holds_for_random_invertible_impacts_and_signs_cancel() -> Result<(), IdentError> {
    let fx = load();
    let case = &fx["hd_4var_p2"];
    let p = case["p"].as_u64().unwrap() as usize;
    let n = case["n"].as_u64().unwrap() as usize;
    let horizon = case["horizon"].as_u64().unwrap() as usize;
    let data = mat_from(&case["data"]);
    let b = mat_from(&case["b"]);
    let mut st = 0x2545_F491_4F6C_DD1Du64;
    for trial in 0..5 {
        let a0 = Mat::from_fn(n, n, |i, j| {
            xorshift(&mut st) * 2.0 + if i == j { 1.5 } else { 0.0 }
        });
        let hd = decompose_with_impact(data.as_ref(), b.as_ref(), a0.as_ref(), p, horizon)?;
        let r = hd.adding_up_residual(data.as_ref(), p);
        assert!(r < 1e-12, "trial {trial}: adding-up residual {r}");
        // Flip the sign of column 1: |hd| and the baseline are invariant.
        let flipped = Mat::from_fn(n, n, |i, j| if j == 1 { -a0[(i, j)] } else { a0[(i, j)] });
        let hf = decompose_with_impact(data.as_ref(), b.as_ref(), flipped.as_ref(), p, horizon)?;
        for t in 0..hd.t_eff() {
            for i in 0..n {
                for j in 0..n {
                    let d = (hd.hd()[t][(i, j)].abs() - hf.hd()[t][(i, j)].abs()).abs();
                    assert!(d < 1e-12, "|hd| not sign-invariant: {d:.3e}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn impact_horizon_beyond_t_eff_is_inert_and_allocates_nothing() -> Result<(), IdentError> {
    let fx = load_named("historical_decomposition_impact.json");
    let case = &fx["random_3var_p2"];
    let p = case["p"].as_u64().unwrap() as usize;
    let t_eff = case["t_eff"].as_u64().unwrap() as usize;
    let data = mat_from(&case["data"]);
    let b = mat_from(&case["b"]);
    let a0 = mat_from(&case["impact"]);
    let base = decompose_with_impact(data.as_ref(), b.as_ref(), a0.as_ref(), p, t_eff - 1)?;
    // usize::MAX - 1 would need ~10^19 matrices if it reached the allocator.
    for h in [t_eff, 10_000_000_000usize, usize::MAX - 1] {
        let big = decompose_with_impact(data.as_ref(), b.as_ref(), a0.as_ref(), p, h)?;
        assert_eq!(big.theta().len(), t_eff);
        assert_hd_equal(
            &base,
            &big,
            0.0 + f64::MIN_POSITIVE,
            &format!("horizon {h}"),
        );
    }
    // A horizon below T_eff - 1 is still refused.
    let err =
        decompose_with_impact(data.as_ref(), b.as_ref(), a0.as_ref(), p, t_eff - 2).unwrap_err();
    assert!(matches!(err, IdentError::InvalidArgument { .. }), "{err:?}");
    Ok(())
}

#[test]
fn impact_refusals_singular_shape_and_nonfinite() {
    let fx = load_named("historical_decomposition_impact.json");
    let case = &fx["random_3var_p2"];
    let p = case["p"].as_u64().unwrap() as usize;
    let t_eff = case["t_eff"].as_u64().unwrap() as usize;
    let data = mat_from(&case["data"]);
    let b = mat_from(&case["b"]);
    let h = t_eff - 1;
    // Exactly singular: two equal columns.
    let sing = Mat::from_fn(3, 3, |i, j| {
        if j == 2 {
            (i + 1) as f64
        } else {
            (i + j + 1) as f64
        }
    });
    let sing = Mat::from_fn(
        3,
        3,
        |i, j| if j == 1 { sing[(i, 0)] } else { sing[(i, j)] },
    );
    let e = decompose_with_impact(data.as_ref(), b.as_ref(), sing.as_ref(), p, h).unwrap_err();
    assert!(
        matches!(e, IdentError::InvalidArgument { what } if what.contains("singular")),
        "{e:?}"
    );
    // Numerically singular by the matrix_rank convention: sigma_min ~ 1e-17.
    let near = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => 1.0,
        (1, 1) => 1.0,
        (2, 2) => 1e-17,
        _ => 0.0,
    });
    let e = decompose_with_impact(data.as_ref(), b.as_ref(), near.as_ref(), p, h).unwrap_err();
    assert!(matches!(e, IdentError::InvalidArgument { .. }), "{e:?}");
    // ... but 1e-12 is above the tolerance (3 * eps ~ 6.7e-16) and accepted.
    let ok = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => 1.0,
        (1, 1) => 1.0,
        (2, 2) => 1e-12,
        _ => 0.0,
    });
    assert!(decompose_with_impact(data.as_ref(), b.as_ref(), ok.as_ref(), p, h).is_ok());
    // All zeros.
    let zero = Mat::<f64>::zeros(3, 3);
    assert!(decompose_with_impact(data.as_ref(), b.as_ref(), zero.as_ref(), p, h).is_err());
    // Wrong shape.
    let rect = Mat::<f64>::identity(3, 2);
    let e = decompose_with_impact(data.as_ref(), b.as_ref(), rect.as_ref(), p, h).unwrap_err();
    assert!(matches!(e, IdentError::Dimension { .. }), "{e:?}");
    let big = Mat::<f64>::identity(4, 4);
    assert!(matches!(
        decompose_with_impact(data.as_ref(), b.as_ref(), big.as_ref(), p, h),
        Err(IdentError::Dimension { .. })
    ));
    // Non-finite entries.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let m = Mat::from_fn(3, 3, |i, j| {
            if (i, j) == (1, 2) {
                bad
            } else if i == j {
                1.0
            } else {
                0.0
            }
        });
        let e = decompose_with_impact(data.as_ref(), b.as_ref(), m.as_ref(), p, h).unwrap_err();
        assert!(
            matches!(e, IdentError::NonFinite { what: "impact" }),
            "{e:?}"
        );
    }
}
