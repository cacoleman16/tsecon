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

use serde_json::Value;
use tsecon_ident::histdecomp::decompose;
use tsecon_ident::IdentError;
use tsecon_linalg::faer::{Mat, MatRef};

const RTOL: f64 = 1e-8;
const ATOL: f64 = 1e-10;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/historical_decomposition_chol.json",
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
