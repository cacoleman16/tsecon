//! JSZ (Joslin-Singleton-Zhu 2011) golden tests against `fixtures/jsz.json`.
//!
//! The fixture is produced by `fixtures/generate_jsz_fixtures.py`, which never
//! calls tsecon. Its blocks and their honest grades:
//!
//! - `loadings`: documented-formula golden of the Riccati recursions at stated
//!   parameters (distinct eigenvalues, the AFNS Jordan block, N = 2, N = 4
//!   with an interior tie) — pinned at 1e-12.
//! - `afns`: the AFNS special case `lambda^Q = (1, rho, rho)`: the JSZ yield
//!   loadings span the Nelson-Siegel loadings exactly (a stored rotation), and
//!   the discrete convexity intercept converges at first order in the period
//!   length to the Christensen-Diebold-Rudebusch (2011) closed form — checked
//!   here through the crate's own independent `afns_yield_adjustment`.
//! - `sim`: a simulated canonical model. The portfolio VAR(1) is pinned to
//!   statsmodels `VAR(1)` (independent package, 1e-9); the likelihood at two
//!   stated points to the documented formula (1e-7 absolute on a value of
//!   order 3e4, i.e. ~1e-11 relative); the SciPy maximum-likelihood estimate
//!   is a cross-optimizer target (two optimizers on one likelihood agree to
//!   their stopping tolerances); recovery of the truth is asserted at the
//!   tolerances the generator measured.
//! - `gsw`: the real GSW panel, 1990-2007 (JSZ's window): PCA weights pinned
//!   to NumPy, the VAR(1) to statsmodels, the MLE to SciPy (cross-optimizer),
//!   and the illustration numbers quoted in the docs reproduced.

use serde_json::Value;
use tsecon_termstructure::{afns_yield_adjustment, fit_jsz, jsz_loadings, jsz_loglik, JszFit};

/// Starts used throughout (the binding's default).
const STARTS: usize = 5;

type Matrix = Vec<Vec<f64>>;

fn load() -> Value {
    let path = format!("{}/../../fixtures/jsz.json", env!("CARGO_MANIFEST_DIR"));
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

fn f64_matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("integer") as usize)
        .collect()
}

fn num(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn assert_close(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > atol {atol:e}"
    );
}

fn assert_rel(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    let tol = rtol * expected.abs().max(1e-300);
    assert!(
        err <= tol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > rtol {rtol:e} * |expected|"
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64], atol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, atol, &format!("{ctx}[{i}]"));
    }
}

fn assert_vec_rel(actual: &[f64], expected: &[f64], rtol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_rel(a, e, rtol, &format!("{ctx}[{i}]"));
    }
}

fn assert_mat_close(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: rows");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_vec_close(a, e, atol, &format!("{ctx}[{i}]"));
    }
}

fn assert_mat_rel(actual: &[Vec<f64>], expected: &[Vec<f64>], rtol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: rows");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_vec_rel(a, e, rtol, &format!("{ctx}[{i}]"));
    }
}

fn max_abs(m: &[Vec<f64>]) -> f64 {
    m.iter()
        .flat_map(|r| r.iter().map(|v| v.abs()))
        .fold(0.0, f64::max)
}

fn max_abs_diff_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(r, s)| r.iter().zip(s).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (r, inner, c) = (a.len(), b.len(), b[0].len());
    let mut out = vec![vec![0.0; c]; r];
    for i in 0..r {
        for k in 0..inner {
            for j in 0..c {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (r, c) = (a.len(), a[0].len());
    (0..c).map(|j| (0..r).map(|i| a[i][j]).collect()).collect()
}

/// statsmodels `VAR(1)` layout: `params[0]` is the constant row, `params[1 +
/// j]` the lag-1 coefficients of regressor `j`, columns are equations.
fn check_var(fit: &JszFit, var: &Value, rtol: f64, ctx: &str) {
    let params = f64_matrix(&var["params"]);
    let stderr = f64_matrix(&var["stderr"]);
    let n = fit.n_factors;
    assert_vec_rel(&fit.mu_p, &params[0], rtol, &format!("{ctx} mu_p"));
    assert_vec_rel(&fit.mu_p_se, &stderr[0], rtol, &format!("{ctx} mu_p_se"));
    for i in 0..n {
        for j in 0..n {
            assert_rel(
                fit.phi_p[i][j],
                params[1 + j][i],
                rtol,
                &format!("{ctx} phi_p[{i}][{j}]"),
            );
            assert_rel(
                fit.phi_p_se[i][j],
                stderr[1 + j][i],
                rtol,
                &format!("{ctx} phi_p_se[{i}][{j}]"),
            );
        }
    }
    assert_mat_rel(
        &fit.sigma_ols,
        &f64_matrix(&var["sigma_u_mle"]),
        rtol,
        &format!("{ctx} sigma_ols"),
    );
}

// ---------------------------------------------------------------------------
// Loadings
// ---------------------------------------------------------------------------

#[test]
fn jsz_loadings_match_the_documented_recursions() {
    let fx = load();
    let cases = fx["loadings"].as_array().expect("cases");
    assert_eq!(cases.len(), 4);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let lambda_q = f64s(&case["lambda_q"]);
        let k_inf = num(&case["k_inf_q"]);
        let sigma_x = f64_matrix(&case["sigma_x"]);
        let mats = usizes(&case["maturities"]);
        let ppy = num(&case["periods_per_year"]);
        let l = jsz_loadings(&lambda_q, k_inf, &sigma_x, &mats, ppy).expect("loadings");
        assert_vec_close(&l.a, &f64s(&case["a"]), 1e-12, &format!("{name} a"));
        assert_mat_close(&l.b, &f64_matrix(&case["b"]), 1e-12, &format!("{name} b"));
        assert_vec_close(&l.k0_q, &f64s(&case["k0_q"]), 0.0, &format!("{name} k0_q"));
        assert_mat_close(
            &l.k1_q,
            &f64_matrix(&case["k1_q"]),
            0.0,
            &format!("{name} k1_q"),
        );
        assert_eq!(l.maturities, mats);
    }
}

// ---------------------------------------------------------------------------
// The AFNS special case
// ---------------------------------------------------------------------------

#[test]
fn jsz_afns_case_spans_nelson_siegel_and_converges_to_the_cdr_adjustment() {
    let fx = load();
    let afns = &fx["afns"];
    let lam_annual = num(&afns["lambda_annual"]);
    let sig = f64s(&afns["sigma_annual"]);
    let tau = f64s(&afns["tau_years"]);
    let ns = f64_matrix(&afns["ns_loadings"]);
    // The CDR closed form through the crate's own independent implementation.
    let cdr = afns_yield_adjustment(&tau, lam_annual, [sig[0], sig[1], sig[2]]).expect("cdr");
    let grids = afns["grids"].as_array().expect("grids");
    let mut gaps = Vec::new();
    for grid in grids {
        let ppy = num(&grid["periods_per_year"]);
        let rho = num(&grid["rho"]);
        let mats = usizes(&grid["maturities"]);
        let zero = vec![vec![0.0; 3]; 3];
        let lambda_q = [1.0, rho, rho];
        // (i) The loadings span the Nelson-Siegel loadings exactly: b C = NS.
        let l0 = jsz_loadings(&lambda_q, 0.0, &zero, &mats, ppy).expect("loadings");
        assert_eq!(l0.k1_q[1][2], 1.0, "Jordan block at ppy {ppy}");
        let c = f64_matrix(&grid["rotation_c"]);
        let bc = mat_mul(&l0.b, &c);
        let resid = max_abs_diff_mat(&bc, &ns);
        assert!(resid < 1e-10, "ppy {ppy}: NS span residual {resid:e}");
        // (ii) The intercept: pinned to the generator, and its gap to the CDR
        // closed form is the stored (first-order) discretization error.
        let sigma_x = f64_matrix(&grid["sigma_x"]);
        let l = jsz_loadings(&lambda_q, 0.0, &sigma_x, &mats, ppy).expect("loadings");
        assert_vec_close(
            &l.a,
            &f64s(&grid["a_jsz"]),
            1e-12,
            &format!("ppy {ppy} a_jsz"),
        );
        assert_vec_close(
            &cdr,
            &f64s(&grid["a_cdr"]),
            1e-12,
            &format!("ppy {ppy} a_cdr"),
        );
        let gap =
            l.a.iter()
                .zip(&cdr)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
        let stored = num(&grid["max_abs_gap"]);
        assert!(
            gap <= stored * (1.0 + 1e-6),
            "ppy {ppy}: gap {gap:e} > stored {stored:e}"
        );
        // The adjustment is negative and deepens with maturity on both sides.
        assert!(l.a.iter().all(|v| *v < 0.0));
        assert!(l.a[l.a.len() - 1] < l.a[0]);
        gaps.push(gap);
    }
    // First-order convergence: quartering the period length quarters the gap.
    for w in gaps.windows(2) {
        let ratio = w[1] / w[0];
        assert!(
            (0.2..0.32).contains(&ratio),
            "convergence ratio {ratio} not ~0.25"
        );
    }
    // At monthly resolution the gap is below 1bp; at 1/192 below 0.01bp.
    assert!(gaps[0] < 1e-4 && gaps[2] < 1e-6, "gaps {gaps:?}");
}

// ---------------------------------------------------------------------------
// The simulated canonical model
// ---------------------------------------------------------------------------

fn sim_inputs(fx: &Value) -> (Matrix, Vec<usize>, f64, Matrix) {
    let sim = &fx["sim"];
    (
        f64_matrix(&sim["yields"]),
        usizes(&sim["maturities"]),
        num(&sim["periods_per_year"]),
        f64_matrix(&sim["truth"]["w"]),
    )
}

#[test]
fn jsz_sim_portfolio_var_matches_statsmodels() {
    let fx = load();
    let (y, mats, ppy, w) = sim_inputs(&fx);
    let fit = fit_jsz(&y, &mats, 3, ppy, Some(&w), STARTS, 0).expect("fit");
    check_var(&fit, &fx["sim"]["statsmodels_var"], 1e-9, "sim");
    assert_eq!(
        fx["sim"]["statsmodels_var"]["nobs"].as_u64().expect("nobs"),
        y.len() as u64 - 1
    );
}

#[test]
fn jsz_sim_loglik_matches_the_documented_formula() {
    let fx = load();
    let (y, mats, ppy, w) = sim_inputs(&fx);
    let truth = &fx["sim"]["truth"];
    let ll = &fx["sim"]["loglik"];
    let lambda_q = f64s(&truth["lambda_q"]);
    let k_inf = num(&truth["k_inf_q"]);
    let sigma = f64_matrix(&truth["sigma"]);
    let sigma_e = num(&truth["sigma_e"]);
    let at_truth = jsz_loglik(
        &y,
        &mats,
        3,
        ppy,
        Some(&w),
        &lambda_q,
        k_inf,
        &sigma,
        sigma_e,
    )
    .expect("loglik");
    let expected = num(&ll["at_truth"]);
    assert_close(at_truth, expected, 1e-7, "llf at truth");
    // The same likelihood in a different basis of the portfolio space.
    let g = f64_matrix(&ll["rotated"]["g"]);
    let w2 = mat_mul(&g, &w);
    let sigma2 = mat_mul(&mat_mul(&g, &sigma), &transpose(&g));
    let rotated = jsz_loglik(
        &y,
        &mats,
        3,
        ppy,
        Some(&w2),
        &lambda_q,
        k_inf,
        &sigma2,
        sigma_e,
    )
    .expect("loglik");
    assert_close(
        rotated,
        num(&ll["rotated"]["llf"]),
        1e-7,
        "llf in a rotated basis",
    );
    assert_close(rotated, at_truth, 1e-7, "invariance of the likelihood");
    // A second stated point.
    let p2 = &ll["point2"];
    let llf2 = jsz_loglik(
        &y,
        &mats,
        3,
        ppy,
        Some(&w),
        &f64s(&p2["lambda_q"]),
        num(&p2["k_inf_q"]),
        &f64_matrix(&p2["sigma"]),
        num(&p2["sigma_e"]),
    )
    .expect("loglik");
    assert_close(llf2, num(&p2["llf"]), 1e-7, "llf at point 2");
    assert!(llf2 < at_truth, "the arbitrary point beats the truth?");
}

#[test]
fn jsz_sim_mle_agrees_with_scipy_and_recovers_the_truth() {
    let fx = load();
    let (y, mats, ppy, w) = sim_inputs(&fx);
    let sim = &fx["sim"];
    let truth = &sim["truth"];
    let mle = &sim["mle"];
    let fit = fit_jsz(&y, &mats, 3, ppy, Some(&w), STARTS, 0).expect("fit");
    assert!(
        fit.converged,
        "not converged after {} iterations",
        fit.n_iter
    );
    // Cross-optimizer: the Rust optimum is at least as good as SciPy's and
    // lands at the same point to the optimizers' stopping tolerances.
    let scipy_llf = num(&mle["llf"]);
    assert!(
        fit.llf >= scipy_llf - 1e-4,
        "Rust llf {} below SciPy's {}",
        fit.llf,
        scipy_llf
    );
    assert!(
        (fit.llf - scipy_llf).abs() < 1e-2,
        "llf gap {}",
        fit.llf - scipy_llf
    );
    assert_vec_close(
        &fit.lambda_q,
        &f64s(&mle["lambda_q"]),
        1e-5,
        "lambda_q vs SciPy",
    );
    assert_rel(fit.k_inf_q, num(&mle["k_inf_q"]), 1e-3, "k_inf_q vs SciPy");
    assert_rel(fit.sigma_e, num(&mle["sigma_e"]), 1e-4, "sigma_e vs SciPy");
    let sig_scipy = f64_matrix(&mle["sigma"]);
    let dsig = max_abs_diff_mat(&fit.sigma, &sig_scipy) / max_abs(&sig_scipy);
    assert!(dsig < 1e-3, "sigma vs SciPy: relative {dsig:e}");
    // Recovery of the truth, at the tolerances the generator measured.
    let lam_true = f64s(&truth["lambda_q"]);
    let rec = &sim["recovery"];
    let dl = fit
        .lambda_q
        .iter()
        .zip(&lam_true)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(dl < 2e-4, "lambda_q recovery {dl:e}");
    assert!(dl <= 2.0 * num(&rec["lambda_max_abs_err"]) + 1e-6);
    let dk = (fit.k_inf_q - num(&truth["k_inf_q"])).abs() / num(&truth["k_inf_q"]);
    assert!(dk < 0.03, "k_inf_q recovery {dk:e}");
    let ds = (fit.sigma_e - num(&truth["sigma_e"])).abs() / num(&truth["sigma_e"]);
    assert!(ds < 0.02, "sigma_e recovery {ds:e}");
    let sig_true = f64_matrix(&truth["sigma"]);
    let dsig = max_abs_diff_mat(&fit.sigma, &sig_true) / max_abs(&sig_true);
    assert!(dsig < 0.2, "sigma recovery {dsig:e}");
    // The rotated loadings agree with the DGP's.
    let db = max_abs_diff_mat(&fit.b_p, &f64_matrix(&truth["b_p"]));
    assert!(db < 2e-3, "b_p recovery {db:e}");
    let a_true = f64s(&truth["a_p"]);
    let da = fit
        .a_p
        .iter()
        .zip(&a_true)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(da < 1e-4, "a_p recovery {da:e}");
    // Self-consistency: jsz_loglik at the fit's parameters is fit.llf.
    let again = jsz_loglik(
        &y,
        &mats,
        3,
        ppy,
        Some(&w),
        &fit.lambda_q,
        fit.k_inf_q,
        &fit.sigma,
        fit.sigma_e,
    )
    .expect("loglik");
    assert_close(again, fit.llf, 1e-7, "jsz_loglik at the optimum");
}

// ---------------------------------------------------------------------------
// The real GSW panel (1990-2007)
// ---------------------------------------------------------------------------

#[test]
fn jsz_gsw_pca_weights_and_var_match_numpy_and_statsmodels() {
    let fx = load();
    let gsw = &fx["gsw"];
    let y = f64_matrix(&gsw["yields"]);
    let mats = usizes(&gsw["maturities"]);
    assert_eq!(y.len(), 216);
    let fit = fit_jsz(&y, &mats, 3, 12.0, None, STARTS, 0).expect("fit");
    assert_mat_close(&fit.w, &f64_matrix(&gsw["w"]), 1e-10, "gsw PCA weights");
    check_var(&fit, &gsw["statsmodels_var"], 1e-9, "gsw");
}

#[test]
fn jsz_gsw_mle_agrees_with_scipy_and_reproduces_the_illustration() {
    let fx = load();
    let gsw = &fx["gsw"];
    let y = f64_matrix(&gsw["yields"]);
    let mats = usizes(&gsw["maturities"]);
    let mle = &gsw["mle"];
    let ill = &gsw["illustration"];
    let fit = fit_jsz(&y, &mats, 3, 12.0, None, STARTS, 0).expect("fit");
    assert!(
        fit.converged,
        "not converged after {} iterations",
        fit.n_iter
    );
    let scipy_llf = num(&mle["llf"]);
    assert!(
        fit.llf >= scipy_llf - 1e-4,
        "Rust llf {} below SciPy's {}",
        fit.llf,
        scipy_llf
    );
    assert!(
        (fit.llf - scipy_llf).abs() < 1e-2,
        "llf gap {}",
        fit.llf - scipy_llf
    );
    assert_vec_close(
        &fit.lambda_q,
        &f64s(&mle["lambda_q"]),
        1e-5,
        "gsw lambda_q vs SciPy",
    );
    assert_rel(
        fit.k_inf_q,
        num(&mle["k_inf_q"]),
        1e-3,
        "gsw k_inf_q vs SciPy",
    );
    assert_rel(
        fit.sigma_e,
        num(&mle["sigma_e"]),
        1e-4,
        "gsw sigma_e vs SciPy",
    );
    // Fitted rows and the illustration numbers quoted in the docs.
    assert_vec_close(
        &fit.fitted[0],
        &f64s(&gsw["fitted_row0"]),
        1e-7,
        "gsw fitted row 0",
    );
    assert_vec_close(
        &fit.fitted[fit.fitted.len() - 1],
        &f64s(&gsw["fitted_row_last"]),
        1e-7,
        "gsw fitted last row",
    );
    assert_close(
        fit.sigma_e * 1e4,
        num(&ill["sigma_e_bp"]),
        1e-3,
        "sigma_e in bp",
    );
    let rmse_bp: Vec<f64> = fit.rmse.iter().map(|r| r * 1e4).collect();
    assert_vec_close(&rmse_bp, &f64s(&ill["rmse_bp"]), 1e-3, "rmse in bp");
    let j10 = mats.iter().position(|&m| m == 120).expect("10y");
    let tp10: Vec<f64> = fit.term_premium.iter().map(|r| r[j10]).collect();
    assert_vec_close(
        &tp10,
        &f64s(&gsw["term_premium_120"]),
        1e-7,
        "10y term premium path",
    );
    let mean_pp = tp10.iter().sum::<f64>() / tp10.len() as f64 * 100.0;
    assert_close(
        mean_pp,
        num(&ill["tp10_mean_pp"]),
        1e-4,
        "mean 10y premium (pp)",
    );
    let half_life = (0.5f64).ln() / fit.lambda_q[0].ln() / 12.0;
    assert_close(
        half_life,
        num(&ill["q_half_life_years"]),
        1e-3,
        "Q half-life (years)",
    );
    // Structural sanity on real data: an ordered, sub-unit-root Q spectrum,
    // basis-point pricing errors, and a fit that beats every single-maturity
    // RMSE of 5bp.
    assert!(fit.lambda_q[0] > fit.lambda_q[1] && fit.lambda_q[1] > fit.lambda_q[2]);
    assert!(fit.lambda_q[0] < 1.0 && fit.lambda_q[0] > 0.99);
    assert!(fit.rmse.iter().all(|r| *r < 5e-4));
}
