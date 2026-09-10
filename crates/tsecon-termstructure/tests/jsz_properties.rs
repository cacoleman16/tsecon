//! Property tests for the JSZ canonical affine term-structure fit: recovery
//! on a simulated canonical model, invariance to the basis of the portfolio
//! space (the whole point of the JSZ normalization), exact pricing of the
//! portfolios, determinism, the self-checking local-optimality of the
//! maximized likelihood, the unit-root level factor, the Jordan-block
//! loadings, and teaching errors on invalid inputs.
//!
//! The DGP is simulated HERE (xorshift + Box-Muller — no fixture, no NumPy)
//! in the portfolio rotation: `P_t` follows a VAR(1), yields are the exact
//! affine prices `a_p + b_p P_t` built from the public [`jsz_loadings`]
//! recursions and the rotation algebra transcribed independently below,
//! plus an iid error projected onto the complement of the portfolio space
//! (so the portfolios are priced without error, exactly as the model
//! assumes).

use tsecon_termstructure::{fit_jsz, jsz_loadings, jsz_loglik, JszFit, TermStructureError};

// ---------------------------------------------------------------------------
// Tiny linear algebra for the DGP.
// ---------------------------------------------------------------------------

type Matrix = Vec<Vec<f64>>;

/// Starts used throughout (the binding's default).
const STARTS: usize = 5;

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
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

fn transpose(a: &[Vec<f64>]) -> Matrix {
    let (r, c) = (a.len(), a[0].len());
    (0..c).map(|j| (0..r).map(|i| a[i][j]).collect()).collect()
}

fn mat_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v).map(|(x, y)| x * y).sum())
        .collect()
}

fn invert(a: &[Vec<f64>]) -> Matrix {
    let n = a.len();
    let mut w: Matrix = a.to_vec();
    let mut inv: Matrix = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if w[r][col].abs() > w[piv][col].abs() {
                piv = r;
            }
        }
        w.swap(col, piv);
        inv.swap(col, piv);
        let p = w[col][col];
        assert!(p.abs() > 1e-14, "singular matrix in the test DGP");
        for j in 0..n {
            w[col][j] /= p;
            inv[col][j] /= p;
        }
        for r in 0..n {
            if r != col {
                let f = w[r][col];
                for j in 0..n {
                    w[r][j] -= f * w[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    inv
}

/// Orthonormal portfolio weights: level / ramp / hump, Gram-Schmidt.
fn orthonormal_weights(mats: &[usize]) -> Matrix {
    let m = mats.len();
    let raw: Matrix = vec![
        vec![1.0; m],
        mats.iter().map(|&n| n as f64 / 120.0).collect(),
        mats.iter()
            .map(|&n| {
                let x = n as f64 / 24.0;
                x * (-x).exp()
            })
            .collect(),
    ];
    let mut q: Matrix = Vec::new();
    for v in raw {
        let mut u = v.clone();
        for prev in &q {
            let proj: f64 = prev.iter().zip(&v).map(|(a, b)| a * b).sum();
            for j in 0..m {
                u[j] -= proj * prev[j];
            }
        }
        let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        q.push(u.iter().map(|x| x / norm).collect());
    }
    q
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next();
        let u2 = self.next();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

struct Dgp {
    mats: Vec<usize>,
    ppy: f64,
    lambda: Vec<f64>,
    k_inf: f64,
    w: Matrix,
    mu: Vec<f64>,
    phi: Matrix,
    chol: Matrix,
    sigma_e: f64,
}

struct Truth {
    a_p: Vec<f64>,
    b_p: Matrix,
    sigma_p: Matrix,
}

/// The rotation algebra of the module docs, transcribed independently:
/// `D = W b_x`, `Sigma_X = D^{-1} (Sigma_P / ppy^2) D^{-T}`, `c = W a_x`,
/// `b_p = b_x D^{-1}`, `a_p = a_x - b_p c`.
fn truth(d: &Dgp) -> Truth {
    let n = d.lambda.len();
    let zero = vec![vec![0.0; n]; n];
    let l0 = jsz_loadings(&d.lambda, 0.0, &zero, &d.mats, d.ppy).expect("loadings");
    let dm = mat_mul(&d.w, &l0.b);
    let d_inv = invert(&dm);
    let sigma_p = mat_mul(&d.chol, &transpose(&d.chol));
    let sigma_pp: Matrix = sigma_p
        .iter()
        .map(|r| r.iter().map(|v| v / (d.ppy * d.ppy)).collect())
        .collect();
    let sigma_x = mat_mul(&mat_mul(&d_inv, &sigma_pp), &transpose(&d_inv));
    let lx = jsz_loadings(&d.lambda, d.k_inf, &sigma_x, &d.mats, d.ppy).expect("loadings");
    let c = mat_vec(&d.w, &lx.a);
    let b_p = mat_mul(&lx.b, &d_inv);
    let bc = mat_vec(&b_p, &c);
    let a_p: Vec<f64> = lx.a.iter().zip(&bc).map(|(a, b)| a - b).collect();
    Truth { a_p, b_p, sigma_p }
}

fn simulate(d: &Dgp, t_len: usize, seed: u64) -> (Matrix, Truth) {
    let tr = truth(d);
    let n = d.lambda.len();
    let m = d.mats.len();
    let mut rng = Rng(seed);
    // Stationary start: mean of the P-VAR.
    let eye_minus_phi: Matrix = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { 1.0 } else { 0.0 } - d.phi[i][j])
                .collect()
        })
        .collect();
    let mut p = mat_vec(&invert(&eye_minus_phi), &d.mu);
    let step = |p: &[f64], rng: &mut Rng| -> Vec<f64> {
        let eps: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let shock = mat_vec(&d.chol, &eps);
        let drift = mat_vec(&d.phi, p);
        (0..n).map(|i| d.mu[i] + drift[i] + shock[i]).collect()
    };
    for _ in 0..300 {
        p = step(&p, &mut rng);
    }
    let mut yields = vec![vec![0.0; m]; t_len];
    for row in yields.iter_mut() {
        p = step(&p, &mut rng);
        let exact = mat_vec(&tr.b_p, &p);
        // Error projected onto the complement of the (orthonormal) W rows.
        let e: Vec<f64> = (0..m).map(|_| d.sigma_e * rng.normal()).collect();
        let we = mat_vec(&d.w, &e);
        let wte = mat_vec(&transpose(&d.w), &we);
        for j in 0..m {
            row[j] = tr.a_p[j] + exact[j] + e[j] - wte[j];
        }
    }
    (yields, tr)
}

fn baseline_dgp() -> Dgp {
    let mats = vec![1usize, 3, 6, 12, 24, 36, 60, 84, 120];
    let w = orthonormal_weights(&mats);
    Dgp {
        mats,
        ppy: 12.0,
        lambda: vec![0.995, 0.96, 0.85],
        k_inf: 2.0e-5,
        // P-dynamics of the (level-ish, slope-ish, curvature-ish) portfolios.
        mu: vec![0.0025, 0.0004, -0.0001],
        phi: vec![
            vec![0.980, 0.010, -0.010],
            vec![0.005, 0.930, 0.020],
            vec![0.000, -0.010, 0.850],
        ],
        chol: vec![
            vec![0.0022, 0.0, 0.0],
            vec![0.0006, 0.0012, 0.0],
            vec![-0.0002, 0.0003, 0.0008],
        ],
        sigma_e: 1.0e-5,
        w,
    }
}

fn max_abs_diff_mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(r, s)| r.iter().zip(s).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn max_abs_mat(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|r| r.iter().map(|x| x.abs()))
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

#[test]
fn jsz_recovers_a_simulated_canonical_model_with_the_true_portfolios() {
    let d = baseline_dgp();
    let (y, tr) = simulate(&d, 400, 20260910);
    let start = std::time::Instant::now();
    let fit = fit_jsz(&y, &d.mats, 3, d.ppy, Some(&d.w), STARTS, 0).expect("fit");
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "fit: {secs:.2}s, converged {}, iters {}, lambda {:?}, k_inf {:.3e}, sigma_e {:.3e}, llf {:.3}",
        fit.converged, fit.n_iter, fit.lambda_q, fit.k_inf_q, fit.sigma_e, fit.llf
    );
    assert!(fit.converged, "optimizer did not converge");
    assert!(fit.lambda_q.iter().all(|v| v.is_finite()));
    // The Q parameters are pinned by the cross-section: tight recovery.
    let dl = max_abs_diff(&fit.lambda_q, &d.lambda);
    assert!(dl < 2e-4, "lambda_q recovery: max abs error {dl:e}");
    let dk = (fit.k_inf_q - d.k_inf).abs() / d.k_inf;
    assert!(dk < 0.03, "k_inf_q recovery: relative error {dk:e}");
    let ds = (fit.sigma_e - d.sigma_e).abs() / d.sigma_e;
    assert!(ds < 0.06, "sigma_e recovery: relative error {ds:e}");
    // Sigma_P is a T-observation covariance: sampling-error accuracy.
    let dsig = max_abs_diff_mat(&fit.sigma, &tr.sigma_p) / max_abs_mat(&tr.sigma_p);
    assert!(dsig < 0.15, "sigma recovery: relative error {dsig:e}");
    // The rotated loadings agree with the DGP's (transcribed independently).
    let db = max_abs_diff_mat(&fit.b_p, &tr.b_p);
    let da = max_abs_diff(&fit.a_p, &tr.a_p);
    assert!(db < 2e-3, "b_p recovery: max abs error {db:e}");
    assert!(da < 5e-5, "a_p recovery: max abs error {da:e}");
    // The pricing error is what the DGP put in.
    for (j, r) in fit.rmse.iter().enumerate() {
        assert!(*r < 3.0 * d.sigma_e, "rmse[{j}] = {r:e} exceeds 3 sigma_e");
    }
}

#[test]
fn jsz_default_principal_component_portfolios_recover_the_model_too() {
    let d = baseline_dgp();
    let (y, _) = simulate(&d, 400, 7);
    let fit = fit_jsz(&y, &d.mats, 3, d.ppy, None, STARTS, 0).expect("fit");
    assert!(fit.converged);
    // The PCA rows are orthonormal and priced exactly.
    for i in 0..3 {
        for k in 0..3 {
            let dotp: f64 = (0..9).map(|j| fit.w[i][j] * fit.w[k][j]).sum();
            let expect = if i == k { 1.0 } else { 0.0 };
            assert!((dotp - expect).abs() < 1e-12);
        }
    }
    let dl = max_abs_diff(&fit.lambda_q, &d.lambda);
    assert!(dl < 5e-4, "lambda_q recovery with PCA portfolios: {dl:e}");
    let dk = (fit.k_inf_q - d.k_inf).abs() / d.k_inf;
    assert!(dk < 0.05, "k_inf_q recovery with PCA portfolios: {dk:e}");
}

// ---------------------------------------------------------------------------
// Invariance, exact pricing, determinism, self-checking optimum
// ---------------------------------------------------------------------------

#[test]
fn jsz_is_invariant_to_the_basis_of_the_portfolio_space() {
    let d = baseline_dgp();
    let (y, _) = simulate(&d, 300, 11);
    let g: Matrix = vec![
        vec![2.0, 0.5, -0.3],
        vec![0.1, -1.5, 0.2],
        vec![0.4, 0.3, 3.0],
    ];
    let w2 = mat_mul(&g, &d.w);
    let f1 = fit_jsz(&y, &d.mats, 3, d.ppy, Some(&d.w), STARTS, 0).expect("fit 1");
    let f2 = fit_jsz(&y, &d.mats, 3, d.ppy, Some(&w2), STARTS, 0).expect("fit 2");
    assert!(f1.converged && f2.converged);
    let tol = 1e-8;
    assert!(
        (f1.llf - f2.llf).abs() < tol * f1.llf.abs().max(1.0),
        "llf differs across bases: {} vs {}",
        f1.llf,
        f2.llf
    );
    assert!(max_abs_diff(&f1.lambda_q, &f2.lambda_q) < tol);
    // k_inf_q ~ 2e-5: within 1e-8 absolutely and 1e-7 relatively (its
    // profile value inherits the Sigma_P flatness; measured 6e-9 relative).
    assert!((f1.k_inf_q - f2.k_inf_q).abs() < tol);
    assert!((f1.k_inf_q - f2.k_inf_q).abs() < 1e-7 * f1.k_inf_q.abs());
    assert!((f1.sigma_e - f2.sigma_e).abs() < tol * f1.sigma_e);
    assert!(max_abs_diff_mat(&f1.fitted, &f2.fitted) < tol);
    assert!(max_abs_diff_mat(&f1.risk_neutral, &f2.risk_neutral) < tol);
    assert!(max_abs_diff_mat(&f1.term_premium, &f2.term_premium) < tol);
    assert!(max_abs_diff(&f1.a_x, &f2.a_x) < tol);
    assert!(max_abs_diff_mat(&f1.b_x, &f2.b_x) < tol);
    // The rotation-dependent objects transform exactly: P2 = G P1,
    // Sigma2 = G Sigma1 G', b_p2 = b_p1 G^{-1}, mu2 = G mu1, phi2 = G phi1 G^{-1}.
    let p2_from_p1 = mat_mul(&f1.factors, &transpose(&g));
    assert!(max_abs_diff_mat(&f2.factors, &p2_from_p1) < 1e-12);
    let s2 = mat_mul(&mat_mul(&g, &f1.sigma), &transpose(&g));
    // Sigma_P is the flattest direction of the likelihood: a 1e-11 change in
    // llf moves it by ~3e-7 relative at T = 300, so the two optimizer runs
    // agree on it only to the optimizer's tolerance, not to 1e-8. Measured:
    // 2.1e-7 relative (llf differs by 1.1e-11, lambda_q by 1.6e-10).
    assert!(max_abs_diff_mat(&f2.sigma, &s2) < 1e-6 * max_abs_mat(&s2));
    let g_inv = invert(&g);
    let b2 = mat_mul(&f1.b_p, &g_inv);
    assert!(max_abs_diff_mat(&f2.b_p, &b2) < tol);
    assert!(max_abs_diff(&f2.a_p, &f1.a_p) < tol);
    let mu2 = mat_vec(&g, &f1.mu_p);
    assert!(max_abs_diff(&f2.mu_p, &mu2) < 1e-10);
    let phi2 = mat_mul(&mat_mul(&g, &f1.phi_p), &g_inv);
    assert!(max_abs_diff_mat(&f2.phi_p, &phi2) < 1e-10);
    let k1_2 = mat_mul(&mat_mul(&g, &f1.k1_q_p), &g_inv);
    assert!(max_abs_diff_mat(&f2.k1_q_p, &k1_2) < 1e-8);
    let k0_2 = mat_vec(&g, &f1.k0_q_p);
    assert!(max_abs_diff(&f2.k0_q_p, &k0_2) < 1e-10);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn jsz_prices_the_portfolios_exactly_and_the_decomposition_is_exact() {
    let d = baseline_dgp();
    let (y, _) = simulate(&d, 200, 3);
    for w in [None, Some(d.w.as_slice())] {
        let fit = fit_jsz(&y, &d.mats, 3, d.ppy, w, STARTS, 0).expect("fit");
        let scale = y
            .iter()
            .flat_map(|r| r.iter().map(|v| v.abs()))
            .fold(0.0, f64::max);
        for t in 0..y.len() {
            let wy = mat_vec(&fit.w, &y[t]);
            let wf = mat_vec(&fit.w, &fit.fitted[t]);
            assert!(max_abs_diff(&wy, &wf) < 1e-12 * scale.max(1.0));
            for j in 0..d.mats.len() {
                let s = fit.risk_neutral[t][j] + fit.term_premium[t][j];
                assert!((fit.fitted[t][j] - s).abs() < 1e-14);
            }
        }
        // W b_p = I and W a_p = 0.
        let wb = mat_mul(&fit.w, &fit.b_p);
        for i in 0..3 {
            for k in 0..3 {
                let e = if i == k { 1.0 } else { 0.0 };
                assert!(
                    (wb[i][k] - e).abs() < 1e-10,
                    "W b_p != I at ({i},{k}): {}",
                    wb[i][k]
                );
            }
        }
        let wa = mat_vec(&fit.w, &fit.a_p);
        assert!(wa.iter().all(|v| v.abs() < 1e-12));
        // The market prices of risk are the P-Q gap.
        for i in 0..3 {
            assert!((fit.lambda0[i] - (fit.mu_p[i] - fit.k0_q_p[i])).abs() < 1e-15);
            for j in 0..3 {
                assert!((fit.lambda1[i][j] - (fit.phi_p[i][j] - fit.k1_q_p[i][j])).abs() < 1e-15);
            }
        }
    }
}

#[test]
fn jsz_is_deterministic() {
    let d = baseline_dgp();
    let (y, _) = simulate(&d, 150, 5);
    let a = fit_jsz(&y, &d.mats, 3, d.ppy, None, STARTS, 0).expect("fit");
    let b = fit_jsz(&y, &d.mats, 3, d.ppy, None, STARTS, 0).expect("fit");
    assert_eq!(a, b);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn jsz_maximized_likelihood_is_a_local_maximum_in_every_parameter() {
    // The self-checking property: `jsz_loglik` at the fit's own parameters
    // reproduces `llf`, and moving any Q parameter, sigma_e, or an entry of
    // Sigma_P away from the optimum lowers the likelihood.
    let d = baseline_dgp();
    let (y, _) = simulate(&d, 300, 17);
    let fit = fit_jsz(&y, &d.mats, 3, d.ppy, Some(&d.w), STARTS, 0).expect("fit");
    assert!(fit.converged);
    let at = |lam: &[f64], k: f64, sig: &[Vec<f64>], se: f64| -> f64 {
        jsz_loglik(&y, &d.mats, 3, d.ppy, Some(&d.w), lam, k, sig, se).expect("loglik")
    };
    let llf0 = at(&fit.lambda_q, fit.k_inf_q, &fit.sigma, fit.sigma_e);
    assert!(
        (llf0 - fit.llf).abs() < 1e-7 * fit.llf.abs(),
        "jsz_loglik at the optimum {llf0} != fit.llf {}",
        fit.llf
    );
    let mut checked = 0;
    for k in 0..3 {
        for sign in [-1.0, 1.0] {
            let mut lam = fit.lambda_q.clone();
            lam[k] += sign * 1e-4;
            if k > 0 && lam[k] > lam[k - 1] || k < 2 && lam[k] < lam[k + 1] {
                continue;
            }
            let l = at(&lam, fit.k_inf_q, &fit.sigma, fit.sigma_e);
            assert!(
                l < llf0,
                "moving lambda_q[{k}] by {sign:+}e-4 raised llf: {l} > {llf0}"
            );
            checked += 1;
        }
    }
    for sign in [-1.0, 1.0] {
        let l = at(
            &fit.lambda_q,
            fit.k_inf_q * (1.0 + sign * 0.02),
            &fit.sigma,
            fit.sigma_e,
        );
        assert!(l < llf0, "moving k_inf_q raised llf");
        let l = at(
            &fit.lambda_q,
            fit.k_inf_q,
            &fit.sigma,
            fit.sigma_e * (1.0 + sign * 0.02),
        );
        assert!(l < llf0, "moving sigma_e raised llf");
        for i in 0..3 {
            for j in 0..=i {
                let mut sig = fit.sigma.clone();
                let delta = sign * 0.02 * fit.sigma[i][j].abs().max(1e-3 * fit.sigma[i][i]);
                sig[i][j] += delta;
                if i != j {
                    sig[j][i] += delta;
                }
                let l = at(&fit.lambda_q, fit.k_inf_q, &sig, fit.sigma_e);
                assert!(
                    l < llf0,
                    "moving sigma[{i}][{j}] by {delta:e} raised llf: {l} > {llf0}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 12);
}

// ---------------------------------------------------------------------------
// The unit-root level factor and the Jordan block
// ---------------------------------------------------------------------------

#[test]
fn jsz_handles_a_unit_root_level_factor() {
    // At lambda_1 = 1 the loading recursion is evaluated by recurrence (no
    // 1/(1 - lambda) closed form), k_inf becomes the drift of a unit-root
    // level and the intercept grows linearly in maturity:
    // a_n(k_inf) - a_n(0) = ppy * k_inf * (n - 1) / 2.
    let mats = [1usize, 2, 12, 60, 120, 360];
    let zero = vec![vec![0.0; 3]; 3];
    let l1 = jsz_loadings(&[1.0, 0.9, 0.8], 3e-5, &zero, &mats, 12.0).expect("loadings");
    let l0 = jsz_loadings(&[1.0, 0.9, 0.8], 0.0, &zero, &mats, 12.0).expect("loadings");
    for (i, &n) in mats.iter().enumerate() {
        let expect = 12.0 * 3e-5 * (n as f64 - 1.0) / 2.0;
        assert!(((l1.a[i] - l0.a[i]) - expect).abs() < 1e-12, "maturity {n}");
        assert!(
            (l1.b[i][0] - 1.0).abs() < 1e-12,
            "unit-root level loading is flat"
        );
    }
    // A fit on a unit-root-under-Q DGP does not break and lands near 1.
    let mut d = baseline_dgp();
    d.lambda = vec![1.0, 0.95, 0.85];
    let (y, _) = simulate(&d, 300, 23);
    let fit = fit_jsz(&y, &d.mats, 3, d.ppy, Some(&d.w), STARTS, 0).expect("fit");
    assert!(fit.lambda_q.iter().all(|v| v.is_finite()));
    assert!(
        (fit.lambda_q[0] - 1.0).abs() < 1e-3,
        "unit-root level: lambda_1 = {}",
        fit.lambda_q[0]
    );
    assert!((fit.lambda_q[1] - 0.95).abs() < 1e-3);
    assert!(fit.k_inf_q.is_finite() && (fit.k_inf_q - d.k_inf).abs() < 0.05 * d.k_inf);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn jsz_loadings_jordan_block_matches_the_closed_form_derivative() {
    // K1 = diag(1) (+) [[rho, 1], [0, rho]]: (iota' K1^j)_3 = j rho^{j-1} + rho^j,
    // so b[n][2] = (1/n) sum_{j<n} (j rho^{j-1} + rho^j), and
    // b[n][1] = (1 - rho^n) / (n (1 - rho)).
    let rho = (-0.0609f64).exp();
    let mats = [1usize, 3, 12, 60, 120];
    let zero = vec![vec![0.0; 3]; 3];
    let l = jsz_loadings(&[1.0, rho, rho], 0.0, &zero, &mats, 1.0).expect("loadings");
    assert_eq!(l.k1_q[1][2], 1.0);
    assert_eq!(l.k1_q[0][1], 0.0);
    for (i, &n) in mats.iter().enumerate() {
        let nf = n as f64;
        let slope = (1.0 - rho.powi(n as i32)) / (nf * (1.0 - rho));
        let curv: f64 = (0..n)
            .map(|j| j as f64 * rho.powi(j as i32 - 1) + rho.powi(j as i32))
            .sum::<f64>()
            / nf;
        assert!((l.b[i][0] - 1.0).abs() < 1e-13);
        assert!((l.b[i][1] - slope).abs() < 1e-12, "slope at n={n}");
        assert!((l.b[i][2] - curv).abs() < 1e-12, "curvature at n={n}");
    }
    // Distinct eigenvalues: b[n][k] = (1/n) sum_{j<n} lambda_k^j (diagonal form).
    let lam = [0.995, 0.9, 0.7];
    let ld = jsz_loadings(&lam, 0.0, &zero, &mats, 1.0).expect("loadings");
    assert_eq!(ld.k1_q[0][1], 0.0);
    assert_eq!(ld.k1_q[1][2], 0.0);
    for (i, &n) in mats.iter().enumerate() {
        for k in 0..3 {
            let expect: f64 = (0..n).map(|j| lam[k].powi(j as i32)).sum::<f64>() / n as f64;
            assert!((ld.b[i][k] - expect).abs() < 1e-12);
        }
    }
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

fn small_panel() -> (Matrix, Vec<usize>) {
    let d = baseline_dgp();
    let (y, _) = simulate(&d, 40, 99);
    (y, d.mats)
}

#[test]
fn jsz_rejects_invalid_inputs_with_teaching_errors() {
    let (y, mats) = small_panel();
    let m = mats.len();
    let fit = |y: &Matrix,
               mats: &[usize],
               n: usize,
               ppy: f64,
               w: Option<&[Vec<f64>]>,
               starts: usize|
     -> Result<JszFit, TermStructureError> { fit_jsz(y, mats, n, ppy, w, starts, 0) };
    // Maturities: empty, zero, unsorted, duplicated.
    assert!(matches!(
        fit(&y, &[], 3, 12.0, None, STARTS),
        Err(TermStructureError::EmptyMaturities)
    ));
    let mut zero = mats.clone();
    zero[0] = 0;
    assert!(matches!(
        fit(&y, &zero, 3, 12.0, None, STARTS),
        Err(TermStructureError::InvalidMaturity { index: 0, .. })
    ));
    let mut unsorted = mats.clone();
    unsorted.swap(2, 3);
    assert!(matches!(
        fit(&y, &unsorted, 3, 12.0, None, STARTS),
        Err(TermStructureError::MaturitiesNotAscending { index: 3 })
    ));
    let mut dup = mats.clone();
    dup[4] = dup[3];
    assert!(matches!(
        fit(&y, &dup, 3, 12.0, None, STARTS),
        Err(TermStructureError::MaturitiesNotAscending { index: 4 })
    ));
    // Factor count.
    assert!(matches!(
        fit(&y, &mats, 0, 12.0, None, STARTS),
        Err(TermStructureError::InvalidFactorCount { requested: 0, .. })
    ));
    let err = fit(&y, &mats, m, 12.0, None, STARTS).unwrap_err();
    assert!(
        matches!(err, TermStructureError::InvalidFactorCount { requested, max } if requested == m && max == m - 1)
    );
    assert!(err.to_string().contains("n_factors"));
    // periods_per_year.
    assert!(matches!(
        fit(&y, &mats, 3, 0.0, None, STARTS),
        Err(TermStructureError::InvalidPeriodsPerYear { .. })
    ));
    assert!(matches!(
        fit(&y, &mats, 3, f64::NAN, None, STARTS),
        Err(TermStructureError::InvalidPeriodsPerYear { .. })
    ));
    // Too short.
    let short: Matrix = y[..8].to_vec();
    let err = fit(&short, &mats, 3, 12.0, None, STARTS).unwrap_err();
    assert!(matches!(
        err,
        TermStructureError::PanelTooShort {
            dates: 8,
            needed: 9,
            ..
        }
    ));
    assert!(err.to_string().contains("needs at least 9"));
    // Ragged row and NaN.
    let mut ragged = y.clone();
    ragged[3].pop();
    assert!(matches!(
        fit(&ragged, &mats, 3, 12.0, None, STARTS),
        Err(TermStructureError::DimensionMismatch { .. })
    ));
    let mut nan = y.clone();
    nan[5][2] = f64::NAN;
    let err = fit(&nan, &mats, 3, 12.0, None, STARTS).unwrap_err();
    assert!(matches!(err, TermStructureError::NonFinite { index, .. } if index == 5 * m + 2));
    assert!(err.to_string().contains("non-finite"));
    // Portfolio weights: wrong rows, wrong columns, non-finite, rank-deficient.
    let d = baseline_dgp();
    let two_rows: Matrix = d.w[..2].to_vec();
    let err = fit(&y, &mats, 3, 12.0, Some(&two_rows), STARTS).unwrap_err();
    assert!(matches!(
        err,
        TermStructureError::InvalidWeights { rows: 2, .. }
    ));
    assert!(err.to_string().contains("n_factors x n_maturities"));
    let mut short_cols = d.w.clone();
    short_cols[1].pop();
    assert!(matches!(
        fit(&y, &mats, 3, 12.0, Some(&short_cols), STARTS),
        Err(TermStructureError::InvalidWeights { .. })
    ));
    let mut inf_w = d.w.clone();
    inf_w[0][0] = f64::INFINITY;
    assert!(matches!(
        fit(&y, &mats, 3, 12.0, Some(&inf_w), STARTS),
        Err(TermStructureError::InvalidWeights { .. })
    ));
    let mut dependent = d.w.clone();
    dependent[2] = d.w[0].iter().zip(&d.w[1]).map(|(a, b)| a + b).collect();
    let err = fit(&y, &mats, 3, 12.0, Some(&dependent), STARTS).unwrap_err();
    assert!(matches!(err, TermStructureError::InvalidWeights { .. }));
    assert!(err.to_string().contains("linearly dependent"));
    // Start count and maturity caps (allocation guards).
    let err = fit(&y, &mats, 3, 12.0, None, 0).unwrap_err();
    assert!(matches!(
        err,
        TermStructureError::InvalidStartCount { requested: 0, .. }
    ));
    assert!(err.to_string().contains("n_starts = 0"));
    assert!(matches!(
        fit(&y, &mats, 3, 12.0, None, 1_000_001),
        Err(TermStructureError::InvalidStartCount { .. })
    ));
    let mut huge = mats.clone();
    huge[m - 1] = 10_000_000_000;
    let err = fit(&y, &huge, 3, 12.0, None, 1).unwrap_err();
    assert!(matches!(
        err,
        TermStructureError::MaturityTooLarge {
            value: 10_000_000_000,
            ..
        }
    ));
    assert!(err.to_string().contains("exceeds the supported maximum"));
    assert!(matches!(
        jsz_loadings(&[0.9, 0.8, 0.7], 0.0, &vec![vec![0.0; 3]; 3], &huge, 1.0),
        Err(TermStructureError::MaturityTooLarge { .. })
    ));
    // Constant panel: no portfolio variation.
    let flat: Matrix = vec![vec![0.05; m]; 40];
    assert!(fit(&flat, &mats, 3, 12.0, None, STARTS).is_err());

    // jsz_loadings refusals.
    let zero3 = vec![vec![0.0; 3]; 3];
    let err = jsz_loadings(&[0.9, 0.95, 0.8], 0.0, &zero3, &mats, 1.0).unwrap_err();
    assert!(matches!(
        err,
        TermStructureError::QEigenvaluesNotOrdered { index: 1 }
    ));
    assert!(err.to_string().contains("descending"));
    assert!(matches!(
        jsz_loadings(&[0.9, f64::NAN, 0.8], 0.0, &zero3, &mats, 1.0),
        Err(TermStructureError::InvalidQEigenvalue { index: 1, .. })
    ));
    assert!(matches!(
        jsz_loadings(&[], 0.0, &zero3, &mats, 1.0),
        Err(TermStructureError::InvalidQEigenvalue { index: 0, .. })
    ));
    assert!(matches!(
        jsz_loadings(&[0.9, 0.8], 0.0, &zero3, &mats, 1.0),
        Err(TermStructureError::DimensionMismatch {
            expected: 2,
            got: 3,
            ..
        })
    ));
    assert!(matches!(
        jsz_loadings(&[0.9, 0.8, 0.7], f64::NAN, &zero3, &mats, 1.0),
        Err(TermStructureError::NonFinite { .. })
    ));
    assert!(matches!(
        jsz_loadings(&[0.9, 0.8, 0.7], 0.0, &zero3, &mats, -1.0),
        Err(TermStructureError::InvalidPeriodsPerYear { .. })
    ));
    // jsz_loglik refusals: sigma_e must be positive, sigma_p the right shape.
    assert!(matches!(
        jsz_loglik(
            &y,
            &mats,
            3,
            12.0,
            None,
            &[0.99, 0.9, 0.8],
            1e-5,
            &zero3,
            0.0
        ),
        Err(TermStructureError::NonFinite { .. })
    ));
    assert!(matches!(
        jsz_loglik(&y, &mats, 3, 12.0, None, &[0.99, 0.9], 1e-5, &zero3, 1e-4),
        Err(TermStructureError::DimensionMismatch { .. })
    ));
    // A zero sigma_p is not positive definite.
    assert!(matches!(
        jsz_loglik(
            &y,
            &mats,
            3,
            12.0,
            None,
            &[0.99, 0.9, 0.8],
            1e-5,
            &zero3,
            1e-4
        ),
        Err(TermStructureError::SingularDesign { .. })
    ));
}
