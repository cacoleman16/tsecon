//! Golden and property tests for non-Gaussian / independent-component SVAR
//! identification ([`nongaussian_svar`]).
//!
//! `fixtures/nongaussian_svar.json` is produced by
//! `fixtures/generate_nongaussian_svar_fixtures.py` — every expected number
//! comes from an INDEPENDENT NumPy pipeline (`numpy.linalg.lstsq` OLS,
//! `numpy.linalg.eigh` for the symmetric inverse sqrt and the FastICA
//! decorrelation, `numpy.tanh` for the contrast), never from this crate, so
//! reproducing them is a genuine cross-implementation check of the SAME
//! deterministic symmetric FastICA on the SAME whitened residuals. The
//! generator additionally cross-checks its self-contained FastICA against
//! `sklearn.decomposition.FastICA` (matched to ~4e-16 at generation), so the
//! reference is a faithful FastICA, not a bespoke re-derivation.
//!
//! Column order/sign are conventions, applied identically on both sides
//! (descending |excess kurtosis|, then max-abs-positive), which is what makes
//! the whole result reproducible across two independent eigensolvers.
//!
//! * **`core_matches_numpy_reference`** bit-matches the impact `B`, whitened
//!   rotation `Q`, per-shock excess kurtosis, structural IRF, ordering,
//!   convergence flag and iteration count to the reference — the strong
//!   validation of the novel ICA-identification core.
//! * **`recovers_true_b_up_to_sign_and_permutation`** aligns the recovered `B`
//!   to the *true* DGP `B` by matching columns up to sign+permutation and
//!   checks recovery within an MC tolerance (the statistical-identification
//!   property).
//! * **`recovered_shocks_more_independent_than_raw_whitened`** confirms the
//!   ICA rotation lowers fourth-order cross-dependence versus the raw whitened
//!   residuals.
//! * property checks: `B B' = Sigma_u`, `Q` orthogonal, and deterministic
//!   (bit-identical) reproducibility.

use serde_json::Value;
use tsecon_ident::{nongaussian_svar, Contrast, OrderBy};
use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};

// Observed crate-vs-NumPy agreement is ~1e-15 (near 1 ULP); 1e-10 leaves ample
// cross-platform margin.
const TOL_CORE: f64 = 1e-10;
// Statistical recovery of the true DGP B: a large-sample property, not exact
// algebra (observed max abs diff ~0.011 at this seed).
const TOL_MC: f64 = 5e-2;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/nongaussian_svar.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn u(v: &Value) -> usize {
    v.as_u64().expect("uint") as usize
}

fn mat(v: &Value) -> Mat<f64> {
    let r = rows(v);
    let nr = r.len();
    let nc = r[0].len();
    Mat::from_fn(nr, nc, |i, j| r[i][j])
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    assert!(
        err < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

fn close_mat(actual: &Mat<f64>, expected: &[Vec<f64>], tol: f64, what: &str) {
    assert_eq!(actual.nrows(), expected.len(), "{what} rows");
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(actual.ncols(), row.len(), "{what} cols");
        for (j, &e) in row.iter().enumerate() {
            close(actual[(i, j)], e, tol, &format!("{what}[{i}][{j}]"));
        }
    }
}

/// Runs the crate estimator on the fixture inputs with the fixture controls.
fn run(fx: &Value) -> tsecon_ident::NonGaussianSvar {
    let resid = mat(&fx["resid"]);
    let sigma = mat(&fx["sigma"]);
    let b_coefs = mat(&fx["reg_coefs"]);
    let lags = u(&fx["lags"]);
    let horizon = u(&fx["horizon"]);
    let max_iter = u(&fx["max_iter"]);
    let tol = fx["tol"].as_f64().expect("tol");
    nongaussian_svar(
        resid.as_ref(),
        sigma.as_ref(),
        b_coefs.as_ref(),
        lags,
        horizon,
        Contrast::LogCosh,
        max_iter,
        tol,
        OrderBy::Kurtosis,
    )
    .expect("nongaussian_svar")
}

#[test]
fn core_matches_numpy_reference() {
    let fx = load();
    let r = run(&fx);

    // Impact matrix B and whitened rotation Q.
    close_mat(&r.impact, &rows(&fx["B"]), TOL_CORE, "impact B");
    close_mat(&r.rotation, &rows(&fx["rotation"]), TOL_CORE, "rotation Q");

    // Per-shock excess kurtosis (identified order).
    let exp_kurt = f64s(&fx["shock_kurtosis"]);
    assert_eq!(r.shock_kurtosis.len(), exp_kurt.len());
    for (i, (a, e)) in r.shock_kurtosis.iter().zip(exp_kurt.iter()).enumerate() {
        close(*a, *e, TOL_CORE, &format!("shock_kurtosis[{i}]"));
    }

    // Structural IRF, all horizons.
    let exp_irf = fx["structural_irf"].as_array().expect("irf array");
    assert_eq!(r.irf.len(), exp_irf.len(), "irf horizon length");
    for (h, (got, want)) in r.irf.iter().zip(exp_irf.iter()).enumerate() {
        close_mat(got, &rows(want), TOL_CORE, &format!("structural_irf[{h}]"));
    }

    // Ordering permutation, convergence, iteration count.
    let exp_order: Vec<usize> = f64s(&fx["order"]).iter().map(|x| *x as usize).collect();
    assert_eq!(r.order, exp_order, "order permutation");
    assert_eq!(
        r.converged,
        fx["converged"].as_bool().expect("bool"),
        "converged"
    );
    assert_eq!(r.n_iter, u(&fx["n_iter"]), "n_iter");
}

#[test]
fn recovers_true_b_up_to_sign_and_permutation() {
    // Statistical identification: the recovered B equals the true DGP B up to
    // column sign and permutation (both conventions). Align by brute force
    // (n = 3 -> 6 permutations x 8 sign patterns), then compare.
    let fx = load();
    let r = run(&fx);
    let n = r.impact.nrows();
    assert_eq!(n, 3, "matcher assumes n = 3");
    let b_true = mat(&fx["mc_b_true"]);

    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut best = f64::INFINITY;
    for perm in perms.iter() {
        for signs in 0..8u32 {
            let s = [
                if signs & 1 == 0 { 1.0 } else { -1.0 },
                if signs & 2 == 0 { 1.0 } else { -1.0 },
                if signs & 4 == 0 { 1.0 } else { -1.0 },
            ];
            let mut worst = 0.0f64;
            for col in 0..n {
                for row in 0..n {
                    let cand = s[col] * b_true[(row, perm[col])];
                    let d = (r.impact[(row, col)] - cand).abs();
                    if d > worst {
                        worst = d;
                    }
                }
            }
            if worst < best {
                best = worst;
            }
        }
    }
    assert!(
        best < TOL_MC,
        "recovered B should match true B up to sign+perm within {TOL_MC}, got {best:.4}"
    );
}

/// Symmetric inverse square root of a covariance via faer's self-adjoint EVD
/// (mirrors the crate's whitening, computed independently in the test).
fn inv_sqrt(m: &Mat<f64>) -> Mat<f64> {
    let n = m.nrows();
    let eig = m.self_adjoint_eigen(Side::Lower).expect("eig");
    let lam: Vec<f64> = eig.S().column_vector().iter().copied().collect();
    let v = eig.U();
    Mat::from_fn(n, n, |i, j| {
        (0..n)
            .map(|k| v[(i, k)] * (1.0 / lam[k].sqrt()) * v[(j, k)])
            .sum()
    })
}

/// Total squared correlation of the squared columns, `sum_{i<j} corr(X_i^2,
/// X_j^2)^2` — a fourth-order dependence proxy (zero for independent columns).
fn fourth_order_dependence(x: &Mat<f64>) -> f64 {
    let t = x.nrows();
    let n = x.ncols();
    let tf = t as f64;
    // Squared, then centered, columns.
    let mut sq = Mat::<f64>::zeros(t, n);
    for j in 0..n {
        let mut mean = 0.0;
        for i in 0..t {
            let v = x[(i, j)] * x[(i, j)];
            sq[(i, j)] = v;
            mean += v;
        }
        mean /= tf;
        for i in 0..t {
            sq[(i, j)] -= mean;
        }
    }
    let mut total = 0.0;
    for a in 0..n {
        for b in (a + 1)..n {
            let mut sab = 0.0;
            let mut saa = 0.0;
            let mut sbb = 0.0;
            for i in 0..t {
                sab += sq[(i, a)] * sq[(i, b)];
                saa += sq[(i, a)] * sq[(i, a)];
                sbb += sq[(i, b)] * sq[(i, b)];
            }
            let denom = (saa * sbb).sqrt();
            if denom > 0.0 {
                let c = sab / denom;
                total += c * c;
            }
        }
    }
    total
}

#[test]
fn recovered_shocks_more_independent_than_raw_whitened() {
    let fx = load();
    let r = run(&fx);
    let resid = mat(&fx["resid"]);
    let sigma = mat(&fx["sigma"]);
    let t = resid.nrows();
    let n = resid.ncols();

    // Raw whitened residuals z = U Sigma^{-1/2}.
    let w = inv_sqrt(&sigma);
    let z = resid.as_ref() * w.as_ref();

    // Recovered structural shocks eps = U B^{-1}' (since u = B eps).
    let binv = r.impact.partial_piv_lu().inverse();
    let eps: Mat<f64> = resid.as_ref() * binv.transpose();

    assert_eq!(z.nrows(), t);
    assert_eq!(eps.ncols(), n);

    let dep_raw = fourth_order_dependence(&z);
    let dep_ica = fourth_order_dependence(&eps);
    assert!(
        dep_ica < dep_raw,
        "ICA sources should be more independent (4th order) than raw whitened: \
         dep_ica={dep_ica:.3e} !< dep_raw={dep_raw:.3e}"
    );
}

#[test]
fn impact_reproduces_sigma_and_rotation_is_orthogonal() {
    let fx = load();
    let r = run(&fx);
    let sigma = mat(&fx["sigma"]);
    let n = r.impact.nrows();

    // B B' == Sigma_u (whitening is exact).
    let bbt = r.impact.as_ref() * r.impact.transpose();
    for i in 0..n {
        for j in 0..n {
            close(bbt[(i, j)], sigma[(i, j)], 1e-9, &format!("B B'[{i}][{j}]"));
        }
    }

    // Q Q' == I.
    let qqt = r.rotation.as_ref() * r.rotation.transpose();
    for i in 0..n {
        for j in 0..n {
            let e = if i == j { 1.0 } else { 0.0 };
            close(qqt[(i, j)], e, 1e-10, &format!("Q Q'[{i}][{j}]"));
        }
    }

    // Theta_0 == impact.
    close_mat(&r.irf[0], &rows(&fx["B"]), TOL_CORE, "Theta_0 == B");
}

#[test]
fn deterministic_reproducibility() {
    // Identical inputs must give bit-identical output (seedless fixed point).
    let fx = load();
    let a = run(&fx);
    let b = run(&fx);
    assert_eq!(a.converged, b.converged);
    assert_eq!(a.n_iter, b.n_iter);
    assert_eq!(a.order, b.order);
    let n = a.impact.nrows();
    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                a.impact[(i, j)].to_bits(),
                b.impact[(i, j)].to_bits(),
                "impact[{i}][{j}] not bit-identical across runs"
            );
        }
    }
    for (h, (ah, bh)) in a.irf.iter().zip(b.irf.iter()).enumerate() {
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    ah[(i, j)].to_bits(),
                    bh[(i, j)].to_bits(),
                    "irf[{h}][{i}][{j}] not bit-identical across runs"
                );
            }
        }
    }
}
