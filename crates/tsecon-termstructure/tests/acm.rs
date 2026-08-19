//! ACM (Adrian-Crump-Moench 2013) tests against `fixtures/acm.json`, plus
//! recovery, invariance, and degenerate-input properties.
//!
//! The fixture is a DOCUMENTED-FORMULA golden: `fixtures/generate_acm_fixtures.py`
//! builds the ENTIRE three-step pipeline — PCA factors, VAR(1), excess-return
//! regressions, the lambda0/lambda1 price-of-risk OLS, and the affine
//! log-price recursions — independently in NumPy (never calling tsecon) on
//! (a) a simulated affine DGP with KNOWN prices of risk and (b) the real
//! monthly GSW zero-coupon panel (1961-2014, vendored with attribution). The
//! crate must reproduce every stored quantity to 1e-8. The GSW leg is
//! additionally compared, loosely, against the NY Fed's PUBLISHED ACM
//! 10-year term premium (level/shape only — the published series is
//! estimated on the Fed's own FFR-spliced inputs and re-estimated as the
//! sample grows, so it is a validation target, not a bit-exact golden).

use serde_json::Value;
use tsecon_termstructure::{acm_term_premium, AcmFit, TermStructureError};

fn load() -> Value {
    let path = format!("{}/../../fixtures/acm.json", env!("CARGO_MANIFEST_DIR"));
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

fn assert_close(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > atol {atol:e}"
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64], atol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, atol, &format!("{ctx}[{i}]"));
    }
}

fn assert_mat_close(actual: &[Vec<f64>], expected: &[Vec<f64>], atol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: rows");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_vec_close(a, e, atol, &format!("{ctx}[{i}]"));
    }
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        cov += (ai - ma) * (bi - mb);
        va += (ai - ma).powi(2);
        vb += (bi - mb).powi(2);
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Run the estimator on one fixture leg (`"sim"` or `"gsw"`).
fn fit_leg(fx: &Value, leg: &str) -> (AcmFit, Value) {
    let case = &fx[leg];
    let yields = f64_matrix(&case["yields"]);
    let maturities = usizes(&case["maturities"]);
    let k = case["n_factors"].as_u64().expect("n_factors") as usize;
    let ppy = case["periods_per_year"].as_f64().expect("ppy");
    let fit = acm_term_premium(&yields, &maturities, k, ppy).expect("ACM fit");
    (fit, case["golden"].clone())
}

/// Compare every golden quantity of one leg at the given tolerance.
fn check_golden(fit: &AcmFit, golden: &Value, atol: f64, leg: &str) {
    // Basis-dependent quantities (factors and everything expressed in the
    // factor basis). The fixture pins the exact NumPy SVD basis; faer's SVD
    // reproduces it because the leading singular values are well separated.
    assert_mat_close(
        &fit.factors,
        &f64_matrix(&golden["factors"]),
        atol,
        &format!("{leg} factors"),
    );
    assert_mat_close(
        &fit.factor_loadings,
        &f64_matrix(&golden["loadings"]),
        atol,
        &format!("{leg} loadings"),
    );
    assert_vec_close(&fit.mu, &f64s(&golden["mu"]), atol, &format!("{leg} mu"));
    assert_mat_close(
        &fit.phi,
        &f64_matrix(&golden["phi"]),
        atol,
        &format!("{leg} phi"),
    );
    assert_mat_close(
        &fit.sigma,
        &f64_matrix(&golden["sigma"]),
        atol,
        &format!("{leg} sigma"),
    );
    assert_eq!(
        fit.rx_maturities,
        usizes(&golden["rx_maturities"]),
        "{leg} rx_maturities"
    );
    assert_vec_close(&fit.rx_a, &f64s(&golden["a"]), atol, &format!("{leg} a"));
    assert_mat_close(
        &fit.rx_beta,
        &f64_matrix(&golden["beta"]),
        atol,
        &format!("{leg} beta"),
    );
    assert_mat_close(
        &fit.rx_c,
        &f64_matrix(&golden["c"]),
        atol,
        &format!("{leg} c"),
    );
    assert_close(
        fit.sigma2,
        golden["sigma2"].as_f64().expect("sigma2"),
        atol,
        &format!("{leg} sigma2"),
    );
    assert_vec_close(
        &fit.lambda0,
        &f64s(&golden["lambda0"]),
        atol,
        &format!("{leg} lambda0"),
    );
    assert_mat_close(
        &fit.lambda1,
        &f64_matrix(&golden["lambda1"]),
        atol,
        &format!("{leg} lambda1"),
    );
    assert_close(
        fit.delta0,
        golden["delta0"].as_f64().expect("delta0"),
        atol,
        &format!("{leg} delta0"),
    );
    assert_vec_close(
        &fit.delta1,
        &f64s(&golden["delta1"]),
        atol,
        &format!("{leg} delta1"),
    );
    assert_vec_close(&fit.price_a, &f64s(&golden["A"]), atol, &format!("{leg} A"));
    assert_mat_close(
        &fit.price_b,
        &f64_matrix(&golden["B"]),
        atol,
        &format!("{leg} B"),
    );
    assert_vec_close(
        &fit.price_a_rn,
        &f64s(&golden["A_rn"]),
        atol,
        &format!("{leg} A_rn"),
    );
    assert_mat_close(
        &fit.price_b_rn,
        &f64_matrix(&golden["B_rn"]),
        atol,
        &format!("{leg} B_rn"),
    );

    // Diagnostics.
    assert_vec_close(
        &fit.var_rsquared,
        &f64s(&golden["var_rsquared"]),
        atol,
        &format!("{leg} var_rsquared"),
    );
    assert_vec_close(
        &fit.rx_rsquared,
        &f64s(&golden["rx_rsquared"]),
        atol,
        &format!("{leg} rx_rsquared"),
    );
    assert_close(
        fit.short_rate_rsquared,
        golden["short_rate_rsquared"].as_f64().expect("sr r2"),
        atol,
        &format!("{leg} short_rate_rsquared"),
    );
    assert_vec_close(
        &fit.yield_rsquared,
        &f64s(&golden["yield_rsquared"]),
        atol,
        &format!("{leg} yield_rsquared"),
    );

    // The decomposition: full first/last rows, plus whole time series at the
    // stored report maturities.
    let t_last = fit.fitted.len() - 1;
    assert_vec_close(
        &fit.fitted[0],
        &f64s(&golden["fitted_row0"]),
        atol,
        &format!("{leg} fitted_row0"),
    );
    assert_vec_close(
        &fit.fitted[t_last],
        &f64s(&golden["fitted_row_last"]),
        atol,
        &format!("{leg} fitted_row_last"),
    );
    assert_vec_close(
        &fit.term_premium[0],
        &f64s(&golden["term_premium_row0"]),
        atol,
        &format!("{leg} term_premium_row0"),
    );
    assert_vec_close(
        &fit.term_premium[t_last],
        &f64s(&golden["term_premium_row_last"]),
        atol,
        &format!("{leg} term_premium_row_last"),
    );
    for key in ["fitted", "risk_neutral", "term_premium"] {
        for n in fit.maturities.iter() {
            let field = format!("{key}_{n}");
            if golden.get(&field).is_none() {
                continue;
            }
            let expected = f64s(&golden[&field]);
            let j = fit
                .maturities
                .iter()
                .position(|m| m == n)
                .expect("maturity in grid");
            let series: Vec<f64> = match key {
                "fitted" => fit.fitted.iter().map(|row| row[j]).collect(),
                "risk_neutral" => fit.risk_neutral.iter().map(|row| row[j]).collect(),
                _ => fit.term_premium.iter().map(|row| row[j]).collect(),
            };
            assert_vec_close(&series, &expected, atol, &format!("{leg} {field}"));
        }
    }
}

#[test]
fn acm_sim_golden_matches_numpy_pipeline() {
    // Reference-matched: every quantity of the three-step pipeline reproduces
    // the independent NumPy transcription to 1e-8 on the simulated affine DGP.
    let fx = load();
    let (fit, golden) = fit_leg(&fx, "sim");
    check_golden(&fit, &golden, 1e-8, "sim");
}

#[test]
fn acm_gsw_golden_matches_numpy_pipeline() {
    // Reference-matched on REAL data: the 1961-2014 monthly GSW zero-coupon
    // panel (24 maturities, five factors — the ACM baseline).
    let fx = load();
    let (fit, golden) = fit_leg(&fx, "gsw");
    check_golden(&fit, &golden, 1e-8, "gsw");
}

#[test]
fn acm_recovers_the_true_term_premium_of_the_affine_dgp() {
    // Recovery: the fixture's simulated leg has KNOWN prices of risk, so the
    // true term premium is stored. The estimated premium must track it — the
    // MC evidence in the generator (30 reps) has corr(TP60) in [0.93, 1.00]
    // and MAE well under the ~3.7% premium level; this fixed seed measures
    // corr 0.990 / MAE 32bp.
    let fx = load();
    let (fit, _) = fit_leg(&fx, "sim");
    let true_tp60 = f64s(&fx["sim"]["true"]["term_premium_60"]);
    let true_tp36 = f64s(&fx["sim"]["true"]["term_premium_36"]);
    let j60 = fit.maturities.iter().position(|&n| n == 60).expect("60");
    let j36 = fit.maturities.iter().position(|&n| n == 36).expect("36");
    let est60: Vec<f64> = fit.term_premium.iter().map(|r| r[j60]).collect();
    let est36: Vec<f64> = fit.term_premium.iter().map(|r| r[j36]).collect();

    let corr60 = correlation(&est60, &true_tp60);
    assert!(corr60 > 0.95, "TP60 recovery correlation = {corr60}");
    let corr36 = correlation(&est36, &true_tp36);
    assert!(corr36 > 0.9, "TP36 recovery correlation = {corr36}");

    let mae60 = est60
        .iter()
        .zip(true_tp60.iter())
        .map(|(&e, &t)| (e - t).abs())
        .sum::<f64>()
        / est60.len() as f64;
    // 60bp against a ~367bp true premium (measured: 32bp).
    assert!(mae60 < 0.0060, "TP60 recovery MAE = {mae60}");
}

#[test]
fn acm_matches_the_published_ny_fed_series_in_level_and_shape() {
    // The NY Fed's published ACM 10-year decomposition (2021 vintage via the
    // Brookings mirror; quarterly) is a level/shape target with documented
    // caveats: the published series uses the Fed's own FFR-spliced curve
    // inputs and is re-estimated as the sample grows. Measured on 1961-2014:
    // TP10 corr 0.985, mean gap -0.10pp, RMSE 0.31pp; fitted-10y corr
    // 0.99999, RMSE 1.3bp.
    let fx = load();
    let (fit, _) = fit_leg(&fx, "gsw");
    let idx = usizes(&fx["gsw"]["published"]["quarter_row_idx"]);
    let pub_tp10 = f64s(&fx["gsw"]["published"]["acmtp10"]);
    let pub_y10 = f64s(&fx["gsw"]["published"]["acmy10"]);
    let j10 = fit.maturities.iter().position(|&n| n == 120).expect("120");
    let ours_tp: Vec<f64> = idx
        .iter()
        .map(|&t| fit.term_premium[t][j10] * 100.0)
        .collect();
    let ours_y: Vec<f64> = idx.iter().map(|&t| fit.fitted[t][j10] * 100.0).collect();

    let corr_tp = correlation(&ours_tp, &pub_tp10);
    assert!(corr_tp > 0.97, "TP10 corr vs published = {corr_tp}");
    let mean_gap = ours_tp
        .iter()
        .zip(pub_tp10.iter())
        .map(|(&o, &p)| o - p)
        .sum::<f64>()
        / ours_tp.len() as f64;
    assert!(mean_gap.abs() < 0.35, "TP10 mean level gap = {mean_gap}pp");
    let rmse = (ours_tp
        .iter()
        .zip(pub_tp10.iter())
        .map(|(&o, &p)| (o - p).powi(2))
        .sum::<f64>()
        / ours_tp.len() as f64)
        .sqrt();
    assert!(rmse < 0.5, "TP10 RMSE vs published = {rmse}pp");

    let corr_y = correlation(&ours_y, &pub_y10);
    assert!(corr_y > 0.999, "fitted 10y corr vs published = {corr_y}");
}

#[test]
fn acm_parallel_shift_moves_yields_one_for_one_and_leaves_the_premium_invariant() {
    // Exact invariance: adding a constant to every yield leaves excess
    // returns, factors, and prices of risk unchanged; delta0 absorbs the
    // shift, so fitted and risk-neutral yields move one-for-one and the term
    // premium is bit-for-bit stable (up to fp roundoff).
    let fx = load();
    let case = &fx["sim"];
    let yields = f64_matrix(&case["yields"]);
    let maturities = usizes(&case["maturities"]);
    let k = case["n_factors"].as_u64().expect("k") as usize;
    let ppy = case["periods_per_year"].as_f64().expect("ppy");

    let base = acm_term_premium(&yields, &maturities, k, ppy).expect("base");
    let shift = 0.01;
    let shifted: Vec<Vec<f64>> = yields
        .iter()
        .map(|row| row.iter().map(|y| y + shift).collect())
        .collect();
    let moved = acm_term_premium(&shifted, &maturities, k, ppy).expect("shifted");

    for t in 0..yields.len() {
        for j in 0..maturities.len() {
            assert_close(
                moved.term_premium[t][j],
                base.term_premium[t][j],
                1e-12,
                &format!("term premium invariance [{t}][{j}]"),
            );
            assert_close(
                moved.fitted[t][j],
                base.fitted[t][j] + shift,
                1e-12,
                &format!("fitted shift [{t}][{j}]"),
            );
        }
    }
}

#[test]
fn acm_scale_behavior_is_first_order_homogeneous_up_to_convexity() {
    // Teaching property: the pipeline is linear except the Jensen convexity
    // terms, which are quadratic. Doubling the yields therefore doubles the
    // term premium only up to a convexity-order remainder (measured max
    // 0.94bp here) — which is exactly why the input units (decimal, not
    // percent) are load-bearing: at scale 100 the convexity terms are 100x
    // too large relative to everything else.
    let fx = load();
    let case = &fx["sim"];
    let yields = f64_matrix(&case["yields"]);
    let maturities = usizes(&case["maturities"]);
    let k = case["n_factors"].as_u64().expect("k") as usize;
    let ppy = case["periods_per_year"].as_f64().expect("ppy");

    let base = acm_term_premium(&yields, &maturities, k, ppy).expect("base");
    let doubled: Vec<Vec<f64>> = yields
        .iter()
        .map(|row| row.iter().map(|y| 2.0 * y).collect())
        .collect();
    let big = acm_term_premium(&doubled, &maturities, k, ppy).expect("doubled");

    let mut max_dev = 0.0f64;
    for t in 0..yields.len() {
        for j in 0..maturities.len() {
            let dev = (big.term_premium[t][j] / 2.0 - base.term_premium[t][j]).abs();
            max_dev = max_dev.max(dev);
        }
    }
    // Not exact (convexity), but far smaller than the premium itself.
    assert!(
        max_dev < 3e-4,
        "scale deviation {max_dev} (should be convexity-order)"
    );
    assert!(max_dev > 0.0, "scaling should not be exactly homogeneous");
}

#[test]
fn acm_maturity_one_is_priced_by_the_short_rate_equation_exactly() {
    // A_1 = -delta0 and B_1 = -delta1 by construction, so the fitted yield at
    // the one-period maturity is exactly the short-rate regression fit.
    let fx = load();
    let (fit, _) = fit_leg(&fx, "sim");
    for (t, factor_row) in fit.factors.iter().enumerate() {
        let sr_fit: f64 = fit.delta0
            + fit
                .delta1
                .iter()
                .zip(factor_row.iter())
                .map(|(&d, &x)| d * x)
                .sum::<f64>();
        assert_close(
            fit.fitted[t][0],
            sr_fit * fit.periods_per_year,
            1e-12,
            &format!("maturity-1 fitted vs short-rate fit at t={t}"),
        );
    }
}

#[test]
fn acm_recovery_from_an_independent_rust_simulated_dgp() {
    // Recovery on a DGP simulated HERE (xorshift + Box-Muller — no fixture,
    // no NumPy): the estimated 5-year premium must track the DGP truth. This
    // complements the fixture-based recovery test with a draw the generator
    // never saw.
    let k = 3usize;
    let n_max = 60usize;
    let t_len = 400usize;
    let ppy = 12.0;

    let mu = [0.02, -0.01, 0.005];
    let phi = [[0.97, 0.02, 0.00], [0.00, 0.90, 0.03], [0.01, 0.00, 0.80]];
    let chol = [[0.16, 0.00, 0.00], [0.02, 0.11, 0.00], [-0.01, 0.01, 0.09]];
    let mut sigma = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            sigma[i][j] = (0..3).map(|l| chol[i][l] * chol[j][l]).sum();
        }
    }
    let delta0 = 0.003;
    let delta1 = [0.0011, 0.0006, 0.0004];
    let lambda0 = [-0.12, 0.08, -0.05];
    let lambda1 = [
        [-0.020, 0.015, -0.010],
        [0.012, -0.018, 0.008],
        [-0.006, 0.010, -0.015],
    ];

    // True recursions (no sigma^2: the DGP prices without error).
    let recur = |l0: [f64; 3], l1: [[f64; 3]; 3]| -> (Vec<f64>, Vec<[f64; 3]>) {
        let mut a = vec![0.0f64; n_max];
        let mut b = vec![[0.0f64; 3]; n_max];
        a[0] = -delta0;
        b[0] = [-delta1[0], -delta1[1], -delta1[2]];
        for n in 1..n_max {
            let bp = b[n - 1];
            let drift: f64 = (0..3).map(|i| bp[i] * (mu[i] - l0[i])).sum();
            let conv: f64 = (0..3)
                .map(|i| (0..3).map(|j| bp[i] * sigma[i][j] * bp[j]).sum::<f64>())
                .sum();
            a[n] = a[n - 1] + drift + 0.5 * conv - delta0;
            for j in 0..3 {
                b[n][j] = (0..3).map(|i| (phi[i][j] - l1[i][j]) * bp[i]).sum::<f64>() - delta1[j];
            }
        }
        (a, b)
    };
    let (a_p, b_p) = recur(lambda0, lambda1);
    let (a_q, b_q) = recur([0.0; 3], [[0.0; 3]; 3]);

    // xorshift64 + Box-Muller standard normals.
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut gauss = move || {
        let (u1, u2) = (uniform(), uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // Simulate factors (with burn-in), yields + 5bp measurement noise.
    let mut x = [0.0f64; 3];
    // Start at the unconditional mean (I - phi)^{-1} mu, approximated by a
    // long deterministic pass, then burn in with noise.
    for _ in 0..500 {
        let mut nx = [0.0f64; 3];
        for i in 0..3 {
            nx[i] = mu[i] + (0..3).map(|j| phi[i][j] * x[j]).sum::<f64>();
        }
        x = nx;
    }
    let step = |x: &[f64; 3], gauss: &mut dyn FnMut() -> f64| -> [f64; 3] {
        let e = [gauss(), gauss(), gauss()];
        let mut nx = [0.0f64; 3];
        for i in 0..3 {
            nx[i] = mu[i]
                + (0..3).map(|j| phi[i][j] * x[j]).sum::<f64>()
                + (0..3).map(|j| chol[i][j] * e[j]).sum::<f64>();
        }
        nx
    };
    for _ in 0..200 {
        x = step(&x, &mut gauss);
    }
    let maturities: Vec<usize> = (1..=n_max).collect();
    let mut yields = Vec::with_capacity(t_len);
    let mut true_tp60 = Vec::with_capacity(t_len);
    for _ in 0..t_len {
        x = step(&x, &mut gauss);
        let mut row = Vec::with_capacity(n_max);
        for &n in &maturities {
            let bx: f64 = (0..3).map(|j| b_p[n - 1][j] * x[j]).sum();
            let y = -(a_p[n - 1] + bx) * ppy / n as f64;
            row.push(y + 0.0005 * gauss());
        }
        let bx_p: f64 = (0..3).map(|j| b_p[n_max - 1][j] * x[j]).sum();
        let bx_q: f64 = (0..3).map(|j| b_q[n_max - 1][j] * x[j]).sum();
        let y_p = -(a_p[n_max - 1] + bx_p) * ppy / n_max as f64;
        let y_q = -(a_q[n_max - 1] + bx_q) * ppy / n_max as f64;
        true_tp60.push(y_p - y_q);
        yields.push(row);
    }

    let fit = acm_term_premium(&yields, &maturities, k, ppy).expect("fit");
    let est_tp60: Vec<f64> = fit.term_premium.iter().map(|r| r[n_max - 1]).collect();
    let corr = correlation(&est_tp60, &true_tp60);
    assert!(corr > 0.9, "independent-DGP TP60 recovery corr = {corr}");
    let mae = est_tp60
        .iter()
        .zip(true_tp60.iter())
        .map(|(&e, &t)| (e - t).abs())
        .sum::<f64>()
        / t_len as f64;
    assert!(mae < 0.01, "independent-DGP TP60 recovery MAE = {mae}");
}

#[test]
fn acm_rejects_degenerate_inputs_with_teaching_errors() {
    let good_mats: Vec<usize> = (1..=8).collect();
    let good_yields: Vec<Vec<f64>> = (0..30)
        .map(|t| {
            (1..=8)
                .map(|n| 0.03 + 0.001 * n as f64 + 1e-4 * ((t * 8 + n) as f64).sin())
                .collect()
        })
        .collect();

    // Empty grid.
    assert!(matches!(
        acm_term_premium(&good_yields, &[], 2, 12.0),
        Err(TermStructureError::EmptyMaturities)
    ));
    // A zero maturity.
    assert!(matches!(
        acm_term_premium(&good_yields, &[0, 1, 2, 3, 4, 5, 6, 7], 2, 12.0),
        Err(TermStructureError::InvalidMaturity { .. })
    ));
    // Not strictly ascending.
    assert!(matches!(
        acm_term_premium(&good_yields, &[1, 2, 2, 3, 4, 5, 6, 7], 2, 12.0),
        Err(TermStructureError::MaturitiesNotAscending { index: 2 })
    ));
    // Missing the one-period maturity.
    assert!(matches!(
        acm_term_premium(&good_yields, &[2, 3, 4, 5, 6, 7, 8, 9], 2, 12.0),
        Err(TermStructureError::MissingShortMaturity { shortest: 2 })
    ));
    // Too many / zero factors.
    assert!(matches!(
        acm_term_premium(&good_yields, &good_mats, 0, 12.0),
        Err(TermStructureError::InvalidFactorCount { .. })
    ));
    assert!(matches!(
        acm_term_premium(&good_yields, &good_mats, 8, 12.0),
        Err(TermStructureError::InvalidFactorCount { .. })
    ));
    // Bad periods_per_year.
    assert!(matches!(
        acm_term_premium(&good_yields, &good_mats, 2, 0.0),
        Err(TermStructureError::InvalidPeriodsPerYear { .. })
    ));
    assert!(matches!(
        acm_term_premium(&good_yields, &good_mats, 2, f64::NAN),
        Err(TermStructureError::InvalidPeriodsPerYear { .. })
    ));
    // Too few observations: 2K + 3 = 7 needed for K = 2.
    assert!(matches!(
        acm_term_premium(&good_yields[..6], &good_mats, 2, 12.0),
        Err(TermStructureError::PanelTooShort { .. })
    ));
    // A ragged row.
    let mut ragged = good_yields.clone();
    ragged[3].pop();
    assert!(matches!(
        acm_term_premium(&ragged, &good_mats, 2, 12.0),
        Err(TermStructureError::DimensionMismatch { .. })
    ));
    // A NaN yield.
    let mut nan = good_yields.clone();
    nan[5][2] = f64::NAN;
    assert!(matches!(
        acm_term_premium(&nan, &good_mats, 2, 12.0),
        Err(TermStructureError::NonFinite { .. })
    ));
    // No adjacent maturity pairs at all: no excess returns to price.
    assert!(matches!(
        acm_term_premium(
            &good_yields
                .iter()
                .map(|r| r[..5].to_vec())
                .collect::<Vec<_>>(),
            &[1, 3, 5, 7, 9],
            2,
            12.0
        ),
        Err(TermStructureError::Underdetermined { .. })
    ));
    // Too few pairs for the price-of-risk cross-section (N_rx = 2 < K + 1 = 4).
    assert!(matches!(
        acm_term_premium(
            &good_yields
                .iter()
                .map(|r| r[..5].to_vec())
                .collect::<Vec<_>>(),
            &[1, 2, 3, 5, 7],
            3,
            12.0
        ),
        Err(TermStructureError::Underdetermined { .. })
    ));
    // A constant panel: factors carry no variation.
    let flat = vec![vec![0.04; 8]; 30];
    assert!(matches!(
        acm_term_premium(&flat, &good_mats, 2, 12.0),
        Err(TermStructureError::SingularDesign { .. })
    ));
}
