//! Golden-value and optimality-certificate tests for the structured-penalty
//! and post-selection slice, against `fixtures/structured.json`.
//!
//! * `group_lasso`: every fixture case is (1) certified optimal by an
//!   **independent** evaluation of the subgradient KKT conditions written
//!   here — rigorous for a convex problem and the primary grade — and (2)
//!   cross-checked against `skglm` (an independent package; the
//!   `l1_ratio = 1` case against scikit-learn `Lasso`). The fixture records
//!   each reference solution's own KKT residual so the cross-package
//!   tolerance is honest about which side limits it.
//! * `post_lasso`: the refit against scikit-learn
//!   `LinearRegression(fit_intercept=False)` on the selected columns.
//! * `pds_lasso`: the exact leg against statsmodels HAC / nonrobust OLS on
//!   the (known or BIC-selected) union of supports.

mod common;

use common::{as_f64_vec, as_mat, assert_slice_close, load_fixture};
use serde_json::Value;
use tsecon_ml::faer::Mat;
use tsecon_ml::{
    group_lasso, group_lasso_alpha_max, pds_lasso, post_lasso, CoordDescentOptions, GroupWeights,
    PdsAlpha,
};

const OPTS: CoordDescentOptions = CoordDescentOptions {
    tol: 1e-11,
    max_iter: 100_000,
};

fn as_i64_vec(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn as_usize_vec(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn weights_of(v: &Value) -> GroupWeights {
    match v {
        Value::String(s) if s == "sqrt_size" => GroupWeights::SqrtSize,
        Value::String(s) if s == "none" => GroupWeights::Uniform,
        Value::Array(_) => GroupWeights::Custom(as_f64_vec(v)),
        other => panic!("unexpected group_weights {other:?}"),
    }
}

/// Distinct labels ascending and their member columns.
fn members_of(groups: &[i64]) -> (Vec<i64>, Vec<Vec<usize>>) {
    let mut labels = groups.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let members = labels
        .iter()
        .map(|&l| (0..groups.len()).filter(|&j| groups[j] == l).collect())
        .collect();
    (labels, members)
}

fn resolve(w: &GroupWeights, members: &[Vec<usize>]) -> Vec<f64> {
    match w {
        GroupWeights::SqrtSize => members.iter().map(|m| (m.len() as f64).sqrt()).collect(),
        GroupWeights::Uniform => vec![1.0; members.len()],
        GroupWeights::Custom(v) => v.clone(),
    }
}

fn soft(z: f64, t: f64) -> f64 {
    if z > t {
        z - t
    } else if z < -t {
        z + t
    } else {
        0.0
    }
}

/// Independent subgradient-KKT residual for
/// `(1/(2n))||y - Xb||^2 + alpha[(1 - l1) sum_g w_g ||b_g|| + l1 ||b||_1]`,
/// written from the optimality conditions rather than shared with the
/// solver. With `grad = -X'(y - Xb)/n`:
///
/// * `b_g = 0`: `||S(-grad_g, alpha l1)||_2 <= alpha (1 - l1) w_g`;
/// * `b_g != 0`, `b_j != 0`: `grad_j + alpha (1 - l1) w_g b_j / ||b_g|| +
///   alpha l1 sign(b_j) = 0`;
/// * `b_g != 0`, `b_j = 0`: `|grad_j| <= alpha l1`.
fn kkt_residual(
    x: &Mat<f64>,
    y: &[f64],
    b: &[f64],
    members: &[Vec<usize>],
    w: &[f64],
    alpha: f64,
    l1: f64,
) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    let nf = n as f64;
    let resid: Vec<f64> = (0..n)
        .map(|i| y[i] - (0..p).map(|j| x[(i, j)] * b[j]).sum::<f64>())
        .collect();
    let grad: Vec<f64> = (0..p)
        .map(|j| -(0..n).map(|i| x[(i, j)] * resid[i]).sum::<f64>() / nf)
        .collect();
    let lam1 = alpha * l1;
    let lam2 = alpha * (1.0 - l1);
    let mut worst = 0.0f64;
    for (g, m) in members.iter().enumerate() {
        let nb = m.iter().map(|&j| b[j] * b[j]).sum::<f64>().sqrt();
        if nb == 0.0 {
            let s = m
                .iter()
                .map(|&j| soft(-grad[j], lam1).powi(2))
                .sum::<f64>()
                .sqrt();
            worst = worst.max(s - lam2 * w[g]);
        } else {
            for &j in m {
                let v = if b[j] != 0.0 {
                    (grad[j] + lam2 * w[g] * b[j] / nb + lam1 * b[j].signum()).abs()
                } else {
                    grad[j].abs() - lam1
                };
                worst = worst.max(v);
            }
        }
    }
    worst.max(0.0)
}

fn objective(
    x: &Mat<f64>,
    y: &[f64],
    b: &[f64],
    members: &[Vec<usize>],
    w: &[f64],
    alpha: f64,
    l1: f64,
) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    let rss: f64 = (0..n)
        .map(|i| {
            let r = y[i] - (0..p).map(|j| x[(i, j)] * b[j]).sum::<f64>();
            r * r
        })
        .sum();
    let grp: f64 = members
        .iter()
        .zip(w)
        .map(|(m, &wg)| wg * m.iter().map(|&j| b[j] * b[j]).sum::<f64>().sqrt())
        .sum();
    let l1n: f64 = b.iter().map(|v| v.abs()).sum();
    rss / (2.0 * n as f64) + alpha * ((1.0 - l1) * grp + l1 * l1n)
}

/// **Optimality certificate.** Every fixture case's solution satisfies the
/// independently evaluated KKT conditions to 1e-8 (achieved is printed),
/// converges, reproduces the reference's active groups, and its objective is
/// no worse than the reference's beyond rounding.
#[test]
fn group_lasso_kkt_certificate_on_every_fixture_case() {
    let fx = load_fixture("structured.json");
    let gl = &fx["group_lasso"];
    let mut worst_kkt = 0.0f64;
    let mut worst_obj_gap = f64::NEG_INFINITY;
    for case in gl["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let design = &gl["designs"][case["design"].as_str().unwrap()];
        let x = as_mat(&design["X"]);
        let y = as_f64_vec(&design["y"]);
        let groups = as_i64_vec(&design["groups"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let l1 = case["l1_ratio"].as_f64().unwrap();
        let weights = weights_of(&case["group_weights"]);
        let (labels, members) = members_of(&groups);
        let w = resolve(&weights, &members);

        let fit = group_lasso(x.as_ref(), &y, &groups, alpha, l1, &weights, OPTS).unwrap();
        assert!(fit.converged, "{name}: not converged");
        assert!(
            fit.max_rel_change <= OPTS.tol,
            "{name}: max_rel_change > tol"
        );

        let kkt = kkt_residual(&x, &y, &fit.coef, &members, &w, alpha, l1);
        assert!(
            kkt <= 1e-8,
            "{name}: independent KKT residual {kkt:e} > 1e-8"
        );
        worst_kkt = worst_kkt.max(kkt);
        // The solver's self-reported certificate agrees with the independent one.
        assert!(
            (fit.kkt_violation - kkt).abs() <= 1e-12,
            "{name}: reported kkt {:e} vs independent {kkt:e}",
            fit.kkt_violation
        );

        let expected_groups = as_i64_vec(&case["active_groups"]);
        assert_eq!(fit.active_groups, expected_groups, "{name}: active groups");
        for &lab in &fit.active_groups {
            assert!(labels.contains(&lab));
        }
        let active_from_coef: Vec<usize> = (0..fit.coef.len())
            .filter(|&j| fit.coef[j] != 0.0)
            .collect();
        assert_eq!(fit.active_set, active_from_coef, "{name}: active set");

        // Objective: recomputed here, equal to the solver's report, and no
        // worse than the reference solution's beyond rounding.
        let obj = objective(&x, &y, &fit.coef, &members, &w, alpha, l1);
        assert!((obj - fit.objective).abs() <= 1e-12 * obj.abs().max(1.0));
        let ref_obj = case["objective"].as_f64().unwrap();
        let gap = obj - ref_obj;
        worst_obj_gap = worst_obj_gap.max(gap);
        assert!(
            gap <= 1e-12,
            "{name}: objective {obj} exceeds reference {ref_obj} by {gap:e}"
        );
    }
    println!("group_lasso worst independent KKT residual: {worst_kkt:e}");
    println!("group_lasso worst objective gap vs reference: {worst_obj_gap:e}");
}

/// **Cross-package golden.** Coefficients agree with skglm (l1_ratio < 1)
/// and scikit-learn Lasso (l1_ratio = 1) to 1e-8 absolute; the achieved
/// figure is printed. The fixture's `reference_kkt` (skglm's own residual,
/// ~1e-13) is what bounds the agreement, not this solver.
#[test]
fn group_lasso_matches_skglm_and_sklearn() {
    let fx = load_fixture("structured.json");
    let gl = &fx["group_lasso"];
    let mut worst = 0.0f64;
    let mut worst_ref_kkt = 0.0f64;
    for case in gl["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let design = &gl["designs"][case["design"].as_str().unwrap()];
        let x = as_mat(&design["X"]);
        let y = as_f64_vec(&design["y"]);
        let groups = as_i64_vec(&design["groups"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let l1 = case["l1_ratio"].as_f64().unwrap();
        let weights = weights_of(&case["group_weights"]);
        let expected = as_f64_vec(&case["coef"]);
        let ref_kkt = case["reference_kkt"].as_f64().unwrap();
        worst_ref_kkt = worst_ref_kkt.max(ref_kkt);
        assert!(
            ref_kkt <= 1e-10,
            "{name}: reference itself is not converged ({ref_kkt:e})"
        );

        let fit = group_lasso(x.as_ref(), &y, &groups, alpha, l1, &weights, OPTS).unwrap();
        let d = assert_slice_close(&fit.coef, &expected, 1e-8, name);
        worst = worst.max(d);
    }
    println!("group_lasso achieved max abs coefficient error vs reference: {worst:e}");
    println!("reference solutions' own worst KKT residual: {worst_ref_kkt:e}");
    assert!(worst <= 1e-8);
}

/// `alpha_max` (closed form / bisection) matches the NumPy transcription to
/// 1e-12 relative, both from the standalone function and as reported by the
/// fit; at `alpha_max * (1 + 1e-9)` the solution is identically zero and at
/// `alpha_max * (1 - 1e-3)` it is not.
#[test]
fn group_lasso_alpha_max_matches_fixture_and_zeroes_the_fit() {
    let fx = load_fixture("structured.json");
    let gl = &fx["group_lasso"];
    for case in gl["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let design = &gl["designs"][case["design"].as_str().unwrap()];
        let x = as_mat(&design["X"]);
        let y = as_f64_vec(&design["y"]);
        let groups = as_i64_vec(&design["groups"]);
        let l1 = case["l1_ratio"].as_f64().unwrap();
        let weights = weights_of(&case["group_weights"]);
        let expected = case["alpha_max"].as_f64().unwrap();

        let am = group_lasso_alpha_max(x.as_ref(), &y, &groups, l1, &weights).unwrap();
        assert!(
            ((am - expected) / expected).abs() <= 1e-12,
            "{name}: alpha_max {am} vs {expected}"
        );
        let above = group_lasso(
            x.as_ref(),
            &y,
            &groups,
            am * (1.0 + 1e-9),
            l1,
            &weights,
            OPTS,
        )
        .unwrap();
        assert!(above.converged);
        assert!((above.alpha_max - am).abs() <= 1e-15 * am);
        assert!(
            above.coef.iter().all(|&b| b == 0.0),
            "{name}: nonzero coefficient above alpha_max"
        );
        assert!(above.active_groups.is_empty() && above.active_set.is_empty());
        let below = group_lasso(
            x.as_ref(),
            &y,
            &groups,
            am * (1.0 - 1e-3),
            l1,
            &weights,
            OPTS,
        )
        .unwrap();
        assert!(
            below.coef.iter().any(|&b| b != 0.0),
            "{name}: all-zero coefficient below alpha_max"
        );
    }
}

/// post-LASSO OLS refit matches scikit-learn `LinearRegression` on the
/// scikit-learn `Lasso`/`ElasticNet` support to 1e-10 (support exact).
#[test]
fn post_lasso_refit_matches_sklearn_linear_regression() {
    let fx = load_fixture("structured.json");
    let design = &fx["group_lasso"]["designs"]["blocks"];
    let x = as_mat(&design["X"]);
    let y = as_f64_vec(&design["y"]);
    let mut worst_ols = 0.0f64;
    let mut worst_lasso = 0.0f64;
    for case in fx["post_lasso"]["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let alpha = case["alpha"].as_f64().unwrap();
        let l1 = case["l1_ratio"].as_f64().unwrap();
        let fit = post_lasso(x.as_ref(), &y, alpha, l1, OPTS).unwrap();
        assert_eq!(
            fit.support,
            as_usize_vec(&case["support"]),
            "{name}: support"
        );
        assert_eq!(fit.n_selected, fit.support.len());
        let d = assert_slice_close(&fit.coef_ols, &as_f64_vec(&case["coef_ols"]), 1e-10, name);
        worst_ols = worst_ols.max(d);
        let d = assert_slice_close(
            &fit.coef_lasso,
            &as_f64_vec(&case["coef_lasso"]),
            1e-6,
            name,
        );
        worst_lasso = worst_lasso.max(d);
        let rss = case["rss"].as_f64().unwrap();
        assert!(
            ((fit.rss - rss) / rss).abs() <= 1e-10,
            "{name}: rss {} vs {rss}",
            fit.rss
        );
        for j in 0..fit.coef_ols.len() {
            if !fit.support.contains(&j) {
                assert_eq!(
                    fit.coef_ols[j], 0.0,
                    "{name}: nonzero off-support refit at {j}"
                );
            }
        }
    }
    println!("post_lasso achieved max abs refit error: {worst_ols:e}");
    println!("post_lasso achieved max abs first-stage error vs sklearn: {worst_lasso:e}");
}

fn check_pds_case(case: &Value, fit: &tsecon_ml::PdsFit, name: &str) -> f64 {
    let mut worst = 0.0f64;
    let mut rel = |got: f64, key: &str| {
        let want = case[key].as_f64().unwrap();
        let d = ((got - want) / want).abs();
        assert!(d <= 1e-8, "{name}: {key} {got} vs {want} (rel {d:e})");
        worst = worst.max(d);
    };
    rel(fit.coef, "coef");
    rel(fit.se, "se");
    rel(fit.t_stat, "t_stat");
    rel(fit.conf_int.0, "conf_int_lo");
    rel(fit.conf_int.1, "conf_int_hi");
    // p-values are compared absolutely: they can be tiny.
    let p_want = case["p_value"].as_f64().unwrap();
    assert!(
        (fit.p_value - p_want).abs() <= 1e-12,
        "{name}: p_value {} vs {p_want}",
        fit.p_value
    );
    assert_eq!(
        fit.support_y,
        as_usize_vec(&case["support_y"]),
        "{name}: support_y"
    );
    assert_eq!(
        fit.support_d,
        as_usize_vec(&case["support_d"]),
        "{name}: support_d"
    );
    assert_eq!(
        fit.union_support,
        as_usize_vec(&case["union_support"]),
        "{name}: union_support"
    );
    assert_eq!(fit.n_controls_selected, fit.union_support.len());
    worst
}

/// The statsmodels leg: with every control forced into the union (tiny
/// alpha) and with the BIC-selected union, the coefficient, HAC / nonrobust
/// standard error, t-statistic, normal p-value and 95% interval match
/// statsmodels to 1e-8 relative (p-values 1e-12 absolute). The convention
/// matched is `cov_type="HAC"`, Bartlett, `maxlags=L`,
/// `use_correction=True`, `use_t=False`; `hac_lags=0` is
/// `cov_type="nonrobust"`, `use_t=False`.
#[test]
fn pds_lasso_inference_matches_statsmodels() {
    let fx = load_fixture("structured.json");
    let pds = &fx["pds"];
    let x = as_mat(&pds["X"]);
    let y = as_f64_vec(&pds["y"]);
    let d = as_f64_vec(&pds["d"]);
    let n = y.len();
    let p = x.ncols();
    let nw_rule = pds["newey_west_rule_maxlags"].as_u64().unwrap() as usize;
    let mut worst = 0.0f64;
    for case in pds["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let lags = case["hac_lags"].as_u64().unwrap() as usize;
        let alpha = match &case["alpha"] {
            Value::String(s) => {
                assert_eq!(s, "bic");
                PdsAlpha::Bic
            }
            v => PdsAlpha::Fixed(v.as_f64().unwrap()),
        };
        // Exercise the `None -> Newey-West rule` resolution on the rule's
        // own value so both paths are pinned.
        let hac_lags = if lags == nw_rule { None } else { Some(lags) };
        let fit = pds_lasso(&y, &d, x.as_ref(), alpha, hac_lags, OPTS).unwrap();
        assert_eq!(fit.hac_lags_resolved, lags, "{name}: resolved lags");
        // Add the interval bounds as pseudo-keys for the shared checker.
        let mut case = case.clone();
        let ci = case["conf_int"].as_array().unwrap().clone();
        case["conf_int_lo"] = ci[0].clone();
        case["conf_int_hi"] = ci[1].clone();
        worst = worst.max(check_pds_case(&case, &fit, name));
        match alpha {
            PdsAlpha::Fixed(a) => {
                assert_eq!(fit.alpha_y, a);
                assert_eq!(fit.alpha_d, a);
                assert_eq!(fit.union_support.len(), p, "{name}: forced union");
            }
            PdsAlpha::Bic => {
                let ay = case["alpha_y"].as_f64().unwrap();
                let ad = case["alpha_d"].as_f64().unwrap();
                assert!(((fit.alpha_y - ay) / ay).abs() <= 1e-10, "{name}: alpha_y");
                assert!(((fit.alpha_d - ad) / ad).abs() <= 1e-10, "{name}: alpha_d");
            }
        }
    }
    assert_eq!(nw_rule, tsecon_hac_rule(n));
    println!("pds_lasso achieved max relative error vs statsmodels: {worst:e}");
}

/// `floor(4 (n/100)^(2/9))`, written out so the fixture's rule value is
/// checked against the formula and not just against the crate.
fn tsecon_hac_rule(n: usize) -> usize {
    (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize
}
