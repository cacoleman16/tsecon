//! Golden and certificate tests for the convex / greedy estimators against
//! `fixtures/convex.json`.
//!
//! * **L1 trend filtering** — three legs, graded separately:
//!   1. an **optimality certificate** re-derived here from scratch for the
//!      crate's own trend (the primary grade for a convex problem): the
//!      dual variable is recovered from the residual by `order` negative
//!      cumulative sums, clipped into the dual box, and the relative
//!      duality gap `P(x) - G(v)` must be `<= 1e-8` on every case;
//!   2. **cvxpy + Clarabel** (third-party interior-point solver, converged
//!      at its 1e-14 tolerance; the fixture records each reference's own
//!      relative gap, 1e-13 to 1e-16) — the trend must agree at 1e-8
//!      absolute;
//!   3. the closed-form **limits**: `lam_max` and, for `lam >= lam_max`,
//!      the least-squares polynomial (`np.polyfit`) at 1e-8.
//! * **L2 / Hodrick-Prescott** — the dense `np.linalg.solve(I + lam D'D, y)`
//!   closed form at 1e-10.
//! * **Boosting** — a dense NumPy transcription with the boosting operator
//!   formed explicitly (so its trace is exact by construction) at 1e-12 on
//!   `coef_path`, `df_path`, `aic_path`; `selected` and `best_step` exact;
//!   `fitted` must reproduce `B_best y` from the dense operator at 1e-10.
//!   This is a transcription grade, not a third-party run (R mboost is not
//!   runnable in the build environment).

mod common;

use common::{as_f64_vec, as_mat, assert_slice_close, load_fixture};
use tsecon_ml::{
    boosting, l1_trend_filter, BoostStop, BoostingOptions, Penalty, TrendFilterOptions,
};

/// `D_k x` by `k` successive first differences (independent of the crate).
fn diff_k(x: &[f64], k: usize) -> Vec<f64> {
    let mut v = x.to_vec();
    for _ in 0..k {
        v = v.windows(2).map(|w| w[1] - w[0]).collect();
    }
    v
}

/// `D_k' v` (independent of the crate).
fn diff_k_t(v: &[f64], k: usize) -> Vec<f64> {
    let mut w = v.to_vec();
    for _ in 0..k {
        let l = w.len();
        let mut out = vec![0.0; l + 1];
        for (j, o) in out.iter_mut().enumerate() {
            let a = if j >= 1 { w[j - 1] } else { 0.0 };
            let b = if j < l { w[j] } else { 0.0 };
            *o = a - b;
        }
        w = out;
    }
    w
}

/// The KKT / duality-gap certificate for a candidate trend `x`, evaluated
/// from scratch: `(pobj, dobj, v_raw)` with `v_raw` the dual variable
/// recovered from `y - x = D' v` and the dual objective taken at
/// `clip(v_raw, -lam, lam)` so it is feasible whatever `x` is.
fn certificate(y: &[f64], x: &[f64], lam: f64, k: usize) -> (f64, f64, Vec<f64>) {
    let r: Vec<f64> = y.iter().zip(x).map(|(a, b)| a - b).collect();
    let mut v = r.clone();
    for _ in 0..k {
        let mut acc = 0.0;
        let cs: Vec<f64> = v
            .iter()
            .map(|t| {
                acc += t;
                -acc
            })
            .collect();
        v = cs[..cs.len() - 1].to_vec();
    }
    let vc: Vec<f64> = v.iter().map(|t| t.clamp(-lam, lam)).collect();
    let dx = diff_k(x, k);
    let pobj =
        0.5 * r.iter().map(|t| t * t).sum::<f64>() + lam * dx.iter().map(|t| t.abs()).sum::<f64>();
    let dtv = diff_k_t(&vc, k);
    let dy = diff_k(y, k);
    let dobj = -0.5 * dtv.iter().map(|t| t * t).sum::<f64>()
        + vc.iter().zip(&dy).map(|(a, b)| a * b).sum::<f64>();
    (pobj, dobj, v)
}

fn series(fx: &serde_json::Value, name: &str) -> Vec<f64> {
    as_f64_vec(&fx["series"][name])
}

fn l1_opts(order: usize) -> TrendFilterOptions {
    TrendFilterOptions {
        order,
        penalty: Penalty::L1,
        ..TrendFilterOptions::default()
    }
}

/// Leg 1 — the optimality certificate. For every L1 case the relative
/// duality gap of the crate's trend, re-derived here, is `<= 1e-8`; the
/// recovered dual variable is feasible; and at every knot it sits on the
/// bound with the sign of the kink (complementary slackness). Achieved:
/// machine precision (the polish lands the exact active-set solution).
#[test]
fn golden_l1_trend_filter_certificate() {
    let fx = load_fixture("convex.json");
    let mut worst_gap = 0.0f64;
    let mut worst_kkt = 0.0f64;
    for case in fx["l1_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = series(&fx, case["series"].as_str().unwrap());
        let k = case["order"].as_u64().unwrap() as usize;
        let lam = case["lam"].as_f64().unwrap();
        let fit = l1_trend_filter(&y, lam, l1_opts(k)).unwrap();
        assert!(fit.converged, "{name}: not converged");

        let (pobj, dobj, v) = certificate(&y, &fit.trend, lam, k);
        let rel = (pobj - dobj) / pobj;
        assert!(rel <= 1e-8, "{name}: relative duality gap {rel:e} > 1e-8");
        worst_gap = worst_gap.max(rel);
        // The crate's own objective and certificate agree with the
        // independent evaluation.
        assert!(
            (fit.objective - pobj).abs() <= 1e-10 * pobj,
            "{name}: reported objective {} vs independent {pobj}",
            fit.objective
        );
        assert!(
            fit.duality_gap <= 1e-8 * pobj,
            "{name}: reported gap too large"
        );

        // Dual feasibility of the recovered multiplier.
        let vmax = v.iter().fold(0.0f64, |m, t| m.max(t.abs()));
        assert!(
            vmax <= lam * (1.0 + 1e-8),
            "{name}: recovered dual |v|_inf = {vmax} exceeds lam = {lam}"
        );
        // Complementary slackness at every reported knot.
        let dx = diff_k(&fit.trend, k);
        for &i in &fit.knots {
            let want = lam * dx[i].signum();
            let d = (v[i] - want).abs() / lam;
            worst_kkt = worst_kkt.max(d);
            assert!(
                d <= 1e-6,
                "{name}: knot {i}: v = {} but lam*sign(dx) = {want}",
                v[i]
            );
        }
        // Every kink above the knot threshold is reported, and nothing else.
        let dy = diff_k(&y, k);
        let thr = (1e-6 * dy.iter().fold(0.0f64, |m, t| m.max(t.abs())))
            .max(1e-12 * y.iter().fold(0.0f64, |m, t| m.max(t.abs())));
        let expect: Vec<usize> = (0..dx.len()).filter(|&i| dx[i].abs() > thr).collect();
        assert_eq!(fit.knots, expect, "{name}: knot set");
        println!(
            "  {name}: n_iter {}, relative gap {rel:.2e}, {} knots",
            fit.n_iter,
            fit.knots.len()
        );
    }
    println!(
        "L1 certificate: worst relative duality gap {worst_gap:e}, worst knot KKT {worst_kkt:e}"
    );
}

/// Leg 2 — the third-party trend. cvxpy + Clarabel at its tightest
/// tolerance (1e-14; each reference's own relative gap is recorded in the
/// fixture and asserted here to be below 1e-11, so the comparison is
/// against a converged reference). The trends agree at 1e-8 absolute;
/// achieved is printed.
#[test]
fn golden_l1_trend_filter_matches_clarabel() {
    let fx = load_fixture("convex.json");
    let mut n_pinned = 0usize;
    let mut worst = 0.0f64;
    for case in fx["l1_cases"].as_array().unwrap() {
        if case["trend_ref"].is_null() {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let y = series(&fx, case["series"].as_str().unwrap());
        let k = case["order"].as_u64().unwrap() as usize;
        let lam = case["lam"].as_f64().unwrap();
        assert_eq!(
            case["ref_status"].as_str().unwrap(),
            "optimal",
            "{name}: reference status"
        );
        assert!(
            case["ref_gap_rel"].as_f64().unwrap() <= 1e-11,
            "{name}: the reference itself is not converged"
        );
        let expected = as_f64_vec(&case["trend_ref"]);
        let fit = l1_trend_filter(&y, lam, l1_opts(k)).unwrap();
        let d = assert_slice_close(&fit.trend, &expected, 1e-8, name);
        worst = worst.max(d);
        n_pinned += 1;
    }
    assert!(n_pinned >= 10, "fixture carries no third-party trends");
    println!("L1 vs Clarabel: {n_pinned} cases, worst abs trend error {worst:e}");
}

/// Leg 3 — the closed-form limits. `lam_max = ||(DD')^{-1} D y||_inf`
/// matches the dense NumPy value at 1e-10 relative on every case, and at
/// `lam >= lam_max` the trend is the least-squares polynomial of degree
/// `order - 1` (`np.polyfit`) at 1e-8 with no knots and no iterations.
#[test]
fn golden_l1_trend_filter_lam_max_and_polynomial_limit() {
    let fx = load_fixture("convex.json");
    let mut n_limit = 0usize;
    let mut worst = 0.0f64;
    for case in fx["l1_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = series(&fx, case["series"].as_str().unwrap());
        let k = case["order"].as_u64().unwrap() as usize;
        let lam = case["lam"].as_f64().unwrap();
        let lam_max = case["lam_max"].as_f64().unwrap();
        let fit = l1_trend_filter(&y, lam, l1_opts(k)).unwrap();
        assert!(
            (fit.lam_max - lam_max).abs() <= 1e-10 * lam_max,
            "{name}: lam_max {} vs {lam_max}",
            fit.lam_max
        );
        if case["lam_frac"].as_f64().unwrap() >= 1.0 {
            let poly = as_f64_vec(&case["poly_fit"]);
            let d = assert_slice_close(&fit.trend, &poly, 1e-8, name);
            worst = worst.max(d);
            assert_eq!(fit.n_iter, 0, "{name}: the polynomial limit is closed-form");
            assert!(fit.knots.is_empty(), "{name}: knots {:?}", fit.knots);
            n_limit += 1;
        }
    }
    assert!(n_limit >= 3);
    println!("polynomial limit: {n_limit} cases, worst abs error {worst:e}");
}

/// L2 penalty (the Hodrick-Prescott form): the banded solve reproduces the
/// dense closed form `solve(I + lam D'D, y)` at 1e-10, the objective at
/// 1e-10 relative, and the dual certificate is at rounding.
#[test]
fn golden_l2_penalty_matches_dense_closed_form() {
    let fx = load_fixture("convex.json");
    let mut worst = 0.0f64;
    for case in fx["l2_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = series(&fx, case["series"].as_str().unwrap());
        let k = case["order"].as_u64().unwrap() as usize;
        let lam = case["lam"].as_f64().unwrap();
        let opts = TrendFilterOptions {
            order: k,
            penalty: Penalty::L2,
            ..TrendFilterOptions::default()
        };
        let fit = l1_trend_filter(&y, lam, opts).unwrap();
        let expected = as_f64_vec(&case["trend_ref"]);
        let d = assert_slice_close(&fit.trend, &expected, 1e-10, name);
        worst = worst.max(d);
        let obj = case["objective"].as_f64().unwrap();
        assert!(
            (fit.objective - obj).abs() <= 1e-10 * obj,
            "{name}: objective"
        );
        assert!(
            fit.duality_gap.abs() <= 1e-10 * obj,
            "{name}: gap {}",
            fit.duality_gap
        );
        assert!(fit.converged && fit.n_iter == 0);
    }
    println!("L2 vs dense closed form: worst abs trend error {worst:e}");
}

/// Boosting against the dense transcription: coefficient paths, the
/// operator trace, and the corrected AIC at 1e-12; the selection sequence
/// and the AIC-chosen step exactly; the fit reproduces `B_best y`.
#[test]
fn golden_boosting_matches_dense_transcription() {
    let fx = load_fixture("convex.json");
    let mut worst_coef = 0.0f64;
    let mut worst_df = 0.0f64;
    let mut worst_aic = 0.0f64;
    let mut worst_fit = 0.0f64;
    for case in fx["boost_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let design = &fx["boost_designs"][case["design"].as_str().unwrap()];
        let x = as_mat(&design["X"]);
        let y = as_f64_vec(&design["y"]);
        let xt = if design["X_test"].is_null() {
            None
        } else {
            Some(as_mat(&design["X_test"]))
        };
        let opts = BoostingOptions {
            learning_rate: case["learning_rate"].as_f64().unwrap(),
            n_steps: case["n_steps"].as_u64().unwrap() as usize,
            stop: BoostStop::Aic,
        };
        let fit = boosting(x.as_ref(), &y, opts, xt.as_ref().map(|m| m.as_ref())).unwrap();

        let sel: Vec<usize> = case["selected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(fit.selected, sel, "{name}: selected");
        assert_eq!(
            fit.best_step,
            case["best_step"].as_u64().unwrap() as usize,
            "{name}: best_step"
        );
        for (m, row) in case["coef_path"].as_array().unwrap().iter().enumerate() {
            let d = assert_slice_close(
                &fit.coef_path[m],
                &as_f64_vec(row),
                1e-12,
                &format!("{name} step {m}"),
            );
            worst_coef = worst_coef.max(d);
        }
        worst_df = worst_df.max(assert_slice_close(
            &fit.df_path,
            &as_f64_vec(&case["df_path"]),
            1e-12,
            name,
        ));
        worst_aic = worst_aic.max(assert_slice_close(
            &fit.aic_path,
            &as_f64_vec(&case["aic_path"]),
            1e-12,
            name,
        ));
        let rss = as_f64_vec(&case["rss_path"]);
        for (m, (a, e)) in fit.rss_path.iter().zip(&rss).enumerate() {
            assert!(
                (a - e).abs() <= 1e-12 * e.max(1.0),
                "{name}: rss step {m}: {a} vs {e}"
            );
        }
        assert_slice_close(&fit.coef, &as_f64_vec(&case["coef"]), 1e-12, name);
        worst_fit = worst_fit.max(assert_slice_close(
            &fit.fitted,
            &as_f64_vec(&case["fitted_operator"]),
            1e-10,
            name,
        ));
        if let Some(pred) = &fit.predicted {
            assert_slice_close(pred, &as_f64_vec(&case["predicted"]), 1e-12, name);
        } else {
            assert!(case["predicted"].is_null());
        }
    }
    println!(
        "boosting vs dense transcription: worst coef {worst_coef:e}, df {worst_df:e}, \
         aic {worst_aic:e}, fitted-vs-operator {worst_fit:e}"
    );
}
