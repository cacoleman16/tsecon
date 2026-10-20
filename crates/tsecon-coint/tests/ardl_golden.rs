//! Golden-value tests for the ARDL / UECM fit and the Pesaran-Shin-Smith
//! bounds test (`ardl_bounds_test`) against `fixtures/ardl.json`,
//! generated from statsmodels 0.15.0 (`statsmodels.tsa.ardl.{UECM, ARDL,
//! ardl_select_order, pss_critical_values}` and `OLS`) by
//! `fixtures/generate_ardl_fixtures.py`.
//!
//! Every case pins the UECM estimates (params, covariance, hence the
//! standard errors and t values, the residuals), the fit statistics, the
//! PSS F and t statistics through two independent statsmodels routes (the
//! UECM Wald and the levels-ARDL affine `f_test`), the long run, the
//! critical values and response-surface p-values at the corrected key
//! `k = n_x`, the verdict and — for searched specifications — the selected
//! orders and the minimised criterion, across seven systems (macrodata,
//! the Danish money-demand data, five seeded systems including 10 and 11
//! regressors), all five PSS cases and every search mode. Beside them:
//! every constant of the transcribed PSS surface, the surface on a grid of
//! statistics around every switch point, and the evidence for the two
//! statsmodels defects the crate documents rather than inherits (the table
//! key `1 + n_x`; the reported `nobs = T - p`).

mod common;

use serde_json::Value;
use tsecon_coint::tsecon_linalg::faer::Mat;
use tsecon_coint::{ardl_bounds_test, pss, ArdlOptions, ArdlVerdict, DolsIc, PssCase};

use common::{as_mat, as_vec, load_fixture};

const TOL: f64 = 1e-10;

fn fx() -> Value {
    load_fixture("ardl.json")
}

fn system(fx: &Value, name: &str) -> (Vec<f64>, Mat<f64>) {
    let s = &fx["systems"][name];
    (as_vec(&s["y"]), as_mat(&s["x"]))
}

/// A JSON number, `null` read as NaN.
fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn fvec(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(f).collect()
}

fn opt_usize(v: &Value) -> Option<usize> {
    v.as_u64().map(|u| u as usize)
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|e| e.as_u64().expect("int") as usize)
        .collect()
}

/// Scalar: `|a - e| <= tol * max(|e|, floor)`.
fn close(a: f64, e: f64, tol: f64, floor: f64, ctx: &str) {
    if e.is_nan() {
        assert!(a.is_nan(), "{ctx}: expected NaN, got {a}");
        return;
    }
    let scale = e.abs().max(floor);
    assert!(
        (a - e).abs() <= tol * scale,
        "{ctx}: actual {a}, expected {e}, rel {:e}",
        (a - e).abs() / scale
    );
}

/// Vector: relative per entry, with a floor of 1e-3 of the vector's
/// largest entry so a coefficient that is near zero by chance is compared
/// on the scale of its neighbours.
fn close_vec(a: &[f64], e: &[f64], tol: f64, ctx: &str) {
    assert_eq!(a.len(), e.len(), "{ctx}: length");
    let big = e.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(e).enumerate() {
        close(*x, *y, tol, 1e-3 * big, &format!("{ctx}[{i}]"));
    }
}

fn close_mat(a: &[Vec<f64>], e: &Value, tol: f64, ctx: &str) {
    let rows = e.as_array().expect("rows");
    assert_eq!(a.len(), rows.len(), "{ctx}: rows");
    let big = rows
        .iter()
        .flat_map(fvec)
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    for (i, row) in rows.iter().enumerate() {
        for (j, (x, y)) in a[i].iter().zip(fvec(row)).enumerate() {
            close(*x, y, tol, 1e-3 * big, &format!("{ctx}[{i}][{j}]"));
        }
    }
}

/// p-values: atol 1e-14 / rtol 1e-9 (statsmodels evaluates `1 - cdf`; the
/// fixture stores the `sf` values, which differ by a relative 7.6e-11 at
/// p = 5.7e-7).
fn close_p(a: f64, e: f64, ctx: &str) {
    if e.is_nan() {
        assert!(a.is_nan(), "{ctx}: expected NaN, got {a}");
        return;
    }
    assert!(
        (a - e).abs() <= 1e-14 + 1e-9 * e.abs(),
        "{ctx}: actual {a}, expected {e}"
    );
}

fn ic_of(code: &str) -> DolsIc {
    DolsIc::parse(code).unwrap_or_else(|| panic!("unknown ic {code}"))
}

fn options(case: &Value) -> ArdlOptions {
    ArdlOptions {
        lags: opt_usize(&case["lags_arg"]),
        order: if case["order_arg"].is_null() {
            None
        } else {
            Some(usizes(&case["order_arg"]))
        },
        case: PssCase::from_index(case["case"].as_u64().expect("case") as usize).expect("case"),
        ic: case["ic_arg"].as_str().map_or(DolsIc::Bic, ic_of),
        max_lag: opt_usize(&case["max_lag_arg"]),
        max_order: opt_usize(&case["max_order_arg"]),
        alpha: f(&case["alpha"]),
    }
}

fn case_id(c: &Value) -> String {
    format!(
        "{}-case{}-lags{:?}-order{:?}-ic{:?}-caps{:?}/{:?}",
        c["system"].as_str().unwrap_or("?"),
        c["case"],
        opt_usize(&c["lags_arg"]),
        c["order_arg"],
        c["ic_arg"].as_str(),
        opt_usize(&c["max_lag_arg"]),
        opt_usize(&c["max_order_arg"])
    )
}

#[test]
fn every_case_matches_the_statsmodels_golden() {
    let fx = fx();
    let cases = fx["cases"].as_array().expect("cases");
    assert!(cases.len() >= 120, "fixture shrank: {} cases", cases.len());
    for c in cases {
        let ctx = case_id(c);
        let (y, x) = system(&fx, c["system"].as_str().expect("system"));
        let r =
            ardl_bounds_test(&y, x.as_ref(), &options(c)).unwrap_or_else(|e| panic!("{ctx}: {e}"));

        // orders, selection, shapes
        assert_eq!(
            r.lags,
            c["lags"].as_u64().expect("lags") as usize,
            "{ctx}: lags"
        );
        assert_eq!(r.order, usizes(&c["order"]), "{ctx}: order");
        assert_eq!(r.selected, c["selected"].as_bool().expect("sel"), "{ctx}");
        assert_eq!(r.ic.code(), c["ic"].as_str().expect("ic"), "{ctx}: ic");
        close(
            r.ic_value,
            f(&c["ic_value"]),
            1e-8,
            1.0,
            &format!("{ctx}: ic_value"),
        );
        for key in [
            "max_lag",
            "max_order",
            "n_models",
            "nobs",
            "n_total",
            "hold_back",
            "df_resid",
            "n_params",
            "n_det",
            "n_x",
            "n_restrictions",
        ] {
            let got = match key {
                "max_lag" => r.max_lag,
                "max_order" => r.max_order,
                "n_models" => r.n_models,
                "nobs" => r.nobs,
                "n_total" => r.n_total,
                "hold_back" => r.hold_back,
                "df_resid" => r.df_resid,
                "n_params" => r.n_params,
                "n_det" => r.n_det,
                "n_x" => r.n_x,
                _ => r.n_restrictions,
            };
            assert_eq!(got, c[key].as_u64().expect(key) as usize, "{ctx}: {key}");
        }
        assert_eq!(
            r.case.trend_code(),
            c["trend"].as_str().expect("trend"),
            "{ctx}"
        );
        let names: Vec<&str> = c["param_names"]
            .as_array()
            .expect("names")
            .iter()
            .map(|v| v.as_str().expect("name"))
            .collect();
        assert_eq!(r.param_names, names, "{ctx}: param_names");
        let lr_names: Vec<&str> = c["long_run_names"]
            .as_array()
            .expect("lr names")
            .iter()
            .map(|v| v.as_str().expect("name"))
            .collect();
        assert_eq!(r.long_run_names, lr_names, "{ctx}: long_run_names");

        // estimates
        close_vec(
            &r.params,
            &fvec(&c["params"]),
            TOL,
            &format!("{ctx}: params"),
        );
        close_vec(&r.se, &fvec(&c["se"]), TOL, &format!("{ctx}: se"));
        close_vec(
            &r.tvalues,
            &fvec(&c["tvalues"]),
            TOL,
            &format!("{ctx}: tvalues"),
        );
        for (i, (a, e)) in r.pvalues.iter().zip(fvec(&c["pvalues"])).enumerate() {
            close_p(*a, e, &format!("{ctx}: pvalues[{i}]"));
        }
        close_mat(&r.cov, &c["cov"], TOL, &format!("{ctx}: cov"));
        // residuals: relative to the largest one (a residual near zero carries
        // the absolute rounding of the fitted value, ~1e-14 here)
        let e_resid = fvec(&c["resid"]);
        let big = e_resid.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert_eq!(r.resid.len(), e_resid.len(), "{ctx}: resid length");
        for (i, (a, e)) in r.resid.iter().zip(&e_resid).enumerate() {
            close(*a, *e, TOL, big, &format!("{ctx}: resid[{i}]"));
        }
        for (name, got) in [
            ("ssr", r.ssr),
            ("llf", r.llf),
            ("aic", r.aic),
            ("bic", r.bic),
            ("hqic", r.hqic),
            ("adjustment", r.adjustment),
        ] {
            close(got, f(&c[name]), TOL, 1e-300, &format!("{ctx}: {name}"));
        }

        // the bounds test: against the exact-rational referee at 1e-10, and
        // through both floating-point statsmodels routes — the UECM Wald at
        // 5e-10 (its own rounding reaches 1.5e-10 where the restriction
        // block is ill-conditioned: case 2, the constant beside levels near
        // 800) and the levels-ARDL f_test at 1e-8 (cond up to 6e5).
        close(
            r.statistic,
            f(&c["statistic_exact"]),
            TOL,
            1e-300,
            &format!("{ctx}: F (exact)"),
        );
        close(
            r.statistic,
            f(&c["statistic"]),
            5e-10,
            1e-300,
            &format!("{ctx}: F"),
        );
        close(
            r.statistic,
            f(&c["statistic_levels"]),
            1e-8,
            1e-300,
            &format!("{ctx}: F (levels route)"),
        );
        close(
            r.t_statistic,
            f(&c["t_statistic_exact"]),
            TOL,
            1e-300,
            &format!("{ctx}: t (exact)"),
        );
        close(
            r.t_statistic,
            f(&c["t_statistic"]),
            TOL,
            1e-300,
            &format!("{ctx}: t"),
        );
        close(
            r.t_statistic,
            f(&c["t_statistic_levels"]),
            1e-8,
            1e-300,
            &format!("{ctx}: t (levels route)"),
        );

        // long run
        close_vec(
            &r.long_run,
            &fvec(&c["long_run"]),
            TOL,
            &format!("{ctx}: long_run"),
        );
        close_vec(
            &r.long_run_se,
            &fvec(&c["long_run_se"]),
            TOL,
            &format!("{ctx}: long_run_se"),
        );
        close_vec(
            &r.long_run_tvalues,
            &fvec(&c["long_run_tvalues"]),
            TOL,
            &format!("{ctx}: long_run_tvalues"),
        );
        for (i, (a, e)) in r
            .long_run_pvalues
            .iter()
            .zip(fvec(&c["long_run_pvalues"]))
            .enumerate()
        {
            close_p(*a, e, &format!("{ctx}: long_run_pvalues[{i}]"));
        }

        // critical values, p-values, verdict at the corrected key
        let lo = fvec(&c["crit_lower"]);
        let up = fvec(&c["crit_upper"]);
        for i in 0..4 {
            close(
                r.crit_lower[i],
                lo[i],
                1e-12,
                1e-300,
                &format!("{ctx}: crit_lower[{i}]"),
            );
            close(
                r.crit_upper[i],
                up[i],
                1e-12,
                1e-300,
                &format!("{ctx}: crit_upper[{i}]"),
            );
        }
        assert_eq!(r.crit_levels, [0.90, 0.95, 0.99, 0.999]);
        close_p(
            r.p_value_lower,
            f(&c["p_value_lower"]),
            &format!("{ctx}: p_value_lower"),
        );
        close_p(
            r.p_value_upper,
            f(&c["p_value_upper"]),
            &format!("{ctx}: p_value_upper"),
        );
        assert_eq!(
            r.verdict.code(),
            c["verdict"].as_str().expect("verdict"),
            "{ctx}: verdict"
        );
        assert_eq!(r.alpha, f(&c["alpha"]), "{ctx}: alpha");
        if r.n_x > 10 {
            assert_eq!(r.verdict, ArdlVerdict::Unavailable, "{ctx}");
            let note = r.crit_note.as_deref().expect("crit_note");
            assert!(note.contains("n_x = 11"), "{ctx}: {note}");
        } else {
            assert!(r.crit_note.is_none(), "{ctx}");
        }
    }
}

#[test]
fn the_search_winner_beats_its_runner_up_by_a_clear_margin() {
    // The fixture asserts every margin > 1e-6; pin that the stored winner
    // is the one tsecon picks (the main test) AND that the margins are
    // what make the comparison robust to the FWL-vs-QR rounding (3e-13).
    let fx = fx();
    let mut n = 0;
    for c in fx["cases"].as_array().expect("cases") {
        if c["selected"].as_bool() == Some(true) {
            assert!(f(&c["ic_margin"]) > 1e-6, "{}", case_id(c));
            n += 1;
        }
    }
    assert!(n >= 40, "only {n} searched cases");
}

#[test]
fn pss_table_constants_are_statsmodels_verbatim() {
    let fx = fx();
    let rows = fx["pss_table"].as_array().expect("table");
    assert_eq!(rows.len(), 100);
    assert_eq!(pss::PSS_TABLE.len(), 100);
    for row in rows {
        let k = row["k"].as_u64().expect("k") as usize;
        let case = row["case"].as_u64().expect("case") as usize;
        let i1 = row["i1"].as_bool().expect("i1");
        let e = pss::entry(k, case, i1).expect("key");
        assert_eq!((e.k as usize, e.case as usize, e.i1), (k, case, i1));
        assert_eq!(e.crit.to_vec(), fvec(&row["crit"]), "crit {k} {case} {i1}");
        assert_eq!(
            e.large_p.to_vec(),
            fvec(&row["large_p"]),
            "large_p {k} {case} {i1}"
        );
        assert_eq!(
            e.small_p.to_vec(),
            fvec(&row["small_p"]),
            "small_p {k} {case} {i1}"
        );
        assert_eq!(
            e.stat_star,
            f(&row["stat_star"]),
            "stat_star {k} {case} {i1}"
        );
        assert_eq!(pss::crit_values(k, case, i1), Some(e.crit));
    }
    assert!(pss::entry(0, 3, true).is_none());
    assert!(pss::entry(11, 3, true).is_none());
    assert!(pss::entry(1, 0, true).is_none());
    assert!(pss::entry(1, 6, false).is_none());
}

#[test]
fn pss_pvalue_surface_matches_on_a_grid_around_every_switch_point() {
    let fx = fx();
    for g in fx["pss_pvalue_grid"].as_array().expect("grid") {
        let k = g["k"].as_u64().expect("k") as usize;
        let case = g["case"].as_u64().expect("case") as usize;
        let i1 = g["i1"].as_bool().expect("i1");
        for (s, p) in fvec(&g["stat"]).iter().zip(fvec(&g["p"])) {
            let got = pss::pvalue(*s, k, case, i1).expect("key");
            close_p(got, p, &format!("p({s}; {k}, {case}, {i1})"));
        }
    }
}

#[test]
fn statsmodels_table_key_defect_is_pinned_as_a_canary() {
    // DEVIATION 1: statsmodels' bounds_test reads the table at 1 + n_x.
    // If upstream fixes the key these assertions (and the generator's) fail.
    let fx = fx();
    let mut n = 0;
    for c in fx["cases"].as_array().expect("cases") {
        let Some(can) = c.get("canary_key") else {
            continue;
        };
        let kx = c["n_x"].as_u64().expect("n_x") as usize;
        let case = c["case"].as_u64().expect("case") as usize;
        assert_eq!(can["sm_key_k"].as_u64(), Some(kx as u64 + 1));
        let wrong_lo = pss::crit_values(kx + 1, case, false).expect("key");
        let wrong_up = pss::crit_values(kx + 1, case, true).expect("key");
        assert_eq!(
            fvec(&can["sm_crit_lower"]),
            wrong_lo.to_vec(),
            "{}",
            case_id(c)
        );
        assert_eq!(
            fvec(&can["sm_crit_upper"]),
            wrong_up.to_vec(),
            "{}",
            case_id(c)
        );
        // and what tsecon reports is the n_x row, which differs
        assert_ne!(fvec(&c["crit_upper"]), wrong_up.to_vec());
        let stat = f(&c["statistic"]);
        let sm_pu = f(&can["sm_p_value_upper"]);
        close_p(
            pss::pvalue(stat, kx + 1, case, true).expect("key"),
            sm_pu,
            "sm p_upper",
        );
        n += 1;
    }
    assert!(n >= 110, "only {n} canary cases");

    // statsmodels' own simulator, at the key it passes (1 + n_x),
    // reproduces the n_x row of the table, not the 1 + n_x row.
    for s in fx["canary_sim"].as_array().expect("sim") {
        let kx = s["n_x"].as_u64().expect("n_x") as usize;
        let lo = fvec(&s["sim_lower"]);
        let up = fvec(&s["sim_upper"]);
        let right = (
            pss::crit_values(kx, 3, false).expect("k"),
            pss::crit_values(kx, 3, true).expect("k"),
        );
        let wrong = (
            pss::crit_values(kx + 1, 3, false).expect("k"),
            pss::crit_values(kx + 1, 3, true).expect("k"),
        );
        // 40 000 draws at nobs = 1000: within 0.12 of the n_x row at the
        // 90/95/99% levels (measured worst 0.116), more than 0.3 from the
        // n_x + 1 row at 95% and 99% (measured nearest 0.306; at 90% the
        // k = 3 and k = 4 upper rows are only 0.23 apart).
        for i in 0..3 {
            assert!((lo[i] - right.0[i]).abs() < 0.12, "sim lower[{i}] n_x={kx}");
            assert!((up[i] - right.1[i]).abs() < 0.12, "sim upper[{i}] n_x={kx}");
        }
        for i in 1..3 {
            assert!((lo[i] - wrong.0[i]).abs() > 0.3, "sim lower[{i}] vs n_x+1");
            assert!((up[i] - wrong.1[i]).abs() > 0.3, "sim upper[{i}] vs n_x+1");
        }
    }

    // PSS (2001) Table CI(iii)/CI(v), 5% bounds, k = 1..4 (hand
    // transcription; a reviewer verifies it against the PDF).
    for r in fx["published"]["rows"].as_array().expect("rows") {
        let k = r["k"].as_u64().expect("k") as usize;
        let case = r["case"].as_u64().expect("case") as usize;
        let lo = pss::crit_values(k, case, false).expect("key")[1];
        let up = pss::crit_values(k, case, true).expect("key")[1];
        assert!(
            (lo - f(&r["lower"])).abs() < 0.05,
            "published lower k={k} case={case}"
        );
        assert!(
            (up - f(&r["upper"])).abs() < 0.05,
            "published upper k={k} case={case}"
        );
    }
}

#[test]
fn statsmodels_nobs_defect_is_pinned_as_a_canary() {
    // DEVIATION 2: statsmodels' ARDL/UECM report nobs = T - p (AutoReg's
    // hold-back), not the T - max(p, q) rows the regression has.
    let fx = fx();
    let mut differs = 0;
    for c in fx["cases"].as_array().expect("cases") {
        let sm = &c["sm_reported"];
        let lags = c["lags"].as_u64().expect("lags") as usize;
        let qmax = usizes(&c["order"]).into_iter().max().expect("order");
        let t = c["n_total"].as_u64().expect("T") as usize;
        let nobs = c["nobs"].as_u64().expect("nobs") as usize;
        assert_eq!(sm["nobs"].as_u64(), Some((t - lags) as u64));
        if lags >= qmax {
            assert_eq!(nobs, t - lags);
            close(f(&sm["llf"]), f(&c["llf"]), 1e-9, 1.0, "sm llf");
        } else {
            assert_eq!(nobs, t - qmax);
            assert!((f(&sm["llf"]) - f(&c["llf"])).abs() > 1e-6);
            assert_eq!(
                f(&sm["df_resid"]) as usize,
                c["df_resid"].as_u64().expect("df") as usize + (qmax - lags)
            );
            differs += 1;
        }
    }
    assert!(
        differs >= 20,
        "only {differs} cases exercise the nobs defect"
    );
}

#[test]
fn the_documented_stories_reproduce() {
    let fx = fx();
    // macrodata: BIC picks ARDL(1, 1, 1); F rejects at 5% while t(y.L1) is
    // far from any PSS t bound — degenerate case #1, the card's headline.
    let s = &fx["stories"]["macrodata_bic_case3"];
    let (y, x) = system(&fx, "macrodata");
    let r = ardl_bounds_test(&y, x.as_ref(), &ArdlOptions::default()).expect("fit");
    assert_eq!((r.lags, r.order.clone()), (1, vec![1, 1]));
    close(r.statistic, f(&s["statistic"]), TOL, 1.0, "macrodata F");
    close(r.t_statistic, f(&s["t_statistic"]), TOL, 1.0, "macrodata t");
    close_p(r.p_value_upper, f(&s["p_value_upper"]), "macrodata p_upper");
    assert!(r.p_value_upper < 0.05 && r.t_statistic > -2.0);
    assert_eq!(r.verdict, ArdlVerdict::LevelRelationship);
    // statsmodels' key would have read p_upper = 0.024 off the k = 3 row.
    assert!(f(&s["sm_p_value_upper"]) < 0.03);

    // Danish: BIC's unrestricted pick (3, 1, 0) has no UECM; the
    // admissible subgrid picks (3, (1, 1, 1)).
    let d = &fx["stories"]["danish_bic_case3"];
    let (y, x) = system(&fx, "danish");
    let r = ardl_bounds_test(&y, x.as_ref(), &ArdlOptions::default()).expect("fit");
    assert_eq!((r.lags, r.order.clone()), (3, vec![1, 1, 1]));
    close(r.ic_value, f(&d["ic_value"]), 1e-8, 1.0, "danish bic");
    assert_eq!(usizes(&d["unrestricted"]), vec![3, 1, 0]);
}

#[test]
fn level_offsets_keep_the_f_at_the_exact_referee() {
    // Audit round 15 (F1): a level L far from zero beside the series'
    // variation (increments' sd about 0.015, L up to 1e6, y alone or y and
    // x shifted). With a constant (cases 2-5) the F is invariant to the
    // shift; the old Wald route through an explicit inverse of the
    // restriction block's covariance returned F = -35.8 (exact 9.548) in
    // case 2 at L = 1e6 and refused the design with x shifted too as
    // "exact linear combinations".
    // Golden: the exact-rational SSR-difference F of the shifted float
    // design. Cases 2-5 hold 1e-10 at every L (measured worst 1e-15: the
    // restricted columns are centred before the QR); case 1 (no constant,
    // nothing to centre: the F genuinely moves with L) and the t on y.L1
    // (from the raw design's QR, like every estimate) carry its
    // conditioning, about eps * L / variation: 1e-10 * max(1, L / 1e4)
    // (measured worst 2.4e-9 at L = 1e6, the t).
    let fx = fx();
    let o = &fx["offsets"];
    let y0 = as_vec(&o["y0"]);
    let x0 = as_mat(&o["x"]);
    let lags = o["lags"].as_u64().expect("lags") as usize;
    let order = usizes(&o["order"]);
    let rows = o["rows"].as_array().expect("rows");
    assert_eq!(rows.len(), 50);
    let (mut worst_f, mut worst_f1, mut worst_t) = (0.0_f64, 0.0_f64, 0.0_f64);
    for row in rows {
        let level = f(&row["level"]);
        let shift_x = row["shift_x"].as_bool().expect("shift_x");
        let case = row["case"].as_u64().expect("case") as usize;
        let ctx = format!("offset L = {level:e}, shift_x = {shift_x}, case {case}");
        let y: Vec<f64> = y0.iter().map(|v| v + level).collect();
        let x = Mat::from_fn(x0.nrows(), x0.ncols(), |i, j| {
            x0[(i, j)] + if shift_x { level } else { 0.0 }
        });
        let opts = ArdlOptions {
            lags: Some(lags),
            order: Some(order.clone()),
            case: PssCase::from_index(case).expect("case"),
            ..ArdlOptions::default()
        };
        let r = ardl_bounds_test(&y, x.as_ref(), &opts).unwrap_or_else(|e| panic!("{ctx}: {e}"));
        let loose = TOL * (level / 1e4).max(1.0);
        let e_f = f(&row["statistic_exact"]);
        let e_t = f(&row["t_statistic_exact"]);
        let tol_f = if case == 1 { loose } else { TOL };
        close(
            r.statistic,
            e_f,
            tol_f,
            1e-300,
            &format!("{ctx}: F (exact)"),
        );
        close(
            r.t_statistic,
            e_t,
            loose,
            1e-300,
            &format!("{ctx}: t (exact)"),
        );
        let err_f = (r.statistic - e_f).abs() / e_f.abs();
        if case >= 2 {
            worst_f = worst_f.max(err_f);
        } else {
            worst_f1 = worst_f1.max(err_f);
        }
        worst_t = worst_t.max((r.t_statistic - e_t).abs() / e_t.abs());
        if level == 0.0 {
            // the referee's design is statsmodels' own UECM design
            close(
                r.statistic,
                f(&row["sm_statistic"]),
                5e-10,
                1e-300,
                &format!("{ctx}: F (statsmodels)"),
            );
        }
    }
    println!(
        "offsets: worst relative error of F in cases 2-5 {worst_f:e}, of F in case 1 \
         {worst_f1:e}, of t {worst_t:e}"
    );
}
