//! Golden tests against `fixtures/tsecon-breaks.json`.
//!
//! The fixture is produced by `fixtures/generate_tsecon-breaks_fixtures.py`
//! with numpy/scipy only (no tsecon import): global partitions come from
//! EXACT brute-force enumeration over all admissible break placements with
//! `numpy.linalg.lstsq` segment regressions — an independent algorithmic
//! path the crate's dynamic program must reproduce exactly (same minimal
//! SSR to 1e-8 relative, same dates). Sequential statistics, selection,
//! per-regime coefficients (1e-8), Hansen p-values (1e-10), and the Bai
//! argmax cdf / CI critical values (1e-10 / 1e-8) are pinned to the
//! documented published formulas evaluated independently in Python.

use serde_json::Value;
use tsecon_breaks::{
    bai_argmax_cdf, bai_argmax_two_sided_crit, bai_perron, hansen_supf_pvalue, sup_f_test,
    BaiPerronConfig,
};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-breaks.json",
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

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("integer") as usize)
        .collect()
}

fn cols(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .expect("array of columns")
        .iter()
        .map(f64s)
        .collect()
}

fn g(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

#[test]
fn dp_matches_bruteforce_enumeration() {
    let fx = load();
    for case in fx["dp_cases"].as_array().expect("dp_cases") {
        let name = case["name"].as_str().expect("name");
        let y = f64s(&case["y"]);
        let x = cols(&case["x"]);
        let trim = g(&case["trim"]);
        let optima = case["optima"].as_array().expect("optima");
        let max_m = optima
            .iter()
            .map(|o| o["m"].as_u64().expect("m") as usize)
            .max()
            .expect("nonempty");
        let bp = bai_perron(
            &y,
            &x,
            BaiPerronConfig {
                max_breaks: max_m,
                trim,
            },
        )
        .expect(name);
        assert_eq!(bp.h as u64, case["h"].as_u64().expect("h"), "{name}: h");
        close(
            bp.ssr_path[0],
            g(&case["ssr0"]),
            1e-8,
            &format!("{name}: ssr0"),
        );
        for o in optima {
            let m = o["m"].as_u64().expect("m") as usize;
            close(
                bp.ssr_path[m],
                g(&o["ssr"]),
                1e-8,
                &format!("{name}: global SSR, m={m}"),
            );
            assert_eq!(
                bp.break_dates_by_m[m - 1],
                usizes(&o["dates"]),
                "{name}: break dates, m={m} (DP must equal brute force exactly)"
            );
        }
    }
}

fn check_bai_perron_case(case: &Value) {
    let name = case["name"].as_str().expect("name");
    let y = f64s(&case["y"]);
    let x = cols(&case["x"]);
    let bp = bai_perron(
        &y,
        &x,
        BaiPerronConfig {
            max_breaks: case["max_breaks"].as_u64().expect("max_breaks") as usize,
            trim: g(&case["trim"]),
        },
    )
    .expect(name);
    assert_eq!(bp.h as u64, case["h"].as_u64().expect("h"), "{name}: h");
    let ssr_path = f64s(&case["ssr_path"]);
    assert_eq!(bp.ssr_path.len(), ssr_path.len(), "{name}: ssr_path length");
    for (m, (a, e)) in bp.ssr_path.iter().zip(&ssr_path).enumerate() {
        close(*a, *e, 1e-8, &format!("{name}: ssr_path[{m}]"));
    }
    for (m1, dates) in case["break_dates_by_m"]
        .as_array()
        .expect("dates by m")
        .iter()
        .enumerate()
    {
        assert_eq!(
            bp.break_dates_by_m[m1],
            usizes(dates),
            "{name}: break_dates_by_m[{m1}]"
        );
    }
    let seq = f64s(&case["sup_f_seq"]);
    assert_eq!(bp.sup_f_seq.len(), seq.len(), "{name}: sup_f_seq length");
    for (l, (a, e)) in bp.sup_f_seq.iter().zip(&seq).enumerate() {
        close(*a, *e, 1e-7, &format!("{name}: supF({}|{l})", l + 1));
    }
    let crit = f64s(&case["sup_f_crit"]);
    for (l, (a, e)) in bp.sup_f_crit.iter().zip(&crit).enumerate() {
        close(*a, *e, 1e-12, &format!("{name}: sup_f_crit[{l}]"));
    }
    assert_eq!(
        bp.n_breaks as u64,
        case["n_breaks"].as_u64().expect("n_breaks"),
        "{name}: selected number of breaks"
    );
    assert_eq!(
        bp.break_dates,
        usizes(&case["break_dates"]),
        "{name}: selected break dates"
    );
    let regimes = case["regimes"].as_array().expect("regimes");
    assert_eq!(bp.regimes.len(), regimes.len(), "{name}: regime count");
    for (r, (actual, expected)) in bp.regimes.iter().zip(regimes).enumerate() {
        assert_eq!(
            actual.start as u64,
            expected["start"].as_u64().expect("start"),
            "{name}: regime {r} start"
        );
        assert_eq!(
            actual.end as u64,
            expected["end"].as_u64().expect("end"),
            "{name}: regime {r} end"
        );
        close(
            actual.ssr,
            g(&expected["ssr"]),
            1e-8,
            &format!("{name}: regime {r} ssr"),
        );
        for (j, (a, e)) in actual
            .params
            .iter()
            .zip(f64s(&expected["params"]))
            .enumerate()
        {
            close(*a, e, 1e-8, &format!("{name}: regime {r} beta[{j}]"));
        }
        for (j, (a, e)) in actual.se.iter().zip(f64s(&expected["se"])).enumerate() {
            close(*a, e, 1e-8, &format!("{name}: regime {r} se[{j}]"));
        }
    }
    let cis = case["ci"].as_array().expect("ci");
    assert_eq!(bp.ci.len(), cis.len(), "{name}: CI count");
    for (i, (a, e)) in bp.ci.iter().zip(cis).enumerate() {
        assert_eq!(
            a.date as u64,
            e["date"].as_u64().expect("date"),
            "{name}: ci[{i}] date"
        );
        close(
            a.scale,
            g(&e["scale"]),
            1e-7,
            &format!("{name}: ci[{i}] scale"),
        );
        assert_eq!(
            (a.lower90 as u64, a.upper90 as u64),
            (
                e["lower90"].as_u64().expect("l90"),
                e["upper90"].as_u64().expect("u90")
            ),
            "{name}: ci[{i}] 90% interval"
        );
        assert_eq!(
            (a.lower95 as u64, a.upper95 as u64),
            (
                e["lower95"].as_u64().expect("l95"),
                e["upper95"].as_u64().expect("u95")
            ),
            "{name}: ci[{i}] 95% interval"
        );
    }
}

#[test]
fn bai_perron_two_break_case_matches_reference() {
    let fx = load();
    check_bai_perron_case(&fx["bai_perron_case"]);
}

#[test]
fn bai_perron_null_case_selects_zero_breaks() {
    let fx = load();
    check_bai_perron_case(&fx["bai_perron_null_case"]);
}

#[test]
fn sup_f_matches_documented_formula() {
    let fx = load();
    for case in fx["sup_f_cases"].as_array().expect("sup_f_cases") {
        let name = case["name"].as_str().expect("name");
        let y = f64s(&case["y"]);
        let x = cols(&case["x"]);
        let r = sup_f_test(&y, &x, g(&case["trim"])).expect(name);
        assert_eq!(r.h as u64, case["h"].as_u64().expect("h"), "{name}: h");
        close(r.stat, g(&case["stat"]), 1e-8, &format!("{name}: sup-F"));
        assert_eq!(
            r.break_date as u64,
            case["break_date"].as_u64().expect("break_date"),
            "{name}: argmax date"
        );
        close(
            r.p_value,
            g(&case["p_value"]),
            1e-10,
            &format!("{name}: Hansen p-value"),
        );
        assert_eq!(r.dates, usizes(&case["dates"]), "{name}: candidate dates");
        let path = f64s(&case["f_path"]);
        assert_eq!(r.f_path.len(), path.len(), "{name}: f_path length");
        for (i, (a, e)) in r.f_path.iter().zip(&path).enumerate() {
            close(*a, *e, 1e-8, &format!("{name}: f_path[{i}]"));
        }
    }
}

#[test]
fn hansen_pvalues_match_response_surface() {
    let fx = load();
    for case in fx["hansen_pvalue_cases"].as_array().expect("hansen cases") {
        let stat = g(&case["stat"]);
        let q = case["q"].as_u64().expect("q") as usize;
        let tau = g(&case["tau"]);
        let p = hansen_supf_pvalue(stat, q, tau).expect("hansen p-value");
        close(
            p,
            g(&case["p"]),
            1e-10,
            &format!("hansen p (stat={stat}, q={q}, tau={tau})"),
        );
    }
}

#[test]
fn bai_argmax_cdf_and_crits_match_closed_form() {
    let fx = load();
    let grid = &fx["argmax_cdf"];
    let xs = f64s(&grid["x"]);
    let cdf = f64s(&grid["cdf"]);
    for (x, e) in xs.iter().zip(&cdf) {
        close(bai_argmax_cdf(*x), *e, 1e-10, &format!("G({x})"));
    }
    close(
        bai_argmax_two_sided_crit(0.90).expect("crit90"),
        g(&grid["crit90"]),
        1e-8,
        "two-sided 90% critical value (published anchor 7.7)",
    );
    close(
        bai_argmax_two_sided_crit(0.95).expect("crit95"),
        g(&grid["crit95"]),
        1e-8,
        "two-sided 95% critical value (published anchor 11.03)",
    );
}
