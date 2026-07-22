//! Golden tests against `fixtures/tsecon-panelroot.json`.
//!
//! The fixture is produced by `fixtures/generate_tsecon-panelroot_fixtures.py`
//! with numpy/scipy/statsmodels only (no tsecon import). The per-unit ADF
//! comes from statsmodels `adfuller` — which `tsecon_diag::adf` reproduces —
//! so the per-unit tau/p-value/lag/nobs are pinned tightly. Fisher is exact
//! arithmetic on those p-values (the strong sub-golden). IPS `Wtbar` and LLC
//! `t*_delta` are computed independently in Python from the SAME transcribed
//! moment tables (IPS 2003 Table 3, LLC 2002 Table 2), pinning the Rust
//! combination pipeline to the shared tables. The `plm_anchor` block holds
//! `plm::purtest` outputs (an independent R implementation) for three
//! fixed-lag panels; the crate must reproduce them within the stated
//! tolerance, an external cross-check on both the tables and the pipeline.

use serde_json::Value;
use tsecon_diag::{AdfLagSelection, AdfRegression};
use tsecon_panelroot::{
    panel_unit_root, PanelRootDetail, PanelRootOpts, PanelRootResult, PanelRootTest,
};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-panelroot.json",
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

fn units(case: &Value) -> Vec<Vec<f64>> {
    case["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(f64s)
        .collect()
}

fn regression(case: &Value) -> AdfRegression {
    match case["regression"].as_str().expect("regression") {
        "n" => AdfRegression::NoConstant,
        "c" => AdfRegression::Constant,
        "ct" => AdfRegression::ConstantTrend,
        other => panic!("bad regression {other}"),
    }
}

fn lag_selection(case: &Value) -> AdfLagSelection {
    match case["lag_mode"].as_str().expect("lag_mode") {
        "fixed" => AdfLagSelection::Fixed(case["lag"].as_u64().expect("lag") as usize),
        "aic" => AdfLagSelection::Aic(case["max_lags"].as_u64().map(|m| m as usize)),
        "bic" => AdfLagSelection::Bic(case["max_lags"].as_u64().map(|m| m as usize)),
        other => panic!("bad lag_mode {other}"),
    }
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e}"
    );
}

fn run(case: &Value, test: PanelRootTest) -> PanelRootResult {
    panel_unit_root(
        &units(case),
        test,
        regression(case),
        lag_selection(case),
        &PanelRootOpts::default(),
    )
    .expect("panel_unit_root ok")
}

/// The per-unit ADF front half (tau, p, lag, nobs) must match statsmodels
/// for every case — this pins the shared foundation all three tests rest on.
#[test]
#[allow(clippy::needless_range_loop)] // i indexes several parallel arrays.
fn per_unit_adf_matches_statsmodels() {
    let fx = load();
    for case in fx["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let r = run(case, PanelRootTest::Fisher);
        let exp_t = f64s(&case["per_unit"]["tstat"]);
        let exp_lag = usizes(&case["per_unit"]["lags"]);
        let exp_nobs = usizes(&case["per_unit"]["nobs"]);
        for i in 0..r.n_units {
            close(
                r.per_unit_tstat[i],
                exp_t[i],
                1e-7,
                &format!("{name} tstat[{i}]"),
            );
            assert_eq!(r.per_unit_lags[i], exp_lag[i], "{name} lag[{i}]");
            assert_eq!(r.per_unit_nobs[i], exp_nobs[i], "{name} nobs[{i}]");
        }
    }
}

/// Fisher — the strong sub-golden: Maddala-Wu P, its chi2 p-value, Choi Z,
/// and Choi's p-value are exact functions of the (clamped) per-unit p-values.
#[test]
#[allow(clippy::needless_range_loop)] // i indexes several parallel arrays.
fn fisher_matches_reference() {
    let fx = load();
    for case in fx["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let r = run(case, PanelRootTest::Fisher);
        let fi = &case["fisher"];
        close(
            r.statistic,
            fi["maddala_wu"].as_f64().unwrap(),
            1e-6,
            &format!("{name} madwu"),
        );
        close(
            r.p_value,
            fi["mw_pvalue"].as_f64().unwrap(),
            1e-6,
            &format!("{name} madwu_p"),
        );
        let exp_clamped = f64s(&fi["clamped_pvalue"]);
        for i in 0..r.n_units {
            close(
                r.per_unit_pvalue[i],
                exp_clamped[i],
                1e-7,
                &format!("{name} clamped_p[{i}]"),
            );
        }
        match r.detail {
            PanelRootDetail::Fisher {
                choi_z,
                choi_z_pvalue,
            } => {
                close(
                    choi_z,
                    fi["choi_z"].as_f64().unwrap(),
                    1e-6,
                    &format!("{name} choi_z"),
                );
                close(
                    choi_z_pvalue,
                    fi["choi_z_pvalue"].as_f64().unwrap(),
                    1e-6,
                    &format!("{name} choi_p"),
                );
            }
            _ => panic!("expected Fisher detail"),
        }
    }
}

/// IPS — t_bar and the Table-3-standardized W_tbar and its p-value.
#[test]
fn ips_matches_reference() {
    let fx = load();
    for case in fx["cases"].as_array().expect("cases") {
        if case["regression"].as_str() == Some("n") || !case.get("ips").is_some() {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let r = run(case, PanelRootTest::Ips);
        let ip = &case["ips"];
        close(
            r.statistic,
            ip["w_tbar"].as_f64().unwrap(),
            1e-9,
            &format!("{name} wtbar"),
        );
        close(
            r.p_value,
            ip["p_value"].as_f64().unwrap(),
            1e-9,
            &format!("{name} ips_p"),
        );
        match r.detail {
            PanelRootDetail::Ips { t_bar } => {
                close(
                    t_bar,
                    ip["t_bar"].as_f64().unwrap(),
                    1e-7,
                    &format!("{name} t_bar"),
                );
            }
            _ => panic!("expected Ips detail"),
        }
    }
}

/// LLC — the six-step pipeline reproduces the independent NumPy reference
/// (which itself matches plm::purtest levinlin) on the mechanical quantities.
#[test]
fn llc_matches_reference() {
    let fx = load();
    for case in fx["cases"].as_array().expect("cases") {
        if !case.get("llc").is_some() {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let r = run(case, PanelRootTest::Llc);
        let lc = &case["llc"];
        close(
            r.statistic,
            lc["t_star"].as_f64().unwrap(),
            1e-8,
            &format!("{name} t_star"),
        );
        close(
            r.p_value,
            lc["p_value"].as_f64().unwrap(),
            1e-8,
            &format!("{name} llc_p"),
        );
        match r.detail {
            PanelRootDetail::Llc {
                delta_hat,
                t_delta,
                s_n,
                t_bar_periods,
            } => {
                close(
                    delta_hat,
                    lc["delta_hat"].as_f64().unwrap(),
                    1e-8,
                    &format!("{name} delta"),
                );
                close(
                    t_delta,
                    lc["t_delta"].as_f64().unwrap(),
                    1e-8,
                    &format!("{name} t_delta"),
                );
                close(
                    s_n,
                    lc["s_n"].as_f64().unwrap(),
                    1e-8,
                    &format!("{name} s_n"),
                );
                close(
                    t_bar_periods,
                    lc["t_bar_periods"].as_f64().unwrap(),
                    1e-9,
                    &format!("{name} t_bar_periods"),
                );
            }
            _ => panic!("expected Llc detail"),
        }
    }
}

/// The unbalanced case exercises the IPS/Fisher ragged-panel path.
#[test]
fn unbalanced_ips_fisher() {
    let fx = load();
    let case = fx["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"].as_str() == Some("unbal_c_L1"))
        .expect("unbalanced case present");
    let f = run(case, PanelRootTest::Fisher);
    close(
        f.statistic,
        case["fisher"]["maddala_wu"].as_f64().unwrap(),
        1e-6,
        "unbal madwu",
    );
    let i = run(case, PanelRootTest::Ips);
    close(
        i.statistic,
        case["ips"]["w_tbar"].as_f64().unwrap(),
        1e-9,
        "unbal wtbar",
    );
}

/// External cross-check: reproduce `plm::purtest` (R) outputs for the three
/// fixed-lag anchor panels. This validates the transcribed tables and the
/// pipeline against an independent, published implementation.
#[test]
fn matches_plm_anchor() {
    let fx = load();
    let by_name = |name: &str| {
        fx["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("anchor case {name} present"))
            .clone()
    };
    for a in fx["plm_anchor"].as_array().expect("plm_anchor") {
        let name = a["case"].as_str().unwrap();
        let case = by_name(name);
        let tol = a["tol"].as_f64().unwrap();

        let f = run(&case, PanelRootTest::Fisher);
        close(
            f.statistic,
            a["madwu"].as_f64().unwrap(),
            tol,
            &format!("{name} plm madwu"),
        );
        close(
            f.p_value,
            a["madwu_p"].as_f64().unwrap(),
            tol,
            &format!("{name} plm madwu_p"),
        );
        if let PanelRootDetail::Fisher {
            choi_z,
            choi_z_pvalue,
        } = f.detail
        {
            close(
                choi_z,
                a["choi_z"].as_f64().unwrap(),
                tol,
                &format!("{name} plm choi_z"),
            );
            close(
                choi_z_pvalue,
                a["choi_p"].as_f64().unwrap(),
                tol,
                &format!("{name} plm choi_p"),
            );
        }

        let l = run(&case, PanelRootTest::Llc);
        close(
            l.statistic,
            a["llc_tstar"].as_f64().unwrap(),
            tol,
            &format!("{name} plm llc"),
        );

        if a.get("ips_wtbar").is_some() {
            let ip = run(&case, PanelRootTest::Ips);
            close(
                ip.statistic,
                a["ips_wtbar"].as_f64().unwrap(),
                tol,
                &format!("{name} plm wtbar"),
            );
        }
    }
}
