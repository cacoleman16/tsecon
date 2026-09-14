//! Golden tests for the VAR residual diagnostics against
//! `fixtures/var_diag.json` (statsmodels `VARResults.test_whiteness` with
//! `adjusted=False/True`, `test_normality`, `roots`, `is_stable`, and
//! `VAR.select_order` — independent package), on a seeded simulated
//! VAR(2), a Student-t-driven VAR(2), and the transformed US macro system.

mod common;

use common::{as_mat, assert_rel_close, load_fixture};
use serde_json::Value;
use tsecon_var::{select_order, Trend, VarResults, VarSpec};

fn trend_of(case: &Value) -> Trend {
    if case["trend"] == "c" {
        Trend::Constant
    } else {
        Trend::None
    }
}

fn fit_case(fx: &Value, case: &Value) -> VarResults {
    let data = as_mat(&fx["series"][case["series"].as_str().expect("series name")]);
    let p = case["p"].as_u64().expect("p") as usize;
    VarSpec::new(p, trend_of(case))
        .unwrap()
        .fit(data.as_ref())
        .unwrap()
}

/// p-values: absolute 1e-10 (via the relative-with-floor helper) plus a
/// relative 1e-6 check on the log scale whenever both are representable,
/// so a tiny tail probability cannot pass on the absolute floor alone.
fn assert_pvalue_close(actual: f64, expected: f64, what: &str) {
    assert_rel_close(actual, expected, 1e-10, what);
    if actual > 1e-300 && expected > 1e-300 {
        let d = (actual.ln() - expected.ln()).abs();
        assert!(d < 1e-6, "{what}: log p-value differs by {d:e}");
    }
}

fn cases() -> (Value, Vec<Value>) {
    let fx = load_fixture("var_diag.json");
    let cs = fx["cases"].as_array().expect("cases").clone();
    assert_eq!(cs.len(), 6);
    (fx, cs)
}

/// Both Portmanteau statistics, their degrees of freedom and p-values match
/// statsmodels `test_whiteness` at every (case, nlags) block.
#[test]
fn golden_portmanteau() {
    let (fx, cs) = cases();
    let mut n_blocks = 0;
    for case in &cs {
        let name = case["name"].as_str().unwrap();
        let res = fit_case(&fx, case);
        assert_eq!(
            res.nobs,
            case["nobs"].as_u64().unwrap() as usize,
            "{name}: nobs"
        );
        for block in case["portmanteau"].as_array().unwrap() {
            let nlags = block["nlags"].as_u64().unwrap() as usize;
            let q = res.portmanteau_test(nlags).unwrap();
            let what = format!("{name} nlags={nlags}");
            assert_eq!(q.nlags, nlags);
            assert_eq!(q.df, block["df"].as_u64().unwrap() as usize, "{what}: df");
            assert_rel_close(
                q.statistic,
                block["statistic"].as_f64().unwrap(),
                1e-10,
                &format!("{what}: Q"),
            );
            assert_rel_close(
                q.adjusted,
                block["adjusted"].as_f64().unwrap(),
                1e-10,
                &format!("{what}: adjusted Q"),
            );
            assert_pvalue_close(
                q.pvalue,
                block["pvalue"].as_f64().unwrap(),
                &format!("{what}: p"),
            );
            assert_pvalue_close(
                q.adjusted_pvalue,
                block["adjusted_pvalue"].as_f64().unwrap(),
                &format!("{what}: adjusted p"),
            );
            // The adjustment only ever raises the statistic.
            assert!(q.adjusted > q.statistic, "{what}: adjusted <= unadjusted");
            n_blocks += 1;
        }
    }
    assert_eq!(n_blocks, 11);
}

/// The multivariate Jarque-Bera omnibus statistic, degrees of freedom and
/// p-value match statsmodels `test_normality`; the skewness and kurtosis
/// components and their per-series moments match the transcription of the
/// same code; the components add up to the omnibus statistic.
#[test]
fn golden_normality() {
    let (fx, cs) = cases();
    for case in &cs {
        let name = case["name"].as_str().unwrap();
        let res = fit_case(&fx, case);
        let n = res.normality_test().unwrap();
        let e = &case["normality"];
        assert_eq!(n.df, e["df"].as_u64().unwrap() as usize, "{name}: df");
        assert_eq!(n.df, 2 * res.neqs);
        assert_rel_close(
            n.statistic,
            e["statistic"].as_f64().unwrap(),
            1e-10,
            &format!("{name}: JB"),
        );
        assert_pvalue_close(
            n.pvalue,
            e["pvalue"].as_f64().unwrap(),
            &format!("{name}: JB p"),
        );
        assert_rel_close(
            n.skewness,
            e["skewness"].as_f64().unwrap(),
            1e-10,
            &format!("{name}: skew"),
        );
        assert_rel_close(
            n.kurtosis,
            e["kurtosis"].as_f64().unwrap(),
            1e-10,
            &format!("{name}: kurt"),
        );
        assert_pvalue_close(
            n.skewness_pvalue,
            e["skewness_pvalue"].as_f64().unwrap(),
            &format!("{name}: skew p"),
        );
        assert_pvalue_close(
            n.kurtosis_pvalue,
            e["kurtosis_pvalue"].as_f64().unwrap(),
            &format!("{name}: kurt p"),
        );
        for (i, v) in e["skewness_components"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            assert_rel_close(
                n.skewness_components[i],
                v.as_f64().unwrap(),
                1e-10,
                &format!("{name}: b1[{i}]"),
            );
        }
        for (i, v) in e["kurtosis_components"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            assert_rel_close(
                n.kurtosis_components[i],
                v.as_f64().unwrap(),
                1e-10,
                &format!("{name}: b2[{i}]"),
            );
        }
        assert!((n.skewness + n.kurtosis - n.statistic).abs() < 1e-12 * n.statistic.max(1.0));
    }
    // The Student-t(4) system is rejected, the Gaussian one is not, and the
    // t case's rejection is kurtosis-driven.
    let t4 = cs.iter().find(|c| c["name"] == "var2c_sim_t4").unwrap();
    let nt = fit_case(&fx, t4).normality_test().unwrap();
    assert!(
        nt.pvalue < 1e-6,
        "t(4) innovations not rejected: p = {}",
        nt.pvalue
    );
    assert!(nt.kurtosis > nt.skewness);
    let g = cs.iter().find(|c| c["name"] == "var2c_sim").unwrap();
    assert!(fit_case(&fx, g).normality_test().unwrap().pvalue > 0.05);
}

/// Root moduli, companion eigenvalue moduli and the stability flag match
/// statsmodels `roots` / `is_stable`, and the bundle agrees with its parts.
#[test]
fn golden_roots_and_bundle() {
    let (fx, cs) = cases();
    for case in &cs {
        let name = case["name"].as_str().unwrap();
        let res = fit_case(&fx, case);
        let first_nlags = case["portmanteau"][0]["nlags"].as_u64().unwrap() as usize;
        let d = res.diagnostics(first_nlags).unwrap();
        let roots = case["roots"].as_array().unwrap();
        assert_eq!(d.roots.len(), roots.len(), "{name}: root count");
        for (i, r) in roots.iter().enumerate() {
            assert_rel_close(
                d.roots[i],
                r.as_f64().unwrap(),
                1e-8,
                &format!("{name}: root[{i}]"),
            );
        }
        for (i, r) in case["eigenvalue_moduli"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            assert_rel_close(
                d.eigenvalue_moduli[i],
                r.as_f64().unwrap(),
                1e-8,
                &format!("{name}: eig[{i}]"),
            );
        }
        assert_eq!(
            d.is_stable,
            case["is_stable"].as_bool().unwrap(),
            "{name}: is_stable"
        );
        assert_eq!(d.is_stable, *d.roots.last().unwrap() > 1.0);
        assert_eq!(d.is_stable, d.eigenvalue_moduli[0] < 1.0);
        assert_eq!(d.portmanteau, res.portmanteau_test(first_nlags).unwrap());
        assert_eq!(d.normality, res.normality_test().unwrap());
        assert_eq!(
            (d.nobs, d.neqs, d.lags),
            (res.nobs, res.neqs, res.spec.lags)
        );
    }
}

/// The lag-order selection table and picks match statsmodels
/// `VAR.select_order` on the common sample (the surface `var_select_order`
/// binds).
#[test]
fn golden_select_order_table() {
    let fx = load_fixture("var_diag.json");
    for block in fx["select_order"].as_array().unwrap() {
        let data = as_mat(&fx["series"][block["series"].as_str().unwrap()]);
        let maxlags = block["max_lags"].as_u64().unwrap() as usize;
        let sel = select_order(data.as_ref(), maxlags, trend_of(block)).unwrap();
        let cands: Vec<usize> = block["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let got: Vec<usize> = sel.candidates.iter().map(|c| c.lags).collect();
        assert_eq!(got, cands);
        for (i, c) in sel.candidates.iter().enumerate() {
            let what = format!("select maxlags={maxlags} p={}", c.lags);
            assert_rel_close(
                c.aic,
                block["aic_values"][i].as_f64().unwrap(),
                1e-8,
                &format!("{what} aic"),
            );
            assert_rel_close(
                c.bic,
                block["bic_values"][i].as_f64().unwrap(),
                1e-8,
                &format!("{what} bic"),
            );
            assert_rel_close(
                c.hqic,
                block["hqic_values"][i].as_f64().unwrap(),
                1e-8,
                &format!("{what} hqic"),
            );
            assert_rel_close(
                c.fpe,
                block["fpe_values"][i].as_f64().unwrap(),
                1e-8,
                &format!("{what} fpe"),
            );
        }
        let s = &block["selected"];
        assert_eq!(sel.aic, s["aic"].as_u64().unwrap() as usize);
        assert_eq!(sel.bic, s["bic"].as_u64().unwrap() as usize);
        assert_eq!(sel.hqic, s["hqic"].as_u64().unwrap() as usize);
        assert_eq!(sel.fpe, s["fpe"].as_u64().unwrap() as usize);
    }
}

/// Refusals name `nlags` and say what to pass.
#[test]
fn portmanteau_refusals_name_nlags() {
    let (fx, cs) = cases();
    let res = fit_case(&fx, &cs[0]);
    let p = res.spec.lags;
    for bad in [0, p] {
        let e = res.portmanteau_test(bad).unwrap_err().to_string();
        assert!(e.contains("nlags"), "{e}");
        assert!(e.contains("k^2 (nlags - p)"), "{e}");
    }
    let e = res.portmanteau_test(res.nobs).unwrap_err().to_string();
    assert!(
        e.contains("nlags") && e.contains("effective sample size"),
        "{e}"
    );
    assert!(res.portmanteau_test(p + 1).is_ok());
}
