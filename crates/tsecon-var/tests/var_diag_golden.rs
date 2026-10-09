//! Golden tests for the VAR residual diagnostics against
//! `fixtures/var_diag.json` (statsmodels `VARResults.test_whiteness` with
//! `adjusted=False/True`, `test_normality`, `roots`, `is_stable`, and
//! `VAR.select_order` — independent package), on a seeded simulated
//! VAR(2), a Student-t-driven VAR(2), and the transformed US macro system.
//!
//! The residual-normality family added in 0.11.0 is pinned on two more
//! legs: JMulTi's own printed output (independent program, bundled with
//! statsmodels 0.15.0) on the 16 stored residual matrices of the `jmulti`
//! list — the symmetric-root Jarque-Bera against the block JMulTi labels
//! "Doornik & Hansen (1994)", the Cholesky form against its "Lütkepohl
//! (1993)" block and the k = 1 form against its per-variable Jarque-Bera
//! rows, each at the printed precision `5e-5` — and the generator's
//! transcriptions (statsmodels-convention symmetric root; the Doornik-Hansen
//! 2008 formulas, whose z1 the generator matched to `scipy.stats.skewtest`)
//! at 1e-10.

mod common;

use common::{as_mat, assert_rel_close, load_fixture};
use serde_json::Value;
use tsecon_var::tsecon_linalg::faer::Mat;
use tsecon_var::{
    doornik_hansen_test, normality_test_cholesky, normality_test_symmetric, select_order,
    DoornikHansenTest, NormalityTest, Trend, VarResults, VarSpec,
};

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
            d.normality_symmetric,
            Some(res.normality_test_symmetric().unwrap())
        );
        assert_eq!(d.doornik_hansen, Some(res.doornik_hansen_test().unwrap()));
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

// ------------------------------------------------------- normality family

/// JMulTi prints four decimals: the pin on its numbers.
const PRINTED: f64 = 5e-5 + 1e-12;

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn assert_printed(actual: f64, printed: f64, what: &str) {
    assert!(
        (actual - printed).abs() <= PRINTED,
        "{what}: {actual} misses JMulTi's printed {printed} by {:e} (> 5e-5)",
        (actual - printed).abs()
    );
}

/// A normality block against a stored transcription: statistics and
/// components at 1e-10 relative, p-values on the log scale, df exact.
fn assert_normality_block(n: &NormalityTest, e: &Value, k: usize, what: &str) {
    assert_eq!(n.df, 2 * k, "{what}: df");
    assert_eq!(n.df, e["df"].as_u64().unwrap() as usize, "{what}: df");
    for (got, key) in [
        (n.statistic, "statistic"),
        (n.skewness, "skewness"),
        (n.kurtosis, "kurtosis"),
    ] {
        assert_rel_close(
            got,
            e[key].as_f64().unwrap(),
            1e-10,
            &format!("{what}: {key}"),
        );
    }
    for (got, key) in [
        (n.pvalue, "pvalue"),
        (n.skewness_pvalue, "skewness_pvalue"),
        (n.kurtosis_pvalue, "kurtosis_pvalue"),
    ] {
        assert_pvalue_close(got, e[key].as_f64().unwrap(), &format!("{what}: {key}"));
    }
    for (got, key) in [
        (&n.skewness_components, "skewness_components"),
        (&n.kurtosis_components, "kurtosis_components"),
    ] {
        let want = f64s(&e[key]);
        assert_eq!(got.len(), k, "{what}: {key} length");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_rel_close(*a, *b, 1e-10, &format!("{what}: {key}[{i}]"));
        }
    }
    assert!((n.skewness + n.kurtosis - n.statistic).abs() <= 1e-12 * n.statistic.max(1.0));
}

/// The Doornik-Hansen (2008) result against the generator's transcription.
fn assert_dh_block(d: &DoornikHansenTest, e: &Value, k: usize, what: &str) {
    assert_eq!(d.df, 2 * k, "{what}: df");
    assert_eq!(d.df, e["df"].as_u64().unwrap() as usize, "{what}: df");
    for (got, key) in [
        (d.statistic, "statistic"),
        (d.skewness, "skewness"),
        (d.kurtosis, "kurtosis"),
    ] {
        assert_rel_close(
            got,
            e[key].as_f64().unwrap(),
            1e-10,
            &format!("{what}: DH {key}"),
        );
    }
    for (got, key) in [
        (d.pvalue, "pvalue"),
        (d.skewness_pvalue, "skewness_pvalue"),
        (d.kurtosis_pvalue, "kurtosis_pvalue"),
    ] {
        assert_pvalue_close(got, e[key].as_f64().unwrap(), &format!("{what}: DH {key}"));
    }
    for (got, key) in [
        (&d.z_skewness, "z_skewness"),
        (&d.z_kurtosis, "z_kurtosis"),
        (&d.sqrt_b1, "sqrt_b1"),
        (&d.b2, "b2"),
    ] {
        let want = f64s(&e[key]);
        assert_eq!(got.len(), k, "{what}: {key} length");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_rel_close(*a, *b, 1e-10, &format!("{what}: DH {key}[{i}]"));
        }
    }
    let zz: f64 = d.z_skewness.iter().map(|z| z * z).sum();
    assert_eq!(zz.to_bits(), d.skewness.to_bits(), "{what}: z1'z1");
    assert!((d.skewness + d.kurtosis - d.statistic).abs() <= 1e-12 * d.statistic.max(1.0));
}

fn jmulti_cases() -> Vec<Value> {
    let fx = load_fixture("var_diag.json");
    let cs = fx["jmulti"].as_array().expect("jmulti").clone();
    assert_eq!(cs.len(), 16, "all 16 JMulTi specifications");
    cs
}

/// The symmetric-root Jarque-Bera reproduces every "Doornik & Hansen (1994)"
/// block JMulTi printed (joint, skewness, kurtosis statistics AND their three
/// p-values, df = 2k) on all 16 bundled specifications, to the printed
/// precision; and the generator's transcription at 1e-10.
#[test]
fn golden_symmetric_reproduces_jmulti_doornik_hansen_1994_blocks() {
    let mut worst: f64 = 0.0;
    for c in jmulti_cases() {
        let name = c["name"].as_str().unwrap();
        let u = as_mat(&c["resid"]);
        let k = c["k"].as_u64().unwrap() as usize;
        assert_eq!(
            (u.nrows(), u.ncols()),
            (c["nobs"].as_u64().unwrap() as usize, k)
        );
        let r = normality_test_symmetric(u.as_ref()).unwrap();
        let p = &c["printed_symmetric"];
        assert_eq!(r.df as f64, p["df"].as_f64().unwrap(), "{name}: df");
        for (got, key) in [
            (r.statistic, "joint"),
            (r.skewness, "skew"),
            (r.kurtosis, "kurt"),
            (r.pvalue, "p_joint"),
            (r.skewness_pvalue, "p_skew"),
            (r.kurtosis_pvalue, "p_kurt"),
        ] {
            let want = p[key].as_f64().unwrap();
            assert_printed(got, want, &format!("{name}: symmetric {key}"));
            worst = worst.max((got - want).abs());
        }
        assert_normality_block(&r, &c["symmetric"], k, &format!("{name}: symmetric"));
    }
    println!("JMulTi 'Doornik & Hansen (1994)' blocks, 16 specifications: worst |diff| vs print {worst:.3e}");
}

/// The Cholesky form reproduces JMulTi's "Lütkepohl (1993)" block on the same
/// residuals (which calibrates the stored residuals: they are the ones JMulTi
/// tested), statsmodels `test_normality`, and the transcription; at k = 1 it
/// is JMulTi's per-variable univariate Jarque-Bera row.
#[test]
fn golden_cholesky_reproduces_jmulti_lutkepohl_blocks_and_univariate_rows() {
    for c in jmulti_cases() {
        let name = c["name"].as_str().unwrap();
        let u = as_mat(&c["resid"]);
        let k = c["k"].as_u64().unwrap() as usize;
        let r = normality_test_cholesky(u.as_ref()).unwrap();
        let p = &c["printed_lutkepohl"];
        for (got, key) in [
            (r.statistic, "joint"),
            (r.skewness, "skew"),
            (r.kurtosis, "kurt"),
            (r.pvalue, "p_joint"),
            (r.skewness_pvalue, "p_skew"),
            (r.kurtosis_pvalue, "p_kurt"),
        ] {
            assert_printed(
                got,
                p[key].as_f64().unwrap(),
                &format!("{name}: Cholesky {key}"),
            );
        }
        assert_normality_block(&r, &c["cholesky"], k, &format!("{name}: Cholesky"));
        let sm = &c["statsmodels_test_normality"];
        assert_rel_close(
            r.statistic,
            sm["statistic"].as_f64().unwrap(),
            1e-10,
            &format!("{name}: statsmodels test_normality"),
        );
        // k = 1: each residual column alone is JMulTi's univariate row, and
        // all three Jarque-Bera forms coincide there.
        for (j, row) in c["printed_jarque_bera"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let col = Mat::from_fn(u.nrows(), 1, |i, _| u[(i, j)]);
            let one = normality_test_cholesky(col.as_ref()).unwrap();
            let sym = normality_test_symmetric(col.as_ref()).unwrap();
            assert_rel_close(
                sym.statistic,
                one.statistic,
                1e-12,
                &format!("{name} u{j}: k=1 forms"),
            );
            assert_printed(
                one.statistic,
                row["statistic"].as_f64().unwrap(),
                &format!("{name} u{j}: univariate JB"),
            );
            assert_printed(
                one.pvalue,
                row["pvalue"].as_f64().unwrap(),
                &format!("{name} u{j}: univariate JB p"),
            );
            assert_printed(
                one.skewness_components[0],
                row["skewness"].as_f64().unwrap(),
                &format!("{name} u{j}: skewness"),
            );
            assert_printed(
                one.kurtosis_components[0] + 3.0,
                row["kurtosis"].as_f64().unwrap(),
                &format!("{name} u{j}: kurtosis"),
            );
            let full = &c["jarque_bera_rows"][j];
            assert_rel_close(
                one.statistic,
                full["statistic"].as_f64().unwrap(),
                1e-10,
                &format!("{name} u{j}: univariate JB transcription"),
            );
        }
    }
}

/// Doornik-Hansen (2008): the six VAR cases through `VarResults` (and the
/// bundle) and the 16 JMulTi residual matrices through the free function,
/// against the generator's transcription.
#[test]
fn golden_doornik_hansen() {
    let (fx, cs) = cases();
    for case in &cs {
        let name = case["name"].as_str().unwrap();
        let res = fit_case(&fx, case);
        let d = res.doornik_hansen_test().unwrap();
        assert_dh_block(&d, &case["doornik_hansen"], res.neqs, name);
        let s = res.normality_test_symmetric().unwrap();
        assert_normality_block(
            &s,
            &case["normality_symmetric"],
            res.neqs,
            &format!("{name}: symmetric"),
        );
        let first_nlags = case["portmanteau"][0]["nlags"].as_u64().unwrap() as usize;
        let bundle = res.diagnostics(first_nlags).unwrap();
        assert_eq!(bundle.doornik_hansen.as_ref(), Some(&d));
        assert_eq!(bundle.normality_symmetric.as_ref(), Some(&s));
        // The free functions on the fit's residuals are the methods.
        assert_eq!(doornik_hansen_test(res.resid.as_ref()).unwrap(), d);
        assert_eq!(normality_test_symmetric(res.resid.as_ref()).unwrap(), s);
        assert_eq!(
            normality_test_cholesky(res.resid.as_ref()).unwrap(),
            res.normality_test().unwrap()
        );
    }
    for c in jmulti_cases() {
        let name = c["name"].as_str().unwrap();
        let u = as_mat(&c["resid"]);
        let k = c["k"].as_u64().unwrap() as usize;
        let d = doornik_hansen_test(u.as_ref()).unwrap();
        assert_dh_block(&d, &c["doornik_hansen"], k, name);
    }
    // The t(4) system is rejected by DH too, kurtosis-driven; the Gaussian
    // one is not.
    let t4 = cs.iter().find(|c| c["name"] == "var2c_sim_t4").unwrap();
    let d = fit_case(&fx, t4).doornik_hansen_test().unwrap();
    assert!(d.pvalue < 1e-6 && d.kurtosis > d.skewness, "{d:?}");
    let g = cs.iter().find(|c| c["name"] == "var2c_sim").unwrap();
    assert!(fit_case(&fx, g).doornik_hansen_test().unwrap().pvalue > 0.05);
}

/// Each free function refuses malformed residual matrices naming `resid`.
#[test]
fn normality_refusals_name_resid() {
    let base: Vec<f64> = (0..40)
        .map(|i| ((i * 7919) % 101) as f64 / 13.0 - 3.5)
        .collect();
    let two = |t: usize| {
        Mat::from_fn(t, 2, |i, j| {
            base[i] + if j == 1 { base[(i * 3 + 5) % 40] } else { 0.0 }
        })
    };
    let all = |m: &Mat<f64>| -> Vec<String> {
        vec![
            normality_test_cholesky(m.as_ref()).unwrap_err().to_string(),
            normality_test_symmetric(m.as_ref())
                .unwrap_err()
                .to_string(),
            doornik_hansen_test(m.as_ref()).unwrap_err().to_string(),
        ]
    };
    // No columns.
    for e in all(&Mat::<f64>::zeros(20, 0)) {
        assert!(e.contains("resid") && e.contains("column"), "{e}");
    }
    // A NaN.
    let mut m = two(20);
    m[(4, 1)] = f64::NAN;
    for e in all(&m) {
        assert!(e.contains("resid") && e.contains("row 4, column 1"), "{e}");
    }
    // A constant column.
    let m = Mat::from_fn(20, 2, |i, j| if j == 0 { 0.1 } else { base[i] });
    for e in all(&m) {
        assert!(e.contains("resid") && e.contains("constant column"), "{e}");
    }
    // Entries so small that their squares are subnormal: refused by name
    // (left to the eigensolver, the subnormal covariance makes it fail to
    // converge, in an error that does not name resid). At 1e-150 the squares
    // are normal again and every statistic equals the unit-scale one, being
    // invariant to a common rescaling.
    let unit = two(20);
    for scale in [1e-155, 1e-160, 1e-200, 1e-300] {
        let m = Mat::from_fn(20, 2, |i, j| scale * unit[(i, j)]);
        for e in all(&m) {
            assert!(
                e.contains("resid") && e.contains("underflow"),
                "{scale}: {e}"
            );
        }
    }
    let small = Mat::from_fn(20, 2, |i, j| 1e-150 * unit[(i, j)]);
    for (a, b) in [
        (
            doornik_hansen_test(small.as_ref()).unwrap().statistic,
            doornik_hansen_test(unit.as_ref()).unwrap().statistic,
        ),
        (
            normality_test_symmetric(small.as_ref()).unwrap().statistic,
            normality_test_symmetric(unit.as_ref()).unwrap().statistic,
        ),
        (
            normality_test_cholesky(small.as_ref()).unwrap().statistic,
            normality_test_cholesky(unit.as_ref()).unwrap().statistic,
        ),
    ] {
        assert!((a - b).abs() <= 1e-10 * b, "{a} vs {b}");
    }
    // A duplicated column: collinear.
    let m = Mat::from_fn(20, 2, |i, _| base[i]);
    for e in all(&m) {
        assert!(
            e.contains("resid") && e.contains("correlation matrix"),
            "{e}"
        );
    }
    // T <= k (eight rows, so the DH row floor is met).
    let m = Mat::from_fn(8, 9, |i, j| base[(i * 9 + j) % 40]);
    for e in all(&m) {
        assert!(
            e.contains("resid") && e.contains("more rows than columns"),
            "{e}"
        );
    }
    // Seven rows: only Doornik-Hansen refuses.
    let m = two(7);
    assert!(normality_test_cholesky(m.as_ref()).is_ok());
    assert!(normality_test_symmetric(m.as_ref()).is_ok());
    let e = doornik_hansen_test(m.as_ref()).unwrap_err().to_string();
    assert!(e.contains("resid") && e.contains("8 rows"), "{e}");
    // Columns on wildly different scales: only the unit-dependent symmetric
    // root refuses (the Cholesky and DH 2008 forms are scale-invariant).
    let m = Mat::from_fn(20, 2, |i, j| {
        if j == 0 {
            1e9 * base[i]
        } else {
            1e-9 * base[(i * 3 + 5) % 40]
        }
    });
    assert!(normality_test_cholesky(m.as_ref()).is_ok());
    assert!(doornik_hansen_test(m.as_ref()).is_ok());
    let e = normality_test_symmetric(m.as_ref())
        .unwrap_err()
        .to_string();
    assert!(e.contains("resid") && e.contains("rescale"), "{e}");
}
