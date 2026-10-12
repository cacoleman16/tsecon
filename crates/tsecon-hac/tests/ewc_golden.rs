//! Golden-value tests for equal-weighted cosine (EWC) HAR inference against
//! `fixtures/ewc.json` (generator `fixtures/generate_ewc_fixtures.py`).
//!
//! **Grade: documented closed form, not an independent package.** No
//! reference library implements EWC (the generator's absence canary greps
//! statsmodels 0.15.0, `arch` 8.0.0 and `linearmodels` 7.0 and finds
//! nothing), so the fixture evaluates the Lazarus-Lewis-Stock-Watson (2018)
//! closed form through an *independent code path* — SciPy's FFT-based
//! type-II DCT (`scipy.fft.dct(type=2, norm="ortho")`) rather than the direct
//! cosine sum this crate runs. The OLS block takes its coefficients and
//! residuals from statsmodels `OLS(y, X).fit()`.
//!
//! Pinned:
//! * `ewc_lrv` on the Nile series (read from `diagnostics.json`) and on a
//!   seeded AR(1), `b` in {1, 4, 9, 16, n - 1}, demeaned and raw — 1e-12;
//! * `ewc_default_b` — exact, on the spec's `n` and on the ten `n` below
//!   10^6 whose `0.4 n^(2/3)` lies closest to a half-integer, and on the ten
//!   `n` in 10^6..3e10 closest above a half-integer (where a float `n^(2/3)`
//!   rounds the wrong way), the ten closest below one, and the
//!   audit-round-15 cases (the reference B is integer-exact);
//! * `ols` + [`SeType::Ewc`] on a `k = 2` and a `k = 3` design at the default
//!   `B` and at `B = 8`: params / bse / tvalues / covariance at 1e-10 and the
//!   `ewc_lrv_matrix` meat at 1e-10; the Student-t_B p-values built with
//!   tsecon-stats `StudentT::sf` (exactly as the Python binding builds them)
//!   at rtol 1e-9 with a 1e-15 absolute floor.

use serde_json::Value;
use tsecon_hac::{ewc_default_b, ewc_lrv, ewc_lrv_matrix, ols, HacError, SeType};
use tsecon_stats::{ContinuousDist, StudentT};

fn load(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
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

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        ((actual - expected) / expected).abs()
    }
}

fn assert_close(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    let rel = rel_err(actual, expected);
    assert!(
        rel <= rtol,
        "{ctx}: actual {actual}, expected {expected}, rel err {rel:e} > {rtol:e}"
    );
}

fn assert_all_close(actual: &[f64], expected: &[f64], rtol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, rtol, &format!("{ctx}[{i}]"));
    }
}

fn demean(x: &[f64]) -> Vec<f64> {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| v - m).collect()
}

fn lrv_series(fx: &Value, name: &str) -> Vec<f64> {
    match name {
        "nile" => f64s(&load("diagnostics.json")["nile"]),
        "ar1_n250" => f64s(&fx["series"]["ar1_n250"]),
        other => panic!("unknown series {other}"),
    }
}

#[test]
fn ewc_lrv_matches_the_dct_closed_form_on_nile_and_ar1() {
    let fx = load("ewc.json");
    let mut worst = 0.0_f64;
    for name in ["nile", "ar1_n250"] {
        let block = &fx["lrv"][name];
        let x = lrv_series(&fx, name);
        assert_eq!(x.len() as u64, block["n"].as_u64().expect("n"));
        let z = demean(&x);
        for b in block["b"].as_array().expect("b list") {
            let b = b.as_u64().expect("b") as usize;
            let key = b.to_string();
            let want_dem = block["ewc_demeaned"][&key].as_f64().expect("demeaned");
            let got = ewc_lrv(&z, b).unwrap();
            worst = worst.max(rel_err(got, want_dem));
            assert_close(got, want_dem, 1e-12, &format!("{name} demeaned b={b}"));
            // Raw input: the j = 0 vector is excluded, so the level drops
            // out exactly in exact arithmetic; in floating point the raw sum
            // carries the mean's rounding, measured below 1e-12 here too.
            let want_raw = block["ewc_raw"][&key].as_f64().expect("raw");
            let got_raw = ewc_lrv(&x, b).unwrap();
            assert_close(got_raw, want_raw, 1e-12, &format!("{name} raw b={b}"));
        }
        let b0 = block["default_b"].as_u64().expect("default b") as usize;
        assert_eq!(ewc_default_b(x.len()), b0, "{name} default b");
        assert_close(
            ewc_lrv(&z, b0).unwrap(),
            block["ewc_demeaned_at_default_b"].as_f64().expect("value"),
            1e-12,
            &format!("{name} at the default b"),
        );
    }
    assert!(worst < 1e-12, "worst {worst:e}");
}

#[test]
fn ewc_default_b_matches_the_llsw_rule_exactly() {
    let fx = load("ewc.json");
    let cases = fx["default_b"].as_object().expect("object");
    assert!(cases.len() >= 18, "spec n plus ten near-half n");
    for (n, b) in cases {
        let n: usize = n.parse().expect("integer key");
        let b = b.as_u64().expect("integer") as usize;
        assert_eq!(ewc_default_b(n), b, "ewc_default_b({n})");
    }
}

/// Beyond 10^6 a double-precision `0.4 n^(2/3)` misrounds near a
/// half-integer (audit round 15: `n = 1_755_027_219` gave 581990, not
/// 581991). The fixture's B is the integer rule `125 (2r - 1)^3 <= 64 n^2 <
/// 125 (2r + 1)^3`. Every case sits within 5.1e-10 of a tie (the auditor's
/// second within 1.2e-7): ten just above one, each of which the float
/// formula rounds down by one (the generator asserts it: 2/3 rounded to a
/// double is below 2/3, so the float power runs low), and ten just below
/// one, which catch an exact correction that steps up too eagerly.
#[test]
fn ewc_default_b_is_exact_where_a_float_power_misrounds() {
    let fx = load("ewc.json");
    let cases = fx["default_b_large_n"].as_object().expect("object");
    let gaps = fx["default_b_large_n_gap"].as_object().expect("object");
    assert!(cases.contains_key("1755027219") && cases.contains_key("100292764931117"));
    let signed: Vec<f64> = gaps.values().map(|g| g.as_f64().expect("gap")).collect();
    let above = signed.iter().filter(|&&g| g > 0.0).count();
    let below = signed.iter().filter(|&&g| g < 0.0).count();
    assert!(
        above >= 11 && below >= 10,
        "{above} above / {below} below a tie"
    );
    for (n, b) in cases {
        let gap = gaps[n].as_f64().expect("gap");
        assert!(gap.abs() < 2e-7, "{n} is not a near-tie case (gap {gap:e})");
        let n: usize = n.parse().expect("integer key");
        let b = b.as_u64().expect("integer") as usize;
        assert_eq!(ewc_default_b(n), b, "ewc_default_b({n})");
    }
}

/// OLS-EWC: params, the EWC meat, the sandwich and its t-statistics, and the
/// Student-t_B p-values the binding derives from them.
#[test]
fn ols_ewc_matches_the_closed_form_on_both_designs() {
    let fx = load("ewc.json");
    let blocks = fx["ols"].as_object().expect("ols blocks");
    assert_eq!(blocks.len(), 2);
    let mut worst_p = 0.0_f64;
    for (name, block) in blocks {
        let y = f64s(&block["y"]);
        let x: Vec<Vec<f64>> = block["x_cols"]
            .as_array()
            .expect("columns")
            .iter()
            .map(f64s)
            .collect();
        let k = x.len();
        let n = y.len();
        let fit = ols(&y, &x).unwrap();
        assert_all_close(
            &fit.params,
            &f64s(&block["params"]),
            1e-10,
            &format!("{name} params"),
        );
        assert_eq!(
            ewc_default_b(n) as u64,
            block["default_b"].as_u64().expect("default b")
        );
        let scores: Vec<Vec<f64>> = x
            .iter()
            .map(|col| {
                col.iter()
                    .zip(&fit.residuals)
                    .map(|(&a, &u)| a * u)
                    .collect()
            })
            .collect();
        let cases = block["cases"].as_array().expect("cases");
        assert!(cases.iter().any(|c| c["b_is_default"] == true));
        for case in cases {
            let b = case["b"].as_u64().expect("b") as usize;
            let ctx = format!("{name} b={b}");
            let omega = ewc_lrv_matrix(&scores, b).unwrap();
            assert_all_close(
                &omega,
                &f64s(&case["omega"]),
                1e-10,
                &format!("{ctx} omega"),
            );
            let inf = fit.inference(SeType::Ewc { b }).unwrap();
            assert_eq!(inf.cov.len(), k * k);
            assert_all_close(&inf.cov, &f64s(&case["cov"]), 1e-10, &format!("{ctx} cov"));
            assert_all_close(&inf.bse, &f64s(&case["bse"]), 1e-10, &format!("{ctx} bse"));
            assert_all_close(
                &inf.tvalues,
                &f64s(&case["tvalues"]),
                1e-10,
                &format!("{ctx} tvalues"),
            );
            let t_b = StudentT::new(b as f64).unwrap();
            for (i, (&t, &want)) in inf
                .tvalues
                .iter()
                .zip(f64s(&case["pvalues"]).iter())
                .enumerate()
            {
                let p = 2.0 * t_b.sf(t.abs());
                let err = (p - want).abs();
                worst_p = worst_p.max(err / want);
                assert!(
                    err <= 1e-9 * want + 1e-15,
                    "{ctx} pvalue[{i}]: {p} vs {want} (abs err {err:e})"
                );
            }
            let crit = t_b.ppf(0.975).unwrap();
            assert_close(
                crit,
                case["t_crit_975"].as_f64().expect("crit"),
                1e-10,
                &format!("{ctx} t_B critical value"),
            );
        }
    }
    assert!(worst_p < 1e-9, "worst p-value rel err {worst_p:e}");
}

#[test]
fn ewc_inference_refuses_an_invalid_b_by_value() {
    let fx = load("ewc.json");
    let block = &fx["ols"]["k2_t250"];
    let y = f64s(&block["y"]);
    let x: Vec<Vec<f64>> = block["x_cols"]
        .as_array()
        .expect("columns")
        .iter()
        .map(f64s)
        .collect();
    let fit = ols(&y, &x).unwrap();
    for b in [0, y.len(), y.len() + 7] {
        match fit.inference(SeType::Ewc { b }) {
            Err(HacError::InvalidDof { b: got, n, what }) => {
                assert_eq!((got, n), (b, y.len()));
                assert_eq!(what, "OLS EWC covariance");
            }
            other => panic!("b = {b} should be refused, got {other:?}"),
        }
    }
    assert!(fit.inference(SeType::Ewc { b: y.len() - 1 }).is_ok());
}
