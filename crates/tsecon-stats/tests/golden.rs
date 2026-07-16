//! Golden-value tests against `fixtures/distributions.json` (generated with
//! SciPy 1.17.1; see `fixtures/generate_fixtures.py`).
//!
//! Tolerances: pdf/logpdf/cdf at 1e-12 relative, ppf at 1e-9 relative,
//! special functions at 1e-12 relative (absolute at expected == 0).

use serde_json::Value;
use tsecon_stats::{ContinuousDist, Ged, StdNormal, StudentT};

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/distributions.json"
    );
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

/// Relative comparison; falls back to absolute when the reference is 0.
fn assert_close(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    if expected == 0.0 {
        assert!(
            actual.abs() <= rtol,
            "{ctx}: actual {actual}, expected 0 (atol {rtol})"
        );
    } else {
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel <= rtol,
            "{ctx}: actual {actual}, expected {expected}, rel err {rel:e} > {rtol:e}"
        );
    }
}

fn check_dist(d: &dyn ContinuousDist, block: &Value, name: &str) {
    let xs = f64s(&block["x"]);
    let pdf = f64s(&block["pdf"]);
    let logpdf = f64s(&block["logpdf"]);
    let cdf = f64s(&block["cdf"]);
    for (i, &x) in xs.iter().enumerate() {
        assert_close(d.pdf(x), pdf[i], 1e-12, &format!("{name} pdf({x})"));
        assert_close(
            d.ln_pdf(x),
            logpdf[i],
            1e-12,
            &format!("{name} logpdf({x})"),
        );
        assert_close(d.cdf(x), cdf[i], 1e-12, &format!("{name} cdf({x})"));
    }
    let qs = f64s(&block["q"]);
    let ppf = f64s(&block["ppf"]);
    for (i, &q) in qs.iter().enumerate() {
        assert_close(d.ppf(q).unwrap(), ppf[i], 1e-9, &format!("{name} ppf({q})"));
    }
}

#[test]
fn std_normal_golden() {
    let fx = fixture();
    check_dist(&StdNormal, &fx["std_normal"], "std_normal");
}

#[test]
fn student_t_golden() {
    let fx = fixture();
    let block = &fx["student_t"];
    let df = block["df"].as_f64().unwrap();
    let d = StudentT::new(df).unwrap();
    check_dist(&d, block, "student_t");
}

#[test]
fn student_t_fractional_df_golden() {
    let fx = fixture();
    let block = &fx["student_t_frac_df"];
    let df = block["df"].as_f64().unwrap();
    assert_eq!(df, 4.3);
    let d = StudentT::new(df).unwrap();
    check_dist(&d, block, "student_t_frac_df");
}

#[test]
fn ged_gennorm_golden() {
    let fx = fixture();
    let block = &fx["ged_gennorm"];
    let nu = block["beta"].as_f64().unwrap();
    let d = Ged::new(nu).unwrap();
    check_dist(&d, block, "ged_gennorm");
}

#[test]
fn special_functions_golden() {
    let fx = fixture();
    let sp = &fx["special_functions"];

    let erf_x = f64s(&sp["erf_x"]);
    let erf_v = f64s(&sp["erf"]);
    for (i, &x) in erf_x.iter().enumerate() {
        assert_close(
            tsecon_stats::special::erf(x),
            erf_v[i],
            1e-12,
            &format!("erf({x})"),
        );
    }

    let lg_x = f64s(&sp["lgamma_x"]);
    let lg_v = f64s(&sp["lgamma"]);
    for (i, &x) in lg_x.iter().enumerate() {
        assert_close(
            tsecon_stats::special::ln_gamma(x),
            lg_v[i],
            1e-12,
            &format!("lgamma({x})"),
        );
    }

    let bi_args = sp["betainc_args"].as_array().unwrap();
    let bi_v = f64s(&sp["betainc"]);
    for (i, args) in bi_args.iter().enumerate() {
        let args = f64s(args);
        let (a, b, x) = (args[0], args[1], args[2]);
        assert_close(
            tsecon_stats::special::beta_inc(a, b, x).unwrap(),
            bi_v[i],
            1e-12,
            &format!("betainc({a},{b},{x})"),
        );
    }
}

/// `sample_from_uniform` must agree with `ppf` (inverse-transform default).
#[test]
fn sample_from_uniform_is_ppf() {
    let fx = fixture();
    let d = StudentT::new(fx["student_t"]["df"].as_f64().unwrap()).unwrap();
    for &u in &[0.01, 0.3, 0.5, 0.77, 0.999] {
        assert_eq!(d.sample_from_uniform(u).unwrap(), d.ppf(u).unwrap());
    }
}
