//! Golden-value tests against `fixtures/filters.json` (generated with
//! statsmodels 0.14.6 on 100*log US real GDP, 203 quarterly
//! observations; see `fixtures/generate_fixtures.py`).
//!
//! Tolerance: 1e-8 (absolute for cycles, which cross zero; relative for
//! trends and coefficients).

use serde_json::Value;
use tsecon_filters::{bk_filter, cf_filter, hamilton_filter, hp_filter};

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/filters.json");
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

/// Absolute-or-relative comparison: |a - e| <= tol * max(1, |e|).
fn assert_close(actual: f64, expected: f64, tol: f64, ctx: &str) {
    let err = (actual - expected).abs() / expected.abs().max(1.0);
    assert!(
        err <= tol,
        "{ctx}: actual {actual}, expected {expected}, err {err:e} > {tol:e}"
    );
}

fn assert_all_close(actual: &[f64], expected: &[f64], tol: f64, ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert_close(a, e, tol, &format!("{ctx}[{i}]"));
    }
}

#[test]
fn hp_1600_golden() {
    let fx = fixture();
    let y = f64s(&fx["y_100_log_realgdp"]);
    let dec = hp_filter(&y, 1600.0).expect("hp_filter succeeds");
    let trend = dec.trend.expect("hp has a trend");
    assert_all_close(&trend, &f64s(&fx["hp_1600"]["trend"]), 1e-8, "hp trend");
    assert_all_close(&dec.cycle, &f64s(&fx["hp_1600"]["cycle"]), 1e-8, "hp cycle");
    assert_eq!(dec.alignment.lost_start, 0);
    assert_eq!(dec.alignment.lost_end, 0);
    assert_eq!(dec.alignment.input_len, y.len());
}

#[test]
fn bk_6_32_k12_golden() {
    let fx = fixture();
    let y = f64s(&fx["y_100_log_realgdp"]);
    let expected = f64s(&fx["bk_6_32_K12"]);
    let dec = bk_filter(&y, 6.0, 32.0, 12).expect("bk_filter succeeds");
    assert!(dec.trend.is_none(), "BK defines no trend component");
    assert_all_close(&dec.cycle, &expected, 1e-8, "bk cycle");
    // statsmodels returns the truncated series: K lost at each end.
    assert_eq!(dec.alignment.lost_start, 12);
    assert_eq!(dec.alignment.lost_end, 12);
    assert_eq!(dec.cycle.len(), y.len() - 24);
    assert_eq!(dec.alignment.output_len(), dec.cycle.len());
    // Output element 0 corresponds to input observation 12.
    assert_eq!(dec.alignment.first_index(), 12);
    assert_eq!(dec.alignment.input_index(0), Some(12));
    assert_eq!(dec.alignment.input_index(dec.cycle.len()), None);
}

#[test]
fn cf_6_32_drift_golden() {
    let fx = fixture();
    let y = f64s(&fx["y_100_log_realgdp"]);
    let dec = cf_filter(&y, 6.0, 32.0, true).expect("cf_filter succeeds");
    let trend = dec.trend.expect("cf has a trend");
    assert_all_close(
        &dec.cycle,
        &f64s(&fx["cf_6_32_drift"]["cycle"]),
        1e-8,
        "cf cycle",
    );
    assert_all_close(
        &trend,
        &f64s(&fx["cf_6_32_drift"]["trend"]),
        1e-8,
        "cf trend",
    );
    assert_eq!(dec.alignment.lost_start, 0);
    assert_eq!(dec.alignment.lost_end, 0);
}

#[test]
fn hamilton_h8_p4_golden() {
    let fx = fixture();
    let y = f64s(&fx["y_100_log_realgdp"]);
    let block = &fx["hamilton_h8_p4"];
    let res = hamilton_filter(&y, 8, 4).expect("hamilton_filter succeeds");
    assert_all_close(&res.beta, &f64s(&block["beta"]), 1e-8, "hamilton beta");
    assert_all_close(
        &res.decomposition.cycle,
        &f64s(&block["cycle"]),
        1e-8,
        "hamilton cycle",
    );
    let first = block["first_cycle_index"].as_u64().expect("index") as usize;
    assert_eq!(res.decomposition.alignment.lost_start, first);
    assert_eq!(res.decomposition.alignment.first_index(), first);
    assert_eq!(res.decomposition.alignment.lost_end, 0);
    assert_eq!(res.decomposition.cycle.len(), y.len() - first);
}
