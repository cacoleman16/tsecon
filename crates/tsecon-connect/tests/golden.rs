//! Golden-value tests against the NumPy fixture (connect.json): a
//! Pesaran-Shin (1998) generalized FEVD, row-normalized, and the
//! Diebold-Yilmaz (2012) connectedness measures, on a VAR(2)-with-constant
//! of the macro data carried in the fixture (horizon 10).

mod common;

use common::{as_mat, assert_mat_close, assert_rel_close, load_fixture};
use tsecon_connect::{generalized_fevd, ConnectednessTable};
use tsecon_linalg::faer::Mat;
use tsecon_var::{Trend, VarSpec};

/// Loads the fixture and fits the VAR. The fixture stores `data` as
/// `k x T` (one row per variable); `tsecon-var` wants observations in
/// rows, so the design matrix is the transpose.
fn setup() -> (serde_json::Value, tsecon_var::VarResults, usize) {
    let fx = load_fixture("connect.json");
    let raw = as_mat(&fx["data"]); // k x T
    let endog = Mat::from_fn(raw.ncols(), raw.nrows(), |i, j| raw[(j, i)]); // T x k
    let lags = fx["lags"].as_u64().unwrap() as usize;
    let horizon = fx["horizon"].as_u64().unwrap() as usize;
    let res = VarSpec::new(lags, Trend::Constant)
        .unwrap()
        .fit(endog.as_ref())
        .unwrap();
    (fx, res, horizon)
}

/// The row-normalized generalized FEVD reproduces `gfevd_normalized` to
/// 1e-8 (the task's golden tolerance; the actual agreement is ~1e-16).
#[test]
fn golden_gfevd_normalized() {
    let (fx, res, horizon) = setup();
    let psi = res.ma_rep(horizon).unwrap();
    let theta = generalized_fevd(&psi, res.sigma_u.as_ref()).unwrap();
    assert_mat_close(&theta, &fx["gfevd_normalized"], 1e-8, "gfevd_normalized");
}

/// Total connectedness and the directional "to others" / "from others"
/// vectors match the fixture to 1e-6.
#[test]
fn golden_connectedness_measures() {
    let (fx, res, horizon) = setup();
    let table = ConnectednessTable::from_var(&res, horizon).unwrap();

    assert_rel_close(
        table.total,
        fx["total_connectedness"].as_f64().unwrap(),
        1e-6,
        "total_connectedness",
    );

    let to = fx["to_others"].as_array().unwrap();
    let from = fx["from_others"].as_array().unwrap();
    assert_eq!(table.to_others.len(), to.len());
    for (i, v) in to.iter().enumerate() {
        assert_rel_close(table.to_others[i], v.as_f64().unwrap(), 1e-6, "to_others");
    }
    for (i, v) in from.iter().enumerate() {
        assert_rel_close(
            table.from_others[i],
            v.as_f64().unwrap(),
            1e-6,
            "from_others",
        );
    }
}

/// The `from_var` convenience path and the two-step `generalized_fevd` +
/// `from_gfevd` path agree exactly, and the Display renders the spillover
/// table with the total in the lower-right corner.
#[test]
fn golden_table_consistency_and_display() {
    let (_fx, res, horizon) = setup();
    let psi = res.ma_rep(horizon).unwrap();
    let theta = generalized_fevd(&psi, res.sigma_u.as_ref()).unwrap();
    let a = ConnectednessTable::from_gfevd(theta.as_ref()).unwrap();
    let b = ConnectednessTable::from_var(&res, horizon).unwrap();
    assert_rel_close(a.total, b.total, 0.0, "total match");

    let rendered = format!(
        "{}",
        b.with_labels(vec!["gdp".into(), "cons".into(), "inv".into()])
            .unwrap()
    );
    assert!(rendered.contains("FROM"));
    assert!(rendered.contains("TO"));
    assert!(rendered.contains("gdp"));
    // The total connectedness index prints in the corner.
    assert!(rendered.contains(&format!("{:.2}", a.total)));
}
