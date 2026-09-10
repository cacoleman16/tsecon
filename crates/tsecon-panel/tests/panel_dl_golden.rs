//! Golden-value tests against `fixtures/panel_dl.json` (generated with
//! linearmodels 7.0 by `fixtures/generate_panel_dl_fixtures.py`).
//!
//! Nine cases cover the `panel_distributed_lag` surface: linear responses
//! at `L = 0, 1, 2, 3` under two-way, entity-only and time-only effects,
//! entity trends with and without time effects, two regressors, and the
//! quadratic response with explicit and default (sample-mean) evaluation
//! points. Every case pins slopes, standard errors, t-statistics and the
//! full covariance at 1e-10 relative against `PanelOLS` for the three
//! covariance estimators (nonrobust, entity-clustered, Driscoll-Kraay at
//! bandwidth 4), `nobs` and `df_resid` exactly, and — the
//! documented-formula leg — the cumulative effect, its delta-method
//! standard error and interval, the marginal effects and the turning
//! point against the generator's NumPy transcription at 1e-10.
//!
//! The inputs are stored at full double precision, so unlike
//! `fixtures/panel.json` the ceiling here is the estimator, not the
//! fixture.

use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_panel::{
    panel_distributed_lag, DistributedLagConfig, DistributedLagResult, FixedEffects, PanelData,
    PanelSeType,
};

const RTOL: f64 = 1e-10;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/panel_dl.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(f64s).collect()
}

fn to_mat(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn assert_close(got: f64, want: f64, rtol: f64, floor: f64, what: &str) {
    let denom = want.abs().max(floor);
    let rel = ((got - want) / denom).abs();
    assert!(rel < rtol, "{what}: got {got}, want {want} (rel {rel:e})");
}

/// Rebuilds the case's panel and effects menu from the fixture.
fn build(fx: &Value, case: &Value) -> (PanelData, FixedEffects) {
    let inputs = &fx["inputs"];
    let y = to_mat(&matrix(&inputs[case["outcome"].as_str().unwrap()]));
    let regs: Vec<(String, Mat<f64>)> = case["regressors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            let name = n.as_str().unwrap();
            (name.to_string(), to_mat(&matrix(&inputs[name])))
        })
        .collect();
    let data = PanelData::balanced(y, regs).expect("balanced panel");
    let effects = FixedEffects {
        entity: case["entity_effects"].as_bool().unwrap(),
        time: case["time_effects"].as_bool().unwrap(),
        entity_trends: case["entity_trends"].as_bool().unwrap(),
    };
    (data, effects)
}

fn config(case: &Value, effects: FixedEffects, se_type: PanelSeType) -> DistributedLagConfig {
    let eval_points = match &case["eval_points"] {
        Value::Array(pts) => Some(pts.iter().map(|v| v.as_f64().unwrap()).collect()),
        _ => None, // null or "mean": the crate's default (sample mean)
    };
    DistributedLagConfig {
        lags: case["lags"].as_u64().unwrap() as usize,
        powers: case["powers"].as_u64().unwrap() as usize,
        effects,
        se_type,
        eval_points,
    }
}

fn se_cases(bw: f64) -> [(&'static str, PanelSeType); 3] {
    [
        ("nonrobust", PanelSeType::NonRobust),
        ("cluster_entity", PanelSeType::ClusterEntity),
        (
            "driscoll_kraay",
            PanelSeType::DriscollKraay { bandwidth: bw },
        ),
    ]
}

fn fit(fx: &Value, case: &Value, se_type: PanelSeType) -> DistributedLagResult {
    let (data, effects) = build(fx, case);
    let cfg = config(case, effects, se_type);
    panel_distributed_lag(&data, &cfg).unwrap_or_else(|e| panic!("{}: {e}", case["name"]))
}

#[test]
fn every_case_matches_linearmodels_panelols() {
    let fx = load();
    let n_ent = fx["n_entities"].as_u64().unwrap() as usize;
    let mut n_checked = 0usize;
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let bw = case["bandwidth"].as_f64().unwrap();
        let want_names: Vec<&str> = case["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for (key, se_type) in se_cases(bw) {
            let res = fit(&fx, case, se_type);
            let block = &case["fits"][key];
            assert_eq!(res.names, want_names, "{name}/{key}: column names");
            assert_eq!(
                res.nobs,
                block["nobs"].as_u64().unwrap() as usize,
                "{name}/{key}: nobs"
            );
            assert_eq!(
                res.df_resid,
                block["df_resid"].as_u64().unwrap() as usize,
                "{name}/{key}: df_resid"
            );
            assert_eq!(res.n_entities, n_ent);
            assert_eq!(
                res.n_periods_used,
                fx["n_periods"].as_u64().unwrap() as usize - res.lags
            );
            let params = f64s(&block["params"]);
            let bse = f64s(&block["bse"]);
            let tstats = f64s(&block["tstats"]);
            let cov = matrix(&block["cov"]);
            let cov_scale = cov.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
            let k = params.len();
            assert_eq!(res.params.len(), k);
            for a in 0..k {
                assert_close(
                    res.params[a],
                    params[a],
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} param {a}"),
                );
                assert_close(
                    res.bse[a],
                    bse[a],
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} bse {a}"),
                );
                assert_close(
                    res.tvalues[a],
                    tstats[a],
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} t {a}"),
                );
                for (b, want) in cov[a].iter().enumerate() {
                    assert_close(
                        res.cov[(a, b)],
                        *want,
                        RTOL,
                        cov_scale,
                        &format!("{name}/{key} cov[{a},{b}]"),
                    );
                }
            }
            n_checked += 1;
        }
    }
    assert_eq!(n_checked, 27, "nine cases x three covariance estimators");
}

#[test]
fn delta_method_matches_documented_transcription() {
    let fx = load();
    let z = fx["z975"].as_f64().unwrap();
    let mut n_checked = 0usize;
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if case["regressors"].as_array().unwrap().len() != 1 {
            continue;
        }
        let bw = case["bandwidth"].as_f64().unwrap();
        for (key, se_type) in se_cases(bw) {
            let res = fit(&fx, case, se_type);
            let d = &case["delta"][key];
            let powers = res.powers;
            let cum = f64s(&d["cumulative_effect"]);
            let cse = f64s(&d["cumulative_se"]);
            let lo = f64s(&d["cumulative_ci_low"]);
            let hi = f64s(&d["cumulative_ci_high"]);
            for p in 0..powers {
                assert_close(
                    res.cumulative_effect[0][p],
                    cum[p],
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} cum {p}"),
                );
                assert_close(
                    res.cumulative_se[0][p],
                    cse[p],
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} cum se {p}"),
                );
                assert_close(
                    res.cumulative_ci_low[0][p],
                    lo[p],
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} ci lo {p}"),
                );
                assert_close(
                    res.cumulative_ci_high[0][p],
                    hi[p],
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} ci hi {p}"),
                );
                // The interval is the documented normal one.
                assert_close(
                    res.cumulative_ci_high[0][p] - res.cumulative_effect[0][p],
                    z * res.cumulative_se[0][p],
                    1e-12,
                    1e-300,
                    &format!("{name}/{key} half-width {p}"),
                );
                // Lag-level views agree with the flat parameter vector.
                let sum: f64 = res.lag_effects[0][p].iter().sum();
                assert_close(sum, res.cumulative_effect[0][p], 1e-12, 1e-300, "lag sum");
            }
            if powers == 2 {
                let pts = f64s(&d["eval_points"]);
                let me = f64s(&d["marginal_effect"]);
                let mse = f64s(&d["marginal_se"]);
                let got_pts = res.eval_points.as_ref().unwrap();
                let got_me = res.marginal_effect.as_ref().unwrap();
                let got_mse = res.marginal_se.as_ref().unwrap();
                assert_eq!(got_pts[0].len(), pts.len(), "{name}: eval point count");
                for q in 0..pts.len() {
                    assert_close(
                        got_pts[0][q],
                        pts[q],
                        RTOL,
                        1e-300,
                        &format!("{name} eval point {q}"),
                    );
                    assert_close(
                        got_me[0][q],
                        me[q],
                        RTOL,
                        1e-300,
                        &format!("{name}/{key} marginal {q}"),
                    );
                    assert_close(
                        got_mse[0][q],
                        mse[q],
                        RTOL,
                        1e-300,
                        &format!("{name}/{key} marginal se {q}"),
                    );
                }
                assert_close(
                    res.turning_point.as_ref().unwrap()[0],
                    d["turning_point"].as_f64().unwrap(),
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} turning point"),
                );
                assert_close(
                    res.turning_point_se.as_ref().unwrap()[0],
                    d["turning_point_se"].as_f64().unwrap(),
                    RTOL,
                    1e-300,
                    &format!("{name}/{key} turning point se"),
                );
            } else {
                assert!(res.eval_points.is_none() && res.marginal_effect.is_none());
                assert!(res.marginal_se.is_none() && res.turning_point.is_none());
                assert!(res.turning_point_se.is_none());
            }
            n_checked += 1;
        }
    }
    assert_eq!(
        n_checked, 24,
        "eight single-regressor cases x three estimators"
    );
}

#[test]
fn two_regressor_case_orders_columns_regressor_major() {
    // The two-regressor case has no delta block in the fixture (the
    // transcription is single-regressor); check the per-regressor views
    // against the flat vector directly.
    let fx = load();
    let case = fx["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "linear_L1_two_regressors")
        .unwrap();
    let res = fit(&fx, case, PanelSeType::ClusterEntity);
    assert_eq!(res.names, ["x_L0", "x_L1", "z_L0", "z_L1"]);
    assert_eq!(res.lag_effects.len(), 2);
    for j in 0..2 {
        for l in 0..2 {
            assert_eq!(res.lag_effects[j][0][l], res.params[j * 2 + l]);
            assert_eq!(res.lag_se[j][0][l], res.bse[j * 2 + l]);
        }
        let var = (0..2)
            .flat_map(|a| (0..2).map(move |b| (a, b)))
            .map(|(a, b)| res.cov[(j * 2 + a, j * 2 + b)])
            .sum::<f64>();
        assert_close(res.cumulative_se[j][0], var.sqrt(), 1e-12, 1e-300, "cum se");
    }
}
