//! Golden-value tests for `panel_lp` with `bias_correction = Spj`
//! against `fixtures/panel_spj.json`.
//!
//! The fixture is a **transcription golden** (see
//! `fixtures/generate_panel_spj_fixtures.py`): an independent NumPy
//! reimplementation of the Mei-Sheng-Shi split-panel jackknife algebra as
//! shipped in their reference `pLP` R package (`panelLP.R`,
//! github.com/zhentaoshi/panel-local-projection, fetched 2026-08-18) —
//! the median row split, the full-panel leads/lags in each half, the
//! `2F - (A+B)/2` combination, and both adjusted-score sandwiches
//! (cluster-by-entity with the Stata-style `(N/(N-1))((n-1)/(n-k))`
//! factor; Driscoll-Kraay with no small-sample factor). The R repository
//! commits no numeric outputs and no R interpreter is available in the
//! test environment, so the honest validation grade is transcription +
//! Monte Carlo (`spj_properties.rs`), not a stored-output match.
//!
//! Tolerance is 1e-10 relative: the generator stores full-precision
//! doubles and both sides compute the same estimator through different
//! numerical paths (NumPy lstsq/inv vs faer QR/Cholesky).

use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_panel::{panel_lp, LpBiasCorrection, PanelData, PanelLpConfig, PanelSeType};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/panel_spj.json",
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

fn assert_close(got: f64, want: f64, rtol: f64, what: &str) {
    let denom = want.abs().max(1e-12);
    assert!(
        ((got - want) / denom).abs() < rtol,
        "{what}: got {got}, want {want} (rel {})",
        ((got - want) / denom).abs()
    );
}

fn build(fx: &Value) -> (PanelData, Vec<f64>, PanelLpConfig) {
    let y: Vec<Vec<f64>> = fx["y"].as_array().expect("y").iter().map(f64s).collect();
    let shock = f64s(&fx["shock"]);
    let n = y.len();
    let t = y[0].len();
    let outcome = Mat::from_fn(n, t, |i, tt| y[i][tt]);
    let data = PanelData::balanced(outcome, vec![]).expect("balanced panel");

    let design = &fx["design"];
    let mut cfg = PanelLpConfig::new(
        design["max_horizon"].as_u64().expect("hmax") as usize,
        0,
        PanelSeType::ClusterEntity, // per-case below
    );
    cfg.shock_lags = design["shock_lags"].as_u64().expect("Ls") as usize;
    cfg.outcome_lags = design["outcome_lags"].as_u64().expect("Ly") as usize;
    cfg.bias_correction = LpBiasCorrection::Spj;
    (data, shock, cfg)
}

#[test]
fn spj_matches_the_transcription_at_1e10() {
    let fx = load();
    let (data, shock, base_cfg) = build(&fx);

    for (name, case) in fx["cases"].as_object().expect("cases") {
        let mut cfg = base_cfg;
        cfg.cumulative = case["cumulative"].as_bool().expect("cumulative");
        cfg.cov = match case["se_type"].as_str().expect("se_type") {
            "cluster" => PanelSeType::ClusterEntity,
            "driscoll_kraay" => PanelSeType::DriscollKraay {
                bandwidth: case["bandwidth"].as_f64().expect("bandwidth"),
            },
            other => panic!("unknown se_type {other}"),
        };

        let res = panel_lp(&data, &shock, &cfg).expect("spj panel lp");
        assert_eq!(res.bias_correction, LpBiasCorrection::Spj, "{name}: stamp");
        assert!(!res.jackknife, "{name}: SPJ must not claim the DJ flag");

        let horizons = case["horizons"].as_array().expect("horizons");
        assert_eq!(res.irf.len(), horizons.len(), "{name}: horizon count");
        for (h, want) in horizons.iter().enumerate() {
            let beta_spj = f64s(&want["beta_spj"]);
            let se_spj = f64s(&want["se_spj"]);
            assert_close(res.irf[h], beta_spj[0], 1e-10, &format!("{name} h={h} irf"));
            for (j, &b) in beta_spj.iter().enumerate() {
                assert_close(
                    res.params[h][j],
                    b,
                    1e-10,
                    &format!("{name} h={h} params[{j}]"),
                );
            }
            assert_close(res.se[h], se_spj[0], 1e-10, &format!("{name} h={h} se"));
            assert_eq!(
                res.nobs[h],
                want["nobs"].as_u64().expect("nobs") as usize,
                "{name} h={h}: nobs"
            );
        }
    }
}

/// The full-sample coefficients inside the correction must agree with the
/// uncorrected route (same within estimator, same rows): the identity
/// `beta_spj = 2 beta_full - (beta_a + beta_b)/2` pinned by the fixture
/// then fixes the half fits too.
#[test]
fn spj_full_sample_leg_equals_the_uncorrected_fit() {
    let fx = load();
    let (data, shock, mut cfg) = build(&fx);
    cfg.cov = PanelSeType::ClusterEntity;
    cfg.bias_correction = LpBiasCorrection::None;
    let plain = panel_lp(&data, &shock, &cfg).expect("plain lp");

    let horizons = fx["cases"]["spj_cluster"]["horizons"]
        .as_array()
        .expect("horizons");
    for (h, want) in horizons.iter().enumerate() {
        let beta_full = f64s(&want["beta_full"]);
        for (j, &b) in beta_full.iter().enumerate() {
            assert_close(
                plain.params[h][j],
                b,
                1e-10,
                &format!("h={h} full-leg params[{j}]"),
            );
        }
    }
}
