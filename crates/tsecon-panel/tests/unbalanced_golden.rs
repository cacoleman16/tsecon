//! Golden-value tests for the observation mask (unbalanced panels)
//! against `fixtures/panel_unbalanced.json` (linearmodels 7.0 by
//! `fixtures/generate_panel_unbalanced_fixtures.py`), plus the
//! bit-identity of a fully observed mask with the balanced path and the
//! refusals that name `mask`.
//!
//! * `fe_cases`: the within estimator under every effects menu (entity,
//!   two-way, time-only, entity trends, trends + time) on the
//!   Arellano-Bond `EmplUK` panel (140 firms, 1976-1984, 1031 firm-years)
//!   and on a seeded panel with entry, exit and internal gaps — slopes,
//!   standard errors, t-statistics and the full covariance at 1e-10
//!   relative for the nonrobust, entity-clustered and Driscoll-Kraay
//!   (Bartlett, bandwidth 4) covariances; `nobs` and `df_resid` exact.
//! * `dl_cases`: the distributed-lag design on the same panels (lags
//!   0-2, linear and quadratic, one and two regressors, every effects
//!   menu), the same pins plus the documented delta-method leg.
//! * `lp_cases`: the panel local projection per horizon against
//!   `PanelOLS` on exactly the rows the crate keeps (target and lags
//!   observed) — `irf`, `se` and `nobs` per horizon and covariance.

use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_panel::{
    lp_did, mean_group_var, panel_distributed_lag, panel_lp, panel_ols_fe, panel_ols_fe_with,
    DistributedLagConfig, FixedEffects, LpBiasCorrection, LpDidConfig, PanelData, PanelError,
    PanelLpConfig, PanelSeType,
};
use tsecon_var::Trend;

const RTOL: f64 = 1e-10;

fn load(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

/// A JSON row; `null` (a masked-out cell) becomes NaN.
fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(f64s).collect()
}

fn to_mat(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn mask_of(v: &Value) -> Vec<Vec<bool>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() == 1)
                .collect()
        })
        .collect()
}

fn assert_close(got: f64, want: f64, rtol: f64, floor: f64, what: &str) {
    let denom = want.abs().max(floor);
    let rel = ((got - want) / denom).abs();
    assert!(rel < rtol, "{what}: got {got}, want {want} (rel {rel:e})");
}

fn se_menu() -> [(&'static str, PanelSeType); 3] {
    [
        ("nonrobust", PanelSeType::NonRobust),
        ("cluster_entity", PanelSeType::ClusterEntity),
        (
            "driscoll_kraay",
            PanelSeType::DriscollKraay { bandwidth: 4.0 },
        ),
    ]
}

fn effects_of(case: &Value) -> FixedEffects {
    FixedEffects {
        entity: case["entity_effects"].as_bool().unwrap(),
        time: case["time_effects"].as_bool().unwrap(),
        entity_trends: case["entity_trends"].as_bool().unwrap(),
    }
}

/// The case's unbalanced panel from its dataset block.
fn build(fx: &Value, case: &Value) -> PanelData {
    let ds = &fx["datasets"][case["dataset"].as_str().unwrap()];
    let y = to_mat(&matrix(&ds[case["outcome"].as_str().unwrap()]));
    let regs: Vec<(String, Mat<f64>)> = case["regressors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            let name = n.as_str().unwrap();
            (name.to_string(), to_mat(&matrix(&ds[name])))
        })
        .collect();
    let mask = mask_of(&ds["mask"]);
    let data = PanelData::unbalanced(y, regs, &mask).expect("unbalanced panel");
    assert!(!data.is_balanced());
    assert_eq!(data.nobs(), ds["n_obs"].as_u64().unwrap() as usize);
    data
}

#[test]
fn empluk_panel_is_the_arellano_bond_one() {
    let fx = load("panel_unbalanced.json");
    let ds = &fx["datasets"]["empluk"];
    assert_eq!(ds["n_entities"], 140);
    assert_eq!(ds["n_periods"], 9);
    assert_eq!(ds["n_obs"], 1031);
    let mask = mask_of(&ds["mask"]);
    let counts: Vec<usize> = mask
        .iter()
        .map(|r| r.iter().filter(|&&m| m).count())
        .collect();
    assert_eq!(counts.iter().filter(|&&c| c == 7).count(), 103);
    assert_eq!(counts.iter().filter(|&&c| c == 8).count(), 23);
    assert_eq!(counts.iter().filter(|&&c| c == 9).count(), 14);
}

#[test]
fn fe_cases_match_linearmodels_panelols() {
    let fx = load("panel_unbalanced.json");
    let cases = fx["fe_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let data = build(&fx, case);
        let effects = effects_of(case);
        let fit = panel_ols_fe_with(&data, effects).unwrap_or_else(|e| panic!("{name}: {e}"));
        let k = fit.nparams;
        for (key, se_type) in se_menu() {
            let want = &case["fits"][key];
            let inf = fit.inference(se_type).unwrap();
            assert_eq!(
                fit.nobs,
                want["nobs"].as_u64().unwrap() as usize,
                "{name}/{key} nobs"
            );
            assert_eq!(
                fit.df_resid,
                want["df_resid"].as_u64().unwrap() as usize,
                "{name}/{key} df_resid"
            );
            let (wp, wb, wt) = (
                f64s(&want["params"]),
                f64s(&want["bse"]),
                f64s(&want["tstats"]),
            );
            let wc = matrix(&want["cov"]);
            for j in 0..k {
                assert_close(
                    fit.params[j],
                    wp[j],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} params[{j}]"),
                );
                assert_close(
                    inf.bse[j],
                    wb[j],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} bse[{j}]"),
                );
                assert_close(
                    inf.tvalues[j],
                    wt[j],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} t[{j}]"),
                );
                for b in 0..k {
                    assert_close(
                        inf.cov[(j, b)],
                        wc[j][b],
                        RTOL,
                        1e-14,
                        &format!("{name}/{key} cov[{j},{b}]"),
                    );
                }
            }
        }
    }
}

#[test]
fn dl_cases_match_linearmodels_and_the_delta_method() {
    let fx = load("panel_unbalanced.json");
    let cases = fx["dl_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let data = build(&fx, case);
        let lags = case["lags"].as_u64().unwrap() as usize;
        let powers = case["powers"].as_u64().unwrap() as usize;
        let eval_points: Option<Vec<f64>> = case["eval_points"]
            .is_array()
            .then(|| f64s(&case["eval_points"]));
        for (key, se_type) in se_menu() {
            let cfg = DistributedLagConfig {
                lags,
                powers,
                effects: effects_of(case),
                se_type,
                eval_points: eval_points.clone(),
            };
            let r =
                panel_distributed_lag(&data, &cfg).unwrap_or_else(|e| panic!("{name}/{key}: {e}"));
            let want = &case["fits"][key];
            let names: Vec<&str> = case["names"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(r.names, names, "{name}/{key} names");
            assert_eq!(
                r.nobs,
                want["nobs"].as_u64().unwrap() as usize,
                "{name}/{key} nobs"
            );
            assert_eq!(r.nobs, case["n_obs_lagged"].as_u64().unwrap() as usize);
            assert_eq!(
                r.df_resid,
                want["df_resid"].as_u64().unwrap() as usize,
                "{name}/{key} df"
            );
            let (wp, wb, wt) = (
                f64s(&want["params"]),
                f64s(&want["bse"]),
                f64s(&want["tstats"]),
            );
            let wc = matrix(&want["cov"]);
            let kk = r.params.len();
            for j in 0..kk {
                assert_close(
                    r.params[j],
                    wp[j],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} params[{j}]"),
                );
                assert_close(
                    r.bse[j],
                    wb[j],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} bse[{j}]"),
                );
                assert_close(
                    r.tvalues[j],
                    wt[j],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} t[{j}]"),
                );
                for b in 0..kk {
                    assert_close(
                        r.cov[(j, b)],
                        wc[j][b],
                        RTOL,
                        1e-14,
                        &format!("{name}/{key} cov[{j},{b}]"),
                    );
                }
            }
            // Documented-formula leg on regressor 0.
            let delta = &case["delta"][key];
            let ce = f64s(&delta["cumulative_effect"]);
            let cs = f64s(&delta["cumulative_se"]);
            let lo = f64s(&delta["cumulative_ci_low"]);
            let hi = f64s(&delta["cumulative_ci_high"]);
            for p in 0..powers {
                assert_close(
                    r.cumulative_effect[0][p],
                    ce[p],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} B_{p}"),
                );
                assert_close(
                    r.cumulative_se[0][p],
                    cs[p],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} se(B_{p})"),
                );
                assert_close(
                    r.cumulative_ci_low[0][p],
                    lo[p],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} lo_{p}"),
                );
                assert_close(
                    r.cumulative_ci_high[0][p],
                    hi[p],
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} hi_{p}"),
                );
            }
            if powers == 2 {
                let pts_used = case
                    .get("eval_points_used")
                    .filter(|v| v.is_array())
                    .map(f64s)
                    .or_else(|| eval_points.clone())
                    .unwrap();
                let got_pts = &r.eval_points.as_ref().unwrap()[0];
                let me = f64s(&delta["marginal_effect"]);
                let mse = f64s(&delta["marginal_se"]);
                for (q, &x) in pts_used.iter().enumerate() {
                    assert_close(
                        got_pts[q],
                        x,
                        1e-12,
                        1e-12,
                        &format!("{name}/{key} eval_points[{q}]"),
                    );
                    assert_close(
                        r.marginal_effect.as_ref().unwrap()[0][q],
                        me[q],
                        RTOL,
                        1e-12,
                        "marginal",
                    );
                    assert_close(
                        r.marginal_se.as_ref().unwrap()[0][q],
                        mse[q],
                        RTOL,
                        1e-12,
                        "marginal se",
                    );
                }
                assert_close(
                    r.turning_point.as_ref().unwrap()[0],
                    delta["turning_point"].as_f64().unwrap(),
                    RTOL,
                    1e-12,
                    "x*",
                );
                assert_close(
                    r.turning_point_se.as_ref().unwrap()[0],
                    delta["turning_point_se"].as_f64().unwrap(),
                    RTOL,
                    1e-12,
                    "se(x*)",
                );
            } else {
                assert!(r.eval_points.is_none() && r.turning_point.is_none());
            }
        }
    }
}

#[test]
fn lp_cases_match_linearmodels_per_horizon() {
    let fx = load("panel_unbalanced.json");
    let ds = &fx["datasets"]["lp_synthetic"];
    let y = to_mat(&matrix(&ds["y"]));
    let shock = f64s(&ds["shock"]);
    let mask = mask_of(&ds["mask"]);
    let data = PanelData::unbalanced(y, vec![], &mask).unwrap();
    let cases = fx["lp_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let hmax = case["max_horizon"].as_u64().unwrap() as usize;
        let n_lags = case["n_lag_controls"].as_u64().unwrap() as usize;
        for (key, se_type) in se_menu() {
            let mut cfg = PanelLpConfig::new(hmax, n_lags, se_type);
            cfg.cumulative = case["cumulative"].as_bool().unwrap();
            let r = panel_lp(&data, &shock, &cfg).unwrap_or_else(|e| panic!("{name}/{key}: {e}"));
            assert_eq!(r.bias_correction, LpBiasCorrection::None);
            for (h, want) in case["horizons"].as_array().unwrap().iter().enumerate() {
                assert_eq!(
                    r.nobs[h],
                    want["nobs"].as_u64().unwrap() as usize,
                    "{name}/{key} nobs[{h}]"
                );
                assert_close(
                    r.irf[h],
                    want["irf"][key].as_f64().unwrap(),
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} irf[{h}]"),
                );
                assert_close(
                    r.se[h],
                    want["se"][key].as_f64().unwrap(),
                    RTOL,
                    1e-12,
                    &format!("{name}/{key} se[{h}]"),
                );
                let wp = f64s(&want["params"]);
                for (j, &p) in wp.iter().enumerate() {
                    assert_close(
                        r.params[h][j],
                        p,
                        RTOL,
                        1e-12,
                        &format!("{name}/{key} params[{h}][{j}]"),
                    );
                }
            }
        }
    }
}

/// The balanced inputs of the 0.9.0 bitwise snapshot.
fn snapshot_inputs() -> (
    Mat<f64>,
    Mat<f64>,
    Mat<f64>,
    Vec<f64>,
    Mat<f64>,
    Vec<Mat<f64>>,
) {
    let fx = load("panel_balanced_snapshot.json");
    let inp = &fx["inputs"];
    let y = to_mat(&matrix(&inp["y"]));
    let temp = to_mat(&matrix(&inp["temp"]));
    let x2 = to_mat(&matrix(&inp["x2"]));
    let shock = f64s(&inp["shock"]);
    let treat = to_mat(&matrix(&inp["treat"]));
    let ents: Vec<Mat<f64>> = inp["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| to_mat(&matrix(e)))
        .collect();
    (y, temp, x2, shock, treat, ents)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn mat_bits(m: &Mat<f64>) -> Vec<u64> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.push(m[(i, j)].to_bits());
        }
    }
    out
}

#[test]
fn fully_observed_mask_is_bit_identical_to_the_balanced_path() {
    let (y, temp, x2, shock, treat, _) = snapshot_inputs();
    let (n, t) = (y.nrows(), y.ncols());
    let all = vec![vec![true; t]; n];
    let regs = || {
        vec![
            ("temp".to_string(), temp.clone()),
            ("x2".to_string(), x2.clone()),
        ]
    };
    let bal = PanelData::balanced(y.clone(), regs()).unwrap();
    let unb = PanelData::unbalanced(y.clone(), regs(), &all).unwrap();
    assert!(unb.is_balanced() && unb.mask().is_none() && unb.nobs() == n * t);
    let menus = [
        FixedEffects::ENTITY,
        FixedEffects::TWO_WAY,
        FixedEffects {
            entity: false,
            time: true,
            entity_trends: false,
        },
        FixedEffects {
            entity: true,
            time: false,
            entity_trends: true,
        },
        FixedEffects {
            entity: true,
            time: true,
            entity_trends: true,
        },
    ];
    for effects in menus {
        let a = panel_ols_fe_with(&bal, effects).unwrap();
        let b = panel_ols_fe_with(&unb, effects).unwrap();
        assert_eq!(bits(&a.params), bits(&b.params));
        assert_eq!(bits(a.within_residuals()), bits(b.within_residuals()));
        assert_eq!(
            (a.nobs, a.df_resid, a.n_absorbed),
            (b.nobs, b.df_resid, b.n_absorbed)
        );
        for (_, se_type) in se_menu() {
            let ia = a.inference(se_type).unwrap();
            let ib = b.inference(se_type).unwrap();
            assert_eq!(bits(&ia.bse), bits(&ib.bse));
            assert_eq!(bits(&ia.tvalues), bits(&ib.tvalues));
            assert_eq!(mat_bits(&ia.cov), mat_bits(&ib.cov));
        }
        let cfg = DistributedLagConfig {
            lags: 2,
            powers: 2,
            effects,
            se_type: PanelSeType::ClusterEntity,
            eval_points: None,
        };
        let da = panel_distributed_lag(&bal, &cfg).unwrap();
        let db = panel_distributed_lag(&unb, &cfg).unwrap();
        assert_eq!(bits(&da.params), bits(&db.params));
        assert_eq!(bits(&da.bse), bits(&db.bse));
        assert_eq!(mat_bits(&da.cov), mat_bits(&db.cov));
        assert_eq!(bits(&da.cumulative_se[0]), bits(&db.cumulative_se[0]));
        assert_eq!(
            bits(&da.turning_point.unwrap()),
            bits(&db.turning_point.unwrap())
        );
        assert_eq!(
            bits(&da.eval_points.unwrap()[0]),
            bits(&db.eval_points.unwrap()[0])
        );
    }
    let bal0 = PanelData::balanced(y.clone(), vec![]).unwrap();
    let unb0 = PanelData::unbalanced(y.clone(), vec![], &all).unwrap();
    for (jk, bc) in [
        (false, LpBiasCorrection::None),
        (true, LpBiasCorrection::None),
        (false, LpBiasCorrection::Spj),
    ] {
        let mut cfg = PanelLpConfig::new(4, 2, PanelSeType::DriscollKraay { bandwidth: 4.0 });
        cfg.jackknife = jk;
        cfg.bias_correction = bc;
        let a = panel_lp(&bal0, &shock, &cfg).unwrap();
        let b = panel_lp(&unb0, &shock, &cfg).unwrap();
        assert_eq!(bits(&a.irf), bits(&b.irf));
        assert_eq!(bits(&a.se), bits(&b.se));
        assert_eq!(a.nobs, b.nobs);
    }
    let mut did = LpDidConfig::new(3, 4);
    did.pooled = true;
    let a = lp_did(&bal0, treat.as_ref(), &did).unwrap();
    let b = lp_did(&unb0, treat.as_ref(), &did).unwrap();
    assert_eq!(bits(&a.coef), bits(&b.coef));
    assert_eq!(bits(&a.se), bits(&b.se));
}

/// A masked cell may hold anything: NaN there changes nothing, while NaN
/// in an observed cell is refused naming the input.
#[test]
fn masked_out_cells_are_never_read() {
    let fx = load("panel_unbalanced.json");
    let ds = &fx["datasets"]["synthetic"];
    let y = matrix(&ds["y"]);
    let x = matrix(&ds["x"]);
    let mask = mask_of(&ds["mask"]);
    let regs = |x: &Vec<Vec<f64>>| vec![("x".to_string(), to_mat(x))];
    let a = PanelData::unbalanced(to_mat(&y), regs(&x), &mask).unwrap();
    // Overwrite every masked-out cell with garbage of both kinds.
    let mut y2 = y.clone();
    let mut x2 = x.clone();
    for (i, row) in mask.iter().enumerate() {
        for (t, &m) in row.iter().enumerate() {
            if !m {
                y2[i][t] = if (i + t) % 2 == 0 { 1e300 } else { f64::NAN };
                x2[i][t] = f64::INFINITY;
            }
        }
    }
    let b = PanelData::unbalanced(to_mat(&y2), regs(&x2), &mask).unwrap();
    let fa = panel_ols_fe_with(&a, FixedEffects::TWO_WAY).unwrap();
    let fb = panel_ols_fe_with(&b, FixedEffects::TWO_WAY).unwrap();
    assert_eq!(bits(&fa.params), bits(&fb.params));
    assert_eq!(
        bits(
            &fa.inference(PanelSeType::DriscollKraay { bandwidth: 4.0 })
                .unwrap()
                .bse
        ),
        bits(
            &fb.inference(PanelSeType::DriscollKraay { bandwidth: 4.0 })
                .unwrap()
                .bse
        )
    );
    let cfg = DistributedLagConfig::new(2, PanelSeType::ClusterEntity);
    let da = panel_distributed_lag(&a, &cfg).unwrap();
    let db = panel_distributed_lag(&b, &cfg).unwrap();
    assert_eq!(bits(&da.params), bits(&db.params));
    assert_eq!(bits(&da.cumulative_se[0]), bits(&db.cumulative_se[0]));
    // NaN in an OBSERVED cell is still refused, naming the input.
    let (i0, t0) = (0..mask.len())
        .flat_map(|i| (0..mask[i].len()).map(move |t| (i, t)))
        .find(|&(i, t)| mask[i][t])
        .unwrap();
    let mut y3 = y.clone();
    y3[i0][t0] = f64::NAN;
    let err = PanelData::unbalanced(to_mat(&y3), regs(&x), &mask).unwrap_err();
    assert!(
        matches!(err, PanelError::NonFinite { what: "outcome" }),
        "{err}"
    );
    assert!(err.to_string().contains("mask="), "{err}");
    let mut x3 = x.clone();
    x3[i0][t0] = f64::INFINITY;
    let err = PanelData::unbalanced(to_mat(&y), regs(&x3), &mask).unwrap_err();
    assert!(
        matches!(err, PanelError::NonFinite { what: "regressors" }),
        "{err}"
    );
    // And the balanced constructor refuses NaN anywhere, pointing at the mask.
    let err = PanelData::balanced(to_mat(&y), regs(&x)).unwrap_err();
    assert!(err.to_string().contains("observation mask"), "{err}");
}

#[test]
fn refusals_name_the_mask_parameter() {
    let fx = load("panel_unbalanced.json");
    let ds = &fx["datasets"]["lp_synthetic"];
    let y = to_mat(&matrix(&ds["y"]));
    let shock = f64s(&ds["shock"]);
    let mask = mask_of(&ds["mask"]);
    let (n, t) = (y.nrows(), y.ncols());
    // Mask dimensions.
    let short = mask[..n - 1].to_vec();
    let err = PanelData::unbalanced(y.clone(), vec![], &short).unwrap_err();
    assert!(
        err.to_string().starts_with("dimension mismatch: mask:"),
        "{err}"
    );
    let mut ragged = mask.clone();
    ragged[0].pop();
    let err = PanelData::unbalanced(y.clone(), vec![], &ragged).unwrap_err();
    assert!(err.to_string().contains("mask: every row"), "{err}");
    let none = vec![vec![false; t]; n];
    let err = PanelData::unbalanced(y.clone(), vec![], &none).unwrap_err();
    assert!(
        err.to_string()
            .contains("mask: the observation mask marks no cell"),
        "{err}"
    );
    // LP-DiD and the half-panel jackknives refuse unbalanced panels.
    let data = PanelData::unbalanced(y.clone(), vec![], &mask).unwrap();
    let treat = Mat::from_fn(n, t, |i, tt| f64::from(tt >= 10 + 3 * i));
    let err = lp_did(&data, treat.as_ref(), &LpDidConfig::new(2, 3)).unwrap_err();
    assert!(matches!(err, PanelError::Unbalanced { .. }));
    assert!(err.to_string().contains("mask=None"), "{err}");
    for (jk, bc) in [
        (true, LpBiasCorrection::None),
        (false, LpBiasCorrection::DhaeneJochmans),
        (false, LpBiasCorrection::Spj),
    ] {
        let mut cfg = PanelLpConfig::new(3, 1, PanelSeType::ClusterEntity);
        cfg.jackknife = jk;
        cfg.bias_correction = bc;
        let err = panel_lp(&data, &shock, &cfg).unwrap_err();
        assert!(matches!(err, PanelError::Unbalanced { .. }), "{err}");
        assert!(
            err.to_string().contains("bias_correction") && err.to_string().contains("mask"),
            "{err}"
        );
    }
    // A mask that leaves no lagged row (on the NaN-free snapshot panel).
    let (yb, temp, _, _, _, _) = snapshot_inputs();
    let (nb, tb) = (yb.nrows(), yb.ncols());
    let sparse: Vec<Vec<bool>> = (0..nb)
        .map(|_| (0..tb).map(|tt| tt % 3 == 0).collect())
        .collect();
    let data = PanelData::unbalanced(yb, vec![("temp".to_string(), temp)], &sparse).unwrap();
    let err = panel_distributed_lag(
        &data,
        &DistributedLagConfig::new(2, PanelSeType::ClusterEntity),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            PanelError::InsufficientObservations {
                needed: 3,
                got: 1,
                ..
            }
        ),
        "{err}"
    );
    assert!(err.to_string().contains("(mask)"), "{err}");
    // Mean-group VAR: an internal NaN names the entity and the ragged-list route.
    let (_, _, _, _, _, ents) = snapshot_inputs();
    let mut bad = ents.clone();
    bad[1][(5, 0)] = f64::NAN;
    let err = mean_group_var(&bad, 1, Trend::Constant, 3).unwrap_err();
    assert!(matches!(err, PanelError::NonFiniteEntity { entity: 1 }));
    assert!(err.to_string().starts_with("entities[1]:"), "{err}");
    // A shorter entity matrix (entry/exit) is the supported ragged form.
    let mut ragged = ents.clone();
    ragged[0] = Mat::from_fn(20, 2, |i, j| ents[0][(i + 5, j)]);
    assert!(mean_group_var(&ragged, 1, Trend::Constant, 3).is_ok());
}

/// Horizon 0 without lag controls is the within regression of `y` on
/// the broadcast shock: the masked LP and the masked FE paths must agree
/// exactly (both use the observed rows in the same order).
#[test]
fn masked_lp_at_horizon_zero_is_the_masked_within_regression() {
    let fx = load("panel_unbalanced.json");
    let ds = &fx["datasets"]["lp_synthetic"];
    let y = to_mat(&matrix(&ds["y"]));
    let shock = f64s(&ds["shock"]);
    let mask = mask_of(&ds["mask"]);
    let n = y.nrows();
    let data = PanelData::unbalanced(y.clone(), vec![], &mask).unwrap();
    let cfg = PanelLpConfig::new(0, 0, PanelSeType::DriscollKraay { bandwidth: 4.0 });
    let lp = panel_lp(&data, &shock, &cfg).unwrap();
    let regs = vec![("shock".to_string(), PanelData::broadcast_common(&shock, n))];
    let fe_data = PanelData::unbalanced(y, regs, &mask).unwrap();
    let fe = panel_ols_fe(&fe_data).unwrap();
    let inf = fe
        .inference(PanelSeType::DriscollKraay { bandwidth: 4.0 })
        .unwrap();
    assert_eq!(lp.irf[0].to_bits(), fe.params[0].to_bits());
    assert_eq!(lp.se[0].to_bits(), inf.bse[0].to_bits());
    assert_eq!(lp.nobs[0], fe.nobs);
    assert_eq!(fe.nobs, ds["n_obs"].as_u64().unwrap() as usize);
}
