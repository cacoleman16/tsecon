//! Golden-value tests for the static-panel menu against
//! `fixtures/panel_static.json` (linearmodels 7.0 by
//! `fixtures/generate_panel_static_fixtures.py`; the EmplUK inputs come
//! from `fixtures/panel_unbalanced.json`).
//!
//! * `re_cases`: `RandomEffects` with `small_sample` off and on, under the
//!   nonrobust, HC1 (`"robust"`), entity-clustered and Driscoll-Kraay
//!   (Bartlett, bandwidth 4) covariances, on EmplUK, a seeded ragged
//!   panel, a balanced one, a time-invariant regressor, a truncated effect
//!   variance (`sigma2_effects == 0` exactly), a common trend (rank-
//!   deficient between step), year dummies (between rank 7 of 11), the
//!   Mundlak design, no constant, and the intercept-only model;
//! * `be_cases`: `BetweenOLS` with `reweight` off and on, three
//!   covariances;
//! * `fd_cases`: `FirstDifferenceOLS` (and `PooledOLS` on the
//!   hand-differenced frame for the intercept variant), four covariances;
//! * `fm_cases`: `FamaMacBeth` with the Fama-MacBeth and Bartlett-HAC
//!   standard errors, including a skipped period.
//!
//! Coefficients, standard errors, t-statistics and the full covariance at
//! 1e-10 relative (covariance floor 1e-14); counts exact.

mod static_common;

use serde_json::Value;
use static_common::{assert_close, case_panel, f64s, load, matrix};
use tsecon_panel::{
    fama_macbeth, panel_between, panel_first_difference, panel_random_effects, BetweenConfig,
    FamaMacBethSe, RandomEffectsConfig, StaticInference, StaticSeType,
};

const RTOL: f64 = 1e-10;
const COV_FLOOR: f64 = 1e-14;

fn static_menu() -> [(&'static str, StaticSeType); 4] {
    [
        ("nonrobust", StaticSeType::NonRobust),
        ("hc1", StaticSeType::Hc1),
        ("cluster", StaticSeType::ClusterEntity),
        (
            "driscoll_kraay",
            StaticSeType::DriscollKraay { bandwidth: 4.0 },
        ),
    ]
}

fn check_vec(got: &[f64], want: &Value, what: &str) {
    let want = f64s(want);
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (j, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_close(*g, *w, RTOL, COV_FLOOR, &format!("{what}[{j}]"));
    }
}

fn check_inference(
    params: &[f64],
    inf_cov: &[Vec<f64>],
    bse: &[f64],
    tv: &[f64],
    want: &Value,
    what: &str,
) {
    check_vec(params, &want["params"], &format!("{what} params"));
    check_vec(bse, &want["bse"], &format!("{what} bse"));
    check_vec(tv, &want["tvalues"], &format!("{what} tvalues"));
    let wc = matrix(&want["cov"]);
    assert_eq!(wc.len(), inf_cov.len(), "{what}: cov shape");
    for (a, row) in wc.iter().enumerate() {
        for (b, w) in row.iter().enumerate() {
            // Off-diagonal entries are compared on the correlation scale
            // (relative to sqrt(V_aa V_bb)) when they are smaller than it:
            // a covariance that is zero in exact arithmetic (the constant
            // against a slope in the Mundlak design) carries ~1e-15 of
            // summation-order noise in both implementations.
            let scale = (wc[a][a].abs() * wc[b][b].abs()).sqrt().max(COV_FLOOR);
            assert_close(
                inf_cov[a][b],
                *w,
                RTOL,
                scale,
                &format!("{what} cov[{a}][{b}]"),
            );
        }
    }
}

fn cov_rows(inf: &StaticInference) -> Vec<Vec<f64>> {
    let k = inf.cov.nrows();
    (0..k)
        .map(|a| (0..k).map(|b| inf.cov[(a, b)]).collect())
        .collect()
}

fn names_of(case: &Value) -> Vec<String> {
    case["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

fn usize_of(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

#[test]
fn random_effects_match_linearmodels() {
    let fx = load("panel_static.json");
    let cases = fx["re_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 14);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let data = case_panel(&fx, case);
        let cfg = RandomEffectsConfig {
            constant: case["constant"].as_bool().unwrap(),
            small_sample: case["small_sample"].as_bool().unwrap(),
        };
        let fit = panel_random_effects(&data, cfg).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(fit.nobs, usize_of(&case["nobs"]), "{name} nobs");
        assert_eq!(
            fit.n_entities,
            usize_of(&case["n_entities"]),
            "{name} n_entities"
        );
        assert_eq!(fit.df_resid, usize_of(&case["df_resid"]), "{name} df_resid");
        assert_eq!(fit.names, names_of(case), "{name} names");
        assert_close(
            fit.sigma2_eps,
            case["sigma2_eps"].as_f64().unwrap(),
            RTOL,
            0.0,
            &format!("{name} sigma2_eps"),
        );
        let s2u = case["sigma2_effects"].as_f64().unwrap();
        if s2u == 0.0 {
            assert_eq!(
                fit.sigma2_effects, 0.0,
                "{name}: truncated sigma2_effects must be exactly 0"
            );
            assert_eq!(fit.rho, 0.0, "{name} rho");
        } else {
            assert_close(
                fit.sigma2_effects,
                s2u,
                RTOL,
                0.0,
                &format!("{name} sigma2_effects"),
            );
            assert_close(
                fit.rho,
                case["rho"].as_f64().unwrap(),
                RTOL,
                0.0,
                &format!("{name} rho"),
            );
        }
        let theta = f64s(&case["theta"]);
        assert_eq!(fit.theta.len(), theta.len(), "{name} theta length");
        for (i, (g, w)) in fit.theta.iter().zip(&theta).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "{name} theta[{i}] should be NaN");
            } else if *w == 0.0 {
                assert_eq!(*g, 0.0, "{name} theta[{i}]");
            } else {
                assert_close(*g, *w, RTOL, 0.0, &format!("{name} theta[{i}]"));
            }
        }
        for (key, se) in static_menu() {
            let inf = fit.inference(se).unwrap();
            check_inference(
                &fit.params,
                &cov_rows(&inf),
                &inf.bse,
                &inf.tvalues,
                &case["se"][key],
                &format!("RE {name} {key}"),
            );
        }
    }
}

#[test]
fn between_matches_linearmodels() {
    let fx = load("panel_static.json");
    let cases = fx["be_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let data = case_panel(&fx, case);
        let cfg = BetweenConfig {
            constant: case["constant"].as_bool().unwrap(),
            reweight: case["reweight"].as_bool().unwrap(),
        };
        let fit = panel_between(&data, cfg).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(fit.nobs, usize_of(&case["nobs"]), "{name} nobs");
        assert_eq!(
            fit.n_entities,
            usize_of(&case["n_entities"]),
            "{name} n_entities"
        );
        assert_eq!(fit.df_resid, usize_of(&case["df_resid"]), "{name} df_resid");
        assert_eq!(fit.names, names_of(case), "{name} names");
        for (key, se) in static_menu().into_iter().take(3) {
            let inf = fit.inference(se).unwrap();
            check_inference(
                &fit.params,
                &cov_rows(&inf),
                &inf.bse,
                &inf.tvalues,
                &case["se"][key],
                &format!("BE {name} {key}"),
            );
        }
        assert!(fit
            .inference(StaticSeType::DriscollKraay { bandwidth: 4.0 })
            .is_err());
    }
}

#[test]
fn first_difference_matches_linearmodels() {
    let fx = load("panel_static.json");
    let cases = fx["fd_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let data = case_panel(&fx, case);
        let fit = panel_first_difference(&data, case["constant"].as_bool().unwrap())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(fit.nobs, usize_of(&case["nobs"]), "{name} nobs");
        assert_eq!(
            fit.n_entities,
            usize_of(&case["n_entities"]),
            "{name} n_entities"
        );
        assert_eq!(fit.df_resid, usize_of(&case["df_resid"]), "{name} df_resid");
        assert_eq!(fit.names, names_of(case), "{name} names");
        for (key, se) in static_menu() {
            let inf = fit.inference(se).unwrap();
            check_inference(
                &fit.params,
                &cov_rows(&inf),
                &inf.bse,
                &inf.tvalues,
                &case["se"][key],
                &format!("FD {name} {key}"),
            );
        }
    }
}

#[test]
fn fama_macbeth_matches_linearmodels() {
    let fx = load("panel_static.json");
    let cases = fx["fm_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let data = case_panel(&fx, case);
        let fit = fama_macbeth(&data, case["constant"].as_bool().unwrap())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(fit.nobs, usize_of(&case["nobs"]), "{name} nobs");
        assert_eq!(
            fit.n_periods_used,
            usize_of(&case["n_periods_used"]),
            "{name} n_periods_used"
        );
        assert_eq!(fit.names, names_of(case), "{name} names");
        let used: Vec<bool> = case["periods_used"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        assert_eq!(fit.periods_used, used, "{name} periods_used");
        let ppp = matrix(&case["params_per_period"]);
        for (t, (g, w)) in fit.params_per_period.iter().zip(&ppp).enumerate() {
            for (j, (gv, wv)) in g.iter().zip(w).enumerate() {
                if wv.is_nan() {
                    assert!(
                        gv.is_nan(),
                        "{name} params_per_period[{t}][{j}] should be NaN"
                    );
                } else {
                    assert_close(
                        *gv,
                        *wv,
                        RTOL,
                        COV_FLOOR,
                        &format!("{name} params_per_period[{t}][{j}]"),
                    );
                }
            }
        }
        for (key, se) in [
            ("fama_macbeth", FamaMacBethSe::Standard),
            ("hac", FamaMacBethSe::Hac { bandwidth: 4.0 }),
        ] {
            let inf = fit.inference(se).unwrap();
            let k = inf.cov.nrows();
            let cov: Vec<Vec<f64>> = (0..k)
                .map(|a| (0..k).map(|b| inf.cov[(a, b)]).collect())
                .collect();
            check_inference(
                &fit.params,
                &cov,
                &inf.bse,
                &inf.tvalues,
                &case["se"][key],
                &format!("FM {name} {key}"),
            );
        }
    }
}

#[test]
fn the_headline_numbers_are_the_specs() {
    // The numbers the model card quotes, read back from the fixture so a
    // regeneration that moved them would fail here first.
    let fx = load("panel_static.json");
    let re = &fx["re_cases"][0];
    assert_eq!(re["name"], "empluk/ss=False");
    let bse = f64s(&re["se"]["cluster"]["bse"]);
    assert!((bse[0] - 0.33567743).abs() < 5e-9 && (bse[2] - 0.0328584).abs() < 5e-9);
    let fd = &fx["fd_cases"][0];
    assert_eq!(fd["nobs"], 891);
    let m = &fx["mundlak"];
    assert!((m["cluster_wald_mean_coefficients"].as_f64().unwrap() - 14.400176).abs() < 1e-5);
}

/// Driscoll-Kraay at the largest admissible bandwidth — `bandwidth + 1`
/// equal to the periods that carry scores (8 for random effects, 7 for
/// first differences on the balanced 25 x 8 panel), every lag weighted —
/// against linearmodels, which raises one step beyond. tsecon refuses
/// there too, naming `bandwidth`: past the edge the score sums' telescoping
/// (`sum_t a_t = X'e = 0`) makes every standard error shrink like
/// `sqrt(S / (bandwidth + 1))` (the fixture's `beyond` block, at
/// bandwidth 24), to 0 in the limit.
#[test]
fn driscoll_kraay_at_the_largest_admissible_bandwidth_matches_linearmodels() {
    let fx = load("panel_static.json");
    let edge = &fx["dk_bandwidth_edge"];
    let raw = static_common::dataset(&fx, edge["dataset"].as_str().unwrap());
    let data = raw.panel();
    let re = panel_random_effects(&data, RandomEffectsConfig::default()).unwrap();
    let fd = panel_first_difference(&data, false).unwrap();
    type Infer<'a> =
        Box<dyn Fn(StaticSeType) -> Result<StaticInference, tsecon_panel::PanelError> + 'a>;
    let cases: [(&str, &[f64], Infer<'_>); 2] = [
        ("re", &re.params, Box::new(|se| re.inference(se))),
        ("fd", &fd.params, Box::new(|se| fd.inference(se))),
    ];
    for (key, params, infer) in cases {
        let want = &edge[key];
        let s = usize_of(&want["score_periods"]);
        let bw = want["bandwidth"].as_f64().unwrap();
        assert_eq!(bw, (s - 1) as f64);
        let inf = infer(StaticSeType::DriscollKraay { bandwidth: bw }).unwrap();
        check_inference(
            params,
            &cov_rows(&inf),
            &inf.bse,
            &inf.tvalues,
            want,
            &format!("dk edge {key}"),
        );
        // One step beyond — where linearmodels raises — and far beyond.
        for beyond in [s as f64, bw + 0.5, 1e6, 1e300] {
            let msg = infer(StaticSeType::DriscollKraay { bandwidth: beyond })
                .unwrap_err()
                .to_string();
            assert!(
                msg.starts_with("bandwidth:") && msg.contains(&format!("S = {s} here")),
                "{key} at {beyond}: {msg}"
            );
            assert!(msg.contains(&format!("bandwidth <= {}", s - 1)), "{msg}");
        }
    }
    // The documented-formula leg of the degeneracy past the edge.
    let beyond = &edge["beyond"];
    let r = beyond["sqrt_T_over_bandwidth_plus_one"].as_f64().unwrap();
    for v in f64s(&beyond["bse_ratio_to_edge"]) {
        assert!((v / r - 1.0).abs() < 1e-10, "{v} vs {r}");
    }
}
