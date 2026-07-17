//! Golden-value tests against `fixtures/panel.json` (generated with
//! `linearmodels` 7.0; see `fixtures/generate_panel_fixtures.py`).
//!
//! The `panel_ols_fe_s0_s1_drop_t0` block pins the within (entity-FE)
//! estimator's slopes, standard errors, and t-statistics for three
//! covariance estimators — nonrobust, entity-clustered, and
//! Driscoll-Kraay (Bartlett, bandwidth 4). The design regresses
//! `y_{i,t}` on `shock_t` and `shock_{t-1}` with entity effects, dropping
//! period 0, exactly as the fixture generator does.
//!
//! Tolerance is 1e-6 relative on slopes, standard errors, and
//! t-statistics — and that ceiling is set by the FIXTURE, not the
//! estimator. The generator wrote `y` and `shock` rounded to six decimal
//! places but computed the golden PanelOLS fit from the full-precision
//! arrays, so a from-scratch fit on the stored (rounded) inputs can only
//! agree to ~1e-6. The estimator itself is exact: the companion
//! `panel_lp_recovers_known_irf` test confirms the within machinery
//! recovers the panel's analytically-known impulse response 0.8*0.6^h.

use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_panel::{panel_lp, panel_ols_fe, PanelData, PanelLpConfig, PanelSeType};

fn load() -> Value {
    let path = format!("{}/../../fixtures/panel.json", env!("CARGO_MANIFEST_DIR"));
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

/// Rebuild the fixture's PanelOLS design: outcome `y[:, 1..]`, regressors
/// `s0 = shock[1..]` and `s1 = shock[..T-1]`, broadcast across entities.
fn build_panel(fx: &Value) -> PanelData {
    let panel = &fx["panel"];
    let y = matrix(&panel["y"]); // N x T
    let shock = f64s(&panel["shock"]); // T
    let n = y.len();
    let t = y[0].len();
    let t1 = t - 1;

    let outcome = Mat::from_fn(n, t1, |i, tt| y[i][tt + 1]);
    let s0 = Mat::from_fn(n, t1, |_, tt| shock[tt + 1]);
    let s1 = Mat::from_fn(n, t1, |_, tt| shock[tt]);
    PanelData::balanced(
        outcome,
        vec![("s0".to_string(), s0), ("s1".to_string(), s1)],
    )
    .expect("balanced panel")
}

fn assert_close(got: f64, want: f64, rtol: f64, what: &str) {
    let denom = want.abs().max(1e-12);
    assert!(
        ((got - want) / denom).abs() < rtol,
        "{what}: got {got}, want {want} (rel {})",
        ((got - want) / denom).abs()
    );
}

#[test]
fn fe_within_matches_linearmodels() {
    let fx = load();
    let data = build_panel(&fx);
    let fit = panel_ols_fe(&data).expect("fit");
    let block = &fx["panel_ols_fe_s0_s1_drop_t0"];

    // Slopes are shared across covariance estimators.
    let want_params = &block["nonrobust"]["params"];
    for (j, name) in ["s0", "s1"].iter().enumerate() {
        assert_close(
            fit.params[j],
            want_params[name].as_f64().unwrap(),
            1e-6,
            &format!("param {name}"),
        );
    }

    let cases = [
        ("nonrobust", PanelSeType::NonRobust),
        ("cluster_entity", PanelSeType::ClusterEntity),
        (
            "driscoll_kraay",
            PanelSeType::DriscollKraay { bandwidth: 4.0 },
        ),
    ];
    for (key, se) in cases {
        let inf = fit.inference(se).expect("inference");
        let wb = &fx["panel_ols_fe_s0_s1_drop_t0"][key];
        for (j, name) in ["s0", "s1"].iter().enumerate() {
            assert_close(
                inf.bse[j],
                wb["bse"][name].as_f64().unwrap(),
                1e-6,
                &format!("{key} bse {name}"),
            );
            assert_close(
                inf.tvalues[j],
                wb["tstats"][name].as_f64().unwrap(),
                1e-6,
                &format!("{key} tstat {name}"),
            );
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn panel_lp_recovers_known_irf() {
    // The fixture panel has a known true response psi_h = 0.8 * 0.6^h to the
    // common shock; a panel LP must cover it within Driscoll-Kraay bands.
    let fx = load();
    let panel = &fx["panel"];
    let y = matrix(&panel["y"]);
    let shock = f64s(&panel["shock"]);
    let true_irf = f64s(&panel["true_irf_psi"]);
    let n = y.len();
    let t = y[0].len();

    let outcome = Mat::from_fn(n, t, |i, tt| y[i][tt]);
    let data = PanelData::balanced(outcome, vec![]).expect("outcome-only panel");

    let cfg = PanelLpConfig::new(6, 2, PanelSeType::DriscollKraay { bandwidth: 4.0 });
    let res = panel_lp(&data, &shock, &cfg).expect("panel lp");

    // Impact and one-period responses should be close to the truth; every
    // horizon should be covered by a 4-sigma Driscoll-Kraay band.
    for h in 0..=6usize {
        let irf = res.irf[h];
        let se = res.se[h];
        let psi = true_irf[h];
        assert!(
            (irf - psi).abs() < 4.0 * se + 0.05,
            "h={h}: irf {irf} not within 4 s.e. ({se}) of true {psi}"
        );
    }
    // Impact estimate is tight.
    assert!(
        (res.irf[0] - true_irf[0]).abs() < 0.1,
        "impact estimate off"
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn mean_group_var_recovers_common_dynamics() {
    // A homogeneous panel: every entity is the same stable VAR(1) driven by
    // its own shocks. The mean-group estimator must recover the common
    // coefficient matrix, and its cross-entity dispersion must be positive
    // and modest. Data are generated with a tiny in-test LCG (no dependency
    // on tsecon-rng) so the test is self-contained and deterministic.
    use tsecon_panel::{mean_group_var, tsecon_var::Trend};

    let a = [[0.5_f64, 0.1], [0.0, 0.4]];
    let (n_ent, t_len) = (25usize, 300usize);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut normal = || {
        // Two uniforms via an LCG -> Box-Muller.
        let mut u = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
        };
        let (u1, u2) = (u().max(1e-12), u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };

    let mut entities = Vec::with_capacity(n_ent);
    for _ in 0..n_ent {
        let mut y = vec![[0.0_f64; 2]; t_len];
        for tt in 1..t_len {
            for i in 0..2 {
                y[tt][i] = a[i][0] * y[tt - 1][0] + a[i][1] * y[tt - 1][1] + normal();
            }
        }
        entities.push(Mat::from_fn(t_len, 2, |r, c| y[r][c]));
    }

    let mg = mean_group_var(&entities, 1, Trend::None, 6).expect("mean-group fit");
    assert_eq!(mg.n_entities, n_ent);
    // Recover the common A_1 within Monte Carlo tolerance.
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (mg.coefs[0][(i, j)] - a[i][j]).abs() < 0.05,
                "A1[{i},{j}] = {} not near {}",
                mg.coefs[0][(i, j)],
                a[i][j]
            );
            // Dispersion SE is positive and small on a homogeneous panel.
            assert!(mg.coefs_se[0][(i, j)] > 0.0 && mg.coefs_se[0][(i, j)] < 0.05);
        }
    }
    // Impact IRF is the Cholesky factor (lower-triangular): upper-right is 0.
    assert!(mg.orth_irfs[0][(0, 1)].abs() < 1e-9);
}
