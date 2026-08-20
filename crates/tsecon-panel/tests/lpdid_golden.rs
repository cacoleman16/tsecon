//! Golden-value tests for `lp_did` against `fixtures/lpdid.json`.
//!
//! The fixture is a **reference-run golden** (see
//! `fixtures/generate_lpdid_fixtures.py`): the stored values come from an
//! R/fixest run of the Dube-Girardi-Jordà-Taylor LP-DiD conventions,
//! transcribed line-by-line from the authors' own example implementations
//! (github.com/danielegirardi/lpdid — `LP_DiD_R_example_VW.R`,
//! `LP_DiD_R_example_EW.R`, `LPDiD_nonabsorbing_example.do`; fetched
//! 2026-08-19) by `fixtures/generate_lpdid_fixtures.R`, and cross-checked
//! against an independent NumPy reimplementation (agreement 5.3e-15) at
//! generation time. It is not a run of the SSC-only Stata ado (stated in
//! the generator).
//!
//! Six cases: absorbing staggered adoption — variance-weighted,
//! reweighted (equally-weighted ATT), never-treated-only controls, each
//! with pooled estimates; non-absorbing treatment with stabilization lag
//! 3 — variance-weighted (pooled), reweighted, never-treated-only.
//!
//! Tolerance 1e-10 relative: both sides compute the same estimator
//! through different numerical paths (fixest weighted demeaning vs this
//! crate's explicit cell algebra); `nobs`/`n_switchers` are exact.

use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_panel::{lp_did, LpDidConfig, PanelData};

fn load() -> Value {
    let path = format!("{}/../../fixtures/lpdid.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn mat(v: &Value) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .expect("outer array")
        .iter()
        .map(|r| {
            r.as_array()
                .expect("inner array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
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
fn lp_did_matches_the_fixest_reference_run_at_1e10() {
    let fx = load();
    let panels = [
        ("A", mat(&fx["panel_a"]["y"]), mat(&fx["panel_a"]["d"])),
        ("B", mat(&fx["panel_b"]["y"]), mat(&fx["panel_b"]["d"])),
    ];

    for (name, case) in fx["cases"].as_object().expect("cases") {
        let (_, y, d) = panels
            .iter()
            .find(|(p, _, _)| name.starts_with(*p))
            .expect("panel for case");
        let data = PanelData::balanced(y.clone(), vec![]).expect("balanced panel");

        let mut cfg = LpDidConfig::new(
            case["pre_window"].as_u64().expect("Q") as usize,
            case["post_window"].as_u64().expect("H") as usize,
        );
        cfg.absorbing = case["absorbing"].as_bool().expect("absorbing");
        cfg.nonabsorbing_lag = case["nonabsorbing_lag"].as_u64().expect("lag") as usize;
        cfg.reweight = case["reweight"].as_bool().expect("reweight");
        cfg.never_treated_only = case["never_treated_only"].as_bool().expect("nt");
        cfg.pooled = case["pooled"].as_bool().expect("pooled");

        let res = lp_did(&data, d.as_ref(), &cfg).expect("lp_did");

        // Stamps.
        assert_eq!(res.absorbing, cfg.absorbing, "{name}: absorbing stamp");
        assert_eq!(res.reweight, cfg.reweight, "{name}: reweight stamp");
        assert_eq!(
            res.never_treated_only, cfg.never_treated_only,
            "{name}: never_treated_only stamp"
        );
        assert_eq!(
            res.nonabsorbing_lag, cfg.nonabsorbing_lag,
            "{name}: nonabsorbing_lag stamp"
        );

        // Horizon grid: -Q..=H, with the -1 baseline identically zero.
        let q = cfg.pre_window as i64;
        let grid: Vec<i64> = (-q..=cfg.post_window as i64).collect();
        assert_eq!(res.horizons, grid, "{name}: horizon grid");
        let at = |h: i64| (h + q) as usize;
        assert_eq!(res.coef[at(-1)], 0.0, "{name}: baseline coef");
        assert_eq!(res.se[at(-1)], 0.0, "{name}: baseline se");
        assert_eq!(res.nobs[at(-1)], 0, "{name}: baseline nobs");

        let results = case["results"].as_object().expect("results");
        for (key, want) in results {
            let coef = want["coef"].as_f64().expect("coef");
            let se = want["se"].as_f64().expect("se");
            let nobs = want["nobs"].as_u64().expect("nobs") as usize;
            let nsw = want["n_switchers"].as_u64().expect("n_switchers") as usize;
            let (got_c, got_s, got_n, got_w) = match key.as_str() {
                "pooled_post" => {
                    let p = res.pooled_post.as_ref().expect("pooled_post present");
                    (p.att, p.se, p.nobs, p.n_switchers)
                }
                "pooled_pre" => {
                    let p = res.pooled_pre.as_ref().expect("pooled_pre present");
                    (p.att, p.se, p.nobs, p.n_switchers)
                }
                h => {
                    let k = at(h.parse::<i64>().expect("horizon key"));
                    (res.coef[k], res.se[k], res.nobs[k], res.n_switchers[k])
                }
            };
            assert_close(got_c, coef, 1e-10, &format!("{name} {key} coef"));
            assert_close(got_s, se, 1e-10, &format!("{name} {key} se"));
            assert_eq!(got_n, nobs, "{name} {key}: nobs");
            assert_eq!(got_w, nsw, "{name} {key}: n_switchers");
        }
        if !cfg.pooled {
            assert!(res.pooled_post.is_none(), "{name}: pooled_post absent");
            assert!(res.pooled_pre.is_none(), "{name}: pooled_pre absent");
        }
    }
}
