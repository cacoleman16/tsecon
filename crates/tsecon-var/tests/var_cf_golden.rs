//! Golden tests for the conditional (hard-path) VAR forecast against
//! `fixtures/var_cf.json`: the Doan-Litterman-Sims / Waggoner-Zha closed
//! form transcribed in NumPy (documented formula — every returned quantity)
//! and the Bańbura-Giannone-Lenza Kalman-conditioning route (statsmodels
//! `VARMAX(...NaN future...).smooth(params)`, an independent package —
//! path, covariance and implied shocks), on a seeded simulated VAR(2) and
//! on the transformed US macro system.

mod common;

use common::{as_mat, assert_mat_close, assert_rel_close, load_fixture};
use serde_json::Value;
use tsecon_var::{ConditionalForecast, Trend, VarResults, VarSpec};

fn conditions_of(v: &Value) -> Vec<Vec<Option<f64>>> {
    v.as_array()
        .expect("conditions rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("conditions row")
                .iter()
                .map(Value::as_f64)
                .collect()
        })
        .collect()
}

fn fit_case(fx: &Value, case: &Value) -> (VarResults, ConditionalForecast) {
    let data = as_mat(&fx["series"][case["series"].as_str().expect("series name")]);
    let trend = if case["trend"] == "c" {
        Trend::Constant
    } else {
        Trend::None
    };
    let p = case["p"].as_u64().expect("p") as usize;
    let res = VarSpec::new(p, trend).unwrap().fit(data.as_ref()).unwrap();
    let steps = case["steps"].as_u64().expect("steps") as usize;
    let alpha = case["alpha"].as_f64().expect("alpha");
    let cf = res
        .conditional_forecast(steps, &conditions_of(&case["conditions"]), alpha)
        .unwrap();
    (res, cf)
}

fn cases() -> (Value, Vec<Value>) {
    let fx = load_fixture("var_cf.json");
    let cs = fx["cases"].as_array().expect("cases").clone();
    assert_eq!(cs.len(), 6);
    (fx, cs)
}

/// Every stored quantity of every case reproduces the NumPy transcription
/// of the closed form at 1e-10: conditional path, unconditional path,
/// per-horizon covariance, standard errors, interval bounds, implied
/// reduced-form and orthogonalised shocks, the constrained-cell grid, and
/// the Mahalanobis plausibility statistic with its chi-squared p-value.
#[test]
fn golden_closed_form_every_case() {
    let (fx, cs) = cases();
    for case in &cs {
        let name = case["name"].as_str().unwrap();
        let (_, cf) = fit_case(&fx, case);
        let steps = case["steps"].as_u64().unwrap() as usize;
        assert_eq!(cf.steps, steps, "{name}: steps");
        assert_eq!(
            cf.n_constrained,
            case["n_constrained"].as_u64().unwrap() as usize,
            "{name}: n_constrained"
        );
        assert_mat_close(&cf.point, &case["point"], 1e-10, &format!("{name}: point"));
        assert_mat_close(
            &cf.unconditional,
            &case["unconditional"],
            1e-10,
            &format!("{name}: unconditional"),
        );
        assert_mat_close(&cf.se, &case["se"], 1e-10, &format!("{name}: se"));
        assert_mat_close(
            &cf.unconditional_se,
            &case["unconditional_se"],
            1e-10,
            &format!("{name}: unconditional_se"),
        );
        assert_mat_close(&cf.lower, &case["lower"], 1e-10, &format!("{name}: lower"));
        assert_mat_close(&cf.upper, &case["upper"], 1e-10, &format!("{name}: upper"));
        assert_mat_close(
            &cf.shocks,
            &case["shocks"],
            1e-10,
            &format!("{name}: shocks"),
        );
        assert_mat_close(
            &cf.orth_shocks,
            &case["orth_shocks"],
            1e-10,
            &format!("{name}: orth_shocks"),
        );
        assert_eq!(cf.cov.len(), steps, "{name}: cov length");
        for h in 0..steps {
            assert_mat_close(
                &cf.cov[h],
                &case["cov"][h],
                1e-10,
                &format!("{name}: cov[{h}]"),
            );
        }
        assert_rel_close(
            cf.mahalanobis,
            case["mahalanobis"].as_f64().unwrap(),
            1e-10,
            &format!("{name}: mahalanobis"),
        );
        assert_rel_close(
            cf.mahalanobis_pvalue,
            case["mahalanobis_pvalue"].as_f64().unwrap(),
            1e-10,
            &format!("{name}: mahalanobis_pvalue"),
        );
        let grid = case["constrained"].as_array().unwrap();
        for h in 0..steps {
            let row = grid[h].as_array().unwrap();
            for (j, c) in row.iter().enumerate() {
                assert_eq!(
                    cf.constrained[h][j],
                    c.as_bool().unwrap(),
                    "{name}: constrained[{h}][{j}]"
                );
            }
        }
    }
}

/// The Kalman-conditioning leg (statsmodels VARMAX smoother with the
/// unconstrained future set missing, Bańbura-Giannone-Lenza 2015) agrees
/// with the crate's closed form at 1e-8 on the path, the per-horizon
/// covariance and the implied shocks — the independent-package golden.
#[test]
fn golden_kalman_conditioning_every_case() {
    let (fx, cs) = cases();
    for case in &cs {
        let name = case["name"].as_str().unwrap();
        let (_, cf) = fit_case(&fx, case);
        let steps = cf.steps;
        assert_mat_close(
            &cf.point,
            &case["bgl_point"],
            1e-8,
            &format!("{name}: point vs VARMAX smoothed_state"),
        );
        assert_mat_close(
            &cf.shocks,
            &case["bgl_shocks"],
            1e-8,
            &format!("{name}: shocks vs VARMAX smoothed_state_disturbance"),
        );
        for h in 0..steps {
            assert_mat_close(
                &cf.cov[h],
                &case["bgl_cov"][h],
                1e-8,
                &format!("{name}: cov[{h}] vs VARMAX smoothed_state_cov"),
            );
        }
        // The generator measured the two legs against each other too; the
        // recorded deviation must be at the roundoff level it was asserted at.
        for key in ["point", "cov", "shocks"] {
            let dev = case["bgl_max_abs_dev"][key].as_f64().unwrap();
            assert!(
                dev < 1e-10,
                "{name}: generator-recorded {key} deviation {dev}"
            );
        }
    }
}

/// The unconditional path is `VarResults::forecast` bitwise (the same
/// recursion) and statsmodels `VARResults.forecast` at 1e-10.
#[test]
fn unconditional_path_is_the_plain_forecast() {
    let (fx, cs) = cases();
    for case in &cs {
        let name = case["name"].as_str().unwrap();
        let (res, cf) = fit_case(&fx, case);
        let fc = res.forecast(cf.steps).unwrap();
        for h in 0..cf.steps {
            for j in 0..res.neqs {
                assert_eq!(
                    cf.unconditional[(h, j)].to_bits(),
                    fc[(h, j)].to_bits(),
                    "{name}: unconditional[({h},{j})] not bitwise forecast"
                );
            }
        }
        assert_mat_close(
            &fc,
            &case["unconditional"],
            1e-10,
            &format!("{name}: forecast"),
        );
        // Its se is the marginal forecast-interval se, also bitwise.
        let fi = res.forecast_interval(cf.steps, cf.alpha).unwrap();
        for h in 0..cf.steps {
            for j in 0..res.neqs {
                assert_eq!(
                    cf.unconditional_se[(h, j)].to_bits(),
                    fi.se[(h, j)].to_bits(),
                    "{name}: unconditional_se[({h},{j})]"
                );
            }
        }
    }
}

/// Constrained cells hold their condition exactly, with an exactly zero
/// standard error, degenerate bounds and a zeroed covariance row/column.
#[test]
fn constrained_cells_are_pinned_exactly() {
    let (fx, cs) = cases();
    for case in &cs {
        let name = case["name"].as_str().unwrap();
        let (res, cf) = fit_case(&fx, case);
        let conds = conditions_of(&case["conditions"]);
        let mut n = 0;
        for (h, row) in conds.iter().enumerate() {
            for (j, c) in row.iter().enumerate() {
                if let Some(v) = c {
                    n += 1;
                    assert!(cf.constrained[h][j], "{name}: ({h},{j}) flagged");
                    assert_eq!(
                        cf.point[(h, j)].to_bits(),
                        v.to_bits(),
                        "{name}: point ({h},{j})"
                    );
                    assert_eq!(cf.se[(h, j)], 0.0, "{name}: se ({h},{j})");
                    assert_eq!(
                        cf.lower[(h, j)].to_bits(),
                        v.to_bits(),
                        "{name}: lower ({h},{j})"
                    );
                    assert_eq!(
                        cf.upper[(h, j)].to_bits(),
                        v.to_bits(),
                        "{name}: upper ({h},{j})"
                    );
                    for l in 0..res.neqs {
                        assert_eq!(cf.cov[h][(j, l)], 0.0, "{name}: cov[{h}] row {j}");
                        assert_eq!(cf.cov[h][(l, j)], 0.0, "{name}: cov[{h}] col {j}");
                    }
                }
            }
        }
        assert_eq!(n, cf.n_constrained, "{name}: n_constrained");
        // Free cells: a positive se no larger than the unconditional one
        // (conditioning never adds variance), and bounds around the point.
        for h in 0..cf.steps {
            for j in 0..res.neqs {
                if !cf.constrained[h][j] {
                    assert!(cf.se[(h, j)] > 0.0, "{name}: free cell ({h},{j}) se");
                    assert!(
                        cf.se[(h, j)] <= cf.unconditional_se[(h, j)] * (1.0 + 1e-12),
                        "{name}: ({h},{j}) conditional se exceeds unconditional"
                    );
                    assert!(
                        cf.lower[(h, j)] < cf.point[(h, j)] && cf.point[(h, j)] < cf.upper[(h, j)]
                    );
                }
            }
        }
    }
}

/// Conditioning a cell at exactly its unconditional forecast is a no-op for
/// the path (`u* = 0`, `mahalanobis = 0`, p-value 1) but still shrinks the
/// variance of the correlated free cells.
#[test]
fn conditioning_at_the_unconditional_value_moves_nothing_but_the_variance() {
    let (fx, cs) = cases();
    let case = cs
        .iter()
        .find(|c| c["name"] == "var2c_at_unconditional")
        .unwrap();
    let (res, cf) = fit_case(&fx, case);
    for h in 0..cf.steps {
        for j in 0..res.neqs {
            assert!(cf.shocks[(h, j)].abs() < 1e-12, "shock ({h},{j})");
            assert!(cf.orth_shocks[(h, j)].abs() < 1e-12, "orth shock ({h},{j})");
            assert!(
                (cf.point[(h, j)] - cf.unconditional[(h, j)]).abs() < 1e-12,
                "point ({h},{j})"
            );
        }
    }
    // The gap is zero to roundoff (~1e-33), so the chi-squared tail
    // probability is one to roundoff, not bitwise.
    assert!(cf.mahalanobis.abs() < 1e-20);
    assert!((cf.mahalanobis_pvalue - 1.0).abs() < 1e-12);
    // Same-horizon cells of the other series are correlated through
    // Sigma_u, so their conditional se is strictly smaller.
    assert!(cf.se[(1, 0)] < cf.unconditional_se[(1, 0)]);
    assert!(cf.se[(1, 2)] < cf.unconditional_se[(1, 2)]);
    // The horizon before the constrained one is not untouched: the h = 1
    // innovations feed the h = 2 cell, so conditioning on it tightens h = 1
    // as well (the smoother runs backwards).
    for j in 0..res.neqs {
        assert!(cf.se[(0, j)] < cf.unconditional_se[(0, j)]);
    }
}

/// A short `conditions` list (fewer rows than `steps`) is padded with free
/// rows: it produces exactly the same object as the explicitly padded grid.
#[test]
fn short_condition_lists_are_padded_with_free_rows() {
    let (fx, cs) = cases();
    let case = cs.iter().find(|c| c["name"] == "var2c_short_rows").unwrap();
    let (res, cf) = fit_case(&fx, case);
    let mut padded = conditions_of(&case["conditions"]);
    assert!(padded.len() < cf.steps);
    while padded.len() < cf.steps {
        padded.push(vec![None; res.neqs]);
    }
    let cf2 = res
        .conditional_forecast(cf.steps, &padded, cf.alpha)
        .unwrap();
    for h in 0..cf.steps {
        for j in 0..res.neqs {
            assert_eq!(cf.point[(h, j)].to_bits(), cf2.point[(h, j)].to_bits());
            assert_eq!(cf.se[(h, j)].to_bits(), cf2.se[(h, j)].to_bits());
        }
    }
    // NaN spells "free" exactly like None.
    let mut nan_grid = padded.clone();
    for row in nan_grid.iter_mut() {
        for c in row.iter_mut() {
            if c.is_none() {
                *c = Some(f64::NAN);
            }
        }
    }
    let cf3 = res
        .conditional_forecast(cf.steps, &nan_grid, cf.alpha)
        .unwrap();
    assert_eq!(cf3.n_constrained, cf.n_constrained);
    for h in 0..cf.steps {
        for j in 0..res.neqs {
            assert_eq!(cf.point[(h, j)].to_bits(), cf3.point[(h, j)].to_bits());
        }
    }
}

/// The macro scenario (inflation held for four quarters, T-bill on hold for
/// two) is a six-cell constraint whose plausibility statistic is an
/// ordinary chi-squared draw under the fitted model.
#[test]
fn macro_scenario_reads_right() {
    let (fx, cs) = cases();
    let case = cs
        .iter()
        .find(|c| c["name"] == "macro_var2c_rates_on_hold")
        .unwrap();
    let (res, cf) = fit_case(&fx, case);
    assert_eq!(res.neqs, 3);
    assert_eq!(cf.n_constrained, 6);
    assert!(cf.mahalanobis > 0.0 && cf.mahalanobis_pvalue > 0.0 && cf.mahalanobis_pvalue < 1.0);
    // GDP growth is never constrained, yet its conditional se is tightened
    // at the conditioned horizons by the correlation with inflation and
    // the rate change.
    for h in 0..4 {
        assert!(!cf.constrained[h][0]);
        assert!(cf.se[(h, 0)] < cf.unconditional_se[(h, 0)]);
    }
}
