//! Golden tests for ADL-MIDAS against `fixtures/adl_midas.json`.
//!
//! The fixture (statsmodels 0.15.0) fits OLS of `y_t` on
//! `[1, y_{t-1..t-P}, hf_lags[t, :]]` over the rows `t = P..n-1`, with the
//! y-lag columns built by `pandas.Series.shift`, under nonrobust and HAC
//! (Bartlett, `use_correction=True`) covariances — at the effective-rows
//! Newey-West default bandwidth and at `maxlags = 8`. Here the same design is
//! assembled the way the Python binding does (y-lag column `p` is
//! `y[P - p .. n - p]`) and passed to [`tsecon_midas::adl_midas`]; that pins
//! the alignment, the parameter order `[const, rho_1..rho_P, b_1..b_K]`, and
//! the standard errors at 1e-10.

use serde_json::Value;
use tsecon_hac::{newey_west_maxlags, Kernel};
use tsecon_midas::{adl_midas, SeType};

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/adl_midas.json",
        env!("CARGO_MANIFEST_DIR")
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

/// `(y_eff, y_lag_columns, hf_columns)` for ADL-MIDAS with `p` AR lags: the
/// binding's alignment.
fn design(y: &[f64], hf_rows: &[Vec<f64>], p: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = y.len();
    let k = hf_rows[0].len();
    let y_eff = y[p..].to_vec();
    let y_lags = (1..=p).map(|lag| y[p - lag..n - lag].to_vec()).collect();
    let hf = (0..k)
        .map(|j| hf_rows[p..].iter().map(|row| row[j]).collect())
        .collect();
    (y_eff, y_lags, hf)
}

fn assert_rel(actual: &[f64], expected: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let mut worst = 0.0f64;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let d = (a - e).abs() / e.abs().max(1.0);
        worst = worst.max(d);
        assert!(d <= tol, "{what}[{i}]: {a} vs {e} (rel {d:e})");
    }
    worst
}

#[test]
fn adl_midas_matches_statsmodels_ols_and_hac() {
    let fx = load();
    let mut worst = 0.0f64;
    for case in fx["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let p = case["ar_lags"].as_u64().unwrap() as usize;
        let y = f64s(&case["y"]);
        let hf_rows: Vec<Vec<f64>> = case["hf_lags"]
            .as_array()
            .unwrap()
            .iter()
            .map(f64s)
            .collect();
        let (y_eff, y_lags, hf) = design(&y, &hf_rows, p);
        let nobs = case["nobs"].as_u64().unwrap() as usize;
        assert_eq!(y_eff.len(), nobs);
        // The binding's default bandwidth is the Newey-West rule on the
        // EFFECTIVE rows n - P, which the fixture records.
        assert_eq!(
            newey_west_maxlags(nobs),
            case["default_maxlags"].as_u64().unwrap() as usize,
            "{name}: default maxlags"
        );
        for (label, fit) in case["fits"].as_object().unwrap() {
            let se = match fit["se_type"].as_str().unwrap() {
                "nonrobust" => SeType::NonRobust,
                _ => SeType::Hac {
                    kernel: Kernel::Bartlett,
                    bandwidth: fit["maxlags"].as_u64().unwrap() as f64,
                    use_correction: true,
                },
            };
            let r = adl_midas(&y_eff, &y_lags, &hf, se).expect("adl_midas fits");
            let what = format!("{name}/{label}");
            assert_eq!(r.nobs, nobs);
            worst = worst.max(assert_rel(
                &r.params,
                &f64s(&case["params"]),
                1e-10,
                &format!("{what} params"),
            ));
            worst = worst.max(assert_rel(
                &r.bse,
                &f64s(&fit["bse"]),
                1e-10,
                &format!("{what} bse"),
            ));
            worst = worst.max(assert_rel(
                &r.tvalues,
                &f64s(&fit["tvalues"]),
                1e-10,
                &format!("{what} tvalues"),
            ));
            worst = worst.max(assert_rel(
                &r.residuals,
                &f64s(&case["residuals"]),
                1e-10,
                &format!("{what} residuals"),
            ));
            worst = worst.max(assert_rel(
                &[r.rsquared],
                &[case["rsquared"].as_f64().unwrap()],
                1e-10,
                &format!("{what} rsquared"),
            ));
        }
    }
    eprintln!("adl_midas worst relative deviation {worst:e}");
    // Measured: 7.9e-15 worst relative deviation on the committed fixture
    // (3.2e-14 before the engine moved from Cholesky normal equations to QR).
    assert!(worst < 1e-11, "worst deviation {worst:e}");
}

#[test]
fn effective_rows_bandwidth_differs_from_the_raw_length_rule() {
    // The n = 275, P = 4 case is where a copy of umidas' rule (on the raw
    // length) would silently pick 5 lags instead of 4.
    let fx = load();
    let case = &fx["cases"][1];
    assert_eq!(case["name"], "n275_p4");
    let n = case["n"].as_u64().unwrap() as usize;
    let p = case["ar_lags"].as_u64().unwrap() as usize;
    assert_eq!(newey_west_maxlags(n - p), 4);
    assert_eq!(newey_west_maxlags(n), 5);
    assert_eq!(case["maxlags_rule_on_raw_n"].as_u64().unwrap(), 5);
}

#[test]
fn hf_column_one_is_the_last_high_frequency_observation() {
    // Within one case, column c + ratio equals column c lagged one
    // low-frequency period: the most-recent-first stacking the names hf1..hfK
    // describe (hf1 = high-frequency lag 0).
    let fx = load();
    for case in fx["cases"].as_array().unwrap() {
        let ratio = case["ratio"].as_u64().unwrap() as usize;
        let rows: Vec<Vec<f64>> = case["hf_lags"]
            .as_array()
            .unwrap()
            .iter()
            .map(f64s)
            .collect();
        let k = rows[0].len();
        for t in 1..rows.len() {
            for c in 0..k - ratio {
                assert_eq!(rows[t][c + ratio].to_bits(), rows[t - 1][c].to_bits());
            }
        }
    }
}

/// Max relative deviation `max_i |a_i - e_i| / |e_i|`.
fn rel_dev(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / e.abs())
        .fold(0.0f64, f64::max)
}

/// **Persistent level targets.** A log-GDP level near 9.5 and an index near
/// 100 make the lagged-y columns nearly collinear with the constant
/// (`cond(X)` 2.4e3 and 1.4e5). The fixture's `exact` block is the
/// least-squares solution of the very same float64 design in rational
/// arithmetic, with its exact nonrobust and HAC covariances.
///
/// Until 0.11.0 the engine solved the normal equations by Cholesky, which
/// squares the condition number, and multiplied the HAC sandwich out: on
/// these two cases the params came out 2.2e-8 / 4.9e-10 and the HAC bse
/// 2.4e-9 / 3.3e-9 relative from the exact answer, where statsmodels' params
/// are within 9.7e-13 / 7.8e-13. Measured now: params 1.2e-12 / 7.0e-13,
/// nonrobust bse 3.7e-14 / 7.2e-15, HAC bse 2.3e-13 / 2.0e-13 — the HAC
/// standard errors closer to the exact answer than statsmodels' own
/// (1.7e-10 / 7.8e-10, its sandwich multiplied out), so against statsmodels
/// they are gated at statsmodels' distance from the exact answer.
#[test]
fn level_targets_match_the_exact_solution_and_statsmodels() {
    let fx = load();
    for case in fx["level_cases"].as_array().expect("level_cases") {
        let name = case["name"].as_str().unwrap();
        let p = case["ar_lags"].as_u64().unwrap() as usize;
        let y = f64s(&case["y"]);
        let hf_rows: Vec<Vec<f64>> = case["hf_lags"]
            .as_array()
            .unwrap()
            .iter()
            .map(f64s)
            .collect();
        let (y_eff, y_lags, hf) = design(&y, &hf_rows, p);
        let exact = &case["exact"];
        let lags = case["default_maxlags"].as_u64().unwrap() as f64;
        let nr = adl_midas(&y_eff, &y_lags, &hf, SeType::NonRobust).unwrap();
        let hac = adl_midas(
            &y_eff,
            &y_lags,
            &hf,
            SeType::Hac {
                kernel: Kernel::Bartlett,
                bandwidth: lags,
                use_correction: true,
            },
        )
        .unwrap();
        let d_params_exact = rel_dev(&nr.params, &f64s(&exact["params"]));
        let d_params_sm = rel_dev(&nr.params, &f64s(&case["params"]));
        let d_nr_exact = rel_dev(&nr.bse, &f64s(&exact["bse_nonrobust"]));
        let d_nr_sm = rel_dev(&nr.bse, &f64s(&case["fits"]["nonrobust"]["bse"]));
        let d_hac_exact = rel_dev(&hac.bse, &f64s(&exact["bse_hac_default"]));
        let d_hac_sm = rel_dev(&hac.bse, &f64s(&case["fits"]["hac_default"]["bse"]));
        let resid_scale = f64s(&case["residuals"])
            .iter()
            .fold(0.0f64, |a, v| a.max(v.abs()));
        let d_resid = nr
            .residuals
            .iter()
            .zip(f64s(&exact["residuals"]))
            .map(|(a, e)| (a - e).abs() / resid_scale)
            .fold(0.0f64, f64::max);
        let sm_hac = case["sm_vs_exact"]["bse_hac_default"].as_f64().unwrap();
        eprintln!(
            "{name}: params vs exact {d_params_exact:e} (statsmodels {:e}), vs statsmodels \
             {d_params_sm:e}; nonrobust bse vs exact {d_nr_exact:e}, vs statsmodels \
             {d_nr_sm:e}; hac bse vs exact {d_hac_exact:e} (statsmodels {sm_hac:e}), vs \
             statsmodels {d_hac_sm:e}; residuals vs exact {d_resid:e}",
            case["sm_vs_exact"]["params"].as_f64().unwrap(),
        );
        assert!(
            d_params_exact <= 1e-11,
            "{name}: params vs exact {d_params_exact:e}"
        );
        assert!(
            d_params_sm <= 1e-10,
            "{name}: params vs statsmodels {d_params_sm:e}"
        );
        assert!(d_resid <= 1e-11, "{name}: residuals vs exact {d_resid:e}");
        assert!(
            d_nr_exact <= 1e-12,
            "{name}: nonrobust bse vs exact {d_nr_exact:e}"
        );
        assert!(
            d_nr_sm <= 1e-10,
            "{name}: nonrobust bse vs statsmodels {d_nr_sm:e}"
        );
        assert!(
            d_hac_exact <= 1e-11,
            "{name}: hac bse vs exact {d_hac_exact:e}"
        );
        assert!(
            d_hac_sm <= sm_hac + 1e-11,
            "{name}: hac bse vs statsmodels {d_hac_sm:e}, statsmodels vs exact {sm_hac:e}"
        );
        let r2 = case["rsquared"].as_f64().unwrap();
        assert!(
            (nr.rsquared - r2).abs() <= 1e-10 * r2.abs(),
            "{name}: rsquared"
        );
        assert_eq!(
            nr.params, hac.params,
            "{name}: one solve for both covariances"
        );
    }
}

/// **Finite inputs at extreme magnitudes.** Scaling `y` by `s` multiplies
/// the constant and the high-frequency coefficients (and their standard
/// errors) by `s` and leaves the AR coefficients, every t-value and R²
/// unchanged; scaling `hf_lags` by `s` divides the high-frequency
/// coefficients by `s`. Before the OLS engine equilibrated its inputs,
/// `y * 1e77` returned NaN HAC standard errors, `y * 1e155` and
/// `hf_lags * 1e154` NaN params, bse and R² as a success, and `y * 1e154` a
/// false "collinear columns" refusal.
#[test]
fn extreme_magnitudes_give_the_rescaled_fit() {
    let fx = load();
    let case = &fx["cases"][0];
    let p = case["ar_lags"].as_u64().unwrap() as usize;
    let y = f64s(&case["y"]);
    let hf_rows: Vec<Vec<f64>> = case["hf_lags"]
        .as_array()
        .unwrap()
        .iter()
        .map(f64s)
        .collect();
    let hac = SeType::Hac {
        kernel: Kernel::Bartlett,
        bandwidth: 4.0,
        use_correction: true,
    };
    let (y_eff, y_lags, hf) = design(&y, &hf_rows, p);
    for se in [hac, SeType::NonRobust] {
        let base = adl_midas(&y_eff, &y_lags, &hf, se).unwrap();
        for (sy, sx) in [
            (1e77, 1.0),
            (1e154, 1.0),
            (1e155, 1.0),
            (1.0, 1e154),
            (1e-300, 1.0),
        ] {
            let ys: Vec<f64> = y.iter().map(|v| v * sy).collect();
            let rows: Vec<Vec<f64>> = hf_rows
                .iter()
                .map(|r| r.iter().map(|v| v * sx).collect())
                .collect();
            let (ys_eff, ys_lags, hfs) = design(&ys, &rows, p);
            let r = adl_midas(&ys_eff, &ys_lags, &hfs, se)
                .unwrap_or_else(|e| panic!("y * {sy:e}, hf * {sx:e}, {se:?}: {e}"));
            let factor = |j: usize| {
                if j == 0 {
                    sy
                } else if j <= p {
                    1.0
                } else {
                    sy / sx
                }
            };
            for j in 0..r.params.len() {
                let (pw, bw) = (base.params[j] * factor(j), base.bse[j] * factor(j));
                assert!(
                    (r.params[j] - pw).abs() <= 1e-11 * pw.abs()
                        && (r.bse[j] - bw).abs() <= 1e-11 * bw
                        && (r.tvalues[j] - base.tvalues[j]).abs() <= 1e-11 * base.tvalues[j].abs(),
                    "y * {sy:e}, hf * {sx:e}, {se:?}: coefficient {j}: {} ({}) vs {pw} ({bw})",
                    r.params[j],
                    r.bse[j]
                );
            }
            assert!(
                (r.rsquared - base.rsquared).abs() <= 1e-12,
                "y * {sy:e}, hf * {sx:e}: R² {} vs {}",
                r.rsquared,
                base.rsquared
            );
        }
    }
}
