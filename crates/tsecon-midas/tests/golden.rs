//! Golden-value tests against `fixtures/midas.json`.
//!
//! The fixture (`_meta`: statsmodels 0.14.6, numpy 2.5.1) pins:
//!
//! * `weight_goldens` — exponential-Almon and Beta normalized weights, matched
//!   to 1e-10 through [`tsecon_midas::exp_almon_weights`] /
//!   [`tsecon_midas::beta_weights`].
//! * `umidas_ols` — OLS `params`, nonrobust `bse`, and centered `rsquared` of
//!   `y` on `[const, X_stacked]` (`K = 6` most-recent-first monthly lags),
//!   matched to 1e-8 through [`tsecon_midas::umidas`].

use serde_json::Value;
use tsecon_midas::{beta_weights, exp_almon_weights, umidas, SeType};

fn load() -> Value {
    let path = format!("{}/../../fixtures/midas.json", env!("CARGO_MANIFEST_DIR"));
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

fn cols(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn assert_close(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > atol {atol:e}"
    );
}

#[test]
fn exp_almon_weights_match_golden() {
    let fx = load();
    let golden = f64s(&fx["weight_goldens"]["exp_almon_0.1_-0.05_K6"]);
    let w = exp_almon_weights(0.1, -0.05, 6).expect("exp-Almon weights");
    assert_eq!(w.len(), golden.len());
    for (k, (a, e)) in w.iter().zip(golden.iter()).enumerate() {
        assert_close(*a, *e, 1e-10, &format!("exp_almon k={k}"));
    }
    let sum: f64 = w.iter().sum();
    assert_close(sum, 1.0, 1e-12, "exp_almon sum");
}

#[test]
fn beta_weights_match_goldens() {
    let fx = load();

    let g_23 = f64s(&fx["weight_goldens"]["beta_2_3_K10"]);
    let w_23 = beta_weights(2.0, 3.0, 10).expect("Beta(2,3) weights");
    assert_eq!(w_23.len(), g_23.len());
    for (k, (a, e)) in w_23.iter().zip(g_23.iter()).enumerate() {
        assert_close(*a, *e, 1e-10, &format!("beta_2_3 k={k}"));
    }

    let g_15 = f64s(&fx["weight_goldens"]["beta_1_5_K8"]);
    let w_15 = beta_weights(1.0, 5.0, 8).expect("Beta(1,5) weights");
    assert_eq!(w_15.len(), g_15.len());
    for (k, (a, e)) in w_15.iter().zip(g_15.iter()).enumerate() {
        assert_close(*a, *e, 1e-10, &format!("beta_1_5 k={k}"));
    }
}

#[test]
fn umidas_matches_statsmodels_ols() {
    let fx = load();
    let y = f64s(&fx["y"]);
    let x_stacked = cols(&fx["X_stacked"]);
    let k = fx["K"].as_u64().expect("K") as usize;
    assert_eq!(x_stacked.len(), k, "fixture K");

    let fit = umidas(&y, &x_stacked, SeType::NonRobust).expect("U-MIDAS fit");

    let g_params = f64s(&fx["umidas_ols"]["params"]);
    let g_bse = f64s(&fx["umidas_ols"]["bse"]);
    let g_r2 = fx["umidas_ols"]["rsquared"].as_f64().expect("rsquared");

    assert_eq!(fit.params.len(), k + 1, "intercept + K coefficients");
    for (i, (a, e)) in fit.params.iter().zip(g_params.iter()).enumerate() {
        assert_close(*a, *e, 1e-8, &format!("umidas param {i}"));
    }
    for (i, (a, e)) in fit.bse.iter().zip(g_bse.iter()).enumerate() {
        assert_close(*a, *e, 1e-8, &format!("umidas bse {i}"));
    }
    assert_close(fit.rsquared, g_r2, 1e-8, "umidas rsquared");
}

/// The fixture's stacked columns follow the most-recent-first convention this
/// crate's design builder implements: at frequency ratio 3, column `c + 3` is
/// column `c` lagged one whole low-frequency period, so
/// `X_stacked[c + 3][t] == X_stacked[c][t - 1]` for `t >= 1`. Locking this
/// cross-checks [`tsecon_midas::stack_high_freq_lags`]'s documented alignment
/// against the fixture generator.
#[test]
fn fixture_stacking_is_most_recent_first_ratio_3() {
    let fx = load();
    let x = cols(&fx["X_stacked"]);
    let ratio = 3usize;
    let n = x[0].len();
    for c in 0..(x.len() - ratio) {
        for t in 1..n {
            assert_close(
                x[c + ratio][t],
                x[c][t - 1],
                0.0,
                &format!("stacking shift col {c} t={t}"),
            );
        }
    }
}
