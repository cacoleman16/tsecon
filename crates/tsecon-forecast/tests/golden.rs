//! Golden-value tests against `fixtures/forecast.json` (generated with
//! statsmodels 0.14.6 / NumPy 1.26.4; see `fixtures/generate_fixtures.py`
//! `gen_forecast`).
//!
//! * `accuracy_small`: the measures are checked against hand values
//!   recomputed here from the documented definitions.
//! * `dm_test`: `dm_stat`, `hln_stat`, and the t(n-1) p-value are pinned
//!   to 1e-10 relative.
//! * `theta_realgdp_p4`: the 8-step statsmodels `ThetaModel` forecast
//!   (period=4, deseasonalize=True, use_test=False) is pinned to 1e-6
//!   relative. statsmodels stops its L-BFGS-B SES fit slightly short of
//!   the least-squares optimum (alpha 0.99989 vs 0.99999997), and the
//!   forecast is flat in alpha there; the exact optimizer used here lands
//!   within ~6e-7 relative of the fixture.

use serde_json::Value;
use tsecon_forecast::{
    dm_test, mae, mape, mase, mdae, me, mse, rmse, rmsse, smape, theta_forecast, DmLoss,
    ForecastComparison,
};

mod realgdp;
use realgdp::REALGDP;

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/forecast.json");
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

/// Relative comparison; falls back to absolute when the reference is 0.
fn assert_close(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    if expected == 0.0 {
        assert!(
            actual.abs() <= rtol,
            "{ctx}: actual {actual}, expected 0 (atol {rtol})"
        );
    } else {
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel <= rtol,
            "{ctx}: actual {actual}, expected {expected}, rel err {rel:e} > {rtol:e}"
        );
    }
}

#[test]
fn accuracy_small_matches_hand_values() {
    let fx = fixture();
    let block = &fx["accuracy_small"];
    let actual = f64s(&block["actual"]);
    let forecast = f64s(&block["forecast"]);
    let insample = f64s(&block["insample_for_mase"]);
    let n = actual.len() as f64;

    // Hand values from the documented definitions, e_t = y_t - yhat_t.
    let e: Vec<f64> = actual
        .iter()
        .zip(forecast.iter())
        .map(|(y, f)| y - f)
        .collect();
    let me_hand = e.iter().sum::<f64>() / n;
    let mse_hand = e.iter().map(|v| v * v).sum::<f64>() / n;
    let mae_hand = e.iter().map(|v| v.abs()).sum::<f64>() / n;
    let mape_hand = 100.0
        * e.iter()
            .zip(actual.iter())
            .map(|(ev, y)| (ev / y).abs())
            .sum::<f64>()
        / n;
    let smape_hand = 200.0
        * e.iter()
            .zip(actual.iter().zip(forecast.iter()))
            .map(|(ev, (y, f))| ev.abs() / (y.abs() + f.abs()))
            .sum::<f64>()
        / n;
    // MdAE: |e| = [1, 0.5, 1, 1.5, 0, 1], sorted [0, .5, 1, 1, 1, 1.5],
    // even count -> midpoint of the two central values = 1.
    let mdae_hand = 1.0;
    // MASE denominator: mean absolute first difference of the insample
    // (period 1), diffs [1.5,-0.5,1.5,-0.5,1.5,-0.5,1] -> mean |d| = 1.
    let d: Vec<f64> = insample.windows(2).map(|w| w[1] - w[0]).collect();
    let mase_denom = d.iter().map(|v| v.abs()).sum::<f64>() / d.len() as f64;
    assert_close(mase_denom, 1.0, 1e-15, "mase denominator");
    let mase_hand = mae_hand / mase_denom;
    let rmsse_denom = d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64;
    let rmsse_hand = (mse_hand / rmsse_denom).sqrt();

    const TOL: f64 = 1e-14;
    assert_close(me(&actual, &forecast).unwrap(), me_hand, TOL, "ME");
    assert_close(mse(&actual, &forecast).unwrap(), mse_hand, TOL, "MSE");
    assert_close(
        rmse(&actual, &forecast).unwrap(),
        mse_hand.sqrt(),
        TOL,
        "RMSE",
    );
    assert_close(mae(&actual, &forecast).unwrap(), mae_hand, TOL, "MAE");
    assert_close(mdae(&actual, &forecast).unwrap(), mdae_hand, TOL, "MdAE");
    assert_close(mape(&actual, &forecast).unwrap(), mape_hand, TOL, "MAPE");
    assert_close(smape(&actual, &forecast).unwrap(), smape_hand, TOL, "sMAPE");
    assert_close(
        mase(&actual, &forecast, &insample, 1).unwrap(),
        mase_hand,
        TOL,
        "MASE",
    );
    assert_close(
        rmsse(&actual, &forecast, &insample, 1).unwrap(),
        rmsse_hand,
        TOL,
        "RMSSE",
    );

    // Fully independent literals for the pinned cases.
    assert_close(me(&actual, &forecast).unwrap(), -1.0 / 6.0, TOL, "ME lit");
    assert_close(mse(&actual, &forecast).unwrap(), 5.5 / 6.0, TOL, "MSE lit");
    assert_close(mae(&actual, &forecast).unwrap(), 5.0 / 6.0, TOL, "MAE lit");
    assert_close(
        mase(&actual, &forecast, &insample, 1).unwrap(),
        5.0 / 6.0,
        TOL,
        "MASE lit",
    );
}

#[test]
fn dm_test_matches_fixture() {
    let fx = fixture();
    let block = &fx["dm_test"];
    let e1 = f64s(&block["e1"]);
    let e2 = f64s(&block["e2"]);
    let h = block["h"].as_u64().expect("h") as usize;
    assert_eq!(block["loss"].as_str().expect("loss"), "squared");

    let res = dm_test(&e1, &e2, h, DmLoss::Squared).unwrap();
    const TOL: f64 = 1e-10;
    assert_close(
        res.dm_stat,
        block["dm_stat"].as_f64().expect("dm_stat"),
        TOL,
        "dm_stat",
    );
    assert_close(
        res.hln_stat,
        block["hln_stat"].as_f64().expect("hln_stat"),
        TOL,
        "hln_stat",
    );
    assert_close(
        res.p_value,
        block["hln_pvalue_t_nminus1"].as_f64().expect("p"),
        TOL,
        "hln p-value",
    );
    assert_eq!(res.n, 120);
    assert_eq!(res.h, 3);
}

#[test]
fn dm_custom_loss_closure_matches_builtin() {
    let fx = fixture();
    let block = &fx["dm_test"];
    let e1 = f64s(&block["e1"]);
    let e2 = f64s(&block["e2"]);

    let builtin = dm_test(&e1, &e2, 3, DmLoss::Squared).unwrap();
    let custom = tsecon_forecast::dm_test_with_loss(&e1, &e2, 3, |e| e * e).unwrap();
    assert_eq!(builtin, custom);

    let builtin_abs = dm_test(&e1, &e2, 3, DmLoss::Absolute).unwrap();
    let custom_abs = tsecon_forecast::dm_test_with_loss(&e1, &e2, 3, |e| e.abs()).unwrap();
    assert_eq!(builtin_abs, custom_abs);
}

#[test]
fn theta_realgdp_matches_statsmodels() {
    let fx = fixture();
    let expected = f64s(&fx["theta_realgdp_p4"]["forecast_8"]);

    let res = theta_forecast(REALGDP, 4, 8).unwrap();
    assert_eq!(res.forecast.len(), 8);
    assert!(res.multiplicative, "realgdp > 0 => multiplicative");
    assert_eq!(res.seasonal.len(), 4);
    for (i, (&a, &e)) in res.forecast.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, 1e-6, &format!("theta forecast[{i}]"));
    }
    // The estimated pieces (statsmodels: b0 = 53.8768, alpha ~ 1 up to
    // optimizer stopping noise, seasonal factors ~ [0.99987, 1.00067,
    // 1.00046, 0.99899]).
    assert_close(res.b0, 53.87678562834662, 1e-10, "b0");
    assert!(res.alpha > 0.999, "alpha pinned near 1, got {}", res.alpha);
    let seas_expected = [
        0.9998702449064452,
        1.0006718119544864,
        1.0004638911751085,
        0.9989940519639596,
    ];
    for (i, (&a, &e)) in res.seasonal.iter().zip(seas_expected.iter()).enumerate() {
        assert_close(a, e, 1e-8, &format!("seasonal[{i}]"));
    }
}

#[test]
fn comparison_report_on_fixture_data() {
    let fx = fixture();
    let block = &fx["accuracy_small"];
    let actual = f64s(&block["actual"]);
    let forecast = f64s(&block["forecast"]);
    let insample = f64s(&block["insample_for_mase"]);
    // A second, worse forecast: shift by a constant.
    let worse: Vec<f64> = forecast.iter().map(|v| v + 2.0).collect();

    let cmp = ForecastComparison::new(
        &actual,
        &[("model", &forecast), ("model+2", &worse)],
        Some((&insample, 1)),
        1,
        0.05,
    )
    .unwrap();

    assert_eq!(cmp.measures.len(), 2);
    assert_eq!(cmp.dm_pairs.len(), 1);
    assert_eq!(cmp.best_rmse, "model");
    assert_close(
        cmp.measures[0].mase.unwrap(),
        5.0 / 6.0,
        1e-14,
        "comparison MASE",
    );
    assert!(cmp.measures[0].rmse < cmp.measures[1].rmse);
    // The shifted forecast has strictly larger squared loss every period,
    // so the mean loss differential d = loss(model) - loss(model+2) < 0.
    assert!(cmp.dm_pairs[0].dm.mean_loss_diff < 0.0);
    assert!(cmp.interpretation.contains("lowest RMSE"));
    assert!(!format!("{cmp}").is_empty());
}
