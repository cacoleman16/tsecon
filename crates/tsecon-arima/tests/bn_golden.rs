//! Beveridge-Nelson decomposition: golden re-pins against
//! `fixtures/bn_filters.json`, the exact finite-sample identities, the
//! ARIMA(0,1,1) textbook closed form, and the classic-vs-KMW amplitude
//! contrast.
//!
//! The `bn_arma` fixture block is a **documented-formula transcription
//! golden with a partial statsmodels pin**: statsmodels has no BN
//! decomposition (the fixture's absence canary pins that), so
//! trend/cycle come from an independent NumPy transcription of the
//! Morley (2002) companion-form computation — but the long-run
//! multiplier `psi(1)`, the number that *defines* the decomposition, is
//! pinned to the cumulative sum of statsmodels' `arma_impulse_response`
//! (a genuine third-party leg, stored alongside).

use serde_json::Value;
use tsecon_arima::{bn_decomposition, bn_from_arma, ArimaError};

fn fixture(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
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

fn max_err(actual: &[f64], expected: &[f64], ctx: &str) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (a - e).abs() / e.abs().max(1.0))
        .fold(0.0_f64, f64::max)
}

fn gdp() -> Vec<f64> {
    f64s(&fixture("filters.json")["y_100_log_realgdp"])
}

fn sim() -> Vec<f64> {
    f64s(&fixture("bn_filters.json")["sim_series"])
}

fn check_fixed_case(y: &[f64], case: &Value, ctx: &str) {
    let ar = f64s(&case["ar"]);
    let ma = f64s(&case["ma"]);
    let drift = case["drift"].as_f64().expect("drift");
    let d = bn_from_arma(y, drift, &ar, &ma).expect("bn_from_arma succeeds");

    let trend_err = max_err(&d.trend, &f64s(&case["trend"]), ctx);
    let cycle_err = max_err(&d.cycle, &f64s(&case["cycle"]), ctx);
    let eps_err = max_err(&d.innovations, &f64s(&case["innovations"]), ctx);
    println!("{ctx}: trend {trend_err:e}, cycle {cycle_err:e}, eps {eps_err:e}");
    assert!(trend_err <= 1e-8, "{ctx} trend: {trend_err:e}");
    assert!(cycle_err <= 1e-8, "{ctx} cycle: {cycle_err:e}");
    assert!(eps_err <= 1e-8, "{ctx} innovations: {eps_err:e}");

    // The closed form psi(1) = theta(1)/phi(1) — against the stored value
    // AND the statsmodels cumulative-impulse-response pin.
    let psi1 = case["long_run_multiplier"].as_f64().expect("psi1");
    let sm_cum = case["long_run_multiplier_sm_cum_irf"]
        .as_f64()
        .expect("sm irf");
    assert!(
        (d.long_run_multiplier - psi1).abs() <= 1e-10 * psi1.abs().max(1.0),
        "{ctx} psi1: {} vs {psi1}",
        d.long_run_multiplier
    );
    assert!(
        (d.long_run_multiplier - sm_cum).abs() <= 1e-7 * sm_cum.abs().max(1.0),
        "{ctx} psi1 vs statsmodels cum IRF: {} vs {sm_cum}",
        d.long_run_multiplier
    );

    // Exact identities, on the crate's own output.
    assert_identities(y, &d, ctx);
}

fn assert_identities(y: &[f64], d: &tsecon_arima::BnDecomposition, ctx: &str) {
    assert_eq!(d.lost_start, 1);
    assert_eq!(d.input_len, y.len());
    // (1) trend + cycle recovers y up to at most one final rounding
    //     (trend is stored as y - cycle, so re-adding can re-round).
    for (i, (t, c)) in d.trend.iter().zip(d.cycle.iter()).enumerate() {
        assert!(
            (t + c - y[i + 1]).abs() <= 1e-15 * y[i + 1].abs(),
            "{ctx}: reconstruction at {i}: {} vs {}",
            t + c,
            y[i + 1]
        );
    }
    // (2) the BN trend is a random walk with drift:
    //     Delta tau_t = mu + psi(1) eps_t, observation by observation.
    for i in 1..d.trend.len() {
        let dt = d.trend[i] - d.trend[i - 1];
        let rhs = d.drift + d.long_run_multiplier * d.innovations[i];
        assert!(
            (dt - rhs).abs() <= 1e-9 * rhs.abs().max(1.0),
            "{ctx}: trend increment at {i}: {dt} vs {rhs}"
        );
    }
}

#[test]
fn bn_arma_gdp_arima212_golden() {
    let fx = fixture("bn_filters.json");
    check_fixed_case(&gdp(), &fx["bn_arma"]["gdp_arima212"], "gdp_arima212");
}

#[test]
fn bn_arma_sim_arma11_fixed_golden() {
    let fx = fixture("bn_filters.json");
    check_fixed_case(
        &sim(),
        &fx["bn_arma"]["sim_arma11_fixed"],
        "sim_arma11_fixed",
    );
}

#[test]
fn bn_arma_sim_ar2_fixed_golden() {
    let fx = fixture("bn_filters.json");
    check_fixed_case(&sim(), &fx["bn_arma"]["sim_ar2_fixed"], "sim_ar2_fixed");
}

#[test]
fn ima11_matches_the_textbook_closed_form_exactly() {
    // For ARIMA(0,1,1): cycle_t = -theta eps_t, trend_t = y_t + theta eps_t,
    // psi(1) = 1 + theta — the classic Beveridge-Nelson worked example.
    let y = gdp();
    let theta = 0.4;
    let drift = 0.8;
    let d = bn_from_arma(&y, drift, &[], &[theta]).expect("bn_from_arma succeeds");
    assert_eq!(d.long_run_multiplier, 1.0 + theta);
    for (i, (&c, &e)) in d.cycle.iter().zip(d.innovations.iter()).enumerate() {
        let expected = -theta * e;
        assert!(
            (c - expected).abs() <= 1e-14 * expected.abs().max(1.0),
            "cycle at {i}: {c} vs {expected}"
        );
    }
}

#[test]
fn random_walk_with_drift_has_zero_cycle() {
    // ARIMA(0,1,0): the series is its own trend.
    let y = gdp();
    let d = bn_from_arma(&y, 0.8, &[], &[]).expect("bn_from_arma succeeds");
    assert_eq!(d.long_run_multiplier, 1.0);
    assert!(d.cycle.iter().all(|&c| c == 0.0));
    assert_eq!(d.trend, y[1..].to_vec());
}

#[test]
fn library_fit_path_reproduces_the_mnz_spec_on_gdp() {
    // bn_decomposition fits ARIMA(2,1,2)+c by this crate's own exact MLE.
    // The fixture records statsmodels' MLE of the same spec; two
    // optimizers' stopping points differ, so the pin is loose (the tight
    // pins live in the fixed-coefficient goldens above) — but psi(1) and
    // the drift must land near the statsmodels optimum, and the exact
    // identities must hold at whatever the fit returned.
    let y = gdp();
    let fx = fixture("bn_filters.json");
    let case = &fx["bn_arma"]["gdp_arima212"];
    let d = bn_decomposition(&y, 2, 2).expect("bn_decomposition succeeds");
    assert_identities(&y, &d, "fit path");
    let psi1 = case["long_run_multiplier"].as_f64().expect("psi1");
    let drift = case["drift"].as_f64().expect("drift");
    println!(
        "fit path: psi1 {} vs statsmodels {psi1}; drift {} vs {drift}",
        d.long_run_multiplier, d.drift
    );
    assert!(
        (d.long_run_multiplier - psi1).abs() < 0.05,
        "psi1 {} vs statsmodels {psi1}",
        d.long_run_multiplier
    );
    assert!(
        (d.drift - drift).abs() < 0.02,
        "drift {} vs statsmodels {drift}",
        d.drift
    );
    let results = d.results.as_ref().expect("fit path carries the results");
    assert_eq!(results.spec.p(), 2);
    assert_eq!(results.spec.d(), 1);
    assert_eq!(results.spec.q(), 2);
    assert!(results.spec.include_constant());
}

#[test]
fn classic_cycle_is_small_where_the_kmw_filter_finds_a_large_gap() {
    // The reason bn_filter exists (KMW 2018): on a drifting series the
    // freely estimated ARMA attributes nearly everything to the trend
    // and the classic BN cycle is tiny; pinning delta recovers a large
    // amplitude gap. Measured on the fixture's simulated series.
    // AR(2) growth: the ARMA(2,2) MLE on this series piles up on the MA
    // unit circle and is (correctly) refused — see the error-surface
    // test below — so the freely-estimated comparison uses the pure AR.
    let y = sim();
    let classic = bn_decomposition(&y, 2, 0).expect("classic BN succeeds");
    let kmw = tsecon_filters::bn_filter(&y, 12, tsecon_filters::BnDelta::auto(), true)
        .expect("bn_filter succeeds");
    let var = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64
    };
    let vc = var(&classic.cycle);
    let vk = var(&kmw.decomposition.cycle);
    println!(
        "classic cycle var {vc:.4}, KMW cycle var {vk:.4}, ratio {:.1}",
        vk / vc
    );
    assert!(
        vk > 3.0 * vc,
        "expected the KMW gap to dwarf the classic cycle: classic {vc}, kmw {vk}"
    );
}

#[test]
fn fit_path_refuses_an_ma_boundary_fit_with_a_teaching_error() {
    // On the simulated series the ARMA(2,2) MLE piles up on the MA unit
    // circle (near-cancelling roots); bn_decomposition must refuse with
    // an error that names the cure (lower q), not silently decompose
    // with a divergence-prone innovation recursion.
    let y = sim();
    match bn_decomposition(&y, 2, 2) {
        Err(ArimaError::InvalidArgument { what }) => {
            assert!(what.contains("MA"), "unexpected message: {what}");
            assert!(what.contains("lower q"), "unexpected message: {what}");
        }
        Ok(_) => panic!("expected the MA-boundary refusal"),
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn rejects_nonstationary_ar_and_noninvertible_ma() {
    let y = gdp();
    assert!(matches!(
        bn_from_arma(&y, 0.0, &[1.05], &[]),
        Err(ArimaError::InvalidArgument { .. })
    ));
    assert!(matches!(
        bn_from_arma(&y, 0.0, &[0.5, 0.5], &[]), // phi(1) = 0: unit root
        Err(ArimaError::InvalidArgument { .. })
    ));
    assert!(matches!(
        bn_from_arma(&y, 0.0, &[], &[-1.2]),
        Err(ArimaError::InvalidArgument { .. })
    ));
}

#[test]
fn rejects_short_and_non_finite_input() {
    assert!(matches!(
        bn_from_arma(&[1.0], 0.0, &[0.3], &[]),
        Err(ArimaError::InsufficientObservations { .. })
    ));
    let mut y = gdp();
    y[10] = f64::INFINITY;
    assert!(matches!(
        bn_from_arma(&y, 0.0, &[0.3], &[]),
        Err(ArimaError::NonFinite { at: Some(10), .. })
    ));
    assert!(matches!(
        bn_from_arma(&gdp(), f64::NAN, &[0.3], &[]),
        Err(ArimaError::NonFinite { .. })
    ));
    assert!(matches!(
        bn_from_arma(&gdp(), 0.0, &[f64::NAN], &[]),
        Err(ArimaError::NonFinite { .. })
    ));
}
