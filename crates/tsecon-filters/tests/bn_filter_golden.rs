//! Golden-value tests for the Kamber-Morley-Wong (2018) BN filter
//! against `fixtures/bn_filters.json` — **reference-run goldens**: the
//! stored cycle/delta/AR/SE values come from actual R runs of the
//! authors' replication code (bnfiltering.com lineage, packaged as
//! github.com/kletts/bnfilter@8af7924) at the KMW-2018 baseline options,
//! cross-checked at generation time against an independent NumPy
//! transcription at 1e-9 (see `fixtures/generate_bn_filters_fixtures.py`).
//!
//! Tolerance: 1e-8 (absolute for the cycle, which crosses zero;
//! relative for coefficients and scalars). Measured agreement is
//! recorded in the diagnostics model card.

use serde_json::Value;
use tsecon_filters::{bn_filter, BnDelta};

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

fn assert_close(actual: f64, expected: f64, tol: f64, ctx: &str) {
    let err = (actual - expected).abs() / expected.abs().max(1.0);
    assert!(
        err <= tol,
        "{ctx}: actual {actual}, expected {expected}, err {err:e} > {tol:e}"
    );
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

fn check_case(y: &[f64], case: &Value, ctx: &str) {
    let p = case["p"].as_u64().expect("p") as usize;
    let demean = case["demean"].as_bool().expect("demean");
    let expected_delta = case["delta"].as_f64().expect("delta");
    let delta = if case["delta_mode"].as_str() == Some("auto") {
        BnDelta::auto()
    } else {
        BnDelta::Fixed(expected_delta)
    };
    let r = bn_filter(y, p, delta, demean).expect("bn_filter succeeds");

    // Automatic selection must land on the same grid point as the
    // authors' R code (the grid is 0.0005-spaced, so 1e-12 is "same
    // point", not "close").
    assert_close(r.delta, expected_delta, 1e-12, &format!("{ctx} delta"));

    let cycle_err = max_err(&r.decomposition.cycle, &f64s(&case["cycle"]), ctx);
    let ar_err = max_err(&r.ar, &f64s(&case["ar"]), ctx);
    println!("{ctx}: cycle max err {cycle_err:e}, ar max err {ar_err:e}");
    assert!(cycle_err <= 1e-8, "{ctx} cycle: max err {cycle_err:e}");
    assert!(ar_err <= 1e-8, "{ctx} ar: max err {ar_err:e}");

    assert_close(
        r.cycle_se,
        case["cycle_se"].as_f64().expect("cycle_se"),
        1e-8,
        &format!("{ctx} cycle_se"),
    );
    assert_close(
        r.amplitude_to_noise,
        case["amp_to_noise"].as_f64().expect("amp_to_noise"),
        1e-8,
        &format!("{ctx} amp_to_noise"),
    );
    assert_close(
        r.drift,
        case["drift"].as_f64().expect("drift"),
        1e-12,
        &format!("{ctx} drift"),
    );

    // Alignment: one observation lost to differencing.
    assert_eq!(r.decomposition.alignment.lost_start, 1);
    assert_eq!(r.decomposition.alignment.lost_end, 0);
    assert_eq!(r.decomposition.alignment.input_len, y.len());
    assert_eq!(r.decomposition.cycle.len(), y.len() - 1);
}

#[test]
fn kmw_usgdp_p12_auto_sample_mean_reference_run() {
    let fx = fixture("bn_filters.json");
    check_case(&gdp(), &fx["kmw"]["usgdp_p12_auto_sm"], "usgdp_p12_auto_sm");
}

#[test]
fn kmw_usgdp_p12_fixed_delta_reference_run() {
    let fx = fixture("bn_filters.json");
    check_case(
        &gdp(),
        &fx["kmw"]["usgdp_p12_fixed025_sm"],
        "usgdp_p12_fixed025_sm",
    );
}

#[test]
fn kmw_sim_p12_auto_reference_run() {
    let fx = fixture("bn_filters.json");
    check_case(&sim(), &fx["kmw"]["sim_p12_auto_sm"], "sim_p12_auto_sm");
}

#[test]
fn kmw_sim_p8_fixed_no_drift_reference_run() {
    let fx = fixture("bn_filters.json");
    check_case(
        &sim(),
        &fx["kmw"]["sim_p8_fixed005_nd"],
        "sim_p8_fixed005_nd",
    );
}

#[test]
fn hamilton_hac_bse_match_statsmodels() {
    // The Hamilton regression's coefficient inference IS statsmodels
    // territory (plain OLS with HAC): pin bse and t-values for the
    // nonrobust and three HAC settings, including the h-overlap default
    // maxlags = h = 8.
    use tsecon_filters::{hamilton_filter, hamilton_filter_with_se, HamiltonSe};

    let fx = fixture("bn_filters.json");
    let block = &fx["hamilton_hac"];
    let y = gdp();
    let h = block["h"].as_u64().expect("h") as usize;
    let p = block["p"].as_u64().expect("p") as usize;

    let cases: [(&str, HamiltonSe, &Value); 4] = [
        ("nonrobust", HamiltonSe::NonRobust, &block["nonrobust"]),
        (
            "hac_h8_corr",
            HamiltonSe::Hac {
                maxlags: None, // resolves to the h-overlap default h = 8
                use_correction: true,
            },
            &block["hac_h8_corr"],
        ),
        (
            "hac_h8_nocorr",
            HamiltonSe::Hac {
                maxlags: Some(8),
                use_correction: false,
            },
            &block["hac_h8_nocorr"],
        ),
        (
            "hac_l4_corr",
            HamiltonSe::Hac {
                maxlags: Some(4),
                use_correction: true,
            },
            &block["hac_l4_corr"],
        ),
    ];

    let plain = hamilton_filter(&y, h, p).expect("hamilton_filter succeeds");
    for (name, se, expected) in cases {
        let (result, inf) = hamilton_filter_with_se(&y, h, p, se).expect("with_se succeeds");
        // The filter output is bit-identical to the plain call, whatever
        // inference was requested.
        assert_eq!(
            result.beta, plain.beta,
            "{name}: beta must be bit-identical"
        );
        assert_eq!(
            result.decomposition, plain.decomposition,
            "{name}: decomposition must be bit-identical"
        );
        // Tolerance 1e-6, measured ~2e-8 (bse) / ~5e-8 (tvalues): the
        // design is raw *levels* of a trending series, so X'X is
        // enormous-condition-number territory, and statsmodels (pinv)
        // vs tsecon-hac (refined Cholesky) agree to ~1e-8 here rather
        // than the engine's 1e-10 on its own calmer golden designs.
        let bse_err = max_err(&inf.bse, &f64s(&expected["bse"]), name);
        let t_err = max_err(&inf.tvalues, &f64s(&expected["tvalues"]), name);
        println!("{name}: bse max err {bse_err:e}, tvalues max err {t_err:e}");
        assert!(bse_err <= 1e-6, "{name} bse: max err {bse_err:e}");
        assert!(t_err <= 1e-6, "{name} tvalues: max err {t_err:e}");
        match se {
            HamiltonSe::NonRobust => assert_eq!(inf.maxlags, None),
            HamiltonSe::Hac { maxlags, .. } => {
                assert_eq!(inf.maxlags, Some(maxlags.unwrap_or(h)));
            }
        }
        // The HAC engine's point estimates (Cholesky normal equations +
        // refinement) agree with the filter's Householder QR solve well
        // within the golden tolerance: t = beta/bse. The lag columns are
        // highly collinear (levels of a trending series), so the two
        // solvers differ at the ~1e-9-relative level on the intercept —
        // asserted at 5e-8, an order under the 1e-8 fixture pin's scale.
        for j in 0..inf.bse.len() {
            let t_from_beta = plain.beta[j] / inf.bse[j];
            assert!(
                (t_from_beta - inf.tvalues[j]).abs() <= 5e-8 * inf.tvalues[j].abs().max(1.0),
                "{name}: engine params differ from filter beta at slot {j}: \
                 {t_from_beta} vs {}",
                inf.tvalues[j]
            );
        }
    }

    // And the point estimates themselves match the statsmodels beta.
    let beta_err = max_err(&plain.beta, &f64s(&block["beta"]), "beta");
    assert!(beta_err <= 1e-8, "beta: max err {beta_err:e}");
}
