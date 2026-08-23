//! Candidate-level goldens for `auto_arima` against the statsmodels
//! fixture (`fixtures/auto_arima.json`).
//!
//! What is (and is not) pinned, honestly: the *selection loop* has no
//! third-party parity gate — R's `forecast::auto.arima` and `pmdarima`
//! disagree with each other on real series, so "parity" would pin an
//! implementation accident; the loop is graded by Monte-Carlo order
//! recovery instead (`tests/auto.rs`, and the Python MC study quoted in
//! the model card). What CAN be pinned is the **candidate level**: the
//! quantity the loop minimizes. For every fixture (series, order) pair,
//!
//! * the exact log-likelihood at statsmodels' recorded MLE parameters
//!   matches at **1e-8 relative** — same gate as `tests/golden.rs`;
//! * the AICc/AIC/BIC implied by that likelihood match at 1e-8 — this is
//!   the `n` (post-differencing) and `k` (`sigma2` counted) convention
//!   check, the place selection loops classically drift;
//! * the crate's own free fit **matches or beats** statsmodels'
//!   Nelder-Mead-polished optimum on the criterion the search minimizes
//!   (ARMA likelihoods are multimodal; equality gates on free fits are
//!   the pmdarima-vs-R trap, and the Nile golden documents a live
//!   statsmodels stall).

mod common;

use common::{as_vec, assert_rel_close, load_fixture};
use serde_json::Value;
use tsecon_arima::{ArimaSpec, SelectionIc};

/// Builds the crate spec for a fixture case.
fn spec_for(case: &Value) -> ArimaSpec {
    let order = as_vec(&case["order"]);
    let seasonal = as_vec(&case["seasonal_order"]);
    let trend = case["trend"].as_str().unwrap();
    let spec = ArimaSpec::new(order[0] as usize, order[1] as usize, order[2] as usize)
        .unwrap()
        .with_constant(trend == "c");
    spec.seasonal(
        seasonal[0] as usize,
        seasonal[1] as usize,
        seasonal[2] as usize,
        seasonal[3] as usize,
    )
    .unwrap()
}

/// The fixture parameter vector reordered into the crate's packed layout.
///
/// statsmodels orders SARIMAX params `[const?, ar, ma, sar, sma, sigma2]`
/// — the same layout this crate uses — so this is a straight copy; the
/// name check below asserts that assumption instead of trusting it.
fn params_for(case: &Value) -> Vec<f64> {
    as_vec(&case["params"])
}

fn series(fx: &Value, case_name: &str) -> Vec<f64> {
    let key = case_name.split("__").next().unwrap();
    as_vec(&fx["series"][key])
}

const CASES: [&str; 9] = [
    "ar1__100c",
    "ar1__001c",
    "ar1__202c",
    "arma11c__101c",
    "arma11c__100c",
    "arima111__111",
    "arima111__011",
    "sarima__100_100c",
    "sarima__100_000c",
];

/// Fixed-parameter pins: exact log-likelihood and the implied
/// AICc/AIC/BIC at statsmodels' recorded parameters, 1e-8 relative, for
/// every case. This is the value the selection loop compares between
/// candidates, pinned to an independent implementation on the same
/// series and orders.
#[test]
fn golden_candidate_loglik_and_ic_at_recorded_params() {
    let fx = load_fixture("auto_arima.json");
    for name in CASES {
        let case = &fx["cases"][name];
        let y = series(&fx, name);
        let spec = spec_for(case);
        let params = params_for(case);

        // The layout assumption is asserted, not trusted.
        let sm_names: Vec<&str> = case["param_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let our_names = spec.param_names();
        assert_eq!(
            our_names.len(),
            sm_names.len(),
            "{name}: parameter count mismatch"
        );
        for (ours, sm) in our_names.iter().zip(&sm_names) {
            // statsmodels calls the trend term "intercept"; every other
            // name matches the crate's statsmodels-style names.
            let sm_mapped = if *sm == "intercept" { "const" } else { sm };
            assert_eq!(ours, sm_mapped, "{name}: parameter order mismatch");
        }

        let ll = spec.loglike(&y, &params).unwrap();
        assert_rel_close(
            ll,
            case["loglike_fixed"].as_f64().unwrap(),
            1e-8,
            &format!("{name} loglike_fixed"),
        );

        // ICs through the crate's own results object at those params:
        // pins the (n, k) conventions the search depends on.
        let res = spec.at_params(&y, &params).unwrap();
        assert_eq!(
            res.nobs,
            case["nobs"].as_u64().unwrap() as usize,
            "{name} nobs"
        );
        assert_eq!(
            res.k_params,
            case["k_params"].as_u64().unwrap() as usize,
            "{name} k_params"
        );
        assert_rel_close(
            res.aic,
            case["aic_fixed"].as_f64().unwrap(),
            1e-8,
            &format!("{name} aic_fixed"),
        );
        assert_rel_close(
            res.bic,
            case["bic_fixed"].as_f64().unwrap(),
            1e-8,
            &format!("{name} bic_fixed"),
        );
        let aicc = SelectionIc::Aicc.evaluate(&res).unwrap();
        assert_rel_close(
            aicc,
            case["aicc_fixed"].as_f64().unwrap(),
            1e-8,
            &format!("{name} aicc_fixed"),
        );
    }
}

/// Free-fit match-or-beat: on a subset of cases (kept small — each
/// debug-mode MLE costs seconds) the crate's own fit must reach a
/// log-likelihood at least as good as statsmodels' polished optimum, and
/// hence an AICc at least as small (same k, same n). The subset spans
/// the search's regimes: pure AR, integrated, and seasonal.
#[test]
fn golden_candidate_free_fit_match_or_beat() {
    let fx = load_fixture("auto_arima.json");
    for name in ["ar1__100c", "arima111__111", "sarima__100_100c"] {
        let case = &fx["cases"][name];
        let y = series(&fx, name);
        let spec = spec_for(case);
        let res = spec.fit(&y).unwrap();

        let ll_ref = case["loglike_fit"].as_f64().unwrap();
        assert!(
            res.loglik >= ll_ref - 1e-5 * ll_ref.abs(),
            "{name}: fit loglik {} worse than the statsmodels floor {ll_ref}",
            res.loglik
        );
        let aicc = SelectionIc::Aicc.evaluate(&res).unwrap();
        let aicc_ref = case["aicc_fit"].as_f64().unwrap();
        assert!(
            aicc <= aicc_ref + 1e-5 * aicc_ref.abs(),
            "{name}: fit AICc {aicc} worse than the statsmodels floor {aicc_ref}"
        );
    }
}
