//! Golden tests for `tsecon-ets` against `fixtures/ets.json`
//! (`fixtures/generate_ets_fixtures.py`, which never calls tsecon).
//!
//! Blocks and their honest grades (see the generator's docstring):
//!
//! * `fixed[*].statsmodels` — **independent package**: for the twenty
//!   models without a multiplicative seasonal, statsmodels `ETSModel`
//!   at fixed parameters and known initial states — log-likelihood,
//!   fitted values, residuals, state paths, `forecast(h)` and `simulate`
//!   along given innovations, all at 1e-10; for the six class-1 models
//!   the exact forecast variance of `get_prediction` at 1e-10.
//! * `fixed[*].statsmodels_simulate_start` — **independent package, all
//!   thirty models**: statsmodels' `simulate` (written in the innovations
//!   form) from the initial state along given innovations, 1e-10.
//! * `fixed[*].transcription` — **documented-formula transcription, all
//!   thirty models**: the Hyndman et al. (2008) recursion in R's
//!   `etscalc.c` arithmetic, 1e-12 relative.
//! * `fixed[*].class1_variance_table61` — the Table 6.1 closed forms,
//!   transcribed and self-checked against the general `w' F^{j-1} g`
//!   formula in the generator, 1e-12 relative.
//! * `heuristic[*]` — **independent package**: `holtwinters.
//!   ExponentialSmoothing` heuristic initial states, 1e-10; the simple
//!   rule for short samples, 1e-12.
//! * `mle[*]` — **independent package, cross-optimizer**: the crate's
//!   maximum likelihood matches-or-beats statsmodels' L-BFGS-B optimum
//!   (slack 1e-5 relative on the log-likelihood, the `auto_arima` free-fit
//!   precedent; measured worst shortfall 2.7e-6 on a 17-parameter case) with two optimizers
//!   (`auto`: L-BFGS + Nelder-Mead + BFGS polish; and BFGS alone), and the
//!   parameters agree at
//!   the tolerance stated in `MLE_PARAM_TOL`; the criteria reproduce the
//!   documented formulas with the crate's parameter count.
//! * `candidates[*]` — the R `forecast::ets` candidate loop, exact.

use serde_json::Value;
use tsecon_ets::{
    candidate_specs, class1_forecast_variance, ets_at, ets_fit, forecast, forecast_from_state,
    heuristic_initial_states, loglik, simple_initial_states, simulate_paths, smooth,
    AutoEtsOptions, Component, ErrorType, EtsParams, EtsSpec, EtsStates, FitOptions, Ic,
    Initialization, IntervalMethod, Optimizer,
};

fn load() -> Value {
    let path = format!("{}/../../fixtures/ets.json", env!("CARGO_MANIFEST_DIR"));
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

fn opt_f64s(v: &Value) -> Option<Vec<f64>> {
    if v.is_null() {
        None
    } else {
        Some(f64s(v))
    }
}

fn num(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn opt_num(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn component(v: &Value) -> Component {
    match v.as_str() {
        None => Component::None,
        Some("add") => Component::Additive,
        Some("mul") => Component::Multiplicative,
        Some(other) => panic!("unknown component {other}"),
    }
}

fn spec_of(case: &Value) -> EtsSpec {
    let error = match case["error"].as_str().expect("error") {
        "add" => ErrorType::Additive,
        _ => ErrorType::Multiplicative,
    };
    EtsSpec::new(
        error,
        component(&case["trend"]),
        case["damped"].as_bool().unwrap_or(false),
        component(&case["seasonal"]),
        case["seasonal_periods"].as_u64().map(|m| m as usize),
    )
    .expect("fixture spec is valid")
}

fn params_of(case: &Value) -> EtsParams {
    EtsParams {
        alpha: num(&case["alpha"]),
        beta: opt_num(&case["beta"]),
        gamma: opt_num(&case["gamma"]),
        phi: opt_num(&case["phi"]),
    }
}

fn states_of(case: &Value, prefix: &str) -> EtsStates {
    EtsStates {
        level: num(&case[format!("{prefix}level")]),
        trend: opt_num(&case[format!("{prefix}trend")]),
        seasonal: opt_f64s(&case[format!("{prefix}seasonal")]),
    }
}

fn series(fx: &Value, name: &str) -> Vec<f64> {
    f64s(&fx["series"][name])
}

fn assert_close(what: &str, got: &[f64], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0_f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let scale = w.abs().max(1.0);
        let err = (g - w).abs() / scale;
        worst = worst.max(err);
        assert!(
            err <= rel,
            "{what}[{i}]: got {g}, want {w} (rel err {err:.3e} > {rel:.1e})"
        );
    }
    let _ = worst;
}

fn assert_scalar(what: &str, got: f64, want: f64, rel: f64) {
    let err = (got - want).abs() / want.abs().max(1.0);
    assert!(
        err <= rel,
        "{what}: got {got}, want {want} (rel err {err:.3e} > {rel:.1e})"
    );
}

const SM_TOL: f64 = 1e-10;
const TRANSCRIPTION_TOL: f64 = 1e-12;

// ---------------------------------------------------------------- fixed

#[test]
fn fixed_parameter_recursion_matches_transcription_for_all_thirty_models() {
    let fx = load();
    let cases = fx["fixed"].as_array().expect("fixed");
    let mut seen = std::collections::BTreeSet::new();
    let mut n_paths = 0;
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let spec = spec_of(case);
        seen.insert(spec.short_name());
        assert_eq!(
            spec.short_name(),
            case["short_name"].as_str().expect("short")
        );
        let y = series(&fx, case["series"].as_str().expect("series"));
        let params = params_of(case);
        let init = states_of(case, "initial_");
        let tr = &case["transcription"];
        let sm = smooth(&spec, &y, &params, &init).expect("smooth");
        assert_scalar(
            &format!("{name} loglik"),
            sm.loglik,
            num(&tr["loglik"]),
            TRANSCRIPTION_TOL,
        );
        assert_scalar(
            &format!("{name} sigma2"),
            sm.sigma2,
            num(&tr["sigma2"]),
            TRANSCRIPTION_TOL,
        );
        // Per-period paths are stored only where statsmodels is not exact
        // (the ten multiplicative-seasonal models); elsewhere the generator
        // asserted them equal to the statsmodels paths pinned below.
        if let Some(want) = opt_f64s(&tr["fitted"]) {
            assert!(spec.seasonal == Component::Multiplicative);
            assert_close(
                &format!("{name} fitted"),
                &sm.fitted,
                &want,
                TRANSCRIPTION_TOL,
            );
            assert_close(
                &format!("{name} resid"),
                &sm.resid,
                &f64s(&tr["resid"]),
                TRANSCRIPTION_TOL,
            );
            assert_close(
                &format!("{name} level"),
                &sm.level,
                &f64s(&tr["level"]),
                TRANSCRIPTION_TOL,
            );
            if let Some(want) = opt_f64s(&tr["trend"]) {
                assert_close(
                    &format!("{name} trend"),
                    sm.trend.as_ref().expect("trend"),
                    &want,
                    TRANSCRIPTION_TOL,
                );
            } else {
                assert!(sm.trend.is_none());
            }
            assert_close(
                &format!("{name} seasonal"),
                sm.seasonal.as_ref().expect("seasonal"),
                &f64s(&tr["seasonal"]),
                TRANSCRIPTION_TOL,
            );
            n_paths += 1;
        } else {
            assert!(spec.seasonal != Component::Multiplicative);
        }
        let final_want = states_of(tr, "final_");
        assert_scalar(
            &format!("{name} final level"),
            sm.final_state.level,
            final_want.level,
            TRANSCRIPTION_TOL,
        );
        assert_eq!(sm.final_state.trend.is_some(), final_want.trend.is_some());
        if let (Some(g), Some(w)) = (sm.final_state.trend, final_want.trend) {
            assert_scalar(&format!("{name} final trend"), g, w, TRANSCRIPTION_TOL);
        }
        if let (Some(g), Some(w)) = (&sm.final_state.seasonal, &final_want.seasonal) {
            assert_close(&format!("{name} final seasonal"), g, w, TRANSCRIPTION_TOL);
        }
        // loglik() is the same number without the paths.
        let ll = loglik(&spec, &y, &params, &init).expect("loglik");
        assert_eq!(ll, sm.loglik);
        // Point forecast and given-innovation paths from the final state and
        // from the initial state.
        let h = case["h"].as_u64().expect("h") as usize;
        let fc = forecast_from_state(&spec, &params, &sm.final_state, h).expect("forecast");
        assert_close(
            &format!("{name} forecast"),
            &fc,
            &f64s(&tr["forecast"]),
            TRANSCRIPTION_TOL,
        );
        let errors: Vec<Vec<f64>> = case["errors"]
            .as_array()
            .expect("errors")
            .iter()
            .map(f64s)
            .collect();
        let paths = simulate_paths(&spec, &params, &sm.final_state, &errors).expect("paths");
        let want: Vec<Vec<f64>> = tr["paths_from_end"]
            .as_array()
            .expect("paths")
            .iter()
            .map(f64s)
            .collect();
        for (i, (g, w)) in paths.iter().zip(&want).enumerate() {
            assert_close(
                &format!("{name} path_from_end[{i}]"),
                g,
                w,
                TRANSCRIPTION_TOL,
            );
        }
        let paths0 = simulate_paths(&spec, &params, &init, &errors).expect("paths from start");
        let want0: Vec<Vec<f64>> = tr["paths_from_start"]
            .as_array()
            .expect("paths")
            .iter()
            .map(f64s)
            .collect();
        for (i, (g, w)) in paths0.iter().zip(&want0).enumerate() {
            assert_close(
                &format!("{name} path_from_start[{i}]"),
                g,
                w,
                TRANSCRIPTION_TOL,
            );
        }
    }
    assert_eq!(
        seen.len(),
        30,
        "every member of the taxonomy is pinned: {seen:?}"
    );
    assert_eq!(
        n_paths, 13,
        "per-period transcription paths for every multiplicative-seasonal case"
    );
}

#[test]
fn fixed_parameter_recursion_matches_statsmodels_where_its_smoother_is_the_innovations_form() {
    let fx = load();
    let mut exact = std::collections::BTreeSet::new();
    let mut gap = std::collections::BTreeSet::new();
    for case in fx["fixed"].as_array().expect("fixed") {
        let name = case["name"].as_str().expect("name");
        let spec = spec_of(case);
        let y = series(&fx, case["series"].as_str().expect("series"));
        let params = params_of(case);
        let init = states_of(case, "initial_");
        let sm = smooth(&spec, &y, &params, &init).expect("smooth");
        let h = case["h"].as_u64().expect("h") as usize;
        let errors: Vec<Vec<f64>> = case["errors"]
            .as_array()
            .expect("errors")
            .iter()
            .map(f64s)
            .collect();
        // simulate(anchor="start") is exact for all thirty.
        let paths0 = simulate_paths(&spec, &params, &init, &errors).expect("paths from start");
        let want0: Vec<Vec<f64>> = case["statsmodels_simulate_start"]
            .as_array()
            .expect("sim")
            .iter()
            .map(f64s)
            .collect();
        for (i, (g, w)) in paths0.iter().zip(&want0).enumerate() {
            assert_close(
                &format!("{name} statsmodels simulate start[{i}]"),
                g,
                w,
                SM_TOL,
            );
        }
        let block = &case["statsmodels"];
        if block.is_null() {
            assert_eq!(spec.seasonal, Component::Multiplicative);
            // The gap is a recorded convention difference, not a golden;
            // it must be visible (non-zero) or the note would be stale.
            let g = &case["statsmodels_smoother_gap"];
            assert!(
                num(&g["max_abs_fitted_gap"]) > 1e-6,
                "{name}: recorded gap vanished"
            );
            gap.insert(spec.short_name());
            continue;
        }
        exact.insert(spec.short_name());
        assert_scalar(
            &format!("{name} statsmodels llf"),
            sm.loglik,
            num(&block["loglik"]),
            SM_TOL,
        );
        assert_scalar(
            &format!("{name} statsmodels mse"),
            sm.sigma2,
            num(&block["mse"]),
            SM_TOL,
        );
        assert_close(
            &format!("{name} statsmodels fitted"),
            &sm.fitted,
            &f64s(&block["fitted"]),
            SM_TOL,
        );
        assert_close(
            &format!("{name} statsmodels resid"),
            &sm.resid,
            &f64s(&block["resid"]),
            SM_TOL,
        );
        assert_close(
            &format!("{name} statsmodels level"),
            &sm.level,
            &f64s(&block["level"]),
            SM_TOL,
        );
        if let Some(want) = opt_f64s(&block["trend"]) {
            assert_close(
                &format!("{name} statsmodels slope"),
                sm.trend.as_ref().expect("trend"),
                &want,
                SM_TOL,
            );
        }
        if let Some(want) = opt_f64s(&block["seasonal"]) {
            assert_close(
                &format!("{name} statsmodels season"),
                sm.seasonal.as_ref().expect("seasonal"),
                &want,
                SM_TOL,
            );
        }
        let fc = forecast_from_state(&spec, &params, &sm.final_state, h).expect("forecast");
        assert_close(
            &format!("{name} statsmodels forecast"),
            &fc,
            &f64s(&block["forecast"]),
            SM_TOL,
        );
        let paths = simulate_paths(&spec, &params, &sm.final_state, &errors).expect("paths");
        let want: Vec<Vec<f64>> = block["simulate_end"]
            .as_array()
            .expect("sim")
            .iter()
            .map(f64s)
            .collect();
        for (i, (g, w)) in paths.iter().zip(&want).enumerate() {
            assert_close(
                &format!("{name} statsmodels simulate end[{i}]"),
                g,
                w,
                SM_TOL,
            );
        }
        if !block["forecast_variance"].is_null() {
            assert!(spec.is_class1());
            let v = class1_forecast_variance(&spec, &params, sm.sigma2, h).expect("variance");
            assert_close(
                &format!("{name} statsmodels forecast variance"),
                &v,
                &f64s(&block["forecast_variance"]),
                SM_TOL,
            );
            assert_close(
                &format!("{name} Table 6.1 variance"),
                &v,
                &f64s(&case["class1_variance_table61"]),
                TRANSCRIPTION_TOL,
            );
            let rel: Vec<f64> = v.iter().map(|x| x / sm.sigma2).collect();
            assert_close(
                &format!("{name} general c_j variance"),
                &rel,
                &f64s(&case["class1_relative_variance_general"]),
                TRANSCRIPTION_TOL,
            );
            // The forecast() surface reproduces the same numbers with an
            // exact Gaussian interval.
            let fit = ets_at(&spec, &y, &params, &init).expect("ets_at");
            let f = forecast(&fit, h, 0.95, 0, 0).expect("forecast");
            assert_eq!(f.method, IntervalMethod::Exact);
            assert_close(&format!("{name} forecast() mean"), &f.mean, &fc, 0.0);
            assert_close(&format!("{name} forecast() variance"), &f.variance, &v, 0.0);
            let z = 1.959963984540054_f64;
            for (j, vj) in v.iter().enumerate().take(h) {
                assert_scalar(
                    &format!("{name} lower[{j}]"),
                    f.lower[j],
                    f.mean[j] - z * vj.sqrt(),
                    1e-12,
                );
                assert_scalar(
                    &format!("{name} upper[{j}]"),
                    f.upper[j],
                    f.mean[j] + z * vj.sqrt(),
                    1e-12,
                );
            }
        } else {
            assert!(!spec.is_class1());
            assert!(matches!(
                class1_forecast_variance(&spec, &params, sm.sigma2, h),
                Err(tsecon_ets::EtsError::NotClass1 { .. })
            ));
        }
    }
    assert_eq!(
        exact.len(),
        20,
        "twenty models are statsmodels-exact: {exact:?}"
    );
    assert_eq!(
        gap.len(),
        10,
        "ten multiplicative-seasonal models carry the recorded gap: {gap:?}"
    );
}

// ------------------------------------------------------------ heuristic

#[test]
fn heuristic_and_simple_initial_states_match_statsmodels() {
    let fx = load();
    let mut n_heur = 0;
    let mut n_simple = 0;
    for case in fx["heuristic"].as_array().expect("heuristic") {
        let sname = case["series"].as_str().expect("series");
        let (y, tol) = if let Some(base) = sname.strip_suffix("[:12]") {
            (series(&fx, base)[..12].to_vec(), 1e-12)
        } else {
            (series(&fx, sname), SM_TOL)
        };
        let spec = EtsSpec::new(
            ErrorType::Additive,
            component(&case["trend"]),
            false,
            component(&case["seasonal"]),
            case["seasonal_periods"].as_u64().map(|m| m as usize),
        )
        .expect("spec");
        let got = match case["method"].as_str().expect("method") {
            "heuristic" => {
                n_heur += 1;
                heuristic_initial_states(&spec, &y).expect("heuristic")
            }
            _ => {
                n_simple += 1;
                simple_initial_states(&spec, &y).expect("simple")
            }
        };
        let what = format!("{sname} {} {}", spec.short_name(), case["method"]);
        assert_scalar(
            &format!("{what} level"),
            got.level,
            num(&case["initial_level"]),
            tol,
        );
        match (got.trend, opt_num(&case["initial_trend"])) {
            (Some(g), Some(w)) => assert_scalar(&format!("{what} trend"), g, w, tol),
            (None, None) => {}
            other => panic!("{what}: trend presence mismatch {other:?}"),
        }
        match (&got.seasonal, opt_f64s(&case["initial_seasonal"])) {
            (Some(g), Some(w)) => assert_close(&format!("{what} seasonal"), g, &w, tol),
            (None, None) => {}
            _ => panic!("{what}: seasonal presence mismatch"),
        }
    }
    assert!(
        n_heur >= 20 && n_simple >= 6,
        "{n_heur} heuristic, {n_simple} simple"
    );
}

// ------------------------------------------------------------------ mle

/// Parameter agreement between two optimizers on one likelihood: the
/// tolerance measured on the fixture (smoothing parameters, absolute).
const MLE_PARAM_TOL: f64 = 1e-3;
/// Log-likelihood slack for the match-or-beat gate (relative).
const MLE_LL_SLACK: f64 = 1e-5;

fn mle_fit_options(case: &Value, optimizer: Optimizer) -> FitOptions {
    FitOptions {
        initialization: match case["initialization"].as_str().expect("init") {
            "estimated" => Initialization::Estimated,
            _ => Initialization::Heuristic,
        },
        optimizer,
        max_iter: None,
    }
}

#[test]
fn maximum_likelihood_matches_or_beats_statsmodels_with_two_optimizers() {
    let fx = load();
    let mut worst_alpha = 0.0_f64;
    // The same gap over only the (case, optimizer) pairs where the two
    // optima coincide -- the subset the parameter gate applies to, and the
    // number quoted on the model card.
    let mut worst_alpha_gated = 0.0_f64;
    let mut n_gated = 0usize;
    let mut best_gain = 0.0_f64;
    let mut best_gain_case = String::new();
    let mut worst_ll_rel = f64::NEG_INFINITY;
    let mut n = 0;
    for case in fx["mle"].as_array().expect("mle") {
        n += 1;
        let spec = spec_of(case);
        let y = series(&fx, case["series"].as_str().expect("series"));
        let what = format!(
            "{} {} ({})",
            case["series"],
            spec.short_name(),
            case["initialization"]
        );
        let ll_sm = num(&case["loglik"]);
        for optimizer in [Optimizer::TwoStage, Optimizer::Bfgs] {
            let fit = ets_fit(&spec, &y, &mle_fit_options(case, optimizer)).expect("fit");
            let rel = (ll_sm - fit.loglik) / ll_sm.abs().max(1.0);
            worst_ll_rel = worst_ll_rel.max(rel);
            eprintln!(
                "mle {what} [{}]: loglik {:.6} (statsmodels {ll_sm:.6}, shortfall rel {rel:.2e}) alpha {:.5} vs {:.5} converged {} iters {} fevals {}",
                optimizer.name(),
                fit.loglik,
                fit.params.alpha,
                num(&case["alpha"]),
                fit.converged,
                fit.n_iterations,
                fit.n_fevals
            );
            assert!(
                fit.loglik >= ll_sm - MLE_LL_SLACK * ll_sm.abs().max(1.0),
                "{what} [{}]: loglik {} below statsmodels {ll_sm}",
                optimizer.name(),
                fit.loglik
            );
            let d_alpha = (fit.params.alpha - num(&case["alpha"])).abs();
            worst_alpha = worst_alpha.max(d_alpha);
            // Parameters agree at the cross-optimizer tolerance only when
            // the crate's optimum is not materially better (a better
            // optimum is a legitimately different point).
            if fit.loglik - ll_sm > 1e-3 * ll_sm.abs().max(1.0) && fit.loglik - ll_sm > best_gain {
                best_gain = fit.loglik - ll_sm;
                best_gain_case = format!("{what} [{}]", optimizer.name());
            }
            if fit.loglik - ll_sm <= 1e-3 * ll_sm.abs().max(1.0) {
                n_gated += 1;
                worst_alpha_gated = worst_alpha_gated.max(d_alpha);
                assert!(
                    d_alpha <= MLE_PARAM_TOL,
                    "{what} [{}]: alpha {} vs {}",
                    optimizer.name(),
                    fit.params.alpha,
                    case["alpha"]
                );
                if let (Some(b), Some(w)) = (fit.params.beta, opt_num(&case["beta"])) {
                    assert!((b - w).abs() <= MLE_PARAM_TOL, "{what}: beta {b} vs {w}");
                }
                if let (Some(g), Some(w)) = (fit.params.gamma, opt_num(&case["gamma"])) {
                    assert!((g - w).abs() <= MLE_PARAM_TOL, "{what}: gamma {g} vs {w}");
                }
                if let (Some(p), Some(w)) = (fit.params.phi, opt_num(&case["phi"])) {
                    assert!(
                        (p - w).abs() <= 2.0 * MLE_PARAM_TOL,
                        "{what}: phi {p} vs {w}"
                    );
                }
                // Initial states in the crate's normalisation, scaled by the level.
                let scale = fit.initial_state.level.abs().max(1.0);
                let dl =
                    (fit.initial_state.level - num(&case["converted_initial_level"])).abs() / scale;
                assert!(
                    dl <= 5e-2,
                    "{what}: initial level {} vs {}",
                    fit.initial_state.level,
                    case["converted_initial_level"]
                );
            }
            // The criteria follow the documented formulas with the crate's count.
            assert_eq!(
                fit.k_params,
                case["k_params_crate"].as_u64().expect("k") as usize,
                "{what}: k"
            );
            let nf = fit.nobs as f64;
            let k = fit.k_params as f64;
            assert_scalar(
                &format!("{what} aic"),
                fit.aic,
                -2.0 * fit.loglik + 2.0 * k,
                1e-12,
            );
            assert_scalar(
                &format!("{what} bic"),
                fit.bic,
                -2.0 * fit.loglik + k * nf.ln(),
                1e-12,
            );
            assert_scalar(
                &format!("{what} aicc"),
                fit.aicc,
                fit.aic + 2.0 * k * (k + 1.0) / (nf - k - 1.0),
                1e-12,
            );
            // statsmodels counts every seasonal state: its criteria differ by
            // exactly the count difference at equal likelihoods.
            let k_sm = case["k_params_statsmodels"].as_u64().expect("k_sm") as f64;
            let aic_sm_at_crate_ll = -2.0 * fit.loglik + 2.0 * k_sm;
            assert!(
                (aic_sm_at_crate_ll - num(&case["aic"])).abs()
                    <= 2.0 * MLE_LL_SLACK * ll_sm.abs().max(1.0)
                        + 2.0 * (fit.loglik - ll_sm).abs()
                        + 1e-9,
                "{what}: statsmodels aic {} vs {aic_sm_at_crate_ll}",
                case["aic"]
            );
        }
        // Refit determinism: the same call twice is bit-identical.
        let a = ets_fit(&spec, &y, &mle_fit_options(case, Optimizer::TwoStage)).expect("fit");
        let b = ets_fit(&spec, &y, &mle_fit_options(case, Optimizer::TwoStage)).expect("fit");
        assert_eq!(a, b, "{what}: refit is not deterministic");
    }
    assert!(n >= 15);
    eprintln!(
        "mle: {n} cases x 2 optimizers; worst loglik shortfall (rel) {worst_ll_rel:.2e}; \
         worst |alpha gap| {worst_alpha:.2e} over all pairs, {worst_alpha_gated:.2e} over the \
         {n_gated} pairs whose optima coincide; largest gain over statsmodels {best_gain:.4} \
         on {best_gain_case}"
    );
}

// ------------------------------------------------------------ candidates

#[test]
fn candidate_set_reproduces_r_forecast_ets_loop() {
    let fx = load();
    let mut n = 0;
    for case in fx["candidates"].as_array().expect("candidates") {
        n += 1;
        let opts = AutoEtsOptions {
            ic: Ic::Aicc,
            seasonal_periods: case["seasonal_periods"].as_u64().map(|m| m as usize),
            allow_multiplicative_trend: case["allow_multiplicative_trend"].as_bool().expect("amt"),
            restrict: case["restrict"].as_bool().expect("restrict"),
            damped: case["damped"].as_bool(),
            initialization: Initialization::Estimated,
            optimizer: Optimizer::TwoStage,
        };
        let got: Vec<String> =
            candidate_specs(&opts, case["data_positive"].as_bool().expect("pos"))
                .iter()
                .map(EtsSpec::short_name)
                .collect();
        let want: Vec<String> = case["candidates"]
            .as_array()
            .expect("list")
            .iter()
            .map(|v| v.as_str().expect("str").to_string())
            .collect();
        assert_eq!(got, want, "candidate set for {case}");
    }
    assert_eq!(n, 48);
}
