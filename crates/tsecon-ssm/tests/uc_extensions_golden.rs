//! Golden-value tests for the 0.11.0 follow-ups of
//! `unobserved_components` against `fixtures/uc_extensions.json`
//! (generator: `fixtures/generate_uc_extensions_fixtures.py`, which
//! states the references and the honest grade):
//!
//! * **`autoregressive = p`** — statsmodels' own AR(p) component
//!   (row-companion form, always stationary-initialized), at fixed
//!   parameters on Nile, inflation, GDP, GDP growth (every state
//!   stationary) and simulated specifications, and three MLE cases;
//! * **`cycle_init = "stationary"`** — the damped stochastic cycle from
//!   its unconditional (Lyapunov) distribution. statsmodels has no such
//!   keyword; the reference is statsmodels' own block `Initialization`
//!   ("diffuse" everywhere except the cycle pair, "stationary") through
//!   its own exact-diffuse filter and smoother, plus the closed form
//!   `P_* = sigma2_cycle / (1 - rho^2) I_2` (pinned in the crate's unit
//!   test, since the fit does not expose `P_1`).
//!
//! Fixed-parameter quantities are pinned at [`TOL_FIXED`] with the
//! fixture's `smoother_spread` as the smoothed-variance tolerance inside
//! the diffuse period, exactly as in `uc_golden.rs`; the MLE reaches the
//! better of statsmodels' L-BFGS and a SciPy re-optimization of the same
//! criterion within [`TOL_MLE_LL`], parameters within [`TOL_MLE_PARAM`],
//! the pile-up flags exactly and the conditional standard errors within
//! [`TOL_SE`].

mod common;

use common::{as_f64_vec, assert_rel_close, load_fixture};
use serde_json::Value;
use tsecon_ssm::{
    unobserved_components, CycleInit, FreqSeasonalSpec, TrendSpec, UcComponent, UcFit, UcOptions,
    UcSpec,
};

/// Fixed-parameter parity tier (relative, absolute floor at 1).
const TOL_FIXED: f64 = 1e-8;
/// Rows past the end of the diffuse period on which the exact-diffuse
/// smoother's conditioning can still be visible (as in `uc_golden.rs`).
const DIFFUSE_SMOOTH_ROWS: usize = 2;
/// The Rust optimum may not fall below the better reference optimum by
/// more than this (absolute log-likelihood units).
const TOL_MLE_LL: f64 = 1e-5;
/// Parameter agreement at a shared interior optimum.
const TOL_MLE_PARAM: f64 = 2e-3;
/// Conditional standard errors vs the fixture's `se_conditional`.
const TOL_SE: f64 = 2e-2;

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .expect("array of rows")
        .iter()
        .map(as_f64_vec)
        .collect()
}

fn columns(v: &Value) -> Vec<Vec<f64>> {
    let r = rows(v);
    let k = r[0].len();
    (0..k)
        .map(|j| r.iter().map(|row| row[j]).collect())
        .collect()
}

fn names(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|s| s.as_str().expect("string").to_string())
        .collect()
}

fn series(fx: &Value, case: &Value) -> Vec<f64> {
    let data = case["data"].as_str().expect("data");
    let key = match data {
        "sim" => "sim_y",
        other => other,
    };
    as_f64_vec(&fx["data"][key])
}

fn spec_from_json(fx: &Value, spec: &Value) -> UcSpec {
    let mut s = UcSpec {
        trend: TrendSpec::parse(spec["level"].as_str().expect("level")).expect("valid level"),
        ..UcSpec::default()
    };
    if let Some(p) = spec.get("seasonal") {
        s.seasonal = Some(p.as_u64().expect("period") as usize);
    }
    if let Some(b) = spec.get("stochastic_seasonal") {
        s.stochastic_seasonal = b.as_bool().expect("bool");
    }
    if let Some(fs) = spec.get("freq_seasonal") {
        for f in fs.as_array().expect("array") {
            let mut fspec = FreqSeasonalSpec::new(f["period"].as_f64().expect("period"));
            if let Some(h) = f.get("harmonics") {
                fspec.harmonics = h.as_u64().expect("harmonics") as usize;
            }
            s.freq_seasonal.push(fspec);
        }
    }
    let flag = |k: &str| spec.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    s.cycle = flag("cycle");
    s.damped_cycle = flag("damped_cycle");
    s.stochastic_cycle = flag("stochastic_cycle");
    if let Some(b) = spec.get("cycle_period_bounds") {
        let v = as_f64_vec(b);
        s.cycle_period_bounds = (v[0], v[1]);
    }
    if let Some(p) = spec.get("autoregressive") {
        s.autoregressive = Some(p.as_u64().expect("order") as usize);
    }
    if let Some(c) = spec.get("cycle_init") {
        s.cycle_init = CycleInit::parse(c.as_str().expect("cycle_init")).expect("valid");
    }
    if let Some(x) = spec.get("exog") {
        s.exog = columns(&fx["data"][x.as_str().expect("exog key")]);
    }
    s
}

fn forecast_exog(fx: &Value, case: &Value) -> Vec<Vec<f64>> {
    match case["spec"].get("exog").and_then(|v| v.as_str()) {
        Some(_) => {
            let h = case["forecast"].as_array().map_or(0, |a| a.len());
            columns(&fx["data"]["sim_x_forecast"])
                .into_iter()
                .map(|c| c[..h].to_vec())
                .collect()
        }
        None => Vec::new(),
    }
}

fn assert_series(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (t, (a, e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{what}[{t}]: expected NaN, got {a}");
        } else {
            assert_rel_close(*a, *e, tol, &format!("{what}[{t}]"));
        }
    }
}

fn assert_paths(actual: &[Vec<f64>], expected: &Value, tol: f64, what: &str) {
    let exp = rows(expected);
    assert_eq!(actual.len(), exp.len(), "{what}: periods");
    for (t, (a, e)) in actual.iter().zip(&exp).enumerate() {
        assert_series(a, e, tol, &format!("{what}[{t}]"));
    }
}

/// Smoothed variances: [`TOL_FIXED`] except inside the diffuse period,
/// where the tolerance is the reference's own smoother spread.
fn assert_smoothed_var(
    actual: &[f64],
    expected: &[f64],
    spread: f64,
    d: usize,
    t0: usize,
    what: &str,
) {
    let relaxed = spread.max(TOL_FIXED);
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        let t = t0 + k;
        let tol = if t <= d + DIFFUSE_SMOOTH_ROWS {
            relaxed
        } else {
            TOL_FIXED
        };
        assert_series(&[*a], &[*e], tol, &format!("{what}[{t}]"));
    }
}

fn assert_component(c: &UcComponent, block: &Value, spread: f64, d: usize, what: &str) {
    assert_series(
        &c.filtered,
        &as_f64_vec(&block["filtered"]),
        TOL_FIXED,
        &format!("{what}.filtered"),
    );
    assert_series(
        &c.filtered_var,
        &as_f64_vec(&block["filtered_var"]),
        TOL_FIXED,
        &format!("{what}.filtered_var"),
    );
    assert_series(
        &c.smoothed,
        &as_f64_vec(&block["smoothed"]),
        TOL_FIXED,
        &format!("{what}.smoothed"),
    );
    let exp = as_f64_vec(&block["smoothed_var"]);
    assert_eq!(
        c.smoothed_var.len(),
        exp.len(),
        "{what}.smoothed_var: length"
    );
    assert_smoothed_var(
        &c.smoothed_var,
        &exp,
        spread,
        d,
        0,
        &format!("{what}.smoothed_var"),
    );
}

/// Every fixed-parameter quantity of a block against the fit.
fn check_fixed_block(fit: &UcFit, block: &Value, what: &str) {
    let int = |k: &str| block[k].as_u64().unwrap_or_else(|| panic!("{what}: {k}")) as usize;
    assert_eq!(fit.k_states, int("k_states"), "{what}: k_states");
    assert_eq!(
        fit.nobs_diffuse,
        int("nobs_diffuse"),
        "{what}: nobs_diffuse"
    );
    assert_eq!(fit.k_diffuse, int("k_diffuse"), "{what}: k_diffuse");
    // statsmodels' df_model is k_params + k_diffuse_states.
    assert_eq!(
        fit.k_params + fit.k_diffuse,
        int("df_model"),
        "{what}: df_model"
    );
    assert_eq!(
        fit.param_names,
        names(&block["param_names"]),
        "{what}: param_names"
    );
    for key in ["loglike", "aic", "bic"] {
        let got = match key {
            "loglike" => fit.loglik,
            "aic" => fit.aic,
            _ => fit.bic,
        };
        assert_rel_close(
            got,
            block[key].as_f64().unwrap(),
            TOL_FIXED,
            &format!("{what}: {key}"),
        );
    }
    let spread = block
        .get("smoother_spread")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    if let Some(v) = block.get("filtered_state") {
        assert_paths(
            &fit.filtered_state,
            v,
            TOL_FIXED,
            &format!("{what}: filtered_state"),
        );
        assert_paths(
            &fit.filtered_state_var,
            &block["filtered_state_var"],
            TOL_FIXED,
            &format!("{what}: filtered_state_var"),
        );
        assert_paths(
            &fit.smoothed_state,
            &block["smoothed_state"],
            TOL_FIXED,
            &format!("{what}: smoothed_state"),
        );
        let exp = rows(&block["smoothed_state_var"]);
        assert_eq!(fit.smoothed_state_var.len(), exp.len());
        for (t, (a, e)) in fit.smoothed_state_var.iter().zip(&exp).enumerate() {
            assert_smoothed_var(
                a,
                e,
                spread,
                fit.nobs_diffuse,
                t,
                &format!("{what}: smoothed_state_var[{t}]"),
            );
        }
    }
    assert_series(
        &fit.fitted,
        &as_f64_vec(&block["fitted"]),
        TOL_FIXED,
        &format!("{what}: fitted"),
    );
    assert_series(
        &fit.resid,
        &as_f64_vec(&block["resid"]),
        TOL_FIXED,
        &format!("{what}: resid"),
    );
    // statsmodels writes 0.0 inside the diffuse period and at missing
    // periods; this crate NaN (no finite prediction variance / no error).
    let sr = as_f64_vec(&block["std_resid"]);
    let ref_resid = as_f64_vec(&block["resid"]);
    for (t, (a, e)) in fit.std_resid.iter().zip(&sr).enumerate() {
        if t < fit.nobs_diffuse || e.is_nan() || ref_resid[t].is_nan() {
            assert!(a.is_nan(), "{what}: std_resid[{t}] must be NaN");
        } else {
            assert_rel_close(*a, *e, TOL_FIXED, &format!("{what}: std_resid[{t}]"));
        }
    }
    if let Some(fc) = block.get("forecast") {
        assert_series(
            &fit.forecast,
            &as_f64_vec(fc),
            TOL_FIXED,
            &format!("{what}: forecast"),
        );
        assert_series(
            &fit.forecast_var,
            &as_f64_vec(&block["forecast_var"]),
            TOL_FIXED,
            &format!("{what}: forecast_var"),
        );
    }
    if let Some(comp) = block.get("components") {
        let d = fit.nobs_diffuse;
        let pairs: [(&str, Option<&UcComponent>); 5] = [
            ("level", fit.level.as_ref()),
            ("trend", fit.slope.as_ref()),
            ("seasonal", fit.seasonal.as_ref()),
            ("cycle", fit.cycle.as_ref()),
            ("autoregressive", fit.autoregressive.as_ref()),
        ];
        for (key, c) in pairs {
            if let Some(b) = comp.get(key) {
                let c = c.unwrap_or_else(|| panic!("{what}: {key} missing"));
                assert_component(c, b, spread, d, &format!("{what}: {key}"));
            }
        }
        if let Some(bs) = comp.get("freq_seasonal") {
            let bs = bs.as_array().unwrap();
            assert_eq!(fit.freq_seasonal.len(), bs.len(), "{what}: freq blocks");
            for (i, (c, b)) in fit.freq_seasonal.iter().zip(bs).enumerate() {
                assert_component(c, b, spread, d, &format!("{what}: freq_seasonal[{i}]"));
            }
        }
    }
}

fn fixed_fit(fx: &Value, case: &Value) -> UcFit {
    let name = case["name"].as_str().unwrap();
    let y = series(fx, case);
    let spec = spec_from_json(fx, &case["spec"]);
    let h = case["forecast"].as_array().map_or(0, |a| a.len());
    let opts = UcOptions {
        forecast_steps: h,
        forecast_exog: forecast_exog(fx, case),
        fixed_params: Some(as_f64_vec(&case["params"])),
        n_starts: 3,
    };
    unobserved_components(&y, &spec, &opts).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn assert_params_close(actual: &[f64], expected: &[f64], scale: f64, what: &str) {
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let denom = e.abs().max(1e-3 * scale);
        assert!(
            (a - e).abs() <= TOL_MLE_PARAM * denom,
            "{what}[{i}]: {a} vs {e} (rel diff {:e})",
            (a - e).abs() / denom
        );
    }
}

fn check_mle(fit: &UcFit, mle: &Value, what: &str) {
    let best = &mle["best"];
    let ll_ref = best["llf"].as_f64().unwrap();
    let ll_sm = mle["statsmodels"]["llf"].as_f64().unwrap();
    let ll_sc = mle["scipy"]["llf"].as_f64().unwrap();
    assert!(
        ll_ref >= ll_sm && ll_ref >= ll_sc,
        "{what}: fixture picks the better"
    );
    eprintln!(
        "{what}: rust {} vs best {ll_ref} (statsmodels {ll_sm}, scipy {ll_sc}); converged {}, \
         iters {}, params {:?}, se {:?}, at_boundary {:?}",
        fit.loglik, fit.converged, fit.n_iter, fit.params, fit.se, fit.at_boundary
    );
    assert!(
        fit.loglik >= ll_ref - TOL_MLE_LL,
        "{what}: loglik {} below the better reference optimum {ll_ref}",
        fit.loglik
    );
    assert_eq!(fit.param_names, names(&mle["param_names"]), "{what}: names");
    let p_ref = as_f64_vec(&best["params"]);
    let scale = p_ref.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(fit.loglik - ll_ref < 1e-3, "{what}: a different optimum");
    assert_params_close(&fit.params, &p_ref, scale, &format!("{what}: params"));
    let want: Vec<bool> = best["at_boundary"]
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_bool().expect("bool"))
        .collect();
    assert_eq!(fit.at_boundary, want, "{what}: at_boundary");
    let se_ref = as_f64_vec(&best["se_conditional"]);
    for (i, (a, e)) in fit.se.iter().zip(&se_ref).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{what}: se[{i}] must be NaN like the reference");
            continue;
        }
        assert_rel_close(
            *a,
            *e,
            TOL_SE,
            &format!("{what}: se[{i}] ({})", fit.param_names[i]),
        );
    }
    let int = |k: &str| best[k].as_u64().unwrap() as usize;
    assert_eq!(
        fit.nobs_diffuse,
        int("nobs_diffuse"),
        "{what}: nobs_diffuse"
    );
    assert_eq!(fit.k_diffuse, int("k_diffuse"), "{what}: k_diffuse");
    // AIC / BIC at the better reference optimum, re-evaluated: they follow
    // the log-likelihood (at optimizer tolerance) and df exactly.
    let df = (fit.k_params + fit.k_diffuse) as f64;
    assert_rel_close(fit.aic, 2.0 * df - 2.0 * fit.loglik, 1e-12, "aic identity");
    assert!(
        (fit.aic - best["aic"].as_f64().unwrap()).abs() < 2.0 * TOL_MLE_LL + 1e-3,
        "{what}: aic {} vs {}",
        fit.aic,
        best["aic"]
    );
}

// ------------------------------------------------------ (a) AR component

#[test]
fn ar_component_fixed_parameters_match_statsmodels() {
    let fx = load_fixture("uc_extensions.json");
    let cases = fx["autoregressive"].as_array().unwrap();
    assert!(cases.len() >= 9);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let fit = fixed_fit(&fx, case);
        let p = case["spec"]["autoregressive"].as_u64().unwrap() as usize;
        // The AR states are the stationary block; everything else diffuse.
        let pinf = as_f64_vec(&case["initial_diffuse_state_cov_diag"]);
        assert_eq!(
            fit.k_diffuse,
            pinf.iter().filter(|&&v| v == 1.0).count(),
            "{name}: k_diffuse"
        );
        assert_eq!(fit.k_diffuse + p, fit.k_states, "{name}: AR stationary");
        let names: Vec<String> = (1..=p).map(|j| format!("ar.L{j}")).collect();
        assert_eq!(
            &fit.state_names[fit.k_states - p..],
            &names[..],
            "{name}: state names"
        );
        assert!(fit.autoregressive.is_some(), "{name}");
        check_fixed_block(&fit, case, &format!("autoregressive/{name}"));
    }
    // The all-stationary specification: nothing diffuse at all.
    let all = cases
        .iter()
        .find(|c| c["name"] == "dlgdp_ntrend_ar3")
        .unwrap();
    let fit = fixed_fit(&fx, all);
    assert_eq!((fit.k_diffuse, fit.nobs_diffuse), (0, 0));
    assert!(fit.std_resid.iter().all(|v| v.is_finite()));
}

#[test]
fn ar_component_mle_reaches_the_better_reference_optimum() {
    let fx = load_fixture("uc_extensions.json");
    for case in fx["autoregressive_mle"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = series(&fx, case);
        let spec = spec_from_json(&fx, &case["spec"]);
        let fit = unobserved_components(&y, &spec, &UcOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        check_mle(&fit, case, name);
    }
}

// ------------------------------------------------- (b) stationary cycle

#[test]
fn stationary_cycle_fixed_parameters_match_statsmodels_block_initialization() {
    let fx = load_fixture("uc_extensions.json");
    let cases = fx["stationary_cycle"].as_array().unwrap();
    assert!(cases.len() >= 3);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let fit = fixed_fit(&fx, case);
        // The cycle states are not diffuse: two states fewer than the
        // all-diffuse count, and P_* on the cycle is the closed form.
        let pinf = as_f64_vec(&case["initial_diffuse_state_cov_diag"]);
        assert_eq!(
            fit.k_diffuse,
            pinf.iter().filter(|&&v| v == 1.0).count(),
            "{name}: k_diffuse = #(P_inf diag == 1)"
        );
        let p = case["spec"]
            .get("autoregressive")
            .map_or(0, |v| v.as_u64().unwrap() as usize);
        assert_eq!(
            fit.k_diffuse + 2 + p,
            fit.k_states,
            "{name}: cycle (and AR) stationary"
        );
        check_fixed_block(&fit, case, &format!("stationary_cycle/{name}"));
    }
}

#[test]
fn stationary_and_diffuse_cycle_likelihoods_differ_by_definition() {
    let fx = load_fixture("uc_extensions.json");
    let case = &fx["stationary_cycle"][0];
    assert_eq!(case["name"], "lgdp_llevel");
    let stat = fixed_fit(&fx, case);
    let mut spec = spec_from_json(&fx, &case["spec"]);
    spec.cycle_init = CycleInit::Diffuse;
    let y = series(&fx, case);
    let diffuse = unobserved_components(
        &y,
        &spec,
        &UcOptions {
            fixed_params: Some(as_f64_vec(&case["params"])),
            ..UcOptions::default()
        },
    )
    .unwrap();
    let d = &case["diffuse_cycle"];
    assert_rel_close(
        diffuse.loglik,
        d["loglike"].as_f64().unwrap(),
        TOL_FIXED,
        "diffuse cycle loglik",
    );
    assert_rel_close(diffuse.aic, d["aic"].as_f64().unwrap(), TOL_FIXED, "aic");
    assert_eq!(
        diffuse.k_params + diffuse.k_diffuse,
        d["df_model"].as_u64().unwrap() as usize
    );
    assert_eq!(
        diffuse.nobs_diffuse,
        d["nobs_diffuse"].as_u64().unwrap() as usize
    );
    // Same parameters, different likelihood definition: -454.98 vs -444.16.
    assert!((stat.loglik - diffuse.loglik).abs() > 10.0);
    assert_eq!(stat.k_diffuse + 2, diffuse.k_diffuse);
}

#[test]
fn stationary_cycle_gdp_mle_reaches_the_better_reference_optimum() {
    let fx = load_fixture("uc_extensions.json");
    for case in fx["stationary_cycle_mle"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = series(&fx, case);
        let spec = spec_from_json(&fx, &case["spec"]);
        let fit = unobserved_components(&y, &spec, &UcOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(fit.estimated);
        check_mle(&fit, case, name);
    }
}

#[test]
fn stationary_cycle_removes_the_low_frequency_singularity() {
    // statsmodels' own log-likelihood with only the frequency moved toward
    // zero at the GDP optimum: flat under the stationary initialization.
    let fx = load_fixture("uc_extensions.json");
    let y = as_f64_vec(&fx["data"]["lgdp"]);
    let spec = UcSpec {
        trend: TrendSpec::SmoothTrend,
        cycle: true,
        damped_cycle: true,
        stochastic_cycle: true,
        cycle_period_bounds: (2.0, 1e9),
        cycle_init: CycleInit::Stationary,
        ..UcSpec::default()
    };
    for s in fx["stationary_cycle_lambda_sweep"].as_array().unwrap() {
        let fit = unobserved_components(
            &y,
            &spec,
            &UcOptions {
                fixed_params: Some(as_f64_vec(&s["params"])),
                ..UcOptions::default()
            },
        )
        .unwrap();
        assert_rel_close(
            fit.loglik,
            s["stationary"].as_f64().unwrap(),
            TOL_FIXED,
            &format!("lambda {}", s["frequency"]),
        );
    }
}
