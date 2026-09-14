//! Golden-value tests for the structural time-series
//! (`unobserved_components`) and TVP-regression (`tvp_regression`)
//! estimators against `fixtures/uc.json` (generator:
//! `fixtures/generate_uc_fixtures.py`, which states the references and
//! the honest grade).
//!
//! Legs, graded per leg:
//!
//! 1. **Fixed parameters — independent-package golden.** statsmodels
//!    `UnobservedComponents(..., use_exact_diffuse=True).smooth(params)`
//!    at fixed parameters: log-likelihood, filtered/smoothed states and
//!    variances, one-step predictions and residuals, standardized
//!    residuals, forecasts and their variances, AIC/BIC and the component
//!    paths, for 26 component combinations (every level/trend
//!    specification, dummy and trigonometric seasonals, deterministic and
//!    stochastic, damped and undamped cycles, regressors) and three
//!    NaN-inserted series — pinned at [`TOL_FIXED`], with one measured
//!    exception: the *smoothed* variances inside the diffuse period, where
//!    the exact-diffuse smoother is ill-conditioned and statsmodels' own
//!    univariate and conventional smoothers disagree with each other (up to
//!    4.5e-3 relative on the eight-diffuse-state combination, while their
//!    filters agree to 2.9e-11). The fixture records that internal spread
//!    per case as `smoother_spread` and the test uses it as the tolerance
//!    there, so the requirement is "at least as close to the reference as
//!    the reference is to itself" rather than a guessed number; it is
//!    exactly [`TOL_FIXED`] on every specification whose two reference
//!    paths agree bit for bit, which is all but one.
//! 2. **The MLE — one criterion, two optimizers.** The Rust optimum must
//!    reach the BETTER of statsmodels' own L-BFGS and a SciPy
//!    Nelder-Mead + L-BFGS-B polish of the identical criterion (the
//!    fixture records both; on the damped-cycle case they land 0.0031
//!    log-likelihood apart, and on `llevel`/`lltrend_seasonal4`/`strend`
//!    SciPy's polish is the better one by 1e-8 to 6e-8, which is why two
//!    are recorded) within
//!    [`TOL_MLE_LL`], and the parameters within [`TOL_MLE_PARAM`] where the
//!    optimum is interior; the pile-up flags against an independent
//!    implementation of the same documented criterion; and
//!    observed-information standard errors against the same statsmodels
//!    Hessian, inverted over the non-boundary parameters (the quantity
//!    this crate reports), within [`TOL_SE`].
//! 3. **Nile, Durbin & Koopman (2012).** The local-level MLE reproduces
//!    the values printed in the book (15099, 1469.1) to their printed
//!    precision.
//! 4. **UK Seatbelts, Harvey & Durbin (1986) BSM** — the series is not
//!    redistributed (R's `datasets` is GPL), so that leg lives in the
//!    Python test (`bindings/python/tests/test_uc.py`), which fetches it
//!    through statsmodels' Rdatasets loader and pins the fixture's
//!    derived optimum (parameters, log-likelihood, the pile-ups on the
//!    slope and seasonal variances).
//! 5. **TVP regression.** Fixed-parameter parity with a custom statsmodels
//!    `MLEModel` transcription (documented state-space form), the
//!    `RecursiveLS` zero-state-variance limit (filtered coefficients and
//!    concentrated log-likelihood), the two-optimizer MLE with the
//!    pile-up flag on the true-zero variance, and a NaN-inserted series.

mod common;

use common::{as_f64_vec, assert_rel_close, load_fixture};
use serde_json::Value;
use tsecon_ssm::{
    tvp_regression, unobserved_components, FreqSeasonalSpec, TrendSpec, TvpOptions, UcComponent,
    UcFit, UcOptions, UcSpec,
};

/// Fixed-parameter parity tier (relative, absolute floor at 1).
const TOL_FIXED: f64 = 1e-8;
/// How many rows past the end of the diffuse period the diffuse
/// smoother's conditioning is still visible (measured: the disagreement
/// falls back under [`TOL_FIXED`] by `nobs_diffuse + 3`).
const DIFFUSE_SMOOTH_ROWS: usize = 2;
/// The Rust optimum may not fall below the better reference optimum by
/// more than this (absolute log-likelihood units).
const TOL_MLE_LL: f64 = 1e-5;
/// Parameter agreement at a shared interior optimum (relative, floor 1e-3
/// absolute for variances near zero).
const TOL_MLE_PARAM: f64 = 2e-3;
/// Observed-information standard errors vs the fixture's
/// `se_conditional` — statsmodels' own complex-step Hessian inverted over
/// the non-boundary parameters, which is the quantity this crate reports.
/// (`se_approx`, the full-Hessian inverse statsmodels prints, is a
/// different quantity whenever anything is at a boundary; the fixture
/// keeps both.)
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

fn spec_from_json(spec: &Value, x: &[Vec<f64>]) -> UcSpec {
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
        for (i, f) in fs.as_array().expect("array").iter().enumerate() {
            let period = f["period"].as_f64().expect("period");
            let mut fspec = FreqSeasonalSpec::new(period);
            if let Some(h) = f.get("harmonics") {
                fspec.harmonics = h.as_u64().expect("harmonics") as usize;
            }
            if let Some(st) = spec.get("stochastic_freq_seasonal") {
                fspec.stochastic = st[i].as_bool().expect("bool");
            }
            s.freq_seasonal.push(fspec);
        }
    }
    s.cycle = spec.get("cycle").and_then(|v| v.as_bool()).unwrap_or(false);
    s.damped_cycle = spec
        .get("damped_cycle")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    s.stochastic_cycle = spec
        .get("stochastic_cycle")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if let Some(b) = spec.get("cycle_period_bounds") {
        let v = as_f64_vec(b);
        s.cycle_period_bounds = (v[0], v[1]);
    }
    if spec.get("exog").is_some() {
        s.exog = x.to_vec();
    }
    s
}

/// NaN-aware series comparison: where the reference is NaN the estimate
/// must be NaN; elsewhere relative closeness.
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

/// The same, with the *smoothed*-variance tolerance relaxed inside the
/// diffuse period to the fixture's recorded `smoother_spread` — the
/// distance between statsmodels' OWN two smoother implementations on this
/// model at these parameters. The exact-diffuse smoother recursion is the
/// ill-conditioned part of this family (its filter is not: the filtered
/// variances hold 1e-8 everywhere), and where the reference disagrees with
/// itself by 5e-3 a 1e-8 claim against one of its two answers would be
/// theatre. Outside the diffuse period, and on every specification whose
/// two reference paths agree bit for bit, this is exactly [`TOL_FIXED`].
fn assert_smoothed_var_paths(
    actual: &[Vec<f64>],
    expected: &Value,
    spread: f64,
    d_diffuse: usize,
    what: &str,
) {
    let exp = rows(expected);
    assert_eq!(actual.len(), exp.len(), "{what}: periods");
    let relaxed = spread.max(TOL_FIXED);
    for (t, (a, e)) in actual.iter().zip(&exp).enumerate() {
        let tol = if t <= d_diffuse + DIFFUSE_SMOOTH_ROWS {
            relaxed
        } else {
            TOL_FIXED
        };
        assert_series(a, e, tol, &format!("{what}[{t}]"));
    }
}

fn assert_component(c: &UcComponent, block: &Value, tol: f64, spread: f64, d: usize, what: &str) {
    assert_series(
        &c.filtered,
        &as_f64_vec(&block["filtered"]),
        tol,
        &format!("{what}.filtered"),
    );
    assert_series(
        &c.filtered_var,
        &as_f64_vec(&block["filtered_var"]),
        tol,
        &format!("{what}.filtered_var"),
    );
    assert_series(
        &c.smoothed,
        &as_f64_vec(&block["smoothed"]),
        tol,
        &format!("{what}.smoothed"),
    );
    // A component variance is a sum of state variances, so it inherits the
    // diffuse smoother's conditioning; same treatment.
    let exp = as_f64_vec(&block["smoothed_var"]);
    let relaxed = spread.max(tol);
    for (t, (a, e)) in c.smoothed_var.iter().zip(&exp).enumerate() {
        let tt = if t <= d + DIFFUSE_SMOOTH_ROWS {
            relaxed
        } else {
            tol
        };
        assert_series(&[*a], &[*e], tt, &format!("{what}.smoothed_var[{t}]"));
    }
}

/// Every fixed-parameter quantity of a block against the fit.
fn check_fixed_block(fit: &UcFit, block: &Value, what: &str) {
    assert_eq!(
        fit.k_states,
        block["k_states"].as_u64().unwrap() as usize,
        "{what}: k_states"
    );
    assert_eq!(
        fit.nobs_diffuse,
        block["nobs_diffuse"].as_u64().unwrap() as usize,
        "{what}: nobs_diffuse"
    );
    assert_eq!(
        fit.param_names,
        names(&block["param_names"]),
        "{what}: param_names"
    );
    assert_rel_close(
        fit.loglik,
        block["loglike"].as_f64().unwrap(),
        TOL_FIXED,
        &format!("{what}: loglike"),
    );
    assert_rel_close(
        fit.aic,
        block["aic"].as_f64().unwrap(),
        TOL_FIXED,
        &format!("{what}: aic"),
    );
    assert_rel_close(
        fit.bic,
        block["bic"].as_f64().unwrap(),
        TOL_FIXED,
        &format!("{what}: bic"),
    );
    assert_paths(
        &fit.filtered_state,
        &block["filtered_state"],
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
    let spread = block
        .get("smoother_spread")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    assert_smoothed_var_paths(
        &fit.smoothed_state_var,
        &block["smoothed_state_var"],
        spread,
        fit.nobs_diffuse,
        &format!("{what}: smoothed_state_var"),
    );
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
    // statsmodels writes 0.0 for the standardized residual inside the
    // diffuse period and at a missing period; this crate reports NaN at
    // both (no finite prediction variance exists inside the diffuse
    // period, and no prediction error exists where there is no
    // observation). The missing periods are the ones whose reference
    // prediction error is NaN. Compare everywhere else.
    let sr = as_f64_vec(&block["std_resid"]);
    let ref_resid = as_f64_vec(&block["resid"]);
    for (t, (a, e)) in fit.std_resid.iter().zip(&sr).enumerate() {
        if t < fit.nobs_diffuse {
            assert!(
                a.is_nan(),
                "{what}: std_resid[{t}] inside the diffuse period must be NaN"
            );
        } else if e.is_nan() || ref_resid[t].is_nan() {
            assert!(
                a.is_nan(),
                "{what}: std_resid[{t}] at a missing period must be NaN"
            );
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
        if let Some(b) = comp.get("level") {
            assert_component(
                fit.level.as_ref().expect("level"),
                b,
                TOL_FIXED,
                spread,
                fit.nobs_diffuse,
                &format!("{what}: level"),
            );
        }
        if let Some(b) = comp.get("trend") {
            assert_component(
                fit.slope.as_ref().expect("slope"),
                b,
                TOL_FIXED,
                spread,
                fit.nobs_diffuse,
                &format!("{what}: slope"),
            );
        }
        if let Some(b) = comp.get("seasonal") {
            assert_component(
                fit.seasonal.as_ref().expect("seasonal"),
                b,
                TOL_FIXED,
                spread,
                fit.nobs_diffuse,
                &format!("{what}: seasonal"),
            );
        }
        if let Some(b) = comp.get("cycle") {
            assert_component(
                fit.cycle.as_ref().expect("cycle"),
                b,
                TOL_FIXED,
                spread,
                fit.nobs_diffuse,
                &format!("{what}: cycle"),
            );
        }
        if let Some(bs) = comp.get("freq_seasonal") {
            let bs = bs.as_array().unwrap();
            assert_eq!(fit.freq_seasonal.len(), bs.len(), "{what}: freq blocks");
            for (i, (c, b)) in fit.freq_seasonal.iter().zip(bs).enumerate() {
                assert_component(
                    c,
                    b,
                    TOL_FIXED,
                    spread,
                    fit.nobs_diffuse,
                    &format!("{what}: freq_seasonal[{i}]"),
                );
            }
        }
    }
}

fn fixed_opts(block: &Value, h: usize, forecast_exog: Vec<Vec<f64>>) -> UcOptions {
    UcOptions {
        forecast_steps: h,
        forecast_exog,
        fixed_params: Some(as_f64_vec(&block["params"])),
        n_starts: 3,
    }
}

/// Parameters at a shared interior optimum agree to `TOL_MLE_PARAM`
/// (relative, with an absolute floor for variances that are numerically
/// zero on either side).
fn assert_params_close(actual: &[f64], expected: &[f64], scale: f64, what: &str) {
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let floor = 1e-3 * scale;
        let denom = e.abs().max(floor);
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
        "{what}: fixture picks the better optimum"
    );
    assert!(
        fit.loglik >= ll_ref - TOL_MLE_LL,
        "{what}: loglik {} below the better reference optimum {ll_ref} (statsmodels {ll_sm}, scipy {ll_sc})",
        fit.loglik
    );
    eprintln!(
        "{what}: rust {} vs best {ll_ref} (statsmodels {ll_sm}, scipy {ll_sc}); converged {}, iters {}",
        fit.loglik, fit.converged, fit.n_iter
    );
    let p_ref = as_f64_vec(&best["params"]);
    let scale = p_ref.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let same_optimum = fit.loglik - ll_ref < 1e-3;
    if same_optimum {
        assert_params_close(&fit.params, &p_ref, scale, &format!("{what}: params"));
        // The pile-up flags, against an independent implementation of the
        // same documented criterion in the fixture generator.
        if let Some(b) = best.get("at_boundary") {
            let want: Vec<bool> = b
                .as_array()
                .expect("array")
                .iter()
                .map(|v| v.as_bool().expect("bool"))
                .collect();
            assert_eq!(fit.at_boundary, want, "{what}: at_boundary");
        }
    }
    let se_ref = as_f64_vec(&best["se_conditional"]);
    for (i, (a, e)) in fit.se.iter().zip(&se_ref).enumerate() {
        if fit.at_boundary[i] {
            assert!(a.is_nan(), "{what}: se[{i}] at a boundary must be NaN");
            continue;
        }
        if e.is_nan() || !same_optimum {
            continue;
        }
        assert_rel_close(
            *a,
            *e,
            TOL_SE,
            &format!("{what}: se[{i}] ({})", fit.param_names[i]),
        );
    }
}

// ------------------------------------------------------------------ Nile

#[test]
fn nile_local_level_fixed_at_durbin_koopman_values() {
    let fx = load_fixture("uc.json");
    let nile = as_f64_vec(&fx["nile"]["y"]);
    let block = &fx["nile"]["fixed"];
    let fit = unobserved_components(
        &nile,
        &UcSpec::default(),
        &fixed_opts(block, 10, Vec::new()),
    )
    .unwrap();
    assert!(!fit.estimated && fit.n_iter == 0);
    assert!(fit.se.iter().all(|v| v.is_nan()));
    check_fixed_block(&fit, block, "nile fixed");
}

#[test]
fn nile_local_level_mle_reproduces_durbin_koopman_and_both_optimizers() {
    let fx = load_fixture("uc.json");
    let nile = as_f64_vec(&fx["nile"]["y"]);
    let fit = unobserved_components(&nile, &UcSpec::default(), &UcOptions::default()).unwrap();
    assert!(fit.estimated && fit.converged);
    check_mle(&fit, &fx["nile"]["mle"], "nile mle");
    // Durbin & Koopman (2012, sec. 2.2): sigma2_eps = 15099, sigma2_eta =
    // 1469.1 as printed (5 significant figures); matched to within their
    // rounding.
    let dk = as_f64_vec(&fx["nile"]["dk_params"]);
    assert!(
        (fit.params[0] - dk[0]).abs() < 1.0,
        "sigma2_eps {} vs DK {}",
        fit.params[0],
        dk[0]
    );
    assert!(
        (fit.params[1] - dk[1]).abs() < 0.2,
        "sigma2_eta {} vs DK {}",
        fit.params[1],
        dk[1]
    );
    assert!(fit.at_boundary.iter().all(|b| !b));
    assert!(fit.se.iter().all(|v| v.is_finite() && *v > 0.0));
    assert_eq!(fit.nobs_diffuse, 1);
    assert_eq!(fit.k_diffuse, 1);
}

// --------------------------------------------------------- fixed params

#[test]
fn fixed_parameter_component_combinations_match_statsmodels() {
    let fx = load_fixture("uc.json");
    let y = as_f64_vec(&fx["sim"]["y"]);
    let x = columns(&fx["sim"]["x"]);
    let xf = columns(&fx["sim"]["x_forecast"]);
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 26);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let spec = spec_from_json(&case["spec"], &x);
        let fexog = if spec.exog.is_empty() {
            Vec::new()
        } else {
            xf.clone()
        };
        let fit = unobserved_components(&y, &spec, &fixed_opts(case, 8, fexog))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        check_fixed_block(&fit, case, name);
    }
}

#[test]
fn missing_observations_fixed_parameters_match_statsmodels() {
    let fx = load_fixture("uc.json");
    let y = as_f64_vec(&fx["sim"]["y_missing"]);
    assert_eq!(y.iter().filter(|v| v.is_nan()).count(), 12);
    // An interior run, two isolated holes, and a missing tail, so the
    // forecast origin itself is unobserved.
    assert!(y[10].is_nan() && y[14].is_nan() && !y[15].is_nan());
    assert!(y[40].is_nan() && y[77].is_nan());
    assert!(y[115].is_nan() && y[119].is_nan() && !y[114].is_nan());
    let x = columns(&fx["sim"]["x"]);
    let xf = columns(&fx["sim"]["x_forecast"]);
    for case in fx["missing"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let spec = spec_from_json(&case["spec"], &x);
        let fexog = if spec.exog.is_empty() {
            Vec::new()
        } else {
            xf.clone()
        };
        let fit = unobserved_components(&y, &spec, &fixed_opts(case, 8, fexog))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            fit.nobs_observed,
            y.iter().filter(|v| v.is_finite()).count()
        );
        check_fixed_block(&fit, case, &format!("missing/{name}"));
    }
}

// ------------------------------------------------------------------ MLE

#[test]
fn mle_cases_reach_the_better_of_two_reference_optimizers() {
    let fx = load_fixture("uc.json");
    let y = as_f64_vec(&fx["sim"]["y"]);
    let x = columns(&fx["sim"]["x"]);
    for case in fx["mle_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let spec = spec_from_json(&case["spec"], &x);
        let fit = unobserved_components(&y, &spec, &UcOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(fit.estimated);
        assert_eq!(fit.param_names, names(&case["param_names"]));
        check_mle(&fit, case, name);
    }
}

// ------------------------------------------------------------------ TVP

fn check_tvp_fixed(fit: &tsecon_ssm::TvpFit, block: &Value, what: &str) {
    assert_rel_close(
        fit.loglik,
        block["loglike"].as_f64().unwrap(),
        TOL_FIXED,
        &format!("{what}: loglike"),
    );
    assert_rel_close(
        fit.aic,
        block["aic"].as_f64().unwrap(),
        TOL_FIXED,
        &format!("{what}: aic"),
    );
    assert_rel_close(
        fit.bic,
        block["bic"].as_f64().unwrap(),
        TOL_FIXED,
        &format!("{what}: bic"),
    );
    assert_eq!(
        fit.nobs_diffuse,
        block["nobs_diffuse"].as_u64().unwrap() as usize
    );
    assert_paths(
        &fit.beta_filtered,
        &block["beta_filtered"],
        TOL_FIXED,
        &format!("{what}: beta_filtered"),
    );
    assert_paths(
        &fit.beta_filtered_var,
        &block["beta_filtered_var"],
        TOL_FIXED,
        &format!("{what}: beta_filtered_var"),
    );
    assert_paths(
        &fit.beta_smoothed,
        &block["beta_smoothed"],
        TOL_FIXED,
        &format!("{what}: beta_smoothed"),
    );
    assert_smoothed_var_paths(
        &fit.beta_smoothed_var,
        &block["beta_smoothed_var"],
        block
            .get("smoother_spread")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        fit.nobs_diffuse,
        &format!("{what}: beta_smoothed_var"),
    );
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
    // As for the structural model: statsmodels writes 0.0 inside the
    // diffuse period and at missing periods, this crate NaN. The missing
    // periods are the ones with a NaN reference prediction error.
    let sr = as_f64_vec(&block["std_resid"]);
    let ref_resid = as_f64_vec(&block["resid"]);
    for (t, (a, e)) in fit.std_resid.iter().zip(&sr).enumerate() {
        if t < fit.nobs_diffuse || e.is_nan() || ref_resid[t].is_nan() {
            assert!(a.is_nan(), "{what}: std_resid[{t}] must be NaN");
        } else {
            assert_rel_close(*a, *e, TOL_FIXED, &format!("{what}: std_resid[{t}]"));
        }
    }
}

#[test]
fn tvp_fixed_parameters_match_the_statsmodels_transcription() {
    let fx = load_fixture("uc.json");
    let tv = &fx["tvp"];
    let y = as_f64_vec(&tv["y"]);
    let x = columns(&tv["x"]);
    let opts = TvpOptions {
        constant: true,
        fixed_params: Some(as_f64_vec(&tv["fixed"]["params"])),
        n_starts: 3,
    };
    let fit = tvp_regression(&y, &x, &opts).unwrap();
    assert_eq!(fit.k, 3);
    assert_eq!(fit.coef_names, vec!["const", "x1", "x2"]);
    assert!(!fit.estimated);
    check_tvp_fixed(&fit, &tv["fixed"], "tvp fixed");

    let ym = as_f64_vec(&tv["y_missing"]);
    let fit_m = tvp_regression(&ym, &x, &opts).unwrap();
    assert_eq!(
        fit_m.nobs_observed,
        ym.iter().filter(|v| v.is_finite()).count()
    );
    check_tvp_fixed(&fit_m, &tv["missing"], "tvp missing");
}

#[test]
fn tvp_zero_state_variance_is_recursive_least_squares() {
    let fx = load_fixture("uc.json");
    let tv = &fx["tvp"];
    let y = as_f64_vec(&tv["y"]);
    let x = columns(&tv["x"]);
    let rls = &tv["rls"];
    let scale = rls["scale"].as_f64().unwrap();
    let opts = TvpOptions {
        constant: true,
        fixed_params: Some(vec![scale, 0.0, 0.0, 0.0]),
        n_starts: 3,
    };
    let fit = tvp_regression(&y, &x, &opts).unwrap();
    assert!(fit.pile_up.iter().all(|b| *b) && fit.at_boundary == vec![false, true, true, true]);
    // statsmodels RecursiveLS: exact-diffuse filter with the scale
    // concentrated out; at sigma2_eps = scale its concentrated
    // log-likelihood is the full log-likelihood, and its filtered
    // coefficients are the TVP filter's.
    assert_rel_close(
        fit.loglik,
        rls["llf"].as_f64().unwrap(),
        TOL_FIXED,
        "rls llf",
    );
    assert_paths(
        &fit.beta_filtered,
        &rls["filtered_coefficients"],
        TOL_FIXED,
        "rls filtered coefficients",
    );
    assert_eq!(
        fit.nobs_diffuse,
        rls["nobs_diffuse"].as_u64().unwrap() as usize
    );
    // The final filtered coefficient is the full-sample OLS estimate.
    let params = as_f64_vec(&rls["params"]);
    for (a, e) in fit.beta_filtered.last().unwrap().iter().zip(&params) {
        assert_rel_close(*a, *e, TOL_FIXED, "rls final = OLS");
    }
    check_tvp_fixed(&fit, &tv["rls_limit"], "rls limit (MLEModel)");
}

#[test]
fn tvp_mle_reaches_both_optimizers_and_flags_the_true_zero_variance() {
    let fx = load_fixture("uc.json");
    let tv = &fx["tvp"];
    let y = as_f64_vec(&tv["y"]);
    let x = columns(&tv["x"]);
    let fit = tvp_regression(&y, &x, &TvpOptions::default()).unwrap();
    assert!(fit.estimated);
    let mle = &tv["mle"];
    let best = &mle["best"];
    let ll_ref = best["llf"].as_f64().unwrap();
    eprintln!(
        "tvp mle: rust {} vs best {ll_ref} (statsmodels {}, scipy {}); params {:?}; pile_up {:?}",
        fit.loglik, mle["statsmodels"]["llf"], mle["scipy"]["llf"], fit.params, fit.pile_up
    );
    assert!(
        fit.loglik >= ll_ref - TOL_MLE_LL,
        "loglik {} vs {ll_ref}",
        fit.loglik
    );
    let p_ref = as_f64_vec(&best["params"]);
    // The true-zero third coefficient variance piles up: statsmodels'
    // optimum has it at ~3e-17; flagged here, NaN standard error.
    assert!(fit.pile_up[2] && fit.at_boundary[3]);
    assert!(fit.se[3].is_nan());
    assert!(!fit.pile_up[0] && !fit.pile_up[1]);
    assert_params_close(&fit.params[..3], &p_ref[..3], 1.0, "tvp params");
    let se_ref = as_f64_vec(&best["se_conditional"]);
    for (i, e) in se_ref.iter().enumerate().take(3) {
        assert_rel_close(fit.se[i], *e, TOL_SE, &format!("tvp se[{i}]"));
    }
    let want: Vec<bool> = best["at_boundary"]
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_bool().expect("bool"))
        .collect();
    assert_eq!(fit.at_boundary, want, "tvp at_boundary");
    assert_eq!(
        fit.param_names,
        vec!["sigma2.irregular", "sigma2.const", "sigma2.x1", "sigma2.x2"]
    );
}
