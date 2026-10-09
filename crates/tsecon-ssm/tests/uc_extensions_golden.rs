//! Golden-value tests for the 0.11.0 follow-ups of
//! `unobserved_components` against `fixtures/uc_extensions.json`
//! (generator: `fixtures/generate_uc_extensions_fixtures.py`, which
//! states the references and the honest grade):
//!
//! * **`autoregressive = p`** — statsmodels' own AR(p) component
//!   (row-companion form, always stationary-initialized), at fixed
//!   parameters on Nile, inflation, GDP, GDP growth (every state
//!   stationary) and simulated specifications, and three MLE cases;
//! * **`mle_regression = false`** — statsmodels' regressors in the state
//!   vector (exactly diffuse constant states, design row `x_t'`), at fixed
//!   parameters, plus the closed forms (white noise + regressors is OLS:
//!   coefficients, covariance, log-likelihood, forecast and its variance),
//!   the marginal/profile identity `l_state - l_mle(beta_gls) = ln det
//!   P_beta / 2` and the regressor-units identity, and one MLE case;
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
/// A block's `rounding_spread` — how far statsmodels' own paths move under
/// a one-ulp change of an input, measured by the generator — times this is
/// the path tolerance where it exceeds [`TOL_FIXED`] (as in the Python
/// leg's `SPREAD_FACTOR`). Only weakly resolved diffuse directions get
/// there (`long_diffuse`, the two N(0, 1)-regressor designs).
const ROUNDING_FACTOR: f64 = 10.0;

/// The tolerances of one fixed block: `path` for every path, and
/// `smoothed_diffuse` for the smoothed variances inside the diffuse period
/// and the rows just after it (the reference's own smoother spread).
#[derive(Clone, Copy)]
struct Tol {
    path: f64,
    smoothed_diffuse: f64,
}

impl Tol {
    fn of(block: &Value) -> Self {
        let get = |k: &str| block.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let path = TOL_FIXED.max(ROUNDING_FACTOR * get("rounding_spread"));
        Tol {
            path,
            smoothed_diffuse: get("smoother_spread").max(path),
        }
    }
}

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
        "sim2" => "sim2_y",
        "sim2_missing" => "sim2_y_missing",
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
    if let Some(b) = spec.get("mle_regression") {
        s.mle_regression = b.as_bool().expect("bool");
    }
    s
}

fn forecast_exog(fx: &Value, case: &Value) -> Vec<Vec<f64>> {
    match case["spec"].get("exog").and_then(|v| v.as_str()) {
        Some(x) => {
            let h = case["forecast"].as_array().map_or(0, |a| a.len());
            columns(&fx["data"][format!("{x}_forecast").as_str()])
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

/// A smoothed-variance *path* (index = period): the path tolerance except
/// inside the diffuse period and the rows just after it, where the
/// tolerance is the reference's own smoother spread.
fn assert_smoothed_var(actual: &[f64], expected: &[f64], tol: Tol, d: usize, what: &str) {
    for (t, (a, e)) in actual.iter().zip(expected).enumerate() {
        let tol = if t <= d + DIFFUSE_SMOOTH_ROWS {
            tol.smoothed_diffuse
        } else {
            tol.path
        };
        assert_series(&[*a], &[*e], tol, &format!("{what}[{t}]"));
    }
}

fn assert_component(c: &UcComponent, block: &Value, tol: Tol, d: usize, what: &str) {
    assert_series(
        &c.filtered,
        &as_f64_vec(&block["filtered"]),
        tol.path,
        &format!("{what}.filtered"),
    );
    assert_series(
        &c.filtered_var,
        &as_f64_vec(&block["filtered_var"]),
        tol.path,
        &format!("{what}.filtered_var"),
    );
    assert_series(
        &c.smoothed,
        &as_f64_vec(&block["smoothed"]),
        tol.path,
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
        tol,
        d,
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
    let tol = Tol::of(block);
    if let Some(v) = block.get("filtered_state") {
        assert_paths(
            &fit.filtered_state,
            v,
            tol.path,
            &format!("{what}: filtered_state"),
        );
        assert_paths(
            &fit.filtered_state_var,
            &block["filtered_state_var"],
            tol.path,
            &format!("{what}: filtered_state_var"),
        );
        assert_paths(
            &fit.smoothed_state,
            &block["smoothed_state"],
            tol.path,
            &format!("{what}: smoothed_state"),
        );
        let exp = rows(&block["smoothed_state_var"]);
        assert_eq!(fit.smoothed_state_var.len(), exp.len());
        for (t, (a, e)) in fit.smoothed_state_var.iter().zip(&exp).enumerate() {
            // Every state of row t is at period t.
            let tol = if t <= fit.nobs_diffuse + DIFFUSE_SMOOTH_ROWS {
                tol.smoothed_diffuse
            } else {
                tol.path
            };
            assert_series(a, e, tol, &format!("{what}: smoothed_state_var[{t}]"));
        }
    }
    assert_series(
        &fit.fitted,
        &as_f64_vec(&block["fitted"]),
        tol.path,
        &format!("{what}: fitted"),
    );
    assert_series(
        &fit.resid,
        &as_f64_vec(&block["resid"]),
        tol.path,
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
            assert_rel_close(*a, *e, tol.path, &format!("{what}: std_resid[{t}]"));
        }
    }
    if let Some(fc) = block.get("forecast") {
        assert_series(
            &fit.forecast,
            &as_f64_vec(fc),
            tol.path,
            &format!("{what}: forecast"),
        );
        assert_series(
            &fit.forecast_var,
            &as_f64_vec(&block["forecast_var"]),
            tol.path,
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
                assert_component(c, b, tol, d, &format!("{what}: {key}"));
            }
        }
        if let Some(bs) = comp.get("freq_seasonal") {
            let bs = bs.as_array().unwrap();
            assert_eq!(fit.freq_seasonal.len(), bs.len(), "{what}: freq blocks");
            for (i, (c, b)) in fit.freq_seasonal.iter().zip(bs).enumerate() {
                assert_component(c, b, tol, d, &format!("{what}: freq_seasonal[{i}]"));
            }
        }
        if let Some(rc) = comp.get("regression_coefficients") {
            check_regression_coefficients(fit, rc, tol, &format!("{what}: regression"));
        }
    }
}

/// The regression-coefficient states (`mle_regression = false`) against
/// statsmodels' `results.regression_coefficients`: per-regressor paths as
/// `nobs x k` rows, and the `k x k` smoothed covariance at the last period.
fn check_regression_coefficients(fit: &UcFit, rc: &Value, tol: Tol, what: &str) {
    let k = fit.regression_coefficients.len();
    assert!(k > 0, "{what}: components missing");
    let col = |key: &str, j: usize| -> Vec<f64> { rows(&rc[key]).iter().map(|r| r[j]).collect() };
    for (j, c) in fit.regression_coefficients.iter().enumerate() {
        let block = serde_json::json!({
            "filtered": col("filtered", j),
            "filtered_var": col("filtered_var", j),
            "smoothed": col("smoothed", j),
            "smoothed_var": col("smoothed_var", j),
        });
        assert_component(c, &block, tol, fit.nobs_diffuse, &format!("{what}[{j}]"));
    }
    let cov = fit.regression_coefficients_cov.as_ref().expect("cov");
    let want = rows(&rc["smoothed_cov_last"]);
    for i in 0..k {
        assert_series(&cov[i], &want[i], tol.path, &format!("{what}: cov[{i}]"));
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

// --------------------------------------------- (c) state-vector regression

/// `ln det` of a small symmetric positive-definite matrix (Cholesky).
fn logdet(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    let mut out = 0.0;
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                l[i][i] = (a[i][i] - s).sqrt();
                out += 2.0 * l[i][i].ln();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    out
}

#[test]
fn state_regression_fixed_parameters_match_statsmodels() {
    let fx = load_fixture("uc_extensions.json");
    let cases = fx["state_regression"].as_array().unwrap();
    assert!(cases.len() >= 4);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let fit = fixed_fit(&fx, case);
        let k = fit.regression_coefficients.len();
        assert_eq!(k, 2, "{name}");
        // No beta parameters; the coefficient states close the state
        // vector and are exactly diffuse.
        assert!(fit.param_names.iter().all(|p| !p.starts_with("beta")));
        assert_eq!(
            &fit.state_names[fit.k_states - 2..],
            &["beta.x1", "beta.x2"]
        );
        assert!(case["components"].get("regression_coefficients").is_some());
        check_fixed_block(&fit, case, &format!("state_regression/{name}"));
        // The smoothed coefficient path is constant in t (a constant state
        // conditioned on the whole sample).
        for c in &fit.regression_coefficients {
            let last = *c.smoothed.last().unwrap();
            assert!(c
                .smoothed
                .iter()
                .all(|v| (v - last).abs() <= 1e-9 * last.abs().max(1.0)));
        }
    }
}

#[test]
fn white_noise_plus_state_regression_is_ols() {
    let fx = load_fixture("uc_extensions.json");
    let cf = &fx["state_regression_closed_form"];
    let y = as_f64_vec(&fx["data"]["sim2_y"]);
    let x = columns(&fx["data"]["sim2_x"]);
    let xf = columns(&fx["data"]["sim2_x_forecast"]);
    let s2 = cf["sigma2"].as_f64().unwrap();
    let fit = unobserved_components(
        &y,
        &UcSpec {
            trend: TrendSpec::Irregular,
            exog: x,
            mle_regression: false,
            ..UcSpec::default()
        },
        &UcOptions {
            forecast_steps: xf[0].len(),
            forecast_exog: xf,
            fixed_params: Some(vec![s2]),
            ..UcOptions::default()
        },
    )
    .unwrap();
    let tol = 1e-10;
    assert_rel_close(
        fit.loglik,
        cf["loglike"].as_f64().unwrap(),
        tol,
        "loglik = closed form",
    );
    let beta = as_f64_vec(&cf["beta_ols"]);
    let cov = rows(&cf["cov"]);
    for (j, c) in fit.regression_coefficients.iter().enumerate() {
        for t in [0, y.len() - 1] {
            assert_rel_close(
                c.smoothed[t],
                beta[j],
                tol,
                &format!("beta[{j}] at t={t} = OLS"),
            );
            assert_rel_close(
                c.smoothed_var[t],
                cov[j][j],
                tol,
                &format!("var beta[{j}] at t={t} = s2 (X'X)^-1"),
            );
        }
    }
    let got = fit.regression_coefficients_cov.as_ref().unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert_rel_close(got[i][j], cov[i][j], tol, &format!("cov[{i}][{j}]"));
        }
    }
    assert_series(
        &fit.forecast,
        &as_f64_vec(&cf["forecast"]),
        tol,
        "forecast = x' beta",
    );
    assert_series(
        &fit.forecast_var,
        &as_f64_vec(&cf["forecast_var"]),
        tol,
        "forecast var = s2 (1 + x'(X'X)^-1 x)",
    );
    assert_eq!(fit.nobs_diffuse, 2);
    assert_eq!(fit.k_params + fit.k_diffuse, 3, "statsmodels df_model");
}

#[test]
fn state_and_mle_regression_differ_by_half_logdet_of_the_coefficient_covariance() {
    let fx = load_fixture("uc_extensions.json");
    let y = as_f64_vec(&fx["data"]["sim2_y"]);
    let x = columns(&fx["data"]["sim2_x"]);
    for c in fx["state_regression_identity"].as_array().unwrap() {
        let trend = TrendSpec::parse(c["level"].as_str().unwrap()).unwrap();
        let ar = c["autoregressive"].as_u64().map(|p| p as usize);
        let what = format!("{trend:?} AR {ar:?}");
        let params = as_f64_vec(&c["params"]);
        let base = UcSpec {
            trend,
            autoregressive: ar,
            exog: x.clone(),
            ..UcSpec::default()
        };
        let state = unobserved_components(
            &y,
            &UcSpec {
                mle_regression: false,
                ..base.clone()
            },
            &UcOptions {
                fixed_params: Some(params.clone()),
                ..UcOptions::default()
            },
        )
        .unwrap();
        assert_rel_close(
            state.loglik,
            c["loglike_state"].as_f64().unwrap(),
            TOL_FIXED,
            &format!("{what}: l_state"),
        );
        // The GLS beta fed back as MLE-regression parameters.
        let beta: Vec<f64> = state
            .regression_coefficients
            .iter()
            .map(|r| *r.smoothed.last().unwrap())
            .collect();
        assert_series(
            &beta,
            &as_f64_vec(&c["beta_gls"]),
            TOL_FIXED,
            &format!("{what}: beta_gls"),
        );
        let mut p = params.clone();
        p.extend(&beta);
        let mle = unobserved_components(
            &y,
            &base,
            &UcOptions {
                fixed_params: Some(p),
                ..UcOptions::default()
            },
        )
        .unwrap();
        assert_rel_close(
            mle.loglik,
            c["loglike_mle_at_gls"].as_f64().unwrap(),
            TOL_FIXED,
            &format!("{what}: l_mle"),
        );
        let half = 0.5 * logdet(state.regression_coefficients_cov.as_ref().unwrap());
        assert_rel_close(
            half,
            c["half_logdet_p_beta"].as_f64().unwrap(),
            TOL_FIXED,
            &format!("{what}: half ln det"),
        );
        let gap = state.loglik - mle.loglik - half;
        assert!(gap.abs() < 1e-10, "{what}: identity gap {gap:e}");
    }
}

#[test]
fn state_regression_likelihood_moves_with_the_units_of_the_regressors() {
    let fx = load_fixture("uc_extensions.json");
    let u = &fx["state_regression_units"];
    let y = as_f64_vec(&fx["data"]["sim2_y"]);
    let x = columns(&fx["data"]["sim2_x"]);
    let c = u["scale"].as_f64().unwrap();
    let mut xc = x.clone();
    for v in xc[0].iter_mut() {
        *v *= c;
    }
    let fit = |cols: &[Vec<f64>], mle_regression: bool, params: Vec<f64>| {
        unobserved_components(
            &y,
            &UcSpec {
                exog: cols.to_vec(),
                mle_regression,
                ..UcSpec::default()
            },
            &UcOptions {
                fixed_params: Some(params),
                ..UcOptions::default()
            },
        )
        .unwrap()
    };
    let a = fit(&x, false, vec![0.8, 0.3]);
    let b = fit(&xc, false, vec![0.8, 0.3]);
    assert_rel_close(
        a.loglik,
        u["loglike_state"].as_f64().unwrap(),
        TOL_FIXED,
        "state",
    );
    assert_rel_close(
        b.loglik,
        u["loglike_state_scaled"].as_f64().unwrap(),
        TOL_FIXED,
        "scaled",
    );
    // Exactly -ln|c| per rescaled column, and the coefficient by 1/c.
    assert!(
        (a.loglik - b.loglik - c.ln()).abs() < 1e-10,
        "{}",
        a.loglik - b.loglik
    );
    let (ba, bb) = (
        a.regression_coefficients[0].smoothed[0],
        b.regression_coefficients[0].smoothed[0],
    );
    assert_rel_close(bb * c, ba, 1e-10, "beta scales by 1/c");
    // The MLE form does not move.
    let m = fit(&x, true, vec![0.8, 0.3, 0.5, -0.3]);
    let mc = fit(&xc, true, vec![0.8, 0.3, 0.5 / c, -0.3]);
    assert_rel_close(
        m.loglik,
        u["loglike_mle"].as_f64().unwrap(),
        TOL_FIXED,
        "mle",
    );
    assert!((m.loglik - mc.loglik).abs() < 1e-10);
}

#[test]
fn state_regression_mle_reaches_the_better_reference_optimum() {
    let fx = load_fixture("uc_extensions.json");
    for case in fx["state_regression_mle"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = series(&fx, case);
        let spec = spec_from_json(&fx, &case["spec"]);
        let fit = unobserved_components(&y, &spec, &UcOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        check_mle(&fit, case, name);
        let best = &case["best"];
        let beta: Vec<f64> = fit
            .regression_coefficients
            .iter()
            .map(|c| *c.smoothed.last().unwrap())
            .collect();
        for (a, e) in beta
            .iter()
            .zip(as_f64_vec(&best["regression_coefficients"]))
        {
            assert!(
                (a - e).abs() < 2e-3 * e.abs().max(0.1),
                "{name}: beta {a} vs {e}"
            );
        }
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
    let diffuse_spec = UcSpec {
        cycle_init: CycleInit::Diffuse,
        ..spec.clone()
    };
    let mut sweep = Vec::new();
    for s in fx["stationary_cycle_lambda_sweep"].as_array().unwrap() {
        let opts = UcOptions {
            fixed_params: Some(as_f64_vec(&s["params"])),
            ..UcOptions::default()
        };
        let fit = unobserved_components(&y, &spec, &opts).unwrap();
        assert_rel_close(
            fit.loglik,
            s["stationary"].as_f64().unwrap(),
            TOL_FIXED,
            &format!("lambda {}", s["frequency"]),
        );
        let diffuse = unobserved_components(&y, &diffuse_spec, &opts).unwrap();
        sweep.push((s["frequency"].as_f64().unwrap(), fit.loglik, diffuse.loglik));
    }
    // The same sweep under the diffuse cycle keeps climbing toward the
    // singularity (measured: -258.509, -253.968, -251.545 here; statsmodels'
    // absolute 1e-10 tolerance on F_inf clips its own diffuse numbers at
    // -258.509, -253.642, -253.059), while the stationary one is flat. The
    // climb is at most the `ln 100` per two decades of the one diffuse
    // element that resolves the weakly observed cycle state (less here,
    // where that direction is also nearly collinear with the smooth trend
    // and double precision loses part of it); more would be annihilated
    // diffuse mass taken for a diffuse element, which is what the filter's
    // pre-0.11.0 diffuse-element test did at 1e-4 (-240.094, 13.4 points
    // up in one window).
    eprintln!("GDP lambda sweep (frequency, stationary, diffuse): {sweep:?}");
    for w in sweep.windows(2) {
        let climb = w[1].2 - w[0].2;
        assert!(
            climb > 0.0 && climb < 100f64.ln() + 1.0,
            "diffuse climb {climb}: {sweep:?}"
        );
    }
    assert!(sweep[2].2 - sweep[0].2 > 5.0);
    assert!((sweep[2].1 - sweep[0].1).abs() < 0.2);
}

// ------------------------------------------ (d) long exact-diffuse periods

/// The shared exact-diffuse filter on diffuse periods where a *genuine*
/// diffuse element arrives late, against statsmodels at fixed parameters:
/// the same calendar month missing in consecutive years keeps a seasonal-12
/// model diffuse for 27 (`t = 2, 14` missing) or 39 (`t = 2, 14, 26`)
/// observations — fixed and stochastic seasonal, with an AR(1), with a
/// regressor in the state — and two N(0, 1)-regressor designs with no
/// missing value at all (`default_rng(207)`: 15; `default_rng(69)`: 13).
///
/// Before 0.11.0 the diffuse-element test measured `F_inf` against a
/// magnitude bound carried as `|T| P_mag |T|'`, which the dummy-seasonal
/// transition (`rho(|T|) = 1.9995`) inflates about fourfold per period:
/// the genuine `F_inf = 1.091` at `t = 26` met a threshold of `203.8`
/// (`1e-14` times a bound of `2.0e16`) and was refused, so `P_inf` never
/// washed out. The seven cases with missing months reported
/// `nobs_diffuse = 120` with log-likelihoods 0.64 to 2.87 too low, the
/// seed-207 case 16 diffuse observations instead of 15 (`2.0e-4` low),
/// and the seed-69 case was refused by the units re-check.
#[test]
fn long_diffuse_periods_match_statsmodels() {
    let fx = load_fixture("uc_extensions.json");
    let cases = fx["long_diffuse"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let fit = fixed_fit(&fx, case);
        check_fixed_block(&fit, case, &format!("long_diffuse/{name}"));
        // The diffuse period ends inside the sample, so the standardized
        // residuals after it are finite (they were NaN everywhere).
        assert!(
            fit.nobs_diffuse < fit.nobs,
            "{name}: the diffuse period ran to the end"
        );
        assert!(fit.std_resid[fit.nobs_diffuse..]
            .iter()
            .zip(&fit.resid[fit.nobs_diffuse..])
            .all(|(s, r)| s.is_finite() || r.is_nan()));
    }
    let nd = |n: &str| {
        cases
            .iter()
            .find(|c| c["name"] == n)
            .map(|c| c["nobs_diffuse"].as_u64().unwrap())
            .unwrap()
    };
    assert_eq!(nd("llevel_seasonal12_fixed_nan2_14"), 27);
    assert_eq!(nd("llevel_seasonal12_nan2_14_26"), 39);
    assert_eq!(nd("seed207_lltrend_seasonal12_state_regression"), 15);
    assert_eq!(nd("seed69_llevel_seasonal12_state_regression"), 13);
}

/// The MLE of the stochastic-seasonal `[2, 14]` case reaches the better of
/// statsmodels' L-BFGS and the SciPy re-optimization (the seasonal
/// variance at its zero boundary). Before the fix the optimizer climbed a
/// different surface — the wrong exact-diffuse likelihood — and reported
/// `-180.97` with `nobs_diffuse = 120` against the reference `-178.84`.
#[test]
fn long_diffuse_mle_reaches_the_better_reference_optimum() {
    let fx = load_fixture("uc_extensions.json");
    for case in fx["long_diffuse_mle"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = series(&fx, case);
        let spec = spec_from_json(&fx, &case["spec"]);
        let fit = unobserved_components(&y, &spec, &UcOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(fit.estimated);
        check_mle(&fit, case, &format!("long_diffuse_mle/{name}"));
    }
}
