//! Golden tests for conditional-sum-of-squares estimation
//! ([`ArimaSpec::fit_css`], [`ArimaSpec::css_residuals`]) against
//! `fixtures/arima_css.json`.
//!
//! * **Leg A — the recursion, optimizer-free, third-party.** statsforecast
//!   2.1.1's `arima_css` (its C++ port of R's `ARIMA_CSS`) evaluated at fixed
//!   parameters — the leg-C optimum and a perturbed point — for ARMA(1,1)+c,
//!   SARIMA(1,0,0)(1,0,0)_4+c, MA(1) and ARIMA(1,1,1): residuals, SSR,
//!   `sigma2` and the residual count at 1e-12. R's mean form is mapped to the
//!   crate's intercept by `mu = c / (1 - sum phi*)`.
//! * **Leg B — AR(2)+c.** statsmodels `AutoReg(trend="c")`: OLS is the CSS
//!   optimum, so params are pinned at 1e-6 and `sigma2`/`loglik`/AIC/BIC at
//!   1e-9.
//! * **Leg C — the optimum.** It is optimizer-defined (no closed form): the
//!   reference is SciPy's Levenberg-Marquardt at 1e-15 tolerances (BFGS
//!   agrees to <= 1.2e-8). Gates: params 1e-6 (achieved <= 5.5e-9, see
//!   `achieved_parameter_precision`), `loglik`/AIC/BIC 1e-9 (the optimum is
//!   flat; achieved 5e-16), `bse` 1e-4 relative against `approx_hess3` of
//!   the same conditional likelihood with `sigma2` free (achieved 2.3e-7).

mod common;

use common::{as_vec, load_fixture};
use serde_json::Value;
use tsecon_arima::{ArimaSpec, EstimationMethod};

const FIXTURE: &str = "arima_css.json";

fn spec_of(c: &Value) -> ArimaSpec {
    let order: Vec<usize> = c["order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut spec = ArimaSpec::new(order[0], order[1], order[2])
        .unwrap()
        .with_constant(c["constant"].as_bool().unwrap());
    if let Some(s) = c["seasonal"].as_array() {
        let s: Vec<usize> = s.iter().map(|v| v.as_u64().unwrap() as usize).collect();
        spec = spec.seasonal(s[0], s[1], s[2], s[3]).unwrap();
    }
    spec
}

fn rel(a: f64, e: f64) -> f64 {
    (a - e).abs() / e.abs().max(1.0)
}

#[test]
fn css_recursion_matches_statsforecast_arima_css_at_fixed_parameters() {
    let fx = load_fixture(FIXTURE);
    let mut worst = 0.0f64;
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let spec = spec_of(c);
        let y = as_vec(&c["y"]);
        for leg in c["fixed_point_statsforecast"].as_array().unwrap() {
            let point = leg["point"].as_str().unwrap();
            let params = as_vec(&leg["mean_params"]);
            let ev = spec.css_residuals(&y, &params).unwrap();
            let expected = as_vec(&leg["residuals"]);
            assert_eq!(
                ev.nobs,
                leg["nobs"].as_u64().unwrap() as usize,
                "{name}@{point}"
            );
            assert_eq!(ev.residuals.len(), expected.len(), "{name}@{point}");
            for (t, (a, e)) in ev.residuals.iter().zip(&expected).enumerate() {
                let d = rel(*a, *e);
                worst = worst.max(d);
                assert!(d <= 1e-12, "{name}@{point} residual {t}: {a} vs {e}");
            }
            let s2 = leg["sigma2"].as_f64().unwrap();
            let d = (ev.sigma2 - s2).abs() / s2;
            worst = worst.max(d);
            assert!(d <= 1e-12, "{name}@{point} sigma2 {} vs {s2}", ev.sigma2);
            let ssr = leg["ssr"].as_f64().unwrap();
            assert!((ev.ssr - ssr).abs() / ssr <= 1e-12, "{name}@{point} ssr");
            // R's ncond adds d + D s to the crate's p + s P.
            let (d_, sd, s) = (spec.d(), spec.seasonal_d(), spec.period());
            assert_eq!(
                ev.n_cond + d_ + sd * s,
                leg["ncond_r"].as_u64().unwrap() as usize
            );
        }
    }
    eprintln!("leg A worst relative deviation {worst:e}");
    // Measured 1.6e-15 on the committed fixture.
    assert!(worst < 1e-13, "{worst:e}");
}

#[test]
fn css_ar2_equals_autoreg_ols() {
    let fx = load_fixture(FIXTURE);
    let c = &fx["ar_autoreg"];
    let y = as_vec(&c["y"]);
    let spec = ArimaSpec::new(2, 0, 0).unwrap().with_constant(true);
    let r = spec.fit_css(&y).unwrap();
    assert_eq!(r.method, EstimationMethod::Css);
    assert_eq!(r.nobs, c["nobs"].as_u64().unwrap() as usize);
    let expected = as_vec(&c["params"]);
    let mut worst = 0.0f64;
    for (i, (a, e)) in r.params().iter().zip(&expected).enumerate() {
        let d = rel(*a, *e);
        worst = worst.max(d);
        assert!(d <= 1e-6, "param {i}: {a} vs {e}");
    }
    eprintln!("AR(2) params worst relative deviation {worst:e}");
    for (key, got) in [
        ("sigma2", r.sigma2()),
        ("loglik", r.loglik),
        ("aic", r.aic),
        ("bic", r.bic),
    ] {
        let e = c[key].as_f64().unwrap();
        assert!(rel(got, e) <= 1e-9, "{key}: {got} vs {e}");
    }
}

/// Max relative parameter deviation of every leg-C case.
fn leg_c_param_devs() -> Vec<(String, f64)> {
    let fx = load_fixture(FIXTURE);
    let mut out = Vec::new();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap().to_owned();
        let spec = spec_of(c);
        let y = as_vec(&c["y"]);
        let r = spec.fit_css(&y).unwrap();
        let expected = as_vec(&c["optimum"]["params"]);
        let worst = r
            .params()
            .iter()
            .zip(&expected)
            .map(|(a, e)| rel(*a, *e))
            .fold(0.0f64, f64::max);
        out.push((name, worst));
    }
    out
}

#[test]
fn css_optimum_matches_the_levenberg_marquardt_reference() {
    let fx = load_fixture(FIXTURE);
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let spec = spec_of(c);
        let y = as_vec(&c["y"]);
        let r = spec.fit_css(&y).unwrap();
        let o = &c["optimum"];
        assert!(r.converged, "{name}: optimizer did not certify convergence");
        assert_eq!(r.nobs, o["nobs"].as_u64().unwrap() as usize, "{name} nobs");
        let names: Vec<String> = c["param_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        assert_eq!(r.param_names(), names.as_slice(), "{name} names");
        for (i, (a, e)) in r.params().iter().zip(&as_vec(&o["params"])).enumerate() {
            assert!(rel(*a, *e) <= 1e-6, "{name} param {i}: {a} vs {e}");
        }
        for (key, got) in [("loglik", r.loglik), ("aic", r.aic), ("bic", r.bic)] {
            let e = o[key].as_f64().unwrap();
            assert!(rel(got, e) <= 1e-9, "{name} {key}: {got} vs {e}");
        }
        let bse = r.bse().unwrap();
        let mut worst_bse = 0.0f64;
        for (i, (a, e)) in bse.iter().zip(&as_vec(&o["bse"])).enumerate() {
            let d = (a - e).abs() / e.abs();
            worst_bse = worst_bse.max(d);
            assert!(d <= 1e-4, "{name} bse {i}: {a} vs {e} (rel {d:e})");
        }
        let worst_ll = [("loglik", r.loglik), ("aic", r.aic), ("bic", r.bic)]
            .iter()
            .map(|(k, v)| rel(*v, o[*k].as_f64().unwrap()))
            .fold(0.0f64, f64::max);
        eprintln!("{name}: bse worst rel {worst_bse:e}, loglik/aic/bic worst rel {worst_ll:e}");
        // The fit's own loglik is the recursion's at its optimum, bitwise:
        // one code path for the objective and for css_residuals.
        let mean = &r.params()[..r.params().len() - 1];
        let ev = spec.css_residuals(&y, mean).unwrap();
        assert_eq!(ev.loglik.to_bits(), r.loglik.to_bits(), "{name}");
        assert_eq!(ev.sigma2.to_bits(), r.sigma2().to_bits(), "{name}");
    }
}

#[test]
fn achieved_parameter_precision() {
    // The measurement behind the 1e-6 leg-C gate (the exact-MLE parity gate
    // is 1e-4; the CSS optimum is pinned two orders tighter). Measured on
    // the committed fixture: 5.5e-9 (ARMA(1,1)+c), 1.6e-9 (SARIMA), 1.1e-10
    // (MA(1)), 7.8e-10 (ARIMA(1,1,1)) — inside the LM-vs-BFGS spread of the
    // reference itself (<= 1.2e-8).
    for (name, worst) in leg_c_param_devs() {
        eprintln!("{name}: worst relative parameter deviation vs LM {worst:e}");
        assert!(worst <= 1e-6, "{name}: {worst:e}");
    }
}

#[test]
fn css_residuals_refuses_malformed_input() {
    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let y: Vec<f64> = (0..40).map(|t| ((t * 7919) % 13) as f64).collect();
    // sigma2 is not a CSS mean parameter.
    assert!(spec.css_residuals(&y, &[0.1, 0.2, 0.3, 1.0]).is_err());
    assert!(spec.css_residuals(&y, &[0.1, f64::NAN, 0.3]).is_err());
    assert!(spec.css_residuals(&y[..1], &[0.1, 0.2, 0.3]).is_err());
    let mut bad = y.clone();
    bad[3] = f64::INFINITY;
    assert!(spec.css_residuals(&bad, &[0.1, 0.2, 0.3]).is_err());
    // An explosive point overflows the recursion and is refused, not NaN.
    let long: Vec<f64> = (0..4000).map(|t| (t as f64).sin()).collect();
    assert!(spec.css_residuals(&long, &[0.0, 0.5, 50.0]).is_err());
}
