//! Golden tests for the VAR forecast-error covariance with and without
//! Lütkepohl's parameter-uncertainty term, against
//! `fixtures/var_forecast_cov.json` (generator
//! `fixtures/generate_var_forecast_cov_fixtures.py`).
//!
//! Reference: statsmodels 0.15.0 on the `var.json` system (100 x dlog of real
//! GDP, consumption and investment) — `VARResults.mse(steps)` for the
//! default covariance and `forecast_cov(steps, method="auto")` for
//! `ForecastOptions::with_param_uncertainty(true)` under `trend = "c"`. Under
//! `trend = "n"` statsmodels' public `"auto"` silently drops the term (its
//! `k_exog == 1` gate), so that golden is `mse + _omega_forc_cov / nobs`; the
//! generator asserts the quirk still holds and re-derives every `Omega(h)` in
//! NumPy (agreement <= 1e-15 relative at generation).
//!
//! Cases VAR(2)c, VAR(2)n, VAR(1)c, VAR(4)c at steps 1, 8 and 12 (the fixture
//! stores 12; shorter requests must reproduce its prefix). Everything at
//! 1e-10 relative, the `h = 1` identity `cov(1) = (1 + m/T) Sigma_u` at
//! 1e-14.

mod common;

use common::{as_mat, load_fixture};
use serde_json::Value;
use tsecon_linalg::faer::Mat;
use tsecon_var::{ForecastOptions, Trend, VarResults, VarSpec};

const TOL: f64 = 1e-10;

fn fit(fx_var: &Value, lags: usize, trend: &str) -> VarResults {
    let data = as_mat(&fx_var["data_100dlog_gdp_cons_inv"]);
    let trend = match trend {
        "c" => Trend::Constant,
        "n" => Trend::None,
        other => panic!("unexpected trend {other}"),
    };
    VarSpec::new(lags, trend)
        .unwrap()
        .fit(data.as_ref())
        .unwrap()
}

/// Max over a cube of `|a - e| / max|e|` (relative to the cube's scale, so a
/// near-zero off-diagonal does not demand absolute 1e-10 digits it cannot
/// carry), plus the per-entry check against `tol`.
fn assert_cube_close(actual: &[Mat<f64>], expected: &Value, tol: f64, what: &str) -> f64 {
    let rows = expected.as_array().expect("cube");
    assert!(actual.len() <= rows.len(), "{what}: too many horizons");
    let mut worst = 0.0_f64;
    for (h, a) in actual.iter().enumerate() {
        let e = as_mat(&rows[h]);
        let mut scale = 0.0_f64;
        for j in 0..e.ncols() {
            for i in 0..e.nrows() {
                scale = scale.max(e[(i, j)].abs());
            }
        }
        for j in 0..e.ncols() {
            for i in 0..e.nrows() {
                let err = (a[(i, j)] - e[(i, j)]).abs() / scale;
                worst = worst.max(err);
                assert!(
                    err <= tol,
                    "{what}[h={h}][{i},{j}]: {} vs {} (rel {err:e})",
                    a[(i, j)],
                    e[(i, j)]
                );
            }
        }
    }
    worst
}

fn assert_paths_close(actual: &Mat<f64>, expected: &Value, tol: f64, what: &str) -> f64 {
    let e = as_mat(expected);
    let mut worst = 0.0_f64;
    for h in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let scale = e[(h, j)].abs().max(1.0);
            let err = (actual[(h, j)] - e[(h, j)]).abs() / scale;
            worst = worst.max(err);
            assert!(
                err <= tol,
                "{what}[{h},{j}]: {} vs {} (rel {err:e})",
                actual[(h, j)],
                e[(h, j)]
            );
        }
    }
    worst
}

#[test]
fn forecast_covariances_match_statsmodels_with_and_without_the_omega_term() {
    let fx = load_fixture("var_forecast_cov.json");
    let fx_var = load_fixture("var.json");
    let cases = fx["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 4);
    let mut worst = [0.0_f64; 3];
    let with = ForecastOptions::new().with_param_uncertainty(true);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let lags = case["lags"].as_u64().expect("lags") as usize;
        let res = fit(&fx_var, lags, case["trend"].as_str().expect("trend"));
        assert_eq!(res.nobs as u64, case["nobs"].as_u64().expect("nobs"));
        assert_eq!(res.df_model as u64, case["df_model"].as_u64().expect("m"));
        for steps in [1_usize, 8, 12] {
            let ctx = format!("{name} steps={steps}");
            let mse = res
                .forecast_cov_with(steps, ForecastOptions::new())
                .unwrap();
            assert_eq!(mse.len(), steps);
            worst[0] = worst[0].max(assert_cube_close(
                &mse,
                &case["mse"],
                TOL,
                &format!("{ctx} mse"),
            ));
            let omega = res.forecast_param_uncertainty(steps).unwrap();
            worst[1] = worst[1].max(assert_cube_close(
                &omega,
                &case["omega"],
                TOL,
                &format!("{ctx} omega"),
            ));
            let cov = res.forecast_cov_with(steps, with).unwrap();
            worst[2] = worst[2].max(assert_cube_close(
                &cov,
                &case["cov_param"],
                TOL,
                &format!("{ctx} cov"),
            ));
        }
    }
    eprintln!(
        "worst rel dev: mse {:.1e}, omega {:.1e}, cov {:.1e}",
        worst[0], worst[1], worst[2]
    );
}

#[test]
fn forecast_bands_match_statsmodels_at_both_levels() {
    let fx = load_fixture("var_forecast_cov.json");
    let fx_var = load_fixture("var.json");
    let with = ForecastOptions::new().with_param_uncertainty(true);
    for case in fx["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let lags = case["lags"].as_u64().expect("lags") as usize;
        let res = fit(&fx_var, lags, case["trend"].as_str().expect("trend"));
        let bands = case["bands"].as_object().expect("bands");
        for (alpha, band) in bands {
            let alpha: f64 = alpha.parse().expect("alpha key");
            let plain = res
                .forecast_interval_with(12, alpha, ForecastOptions::new())
                .unwrap();
            let param = res.forecast_interval_with(12, alpha, with).unwrap();
            let ctx = format!("{name} alpha={alpha}");
            assert_paths_close(&plain.point, &case["point"], TOL, &format!("{ctx} point"));
            assert_paths_close(&param.point, &case["point"], TOL, &format!("{ctx} point"));
            assert_paths_close(&plain.lower, &band["lower_mse"], TOL, &format!("{ctx} lo"));
            assert_paths_close(&plain.upper, &band["upper_mse"], TOL, &format!("{ctx} up"));
            assert_paths_close(
                &param.lower,
                &band["lower_param"],
                TOL,
                &format!("{ctx} plo"),
            );
            assert_paths_close(
                &param.upper,
                &band["upper_param"],
                TOL,
                &format!("{ctx} pup"),
            );
            // The interval carries the covariance it was built from.
            assert_cube_close(&param.cov, &case["cov_param"], TOL, &format!("{ctx} cov"));
            assert_cube_close(&plain.cov, &case["mse"], TOL, &format!("{ctx} mse"));
        }
    }
}

/// `Omega(1) = m Sigma_u`, so `cov(1) = (1 + m/T) Sigma_u` — the identity
/// that needs `Q_0 = I` exactly (statsmodels special-cases `h = 1` the same
/// way). Pinned at 1e-14 relative to the largest entry of `Sigma_u`.
#[test]
fn one_step_covariance_is_one_plus_m_over_t_times_sigma_u() {
    let fx_var = load_fixture("var.json");
    for (lags, trend) in [(2, "c"), (2, "n"), (1, "c"), (4, "c")] {
        let res = fit(&fx_var, lags, trend);
        let m = res.df_model as f64;
        let t = res.nobs as f64;
        let cov = res
            .forecast_cov_with(1, ForecastOptions::new().with_param_uncertainty(true))
            .unwrap();
        let omega = res.forecast_param_uncertainty(1).unwrap();
        let k = res.neqs;
        let mut scale = 0.0_f64;
        for j in 0..k {
            for i in 0..k {
                scale = scale.max(res.sigma_u[(i, j)].abs());
            }
        }
        for j in 0..k {
            for i in 0..k {
                let want = (1.0 + m / t) * res.sigma_u[(i, j)];
                assert!(
                    (cov[0][(i, j)] - want).abs() <= 1e-14 * scale,
                    "VAR({lags}){trend} [{i},{j}]: {} vs {want}",
                    cov[0][(i, j)]
                );
                let want_om = m * res.sigma_u[(i, j)];
                assert!(
                    (omega[0][(i, j)] - want_om).abs() <= 1e-14 * scale,
                    "Omega(1) [{i},{j}]"
                );
            }
        }
    }
}
