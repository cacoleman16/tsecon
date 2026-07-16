//! Golden-value tests against the statsmodels fixture (var.json): a
//! VAR(2) with constant on 100 * dlog US macro data (gdp, cons, inv) —
//! coefficients, residual covariance, log-likelihood, information
//! criteria, stability roots, lag-order selection, Granger causality,
//! impulse responses, FEVD, and forecast intervals.

mod common;

use common::{as_mat, assert_mat_close, assert_rel_close, load_fixture};
use tsecon_var::{select_order, Trend, VarSpec};

fn fitted() -> (serde_json::Value, tsecon_var::VarResults) {
    let fx = load_fixture("var.json");
    let data = as_mat(&fx["data_100dlog_gdp_cons_inv"]);
    let spec = VarSpec::new(2, Trend::Constant).unwrap();
    let res = spec.fit(data.as_ref()).unwrap();
    (fx, res)
}

/// OLS coefficients (statsmodels `params` layout), the df-adjusted
/// residual covariance, log-likelihood, and all four information
/// criteria match statsmodels `VAR(data).fit(2, trend="c")` to 1e-8.
#[test]
fn golden_estimation() {
    let (fx, res) = fitted();
    let block = &fx["var2c"];

    assert_eq!(res.neqs, 3);
    assert_eq!(res.nobs, 200);
    assert_eq!(res.df_model, 7);
    assert_eq!(res.df_resid, 193);

    assert_mat_close(&res.params, &block["params"], 1e-8, "params");
    assert_mat_close(&res.sigma_u, &block["sigma_u"], 1e-8, "sigma_u");
    assert_rel_close(res.llf, block["llf"].as_f64().unwrap(), 1e-8, "llf");
    assert_rel_close(res.aic, block["aic"].as_f64().unwrap(), 1e-8, "aic");
    assert_rel_close(res.bic, block["bic"].as_f64().unwrap(), 1e-8, "bic");
    assert_rel_close(res.hqic, block["hqic"].as_f64().unwrap(), 1e-8, "hqic");
    assert_rel_close(res.fpe, block["fpe"].as_f64().unwrap(), 1e-8, "fpe");

    // Intercept and lag matrices are views into params.
    for j in 0..3 {
        assert_rel_close(
            res.intercept[j],
            res.params[(0, j)],
            0.0,
            &format!("intercept[{j}]"),
        );
    }
    assert_eq!(res.coefs.len(), 2);
    assert_rel_close(res.coefs[0][(2, 1)], res.params[(2, 2)], 0.0, "A_1[2,1]");
}

/// The largest modulus among the characteristic roots (statsmodels
/// `VARResults.roots`) matches the fixture, and the fitted growth-rate
/// system is stable (all companion eigenvalues inside the unit circle).
#[test]
fn golden_stability_roots() {
    let (fx, res) = fitted();
    let roots = res.roots_moduli().unwrap();
    assert_eq!(roots.len(), 6);
    assert_rel_close(
        roots[0],
        fx["var2c"]["stable_max_root"].as_f64().unwrap(),
        1e-8,
        "max root modulus",
    );
    // Stability: every root outside the unit circle.
    assert!(roots.iter().all(|&r| r > 1.0));
    assert!(res.is_stable().unwrap());
}

/// Lag-order selection at maxlags = 8 with a constant reproduces the
/// statsmodels `select_order` picks (common estimation sample across
/// candidate orders) for all four criteria.
#[test]
fn golden_lag_selection() {
    let fx = load_fixture("var.json");
    let data = as_mat(&fx["data_100dlog_gdp_cons_inv"]);
    let sel = select_order(data.as_ref(), 8, Trend::Constant).unwrap();
    let block = &fx["lag_selection_maxlags_8"];
    assert_eq!(sel.aic, block["aic"].as_u64().unwrap() as usize, "aic pick");
    assert_eq!(sel.bic, block["bic"].as_u64().unwrap() as usize, "bic pick");
    assert_eq!(
        sel.hqic,
        block["hqic"].as_u64().unwrap() as usize,
        "hqic pick"
    );
    assert_eq!(sel.fpe, block["fpe"].as_u64().unwrap() as usize, "fpe pick");
    // Candidate table covers p = 0..=8 (constant included).
    assert_eq!(sel.candidates.len(), 9);
    assert_eq!(sel.candidates[0].lags, 0);
}

/// Granger-causality F test that consumption growth does not cause GDP
/// growth: statistic, p-value (relative accuracy — it is ~7e-8), and
/// both degrees of freedom match statsmodels
/// `test_causality("y1", ["y2"], kind="f")`.
#[test]
fn golden_granger_causality() {
    let (fx, res) = fitted();
    let block = &fx["granger_cons_causes_gdp"];
    let test = res.test_causality(&[0], &[1]).unwrap();

    assert_rel_close(
        test.statistic,
        block["stat"].as_f64().unwrap(),
        1e-8,
        "F statistic",
    );
    let p_expected = block["pvalue"].as_f64().unwrap();
    assert!(
        (test.pvalue - p_expected).abs() <= 1e-6 * p_expected,
        "pvalue: {} vs {p_expected}",
        test.pvalue
    );
    assert_eq!(test.df_num, block["df"][0].as_u64().unwrap() as usize);
    assert_eq!(test.df_den, block["df"][1].as_u64().unwrap() as usize);
}

/// Non-orthogonalized and Cholesky-orthogonalized impulse responses to
/// horizon 10 match statsmodels `irf(10)` elementwise to 1e-8.
#[test]
fn golden_irf() {
    let (fx, res) = fitted();
    let irf = res.irf(10).unwrap();
    assert_eq!(irf.irfs.len(), 11);
    assert_eq!(irf.orth_irfs.len(), 11);
    let e_nonorth = fx["irf_nonorth_h10"].as_array().unwrap();
    let e_orth = fx["irf_orth_h10"].as_array().unwrap();
    for h in 0..=10 {
        assert_mat_close(&irf.irfs[h], &e_nonorth[h], 1e-8, &format!("irfs[{h}]"));
        assert_mat_close(
            &irf.orth_irfs[h],
            &e_orth[h],
            1e-8,
            &format!("orth_irfs[{h}]"),
        );
    }
}

/// The forecast-error variance decomposition to 10 periods matches
/// statsmodels `fevd(10).decomp` elementwise to 1e-8.
#[test]
fn golden_fevd() {
    let (fx, res) = fitted();
    let fevd = res.fevd(10).unwrap();
    let expected = fx["fevd_h10"].as_array().unwrap();
    assert_eq!(fevd.decomp.len(), 3);
    for (i, decomp) in fevd.decomp.iter().enumerate() {
        assert_mat_close(decomp, &expected[i], 1e-8, &format!("fevd[{i}]"));
    }
}

/// Iterated point forecasts and 95% asymptotic intervals to 8 steps
/// match statsmodels `forecast_interval(data[-2:], 8)` to 1e-8 (the
/// intervals carry innovation uncertainty only — the Psi-weight
/// accumulation of sigma_u, no parameter-uncertainty term).
#[test]
fn golden_forecast_interval() {
    let (fx, res) = fitted();
    let fc = res.forecast_interval(8, 0.05).unwrap();
    let block = &fx["forecast_8"];
    assert_mat_close(&fc.point, &block["point"], 1e-8, "point");
    assert_mat_close(&fc.lower, &block["lower95"], 1e-8, "lower95");
    assert_mat_close(&fc.upper, &block["upper95"], 1e-8, "upper95");
}
