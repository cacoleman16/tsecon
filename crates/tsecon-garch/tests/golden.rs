//! Golden-value tests against the `arch` package fixture
//! (`fixtures/garch.json`, arch 8.0.0): fixed-parameter log-likelihoods,
//! conditional volatilities, QMLE fits, and robust standard errors for
//! GARCH(1,1)/GJR(1,1,1)/EGARCH(1,1,1) under zero/constant means and
//! normal/standardized-t innovations.

mod common;

use common::{
    as_f64_vec, assert_abs_close, assert_rel_close, find_case, load_fixture, params_by_name,
    spec_for,
};
use tsecon_garch::GarchModel;

const CASES: [&str; 5] = [
    "garch11_zero_normal",
    "garch11_const_normal",
    "gjr111_zero_normal",
    "egarch111_zero_normal",
    "garch11_zero_t",
];

/// Fixed-parameter log-likelihoods match `arch` to 1e-8 relative — this
/// pins the variance recursion, the decay-weighted backcast (computed at
/// the mean model's starting values), and every likelihood constant.
/// (Measured agreement on macOS arm64 is <= 5e-16 relative.)
#[test]
fn golden_loglike_fixed() {
    let fx = load_fixture("garch.json");
    let y = as_f64_vec(&fx["returns"]);
    for name in CASES {
        let case = find_case(&fx, name);
        let spec = spec_for(name);
        let model = GarchModel::new(&y, spec).unwrap();
        let params = as_f64_vec(&case["fixed_params"]);
        let ll = model.loglike(&params).unwrap();
        let expected = case["loglike_fixed"].as_f64().unwrap();
        assert_rel_close(ll, expected, 1e-8, &format!("{name}: loglike_fixed"));
    }
}

/// Conditional volatility at `arch`'s fitted parameters matches the
/// fixture head/tail to 1e-6, and the log-likelihood at those parameters
/// reproduces `fit_loglike` to 1e-8 relative.
#[test]
fn golden_conditional_volatility_and_loglike_at_fit() {
    let fx = load_fixture("garch.json");
    let y = as_f64_vec(&fx["returns"]);
    for name in CASES {
        let case = find_case(&fx, name);
        let spec = spec_for(name);
        let model = GarchModel::new(&y, spec).unwrap();
        let params = params_by_name(case, "fit_params", &spec);

        let vol: Vec<f64> = model
            .conditional_variance(&params)
            .unwrap()
            .iter()
            .map(|s| s.sqrt())
            .collect();
        let head = as_f64_vec(&case["conditional_volatility_first5"]);
        let tail = as_f64_vec(&case["conditional_volatility_last5"]);
        for (i, &e) in head.iter().enumerate() {
            assert_abs_close(vol[i], e, 1e-6, &format!("{name}: cond vol head[{i}]"));
        }
        let n = vol.len();
        for (i, &e) in tail.iter().enumerate() {
            assert_abs_close(
                vol[n - 5 + i],
                e,
                1e-6,
                &format!("{name}: cond vol tail[{i}]"),
            );
        }

        let ll = model.loglike(&params).unwrap();
        let expected = case["fit_loglike"].as_f64().unwrap();
        assert_rel_close(ll, expected, 1e-8, &format!("{name}: loglike at fit"));
    }
}

/// Robust (Bollerslev-Wooldridge) standard errors at `arch`'s fitted
/// parameters match the fixture within 5e-3 relative.
///
/// Fixture honesty notes, verified against a NumPy replication of
/// `arch.compute_param_cov`:
///
/// * the fixture's `fit_bse_mle` is byte-identical to `fit_bse_robust` in
///   every case — the generator stored `arch`'s default (robust) standard
///   errors twice, so the robust sandwich is the only covariance the
///   fixture actually pins (asserted below to document the artifact);
///   true inverse-Hessian MLE standard errors differ from these values by
///   9-14% on the normal cases and are sanity-checked separately.
/// * the Student-t case's `nu = 293.9` sits on a numerically flat ridge
///   (the data are simulated normal): its "standard error" moves from 26
///   to 1600 as the Hessian step varies around statsmodels' default, so
///   `nu`'s tolerance is 0.10 relative (measured agreement 5.6e-2) rather
///   than 5e-3. The remaining parameters meet 5e-3 (measured <= 4e-5).
#[test]
fn golden_robust_standard_errors() {
    let fx = load_fixture("garch.json");
    let y = as_f64_vec(&fx["returns"]);
    for name in CASES {
        let case = find_case(&fx, name);
        let spec = spec_for(name);
        let model = GarchModel::new(&y, spec).unwrap();
        let params = params_by_name(case, "fit_params", &spec);
        let bse_robust = params_by_name(case, "fit_bse_robust", &spec);
        let bse_mle = params_by_name(case, "fit_bse_mle", &spec);

        // Document the fixture artifact: both columns hold arch's default
        // (robust) standard errors.
        for (i, (m, r)) in bse_mle.iter().zip(&bse_robust).enumerate() {
            assert!(
                m == r,
                "{name}: fixture bse columns diverge at {i} ({m} vs {r}); \
                 the generator artifact assumption no longer holds"
            );
        }

        let se = model.standard_errors(&params).unwrap();
        let names = spec.param_names();
        for (i, nm) in names.iter().enumerate() {
            let tol = if name == "garch11_zero_t" && nm == "nu" {
                0.10
            } else {
                5e-3
            };
            assert_rel_close(
                se.robust[i] / bse_robust[i],
                1.0,
                tol,
                &format!("{name}: robust se[{nm}]"),
            );
            // MLE standard errors: finite, positive, same scale as robust
            // (information-matrix equality holds approximately under these
            // correctly specified simulated data; measured gap 9-14% on
            // the normal cases).
            assert!(
                se.mle[i].is_finite() && se.mle[i] > 0.0,
                "{name}: mle se[{nm}] = {}",
                se.mle[i]
            );
            if name != "garch11_zero_t" {
                assert_rel_close(
                    se.mle[i] / se.robust[i],
                    1.0,
                    0.30,
                    &format!("{name}: mle-vs-robust se[{nm}]"),
                );
            }
        }
    }
}

/// Full QMLE fits: the optimized log-likelihood is within 1e-6 absolute of
/// `arch`'s (or better), and the parameters match to 1e-3 relative on the
/// normal-innovation cases.
///
/// The Student-t case is special: its likelihood is monotonically
/// increasing in `nu` on this (simulated normal) data, so the true optimum
/// sits at the `nu` upper bound (500, `arch`'s own bound) while `arch`'s
/// SLSQP stopped early on the flat ridge at `nu = 293.9` with a
/// log-likelihood 0.11 *below* its own normal-case fit. We therefore
/// assert a log-likelihood at least as good as the fixture's and variance
/// parameters within 2e-2 relative (their drift along the ridge between
/// `nu = 294` and `nu = 500`), and `nu` large.
#[test]
fn golden_fit() {
    let fx = load_fixture("garch.json");
    let y = as_f64_vec(&fx["returns"]);
    for name in CASES {
        let case = find_case(&fx, name);
        let spec = spec_for(name);
        let model = GarchModel::new(&y, spec).unwrap();
        let res = model.fit().unwrap();
        let expected_ll = case["fit_loglike"].as_f64().unwrap();
        assert!(
            res.loglik >= expected_ll - 1e-6,
            "{name}: fit loglik {} vs arch {expected_ll}",
            res.loglik
        );
        let expected_params = params_by_name(case, "fit_params", &spec);
        let names = spec.param_names();
        for (i, nm) in names.iter().enumerate() {
            if name == "garch11_zero_t" {
                if nm == "nu" {
                    assert!(
                        res.params[i] > 100.0,
                        "{name}: nu = {} should be on the large-nu ridge",
                        res.params[i]
                    );
                } else {
                    assert_rel_close(
                        res.params[i] / expected_params[i],
                        1.0,
                        2e-2,
                        &format!("{name}: fit param {nm}"),
                    );
                }
            } else {
                assert_rel_close(
                    res.params[i] / expected_params[i],
                    1.0,
                    1e-3,
                    &format!("{name}: fit param {nm}"),
                );
            }
        }
        assert_eq!(res.param_names, names, "{name}: param names");
        assert_eq!(res.nobs, y.len());
        // aic/bic consistency with the reported loglik.
        let k = names.len() as f64;
        assert_abs_close(
            res.aic,
            2.0 * k - 2.0 * res.loglik,
            1e-10,
            &format!("{name}: aic"),
        );
        assert_abs_close(
            res.bic,
            k * (y.len() as f64).ln() - 2.0 * res.loglik,
            1e-10,
            &format!("{name}: bic"),
        );
        // Standardized residuals are residuals over conditional volatility.
        for t in [0, y.len() / 2, y.len() - 1] {
            assert_abs_close(
                res.std_residuals[t] * res.conditional_volatility[t],
                res.residuals()[t],
                1e-12,
                &format!("{name}: std resid identity at {t}"),
            );
        }
    }
}
