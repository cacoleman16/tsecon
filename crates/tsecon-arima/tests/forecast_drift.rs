//! The drift-uncertainty forecast correction
//! ([`ArimaResults::forecast_with`]): closed-form anchors, the
//! byte-for-byte invariance of the default path, and a Monte Carlo that
//! reproduces the interval-coverage shortfall and its repair.
//!
//! # The anchor, derived
//!
//! Take ARIMA(0, 1, 0) with a constant on `y_1..y_T`. Simple
//! differencing leaves `n = T - 1` observations
//!
//! ```text
//! x_t = c + eps_t,   eps_t ~ N(0, sigma2)  iid,
//! ```
//!
//! whose exact Gaussian log-likelihood is
//! `-n/2 (ln 2*pi + ln sigma2) - S(c) / (2 sigma2)` with
//! `S(c) = sum_t (x_t - c)^2`. Its maximizers are `c_hat = xbar` and
//! `sigma2_hat = S(c_hat)/n`, and at that point
//!
//! ```text
//! d^2 l / dc^2        = -n / sigma2
//! d^2 l / dc dsigma2  = -n (xbar - c) / sigma2^2 = 0   (at c = xbar)
//! ```
//!
//! so the observed information is block diagonal and
//! `Var(c_hat) = sigma2 / n` exactly — no matrix inversion subtlety, and
//! `l` is a *quadratic* in `c`, so the four-point difference recovers
//! `d^2 l / dc^2` to roundoff rather than to `O(h^2)`.
//!
//! The level forecast is `yhat_{T+h} = y_T + h c_hat`, giving
//! `d yhat_{T+h} / dc = h`, while the cumulated innovation variance is
//! `h sigma2`. The delta method then gives
//!
//! ```text
//! Var(yhat_{T+h}) = h sigma2 + h^2 sigma2 / n
//!                 = sigma2 (h + h^2 / n),
//! se_h            = sigma sqrt(h + h^2 / n),   n = T - 1.
//! ```
//!
//! The parameters-known convention drops the second term entirely and
//! reports `sigma sqrt(h)`. The ratio of the two is `1/sqrt(1 + h/n)`,
//! so a nominally `1 - alpha` band actually covers
//! `2 Phi(z_{1-alpha/2} / sqrt(1 + h/n)) - 1` — 90.2% at `T = 60`,
//! `h = 24`, `alpha = 0.05`. That is the closed form the coverage audit
//! measured (90.3%), and it is asserted below.

mod common;

use common::{assert_rel_close, Lcg};
use tsecon_arima::{ArimaError, ArimaSpec, ForecastOptions};

/// Analytic exact MLE of ARIMA(0, 1, 0) + constant: the mean and the
/// divide-by-n variance of the first differences.
///
/// Using this with `at_params` rather than `fit` puts the Hessian at the
/// *exact* stationary point, so the anchor tests measure the correction
/// itself and not the optimizer's stopping tolerance. `fit` is checked
/// separately, at the looser tolerance its convergence test earns.
fn rw_drift_mle(y: &[f64]) -> (Vec<f64>, usize) {
    let dx: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let n = dx.len();
    let c = dx.iter().sum::<f64>() / n as f64;
    let s2 = dx.iter().map(|v| (v - c) * (v - c)).sum::<f64>() / n as f64;
    (vec![c, s2], n)
}

/// A random walk with drift of length `t_obs`, drawn from an existing
/// generator.
///
/// Taking the generator by reference rather than a seed matters for the
/// Monte Carlo below: `Lcg::new(s)` is a single multiply-add and
/// `uniform()` is a top-bits shift, so seeding per replication with
/// `1000 + rep` makes consecutive replications differ by a fixed additive
/// constant, and their first draws lie on a Weyl-type lattice rather than
/// being independent. One generator threaded through every replication
/// gives the sequence its full period.
fn walk_from(rng: &mut Lcg, t_obs: usize, drift: f64, sd: f64) -> Vec<f64> {
    let mut y = Vec::with_capacity(t_obs);
    let mut level = 0.0;
    y.push(level);
    for _ in 1..t_obs {
        level += drift + sd * rng.gaussian();
        y.push(level);
    }
    y
}

/// A seeded random walk with drift of length `t_obs` — one path, one
/// generator, for the single-path anchor tests.
fn random_walk_with_drift(seed: u64, t_obs: usize, drift: f64, sd: f64) -> Vec<f64> {
    walk_from(&mut Lcg::new(seed), t_obs, drift, sd)
}

/// **The anchor.** For ARIMA(0, 1, 0) with a constant the corrected
/// standard error must equal `sigma sqrt(h + h^2 / n)` with
/// `n = T - 1` — the closed form derived in the module docs and the one
/// the coverage audit verified.
///
/// The default leg is gated at 1e-12: `sigma sqrt(h)` involves no
/// numerical differentiation at all.
///
/// The corrected leg is gated at 5e-8, and that bound is derived rather
/// than tuned. The only inexact input is `Var(c_hat)`, whose relative
/// error is the roundoff floor of the four-point difference,
/// `eps |l| / (4 h_c^2 |l_cc|) ~ 3e-8` (the log-likelihood is exactly
/// quadratic in `c`, so there is no truncation term, and the forecast
/// mean is exactly affine in `c`, so the derivative `= h` is exact). The
/// drift term is a fraction `h / (n + h)` of the total variance, and
/// `se = sqrt(var)` halves relative errors, so the worst case over
/// horizons is `0.5 * h/(n+h) * 3e-8 ~ 5e-9` at `h = 24`. The measured
/// maximum is 7.2e-9, printed by `cargo test -- --nocapture`.
#[test]
fn drift_anchor_random_walk_closed_form() {
    let t_obs = 60;
    let y = random_walk_with_drift(7, t_obs, 0.4, 1.0);
    let (params, n) = rw_drift_mle(&y);
    assert_eq!(n, t_obs - 1);

    let spec = ArimaSpec::new(0, 1, 0).unwrap().with_constant(true);
    let res = spec.at_params(&y, &params).unwrap();

    let steps = 24;
    let plain = res.forecast(steps).unwrap();
    let corrected = res
        .forecast_with(steps, ForecastOptions::new().with_drift_uncertainty(true))
        .unwrap();

    let sigma = params[1].sqrt();
    let n = n as f64;
    let mut worst = 0.0_f64;
    for h in 1..=steps {
        let hf = h as f64;
        // Parameters known: sigma sqrt(h). This is what the audit found
        // the crate reporting, to 2.7e-15.
        assert_rel_close(
            plain.se[h - 1],
            sigma * hf.sqrt(),
            1e-12,
            &format!("default se[h={h}] = sigma sqrt(h)"),
        );
        // With the drift term: sigma sqrt(h + h^2/n).
        let want = sigma * (hf + hf * hf / n).sqrt();
        let rel = (corrected.se[h - 1] - want).abs() / want;
        worst = worst.max(rel);
        assert!(
            rel <= 5e-8,
            "corrected se[h={h}] = sigma sqrt(h + h^2/n): {} vs {want} (rel {rel:e})",
            corrected.se[h - 1]
        );
    }
    println!("drift anchor: worst relative error over h=1..{steps} is {worst:e}");

    // Var(c_hat) = sigma2 / n exactly, and it is the (0, 0) entry.
    let var_c = res.param_cov().unwrap().get(0, 0).unwrap();
    assert_rel_close(var_c, params[1] / n, 1e-9, "Var(c_hat) = sigma2 / n");

    // The point forecasts are untouched, bit for bit.
    assert_eq!(plain.mean, corrected.mean, "drift term moved the means");
}

/// The same closed form must hold on the *shipped* path — `fit`, not
/// `at_params` — since that is what users call. Looser tolerance
/// (1e-6): the optimizer stops on a convergence test, not on the exact
/// stationary point, so `sigma_hat` itself carries that error.
#[test]
fn drift_anchor_holds_through_fit() {
    let t_obs = 60;
    let y = random_walk_with_drift(11, t_obs, 0.4, 1.0);
    let spec = ArimaSpec::new(0, 1, 0).unwrap().with_constant(true);
    let res = spec.fit(&y).unwrap();
    assert!(res.converged);

    let (analytic, n) = rw_drift_mle(&y);
    assert_rel_close(res.params()[0], analytic[0], 1e-6, "c_hat = xbar");
    assert_rel_close(res.params()[1], analytic[1], 1e-6, "sigma2_hat");

    let steps = 24;
    let corrected = res
        .forecast_with(steps, ForecastOptions::new().with_drift_uncertainty(true))
        .unwrap();
    let sigma = res.sigma2().sqrt();
    let n = n as f64;
    for h in 1..=steps {
        let hf = h as f64;
        assert_rel_close(
            corrected.se[h - 1],
            sigma * (hf + hf * hf / n).sqrt(),
            1e-6,
            &format!("fitted corrected se[h={h}]"),
        );
    }
}

/// A second, independent closed form for the derivative, on a
/// *stationary* model where the answer is different: for AR(1) with a
/// constant and `d = 0` the forecast recursion is `a <- c + phi a` from
/// `a_{T|T} = y_T` (the measurement equation is noiseless), so
///
/// ```text
/// yhat_{T+h} = c (1 - phi^h) / (1 - phi) + phi^h y_T,
/// d yhat_{T+h} / dc = (1 - phi^h) / (1 - phi).
/// ```
///
/// This holds at any admissible parameter point, not just the MLE, so it
/// isolates the derivative from the Hessian: the test reads `Var(c_hat)`
/// off `param_cov` and checks that the variance the correction added is
/// exactly `(d yhat / dc)^2 Var(c_hat)`.
#[test]
fn drift_derivative_matches_ar1_geometric_sum() {
    let y = random_walk_with_drift(23, 200, 0.0, 1.0);
    // A stationary series: use the differences, which are iid, and fit an
    // AR(1) to them at fixed parameters.
    let x: Vec<f64> = y.windows(2).map(|w| w[1] - w[0]).collect();
    let phi = 0.6_f64;
    let params = vec![0.3, phi, 1.1];

    let spec = ArimaSpec::new(1, 0, 0).unwrap().with_constant(true);
    let res = spec.at_params(&x, &params).unwrap();
    let var_c = res.param_cov().unwrap().get(0, 0).unwrap();
    assert!(var_c > 0.0, "Var(c_hat) = {var_c}");

    let steps = 30;
    let plain = res.forecast(steps).unwrap();
    let corrected = res
        .forecast_with(steps, ForecastOptions::new().with_drift_uncertainty(true))
        .unwrap();
    for h in 1..=steps {
        let dydc = (1.0 - phi.powi(h as i32)) / (1.0 - phi);
        let added = corrected.se[h - 1].powi(2) - plain.se[h - 1].powi(2);
        assert_rel_close(
            added,
            dydc * dydc * var_c,
            1e-8,
            &format!("added variance[h={h}] = ((1-phi^h)/(1-phi))^2 Var(c_hat)"),
        );
    }
}

/// The default path must be untouched: `forecast_with` under the default
/// options has to return the *same bits* as `forecast`, on a model where
/// the drift correction would otherwise be large. This is the guard on
/// the statsmodels parity gate in `golden.rs`.
#[test]
fn default_options_are_byte_identical_to_forecast() {
    let y = random_walk_with_drift(31, 80, 0.7, 1.3);
    let spec = ArimaSpec::new(0, 1, 1).unwrap().with_constant(true);
    let res = spec.fit(&y).unwrap();

    let plain = res.forecast(18).unwrap();
    for opts in [
        ForecastOptions::default(),
        ForecastOptions::new(),
        ForecastOptions::new().with_drift_uncertainty(false),
    ] {
        let same = res.forecast_with(18, opts).unwrap();
        assert_eq!(plain, same, "default options changed the forecast");
        for h in 0..18 {
            assert_eq!(
                plain.se[h].to_bits(),
                same.se[h].to_bits(),
                "se[{h}] differs in the last bit"
            );
        }
    }

    // And the correction, when asked for, is a strict widening at every
    // horizon (Var(c_hat) > 0 and the derivative is nonzero for d = 1).
    let wide = res
        .forecast_with(18, ForecastOptions::new().with_drift_uncertainty(true))
        .unwrap();
    for h in 0..18 {
        assert!(
            wide.se[h] > plain.se[h],
            "corrected se[{h}] = {} not wider than {}",
            wide.se[h],
            plain.se[h]
        );
    }
}

/// Asking for the drift term without a constant is an error, not a
/// silent no-op: the whole point of the option is wider bands, and
/// returning the identical narrow ones would repeat the bug it fixes.
#[test]
fn drift_uncertainty_without_a_constant_is_an_error() {
    let y = random_walk_with_drift(41, 80, 0.0, 1.0);
    let spec = ArimaSpec::new(0, 1, 0).unwrap();
    let res = spec.fit(&y).unwrap();
    assert!(res.forecast(5).is_ok());
    let err = res
        .forecast_with(5, ForecastOptions::new().with_drift_uncertainty(true))
        .unwrap_err();
    assert!(
        matches!(err, ArimaError::InvalidArgument { .. }),
        "expected InvalidArgument, got {err:?}"
    );
    assert!(
        err.to_string().contains("with_constant"),
        "the error must name the fix: {err}"
    );

    // steps = 0 still fails the same way through the options path.
    let spec_c = ArimaSpec::new(0, 1, 0).unwrap().with_constant(true);
    let res_c = spec_c.fit(&y).unwrap();
    assert!(matches!(
        res_c.forecast_with(0, ForecastOptions::new().with_drift_uncertainty(true)),
        Err(ArimaError::InvalidArgument { .. })
    ));
}

/// Reproduces the interval-coverage audit and its repair, on the audit's
/// own design (random walk with drift, `T = 60`, nominal 95%, `h = 24`).
///
/// Each replication uses the analytic exact MLE via `at_params` — for
/// this model that *is* the maximizer, so the Monte Carlo measures the
/// interval formula rather than the optimizer.
///
/// The predicted default coverage is the closed form from the module
/// docs, `2 Phi(z / sqrt(1 + h/n)) - 1 = 0.9016`; the audit measured
/// 90.3%. This test measures **89.35%** for the default band and
/// **95.00%** for the corrected one — the audit's own before/after
/// (90.2% -> 94.5%).
///
/// Those numbers moved (from 89.0% / 94.6%) when the replications stopped
/// being independently seeded. Each used to come from `Lcg::new(1000 +
/// rep)`, and since `Lcg::new` is one multiply-add and `uniform()` is a
/// top-bits shift, consecutive replications differed by a fixed additive
/// constant and their first draws formed a Weyl-type lattice rather than
/// an iid sample. Drawing every replication from one generator moved the
/// corrected coverage onto the nominal level exactly and closed a third
/// of the default band's gap to the closed form; the rest is within
/// Monte Carlo error, plus the small downward bias from plugging in
/// `sigma_hat` where the closed form assumes `sigma` known.
///
/// The bands below are wide enough for the Monte Carlo error (2000
/// replications, ~0.7pp standard error) and would still fail loudly if
/// the drift term were ever dropped again. 2000 replications is the bulk
/// of this file's runtime (~35 s in debug); it is worth it, because this
/// is the only test that checks the *consequence* rather than the
/// formula.
#[test]
fn coverage_shortfall_and_repair() {
    let reps = 2000;
    let t_obs = 60;
    let h = 24_usize;
    let drift = 0.4;
    let sd = 1.0;
    let spec = ArimaSpec::new(0, 1, 0).unwrap().with_constant(true);
    let opts = ForecastOptions::new().with_drift_uncertainty(true);

    let mut hit_plain = 0_usize;
    let mut hit_corrected = 0_usize;
    // One generator for the whole experiment: see `walk_from`. Reseeding
    // per replication put the first draws on a lattice, which is a
    // plausible part of why the measured default coverage sat 1.2pp below
    // the closed form.
    let mut rng = Lcg::new(1000);
    for _ in 0..reps {
        // One long path: the first `t_obs` points are the estimation
        // sample, the next `h` are the realized future.
        let path = walk_from(&mut rng, t_obs + h, drift, sd);
        let y = &path[..t_obs];
        let truth = path[t_obs + h - 1];

        let (params, _) = rw_drift_mle(y);
        let res = spec.at_params(y, &params).unwrap();
        let plain = res.forecast(h).unwrap().conf_int(0.05).unwrap();
        let corrected = res.forecast_with(h, opts).unwrap().conf_int(0.05).unwrap();
        if plain[h - 1].0 <= truth && truth <= plain[h - 1].1 {
            hit_plain += 1;
        }
        if corrected[h - 1].0 <= truth && truth <= corrected[h - 1].1 {
            hit_corrected += 1;
        }
    }

    let cov_plain = hit_plain as f64 / reps as f64;
    let cov_corrected = hit_corrected as f64 / reps as f64;
    // Closed-form prediction for the parameters-known band.
    let n = (t_obs - 1) as f64;
    let predicted = 2.0 * std_normal_cdf(1.959963984540054 / (1.0 + h as f64 / n).sqrt()) - 1.0;
    println!(
        "coverage at h={h}, T={t_obs}, nominal 95%: default {cov_plain:.4} \
         (closed form {predicted:.4}), corrected {cov_corrected:.4}"
    );
    assert!(
        (predicted - 0.9016).abs() < 5e-4,
        "closed-form prediction drifted: {predicted}"
    );

    assert!(
        (cov_plain - predicted).abs() < 0.025,
        "default-band coverage {cov_plain} is not near the predicted {predicted}"
    );
    assert!(
        (0.925..=0.975).contains(&cov_corrected),
        "corrected-band coverage {cov_corrected} is not near the 95% nominal level"
    );
    assert!(
        cov_corrected > cov_plain + 0.02,
        "the correction bought nothing: {cov_plain} -> {cov_corrected}"
    );
}

/// Standard normal CDF via the error function's Abramowitz-Stegun 7.1.26
/// rational approximation (|error| < 1.5e-7) — enough to check a
/// coverage prediction to three decimals without pulling the stats crate
/// into this test's dependency set.
fn std_normal_cdf(z: f64) -> f64 {
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}
