//! Seeded Monte-Carlo property tests for `tsecon-ets`: the statistical
//! claims a fixed-parameter golden cannot make.
//!
//! * (a) **Parameter recovery at large T**: data simulated from the
//!   innovations form at known parameters, refit; the mean estimate over
//!   replications sits within a stated band of the truth and the spread
//!   shrinks with T (ETS(A,N,N), ETS(A,A,N), ETS(A,N,A), ETS(M,N,N),
//!   ETS(M,Ad,M) — the damped form: in Hyndman's (M,A,M) the slope
//!   innovation `beta (l + b) e` random-walks with the level, so an
//!   undamped design at these parameters wanders below zero).
//! * (b) **Interval coverage for the class-1 models**: 80% and 95%
//!   prediction intervals from the fitted model cover the realised future
//!   at close to the nominal rate over seeded replications (ETS(A,N,N),
//!   ETS(A,A,N), ETS(A,Ad,N), ETS(A,N,A)); parameter uncertainty is
//!   ignored by the closed forms, so coverage sits a little below nominal
//!   in finite samples, as Hyndman et al. (2008, section 6.4) say, and the
//!   simulated intervals of a class-2 model track the exact ones of its
//!   additive-error twin where the two models coincide.
//! * (c) **`auto_ets` recovers the generating component form** with high
//!   frequency at large T for well-separated designs (a level-only
//!   series, a trending series, a seasonal series, a multiplicative-
//!   seasonal series) — the grade the selection loop carries, as
//!   `auto_arima`'s does (no runnable third-party auto-ETS exists here).
//! * (d) Invariances and identities: scale equivariance of the additive
//!   models; the estimated seasonal normalisation; refit determinism; the
//!   forecast from the final state equals the simulation with zero errors;
//!   simulated intervals are seed-reproducible and widen with the horizon.
//! * (e) Degenerate input raises the documented, parameter-naming errors.
//!
//! Run with `--nocapture` to see the measured rates quoted in the model
//! card.

use tsecon_ets::{
    auto_ets, ets_fit, forecast, forecast_from_state, simulate_paths, AutoEtsOptions, Component,
    ErrorType, EtsError, EtsParams, EtsSpec, EtsStates, FitOptions, Ic, Initialization,
    IntervalMethod, Optimizer,
};
use tsecon_rng::Stream;

fn normal(stream: &mut Stream) -> f64 {
    let u1 = 1.0 - stream.uniform_f64();
    let u2 = stream.uniform_f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// Simulates `n` observations (plus `h` future values) from `spec` at
/// `params` from `init` with Gaussian innovations of standard deviation
/// `sigma` through the crate's own innovations recursion (validated
/// against statsmodels `simulate` in the golden tests).
fn simulate(
    spec: &EtsSpec,
    params: &EtsParams,
    init: &EtsStates,
    sigma: f64,
    n: usize,
    stream: &mut Stream,
) -> Vec<f64> {
    let errors: Vec<Vec<f64>> = vec![(0..n).map(|_| sigma * normal(stream)).collect()];
    simulate_paths(spec, params, init, &errors).expect("simulate")[0].clone()
}

fn spec(e: ErrorType, t: Component, d: bool, s: Component, m: Option<usize>) -> EtsSpec {
    EtsSpec::new(e, t, d, s, m).expect("spec")
}

/// One `auto_ets` recovery design: a label, the DGP (spec, parameters,
/// initial states, innovation scale), the `seasonal_periods` the search is
/// given, and the component forms counted as a recovery of the generating
/// one (the error type and the damped/undamped distinction are not graded).
type AutoCase = (
    &'static str,
    EtsSpec,
    EtsParams,
    EtsStates,
    f64,
    Option<usize>,
    Vec<&'static str>,
);

/// `crates/tsecon-ets/src/forecast.rs`'s `MAX_HORIZON` (private there):
/// the allocation guard the refusal tests below pin.
const MAX_H: usize = 1_000_000;

const A: ErrorType = ErrorType::Additive;
const M: ErrorType = ErrorType::Multiplicative;
const N: Component = Component::None;
const AD: Component = Component::Additive;
const MU: Component = Component::Multiplicative;

// ------------------------------------------------------------ (a) recovery

struct Design {
    name: &'static str,
    spec: EtsSpec,
    params: EtsParams,
    init: EtsStates,
    sigma: f64,
}

fn designs() -> Vec<Design> {
    vec![
        Design {
            name: "ANN",
            spec: spec(A, N, false, N, None),
            params: EtsParams {
                alpha: 0.5,
                beta: None,
                gamma: None,
                phi: None,
            },
            init: EtsStates {
                level: 50.0,
                trend: None,
                seasonal: None,
            },
            sigma: 2.0,
        },
        Design {
            name: "AAN",
            spec: spec(A, AD, false, N, None),
            params: EtsParams {
                alpha: 0.4,
                beta: Some(0.1),
                gamma: None,
                phi: None,
            },
            init: EtsStates {
                level: 50.0,
                trend: Some(0.3),
                seasonal: None,
            },
            sigma: 2.0,
        },
        Design {
            name: "ANA",
            spec: spec(A, N, false, AD, Some(4)),
            params: EtsParams {
                alpha: 0.3,
                beta: None,
                gamma: Some(0.2),
                phi: None,
            },
            init: EtsStates {
                level: 50.0,
                trend: None,
                seasonal: Some(vec![3.0, -1.0, -4.0, 2.0]),
            },
            sigma: 2.0,
        },
        Design {
            name: "MNN",
            spec: spec(M, N, false, N, None),
            params: EtsParams {
                alpha: 0.5,
                beta: None,
                gamma: None,
                phi: None,
            },
            init: EtsStates {
                level: 100.0,
                trend: None,
                seasonal: None,
            },
            sigma: 0.05,
        },
        Design {
            name: "MAdM",
            spec: spec(M, AD, true, MU, Some(4)),
            params: EtsParams {
                alpha: 0.3,
                beta: Some(0.03),
                gamma: Some(0.15),
                phi: Some(0.9),
            },
            init: EtsStates {
                level: 100.0,
                trend: Some(0.5),
                seasonal: Some(vec![0.9, 1.1, 1.05, 0.95]),
            },
            sigma: 0.04,
        },
    ]
}

#[test]
fn smoothing_parameters_are_recovered_at_large_t() {
    let reps = 40;
    for d in designs() {
        let mut streams = Stream::substreams(101, reps).expect("streams");
        let mut sums = [0.0_f64; 3];
        let mut sq = [0.0_f64; 3];
        let mut n_ok = 0;
        for st in streams.iter_mut() {
            let y = simulate(&d.spec, &d.params, &d.init, d.sigma, 800, st);
            let fit = ets_fit(&d.spec, &y, &FitOptions::default()).expect("fit");
            let est = [
                fit.params.alpha,
                fit.params.beta.unwrap_or(0.0),
                fit.params.gamma.unwrap_or(0.0),
            ];
            for i in 0..3 {
                sums[i] += est[i];
                sq[i] += est[i] * est[i];
            }
            n_ok += 1;
        }
        let truth = [
            d.params.alpha,
            d.params.beta.unwrap_or(0.0),
            d.params.gamma.unwrap_or(0.0),
        ];
        let names = ["alpha", "beta", "gamma"];
        for i in 0..3 {
            let mean = sums[i] / n_ok as f64;
            let sd = (sq[i] / n_ok as f64 - mean * mean).max(0.0).sqrt();
            eprintln!(
                "recovery {} {}: truth {:.3} mean {:.4} sd {:.4} (T = 800, {reps} reps)",
                d.name, names[i], truth[i], mean, sd
            );
            if truth[i] > 0.0 {
                // The mean estimate lies within 0.05 of the truth: a band
                // wide enough for the boundary-censored beta/gamma at this
                // T, tight enough to catch a wrong recursion or a mis-scaled
                // parameter (beta vs beta*, gamma vs gamma*).
                assert!(
                    (mean - truth[i]).abs() < 0.05,
                    "{} {}: mean {mean} vs truth {}",
                    d.name,
                    names[i],
                    truth[i]
                );
            }
        }
    }
}

#[test]
fn recovery_spread_shrinks_with_t() {
    let d = &designs()[0]; // ANN, alpha = 0.5
    let reps = 60;
    let mut sds = Vec::new();
    for &n in &[100usize, 800] {
        let mut streams = Stream::substreams(202, reps).expect("streams");
        let mut ests = Vec::with_capacity(reps);
        for st in streams.iter_mut() {
            let y = simulate(&d.spec, &d.params, &d.init, d.sigma, n, st);
            ests.push(
                ets_fit(&d.spec, &y, &FitOptions::default())
                    .expect("fit")
                    .params
                    .alpha,
            );
        }
        let mean = ests.iter().sum::<f64>() / reps as f64;
        let sd = (ests.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / reps as f64).sqrt();
        eprintln!("ANN alpha at T = {n}: mean {mean:.4} sd {sd:.4}");
        sds.push(sd);
    }
    assert!(
        sds[1] < 0.6 * sds[0],
        "sd at T=800 ({}) vs T=100 ({})",
        sds[1],
        sds[0]
    );
}

// ------------------------------------------------------------ (b) coverage

#[test]
fn class1_interval_coverage_is_near_nominal() {
    let reps = 300;
    let h = 8;
    let n = 300;
    let cases: Vec<(&str, EtsSpec, EtsParams, EtsStates)> = vec![
        (
            "ANN",
            spec(A, N, false, N, None),
            EtsParams {
                alpha: 0.5,
                beta: None,
                gamma: None,
                phi: None,
            },
            EtsStates {
                level: 50.0,
                trend: None,
                seasonal: None,
            },
        ),
        (
            "AAN",
            spec(A, AD, false, N, None),
            EtsParams {
                alpha: 0.4,
                beta: Some(0.1),
                gamma: None,
                phi: None,
            },
            EtsStates {
                level: 50.0,
                trend: Some(0.2),
                seasonal: None,
            },
        ),
        (
            "AAdN",
            spec(A, AD, true, N, None),
            EtsParams {
                alpha: 0.4,
                beta: Some(0.1),
                gamma: None,
                phi: Some(0.9),
            },
            EtsStates {
                level: 50.0,
                trend: Some(0.5),
                seasonal: None,
            },
        ),
        (
            "ANA",
            spec(A, N, false, AD, Some(4)),
            EtsParams {
                alpha: 0.3,
                beta: None,
                gamma: Some(0.2),
                phi: None,
            },
            EtsStates {
                level: 50.0,
                trend: None,
                seasonal: Some(vec![3.0, -1.0, -4.0, 2.0]),
            },
        ),
    ];
    for (name, sp, params, init) in cases {
        let mut streams = Stream::substreams(303, reps).expect("streams");
        let mut hit80 = vec![0usize; h];
        let mut hit95 = vec![0usize; h];
        for st in streams.iter_mut() {
            let full = simulate(&sp, &params, &init, 2.0, n + h, st);
            let (y, future) = full.split_at(n);
            let fit = ets_fit(&sp, y, &FitOptions::default()).expect("fit");
            let f80 = forecast(&fit, h, 0.80, 0, 0).expect("forecast");
            let f95 = forecast(&fit, h, 0.95, 0, 0).expect("forecast");
            assert_eq!(f80.method, IntervalMethod::Exact);
            for j in 0..h {
                if future[j] >= f80.lower[j] && future[j] <= f80.upper[j] {
                    hit80[j] += 1;
                }
                if future[j] >= f95.lower[j] && future[j] <= f95.upper[j] {
                    hit95[j] += 1;
                }
            }
        }
        let c80: Vec<f64> = hit80.iter().map(|&k| k as f64 / reps as f64).collect();
        let c95: Vec<f64> = hit95.iter().map(|&k| k as f64 / reps as f64).collect();
        let mean80 = c80.iter().sum::<f64>() / h as f64;
        let mean95 = c95.iter().sum::<f64>() / h as f64;
        eprintln!(
            "coverage {name} (T = {n}, h = 1..{h}, {reps} reps): 80% -> {mean80:.3} (h1 {:.3}, h{h} {:.3}); 95% -> {mean95:.3} (h1 {:.3}, h{h} {:.3})",
            c80[0],
            c80[h - 1],
            c95[0],
            c95[h - 1]
        );
        // Bands around what was measured: nominal minus the parameter-
        // uncertainty shortfall of a plug-in interval, wide enough for the
        // binomial noise of 300 replications (se ~ 0.013 at 95%).
        assert!(
            mean80 > 0.72 && mean80 < 0.88,
            "{name}: 80% coverage {mean80}"
        );
        assert!(
            mean95 > 0.90 && mean95 < 0.99,
            "{name}: 95% coverage {mean95}"
        );
    }
}

#[test]
fn simulated_intervals_of_a_multiplicative_model_track_the_exact_ones_at_small_noise() {
    // ETS(M,N,N) with a small relative error at a high level is, to first
    // order, ETS(A,N,N) with sigma_add = level * sigma_rel: the simulated
    // interval of the former must sit close to the exact interval of the
    // latter fitted to the same data.
    let sp_m = spec(M, N, false, N, None);
    let sp_a = spec(A, N, false, N, None);
    let params = EtsParams {
        alpha: 0.4,
        beta: None,
        gamma: None,
        phi: None,
    };
    let init = EtsStates {
        level: 1000.0,
        trend: None,
        seasonal: None,
    };
    let mut st = Stream::new(404);
    let y = simulate(&sp_m, &params, &init, 0.01, 400, &mut st);
    let fm = ets_fit(&sp_m, &y, &FitOptions::default()).expect("fit m");
    let fa = ets_fit(&sp_a, &y, &FitOptions::default()).expect("fit a");
    let h = 6;
    let fcm = forecast(&fm, h, 0.95, 20_000, 7).expect("sim");
    let fca = forecast(&fa, h, 0.95, 0, 0).expect("exact");
    assert_eq!(fcm.method, IntervalMethod::Simulated);
    assert_eq!(fca.method, IntervalMethod::Exact);
    for j in 0..h {
        let width_m = fcm.upper[j] - fcm.lower[j];
        let width_a = fca.upper[j] - fca.lower[j];
        eprintln!(
            "h = {}: simulated width {:.3} exact width {:.3} (mean {:.3} vs {:.3})",
            j + 1,
            width_m,
            width_a,
            fcm.mean[j],
            fca.mean[j]
        );
        assert!(
            (width_m / width_a - 1.0).abs() < 0.08,
            "h = {}: {width_m} vs {width_a}",
            j + 1
        );
        assert!((fcm.mean[j] - fca.mean[j]).abs() < 0.5);
    }
}

// ------------------------------------------------------- (c) auto recovery

#[test]
fn auto_ets_recovers_the_generating_form_at_large_t() {
    let reps = 30;
    let n = 600;
    // (name, DGP, options, the set of component forms counted as a hit)
    let level = spec(A, N, false, N, None);
    let trend = spec(A, AD, false, N, None);
    let seas = spec(A, N, false, AD, Some(4));
    let mam = spec(M, AD, true, MU, Some(4));
    let cases: Vec<AutoCase> = vec![
        (
            "level (ANN)",
            level,
            EtsParams {
                alpha: 0.5,
                beta: None,
                gamma: None,
                phi: None,
            },
            EtsStates {
                level: 50.0,
                trend: None,
                seasonal: None,
            },
            2.0,
            Some(4),
            vec!["ANN", "MNN"],
        ),
        (
            "trend (AAN)",
            trend,
            EtsParams {
                alpha: 0.4,
                beta: Some(0.15),
                gamma: None,
                phi: None,
            },
            EtsStates {
                level: 50.0,
                trend: Some(0.5),
                seasonal: None,
            },
            2.0,
            Some(4),
            vec!["AAN", "AAdN", "MAN", "MAdN"],
        ),
        (
            "seasonal (ANA)",
            seas,
            EtsParams {
                alpha: 0.3,
                beta: None,
                gamma: Some(0.2),
                phi: None,
            },
            EtsStates {
                level: 50.0,
                trend: None,
                seasonal: Some(vec![6.0, -2.0, -8.0, 4.0]),
            },
            2.0,
            Some(4),
            vec!["ANA", "MNA"],
        ),
        (
            "multiplicative seasonal (MAdM)",
            mam,
            EtsParams {
                alpha: 0.3,
                beta: Some(0.03),
                gamma: Some(0.15),
                phi: Some(0.9),
            },
            EtsStates {
                level: 100.0,
                trend: Some(0.8),
                seasonal: Some(vec![0.7, 1.3, 1.1, 0.9]),
            },
            0.04,
            Some(4),
            vec!["MAM", "MAdM"],
        ),
    ];
    for (name, sp, params, init, sigma, m, hits) in cases {
        let mut streams = Stream::substreams(505, reps).expect("streams");
        let mut n_hit = 0;
        let mut picks: std::collections::BTreeMap<String, usize> = Default::default();
        for st in streams.iter_mut() {
            let y = simulate(&sp, &params, &init, sigma, n, st);
            let opts = AutoEtsOptions {
                seasonal_periods: m,
                ..AutoEtsOptions::default()
            };
            let r = auto_ets(&y, &opts).expect("auto");
            let chosen = r.best.spec.short_name();
            *picks.entry(chosen.clone()).or_insert(0) += 1;
            if hits.contains(&chosen.as_str()) {
                n_hit += 1;
            }
            // The winner is the trace minimum and its criterion is finite.
            assert_eq!(r.candidates[0].short_name, chosen);
            assert!(r.candidates[0].ic_value.is_finite());
            assert!(r
                .candidates
                .windows(2)
                .all(|w| w[0].ic_value <= w[1].ic_value));
        }
        let rate = n_hit as f64 / reps as f64;
        eprintln!("auto_ets recovery {name} (T = {n}, {reps} reps): {rate:.2}; picks {picks:?}");
        assert!(rate >= 0.7, "{name}: recovery rate {rate}");
    }
}

#[test]
fn auto_ets_refit_of_the_winner_reproduces_its_numbers_exactly() {
    // A damped trend keeps the simulated series positive (an undamped
    // additive slope random-walks by beta * e per period and wanders
    // below zero over 200 periods at these parameters), so every
    // candidate is admissible.
    let sp = spec(A, AD, true, AD, Some(4));
    let params = EtsParams {
        alpha: 0.4,
        beta: Some(0.1),
        gamma: Some(0.2),
        phi: Some(0.9),
    };
    let init = EtsStates {
        level: 500.0,
        trend: Some(0.3),
        seasonal: Some(vec![3.0, -1.0, -4.0, 2.0]),
    };
    let mut st = Stream::new(606);
    let y = simulate(&sp, &params, &init, 2.0, 200, &mut st);
    let opts = AutoEtsOptions {
        seasonal_periods: Some(4),
        ..AutoEtsOptions::default()
    };
    let r = auto_ets(&y, &opts).expect("auto");
    let refit = ets_fit(&r.best.spec, &y, &FitOptions::default()).expect("refit");
    assert_eq!(refit, r.best);
    let again = auto_ets(&y, &opts).expect("auto");
    assert_eq!(again, r);
    assert_eq!(r.n_candidates, 15);
    assert_eq!(r.ic, Ic::Aicc);
    // Restricting to additive-only data drops the multiplicative candidates.
    let y_neg: Vec<f64> = y.iter().map(|v| v - 600.0).collect();
    let r2 = auto_ets(&y_neg, &opts).expect("auto");
    assert_eq!(r2.n_candidates, 6);
    assert!(r2.candidates.iter().all(|c| c.spec.error == A));
}

// ------------------------------------------------------- (d) invariances

#[test]
fn additive_models_are_scale_and_location_equivariant() {
    let sp = spec(A, AD, true, AD, Some(4));
    let params = EtsParams {
        alpha: 0.4,
        beta: Some(0.1),
        gamma: Some(0.2),
        phi: Some(0.9),
    };
    let init = EtsStates {
        level: 50.0,
        trend: Some(0.3),
        seasonal: Some(vec![3.0, -1.0, -4.0, 2.0]),
    };
    let mut st = Stream::new(707);
    let y = simulate(&sp, &params, &init, 2.0, 160, &mut st);
    let fit = ets_fit(&sp, &y, &FitOptions::default()).expect("fit");
    let c = 7.0;
    let y2: Vec<f64> = y.iter().map(|v| c * v + 100.0).collect();
    let fit2 = ets_fit(&sp, &y2, &FitOptions::default()).expect("fit");
    assert!((fit2.params.alpha - fit.params.alpha).abs() < 2e-3);
    assert!(
        (fit2.loglik - (fit.loglik - 160.0 * c.ln())).abs() < 5e-3,
        "{} vs {}",
        fit2.loglik,
        fit.loglik - 160.0 * c.ln()
    );
    for (a, b) in fit.fitted.iter().zip(&fit2.fitted) {
        assert!((c * a + 100.0 - b).abs() < 0.2, "{a} {b}");
    }
    // The estimated seasonal states obey the normalisation exactly.
    let s = fit.initial_state.seasonal.as_ref().expect("seasonal");
    assert!(s.iter().sum::<f64>().abs() < 1e-9);
}

#[test]
fn multiplicative_seasonal_states_average_one_and_forecast_is_the_zero_error_path() {
    let sp = spec(M, AD, true, MU, Some(4));
    let params = EtsParams {
        alpha: 0.3,
        beta: Some(0.03),
        gamma: Some(0.15),
        phi: Some(0.9),
    };
    let init = EtsStates {
        level: 100.0,
        trend: Some(0.5),
        seasonal: Some(vec![0.9, 1.1, 1.05, 0.95]),
    };
    let mut st = Stream::new(808);
    let y = simulate(&sp, &params, &init, 0.04, 200, &mut st);
    let fit = ets_fit(&sp, &y, &FitOptions::default()).expect("fit");
    let s = fit.initial_state.seasonal.as_ref().expect("seasonal");
    assert!((s.iter().sum::<f64>() / 4.0 - 1.0).abs() < 1e-9);
    assert!(s.iter().all(|&v| v > 0.0));
    let h = 8;
    let fc = forecast(&fit, h, 0.9, 3000, 11).expect("forecast");
    let zero = forecast_from_state(&sp, &fit.params, &fit.final_state, h).expect("path");
    assert_eq!(fc.mean, zero);
    let sim0 = simulate_paths(&sp, &fit.params, &fit.final_state, &[vec![0.0; h]]).expect("sim");
    assert_eq!(sim0[0], zero);
    // Seed-reproducible, seed-sensitive, widening with the horizon.
    let fc2 = forecast(&fit, h, 0.9, 3000, 11).expect("forecast");
    assert_eq!(fc, fc2);
    let fc3 = forecast(&fit, h, 0.9, 3000, 12).expect("forecast");
    assert_ne!(fc.lower, fc3.lower);
    assert!(fc.upper[h - 1] - fc.lower[h - 1] > fc.upper[0] - fc.lower[0]);
    assert!(fc.variance.iter().all(|v| *v > 0.0));
    assert_eq!(fc.n_sim, 3000);
    assert_eq!(fc.seed, 11);
}

#[test]
fn heuristic_and_known_initialisations_estimate_only_the_smoothing_parameters() {
    let sp = spec(A, AD, false, N, None);
    let mut st = Stream::new(909);
    let params = EtsParams {
        alpha: 0.4,
        beta: Some(0.1),
        gamma: None,
        phi: None,
    };
    let init = EtsStates {
        level: 50.0,
        trend: Some(0.3),
        seasonal: None,
    };
    let y = simulate(&sp, &params, &init, 2.0, 150, &mut st);
    let heur = ets_fit(
        &sp,
        &y,
        &FitOptions {
            initialization: Initialization::Heuristic,
            optimizer: Optimizer::TwoStage,
            max_iter: None,
        },
    )
    .expect("fit");
    assert_eq!(heur.initialization, "heuristic");
    assert_eq!(heur.k_params, 3);
    let h0 = tsecon_ets::heuristic_initial_states(&sp, &y).expect("heuristic");
    assert_eq!(heur.initial_state, h0);
    let known = ets_fit(
        &sp,
        &y,
        &FitOptions {
            initialization: Initialization::Known(init.clone()),
            optimizer: Optimizer::Bfgs,
            max_iter: None,
        },
    )
    .expect("fit");
    assert_eq!(known.initialization, "known");
    assert_eq!(known.initial_state, init);
    assert_eq!(known.optimizer, "bfgs");
    let est = ets_fit(&sp, &y, &FitOptions::default()).expect("fit");
    assert_eq!(est.k_params, 5);
    assert!(est.loglik >= heur.loglik - 1e-9);
    assert!(est.loglik >= known.loglik - 1e-9);
}

// --------------------------------------------------------- (e) refusals

#[test]
fn degenerate_input_raises_teaching_errors_naming_the_argument() {
    let y: Vec<f64> = (0..40).map(|t| 10.0 + (t as f64).sin()).collect();
    // Inert options are refused at the spec.
    let e = EtsSpec::new(A, N, true, N, None).unwrap_err();
    assert!(matches!(e, EtsError::InvalidSpec { .. }) && e.to_string().contains("damped"));
    let e = EtsSpec::new(A, N, false, N, Some(4)).unwrap_err();
    assert!(e.to_string().contains("seasonal_periods"));
    let e = EtsSpec::new(A, N, false, AD, None).unwrap_err();
    assert!(e.to_string().contains("seasonal_periods"));
    let e = EtsSpec::new(A, N, false, AD, Some(1)).unwrap_err();
    assert!(e.to_string().contains("seasonal_periods"));
    // Multiplicative components need positive data: the message names y.
    let mut y_neg = y.clone();
    y_neg[5] = -1.0;
    let e = ets_fit(&spec(M, N, false, N, None), &y_neg, &FitOptions::default()).unwrap_err();
    assert!(matches!(e, EtsError::NonPositiveData { index: 5, .. }));
    assert!(e.to_string().contains("y[5]"));
    // NaN is refused, naming y.
    let mut y_nan = y.clone();
    y_nan[3] = f64::NAN;
    let e = ets_fit(&spec(A, N, false, N, None), &y_nan, &FitOptions::default()).unwrap_err();
    assert!(matches!(
        e,
        EtsError::NonFinite {
            what: "y",
            index: 3,
            ..
        }
    ));
    // Too few observations.
    let e = ets_fit(
        &spec(A, AD, true, AD, Some(12)),
        &y[..12],
        &FitOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(e, EtsError::TooFewObservations { .. }));
    // Parameters outside the box, named.
    let e = EtsParams::from_slice(&spec(A, AD, false, N, None), &[0.5, 0.7]).unwrap_err();
    assert!(e.to_string().starts_with("beta = 0.7"));
    let e = EtsParams::from_slice(&spec(A, N, false, N, None), &[1.5]).unwrap_err();
    assert!(e.to_string().starts_with("alpha = 1.5"));
    let e = EtsParams::from_slice(&spec(A, N, false, N, None), &[0.5, 0.1]).unwrap_err();
    assert!(matches!(
        e,
        EtsError::DimensionMismatch {
            expected: 1,
            got: 2,
            ..
        }
    ));
    let e = EtsStates::from_slice(&spec(A, N, false, AD, Some(4)), &[1.0, 0.0, 0.0]).unwrap_err();
    assert!(matches!(
        e,
        EtsError::DimensionMismatch {
            expected: 5,
            got: 3,
            ..
        }
    ));
    let e = EtsStates::from_slice(&spec(M, N, false, MU, Some(4)), &[1.0, 1.0, -1.0, 1.0, 1.0])
        .unwrap_err();
    assert!(e.to_string().contains("initial seasonal[1]"));
    // Forecast options.
    let fit = ets_fit(&spec(A, N, false, N, None), &y, &FitOptions::default()).expect("fit");
    let e = forecast(&fit, 0, 0.95, 0, 0).unwrap_err();
    assert!(e.to_string().starts_with("horizon = 0"));
    let e = forecast(&fit, 3, 1.5, 0, 0).unwrap_err();
    assert!(e.to_string().starts_with("level = 1.5"));
    let fit_m = ets_fit(&spec(M, N, false, N, None), &y, &FitOptions::default()).expect("fit");
    let e = forecast(&fit_m, 3, 0.95, 1, 0).unwrap_err();
    assert!(e.to_string().starts_with("n_sim = 1"));
    // Allocation budgets: a horizon or an n_sim x horizon buffer that would
    // abort the allocator is refused by name instead. (The Python layer
    // stops integer counts at 2^48; everything below that is this crate's
    // job.) 2^47 paths of one step, and a million and one steps, are both
    // inside what the binding forwards.
    let e = forecast(&fit_m, 1, 0.95, 1 << 47, 0).unwrap_err();
    assert!(e.to_string().starts_with("n_sim = 140737488355328"), "{e}");
    let e = forecast(&fit_m, 4, 0.95, 1 << 40, 0).unwrap_err();
    assert!(e.to_string().starts_with("n_sim = "), "{e}");
    for f in [&fit, &fit_m] {
        let e = forecast(f, MAX_H + 1, 0.95, 10, 0).unwrap_err();
        assert!(e.to_string().starts_with("horizon = 1000001"), "{e}");
    }
    let e = forecast_from_state(&fit.spec, &fit.params, &fit.final_state, MAX_H + 1).unwrap_err();
    assert!(e.to_string().starts_with("horizon = 1000001"), "{e}");
    // The budget binds only absurd requests: the documented default
    // n_sim = 5000 at a long-but-sane horizon still runs.
    let ok = forecast(&fit_m, 200, 0.95, 5000, 1).expect("5000 x 200 paths");
    assert_eq!(ok.mean.len(), 200);
    // auto_ets refusals.
    let e = auto_ets(
        &y,
        &AutoEtsOptions {
            seasonal_periods: Some(0),
            ..AutoEtsOptions::default()
        },
    )
    .unwrap_err();
    assert!(e.to_string().starts_with("seasonal_periods = 0"));
    let e = auto_ets(
        &y,
        &AutoEtsOptions {
            initialization: Initialization::Known(EtsStates {
                level: 1.0,
                trend: None,
                seasonal: None,
            }),
            ..AutoEtsOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        e,
        EtsError::InvalidOption {
            name: "initialization",
            ..
        }
    ));
    let e = auto_ets(&y_nan, &AutoEtsOptions::default()).unwrap_err();
    assert!(matches!(e, EtsError::NonFinite { what: "y", .. }));
}
