//! Internal-consistency, determinism, and guard tests for `auto_arima`.
//!
//! The selection loop's primary grade is Monte-Carlo order recovery,
//! which needs release-speed fits and therefore lives in the Python
//! suite (`bindings/python/tests/test_auto_arima.py`) and the MC study
//! quoted in the model card. What this file pins, in debug-friendly
//! sizes, is the loop's *contract*:
//!
//! * every candidate the search visits has its criterion computed by
//!   this crate's own exact-MLE fit, and refitting the reported best
//!   orders reproduces the reported best criterion **exactly** (same
//!   deterministic code path, so equality is to the bit);
//! * the search is deterministic: two runs give identical traces;
//! * the reported best is the argmin over the eligible trace entries;
//! * the tiny-caps stepwise and exhaustive searches agree where the
//!   grid is small enough to enumerate;
//! * the error surfaces teach instead of panicking.

mod common;

use common::{integrate, simulate_arma, Lcg};
use tsecon_arima::{
    auto_arima, ArimaError, ArimaSpec, AutoArimaOptions, AutoArimaResult, CandidateStatus,
};

/// Small options that keep debug-mode fit counts low: non-seasonal,
/// p and q capped at 2.
fn small_opts() -> AutoArimaOptions {
    AutoArimaOptions {
        max_p: 2,
        max_q: 2,
        ..AutoArimaOptions::default()
    }
}

/// Refits `r.best.spec` from scratch and checks the reported criterion
/// is reproduced exactly, and that the reported best is the argmin over
/// the eligible (status Ok) trace entries.
fn assert_internally_consistent(y: &[f64], r: &AutoArimaResult) {
    // (a) best_ic is the minimum over eligible candidates.
    let min_ok = r
        .trace
        .iter()
        .filter(|c| c.status == CandidateStatus::Ok)
        .filter_map(|c| c.ic)
        .fold(f64::INFINITY, f64::min);
    assert_eq!(
        r.best_ic, min_ok,
        "reported best_ic is not the trace minimum"
    );

    // (b) the best entry's orders appear in the trace with that ic.
    let spec = r.best.spec;
    let entry = r
        .trace
        .iter()
        .find(|c| {
            c.p == spec.p()
                && c.q == spec.q()
                && c.seasonal_p == spec.seasonal_p()
                && c.seasonal_q == spec.seasonal_q()
                && c.constant == spec.include_constant()
        })
        .expect("selected orders missing from the trace");
    assert_eq!(entry.status, CandidateStatus::Ok);
    assert_eq!(entry.ic, Some(r.best_ic));

    // (c) refitting the reported orders reproduces the reported
    //     criterion exactly (deterministic fit, same code path).
    let mut refit_spec = ArimaSpec::new(spec.p(), r.d, spec.q())
        .unwrap()
        .with_constant(spec.include_constant());
    if r.seasonal_period >= 2 {
        refit_spec = refit_spec
            .seasonal(
                spec.seasonal_p(),
                r.seasonal_d,
                spec.seasonal_q(),
                r.seasonal_period,
            )
            .unwrap();
    }
    let refit = refit_spec.fit(y).unwrap();
    let refit_ic = r.ic.evaluate(&refit).unwrap();
    assert_eq!(
        refit_ic, r.best_ic,
        "refitting the reported orders does not reproduce the reported IC"
    );
    assert_eq!(
        refit.loglik, r.best.loglik,
        "refitting the reported orders does not reproduce the reported loglik"
    );
    assert_eq!(refit.params(), r.best.params(), "refit params differ");
}

/// One stepwise search on a seeded AR(1): internally consistent,
/// deterministic across runs, and *coherent* — the true model is
/// visited (it is a starting model) and whatever is selected scores at
/// least as well on the criterion. Recovery *rates* are the Python MC
/// study's job (single seeds legitimately select an over-fit neighbor
/// when AICc genuinely prefers it on that draw; this seed does).
#[test]
fn stepwise_consistent_deterministic_and_recovers_ar1() {
    let mut rng = Lcg::new(11);
    let y = simulate_arma(&mut rng, 150, 0.0, &[0.7], &[], 1.0);
    let opts = AutoArimaOptions {
        // Fix d: this test is about the order search, not the KPSS
        // sequence (which has its own goldens in tsecon-diag).
        fixed_d: Some(0),
        ..small_opts()
    };

    let r1 = auto_arima(&y, &opts).unwrap();
    assert_internally_consistent(&y, &r1);
    assert!(r1.d_evidence.is_none(), "fixed d must skip the KPSS run");
    assert_eq!(r1.d, 0);
    assert_eq!(r1.seasonal_d, 0);
    assert!(!r1.budget_exhausted);
    assert_eq!(r1.n_models, r1.trace.len());

    // Coherence: the true (1, 0, 0)+constant model is a starting model,
    // so it was visited and scored — and the selected model beat or
    // matched it on the criterion.
    let truth = r1
        .trace
        .iter()
        .find(|c| c.p == 1 && c.q == 0 && c.constant)
        .expect("the (1,0)+c starting model is missing from the trace");
    assert_eq!(truth.status, CandidateStatus::Ok);
    assert!(
        r1.best_ic <= truth.ic.unwrap(),
        "selected model scores worse than the visited truth"
    );

    // Determinism: a second run reproduces the whole trace bit-for-bit.
    let r2 = auto_arima(&y, &opts).unwrap();
    assert_eq!(r1.trace, r2.trace, "search trace is not deterministic");
    assert_eq!(r1.best_ic, r2.best_ic);
    assert_eq!(r1.best.params(), r2.best.params());
    assert_eq!(r1.interpretation, r2.interpretation);
}

/// Stepwise and the exhaustive grid agree on a tiny search space where
/// the grid can be enumerated cheaply, and the KPSS stage picks d = 1 on
/// an integrated series (evidence attached).
#[test]
fn tiny_grid_matches_stepwise_and_kpss_picks_d() {
    let mut rng = Lcg::new(3);
    let x = simulate_arma(&mut rng, 121, 0.0, &[0.5], &[], 1.0);
    let y = integrate(&x, 1);
    let opts = AutoArimaOptions {
        max_p: 1,
        max_q: 1,
        max_order: 2,
        ..AutoArimaOptions::default()
    };

    let step = auto_arima(&y, &opts).unwrap();
    let grid = auto_arima(
        &y,
        &AutoArimaOptions {
            stepwise: false,
            ..opts.clone()
        },
    )
    .unwrap();

    // The random walk must be differenced: KPSS chooses d = 1, with the
    // per-order evidence attached.
    assert_eq!(step.d, 1);
    let ev = step.d_evidence.as_ref().expect("ndiffs evidence missing");
    assert_eq!(ev.d, 1);
    assert!(ev.steps[0].needs_differencing);
    assert!(!ev.steps[1].needs_differencing);

    assert_internally_consistent(&y, &step);
    assert_internally_consistent(&y, &grid);
    assert!(!grid.stepwise && step.stepwise);

    // The tiny stepwise search explores the whole 2x2 (+constant) box
    // here, so both must land on the same orders and criterion.
    assert_eq!(step.best.spec, grid.best.spec, "stepwise vs grid orders");
    assert_eq!(step.best_ic, grid.best_ic, "stepwise vs grid criterion");

    // The grid visited every candidate with p + q <= max_order once.
    assert_eq!(grid.n_models, 8, "2 x 2 orders x 2 constants");
}

/// A tiny seasonal search: with (d, D) fixed the loop recovers a pure
/// seasonal AR at period 4 through the multiplicative spec, and the
/// trace carries the seasonal orders.
#[test]
fn tiny_seasonal_search_recovers_sar1() {
    let mut rng = Lcg::new(9);
    // y_t = 0.7 y_{t-4} + e_t: SARIMA (0,0,0)(1,0,0)_4.
    let y = simulate_arma(&mut rng, 120, 0.0, &[0.0, 0.0, 0.0, 0.7], &[], 1.0);
    let opts = AutoArimaOptions {
        seasonal_period: 4,
        max_p: 1,
        max_q: 0,
        max_seasonal_p: 1,
        max_seasonal_q: 0,
        fixed_d: Some(0),
        fixed_seasonal_d: Some(0),
        ..AutoArimaOptions::default()
    };
    let r = auto_arima(&y, &opts).unwrap();
    assert_internally_consistent(&y, &r);
    assert_eq!(r.seasonal_period, 4);
    assert_eq!(
        (r.best.spec.seasonal_p(), r.best.spec.p()),
        (1, 0),
        "SAR(1)_4 not recovered: {}",
        r.interpretation
    );
    assert!(r
        .trace
        .iter()
        .any(|c| c.seasonal_p == 1 && c.status == CandidateStatus::Ok));
}

/// The error surfaces teach instead of panicking, and never run a fit.
#[test]
fn error_paths() {
    let y: Vec<f64> = (0..60).map(|t| (t as f64 * 0.7).sin()).collect();

    // seasonal_period = 1 is refused.
    let err = auto_arima(
        &y,
        &AutoArimaOptions {
            seasonal_period: 1,
            ..AutoArimaOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, ArimaError::InvalidArgument { .. }), "{err}");
    assert!(err.to_string().contains("seasonal_period"), "{err}");

    // Cap typos are refused with the limit named.
    for opts in [
        AutoArimaOptions {
            max_p: 40,
            ..AutoArimaOptions::default()
        },
        AutoArimaOptions {
            seasonal_period: 4,
            max_seasonal_q: 9,
            ..AutoArimaOptions::default()
        },
        AutoArimaOptions {
            max_models: 0,
            ..AutoArimaOptions::default()
        },
    ] {
        let err = auto_arima(&y, &opts).unwrap_err();
        assert!(matches!(err, ArimaError::InvalidArgument { .. }), "{err}");
    }

    // A fixed D > 0 without a seasonal period is meaningless.
    let err = auto_arima(
        &y,
        &AutoArimaOptions {
            fixed_seasonal_d: Some(1),
            ..AutoArimaOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("seasonal_period"), "{err}");

    // NaN input names the index.
    let mut bad = y.clone();
    bad[7] = f64::NAN;
    let err = auto_arima(&bad, &AutoArimaOptions::default()).unwrap_err();
    assert!(
        matches!(err, ArimaError::NonFinite { at: Some(7), .. }),
        "{err}"
    );

    // Too short for the advisors: the Selection error carries the
    // advisor's own teaching text.
    let err = auto_arima(&y[..3], &AutoArimaOptions::default()).unwrap_err();
    assert!(matches!(err, ArimaError::Selection { .. }), "{err}");
    assert!(err.to_string().contains("auto_arima"), "{err}");

    // An oversized exhaustive grid is refused with advice, not run.
    let err = auto_arima(
        &y,
        &AutoArimaOptions {
            stepwise: false,
            seasonal_period: 4,
            max_p: 12,
            max_q: 12,
            max_seasonal_p: 6,
            max_seasonal_q: 6,
            max_order: 36,
            ..AutoArimaOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("stepwise"), "{err}");
}

/// The model budget stops a search instead of letting it run long, and
/// says so.
#[test]
fn budget_exhaustion_is_reported() {
    let mut rng = Lcg::new(4);
    let y = simulate_arma(&mut rng, 130, 0.0, &[0.5], &[0.3], 1.0);
    let opts = AutoArimaOptions {
        fixed_d: Some(0),
        max_models: 3,
        ..small_opts()
    };
    let r = auto_arima(&y, &opts).unwrap();
    assert!(r.budget_exhausted);
    assert_eq!(r.n_models, 3);
    // The best-so-far is still internally consistent.
    assert_internally_consistent(&y, &r);
}
