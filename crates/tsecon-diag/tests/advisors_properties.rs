//! Property / contract tests for the preprocessing advisors that need no
//! external golden: the invariances the two criteria are supposed to have,
//! the sequential rule's contract with its own evidence, and the teaching
//! error surface.

use tsecon_diag::{
    box_cox_lambda, box_cox_llf, guerrero_cv, ndiffs, AdvisorError, BoxCoxMethod, DiagError,
    NdiffsStop, NdiffsTest,
};

/// Deterministic pseudo-noise in (-1, 1): the logistic map at `r = 3.9999`,
/// which is chaotic (no fixed point, no cycle) and — being quadratic — is
/// never fitted exactly by a finite autoregression, so the lag regressions
/// inside the ADF stay non-degenerate. A fixed test vector, not a random
/// draw: no RNG is involved anywhere in this file.
fn noise(n: usize, seed: f64) -> Vec<f64> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = 3.9999 * x * (1.0 - x);
            2.0 * x - 1.0
        })
        .collect()
}

fn walk(n: usize, seed: f64) -> Vec<f64> {
    let mut acc = 0.0;
    noise(n, seed)
        .into_iter()
        .map(|e| {
            acc += e;
            acc
        })
        .collect()
}

/// A growing level with *multiplicative* noise: the spread scales with the
/// level, so the variance-stabilising transform is the log (lambda = 0).
fn multiplicative(n: usize, seed: f64) -> Vec<f64> {
    noise(n, seed)
        .into_iter()
        .enumerate()
        .map(|(t, e)| (2.0 + 0.02 * t as f64).exp() * (1.0 + 0.25 * e))
        .collect()
}

/// The same growing level with *additive* noise: the spread is constant,
/// so no transform is needed (lambda = 1).
fn additive(n: usize, seed: f64) -> Vec<f64> {
    noise(n, seed)
        .into_iter()
        .enumerate()
        .map(|(t, e)| 100.0 + 2.0 * t as f64 + 8.0 * e)
        .collect()
}

// ------------------------------------------------------------------ ndiffs

#[test]
fn ndiffs_evidence_matches_the_decision_rule_it_documents() {
    // The exposed evidence must imply the exposed decision: KPSS (null =
    // stationarity) differences while it rejects, ADF/PP (null = unit
    // root) difference unless they reject.
    let series = [walk(180, 0.41), noise(180, 0.41)];
    for y in &series {
        for (test, kpss_like) in [
            (NdiffsTest::Kpss, true),
            (NdiffsTest::Adf, false),
            (NdiffsTest::Pp, false),
        ] {
            let r = ndiffs(y, test, 0.05, 2).expect("ndiffs succeeds");
            for s in &r.steps {
                let expected = if kpss_like {
                    s.p_value < r.alpha
                } else {
                    s.p_value > r.alpha
                };
                assert_eq!(
                    s.needs_differencing,
                    expected,
                    "{}: step d = {} decision must follow from its own p-value",
                    test.code(),
                    s.d
                );
            }
            // Every step but the last must have called for a difference.
            for s in &r.steps[..r.steps.len() - 1] {
                assert!(s.needs_differencing, "an early step stopped the sequence");
            }
            assert_eq!(r.steps.len(), r.d + 1, "one evidence row per order tried");
        }
    }
}

#[test]
fn ndiffs_stops_at_max_d_and_says_so() {
    let y = walk(180, 0.41);
    let capped = ndiffs(&y, NdiffsTest::Kpss, 0.05, 0).expect("max_d = 0 is legal");
    assert_eq!(capped.d, 0, "the cap binds");
    assert_eq!(capped.stop, NdiffsStop::MaxD);
    assert_eq!(
        capped.steps.len(),
        1,
        "the level evidence is still reported"
    );
    assert!(
        capped.steps[0].needs_differencing,
        "a random walk asks for a difference"
    );
    assert!(
        capped.interpretation.contains("floor"),
        "a capped answer must be flagged as a floor, got: {}",
        capped.interpretation
    );

    let free = ndiffs(&y, NdiffsTest::Kpss, 0.05, 2).expect("uncapped");
    assert_eq!(free.d, 1, "a random walk is I(1)");
    assert_eq!(free.stop, NdiffsStop::Stationary);
}

#[test]
fn ndiffs_treats_a_deterministic_trend_as_a_dead_end() {
    // y_t = t is a *deterministic* trend: one difference leaves a constant
    // series, where no unit-root test is defined.
    let y: Vec<f64> = (0..60).map(|t| t as f64).collect();
    let r = ndiffs(&y, NdiffsTest::Kpss, 0.05, 2).expect("ndiffs succeeds");
    assert_eq!(r.stop, NdiffsStop::Constant);
    assert_eq!(r.d, 1);
    assert_eq!(r.steps.len(), 1, "only the level was testable");
    assert!(
        r.interpretation.contains("deterministic"),
        "the report must name the deterministic trend, got: {}",
        r.interpretation
    );

    // A constant input never even reaches a test.
    let flat = vec![7.5; 40];
    let r = ndiffs(&flat, NdiffsTest::Adf, 0.05, 2).expect("constant input is not an error");
    assert_eq!(r.d, 0);
    assert_eq!(r.stop, NdiffsStop::Constant);
    assert!(r.steps.is_empty());
}

#[test]
fn ndiffs_rejects_an_invalid_alpha_and_a_stub_series() {
    let y = walk(80, 0.21);
    assert!(matches!(
        ndiffs(&y, NdiffsTest::Kpss, 0.0, 2),
        Err(AdvisorError::Diag(DiagError::InvalidAlpha { .. }))
    ));
    assert!(matches!(
        ndiffs(&y, NdiffsTest::Kpss, 1.0, 2),
        Err(AdvisorError::Diag(DiagError::InvalidAlpha { .. }))
    ));
    assert!(matches!(
        ndiffs(&[1.0, 2.0, 3.0], NdiffsTest::Kpss, 0.05, 2),
        Err(AdvisorError::Diag(DiagError::SeriesTooShort { .. }))
    ));
    let nan = [1.0, f64::NAN, 3.0, 4.0, 5.0];
    let err = ndiffs(&nan, NdiffsTest::Kpss, 0.05, 2).unwrap_err();
    assert!(
        err.to_string().contains("index 1"),
        "the error must name the offending index, got: {err}"
    );
}

#[test]
fn ndiffs_display_shows_the_evidence_not_just_the_integer() {
    let y = walk(180, 0.41);
    let shown = ndiffs(&y, NdiffsTest::Kpss, 0.05, 2)
        .expect("ndiffs succeeds")
        .to_string();
    assert!(shown.starts_with("ndiffs(kpss) = 1 ["), "got: {shown}");
    assert!(shown.contains("d = 0: stat ="), "got: {shown}");
    assert!(shown.contains("d = 1: stat ="), "got: {shown}");
}

// ---------------------------------------------------------------- box-cox

#[test]
fn box_cox_llf_equals_its_textbook_definition() {
    // The implementation drops the -1/lambda offset (it cancels in a
    // variance); check that against the literal transform.
    let y = multiplicative(120, 0.31);
    let n = y.len() as f64;
    for lambda in [-1.5, -0.5, 0.25, 0.75, 1.0, 1.75] {
        let w: Vec<f64> = y.iter().map(|&v| (v.powf(lambda) - 1.0) / lambda).collect();
        let mean = w.iter().sum::<f64>() / n;
        let var = w.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let sum_log = y.iter().map(|&v| v.ln()).sum::<f64>();
        let expected = (lambda - 1.0) * sum_log - 0.5 * n * var.ln();
        let got = box_cox_llf(&y, lambda).expect("llf succeeds");
        let rel = ((got - expected) / expected).abs();
        assert!(
            rel < 1e-12,
            "lambda = {lambda}: {got} vs {expected} ({rel:e})"
        );
    }
}

#[test]
fn box_cox_llf_is_smooth_through_zero() {
    // The lambda = 0 branch must be the limit of the lambda != 0 branch,
    // not a jump: this is where a naive exp() implementation breaks down.
    let y = multiplicative(120, 0.31);
    let at_zero = box_cox_llf(&y, 0.0).expect("llf at 0");
    for eps in [1e-6, 1e-8, 1e-10, 1e-12] {
        for signed in [eps, -eps] {
            let got = box_cox_llf(&y, signed).expect("llf near 0");
            assert!(
                (got - at_zero).abs() < 1e-4,
                "lambda = {signed:e}: {got} vs {at_zero} at 0"
            );
        }
    }
}

#[test]
fn mle_lambda_is_scale_invariant_and_shifts_the_likelihood_exactly() {
    // l(lambda; c y) = l(lambda; y) - n log c for every lambda, so the
    // argmax cannot move: a genuine invariance, not an approximation.
    let y = multiplicative(150, 0.13);
    let c = 100.0;
    let scaled: Vec<f64> = y.iter().map(|&v| v * c).collect();
    let n = y.len() as f64;

    let base = box_cox_lambda(&y, BoxCoxMethod::Mle, (-2.0, 2.0)).expect("mle");
    let big = box_cox_lambda(&scaled, BoxCoxMethod::Mle, (-2.0, 2.0)).expect("mle");
    assert!(
        (base.lambda - big.lambda).abs() < 1e-6,
        "lambda moved under rescaling: {} vs {}",
        base.lambda,
        big.lambda
    );
    let shift = big.objective - base.objective;
    assert!(
        (shift + n * c.ln()).abs() < 1e-6,
        "likelihood shift {shift} != -n log c = {}",
        -n * c.ln()
    );
}

#[test]
fn guerrero_criterion_is_scale_invariant() {
    // r_k -> c^lambda r_k under y -> c y, a common factor, so the
    // coefficient of variation — and the lambda minimising it — is
    // unchanged.
    let y = multiplicative(150, 0.13);
    let scaled: Vec<f64> = y.iter().map(|&v| v * 250.0).collect();
    for lambda in [-1.0, -0.25, 0.0, 0.5, 1.0, 1.75] {
        let a = guerrero_cv(&y, lambda, 4).expect("cv");
        let b = guerrero_cv(&scaled, lambda, 4).expect("cv");
        assert!(
            ((a - b) / a).abs() < 1e-12,
            "lambda = {lambda}: {a} vs {b} after rescaling"
        );
    }
    let base = box_cox_lambda(&y, BoxCoxMethod::Guerrero { period: 4 }, (-2.0, 2.0)).expect("g");
    let big =
        box_cox_lambda(&scaled, BoxCoxMethod::Guerrero { period: 4 }, (-2.0, 2.0)).expect("g");
    assert!((base.lambda - big.lambda).abs() < 1e-6);
    assert!(((base.objective - big.objective) / base.objective).abs() < 1e-12);
}

#[test]
fn both_methods_recover_the_transform_the_data_were_built_with() {
    // Multiplicative noise on a growing level -> the log (lambda near 0);
    // additive noise on the same level -> no transform (lambda near 1).
    let mult = multiplicative(200, 0.13);
    let add = additive(200, 0.29);

    for method in [BoxCoxMethod::Mle, BoxCoxMethod::Guerrero { period: 4 }] {
        let m = box_cox_lambda(&mult, method, (-2.0, 2.0)).expect("mult");
        assert!(
            m.lambda.abs() < 0.35,
            "{}: multiplicative noise should want the log, got {}",
            method.code(),
            m.lambda
        );
        let a = box_cox_lambda(&add, method, (-2.0, 2.0)).expect("add");
        assert!(
            (a.lambda - 1.0).abs() < 0.35,
            "{}: additive noise should want no transform, got {}",
            method.code(),
            a.lambda
        );
        assert!(!m.at_bound && !a.at_bound, "these optima are interior");
    }
}

#[test]
fn the_reported_optimum_beats_a_dense_grid() {
    // The defining property of the answer, independent of the optimiser.
    let y = multiplicative(150, 0.41);
    let mle = box_cox_lambda(&y, BoxCoxMethod::Mle, (-2.0, 2.0)).expect("mle");
    let guer = box_cox_lambda(&y, BoxCoxMethod::Guerrero { period: 4 }, (-2.0, 2.0)).expect("g");
    for k in 0..=400 {
        let lambda = -2.0 + 4.0 * (k as f64) / 400.0;
        let llf = box_cox_llf(&y, lambda).expect("llf");
        assert!(
            llf <= mle.objective + 1e-9,
            "grid lambda {lambda} beats the MLE: {llf} > {}",
            mle.objective
        );
        let cv = guerrero_cv(&y, lambda, 4).expect("cv");
        assert!(
            cv >= guer.objective - 1e-9,
            "grid lambda {lambda} beats Guerrero: {cv} < {}",
            guer.objective
        );
    }
}

#[test]
fn a_binding_bound_is_reported_as_binding() {
    // The unconstrained optimum for this series is near 0, so a search
    // restricted to [0.9, 1.0] must return the lower bound and say so.
    let y = multiplicative(150, 0.13);
    let r = box_cox_lambda(&y, BoxCoxMethod::Mle, (0.9, 1.0)).expect("mle");
    assert_eq!(r.lambda, 0.9, "the constrained optimum is the bound itself");
    assert!(r.at_bound);
    assert!(
        r.interpretation.contains("constrained"),
        "got: {}",
        r.interpretation
    );
}

#[test]
fn box_cox_names_the_first_non_positive_observation() {
    let mut y = multiplicative(60, 0.13);
    y[17] = 0.0;
    y[41] = -3.5;
    let err = box_cox_lambda(&y, BoxCoxMethod::Mle, (-2.0, 2.0)).unwrap_err();
    match err {
        AdvisorError::NonPositive {
            index,
            value,
            count,
            ..
        } => {
            assert_eq!(index, 17);
            assert_eq!(value, 0.0);
            assert_eq!(count, 2);
        }
        other => panic!("expected NonPositive, got {other:?}"),
    }
    let msg = err.to_string();
    assert!(msg.contains("y[17]"), "the index must be named: {msg}");
    for hint in ["shift", "log1p", "Yeo-Johnson"] {
        assert!(msg.contains(hint), "the fix must be taught ({hint}): {msg}");
    }
    // Same contract on the two objective functions.
    assert!(matches!(
        box_cox_llf(&y, 0.5),
        Err(AdvisorError::NonPositive { index: 17, .. })
    ));
    assert!(matches!(
        guerrero_cv(&y, 0.5, 2),
        Err(AdvisorError::NonPositive { index: 17, .. })
    ));
}

#[test]
fn box_cox_rejects_impossible_settings() {
    let y = multiplicative(60, 0.13);
    assert!(matches!(
        box_cox_lambda(&y, BoxCoxMethod::Mle, (2.0, -2.0)),
        Err(AdvisorError::InvalidBounds { .. })
    ));
    assert!(matches!(
        box_cox_lambda(&y, BoxCoxMethod::Mle, (0.0, 0.0)),
        Err(AdvisorError::InvalidBounds { .. })
    ));
    assert!(matches!(
        box_cox_lambda(&y, BoxCoxMethod::Mle, (f64::NAN, 2.0)),
        Err(AdvisorError::InvalidBounds { .. })
    ));
    // period < 2 and "fewer than two complete groups" are both unusable.
    assert!(matches!(
        box_cox_lambda(&y, BoxCoxMethod::Guerrero { period: 1 }, (-2.0, 2.0)),
        Err(AdvisorError::InvalidPeriod { period: 1, .. })
    ));
    assert!(matches!(
        box_cox_lambda(&y, BoxCoxMethod::Guerrero { period: 40 }, (-2.0, 2.0)),
        Err(AdvisorError::InvalidPeriod { period: 40, .. })
    ));
    assert!(matches!(
        box_cox_llf(&y, f64::INFINITY),
        Err(AdvisorError::InvalidLambda { .. })
    ));
    // A constant series has no variance to stabilise.
    let flat = vec![3.0; 40];
    assert!(matches!(
        box_cox_lambda(&flat, BoxCoxMethod::Mle, (-2.0, 2.0)),
        Err(AdvisorError::Diag(DiagError::ConstantSeries { .. }))
    ));
    // Non-finite input is caught before anything is transformed.
    let nan = [1.0, 2.0, f64::INFINITY, 4.0, 5.0];
    assert!(matches!(
        box_cox_lambda(&nan, BoxCoxMethod::Mle, (-2.0, 2.0)),
        Err(AdvisorError::Diag(DiagError::NonFinite { index: 2, .. }))
    ));
}

#[test]
fn box_cox_display_states_the_method_and_the_objective() {
    let y = multiplicative(150, 0.13);
    let shown = box_cox_lambda(&y, BoxCoxMethod::Mle, (-2.0, 2.0))
        .expect("mle")
        .to_string();
    assert!(
        shown.starts_with("box_cox_lambda(mle): lambda ="),
        "{shown}"
    );
    assert!(shown.contains("objective ="), "{shown}");
    let shown = box_cox_lambda(&y, BoxCoxMethod::Guerrero { period: 4 }, (-2.0, 2.0))
        .expect("guerrero")
        .to_string();
    assert!(shown.starts_with("box_cox_lambda(guerrero):"), "{shown}");
}
