//! Property and validation-error tests.
//!
//! The DYNAMIC probit (Kauppi & Saikkonen 2008) has NO statsmodels reference,
//! so it is validated PROPERTY-ONLY here:
//!
//! * RECOVERY — on data simulated from a known dynamic-probit DGP the mean
//!   estimates of `rho` and `b` (over independent Monte-Carlo replications)
//!   sit within finite-sample bands of the truth.
//! * NESTING / FIT — the dynamic model nests the static probit at `rho = 0`,
//!   so its maximized log-likelihood is at least the static model's; on
//!   genuinely persistent data it is strictly larger by a clear margin.
//!
//! Plus the validation-layer error tests for both estimators.
//!
//! All randomness is the library's seeded Philox stream (`tsecon_rng`), so the
//! numbers are reproducible run to run.

use tsecon_recession::{fit_dynamic_probit, fit_static, Link, RecessionError};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

/// Standard normal draw via the inverse-CDF of a Philox uniform.
fn gaussian(s: &mut Stream) -> f64 {
    let u = s.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Simulate the dynamic-probit DGP over `n` periods:
///
/// ```text
/// x_t     = px * x_{t-1} + noise         (a persistent AR(1) predictor)
/// index_t = w + b * x_t + rho * index_{t-1},   index_{-1} = w / (1 - rho)
/// y_t     = 1{ index_t + eps_t > 0 },    eps_t ~ N(0, 1)  =>  P(y=1)=Phi(index)
/// ```
///
/// Returns `(y, x)`.
fn simulate_dynamic(
    s: &mut Stream,
    n: usize,
    w: f64,
    b: f64,
    rho: f64,
    px: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0_f64; n];
    for t in 1..n {
        x[t] = px * x[t - 1] + gaussian(s);
    }
    let mut y = vec![0.0_f64; n];
    let mut prev = w / (1.0 - rho);
    for t in 0..n {
        let index = w + b * x[t] + rho * prev;
        let latent = index + gaussian(s);
        y[t] = if latent > 0.0 { 1.0 } else { 0.0 };
        prev = index;
    }
    (y, x)
}

#[test]
fn dynamic_probit_recovers_rho_and_b() {
    // Known DGP; recover (b, rho) on average across replications.
    let (w, b, rho, px) = (-0.3, 1.0, 0.6, 0.5);
    let n = 1000;
    let reps = 30;

    let mut sum_rho = 0.0;
    let mut sum_b = 0.0;
    let mut used = 0usize;
    for r in 0..reps {
        let mut s = Stream::new(0xD1FF_0000 ^ r as u64);
        let (y, x) = simulate_dynamic(&mut s, n, w, b, rho, px);
        // Skip the rare degenerate draw (all-0 / all-1); does not happen here.
        let fit = match fit_dynamic_probit(&y, &[x]) {
            Ok(f) => f,
            Err(_) => continue,
        };
        sum_rho += fit.rho;
        sum_b += fit.beta[0];
        used += 1;
    }
    assert!(
        used >= reps - 2,
        "too many failed replications: {used}/{reps}"
    );
    let mean_rho = sum_rho / used as f64;
    let mean_b = sum_b / used as f64;

    // Finite-sample bands: dynamic-probit ML is consistent but biased in finite
    // samples, so the bands are generous (they are the Monte-Carlo claim, not a
    // point identity).
    assert!(
        (mean_rho - rho).abs() < 0.15,
        "mean rho_hat = {mean_rho:.4} not within 0.15 of {rho}"
    );
    assert!(
        (mean_b - b).abs() < 0.30,
        "mean b_hat = {mean_b:.4} not within 0.30 of {b}"
    );
    // rho is estimated as clearly positive (persistence detected).
    assert!(
        mean_rho > 0.25,
        "mean rho_hat = {mean_rho:.4} not clearly positive"
    );
}

#[test]
fn dynamic_loglik_beats_static_on_persistent_data() {
    // On a strongly persistent dynamic-probit DGP the dynamic model should fit
    // strictly better than the static probit with the same covariate.
    let mut s = Stream::new(0xBEEF_1234);
    let (y, x) = simulate_dynamic(&mut s, 1200, -0.3, 1.0, 0.7, 0.5);

    let dynamic = fit_dynamic_probit(&y, std::slice::from_ref(&x)).expect("dynamic fit");
    let c = vec![1.0; y.len()];
    let static_fit = fit_static(&y, &[c, x], Link::Probit).expect("static fit");

    // Nesting: the dynamic model contains the static one at rho = 0, so its
    // maximized log-likelihood cannot be materially lower.
    assert!(
        dynamic.loglik >= static_fit.loglik - 1e-4,
        "dynamic llf {} below static llf {} (nesting violated)",
        dynamic.loglik,
        static_fit.loglik
    );
    // On persistent data the improvement is real and sizable.
    assert!(
        dynamic.loglik > static_fit.loglik + 1.0,
        "dynamic llf {} did not clearly beat static llf {}",
        dynamic.loglik,
        static_fit.loglik
    );
    // And the estimated persistence is substantial.
    assert!(
        dynamic.rho > 0.3,
        "rho_hat = {} not persistent",
        dynamic.rho
    );
}

#[test]
fn dynamic_fit_summary_is_well_formed() {
    let mut s = Stream::new(0x0101_0101);
    let (y, x) = simulate_dynamic(&mut s, 600, -0.2, 1.0, 0.5, 0.5);
    let fit = fit_dynamic_probit(&y, &[x]).expect("dynamic fit");

    assert_eq!(fit.params.len(), 3); // [w, b, rho]
    assert_eq!(fit.bse.len(), 3);
    assert_eq!(fit.zstats.len(), 3);
    assert!(fit.rho.abs() < 1.0);
    assert!(fit.bse.iter().all(|&se| se.is_finite() && se > 0.0));
    assert!(fit.fitted.iter().all(|&p| (0.0..=1.0).contains(&p)));
    assert!((0.0..=1.0).contains(&fit.pseudo_r2));
    assert!(fit.loglik < 0.0);
}

// ---------------------------------------------------------------------------
// Validation-layer error tests.
// ---------------------------------------------------------------------------

#[test]
fn rejects_empty_response() {
    let err = fit_static(&[], &[vec![]], Link::Probit).unwrap_err();
    assert!(matches!(err, RecessionError::EmptyInput { .. }));
}

#[test]
fn rejects_no_regressors() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let err = fit_static(&y, &[], Link::Logit).unwrap_err();
    assert!(matches!(err, RecessionError::NoRegressors));
    let err_dyn = fit_dynamic_probit(&y, &[]).unwrap_err();
    assert!(matches!(err_dyn, RecessionError::NoRegressors));
}

#[test]
fn rejects_dimension_mismatch() {
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0];
    let short = vec![1.0, 1.0, 1.0, 1.0]; // length 4 != 5
    let err = fit_static(&y, std::slice::from_ref(&short), Link::Probit).unwrap_err();
    assert!(matches!(
        err,
        RecessionError::DimensionMismatch {
            expected: 5,
            got: 4,
            ..
        }
    ));
    let err_dyn = fit_dynamic_probit(&y, &[short]).unwrap_err();
    assert!(matches!(err_dyn, RecessionError::DimensionMismatch { .. }));
}

#[test]
fn rejects_non_binary_response() {
    let y = vec![0.0, 1.0, 0.5, 1.0, 0.0, 1.0]; // 0.5 is not a valid indicator
    let c = vec![1.0; 6];
    let err = fit_static(&y, &[c], Link::Probit).unwrap_err();
    match err {
        RecessionError::NonBinaryResponse { index, value } => {
            assert_eq!(index, 2);
            assert_eq!(value, 0.5);
        }
        other => panic!("expected NonBinaryResponse, got {other:?}"),
    }
}

#[test]
fn rejects_non_finite_inputs() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let bad = vec![1.0, f64::NAN, 1.0, 1.0];
    let err = fit_static(&y, &[bad], Link::Probit).unwrap_err();
    assert!(matches!(err, RecessionError::NonFinite { .. }));

    let bad_y = vec![0.0, 1.0, f64::INFINITY, 1.0];
    let c = vec![1.0; 4];
    let err2 = fit_static(&bad_y, &[c], Link::Logit).unwrap_err();
    assert!(matches!(err2, RecessionError::NonFinite { .. }));
}

#[test]
fn rejects_degenerate_response() {
    let y = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // no recessions at all
    let c = vec![1.0; 5];
    let spread = vec![0.2, -0.1, 0.4, -0.3, 0.5];
    let err = fit_static(&y, &[c, spread], Link::Probit).unwrap_err();
    assert!(matches!(err, RecessionError::Degenerate { ones: 0, n: 5 }));
}

#[test]
fn rejects_insufficient_degrees_of_freedom() {
    // n = 2 observations, k = 2 parameters (const + spread): no residual dof.
    let y = vec![0.0, 1.0];
    let c = vec![1.0, 1.0];
    let spread = vec![0.5, -0.5];
    let err = fit_static(&y, &[c, spread], Link::Probit).unwrap_err();
    assert!(matches!(
        err,
        RecessionError::DegreesOfFreedom { n: 2, k: 2 }
    ));
}

#[test]
fn detects_complete_separation() {
    // y is perfectly predicted by the sign of the single predictor, so the MLE
    // coefficient diverges — there is no finite maximum.
    let n = 24;
    let mut c = Vec::with_capacity(n);
    let mut spread = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for t in 0..n {
        c.push(1.0);
        // Predictor strictly separated away from zero on each side.
        let v = if t % 2 == 0 {
            1.0 + t as f64
        } else {
            -1.0 - t as f64
        };
        spread.push(v);
        y.push(if v > 0.0 { 1.0 } else { 0.0 });
    }
    let err = fit_static(&y, &[c, spread], Link::Probit).unwrap_err();
    assert!(
        matches!(err, RecessionError::Separation),
        "expected Separation, got {err:?}"
    );
}

#[test]
fn error_messages_are_nonempty() {
    // The Display impl (errors-that-teach) produces a message for each variant.
    let e = fit_static(&[], &[vec![]], Link::Probit).unwrap_err();
    assert!(!format!("{e}").is_empty());
}
