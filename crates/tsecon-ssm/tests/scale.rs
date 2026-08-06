//! Scale invariance of the log-likelihood, and the empty-sample error.
//!
//! `golden.rs` pins agreement with statsmodels at one scale. It cannot see
//! a units bug, because every fixture is unit-scale: the Nile data are
//! thousands of cubic metres and the fitted variances are in the thousands.
//! The tests here vary the *scale* of the same problem and check the
//! log-likelihood against the closed form it must obey.
//!
//! # The closed form
//!
//! The filter accumulates one term per informative element,
//! `-(ln 2*pi + ln F + v^2 / F) / 2`. Rescale the observations by `c > 0`
//! and rescale every variance in the model by `c^2` (`H`, `Q`, `P_1`) and
//! every location by `c` (`a_1`, the intercepts): then every `v -> c v` and
//! every `F -> c^2 F`, so `v^2 / F` is unchanged and `ln F -> ln F + 2 ln c`.
//! Each term therefore drops by exactly `ln c`, and
//!
//! ```text
//! loglik(c y; scaled model) = loglik(y; model) - n_informative * ln c.
//! ```
//!
//! This is the Jacobian of the change of variables `y -> c y`, so it holds
//! to the last few bits and needs no external package to check against.
//!
//! One wrinkle, and it is the interesting one. Under *exact-diffuse*
//! initialization the diffuse prior `P_inf = I` is a limit
//! (`P_1 = P_* + kappa P_inf`, `kappa -> infinity`), not a variance, so it
//! does **not** scale with `c`. A diffuse element contributes
//! `-(ln 2*pi + ln F_inf) / 2` with `F_inf = Z P_inf Z'`, which is
//! scale-free — those elements do not pick up the `-ln c`. The identity
//! above is therefore exact with a proper (`Known`) initialization, and
//! carries a correction of exactly one term per diffuse element under
//! `Diffuse`. Both variants are asserted below; getting this wrong in
//! either direction would look like a scale bug.
//!
//! # What these tests would have caught
//!
//! Before the fix, `F_*` was compared against an *absolute* `1e-10`. Any
//! series whose prediction variances fell below that had its observations
//! discarded as uninformative, and the two failure modes are both silent:
//!
//! * **partial.** When only the converged `F` falls under the floor, the
//!   first few observations are kept and the rest are dropped, so `loglik`
//!   comes back as a plausible finite number computed from a truncated
//!   sample. Measured on the sweep below at `c = 1e-5`: `772.13` against a
//!   true `1581.34`, wrong by 51%, with no error and no NaN. This one is
//!   worse than the collapse, because nothing about the output looks odd.
//! * **total.** When every `F` falls under the floor, every observation is
//!   dropped and `loglik` is exactly `0.0` — an empty sum returned as a
//!   success, which a caller can optimize and report as a fit.
//!
//! [`loglik_shifts_by_minus_n_ln_c_under_rescaling`] catches both, because
//! it checks the *value* against the closed-form line rather than checking
//! that the filter returned without complaining.
//!
//! # Why the single-state tests are not enough
//!
//! Every test above uses `m = 1`. That is a blind spot, and it let a second
//! units bug through: with one state, `||Z||_1^2 max_j P_jj` and
//! `|Z|' |P| |Z|` are the *same number*, so a reference scale that looks at
//! states the observation equation does not load on is indistinguishable
//! from one that does not. The first version of the relative tolerance used
//! the former, and on `m = 1` models — these ones — it was exactly right.
//!
//! On `m >= 2` it is not. `F_{*,i}` is a variance along the single
//! direction `Z_i`; measuring it against the largest variance over *all*
//! states lets one large-variance state veto every observation in the
//! sample, and states in different units are routine (a regression
//! coefficient beside a level, an approximate-diffuse prior of `1e6`
//! beside a unit-scale level). The three failures that produced —
//! a nuisance state vetoing observations, the same on the diffuse side, and
//! the partial silent drop coming back through a *decaying* nuisance state
//! — are pinned by the four multi-state tests at the end of this file:
//!
//! * [`a_nuisance_state_cannot_change_the_likelihood`] and
//!   [`a_decaying_nuisance_state_cannot_silently_drop_observations`] —
//!   a state `Z` does not load on, and `T` cannot feed into the states it
//!   does, is provably irrelevant to `y`; the likelihood must not move
//!   however large its variance.
//! * [`loglik_is_invariant_to_rescaling_one_state_coordinate`] — two models
//!   related by `alpha~ = S^{-1} alpha`, `S = diag(1, x)`, are the same
//!   model written in different units, so they must return the same number.
//! * [`the_diffuse_period_is_invariant_to_rescaling_one_state_coordinate`]
//!   — the same for the *length* of the diffuse period, which is a rank
//!   count and cannot depend on the units the states are written in.
//!
//! These four are the tests that would have caught the defect.

mod common;

use common::Lcg;
use tsecon_linalg::faer::Mat;
use tsecon_ssm::{Initialization, LinearGaussianSSM, SsmError};

/// A deterministic unit-scale local-level series: a random-walk level plus
/// observation noise, from the seeded LCG (no dependency on `tsecon-rng`).
fn local_level_series(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut level = 0.0;
    (0..n)
        .map(|_| {
            level += 0.7 * rng.symmetric();
            level + 0.4 * rng.symmetric()
        })
        .collect()
}

/// A slice as an `n x 1` observation matrix.
fn col(y: &[f64]) -> Mat<f64> {
    Mat::from_fn(y.len(), 1, |i, _| y[i])
}

/// Local level with a *proper* (`Known`) initialization, every variance
/// scaled by `c^2` and every location by `c`. No diffuse elements, so the
/// closed-form shift is exact for all `n` observations.
fn scaled_known_local_level(c: f64) -> LinearGaussianSSM {
    let one = Mat::from_fn(1, 1, |_, _| 1.0);
    LinearGaussianSSM::builder(1, 1, 1)
        .z(one.clone())
        .h(Mat::from_fn(1, 1, |_, _| 0.16 * c * c))
        .t(one.clone())
        .r(one)
        .q(Mat::from_fn(1, 1, |_, _| 0.49 * c * c))
        .initialization(Initialization::Known {
            a1: vec![0.0],
            p1: Mat::from_fn(1, 1, |_, _| 2.0 * c * c),
        })
        .build()
        .expect("the scaled local level is a valid model")
}

/// **The scale sweep.** Over twenty-two decades of prediction variance,
/// the log-likelihood must sit on the closed-form line
/// `loglik(c y) = loglik(y) - n ln c` rather than degrading.
///
/// The sweep runs `c` from `1e5` down to `1e-6`. This model's steady-state
/// prediction variance is `F = P + H` with `P` the positive root of
/// `P^2 - Q P - Q H = 0`, so `F ~ 0.777 c^2`: the sweep spans `F` from
/// about `7.8e9` down to `7.8e-13`, straddling the old absolute `1e-10`
/// floor from both sides. The first point the old floor got wrong is
/// `c = 1e-5` (`F ~ 7.8e-11`), and it got it wrong *partially* — `P_1`
/// starts at `2e-10`, above the floor, so the opening observations were
/// kept and the rest dropped, returning `772.13` where the truth is
/// `1581.34`. Deeper in, the sample empties completely and the answer is
/// exactly `0.0`.
#[test]
fn loglik_shifts_by_minus_n_ln_c_under_rescaling() {
    let y = local_level_series(11, 150);
    let n = y.len() as f64;

    let base = scaled_known_local_level(1.0)
        .loglike(col(&y).as_ref())
        .expect("the unit-scale likelihood is well defined");
    assert!(base.is_finite());

    let mut worst: f64 = 0.0;
    for k in -5..=6 {
        let c = 10f64.powi(-k);
        let ys: Vec<f64> = y.iter().map(|v| v * c).collect();
        let got = scaled_known_local_level(c)
            .loglike(col(&ys).as_ref())
            .unwrap_or_else(|e| panic!("c = {c:e}: the filter failed with {e}"));
        let want = base - n * c.ln();

        assert!(got.is_finite(), "c = {c:e}: loglik is not finite ({got})");
        assert_ne!(
            got, 0.0,
            "c = {c:e}: loglik collapsed to exactly 0.0 — every element was \
             skipped, which is the absolute-floor defect"
        );
        // Relative to the magnitude of the shift itself, so the tolerance
        // does not quietly loosen as |loglik| grows with the scale.
        let rel = (got - want).abs() / want.abs().max(1.0);
        worst = worst.max(rel);
        assert!(
            rel <= 1e-12,
            "c = {c:e}: loglik {got} vs the closed form {want} (rel {rel:e})"
        );
    }
    println!("known-init scale sweep: worst relative deviation {worst:e}");
}

/// The same identity under *exact-diffuse* initialization, where it is not
/// the same identity: the one diffuse element contributes a scale-free
/// `-(ln 2*pi + ln F_inf) / 2`, so the shift is `-(n - 1) ln c`, not
/// `-n ln c`.
///
/// This pins two things at once — that the diffuse contribution really is
/// scale-free, and that *which* elements take the diffuse branch does not
/// drift with the scale of the data (`d_diffuse` is asserted constant).
#[test]
fn diffuse_loglik_shifts_by_minus_n_minus_d_ln_c() {
    let y = local_level_series(23, 120);
    let n = y.len() as f64;

    let base_out = LinearGaussianSSM::local_level(0.16, 0.49)
        .unwrap()
        .filter(col(&y).as_ref())
        .expect("the unit-scale diffuse filter succeeds");
    // p = 1, so one diffuse period is exactly one diffuse element.
    assert_eq!(base_out.d_diffuse, 1);

    for k in -5..=6 {
        let c = 10f64.powi(-k);
        let ys: Vec<f64> = y.iter().map(|v| v * c).collect();
        let out = LinearGaussianSSM::local_level(0.16 * c * c, 0.49 * c * c)
            .unwrap()
            .filter(col(&ys).as_ref())
            .unwrap_or_else(|e| panic!("c = {c:e}: the diffuse filter failed with {e}"));

        assert_eq!(
            out.d_diffuse, 1,
            "c = {c:e}: the diffuse period changed length with the data scale"
        );
        let want = base_out.loglik - (n - 1.0) * c.ln();
        let rel = (out.loglik - want).abs() / want.abs().max(1.0);
        assert!(
            rel <= 1e-12,
            "c = {c:e}: diffuse loglik {} vs the closed form {want} (rel {rel:e})",
            out.loglik
        );
    }
}

/// The single case the bug report reproduced: a series whose prediction
/// variances all sit below the old `1e-10` floor. It must not come back as
/// a successful `0.0`.
///
/// The assertion is not merely "nonzero" — the value is checked against the
/// closed form, so a wrong-but-nonzero answer fails too.
#[test]
fn a_tiny_variance_series_returns_a_correct_loglik_not_zero() {
    let y = local_level_series(5, 80);
    let n = y.len() as f64;
    let c = 1e-7; // prediction variances land around 1e-15.

    let base = scaled_known_local_level(1.0)
        .loglike(col(&y).as_ref())
        .unwrap();
    let ys: Vec<f64> = y.iter().map(|v| v * c).collect();
    let got = scaled_known_local_level(c)
        .loglike(col(&ys).as_ref())
        .expect("a small-variance series is filterable, not an error");

    assert_ne!(got, 0.0, "the silent-zero path is back");
    let want = base - n * c.ln();
    let rel = (got - want).abs() / want.abs();
    assert!(
        rel <= 1e-12,
        "tiny-variance loglik {got} vs the closed form {want} (rel {rel:e})"
    );
    // And it is a large number, so "nonzero" above is not a near-miss.
    assert!(got > 1000.0, "expected a large positive loglik, got {got}");
}

/// The smoother rides on the same per-element branch decisions, so it must
/// survive the same rescaling. Smoothed states are equivariant: smoothing
/// `c y` under the `c`-scaled model returns `c` times the smoothed states
/// of `y`, and `c^2` times their covariances.
#[test]
fn smoothed_moments_are_equivariant_under_rescaling() {
    let y = local_level_series(31, 90);
    let c = 1e-6;

    let base = scaled_known_local_level(1.0)
        .smooth(col(&y).as_ref())
        .unwrap();
    let ys: Vec<f64> = y.iter().map(|v| v * c).collect();
    let scaled = scaled_known_local_level(c)
        .smooth(col(&ys).as_ref())
        .expect("smoothing a small-variance series must not fail");

    for t in 0..y.len() {
        let want_mean = c * base.smoothed_state[t][0];
        let got_mean = scaled.smoothed_state[t][0];
        assert!(
            (got_mean - want_mean).abs() <= 1e-10 * want_mean.abs().max(1e-30),
            "smoothed_state[{t}]: {got_mean} vs {want_mean}"
        );
        let want_cov = c * c * base.smoothed_state_cov[t][(0, 0)];
        let got_cov = scaled.smoothed_state_cov[t][(0, 0)];
        assert!(
            (got_cov - want_cov).abs() <= 1e-10 * want_cov.abs(),
            "smoothed_state_cov[{t}]: {got_cov} vs {want_cov}"
        );
    }
}

/// A genuinely degenerate model — zero observation noise, zero state noise,
/// zero initial covariance — makes every prediction variance exactly zero,
/// so every observed element is skipped and the log-likelihood is an empty
/// sum. That must be an error, never `Ok(0.0)`.
///
/// This is the residual case the relative floor cannot rescue: the problem
/// is the model, not the units, and the message has to say so.
#[test]
fn an_entirely_uninformative_sample_is_an_error_not_zero() {
    let one = Mat::from_fn(1, 1, |_, _| 1.0);
    let degenerate = LinearGaussianSSM::builder(1, 1, 1)
        .z(one.clone())
        .h(Mat::zeros(1, 1))
        .t(one.clone())
        .r(one)
        .q(Mat::zeros(1, 1))
        .initialization(Initialization::Known {
            a1: vec![0.0],
            p1: Mat::zeros(1, 1),
        })
        .build()
        .unwrap();

    let y = local_level_series(3, 40);
    let err = degenerate
        .loglike(col(&y).as_ref())
        .expect_err("a fully degenerate model must not report a likelihood");
    assert_eq!(err, SsmError::NoInformation);

    // The message has to name the cause and steer away from the wrong fix.
    let msg = err.to_string();
    for needle in [
        "uninformative",
        "prediction variance",
        "empty sum",
        "rescaling y will not change this",
        "degenerate model",
    ] {
        assert!(
            msg.contains(needle),
            "the error message must contain {needle:?}: {msg}"
        );
    }
}

/// Rescaling never turns the empty-sample error on or off: it is a property
/// of the model, and the message says as much. A model that errors at unit
/// scale errors at every scale, and one that works at unit scale works at
/// every scale — which is the whole point of a relative tolerance.
#[test]
fn the_empty_sample_verdict_is_scale_invariant() {
    let y = local_level_series(3, 40);
    let one = Mat::from_fn(1, 1, |_, _| 1.0);

    for k in -4..=8 {
        let c = 10f64.powi(-k);
        let ys: Vec<f64> = y.iter().map(|v| v * c).collect();

        let degenerate = LinearGaussianSSM::builder(1, 1, 1)
            .z(one.clone())
            .h(Mat::zeros(1, 1))
            .t(one.clone())
            .r(one.clone())
            .q(Mat::zeros(1, 1))
            .initialization(Initialization::Known {
                a1: vec![0.0],
                p1: Mat::zeros(1, 1),
            })
            .build()
            .unwrap();
        assert_eq!(
            degenerate.loglike(col(&ys).as_ref()),
            Err(SsmError::NoInformation),
            "c = {c:e}: the degenerate model stopped erroring"
        );

        assert!(
            scaled_known_local_level(c)
                .loglike(col(&ys).as_ref())
                .is_ok(),
            "c = {c:e}: a well-posed model started erroring"
        );
    }
}

/// An all-missing `y` is *not* the empty-sample error. NaN is the
/// documented encoding for "missing", so running the recursions with
/// nothing to condition on is an explicit request, and the log-likelihood
/// of no observations really is zero. Keeping these two cases distinct is
/// deliberate; this test is what stops them being merged by accident.
#[test]
fn an_all_missing_series_is_still_a_legitimate_zero() {
    let model = LinearGaussianSSM::local_level(0.16, 0.49).unwrap();
    let y = vec![f64::NAN; 25];
    let out = model
        .filter(col(&y).as_ref())
        .expect("an all-missing series is a valid, if empty, filtering problem");
    assert_eq!(out.loglik, 0.0);
    // The recursions still ran: the predicted covariance grew by Q each
    // period, so this is a real forward pass and not an early exit.
    assert_eq!(out.predicted_state.len(), y.len() + 1);
}

/// A partially informative sample is not an error. Only a *completely*
/// empty effective sample is, so an ordinary missing stretch — or one
/// genuinely singular element among many good ones — still filters.
#[test]
fn a_partially_missing_series_still_filters() {
    let model = LinearGaussianSSM::local_level(0.16, 0.49).unwrap();
    let mut y = local_level_series(17, 60);
    for v in y.iter_mut().take(30).skip(10) {
        *v = f64::NAN;
    }
    let out = model.filter(col(&y).as_ref()).unwrap();
    assert!(out.loglik.is_finite() && out.loglik != 0.0);
}

// ---------------------------------------------------------------------
// Multi-state: the class of scale bug that `m = 1` cannot see.
// ---------------------------------------------------------------------

/// [`scaled_known_local_level`] with `c = 1`, plus a second state that
/// **cannot influence `y`**: `Z = [1, 0]` does not load on it and `T` is
/// diagonal, so it never feeds into state 1 either. Its initial variance is
/// `v2` and it evolves as `alpha_2,t+1 = t22 alpha_2,t + eta_2` with
/// `Var(eta_2) = q2`.
///
/// Whatever `(v2, t22, q2)` are, the observations have exactly the law of
/// the one-state model — so the log-likelihood is pinned to the last bit,
/// with no tolerance argument to make.
fn local_level_plus_nuisance(v2: f64, t22: f64, q2: f64) -> LinearGaussianSSM {
    LinearGaussianSSM::builder(1, 2, 2)
        .z(Mat::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 0.0 }))
        .h(Mat::from_fn(1, 1, |_, _| 0.16))
        .t(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (1, 1) => t22,
            _ => 0.0,
        }))
        .r(Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }))
        .q(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 0.49,
            (1, 1) => q2,
            _ => 0.0,
        }))
        .initialization(Initialization::Known {
            a1: vec![0.0, 0.0],
            p1: Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => 2.0,
                (1, 1) => v2,
                _ => 0.0,
            }),
        })
        .build()
        .expect("the local level plus a nuisance state is a valid model")
}

/// **A state the observation equation does not load on cannot change the
/// answer** — not by a bit, and not at any variance.
///
/// This is the multi-state analogue of the scale sweep: instead of moving
/// the scale of `y`, it moves the scale of a state that provably does not
/// enter `y`, over fifteen decades. A reference scale taken over all states
/// (`||Z||_1^2 max_j P_jj`) fails here: at `v2 = 1e10` it drops the tail of
/// the sample, and past `1e11` it drops everything and raises
/// `NoInformation` on a model whose observed states are entirely healthy
/// (`H = 0.16`, `Q_11 = 0.49`, `P_1,11 = 2`).
#[test]
fn a_nuisance_state_cannot_change_the_likelihood() {
    let y = local_level_series(11, 100);
    let truth = scaled_known_local_level(1.0)
        .loglike(col(&y).as_ref())
        .expect("the one-state likelihood is well defined");

    for k in 0..=15 {
        let v2 = 10f64.powi(k);
        let model = local_level_plus_nuisance(v2, 1.0, 0.0);
        let got = model
            .loglike(col(&y).as_ref())
            .unwrap_or_else(|e| panic!("v2 = 1e{k}: an irrelevant state broke the filter: {e}"));
        let rel = (got - truth).abs() / truth.abs();
        assert!(
            rel <= 1e-13,
            "v2 = 1e{k}: loglik {got} vs the one-state truth {truth} (rel {rel:e}) — \
             a state y does not depend on moved the likelihood"
        );

        // And the Joseph-form matrix filter, which has no rank floor at
        // all, agrees: the univariate path is not merely self-consistent.
        let oracle = model
            .filter_matrix(col(&y).as_ref())
            .expect("the matrix filter accepts this proper initialization")
            .loglik;
        let rel_oracle = (got - oracle).abs() / oracle.abs();
        assert!(
            rel_oracle <= 1e-10,
            "v2 = 1e{k}: univariate {got} vs Joseph-form matrix filter {oracle} \
             (rel {rel_oracle:e})"
        );
    }
}

/// The same nuisance state, but *decaying* (`t22 = 0.5`, `q2 = 0`), so its
/// variance falls through the whole sample instead of sitting still.
///
/// This is the shape that brings back the original defect's worst symptom.
/// When the reference scale tracks the largest variance over all states, a
/// falling nuisance variance means observations cross the threshold
/// *mid-sample*: the opening ones are kept and the rest are dropped, and
/// the filter returns a finite, plausible, wrong number with no error and
/// no NaN — exactly the partial silent drop this whole change exists to
/// remove. Measured with that scale on this fixture: `-93.408` at
/// `v2 = 1e12` and `-92.067` at `1e13`, against a truth of `-96.189`.
#[test]
fn a_decaying_nuisance_state_cannot_silently_drop_observations() {
    let y = local_level_series(11, 100);
    let truth = scaled_known_local_level(1.0)
        .loglike(col(&y).as_ref())
        .expect("the one-state likelihood is well defined");

    for k in 0..=15 {
        let v2 = 10f64.powi(k);
        let got = local_level_plus_nuisance(v2, 0.5, 0.0)
            .loglike(col(&y).as_ref())
            .unwrap_or_else(|e| panic!("v2 = 1e{k}: a decaying nuisance state errored: {e}"));
        let rel = (got - truth).abs() / truth.abs();
        assert!(
            rel <= 1e-13,
            "v2 = 1e{k}: loglik {got} vs the one-state truth {truth} (rel {rel:e}) — \
             observations were dropped partway through the sample"
        );
    }
}

/// Two independent AR(1) components observed as their sum, written in
/// state coordinates `alpha~ = S^{-1} alpha` with `S = diag(1, x)`.
///
/// `x = 1` is the base model. Under the reparametrization `Z~ = Z S`,
/// `T~ = S^{-1} T S` (unchanged, `T` is diagonal), `R~ = S^{-1} R` and
/// `P_1~ = S^{-1} P_1 S^{-1}`. Only the *units* of the second state change,
/// so `y` has exactly the same law for every `x`. `T` is kept diagonal on
/// purpose: the reparametrization then leaves `T` alone, and any
/// disagreement between two `x` values is the rank test talking, not the
/// conditioning of `T~`.
fn two_component_rescaled(x: f64) -> LinearGaussianSSM {
    LinearGaussianSSM::builder(1, 2, 2)
        .z(Mat::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { x }))
        .h(Mat::from_fn(1, 1, |_, _| 0.25))
        .t(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 0.9,
            (1, 1) => 0.5,
            _ => 0.0,
        }))
        .r(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (1, 1) => 1.0 / x,
            _ => 0.0,
        }))
        .q(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 0.30,
            (1, 1) => 0.20,
            _ => 0.0,
        }))
        .initialization(Initialization::Known {
            a1: vec![0.0, 0.0],
            p1: Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => 1.5,
                (1, 1) => 0.8 / (x * x),
                _ => 0.0,
            }),
        })
        .build()
        .expect("the rescaled two-component model is valid")
}

/// **Rescaling one state coordinate is a change of units, not a change of
/// model.** The log-likelihood must not notice, over twelve decades of `x`.
///
/// This is the invariance the reference scale has to have and
/// `||Z||_1^2 max_j P_jj` does not: it mixes `Z~_2 = x` with
/// `P~_{1,11} = 1.5`, two quantities that no longer live in the same units,
/// and the product `x^2 * 1.5` runs away in one direction while
/// `P~_{2,22} = 0.8 / x^2` runs away in the other. Both tails of this sweep
/// fail under it; the cancellation-free `|Z~|' |P~| |Z~|` is invariant by
/// construction, because rescaling state `a` multiplies `|Z~_a|` and
/// divides every `|P~_{ab}|` by exactly offsetting factors.
#[test]
fn loglik_is_invariant_to_rescaling_one_state_coordinate() {
    let y = local_level_series(29, 120);
    let base = two_component_rescaled(1.0)
        .loglike(col(&y).as_ref())
        .expect("the base parametrization filters");

    let mut worst: f64 = 0.0;
    for k in -6..=6 {
        let x = 10f64.powi(k);
        let got = two_component_rescaled(x)
            .loglike(col(&y).as_ref())
            .unwrap_or_else(|e| panic!("x = 1e{k}: rewriting the state units broke it: {e}"));
        let rel = (got - base).abs() / base.abs();
        worst = worst.max(rel);
        assert!(
            rel <= 1e-11,
            "x = 1e{k}: loglik {got} vs the base parametrization {base} (rel {rel:e}) — \
             the same model in different state units gave a different answer"
        );
    }
    println!("state-coordinate rescaling: worst relative deviation {worst:e}");
}

/// The local linear trend `y_t = mu_t + beta_t`, `mu_{t+1} = mu_t + beta_t`,
/// `beta_{t+1} = beta_t`, fully diffuse, in state coordinates
/// `alpha~ = S^{-1} alpha` with `S = diag(1, s)`: `Z~ = [1, s]`,
/// `T~ = S^{-1} T S`, `R~ = S^{-1}`.
///
/// Two diffuse directions and a `T` that rotates them into `Z`'s line of
/// sight, so the diffuse period genuinely terminates — at `t = 2`, in every
/// parametrization.
fn diffuse_trend_rescaled(s: f64) -> LinearGaussianSSM {
    LinearGaussianSSM::builder(1, 2, 2)
        .z(Mat::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { s }))
        .h(Mat::from_fn(1, 1, |_, _| 0.36))
        .t(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 1) => s,
            (1, 1) => 1.0,
            _ => 0.0,
        }))
        .r(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (1, 1) => 1.0 / s,
            _ => 0.0,
        }))
        .q(Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 0.25,
            (1, 1) => 0.04,
            _ => 0.0,
        }))
        .initialization(Initialization::Diffuse)
        .build()
        .expect("the rescaled diffuse trend is valid")
}

/// **The diffuse period is a rank count, so its length cannot depend on the
/// units the states are written in.** `d_diffuse` must be `2` for every `s`.
///
/// The likelihood has one wrinkle here and it is worth stating precisely.
/// `Initialization::Diffuse` sets `P_inf = I` in each model's *own*
/// coordinates, so the two models carry diffuse priors that differ by
/// `S S'`. The exact-diffuse contribution `-(ln 2*pi + ln F_inf)/2` is
/// therefore not invariant — it is the log-density of an improper prior and
/// shifts by the Jacobian of the change of state variables:
///
/// ```text
/// loglik(s) = loglik(1) - ln|det S| = loglik(1) - ln|s|.
/// ```
///
/// Asserting that exact line, rather than plain equality, is what makes the
/// test able to fail: a filter that quietly skipped a diffuse element would
/// still get `d_diffuse` wrong *and* miss the line.
///
/// With the all-states reference scale, `s = 1e3` returned `d = 1` and
/// `s = 1e-5` returned `d = 100` — the diffuse period never terminated —
/// on the same two models. See the note on the range below.
#[test]
fn the_diffuse_period_is_invariant_to_rescaling_one_state_coordinate() {
    let y = local_level_series(7, 100);
    let base = diffuse_trend_rescaled(1.0)
        .filter(col(&y).as_ref())
        .expect("the base parametrization filters");
    assert_eq!(
        base.d_diffuse, 2,
        "two diffuse states, resolved one per step"
    );

    // Range note: this sweep stops at `1e-5` and `1e2` because outside it
    // the *representation* is genuinely lost, not the rank test. `T~` and
    // `R~` carry factors of `s` and `1/s`, so the recursions themselves
    // cancel to nothing: at `s = 1e-6` the residual `P_inf` after the
    // first update is roundoff of order `1e-16` sitting in an entry that
    // `Z~` weights by 1, against a live `F_inf` of `1e-12` — the signal is
    // below the noise before any threshold is consulted. What the sweep
    // does pin is that nothing *else* — no threshold, no reference scale —
    // narrows the range in which the identity holds.
    for k in -5..=2 {
        let s = 10f64.powi(k);
        let out = diffuse_trend_rescaled(s)
            .filter(col(&y).as_ref())
            .unwrap_or_else(|e| panic!("s = 1e{k}: rewriting the state units broke it: {e}"));

        assert_eq!(
            out.d_diffuse, 2,
            "s = 1e{k}: the diffuse period changed length ({}) when the states \
             were rewritten in different units",
            out.d_diffuse
        );
        let want = base.loglik - s.ln();
        let rel = (out.loglik - want).abs() / want.abs();
        assert!(
            rel <= 1e-9,
            "s = 1e{k}: diffuse loglik {} vs the Jacobian-shifted base {want} (rel {rel:e})",
            out.loglik
        );
    }
}
