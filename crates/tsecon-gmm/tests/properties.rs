//! Property / behavioural tests for the GMM estimators — the statistical
//! invariances that no single golden value pins:
//!
//! * exact identification collapses GMM to IV / 2SLS for *any* weight;
//! * iterated GMM's first re-weight reproduces the two-step estimator, and
//!   the full iteration converges near it in a couple of steps;
//! * the nonlinear driver recovers the analytic mean/variance method-of-
//!   moments solution;
//! * a zero-bandwidth HAC weight is rejected (it *is* the White estimator)
//!   and the automatic bandwidth is the documented Newey-West rule;
//! * the first-stage F covers exactly the instrumented regressors, equals
//!   the squared robust t when just identified, and tracks instrument
//!   strength;
//! * the first-stage F is **omitted rather than fabricated** when the
//!   instruments reproduce a regressor exactly, and a broken diagnostic never
//!   fails an otherwise-valid estimation — while an estimator that genuinely
//!   cannot handle a rank-deficient `Z` still raises its own accurate error;
//! * the exogenous/endogenous classification is scale-invariant, since its
//!   tolerance leaks into every reported degrees-of-freedom;
//! * the validation layer rejects malformed inputs.

use serde_json::Value;
use tsecon_gmm::{
    gmm_nonlinear, iterated_gmm, one_step_gmm, two_stage_least_squares, two_step_gmm, GmmError,
    GmmWeight,
};
use tsecon_hac::{ols, Kernel, SeType};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

fn load_fixture() -> Value {
    let path = format!("{}/../../fixtures/gmm.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn gaussian(stream: &mut Stream) -> f64 {
    let u = stream.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

fn fixture_design() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let fx = load_fixture();
    let y = f64s(&fx["y"]);
    let x = f64s(&fx["x"]);
    let w = f64s(&fx["w"]);
    let z1 = f64s(&fx["z1"]);
    let z2 = f64s(&fx["z2"]);
    let n = y.len();
    let cst = vec![1.0_f64; n];
    let x_cols = vec![cst.clone(), w.clone(), x];
    let z_cols = vec![cst, w, z1, z2];
    (x_cols, z_cols, y)
}

/// A just-identified DGP: `k = 2` regressors `[const, x_endog]`, instruments
/// `[const, z]`, with `x` endogenous (correlated with the error through a
/// common shock).
fn exactly_identified_data(seed: u64, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut s = Stream::new(seed);
    let cst = vec![1.0_f64; n];
    let mut z = Vec::with_capacity(n);
    let mut xend = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let zi = gaussian(&mut s);
        let v = gaussian(&mut s); // endogeneity shock
        let xi = 0.8 * zi + 0.5 * v;
        let e = 0.6 * v + gaussian(&mut s); // error correlated with v -> x endogenous
        let yi = 1.0 - 0.5 * xi + e;
        z.push(zi);
        xend.push(xi);
        y.push(yi);
    }
    let x_cols = vec![cst.clone(), xend];
    let z_cols = vec![cst, z];
    (x_cols, z_cols, y)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Z'u, the sample moment conditions (should be ~0 for exactly-identified IV).
fn moment_residuals(z_cols: &[Vec<f64>], u: &[f64]) -> Vec<f64> {
    z_cols
        .iter()
        .map(|zc| zc.iter().zip(u.iter()).map(|(z, e)| z * e).sum())
        .collect()
}

#[test]
fn exactly_identified_gmm_equals_iv_regardless_of_weight() {
    let (x_cols, z_cols, y) = exactly_identified_data(20260717, 400);

    // Two very different SPD weights, plus 2SLS / two-step / iterated.
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let skewed = vec![7.0, 0.0, 0.0, 0.3];
    let fit_i = one_step_gmm(&x_cols, &z_cols, &y, &identity, GmmWeight::Robust).unwrap();
    let fit_s = one_step_gmm(&x_cols, &z_cols, &y, &skewed, GmmWeight::Robust).unwrap();
    let fit_2sls = two_stage_least_squares(&x_cols, &z_cols, &y).unwrap();
    let fit_2step = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).unwrap();
    let fit_iter = iterated_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust, 1e-12, 50).unwrap();

    // All point estimates coincide with the (unique) IV estimator.
    for other in [&fit_s, &fit_2sls, &fit_2step, &fit_iter] {
        assert!(
            max_abs_diff(&fit_i.params, &other.params) < 1e-9,
            "exactly-identified estimators must agree: {:?} vs {:?}",
            fit_i.params,
            other.params
        );
    }

    // Defining property of just-identified IV: the moments are satisfied exactly.
    let g = moment_residuals(&z_cols, &fit_i.residuals);
    assert!(
        g.iter().all(|v| v.abs() < 1e-8),
        "Z'u should vanish for exactly-identified IV, got {g:?}"
    );

    // No over-identifying restrictions => no Hansen J-test.
    assert!(fit_i.jtest.is_none());
    assert!(fit_2step.jtest.is_none());
}

#[test]
fn iterated_one_step_reproduces_two_step() {
    // A single re-weight from the 2SLS start is exactly the two-step
    // estimator — same params, same bse, same J-test.
    let (x_cols, z_cols, y) = fixture_design();
    let two = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).unwrap();
    let one_iter = iterated_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust, 1e-15, 1).unwrap();

    assert!(max_abs_diff(&two.params, &one_iter.params) < 1e-12);
    assert!(max_abs_diff(&two.bse, &one_iter.bse) < 1e-12);
    let (j2, ji) = (two.jtest.unwrap(), one_iter.jtest.unwrap());
    assert!((j2.stat - ji.stat).abs() < 1e-12);
    assert_eq!(one_iter.steps, 2);
}

#[test]
fn iterated_gmm_converges_near_two_step() {
    let (x_cols, z_cols, y) = fixture_design();
    let two = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).unwrap();
    let iter = iterated_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust, 1e-10, 100).unwrap();

    // Converges in a handful of re-weights on well-identified data.
    assert!(
        iter.steps <= 10,
        "iterated GMM should converge quickly, took {} steps",
        iter.steps
    );
    // The iterated and two-step estimates are close (they differ only by the
    // higher-order re-weighting terms).
    assert!(
        max_abs_diff(&two.params, &iter.params) < 1e-3,
        "iterated should be near two-step: {:?} vs {:?}",
        iter.params,
        two.params
    );
    // Its Hansen J is still a valid over-identification statistic.
    assert!(iter.jtest.unwrap().stat >= 0.0);
}

/// A zero HAC bandwidth used to be honored, and *was* the White estimator:
/// the coverage audit measured `weight="hac"` at the default bandwidth as
/// bit-identical to `weight="robust"` (max |delta se| = 0 over 3000
/// replications) while callers believed they had bought serial-correlation
/// robustness. It is now a hard error on every entry point that takes a
/// covariance weight. This test replaces `hac_zero_bandwidth_equals_robust`,
/// which asserted the old (silently wrong) behaviour.
#[test]
fn hac_zero_bandwidth_is_rejected_not_silently_white() {
    let (x_cols, z_cols, y) = fixture_design();
    let zero = GmmWeight::Hac {
        kernel: Kernel::Bartlett,
        bandwidth: 0.0,
    };
    let identity16 = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let errs = [
        two_step_gmm(&x_cols, &z_cols, &y, zero).unwrap_err(),
        iterated_gmm(&x_cols, &z_cols, &y, zero, 1e-8, 10).unwrap_err(),
        one_step_gmm(&x_cols, &z_cols, &y, &identity16, zero).unwrap_err(),
    ];
    for err in &errs {
        assert!(
            matches!(err, GmmError::HacBandwidthNoOp { .. }),
            "expected the no-op rejection, got {err:?}"
        );
        // The message has to teach: what happened, and what to do instead.
        let msg = err.to_string();
        for needle in ["White", "no-op", "GmmWeight::HacAuto", "GmmWeight::Robust"] {
            assert!(msg.contains(needle), "message must mention {needle}: {msg}");
        }
        // And it must not oversell the alternative it offers. The audit
        // measured `weight="hac"` at 0.868 coverage against a nominal 0.95
        // (AR(1), phi = 0.8, T = 250, bandwidth = 10), and the automatic rule
        // picks 4 lags at that T — fewer than the setting that under-covered.
        // A caller told "pass HacAuto instead" must not read that as a fix.
        assert!(
            msg.contains("0.868") && msg.contains("not a remedy"),
            "the message must quote the measured coverage and refuse to \
             present HacAuto as the fix: {msg}"
        );
    }
    // A negative or non-finite bandwidth keeps its own (different) error.
    let bad = GmmWeight::Hac {
        kernel: Kernel::Bartlett,
        bandwidth: -1.0,
    };
    assert!(matches!(
        two_step_gmm(&x_cols, &z_cols, &y, bad).unwrap_err(),
        GmmError::InvalidBandwidth { .. }
    ));
}

/// The automatic rule is the Newey-West (1994) `floor(4*(n/100)^(2/9))`
/// lag truncation, it is reported back in `hac_bandwidth`, and it is never
/// zero — otherwise "auto" would reintroduce the no-op the error above
/// exists to prevent.
#[test]
fn hac_auto_bandwidth_is_the_newey_west_rule() {
    for n in [1_usize, 12, 50, 100, 300, 500, 2000, 10_000] {
        let expected = (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor();
        assert_eq!(GmmWeight::auto_bandwidth(n), expected, "rule at n = {n}");
        assert!(expected >= 1.0, "auto bandwidth must never be 0 (n = {n})");
    }

    let (x_cols, z_cols, y) = fixture_design();
    let n = y.len();
    let auto = two_step_gmm(
        &x_cols,
        &z_cols,
        &y,
        GmmWeight::HacAuto {
            kernel: Kernel::Bartlett,
        },
    )
    .unwrap();
    let bw = GmmWeight::auto_bandwidth(n);
    assert_eq!(auto.hac_bandwidth, Some(bw));

    // "auto" is exactly the explicit bandwidth it advertises, not a
    // different code path.
    let explicit = two_step_gmm(
        &x_cols,
        &z_cols,
        &y,
        GmmWeight::Hac {
            kernel: Kernel::Bartlett,
            bandwidth: bw,
        },
    )
    .unwrap();
    assert_eq!(auto.params, explicit.params);
    assert_eq!(auto.bse, explicit.bse);

    // And it is genuinely a different estimator from White.
    let robust = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).unwrap();
    assert_eq!(robust.hac_bandwidth, None);
    assert!(
        max_abs_diff(&robust.bse, &auto.bse) > 0.0,
        "HAC at a positive bandwidth must not reproduce the White standard errors"
    );
}

/// With one endogenous regressor and one excluded instrument the Wald
/// statistic collapses to a squared t, so the reported F must equal the
/// squared HC1 t-statistic on the instrument in the first-stage OLS. That is
/// the same object `tsecon-ident`'s `proxy_svar` reports as its effective F,
/// which is what makes the two comparable across the library.
#[test]
fn just_identified_first_stage_f_is_the_squared_robust_t() {
    let (x_cols, z_cols, y) = exactly_identified_data(31415, 400);
    let fit = two_stage_least_squares(&x_cols, &z_cols, &y).unwrap();
    assert_eq!(fit.first_stage.len(), 1);
    let fs = fit.first_stage[0];
    assert_eq!(fs.regressor, 1); // [const, x_endog]
    assert_eq!(fs.dof_num, 1);
    assert_eq!(fs.dof_den, 400 - 2);

    // Independent construction: OLS of the endogenous regressor on the
    // instruments, HC1 t on the excluded instrument, squared.
    let first = ols(&x_cols[1], &z_cols).unwrap();
    let inf = first.inference(SeType::Hc1).unwrap();
    let t = first.params[1] / inf.bse[1];
    assert!(
        (fs.fstat - t * t).abs() < 1e-9 * (t * t),
        "F {} vs squared robust t {}",
        fs.fstat,
        t * t
    );
    // A strong instrument (loading 0.8, n = 400) should be nowhere near the
    // rule-of-thumb boundary.
    assert!(fs.fstat > 100.0, "F = {}", fs.fstat);
    assert!(fs.pval < 1e-10);
}

/// Exogenous regressors instrument themselves, so they get no first-stage
/// entry; the diagnostic depends only on `(X, Z)`, so every estimator and
/// every weighting reports the same thing.
#[test]
fn first_stage_covers_only_instrumented_regressors() {
    let (x_cols, z_cols, y) = fixture_design(); // X = [const, w, x], Z = [const, w, z1, z2]
    let fits = [
        two_stage_least_squares(&x_cols, &z_cols, &y).unwrap(),
        two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).unwrap(),
        iterated_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust, 1e-10, 20).unwrap(),
        two_step_gmm(
            &x_cols,
            &z_cols,
            &y,
            GmmWeight::HacAuto {
                kernel: Kernel::Bartlett,
            },
        )
        .unwrap(),
    ];
    for fit in &fits {
        assert_eq!(fit.first_stage.len(), 1, "only x is instrumented");
        assert_eq!(fit.first_stage[0].regressor, 2);
        assert_eq!(fit.first_stage[0].dof_num, 2); // z1, z2 excluded
        assert_eq!(fit.first_stage[0], fits[0].first_stage[0]);
    }

    // A model whose regressors are all in the instrument set has no first
    // stage at all: X = [const, w] against Z = [const, w, z1].
    let all_exog = two_step_gmm(
        &[x_cols[0].clone(), x_cols[1].clone()],
        &[z_cols[0].clone(), z_cols[1].clone(), z_cols[2].clone()],
        &y,
        GmmWeight::Robust,
    )
    .unwrap();
    assert!(all_exog.first_stage.is_empty());
}

/// The diagnostic has to be able to say "weak": a small first-stage loading
/// must drive the F below the rule-of-thumb threshold that a large loading
/// clears by orders of magnitude.
///
/// Compared on **medians over 21 replications**, not on a single draw: a
/// first-stage F is itself a random variable (under a useless instrument it
/// is a draw from `F(1, n-2)`, which clears 10 about once in a thousand
/// samples), so a one-sample assertion would be a flaky test rather than a
/// statement about the estimator.
///
/// All 21 replications are drawn from **one** stream, advanced continuously.
/// Re-seeding per replication with consecutive integers
/// (`Stream::new(20260805 + rep)`) is the classic way to get *correlated*
/// "independent" replications out of a counter-based generator: adjacent seeds
/// are adjacent counter states, so the draws are related by construction and
/// the median over them is not the median over 21 independent samples it
/// claims to be.
#[test]
fn first_stage_f_tracks_instrument_strength() {
    let median_f = |loading: f64| {
        let mut s = Stream::new(20260805);
        let mut fs: Vec<f64> = (0..21)
            .map(|_rep| {
                let n = 300;
                let cst = vec![1.0_f64; n];
                let (mut z, mut xend, mut y) = (vec![], vec![], vec![]);
                for _ in 0..n {
                    let zi = gaussian(&mut s);
                    let v = gaussian(&mut s);
                    let xi = loading * zi + v;
                    let e = 0.6 * v + gaussian(&mut s);
                    z.push(zi);
                    xend.push(xi);
                    y.push(1.0 - 0.5 * xi + e);
                }
                let fit = two_stage_least_squares(&[cst.clone(), xend], &[cst, z], &y).unwrap();
                fit.first_stage[0].fstat
            })
            .collect();
        fs.sort_by(f64::total_cmp);
        fs[fs.len() / 2]
    };
    let strong = median_f(0.9);
    let weak = median_f(0.05);
    assert!(strong > 100.0, "strong instrument median F = {strong}");
    assert!(weak < 10.0, "weak instrument median F = {weak}");
}

/// A regressor the instruments reproduce **exactly** must get no first-stage
/// entry — not a fabricated one.
///
/// The guard that is supposed to catch this used to be
/// `residuals.iter().all(|e| *e == 0.0)`, which is dead code: the computed OLS
/// residuals of an exactly collinear regression are rounding noise of order
/// `eps * |x|`, never bit-zero. On this design (`xe = 2 + 3 z1 - 1.5 z2`, an
/// exact linear combination of `Z`) the measured `RSS/TSS` is `4.9e-32` and the
/// crate reported `fstat = 1.19e33, pval = 0` — the weak-instrument diagnostic
/// fabricating infinitely strong instruments, the worst possible direction for
/// it to fail in. The guard is now scale-relative.
#[test]
fn exactly_collinear_regressor_gets_no_fabricated_first_stage_f() {
    let n = 200;
    let cst = vec![1.0_f64; n];
    let z1: Vec<f64> = (0..n).map(|t| (0.31 * t as f64).sin()).collect();
    let z2: Vec<f64> = (0..n).map(|t| (0.17 * t as f64).cos()).collect();
    // xe is an exact linear combination of Z = [const, z1, z2].
    let xe: Vec<f64> = (0..n).map(|t| 2.0 + 3.0 * z1[t] - 1.5 * z2[t]).collect();
    let y: Vec<f64> = (0..n).map(|t| 1.0 + 0.5 * xe[t]).collect();
    let x_cols = vec![cst.clone(), xe];
    let z_cols = vec![cst, z1, z2];

    for fit in [
        two_stage_least_squares(&x_cols, &z_cols, &y).unwrap(),
        two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).unwrap(),
        iterated_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust, 1e-10, 20).unwrap(),
    ] {
        // The estimate itself is fine; only the diagnostic is undefined.
        assert_eq!(fit.params.len(), 2);
        assert!(
            fit.first_stage.is_empty(),
            "an exactly reproduced regressor must be OMITTED, not fabricated; got {:?}",
            fit.first_stage
        );
    }

    // Guard the guard: a genuine (even implausibly strong) first stage still
    // gets its entry. Adding a 1e-4 idiosyncratic component moves RSS/TSS from
    // 4.9e-32 to 5.4e-10, twenty-two orders of magnitude, so the two cases are
    // nowhere near each other and the threshold is not load-bearing.
    let z1b: Vec<f64> = (0..n).map(|t| (0.31 * t as f64).sin()).collect();
    let z2b: Vec<f64> = (0..n).map(|t| (0.17 * t as f64).cos()).collect();
    let xg: Vec<f64> = (0..n)
        .map(|t| 2.0 + 3.0 * z1b[t] - 1.5 * z2b[t] + 1e-4 * (0.9 * t as f64).sin())
        .collect();
    let yg: Vec<f64> = (0..n).map(|t| 1.0 + 0.5 * xg[t]).collect();
    let fit = two_stage_least_squares(&[vec![1.0; n], xg], &[vec![1.0; n], z1b, z2b], &yg).unwrap();
    assert_eq!(fit.first_stage.len(), 1, "a real first stage keeps its F");
    assert!(fit.first_stage[0].fstat.is_finite());
}

/// A **diagnostic must never take down the estimate.**
///
/// `first_stage_f` propagated `ols(...)` and `inv_spd(...)` with `?`, so a
/// rank-deficient instrument matrix failed the whole estimation. That broke
/// `one_step_gmm` with a caller-supplied weight — which never inverts `Z'Z`
/// and legitimately supported a duplicated instrument column — and the error
/// blamed a singular `X'X` in an internal first-stage OLS the caller never
/// requested. Both failures now skip the entry instead.
#[test]
fn duplicated_instrument_does_not_kill_one_step_gmm() {
    let n = 300;
    let cst = vec![1.0_f64; n];
    let z: Vec<f64> = (0..n).map(|t| (0.23 * t as f64).sin()).collect();
    let x: Vec<f64> = (0..n)
        .map(|t| 0.8 * z[t] + 0.3 * (0.11 * t as f64).cos())
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|t| 1.0 - 0.5 * x[t] + 0.1 * (0.07 * t as f64).sin())
        .collect();
    let x_cols = vec![cst.clone(), x];
    // Z = [const, z, z]: L = 3, k = 2, and column 2 duplicates column 1.
    let z_cols = vec![cst, z.clone(), z];

    // `one_step_gmm` never inverts Z'Z, so this design is supported.
    let identity9 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let fit = one_step_gmm(&x_cols, &z_cols, &y, &identity9, GmmWeight::Robust)
        .expect("one-step GMM with an identity weight must still fit a rank-deficient Z");

    // Not a smoke test: the closed form beta = (X'Z W Z'X)^{-1} X'Z W Z'y with
    // W = I is well posed here (the k x k product is full rank even though Z
    // is not), so recompute it independently and pin the coefficients.
    let xz = |j: usize, i: usize| -> f64 {
        x_cols[j]
            .iter()
            .zip(z_cols[i].iter())
            .map(|(a, b)| a * b)
            .sum()
    };
    let zy: Vec<f64> = (0..3)
        .map(|i| z_cols[i].iter().zip(y.iter()).map(|(a, b)| a * b).sum())
        .collect();
    // A = X'Z Z'X (2x2), b = X'Z Z'y (2x1).
    let a = [
        [
            (0..3).map(|i| xz(0, i) * xz(0, i)).sum::<f64>(),
            (0..3).map(|i| xz(0, i) * xz(1, i)).sum::<f64>(),
        ],
        [
            (0..3).map(|i| xz(1, i) * xz(0, i)).sum::<f64>(),
            (0..3).map(|i| xz(1, i) * xz(1, i)).sum::<f64>(),
        ],
    ];
    let b = [
        (0..3).map(|i| xz(0, i) * zy[i]).sum::<f64>(),
        (0..3).map(|i| xz(1, i) * zy[i]).sum::<f64>(),
    ];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let expected = [
        (a[1][1] * b[0] - a[0][1] * b[1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ];
    assert!(
        max_abs_diff(&fit.params, &expected) < 1e-8,
        "one-step params {:?} vs the independently solved closed form {expected:?}",
        fit.params
    );
    // The diagnostic is what is unavailable, and it says so by being absent.
    assert!(
        fit.first_stage.is_empty(),
        "the first stage is not computable here, so it must be omitted: {:?}",
        fit.first_stage
    );
}

/// Where a rank-deficient instrument set genuinely *does* break an estimator,
/// that estimator must still raise its own accurate error naming the real
/// cause — not the singular `X'X` of an internal first-stage OLS the caller
/// never requested. `two_step_gmm` inverts `Z'Z/n` for its step-1 weight, so
/// near-collinear instruments fail there and the message says so.
#[test]
fn near_collinear_instruments_still_report_the_step_one_weight() {
    let n = 300;
    let cst = vec![1.0_f64; n];
    let z: Vec<f64> = (0..n).map(|t| (0.23 * t as f64).sin()).collect();
    // A near-duplicate: relative 1e-10 apart, so `same_column` correctly calls
    // it a different column, but Z'Z is numerically singular.
    let z_dup: Vec<f64> = z
        .iter()
        .enumerate()
        .map(|(i, v)| v * (1.0 + 1e-10 * ((i % 3) as f64)))
        .collect();
    let x: Vec<f64> = (0..n)
        .map(|t| 0.8 * z[t] + 0.3 * (0.11 * t as f64).cos())
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|t| 1.0 - 0.5 * x[t] + 0.1 * (0.07 * t as f64).sin())
        .collect();

    let err = two_step_gmm(&[cst.clone(), x], &[cst, z, z_dup], &y, GmmWeight::Robust).unwrap_err();
    let msg = err.to_string();
    assert!(
        matches!(err, GmmError::SingularMatrix { .. }),
        "expected a singular-matrix error, got {err:?}"
    );
    assert!(
        msg.contains("step-1 weight (Z'Z/n)") && msg.contains("collinear instruments"),
        "the message must name the real cause (the Z'Z inverse), got: {msg}"
    );
    assert!(
        !msg.contains("constant column twice"),
        "must not blame an internal first-stage OLS the caller never requested: {msg}"
    );
}

/// Build `X = [const, w, x]`, `Z = [const, w * (1 + rel), z]` at a chosen
/// magnitude for `w`, and return the reported first-stage classification as
/// `(regressor, dof_num, dof_den)` triples.
///
/// Everything except the scale of `w` is held fixed, so any difference in the
/// output is a difference in how `same_column` classified that one column.
fn classification_at_scale(scale: f64, rel: f64) -> Vec<(usize, usize, usize)> {
    let n = 200;
    let cst = vec![1.0_f64; n];
    let w: Vec<f64> = (0..n)
        .map(|t| scale * (5.0 + (0.3 * t as f64).sin()))
        .collect();
    let z: Vec<f64> = (0..n).map(|t| (0.17 * t as f64).cos()).collect();
    let v: Vec<f64> = (0..n).map(|t| (1.1 * t as f64).sin()).collect();
    let x: Vec<f64> = (0..n)
        .map(|t| 0.6 * z[t] + 0.2 * w[t] / scale + 0.5 * v[t])
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|t| 1.0 - 0.5 * w[t] / scale + 0.5 * x[t] + 0.4 * v[t])
        .collect();
    let w_z: Vec<f64> = w.iter().map(|val| val * (1.0 + rel)).collect();
    let fit =
        two_step_gmm(&[cst.clone(), w, x], &[cst, w_z, z], &y, GmmWeight::Robust).expect("fits");
    fit.first_stage
        .iter()
        .map(|f| (f.regressor, f.dof_num, f.dof_den))
        .collect()
}

/// `same_column` decides the exogenous/endogenous split, so its tolerance
/// leaks straight into the reported degrees of freedom of every endogenous
/// regressor — not just into whether a spurious extra row appears. It must
/// therefore be **per-element relative**, not one absolute tolerance scaled by
/// the column maximum.
///
/// The old rule was `|a_i - b_i| <= 1e-12 * max(1, max_j |a_j|)`. Two
/// consequences, both pinned below.
#[test]
fn same_column_tolerance_is_per_element_relative() {
    // (a) SCALE INVARIANCE. Whether two columns are "the same variable" is a
    // question about their relative agreement; multiplying the column by 1e12
    // must not change the answer. Under the old rule it did: at scale 1e-6 the
    // tolerance floored at 1e-12 while the discrepancy shrank to ~1e-15, so a
    // relative-1e-9 difference was absorbed; at scale 1e6 the tolerance grew to
    // ~1e-6 while the discrepancy grew to ~1e-3, so the same relative
    // difference was rejected. Same question, two answers, and the endogenous
    // regressor's own `dof_num` moved with it.
    let small = classification_at_scale(1e-6, 1e-9);
    let large = classification_at_scale(1e6, 1e-9);
    assert_eq!(
        small, large,
        "classification must not depend on the magnitude of the column: \
         scale 1e-6 gave {small:?}, scale 1e6 gave {large:?}"
    );
    // And a below-tolerance difference is absorbed at every scale.
    let small_same = classification_at_scale(1e-6, 1e-14);
    let large_same = classification_at_scale(1e6, 1e-14);
    assert_eq!(small_same, large_same);
    assert_eq!(
        small_same,
        vec![(2, 1, 200 - 3)],
        "a 1e-14 relative difference is the same column: only x is instrumented, q = 1"
    );
    assert_ne!(
        small, small_same,
        "a 1e-9 relative difference is NOT the same column"
    );

    // The other direction of the old bug — a mixed-magnitude column whose
    // largest element sets a tolerance of ~10 for its smallest, merging two
    // plainly different columns — needs a design that is deliberately
    // ill-conditioned, which the estimator (rightly) refuses to fit. It is
    // pinned directly against `same_column` in the unit tests in `src/linear.rs`.
}

#[test]
fn nonlinear_gmm_recovers_mean_and_variance() {
    // moments: E[y - mu] = 0, E[(y - mu)^2 - s2] = 0.
    // The exactly-identified solution is the sample mean and the *biased*
    // (divide-by-n) sample variance.
    let mut s = Stream::new(90210);
    let n = 500;
    let y: Vec<f64> = (0..n).map(|_| 2.0 + 1.5 * gaussian(&mut s)).collect();

    let moments = |theta: &[f64]| -> Vec<Vec<f64>> {
        let mu = theta[0];
        let s2 = theta[1];
        y.iter()
            .map(|&yi| vec![yi - mu, (yi - mu).powi(2) - s2])
            .collect()
    };
    let fit = gmm_nonlinear(moments, &[0.0, 1.0], None).unwrap();

    let mean = y.iter().sum::<f64>() / n as f64;
    let biased_var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / n as f64;

    assert!(fit.converged);
    assert_eq!(fit.nmoments, 2);
    assert_eq!(fit.nparams, 2);
    assert!(
        (fit.params[0] - mean).abs() < 1e-4,
        "mu {} vs sample mean {mean}",
        fit.params[0]
    );
    assert!(
        (fit.params[1] - biased_var).abs() < 1e-4,
        "s2 {} vs biased variance {biased_var}",
        fit.params[1]
    );
    // Exactly identified => the sample moments are driven to (near) zero.
    assert!(fit.gbar.iter().all(|g| g.abs() < 1e-4));
}

#[test]
fn rejects_underidentified_design() {
    // 1 instrument for 2 parameters.
    let n = 50;
    let cst = vec![1.0; n];
    let xend: Vec<f64> = (0..n).map(|t| t as f64).collect();
    let y = vec![0.0; n];
    let x_cols = vec![cst.clone(), xend];
    let z_cols = vec![cst]; // only 1 instrument
    let err = two_stage_least_squares(&x_cols, &z_cols, &y).unwrap_err();
    assert!(matches!(
        err,
        GmmError::UnderIdentified {
            moments: 1,
            params: 2
        }
    ));
}

#[test]
fn rejects_dimension_mismatch_and_nonfinite() {
    let n = 20;
    let cst = vec![1.0; n];
    let good: Vec<f64> = (0..n).map(|t| t as f64).collect();
    let y = vec![0.0; n];

    // Instrument column too short.
    let short = vec![1.0; n - 1];
    let err = two_stage_least_squares(&[cst.clone(), good.clone()], &[cst.clone(), short], &y)
        .unwrap_err();
    assert!(matches!(err, GmmError::DimensionMismatch { .. }));

    // Non-finite entry.
    let mut bad = good.clone();
    bad[3] = f64::NAN;
    let err = two_stage_least_squares(&[cst.clone(), bad], &[cst.clone(), good], &y).unwrap_err();
    assert!(matches!(err, GmmError::NonFinite { .. }));
}

#[test]
fn rejects_misshaped_weight() {
    let (x_cols, z_cols, y) = fixture_design();
    // L = 4, so the weight must be 4x4 = 16 entries; give 9.
    let bad_weight = vec![0.0; 9];
    let err = one_step_gmm(&x_cols, &z_cols, &y, &bad_weight, GmmWeight::Robust).unwrap_err();
    assert!(matches!(err, GmmError::DimensionMismatch { .. }));
}

#[test]
fn nonlinear_rejects_empty_and_bad_weight() {
    let y = [1.0_f64, 2.0, 3.0];
    let moments =
        |theta: &[f64]| -> Vec<Vec<f64>> { y.iter().map(|&yi| vec![yi - theta[0]]).collect() };
    // Empty initial.
    assert!(matches!(
        gmm_nonlinear(moments, &[], None).unwrap_err(),
        GmmError::EmptyInput { .. }
    ));
    // Weight of the wrong size (m = 1, so must be 1x1).
    let moments2 =
        |theta: &[f64]| -> Vec<Vec<f64>> { y.iter().map(|&yi| vec![yi - theta[0]]).collect() };
    assert!(matches!(
        gmm_nonlinear(moments2, &[0.0], Some(&[1.0, 2.0, 3.0, 4.0])).unwrap_err(),
        GmmError::DimensionMismatch { .. }
    ));
}
