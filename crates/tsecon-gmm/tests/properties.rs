//! Property / behavioural tests for the GMM estimators — the statistical
//! invariances that no single golden value pins:
//!
//! * exact identification collapses GMM to IV / 2SLS for *any* weight;
//! * iterated GMM's first re-weight reproduces the two-step estimator, and
//!   the full iteration converges near it in a couple of steps;
//! * the nonlinear driver recovers the analytic mean/variance method-of-
//!   moments solution;
//! * a zero-bandwidth HAC weight coincides with the robust weight;
//! * the validation layer rejects malformed inputs.

use serde_json::Value;
use tsecon_gmm::{
    gmm_nonlinear, iterated_gmm, one_step_gmm, two_stage_least_squares, two_step_gmm, GmmError,
    GmmWeight,
};
use tsecon_hac::Kernel;
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

#[test]
fn hac_zero_bandwidth_equals_robust() {
    // With bandwidth 0 the HAC kernel keeps only the lag-0 term, so the HAC
    // moment covariance equals the White (robust) one — the two weightings
    // must produce identical estimates and standard errors.
    let (x_cols, z_cols, y) = fixture_design();
    let robust = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).unwrap();
    let hac0 = two_step_gmm(
        &x_cols,
        &z_cols,
        &y,
        GmmWeight::Hac {
            kernel: Kernel::Bartlett,
            bandwidth: 0.0,
        },
    )
    .unwrap();
    assert!(max_abs_diff(&robust.params, &hac0.params) < 1e-12);
    assert!(max_abs_diff(&robust.bse, &hac0.bse) < 1e-12);
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
