//! Monte-Carlo property tests — the real statistical validation of the crate
//! (targets (b)-(d)). Golden fixtures pin the *algebra*; these seeded
//! simulations establish that the algebra is the *statistically correct* one.
//!
//! * (b) SIZE: under `H0: beta = 0`, the IVX-Wald test rejects at ~5% for
//!   every persistence `rho in {0.9, 0.95, 0.99, 1.0}` — including the exact
//!   unit root, where the naive OLS t-test over-rejects several-fold. This
//!   uniform-over-persistence validity is the reason IVX exists.
//! * (c) POWER: under a genuine predictive slope the rejection rate is well
//!   above size.
//! * (d) Stambaugh: with `corr(u, e) < 0` and a persistent predictor, the mean
//!   of `beta_corrected` is closer to the truth than the mean of `beta_ols`.
//!
//! All randomness is the library's seeded Philox stream (`tsecon_rng`); the
//! numbers below are reproducible run to run.

use tsecon_predreg::{ivx, ivx_multi, ols_predictive, stambaugh, IvxConfig, PredRegError};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal};

/// Standard normal via the inverse-CDF of a Philox uniform.
fn gaussian(s: &mut Stream) -> f64 {
    let u = s.uniform_f64().clamp(1e-12, 1.0 - 1e-12);
    StdNormal.ppf(u).expect("ppf on interior point")
}

/// Simulate the Stambaugh DGP:
/// `x_t = rho x_{t-1} + e_t`, `u_t = cue e_t + sqrt(1-cue^2) w_t`,
/// `r_t = beta x_t + u_t`. Returns `(r, x)` of length `n`.
fn simulate(s: &mut Stream, n: usize, rho: f64, cue: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];
    for t in 0..n {
        e[t] = gaussian(s);
        if t > 0 {
            x[t] = rho * x[t - 1] + e[t];
        }
    }
    let root = (1.0 - cue * cue).sqrt();
    let r: Vec<f64> = (0..n)
        .map(|t| {
            let u = cue * e[t] + root * gaussian(s);
            beta * x[t] + u
        })
        .collect();
    (r, x)
}

const CHI2_1_95: f64 = 3.841_459; // chi-square(1) 95th percentile

/// Naive OLS t-test rejection at 5% (two-sided), for the contrast.
fn naive_ols_rejects(r: &[f64], x: &[f64]) -> bool {
    let fit = ols_predictive(r, x).expect("ols");
    fit.tstat.abs() > 1.959_964
}

#[test]
fn ivx_wald_holds_size_uniformly_over_persistence() {
    // Target (b). reps and band chosen for a ~2.5-sigma Monte-Carlo envelope.
    let reps = 3000;
    let n = 250;
    let cue = -0.9; // strong endogeneity — the hard case
    let nominal = 0.05;
    let band = 0.02; // accept [0.030, 0.070]
    let cfg = IvxConfig::default();

    for &rho in &[0.9, 0.95, 0.99, 1.0] {
        let mut s = Stream::new(0xC0FFEE ^ ((rho * 1000.0) as u64));
        let mut ivx_rej = 0usize;
        let mut ols_rej = 0usize;
        for _ in 0..reps {
            let (r, x) = simulate(&mut s, n, rho, cue, 0.0);
            if ivx(&r, &x, cfg).expect("ivx").wald > CHI2_1_95 {
                ivx_rej += 1;
            }
            if naive_ols_rejects(&r, &x) {
                ols_rej += 1;
            }
        }
        let ivx_size = ivx_rej as f64 / reps as f64;
        let ols_size = ols_rej as f64 / reps as f64;
        eprintln!("rho={rho:.2}: IVX size={ivx_size:.3}  naive-OLS size={ols_size:.3}");
        assert!(
            (ivx_size - nominal).abs() < band,
            "IVX-Wald size {ivx_size:.3} at rho={rho} outside nominal {nominal} +/- {band}"
        );
    }
}

#[test]
fn naive_ols_over_rejects_at_the_unit_root() {
    // The failure IVX is designed to fix: at rho=1 the naive OLS t-test
    // rejects a true null far above 5%.
    let reps = 3000;
    let n = 250;
    let mut s = Stream::new(0xBADC0DE);
    let mut ols_rej = 0usize;
    for _ in 0..reps {
        let (r, x) = simulate(&mut s, n, 1.0, -0.9, 0.0);
        if naive_ols_rejects(&r, &x) {
            ols_rej += 1;
        }
    }
    let size = ols_rej as f64 / reps as f64;
    eprintln!("naive-OLS unit-root size = {size:.3}");
    assert!(
        size > 0.12,
        "expected the naive OLS t-test to badly over-reject at the unit root, got {size:.3}"
    );
}

#[test]
fn ivx_wald_has_power_against_a_true_slope() {
    // Target (c). A genuine predictive slope must reject far more often than
    // the ~5% size.
    let reps = 2000;
    let n = 250;
    let cue = -0.9;
    let beta = 0.08;
    let cfg = IvxConfig::default();
    for &rho in &[0.9, 0.99] {
        let mut s = Stream::new(0x50FA ^ ((rho * 1000.0) as u64));
        let mut rej = 0usize;
        for _ in 0..reps {
            let (r, x) = simulate(&mut s, n, rho, cue, beta);
            if ivx(&r, &x, cfg).expect("ivx").wald > CHI2_1_95 {
                rej += 1;
            }
        }
        let power = rej as f64 / reps as f64;
        eprintln!("rho={rho:.2}: IVX power={power:.3}");
        assert!(
            power > 0.5,
            "IVX-Wald power {power:.3} at rho={rho} too low to be useful"
        );
    }
}

#[test]
fn stambaugh_reduces_finite_sample_bias() {
    // Target (d). Mean of beta_corrected must be closer to the truth than the
    // mean of beta_ols across many persistent, endogenous samples.
    let reps = 3000;
    let n = 300;
    let cue = -0.9;
    let beta_true = 0.05;
    for &rho in &[0.95, 0.99] {
        let mut s = Stream::new(0x5AB ^ ((rho * 1000.0) as u64));
        let mut sum_ols = 0.0_f64;
        let mut sum_cor = 0.0_f64;
        for _ in 0..reps {
            let (r, x) = simulate(&mut s, n, rho, cue, beta_true);
            let c = stambaugh(&r, &x).expect("stambaugh");
            sum_ols += c.beta_ols;
            sum_cor += c.beta_corrected;
        }
        let bias_ols = (sum_ols / reps as f64 - beta_true).abs();
        let bias_cor = (sum_cor / reps as f64 - beta_true).abs();
        eprintln!("rho={rho:.2}: |bias(beta_ols)|={bias_ols:.5}  |bias(beta_corr)|={bias_cor:.5}");
        assert!(
            bias_cor < bias_ols,
            "Stambaugh correction did not reduce bias at rho={rho}: \
             |bias_ols|={bias_ols:.5} |bias_corr|={bias_cor:.5}"
        );
    }
}

// ------------------------------------------------------------------ validation

#[test]
fn rejects_length_mismatch() {
    let r = [1.0, 2.0, 3.0];
    let x = [1.0, 2.0];
    assert!(matches!(
        ols_predictive(&r, &x),
        Err(PredRegError::DimensionMismatch { .. })
    ));
}

#[test]
fn rejects_non_finite() {
    let r = [1.0, f64::NAN, 3.0, 4.0];
    let x = [1.0, 2.0, 3.0, 4.0];
    assert!(matches!(
        ivx(&r, &x, IvxConfig::default()),
        Err(PredRegError::NonFinite { .. })
    ));
}

#[test]
fn rejects_too_short() {
    let r = [1.0, 2.0];
    let x = [1.0, 2.0];
    assert!(matches!(
        stambaugh(&r, &x),
        Err(PredRegError::DegreesOfFreedom { .. })
    ));
}

#[test]
fn rejects_bad_ivx_config() {
    let r = [1.0, 2.0, 3.0, 4.0, 5.0];
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    // alpha out of (0,1)
    assert!(matches!(
        ivx(
            &r,
            &x,
            IvxConfig {
                cz: -1.0,
                alpha: 1.5
            }
        ),
        Err(PredRegError::InvalidArgument { .. })
    ));
    // non-negative cz
    assert!(matches!(
        ivx(
            &r,
            &x,
            IvxConfig {
                cz: 0.5,
                alpha: 0.95
            }
        ),
        Err(PredRegError::InvalidArgument { .. })
    ));
}

#[test]
fn ivx_scalar_matches_multi_with_one_regressor() {
    // The multivariate path must specialise to the scalar path for q = 1.
    let mut s = Stream::new(42);
    let (r, x) = simulate(&mut s, 200, 0.97, -0.5, 0.04);
    let cfg = IvxConfig::default();
    let scalar = ivx(&r, &x, cfg).expect("scalar");
    let multi = ivx_multi(&r, std::slice::from_ref(&x), cfg).expect("multi");
    assert!((scalar.beta_ivx - multi.beta_ivx[0]).abs() < 1e-9);
    assert!((scalar.wald - multi.wald).abs() < 1e-9);
}
