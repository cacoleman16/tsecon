//! Property tests for the EVT estimators: invariances, limit behavior,
//! branch continuity, and teaching errors on degenerate input.
//!
//! Data are deterministic inverse-CDF quantile grids (`u_i = (i + 0.5)/n`
//! mapped through the target quantile function) — exact "ideal samples"
//! that need no RNG and make limit statements sharp.

use tsecon_evt::{
    gev_fit, gev_loglik, gpd_fit, gpd_loglik, EvtError, MIN_EXCEEDANCES, MIN_MAXIMA, XI_CUTOFF,
};

/// Midpoint uniform grid on (0, 1).
fn ugrid(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect()
}

/// Exponential(1) quantile grid.
fn exp_grid(n: usize) -> Vec<f64> {
    ugrid(n).iter().map(|&u| -(1.0 - u).ln()).collect()
}

/// Pareto(alpha = 3) quantile grid — true tail index xi = 1/3.
fn pareto3_grid(n: usize) -> Vec<f64> {
    ugrid(n)
        .iter()
        .map(|&u| (1.0 - u).powf(-1.0 / 3.0))
        .collect()
}

/// Bounded-tail grid with true xi = -1/3: y = 1 - (1 - u)^(1/3).
fn bounded_grid(n: usize) -> Vec<f64> {
    ugrid(n)
        .iter()
        .map(|&u| 1.0 - (1.0 - u).powf(1.0 / 3.0))
        .collect()
}

/// Gumbel quantile grid (the exact block-maxima law of exponentials).
fn gumbel_grid(n: usize) -> Vec<f64> {
    ugrid(n).iter().map(|&u| -(-(u.ln())).ln()).collect()
}

// ------------------------------------------------------------------ GPD

#[test]
fn gpd_xi_near_zero_on_exponential_data() {
    // The exponential is the xi = 0 boundary of the GPD family, and its
    // exceedances are again exponential; on an ideal quantile-grid sample
    // the fitted shape must sit tightly at zero.
    let y = exp_grid(2000);
    let fit = gpd_fit(&y, None, 0.90, &[0.99]).expect("fit");
    assert!(
        fit.xi.abs() < 0.05,
        "exponential data should give xi near 0, got {}",
        fit.xi
    );
}

#[test]
fn gpd_default_threshold_equals_explicit_threshold() {
    let y = pareto3_grid(800);
    let a = gpd_fit(&y, None, 0.90, &[0.99, 0.999]).expect("default");
    let b = gpd_fit(&y, Some(a.threshold), 0.90, &[0.99, 0.999]).expect("explicit");
    // Identical threshold => identical exceedances => identical fit,
    // bit for bit (only the reported quantile flavor differs).
    assert_eq!(a.xi, b.xi);
    assert_eq!(a.beta, b.beta);
    assert_eq!(a.loglik, b.loglik);
    assert_eq!(a.var, b.var);
    assert_eq!(a.es, b.es);
    assert_eq!(a.n_exceed, b.n_exceed);
}

#[test]
fn gpd_scale_equivariance() {
    // y -> c y: xi invariant; beta, VaR, ES scale by c. The whole
    // pipeline (quantile threshold, moment starts, (xi, ln beta) search)
    // is scale-equivariant by construction, so this holds to optimizer
    // precision, not just asymptotically.
    let y = pareto3_grid(1200);
    let c = 250.0;
    let yc: Vec<f64> = y.iter().map(|&v| c * v).collect();
    let p = [0.99, 0.995];
    let a = gpd_fit(&y, None, 0.90, &p).expect("base");
    let b = gpd_fit(&yc, None, 0.90, &p).expect("scaled");
    assert!(
        (a.xi - b.xi).abs() < 1e-6,
        "xi not scale invariant: {} vs {}",
        a.xi,
        b.xi
    );
    assert!(
        (b.beta / a.beta / c - 1.0).abs() < 1e-6,
        "beta not equivariant: {} vs {}",
        a.beta,
        b.beta
    );
    for i in 0..p.len() {
        assert!((b.var[i] / a.var[i] / c - 1.0).abs() < 1e-6);
        assert!((b.es[i] / a.es[i] / c - 1.0).abs() < 1e-6);
    }
}

#[test]
fn gpd_es_at_least_var_always() {
    // ES - VaR = (beta + xi (VaR - u)) / (1 - xi) >= 0 for any xi < 1 on
    // the GPD support; check it holds numerically across tail shapes.
    let p = [0.95, 0.99, 0.995, 0.999];
    for (name, y) in [
        ("heavy", pareto3_grid(1500)),
        ("exponential", exp_grid(1500)),
        ("bounded", bounded_grid(1500)),
    ] {
        let fit = gpd_fit(&y, None, 0.90, &p).expect(name);
        for (i, &pi) in p.iter().enumerate() {
            assert!(
                fit.es[i] >= fit.var[i],
                "{name}: es {} < var {} at p={pi}",
                fit.es[i],
                fit.var[i],
            );
            assert!(fit.var[i] > fit.threshold, "{name}: VaR below threshold");
        }
    }
}

#[test]
fn gpd_uniform_tail_irregularity_is_flagged_not_certified() {
    // A raw uniform tail has true xi = -1: no MLE exists (the likelihood
    // supremum is +inf at beta -> -xi max z). The fit must still return,
    // with a strongly negative xi and WITHOUT certifying standard errors.
    let y = ugrid(2000);
    let fit = gpd_fit(&y, None, 0.90, &[]).expect("fit returns best point");
    assert!(
        fit.xi < -0.5,
        "uniform tail should drive xi into the irregular region, got {}",
        fit.xi
    );
    assert!(!fit.se_valid, "SEs must not be certified at xi <= -0.5");
}

#[test]
fn gpd_loglik_continuous_across_xi_cutoff() {
    let y = pareto3_grid(400);
    let fit = gpd_fit(&y, None, 0.90, &[]).expect("fit");
    let z: Vec<f64> = y
        .iter()
        .filter(|&&v| v > fit.threshold)
        .map(|&v| v - fit.threshold)
        .collect();
    let beta = 0.7;
    // (a) No jump across the documented seam.
    let above = gpd_loglik(&z, XI_CUTOFF * 1.0000001, beta).expect("above");
    let below = gpd_loglik(&z, XI_CUTOFF * 0.9999999, beta).expect("below");
    assert!(
        (above - below).abs() < 1e-8,
        "loglik jumps across the cutoff: {above} vs {below}"
    );
    // (b) The limit branch agrees with the exact formula evaluated inside
    // the limit region (the exact form is healthy at any nonzero xi).
    for &xi in &[1e-9, -1e-9, 1e-12] {
        let branch = gpd_loglik(&z, xi, beta).expect("branch");
        let exact: f64 = z
            .iter()
            .map(|&zi| -beta.ln() - (1.0 + 1.0 / xi) * (xi * zi / beta).ln_1p())
            .sum();
        assert!(
            (branch - exact).abs() <= 1e-10 * exact.abs().max(1.0),
            "xi={xi}: branch {branch} vs exact {exact}"
        );
    }
}

#[test]
fn gpd_degenerate_inputs_raise() {
    // Empty.
    assert!(matches!(
        gpd_fit(&[], None, 0.9, &[]),
        Err(EvtError::EmptyInput { .. })
    ));
    // NaN.
    let mut y = exp_grid(100);
    y[3] = f64::NAN;
    assert!(matches!(
        gpd_fit(&y, None, 0.9, &[]),
        Err(EvtError::NonFinite { .. })
    ));
    // Constant series: no strict exceedances over its own quantile.
    let cy = vec![1.5; 200];
    assert!(matches!(
        gpd_fit(&cy, None, 0.9, &[]),
        Err(EvtError::TooFewExceedances { .. })
    ));
    // Too few exceedances (documented minimum).
    let y = exp_grid(MIN_EXCEEDANCES * 10 - 10); // 90 obs -> 9 exceedances
    let err = gpd_fit(&y, None, 0.9, &[]).expect_err("too few");
    assert!(matches!(
        err,
        EvtError::TooFewExceedances { n_exceed: 9, .. }
    ));
    // Threshold above the sample maximum.
    let y = exp_grid(200);
    assert!(matches!(
        gpd_fit(&y, Some(1e9), 0.9, &[]),
        Err(EvtError::TooFewExceedances { n_exceed: 0, .. })
    ));
    // Invalid quantile / tail probabilities.
    assert!(matches!(
        gpd_fit(&y, None, 1.0, &[]),
        Err(EvtError::InvalidQuantile { .. })
    ));
    assert!(matches!(
        gpd_fit(&y, None, 0.9, &[1.5]),
        Err(EvtError::InvalidTailProb { .. })
    ));
    // p not beyond the threshold: 1 - 0.5 >= exceedance rate 0.1.
    assert!(matches!(
        gpd_fit(&y, None, 0.9, &[0.5]),
        Err(EvtError::TailProbNotBeyondThreshold { .. })
    ));
}

// ------------------------------------------------------------------ GEV

#[test]
fn gev_xi_near_zero_on_gumbel_data() {
    let m = gumbel_grid(300);
    let fit = gev_fit(&m, None, &[10.0]).expect("fit");
    assert!(
        fit.xi.abs() < 0.05,
        "Gumbel data should give xi near 0, got {}",
        fit.xi
    );
    assert!(fit.mu.abs() < 0.1 && (fit.sigma - 1.0).abs() < 0.1);
}

#[test]
fn gev_block_path_equals_precomputed_maxima_path() {
    let y = exp_grid(900);
    let b = 30;
    let maxima: Vec<f64> = y
        .chunks_exact(b)
        .map(|c| c.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        .collect();
    let via_blocks = gev_fit(&y, Some(b), &[20.0]).expect("blocks");
    let via_maxima = gev_fit(&maxima, None, &[20.0]).expect("maxima");
    assert_eq!(via_blocks.xi, via_maxima.xi);
    assert_eq!(via_blocks.mu, via_maxima.mu);
    assert_eq!(via_blocks.sigma, via_maxima.sigma);
    assert_eq!(via_blocks.return_levels, via_maxima.return_levels);
    assert_eq!(via_blocks.n_maxima, via_maxima.n_maxima);
    assert_eq!(via_blocks.block_size, Some(b));
    assert_eq!(via_maxima.block_size, None);
}

#[test]
fn gev_shift_and_scale_equivariance() {
    let m = gumbel_grid(250);
    let base = gev_fit(&m, None, &[50.0]).expect("base");
    // Shift: mu translates, xi and sigma untouched.
    let shift = 17.5;
    let ms: Vec<f64> = m.iter().map(|&v| v + shift).collect();
    let sh = gev_fit(&ms, None, &[50.0]).expect("shifted");
    assert!((sh.xi - base.xi).abs() < 1e-6);
    assert!((sh.mu - base.mu - shift).abs() < 1e-6);
    assert!((sh.sigma / base.sigma - 1.0).abs() < 1e-6);
    assert!((sh.return_levels[0] - base.return_levels[0] - shift).abs() < 1e-6);
    // Scale: mu, sigma, return levels scale; xi invariant.
    let c = 40.0;
    let mc: Vec<f64> = m.iter().map(|&v| c * v).collect();
    let sc = gev_fit(&mc, None, &[50.0]).expect("scaled");
    assert!((sc.xi - base.xi).abs() < 1e-5, "{} vs {}", sc.xi, base.xi);
    assert!((sc.mu / (c * base.mu) - 1.0).abs() < 1e-4);
    assert!((sc.sigma / (c * base.sigma) - 1.0).abs() < 1e-5);
    assert!((sc.return_levels[0] / (c * base.return_levels[0]) - 1.0).abs() < 1e-5);
}

#[test]
fn gev_return_levels_increase_with_period() {
    let m = gumbel_grid(200);
    let fit = gev_fit(&m, None, &[5.0, 10.0, 50.0, 100.0, 500.0]).expect("fit");
    for w in fit.return_levels.windows(2) {
        assert!(w[1] > w[0], "return levels must increase with T");
    }
}

#[test]
fn gev_loglik_continuous_across_xi_cutoff() {
    let m = gumbel_grid(150);
    let (mu, sigma) = (0.05, 1.1);
    let above = gev_loglik(&m, XI_CUTOFF * 1.0000001, mu, sigma).expect("above");
    let below = gev_loglik(&m, XI_CUTOFF * 0.9999999, mu, sigma).expect("below");
    assert!(
        (above - below).abs() < 1e-8,
        "loglik jumps across the cutoff: {above} vs {below}"
    );
    for &xi in &[1e-9, -1e-9, 1e-12] {
        let branch = gev_loglik(&m, xi, mu, sigma).expect("branch");
        let exact: f64 = m
            .iter()
            .map(|&mi| {
                let t = (mi - mu) / sigma;
                let a = (xi * t).ln_1p() / xi;
                -sigma.ln() - (1.0 + xi) * a - (-a).exp()
            })
            .sum();
        assert!(
            (branch - exact).abs() <= 1e-10 * exact.abs().max(1.0),
            "xi={xi}: branch {branch} vs exact {exact}"
        );
    }
}

#[test]
fn gev_degenerate_inputs_raise() {
    assert!(matches!(
        gev_fit(&[], None, &[]),
        Err(EvtError::EmptyInput { .. })
    ));
    let m = gumbel_grid(MIN_MAXIMA - 1);
    assert!(matches!(
        gev_fit(&m, None, &[]),
        Err(EvtError::TooFewMaxima { .. })
    ));
    let y = exp_grid(100);
    assert!(matches!(
        gev_fit(&y, Some(0), &[]),
        Err(EvtError::InvalidBlockSize { .. })
    ));
    assert!(matches!(
        gev_fit(&y, Some(101), &[]),
        Err(EvtError::InvalidBlockSize { .. })
    ));
    // 100 obs in blocks of 20 -> 5 maxima < MIN_MAXIMA.
    assert!(matches!(
        gev_fit(&y, Some(20), &[]),
        Err(EvtError::TooFewMaxima { n_maxima: 5, .. })
    ));
    let cm = vec![2.0; 50];
    assert!(matches!(
        gev_fit(&cm, None, &[]),
        Err(EvtError::Degenerate { .. })
    ));
    assert!(matches!(
        gev_fit(&gumbel_grid(50), None, &[1.0]),
        Err(EvtError::InvalidReturnPeriod { .. })
    ));
}
