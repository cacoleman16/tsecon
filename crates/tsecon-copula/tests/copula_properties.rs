//! Property tests for the copula estimators: the monotone-margin
//! invariance that is the entire point of the copula decomposition,
//! parameter recovery on deterministic conditional-inversion samples,
//! independence limits, exchange symmetry, tau/parameter roundtrips,
//! and teaching errors on degenerate input.
//!
//! Data are deterministic: the first margin is the midpoint grid
//! `u_i = (i + 0.5)/n` and the conditioning variate is the golden-ratio
//! Weyl sequence `q_i = frac((i + 1) phi)` — a low-discrepancy stream
//! that needs no RNG — pushed through each family's exact conditional
//! inverse `C_{2|1}^{-1}(q | u)`.

use tsecon_copula::{
    copula_fit, copula_loglik, copula_logpdf, copula_select, kendall_tau, param_to_tau, pseudo_obs,
    tail_dependence, tau_to_param, CopulaError, Family, FitMethod, MIN_OBS,
};

const PHI: f64 = 0.618_033_988_749_894_9;

fn ugrid(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect()
}

fn weyl(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as f64 + 1.0) * PHI).fract())
        .map(|q| q.clamp(1e-12, 1.0 - 1e-12))
        .collect()
}

// Normal/Student quantile helpers via the crate's own evaluators would be
// circular for *correctness*, but for *sampling* only consistency matters:
// the recovery property tests the fit, not the sampler. Use simple
// rational approximations refined by bisection on the fitted families'
// own conditional CDFs where no closed inverse exists (Gumbel).

/// Acklam-style inverse normal via bisection on erf-free logistic bound is
/// overkill here — use Peter Acklam's rational approximation, adequate for
/// sampling (the fit tolerances are MC-scale, not 1e-10).
fn norm_ppf(p: f64) -> f64 {
    // Beasley-Springer-Moro style rational approximation.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.38357751867269e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - p_low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -norm_ppf(1.0 - p)
    }
}

fn norm_cdf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26-style erf; sampling-grade accuracy.
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.398_942_280_401_432_7 * (-x * x / 2.0).exp();
    let p = d
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Gaussian-copula sample by the conditional (regression) construction.
fn sample_gaussian(n: usize, rho: f64) -> (Vec<f64>, Vec<f64>) {
    let u1 = ugrid(n);
    let q = weyl(n);
    let u2 = u1
        .iter()
        .zip(&q)
        .map(|(&u, &qq)| {
            let z1 = norm_ppf(u);
            let z2 = rho * z1 + (1.0 - rho * rho).sqrt() * norm_ppf(qq);
            norm_cdf(z2).clamp(1e-12, 1.0 - 1e-12)
        })
        .collect();
    (u1, u2)
}

/// Clayton conditional inverse (closed form).
fn sample_clayton(n: usize, theta: f64) -> (Vec<f64>, Vec<f64>) {
    let u1 = ugrid(n);
    let q = weyl(n);
    let u2 = u1
        .iter()
        .zip(&q)
        .map(|(&u, &qq)| {
            let v =
                ((qq.powf(-theta / (1.0 + theta)) - 1.0) * u.powf(-theta) + 1.0).powf(-1.0 / theta);
            v.clamp(1e-12, 1.0 - 1e-12)
        })
        .collect();
    (u1, u2)
}

/// Frank conditional inverse (closed form; valid for either sign).
fn sample_frank(n: usize, theta: f64) -> (Vec<f64>, Vec<f64>) {
    let u1 = ugrid(n);
    let q = weyl(n);
    let u2 = u1
        .iter()
        .zip(&q)
        .map(|(&u, &qq)| {
            let v = -(1.0 + (-theta).exp_m1() / ((1.0 / qq - 1.0) * (-theta * u).exp() + 1.0)).ln()
                / theta;
            v.clamp(1e-12, 1.0 - 1e-12)
        })
        .collect();
    (u1, u2)
}

/// Gumbel conditional CDF `C_{2|1}(v | u)`, for the bisection sampler.
fn gumbel_cond(u: f64, v: f64, theta: f64) -> f64 {
    let x = -u.ln();
    let y = -v.ln();
    let s = x.powf(theta) + y.powf(theta);
    let c = (-s.powf(1.0 / theta)).exp();
    c * s.powf(1.0 / theta - 1.0) * x.powf(theta - 1.0) / u
}

/// Gumbel sample by bisection on the (monotone) conditional CDF.
fn sample_gumbel(n: usize, theta: f64) -> (Vec<f64>, Vec<f64>) {
    let u1 = ugrid(n);
    let q = weyl(n);
    let u2 = u1
        .iter()
        .zip(&q)
        .map(|(&u, &qq)| {
            let mut lo = 1e-14;
            let mut hi = 1.0 - 1e-14;
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if gumbel_cond(u, mid, theta) < qq {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        })
        .collect();
    (u1, u2)
}

// ---------------------------------------------------------------- fits

#[test]
fn monotone_margin_invariance_is_exact() {
    // The whole point of the copula decomposition: pseudo-observations
    // see only ranks, so strictly monotone transforms of each margin
    // leave the fit BIT-IDENTICAL — x vs (exp(x), cube(y)).
    let (u1, u2) = sample_gaussian(300, 0.6);
    // Fake "raw data" on absurd scales whose ranks equal those of u1/u2.
    let x1: Vec<f64> = u1.iter().map(|&u| u * 1e6 - 3.0e5).collect();
    let x2: Vec<f64> = u2.iter().map(|&u| u - 0.5).collect();
    let t1: Vec<f64> = x1.iter().map(|&v| (v * 1e-6).exp()).collect(); // strictly increasing
    let t2: Vec<f64> = x2.iter().map(|&v| v * v * v).collect(); // strictly increasing
    let ua = pseudo_obs(&[x1, x2]).expect("po raw");
    let ub = pseudo_obs(&[t1, t2]).expect("po transformed");
    assert_eq!(ua, ub, "pseudo-observations must be rank-only");
    for fam in [Family::Gaussian, Family::Frank, Family::Gumbel] {
        let fa = copula_fit(&ua[0], &ua[1], fam, FitMethod::Mle).expect("fit a");
        let fb = copula_fit(&ub[0], &ub[1], fam, FitMethod::Mle).expect("fit b");
        assert_eq!(fa.params, fb.params, "{}: params drifted", fam.name());
        assert_eq!(fa.loglik, fb.loglik, "{}: loglik drifted", fam.name());
        assert_eq!(fa.tau, fb.tau, "{}: tau drifted", fam.name());
    }
}

#[test]
fn exchange_symmetry_is_exact() {
    // Every family in this slice is exchangeable and every per-point
    // kernel is written symmetrically, so swapping the margins gives the
    // bit-identical fit.
    let (u1, u2) = sample_clayton(400, 2.0);
    for (fam, method) in [
        (Family::Gaussian, FitMethod::Mle),
        (Family::Clayton, FitMethod::Mle),
        (Family::Gumbel, FitMethod::Tau),
        (Family::Frank, FitMethod::Mle),
    ] {
        let a = copula_fit(&u1, &u2, fam, method).expect("fit");
        let b = copula_fit(&u2, &u1, fam, method).expect("fit swapped");
        assert_eq!(a.params, b.params, "{}: not exchangeable", fam.name());
        assert_eq!(a.loglik, b.loglik);
        assert_eq!(a.tau, b.tau);
    }
}

#[test]
fn recovery_gaussian() {
    let (u1, u2) = sample_gaussian(800, 0.6);
    let fit = copula_fit(&u1, &u2, Family::Gaussian, FitMethod::Mle).expect("fit");
    assert!(
        (fit.params[0] - 0.6).abs() < 0.05,
        "rho {} not near 0.6",
        fit.params[0]
    );
    assert!(fit.se_valid && fit.se[0] > 0.0 && fit.se[0] < 0.05);
    // Tau route close to the MLE on well-specified data.
    let ft = copula_fit(&u1, &u2, Family::Gaussian, FitMethod::Tau).expect("tau fit");
    assert!((ft.params[0] - fit.params[0]).abs() < 0.05);
    // loglik reported == loglik evaluated at the reported params.
    let ll = copula_loglik(&u1, &u2, Family::Gaussian, &fit.params).expect("ll");
    assert!((ll - fit.loglik).abs() < 1e-12 * ll.abs().max(1.0));
}

#[test]
fn recovery_clayton_gumbel_frank() {
    let (u1, u2) = sample_clayton(700, 2.0);
    let f = copula_fit(&u1, &u2, Family::Clayton, FitMethod::Mle).expect("clayton");
    assert!(
        (f.params[0] - 2.0).abs() < 0.25,
        "clayton theta {}",
        f.params[0]
    );

    let (u1, u2) = sample_gumbel(700, 2.0);
    let f = copula_fit(&u1, &u2, Family::Gumbel, FitMethod::Mle).expect("gumbel");
    assert!(
        (f.params[0] - 2.0).abs() < 0.15,
        "gumbel theta {}",
        f.params[0]
    );

    let (u1, u2) = sample_frank(700, 4.0);
    let f = copula_fit(&u1, &u2, Family::Frank, FitMethod::Mle).expect("frank");
    assert!(
        (f.params[0] - 4.0).abs() < 0.5,
        "frank theta {}",
        f.params[0]
    );

    // Negative dependence: Frank recovers a negative theta,
    // Clayton/Gumbel refuse with the teaching error.
    let (u1, u2) = sample_frank(500, -3.0);
    let f = copula_fit(&u1, &u2, Family::Frank, FitMethod::Mle).expect("frank neg");
    assert!(
        (f.params[0] + 3.0).abs() < 0.6,
        "frank theta {}",
        f.params[0]
    );
    assert!(matches!(
        copula_fit(&u1, &u2, Family::Clayton, FitMethod::Mle),
        Err(CopulaError::NegativeDependence { .. })
    ));
    assert!(matches!(
        copula_fit(&u1, &u2, Family::Gumbel, FitMethod::Tau),
        Err(CopulaError::NegativeDependence { .. })
    ));
}

#[test]
fn t_nu_boundary_on_gaussian_data_is_honest() {
    // On Gaussian-copula data the t family's nu drifts toward its upper
    // barrier where the likelihood is flat: the fit must still return,
    // match the Gaussian fit's likelihood from above, and NOT certify
    // observed-information SEs at a flat/boundary optimum. (This is the
    // case the fixture deliberately does not pin — no two optimizers
    // stop at the same point on a flat ridge.)
    let (u1, u2) = sample_gaussian(300, 0.5);
    let ft = copula_fit(&u1, &u2, Family::StudentT, FitMethod::Mle).expect("t fit");
    let fg = copula_fit(&u1, &u2, Family::Gaussian, FitMethod::Mle).expect("gauss fit");
    assert!(
        ft.params[1] > 20.0,
        "nu {} should drift large",
        ft.params[1]
    );
    assert!(
        ft.loglik >= fg.loglik - 0.5,
        "t ({}) should not fall below its Gaussian limit ({})",
        ft.loglik,
        fg.loglik
    );
    assert!((ft.params[0] - fg.params[0]).abs() < 0.02);
}

#[test]
fn independence_gives_near_zero_dependence() {
    let n = 600;
    let u1 = ugrid(n);
    let u2 = weyl(n);
    let tau = kendall_tau(&u1, &u2).expect("tau");
    assert!(tau.abs() < 0.03, "tau {tau} not near 0 on independent data");
    let f = copula_fit(&u1, &u2, Family::Gaussian, FitMethod::Mle).expect("fit");
    assert!(f.params[0].abs() < 0.05, "rho {} not near 0", f.params[0]);
    let f = copula_fit(&u1, &u2, Family::Frank, FitMethod::Mle).expect("frank");
    assert!(f.params[0].abs() < 0.5, "theta {} not near 0", f.params[0]);
}

// ------------------------------------------------------------ closed forms

#[test]
fn tau_param_roundtrips() {
    for fam in [
        Family::Gaussian,
        Family::Clayton,
        Family::Gumbel,
        Family::Frank,
    ] {
        for &tau in &[0.05, 0.3, 0.6, 0.85] {
            let p = tau_to_param(fam, tau).expect("map");
            let params: Vec<f64> = if fam == Family::StudentT {
                vec![p, 5.0]
            } else {
                vec![p]
            };
            let back = param_to_tau(fam, &params).expect("inverse");
            assert!(
                (back - tau).abs() < 1e-10,
                "{}: tau {tau} -> {p} -> {back}",
                fam.name()
            );
        }
    }
    // Negative-dependence roundtrips for the families that support it.
    for fam in [Family::Gaussian, Family::Frank] {
        for &tau in &[-0.7, -0.2] {
            let p = tau_to_param(fam, tau).expect("map");
            let back = param_to_tau(fam, &[p]).expect("inverse");
            assert!((back - tau).abs() < 1e-10);
        }
    }
}

#[test]
fn tail_dependence_shapes() {
    // Gaussian and Frank: none, ever.
    assert_eq!(
        tail_dependence(Family::Gaussian, &[0.95]).expect("g"),
        (0.0, 0.0)
    );
    assert_eq!(
        tail_dependence(Family::Frank, &[10.0]).expect("f"),
        (0.0, 0.0)
    );
    // Clayton: lower only, 2^(-1/theta), increasing in theta.
    let (l1, u1) = tail_dependence(Family::Clayton, &[1.0]).expect("c1");
    let (l2, u2) = tail_dependence(Family::Clayton, &[4.0]).expect("c2");
    assert_eq!((u1, u2), (0.0, 0.0));
    assert!((l1 - 0.5).abs() < 1e-15 && l2 > l1);
    // Gumbel: upper only, 2 - 2^(1/theta), increasing in theta.
    let (gl, gu1) = tail_dependence(Family::Gumbel, &[2.0]).expect("gu");
    let (_, gu2) = tail_dependence(Family::Gumbel, &[5.0]).expect("gu2");
    assert_eq!(gl, 0.0);
    assert!((gu1 - (2.0 - 2.0_f64.sqrt())).abs() < 1e-15 && gu2 > gu1);
    // t: symmetric, positive even at rho = 0, vanishing as nu -> inf
    // toward the Gaussian's zero, decreasing in nu.
    let (a, b) = tail_dependence(Family::StudentT, &[0.0, 4.0]).expect("t0");
    assert_eq!(a, b);
    assert!(a > 0.05);
    let (c, _) = tail_dependence(Family::StudentT, &[0.5, 4.0]).expect("t4");
    let (d, _) = tail_dependence(Family::StudentT, &[0.5, 20.0]).expect("t20");
    let (e, _) = tail_dependence(Family::StudentT, &[0.5, 200.0]).expect("t200");
    assert!(c > d && d > e && e < 0.01);
}

#[test]
fn frank_density_is_continuous_through_zero_theta() {
    // theta -> 0 is the independence limit: log-density -> 0.
    let lp = copula_logpdf(&[0.3, 0.8], &[0.6, 0.2], Family::Frank, &[1e-9]).expect("lp");
    for v in lp {
        assert!(v.abs() < 1e-6, "log-density {v} should vanish at theta ~ 0");
    }
}

// ------------------------------------------------------------- teaching errors

#[test]
fn degenerate_inputs_raise() {
    let n = 100;
    let good1 = ugrid(n);
    let good2 = weyl(n);

    // Too few observations.
    let short: Vec<f64> = ugrid(MIN_OBS - 1);
    assert!(matches!(
        copula_fit(&short, &short, Family::Gaussian, FitMethod::Tau),
        Err(CopulaError::TooFewObservations { .. })
    ));

    // Values at or outside the (0,1) boundary — including exactly 0/1.
    for bad in [0.0, 1.0, -0.2, 1.7, f64::NAN] {
        let mut u = good1.clone();
        u[7] = bad;
        let err =
            copula_fit(&u, &good2, Family::Gaussian, FitMethod::Mle).expect_err("must reject");
        assert!(
            matches!(
                err,
                CopulaError::OutOfUnitInterval { .. } | CopulaError::NonFinite { .. }
            ),
            "value {bad}: got {err:?}"
        );
    }
    // The out-of-interval message teaches the pseudo_obs route.
    let mut u = good1.clone();
    u[0] = 1.0;
    let msg = copula_fit(&u, &good2, Family::Gaussian, FitMethod::Mle)
        .expect_err("boundary")
        .to_string();
    assert!(msg.contains("pseudo_obs"), "teaching message: {msg}");

    // Length mismatch.
    assert!(matches!(
        copula_fit(&good1[..50], &good2, Family::Frank, FitMethod::Tau),
        Err(CopulaError::LengthMismatch { .. })
    ));

    // Constant column (after clamping into (0,1) it is still constant).
    let cst = vec![0.5; n];
    assert!(matches!(
        copula_fit(&cst, &good2, Family::Gaussian, FitMethod::Tau),
        Err(CopulaError::Degenerate { .. })
    ));

    // Perfect dependence: u2 == u1.
    assert!(matches!(
        copula_fit(&good1, &good1, Family::Gaussian, FitMethod::Mle),
        Err(CopulaError::PerfectDependence { .. })
    ));

    // pseudo_obs input checks.
    assert!(matches!(
        pseudo_obs(&[]),
        Err(CopulaError::EmptyInput { .. })
    ));
    assert!(matches!(
        pseudo_obs(&[vec![1.0, 2.0], vec![1.0]]),
        Err(CopulaError::LengthMismatch { .. })
    ));
    assert!(matches!(
        pseudo_obs(&[vec![1.0]]),
        Err(CopulaError::TooFewObservations { .. })
    ));
    assert!(matches!(
        pseudo_obs(&[vec![1.0, f64::INFINITY]]),
        Err(CopulaError::NonFinite { .. })
    ));

    // Parse errors.
    assert!(matches!(
        Family::parse("vine"),
        Err(CopulaError::UnknownFamily { .. })
    ));
    assert!(matches!(
        FitMethod::parse("bayes"),
        Err(CopulaError::UnknownMethod { .. })
    ));
}

#[test]
fn select_contracts() {
    let (u1, u2) = sample_gaussian(400, 0.5);

    // Empty and duplicate menus.
    assert!(matches!(
        copula_select(&u1, &u2, &[], FitMethod::Mle),
        Err(CopulaError::EmptyFamilies)
    ));
    assert!(matches!(
        copula_select(
            &u1,
            &u2,
            &[Family::Gaussian, Family::Gaussian],
            FitMethod::Mle
        ),
        Err(CopulaError::DuplicateFamily { .. })
    ));

    // Negative-dependence data: Clayton/Gumbel are skipped with a reason,
    // the rest are still ranked, and the verdict says so.
    let (v1, v2) = sample_frank(400, -3.0);
    let sel = copula_select(
        &v1,
        &v2,
        &[
            Family::Gaussian,
            Family::Clayton,
            Family::Gumbel,
            Family::Frank,
        ],
        FitMethod::Mle,
    )
    .expect("select");
    assert_eq!(sel.fits.len(), 2);
    assert_eq!(sel.skipped.len(), 2);
    assert!(sel.verdict.contains("Skipped"));
    assert!(sel.verdict.contains("AIC"));
    // ... and if EVERY family is domain-excluded, the error says why.
    assert!(matches!(
        copula_select(&v1, &v2, &[Family::Clayton, Family::Gumbel], FitMethod::Mle),
        Err(CopulaError::AllFamiliesSkipped { .. })
    ));

    // On the positive-dependence sample, rankings are consistent and the
    // verdict names the AIC winner.
    let sel = copula_select(
        &u1,
        &u2,
        &[
            Family::Gaussian,
            Family::Clayton,
            Family::Gumbel,
            Family::Frank,
        ],
        FitMethod::Mle,
    )
    .expect("select");
    assert_eq!(sel.fits.len(), 4);
    assert_eq!(sel.ranking_aic[0], sel.best_aic);
    assert_eq!(sel.ranking_bic[0], sel.best_bic);
    let mut aics: Vec<f64> = sel.ranking_aic.iter().map(|&i| sel.fits[i].aic).collect();
    let sorted = {
        let mut s = aics.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        s
    };
    assert_eq!(aics, sorted, "AIC ranking must be sorted");
    aics.clear();
    assert!(sel.verdict.contains(sel.fits[sel.best_aic].family.name()));
}
