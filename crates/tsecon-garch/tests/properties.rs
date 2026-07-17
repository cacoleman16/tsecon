//! Property tests: variance positivity on random admissible parameters,
//! the Student-t -> normal limit, seeded parameter recovery on simulated
//! data, rejection of inadmissible parameters, forecast convergence to the
//! unconditional variance, and equivalence of the t likelihood kernel with
//! `tsecon_stats::Standardized`.

mod common;

use common::{assert_abs_close, assert_rel_close, SplitMix64};
use tsecon_garch::{DistSpec, GarchError, GarchModel, GarchSpec, MeanSpec, VolSpec};
use tsecon_stats::{ContinuousDist, Standardized};

/// A reproducible synthetic return series (not itself a GARCH process;
/// positivity and validation properties do not care).
fn synthetic_returns(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64(seed);
    (0..n)
        .map(|_| rng.normal() * (1.0 + 0.5 * rng.uniform()))
        .collect()
}

/// Simulates a GARCH(1,1) path with normal innovations (500-observation
/// burn-in, unconditional-variance start).
fn simulate_garch11(omega: f64, alpha: f64, beta: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64(seed);
    let burn = 500;
    let mut s2 = omega / (1.0 - alpha - beta);
    let mut eps = 0.0;
    let mut out = Vec::with_capacity(n);
    for t in 0..n + burn {
        s2 = omega + alpha * eps * eps + beta * s2;
        eps = s2.sqrt() * rng.normal();
        if t >= burn {
            out.push(eps);
        }
    }
    out
}

/// Conditional variances are strictly positive and finite for random
/// admissible parameters across all three recursions.
#[test]
fn variance_positive_on_admissible_params() {
    let y = synthetic_returns(300, 7);
    let mut rng = SplitMix64(99);
    for _ in 0..200 {
        // Random admissible parameters: alpha_i >= 0, alpha + gamma >= 0,
        // persistence (with the 0.5 gamma weight for GJR) below one.
        let alpha = 0.4 * rng.uniform();
        let gamma = -alpha + (alpha + 0.3) * rng.uniform();
        let beta = (1.0 - alpha - 1e-3) * rng.uniform();
        let gjr_beta = (1.0 - alpha - 0.5 * gamma - 1e-3).min(1.0) * rng.uniform();
        let omega = 0.001 + rng.uniform();

        let garch = GarchModel::new(
            &y,
            GarchSpec {
                mean: MeanSpec::Zero,
                vol: VolSpec::Garch { p: 1, q: 1 },
                dist: DistSpec::Normal,
            },
        )
        .unwrap();
        let s2 = garch.conditional_variance(&[omega, alpha, beta]).unwrap();
        assert!(s2.iter().all(|&s| s > 0.0 && s.is_finite()));

        let gjr = GarchModel::new(
            &y,
            GarchSpec {
                mean: MeanSpec::Constant,
                vol: VolSpec::Gjr { p: 1, o: 1, q: 1 },
                dist: DistSpec::Normal,
            },
        )
        .unwrap();
        let s2 = gjr
            .conditional_variance(&[0.05, omega, alpha, gamma, gjr_beta])
            .unwrap();
        assert!(s2.iter().all(|&s| s > 0.0 && s.is_finite()));

        // Random admissible EGARCH from the self-stabilizing region
        // (alpha > |gamma| makes the news response increasing in |z|, so
        // large shocks raise rather than collapse the variance; with a
        // negative alpha the recursion can genuinely drive sigma2 to zero
        // and the model reports NonFinite, which is separate honest
        // behavior, not a positivity bug).
        let e_omega = -0.2 + 0.4 * rng.uniform();
        let e_alpha = 0.05 + 0.35 * rng.uniform();
        let e_gamma = e_alpha * (-0.9 + 1.8 * rng.uniform());
        let e_beta = 0.98 * rng.uniform();
        let egarch = GarchModel::new(
            &y,
            GarchSpec {
                mean: MeanSpec::Zero,
                vol: VolSpec::Egarch { p: 1, o: 1, q: 1 },
                dist: DistSpec::Normal,
            },
        )
        .unwrap();
        let s2 = egarch
            .conditional_variance(&[e_omega, e_alpha, e_gamma, e_beta])
            .unwrap();
        assert!(s2.iter().all(|&s| s > 0.0 && s.is_finite()));
    }
}

/// The standardized-t log-likelihood approaches the normal one as
/// `nu -> 1e6` at identical variance parameters.
#[test]
fn t_loglik_approaches_normal_for_large_nu() {
    let y = synthetic_returns(1000, 21);
    let spec_n = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let spec_t = GarchSpec {
        dist: DistSpec::StudentT,
        ..spec_n
    };
    let vol_params = [0.05, 0.08, 0.9];
    let ll_n = GarchModel::new(&y, spec_n)
        .unwrap()
        .loglike(&vol_params)
        .unwrap();
    let model_t = GarchModel::new(&y, spec_t).unwrap();
    let ll_t = model_t
        .loglike(&[vol_params[0], vol_params[1], vol_params[2], 1e6])
        .unwrap();
    assert_abs_close(ll_t, ll_n, 1e-2, "t loglik at nu = 1e6 vs normal");
    // And the gap shrinks monotonically along nu = 10 -> 1e6.
    let ll_t10 = model_t
        .loglike(&[vol_params[0], vol_params[1], vol_params[2], 10.0])
        .unwrap();
    assert!((ll_t - ll_n).abs() < (ll_t10 - ll_n).abs());
}

/// The Student-t likelihood kernel is `ln f_Z(eps/sigma) - ln sigma` with
/// `f_Z` the unit-variance t of `tsecon_stats::Standardized` — checked
/// pointwise through the total likelihood of tiny one-parameter-block
/// models.
#[test]
fn t_loglik_matches_standardized_dist() {
    let y = synthetic_returns(50, 3);
    let nu = 6.5;
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::StudentT,
    };
    let model = GarchModel::new(&y, spec).unwrap();
    let params = [0.1, 0.1, 0.8, nu];
    let ll = model.loglike(&params).unwrap();
    let s2 = model.conditional_variance(&params).unwrap();
    let z = Standardized::student_t(nu).unwrap();
    let expected: f64 = y
        .iter()
        .zip(&s2)
        .map(|(&e, &s)| z.ln_pdf(e / s.sqrt()) - 0.5 * s.ln())
        .sum();
    assert_rel_close(ll, expected, 1e-12, "t loglik vs Standardized");
}

/// Seeded parameter recovery on simulated GARCH(1,1) data (loose bounds:
/// estimates are within a few standard errors of the truth).
#[test]
fn recovers_simulated_garch11_parameters() {
    let (omega, alpha, beta) = (0.1, 0.1, 0.8);
    let y = simulate_garch11(omega, alpha, beta, 3000, 42);
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
    assert!(res.converged, "fit should converge on clean simulated data");
    assert_abs_close(res.params[1], alpha, 0.05, "alpha recovery");
    assert_abs_close(res.params[2], beta, 0.08, "beta recovery");
    assert_abs_close(res.params[0], omega, 0.08, "omega recovery");
    let pers = spec.persistence(&res.params).unwrap();
    assert!(pers < 1.0, "fitted persistence {pers} must be stationary");
    // Standard errors came out finite and positive on this well-behaved
    // problem.
    assert!(res
        .se_robust
        .iter()
        .zip(&res.se_mle)
        .all(|(r, m)| r.is_finite() && *r > 0.0 && m.is_finite() && *m > 0.0));
}

/// Inadmissible parameter vectors are rejected: explosive persistence,
/// negative coefficients, non-positive omega, EGARCH |beta| >= 1, and
/// nu <= 2 all error rather than evaluate.
#[test]
fn inadmissible_params_rejected() {
    let y = synthetic_returns(100, 11);
    let garch = GarchModel::new(
        &y,
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        },
    )
    .unwrap();
    // persistence = 1.05 >= 1.
    assert!(matches!(
        garch.loglike(&[0.05, 0.3, 0.75]),
        Err(GarchError::InvalidParameter {
            name: "persistence",
            ..
        })
    ));
    // persistence exactly 1 (IGARCH) is also rejected in this release.
    assert!(garch.loglike(&[0.05, 0.2, 0.8]).is_err());
    assert!(matches!(
        garch.loglike(&[0.0, 0.1, 0.8]),
        Err(GarchError::InvalidParameter { name: "omega", .. })
    ));
    assert!(matches!(
        garch.loglike(&[0.05, -0.01, 0.8]),
        Err(GarchError::InvalidParameter { name: "alpha", .. })
    ));
    assert!(matches!(
        garch.loglike(&[0.05, 0.1, -0.2]),
        Err(GarchError::InvalidParameter { name: "beta", .. })
    ));

    let gjr = GarchModel::new(
        &y,
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Gjr { p: 1, o: 1, q: 1 },
            dist: DistSpec::Normal,
        },
    )
    .unwrap();
    // alpha + gamma < 0.
    assert!(matches!(
        gjr.loglike(&[0.05, 0.02, -0.05, 0.8]),
        Err(GarchError::InvalidParameter { name: "gamma", .. })
    ));

    let egarch = GarchModel::new(
        &y,
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Egarch { p: 1, o: 1, q: 1 },
            dist: DistSpec::Normal,
        },
    )
    .unwrap();
    assert!(matches!(
        egarch.loglike(&[0.01, 0.1, -0.05, 1.0]),
        Err(GarchError::InvalidParameter {
            name: "sum(beta)",
            ..
        })
    ));

    let t = GarchModel::new(
        &y,
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::StudentT,
        },
    )
    .unwrap();
    assert!(matches!(
        t.loglike(&[0.05, 0.1, 0.8, 2.0]),
        Err(GarchError::InvalidParameter { name: "nu", .. })
    ));

    // Wrong-length vectors and NaN are structural errors.
    assert!(matches!(
        garch.loglike(&[0.05, 0.1]),
        Err(GarchError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        garch.loglike(&[0.05, f64::NAN, 0.8]),
        Err(GarchError::NonFinite { .. })
    ));
}

/// Analytic forecasts: the one-step forecast continues the recursion
/// exactly, long-horizon GARCH/GJR forecasts converge to the
/// unconditional variance, and EGARCH multi-step is an explicit
/// unsupported error (one-step matches a hand computation).
#[test]
fn forecasts_converge_to_unconditional_variance() {
    let y = simulate_garch11(0.1, 0.1, 0.8, 800, 5);

    // GARCH(1,1): one-step continues the recursion; h -> infinity gives
    // omega / (1 - alpha - beta).
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
    let (omega, alpha, beta) = (res.params[0], res.params[1], res.params[2]);
    let f = res.forecast_variance(400).unwrap();
    let n = y.len();
    let eps = res.residuals()[n - 1];
    let s2 = res.conditional_variance()[n - 1];
    assert_rel_close(
        f[0],
        omega + alpha * eps * eps + beta * s2,
        1e-12,
        "one-step forecast continues the recursion",
    );
    let uncond = omega / (1.0 - alpha - beta);
    assert_rel_close(f[399], uncond, 1e-6, "long-horizon GARCH forecast");
    assert!(f.iter().all(|&v| v > 0.0 && v.is_finite()));

    // GJR: unconditional variance uses persistence with the 0.5 gamma
    // weight.
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Gjr { p: 1, o: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
    let f = res.forecast_variance(600).unwrap();
    let pers = spec.persistence(&res.params).unwrap();
    let uncond = res.params[0] / (1.0 - pers);
    assert_rel_close(f[599], uncond, 1e-5, "long-horizon GJR forecast");
    assert!(f.iter().all(|&v| v > 0.0 && v.is_finite()));

    // EGARCH: one-step analytic, multi-step unsupported (TODO(phase0)).
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Egarch { p: 1, o: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
    let f1 = res.forecast_variance(1).unwrap();
    let (omega, alpha, gamma, beta) = (res.params[0], res.params[1], res.params[2], res.params[3]);
    let z = res.std_residuals[n - 1];
    let expected = (omega
        + alpha * (z.abs() - (2.0 / std::f64::consts::PI).sqrt())
        + gamma * z
        + beta * res.conditional_variance()[n - 1].ln())
    .exp();
    assert_rel_close(f1[0], expected, 1e-12, "EGARCH one-step forecast");
    assert!(matches!(
        res.forecast_variance(2),
        Err(GarchError::UnsupportedForecast { .. })
    ));

    // Horizon zero is a structural error.
    assert!(res.forecast_variance(0).is_err());
}

/// Fitted models never report explosive persistence: the constraint set
/// keeps the QMLE search strictly inside the stationary region.
#[test]
fn fit_respects_stationarity() {
    // Near-integrated data: persistence should approach but not reach 1.
    let y = simulate_garch11(0.02, 0.15, 0.84, 1500, 17);
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
    let pers = spec.persistence(&res.params).unwrap();
    assert!(
        pers < 1.0,
        "fitted persistence {pers} escaped the constraint"
    );
    assert!(res.params[0] > 0.0, "omega stayed positive");
}

/// Construction errors: NaN data, too-short series, malformed lag
/// structures, and error display strings.
#[test]
fn construction_and_display() {
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    assert!(matches!(
        GarchModel::new(&[1.0, f64::NAN, 0.5, 1.0, -1.0, 0.3], spec),
        Err(GarchError::NonFinite { .. })
    ));
    assert!(matches!(
        GarchModel::new(&[1.0, -0.5], spec),
        Err(GarchError::InsufficientData { .. })
    ));
    let bad = GarchSpec {
        vol: VolSpec::Garch { p: 0, q: 1 },
        ..spec
    };
    assert!(matches!(
        GarchModel::new(&[1.0; 50], bad),
        Err(GarchError::InvalidSpec { .. })
    ));
    // Constant series has zero variance: no valid backcast.
    assert!(GarchModel::new(&[0.0; 50], spec).is_err());

    let e = GarchError::InvalidParameter {
        name: "omega",
        value: -1.0,
        requirement: "omega > 0",
    };
    assert!(!e.to_string().is_empty());
    assert_eq!(spec.param_names(), vec!["omega", "alpha[1]", "beta[1]"]);
    let full = GarchSpec {
        mean: MeanSpec::Constant,
        vol: VolSpec::Gjr { p: 1, o: 1, q: 1 },
        dist: DistSpec::StudentT,
    };
    assert_eq!(
        full.param_names(),
        vec!["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"]
    );
}
