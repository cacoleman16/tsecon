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

/// **Boundary fits are reported, never silently all-NaN** (audit rounds
/// 1/7). White noise fitted as GARCH(1,1) drives `alpha` to its sign
/// constraint (~1e-14): the observed information is singular in that
/// direction by construction, and the pre-fix behaviour was an unflagged
/// all-NaN `se_mle`/`se_robust` row. Now the boundary parameter is
/// flagged (`boundary`, `se_valid = false`, a teaching note) while the
/// interior parameters keep finite standard errors from the reduced
/// Hessian over the free directions.
#[test]
fn boundary_fit_flags_and_keeps_interior_standard_errors() {
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    // Seeds chosen so the alpha estimate lands at the constraint (the
    // pre-fix all-NaN reproduction); asserted, not assumed.
    let mut boundary_seen = 0;
    for seed in [2, 3, 5] {
        let y: Vec<f64> = {
            let mut rng = SplitMix64(seed);
            (0..750).map(|_| rng.normal()).collect()
        };
        let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
        let alpha = res.params[1];
        if alpha > 1e-6 {
            continue; // this seed did not produce a boundary fit
        }
        boundary_seen += 1;
        // alpha is flagged, and its NaN is a *flagged* NaN.
        assert!(res.boundary[1], "alpha at {alpha:e} must be flagged");
        assert!(!res.se_valid[1]);
        assert!(res.se_mle[1].is_nan() && res.se_robust[1].is_nan());
        // omega is interior and must have finite standard errors — the
        // round-1 defect was exactly this row coming back NaN.
        assert!(
            !res.boundary[0] && res.se_valid[0],
            "omega must be interior/valid, se_mle = {:?}",
            res.se_mle
        );
        assert!(
            res.se_mle[0].is_finite()
                && res.se_mle[0] > 0.0
                && res.se_robust[0].is_finite()
                && res.se_robust[0] > 0.0,
            "interior omega standard errors must be finite: mle {} robust {}",
            res.se_mle[0],
            res.se_robust[0]
        );
        // Flags, values, and note are mutually consistent.
        for i in 0..res.params.len() {
            assert_eq!(
                res.se_valid[i],
                !res.boundary[i] && res.se_mle[i].is_finite() && res.se_robust[i].is_finite(),
                "se_valid[{i}] inconsistent"
            );
        }
        let note = res.boundary_note.as_deref().expect("boundary note");
        assert!(
            note.contains("alpha[1]"),
            "note names the parameter: {note}"
        );
        assert!(
            note.contains("sign constraint"),
            "note states the cause: {note}"
        );
    }
    assert!(
        boundary_seen >= 2,
        "the white-noise DGP no longer produces boundary fits ({boundary_seen}/3); \
         the reproduction has drifted — re-derive the seeds"
    );
}

/// An interior fit carries clean flags: every `se_valid` true, no
/// boundary, no note — so the flags cannot cry wolf on routine data.
#[test]
fn interior_fit_flags_are_clean() {
    let y = simulate_garch11(0.1, 0.1, 0.8, 1500, 42);
    let spec = GarchSpec {
        mean: MeanSpec::Constant,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
    assert!(res.se_valid.iter().all(|&v| v), "{:?}", res.se_valid);
    assert!(res.boundary.iter().all(|&b| !b), "{:?}", res.boundary);
    assert!(res.boundary_note.is_none());
}

/// An integrated (IGARCH-boundary) fit flags every variance coefficient
/// while `omega` keeps a finite standard error, and the note names the
/// persistence bound.
#[test]
fn igarch_boundary_flags_all_coefficients() {
    // An IGARCH DGP (alpha + beta = 1) attracts the QMLE to the
    // persistence constraint on many draws.
    let mut hit = 0;
    for seed in [0u64, 1, 9, 12, 21] {
        let y = {
            // simulate with alpha + beta = 1: variance is a martingale, the
            // path stays finite over 750 draws.
            let mut rng = SplitMix64(seed);
            let (omega, alpha, beta) = (0.02, 0.10, 0.90);
            let mut s2 = 1.0_f64;
            let mut eps = 0.0_f64;
            let mut out = Vec::with_capacity(750);
            for t in 0..1250 {
                s2 = omega + alpha * eps * eps + beta * s2;
                eps = s2.sqrt() * rng.normal();
                if t >= 500 {
                    out.push(eps);
                }
            }
            out
        };
        let spec = GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        };
        let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
        let pers = spec.persistence(&res.params).unwrap();
        if pers < 0.9995 || res.params[0] < 1e-8 {
            // Not attracted to the bound on this draw, or omega itself
            // collapsed toward 0 (a doubly degenerate fit whose reduced
            // problem is honestly all-invalid — covered by the flag
            // consistency assertions in the white-noise test).
            continue;
        }
        hit += 1;
        assert!(
            res.boundary[1] && res.boundary[2],
            "persistence {pers} at the bound must flag alpha and beta: {:?}",
            res.boundary
        );
        assert!(!res.se_valid[1] && !res.se_valid[2]);
        assert!(
            res.se_valid[0] && res.se_mle[0].is_finite() && res.se_robust[0].is_finite(),
            "omega keeps a finite standard error at an IGARCH boundary"
        );
        let note = res.boundary_note.as_deref().expect("boundary note");
        assert!(note.contains("IGARCH"), "note names the bound: {note}");
    }
    assert!(
        hit >= 1,
        "no IGARCH draw reached the persistence bound; re-derive the seeds"
    );
}

/// **Fitting commutes with rescaling — bit-exactly for power-of-two
/// scales** (audit rounds 1/7). `y -> c y` is a pure relabeling of the
/// model (`omega -> c^2 omega`, `mu -> c mu`, coefficients unchanged), so
/// the estimator should commute with it. Since 0.4.0 the optimizer runs
/// on the internally standardized series `y / rms(y)`; for `c = 2^k`
/// every step of the standardization is an exact exponent shift, so the
/// standardized series — and therefore the entire optimizer path — is
/// bit-identical, and the mapped-back parameters are exact. This is a
/// same-run invariant (one binary, one CPU), so bit-equality is portable.
#[test]
fn fit_commutes_bitexactly_with_power_of_two_rescaling() {
    let base = simulate_garch11(0.05, 0.08, 0.88, 900, 13);
    for spec in [
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        },
        GarchSpec {
            mean: MeanSpec::Constant,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        },
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Gjr { p: 1, o: 1, q: 1 },
            dist: DistSpec::Normal,
        },
    ] {
        let reference = GarchModel::new(&base, spec).unwrap().fit().unwrap();
        let names = spec.param_names();
        for k in [-30i32, -8, 8, 30] {
            let c = (2.0_f64).powi(k);
            let y: Vec<f64> = base.iter().map(|v| v * c).collect();
            let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
            for (i, nm) in names.iter().enumerate() {
                let expected = match nm.as_str() {
                    "mu" => reference.params[i] * c,
                    "omega" => reference.params[i] * c * c,
                    _ => reference.params[i],
                };
                assert_eq!(
                    res.params[i].to_bits(),
                    expected.to_bits(),
                    "{spec:?} c=2^{k}: {nm} = {} but the mapped reference is {expected} — \
                     an exact rescaling moved the fit",
                    res.params[i]
                );
            }
        }
    }
}

/// The same commutation under *decade* scalings (the audit's own probe
/// design). `c = 10^k` rounds in `c * y`, so bit-equality is impossible
/// by construction — the two series are genuinely different data — but
/// the standardized optimizer must land at the same point to far beyond
/// statistical resolution. 1e-6 relative on every parameter, eight
/// decades each side.
#[test]
fn fit_commutes_with_decade_rescaling() {
    let base = simulate_garch11(0.05, 0.08, 0.88, 900, 13);
    let spec = GarchSpec {
        mean: MeanSpec::Constant,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let reference = GarchModel::new(&base, spec).unwrap().fit().unwrap();
    let names = spec.param_names();
    let rms = (base.iter().map(|v| v * v).sum::<f64>() / base.len() as f64).sqrt();
    for k in [-8i32, -4, 4, 8] {
        let c = (10.0_f64).powi(k);
        let y: Vec<f64> = base.iter().map(|v| v * c).collect();
        let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
        for (i, nm) in names.iter().enumerate() {
            let mapped = match nm.as_str() {
                "mu" => res.params[i] / c,
                "omega" => res.params[i] / (c * c),
                _ => res.params[i],
            };
            // `mu` is a location whose natural scale is the residual RMS.
            let denom = if nm == "mu" {
                rms
            } else {
                reference.params[i].abs()
            };
            let dev = (mapped - reference.params[i]).abs() / denom;
            assert!(
                dev < 1e-6,
                "c=1e{k}: {nm} mapped to {mapped} vs {} ({dev:.2e} relative)",
                reference.params[i]
            );
        }
    }
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

/// The exponent of `c` that each parameter (and hence its standard error)
/// carries when the data are rescaled `y -> c * y`. `omega` is a variance
/// (`c^2`), `mu` is a location (`c^1`), and every coefficient is
/// dimensionless (`c^0`).
///
/// The EGARCH intercept is deliberately absent: it is a *log*-variance, so
/// rescaling shifts it by `(1 - sum(beta)) * 2 ln c` instead of stretching
/// it, and its standard error genuinely changes with the units. See
/// [`egarch_coefficient_standard_errors_are_scale_invariant`].
fn se_scale_exponents(spec: &GarchSpec) -> Vec<i32> {
    let mut e = Vec::with_capacity(spec.n_params());
    if matches!(spec.mean, MeanSpec::Constant) {
        e.push(1);
    }
    e.push(2); // omega, a variance
    let (p, o, q) = spec.vol.lags();
    e.extend(std::iter::repeat_n(0, p + o + q));
    if matches!(spec.dist, DistSpec::StudentT) {
        e.push(0); // nu
    }
    e
}

/// **Scale equivariance of the standard errors.** Rescaling the data
/// `y -> c * y` is an exact reparameterization of a GARCH/GJR model:
/// `omega -> c^2 omega`, `mu -> c mu`, every coefficient unchanged. The
/// standard errors must follow the same rule, so dividing each by `c` to
/// its parameter's exponent has to give the *same* numbers at every `c`.
///
/// This is a regression test for a silent wrong answer that a single-scale
/// test cannot see. The finite-difference steps behind the covariance used
/// to carry statsmodels' absolute floor, `h_i = eps^(1/4) max(|theta_i|,
/// 0.1)`. Daily equity returns quoted in *decimals* rather than percent
/// put `omega` around `1e-6`, twelve orders of magnitude below that floor:
/// the Hessian probe pushed `omega` negative, the likelihood refused the
/// point, and `fit` reported `se_mle = se_robust = [NaN, ...]` with no
/// error. One decade up (`omega ~ 8e-5`) it did not fail, it just came
/// back 8% (MLE) and 15% (robust) wrong. Steps are now scaled per
/// parameter in that parameter's own units, so the sweep below spans ten
/// decades of `c` -- five orders of magnitude either side of the percent
/// scale `arch` is fitted on -- and holds to 1e-3.
#[test]
fn standard_errors_are_scale_equivariant() {
    // sd(base) ~ 0.0068: daily equity returns in decimals, the scale at
    // which the absolute step floor used to produce all-NaN.
    let base = simulate_garch11(1e-6, 0.08, 0.90, 2500, 7);
    let specs = [
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        },
        GarchSpec {
            mean: MeanSpec::Constant,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        },
        GarchSpec {
            mean: MeanSpec::Constant,
            vol: VolSpec::Gjr { p: 1, o: 1, q: 1 },
            dist: DistSpec::Normal,
        },
    ];
    for spec in specs {
        let names = spec.param_names();
        let exps = se_scale_exponents(&spec);
        let mut reference: Option<(Vec<f64>, Vec<f64>)> = None;
        for &c in &[1e-4_f64, 1e-2, 1.0, 1e2, 1e4, 1e6] {
            let y: Vec<f64> = base.iter().map(|v| v * c).collect();
            let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
            let rescale = |v: &[f64]| -> Vec<f64> {
                v.iter().zip(&exps).map(|(&x, &e)| x / c.powi(e)).collect()
            };
            let (mle, robust) = (rescale(&res.se_mle), rescale(&res.se_robust));
            for (i, nm) in names.iter().enumerate() {
                // The headline symptom: never NaN, never a non-positive
                // "standard error", at any scale.
                for (which, v) in [("mle", mle[i]), ("robust", robust[i])] {
                    assert!(
                        v.is_finite() && v > 0.0,
                        "{spec:?} c={c:e}: se_{which}[{nm}] = {v}"
                    );
                }
            }
            match &reference {
                None => reference = Some((mle, robust)),
                Some((ref_mle, ref_robust)) => {
                    for (i, nm) in names.iter().enumerate() {
                        for (which, v, r) in [
                            ("mle", mle[i], ref_mle[i]),
                            ("robust", robust[i], ref_robust[i]),
                        ] {
                            let dev = ((v - r) / r).abs();
                            assert!(
                                dev < 1e-3,
                                "{spec:?} se_{which}[{nm}]: {v:e} at c={c:e} vs {r:e} at the \
                                 reference scale -- {dev:.2e} relative, but rescaling the data \
                                 cannot move a standard error at all"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// EGARCH's dimensionless coefficients under the same rescaling.
///
/// `alpha`, `gamma` and `beta` are untouched by `y -> c * y` (only the
/// log-variance intercept absorbs the units, and it does so by a *shift*
/// of `(1 - sum(beta)) * 2 ln c`, so `se(omega)` legitimately moves and is
/// not asserted here). The coefficient standard errors are exactly
/// invariant in exact arithmetic; the tolerance is 3e-2 because the
/// intercept's own step still follows statsmodels' `max(|omega|, 0.1)` --
/// correct for an O(1) log-scale quantity, but it does drift as the
/// intercept slides past 0.1, and that drift leaks into the rest of the
/// Hessian. This pins the leak at its measured size (<= 1e-2) so a
/// regression that reintroduced a units mistake here would be caught.
#[test]
fn egarch_coefficient_standard_errors_are_scale_invariant() {
    let base = simulate_garch11(1e-6, 0.08, 0.90, 2500, 7);
    let spec = GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Egarch { p: 1, o: 1, q: 1 },
        dist: DistSpec::Normal,
    };
    let names = spec.param_names();
    let mut reference: Option<Vec<f64>> = None;
    for &c in &[1.0_f64, 1e1, 1e2, 1e3] {
        let y: Vec<f64> = base.iter().map(|v| v * c).collect();
        let res = GarchModel::new(&y, spec).unwrap().fit().unwrap();
        assert!(
            res.se_robust.iter().all(|v| v.is_finite() && *v > 0.0),
            "c={c:e}: se_robust = {:?}",
            res.se_robust
        );
        // Skip the intercept (index 0 under a zero mean).
        let coefs: Vec<f64> = res.se_robust[1..].to_vec();
        match &reference {
            None => reference = Some(coefs),
            Some(rf) => {
                for (i, (&v, &r)) in coefs.iter().zip(rf).enumerate() {
                    let dev = ((v - r) / r).abs();
                    assert!(
                        dev < 3e-2,
                        "egarch se_robust[{}]: {v:e} at c={c:e} vs {r:e} at c=1 -- {dev:.2e} \
                         relative, but the EGARCH coefficients are unit-free",
                        names[i + 1]
                    );
                }
            }
        }
    }
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
