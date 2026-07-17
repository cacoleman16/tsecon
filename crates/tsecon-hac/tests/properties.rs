//! Property and invariant tests beyond the statsmodels goldens, plus
//! error-path coverage.

use tsecon_hac::{
    andrews_bandwidth_ar1, ewc_default_b, ewc_lrv, lrv, lrv_prewhitened_ar1, newey_west_bandwidth,
    newey_west_maxlags, ols, HacError, Kernel, SeType,
};

const KERNELS: [Kernel; 4] = [
    Kernel::Bartlett,
    Kernel::Parzen,
    Kernel::QuadraticSpectral,
    Kernel::Truncated,
];

/// Deterministic pseudo-random uniforms in (-0.5, 0.5) via a 64-bit LCG
/// (Knuth MMIX constants) — no RNG dependency needed at this quality.
fn lcg_series(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
    }
    out
}

/// AR(1) series driven by the LCG innovations.
fn ar1_series(n: usize, phi: f64, seed: u64) -> Vec<f64> {
    let u = lcg_series(n, seed);
    let mut y = Vec::with_capacity(n);
    let mut prev = 0.0;
    for &e in &u {
        prev = phi * prev + e;
        y.push(prev);
    }
    y
}

fn demean(x: &[f64]) -> Vec<f64> {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| v - m).collect()
}

// ---------------------------------------------------------------- kernels

#[test]
fn every_kernel_has_unit_weight_at_lag_zero() {
    for k in KERNELS {
        for bw in [0.0, 1.0, 4.0, 12.5] {
            assert_eq!(k.weight(0, bw), 1.0, "{k:?} weight(0, {bw}) must be 1");
        }
    }
}

#[test]
fn kernel_weights_are_bounded_and_decay() {
    for k in KERNELS {
        for bw in [1.0, 5.0, 20.0] {
            for j in 0..200 {
                let w = k.weight(j, bw);
                assert!(w.is_finite(), "{k:?} weight({j}, {bw}) finite");
                assert!(w.abs() <= 1.0, "{k:?} |weight({j}, {bw})| <= 1");
            }
            // Far beyond the bandwidth the weight is (essentially) gone.
            let far = k.weight(2000, bw);
            assert!(
                far.abs() < 1e-4,
                "{k:?} weight at lag 2000, bw {bw} should be ~0, got {far}"
            );
        }
    }
}

#[test]
fn truncating_kernels_vanish_exactly_beyond_maxlags() {
    for k in [Kernel::Bartlett, Kernel::Parzen, Kernel::Truncated] {
        let bw = 6.0;
        assert!(k.weight(6, bw) > 0.0, "{k:?} keeps lag = maxlags");
        assert_eq!(k.weight(7, bw), 0.0, "{k:?} drops lag = maxlags + 1");
        assert!(k.truncates());
    }
    assert!(!Kernel::QuadraticSpectral.truncates());
    assert!(Kernel::QuadraticSpectral.weight(50, 6.0).abs() > 0.0);
}

#[test]
fn qs_kernel_matches_closed_form_away_from_zero() {
    // Spot-check k_QS(x) at x = 1 against the published formula value.
    let x = 1.0_f64;
    let d = 6.0 * std::f64::consts::PI * x / 5.0;
    let expected = 3.0 / (d * d) * (d.sin() / d - d.cos());
    let got = Kernel::QuadraticSpectral.weight(10, 10.0);
    assert!((got - expected).abs() < 1e-15);
}

#[test]
fn bandwidth_from_scale_reproduces_andrews_weighting() {
    // Bartlett/Parzen: k(j/(b+1)) with b = S_T - 1 equals k(j/S_T).
    let s_t = 7.3;
    for k in [Kernel::Bartlett, Kernel::Parzen] {
        let b = k.bandwidth_from_scale(s_t);
        assert!((b - (s_t - 1.0)).abs() < 1e-15);
    }
    assert_eq!(Kernel::QuadraticSpectral.bandwidth_from_scale(s_t), s_t);
    assert_eq!(Kernel::Truncated.bandwidth_from_scale(s_t), s_t);
    // Bartlett at b = S_T - 1, lag j: 1 - j/S_T.
    let b = Kernel::Bartlett.bandwidth_from_scale(s_t);
    assert!((Kernel::Bartlett.weight(3, b) - (1.0 - 3.0 / s_t)).abs() < 1e-15);
}

// -------------------------------------------------------------------- lrv

#[test]
fn lrv_of_white_noise_is_close_to_its_variance() {
    let z = demean(&lcg_series(4000, 42));
    let n = z.len() as f64;
    let gamma0 = z.iter().map(|v| v * v).sum::<f64>() / n;
    for k in KERNELS {
        let omega = lrv(&z, k, 6.0).unwrap();
        assert!(
            (omega / gamma0 - 1.0).abs() < 0.10,
            "{k:?}: white-noise LRV {omega} should be within 10% of variance {gamma0}"
        );
    }
}

#[test]
fn lrv_at_bandwidth_zero_is_gamma0_for_every_kernel() {
    let z = demean(&ar1_series(300, 0.5, 9));
    let gamma0 = z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64;
    for k in KERNELS {
        let omega = lrv(&z, k, 0.0).unwrap();
        assert!((omega - gamma0).abs() < 1e-12, "{k:?} lrv(bw=0) == gamma0");
    }
}

#[test]
fn lrv_of_persistent_ar1_exceeds_its_short_run_variance() {
    let z = demean(&ar1_series(2000, 0.7, 7));
    let gamma0 = z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64;
    for k in [Kernel::Bartlett, Kernel::Parzen, Kernel::QuadraticSpectral] {
        let omega = lrv(&z, k, 20.0).unwrap();
        assert!(
            omega > gamma0,
            "{k:?}: positive autocorrelation must inflate the LRV \
             (omega {omega} vs gamma0 {gamma0})"
        );
    }
}

// -------------------------------------------------------------------- ewc

#[test]
fn ewc_with_full_dof_recovers_the_naive_sample_variance() {
    // j = 1..n-1 cosine vectors plus the constant form an orthonormal
    // basis, so on a demeaned series EWC(B = n-1) = sum z^2 / (n-1).
    let z = demean(&ar1_series(200, 0.6, 11));
    let n = z.len();
    let naive = z.iter().map(|v| v * v).sum::<f64>() / (n - 1) as f64;
    let omega = ewc_lrv(&z, n - 1).unwrap();
    assert!(
        ((omega - naive) / naive).abs() < 1e-8,
        "EWC(B=n-1) {omega} should equal the naive variance {naive}"
    );
}

#[test]
fn ewc_default_b_follows_the_llsw_rule() {
    assert_eq!(
        ewc_default_b(100),
        (0.4_f64 * 100.0_f64.powf(2.0 / 3.0)).round() as usize
    );
    assert_eq!(ewc_default_b(1000), 40); // 0.4 * 1000^(2/3) = 40
    assert!(ewc_default_b(2) >= 1);
    // Clamped into the valid range for tiny n.
    assert_eq!(ewc_default_b(3), 1);
}

#[test]
fn ewc_of_white_noise_is_close_to_its_variance() {
    let z = demean(&lcg_series(4000, 5));
    let gamma0 = z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64;
    let omega = ewc_lrv(&z, ewc_default_b(z.len())).unwrap();
    assert!(
        (omega / gamma0 - 1.0).abs() < 0.25,
        "EWC white-noise LRV {omega} should be near the variance {gamma0}"
    );
}

// ------------------------------------------------------ automatic bandwidth

#[test]
fn automatic_bandwidths_are_positive_finite_and_stable_on_seeded_ar1() {
    let z = demean(&ar1_series(500, 0.7, 3));
    for k in [Kernel::Bartlett, Kernel::Parzen, Kernel::QuadraticSpectral] {
        let a = andrews_bandwidth_ar1(&z, k).unwrap();
        let nw = newey_west_bandwidth(&z, k).unwrap();
        for (name, bw) in [("andrews", a), ("newey-west", nw)] {
            assert!(bw.is_finite() && bw > 0.0, "{k:?} {name} bandwidth {bw}");
            assert!(
                (0.5..200.0).contains(&bw),
                "{k:?} {name} bandwidth {bw} outside a plausible range"
            );
        }
        // Determinism: same input, same value, bit for bit.
        assert_eq!(a, andrews_bandwidth_ar1(&z, k).unwrap());
        assert_eq!(nw, newey_west_bandwidth(&z, k).unwrap());
    }
    // Andrews also covers the truncated kernel (Table I constant).
    let tr = andrews_bandwidth_ar1(&z, Kernel::Truncated).unwrap();
    assert!(tr.is_finite() && tr > 0.0);
}

#[test]
fn automatic_bandwidths_grow_with_persistence() {
    let calm = demean(&ar1_series(800, 0.2, 17));
    let wild = demean(&ar1_series(800, 0.9, 17));
    for k in [Kernel::Bartlett, Kernel::QuadraticSpectral] {
        assert!(
            andrews_bandwidth_ar1(&wild, k).unwrap() > andrews_bandwidth_ar1(&calm, k).unwrap(),
            "{k:?}: Andrews bandwidth must grow with persistence"
        );
        assert!(
            newey_west_bandwidth(&wild, k).unwrap() > newey_west_bandwidth(&calm, k).unwrap(),
            "{k:?}: Newey-West bandwidth must grow with persistence"
        );
    }
}

#[test]
fn newey_west_maxlags_rule_of_thumb_values() {
    assert_eq!(newey_west_maxlags(100), 4);
    assert_eq!(newey_west_maxlags(200), 4); // 4 * 2^(2/9) = 4.67 -> 4
    assert_eq!(newey_west_maxlags(1000), 6); // 4 * 10^(2/9) = 6.63 -> 6
}

// ----------------------------------------------------------- prewhitening

#[test]
fn prewhitened_lrv_beats_raw_kernel_lrv_on_persistent_ar1() {
    // AR(1), phi = 0.7, uniform(-0.5, 0.5) innovations: sigma2 = 1/12,
    // true LRV = sigma2 / (1 - phi)^2.
    let phi = 0.7;
    let z = demean(&ar1_series(3000, phi, 21));
    let truth = (1.0 / 12.0) / ((1.0 - phi) * (1.0 - phi));

    let raw = lrv(&z, Kernel::Bartlett, 4.0).unwrap();
    let pw = lrv_prewhitened_ar1(&z, Kernel::Bartlett, Some(4.0)).unwrap();

    assert!(pw.value.is_finite() && pw.value > 0.0);
    assert!(
        (pw.rho - phi).abs() < 0.1,
        "prewhitening rho {} should be near {phi}",
        pw.rho
    );
    let err_raw = (raw / truth - 1.0).abs();
    let err_pw = (pw.value / truth - 1.0).abs();
    assert!(
        err_pw < err_raw,
        "prewhitening should cut the small-bandwidth bias \
         (raw rel err {err_raw:.3}, prewhitened {err_pw:.3})"
    );
    assert!(
        err_pw < 0.35,
        "prewhitened estimate {} too far from truth {truth}",
        pw.value
    );
}

#[test]
fn prewhitened_lrv_auto_bandwidth_is_reported_and_valid() {
    let z = demean(&ar1_series(600, 0.6, 33));
    let pw = lrv_prewhitened_ar1(&z, Kernel::QuadraticSpectral, None).unwrap();
    assert!(pw.value.is_finite() && pw.value > 0.0);
    assert!(pw.bandwidth.is_finite() && pw.bandwidth >= 0.0);
    assert!(pw.rho.abs() <= 0.97, "rho must be capped at 0.97");
}

// ------------------------------------------------------------ ols + hac se

fn regression_data(n: usize, seed: u64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let x1 = ar1_series(n, 0.6, seed);
    let x2 = ar1_series(n, 0.4, seed ^ 0xdead_beef);
    let u = ar1_series(n, 0.5, seed ^ 0x1234_5678);
    let y: Vec<f64> = (0..n)
        .map(|t| 1.0 + 0.5 * x1[t] - 0.3 * x2[t] + u[t])
        .collect();
    (y, vec![vec![1.0; n], x1, x2])
}

#[test]
fn hac_bse_are_finite_and_nonnegative_for_psd_kernels() {
    let (y, x) = regression_data(300, 8);
    let fit = ols(&y, &x).unwrap();
    for kernel in [Kernel::Bartlett, Kernel::Parzen, Kernel::QuadraticSpectral] {
        for bandwidth in [0.0, 4.0, 12.0] {
            let inf = fit
                .inference(SeType::Hac {
                    kernel,
                    bandwidth,
                    use_correction: true,
                })
                .unwrap();
            for (i, &se) in inf.bse.iter().enumerate() {
                assert!(
                    se.is_finite() && se >= 0.0,
                    "{kernel:?} bw {bandwidth}: bse[{i}] = {se}"
                );
            }
        }
    }
}

#[test]
fn use_correction_inflates_bse_by_exactly_sqrt_n_over_n_minus_k() {
    let (y, x) = regression_data(250, 4);
    let fit = ols(&y, &x).unwrap();
    let factor = (250.0_f64 / (250.0 - 3.0)).sqrt();
    for bandwidth in [4.0, 8.0] {
        let with = fit
            .inference(SeType::Hac {
                kernel: Kernel::Bartlett,
                bandwidth,
                use_correction: true,
            })
            .unwrap();
        let without = fit
            .inference(SeType::Hac {
                kernel: Kernel::Bartlett,
                bandwidth,
                use_correction: false,
            })
            .unwrap();
        for i in 0..3 {
            assert!(
                (with.bse[i] / without.bse[i] - factor).abs() < 1e-12,
                "correction factor mismatch at param {i}"
            );
        }
    }
}

#[test]
fn hc1_is_hc0_scaled_and_hac_bw0_matches_hc0() {
    let (y, x) = regression_data(220, 15);
    let n = 220.0_f64;
    let fit = ols(&y, &x).unwrap();
    let hc0 = fit.inference(SeType::Hc0).unwrap();
    let hc1 = fit.inference(SeType::Hc1).unwrap();
    let factor = (n / (n - 3.0)).sqrt();
    for i in 0..3 {
        assert!((hc1.bse[i] / hc0.bse[i] - factor).abs() < 1e-12);
    }
    // Bartlett HAC at bandwidth 0 keeps only Gamma_0 = the HC0 meat.
    let hac0 = fit
        .inference(SeType::Hac {
            kernel: Kernel::Bartlett,
            bandwidth: 0.0,
            use_correction: false,
        })
        .unwrap();
    for i in 0..3 {
        assert!(
            ((hac0.bse[i] - hc0.bse[i]) / hc0.bse[i]).abs() < 1e-12,
            "HAC(bw=0) must equal HC0"
        );
    }
}

#[test]
fn tvalues_are_params_over_bse() {
    let (y, x) = regression_data(180, 23);
    let fit = ols(&y, &x).unwrap();
    let inf = fit
        .inference(SeType::Hac {
            kernel: Kernel::Bartlett,
            bandwidth: 6.0,
            use_correction: true,
        })
        .unwrap();
    for i in 0..3 {
        assert!((inf.tvalues[i] - fit.params[i] / inf.bse[i]).abs() < 1e-15);
    }
}

#[test]
fn ols_recovers_exact_coefficients_on_noiseless_data() {
    let n = 50;
    let x1 = ar1_series(n, 0.5, 2);
    let x2 = lcg_series(n, 77);
    let y: Vec<f64> = (0..n).map(|t| 2.0 - 1.5 * x1[t] + 0.25 * x2[t]).collect();
    let fit = ols(&y, &[vec![1.0; n], x1, x2]).unwrap();
    for (b, want) in fit.params.iter().zip([2.0, -1.5, 0.25]) {
        assert!((b - want).abs() < 1e-10, "got {b}, want {want}");
    }
}

// ------------------------------------------------------------- error paths

#[test]
fn error_paths_teach() {
    let z = demean(&ar1_series(50, 0.5, 1));

    assert!(matches!(
        lrv(&[1.0], Kernel::Bartlett, 4.0),
        Err(HacError::SeriesTooShort { .. })
    ));
    assert!(matches!(
        lrv(&[1.0, f64::NAN, 0.5], Kernel::Bartlett, 4.0),
        Err(HacError::NonFinite { index: 1, .. })
    ));
    assert!(matches!(
        lrv(&z, Kernel::Bartlett, -1.0),
        Err(HacError::InvalidBandwidth { .. })
    ));
    assert!(matches!(
        lrv(&z, Kernel::Bartlett, f64::NAN),
        Err(HacError::InvalidBandwidth { .. })
    ));

    assert!(matches!(ewc_lrv(&z, 0), Err(HacError::InvalidDof { .. })));
    assert!(matches!(
        ewc_lrv(&z, z.len()),
        Err(HacError::InvalidDof { .. })
    ));

    assert!(matches!(
        newey_west_bandwidth(&z, Kernel::Truncated),
        Err(HacError::UnsupportedKernel { .. })
    ));
    assert!(matches!(
        andrews_bandwidth_ar1(&vec![0.0; 40], Kernel::Bartlett),
        Err(HacError::ConstantSeries { .. })
    ));

    // OLS: empty design, ragged column, too few observations, collinear.
    assert!(matches!(ols(&z, &[]), Err(HacError::EmptyDesign)));
    assert!(matches!(
        ols(&z, &[vec![1.0; 3]]),
        Err(HacError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        ols(&[1.0, 2.0], &[vec![1.0; 2], vec![0.0, 1.0]]),
        Err(HacError::DegreesOfFreedom { n: 2, k: 2 })
    ));
    let dup = ar1_series(60, 0.3, 5);
    let y60 = ar1_series(60, 0.4, 6);
    assert!(matches!(
        ols(&y60, &[vec![1.0; 60], dup.clone(), dup]),
        Err(HacError::SingularDesign { .. })
    ));

    // Errors display a teaching message, not a bare code.
    let msg = lrv(&z, Kernel::Bartlett, -1.0).unwrap_err().to_string();
    assert!(msg.contains("maxlags"), "unexpected message: {msg}");
}
