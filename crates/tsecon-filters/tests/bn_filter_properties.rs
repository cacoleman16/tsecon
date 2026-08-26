//! Property and error-surface tests for the Kamber-Morley-Wong (2018)
//! BN filter and the Hamilton filter's inference surface.

use tsecon_filters::{bn_filter, hamilton_filter_with_se, BnDelta, FiltersError, HamiltonSe};

/// A random-walk-with-drift + AR(2)-cycle series, seeded (a tiny LCG so
/// the test is dependency-free and deterministic).
fn drifting_series(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut norm = || {
        // Sum of 12 uniforms, centered: near-Gaussian, plenty for a test DGP.
        let mut acc = 0.0f64;
        for _ in 0..12 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            acc += (state >> 11) as f64 / (1u64 << 53) as f64;
        }
        acc - 6.0
    };
    let mut level = 100.0f64;
    let mut c = [0.0f64; 2];
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        level += 0.4 + 0.5 * norm();
        let cnew = 1.3 * c[0] - 0.5 * c[1] + 0.6 * norm();
        c = [cnew, c[0]];
        y.push(level + cnew);
    }
    y
}

#[test]
fn trend_plus_cycle_reconstructs_input_exactly() {
    let y = drifting_series(160, 7);
    let r = bn_filter(&y, 4, BnDelta::Fixed(0.3), true).expect("bn_filter succeeds");
    let trend = r.decomposition.trend.as_ref().expect("bn has a trend");
    assert_eq!(r.decomposition.alignment.lost_start, 1);
    for (i, (t, c)) in trend.iter().zip(r.decomposition.cycle.iter()).enumerate() {
        // trend = y - cycle by construction; re-adding can re-round by
        // at most one ulp.
        assert!(
            (t + c - y[i + 1]).abs() <= 1e-15 * y[i + 1].abs(),
            "reconstruction at output {i}"
        );
    }
}

#[test]
fn ar_coefficients_sum_to_rho_exactly() {
    // The Dickey-Fuller reparameterization pins the sum of the AR
    // coefficients at rho = 1 - 1/sqrt(delta) by construction.
    let y = drifting_series(200, 11);
    for &delta in &[0.05, 0.25, 0.8] {
        let r = bn_filter(&y, 6, BnDelta::Fixed(delta), true).expect("bn_filter succeeds");
        let rho = 1.0 - 1.0 / delta.sqrt();
        let sum: f64 = r.ar.iter().sum();
        assert!(
            (sum - rho).abs() <= 1e-12 * rho.abs().max(1.0),
            "delta {delta}: sum(ar) {sum} vs rho {rho}"
        );
    }
}

#[test]
fn auto_delta_maximizes_amp_to_noise_locally() {
    // The returned delta is a local max of the amplitude-to-noise ratio
    // on the (d0, dt) grid: one step in either direction does not beat it.
    let y = drifting_series(240, 3);
    let auto = bn_filter(&y, 4, BnDelta::auto(), true).expect("auto succeeds");
    let at = |d: f64| {
        bn_filter(&y, 4, BnDelta::Fixed(d), true)
            .expect("fixed succeeds")
            .amplitude_to_noise
    };
    let here = auto.amplitude_to_noise;
    assert!(
        at(auto.delta + 0.0005) <= here,
        "right neighbor beats the max"
    );
    if auto.delta - 0.0005 > 0.0 {
        assert!(
            at(auto.delta - 0.0005) <= here,
            "left neighbor beats the max"
        );
    }
}

#[test]
fn cycle_se_is_positive_and_finite() {
    let y = drifting_series(200, 19);
    let r = bn_filter(&y, 4, BnDelta::auto(), true).expect("bn_filter succeeds");
    assert!(
        r.cycle_se.is_finite() && r.cycle_se > 0.0,
        "se {}",
        r.cycle_se
    );
    assert!(
        r.amplitude_to_noise.is_finite() && r.amplitude_to_noise > 0.0,
        "amp {}",
        r.amplitude_to_noise
    );
}

#[test]
fn no_demean_uses_zero_drift() {
    let y = drifting_series(150, 5);
    let r = bn_filter(&y, 3, BnDelta::Fixed(0.2), false).expect("bn_filter succeeds");
    assert_eq!(r.drift, 0.0);
    let rd = bn_filter(&y, 3, BnDelta::Fixed(0.2), true).expect("bn_filter succeeds");
    let dy_mean = y.windows(2).map(|w| w[1] - w[0]).sum::<f64>() / (y.len() - 1) as f64;
    assert!((rd.drift - dy_mean).abs() < 1e-14);
}

#[test]
fn rejects_p_below_two() {
    let y = drifting_series(100, 1);
    assert!(matches!(
        bn_filter(&y, 1, BnDelta::auto(), true),
        Err(FiltersError::InvalidParameter { name: "p", .. })
    ));
}

#[test]
fn rejects_short_series_with_the_bound_named() {
    let y = drifting_series(20, 1);
    match bn_filter(&y, 12, BnDelta::auto(), true) {
        Err(FiltersError::SeriesTooShort { needed, got, .. }) => {
            assert_eq!(needed, 27); // 2p + 3
            assert_eq!(got, 20);
        }
        other => panic!("expected SeriesTooShort, got {other:?}"),
    }
}

#[test]
fn rejects_non_finite_input() {
    let mut y = drifting_series(100, 1);
    y[40] = f64::NAN;
    assert!(matches!(
        bn_filter(&y, 4, BnDelta::auto(), true),
        Err(FiltersError::NonFiniteInput { index: 40 })
    ));
}

#[test]
fn rejects_invalid_delta_and_grid() {
    let y = drifting_series(100, 1);
    assert!(matches!(
        bn_filter(&y, 4, BnDelta::Fixed(0.0), true),
        Err(FiltersError::InvalidParameter { name: "delta", .. })
    ));
    assert!(matches!(
        bn_filter(&y, 4, BnDelta::Fixed(-1.0), true),
        Err(FiltersError::InvalidParameter { name: "delta", .. })
    ));
    assert!(matches!(
        bn_filter(
            &y,
            4,
            BnDelta::Auto {
                d0: 0.0,
                dt: 0.0005
            },
            true
        ),
        Err(FiltersError::InvalidParameter { name: "d0", .. })
    ));
    assert!(matches!(
        bn_filter(&y, 4, BnDelta::Auto { d0: 0.01, dt: 0.0 }, true),
        Err(FiltersError::InvalidParameter { name: "dt", .. })
    ));
}

#[test]
fn rejects_constant_series() {
    let y = vec![3.0; 60];
    assert!(matches!(
        bn_filter(&y, 4, BnDelta::Fixed(0.2), true),
        Err(FiltersError::RankDeficient { .. })
    ));
}

#[test]
fn hamilton_with_se_error_surface_matches_plain_filter() {
    // Same rejections as hamilton_filter itself.
    let y = drifting_series(100, 1);
    assert!(matches!(
        hamilton_filter_with_se(&y, 0, 4, HamiltonSe::NonRobust),
        Err(FiltersError::InvalidParameter { name: "h", .. })
    ));
    assert!(matches!(
        hamilton_filter_with_se(&y[..10], 8, 4, HamiltonSe::NonRobust),
        Err(FiltersError::SeriesTooShort { .. })
    ));
    let constant = vec![1.0; 60];
    assert!(matches!(
        hamilton_filter_with_se(
            &constant,
            8,
            4,
            HamiltonSe::Hac {
                maxlags: None,
                use_correction: true
            }
        ),
        Err(FiltersError::RankDeficient { .. })
    ));
}
