//! Input-validation tests ("errors that teach"): every guard on the
//! smooth-LP entry point (and the horizon pre-scan every LP entry point
//! shares) returns a typed [`LpError`] whose Display names what happened
//! and what to try, never a panic or an abort.

use tsecon_lp::{
    lp, lp_band, lp_irf_cov, lp_iv, lp_multiplier, lp_state, smooth_lp, BandSpec, LpError, LpSpec,
    SmoothLpSpec,
};

/// A small well-formed pair of series.
fn series(n: usize) -> (Vec<f64>, Vec<f64>) {
    let e: Vec<f64> = (0..n)
        .map(|t| ((t * 37 + 11) % 17) as f64 / 8.5 - 1.0)
        .collect();
    let mut y = vec![0.0; n];
    for t in 1..n {
        y[t] = 0.6 * y[t - 1] + e[t] + 0.1 * ((t % 7) as f64 - 3.0);
    }
    (y, e)
}

#[test]
fn length_mismatch_rejected() {
    let (y, mut e) = series(80);
    e.pop();
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(6, 2)).unwrap_err();
    assert!(matches!(err, LpError::LengthMismatch { .. }));
    assert!(err.to_string().contains("index-aligned"));
}

#[test]
fn non_finite_rejected() {
    let (mut y, e) = series(80);
    y[10] = f64::NAN;
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(6, 2)).unwrap_err();
    assert!(matches!(err, LpError::NonFinite { .. }));
    assert!(
        err.to_string().contains("drop or"),
        "teaches the fix: {err}"
    );

    let (y2, mut e2) = series(80);
    e2[3] = f64::INFINITY;
    let err2 = smooth_lp(&y2, &e2, &SmoothLpSpec::new(6, 2)).unwrap_err();
    assert!(matches!(err2, LpError::NonFinite { .. }));
}

#[test]
fn short_series_and_long_horizon_rejected() {
    let (y, e) = series(4);
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(2, 6)).unwrap_err();
    assert!(matches!(err, LpError::SeriesTooShort { .. }));

    let (y, e) = series(20);
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(18, 2)).unwrap_err();
    assert!(matches!(err, LpError::HorizonTooLong { .. }));
    assert!(
        err.to_string().contains("shorten the maximum horizon"),
        "teaches the fix: {err}"
    );
}

#[test]
fn spline_config_violations_rejected_with_the_constraint_named() {
    let (y, e) = series(120);

    // degree 0 is refused.
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_degree(0)).unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
    assert!(err.to_string().contains("degree >= 1"), "{err}");

    // More basis functions than horizon points.
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_n_basis(12)).unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
    assert!(err.to_string().contains("n_basis <= horizons + 1"), "{err}");

    // Too few basis functions for the degree.
    let err = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2).with_degree(3).with_n_basis(3),
    )
    .unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
    assert!(err.to_string().contains("n_basis >= degree + 1"), "{err}");

    // Penalty order out of range.
    let err = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2)
            .with_n_basis(4)
            .with_penalty_order(4),
    )
    .unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
    assert!(err.to_string().contains("penalty_order < n_basis"), "{err}");

    // Horizons too short for the default cubic interpolating basis:
    // H = 2 gives K = 3 < degree + 1 = 4.
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(2, 2)).unwrap_err();
    assert!(matches!(err, LpError::SplineConfig { .. }));
}

#[test]
fn bad_lambda_rejected() {
    let (y, e) = series(120);
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_lambda(bad)).unwrap_err();
        assert!(matches!(err, LpError::InvalidLambda { .. }), "lambda {bad}");
        assert!(err.to_string().contains("cross-validation"), "{err}");
    }

    // A bad value hiding inside a CV grid is caught too.
    let err = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2).with_cv(Some(vec![1.0, -3.0]), 4),
    )
    .unwrap_err();
    assert!(matches!(err, LpError::InvalidLambda { .. }));
}

#[test]
fn empty_grid_rejected() {
    let (y, e) = series(120);
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_cv(Some(vec![]), 4)).unwrap_err();
    assert!(matches!(err, LpError::EmptyLambdaGrid));
    assert!(err.to_string().contains("default"), "{err}");
}

#[test]
fn infeasible_cv_folds_rejected() {
    let (y, e) = series(120);

    // One fold is not cross-validation.
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(8, 2).with_cv(Some(vec![1.0]), 1)).unwrap_err();
    assert!(matches!(err, LpError::CvConfig { .. }));
    assert!(err.to_string().contains("n_folds"), "{err}");

    // More folds than usable base periods.
    let err = smooth_lp(
        &y,
        &e,
        &SmoothLpSpec::new(8, 2).with_cv(Some(vec![1.0]), 500),
    )
    .unwrap_err();
    assert!(matches!(err, LpError::CvConfig { .. }));
}

#[test]
fn constant_shock_is_a_singular_design_that_teaches() {
    // A constant shock makes the spline block collinear with the intercepts.
    let (y, _) = series(120);
    let e = vec![1.0; 120];
    let err = smooth_lp(&y, &e, &SmoothLpSpec::new(6, 2).with_lambda(1.0)).unwrap_err();
    assert!(matches!(err, LpError::Hac(_)), "got {err:?}");
    assert!(err.to_string().contains("collinear"), "{err}");
}

/// A maximum horizon far beyond the sample (`2^40`, `usize::MAX`) is refused
/// by the arithmetic pre-scan before any output is sized by it — it used to
/// reserve `horizons + 1` slots first and abort the process — at the first
/// infeasible horizon, and the message names `horizons` with the largest
/// value the sample admits. Hand-computed for `n = 60`, `p = 2`:
/// lag-augmented `lp` keeps `60 - h - max(2, h)` rows for `4 + h`
/// regressors (first failure `h = 19`); HAC `lp` and `lp_iv` keep `58 - h`
/// rows for 4 (`h = 54`); `lp_multiplier` keeps `58 - h` for `2 + 2p = 6`
/// (`h = 52`); HAC `lp_state` keeps `58 - h` for `2 (2 + p) = 8` (`h = 50`).
#[test]
fn unreachable_maximum_horizons_are_refused_before_sizing() {
    // Aperiodic draws, so that up to 18 impulse lags stay linearly
    // independent at the largest admissible horizon.
    let scramble = |mut s: u64| -> Vec<f64> {
        (0..60)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    };
    let e = scramble(1);
    let mut y = scramble(2);
    for t in 1..60 {
        y[t] += 0.5 * y[t - 1] + 0.8 * e[t];
    }
    let z: Vec<f64> = e
        .iter()
        .zip(scramble(3))
        .map(|(v, u)| v + 0.3 * u)
        .collect();
    let state: Vec<f64> = (0..60).map(|t| ((t / 3) % 2) as f64).collect();
    for horizons in [1usize << 40, usize::MAX] {
        let la = LpSpec::new(horizons, 2);
        let hac = la.with_hac(None);
        let cases: [(&str, LpError, usize); 8] = [
            ("lp", lp(&y, &e, la).unwrap_err(), 19),
            ("lp hac", lp(&y, &e, hac).unwrap_err(), 54),
            ("lp_iv", lp_iv(&y, &e, &z, la).unwrap_err(), 54),
            (
                "lp_multiplier",
                lp_multiplier(&y, &e, &z, la).unwrap_err(),
                52,
            ),
            ("lp_state", lp_state(&y, &e, &state, hac).unwrap_err(), 50),
            ("lp_irf_cov", lp_irf_cov(&y, &e, la).unwrap_err(), 19),
            (
                "lp_band sup-t",
                lp_band(&y, &e, la, BandSpec::sup_t(0.1)).unwrap_err(),
                19,
            ),
            (
                "smooth_lp",
                smooth_lp(&y, &e, &SmoothLpSpec::new(horizons, 2)).unwrap_err(),
                54,
            ),
        ];
        for (name, err, first) in cases {
            assert!(
                matches!(err, LpError::HorizonTooLong { horizon, .. } if horizon == first),
                "{name}: {err:?}"
            );
            let msg = err.to_string();
            let max = format!("horizons can be at most {}", first - 1);
            assert!(msg.contains(&max), "{name}: {msg}");
        }
    }
    // The largest admissible value really is admissible.
    assert_eq!(lp(&y, &e, LpSpec::new(18, 2)).unwrap().irf.len(), 19);
    assert_eq!(lp_iv(&y, &e, &z, LpSpec::new(53, 2)).unwrap().irf.len(), 54);
    assert_eq!(
        lp_multiplier(&y, &e, &z, LpSpec::new(51, 2))
            .unwrap()
            .multiplier
            .len(),
        52
    );
}

/// When even horizon 0 has too few rows the refusal says no horizon is
/// estimable and points at `n_lag_controls`, not at the horizon.
#[test]
fn infeasible_horizon_zero_points_at_the_lag_controls() {
    let (y, e) = series(10);
    // p = 4: horizon 0 keeps 10 - 4 = 6 rows for 2 + 4 = 6 regressors.
    let err = lp(&y, &e, LpSpec::new(3, 4).with_hac(None)).unwrap_err();
    assert!(
        matches!(err, LpError::HorizonTooLong { horizon: 0, .. }),
        "{err:?}"
    );
    let msg = err.to_string();
    assert!(msg.contains("not even horizons = 0"), "{msg}");
    assert!(msg.contains("reduce n_lag_controls"), "{msg}");
}
