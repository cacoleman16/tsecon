//! Property tests: exact reconstruction, trend-reproduction on
//! polynomial inputs, band-pass annihilation of constants and linear
//! trends, one-sided/expanding-window equivalence, error paths.

use tsecon_filters::{
    bk_filter, cf_filter, hamilton_defaults, hamilton_filter, hamilton_filter_random_walk,
    hp_filter, hp_filter_one_sided, ravn_uhlig_lambda, FiltersError, Frequency,
};

/// Deterministic test series: trend + cycle + bounded pseudo-noise
/// (no RNG dependency; a fixed LCG keeps the test reproducible).
fn test_series(n: usize) -> Vec<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|t| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            100.0 + 0.7 * t as f64 + 3.0 * (t as f64 * 0.35).sin() + u
        })
        .collect()
}

#[test]
fn hp_cycle_plus_trend_reconstructs_input_exactly() {
    // cycle is computed as y - trend with trend within a factor of two of
    // y here, so the subtraction is exact (Sterbenz) and adding the trend
    // back reproduces y bit-for-bit.
    let y = test_series(120);
    let dec = hp_filter(&y, 1600.0).expect("hp_filter succeeds");
    let trend = dec.trend.expect("hp has a trend");
    for i in 0..y.len() {
        assert_eq!(
            dec.cycle[i] + trend[i],
            y[i],
            "hp reconstruction at {i}: cycle {} + trend {}",
            dec.cycle[i],
            trend[i]
        );
    }
}

#[test]
fn hp_trend_of_linear_series_is_the_series_for_any_lambda() {
    // A linear series has zero second differences, so K y = 0 and
    // (I + lambda K'K) y = y: the trend equals the series regardless of
    // lambda, and the cycle is zero.
    let n = 80;
    let y: Vec<f64> = (0..n).map(|t| 3.25 - 0.6 * t as f64).collect();
    for lambda in [
        0.0,
        ravn_uhlig_lambda(Frequency::Annual),
        ravn_uhlig_lambda(Frequency::Quarterly),
        ravn_uhlig_lambda(Frequency::Monthly),
    ] {
        let dec = hp_filter(&y, lambda).expect("hp_filter succeeds");
        let trend = dec.trend.expect("hp has a trend");
        for i in 0..n {
            assert!(
                (trend[i] - y[i]).abs() <= 1e-9 * y[i].abs().max(1.0),
                "lambda {lambda}, index {i}: trend {} vs y {}",
                trend[i],
                y[i]
            );
            assert!(dec.cycle[i].abs() <= 1e-9 * y[i].abs().max(1.0));
        }
    }
}

#[test]
fn hp_lambda_zero_returns_the_series_as_trend() {
    let y = test_series(30);
    let dec = hp_filter(&y, 0.0).expect("hp_filter succeeds");
    let trend = dec.trend.expect("hp has a trend");
    for i in 0..y.len() {
        assert!((trend[i] - y[i]).abs() <= 1e-12 * y[i].abs());
        assert!(dec.cycle[i].abs() <= 1e-12 * y[i].abs());
    }
}

#[test]
fn hp_short_series_trend_equals_series() {
    // n = 1, 2: no second differences exist, penalty is empty, trend = y.
    for n in [1usize, 2] {
        let y = test_series(n);
        let dec = hp_filter(&y, 1600.0).expect("hp_filter succeeds");
        let trend = dec.trend.expect("hp has a trend");
        assert_eq!(trend, y, "n = {n}");
    }
}

#[test]
fn one_sided_hp_matches_last_point_of_expanding_full_sample_hp() {
    let y = test_series(60);
    let lambda = 1600.0;
    let one_sided = hp_filter_one_sided(&y, lambda).expect("one-sided hp succeeds");
    let trend_1s = one_sided.trend.expect("hp has a trend");
    assert_eq!(one_sided.alignment.lost_start, 0);
    assert_eq!(trend_1s.len(), y.len());
    for t in 0..y.len() {
        let full = hp_filter(&y[..=t], lambda).expect("hp_filter succeeds");
        let expected = *full
            .trend
            .expect("hp has a trend")
            .last()
            .expect("nonempty trend");
        assert_eq!(
            trend_1s[t], expected,
            "one-sided trend at {t} differs from expanding-window HP"
        );
    }
}

#[test]
fn bk_cycle_annihilates_constant_and_linear_trend() {
    let n = 90;
    // Demeaned weights sum to zero (kills constants); symmetry makes the
    // first moment sum_j j*b_j vanish (kills linear trends).
    let constant: Vec<f64> = vec![7.5; n];
    let linear: Vec<f64> = (0..n).map(|t| -2.0 + 0.45 * t as f64).collect();
    for (name, series) in [("constant", &constant), ("linear", &linear)] {
        let dec = bk_filter(series, 6.0, 32.0, 12).expect("bk_filter succeeds");
        for (i, c) in dec.cycle.iter().enumerate() {
            assert!(
                c.abs() <= 1e-10,
                "bk {name} series not annihilated at {i}: {c}"
            );
        }
    }
}

#[test]
fn cf_cycle_annihilates_constant_and_linear_trend() {
    let n = 90;
    // Every row of the CF filter sums to zero (endpoint weights are
    // built that way), killing constants; with drift = true a linear
    // trend is removed exactly by the endpoint drift adjustment.
    let constant: Vec<f64> = vec![7.5; n];
    let linear: Vec<f64> = (0..n).map(|t| -2.0 + 0.45 * t as f64).collect();
    for (name, series, drift) in [
        ("constant, no drift", &constant, false),
        ("constant, drift", &constant, true),
        ("linear, drift", &linear, true),
    ] {
        let dec = cf_filter(series, 6.0, 32.0, drift).expect("cf_filter succeeds");
        for (i, c) in dec.cycle.iter().enumerate() {
            assert!(
                c.abs() <= 1e-10,
                "cf {name} series not annihilated at {i}: {c}"
            );
        }
    }
}

#[test]
fn cf_without_drift_reconstructs_input() {
    let y = test_series(70);
    let dec = cf_filter(&y, 6.0, 32.0, false).expect("cf_filter succeeds");
    let trend = dec.trend.expect("cf has a trend");
    for i in 0..y.len() {
        assert!(
            (trend[i] + dec.cycle[i] - y[i]).abs() <= 1e-9,
            "cf reconstruction at {i}"
        );
    }
}

#[test]
fn hamilton_cycle_has_mean_zero() {
    // OLS with an intercept: residuals sum to zero by the normal
    // equations.
    let y = test_series(150);
    let res = hamilton_filter(&y, 8, 4).expect("hamilton_filter succeeds");
    let cycle = &res.decomposition.cycle;
    let mean = cycle.iter().sum::<f64>() / cycle.len() as f64;
    assert!(mean.abs() <= 1e-9, "hamilton cycle mean {mean}");
}

#[test]
fn hamilton_trend_plus_cycle_reconstructs_aligned_input() {
    let y = test_series(100);
    let res = hamilton_filter(&y, 8, 4).expect("hamilton_filter succeeds");
    let dec = &res.decomposition;
    let trend = dec.trend.as_ref().expect("hamilton has a trend");
    assert_eq!(dec.alignment.lost_start, 11);
    for (i, (&tr, &cy)) in trend.iter().zip(&dec.cycle).enumerate() {
        let t = dec.alignment.input_index(i).expect("in range");
        assert!(
            (tr + cy - y[t]).abs() <= 1e-9,
            "hamilton reconstruction at output {i} / input {t}"
        );
    }
}

#[test]
fn hamilton_random_walk_is_h_period_difference() {
    let y = test_series(50);
    let h = 8;
    let dec = hamilton_filter_random_walk(&y, h).expect("rw filter succeeds");
    assert_eq!(dec.alignment.lost_start, h);
    assert_eq!(dec.alignment.lost_end, 0);
    assert_eq!(dec.cycle.len(), y.len() - h);
    let trend = dec.trend.expect("rw has a trend");
    for i in 0..dec.cycle.len() {
        assert_eq!(dec.cycle[i], y[i + h] - y[i]);
        assert_eq!(trend[i], y[i]);
    }
}

#[test]
fn ravn_uhlig_rule_values() {
    assert_eq!(ravn_uhlig_lambda(Frequency::Quarterly), 1600.0);
    assert_eq!(ravn_uhlig_lambda(Frequency::Annual), 6.25);
    assert_eq!(ravn_uhlig_lambda(Frequency::Monthly), 129600.0);
    assert_eq!(hamilton_defaults(Frequency::Quarterly), (8, 4));
    assert_eq!(hamilton_defaults(Frequency::Annual), (2, 1));
    assert_eq!(hamilton_defaults(Frequency::Monthly), (24, 12));
}

#[test]
fn error_paths() {
    let y = test_series(40);

    // Invalid parameters.
    assert!(matches!(
        hp_filter(&y, -1.0),
        Err(FiltersError::InvalidParameter { name: "lambda", .. })
    ));
    assert!(matches!(
        hp_filter(&y, f64::NAN),
        Err(FiltersError::InvalidParameter { name: "lambda", .. })
    ));
    assert!(matches!(
        bk_filter(&y, 1.0, 32.0, 12),
        Err(FiltersError::InvalidParameter { name: "low", .. })
    ));
    assert!(matches!(
        cf_filter(&y, 6.0, 6.0, true),
        Err(FiltersError::InvalidParameter { name: "high", .. })
    ));
    assert!(matches!(
        bk_filter(&y, 6.0, 32.0, 0),
        Err(FiltersError::InvalidParameter { name: "k", .. })
    ));
    assert!(matches!(
        hamilton_filter(&y, 0, 4),
        Err(FiltersError::InvalidParameter { name: "h", .. })
    ));
    assert!(matches!(
        hamilton_filter(&y, 8, 0),
        Err(FiltersError::InvalidParameter { name: "p", .. })
    ));
    assert!(matches!(
        hamilton_filter_random_walk(&y, 0),
        Err(FiltersError::InvalidParameter { name: "h", .. })
    ));

    // Too-short series.
    assert!(matches!(
        hp_filter(&[], 1600.0),
        Err(FiltersError::SeriesTooShort { .. })
    ));
    assert!(matches!(
        bk_filter(&y[..24], 6.0, 32.0, 12),
        Err(FiltersError::SeriesTooShort {
            needed: 25,
            got: 24,
            ..
        })
    ));
    assert!(matches!(
        cf_filter(&y[..2], 6.0, 32.0, true),
        Err(FiltersError::SeriesTooShort { .. })
    ));
    // Hamilton h=8, p=4 needs 11 lost + 5 rows = 16.
    assert!(matches!(
        hamilton_filter(&y[..15], 8, 4),
        Err(FiltersError::SeriesTooShort {
            needed: 16,
            got: 15,
            ..
        })
    ));
    assert!(matches!(
        hamilton_filter_random_walk(&y[..8], 8),
        Err(FiltersError::SeriesTooShort { .. })
    ));

    // Non-finite input.
    let mut bad = y.clone();
    bad[7] = f64::NAN;
    assert!(matches!(
        hp_filter(&bad, 1600.0),
        Err(FiltersError::NonFiniteInput { index: 7 })
    ));
    assert!(matches!(
        cf_filter(&bad, 6.0, 32.0, true),
        Err(FiltersError::NonFiniteInput { index: 7 })
    ));

    // Constant series: Hamilton lag columns collinear with the
    // intercept.
    let flat = vec![3.0; 40];
    assert!(matches!(
        hamilton_filter(&flat, 8, 4),
        Err(FiltersError::RankDeficient { .. })
    ));
}
