//! Property and invariant tests for the unit-root layer, plus error-path
//! and report coverage.

use tsecon_diag::{
    adf, check_stationarity, check_stationarity_at, kpss, AdfLagSelection, AdfRegression,
    DiagError, KpssLags, KpssRegression, Recommendation, StationarityQuadrant,
};

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

/// Pure random walk: cumulative sum of the LCG innovations.
fn random_walk(n: usize, seed: u64) -> Vec<f64> {
    let mut cum = 0.0;
    lcg_series(n, seed)
        .into_iter()
        .map(|e| {
            cum += e;
            cum
        })
        .collect()
}

#[test]
fn adf_fails_and_kpss_rejects_on_a_random_walk() {
    for seed in [11_u64, 42] {
        let y = random_walk(300, seed);
        let a = adf(&y, AdfRegression::Constant, AdfLagSelection::Aic(None)).unwrap();
        assert!(
            a.p_value > 0.05,
            "ADF must not reject the unit root on a pure random walk \
             (seed {seed}, p = {})",
            a.p_value
        );
        let k = kpss(&y, KpssRegression::Constant, KpssLags::Auto).unwrap();
        assert!(
            k.p_value < 0.05,
            "KPSS must reject stationarity on a pure random walk \
             (seed {seed}, p = {})",
            k.p_value
        );
    }
}

#[test]
fn adf_rejects_and_kpss_fails_on_white_noise() {
    for seed in [11_u64, 42] {
        let y = lcg_series(300, seed);
        let a = adf(&y, AdfRegression::Constant, AdfLagSelection::Aic(None)).unwrap();
        assert!(
            a.p_value < 0.01,
            "ADF must reject the unit root on white noise \
             (seed {seed}, p = {})",
            a.p_value
        );
        let k = kpss(&y, KpssRegression::Constant, KpssLags::Auto).unwrap();
        assert!(
            k.p_value > 0.05,
            "KPSS must not reject stationarity on white noise \
             (seed {seed}, p = {})",
            k.p_value
        );
    }
}

#[test]
fn adf_statistic_is_scale_and_shift_invariant() {
    let y = random_walk(250, 42);
    let scaled: Vec<f64> = y.iter().map(|&v| 1000.0 * v).collect();
    let shifted: Vec<f64> = y.iter().map(|&v| v + 500.0).collect();
    for (regression, other, label) in [
        (AdfRegression::Constant, &scaled, "scaled"),
        (AdfRegression::Constant, &shifted, "shifted"),
        (AdfRegression::NoConstant, &scaled, "scaled"),
        (AdfRegression::ConstantTrend, &scaled, "scaled"),
    ] {
        let base = adf(&y, regression, AdfLagSelection::Aic(None)).unwrap();
        let tr = adf(other, regression, AdfLagSelection::Aic(None)).unwrap();
        assert_eq!(
            base.used_lag, tr.used_lag,
            "lag selection must be {label}-invariant ({regression:?})"
        );
        assert!(
            (base.statistic - tr.statistic).abs() < 1e-8,
            "ADF stat must be {label}-invariant ({regression:?}): \
             {} vs {}",
            base.statistic,
            tr.statistic
        );
    }
}

#[test]
fn kpss_statistic_is_scale_invariant() {
    let y = random_walk(200, 11);
    let scaled: Vec<f64> = y.iter().map(|&v| 250.0 * v).collect();
    for regression in [KpssRegression::Constant, KpssRegression::ConstantTrend] {
        let base = kpss(&y, regression, KpssLags::Auto).unwrap();
        let tr = kpss(&scaled, regression, KpssLags::Auto).unwrap();
        assert_eq!(base.lags, tr.lags, "auto bandwidth must be scale-invariant");
        assert!(
            (base.statistic - tr.statistic).abs() < 1e-8,
            "KPSS stat must be scale-invariant: {} vs {}",
            base.statistic,
            tr.statistic
        );
    }
}

#[test]
fn report_quadrants_are_consistent_with_the_underlying_tests() {
    for (series, expected_quadrant, expected_rec) in [
        (
            random_walk(300, 42),
            StationarityQuadrant::UnitRoot,
            Recommendation::Difference,
        ),
        (
            lcg_series(300, 42),
            StationarityQuadrant::Stationary,
            Recommendation::Proceed,
        ),
    ] {
        let rep = check_stationarity(&series).unwrap();
        assert_eq!(rep.alpha, 0.05);
        assert_eq!(rep.adf_rejects, rep.adf.p_value < rep.alpha);
        assert_eq!(rep.kpss_rejects, rep.kpss.p_value < rep.alpha);
        let quadrant = match (rep.adf_rejects, rep.kpss_rejects) {
            (true, false) => StationarityQuadrant::Stationary,
            (false, true) => StationarityQuadrant::UnitRoot,
            (true, true) => StationarityQuadrant::Conflict,
            (false, false) => StationarityQuadrant::Inconclusive,
        };
        assert_eq!(rep.quadrant, quadrant, "quadrant must match the decisions");
        assert_eq!(rep.quadrant, expected_quadrant);
        assert_eq!(rep.recommendation, expected_rec);
        assert!(!rep.interpretation.is_empty());
        let shown = format!("{rep}");
        assert!(shown.contains("ADF") && shown.contains("KPSS"));
    }
}

#[test]
fn conflict_and_inconclusive_map_to_the_documented_recommendations() {
    // A trend-stationary series: ADF(c) rejects nothing it shouldn't, but
    // KPSS(c) sees the trend as nonstationarity — the classic conflict
    // once the trend dominates. Build noise around a strong linear trend.
    let noise = lcg_series(300, 7);
    let trended: Vec<f64> = noise
        .iter()
        .enumerate()
        .map(|(t, &e)| 0.05 * t as f64 + e)
        .collect();
    let rep = check_stationarity(&trended).unwrap();
    // Whatever the exact quadrant on this seed, the mapping must hold.
    let expected = match rep.quadrant {
        StationarityQuadrant::Stationary => Recommendation::Proceed,
        StationarityQuadrant::UnitRoot | StationarityQuadrant::Inconclusive => {
            Recommendation::Difference
        }
        StationarityQuadrant::Conflict => Recommendation::Detrend,
    };
    assert_eq!(rep.recommendation, expected);
    // The KPSS side must flag the deterministic trend at level form.
    assert!(rep.kpss_rejects, "KPSS(c) must reject on a trending series");
}

#[test]
fn fixed_lag_and_fixed_bandwidth_paths_agree_with_their_rules() {
    let y = random_walk(200, 11);
    // Fixed ADF lag is echoed back and changes nobs accordingly.
    let res = adf(&y, AdfRegression::Constant, AdfLagSelection::Fixed(3)).unwrap();
    assert_eq!(res.used_lag, 3);
    assert_eq!(res.nobs, 200 - 1 - 3);
    // KPSS with an explicit bandwidth equal to the legacy rule matches the
    // legacy result exactly.
    let legacy = kpss(&y, KpssRegression::Constant, KpssLags::Legacy).unwrap();
    let fixed = kpss(
        &y,
        KpssRegression::Constant,
        KpssLags::Fixed(legacy.lags),
    )
    .unwrap();
    assert_eq!(legacy.statistic, fixed.statistic);
    assert_eq!(legacy.p_value, fixed.p_value);
}

// ---------------------------------------------------------------- errors

#[test]
fn constant_series_errors() {
    let y = vec![3.25; 60];
    assert!(matches!(
        adf(&y, AdfRegression::Constant, AdfLagSelection::Aic(None)),
        Err(DiagError::ConstantSeries { .. })
    ));
    assert!(matches!(
        kpss(&y, KpssRegression::Constant, KpssLags::Auto),
        Err(DiagError::ConstantSeries { .. })
    ));
    assert!(matches!(
        check_stationarity(&y),
        Err(DiagError::ConstantSeries { .. })
    ));
}

#[test]
fn non_finite_input_errors_with_position() {
    let mut y = random_walk(80, 11);
    y[7] = f64::NAN;
    match adf(&y, AdfRegression::Constant, AdfLagSelection::Aic(None)) {
        Err(DiagError::NonFinite { index, .. }) => assert_eq!(index, 7),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    y[7] = f64::INFINITY;
    assert!(matches!(
        kpss(&y, KpssRegression::ConstantTrend, KpssLags::Legacy),
        Err(DiagError::NonFinite { .. })
    ));
}

#[test]
fn lag_bounds_are_enforced() {
    let y = random_walk(40, 42);
    // statsmodels bound: maxlag <= n/2 - ntrend - 1 = 18 for "c", n = 40.
    assert!(adf(&y, AdfRegression::Constant, AdfLagSelection::Fixed(18)).is_ok());
    assert!(matches!(
        adf(&y, AdfRegression::Constant, AdfLagSelection::Fixed(19)),
        Err(DiagError::InvalidLags { .. })
    ));
    assert!(matches!(
        adf(&y, AdfRegression::Constant, AdfLagSelection::Aic(Some(19))),
        Err(DiagError::InvalidLags { .. })
    ));
    // KPSS: the Bartlett window must stay inside the sample.
    assert!(matches!(
        kpss(&y, KpssRegression::Constant, KpssLags::Fixed(40)),
        Err(DiagError::InvalidLags { .. })
    ));
    assert!(kpss(&y, KpssRegression::Constant, KpssLags::Fixed(39)).is_ok());
    // Far too short a series for any ADF regression.
    let short = [1.0, 2.0, 0.5];
    assert!(matches!(
        adf(&short, AdfRegression::Constant, AdfLagSelection::Aic(None)),
        Err(DiagError::SeriesTooShort { .. })
    ));
}

#[test]
fn check_stationarity_rejects_invalid_alpha() {
    let y = random_walk(100, 11);
    for bad in [0.0, 1.0, -0.05, 1.5, f64::NAN] {
        assert!(matches!(
            check_stationarity_at(&y, bad),
            Err(DiagError::InvalidAlpha { .. })
        ));
    }
}

#[test]
fn errors_display_teaching_messages() {
    let err = adf(
        &vec![1.0; 30],
        AdfRegression::Constant,
        AdfLagSelection::Aic(None),
    )
    .unwrap_err();
    assert!(err.to_string().contains("constant"));
    let err = kpss(
        &random_walk(50, 11),
        KpssRegression::Constant,
        KpssLags::Fixed(50),
    )
    .unwrap_err();
    assert!(err.to_string().contains("Bartlett"));
}
