//! Property / contract tests for the DF-GLS test that do not need an
//! external golden: invariances the GLS-detrending step guarantees,
//! statistical sanity on synthetic series, and the error surface.

use tsecon_diag::{dfgls, AdfLagSelection, DfglsTrend, DiagError};

/// A tiny deterministic LCG (Knuth MMIX constants) so the tests are stable
/// across platforms without a system RNG. The chaotic-map helper used by
/// the sibling test files converges to a fixed point — an exact linear
/// trend — which the GLS-detrending step correctly treats as degenerate,
/// so DF-GLS needs genuinely noisy pseudo-randomness.
struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately standard-normal increment (Irwin-Hall, 12 uniforms).
    fn normalish(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// A deterministic, reproducible random walk.
fn walk(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    let mut acc = 0.0;
    (0..n)
        .map(|_| {
            acc += rng.normalish();
            acc
        })
        .collect()
}

/// A deterministic stationary pseudo-noise series around zero.
fn stationary(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg(seed);
    (0..n).map(|_| rng.normalish()).collect()
}

const AIC: AdfLagSelection = AdfLagSelection::Aic(None);

#[test]
fn statistic_is_scale_invariant() {
    // The tau statistic is a t-ratio and GLS detrending is linear, so
    // rescaling y must not change it (nor the selected lag).
    let y = walk(150, 21);
    for trend in [DfglsTrend::Constant, DfglsTrend::ConstantTrend] {
        let base = dfgls(&y, trend, AIC).unwrap();
        for scale in [1e-6, 3.0, 1e6] {
            let ys: Vec<f64> = y.iter().map(|&v| v * scale).collect();
            let scaled = dfgls(&ys, trend, AIC).unwrap();
            assert_eq!(scaled.used_lag, base.used_lag, "lag under scale {scale}");
            let rel = ((scaled.statistic - base.statistic) / base.statistic).abs();
            assert!(
                rel < 1e-9,
                "scale {scale}: stat {} vs {} (rel {rel:e})",
                scaled.statistic,
                base.statistic
            );
        }
    }
}

#[test]
fn statistic_is_invariant_to_the_deterministics_it_removes() {
    // Adding a constant (both cases) or a linear trend ("ct" case) is
    // absorbed exactly by the detrending regression.
    let y = walk(150, 37);
    for trend in [DfglsTrend::Constant, DfglsTrend::ConstantTrend] {
        let base = dfgls(&y, trend, AIC).unwrap();
        let shifted: Vec<f64> = y.iter().map(|&v| v + 250.0).collect();
        let s = dfgls(&shifted, trend, AIC).unwrap();
        assert_eq!(s.used_lag, base.used_lag);
        let rel = ((s.statistic - base.statistic) / base.statistic).abs();
        assert!(rel < 1e-8, "shift: rel {rel:e}");
    }
    // Linear trend, "ct" only.
    let base = dfgls(&y, DfglsTrend::ConstantTrend, AIC).unwrap();
    let trended: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(t, &v)| v + 5.0 + 0.4 * t as f64)
        .collect();
    let s = dfgls(&trended, DfglsTrend::ConstantTrend, AIC).unwrap();
    assert_eq!(s.used_lag, base.used_lag);
    let rel = ((s.statistic - base.statistic) / base.statistic).abs();
    assert!(rel < 1e-8, "trend: rel {rel:e}");
}

#[test]
fn rejects_on_stationary_noise_constant_case() {
    let y = stationary(250, 61);
    let res = dfgls(&y, DfglsTrend::Constant, AIC).unwrap();
    assert!(
        res.p_value < 0.01,
        "stationary noise should reject: p = {}",
        res.p_value
    );
    assert!(res.statistic < res.crit.pct1);
}

#[test]
fn does_not_reject_on_a_random_walk() {
    let y = walk(200, 11);
    for trend in [DfglsTrend::Constant, DfglsTrend::ConstantTrend] {
        let res = dfgls(&y, trend, AIC).unwrap();
        assert!(
            res.p_value > 0.10,
            "random walk should not reject ({:?}): p = {}",
            trend,
            res.p_value
        );
    }
}

#[test]
fn fixed_lag_at_the_selected_lag_reproduces_the_auto_statistic() {
    let y = walk(180, 47);
    for trend in [DfglsTrend::Constant, DfglsTrend::ConstantTrend] {
        let auto = dfgls(&y, trend, AIC).unwrap();
        let fixed = dfgls(&y, trend, AdfLagSelection::Fixed(auto.used_lag)).unwrap();
        assert_eq!(fixed.used_lag, auto.used_lag);
        assert_eq!(fixed.nobs, auto.nobs);
        assert_eq!(fixed.statistic, auto.statistic);
        assert_eq!(fixed.p_value, auto.p_value);
    }
}

#[test]
fn default_maxlag_follows_schwert_capped_at_arch_bound() {
    // With n = 200 the Schwert rule gives ceil(12 * 2^0.25) = 15, well
    // under the bound (n-1)/2 - 1 = 98; a Fixed run at every lag 0..=15
    // must contain the auto-selected one (i.e. the search space is right).
    let y = walk(200, 53);
    let auto = dfgls(&y, DfglsTrend::Constant, AIC).unwrap();
    assert!(auto.used_lag <= 15, "selected lag {} > 15", auto.used_lag);
    // Explicitly widening max_lags to the same value changes nothing.
    let explicit = dfgls(&y, DfglsTrend::Constant, AdfLagSelection::Aic(Some(15))).unwrap();
    assert_eq!(explicit.used_lag, auto.used_lag);
    assert_eq!(explicit.statistic, auto.statistic);
}

#[test]
fn nobs_is_n_minus_one_minus_lag() {
    let y = walk(120, 71);
    let res = dfgls(&y, DfglsTrend::Constant, AdfLagSelection::Fixed(4)).unwrap();
    assert_eq!(res.used_lag, 4);
    assert_eq!(res.nobs, 120 - 1 - 4);
}

#[test]
fn error_surface() {
    // Empty.
    assert!(matches!(
        dfgls(&[], DfglsTrend::Constant, AIC),
        Err(DiagError::SeriesTooShort { .. })
    ));
    // Constant series.
    let c = vec![3.0; 50];
    assert!(matches!(
        dfgls(&c, DfglsTrend::Constant, AIC),
        Err(DiagError::ConstantSeries { .. })
    ));
    // Non-finite.
    let mut bad = walk(50, 5);
    bad[10] = f64::NAN;
    assert!(matches!(
        dfgls(&bad, DfglsTrend::Constant, AIC),
        Err(DiagError::NonFinite { .. })
    ));
    // Too short for the trend spec (arch minimum 3 + ntrend).
    let short = walk(4, 5);
    assert!(matches!(
        dfgls(&short, DfglsTrend::ConstantTrend, AIC),
        Err(DiagError::SeriesTooShort { .. })
    ));
    // Fixed lag too large for the sample.
    let y = walk(20, 5);
    assert!(matches!(
        dfgls(&y, DfglsTrend::Constant, AdfLagSelection::Fixed(10)),
        Err(DiagError::SeriesTooShort { .. })
    ));
    // An exact linear trend: the GLS detrending fit is exact and the
    // regression degenerates — a teaching error, not a garbage number.
    let det: Vec<f64> = (0..60).map(|t| 2.0 + 0.5 * t as f64).collect();
    assert!(dfgls(&det, DfglsTrend::ConstantTrend, AIC).is_err());
}
