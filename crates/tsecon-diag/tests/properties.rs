//! Property and invariant tests beyond the statsmodels goldens, plus
//! error-path and report coverage.

use tsecon_diag::{acf, arch_lm, jarque_bera, ljung_box, pacf_ols, pacf_yw, DiagError};

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

#[test]
fn acf_lag_zero_is_one_and_se_zero() {
    for seed in [1_u64, 7, 42] {
        let y = ar1_series(150, 0.6, seed);
        for adjusted in [false, true] {
            let res = acf(&y, 25, adjusted).unwrap();
            assert_eq!(res.acf[0], 1.0, "acf[0] must be exactly 1");
            assert_eq!(res.bartlett_se[0], 0.0, "se[0] must be exactly 0");
            assert!(
                (res.bartlett_se[1] - (1.0 / 150.0_f64).sqrt()).abs() < 1e-15,
                "se[1] must be 1/sqrt(n)"
            );
        }
    }
}

#[test]
fn bartlett_se_is_nondecreasing() {
    let y = ar1_series(200, 0.7, 3);
    let res = acf(&y, 30, false).unwrap();
    for k in 1..res.bartlett_se.len() - 1 {
        assert!(
            res.bartlett_se[k + 1] >= res.bartlett_se[k] - 1e-15,
            "Bartlett se must be nondecreasing in the lag (k = {k})"
        );
    }
}

#[test]
fn acf_is_affine_invariant() {
    let y = ar1_series(120, 0.5, 11);
    let scaled: Vec<f64> = y.iter().map(|&v| 3.5 * v - 100.0).collect();
    let a = acf(&y, 15, false).unwrap().acf;
    let b = acf(&scaled, 15, false).unwrap().acf;
    for (i, (&x, &z)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - z).abs() < 1e-10,
            "acf must be invariant to affine transforms (lag {i}: {x} vs {z})"
        );
    }
}

#[test]
fn unadjusted_acf_is_bounded_by_one() {
    // The biased (n-denominator) estimator yields a PSD sequence, so
    // |r_k| <= 1.
    for seed in [2_u64, 9, 33] {
        let y = ar1_series(100, 0.8, seed);
        let res = acf(&y, 40, false).unwrap();
        for (k, &r) in res.acf.iter().enumerate() {
            assert!(r.abs() <= 1.0 + 1e-12, "|acf[{k}]| = {} > 1", r.abs());
        }
    }
}

#[test]
fn pacf_yw_is_bounded_by_one_and_starts_at_acf1() {
    for seed in [1_u64, 5, 21] {
        let y = ar1_series(160, 0.7, seed);
        let p = pacf_yw(&y, 30).unwrap();
        assert_eq!(p[0], 1.0);
        let r = acf(&y, 1, false).unwrap().acf;
        assert!(
            (p[1] - r[1]).abs() < 1e-12,
            "pacf[1] must equal acf[1] ({} vs {})",
            p[1],
            r[1]
        );
        for (k, &v) in p.iter().enumerate() {
            assert!(v.abs() <= 1.0 + 1e-12, "|pacf_yw[{k}]| = {} > 1", v.abs());
        }
    }
}

#[test]
fn pacf_estimators_agree_on_a_strong_ar1() {
    // Different small-sample estimators of the same population quantity:
    // they must agree loosely at lag 1 on a long AR(1) sample.
    let y = ar1_series(2000, 0.7, 13);
    let ywm = pacf_yw(&y, 5).unwrap();
    let ols = pacf_ols(&y, 5).unwrap();
    assert!(
        (ywm[1] - ols[1]).abs() < 0.05,
        "ywm {} vs ols {} at lag 1",
        ywm[1],
        ols[1]
    );
    assert!((ywm[1] - 0.7).abs() < 0.1, "pacf[1] should be near phi");
}

#[test]
fn ljung_box_stat_is_monotone_nondecreasing_and_dominates_box_pierce() {
    for seed in [4_u64, 17, 99] {
        let y = ar1_series(140, 0.4, seed);
        let res = ljung_box(&y, 20).unwrap();
        for i in 1..res.lb_stat.len() {
            assert!(
                res.lb_stat[i] >= res.lb_stat[i - 1],
                "Ljung-Box statistic must be nondecreasing in the lag"
            );
            assert!(
                res.bp_stat[i] >= res.bp_stat[i - 1],
                "Box-Pierce statistic must be nondecreasing in the lag"
            );
        }
        for i in 0..res.lb_stat.len() {
            assert!(
                res.lb_stat[i] >= res.bp_stat[i],
                "the Ljung-Box small-sample correction implies LB >= BP"
            );
            assert!(res.lb_pvalue[i] >= 0.0 && res.lb_pvalue[i] <= 1.0);
            assert!(res.bp_pvalue[i] >= 0.0 && res.bp_pvalue[i] <= 1.0);
        }
    }
}

#[test]
fn jarque_bera_skewness_is_zero_on_symmetric_data() {
    // Pairwise-interleaved (z, -z) cancels the mean and third moment
    // exactly in floating point.
    let z = lcg_series(64, 8);
    let mut x = Vec::with_capacity(128);
    for &v in &z {
        x.push(1.0 + v);
        x.push(1.0 - v);
    }
    let res = jarque_bera(&x).unwrap();
    assert!(
        res.skewness.abs() < 1e-12,
        "symmetric sample must have zero skewness, got {}",
        res.skewness
    );
    assert!(res.statistic >= 0.0);
    assert!(res.p_value >= 0.0 && res.p_value <= 1.0);
}

#[test]
fn jarque_bera_is_location_scale_invariant() {
    let x = lcg_series(200, 15);
    let y: Vec<f64> = x.iter().map(|&v| 250.0 - 12.0 * v).collect();
    let a = jarque_bera(&x).unwrap();
    let b = jarque_bera(&y).unwrap();
    // Negative scale flips the sign of the skewness but not |S| or K.
    assert!((a.skewness + b.skewness).abs() < 1e-10);
    assert!((a.kurtosis - b.kurtosis).abs() < 1e-10);
    assert!((a.statistic - b.statistic).abs() < 1e-8);
}

#[test]
fn arch_lm_basics() {
    let y = ar1_series(300, 0.3, 23);
    let res = arch_lm(&y, 6).unwrap();
    assert_eq!(res.df, 6);
    assert_eq!(res.nobs, 294);
    assert!(res.statistic >= 0.0);
    assert!(res.p_value >= 0.0 && res.p_value <= 1.0);
}

#[test]
fn arch_lm_detects_manufactured_arch() {
    // e_t = u_t * sqrt(1 + 0.9 e_{t-1}^2): a strong ARCH(1) process.
    let u = lcg_series(600, 31);
    let mut e = Vec::with_capacity(600);
    let mut prev: f64 = 0.0;
    for &v in &u {
        // Scale uniforms up so the ARCH recursion has visible variance.
        let shock = 3.0 * v;
        let cur = shock * (1.0 + 0.9 * prev * prev).sqrt();
        e.push(cur);
        prev = cur;
    }
    let res = arch_lm(&e, 2).unwrap();
    assert!(
        res.p_value < 0.05,
        "ARCH-LM should reject on a manufactured ARCH(1) series, p = {}",
        res.p_value
    );
}

// ---------------------------------------------------------------- errors

#[test]
fn constant_series_errors() {
    let y = vec![5.0; 50];
    assert!(matches!(
        acf(&y, 5, false),
        Err(DiagError::ConstantSeries { .. })
    ));
    assert!(matches!(
        pacf_yw(&y, 5),
        Err(DiagError::ConstantSeries { .. })
    ));
    assert!(matches!(
        jarque_bera(&y),
        Err(DiagError::ConstantSeries { .. })
    ));
}

#[test]
fn non_finite_input_errors_with_position() {
    let mut y = lcg_series(60, 40);
    y[13] = f64::NAN;
    match acf(&y, 5, false) {
        Err(DiagError::NonFinite { index, .. }) => assert_eq!(index, 13),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    y[13] = f64::INFINITY;
    assert!(matches!(ljung_box(&y, 5), Err(DiagError::NonFinite { .. })));
    assert!(matches!(arch_lm(&y, 2), Err(DiagError::NonFinite { .. })));
    assert!(matches!(jarque_bera(&y), Err(DiagError::NonFinite { .. })));
}

#[test]
fn lag_bounds_are_enforced() {
    let y = lcg_series(30, 50);
    assert!(matches!(
        acf(&y, 0, false),
        Err(DiagError::InvalidLags { .. })
    ));
    assert!(matches!(
        acf(&y, 30, false),
        Err(DiagError::InvalidLags { .. })
    ));
    // statsmodels convention: pacf lags must stay below n/2.
    assert!(matches!(
        pacf_yw(&y, 15),
        Err(DiagError::InvalidLags { .. })
    ));
    assert!(pacf_yw(&y, 14).is_ok());
    assert!(matches!(
        pacf_ols(&y, 15),
        Err(DiagError::InvalidLags { .. })
    ));
    assert!(matches!(
        ljung_box(&y, 30),
        Err(DiagError::InvalidLags { .. })
    ));
    assert!(matches!(arch_lm(&y, 0), Err(DiagError::InvalidLags { .. })));
    assert!(matches!(
        arch_lm(&y, 15),
        Err(DiagError::SeriesTooShort { .. })
    ));
}

#[test]
fn errors_display_teaching_messages() {
    let err = acf(&[1.0, 1.0, 1.0], 1, false).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("constant"), "message should teach: {msg}");
}

// ---------------------------------------------------------------- reports

#[test]
fn reports_carry_decision_and_interpretation() {
    let y = ar1_series(200, 0.8, 61);

    let lb = ljung_box(&y, 10).unwrap();
    let rep = lb.report(0.05).unwrap();
    assert_eq!(rep.df, 10);
    assert_eq!(rep.alpha, 0.05);
    assert_eq!(rep.reject, rep.p_value < 0.05);
    // A strong AR(1) must be flagged as non-white.
    assert!(rep.reject, "Ljung-Box must reject on an AR(1) series");
    assert!(
        rep.interpretation.contains("ACF"),
        "should point at the next diagnostic"
    );
    assert!(!rep.test.is_empty());
    let shown = format!("{rep}");
    assert!(shown.contains("reject"));

    let jb = jarque_bera(&y).unwrap();
    let rep = jb.report(0.05).unwrap();
    assert_eq!(rep.df, 2);
    assert!(!rep.interpretation.is_empty());

    let al = arch_lm(&y, 4).unwrap();
    let rep = al.report(0.05).unwrap();
    assert_eq!(rep.df, 4);
    assert!(!rep.interpretation.is_empty());
}

#[test]
fn report_rejects_invalid_alpha() {
    let y = ar1_series(100, 0.5, 71);
    let lb = ljung_box(&y, 5).unwrap();
    for bad in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
        assert!(matches!(
            lb.report(bad),
            Err(DiagError::InvalidAlpha { .. })
        ));
    }
}
