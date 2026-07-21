//! "Errors that teach": malformed inputs must be rejected with messages that
//! say what happened, why it matters, and what to try — never a panic, never
//! a silent NaN. (statsmodels, by contrast, silently returns NaN standard
//! errors when the Hall-Sheather offset leaves (0, 1); this crate refuses
//! with instructions.)

use tsecon_quantile::{growth_at_risk, quantile_lp, quantile_regression, QuantileError};

fn ok_y(n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| (t as f64 * 0.37).sin() + 0.01 * t as f64)
        .collect()
}

fn ok_cols(n: usize) -> Vec<Vec<f64>> {
    vec![
        vec![1.0; n],
        (0..n).map(|t| (t as f64 * 0.61).cos()).collect(),
    ]
}

/// Every rejection must carry actionable guidance.
fn assert_teaches(err: &QuantileError, must_contain: &[&str]) {
    let msg = err.to_string();
    for needle in must_contain {
        assert!(
            msg.contains(needle),
            "error message must teach: expected {needle:?} in {msg:?}"
        );
    }
}

#[test]
fn empty_inputs_are_rejected() {
    let err = quantile_regression(&[], &ok_cols(0), &[0.5]).unwrap_err();
    assert!(matches!(err, QuantileError::EmptyInput { .. }));
    assert_teaches(&err, &["empty input", "supply"]);

    let err = quantile_regression(&ok_y(10), &[], &[0.5]).unwrap_err();
    assert!(matches!(err, QuantileError::EmptyInput { .. }));
}

#[test]
fn missing_or_invalid_taus_are_rejected() {
    let y = ok_y(50);
    let cols = ok_cols(50);
    let err = quantile_regression(&y, &cols, &[]).unwrap_err();
    assert!(matches!(err, QuantileError::NoTaus));
    assert_teaches(&err, &["at least one tau", "(0, 1)"]);

    for bad in [0.0, 1.0, -0.2, 1.7, f64::NAN] {
        let err = quantile_regression(&y, &cols, &[0.5, bad]).unwrap_err();
        assert!(matches!(err, QuantileError::InvalidTau { .. }), "tau={bad}");
        assert_teaches(&err, &["strictly inside (0, 1)"]);
    }
}

#[test]
fn nan_and_inf_are_rejected_with_the_index() {
    let mut y = ok_y(50);
    y[7] = f64::NAN;
    let err = quantile_regression(&y, &ok_cols(50), &[0.5]).unwrap_err();
    assert!(matches!(err, QuantileError::NonFinite { index: 7, .. }));
    assert_teaches(&err, &["index 7", "clean or drop"]);

    let mut cols = ok_cols(50);
    cols[1][3] = f64::INFINITY;
    let err = quantile_regression(&ok_y(50), &cols, &[0.5]).unwrap_err();
    assert!(matches!(err, QuantileError::NonFinite { index: 3, .. }));
}

#[test]
fn mismatched_lengths_are_rejected_with_both_sizes() {
    let err = quantile_regression(&ok_y(50), &ok_cols(49), &[0.5]).unwrap_err();
    assert!(matches!(
        err,
        QuantileError::DimensionMismatch {
            expected: 50,
            got: 49,
            ..
        }
    ));
    assert_teaches(&err, &["expected length 50", "got 49"]);

    let err = quantile_lp(&ok_y(50), &ok_y(40), &[0.5], 4, 2).unwrap_err();
    assert!(matches!(err, QuantileError::DimensionMismatch { .. }));

    let err = growth_at_risk(&ok_y(50), &[ok_y(30)], 1, &[0.5], true).unwrap_err();
    assert!(matches!(err, QuantileError::DimensionMismatch { .. }));
    assert_teaches(&err, &["condition series"]);
}

#[test]
fn too_few_observations_are_rejected() {
    let err = quantile_regression(&ok_y(2), &ok_cols(2), &[0.5]).unwrap_err();
    assert!(matches!(
        err,
        QuantileError::DegreesOfFreedom { n: 2, k: 2 }
    ));
    assert_teaches(&err, &["longer series"]);
}

#[test]
fn exhausting_horizons_are_rejected_with_the_arithmetic() {
    // n = 20, p = 4: k = 10, and nobs = n - h - p hits k at h = 6.
    // Lag designs need non-collinear series, so use a scrambled sequence
    // (p lags of a pure sinusoid span only two dimensions).
    let scramble = |n: usize, mut s: u64| -> Vec<f64> {
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    };
    let err = quantile_lp(&scramble(20, 1), &scramble(20, 2), &[0.5], 12, 4).unwrap_err();
    assert!(matches!(err, QuantileError::HorizonExhaustsSample { .. }));
    assert_teaches(&err, &["lower the maximum horizon", "lag controls"]);

    let cond: Vec<f64> = (0..6).map(|t| (t as f64 * 0.83).cos()).collect();
    let err = growth_at_risk(&ok_y(6), &[cond], 4, &[0.5], true).unwrap_err();
    assert!(matches!(err, QuantileError::HorizonExhaustsSample { .. }));
}

#[test]
fn gar_rejects_zero_horizon_and_unsorted_taus() {
    let y = ok_y(60);
    let x = ok_y(60);
    let err = growth_at_risk(&y, std::slice::from_ref(&x), 0, &[0.5], true).unwrap_err();
    assert!(matches!(err, QuantileError::ZeroHorizon));
    assert_teaches(&err, &["horizon >= 1", "periods ahead"]);

    let err = growth_at_risk(&y, std::slice::from_ref(&x), 1, &[0.5, 0.25], true).unwrap_err();
    assert!(matches!(err, QuantileError::TausNotIncreasing { index: 1 }));
    assert_teaches(&err, &["strictly increasing", "sort the taus"]);

    let err = growth_at_risk(&y, &[x], 1, &[0.5, 0.5], true).unwrap_err();
    assert!(matches!(err, QuantileError::TausNotIncreasing { .. }));
}

#[test]
fn extreme_tau_for_the_sample_size_teaches_instead_of_nan() {
    // At n = 60 the Hall-Sheather offset pushes tau = 0.01 outside (0, 1);
    // statsmodels silently returns NaN bse here — we refuse and explain.
    let y = ok_y(60);
    let cols = ok_cols(60);
    let err = quantile_regression(&y, &cols, &[0.01]).unwrap_err();
    assert!(matches!(err, QuantileError::DegenerateBandwidth { .. }));
    assert_teaches(&err, &["less extreme tau", "longer sample"]);
}

#[test]
fn collinear_designs_are_rejected_as_singular() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|t| (t as f64 * 0.61).cos()).collect();
    let doubled: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
    let cols = vec![vec![1.0; n], x, doubled];
    let err = quantile_regression(&ok_y(n), &cols, &[0.5]).unwrap_err();
    // The collinearity surfaces in the first weighted least-squares step,
    // which this crate delegates to tsecon-hac (the single OLS owner).
    assert!(
        matches!(err, QuantileError::Hac(_) | QuantileError::Singular { .. }),
        "collinear design must be rejected, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("singular"),
        "must name the cause: {msg}"
    );
}

#[test]
fn constant_outcome_teaches_about_degenerate_scale() {
    // A constant y has zero residual IQR and zero std: no bandwidth exists.
    let n = 40;
    let y = vec![1.5; n];
    let cols = ok_cols(n);
    let err = quantile_regression(&y, &cols, &[0.5]).unwrap_err();
    assert!(matches!(err, QuantileError::DegenerateBandwidth { .. }));
    assert_teaches(&err, &["near-)constant"]);
}
