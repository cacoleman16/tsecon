//! "Errors that teach": malformed input is rejected with a message that
//! names what went wrong and what to try, and never panics.

use tsecon_funcshock::{
    flp, flp_scenario, functional_pca, fvar_scenario, scenario_response, scenario_weights,
    FuncShockError,
};

fn panel(t: usize, m: usize) -> Vec<Vec<f64>> {
    (0..t)
        .map(|tt| {
            (0..m)
                .map(|mm| ((tt * 31 + mm * 7) % 13) as f64 * 0.1 + (tt as f64 * 0.7).sin())
                .collect()
        })
        .collect()
}

fn msg(e: &FuncShockError) -> String {
    e.to_string()
}

// ------------------------------------------------------------------ fpca

#[test]
fn fpca_rejects_empty_panel() {
    let e = functional_pca(&[], 1).unwrap_err();
    assert!(matches!(e, FuncShockError::EmptyInput { .. }));
    assert!(msg(&e).contains("T x M panel"), "{e}");
}

#[test]
fn fpca_rejects_empty_rows() {
    let e = functional_pca(&[vec![], vec![]], 1).unwrap_err();
    assert!(matches!(e, FuncShockError::EmptyInput { .. }));
    assert!(msg(&e).contains("at least one grid point"), "{e}");
}

#[test]
fn fpca_rejects_single_curve() {
    let e = functional_pca(&panel(1, 4), 1).unwrap_err();
    assert!(matches!(e, FuncShockError::DimensionMismatch { .. }));
    assert!(msg(&e).contains("at least 2 observations"), "{e}");
}

#[test]
fn fpca_rejects_ragged_grid() {
    let mut curves = panel(5, 4);
    curves[3] = vec![1.0, 2.0];
    let e = functional_pca(&curves, 2).unwrap_err();
    assert!(
        matches!(
            e,
            FuncShockError::RaggedRow {
                row: 3,
                expected: 4,
                got: 2,
                ..
            }
        ),
        "{e:?}"
    );
    assert!(msg(&e).contains("interpolate"), "{e}");
}

#[test]
fn fpca_rejects_nan() {
    let mut curves = panel(6, 4);
    curves[2][1] = f64::NAN;
    let e = functional_pca(&curves, 2).unwrap_err();
    assert!(matches!(e, FuncShockError::NonFinite { .. }));
    assert!(msg(&e).contains("clean or interpolate"), "{e}");
}

#[test]
fn fpca_rejects_k_zero_and_k_beyond_m() {
    let curves = panel(10, 4);
    for k in [0usize, 5] {
        let e = functional_pca(&curves, k).unwrap_err();
        assert!(
            matches!(e, FuncShockError::InvalidFactorCount { max: 4, .. }),
            "{e:?}"
        );
        assert!(msg(&e).contains("1 <= n_factors <= 4"), "{e}");
    }
}

#[test]
fn fpca_rejects_constant_panel() {
    let curves = vec![vec![1.0, 2.0, 3.0]; 8];
    let e = functional_pca(&curves, 1).unwrap_err();
    assert!(matches!(e, FuncShockError::ZeroVariance));
    assert!(msg(&e).contains("curves that move"), "{e}");
}

// ------------------------------------------------------------------- flp

fn scores_ok(t: usize, k: usize) -> Vec<Vec<f64>> {
    (0..t)
        .map(|tt| (0..k).map(|kk| ((tt + kk) as f64 * 0.61).sin()).collect())
        .collect()
}

fn y_ok(t: usize) -> Vec<f64> {
    (0..t).map(|tt| (tt as f64 * 0.37).cos()).collect()
}

#[test]
fn flp_rejects_length_mismatch() {
    let e = flp(&y_ok(30), &scores_ok(29, 2), 2, 1, None).unwrap_err();
    assert!(
        matches!(
            e,
            FuncShockError::DimensionMismatch {
                expected: 30,
                got: 29,
                ..
            }
        ),
        "{e:?}"
    );
    assert!(msg(&e).contains("one score vector per outcome"), "{e}");
}

#[test]
fn flp_rejects_ragged_scores_and_nan() {
    let mut s = scores_ok(30, 2);
    s[10] = vec![1.0];
    let e = flp(&y_ok(30), &s, 2, 1, None).unwrap_err();
    assert!(
        matches!(e, FuncShockError::RaggedRow { row: 10, .. }),
        "{e:?}"
    );

    let mut s = scores_ok(30, 2);
    s[4][1] = f64::INFINITY;
    let e = flp(&y_ok(30), &s, 2, 1, None).unwrap_err();
    assert!(matches!(e, FuncShockError::NonFinite { .. }), "{e:?}");

    let mut y = y_ok(30);
    y[7] = f64::NAN;
    let e = flp(&y, &scores_ok(30, 2), 2, 1, None).unwrap_err();
    assert!(matches!(e, FuncShockError::NonFinite { .. }), "{e:?}");
}

#[test]
fn flp_rejects_horizon_that_exhausts_the_sample() {
    // T = 12, p = 2, K = 2: nparams = 5, so h = 6 leaves nobs = 4 <= 5.
    let e = flp(&y_ok(12), &scores_ok(12, 2), 6, 2, None).unwrap_err();
    assert!(matches!(e, FuncShockError::HorizonTooLong { .. }), "{e:?}");
    assert!(
        msg(&e).contains("shorten the horizon"),
        "teaching message: {e}"
    );
}

#[test]
fn flp_rejects_too_many_lag_controls() {
    let e = flp(&y_ok(5), &scores_ok(5, 1), 0, 5, None).unwrap_err();
    assert!(matches!(e, FuncShockError::SeriesTooShort { .. }), "{e:?}");
    assert!(
        msg(&e).contains("longer series or fewer lag controls"),
        "{e}"
    );
}

// -------------------------------------------------------------- scenario

#[test]
fn scenario_weights_rejects_grid_mismatch() {
    let fpca = functional_pca(&panel(20, 5), 2).expect("fpca");
    let e = scenario_weights(&fpca.eigenfunctions, &[1.0, 2.0]).unwrap_err();
    assert!(
        matches!(
            e,
            FuncShockError::DimensionMismatch {
                expected: 5,
                got: 2,
                ..
            }
        ),
        "{e:?}"
    );
    assert!(msg(&e).contains("eigenfunction grid"), "{e}");
}

#[test]
fn scenario_weights_rejects_nan_delta() {
    let fpca = functional_pca(&panel(20, 5), 2).expect("fpca");
    let e = scenario_weights(&fpca.eigenfunctions, &[1.0, 2.0, f64::NAN, 0.0, 1.0]).unwrap_err();
    assert!(matches!(e, FuncShockError::NonFinite { .. }), "{e:?}");
}

#[test]
fn scenario_response_rejects_weight_count_mismatch() {
    let fpca = functional_pca(&panel(40, 5), 2).expect("fpca");
    let fit = flp(&y_ok(40), &fpca.scores, 3, 1, None).expect("flp");
    let e = scenario_response(&fit, &[0.5]).unwrap_err();
    assert!(
        matches!(
            e,
            FuncShockError::DimensionMismatch {
                expected: 2,
                got: 1,
                ..
            }
        ),
        "{e:?}"
    );
    assert!(msg(&e).contains("score regressors K"), "{e}");
}

#[test]
fn flp_scenario_rejects_k_mismatch_between_fpca_and_fit() {
    // fpca kept 3 eigenfunctions but the FLP was fitted on 2 scores: the
    // projection yields 3 weights, which the fit must reject.
    let curves = panel(40, 5);
    let fpca3 = functional_pca(&curves, 3).expect("fpca3");
    let fpca2 = functional_pca(&curves, 2).expect("fpca2");
    let fit2 = flp(&y_ok(40), &fpca2.scores, 3, 1, None).expect("flp");
    let delta = vec![1.0; 5];
    let e = flp_scenario(&fpca3, &fit2, &delta).unwrap_err();
    assert!(
        matches!(
            e,
            FuncShockError::DimensionMismatch {
                expected: 2,
                got: 3,
                ..
            }
        ),
        "{e:?}"
    );
}

// ------------------------------------------------------------------ fvar

#[test]
fn fvar_rejects_mismatches_and_short_samples() {
    let scores = scores_ok(30, 2);
    let y = y_ok(30);

    let e = fvar_scenario(&scores, &y, &[1.0], 2, 4).unwrap_err();
    assert!(
        matches!(
            e,
            FuncShockError::DimensionMismatch {
                expected: 2,
                got: 1,
                ..
            }
        ),
        "{e:?}"
    );

    let e = fvar_scenario(&scores, &y_ok(29), &[1.0, 0.0], 2, 4).unwrap_err();
    assert!(
        matches!(e, FuncShockError::DimensionMismatch { .. }),
        "{e:?}"
    );

    // 5 observations cannot fit a 3-variable VAR(2): the VAR engine's
    // teaching error propagates through the Var wrapper.
    let e = fvar_scenario(&scores_ok(5, 2), &y_ok(5), &[1.0, 0.0], 2, 4).unwrap_err();
    assert!(matches!(e, FuncShockError::Var(_)), "{e:?}");
    assert!(msg(&e).contains("VAR-engine error"), "{e}");
}

#[test]
fn fvar_rejects_nan_inputs() {
    let mut scores = scores_ok(30, 2);
    scores[3][0] = f64::NAN;
    let e = fvar_scenario(&scores, &y_ok(30), &[1.0, 0.0], 1, 4).unwrap_err();
    assert!(matches!(e, FuncShockError::NonFinite { .. }), "{e:?}");

    let e = fvar_scenario(&scores_ok(30, 2), &y_ok(30), &[f64::NAN, 0.0], 1, 4).unwrap_err();
    assert!(matches!(e, FuncShockError::NonFinite { .. }), "{e:?}");
}
