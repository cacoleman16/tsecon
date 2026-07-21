//! Input-validation error tests: every guard in the crate returns a typed
//! [`BreaksError`] that names the problem and the fix, rather than
//! panicking or returning nonsense.

use tsecon_breaks::{
    bai_argmax_two_sided_crit, bai_perron, hansen_supf_pvalue, sup_f_test, BaiPerronConfig,
    BreaksError,
};

/// A small well-formed problem: constant + one regressor, `T = 60`.
fn design() -> (Vec<f64>, Vec<Vec<f64>>) {
    let t = 60;
    let x1: Vec<f64> = (0..t).map(|i| (0.37 * i as f64).sin()).collect();
    let y: Vec<f64> = (0..t)
        .map(|i| 0.4 + 0.9 * x1[i] + 0.3 * ((i * 29 % 13) as f64 - 6.0) / 6.0)
        .collect();
    (y, vec![vec![1.0; t], x1])
}

fn cfg() -> BaiPerronConfig {
    BaiPerronConfig {
        max_breaks: 2,
        trim: 0.15,
    }
}

#[test]
fn empty_inputs_rejected() {
    let (_, x) = design();
    let err = bai_perron(&[], &x, cfg()).unwrap_err();
    assert!(matches!(err, BreaksError::EmptyInput { .. }));

    let (y, _) = design();
    let err = sup_f_test(&y, &[], 0.15).unwrap_err();
    assert!(matches!(err, BreaksError::NoRegressors));
}

#[test]
fn dimension_mismatch_rejected() {
    let (y, mut x) = design();
    x[1].push(0.0);
    let err = bai_perron(&y, &x, cfg()).unwrap_err();
    assert!(matches!(err, BreaksError::DimensionMismatch { .. }));
}

#[test]
fn non_finite_rejected() {
    let (mut y, x) = design();
    y[10] = f64::NAN;
    let err = bai_perron(&y, &x, cfg()).unwrap_err();
    assert!(matches!(err, BreaksError::NonFinite { what: "y" }));

    let (y2, mut x2) = design();
    x2[0][3] = f64::INFINITY;
    let err = sup_f_test(&y2, &x2, 0.15).unwrap_err();
    assert!(matches!(err, BreaksError::NonFinite { what: "x" }));
}

#[test]
fn trim_domain_enforced() {
    let (y, x) = design();
    for bad in [0.0, -0.1, 0.5, 0.9] {
        let err = sup_f_test(&y, &x, bad).unwrap_err();
        assert!(
            matches!(err, BreaksError::InvalidArgument { .. }),
            "trim={bad}"
        );
    }
}

#[test]
fn trim_too_small_names_the_fix() {
    // T = 60, q = 2: trim = 0.02 gives h = 2 < q + 1 = 3.
    let (y, x) = design();
    let err = sup_f_test(&y, &x, 0.02).unwrap_err();
    assert!(matches!(
        err,
        BreaksError::TrimTooSmall { h: 2, q: 2, t: 60 }
    ));
    let msg = err.to_string();
    assert!(
        msg.contains("raise trim"),
        "message must tell the user what to try: {msg}"
    );
}

#[test]
fn infeasible_max_breaks_rejected() {
    // T = 60, trim = 0.25 -> h = 15; 5 breaks need 6 * 15 = 90 > 60.
    let (y, x) = design();
    let err = bai_perron(
        &y,
        &x,
        BaiPerronConfig {
            max_breaks: 5,
            trim: 0.25,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        BreaksError::InfeasibleBreaks {
            max_breaks: 5,
            h: 15,
            t: 60
        }
    ));
    let msg = err.to_string();
    assert!(
        msg.contains("lower max_breaks"),
        "message must suggest the fix: {msg}"
    );
}

#[test]
fn unpublished_trim_rejected_for_bai_perron() {
    let (y, x) = design();
    let err = bai_perron(
        &y,
        &x,
        BaiPerronConfig {
            max_breaks: 2,
            trim: 0.12,
        },
    )
    .unwrap_err();
    assert!(matches!(err, BreaksError::UnsupportedTrim { trim_pct: 12 }));
    let msg = err.to_string();
    assert!(
        msg.contains("0.15"),
        "message must list the supported grid: {msg}"
    );
}

#[test]
fn max_breaks_bounds_enforced() {
    let (y, x) = design();
    for bad in [0_usize, 11] {
        let err = bai_perron(
            &y,
            &x,
            BaiPerronConfig {
                max_breaks: bad,
                trim: 0.15,
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, BreaksError::InvalidArgument { .. }),
            "max_breaks={bad}"
        );
    }
}

#[test]
fn too_many_regressors_rejected() {
    let t = 400;
    let y: Vec<f64> = (0..t).map(|i| (0.17 * i as f64).sin()).collect();
    let x: Vec<Vec<f64>> = (0..11)
        .map(|j| {
            (0..t)
                .map(|i| ((j + 2) as f64 * 0.11 * i as f64).cos())
                .collect()
        })
        .collect();
    let err = sup_f_test(&y, &x, 0.15).unwrap_err();
    assert!(matches!(err, BreaksError::UnsupportedQ { q: 11 }));
}

#[test]
fn sample_too_short_for_supf_search() {
    // T = 10 with trim 0.5 is invalid; with trim 0.45, h = ceil(4.5) = 5
    // and T < 2h leaves no candidate date.
    let y: Vec<f64> = (0..9)
        .map(|i| i as f64 * 0.7 - 2.0 + ((i % 3) as f64))
        .collect();
    let x = vec![vec![1.0; 9]];
    let err = sup_f_test(&y, &x, 0.45).unwrap_err();
    assert!(matches!(err, BreaksError::TooShort { .. }));
}

#[test]
fn collinear_segment_reported_with_location() {
    // Second column duplicates the constant: singular on every segment.
    let t = 60;
    let y: Vec<f64> = (0..t).map(|i| (0.37 * i as f64).sin()).collect();
    let x = vec![vec![1.0; t], vec![1.0; t]];
    let err = bai_perron(&y, &x, cfg()).unwrap_err();
    assert!(matches!(err, BreaksError::Singular { .. }));
    let msg = err.to_string();
    assert!(
        msg.contains("collinear"),
        "message must name the cause: {msg}"
    );
}

#[test]
fn hansen_pvalue_domain_enforced() {
    assert!(matches!(
        hansen_supf_pvalue(1.0, 0, 0.15).unwrap_err(),
        BreaksError::UnsupportedQ { q: 0 }
    ));
    assert!(matches!(
        hansen_supf_pvalue(1.0, 11, 0.15).unwrap_err(),
        BreaksError::UnsupportedQ { q: 11 }
    ));
    assert!(matches!(
        hansen_supf_pvalue(-1.0, 1, 0.15).unwrap_err(),
        BreaksError::InvalidArgument { .. }
    ));
    assert!(matches!(
        hansen_supf_pvalue(f64::NAN, 1, 0.15).unwrap_err(),
        BreaksError::InvalidArgument { .. }
    ));
    assert!(matches!(
        hansen_supf_pvalue(1.0, 1, 0.6).unwrap_err(),
        BreaksError::InvalidArgument { .. }
    ));
}

#[test]
fn ci_level_domain_enforced() {
    for bad in [0.0, 1.0, -0.5, 1.5] {
        assert!(matches!(
            bai_argmax_two_sided_crit(bad).unwrap_err(),
            BreaksError::InvalidArgument { .. }
        ));
    }
}

#[test]
fn errors_display_teaches() {
    // Spot-check that Display output states what happened and what to do.
    let (y, x) = design();
    let msg = bai_perron(
        &y,
        &x,
        BaiPerronConfig {
            max_breaks: 0,
            trim: 0.15,
        },
    )
    .unwrap_err()
    .to_string();
    assert!(msg.contains("max_breaks must be at least 1"));
    let msg = sup_f_test(&[], &x, 0.15).unwrap_err().to_string();
    assert!(msg.contains("empty input"));
    let _ = y;
}
