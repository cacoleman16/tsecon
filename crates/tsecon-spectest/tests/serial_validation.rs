//! Input-validation tests for the serial-correlation tests: every malformed
//! input is a typed [`SpecTestError`], never a panic.

use tsecon_spectest::{
    breusch_godfrey_max_nlags, breusch_godfrey_test, breusch_godfrey_work,
    durbin_watson_regression_statistic, durbin_watson_statistic, durbin_watson_test, DwAlternative,
    SpecTestError, BG_MAX_AUX_CELLS, BG_WORK_BUDGET, DW_EXACT_MAX_NOBS,
};

/// const + one regressor, `n` observations, non-degenerate residuals.
fn design(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let x1: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|t| 0.5 + 0.4 * x1[t] + 0.1 * ((t * t) % 7) as f64)
        .collect();
    (y, vec![vec![1.0; n], x1])
}

const POS: DwAlternative = DwAlternative::Positive;

#[test]
fn empty_and_non_finite_inputs_are_refused() {
    let (y, x) = design(20);
    assert!(matches!(
        breusch_godfrey_test(&[], &x, None).unwrap_err(),
        SpecTestError::EmptyInput { .. }
    ));
    assert!(matches!(
        durbin_watson_test(&[], &x, POS).unwrap_err(),
        SpecTestError::EmptyInput { .. }
    ));
    assert!(matches!(
        breusch_godfrey_test(&y, &[], Some(1)).unwrap_err(),
        SpecTestError::NoRegressors
    ));
    let mut bad = y.clone();
    bad[4] = f64::NAN;
    assert!(matches!(
        breusch_godfrey_test(&bad, &x, None).unwrap_err(),
        SpecTestError::NonFinite { .. }
    ));
    let mut xb = x.clone();
    xb[1][2] = f64::INFINITY;
    assert!(matches!(
        durbin_watson_test(&y, &xb, POS).unwrap_err(),
        SpecTestError::NonFinite { .. }
    ));
    let mut short = x;
    short[1].pop();
    assert!(matches!(
        breusch_godfrey_test(&y, &short, None).unwrap_err(),
        SpecTestError::DimensionMismatch { .. }
    ));
    assert!(durbin_watson_statistic(&[1.0]).is_err());
    assert!(durbin_watson_statistic(&[0.0, 0.0]).is_err());
    assert!(durbin_watson_statistic(&[1.0, f64::NAN]).is_err());
}

#[test]
fn breusch_godfrey_order_errors() {
    let (y, x) = design(20);
    // Explicit zero.
    assert!(matches!(
        breusch_godfrey_test(&y, &x, Some(0)).unwrap_err(),
        SpecTestError::InvalidLags {
            nlags: 0,
            defaulted: false,
            ..
        }
    ));
    // Default of zero on a sample shorter than five.
    let (y4, x4) = design(4);
    assert!(matches!(
        breusch_godfrey_test(&y4, &x4, None).unwrap_err(),
        SpecTestError::InvalidLags {
            defaulted: true,
            ..
        }
    ));
    // Too long for the sample, and absurdly long: refused before any
    // allocation, no overflow.
    for p in [18usize, 1 << 40, usize::MAX] {
        assert!(matches!(
            breusch_godfrey_test(&y, &x, Some(p)).unwrap_err(),
            SpecTestError::DegreesOfFreedom { .. }
        ));
    }
}

#[test]
fn degenerate_and_singular_inputs_are_refused() {
    let n = 20;
    let x1: Vec<f64> = (0..n).map(|t| (0.3 * t as f64).sin()).collect();
    // y exactly in the column space: zero residuals.
    let y: Vec<f64> = x1.iter().map(|v| 1.0 + 2.0 * v).collect();
    let x = vec![vec![1.0; n], x1.clone()];
    assert!(matches!(
        breusch_godfrey_test(&y, &x, None).unwrap_err(),
        SpecTestError::DegenerateResponse { .. }
    ));
    assert!(matches!(
        durbin_watson_test(&y, &x, POS).unwrap_err(),
        SpecTestError::DegenerateResponse { .. }
    ));
    // A design without a constant whose residuals are a nonzero constant
    // (y = 3 + 0.5 z with z orthogonal to the constant): the auxiliary
    // regression adds an intercept, so the centered R^2 is 0/0.
    let z: Vec<f64> = (0..n)
        .map(|t| if t % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let yc: Vec<f64> = z.iter().map(|v| 3.0 + 0.5 * v).collect();
    assert!(matches!(
        breusch_godfrey_test(&yc, std::slice::from_ref(&z), Some(1)).unwrap_err(),
        SpecTestError::DegenerateResponse { .. }
    ));
    // A design without a constant and y an exact linear combination of the
    // design and a constant: the added constant explains the residuals
    // exactly, so both forms would be 0/0 (statsmodels returns LM = T).
    let w: Vec<f64> = x1.iter().map(|v| 2.0 + v).collect();
    let yw: Vec<f64> = w.iter().map(|v| 5.0 + 0.3 * v).collect();
    assert!(matches!(
        breusch_godfrey_test(&yw, std::slice::from_ref(&w), Some(2)).unwrap_err(),
        SpecTestError::DegenerateResponse { .. }
    ));
    // Collinear design.
    let xs = vec![vec![1.0; n], x1.clone(), x1];
    let (y2, _) = design(n);
    assert!(matches!(
        breusch_godfrey_test(&y2, &xs, Some(2)).unwrap_err(),
        SpecTestError::SingularDesign { .. }
    ));
    assert!(matches!(
        durbin_watson_test(&y2, &xs, POS).unwrap_err(),
        SpecTestError::SingularDesign { .. }
    ));
}

#[test]
fn durbin_watson_sample_size_limits() {
    // n = k + 1: the statistic would be a constant.
    let (y, x) = design(3);
    assert!(matches!(
        durbin_watson_test(&y, &x, POS).unwrap_err(),
        SpecTestError::DegreesOfFreedom { .. }
    ));
    // One observation.
    assert!(matches!(
        durbin_watson_test(&[1.0], &[vec![1.0]], POS).unwrap_err(),
        SpecTestError::DegreesOfFreedom { .. }
    ));
    // Beyond the exact-computation cap: refused before any n x n allocation.
    let (y, x) = design(DW_EXACT_MAX_NOBS + 1);
    assert!(matches!(
        durbin_watson_test(&y, &x, POS).unwrap_err(),
        SpecTestError::SampleTooLong { .. }
    ));
    // The smallest admissible sample works.
    let (y, x) = design(4);
    let r = durbin_watson_test(&y, &x, POS).expect("n = k + 2");
    assert!((0.0..=1.0).contains(&r.pvalue));
}

#[test]
fn durbin_watson_statistic_alone_has_no_sample_cap() {
    // Below the cap it is the test's statistic, bit for bit.
    for n in [4, 50, 400] {
        let (y, x) = design(n);
        let full = durbin_watson_test(&y, &x, POS).expect("full test");
        let (d, nobs, df_resid) =
            durbin_watson_regression_statistic(&y, &x).expect("statistic alone");
        assert_eq!(d.to_bits(), full.statistic.to_bits(), "n = {n}");
        assert_eq!((nobs, df_resid), (full.nobs, full.df_resid));
    }
    // Above it, where the test refuses, the statistic is still computed.
    let (y, x) = design(DW_EXACT_MAX_NOBS + 1);
    let (d, nobs, df_resid) = durbin_watson_regression_statistic(&y, &x).expect("uncapped");
    assert!((0.0..=4.0).contains(&d), "d = {d}");
    assert_eq!(
        (nobs, df_resid),
        (DW_EXACT_MAX_NOBS + 1, DW_EXACT_MAX_NOBS + 1 - x.len())
    );
    // The same degrees-of-freedom refusal as the test.
    let (y, x) = design(3);
    assert!(matches!(
        durbin_watson_regression_statistic(&y, &x).unwrap_err(),
        SpecTestError::DegreesOfFreedom { .. }
    ));
}

#[test]
fn error_messages_teach() {
    let (y, x) = design(20);
    let msg = breusch_godfrey_test(&y, &x, Some(0))
        .unwrap_err()
        .to_string();
    assert!(msg.contains("nlags"), "{msg}");
    let (y4, x4) = design(4);
    let msg = breusch_godfrey_test(&y4, &x4, None)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("min(10, n // 5)"), "{msg}");
    let (yl, xl) = design(DW_EXACT_MAX_NOBS + 1);
    let msg = durbin_watson_test(&yl, &xl, POS).unwrap_err().to_string();
    assert!(msg.contains("Breusch-Godfrey"), "{msg}");
}

/// Whether an `n x k_aux` auxiliary design fits both documented budgets,
/// written out from the constants rather than through the crate's helper.
fn fits(n: usize, k_aux: usize) -> bool {
    (n as u128) * (k_aux as u128) <= BG_MAX_AUX_CELLS as u128
        && (n as u128) * (k_aux as u128).pow(2) / 2 + 6 * (k_aux as u128).pow(3) <= BG_WORK_BUDGET
}

#[test]
fn breusch_godfrey_refuses_an_auxiliary_design_past_its_budgets() {
    // T = 1e5 and nlags = 20000 is a valid order (T > k_aux), but its
    // auxiliary design is 2e9 doubles (16 GB): it used to abort the process
    // in the allocator. It is refused before any lag column is built.
    let n = 100_000;
    let y: Vec<f64> = (0..n)
        .map(|t| ((t * t) % 7) as f64 + (0.3 * t as f64).sin())
        .collect();
    let x = vec![vec![1.0; n]];
    let err = breusch_godfrey_test(&y, &x, Some(20_000)).unwrap_err();
    let SpecTestError::AuxiliaryTooLarge {
        n: en,
        k_aux,
        nlags,
        max_nlags,
        defaulted,
    } = err.clone()
    else {
        panic!("expected AuxiliaryTooLarge, got {err:?}");
    };
    assert_eq!((en, k_aux, nlags, defaulted), (n, 20_001, 20_000, false));
    // The largest admissible order: the work budget binds at T = 1e5.
    assert!(
        fits(n, 1 + max_nlags) && !fits(n, 2 + max_nlags),
        "{max_nlags}"
    );
    assert_eq!(max_nlags, breusch_godfrey_max_nlags(n, 1));
    let msg = err.to_string();
    assert!(msg.contains("nlags = 20000"), "{msg}");
    assert!(msg.contains(&format!("nlags <= {max_nlags}")), "{msg}");
    // A 2e9-entry design is past both budgets, and the message says which.
    assert!(msg.contains("past both the memory budget"), "{msg}");
    // A short sample with a long order fits in memory but not in time: the
    // message names the work budget alone.
    let (ys, xs) = (&y[..3000], vec![vec![1.0; 3000]]);
    let msg = breusch_godfrey_test(ys, &xs, Some(2000))
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("past the work budget") && !msg.contains("memory budget"),
        "{msg}"
    );
    // One past the largest admissible order is refused the same way, and an
    // absurd order still fails the degrees-of-freedom check first.
    assert!(matches!(
        breusch_godfrey_test(&y, &x, Some(max_nlags + 1)).unwrap_err(),
        SpecTestError::AuxiliaryTooLarge { .. }
    ));
    assert!(matches!(
        breusch_godfrey_test(&y, &x, Some(usize::MAX)).unwrap_err(),
        SpecTestError::DegreesOfFreedom { .. }
    ));
}

#[test]
fn breusch_godfrey_largest_order_obeys_both_budgets_and_the_sample() {
    // Memory binds for a long sample, work for a short one, the degrees of
    // freedom for a tiny one; the work count saturates instead of
    // overflowing.
    for (n, k_base) in [
        (10_000_000, 3),
        (1_000_000, 2),
        (100_000, 1),
        (20_000, 4),
        (3_000, 2),
        (500, 3),
        (25, 3),
        (5, 4),
        (2, 1),
    ] {
        let m = breusch_godfrey_max_nlags(n, k_base);
        if m > 0 {
            assert!(n > k_base + m && fits(n, k_base + m), "n = {n}: {m}");
        }
        assert!(
            n <= k_base + m + 1 || !fits(n, k_base + m + 1),
            "n = {n}: {m} is not the largest"
        );
    }
    assert_eq!(breusch_godfrey_max_nlags(25, 3), 21);
    assert_eq!(breusch_godfrey_max_nlags(1 << 26, 1), 0);
    assert_eq!(breusch_godfrey_work(usize::MAX, usize::MAX), u128::MAX);
    // A sample so long that even one lag is over the memory budget: the
    // message says so instead of suggesting nlags <= 0.
    let err = SpecTestError::AuxiliaryTooLarge {
        n: 1 << 26,
        k_aux: 3,
        nlags: 1,
        max_nlags: 0,
        defaulted: false,
    };
    assert!(
        err.to_string()
            .contains("even one lag does not fit with this design, so use a shorter sample"),
        "{err}"
    );
    // A design so wide that its 6 k^3 term alone is past the work budget:
    // no sample length helps, and the message does not suggest one.
    let wide = SpecTestError::AuxiliaryTooLarge {
        n: 3000,
        k_aux: 1502,
        nlags: 1,
        max_nlags: 0,
        defaulted: false,
    }
    .to_string();
    assert!(
        wide.contains("at any sample length") && !wide.contains("shorter sample"),
        "{wide}"
    );
    // A default order past the budget (a sample of millions) says so.
    let err = SpecTestError::AuxiliaryTooLarge {
        n: 6_000_000,
        k_aux: 13,
        nlags: 10,
        max_nlags: breusch_godfrey_max_nlags(6_000_000, 3),
        defaulted: true,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("the default order nlags = min(10, n // 5) = 10"),
        "{msg}"
    );
    assert!(msg.contains("pass nlags <= 8"), "{msg}");
    // 7.8e7 entries but only 5e8 multiply-adds: the memory budget alone.
    assert!(
        msg.contains("past the memory budget") && !msg.contains("work budget"),
        "{msg}"
    );
}
