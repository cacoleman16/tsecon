//! Validation tests for the Svensson (1994) cross-sectional fit.
//!
//! Coverage motivation: `SvenssonFit::yield_at` had zero coverage, as did
//! the `DimensionMismatch` and `Underdetermined` guards in `fit_svensson`.
//! The `Underdetermined` guard is the load-bearing one: the Svensson
//! design has four columns, so a fit on exactly four maturities is
//! *exactly* determined and would come back with zero residuals and
//! `R^2 = 1` -- a curve that looks like a flawless fit but carries no
//! information at all. Rejecting it is the difference between an error
//! and a silently meaningless result.

use tsecon_termstructure::{fit_svensson, TermStructureError};

/// A well-conditioned cross-section: eight maturities and two decays that
/// are far enough apart to keep the loadings independent.
fn curve() -> (Vec<f64>, Vec<f64>) {
    let maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
    let yields = vec![1.20, 1.45, 1.85, 2.30, 2.55, 2.80, 2.95, 3.05];
    (maturities, yields)
}

const L1: f64 = 0.7;
const L2: f64 = 0.1;

/// `yield_at` must agree with `fitted` on the estimation grid: both
/// evaluate the same four loadings against the same factors, so any
/// discrepancy means one of them has a transposed or mis-indexed term.
#[test]
fn yield_at_agrees_with_fitted_on_the_estimation_grid() {
    let (maturities, yields) = curve();
    let fit = fit_svensson(&maturities, &yields, L1, L2).expect("well-posed curve");
    let fitted = fit.fitted(&maturities).expect("fitted on the grid");
    for (i, &m) in maturities.iter().enumerate() {
        let point = fit.yield_at(m).expect("yield at an in-sample maturity");
        assert!(
            (point - fitted[i]).abs() <= 1e-12,
            "maturity {m}: yield_at {point} vs fitted {}",
            fitted[i]
        );
    }
}

/// `yield_at` must also work off-grid -- that is its whole purpose --
/// and interpolate monotonically between neighbouring in-sample points on
/// this upward-sloping curve.
#[test]
fn yield_at_interpolates_off_grid() {
    let (maturities, yields) = curve();
    let fit = fit_svensson(&maturities, &yields, L1, L2).expect("well-posed curve");
    let a = fit.yield_at(1.0).expect("ok");
    let mid = fit.yield_at(1.5).expect("ok");
    let b = fit.yield_at(2.0).expect("ok");
    assert!(a.is_finite() && mid.is_finite() && b.is_finite());
    assert!(
        a < mid && mid < b,
        "expected monotone interpolation, got {a} / {mid} / {b}"
    );
}

/// The residuals returned by the fit must actually be `y - yhat`.
#[test]
fn residuals_are_the_fitted_errors() {
    let (maturities, yields) = curve();
    let fit = fit_svensson(&maturities, &yields, L1, L2).expect("well-posed curve");
    for (i, &m) in maturities.iter().enumerate() {
        let implied = yields[i] - fit.yield_at(m).expect("ok");
        assert!(
            (implied - fit.residuals[i]).abs() <= 1e-10,
            "residual[{i}]: reported {} vs implied {implied}",
            fit.residuals[i]
        );
    }
}

/// A non-positive or non-finite maturity has no place on a yield curve
/// (the loadings divide by `lambda * t`), so `yield_at` must reject it
/// rather than return an infinity.
#[test]
fn yield_at_rejects_invalid_maturities() {
    let (maturities, yields) = curve();
    let fit = fit_svensson(&maturities, &yields, L1, L2).expect("well-posed curve");
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            fit.yield_at(bad).is_err(),
            "yield_at accepted maturity {bad}"
        );
    }
}

/// Four maturities exactly determine the four Svensson factors, which
/// would produce zero residuals and a meaningless `R^2 = 1`. The fit must
/// refuse rather than hand back that illusion.
#[test]
fn exactly_determined_cross_section_is_rejected() {
    let maturities = vec![1.0, 2.0, 5.0, 10.0];
    let yields = vec![1.85, 2.30, 2.80, 3.05];
    let err = fit_svensson(&maturities, &yields, L1, L2).unwrap_err();
    match err {
        TermStructureError::Underdetermined {
            maturities: m,
            factors,
            ..
        } => {
            assert_eq!(m, 4);
            assert_eq!(factors, 4);
        }
        other => panic!("expected Underdetermined, got {other:?}"),
    }
    // Three maturities (strictly fewer than the factors) too.
    assert!(matches!(
        fit_svensson(&[1.0, 2.0, 5.0], &[1.8, 2.3, 2.8], L1, L2).unwrap_err(),
        TermStructureError::Underdetermined { .. }
    ));
    // Five is the documented minimum and must be accepted.
    let maturities = vec![1.0, 2.0, 3.0, 5.0, 10.0];
    let yields = vec![1.85, 2.30, 2.55, 2.80, 3.05];
    assert!(fit_svensson(&maturities, &yields, L1, L2).is_ok());
}

#[test]
fn mismatched_yield_and_maturity_lengths_are_rejected() {
    let (maturities, yields) = curve();
    let err = fit_svensson(&maturities, &yields[..6], L1, L2).unwrap_err();
    match err {
        TermStructureError::DimensionMismatch { expected, got, .. } => {
            assert_eq!(expected, 8);
            assert_eq!(got, 6);
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

#[test]
fn non_finite_yields_are_rejected() {
    let (maturities, mut yields) = curve();
    yields[3] = f64::NAN;
    assert!(matches!(
        fit_svensson(&maturities, &yields, L1, L2).unwrap_err(),
        TermStructureError::NonFinite { .. }
    ));
}

/// Two decays that are (near) equal make the two curvature loadings
/// collinear, so the design is singular. That must surface as an error,
/// not as an arbitrary factor split between two indistinguishable humps.
#[test]
fn near_identical_decays_give_a_singular_design() {
    let (maturities, yields) = curve();
    let res = fit_svensson(&maturities, &yields, 0.5, 0.5);
    assert!(
        res.is_err(),
        "identical lambdas must not produce a unique factor split"
    );
}
