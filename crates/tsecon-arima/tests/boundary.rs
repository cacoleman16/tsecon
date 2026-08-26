//! Stationarity/invertibility boundary detection on `ArimaResults`
//! (`boundary()` / `boundary_note()`), the tsecon-garch round-7 pattern
//! ported to ARIMA.
//!
//! The attracting case is the classic one: fitting ARIMA(0,1,1) to white
//! noise over-differences the series, the implied MA root sits at -1, and
//! exact MLE piles the estimate up against the invertibility boundary.
//! The trap this machinery closes: at that boundary the *full-vector*
//! observed information still inverts (`param_cov()` succeeds) and hands
//! back a finite, confident-looking standard error for `ma.L1` — a
//! parameter whose sampling distribution is not normal and whose
//! information is singular in the constrained direction by construction.
//! `boundary()` marks it; the Python binding NaNs the flagged `bse`
//! entries with `se_valid = false` and forwards `boundary_note`.

mod common;

use common::{simulate_arma, Lcg};
use tsecon_arima::ArimaSpec;

/// White noise, over-differenced by the specification: the fitted MA root
/// lands within the documented epsilon (0.1%) of the unit circle and the
/// MA block is flagged, while the observed information still "succeeds"
/// with a finite se — exactly the honest-signaling gap being closed.
#[test]
fn over_differenced_ma_fit_flags_the_invertibility_boundary() {
    // Seed chosen so the pile-up is squarely inside the epsilon (theta at
    // -1 to eight digits); the fit is deterministic, so this is stable.
    let mut rng = Lcg::new(3);
    let y = simulate_arma(&mut rng, 300, 0.0, &[], &[], 1.0);
    let spec = ArimaSpec::new(0, 1, 1).unwrap().with_constant(false);
    let res = spec.fit(&y).unwrap();

    assert!(
        res.ma()[0] <= -0.999,
        "the over-differenced fit should pile up at the MA boundary, got {}",
        res.ma()[0]
    );
    // Packed order [ma.L1, sigma2]: the MA block is flagged, sigma2 never.
    assert_eq!(res.boundary(), vec![true, false]);
    let note = res.boundary_note().expect("a boundary note");
    assert!(
        note.contains("MA (invertibility)") && note.contains("ma.L1"),
        "note should name the block and the parameter: {note}"
    );
    assert!(
        note.contains("over-differenced"),
        "note should teach the over-differencing diagnosis: {note}"
    );
    // The trap, pinned: the full-vector observed information does NOT
    // refuse at this boundary — it returns a finite se for ma.L1. That is
    // why the flags must exist (and why the binding NaNs the flagged se).
    let pc = res.param_cov().expect("full-vector covariance still inverts");
    assert!(
        pc.se()[0].is_finite(),
        "if this starts refusing, the boundary flags are no longer the only \
         guard and this test should be revisited"
    );
}

/// A comfortably interior fit carries no flags and no note.
#[test]
fn interior_fit_has_no_boundary_flags() {
    let mut rng = Lcg::new(20260716);
    let y = simulate_arma(&mut rng, 600, 1.0, &[0.6], &[0.3], 1.5);
    let spec = ArimaSpec::new(1, 0, 1).unwrap().with_constant(true);
    let res = spec.fit(&y).unwrap();
    assert!(
        res.boundary().iter().all(|&b| !b),
        "interior ARMA(1,1)c should not be flagged: {:?}",
        res.boundary()
    );
    assert!(res.boundary_note().is_none());
}
