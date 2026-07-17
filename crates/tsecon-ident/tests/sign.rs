//! Behavioral tests for the sign-restriction checker: validation, the
//! per-shock sign choice (normalization), horizon bands, exact-zero
//! handling, and infeasible patterns.

use tsecon_ident::{IdentError, Sign, SignRestriction, SignRestrictionSet};
use tsecon_linalg::faer::Mat;

/// Builds a horizon-indexed IRF from an explicit closure `f(h, i, j)`.
fn irf(horizon: usize, n: usize, f: impl Fn(usize, usize, usize) -> f64) -> Vec<Mat<f64>> {
    (0..=horizon)
        .map(|h| Mat::from_fn(n, n, |i, j| f(h, i, j)))
        .collect()
}

#[test]
fn out_of_range_indices_are_rejected() {
    let r = vec![SignRestriction::at(3, 0, 0, Sign::Positive)];
    assert!(matches!(
        SignRestrictionSet::new(r, 3, 2),
        Err(IdentError::RestrictionOutOfRange { .. })
    ));
    let r = vec![SignRestriction::at(0, 0, 5, Sign::Positive)];
    assert!(matches!(
        SignRestrictionSet::new(r, 3, 2),
        Err(IdentError::RestrictionOutOfRange { .. })
    ));
    let r = vec![SignRestriction::over(0, 0, 2, 1, Sign::Positive)];
    assert!(matches!(
        SignRestrictionSet::new(r, 3, 3),
        Err(IdentError::InvalidArgument { .. })
    ));
}

#[test]
fn empty_restrictions_rejected() {
    assert!(matches!(
        SignRestrictionSet::new(vec![], 3, 2),
        Err(IdentError::InvalidArgument { .. })
    ));
}

#[test]
fn positive_pattern_accepts_and_flips_when_needed() {
    let set = SignRestrictionSet::new(
        vec![SignRestriction::over(0, 0, 0, 1, Sign::Positive)],
        2,
        1,
    )
    .expect("set");

    let up = irf(1, 2, |_, i, j| if i == 0 && j == 0 { 2.0 } else { 0.1 });
    let o = set.accept_orientations(&up).expect("accepted");
    assert_eq!(o[0], 1.0);

    let down = irf(1, 2, |_, i, j| if i == 0 && j == 0 { -2.0 } else { 0.1 });
    let o = set.accept_orientations(&down).expect("accepted via flip");
    assert_eq!(o[0], -1.0);
}

#[test]
fn sign_flip_within_band_is_rejected() {
    let set = SignRestrictionSet::new(
        vec![SignRestriction::over(0, 0, 0, 1, Sign::Positive)],
        2,
        1,
    )
    .expect("set");
    let zigzag = irf(1, 2, |h, i, j| {
        if i == 0 && j == 0 {
            if h == 0 {
                1.0
            } else {
                -1.0
            }
        } else {
            0.1
        }
    });
    assert!(set.accept_orientations(&zigzag).is_none());
}

#[test]
fn exact_zero_response_fails_both_signs() {
    let set = SignRestrictionSet::new(vec![SignRestriction::at(0, 0, 0, Sign::Positive)], 2, 0)
        .expect("set");
    let flat = irf(0, 2, |_, _, _| 0.0);
    assert!(set.accept_orientations(&flat).is_none());
}

#[test]
fn contradictory_same_cell_signs_never_accept() {
    let set = SignRestrictionSet::new(
        vec![
            SignRestriction::at(0, 0, 0, Sign::Positive),
            SignRestriction::at(0, 0, 0, Sign::Negative),
        ],
        2,
        0,
    )
    .expect("set");
    for v in [-3.0, -0.1, 0.1, 3.0] {
        let m = irf(0, 2, |_, i, j| if i == 0 && j == 0 { v } else { 0.2 });
        assert!(set.accept_orientations(&m).is_none());
    }
}

#[test]
fn multi_shock_orientations_are_independent() {
    // Shock 0 raises var0; shock 1 lowers var1. Both restricted, each
    // oriented independently.
    let set = SignRestrictionSet::new(
        vec![
            SignRestriction::at(0, 0, 0, Sign::Positive),
            SignRestriction::at(1, 1, 0, Sign::Negative),
        ],
        2,
        0,
    )
    .expect("set");
    assert_eq!(set.restricted_shocks(), &[0, 1]);
    // Raw: var0<-shock0 is -1 (needs flip to +), var1<-shock1 is +1 (needs
    // flip to -).
    let m = irf(0, 2, |_, i, j| match (i, j) {
        (0, 0) => -1.0,
        (1, 1) => 1.0,
        _ => 0.05,
    });
    let o = set.accept_orientations(&m).expect("accepted");
    assert_eq!(o[0], -1.0);
    assert_eq!(o[1], -1.0);
}
