//! The within-design rank guard: an absorbed (entity-constant) regressor
//! must raise `SingularDesign`, not return a publishable t-statistic.
//!
//! Audit rounds 2-4, finding 1: the old guard was a Cholesky
//! positive-definiteness test that fired only when the demeaned residue
//! was bit-exactly zero — exactly-representable entity constants raised,
//! ordinary doubles (log land area, a share in [0, 1]) returned, and the
//! default cluster covariance turned the O(1e-16) residue into t-values
//! that reached nominal significance in 19.2% of draws. linearmodels
//! raises `AbsorbingEffectError` on this design at every cov_type, and
//! the docstring promises its conventions.

use tsecon_linalg::faer::Mat;
use tsecon_panel::{panel_ols_fe, PanelData, PanelError};

/// Minimal LCG so the test depends on no RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// A panel with one live regressor and one regressor built per entity by
/// `entity_value` (constant within each entity when that closure ignores
/// the period).
fn panel_with_entity_column(
    n_ent: usize,
    n_per: usize,
    entity_value: impl Fn(usize, usize) -> f64,
) -> PanelData {
    let mut rng = Lcg(20260813);
    let live = Mat::from_fn(n_ent, n_per, |_, _| rng.gaussian());
    let mut outcome = Mat::from_fn(n_ent, n_per, |_, _| rng.gaussian());
    for i in 0..n_ent {
        for t in 0..n_per {
            outcome[(i, t)] += 0.9 * live[(i, t)];
        }
    }
    let dead = Mat::from_fn(n_ent, n_per, entity_value);
    PanelData::balanced(
        outcome,
        vec![("live".to_string(), live), ("dead".to_string(), dead)],
    )
    .expect("balanced panel")
}

fn assert_singular(result: Result<tsecon_panel::FePanelOls, PanelError>, what: &str) {
    match result {
        Err(PanelError::SingularDesign { .. }) => {}
        Err(other) => panic!("{what}: expected SingularDesign, got {other}"),
        Ok(_) => panic!("{what}: expected SingularDesign, got a fitted model"),
    }
}

/// Exactly-representable entity constants (the case the old guard caught)
/// still raise.
#[test]
fn founding_year_raises() {
    let data = panel_with_entity_column(20, 8, |i, _| (1950 + 3 * i) as f64);
    assert_singular(panel_ols_fe(&data), "integer founding year");
}

/// Entity constants that are ordinary doubles — the audit's adversarial
/// set: log land area, latitude-like values, shares in [0, 1] — demean to
/// an O(1e-16) residue that the old positive-definiteness guard accepted
/// and the cluster covariance turned into a publishable t. They must
/// raise.
#[test]
fn ordinary_double_entity_constants_raise() {
    for (name, f) in [
        (
            "log land area",
            Box::new(|i: usize, _t: usize| ((i + 2) as f64 * 7.3).ln())
                as Box<dyn Fn(usize, usize) -> f64>,
        ),
        (
            "share in [0, 1]",
            Box::new(|i: usize, _t: usize| ((i * 37 % 97) as f64 + 0.5) / 98.0),
        ),
        (
            "gaussian entity constant",
            Box::new(|i: usize, _t: usize| {
                let mut rng = Lcg(i as u64 + 7);
                rng.gaussian()
            }),
        ),
    ] {
        let data = panel_with_entity_column(20, 8, &*f);
        assert_singular(panel_ols_fe(&data), name);
    }
}

/// The location-invariance witness from the audit: adding a constant to
/// an absorbed regressor changes nothing the fixed effects do not absorb,
/// so no shift may convert a refusal into a fit.
#[test]
fn absorbed_regressor_shift_invariance() {
    for shift in [0.0, 1.0, 10.0, 100.0] {
        let data = panel_with_entity_column(20, 8, |i, _| ((i + 2) as f64 * 7.3).ln() + shift);
        assert_singular(panel_ols_fe(&data), "shifted absorbed regressor");
    }
}

/// An exactly duplicated regressor is rank-deficient after demeaning and
/// must raise — the old guard accepted exact duplication while rejecting
/// a 1e-9 perturbation of it (anti-monotone).
#[test]
fn duplicated_regressor_raises() {
    let mut rng = Lcg(5);
    let x = Mat::from_fn(15, 6, |_, _| rng.gaussian());
    let outcome = Mat::from_fn(15, 6, |_, _| rng.gaussian());
    let data = PanelData::balanced(
        outcome,
        vec![("x".to_string(), x.clone()), ("x_copy".to_string(), x)],
    )
    .expect("balanced panel");
    assert_singular(panel_ols_fe(&data), "duplicated column");
}

/// A near-duplicate (1e-6 perturbation) is genuinely full rank — badly
/// conditioned, but identified — and must fit: the guard is monotone in
/// the perturbation, unlike its predecessor.
#[test]
fn near_duplicate_still_fits() {
    let mut rng = Lcg(6);
    let x = Mat::from_fn(15, 6, |_, _| rng.gaussian());
    let x2 = Mat::from_fn(15, 6, |i, t| {
        x[(i, t)] + 1e-6 * Lcg((i * 6 + t) as u64).gaussian()
    });
    let outcome = Mat::from_fn(15, 6, |_, _| rng.gaussian());
    let data = PanelData::balanced(
        outcome,
        vec![("x".to_string(), x), ("x_near".to_string(), x2)],
    )
    .expect("balanced panel");
    let fit = panel_ols_fe(&data).expect("near-duplicate design is identified");
    assert!(fit.params.iter().all(|v| v.is_finite()));
}

/// A regressor with genuine but small within variation is identified and
/// must keep fitting: the absorption test is a relative test, not a
/// variance floor.
#[test]
fn small_within_variation_still_fits() {
    // Level ~5e4 per entity with within variation ~0.5: the ratio of
    // demeaned to raw norm is ~1e-5 — far from the absorption tolerance.
    let mut rng = Lcg(9);
    let mut outcome = Mat::from_fn(20, 8, |_, _| rng.gaussian());
    let live = Mat::from_fn(20, 8, |_, _| rng.gaussian());
    let wobble = Mat::from_fn(20, 8, |i, t| {
        (i as f64 + 1.0) * 5.0e4 + 0.5 * Lcg((i * 8 + t) as u64 + 3).gaussian()
    });
    for i in 0..20 {
        for t in 0..8 {
            outcome[(i, t)] += 1.0e-1 * (wobble[(i, t)] - (i as f64 + 1.0) * 5.0e4);
        }
    }
    let data = PanelData::balanced(
        outcome,
        vec![("live".to_string(), live), ("wobble".to_string(), wobble)],
    )
    .expect("balanced panel");
    let fit = panel_ols_fe(&data).expect("small within variation is identified");
    assert!(fit.params.iter().all(|v| v.is_finite()));
}
