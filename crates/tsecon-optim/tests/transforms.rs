//! The reparameterization toolkit: roundtrips, log-Jacobians against
//! numerical differentiation, the Monahan (1984) stationarity transform
//! (roundtrip to 1e-12, stationarity of forward outputs via an independent
//! polynomial-root check, tanh behavior at p = 1), and the
//! TransformedObjective wrapper end-to-end.

mod common;

use common::{det, max_ar_root_modulus, numerical_jacobian, Lcg};
use tsecon_optim::{
    minimize, Bounded, FnObjective, Method, NelderMeadOptions, OptimError, Ordered, Positive,
    StationaryAr, Transform, TransformedObjective, UnitInterval,
};

// ---------- Monahan / StationaryAr ----------

/// Roundtrip identity phi -> z -> phi to 1e-12 for random stationary AR(p)
/// coefficient vectors, p = 1..5 (stationary vectors constructed from
/// partial autocorrelations, which is exactly their parameterization —
/// Barndorff-Nielsen-Schou 1973).
#[test]
fn monahan_roundtrip_phi_to_1e12() {
    let t = StationaryAr;
    let mut rng = Lcg::new(20260716);
    for p in 1..=5usize {
        for _ in 0..100 {
            let z0: Vec<f64> = (0..p).map(|_| rng.uniform(-2.0, 2.0)).collect();
            let phi = t.forward_vec(&z0).unwrap();
            let z = t.inverse_vec(&phi).unwrap();
            let phi2 = t.forward_vec(&z).unwrap();
            for (a, b) in phi.iter().zip(&phi2) {
                assert!((a - b).abs() <= 1e-12, "p = {p}: {phi:?} vs {phi2:?}");
            }
            // And the working-space roundtrip.
            for (a, b) in z0.iter().zip(&z) {
                assert!((a - b).abs() <= 1e-10, "p = {p}: z {z0:?} vs {z:?}");
            }
        }
    }
}

/// Forward of arbitrary unconstrained vectors always yields stationary AR
/// polynomials — verified independently by computing the companion
/// polynomial roots (Durand-Kerner) and checking all moduli < 1.
#[test]
fn monahan_forward_always_stationary() {
    let t = StationaryAr;
    let mut rng = Lcg::new(987654321);
    for p in 1..=5usize {
        for _ in 0..60 {
            let z: Vec<f64> = (0..p).map(|_| rng.uniform(-2.5, 2.5)).collect();
            let phi = t.forward_vec(&z).unwrap();
            let rho = max_ar_root_modulus(&phi);
            assert!(
                rho < 1.0,
                "p = {p}, z = {z:?}: max root modulus {rho} (phi = {phi:?})"
            );
        }
    }
}

/// The AR(1) case is exactly phi_1 = tanh(z_1).
#[test]
fn monahan_ar1_is_tanh() {
    let t = StationaryAr;
    for z in [-3.0, -1.0, -0.2, 0.0, 0.4, 1.7, 5.0] {
        let phi = t.forward_vec(&[z]).unwrap();
        assert_eq!(phi[0], z.tanh());
        let back = t.inverse_vec(&phi).unwrap();
        assert!((back[0] - z).abs() <= 1e-12);
    }
}

/// Hand-computed AR(2) case pins the recursion convention:
/// phi_1 = r_1 (1 - r_2), phi_2 = r_2 with r_k = tanh(z_k)
/// (Monahan 1984, eq. 2).
#[test]
fn monahan_ar2_closed_form() {
    let t = StationaryAr;
    let (z1, z2): (f64, f64) = (0.7, -0.4);
    let (r1, r2) = (z1.tanh(), z2.tanh());
    let phi = t.forward_vec(&[z1, z2]).unwrap();
    assert!((phi[0] - r1 * (1.0 - r2)).abs() <= 1e-15);
    assert!((phi[1] - r2).abs() <= 1e-15);
}

/// Non-stationary coefficients are rejected by the inverse with the
/// offending lag.
#[test]
fn monahan_rejects_nonstationary() {
    let t = StationaryAr;
    // |phi| >= 1 in AR(1) is non-stationary.
    assert!(matches!(
        t.inverse_vec(&[1.05]),
        Err(OptimError::NotStationary { order: 1, .. })
    ));
    // AR(2) with phi_2 = 1.2: fails at lag 2.
    assert!(matches!(
        t.inverse_vec(&[0.0, 1.2]),
        Err(OptimError::NotStationary { order: 2, .. })
    ));
    // Unit root: phi(L) = 1 - L, on the boundary, must also be rejected.
    assert!(t.inverse_vec(&[1.0]).is_err());
}

/// log_jacobian matches log |det| of the numerically differentiated
/// forward map for p = 1..5.
#[test]
fn monahan_log_jacobian_vs_numeric() {
    let t = StationaryAr;
    let mut rng = Lcg::new(5555);
    for p in 1..=5usize {
        for _ in 0..20 {
            let z: Vec<f64> = (0..p).map(|_| rng.uniform(-1.5, 1.5)).collect();
            let lj = t.log_jacobian(&z).unwrap();
            let jac = numerical_jacobian(&t, &z);
            let dnum = det(jac).abs();
            assert!(
                (lj - dnum.ln()).abs() <= 1e-6,
                "p = {p}, z = {z:?}: analytic {lj}, numeric {}",
                dnum.ln()
            );
        }
    }
}

// ---------- Elementwise / ordering transforms ----------

fn check_roundtrip<T: Transform>(t: &T, z: &[f64], tol_z: f64) {
    let theta = t.forward_vec(z).unwrap();
    let z2 = t.inverse_vec(&theta).unwrap();
    for (a, b) in z.iter().zip(&z2) {
        assert!((a - b).abs() <= tol_z, "z {z:?} vs {z2:?}");
    }
    let theta2 = t.forward_vec(&z2).unwrap();
    for (a, b) in theta.iter().zip(&theta2) {
        assert!((a - b).abs() <= 1e-12, "theta {theta:?} vs {theta2:?}");
    }
}

fn check_log_jacobian<T: Transform>(t: &T, z: &[f64], tol: f64) {
    let lj = t.log_jacobian(z).unwrap();
    let dnum = det(numerical_jacobian(t, z)).abs();
    assert!(
        (lj - dnum.ln()).abs() <= tol,
        "z = {z:?}: analytic {lj}, numeric {}",
        dnum.ln()
    );
}

/// Positive: roundtrips and Jacobian.
#[test]
fn positive_roundtrip_and_jacobian() {
    let t = Positive;
    let mut rng = Lcg::new(11);
    for _ in 0..50 {
        let z: Vec<f64> = (0..4).map(|_| rng.uniform(-3.0, 3.0)).collect();
        check_roundtrip(&t, &z, 1e-12);
        check_log_jacobian(&t, &z, 1e-6);
        let theta = t.forward_vec(&z).unwrap();
        assert!(theta.iter().all(|&v| v > 0.0));
    }
    // theta -> z -> theta.
    let theta = [0.001, 2.5, 1e6];
    let z = t.inverse_vec(&theta).unwrap();
    let theta2 = t.forward_vec(&z).unwrap();
    for (a, b) in theta.iter().zip(&theta2) {
        assert!((a - b).abs() <= 1e-12 * a.abs());
    }
    assert!(matches!(
        t.inverse_vec(&[-0.5]),
        Err(OptimError::Domain { .. })
    ));
    assert!(t.inverse_vec(&[0.0]).is_err(), "boundary rejected");
}

/// Bounded(lo, hi): roundtrips, Jacobian, domain checks, bound respect.
#[test]
fn bounded_roundtrip_and_jacobian() {
    let t = Bounded::new(-0.5, 2.0).unwrap();
    let mut rng = Lcg::new(22);
    for _ in 0..50 {
        let z: Vec<f64> = (0..4).map(|_| rng.uniform(-4.0, 4.0)).collect();
        check_roundtrip(&t, &z, 1e-9);
        check_log_jacobian(&t, &z, 1e-6);
        let theta = t.forward_vec(&z).unwrap();
        assert!(theta.iter().all(|&v| v > -0.5 && v < 2.0));
    }
    assert!(matches!(
        t.inverse_vec(&[2.5]),
        Err(OptimError::Domain { .. })
    ));
    assert!(t.inverse_vec(&[-0.5]).is_err(), "boundary rejected");
    assert!(matches!(
        Bounded::new(1.0, 1.0),
        Err(OptimError::InvalidOption { .. })
    ));
    assert!(Bounded::new(2.0, 1.0).is_err());
    assert!(Bounded::new(f64::NEG_INFINITY, 0.0).is_err());
}

/// UnitInterval: roundtrips, Jacobian, saturation at the boundary.
#[test]
fn unit_interval_roundtrip_and_jacobian() {
    let t = UnitInterval;
    let mut rng = Lcg::new(33);
    for _ in 0..50 {
        let z: Vec<f64> = (0..3).map(|_| rng.uniform(-5.0, 5.0)).collect();
        check_roundtrip(&t, &z, 1e-9);
        check_log_jacobian(&t, &z, 1e-6);
    }
    // Far out in working space the map saturates to the boundary in f64;
    // the boundary is then (correctly) rejected by the inverse.
    let theta = t.forward_vec(&[800.0]).unwrap();
    assert_eq!(theta[0], 1.0);
    assert!(t.inverse_vec(&theta).is_err());
}

/// Ordered: strictly increasing output, roundtrips, triangular Jacobian.
#[test]
fn ordered_roundtrip_and_jacobian() {
    let t = Ordered;
    let mut rng = Lcg::new(44);
    for _ in 0..50 {
        let z: Vec<f64> = (0..5).map(|_| rng.uniform(-2.0, 2.0)).collect();
        let theta = t.forward_vec(&z).unwrap();
        for w in theta.windows(2) {
            assert!(w[1] > w[0], "not increasing: {theta:?}");
        }
        check_roundtrip(&t, &z, 1e-10);
        check_log_jacobian(&t, &z, 1e-5);
    }
    assert!(matches!(
        t.inverse_vec(&[0.0, 1.0, 1.0]),
        Err(OptimError::Domain { .. })
    ));
    // Explicit closed form: log|det J| = sum of z[1..].
    let z = [0.3, -1.2, 0.8];
    assert!((t.log_jacobian(&z).unwrap() - (-1.2 + 0.8)).abs() <= 1e-15);
}

/// Degenerate empty blocks are valid no-ops with zero log-Jacobian.
#[test]
fn empty_slices_are_valid() {
    for t in [
        &Positive as &dyn Transform,
        &UnitInterval,
        &Ordered,
        &StationaryAr,
    ] {
        assert_eq!(t.forward_vec(&[]).unwrap(), Vec::<f64>::new());
        assert_eq!(t.inverse_vec(&[]).unwrap(), Vec::<f64>::new());
        assert_eq!(t.log_jacobian(&[]).unwrap(), 0.0);
    }
}

/// Length mismatches and non-finite inputs error.
#[test]
fn transform_input_errors() {
    let t = Positive;
    let mut out = vec![0.0; 3];
    assert!(matches!(
        t.forward(&[1.0, 2.0], &mut out),
        Err(OptimError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        t.forward_vec(&[f64::NAN]),
        Err(OptimError::NonFinite { .. })
    ));
    assert!(matches!(
        StationaryAr.log_jacobian(&[f64::INFINITY]),
        Err(OptimError::NonFinite { .. })
    ));
}

// ---------- TransformedObjective ----------

/// Constrained MLE-style use: minimize (theta - 3)^2 over theta > 0 by
/// optimizing in working space; the solution maps back to theta = 3.
#[test]
fn transformed_objective_positive_mle() {
    let inner = FnObjective::new(|t: &[f64]| (t[0] - 3.0) * (t[0] - 3.0));
    let mut obj = TransformedObjective::new(inner, Positive);
    let opts = NelderMeadOptions {
        x_tol: 1e-10,
        f_tol: 1e-12,
        max_iter: Some(500),
        max_fevals: Some(1000),
        ..NelderMeadOptions::default()
    };
    let res = minimize(&mut obj, &[0.0], &Method::NelderMead(opts)).unwrap();
    assert!(res.converged);
    let theta = obj.constrained(&res.x).unwrap();
    assert!((theta[0] - 3.0).abs() <= 1e-5);
}

/// Bayesian-MAP use: for the density p(theta) proportional to
/// theta^k exp(-theta) (mode k), the transformed density in z = log(theta)
/// is proportional to theta^(k+1) exp(-theta) (mode k + 1) — the
/// log-Jacobian term shifts the mode, and forgetting it is the classic
/// silent bug the toolkit exists to prevent.
#[test]
fn transformed_objective_log_jacobian_shifts_mode() {
    let k = 3.0;
    let nll = move |t: &[f64]| t[0] - k * t[0].ln();
    let opts = NelderMeadOptions {
        x_tol: 1e-10,
        f_tol: 1e-12,
        max_iter: Some(500),
        max_fevals: Some(1000),
        ..NelderMeadOptions::default()
    };

    // Without the Jacobian: plain reparameterized minimum at theta = k.
    let mut plain = TransformedObjective::new(FnObjective::new(nll), Positive);
    let res = minimize(&mut plain, &[0.0], &Method::NelderMead(opts)).unwrap();
    let theta = plain.constrained(&res.x).unwrap();
    assert!((theta[0] - k).abs() <= 1e-4, "theta = {}", theta[0]);

    // With the Jacobian: the z-space density mode is at theta = k + 1.
    let mut with_j = TransformedObjective::with_log_jacobian(FnObjective::new(nll), Positive);
    let res = minimize(&mut with_j, &[0.0], &Method::NelderMead(opts)).unwrap();
    let theta = with_j.constrained(&res.x).unwrap();
    assert!((theta[0] - (k + 1.0)).abs() <= 1e-4, "theta = {}", theta[0]);
}

/// Stationarity-constrained AR(2) objective: the unconstrained optimizer
/// recovers a stationary target from any start, without ever leaving the
/// stationary region.
#[test]
fn transformed_objective_stationary_ar() {
    let target = [0.5, -0.3]; // stationary AR(2)
    let inner = FnObjective::new(move |phi: &[f64]| {
        (phi[0] - target[0]).powi(2) + (phi[1] - target[1]).powi(2)
    });
    let mut obj = TransformedObjective::new(inner, StationaryAr);
    let res = minimize(&mut obj, &[2.0, -2.0], &Method::bfgs()).unwrap();
    let phi = obj.constrained(&res.x).unwrap();
    assert!((phi[0] - 0.5).abs() <= 1e-6 && (phi[1] + 0.3).abs() <= 1e-6, "phi = {phi:?}");
    assert!(max_ar_root_modulus(&phi) < 1.0);
}
