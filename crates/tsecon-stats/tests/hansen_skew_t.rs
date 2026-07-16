//! Verification of the Hansen (1994) skew-t.
//!
//! There is no SciPy reference for this density, so it is verified from
//! first principles: trapezoid integration over a wide grid must give total
//! mass 1, mean 0, and variance 1 (each to 1e-6 — Hansen's a, b constants
//! enforce standardization by construction), quantiles must round-trip
//! through the CDF, and `lambda = 0` must collapse to the standardized
//! Student t.

use tsecon_stats::{ContinuousDist, HansenSkewT, Standardized};

const PARAMS: [(f64, f64); 5] = [
    (6.0, -0.5),
    (8.0, 0.3),
    (12.0, 0.7),
    (5.5, 0.0),
    (30.0, -0.9),
];

/// Trapezoid rule for (mass, mean, variance) of the density on [-z, z].
///
/// The density has polynomial tails ~ |z|^-(eta+1); z = 500 keeps the
/// truncated variance mass below ~1e-7 for eta >= 5.5.
fn moments(d: &HansenSkewT, z_max: f64, h: f64) -> (f64, f64, f64) {
    let n = (2.0 * z_max / h).round() as usize;
    let (mut m0, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for i in 0..=n {
        let z = -z_max + h * i as f64;
        let f = d.pdf(z);
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        m0 += w * f;
        m1 += w * z * f;
        m2 += w * z * z * f;
    }
    (m0 * h, m1 * h, m2 * h)
}

#[test]
fn unit_mass_zero_mean_unit_variance() {
    for &(eta, lambda) in &PARAMS {
        let d = HansenSkewT::new(eta, lambda).unwrap();
        let (mass, mean, var) = moments(&d, 500.0, 0.0025);
        assert!(
            (mass - 1.0).abs() < 1e-6,
            "eta={eta} lambda={lambda}: mass {mass}"
        );
        assert!(mean.abs() < 1e-6, "eta={eta} lambda={lambda}: mean {mean}");
        assert!(
            (var - 1.0).abs() < 1e-6,
            "eta={eta} lambda={lambda}: var {var}"
        );
    }
}

#[test]
fn cdf_ppf_round_trip() {
    for &(eta, lambda) in &PARAMS {
        let d = HansenSkewT::new(eta, lambda).unwrap();
        for k in 1..=999 {
            let q = k as f64 / 1000.0;
            let z = d.ppf(q).unwrap();
            let back = d.cdf(z);
            assert!(
                (back - q).abs() < 1e-8,
                "eta={eta} lambda={lambda} q={q}: ppf {z}, cdf back {back}"
            );
        }
    }
}

/// lambda = 0 must equal the unit-variance (standardized) Student t.
#[test]
fn lambda_zero_is_standardized_t() {
    for &eta in &[4.5, 7.0, 15.0] {
        let skew = HansenSkewT::new(eta, 0.0).unwrap();
        let st = Standardized::student_t(eta).unwrap();
        let mut z = -8.0;
        while z <= 8.0 {
            let (p1, p2) = (skew.pdf(z), st.pdf(z));
            assert!(
                ((p1 - p2) / p2).abs() < 1e-12,
                "eta={eta} z={z}: skew pdf {p1} vs std-t pdf {p2}"
            );
            let (c1, c2) = (skew.cdf(z), st.cdf(z));
            assert!(
                ((c1 - c2) / c2).abs() < 1e-12,
                "eta={eta} z={z}: skew cdf {c1} vs std-t cdf {c2}"
            );
            let (s1, s2) = (skew.sf(z), st.sf(z));
            assert!(
                ((s1 - s2) / s2).abs() < 1e-12,
                "eta={eta} z={z}: skew sf {s1} vs std-t sf {s2}"
            );
            z += 0.25;
        }
        for &q in &[0.001, 0.05, 0.5, 0.95, 0.999] {
            let (q1, q2) = (skew.ppf(q).unwrap(), st.ppf(q).unwrap());
            if q2 == 0.0 {
                assert_eq!(q1, 0.0, "eta={eta} q={q}");
            } else {
                assert!(
                    ((q1 - q2) / q2).abs() < 1e-12,
                    "eta={eta} q={q}: {q1} vs {q2}"
                );
            }
        }
    }
}

/// The CDF is continuous across the density kink at -a/b, and basic shape
/// properties hold.
#[test]
fn shape_and_continuity() {
    for &(eta, lambda) in &PARAMS {
        let d = HansenSkewT::new(eta, lambda).unwrap();
        // The kink sits at F^{-1}((1-lambda)/2).
        let z0 = d.ppf(0.5 * (1.0 - lambda)).unwrap();
        let eps = 1e-9;
        let (left, right) = (d.cdf(z0 - eps), d.cdf(z0 + eps));
        assert!(
            (right - left).abs() < 1e-7,
            "eta={eta} lambda={lambda}: cdf jump at kink {left} vs {right}"
        );
        // cdf(-inf)=0, cdf(inf)=1 limits via far tails.
        assert!(d.cdf(-1e12) < 1e-10);
        assert!(d.cdf(1e12) > 1.0 - 1e-10);
        // With the mean pinned at 0, a left-skewed density (lambda < 0) has
        // its median to the right of the mean (mean < median), and vice
        // versa.
        if lambda < 0.0 {
            assert!(d.cdf(0.0) < 0.5, "lambda<0 should have median > 0");
        } else if lambda > 0.0 {
            assert!(d.cdf(0.0) > 0.5, "lambda>0 should have median < 0");
        }
    }
}

#[test]
fn parameter_validation() {
    assert!(HansenSkewT::new(2.0, 0.0).is_err()); // eta must exceed 2
    assert!(HansenSkewT::new(f64::NAN, 0.0).is_err());
    assert!(HansenSkewT::new(5.0, 1.0).is_err()); // |lambda| < 1
    assert!(HansenSkewT::new(5.0, -1.0).is_err());
    assert!(HansenSkewT::new(5.0, f64::NAN).is_err());
    let d = HansenSkewT::new(5.0, 0.2).unwrap();
    assert!(d.ppf(0.0).is_err());
    assert!(d.ppf(1.0).is_err());
    assert!(d.ppf(f64::NAN).is_err());
}
