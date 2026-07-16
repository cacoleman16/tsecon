//! Property/invariant tests across the distribution zoo and the special
//! functions: monotone CDFs, nonnegative densities, sf + cdf = 1, quantile
//! round-trips, symmetry relations, and closed-form cross-checks.

use tsecon_stats::special;
use tsecon_stats::{
    chi2_cdf, chi2_sf, ChiSquared, ContinuousDist, Ged, HansenSkewT, Normal, Standardized,
    StdNormal, StudentT,
};

fn zoo() -> Vec<(&'static str, Box<dyn ContinuousDist>)> {
    vec![
        ("StdNormal", Box::new(StdNormal)),
        ("Normal(0.3, 2.1)", Box::new(Normal::new(0.3, 2.1).unwrap())),
        ("StudentT(5)", Box::new(StudentT::new(5.0).unwrap())),
        ("StudentT(4.3)", Box::new(StudentT::new(4.3).unwrap())),
        ("StudentT(1)", Box::new(StudentT::new(1.0).unwrap())),
        ("Ged(1.5)", Box::new(Ged::new(1.5).unwrap())),
        ("Ged(0.8)", Box::new(Ged::new(0.8).unwrap())),
        (
            "HansenSkewT(8, 0.3)",
            Box::new(HansenSkewT::new(8.0, 0.3).unwrap()),
        ),
        (
            "Std<StudentT>(7.5)",
            Box::new(Standardized::student_t(7.5).unwrap()),
        ),
        ("Std<Ged>(1.3)", Box::new(Standardized::ged(1.3).unwrap())),
        ("ChiSquared(3.7)", Box::new(ChiSquared::new(3.7).unwrap())),
    ]
}

/// cdf nondecreasing and within [0,1]; pdf >= 0; sf + cdf = 1.
#[test]
fn cdf_monotone_pdf_nonnegative_sf_complement() {
    for (name, d) in zoo() {
        let mut prev = -f64::INFINITY;
        let mut x = -12.0;
        while x <= 12.0 {
            let (p, c, s) = (d.pdf(x), d.cdf(x), d.sf(x));
            assert!(p >= 0.0, "{name}: pdf({x}) = {p} < 0");
            assert!((0.0..=1.0).contains(&c), "{name}: cdf({x}) = {c}");
            assert!((0.0..=1.0).contains(&s), "{name}: sf({x}) = {s}");
            assert!(c >= prev, "{name}: cdf not monotone at {x}: {c} < {prev}");
            assert!(
                (c + s - 1.0).abs() <= 1e-12,
                "{name}: cdf+sf at {x} = {}",
                c + s
            );
            // ln_pdf consistent with pdf.
            let lp = d.ln_pdf(x);
            if p > 0.0 {
                assert!(
                    ((lp.exp() - p) / p).abs() <= 1e-12,
                    "{name}: exp(ln_pdf) != pdf at {x}"
                );
            } else {
                assert_eq!(lp, f64::NEG_INFINITY, "{name}: ln_pdf at pdf==0, x={x}");
            }
            prev = c;
            x += 0.0625;
        }
    }
}

/// cdf(ppf(u)) = u across the zoo.
#[test]
fn quantile_round_trip() {
    for (name, d) in zoo() {
        for k in 1..100 {
            let u = k as f64 / 100.0;
            let x = d.ppf(u).unwrap_or_else(|e| panic!("{name}: ppf({u}): {e}"));
            let back = d.cdf(x);
            assert!((back - u).abs() <= 1e-8, "{name}: cdf(ppf({u})) = {back}");
        }
        // Domain errors at the endpoints and NaN.
        assert!(d.ppf(0.0).is_err(), "{name}: ppf(0) must error");
        assert!(d.ppf(1.0).is_err(), "{name}: ppf(1) must error");
        assert!(d.ppf(-0.1).is_err(), "{name}: ppf(-0.1) must error");
        assert!(d.ppf(f64::NAN).is_err(), "{name}: ppf(NaN) must error");
    }
}

/// NaN propagates through the f64-returning methods.
#[test]
fn nan_propagation() {
    for (name, d) in zoo() {
        assert!(d.pdf(f64::NAN).is_nan(), "{name}: pdf(NaN)");
        assert!(d.ln_pdf(f64::NAN).is_nan(), "{name}: ln_pdf(NaN)");
        assert!(d.cdf(f64::NAN).is_nan(), "{name}: cdf(NaN)");
        assert!(d.sf(f64::NAN).is_nan(), "{name}: sf(NaN)");
    }
}

/// Symmetric distributions: pdf even, ppf odd around 1/2.
#[test]
fn symmetry() {
    let symmetric: Vec<(&str, Box<dyn ContinuousDist>)> = vec![
        ("StdNormal", Box::new(StdNormal)),
        ("StudentT(6.6)", Box::new(StudentT::new(6.6).unwrap())),
        ("Ged(1.7)", Box::new(Ged::new(1.7).unwrap())),
        ("Std<Ged>(0.9)", Box::new(Standardized::ged(0.9).unwrap())),
    ];
    for (name, d) in symmetric {
        let mut x = 0.0625_f64;
        while x <= 8.0 {
            let (l, r) = (d.pdf(-x), d.pdf(x));
            assert!(
                ((l - r) / r).abs() <= 1e-13,
                "{name}: pdf asymmetric at {x}"
            );
            let (cl, sr) = (d.cdf(-x), d.sf(x));
            assert!(
                ((cl - sr) / sr).abs() <= 1e-13,
                "{name}: cdf(-x) != sf(x) at {x}"
            );
            x += 0.375;
        }
        for &u in &[0.01, 0.2, 0.4] {
            let (l, r) = (d.ppf(u).unwrap(), d.ppf(1.0 - u).unwrap());
            assert!(
                ((l + r) / r).abs() <= 1e-9,
                "{name}: ppf({u}) != -ppf(1-{u}): {l} vs {r}"
            );
        }
        assert_eq!(d.ppf(0.5).unwrap(), 0.0, "{name}: median must be exactly 0");
    }
}

/// Standardized wrappers really have unit variance (trapezoid integration)
/// and mean zero.
#[test]
fn standardized_wrappers_unit_variance() {
    let cases: Vec<(&str, Box<dyn ContinuousDist>)> = vec![
        (
            "Std<StudentT>(7.5)",
            Box::new(Standardized::student_t(7.5).unwrap()),
        ),
        (
            "Std<StudentT>(4.2)",
            Box::new(Standardized::student_t(4.2).unwrap()),
        ),
        ("Std<Ged>(1.3)", Box::new(Standardized::ged(1.3).unwrap())),
        ("Std<Ged>(2.0)", Box::new(Standardized::ged(2.0).unwrap())),
    ];
    for (name, d) in cases {
        let (z_max, h) = (800.0, 0.005);
        let n = (2.0 * z_max / h) as usize;
        let (mut m0, mut m1, mut m2) = (0.0, 0.0, 0.0);
        for i in 0..=n {
            let z = -z_max + h * i as f64;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            let f = d.pdf(z);
            m0 += w * f;
            m1 += w * z * f;
            m2 += w * z * z * f;
        }
        let (mass, mean, var) = (m0 * h, m1 * h, m2 * h);
        assert!((mass - 1.0).abs() < 1e-6, "{name}: mass {mass}");
        assert!(mean.abs() < 1e-6, "{name}: mean {mean}");
        // Variance tolerance is looser than mass/mean because the t(4.2)
        // variance integrand decays like |z|^-3.2, so truncating at |z|=800
        // leaves O(1e-5) of variance mass outside the grid.
        assert!((var - 1.0).abs() < 2e-4, "{name}: var {var}");
    }
}

/// Closed-form cross-checks tying different code paths together.
#[test]
fn closed_form_cross_checks() {
    // StudentT(1) is Cauchy: F(x) = 1/2 + atan(x)/pi, f(x) = 1/(pi (1+x²)).
    let cauchy = StudentT::new(1.0).unwrap();
    let mut x = -6.0;
    while x <= 6.0 {
        let f_ref = 1.0 / (core::f64::consts::PI * (1.0 + x * x));
        let c_ref = 0.5 + x.atan() / core::f64::consts::PI;
        assert!(
            ((cauchy.pdf(x) - f_ref) / f_ref).abs() < 1e-12,
            "cauchy pdf {x}"
        );
        assert!(
            ((cauchy.cdf(x) - c_ref) / c_ref).abs() < 1e-12,
            "cauchy cdf {x}"
        );
        x += 0.5;
    }

    // Ged(1) is unit-scale Laplace: F(x) = e^x / 2 for x < 0.
    let laplace = Ged::new(1.0).unwrap();
    for &x in &[-9.0_f64, -4.0, -1.0, -0.2] {
        let c_ref = 0.5 * x.exp();
        assert!(
            ((laplace.cdf(x) - c_ref) / c_ref).abs() < 1e-13,
            "laplace cdf {x}"
        );
    }

    // Ged(2) is N(0, 1/2): F(x) = Phi(x sqrt(2)).
    let g2 = Ged::new(2.0).unwrap();
    for &x in &[-3.0, -0.7, 0.4, 2.5] {
        let c_ref = StdNormal.cdf(x * core::f64::consts::SQRT_2);
        assert!(((g2.cdf(x) - c_ref) / c_ref).abs() < 1e-12, "ged2 cdf {x}");
    }

    // ChiSquared(2) is Exp(1/2): S(x) = e^{-x/2}.
    let chi2_2 = ChiSquared::new(2.0).unwrap();
    for &x in &[0.3_f64, 1.0, 5.0, 20.0] {
        let s_ref = (-0.5 * x).exp();
        assert!(
            ((chi2_2.sf(x) - s_ref) / s_ref).abs() < 1e-13,
            "chi2(2) sf {x}"
        );
    }

    // ChiSquared(1): F(x) = erf(sqrt(x/2)).
    let chi2_1 = ChiSquared::new(1.0).unwrap();
    for &x in &[0.1_f64, 1.0, 4.0, 9.0] {
        let c_ref = special::erf((0.5 * x).sqrt());
        assert!(
            ((chi2_1.cdf(x) - c_ref) / c_ref).abs() < 1e-12,
            "chi2(1) cdf {x}"
        );
    }

    // Large-df StudentT approaches StdNormal (difference is O(1/df)).
    let t_big = StudentT::new(1e7).unwrap();
    for &x in &[-2.0, -0.5, 0.0, 1.0, 3.0] {
        assert!(
            (t_big.cdf(x) - StdNormal.cdf(x)).abs() < 1e-6,
            "t(1e7) vs normal at {x}"
        );
    }
}

/// Free chi-squared p-value helpers agree with the distribution object and
/// complement each other.
#[test]
fn chi2_helpers() {
    let d = ChiSquared::new(5.0).unwrap();
    let mut x = 0.25;
    while x <= 30.0 {
        let c = chi2_cdf(x, 5.0).unwrap();
        let s = chi2_sf(x, 5.0).unwrap();
        assert!((c + s - 1.0).abs() < 1e-13, "chi2 cdf+sf at {x}");
        assert_eq!(c, d.cdf(x), "chi2_cdf vs ChiSquared::cdf at {x}");
        assert_eq!(s, d.sf(x), "chi2_sf vs ChiSquared::sf at {x}");
        x += 0.75;
    }
    assert_eq!(chi2_cdf(-1.0, 3.0).unwrap(), 0.0);
    assert_eq!(chi2_sf(-1.0, 3.0).unwrap(), 1.0);
    assert!(chi2_cdf(1.0, 0.0).is_err());
    assert!(chi2_sf(1.0, -2.0).is_err());
    assert!(chi2_cdf(f64::NAN, 3.0).is_err());

    // Round trip through the quantile.
    for k in 1..40 {
        let u = k as f64 / 40.0;
        let x = d.ppf(u).unwrap();
        assert!((d.cdf(x) - u).abs() < 1e-10, "chi2 round trip {u}");
    }
}

/// Special-function identities.
#[test]
fn special_function_identities() {
    // erf odd; erf + erfc = 1; erfc(-x) = 2 - erfc(x).
    let mut x = -5.0;
    while x <= 5.0 {
        assert!(
            (special::erf(-x) + special::erf(x)).abs() < 1e-15,
            "erf odd {x}"
        );
        assert!(
            (special::erf(x) + special::erfc(x) - 1.0).abs() < 1e-14,
            "erf+erfc {x}"
        );
        assert!(
            (special::erfc(-x) + special::erfc(x) - 2.0).abs() < 1e-14,
            "erfc reflection {x}"
        );
        x += 0.171875;
    }

    // P + Q = 1.
    for &a in &[0.3, 1.0, 4.5, 40.0] {
        for &xx in &[0.01, 0.5, 3.0, 42.0] {
            let p = special::gamma_p(a, xx).unwrap();
            let q = special::gamma_q(a, xx).unwrap();
            assert!((p + q - 1.0).abs() < 1e-14, "P+Q at a={a}, x={xx}");
        }
    }

    // I_x(a,b) = 1 - I_{1-x}(b,a).
    for &(a, b) in &[(2.0, 3.0), (0.5, 0.5), (5.0, 1.5), (0.4, 7.0)] {
        for &xx in &[0.05, 0.3, 0.7, 0.95] {
            let lhs = special::beta_inc(a, b, xx).unwrap();
            let rhs = 1.0 - special::beta_inc(b, a, 1.0 - xx).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-14,
                "beta symmetry a={a} b={b} x={xx}"
            );
        }
    }

    // inv_norm_cdf round trip against the Cody-based normal CDF.
    let mut x = -5.0;
    while x <= 5.0 {
        if x != 0.0 {
            let p = StdNormal.cdf(x);
            let back = special::inv_norm_cdf(p).unwrap();
            assert!(
                ((back - x) / x).abs() < 1e-9,
                "inv_norm_cdf(cdf({x})) = {back}"
            );
        }
        x += 0.203125;
    }
}

/// Constructor validation across the zoo.
#[test]
fn constructor_validation() {
    assert!(Normal::new(f64::NAN, 1.0).is_err());
    assert!(Normal::new(0.0, 0.0).is_err());
    assert!(Normal::new(0.0, -1.0).is_err());
    assert!(Normal::new(0.0, f64::INFINITY).is_err());
    assert!(StudentT::new(0.0).is_err());
    assert!(StudentT::new(-3.0).is_err());
    assert!(StudentT::new(f64::NAN).is_err());
    assert!(Ged::new(0.0).is_err());
    assert!(Ged::new(f64::NAN).is_err());
    assert!(ChiSquared::new(0.0).is_err());
    assert!(Standardized::student_t(2.0).is_err()); // variance must exist
    assert!(Standardized::student_t(1.5).is_err());
    assert!(Standardized::ged(-1.0).is_err());
    assert!(Standardized::from_parts(StdNormal, 0.0).is_err());
    assert!(Standardized::from_parts(StdNormal, f64::NAN).is_err());
}
