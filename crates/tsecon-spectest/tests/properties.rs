//! Property tests: relationships and invariants that hold for the tests
//! independently of any external reference. These do not pin exact numbers
//! (that is `golden.rs`); they check that the statistics behave — reject strong
//! violations, respect algebraic identities, and stay in valid ranges.

use std::f64::consts::PI;

use tsecon_rng::Stream;
use tsecon_spectest::{breusch_pagan_test, chow_test, cusum_test, reset_test, white_test};

/// Standard-normal draws via Box-Muller on the reproducible uniform stream.
fn normals(stream: &mut Stream, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let u1 = 1.0 - stream.uniform_f64(); // in (0, 1]
            let u2 = stream.uniform_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        })
        .collect()
}

fn ar1(stream: &mut Stream, n: usize, rho: f64) -> Vec<f64> {
    let e = normals(stream, n);
    let mut x = vec![0.0; n];
    x[0] = e[0] / (1.0 - rho * rho).sqrt();
    for t in 1..n {
        x[t] = rho * x[t - 1] + e[t];
    }
    x
}

fn in_unit(p: f64) {
    assert!(
        p.is_finite() && (0.0..=1.0).contains(&p),
        "p-value out of range: {p}"
    );
}

#[test]
fn het_tests_reject_strong_heteroskedasticity() {
    let mut s = Stream::new(2026_0717);
    let n = 200;
    let x1 = ar1(&mut s, n, 0.5);
    let x2 = normals(&mut s, n);
    let cst = vec![1.0; n];
    let e = normals(&mut s, n);
    // Error standard deviation rises sharply with x1: strong heteroskedasticity.
    let y: Vec<f64> = (0..n)
        .map(|t| 1.0 + 0.7 * x1[t] - 0.4 * x2[t] + (1.2 * x1[t]).exp() * e[t])
        .collect();
    let x = vec![cst, x1, x2];

    let bp = breusch_pagan_test(&y, &x).expect("bp");
    let white = white_test(&y, &x).expect("white");
    in_unit(bp.pvalue);
    in_unit(white.pvalue);
    assert!(
        bp.pvalue < 1e-2,
        "BP should reject strong het, p={}",
        bp.pvalue
    );
    assert!(
        white.pvalue < 1e-2,
        "White should reject strong het, p={}",
        white.pvalue
    );
    // F-form and LM-form agree on degrees of freedom.
    assert_eq!(bp.df, bp.f_df_num);
    assert_eq!(bp.f_df_den, n - x.len());
}

#[test]
fn het_tests_stay_valid_under_homoskedasticity() {
    let mut s = Stream::new(777);
    let n = 200;
    let x1 = ar1(&mut s, n, 0.4);
    let x2 = normals(&mut s, n);
    let cst = vec![1.0; n];
    let e = normals(&mut s, n);
    let y: Vec<f64> = (0..n)
        .map(|t| 1.0 + 0.5 * x1[t] - 0.3 * x2[t] + e[t])
        .collect();
    let x = vec![cst, x1, x2];

    let bp = breusch_pagan_test(&y, &x).expect("bp");
    let white = white_test(&y, &x).expect("white");
    in_unit(bp.pvalue);
    in_unit(white.pvalue);
    assert!(bp.statistic >= 0.0 && white.statistic >= 0.0);
    // White's auxiliary design has k(k+1)/2 - 1 = 5 slope regressors for k = 3.
    assert_eq!(white.df, 5);
    assert_eq!(bp.df, 2);
}

#[test]
fn reset_rejects_omitted_nonlinearity() {
    let mut s = Stream::new(31415);
    let n = 200;
    let x1 = ar1(&mut s, n, 0.4);
    let x2 = normals(&mut s, n);
    let cst = vec![1.0; n];
    let e = normals(&mut s, n);
    // True model is quadratic in x1 but we fit only the linear design.
    let y: Vec<f64> = (0..n)
        .map(|t| 1.0 + x1[t] + 0.7 * x1[t] * x1[t] - 0.4 * x2[t] + 0.3 * e[t])
        .collect();
    let x = vec![cst, x1, x2];

    let reset = reset_test(&y, &x, 3).expect("reset");
    in_unit(reset.pvalue);
    assert_eq!(reset.df_num, 2);
    assert_eq!(reset.df_den, n - (x.len() + 2));
    assert!(
        reset.pvalue < 1e-2,
        "RESET should reject, p={}",
        reset.pvalue
    );
}

#[test]
fn chow_rejects_strong_break_and_is_split_symmetric_in_range() {
    let mut s = Stream::new(9001);
    let n = 160;
    let split = 80;
    let x1 = ar1(&mut s, n, 0.5);
    let x2 = normals(&mut s, n);
    let cst = vec![1.0; n];
    let e = normals(&mut s, n);
    let mut y = vec![0.0; n];
    for t in 0..n {
        let (a, b, c) = if t < split {
            (1.0, 0.6, -0.3)
        } else {
            (3.0, -0.7, 0.9)
        };
        y[t] = a + b * x1[t] + c * x2[t] + 0.3 * e[t];
    }
    let x = vec![cst, x1, x2];

    let chow = chow_test(&y, &x, split).expect("chow");
    in_unit(chow.pvalue);
    assert_eq!(chow.df_num, 3);
    assert_eq!(chow.df_den, n - 2 * 3);
    assert!(
        chow.pvalue < 1e-3,
        "Chow should reject strong break, p={}",
        chow.pvalue
    );
    // The pooled fit cannot explain more than the regime fits combined.
    assert!(chow.ssr_pooled >= chow.ssr1 + chow.ssr2 - 1e-9);
}

#[test]
fn cusum_identity_and_bounds() {
    let mut s = Stream::new(4242);
    let n = 120;
    let x1 = ar1(&mut s, n, 0.5);
    let cst = vec![1.0; n];
    let e = normals(&mut s, n);
    let y: Vec<f64> = (0..n).map(|t| 0.5 + 0.8 * x1[t] + e[t]).collect();
    let x = vec![cst, x1];
    let k = x.len();

    let cu = cusum_test(&y, &x).expect("cusum");
    assert_eq!(cu.recursive_residuals.len(), n - k);
    assert_eq!(cu.path.len(), n - k);

    // Algebraic identity: sum of squared recursive residuals == SSR == (n-k) sigma^2.
    let sw2: f64 = cu.recursive_residuals.iter().map(|w| w * w).sum();
    let ssr = (n - k) as f64 * cu.sigma * cu.sigma;
    assert!(
        (sw2 - ssr).abs() < 1e-8,
        "sum w^2 = {sw2}, (n-k) sigma^2 = {ssr}"
    );

    // Bounds are symmetric and strictly widening toward the sample end.
    for i in 0..cu.bound_upper.len() {
        assert!((cu.bound_upper[i] + cu.bound_lower[i]).abs() < 1e-12);
        if i > 0 {
            assert!(cu.bound_upper[i] > cu.bound_upper[i - 1]);
        }
    }
    assert_eq!(cu.a, tsecon_spectest::CUSUM_A_5PCT);
}

#[test]
fn cusum_flags_a_large_mean_shift() {
    let mut s = Stream::new(1234);
    let n = 160;
    let x1 = ar1(&mut s, n, 0.5);
    let cst = vec![1.0; n];
    let e = normals(&mut s, n);
    // A large intercept shift halfway through drives the CUSUM out of bounds.
    let y: Vec<f64> = (0..n)
        .map(|t| {
            let level = if t < n / 2 { 0.0 } else { 6.0 };
            level + 0.5 * x1[t] + 0.3 * e[t]
        })
        .collect();
    let x = vec![cst, x1];

    let cu = cusum_test(&y, &x).expect("cusum");
    assert!(cu.rejects_5pct(), "CUSUM should flag a large mean shift");
}
