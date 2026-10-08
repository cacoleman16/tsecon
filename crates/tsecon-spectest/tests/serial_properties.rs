//! Property and seeded Monte-Carlo tests for the serial-correlation tests.
//! `serial_golden.rs` pins the numbers; these check that the statistics
//! behave: the exact Durbin-Watson null distribution matches a simulated
//! one, both tests hold their size under iid errors and reject AR(1) errors,
//! and the statistics are invariant to what they must ignore.

use std::f64::consts::PI;

use tsecon_rng::Stream;
use tsecon_spectest::{breusch_godfrey_test, durbin_watson_test, DwAlternative};

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

/// A fixed design: constant, an AR(1) regressor and a trend.
fn design(stream: &mut Stream, n: usize) -> Vec<Vec<f64>> {
    vec![
        vec![1.0; n],
        ar1(stream, n, 0.6),
        (0..n).map(|t| t as f64 / n as f64).collect(),
    ]
}

fn regress(x: &[Vec<f64>], beta: &[f64], u: &[f64]) -> Vec<f64> {
    (0..u.len())
        .map(|t| beta.iter().zip(x).map(|(b, c)| b * c[t]).sum::<f64>() + u[t])
        .collect()
}

#[test]
fn exact_durbin_watson_pvalue_is_uniform_under_the_null() {
    // Normal iid errors, fixed X: the exact p-value is U(0, 1), so the
    // empirical CDF of the p-values at a, for each a, must be a within Monte
    // Carlo error (4 binomial standard errors).
    let mut s = Stream::new(2026_1008);
    let n = 30;
    let x = design(&mut s, n);
    let reps = 3000;
    let mut p: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let u = normals(&mut s, n);
        let y = regress(&x, &[1.0, 0.5, -0.3], &u);
        p.push(
            durbin_watson_test(&y, &x, DwAlternative::Positive)
                .expect("dw")
                .pvalue,
        );
    }
    for a in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95] {
        let rate = p.iter().filter(|&&v| v <= a).count() as f64 / reps as f64;
        let se = (a * (1.0 - a) / reps as f64).sqrt();
        assert!(
            (rate - a).abs() < 4.0 * se,
            "P(p <= {a}) = {rate:.4} (MC se {se:.4})"
        );
    }
}

#[test]
fn durbin_watson_rejects_ar1_errors_in_the_right_tail() {
    let mut s = Stream::new(77);
    let n = 80;
    let x = design(&mut s, n);
    let reps = 200;
    let (mut pos, mut neg) = (0, 0);
    for _ in 0..reps {
        let y_pos = regress(&x, &[1.0, 0.5, -0.3], &ar1(&mut s, n, 0.6));
        let y_neg = regress(&x, &[1.0, 0.5, -0.3], &ar1(&mut s, n, -0.6));
        let rp = durbin_watson_test(&y_pos, &x, DwAlternative::Positive).expect("dw");
        let rn = durbin_watson_test(&y_neg, &x, DwAlternative::Negative).expect("dw");
        assert!(rp.statistic < 2.0 || rp.pvalue > 0.05);
        pos += usize::from(rp.pvalue < 0.05);
        neg += usize::from(rn.pvalue < 0.05);
        // The two-sided p-value is twice the smaller one-sided p-value.
        let two = durbin_watson_test(&y_pos, &x, DwAlternative::TwoSided).expect("dw");
        let expect = (2.0 * rp.pvalue_positive.min(rp.pvalue_negative)).min(1.0);
        assert!((two.pvalue - expect).abs() < 1e-15);
    }
    assert!(
        pos as f64 / reps as f64 > 0.95,
        "power vs rho=0.6: {pos}/{reps}"
    );
    assert!(
        neg as f64 / reps as f64 > 0.95,
        "power vs rho=-0.6: {neg}/{reps}"
    );
}

#[test]
fn breusch_godfrey_holds_size_and_has_power() {
    let mut s = Stream::new(31_337);
    let n = 100;
    let x = design(&mut s, n);
    let reps = 1500;
    let (mut lm_null, mut f_null, mut lm_alt) = (0usize, 0usize, 0usize);
    for _ in 0..reps {
        let y0 = regress(&x, &[0.5, 1.0, 2.0], &normals(&mut s, n));
        let r0 = breusch_godfrey_test(&y0, &x, Some(4)).expect("bg");
        lm_null += usize::from(r0.pvalue < 0.05);
        f_null += usize::from(r0.f_pvalue < 0.05);
        let y1 = regress(&x, &[0.5, 1.0, 2.0], &ar1(&mut s, n, 0.4));
        let r1 = breusch_godfrey_test(&y1, &x, Some(4)).expect("bg");
        lm_alt += usize::from(r1.pvalue < 0.05);
    }
    let size_lm = lm_null as f64 / reps as f64;
    let size_f = f_null as f64 / reps as f64;
    let power = lm_alt as f64 / reps as f64;
    let se = (0.05 * 0.95 / reps as f64).sqrt();
    assert!((size_lm - 0.05).abs() < 4.0 * se, "LM size {size_lm:.4}");
    assert!((size_f - 0.05).abs() < 4.0 * se, "F size {size_f:.4}");
    assert!(power > 0.8, "power vs rho=0.4 at n=100, p=4: {power:.3}");
}

#[test]
fn statistics_are_invariant_to_scale_and_to_the_fitted_part() {
    // Both tests depend on y only through the OLS residuals, so rescaling y
    // and adding any X b leave them unchanged.
    let mut s = Stream::new(4242);
    let n = 60;
    let x = design(&mut s, n);
    let u = ar1(&mut s, n, 0.3);
    let y = regress(&x, &[1.0, -0.4, 0.8], &u);
    let y2: Vec<f64> = regress(&x, &[-3.0, 2.0, 0.5], &u)
        .iter()
        .map(|v| 7.5 * v)
        .collect();
    let y2b: Vec<f64> = y2
        .iter()
        .zip(regress(&x, &[3.0, 1.0, -1.0], &vec![0.0; n]))
        .map(|(a, b)| a + b)
        .collect();
    for nl in [None, Some(1), Some(5)] {
        let a = breusch_godfrey_test(&y, &x, nl).expect("bg");
        let b = breusch_godfrey_test(&y2b, &x, nl).expect("bg");
        assert!((a.statistic - b.statistic).abs() < 1e-9 * a.statistic.max(1.0));
        assert!((a.fstat - b.fstat).abs() < 1e-9 * a.fstat.max(1.0));
    }
    let a = durbin_watson_test(&y, &x, DwAlternative::Positive).expect("dw");
    let b = durbin_watson_test(&y2b, &x, DwAlternative::Positive).expect("dw");
    assert!((a.statistic - b.statistic).abs() < 1e-12);
    assert!((a.pvalue - b.pvalue).abs() < 1e-10);
    assert!((0.0..=4.0).contains(&a.statistic));
}

#[test]
fn breusch_godfrey_constant_handling() {
    let mut s = Stream::new(99);
    let n = 80;
    let x1 = ar1(&mut s, n, 0.5);
    let u = normals(&mut s, n);
    let y: Vec<f64> = (0..n).map(|t| 2.0 + 0.5 * x1[t] + u[t]).collect();
    // Explicit constant: nothing added, k_aux = k + p.
    let with_c = breusch_godfrey_test(&y, &[vec![1.0; n], x1.clone()], Some(3)).expect("bg");
    assert!(!with_c.constant_added);
    assert_eq!(with_c.f_df_den, n - 2 - 3);
    // No constant: one is added to the auxiliary design.
    let no_c = breusch_godfrey_test(&y, std::slice::from_ref(&x1), Some(3)).expect("bg");
    assert!(no_c.constant_added);
    assert_eq!(no_c.f_df_den, n - 1 - 1 - 3);
    // Implicit constant (complementary dummies) is detected: same numbers as
    // the explicit-constant parametrisation of the same column space.
    let d1: Vec<f64> = (0..n).map(|t| f64::from(u8::from(t < n / 2))).collect();
    let d2: Vec<f64> = d1.iter().map(|v| 1.0 - v).collect();
    let implicit = breusch_godfrey_test(&y, &[d1.clone(), d2, x1.clone()], Some(2)).expect("bg");
    let explicit = breusch_godfrey_test(&y, &[vec![1.0; n], d1, x1], Some(2)).expect("bg");
    assert!(!implicit.constant_added);
    assert!((implicit.statistic - explicit.statistic).abs() < 1e-9);
    assert_eq!(implicit.f_df_den, explicit.f_df_den);
}

#[test]
fn breusch_godfrey_order_runs_to_the_degrees_of_freedom_limit() {
    let mut s = Stream::new(5);
    let n = 25;
    let x = design(&mut s, n);
    let y = regress(&x, &[1.0, 0.5, 0.5], &normals(&mut s, n));
    // k = 3, so the longest admissible order is n - k - 1 = 21.
    let r = breusch_godfrey_test(&y, &x, Some(21)).expect("max order");
    assert_eq!(r.f_df_den, 1);
    assert!(r.pvalue.is_finite() && r.f_pvalue.is_finite());
    assert!(breusch_godfrey_test(&y, &x, Some(22)).is_err());
}
