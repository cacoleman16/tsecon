//! The distribution function of a quadratic form in independent standard
//! normals, by Imhof's (1961) numerical inversion of its characteristic
//! function — the engine of the exact Durbin-Watson p-value.
//!
//! For `Q = sum_i lam_i z_i^2` with `z_i` iid `N(0, 1)`, Imhof (1961, eq.
//! 3.2) gives, at `x = 0`,
//!
//! ```text
//! P(Q <= 0) = 1/2 - (1/pi) int_0^inf sin(theta(u)) / (u rho(u)) du,
//! theta(u)  = (1/2) sum_i arctan(lam_i u),
//! rho(u)    = prod_i (1 + lam_i^2 u^2)^{1/4}.
//! ```
//!
//! The integrand tends to `(1/2) sum_i lam_i` as `u -> 0` and is bounded by
//! `1 / (u rho(u))`, so the integral over `[U, inf)` is at most
//! `2 / (m' U^{m'/2} prod_{i in S} |lam_i|^{1/2})` for any subset `S` of
//! `m'` coefficients (Imhof 1961, eq. 3.3). The upper limit `U` is chosen
//! so that bound is below `1e-16` in p-value units, using the coefficients
//! of at least `1e-3` times the largest magnitude, and `[0, U]` is
//! integrated by a globally adaptive Gauss-Kronrod (7, 15) rule (QUADPACK
//! `qk15`/`qag` error estimate) on a geometric initial partition scaled to
//! the integrand's width `1 / sqrt(sum lam_i^2)`.

use crate::error::SpecTestError;

/// Kronrod abscissae of the 15-point rule on `[-1, 1]` (non-negative half,
/// descending; QUADPACK `qk15`). The odd-indexed entries are the 7-point
/// Gauss abscissae.
const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];

/// Weights of the 15-point Kronrod rule, matching [`XGK`].
const WGK: [f64; 8] = [
    0.022_935_322_010_529_225,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_83,
];

/// Weights of the embedded 7-point Gauss rule (abscissae `XGK[1]`, `XGK[3]`,
/// `XGK[5]`, `XGK[7]`).
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// Absolute tolerance on the integral (it is divided by `pi` into the
/// probability).
const EPSABS: f64 = 1e-14;
/// Largest accepted final error estimate before the evaluation is refused.
const ACCEPT_ERR: f64 = 1e-9;
/// Subdivision budget of the adaptive rule.
const MAX_INTERVALS: usize = 4000;

/// One Gauss-Kronrod (7, 15) panel on `[a, b]`: `(integral, error,
/// integral of |f|)` with the QUADPACK error estimate.
fn gk15<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64, f64) {
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);
    let fc = f(centr);
    let mut resg = fc * WG[3];
    let mut resk = fc * WGK[7];
    let mut resabs = resk.abs();
    let mut fv1 = [0.0_f64; 7];
    let mut fv2 = [0.0_f64; 7];
    for j in 0..7 {
        let absc = hlgth * XGK[j];
        let f1 = f(centr - absc);
        let f2 = f(centr + absc);
        fv1[j] = f1;
        fv2[j] = f2;
        resk += WGK[j] * (f1 + f2);
        resabs += WGK[j] * (f1.abs() + f2.abs());
        if j % 2 == 1 {
            resg += WG[j / 2] * (f1 + f2);
        }
    }
    let reskh = 0.5 * resk;
    let mut resasc = WGK[7] * (fc - reskh).abs();
    for j in 0..7 {
        resasc += WGK[j] * ((fv1[j] - reskh).abs() + (fv2[j] - reskh).abs());
    }
    let result = resk * hlgth;
    resabs *= hlgth.abs();
    resasc *= hlgth.abs();
    let mut abserr = ((resk - resg) * hlgth).abs();
    if resasc != 0.0 && abserr != 0.0 {
        abserr = resasc * (200.0 * abserr / resasc).powf(1.5).min(1.0);
    }
    if resabs > f64::MIN_POSITIVE / (50.0 * f64::EPSILON) {
        abserr = abserr.max(50.0 * f64::EPSILON * resabs);
    }
    (result, abserr, resabs)
}

/// Globally adaptive integration of `f` over the panels `edges[i]..edges[i+1]`:
/// bisect the panel with the largest error estimate until the total estimate
/// is below [`EPSABS`] — or below four times the round-off floor that the
/// QUADPACK estimate itself imposes (`50 eps` per unit of `int |f|`), past
/// which bisection cannot certify more digits — or the budget is spent.
/// Returns `(integral, error)`.
fn adaptive<F: Fn(f64) -> f64>(f: &F, edges: &[f64]) -> (f64, f64) {
    let mut panels: Vec<(f64, f64, f64, f64, f64)> = edges
        .windows(2)
        .map(|w| {
            let (r, e, ra) = gk15(f, w[0], w[1]);
            (w[0], w[1], r, e, ra)
        })
        .collect();
    loop {
        let total_err: f64 = panels.iter().map(|p| p.3).sum();
        let total_abs: f64 = panels.iter().map(|p| p.4).sum();
        let target = EPSABS.max(200.0 * f64::EPSILON * total_abs);
        if total_err <= target || panels.len() >= MAX_INTERVALS {
            let total: f64 = panels.iter().map(|p| p.2).sum();
            return (total, total_err);
        }
        let (worst, _) =
            panels.iter().enumerate().fold(
                (0, -1.0),
                |acc, (i, p)| {
                    if p.3 > acc.1 {
                        (i, p.3)
                    } else {
                        acc
                    }
                },
            );
        let (a, b, _, _, _) = panels[worst];
        let mid = 0.5 * (a + b);
        if !(mid > a && mid < b) {
            // The panel cannot be split further in floating point.
            let total: f64 = panels.iter().map(|p| p.2).sum();
            return (total, total_err);
        }
        let (r1, e1, a1) = gk15(f, a, mid);
        let (r2, e2, a2) = gk15(f, mid, b);
        panels[worst] = (a, mid, r1, e1, a1);
        panels.push((mid, b, r2, e2, a2));
    }
}

/// `P(sum_i lam_i z_i^2 <= 0)` for iid standard normal `z_i`, by Imhof's
/// formula. Coefficients may have either sign; exact zeros contribute
/// nothing. The result is clamped to `[0, 1]`; its absolute accuracy is
/// about `1e-13` (an integral of order one evaluated to `EPSABS`), so a
/// probability below that is numerically zero.
///
/// # Errors
///
/// [`SpecTestError::NumericalBreakdown`] if a coefficient is not finite or
/// the adaptive quadrature cannot bring its error estimate below `1e-9`.
pub(crate) fn prob_le_zero(lam: &[f64]) -> Result<f64, SpecTestError> {
    if lam.iter().any(|v| !v.is_finite()) {
        return Err(SpecTestError::NumericalBreakdown {
            what: "Imhof quadratic-form coefficients (non-finite)",
        });
    }
    let max_abs = lam.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        // Q is identically zero: P(Q <= 0) = 1.
        return Ok(1.0);
    }
    // One-signed forms are degenerate cases of the formula; answer them
    // exactly (Q >= 0 a.s. with P(Q = 0) = 0, or Q <= 0 a.s.).
    if lam.iter().all(|&v| v >= 0.0) {
        return Ok(0.0);
    }
    if lam.iter().all(|&v| v <= 0.0) {
        return Ok(1.0);
    }

    // Truncation point from Imhof's tail bound on the dominant coefficients.
    let cut = 1e-3 * max_abs;
    let (m_big, log_prod) = lam
        .iter()
        .filter(|v| v.abs() >= cut)
        .fold((0usize, 0.0_f64), |(m, s), v| (m + 1, s + v.abs().ln()));
    let half_m = 0.5 * m_big as f64;
    let target = 1e-16_f64;
    let log_u = (-(std::f64::consts::PI * half_m * target).ln() - 0.5 * log_prod) / half_m;
    let upper = log_u.exp().min(1e300);

    // Characteristic width of the integrand near zero.
    let width = 1.0 / lam.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut edges = vec![0.0];
    let mut e = 0.125 * width;
    while e < upper {
        edges.push(e);
        e *= 2.0;
    }
    edges.push(upper.max(0.125 * width * 2.0));
    edges.dedup();

    let half_sum = 0.5 * lam.iter().sum::<f64>();
    let f = |u: f64| -> f64 {
        if u == 0.0 {
            return half_sum;
        }
        let mut theta = 0.0;
        let mut log_rho = 0.0;
        for &l in lam {
            let lu = l * u;
            theta += lu.atan();
            log_rho += (lu * lu).ln_1p();
        }
        (0.5 * theta).sin() * (-0.25 * log_rho).exp() / u
    };
    let (integral, err) = adaptive(&f, &edges);
    if !integral.is_finite() || err > ACCEPT_ERR {
        return Err(SpecTestError::NumericalBreakdown {
            what: "Imhof integral for the exact Durbin-Watson p-value",
        });
    }
    Ok((0.5 - integral / std::f64::consts::PI).clamp(0.0, 1.0))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn kronrod_rule_is_exact_on_polynomials() {
        // K15 integrates x^{2j} over [-1, 1] exactly up to degree 22, G7 up
        // to degree 13.
        for j in 0..=11 {
            let f = |x: f64| x.powi(2 * j);
            let (r, _, _) = gk15(&f, -1.0, 1.0);
            let exact = 2.0 / (2 * j + 1) as f64;
            assert!(
                (r - exact).abs() < 1e-14,
                "degree {}: {r} vs {exact}",
                2 * j
            );
        }
        let s: f64 = 2.0 * WGK[..7].iter().sum::<f64>() + WGK[7];
        assert!((s - 2.0).abs() < 1e-15);
        let g: f64 = 2.0 * WG[..3].iter().sum::<f64>() + WG[3];
        assert!((g - 2.0).abs() < 1e-15);
    }

    #[test]
    fn two_coefficient_closed_form() {
        // For Q = a z1^2 - b z2^2 (a, b > 0): P(Q <= 0) = P(|z1/z2| <= sqrt(b/a))
        // = (2/pi) arctan(sqrt(b/a)).
        for &(a, b) in &[(1.0, 1.0), (2.5, 0.4), (0.3, 3.7), (1.0, 1e-3)] {
            let p = prob_le_zero(&[a, -b]).expect("imhof");
            let exact = 2.0 / std::f64::consts::PI * (b / a).sqrt().atan();
            assert!((p - exact).abs() < 1e-12, "a={a} b={b}: {p} vs {exact}");
        }
    }

    #[test]
    fn symmetric_form_is_one_half_and_signs_are_exact() {
        let p = prob_le_zero(&[1.0, -1.0, 2.0, -2.0, 0.5, -0.5]).expect("imhof");
        assert!((p - 0.5).abs() < 1e-13, "{p}");
        assert_eq!(prob_le_zero(&[1.0, 2.0, 0.0]).expect("pos"), 0.0);
        assert_eq!(prob_le_zero(&[-1.0, -2.0]).expect("neg"), 1.0);
        assert_eq!(prob_le_zero(&[0.0, 0.0]).expect("zero"), 1.0);
        assert!(prob_le_zero(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn chi_square_difference_matches_closed_form() {
        // Q = chi2_2 - c chi2_2' with two unit and two -c coefficients:
        // chi2_2 = 2 Exp(1), so P(E1 <= c E2) = c / (1 + c).
        for &c in &[0.25_f64, 1.0, 3.0] {
            let p = prob_le_zero(&[1.0, 1.0, -c, -c]).expect("imhof");
            assert!((p - c / (1.0 + c)).abs() < 1e-12, "c={c}: {p}");
        }
    }
}
