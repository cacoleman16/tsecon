//! Scalar special functions.
//!
//! This module provides the special functions that the distribution zoo and
//! the diagnostics crate build on: the log-gamma function, the error
//! function pair, the regularized incomplete gamma and beta functions, and
//! the inverse functions needed for quantiles (inverse standard normal CDF,
//! inverse regularized incomplete beta/gamma).
//!
//! Design conventions:
//!
//! * Functions that are total on the reals (`ln_gamma`, `erf`, `erfc`)
//!   return `f64` directly, propagating `NaN` for `NaN` inputs and returning
//!   `+inf` at poles.
//! * Domain-restricted and iterative functions return
//!   `Result<f64, StatsError>`.

// The coefficient tables below are transcribed verbatim from the published
// references (Cody 1969/1993, Wichura 1988, Lanczos/Godfrey); some literals
// carry more digits than an f64 can represent, which is intentional so the
// tables remain diff-able against the sources.
#![allow(clippy::excessive_precision)]

use crate::error::StatsError;
use core::f64::consts::PI;

/// Smallest representable number scaled so that `1/FPMIN` does not overflow;
/// used as the underflow guard in modified-Lentz continued fractions
/// (Press et al. 2007, §5.2).
const FPMIN: f64 = f64::MIN_POSITIVE / f64::EPSILON;

// ---------------------------------------------------------------------------
// Log-gamma (Lanczos)
// ---------------------------------------------------------------------------

/// Lanczos parameter `g = 7` used with the 9-term coefficient set below
/// (Godfrey's coefficients, as used by the GNU Scientific Library and Boost).
const LANCZOS_G: f64 = 7.0;

/// 9-term Lanczos coefficients for `g = 7`.
const LANCZOS_COEF: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

/// `ln(2*pi)`.
const LN_2PI: f64 = 1.8378770664093454835606594728112353;

/// Natural logarithm of the absolute value of the gamma function,
/// `ln |Γ(x)|`.
///
/// Computed with the Lanczos approximation (Lanczos 1964) with `g = 7` and
/// the 9-term coefficient set due to Godfrey (the set used by GSL and Boost):
///
/// ```text
/// Γ(z) = sqrt(2π) · t^(z-1/2) · e^(-t) · A_g(z-1),   t = z + g - 1/2,
/// A_g(w) = c0 + Σ_{k=1..8} c_k / (w + k)
/// ```
///
/// For `x < 0.5` the reflection formula `Γ(x)Γ(1-x) = π / sin(πx)` is used so
/// the Lanczos series is only ever evaluated for arguments ≥ 0.5, where its
/// relative error is below `1e-14`. Observed accuracy against high-precision
/// references is ≲ 1e-14 relative over the positive axis.
///
/// Special values: `ln_gamma(1) = ln_gamma(2) = 0` exactly; poles at
/// `x = 0, -1, -2, ...` return `+inf`; `NaN` propagates.
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 1.0 || x == 2.0 {
        // Exact zeros of ln Γ; short-circuit so golden tests at integer
        // arguments are exact.
        return 0.0;
    }
    if x < 0.5 {
        // Poles of Γ at the non-positive integers.
        if x <= 0.0 && x == x.floor() {
            return f64::INFINITY;
        }
        // Reflection: ln|Γ(x)| = ln π - ln|sin(πx)| - ln|Γ(1-x)|.
        let s = (PI * x).sin();
        return PI.ln() - s.abs().ln() - ln_gamma(1.0 - x);
    }
    let w = x - 1.0;
    let mut a = LANCZOS_COEF[0];
    for (k, &c) in LANCZOS_COEF.iter().enumerate().skip(1) {
        a += c / (w + k as f64);
    }
    let t = w + LANCZOS_G + 0.5;
    0.5 * LN_2PI + (w + 0.5) * t.ln() - t + a.ln()
}

// ---------------------------------------------------------------------------
// Error function (Cody's rational approximations)
// ---------------------------------------------------------------------------

// Coefficient tables from W. J. Cody, "Rational Chebyshev approximation for
// the error function", Math. Comp. 23 (1969) 631-637, as implemented in the
// SPECFUN routine CALERF (Cody 1993, TOMS 715).

/// erf numerator coefficients on `|x| <= 0.46875`.
const ERF_A: [f64; 5] = [
    3.16112374387056560e0,
    1.13864154151050156e2,
    3.77485237685302021e2,
    3.20937758913846947e3,
    1.85777706184603153e-1,
];
/// erf denominator coefficients on `|x| <= 0.46875`.
const ERF_B: [f64; 4] = [
    2.36012909523441209e1,
    2.44024637934444173e2,
    1.28261652607737228e3,
    2.84423683343917062e3,
];
/// erfc numerator coefficients on `0.46875 < x <= 4`.
const ERFC_C: [f64; 9] = [
    5.64188496988670089e-1,
    8.88314979438837594e0,
    6.61191906371416295e1,
    2.98635138197400131e2,
    8.81952221241769090e2,
    1.71204761263407058e3,
    2.05107837782607147e3,
    1.23033935479799725e3,
    2.15311535474403846e-8,
];
/// erfc denominator coefficients on `0.46875 < x <= 4`.
const ERFC_D: [f64; 8] = [
    1.57449261107098347e1,
    1.17693950891312499e2,
    5.37181101862009858e2,
    1.62138957456669019e3,
    3.29079923573345963e3,
    4.36261909014324716e3,
    3.43936767414372164e3,
    1.23033935480374942e3,
];
/// erfc numerator coefficients on `x > 4`.
const ERFC_P: [f64; 6] = [
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
];
/// erfc denominator coefficients on `x > 4`.
const ERFC_Q: [f64; 5] = [
    2.56852019228982242e0,
    1.87295284992346047e0,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
];

/// Switch point between the erf and erfc rational approximations.
const ERF_THRESH: f64 = 0.46875;
/// `1/sqrt(pi)`.
const SQRPI: f64 = 5.6418958354775628695e-1;
/// Above this argument `erfc(x)` underflows to zero in f64.
const ERFC_XBIG: f64 = 26.543;

/// `exp(-y^2)` computed as `exp(-q^2) * exp(-(y-q)(y+q))` with
/// `q = trunc(16 y)/16`, which avoids the roundoff amplification of squaring
/// `y` directly (the trick used in Cody's CALERF).
fn exp_neg_sq(y: f64) -> f64 {
    let q = (y * 16.0).trunc() / 16.0;
    let del = (y - q) * (y + q);
    (-q * q).exp() * (-del).exp()
}

/// erf via the degree-4/4 rational approximation, valid for
/// `|x| <= 0.46875`.
fn erf_small(x: f64) -> f64 {
    let ysq = x * x;
    let mut num = ERF_A[4] * ysq;
    let mut den = ysq;
    for (&a, &b) in ERF_A[..3].iter().zip(ERF_B[..3].iter()) {
        num = (num + a) * ysq;
        den = (den + b) * ysq;
    }
    x * (num + ERF_A[3]) / (den + ERF_B[3])
}

/// erfc(y) for `y > 0.46875`.
fn erfc_abs(y: f64) -> f64 {
    if y <= 4.0 {
        let mut num = ERFC_C[8] * y;
        let mut den = y;
        for (&c, &d) in ERFC_C[..7].iter().zip(ERFC_D[..7].iter()) {
            num = (num + c) * y;
            den = (den + d) * y;
        }
        exp_neg_sq(y) * (num + ERFC_C[7]) / (den + ERFC_D[7])
    } else if y < ERFC_XBIG {
        let ysq = 1.0 / (y * y);
        let mut num = ERFC_P[5] * ysq;
        let mut den = ysq;
        for (&p, &q) in ERFC_P[..4].iter().zip(ERFC_Q[..4].iter()) {
            num = (num + p) * ysq;
            den = (den + q) * ysq;
        }
        let r = ysq * (num + ERFC_P[4]) / (den + ERFC_Q[4]);
        exp_neg_sq(y) * (SQRPI - r) / y
    } else {
        0.0
    }
}

/// The error function `erf(x) = (2/sqrt(pi)) ∫_0^x e^(-t^2) dt`.
///
/// Cody-style rational Chebyshev approximations (Cody 1969; SPECFUN/CALERF,
/// Cody 1993): a degree-4/4 rational on `|x| <= 0.46875`, and
/// `1 - erfc(|x|)` (computed as `(1/2 - erfc) + 1/2` to avoid cancellation)
/// elsewhere. **Maximum error ≈ 1.2e-16 absolute (≈ 1 ulp)** — the rational
/// approximations themselves are accurate to better than `6e-19` relative,
/// so double-precision rounding dominates.
///
/// `NaN` propagates; `erf(±inf) = ±1`.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs();
    if y <= ERF_THRESH {
        return erf_small(x);
    }
    let r = erfc_abs(y);
    let val = (0.5 - r) + 0.5;
    if x < 0.0 {
        -val
    } else {
        val
    }
}

/// The complementary error function `erfc(x) = 1 - erf(x)`, accurate in the
/// right tail (relative accuracy is preserved down to the underflow point
/// `x ≈ 26.54`).
///
/// Same Cody rational approximations as [`erf`]; **maximum error ≈ 1 ulp
/// (≈ 1.2e-16 relative on the right tail)**. For `x < -0.46875` it is
/// computed as `2 - erfc(-x)`, which is exact to rounding since the result
/// is close to 2.
///
/// `NaN` propagates; `erfc(-inf) = 2`, `erfc(+inf) = 0`.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs();
    if y <= ERF_THRESH {
        return 1.0 - erf_small(x);
    }
    let r = erfc_abs(y);
    if x < 0.0 {
        2.0 - r
    } else {
        r
    }
}

// ---------------------------------------------------------------------------
// Regularized incomplete gamma P(a, x), Q(a, x)
// ---------------------------------------------------------------------------

/// Iteration budget for the incomplete gamma series / continued fraction.
/// Generous because convergence slows to O(sqrt(a)) iterations near `x ≈ a`
/// for large `a`.
// TODO(phase0): switch to the Temme uniform asymptotic expansion for very
// large `a` (a > ~1e6) instead of relying on a large iteration budget.
const GAMMA_MAX_ITER: u32 = 2000;

fn check_gamma_args(a: f64, x: f64) -> Result<(), StatsError> {
    if !(a > 0.0 && a.is_finite()) {
        return Err(StatsError::Domain {
            name: "a",
            value: a,
            requirement: "0 < a < inf",
        });
    }
    if x.is_nan() || x < 0.0 {
        return Err(StatsError::Domain {
            name: "x",
            value: x,
            requirement: "x >= 0",
        });
    }
    Ok(())
}

/// Lower series for P(a, x): `P(a,x) = x^a e^{-x}/Γ(a+1) Σ_n x^n / (a+1)…(a+n)`
/// (Abramowitz & Stegun 6.5.29; Press et al. 2007 §6.2, `gser`).
fn gamma_p_series(a: f64, x: f64) -> Result<f64, StatsError> {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..GAMMA_MAX_ITER {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * f64::EPSILON {
            return Ok(sum * (-x + a * x.ln() - ln_gamma(a)).exp());
        }
    }
    Err(StatsError::NoConvergence {
        what: "incomplete gamma series",
        iterations: GAMMA_MAX_ITER,
    })
}

/// Continued fraction for Q(a, x), evaluated with the modified Lentz method
/// (Lentz 1976; Thompson & Barnett 1986; Press et al. 2007 §6.2, `gcf`):
///
/// ```text
/// Q(a,x) = x^a e^{-x}/Γ(a) · 1/(x+1-a- 1·(1-a)/(x+3-a- 2·(2-a)/(x+5-a- …)))
/// ```
fn gamma_q_cf(a: f64, x: f64) -> Result<f64, StatsError> {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = if b.abs() < FPMIN {
        1.0 / FPMIN
    } else {
        1.0 / b
    };
    let mut h = d;
    for i in 1..=GAMMA_MAX_ITER {
        let an = -f64::from(i) * (f64::from(i) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= f64::EPSILON {
            return Ok((-x + a * x.ln() - ln_gamma(a)).exp() * h);
        }
    }
    Err(StatsError::NoConvergence {
        what: "incomplete gamma continued fraction",
        iterations: GAMMA_MAX_ITER,
    })
}

/// Regularized lower incomplete gamma function
/// `P(a, x) = γ(a, x) / Γ(a) = (1/Γ(a)) ∫_0^x t^(a-1) e^(-t) dt`.
///
/// Series representation for `x < a + 1`, modified-Lentz continued fraction
/// for the complement otherwise (Press et al. 2007, §6.2). Relative accuracy
/// is near machine precision away from the crossover; the golden suite pins
/// it (indirectly through the χ² and GED CDFs) at better than `1e-12`.
///
/// Domain: `a > 0` finite, `x >= 0` (`x = +inf` returns 1).
pub fn gamma_p(a: f64, x: f64) -> Result<f64, StatsError> {
    check_gamma_args(a, x)?;
    if x == 0.0 {
        return Ok(0.0);
    }
    if x.is_infinite() {
        return Ok(1.0);
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        Ok(1.0 - gamma_q_cf(a, x)?)
    }
}

/// Regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`,
/// accurate in the right tail (computed directly from the continued fraction
/// for `x >= a + 1`, so relative accuracy does not degrade as `Q -> 0`).
///
/// Domain: `a > 0` finite, `x >= 0` (`x = +inf` returns 0). See [`gamma_p`].
pub fn gamma_q(a: f64, x: f64) -> Result<f64, StatsError> {
    check_gamma_args(a, x)?;
    if x == 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    if x < a + 1.0 {
        Ok(1.0 - gamma_p_series(a, x)?)
    } else {
        gamma_q_cf(a, x)
    }
}

// ---------------------------------------------------------------------------
// Regularized incomplete beta I_x(a, b)
// ---------------------------------------------------------------------------

/// Iteration budget for the incomplete beta continued fraction.
const BETA_MAX_ITER: u32 = 500;

/// Continued fraction for the incomplete beta function, evaluated with the
/// modified Lentz method (Press et al. 2007, §6.4, `betacf`):
///
/// ```text
/// I_x(a,b) = x^a (1-x)^b / (a B(a,b)) · [ 1/(1+ d_1/(1+ d_2/(1+ …))) ]
/// d_{2m+1} = -(a+m)(a+b+m) x / ((a+2m)(a+2m+1))
/// d_{2m}   =  m (b-m) x / ((a+2m-1)(a+2m))
/// ```
fn beta_cf(a: f64, b: f64, x: f64) -> Result<f64, StatsError> {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=BETA_MAX_ITER {
        let mf = f64::from(m);
        let m2 = 2.0 * mf;
        // Even step.
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= f64::EPSILON {
            return Ok(h);
        }
    }
    Err(StatsError::NoConvergence {
        what: "incomplete beta continued fraction",
        iterations: BETA_MAX_ITER,
    })
}

/// Regularized incomplete beta function
/// `I_x(a, b) = (1/B(a,b)) ∫_0^x t^(a-1) (1-t)^(b-1) dt`.
///
/// Modified-Lentz continued fraction with the symmetry switch
/// `I_x(a, b) = 1 - I_{1-x}(b, a)` applied when `x >= (a+1)/(a+b+2)`, which
/// keeps the fraction in its rapidly-convergent region (Press et al. 2007,
/// §6.4). Accuracy near machine precision; pinned at `1e-12` relative by the
/// golden fixture.
///
/// Domain: `a > 0`, `b > 0` finite, `0 <= x <= 1`.
pub fn beta_inc(a: f64, b: f64, x: f64) -> Result<f64, StatsError> {
    if !(a > 0.0 && a.is_finite()) {
        return Err(StatsError::Domain {
            name: "a",
            value: a,
            requirement: "0 < a < inf",
        });
    }
    if !(b > 0.0 && b.is_finite()) {
        return Err(StatsError::Domain {
            name: "b",
            value: b,
            requirement: "0 < b < inf",
        });
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(StatsError::Domain {
            name: "x",
            value: x,
            requirement: "0 <= x <= 1",
        });
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    // ln of the prefactor x^a (1-x)^b / B(a, b); ln(1-x) via ln_1p for
    // accuracy at small x.
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(bt * beta_cf(a, b, x)? / a)
    } else {
        Ok(1.0 - bt * beta_cf(b, a, 1.0 - x)? / b)
    }
}

// ---------------------------------------------------------------------------
// Inverse standard normal CDF (Wichura AS241, PPND16)
// ---------------------------------------------------------------------------

// Coefficients of algorithm AS 241 (Wichura 1988), routine PPND16.
const PPND16_A: [f64; 8] = [
    3.3871328727963666080e0,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
];
const PPND16_B: [f64; 8] = [
    1.0,
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
];
const PPND16_C: [f64; 8] = [
    1.42343711074968357734e0,
    4.63033784615654529590e0,
    5.76949722146069140550e0,
    3.64784832476320460504e0,
    1.27045825245236838258e0,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
];
const PPND16_D: [f64; 8] = [
    1.0,
    2.05319162663775882187e0,
    1.67638483018380384940e0,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
];
const PPND16_E: [f64; 8] = [
    6.65790464350110377720e0,
    5.46378491116411436990e0,
    1.78482653991729133580e0,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
];
const PPND16_F: [f64; 8] = [
    1.0,
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
];

/// Degree-7 polynomial in Horner form; `coef` is lowest-order-first.
fn poly7(coef: &[f64; 8], x: f64) -> f64 {
    coef.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Inverse of the standard normal CDF (the standard normal quantile
/// function), `Φ^{-1}(p)`.
///
/// Algorithm AS 241, routine PPND16 (Wichura 1988): three rational
/// minimax approximations of degree 7/7 — a central branch in
/// `r = 0.180625 - (p - 1/2)^2` for `|p - 1/2| <= 0.425`, and two tail
/// branches in `r = sqrt(-ln(min(p, 1-p)))`. **Accuracy: about 1e-16
/// relative (the approximation error is below double rounding error over
/// the full domain)**.
///
/// Domain: `0 < p < 1`; returns [`StatsError::Domain`] otherwise (including
/// `NaN`). `inv_norm_cdf(0.5) == 0` exactly.
pub fn inv_norm_cdf(p: f64) -> Result<f64, StatsError> {
    if !(p > 0.0 && p < 1.0) {
        return Err(StatsError::Domain {
            name: "p",
            value: p,
            requirement: "0 < p < 1",
        });
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return Ok(q * poly7(&PPND16_A, r) / poly7(&PPND16_B, r));
    }
    let r = if q < 0.0 { p } else { 1.0 - p };
    let r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        poly7(&PPND16_C, r) / poly7(&PPND16_D, r)
    } else {
        let r = r - 5.0;
        poly7(&PPND16_E, r) / poly7(&PPND16_F, r)
    };
    Ok(if q < 0.0 { -val } else { val })
}

// ---------------------------------------------------------------------------
// Inverse regularized incomplete gamma
// ---------------------------------------------------------------------------

/// Iteration budget for the safeguarded Halley iterations of the inverse
/// incomplete gamma/beta functions.
const INV_MAX_ITER: u32 = 100;

/// Inverse of the regularized lower incomplete gamma function: returns `x`
/// such that `P(a, x) = p`.
///
/// Initial guess by the Wilson–Hilferty cube-root normal approximation for
/// `a > 1` and the small-`a` approximation of DiDonato & Morris otherwise
/// (as in Press et al. 2007, §6.2.1), refined by Halley's method on
/// `P(a, x) - p` with a bracketing safeguard: the iterate is kept inside the
/// current sign-change bracket, falling back to bisection (or doubling while
/// the upper bracket is unknown) whenever a step leaves it. Converges to a
/// relative tolerance of `1e-14` in `x`.
///
/// Domain: `a > 0` finite, `0 <= p <= 1`; `p = 0` returns 0 and `p = 1`
/// returns `+inf`.
pub fn inv_gamma_p(a: f64, p: f64) -> Result<f64, StatsError> {
    if !(a > 0.0 && a.is_finite()) {
        return Err(StatsError::Domain {
            name: "a",
            value: a,
            requirement: "0 < a < inf",
        });
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(StatsError::Domain {
            name: "p",
            value: p,
            requirement: "0 <= p <= 1",
        });
    }
    if p == 0.0 {
        return Ok(0.0);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }

    let gln = ln_gamma(a);
    let a1 = a - 1.0;
    // ln-prefactor pieces for the pdf of the Gamma(a, 1) distribution,
    // arranged as in NR `invgammp` to avoid overflow for large `a`.
    let (lna1, afac) = if a > 1.0 {
        let lna1 = a1.ln();
        (lna1, (a1 * (lna1 - 1.0) - gln).exp())
    } else {
        (0.0, 0.0)
    };

    // Initial guess.
    let mut x = if a > 1.0 {
        // Abramowitz & Stegun 26.2.23 rational approximation to the normal
        // quantile, fed into Wilson–Hilferty.
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            z = -z;
        }
        (a * (1.0 - 1.0 / (9.0 * a) - z / (3.0 * a.sqrt())).powi(3)).max(1e-3)
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if p < t {
            (p / t).powf(1.0 / a)
        } else {
            1.0 - (1.0 - (p - t) / (1.0 - t)).ln()
        }
    };

    // Safeguarded Halley iteration on f(x) = P(a, x) - p (increasing in x).
    let mut lo = 0.0_f64;
    let mut hi = f64::INFINITY;
    if x <= lo {
        x = 1e-300;
    }
    for _ in 0..INV_MAX_ITER {
        let f = gamma_p(a, x)? - p;
        if f == 0.0 {
            return Ok(x);
        }
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        // Gamma(a,1) density at x (the derivative of P w.r.t. x).
        let dens = if a > 1.0 {
            afac * (-(x - a1) + a1 * (x.ln() - lna1)).exp()
        } else {
            (-x + a1 * x.ln() - gln).exp()
        };
        let mut x_new = if dens > 0.0 && dens.is_finite() {
            let u = f / dens;
            // Halley correction (second-order): NR eq. 6.2.6-style damping.
            let step = u / (1.0 - 0.5 * f64::min(1.0, u * (a1 / x - 1.0)));
            x - step
        } else {
            f64::NAN
        };
        if !x_new.is_finite() || x_new <= lo || x_new >= hi {
            // Bisection safeguard (doubling while the upper bracket is open).
            x_new = if hi.is_finite() {
                0.5 * (lo + hi)
            } else {
                2.0 * x.max(1.0)
            };
        }
        let dx = (x_new - x).abs();
        x = x_new;
        if dx <= 1e-14 * x.max(1e-300) {
            return Ok(x);
        }
    }
    // Accept only if the residual is genuinely small.
    if (gamma_p(a, x)? - p).abs() <= 1e-9 {
        return Ok(x);
    }
    Err(StatsError::NoConvergence {
        what: "inverse incomplete gamma",
        iterations: INV_MAX_ITER,
    })
}

// ---------------------------------------------------------------------------
// Inverse regularized incomplete beta
// ---------------------------------------------------------------------------

/// Inverse of the regularized incomplete beta function: returns `x` such
/// that `I_x(a, b) = p`.
///
/// Initial guess from the Abramowitz & Stegun 26.5.22 normal approximation
/// when `a, b >= 1`, and from the crossing of the two power-law endpoint
/// expansions otherwise (as in Press et al. 2007, §6.4, `invbetai`), refined
/// by Halley's method on `I_x(a, b) - p` with a bracketing/bisection
/// safeguard on `[0, 1]`. Converges to a relative tolerance of `1e-15` in
/// `x`.
///
/// Domain: `a > 0`, `b > 0` finite, `0 <= p <= 1` (endpoints map to 0
/// and 1).
pub fn inv_beta_inc(a: f64, b: f64, p: f64) -> Result<f64, StatsError> {
    if !(a > 0.0 && a.is_finite()) {
        return Err(StatsError::Domain {
            name: "a",
            value: a,
            requirement: "0 < a < inf",
        });
    }
    if !(b > 0.0 && b.is_finite()) {
        return Err(StatsError::Domain {
            name: "b",
            value: b,
            requirement: "0 < b < inf",
        });
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(StatsError::Domain {
            name: "p",
            value: p,
            requirement: "0 <= p <= 1",
        });
    }
    if p == 0.0 {
        return Ok(0.0);
    }
    if p == 1.0 {
        return Ok(1.0);
    }

    // Initial guess.
    let mut x = if a >= 1.0 && b >= 1.0 {
        // A&S 26.5.22.
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            z = -z;
        }
        let al = (z * z - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = z * (al + h).sqrt() / h
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        a / (a + b * (2.0 * w).exp())
    } else {
        let lna = (a / (a + b)).ln();
        let lnb = (b / (a + b)).ln();
        let t = (a * lna).exp() / a;
        let u = (b * lnb).exp() / b;
        let w = t + u;
        if p < t / w {
            (a * w * p).powf(1.0 / a)
        } else {
            1.0 - (b * w * (1.0 - p)).powf(1.0 / b)
        }
    };
    x = x.clamp(1e-300, 1.0 - 1e-16);

    let afac = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..INV_MAX_ITER {
        let f = beta_inc(a, b, x)? - p;
        if f == 0.0 {
            return Ok(x);
        }
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        // Beta(a, b) density at x.
        let dens = ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() + afac).exp();
        let mut x_new = if dens > 0.0 && dens.is_finite() {
            let u = f / dens;
            let step = u / (1.0 - 0.5 * f64::min(1.0, u * ((a - 1.0) / x - (b - 1.0) / (1.0 - x))));
            x - step
        } else {
            f64::NAN
        };
        if !x_new.is_finite() || x_new <= lo || x_new >= hi {
            // Bisection safeguard.
            x_new = 0.5 * (lo + hi);
        }
        let dx = (x_new - x).abs();
        x = x_new;
        if dx <= 1e-15 * x.max(1e-300) {
            return Ok(x);
        }
    }
    if (beta_inc(a, b, x)? - p).abs() <= 1e-9 {
        return Ok(x);
    }
    Err(StatsError::NoConvergence {
        what: "inverse incomplete beta",
        iterations: INV_MAX_ITER,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rel(actual: f64, expected: f64, rtol: f64) {
        if expected == 0.0 {
            assert!(actual.abs() <= rtol, "actual {actual} vs 0, rtol {rtol}");
        } else {
            let rel = ((actual - expected) / expected).abs();
            assert!(
                rel <= rtol,
                "actual {actual} expected {expected} rel {rel:e}"
            );
        }
    }

    #[test]
    fn ln_gamma_special_values() {
        assert_eq!(ln_gamma(1.0), 0.0);
        assert_eq!(ln_gamma(2.0), 0.0);
        // Γ(0.5) = sqrt(pi)
        assert_rel(ln_gamma(0.5), 0.5 * PI.ln(), 1e-14);
        // ln|Γ(-0.5)| = ln(2 sqrt(pi))
        assert_rel(ln_gamma(-0.5), (2.0 * PI.sqrt()).ln(), 1e-13);
        assert!(ln_gamma(0.0).is_infinite());
        assert!(ln_gamma(-3.0).is_infinite());
        assert!(ln_gamma(f64::NAN).is_nan());
        // Γ(10) = 362880
        assert_rel(ln_gamma(10.0), 362880.0_f64.ln(), 1e-14);
    }

    #[test]
    fn erfc_far_tail() {
        // Reference values (Wolfram): erfc(5), erfc(10).
        assert_rel(erfc(5.0), 1.5374597944280349e-12, 1e-13);
        assert_rel(erfc(10.0), 2.0884875837625447e-45, 1e-13);
        assert_eq!(erfc(30.0), 0.0);
        assert_rel(erfc(-5.0), 2.0 - 1.5374597944280349e-12, 1e-15);
        assert!(erf(f64::NAN).is_nan());
        assert_rel(erf(6.5), 1.0, 1e-15);
    }

    #[test]
    fn gamma_pq_domain_and_edges() {
        assert!(gamma_p(-1.0, 1.0).is_err());
        assert!(gamma_p(1.0, -1.0).is_err());
        assert!(gamma_p(f64::NAN, 1.0).is_err());
        assert_eq!(gamma_p(2.5, 0.0).unwrap(), 0.0);
        assert_eq!(gamma_q(2.5, 0.0).unwrap(), 1.0);
        assert_eq!(gamma_p(2.5, f64::INFINITY).unwrap(), 1.0);
        // P(1, x) = 1 - e^{-x} exactly.
        assert_rel(gamma_p(1.0, 0.7).unwrap(), 1.0 - (-0.7_f64).exp(), 1e-14);
        assert_rel(gamma_q(1.0, 9.0).unwrap(), (-9.0_f64).exp(), 1e-14);
    }

    #[test]
    fn inv_norm_cdf_domain() {
        assert!(inv_norm_cdf(0.0).is_err());
        assert!(inv_norm_cdf(1.0).is_err());
        assert!(inv_norm_cdf(-0.2).is_err());
        assert!(inv_norm_cdf(f64::NAN).is_err());
        assert_eq!(inv_norm_cdf(0.5).unwrap(), 0.0);
    }

    #[test]
    fn inverse_functions_round_trip() {
        for &(a, b) in &[(2.0, 3.0), (0.5, 0.5), (5.0, 1.5), (0.3, 4.0)] {
            for &p in &[1e-6, 0.01, 0.3, 0.5, 0.9, 0.999] {
                let x = inv_beta_inc(a, b, p).unwrap();
                assert_rel(beta_inc(a, b, x).unwrap(), p, 1e-12);
            }
        }
        for &a in &[0.4, 1.0, 2.5, 17.0] {
            for &p in &[1e-6, 0.01, 0.3, 0.5, 0.9, 0.999] {
                let x = inv_gamma_p(a, p).unwrap();
                assert_rel(gamma_p(a, x).unwrap(), p, 1e-12);
            }
        }
        assert_eq!(inv_gamma_p(2.0, 0.0).unwrap(), 0.0);
        assert!(inv_gamma_p(2.0, 1.0).unwrap().is_infinite());
        assert_eq!(inv_beta_inc(2.0, 3.0, 0.0).unwrap(), 0.0);
        assert_eq!(inv_beta_inc(2.0, 3.0, 1.0).unwrap(), 1.0);
    }
}
