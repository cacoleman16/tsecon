//! Numerical kernels the copula families are built on: the Genz (2004)
//! bivariate standard-normal CDF, the bivariate Student-t CDF by the
//! conditional 1-D integral, and the Debye function `D1` behind Frank's
//! Kendall tau.

// The Gauss-Legendre constants below are Genz's published TVPACK digits,
// kept verbatim (some carry a digit beyond double rounding).
#![allow(clippy::excessive_precision)]

use tsecon_stats::{ContinuousDist, StdNormal, StudentT};

/// Standard normal CDF (Cody-accurate `erfc` route from `tsecon-stats`).
#[inline]
pub(crate) fn norm_cdf(x: f64) -> f64 {
    StdNormal.cdf(x)
}

// ---------------------------------------------------------------------------
// Bivariate standard-normal CDF — Genz (2004), TVPACK BVND
// ---------------------------------------------------------------------------

// Gauss-Legendre abscissas/weights on [-1, 1] (half of each symmetric set;
// the `is = -1, +1` inner loop below covers both signs), exactly the sets
// Genz's BVND uses: 6-point for |r| < 0.3, 12-point for |r| < 0.75,
// 20-point otherwise.
const GL06_W: [f64; 3] = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910];
const GL06_X: [f64; 3] = [
    -0.9324695142031521,
    -0.6612093864662645,
    -0.2386191860831969,
];
const GL12_W: [f64; 6] = [
    0.04717533638651183,
    0.1069393259953184,
    0.1600783285433462,
    0.2031674267230659,
    0.2334925365383548,
    0.2491470458134028,
];
const GL12_X: [f64; 6] = [
    -0.9815606342467192,
    -0.9041172563704749,
    -0.7699026741943047,
    -0.5873179542866175,
    -0.3678314989981802,
    -0.1252334085114689,
];
const GL20_W: [f64; 10] = [
    0.01761400713915212,
    0.04060142980038694,
    0.06267204833410907,
    0.08327674157670475,
    0.1019301198172404,
    0.1181945319615184,
    0.1316886384491766,
    0.1420961093183820,
    0.1491729864726037,
    0.1527533871307258,
];
const GL20_X: [f64; 10] = [
    -0.9931285991850949,
    -0.9639719272779138,
    -0.9122344282513259,
    -0.8391169718222188,
    -0.7463319064601508,
    -0.6360536807265150,
    -0.5108670019508271,
    -0.3737060887154195,
    -0.2277858511416451,
    -0.07652652113349734,
];

const TWO_PI: f64 = 2.0 * core::f64::consts::PI;

/// `P(X > dh, Y > dk)` for standard bivariate normal `(X, Y)` with
/// correlation `r` — a transcription of Genz (2004) BVND (TVPACK), the
/// Drezner-Wesolowsky Gauss-Legendre method with the separate `|r| > 0.925`
/// expansion branch. Double-precision accurate (~5e-16 absolute; the golden
/// tests pin it at 1e-10 against the exact Owen's-T closed form evaluated
/// in scipy).
///
/// Requires `|r| < 1` and finite arguments (enforced by the callers).
fn bvnd(dh: f64, dk: f64, r: f64) -> f64 {
    let (w, x): (&[f64], &[f64]) = if r.abs() < 0.3 {
        (&GL06_W, &GL06_X)
    } else if r.abs() < 0.75 {
        (&GL12_W, &GL12_X)
    } else {
        (&GL20_W, &GL20_X)
    };
    let h = dh;
    let mut k = dk;
    let mut hk = h * k;
    let mut bvn = 0.0;
    if r.abs() < 0.925 {
        if r.abs() > 0.0 {
            let hs = (h * h + k * k) / 2.0;
            let asr = r.asin();
            for i in 0..w.len() {
                for is in [-1.0_f64, 1.0] {
                    let sn = (asr * (is * x[i] + 1.0) / 2.0).sin();
                    bvn += w[i] * ((sn * hk - hs) / (1.0 - sn * sn)).exp();
                }
            }
            bvn = bvn * asr / (2.0 * TWO_PI);
        }
        bvn + norm_cdf(-h) * norm_cdf(-k)
    } else {
        if r < 0.0 {
            k = -k;
            hk = -hk;
        }
        if r.abs() < 1.0 {
            let a_s = (1.0 - r) * (1.0 + r);
            let mut a = a_s.sqrt();
            let bs = (h - k) * (h - k);
            let c = (4.0 - hk) / 8.0;
            let d = (12.0 - hk) / 16.0;
            let mut asr = -(bs / a_s + hk) / 2.0;
            if asr > -100.0 {
                bvn = a
                    * asr.exp()
                    * (1.0 - c * (bs - a_s) * (1.0 - d * bs / 5.0) / 3.0 + c * d * a_s * a_s / 5.0);
            }
            if -hk < 100.0 {
                let b = bs.sqrt();
                bvn -= (-hk / 2.0).exp()
                    * TWO_PI.sqrt()
                    * norm_cdf(-b / a)
                    * b
                    * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
            }
            a /= 2.0;
            for i in 0..w.len() {
                for is in [-1.0_f64, 1.0] {
                    let xs = (a * (is * x[i] + 1.0)) * (a * (is * x[i] + 1.0));
                    let rs = (1.0 - xs).sqrt();
                    asr = -(bs / xs + hk) / 2.0;
                    if asr > -100.0 {
                        let sp = 1.0 + c * xs * (1.0 + d * xs);
                        let ep = (-hk * (1.0 - rs) / (2.0 * (1.0 + rs))).exp() / rs;
                        bvn += a * w[i] * asr.exp() * (ep - sp);
                    }
                }
            }
            bvn = -bvn / TWO_PI;
        }
        if r > 0.0 {
            bvn + norm_cdf(-h.max(k))
        } else {
            bvn = -bvn;
            if k > h {
                bvn += norm_cdf(k) - norm_cdf(h);
            }
            bvn
        }
    }
}

/// Bivariate standard-normal CDF `P(X <= x, Y <= y)` at correlation `rho`
/// (`|rho| < 1`), via [`bvnd`] and the reflection `P(X <= x, Y <= y) =
/// P(-X > -x, -Y > -y)`.
pub(crate) fn bvn_cdf(x: f64, y: f64, rho: f64) -> f64 {
    bvnd(-x, -y, rho).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Bivariate Student-t CDF — conditional 1-D integral
// ---------------------------------------------------------------------------

/// Bivariate Student-t CDF `P(T1 <= x, T2 <= y)` at correlation `rho`
/// (`|rho| < 1`) and `nu > 0` degrees of freedom, by the exact conditional
/// decomposition (Joe 2014, sec. 2.2): given `T1 = s`, `T2` is a scaled and
/// shifted t with `nu + 1` degrees of freedom, so
///
/// ```text
/// P = int_-inf^x  f_nu(s) * F_{nu+1}( (y - rho s) sqrt( (nu+1) /
///        ((nu + s^2)(1 - rho^2)) ) )  ds,
/// ```
///
/// evaluated after the substitution `w = F_nu(s)` (a bounded, smooth
/// integrand on `(0, F_nu(x))` with finite endpoint limits) by adaptive
/// Simpson quadrature to ~1e-12. The golden tests pin it at 1e-10 against
/// scipy `quad` on the same closed-form representation, which is itself
/// cross-checked in the fixture generator against
/// `scipy.stats.multivariate_t.cdf` at that reference's own ~2e-4 QMC
/// noise level.
pub(crate) fn bvt_cdf(x: f64, y: f64, rho: f64, nu: f64) -> f64 {
    let Ok(t_nu) = StudentT::new(nu) else {
        return f64::NAN;
    };
    let Ok(t_nu1) = StudentT::new(nu + 1.0) else {
        return f64::NAN;
    };
    let p1 = t_nu.cdf(x);
    if p1 <= 0.0 {
        return 0.0;
    }
    let one_m_r2 = 1.0 - rho * rho;
    // w -> 0 (s -> -inf) limit of the conditional argument.
    let g_limit = rho * ((nu + 1.0) / one_m_r2).sqrt();
    let f = |w: f64| -> f64 {
        if w < 1e-290 {
            return t_nu1.cdf(g_limit);
        }
        match t_nu.ppf(w) {
            Ok(s) => {
                let g = (y - rho * s) * ((nu + 1.0) / ((nu + s * s) * one_m_r2)).sqrt();
                t_nu1.cdf(g)
            }
            Err(_) => f64::NAN,
        }
    };
    adaptive_simpson(&f, 0.0, p1, 1e-13, 32).clamp(0.0, 1.0)
}

/// Adaptive Simpson quadrature with the classical `|S_left + S_right - S|
/// < 15 eps` refinement rule and a depth cap.
fn adaptive_simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, eps: f64, depth: u32) -> f64 {
    let m = 0.5 * (a + b);
    let fa = f(a);
    let fm = f(m);
    let fb = f(b);
    let s = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    simpson_step(f, a, b, fa, fm, fb, s, eps, depth)
}

#[allow(clippy::too_many_arguments)]
fn simpson_step<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    s: f64,
    eps: f64,
    depth: u32,
) -> f64 {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = f(lm);
    let frm = f(rm);
    let sl = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    let sr = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    let s2 = sl + sr;
    if depth == 0 || (s2 - s).abs() <= 15.0 * eps {
        s2 + (s2 - s) / 15.0
    } else {
        simpson_step(f, a, m, fa, flm, fm, sl, eps / 2.0, depth - 1)
            + simpson_step(f, m, b, fm, frm, fb, sr, eps / 2.0, depth - 1)
    }
}

// ---------------------------------------------------------------------------
// Debye function D1 — Frank's Kendall tau
// ---------------------------------------------------------------------------

/// Bernoulli numbers `B_2 .. B_32` (exact rationals evaluated in double).
const B2K: [f64; 16] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
    43867.0 / 798.0,
    -174611.0 / 330.0,
    854513.0 / 138.0,
    -236364091.0 / 2730.0,
    8553103.0 / 6.0,
    -23749461029.0 / 870.0,
    8615841276005.0 / 14322.0,
    -7709321041217.0 / 510.0,
];

/// The Debye function `D1(x) = (1/x) int_0^x t / (e^t - 1) dt` for `x > 0`.
///
/// Two exact expansions, both accurate to ~1e-16 relative and continuous
/// across the documented seam at `x = 2` (property-tested):
///
/// * `x <= 2` — the Bernoulli series
///   `D1(x) = 1 - x/4 + sum_k B_2k x^2k / (2k+1)!` (convergent for
///   `|x| < 2 pi`; truncated below 1e-17 relative, at most 16 terms);
/// * `x > 2` — the exponential tail form
///   `D1(x) = (pi^2/6 - sum_k e^{-kx} (x/k + 1/k^2)) / x` from
///   `int_x^inf t/(e^t - 1) dt = sum_k e^{-kx}(x/k + 1/k^2)`.
pub(crate) fn debye_1(x: f64) -> f64 {
    debug_assert!(x > 0.0);
    if x <= 2.0 {
        let x2 = x * x;
        let mut acc = 1.0 - 0.25 * x;
        let mut xpow = 1.0;
        let mut fact = 1.0; // (2k+1)! running product
        for (k, &b) in B2K.iter().enumerate() {
            let kk = (k + 1) as f64;
            xpow *= x2;
            fact *= 2.0 * kk * (2.0 * kk + 1.0);
            let term = b * xpow / fact;
            acc += term;
            if term.abs() < 1e-17 * acc.abs() {
                break;
            }
        }
        acc
    } else {
        let pi2_6 = core::f64::consts::PI * core::f64::consts::PI / 6.0;
        let mut sum = 0.0;
        for k in 1..400 {
            let kf = k as f64;
            let term = (-kf * x).exp() * (x / kf + 1.0 / (kf * kf));
            sum += term;
            if term < 1e-18 * pi2_6 {
                break;
            }
        }
        (pi2_6 - sum) / x
    }
}

/// Frank's Kendall tau: `tau(theta) = 1 + 4 (D1(theta) - 1) / theta`,
/// odd in `theta` (Genest 1987; Joe 2014 p. 166).
///
/// For `|theta| <= 2` that expression cancels catastrophically (`D1 - 1`
/// loses all precision as `theta -> 0`), so the equivalent direct series
/// `tau = sum_k 4 B_2k theta^{2k-1} / (2k+1)! = theta/9 - theta^3/900 +
/// ...` is used — exact through the origin. Above 2 the tail-form `D1`
/// has no cancellation and the defining expression is used as written.
pub(crate) fn tau_frank(theta: f64) -> f64 {
    let t = theta.abs();
    if t == 0.0 {
        return 0.0;
    }
    let tau = if t <= 2.0 {
        let x2 = t * t;
        let mut acc = 0.0;
        let mut xpow = t; // theta^(2k-1)
        let mut fact = 1.0; // (2k+1)! running product
        for (k, &b) in B2K.iter().enumerate() {
            let kk = (k + 1) as f64;
            fact *= 2.0 * kk * (2.0 * kk + 1.0);
            let term = 4.0 * b * xpow / fact;
            acc += term;
            if term.abs() < 1e-17 * acc.abs() {
                break;
            }
            xpow *= x2;
        }
        acc
    } else {
        1.0 + 4.0 * (debye_1(t) - 1.0) / t
    };
    if theta < 0.0 {
        -tau
    } else {
        tau
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bvn_median_and_independence_closed_forms() {
        for &rho in &[-0.9_f64, -0.5, 0.0, 0.3, 0.8, 0.95, 0.99] {
            let med = 0.25 + rho.asin() / TWO_PI;
            assert!(
                (bvn_cdf(0.0, 0.0, rho) - med).abs() < 1e-14,
                "median formula at rho={rho}"
            );
        }
        for &(x, y) in &[(0.3, -0.7), (1.5, 2.0), (-2.0, -0.4)] {
            let prod = norm_cdf(x) * norm_cdf(y);
            assert!((bvn_cdf(x, y, 0.0) - prod).abs() < 1e-14);
        }
    }

    #[test]
    fn bvt_median_closed_form() {
        for &(rho, nu) in &[(0.5_f64, 4.0_f64), (-0.4, 7.5), (0.9, 2.5)] {
            let med = 0.25 + rho.asin() / TWO_PI;
            assert!(
                (bvt_cdf(0.0, 0.0, rho, nu) - med).abs() < 1e-11,
                "t median formula at rho={rho}, nu={nu}"
            );
        }
    }

    #[test]
    fn debye_branches_agree_at_the_seam() {
        // Series branch at the seam vs the tail branch one ulp above it
        // (the function itself changes by ~1e-16 over that step, so any
        // gap is branch disagreement).
        let below = debye_1(2.0);
        let above = debye_1(2.0 * (1.0 + f64::EPSILON));
        assert!((below - above).abs() < 5e-15, "{below} vs {above}");
        // Exact-rational Bernoulli-series value (40-digit check).
        assert!((debye_1(2.0) - 0.6069472846098100720457858966741557).abs() < 5e-16);
        // Known value D1(1) = 0.7775046341122482 (Abramowitz-Stegun 27.1).
        assert!((debye_1(1.0) - 0.7775046341122482).abs() < 1e-15);
    }

    #[test]
    fn tau_frank_is_odd_and_linear_at_zero() {
        for &th in &[0.3, 1.0, 4.0, 20.0] {
            assert_eq!(tau_frank(-th), -tau_frank(th));
        }
        // The direct series is exact through the origin: theta/9 to first
        // order, no cancellation.
        assert!((tau_frank(1e-13) - 1e-13 / 9.0).abs() < 1e-28);
        // Continuity across the series/tail seam at |theta| = 2.
        let a = tau_frank(2.0);
        let b = tau_frank(2.0 * (1.0 + f64::EPSILON));
        assert!((a - b).abs() < 5e-15, "{a} vs {b}");
    }
}
