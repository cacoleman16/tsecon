//! The Bai (1997) break-date argmax distribution.
//!
//! Under shrinking-break asymptotics with homogeneous regressor second
//! moments and error variance across the two adjacent regimes, the scaled
//! break-date estimation error converges to `argmax_s { W(s) - |s|/2 }`
//! with `W` a two-sided standard Wiener process (Bai 1997, *Review of
//! Economics and Statistics* 79(4), 551–563; Bai & Perron 1998,
//! Proposition 8 with equal moments). The argmax has the known closed-form
//! cdf (Bai 1997, Appendix B; Yao 1987), for `x >= 0`:
//!
//! ```text
//! G(x) = 1 + sqrt(x / (2 pi)) exp(-x / 8)
//!          + (3/2) exp(x) Phi(-(3/2) sqrt(x))
//!          - ((x + 5)/2) Phi(-sqrt(x) / 2),        G(-x) = 1 - G(x),
//! ```
//!
//! whose two-sided 90% and 95% critical values are the published 7.7 and
//! 11.03 (this crate reproduces them to three decimals by inverting `G`
//! rather than hardcoding).

use core::f64::consts::PI;

use tsecon_stats::{ContinuousDist, StdNormal};

use crate::error::BreaksError;

/// Cdf `G(x)` of `argmax_s { W(s) - |s|/2 }` (Bai 1997).
///
/// Symmetric around zero with `G(0) = 1/2`; evaluated by the closed form
/// above (values beyond `x = 200` are indistinguishable from 1).
#[must_use]
pub fn bai_argmax_cdf(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 - bai_argmax_cdf(-x);
    }
    if x > 200.0 {
        return 1.0;
    }
    let phi = StdNormal;
    let sx = x.sqrt();
    let g = 1.0 + (x / (2.0 * PI)).sqrt() * (-x / 8.0).exp() + 1.5 * x.exp() * phi.cdf(-1.5 * sx)
        - 0.5 * (x + 5.0) * phi.cdf(-0.5 * sx);
    g.clamp(0.0, 1.0)
}

/// Two-sided critical value `c` with `P(|argmax| <= c) = level`, found by
/// bisection on the closed-form cdf (`2 G(c) - 1 = level`).
///
/// # Errors
///
/// [`BreaksError::InvalidArgument`] unless `level` is in `(0, 1)`.
pub fn bai_argmax_two_sided_crit(level: f64) -> Result<f64, BreaksError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(BreaksError::InvalidArgument {
            what: "confidence level for the break-date interval must be strictly \
                   between 0 and 1 (use 0.90 or 0.95)",
        });
    }
    let target = |c: f64| 2.0 * bai_argmax_cdf(c) - 1.0 - level;
    let (mut lo, mut hi) = (0.0_f64, 200.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if target(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}
