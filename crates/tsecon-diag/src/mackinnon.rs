//! MacKinnon response-surface p-values and critical values for the
//! (augmented) Dickey-Fuller tau statistic.
//!
//! P-values follow MacKinnon (1994): the tau statistic is mapped through a
//! low-order polynomial `g` fitted by response-surface regression and the
//! p-value is `Phi(g(tau))` with `Phi` the standard normal CDF. Two
//! polynomial regimes are used — a quadratic for the small-p (left) region
//! `tau <= tau_star` and a cubic for the large-p region — with hard
//! saturation to 0 below `tau_min` and to 1 above `tau_max`:
//!
//! ```text
//! p(tau) = 0                                   tau <  tau_min
//!        = Phi(c0 + c1 tau + c2 tau^2)         tau <= tau_star
//!        = Phi(d0 + d1 tau + d2 tau^2 + d3 tau^3)   tau <= tau_max
//!        = 1                                   tau >  tau_max
//! ```
//!
//! Critical values follow MacKinnon (2010): for each regression and level,
//! `cv(n) = b0 + b1/n + b2/n^2 + b3/n^3`, with `b0` the asymptotic value.
//! The coefficients below are the no-cointegration case (`N = 1`, a single
//! I(1) series) exactly as tabulated in `statsmodels.tsa.adfvalues`
//! (statsmodels 0.14.6), which is the golden reference for this crate.
//!
//! References: MacKinnon (1994), "Approximate Asymptotic Distribution
//! Functions for Unit-Root and Cointegration Tests", JBES 12(2);
//! MacKinnon (2010), "Critical Values for Cointegration Tests", Queen's
//! University working paper 1227.

use tsecon_stats::{ContinuousDist, StdNormal};

use crate::unitroot::AdfRegression;

/// MacKinnon (2010) finite-sample critical values for the ADF tau
/// statistic at the conventional levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdfCriticalValues {
    /// The 1% critical value.
    pub pct1: f64,
    /// The 5% critical value.
    pub pct5: f64,
    /// The 10% critical value.
    pub pct10: f64,
}

/// Response-surface coefficients for one deterministic specification
/// (N = 1 rows of the statsmodels tables, scaling already applied).
struct Surface {
    /// Boundary between the small-p and large-p polynomial regions.
    tau_star: f64,
    /// Below this the p-value saturates at 0.
    tau_min: f64,
    /// Above this the p-value saturates at 1.
    tau_max: f64,
    /// Small-p (left tail) quadratic `[c0, c1, c2]`.
    small_p: [f64; 3],
    /// Large-p cubic `[d0, d1, d2, d3]`.
    large_p: [f64; 4],
    /// MacKinnon (2010) `1/n` polynomials, rows = 1% / 5% / 10%, each
    /// `[b0, b1, b2, b3]` for `b0 + b1/n + b2/n^2 + b3/n^3`.
    crit: [[f64; 4]; 3],
}

/// Regression "n": no deterministic terms.
const SURFACE_N: Surface = Surface {
    tau_star: -1.04,
    tau_min: -19.04,
    tau_max: f64::INFINITY,
    small_p: [0.6344, 1.2378, 3.2496 * 1e-2],
    large_p: [0.4797, 9.3557 * 1e-1, -0.6999 * 1e-1, 3.3066 * 1e-2],
    crit: [
        [-2.56574, -2.2358, -3.627, 0.0],
        [-1.94100, -0.2686, -3.365, 31.223],
        [-1.61682, 0.2656, -2.714, 25.364],
    ],
};

/// Regression "c": constant only.
const SURFACE_C: Surface = Surface {
    tau_star: -1.61,
    tau_min: -18.83,
    tau_max: 2.74,
    small_p: [2.1659, 1.4412, 3.8269 * 1e-2],
    large_p: [1.7339, 9.3202 * 1e-1, -1.2745 * 1e-1, -1.0368 * 1e-2],
    crit: [
        [-3.43035, -6.5393, -16.786, -79.433],
        [-2.86154, -2.8903, -4.234, -40.040],
        [-2.56677, -1.5384, -2.809, 0.0],
    ],
};

/// Regression "ct": constant and linear trend.
const SURFACE_CT: Surface = Surface {
    tau_star: -2.89,
    tau_min: -16.18,
    tau_max: 0.7,
    small_p: [3.2512, 1.6047, 4.9588 * 1e-2],
    large_p: [2.5261, 6.1654 * 1e-1, -3.7956 * 1e-1, -6.0285 * 1e-2],
    crit: [
        [-3.95877, -9.0531, -28.428, -134.155],
        [-3.41049, -4.3904, -9.036, -45.374],
        [-3.12705, -2.5856, -3.925, -22.380],
    ],
};

fn surface(regression: AdfRegression) -> &'static Surface {
    match regression {
        AdfRegression::NoConstant => &SURFACE_N,
        AdfRegression::Constant => &SURFACE_C,
        AdfRegression::ConstantTrend => &SURFACE_CT,
    }
}

/// Horner evaluation from the highest-degree coefficient down, matching
/// `numpy.polyval` on the reversed coefficient vector.
fn polyval_ascending(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// MacKinnon (1994) approximate asymptotic p-value for an (augmented)
/// Dickey-Fuller tau statistic, no-cointegration case (N = 1).
///
/// Matches `statsmodels.tsa.adfvalues.mackinnonp(stat, regression, N=1)`:
/// the statistic is pushed through the fitted response-surface polynomial
/// for the given deterministic specification and mapped to a probability
/// with the standard normal CDF, saturating at exactly 0.0 / 1.0 outside
/// `[tau_min, tau_max]`. A NaN statistic yields a NaN p-value.
pub fn mackinnon_p(stat: f64, regression: AdfRegression) -> f64 {
    let s = surface(regression);
    if stat > s.tau_max {
        return 1.0;
    }
    if stat < s.tau_min {
        return 0.0;
    }
    let g = if stat <= s.tau_star {
        polyval_ascending(&s.small_p, stat)
    } else {
        polyval_ascending(&s.large_p, stat)
    };
    StdNormal.cdf(g)
}

/// MacKinnon (2010) critical values for the ADF tau statistic at the
/// 1% / 5% / 10% levels, no-cointegration case (N = 1).
///
/// With `nobs = Some(n)` the finite-sample response surface
/// `b0 + b1/n + b2/n^2 + b3/n^3` is evaluated (statsmodels
/// `mackinnoncrit(N=1, regression, nobs=n)`); `None` (or `Some(0)`)
/// returns the asymptotic values `b0`.
pub fn mackinnon_crit(regression: AdfRegression, nobs: Option<usize>) -> AdfCriticalValues {
    let s = surface(regression);
    let eval = |row: &[f64; 4]| match nobs {
        None | Some(0) => row[0],
        Some(n) => polyval_ascending(row, 1.0 / n as f64),
    };
    AdfCriticalValues {
        pct1: eval(&s.crit[0]),
        pct5: eval(&s.crit[1]),
        pct10: eval(&s.crit[2]),
    }
}
