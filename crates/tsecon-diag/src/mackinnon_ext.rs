//! MacKinnon response surfaces for the Phillips-Perron normalized-bias
//! statistic (the "ADF-z" distribution) and for residual-based
//! cointegration tests (the `N > 1` tau distributions), used by the two
//! tests in [`crate::phillips`].
//!
//! These are the surfaces the N = 1 tau tables in [`crate::mackinnon`] do
//! *not* cover:
//!
//! * **ADF-z** — the null distribution of the Phillips-Perron `Z-alpha`
//!   (normalized-bias) statistic. P-values map the statistic through a
//!   cubic in `ln|stat|` on the left of `z_star` and a cubic in the raw
//!   statistic on the right, then apply the standard normal CDF (there is
//!   no hard saturation: `min = -inf`, `max = +inf`). Critical values use a
//!   `1/n` response surface. Byte-identical to
//!   `arch.unitroot.unitroot.mackinnonp(stat, regression, dist_type="adf-z")`
//!   and `mackinnoncrit(..., dist_type="adf-z")` (arch 8.0.0), which is what
//!   the golden fixtures pin.
//!
//! * **Cointegration tau** — the MacKinnon (1994, 2010) response surfaces
//!   for the Engle-Granger / Phillips-Ouliaris residual test with
//!   `N = 1 + (number of stochastic regressors)`. P-values use the same
//!   quadratic/cubic tau machinery as the `N = 1` case but with the `N`-th
//!   row of the tables (`N = 2..6`); critical values use the MacKinnon
//!   (2010) `1/n` surfaces (`N = 2..12`, constant/constant-trend only — the
//!   no-constant case has no 2010 cointegration table, matching
//!   `statsmodels.tsa.stattools.coint`). Byte-identical to
//!   `statsmodels.tsa.adfvalues.mackinnonp(stat, regression=trend, N)` and
//!   `mackinnoncrit(N, regression=trend, nobs)`.
//!
//! Tables are transcribed verbatim (full-precision doubles) from arch
//! 8.0.0's `dickey_fuller` critical-value module and statsmodels 0.14.6's
//! `adfvalues.tau_2010s`.
//!
//! References: MacKinnon (1994), JBES 12(2); MacKinnon (2010), Queen's
//! University working paper 1227.

use tsecon_stats::{ContinuousDist, StdNormal};

use crate::mackinnon::AdfCriticalValues;
use crate::phillips::PoTrend;
use crate::unitroot::AdfRegression;

/// Horner evaluation from the highest-degree coefficient down, matching
/// `numpy.polyval` on the reversed coefficient vector (as in
/// [`crate::mackinnon`]).
fn polyval_ascending(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

// ------------------------------------------------------------------ ADF-z

/// ADF-z response surface for one deterministic specification (N = 1). The
/// p-value distribution never saturates (`min = -inf`, `max = +inf`), so
/// only the small/large split point `star` is stored.
struct AdfZSurface {
    /// Boundary between the small-p (`ln|stat|`) and large-p (raw `stat`)
    /// polynomial regions.
    star: f64,
    /// Small-p cubic `[c0, c1, c2, c3]` in `ln|stat|`.
    small_p: [f64; 4],
    /// Large-p cubic `[d0, d1, d2, d3]` in `stat`.
    large_p: [f64; 4],
    /// `1/n` critical-value polynomials, rows = 1% / 5% / 10%, each
    /// `[b0, b1, b2, b3]` for `b0 + b1/n + b2/n^2 + b3/n^3`.
    cv: [[f64; 4]; 3],
}

const ADF_Z_N: AdfZSurface = AdfZSurface {
    star: -1.79146,
    small_p: [0.05872, -0.69633, 0.02471, -0.04283],
    large_p: [0.56681, 0.67544, 0.06881, 0.00235],
    cv: [
        [-13.30499, 331.55385, -8807.11008, 88544.76648],
        [-7.82957, 222.75548, -5863.14998, 58024.48295],
        [-5.57486, 170.79065, -4422.48027, 43264.68244],
    ],
};

const ADF_Z_C: AdfZSurface = AdfZSurface {
    star: -5.04709,
    small_p: [1.94205, -1.47677, 0.21163, -0.06288],
    large_p: [1.70059, 0.49465, 0.02636, 0.00055],
    cv: [
        [-20.6258, 119.28619, -442.99517, 1082.9222],
        [-14.09457, 58.46106, -164.08684, 479.02908],
        [-11.25118, 38.48786, -87.62853, 246.93531],
    ],
};

const ADF_Z_CT: AdfZSurface = AdfZSurface {
    star: -9.22766,
    small_p: [4.05596, -2.34128, 0.41403, -0.08312],
    large_p: [2.60323, 0.39217, 0.01321, 0.00019],
    cv: [
        [-29.3568, 232.33249, -1171.21447, 3166.83631],
        [-21.71085, 130.15894, -504.95264, 1170.38109],
        [-18.24475, 93.25896, -305.14509, 581.02963],
    ],
};

fn adf_z_surface(regression: AdfRegression) -> &'static AdfZSurface {
    match regression {
        AdfRegression::NoConstant => &ADF_Z_N,
        AdfRegression::Constant => &ADF_Z_C,
        AdfRegression::ConstantTrend => &ADF_Z_CT,
    }
}

/// MacKinnon approximate p-value for a Phillips-Perron `Z-alpha`
/// (normalized-bias) statistic, no-cointegration case (N = 1).
///
/// Matches `arch.unitroot.unitroot.mackinnonp(stat, regression,
/// dist_type="adf-z")`: below `z_star` the statistic enters as `ln|stat|`,
/// above it as the raw statistic; both branches are cubic and the result is
/// pushed through the standard normal CDF. The distribution never
/// saturates. A NaN statistic yields a NaN p-value.
pub(crate) fn mackinnon_z_p(stat: f64, regression: AdfRegression) -> f64 {
    let s = adf_z_surface(regression);
    let g = if stat <= s.star {
        polyval_ascending(&s.small_p, stat.abs().ln())
    } else {
        polyval_ascending(&s.large_p, stat)
    };
    StdNormal.cdf(g)
}

/// MacKinnon critical values for a Phillips-Perron `Z-alpha` statistic at
/// the 1% / 5% / 10% levels, no-cointegration case (N = 1).
///
/// Evaluates the finite-sample `1/n` response surface at `nobs`, matching
/// `mackinnoncrit(regression, nobs, dist_type="adf-z")`.
pub(crate) fn mackinnon_z_crit(regression: AdfRegression, nobs: usize) -> AdfCriticalValues {
    let s = adf_z_surface(regression);
    let x = 1.0 / nobs as f64;
    AdfCriticalValues {
        pct1: polyval_ascending(&s.cv[0], x),
        pct5: polyval_ascending(&s.cv[1], x),
        pct10: polyval_ascending(&s.cv[2], x),
    }
}

// ---------------------------------------------------------- Cointegration

/// Cointegration tau response surface for one `(trend, N)` pair
/// (`N = 2..6`). Same quadratic/cubic form as the `N = 1` ADF-t case.
struct CointPSurface {
    /// Boundary between the small-p and large-p polynomial regions.
    star: f64,
    /// Below this the p-value saturates at 0.
    min: f64,
    /// Above this the p-value saturates at 1.
    max: f64,
    /// Small-p (left tail) quadratic `[c0, c1, c2]`.
    small_p: [f64; 3],
    /// Large-p cubic `[d0, d1, d2, d3]`.
    large_p: [f64; 4],
}

const COINT_P_N: [CointPSurface; 5] = [
    // N = 2
    CointPSurface {
        star: -1.53,
        min: -19.62,
        max: 1.51,
        small_p: [1.9129, 1.3857, 0.035322],
        large_p: [1.5578, 0.8558, -0.20830000000000004, -0.033549],
    },
    // N = 3
    CointPSurface {
        star: -2.68,
        min: -21.21,
        max: 0.86,
        small_p: [2.7648, 1.4502, 0.034186],
        large_p: [2.2268, 0.68093, -0.32362, -0.054447999999999996],
    },
    // N = 4
    CointPSurface {
        star: -3.09,
        min: -23.25,
        max: 0.88,
        small_p: [3.4336, 1.4835, 0.0319],
        large_p: [2.7654, 0.64502, -0.30811000000000005, -0.044946],
    },
    // N = 5
    CointPSurface {
        star: -3.07,
        min: -21.63,
        max: 1.05,
        small_p: [4.0999, 1.5533, 0.0359],
        large_p: [3.2684, 0.6805100000000001, -0.26778, -0.034971999999999996],
    },
    // N = 6
    CointPSurface {
        star: -3.77,
        min: -25.74,
        max: 1.24,
        small_p: [4.5388, 1.5344, 0.029807],
        large_p: [3.7268, 0.7167, -0.23648, -0.028288000000000004],
    },
];

const COINT_P_C: [CointPSurface; 5] = [
    // N = 2
    CointPSurface {
        star: -2.62,
        min: -18.86,
        max: 0.92,
        small_p: [2.92, 1.5012, 0.039796],
        large_p: [2.1945, 0.64695, -0.29198, -0.042377000000000005],
    },
    // N = 3
    CointPSurface {
        star: -3.13,
        min: -23.48,
        max: 0.55,
        small_p: [3.4699, 1.4856, 0.03164],
        large_p: [2.5893, 0.45168, -0.36529, -0.050074],
    },
    // N = 4
    CointPSurface {
        star: -3.47,
        min: -28.07,
        max: 0.61,
        small_p: [3.9673, 1.4777, 0.026315],
        large_p: [3.0387, 0.45452000000000004, -0.33666, -0.041921],
    },
    // N = 5
    CointPSurface {
        star: -3.78,
        min: -25.96,
        max: 0.79,
        small_p: [4.5509, 1.5338, 0.029545],
        large_p: [3.5049, 0.5209800000000001, -0.29158, -0.033468],
    },
    // N = 6
    CointPSurface {
        star: -3.93,
        min: -23.27,
        max: 1.0,
        small_p: [5.1399, 1.6036, 0.034445],
        large_p: [3.9489, 0.58933, -0.25359, -0.02721],
    },
];

const COINT_P_CT: [CointPSurface; 5] = [
    // N = 2
    CointPSurface {
        star: -3.19,
        min: -21.15,
        max: 0.63,
        small_p: [3.6646, 1.5419, 0.036448],
        large_p: [2.85, 0.5272, -0.36622, -0.051695000000000005],
    },
    // N = 3
    CointPSurface {
        star: -3.5,
        min: -25.37,
        max: 0.71,
        small_p: [4.0983, 1.5173, 0.029897999999999997],
        large_p: [3.221, 0.5255, -0.32685000000000003, -0.041501],
    },
    // N = 4
    CointPSurface {
        star: -3.65,
        min: -26.63,
        max: 0.93,
        small_p: [4.5844, 1.5338, 0.028796],
        large_p: [3.652, 0.59758, -0.27483, -0.032081],
    },
    // N = 5
    CointPSurface {
        star: -3.8,
        min: -26.53,
        max: 1.19,
        small_p: [5.0722, 1.5634, 0.029472],
        large_p: [4.0712, 0.6642800000000001, -0.23464000000000002, -0.02546],
    },
    // N = 6
    CointPSurface {
        star: -4.36,
        min: -26.18,
        max: 1.42,
        small_p: [5.53, 1.5914, 0.030392000000000002],
        large_p: [4.4735, 0.71757, -0.20681, -0.021196000000000003],
    },
];

fn coint_p_table(trend: PoTrend) -> &'static [CointPSurface; 5] {
    match trend {
        PoTrend::None => &COINT_P_N,
        PoTrend::Constant => &COINT_P_C,
        PoTrend::ConstantTrend => &COINT_P_CT,
    }
}

/// MacKinnon (1994) cointegration p-value for a residual-test statistic
/// (Engle-Granger tau or Phillips-Ouliaris `Zt`) with `N` stochastic
/// dimensions.
///
/// Matches `statsmodels.tsa.adfvalues.mackinnonp(stat, regression=trend,
/// N)` for `2 <= N <= 6`. Returns NaN outside that range (the published
/// tables stop at N = 6).
pub(crate) fn mackinnon_coint_p(stat: f64, trend: PoTrend, n_vars: usize) -> f64 {
    if !(2..=6).contains(&n_vars) {
        return f64::NAN;
    }
    let s = &coint_p_table(trend)[n_vars - 2];
    if stat > s.max {
        return 1.0;
    }
    if stat < s.min {
        return 0.0;
    }
    let g = if stat <= s.star {
        polyval_ascending(&s.small_p, stat)
    } else {
        polyval_ascending(&s.large_p, stat)
    };
    StdNormal.cdf(g)
}

/// MacKinnon (2010) `1/n` cointegration critical-value surfaces, constant
/// case, rows = N = 2..12, each `[1%, 5%, 10%] x [b0, b1, b2, b3]`.
const COINT_CRIT_C: [[[f64; 4]; 3]; 11] = [
    // N = 2
    [
        [-3.89644, -10.9519, -33.527, 0.0],
        [-3.33613, -6.1101, -6.823, 0.0],
        [-3.04445, -4.2412, -2.72, 0.0],
    ],
    // N = 3
    [
        [-4.29374, -14.4354, -33.195, 47.433],
        [-3.74066, -8.5632, -10.852, 27.982],
        [-3.45218, -6.2143, -3.718, 0.0],
    ],
    // N = 4
    [
        [-4.64332, -18.1031, -37.972, 0.0],
        [-4.096, -11.2349, -11.175, 0.0],
        [-3.8102, -8.3931, -4.137, 0.0],
    ],
    // N = 5
    [
        [-4.95756, -21.8883, -45.142, 0.0],
        [-4.41519, -14.0405, -12.575, 0.0],
        [-4.13157, -10.7417, -3.784, 0.0],
    ],
    // N = 6
    [
        [-5.24568, -25.6688, -57.737, 88.639],
        [-4.70693, -16.9178, -17.492, 60.007],
        [-4.42501, -13.1875, -5.104, 27.877],
    ],
    // N = 7
    [
        [-5.51233, -29.576, -69.398, 164.295],
        [-4.97684, -19.9021, -22.045, 110.761],
        [-4.69648, -15.7315, -5.104, 27.877],
    ],
    // N = 8
    [
        [-5.76202, -33.5258, -82.189, 256.289],
        [-5.22924, -23.0023, -24.646, 144.479],
        [-4.95007, -18.3959, -7.344, 94.872],
    ],
    // N = 9
    [
        [-5.99742, -37.6572, -87.365, 248.316],
        [-5.46697, -26.2057, -26.627, 176.382],
        [-5.18897, -21.1377, -9.484, 172.704],
    ],
    // N = 10
    [
        [-6.22103, -41.7154, -102.68, 389.33],
        [-5.69244, -29.4521, -30.994, 251.016],
        [-5.41533, -24.0006, -7.514, 163.049],
    ],
    // N = 11
    [
        [-6.43377, -46.0084, -106.809, 352.752],
        [-5.90714, -32.8336, -30.275, 249.994],
        [-5.63086, -26.9693, -4.083, 151.427],
    ],
    // N = 12
    [
        [-6.6379, -50.2095, -124.156, 579.622],
        [-6.11279, -36.2681, -32.505, 314.802],
        [-5.83724, -29.9864, -2.686, 184.116],
    ],
];

/// MacKinnon (2010) `1/n` cointegration critical-value surfaces,
/// constant-trend case, rows = N = 2..12.
const COINT_CRIT_CT: [[[f64; 4]; 3]; 11] = [
    // N = 2
    [
        [-4.32762, -15.4387, -35.679, 0.0],
        [-3.78057, -9.5106, -12.074, 0.0],
        [-3.49631, -7.0815, -7.538, 21.892],
    ],
    // N = 3
    [
        [-4.66305, -18.7688, -49.793, 104.244],
        [-4.1189, -11.8922, -19.031, 77.332],
        [-3.83511, -9.0723, -8.504, 35.403],
    ],
    // N = 4
    [
        [-4.9694, -22.4694, -52.599, 51.314],
        [-4.42871, -14.5876, -18.228, 39.647],
        [-4.14633, -11.25, -9.873, 54.109],
    ],
    // N = 5
    [
        [-5.25276, -26.2183, -59.631, 50.646],
        [-4.71537, -17.3569, -22.66, 91.359],
        [-4.43422, -13.6078, -10.238, 76.781],
    ],
    // N = 6
    [
        [-5.51727, -29.976, -75.222, 202.253],
        [-4.98228, -20.305, -25.224, 132.03],
        [-4.70233, -16.1253, -9.836, 94.272],
    ],
    // N = 7
    [
        [-5.76537, -33.9165, -84.312, 245.394],
        [-5.23299, -23.3328, -28.955, 182.342],
        [-4.95405, -18.7352, -10.168, 120.575],
    ],
    // N = 8
    [
        [-6.00003, -37.8892, -96.428, 335.92],
        [-5.46971, -26.4771, -31.034, 220.165],
        [-5.19183, -21.4328, -10.726, 157.955],
    ],
    // N = 9
    [
        [-6.22288, -41.9496, -109.881, 466.068],
        [-5.69447, -29.7152, -33.784, 273.002],
        [-5.41738, -24.2882, -8.584, 169.891],
    ],
    // N = 10
    [
        [-6.43551, -46.1151, -120.814, 566.823],
        [-5.90887, -33.0251, -37.208, 346.189],
        [-5.63255, -27.2042, -6.792, 177.666],
    ],
    // N = 11
    [
        [-6.63894, -50.4287, -128.997, 642.781],
        [-6.11404, -36.461, -36.246, 348.554],
        [-5.8385, -30.1995, -5.163, 210.338],
    ],
    // N = 12
    [
        [-6.83488, -54.7119, -139.8, 736.376],
        [-6.31127, -39.9676, -37.021, 406.051],
        [-6.0365, -33.2381, -6.606, 317.776],
    ],
];

/// MacKinnon (2010) cointegration critical values at the 1% / 5% / 10%
/// levels for `N` stochastic dimensions and sample size `nobs`.
///
/// Matches `statsmodels.tsa.adfvalues.mackinnoncrit(N, regression=trend,
/// nobs)` for `2 <= N <= 12` in the constant / constant-trend cases.
/// Returns `None` when no published 2010 surface exists: `N < 2`,
/// `N > 12`, or the no-constant (`n`) case for `N > 1` (which
/// `statsmodels.tsa.stattools.coint` also reports as unavailable).
pub(crate) fn mackinnon_coint_crit(
    trend: PoTrend,
    n_vars: usize,
    nobs: usize,
) -> Option<AdfCriticalValues> {
    if !(2..=12).contains(&n_vars) {
        return None;
    }
    let table = match trend {
        PoTrend::Constant => &COINT_CRIT_C,
        PoTrend::ConstantTrend => &COINT_CRIT_CT,
        PoTrend::None => return None,
    };
    let rows = &table[n_vars - 2];
    let x = 1.0 / nobs as f64;
    Some(AdfCriticalValues {
        pct1: polyval_ascending(&rows[0], x),
        pct5: polyval_ascending(&rows[1], x),
        pct10: polyval_ascending(&rows[2], x),
    })
}

#[cfg(test)]
mod tests {
    //! Directly pin the transcribed ADF-z and cointegration surfaces
    //! against `fixtures/phillips.json` (arch 8.0.0 / statsmodels 0.14.6).
    //! These `pub(crate)` maps are unreachable from an integration test, so
    //! the transcription is pinned here in-crate.

    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/phillips.json");
        let text = std::fs::read_to_string(path).expect("fixture file readable");
        serde_json::from_str(&text).expect("fixture is valid JSON")
    }

    fn f64s(v: &Value) -> Vec<f64> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().expect("number"))
            .collect()
    }

    fn adf_reg(code: &str) -> AdfRegression {
        match code {
            "n" => AdfRegression::NoConstant,
            "c" => AdfRegression::Constant,
            "ct" => AdfRegression::ConstantTrend,
            other => panic!("unknown regression {other:?}"),
        }
    }

    fn po_trend(code: &str) -> PoTrend {
        match code {
            "n" => PoTrend::None,
            "c" => PoTrend::Constant,
            "ct" => PoTrend::ConstantTrend,
            other => panic!("unknown trend {other:?}"),
        }
    }

    /// Combined abs+rel p-value comparison (numpy `allclose`): the deep tail
    /// reaches where tsecon-stats vs scipy CDFs differ below the absolute
    /// floor, while moderate p-values are pinned relatively at 1e-8.
    fn assert_pclose(actual: f64, expected: f64, ctx: &str) {
        let tol = 1e-12 + 1e-8 * expected.abs();
        assert!(
            (actual - expected).abs() <= tol,
            "{ctx}: actual {actual}, expected {expected}, |diff| {:e} > {tol:e}",
            (actual - expected).abs()
        );
    }

    fn assert_rel(actual: f64, expected: f64, ctx: &str) {
        let rel = if expected == 0.0 {
            actual.abs()
        } else {
            ((actual - expected) / expected).abs()
        };
        assert!(
            rel <= 1e-8,
            "{ctx}: actual {actual}, expected {expected}, rel {rel:e}"
        );
    }

    #[test]
    fn adf_z_p_and_crit_match_arch() {
        let fx = fixture();
        let block = &fx["adf_z_map"];
        for code in ["n", "c", "ct"] {
            let reg = adf_reg(code);
            let stats = f64s(&block[code]["stat_grid"]);
            let pvals = f64s(&block[code]["pvalues"]);
            assert_eq!(stats.len(), pvals.len());
            for (&s, &p) in stats.iter().zip(&pvals) {
                assert_pclose(mackinnon_z_p(s, reg), p, &format!("adf-z p[{code}]({s})"));
            }
            let nobs = block[code]["nobs"].as_u64().unwrap() as usize;
            let expected = f64s(&block[code]["crit"]);
            let cv = mackinnon_z_crit(reg, nobs);
            assert_rel(cv.pct1, expected[0], &format!("adf-z crit[{code}] 1%"));
            assert_rel(cv.pct5, expected[1], &format!("adf-z crit[{code}] 5%"));
            assert_rel(cv.pct10, expected[2], &format!("adf-z crit[{code}] 10%"));
        }
    }

    #[test]
    fn coint_p_matches_statsmodels() {
        let fx = fixture();
        let block = &fx["coint_p_map"];
        for code in ["n", "c", "ct"] {
            let trend = po_trend(code);
            for n in 2..=6usize {
                let entry = &block[code][n.to_string()];
                let stats = f64s(&entry["stat_grid"]);
                let pvals = f64s(&entry["pvalues"]);
                for (&s, &p) in stats.iter().zip(&pvals) {
                    assert_pclose(
                        mackinnon_coint_p(s, trend, n),
                        p,
                        &format!("coint p[{code}/N{n}]({s})"),
                    );
                }
            }
        }
        // Out-of-range N yields NaN.
        assert!(mackinnon_coint_p(-3.0, PoTrend::Constant, 1).is_nan());
        assert!(mackinnon_coint_p(-3.0, PoTrend::Constant, 7).is_nan());
    }

    #[test]
    fn coint_crit_matches_statsmodels() {
        let fx = fixture();
        let block = &fx["coint_crit_map"];
        for code in ["c", "ct"] {
            let trend = po_trend(code);
            for n in 2..=12usize {
                let entry = &block[code][n.to_string()];
                let nobs = entry["nobs"].as_u64().unwrap() as usize;
                let expected = f64s(&entry["crit"]);
                let cv = mackinnon_coint_crit(trend, n, nobs).expect("crit present");
                assert_rel(cv.pct1, expected[0], &format!("coint crit[{code}/N{n}] 1%"));
                assert_rel(cv.pct5, expected[1], &format!("coint crit[{code}/N{n}] 5%"));
                assert_rel(
                    cv.pct10,
                    expected[2],
                    &format!("coint crit[{code}/N{n}] 10%"),
                );
            }
        }
        // No 2010 surface: no-constant N>1, and out-of-range N.
        assert!(mackinnon_coint_crit(PoTrend::None, 2, 100).is_none());
        assert!(mackinnon_coint_crit(PoTrend::Constant, 13, 100).is_none());
        assert!(mackinnon_coint_crit(PoTrend::Constant, 1, 100).is_none());
    }
}
