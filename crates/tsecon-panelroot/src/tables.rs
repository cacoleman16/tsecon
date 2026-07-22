//! Transcribed moment / bias-adjustment tables for the IPS and LLC tests.
//!
//! THIS IS THE ONLY module in the crate where numbers are hard-coded. The
//! two families are:
//!
//! * [`IPS_E_C`] / [`IPS_V_C`] / [`IPS_E_CT`] / [`IPS_V_CT`] — the mean and
//!   variance of the per-unit ADF t-statistic `t_{iT}(p, 0)` from
//!   Im-Pesaran-Shin (2003, *J. Econometrics* 115, Table 3), for the
//!   intercept (`_C`) and intercept-plus-trend (`_CT`) cases. Rows index the
//!   lag augmentation `p = 0, 1, ..., 8`; columns index the sample size on
//!   the grid [`IPS_T`] = {10, 15, 20, 25, 30, 40, 50, 60, 70, 100}. Cells
//!   that the paper leaves blank (short samples with heavy augmentation) are
//!   stored as `NaN`.
//!
//! * [`LLC_MU`] / [`LLC_SIGMA`] — the mean (`mu*`) and standard-deviation
//!   (`sigma*`) adjustments of Levin-Lin-Chu (2002, *J. Econometrics* 108,
//!   Table 2) for the no-constant, intercept, and intercept-plus-trend cases
//!   (index 0/1/2), over the grid [`LLC_T`] = {25, 30, 35, 40, 45, 50, 60,
//!   70, 80, 90, 100, 250, 500}.
//!
//! Provenance and cross-check: every cell was transcribed from the values
//! carried by the R package `plm` (`purtest`'s internal `adj.ips.wtbar` and
//! `adj.levinlin` objects, themselves transcribed from the two papers). The
//! crate's Python fixture generator embeds the identical numbers and the
//! Rust golden reproduces `plm::purtest`'s `Wtbar` and `levinlin` statistics
//! to floating-point precision (see `tests/golden.rs`). VERIFY CELL-BY-CELL
//! against the published tables before trusting a change here.

use tsecon_diag::AdfRegression;

/// Sample-size grid for the Im-Pesaran-Shin (2003) Table 3 moments.
pub(crate) const IPS_T: [f64; 10] = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 100.0];

const NA: f64 = f64::NAN;

/// IPS (2003) Table 3, `E[t_{iT}(p, 0)]`, intercept case. Rows `p = 0..8`.
pub(crate) const IPS_E_C: [[f64; 10]; 9] = [
    [
        -1.504, -1.514, -1.522, -1.520, -1.526, -1.523, -1.527, -1.519, -1.524, -1.532,
    ],
    [
        -1.488, -1.503, -1.516, -1.514, -1.519, -1.520, -1.524, -1.519, -1.522, -1.530,
    ],
    [
        -1.319, -1.387, -1.428, -1.443, -1.460, -1.476, -1.493, -1.490, -1.498, -1.514,
    ],
    [
        -1.306, -1.366, -1.413, -1.433, -1.453, -1.471, -1.489, -1.486, -1.495, -1.512,
    ],
    [
        -1.171, -1.260, -1.329, -1.363, -1.394, -1.428, -1.454, -1.458, -1.470, -1.495,
    ],
    [
        NA, NA, -1.313, -1.351, -1.384, -1.421, -1.451, -1.454, -1.467, -1.494,
    ],
    [
        NA, NA, NA, -1.289, -1.331, -1.380, -1.418, -1.427, -1.444, -1.476,
    ],
    [
        NA, NA, NA, -1.273, -1.319, -1.371, -1.411, -1.423, -1.441, -1.474,
    ],
    [
        NA, NA, NA, -1.212, -1.266, -1.329, -1.377, -1.393, -1.415, -1.456,
    ],
];

/// IPS (2003) Table 3, `Var[t_{iT}(p, 0)]`, intercept case. Rows `p = 0..8`.
pub(crate) const IPS_V_C: [[f64; 10]; 9] = [
    [
        1.069, 0.923, 0.851, 0.809, 0.789, 0.770, 0.760, 0.749, 0.736, 0.735,
    ],
    [
        1.255, 1.011, 0.915, 0.861, 0.831, 0.803, 0.781, 0.770, 0.753, 0.745,
    ],
    [
        1.421, 1.078, 0.969, 0.905, 0.865, 0.830, 0.798, 0.789, 0.766, 0.754,
    ],
    [
        1.759, 1.181, 1.037, 0.952, 0.907, 0.858, 0.819, 0.802, 0.782, 0.761,
    ],
    [
        2.080, 1.279, 1.097, 1.005, 0.946, 0.886, 0.842, 0.819, 0.801, 0.771,
    ],
    [
        NA, NA, 1.171, 1.055, 0.980, 0.912, 0.863, 0.839, 0.814, 0.781,
    ],
    [NA, NA, NA, 1.114, 1.023, 0.942, 0.886, 0.858, 0.834, 0.795],
    [NA, NA, NA, 1.164, 1.062, 0.968, 0.910, 0.875, 0.851, 0.806],
    [NA, NA, NA, 1.217, 1.105, 0.996, 0.929, 0.896, 0.871, 0.818],
];

/// IPS (2003) Table 3, `E[t_{iT}(p, 0)]`, intercept-plus-trend case.
pub(crate) const IPS_E_CT: [[f64; 10]; 9] = [
    [
        -2.166, -2.167, -2.168, -2.167, -2.172, -2.173, -2.176, -2.174, -2.174, -2.177,
    ],
    [
        -2.173, -2.169, -2.172, -2.172, -2.173, -2.177, -2.180, -2.178, -2.176, -2.179,
    ],
    [
        -1.914, -1.999, -2.047, -2.074, -2.095, -2.120, -2.137, -2.143, -2.146, -2.158,
    ],
    [
        -1.922, -1.977, -2.032, -2.065, -2.091, -2.117, -2.137, -2.142, -2.146, -2.158,
    ],
    [
        -1.750, -1.823, -1.911, -1.968, -2.009, -2.057, -2.091, -2.103, -2.114, -2.135,
    ],
    [
        NA, NA, -1.888, -1.955, -1.998, -2.051, -2.087, -2.101, -2.111, -2.135,
    ],
    [
        NA, NA, NA, -1.868, -1.923, -1.995, -2.042, -2.065, -2.081, -2.113,
    ],
    [
        NA, NA, NA, -1.851, -1.912, -1.986, -2.036, -2.063, -2.079, -2.112,
    ],
    [
        NA, NA, NA, -1.761, -1.835, -1.925, -1.987, -2.024, -2.046, -2.088,
    ],
];

/// IPS (2003) Table 3, `Var[t_{iT}(p, 0)]`, intercept-plus-trend case.
pub(crate) const IPS_V_CT: [[f64; 10]; 9] = [
    [
        1.132, 0.869, 0.763, 0.713, 0.690, 0.655, 0.633, 0.621, 0.610, 0.597,
    ],
    [
        1.453, 0.975, 0.845, 0.769, 0.734, 0.687, 0.654, 0.641, 0.627, 0.605,
    ],
    [
        1.627, 1.036, 0.882, 0.796, 0.756, 0.702, 0.661, 0.653, 0.634, 0.613,
    ],
    [
        2.482, 1.214, 0.983, 0.861, 0.808, 0.735, 0.688, 0.674, 0.650, 0.625,
    ],
    [
        3.947, 1.332, 1.052, 0.913, 0.845, 0.759, 0.705, 0.685, 0.662, 0.629,
    ],
    [
        NA, NA, 1.165, 0.991, 0.899, 0.792, 0.730, 0.705, 0.673, 0.638,
    ],
    [NA, NA, NA, 1.055, 0.945, 0.828, 0.753, 0.725, 0.689, 0.650],
    [NA, NA, NA, 1.145, 1.009, 0.872, 0.786, 0.747, 0.713, 0.661],
    [NA, NA, NA, 1.208, 1.063, 0.902, 0.808, 0.766, 0.728, 0.670],
];

/// Sample-size grid for the Levin-Lin-Chu (2002) Table 2 adjustments.
pub(crate) const LLC_T: [f64; 13] = [
    25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 250.0, 500.0,
];

/// LLC (2002) Table 2 mean adjustment `mu*`, by case (0 = none, 1 = intercept,
/// 2 = intercept + trend).
pub(crate) const LLC_MU: [[f64; 13]; 3] = [
    [
        0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    ],
    [
        -0.554, -0.546, -0.541, -0.537, -0.533, -0.531, -0.527, -0.524, -0.521, -0.520, -0.518,
        -0.509, -0.500,
    ],
    [
        -0.703, -0.674, -0.653, -0.637, -0.624, -0.614, -0.598, -0.587, -0.578, -0.571, -0.566,
        -0.533, -0.500,
    ],
];

/// LLC (2002) Table 2 standard-deviation adjustment `sigma*`, by case
/// (0 = none, 1 = intercept, 2 = intercept + trend).
pub(crate) const LLC_SIGMA: [[f64; 13]; 3] = [
    [
        1.049, 1.035, 1.027, 1.021, 1.017, 1.014, 1.011, 1.008, 1.007, 1.006, 1.005, 1.001, 1.000,
    ],
    [
        0.919, 0.889, 0.867, 0.850, 0.837, 0.826, 0.810, 0.798, 0.789, 0.782, 0.776, 0.742, 0.707,
    ],
    [
        1.003, 0.949, 0.906, 0.871, 0.842, 0.818, 0.780, 0.751, 0.728, 0.710, 0.695, 0.603, 0.500,
    ],
];

/// Clamped piecewise-linear interpolation of `ys` (defined on the ascending
/// grid `xs`) at `x`. Below the grid returns `ys[0]`, above it returns the
/// last value — the `selectT` convention of `plm::purtest`. `xs` and `ys`
/// must be the same non-empty length and contain no `NaN`.
fn clamp_interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let last = xs.len() - 1;
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[last] {
        return ys[last];
    }
    for i in 1..xs.len() {
        if x < xs[i] {
            let w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return ys[i - 1] + w * (ys[i] - ys[i - 1]);
        }
    }
    ys[last]
}

/// Interpolate a single Table 3 row in the sample size, skipping the blank
/// (`NaN`) cells that the paper leaves for short samples with heavy
/// augmentation: the interpolation runs over the sub-grid of tabulated
/// columns only (so a very short, heavily augmented unit clamps to the
/// smallest tabulated `T` for its lag rather than returning `NaN`).
fn interp_row(l: f64, row: &[f64; 10]) -> f64 {
    let mut xs: Vec<f64> = Vec::with_capacity(10);
    let mut ys: Vec<f64> = Vec::with_capacity(10);
    for (t, &v) in IPS_T.iter().zip(row.iter()) {
        if v.is_finite() {
            xs.push(*t);
            ys.push(v);
        }
    }
    clamp_interp(l, &xs, &ys)
}

/// The IPS (2003) standardization moments `(E[t], Var[t])` for a unit whose
/// ADF regression used `p` lag augmentations, had effective sample size `l =
/// T - p - 1`, and the deterministic case `trend` (false = intercept, true =
/// intercept + trend). `p` is clamped to `0..=8` (the tabulated range).
pub(crate) fn ips_moments(l: f64, p: usize, trend: bool) -> (f64, f64) {
    let p = p.min(8);
    let (e_tab, v_tab) = if trend {
        (&IPS_E_CT, &IPS_V_CT)
    } else {
        (&IPS_E_C, &IPS_V_C)
    };
    (interp_row(l, &e_tab[p]), interp_row(l, &v_tab[p]))
}

/// The LLC (2002) Table 2 bias adjustments `(mu*, sigma*)` for the common
/// sample size `t` and deterministic specification.
pub(crate) fn llc_adj(t: f64, regression: AdfRegression) -> (f64, f64) {
    let idx = match regression {
        AdfRegression::NoConstant => 0,
        AdfRegression::Constant => 1,
        AdfRegression::ConstantTrend => 2,
    };
    (
        clamp_interp(t, &LLC_T, &LLC_MU[idx]),
        clamp_interp(t, &LLC_T, &LLC_SIGMA[idx]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ips_exact_grid_point() {
        // Row p = 1, intercept, at T = 50 (a tabulated column): exact cell.
        let (e, v) = ips_moments(50.0, 1, false);
        assert_eq!(e, -1.524);
        assert_eq!(v, 0.781);
    }

    #[test]
    fn ips_linear_interpolation_matches_hand_value() {
        // p = 1, intercept, l = 48 -> between T = 40 (E = -1.520, V = 0.803)
        // and T = 50 (E = -1.524, V = 0.781), weight 0.8.
        let (e, v) = ips_moments(48.0, 1, false);
        let expect_e = -1.520 + 0.8 * (-1.524 - (-1.520));
        let expect_v = 0.803 + 0.8 * (0.781 - 0.803);
        assert!((e - expect_e).abs() < 1e-12);
        assert!((v - expect_v).abs() < 1e-12);
    }

    #[test]
    fn ips_clamps_below_and_above_grid() {
        let (e_lo, _) = ips_moments(5.0, 0, false);
        assert_eq!(e_lo, IPS_E_C[0][0]); // clamp to T = 10
        let (e_hi, _) = ips_moments(1000.0, 0, false);
        assert_eq!(e_hi, IPS_E_C[0][9]); // clamp to T = 100
    }

    #[test]
    fn ips_skips_na_cells_for_short_heavy_augmentation() {
        // p = 8 has NaN for T <= 20; l = 12 must clamp to the first
        // tabulated column (T = 25), not return NaN.
        let (e, v) = ips_moments(12.0, 8, false);
        assert!(e.is_finite() && v.is_finite());
        assert_eq!(e, IPS_E_C[8][3]); // T = 25 column
    }

    #[test]
    fn llc_exact_and_clamped() {
        let (mu, sg) = llc_adj(50.0, AdfRegression::ConstantTrend);
        assert_eq!(mu, -0.614);
        assert_eq!(sg, 0.818);
        let (mu_lo, sg_lo) = llc_adj(10.0, AdfRegression::Constant);
        assert_eq!(mu_lo, -0.554); // clamp to T = 25
        assert_eq!(sg_lo, 0.919);
        let (mu_hi, sg_hi) = llc_adj(9999.0, AdfRegression::NoConstant);
        assert_eq!(mu_hi, 0.000); // clamp to T = 500
        assert_eq!(sg_hi, 1.000);
    }
}
