//! Tabulated asymptotic critical values for the Johansen trace and
//! maximum-eigenvalue statistics.
//!
//! These are the MacKinnon-Haug-Michelis (1999) response-surface values
//! shipped by statsmodels (`statsmodels.tsa.coint_tables`, arrays
//! `tjcp*` / `ejcp*`), generated with MacKinnon's `johdist` program. They
//! are *constants*, not computed, and are reproduced verbatim so the test
//! layer can compare its statistics without a runtime dependency on a
//! p-value surface. Each row `n = 1 .. 12` gives the `[90%, 95%, 99%]`
//! percentiles for a system with `n` common stochastic trends (i.e. null
//! rank `k - n`).

/// Order of the deterministic polynomial assumed under the null, following
/// the statsmodels `det_order` convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetOrder {
    /// No deterministic term (`det_order = -1`).
    NoDeterministic,
    /// Constant in the data (`det_order = 0`) — the case validated against
    /// the golden fixture.
    Constant,
    /// Constant plus linear trend (`det_order = 1`).
    LinearTrend,
}

/// Trace critical values, `det_order = -1` (statsmodels `tjcp0`).
const TRACE_NONE: [[f64; 3]; 12] = [
    [2.9762, 4.1296, 6.9406],
    [10.4741, 12.3212, 16.3640],
    [21.7781, 24.2761, 29.5147],
    [37.0339, 40.1749, 46.5716],
    [56.2839, 60.0627, 67.6367],
    [79.5329, 83.9383, 92.7136],
    [106.7351, 111.7797, 121.7375],
    [137.9954, 143.6691, 154.7977],
    [173.2292, 179.5199, 191.8122],
    [212.4721, 219.4051, 232.8291],
    [255.6732, 263.2603, 277.9962],
    [302.9054, 311.1288, 326.9716],
];

/// Trace critical values, `det_order = 0` (statsmodels `tjcp1`).
const TRACE_CONST: [[f64; 3]; 12] = [
    [2.7055, 3.8415, 6.6349],
    [13.4294, 15.4943, 19.9349],
    [27.0669, 29.7961, 35.4628],
    [44.4929, 47.8545, 54.6815],
    [65.8202, 69.8189, 77.8202],
    [91.1090, 95.7542, 104.9637],
    [120.3673, 125.6185, 135.9825],
    [153.6341, 159.5290, 171.0905],
    [190.8714, 197.3772, 210.0366],
    [232.1030, 239.2468, 253.2526],
    [277.3740, 285.1402, 300.2821],
    [326.5354, 334.9795, 351.2150],
];

/// Trace critical values, `det_order = 1` (statsmodels `tjcp2`).
const TRACE_TREND: [[f64; 3]; 12] = [
    [2.7055, 3.8415, 6.6349],
    [16.1619, 18.3985, 23.1485],
    [32.0645, 35.0116, 41.0815],
    [51.6492, 55.2459, 62.5202],
    [75.1027, 79.3422, 87.7748],
    [102.4674, 107.3429, 116.9829],
    [133.7852, 139.2780, 150.0778],
    [169.0618, 175.1584, 187.1891],
    [208.3582, 215.1268, 228.2226],
    [251.6293, 259.0267, 273.3838],
    [298.8836, 306.8988, 322.4264],
    [350.1125, 358.7190, 375.3203],
];

/// Maximum-eigenvalue critical values, `det_order = -1` (statsmodels
/// `ejcp0`).
const MAXEIG_NONE: [[f64; 3]; 12] = [
    [2.9762, 4.1296, 6.9406],
    [9.4748, 11.2246, 15.0923],
    [15.7175, 17.7961, 22.2519],
    [21.8370, 24.1592, 29.0609],
    [27.9160, 30.4428, 35.7359],
    [33.9271, 36.6301, 42.2333],
    [39.9085, 42.7679, 48.6606],
    [45.8930, 48.8795, 55.0335],
    [51.8528, 54.9629, 61.3449],
    [57.7954, 61.0404, 67.6415],
    [63.7248, 67.0756, 73.8856],
    [69.6513, 73.0946, 80.0937],
];

/// Maximum-eigenvalue critical values, `det_order = 0` (statsmodels
/// `ejcp1`).
const MAXEIG_CONST: [[f64; 3]; 12] = [
    [2.7055, 3.8415, 6.6349],
    [12.2971, 14.2639, 18.5200],
    [18.8928, 21.1314, 25.8650],
    [25.1236, 27.5858, 32.7172],
    [31.2379, 33.8777, 39.3693],
    [37.2786, 40.0763, 45.8662],
    [43.2947, 46.2299, 52.3069],
    [49.2855, 52.3622, 58.6634],
    [55.2412, 58.4332, 64.9960],
    [61.2041, 64.5040, 71.2525],
    [67.1307, 70.5392, 77.4877],
    [73.0563, 76.5734, 83.7105],
];

/// Maximum-eigenvalue critical values, `det_order = 1` (statsmodels
/// `ejcp2`).
const MAXEIG_TREND: [[f64; 3]; 12] = [
    [2.7055, 3.8415, 6.6349],
    [15.0006, 17.1481, 21.7465],
    [21.8731, 24.2522, 29.2631],
    [28.2398, 30.8151, 36.1930],
    [34.4202, 37.1646, 42.8612],
    [40.5244, 43.4183, 49.4095],
    [46.5583, 49.5875, 55.8171],
    [52.5858, 55.7302, 62.1741],
    [58.5316, 61.8051, 68.5030],
    [64.5292, 67.9040, 74.7434],
    [70.4630, 73.9355, 81.0678],
    [76.4081, 79.9878, 87.2395],
];

const NAN_ROW: [f64; 3] = [f64::NAN, f64::NAN, f64::NAN];

/// Returns the `(trace, max_eig)` `[90%, 95%, 99%]` critical-value rows for
/// a system with `n_trends = k - r` common trends under the given
/// deterministic order. Untabulated cases (`n_trends` outside `1 ..= 12`)
/// return rows of NaN, matching statsmodels' out-of-range behaviour.
pub fn critical_values(det: DetOrder, n_trends: usize) -> ([f64; 3], [f64; 3]) {
    if n_trends == 0 || n_trends > 12 {
        return (NAN_ROW, NAN_ROW);
    }
    let idx = n_trends - 1;
    let (trace, maxeig) = match det {
        DetOrder::NoDeterministic => (TRACE_NONE, MAXEIG_NONE),
        DetOrder::Constant => (TRACE_CONST, MAXEIG_CONST),
        DetOrder::LinearTrend => (TRACE_TREND, MAXEIG_TREND),
    };
    (trace[idx], maxeig[idx])
}
