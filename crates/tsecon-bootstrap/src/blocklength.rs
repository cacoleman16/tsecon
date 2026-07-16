//! Automatic block-length selection for the block bootstraps.
//!
//! Implements the Politis-White (2004) plug-in rule with the
//! Patton-Politis-White (2009) correction, following Patton's reference
//! MATLAB implementation (`opt_block_length_REV_dec07`). Block-length
//! selection ships as the library default — users should not have to guess
//! a tuning constant to get valid dependent-data inference.

use crate::error::BootstrapError;

/// Inverse standard normal CDF at 0.975, the critical value in the
/// Politis-White significant-lag search (Patton's Dec-2007 revision; the
/// original 2004 paper used 2.0).
const Z_975: f64 = 1.959_963_984_540_054;

/// Optimal block lengths estimated by [`optimal_block_length`], one per
/// bootstrap flavor (they differ only in the variance constant `D`).
///
/// Values are real-valued; callers round as appropriate (`ceil` is the
/// conservative choice for [`crate::BlockScheme::CircularBlock`] /
/// [`crate::BlockScheme::MovingBlock`], and the stationary bootstrap can
/// use `p = 1 / stationary` directly without rounding).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimalBlockLength {
    /// Optimal expected block length for the stationary bootstrap
    /// (Politis-Romano 1994), i.e. use restart probability `1/stationary`.
    pub stationary: f64,
    /// Optimal block length for the circular block bootstrap
    /// (Politis-Romano 1992); also the standard choice for the
    /// moving-block bootstrap.
    pub circular: f64,
}

/// Automatic block-length selection: the Politis-White (2004) plug-in rule
/// with the Patton-Politis-White (2009) corrected variance constant.
///
/// The estimator minimizes the asymptotic MSE of the bootstrap variance of
/// the sample mean. With biased autocovariances
/// `R(k) = (1/n) sum_t (x_t - xbar)(x_{t+k} - xbar)` and the flat-top lag
/// window `lambda` of Politis-Romano (1995),
///
/// ```text
/// G    = sum_{|k| <= M} lambda(k/M) |k| R(k)
/// gbar = sum_{|k| <= M} lambda(k/M) R(k)          (flat-top LRV estimate)
/// D_SB = 2 gbar^2                                  (PPW 2009 correction)
/// D_CB = (4/3) gbar^2
/// b*   = (2 G^2 / D)^(1/3) n^(1/3)
/// ```
///
/// The bandwidth `M = 2 m_hat` is data-driven: `m_hat` is the smallest lag
/// followed by `K_n = max(5, ceil(sqrt(log10 n)))` consecutive sample
/// autocorrelations that are all insignificant at the two-sided 95% level
/// under the implied-white-noise band `|rho_k| < z_{.975} sqrt(log10(n)/n)`;
/// if no such run exists, `m_hat` falls back to the largest significant
/// lag. `M` is capped at `m_max = ceil(sqrt(n)) + K_n` and the returned
/// lengths are clamped to `[1, ceil(min(3 sqrt(n), n/3))]`, all per
/// Patton's reference implementation.
///
/// # Errors
///
/// - [`BootstrapError::EmptySample`] if `x` is empty.
/// - [`BootstrapError::NonFiniteData`] if `x` contains NaN or infinities.
/// - [`BootstrapError::SampleTooShort`] if `n < m_max + 1` so the required
///   autocovariances do not exist (n >= 9 suffices for all n < 10^25).
/// - [`BootstrapError::DegenerateSeries`] if the sample variance or the
///   flat-top long-run-variance estimate is zero, which would divide by
///   zero in the plug-in formula.
///
/// # References
///
/// Politis and White (2004), *Econometric Reviews* 23(1), 53-70;
/// Patton, Politis and White (2009), *Econometric Reviews* 28(4), 372-375;
/// Politis and Romano (1995), *J. Time Series Analysis* 16, 67-103
/// (flat-top window).
pub fn optimal_block_length(x: &[f64]) -> Result<OptimalBlockLength, BootstrapError> {
    let n = x.len();
    if n == 0 {
        return Err(BootstrapError::EmptySample);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(BootstrapError::NonFiniteData);
    }
    let nf = n as f64;

    // K_n and the maximal bandwidth, per Patton's implementation.
    // K_n = 5 for every n below ~10^25, but keep the formula for fidelity.
    let kn = (nf.log10().sqrt().ceil()).max(5.0) as usize;
    let m_max = nf.sqrt().ceil() as usize + kn;
    if m_max + 1 > n {
        return Err(BootstrapError::SampleTooShort {
            n,
            required: m_max + 1,
        });
    }
    let b_max = (3.0 * nf.sqrt()).min(nf / 3.0).ceil();

    let acov = autocovariances(x, m_max);
    let gamma0 = acov[0];
    // gamma0 is a mean of squares of finite values: >= 0 and never NaN.
    if gamma0 <= 0.0 {
        return Err(BootstrapError::DegenerateSeries);
    }

    // Significant-lag search: an autocorrelation is "insignificant" when it
    // falls inside the +/- z_{.975} sqrt(log10(n)/n) band.
    let crit = Z_975 * (nf.log10() / nf).sqrt();
    let insignificant: Vec<bool> = acov[1..]
        .iter()
        .map(|g| (g / gamma0).abs() < crit)
        .collect();

    // m_hat: smallest lag m >= 1 such that lags m..m+K_n-1 are all
    // insignificant; else the largest significant lag (Patton's fallback).
    let m_hat = (1..=m_max - kn + 1)
        .find(|&m| insignificant[m - 1..m - 1 + kn].iter().all(|&b| b))
        .or_else(|| (1..=m_max).rev().find(|&k| !insignificant[k - 1]))
        // Unreachable in practice: if no lag is significant the first
        // window qualifies; kept as a safe default rather than a panic.
        .unwrap_or(1);
    let m = (2 * m_hat).min(m_max);

    // Flat-top weighted sums over |k| <= M (symmetric, so fold 2x).
    let mut g_hat = 0.0; // sum lambda(k/M) |k| R(k)
    let mut lrv = gamma0; // sum lambda(k/M) R(k)
    for (k, &r) in acov.iter().enumerate().take(m + 1).skip(1) {
        let lam = flat_top(k as f64 / m as f64);
        g_hat += 2.0 * lam * k as f64 * r;
        lrv += 2.0 * lam * r;
    }

    let d_sb = 2.0 * lrv * lrv;
    let d_cb = (4.0 / 3.0) * lrv * lrv;
    if d_sb <= 0.0 || !d_sb.is_finite() {
        return Err(BootstrapError::DegenerateSeries);
    }

    let numerator = 2.0 * g_hat * g_hat;
    Ok(OptimalBlockLength {
        stationary: ((numerator / d_sb).cbrt() * nf.cbrt()).clamp(1.0, b_max),
        circular: ((numerator / d_cb).cbrt() * nf.cbrt()).clamp(1.0, b_max),
    })
}

/// Biased (divide-by-n) sample autocovariances at lags `0..=max_lag`:
/// `R(k) = (1/n) sum_{t=0}^{n-k-1} (x_t - xbar)(x_{t+k} - xbar)`.
///
/// The biased normalization guarantees a positive semi-definite
/// autocovariance sequence and matches `statsmodels.tsa.stattools.acf`
/// with `adjusted=False` (up to the `1/gamma0` scaling).
//
// TODO(phase0): delegate to the shared diagnostics/HAC autocovariance
// kernel once tsecon-diag lands it; this private copy exists only so the
// bootstrap crate has no statistical dependencies in phase 0.
fn autocovariances(x: &[f64], max_lag: usize) -> Vec<f64> {
    debug_assert!(max_lag < x.len());
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    (0..=max_lag)
        .map(|k| {
            let s: f64 = x[..n - k]
                .iter()
                .zip(&x[k..])
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum();
            s / n as f64
        })
        .collect()
}

/// The flat-top (trapezoidal) lag window of Politis-Romano (1995):
/// `lambda(t) = 1` for `|t| <= 1/2`, `2(1 - |t|)` for `1/2 < |t| <= 1`,
/// `0` otherwise. Flat near the origin, so low-order autocovariances enter
/// unshrunk and the implied spectral estimate has bias of arbitrarily high
/// order.
//
// TODO(phase0): fold into the shared HAC kernel-window catalogue in
// tsecon-diag alongside Bartlett/Parzen/QS when that module lands.
fn flat_top(t: f64) -> f64 {
    let a = t.abs();
    if a <= 0.5 {
        1.0
    } else if a <= 1.0 {
        2.0 * (1.0 - a)
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Golden test: the private autocovariance helper against
    /// statsmodels' `acf(nile, adjusted=False)` from the shared fixture
    /// (the fixture stores autocorrelations, i.e. `R(k)/R(0)`).
    #[test]
    fn autocovariances_match_statsmodels_acf_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/diagnostics.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let nile: Vec<f64> = v["nile"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let expected: Vec<f64> = v["acf_20_unadjusted"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();

        let acov = autocovariances(&nile, 20);
        assert_eq!(expected.len(), 21);
        for (k, &e) in expected.iter().enumerate() {
            let got = acov[k] / acov[0];
            assert!(
                (got - e).abs() <= 1e-12,
                "lag {k}: got {got}, expected {e}"
            );
        }
    }

    #[test]
    fn flat_top_window_shape() {
        assert_eq!(flat_top(0.0), 1.0);
        assert_eq!(flat_top(0.5), 1.0);
        assert_eq!(flat_top(-0.5), 1.0);
        assert!((flat_top(0.75) - 0.5).abs() < 1e-15);
        assert_eq!(flat_top(1.0), 0.0);
        assert_eq!(flat_top(1.5), 0.0);
    }

    #[test]
    fn degenerate_and_short_samples_error() {
        assert_eq!(
            optimal_block_length(&[]),
            Err(BootstrapError::EmptySample)
        );
        assert!(matches!(
            optimal_block_length(&[1.0; 8]),
            Err(BootstrapError::SampleTooShort { n: 8, .. })
        ));
        assert_eq!(
            optimal_block_length(&[2.5; 100]),
            Err(BootstrapError::DegenerateSeries)
        );
        let mut x = vec![0.0; 100];
        x[3] = f64::NAN;
        assert_eq!(
            optimal_block_length(&x),
            Err(BootstrapError::NonFiniteData)
        );
    }

    #[test]
    fn nile_block_length_is_finite_positive_and_within_cap() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/diagnostics.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let nile: Vec<f64> = v["nile"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let b = optimal_block_length(&nile).unwrap();
        let n = nile.len() as f64;
        let b_max = (3.0 * n.sqrt()).min(n / 3.0).ceil();
        for len in [b.stationary, b.circular] {
            assert!(len.is_finite());
            assert!((1.0..=b_max).contains(&len), "length {len} out of range");
        }
        // Deterministic function of the data: identical on a second call.
        assert_eq!(b, optimal_block_length(&nile).unwrap());
    }
}
