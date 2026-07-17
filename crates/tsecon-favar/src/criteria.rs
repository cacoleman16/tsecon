//! Factor-number selection: the Bai-Ng (2002) information criteria and
//! the Ahn-Horenstein (2013) eigenvalue-ratio estimator.

use crate::error::FavarError;

/// The Bai & Ng (2002, Econometrica) panel information criteria for the
/// number of static factors, evaluated over candidate counts
/// `k = 0, 1, ..., kmax`.
///
/// With `X` an `n x N` standardized panel decomposed by principal
/// components, let the sum of squared idiosyncratic residuals of the
/// `k`-factor model, per element, be
///
/// ```text
/// V(k) = (1 / (N n)) * sum_{i,t} (X_it - lambda_i' F_t)^2
///      = (1 / N) * sum_{j > k} lambda_j,
/// ```
///
/// where `lambda_j = s_j^2 / n` are the eigenvalues (the second equality
/// holds because the rank-`k` PCA reconstruction leaves exactly the tail
/// eigenvalues as residual variance). Writing `C_{nN}^2 = min(n, N)` and
/// the penalty scale `g = (N + n) / (N n)`, the criteria are
///
/// ```text
/// IC_p1(k) = ln V(k) + k * g * ln( N n / (N + n) )
/// IC_p2(k) = ln V(k) + k * g * ln( C_{nN}^2 )
/// PC_p1(k) = V(k)    + k * sigma^2 * g * ln( N n / (N + n) )
/// PC_p2(k) = V(k)    + k * sigma^2 * g * ln( C_{nN}^2 )
/// ```
///
/// with `sigma^2 = V(kmax)` a consistent estimate of the average
/// idiosyncratic variance. The estimated factor count is the minimizer
/// of each criterion over `0..=kmax` (Bai & Ng 2002, Theorem 2). The
/// criteria are consistent as `n, N -> infinity` with bounded
/// idiosyncratic eigenvalues; in small cross-sections `N` with slowly
/// decaying idiosyncratic eigenvalues they can over-select, in which
/// case the eigenvalue-ratio estimator ([`eigenvalue_ratio`]) is more
/// robust.
#[derive(Debug, Clone, PartialEq)]
pub struct BaiNg {
    /// `IC_p1(k)` for `k = 0..=kmax`.
    pub icp1: Vec<f64>,
    /// `IC_p2(k)` for `k = 0..=kmax`.
    pub icp2: Vec<f64>,
    /// `PC_p1(k)` for `k = 0..=kmax`.
    pub pcp1: Vec<f64>,
    /// `PC_p2(k)` for `k = 0..=kmax`.
    pub pcp2: Vec<f64>,
    /// Minimizer of `IC_p1`.
    pub icp1_hat: usize,
    /// Minimizer of `IC_p2`.
    pub icp2_hat: usize,
    /// Minimizer of `PC_p1`.
    pub pcp1_hat: usize,
    /// Minimizer of `PC_p2`.
    pub pcp2_hat: usize,
}

/// Computes the Bai-Ng criteria from the PCA `eigenvalues` (descending,
/// `lambda_j = s_j^2 / n`) of a standardized `n x N` panel.
///
/// `kmax` is the largest candidate factor count; `sigma^2` is estimated
/// as `V(kmax)`, so `kmax` must be at least 1 and strictly less than the
/// number of eigenvalues (there must be a residual tail to estimate the
/// noise variance).
///
/// # Errors
///
/// * [`FavarError::InvalidArgument`] if `n == 0` or `n_series == 0`;
/// * [`FavarError::NonFinite`] on a NaN/infinite eigenvalue;
/// * [`FavarError::InvalidFactorCount`] if `kmax < 1` or
///   `kmax >= eigenvalues.len()`.
pub fn bai_ng(
    eigenvalues: &[f64],
    n: usize,
    n_series: usize,
    kmax: usize,
) -> Result<BaiNg, FavarError> {
    if n == 0 || n_series == 0 {
        return Err(FavarError::InvalidArgument {
            what: "n and n_series must be positive",
        });
    }
    for &e in eigenvalues {
        if !e.is_finite() {
            return Err(FavarError::NonFinite {
                what: "eigenvalues",
            });
        }
    }
    let m = eigenvalues.len();
    if kmax < 1 || kmax >= m {
        return Err(FavarError::InvalidFactorCount {
            what: "kmax must satisfy 1 <= kmax < number of eigenvalues",
            requested: kmax,
            max: m.saturating_sub(1),
        });
    }

    // V(k) = (1/N) * sum_{j >= k} lambda_j, for k = 0..=kmax.
    let big_n = n_series as f64;
    let mut v = vec![0.0f64; kmax + 1];
    for (k, slot) in v.iter_mut().enumerate() {
        let tail: f64 = eigenvalues[k..].iter().sum();
        *slot = tail / big_n;
    }
    // Guard: with a degenerate (rank-deficient) panel V(kmax) can be 0,
    // which would make ln V and the PC noise scale ill-defined.
    let sigma2 = v[kmax];
    if sigma2 <= 0.0 {
        return Err(FavarError::InvalidArgument {
            what: "residual variance V(kmax) is zero; lower kmax or drop collinear series",
        });
    }

    let nn = big_n * n as f64;
    let g = (big_n + n as f64) / nn;
    let ln_ratio = (nn / (big_n + n as f64)).ln();
    let ln_cmin = (n_series.min(n) as f64).ln();

    let mut icp1 = vec![0.0f64; kmax + 1];
    let mut icp2 = vec![0.0f64; kmax + 1];
    let mut pcp1 = vec![0.0f64; kmax + 1];
    let mut pcp2 = vec![0.0f64; kmax + 1];
    for k in 0..=kmax {
        let kf = k as f64;
        let lnv = v[k].ln();
        icp1[k] = lnv + kf * g * ln_ratio;
        icp2[k] = lnv + kf * g * ln_cmin;
        pcp1[k] = v[k] + kf * sigma2 * g * ln_ratio;
        pcp2[k] = v[k] + kf * sigma2 * g * ln_cmin;
    }

    Ok(BaiNg {
        icp1_hat: argmin(&icp1),
        icp2_hat: argmin(&icp2),
        pcp1_hat: argmin(&pcp1),
        pcp2_hat: argmin(&pcp2),
        icp1,
        icp2,
        pcp1,
        pcp2,
    })
}

/// The Ahn & Horenstein (2013, Econometrica) eigenvalue-ratio (ER)
/// estimator of the number of factors:
///
/// ```text
/// r_hat = argmax_{1 <= k <= kmax}  lambda_k / lambda_{k+1},
/// ```
///
/// where `lambda_k` is the `k`-th largest PCA eigenvalue. A genuine
/// factor produces a diverging eigenvalue while idiosyncratic
/// eigenvalues stay bounded, so the ratio `lambda_r / lambda_{r+1}`
/// spikes exactly at the true factor count. Unlike the Bai-Ng criteria
/// the ER estimator needs no penalty tuning and no consistent
/// noise-variance estimate, which makes it robust in small cross-sections
/// and when the idiosyncratic eigenvalues decay slowly.
///
/// Returns the pair `(r_hat, ratios)` where `ratios[k-1] =
/// lambda_k / lambda_{k+1}` for `k = 1..=kmax`.
///
/// # Errors
///
/// * [`FavarError::NonFinite`] on a NaN/infinite eigenvalue;
/// * [`FavarError::InvalidFactorCount`] if `kmax < 1` or
///   `kmax >= eigenvalues.len()`;
/// * [`FavarError::InvalidArgument`] if any denominator eigenvalue in the
///   scanned range is non-positive (a rank-deficient panel; lower
///   `kmax`).
pub fn eigenvalue_ratio(eigenvalues: &[f64], kmax: usize) -> Result<(usize, Vec<f64>), FavarError> {
    for &e in eigenvalues {
        if !e.is_finite() {
            return Err(FavarError::NonFinite {
                what: "eigenvalues",
            });
        }
    }
    let m = eigenvalues.len();
    if kmax < 1 || kmax >= m {
        return Err(FavarError::InvalidFactorCount {
            what: "kmax must satisfy 1 <= kmax < number of eigenvalues",
            requested: kmax,
            max: m.saturating_sub(1),
        });
    }
    let mut ratios = vec![0.0f64; kmax];
    for k in 1..=kmax {
        let denom = eigenvalues[k];
        if denom <= 0.0 {
            return Err(FavarError::InvalidArgument {
                what: "non-positive eigenvalue in the eigenvalue-ratio range; lower kmax",
            });
        }
        ratios[k - 1] = eigenvalues[k - 1] / denom;
    }
    Ok((argmax(&ratios) + 1, ratios))
}

// Index of the first minimal element (ties resolved to the smaller k,
// i.e. the more parsimonious model).
fn argmin(v: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_val = f64::INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x < best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

// Index of the first maximal element.
fn argmax(v: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}
