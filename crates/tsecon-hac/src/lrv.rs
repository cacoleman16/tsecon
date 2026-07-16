//! Kernel long-run variance estimation with automatic bandwidth selection
//! and Andrews-Monahan (1992) AR(1) prewhitening.
//!
//! The long-run variance (LRV, `2 pi` times the spectral density at
//! frequency zero) of a mean-zero series `x_t` is estimated as
//!
//! ```text
//! omega_hat = gamma_hat(0) + 2 * sum_{j>=1} w(j; bandwidth) * gamma_hat(j),
//! gamma_hat(j) = (1/n) * sum_{t=j+1}^{n} x_t * x_{t-j}   (biased, about zero)
//! ```
//!
//! with kernel weights `w` from [`Kernel::weight`]. Autocovariances are
//! taken **about zero**, not about the sample mean: pass a demeaned series,
//! regression residuals, or scores (all mean zero by construction). This is
//! the statsmodels/Newey-West convention and what the golden fixtures pin.
//!
//! Bandwidth selection: [`newey_west_maxlags`] (the floor-rule everyone
//! calls "maxlags"), [`newey_west_bandwidth`] (Newey-West 1994 nonparametric
//! plug-in), and [`andrews_bandwidth_ar1`] (Andrews 1991 AR(1) parametric
//! plug-in). The latter two return Andrews-style continuous scales `S_T`;
//! convert with [`Kernel::bandwidth_from_scale`] before calling [`lrv`].

use crate::error::HacError;
use crate::kernel::Kernel;
use crate::validate::{check_bandwidth, check_finite, check_min_len};

/// Biased (`1/n`) sample autocovariance about zero at lag `j`.
pub(crate) fn autocov_zero(x: &[f64], j: usize) -> f64 {
    let n = x.len();
    x[j..]
        .iter()
        .zip(x.iter())
        .map(|(&a, &b)| a * b)
        .sum::<f64>()
        / n as f64
}

/// Kernel long-run variance of a mean-zero univariate series.
///
/// Computes `gamma_hat(0) + 2 * sum_{j=1}^{n-1} w(j; bandwidth) *
/// gamma_hat(j)` with biased (`1/n`) autocovariances about zero (demean
/// the series first if it has a nonzero mean; see the module docs) and
/// kernel weights from [`Kernel::weight`]. For Bartlett this is exactly
/// the Newey-West (1987) estimator with `bandwidth` = `maxlags`, i.e.
/// weights `1 - j/(bandwidth + 1)`.
///
/// The bandwidth must be finite and non-negative; it need not be an
/// integer. With a truncating kernel the lag sum stops at the first zero
/// weight; the quadratic spectral kernel uses all `n - 1` lags.
///
/// References: Newey & West (1987), Andrews (1991).
///
/// # Errors
///
/// [`HacError::SeriesTooShort`] if `n < 2`, [`HacError::NonFinite`] on
/// NaN/inf input, [`HacError::InvalidBandwidth`] if the bandwidth is
/// negative or non-finite.
pub fn lrv(x: &[f64], kernel: Kernel, bandwidth: f64) -> Result<f64, HacError> {
    const WHAT: &str = "long-run variance";
    check_min_len(x, 2, WHAT)?;
    check_finite(x, WHAT)?;
    check_bandwidth(bandwidth)?;

    let n = x.len();
    let mut omega = autocov_zero(x, 0);
    for j in 1..n {
        let w = kernel.weight(j, bandwidth);
        if w == 0.0 && kernel.truncates() {
            break;
        }
        omega += 2.0 * w * autocov_zero(x, j);
    }
    Ok(omega)
}

/// The Newey-West rule-of-thumb lag truncation
/// `maxlags = floor(4 * (n/100)^(2/9))`.
///
/// This is the Bartlett-kernel pilot rule of Newey & West (1994) and the
/// default `maxlags` statsmodels users expect for `cov_type="HAC"`. Feed
/// the result (as `f64`) straight to [`lrv`] / HAC standard errors with
/// [`Kernel::Bartlett`].
#[must_use]
pub fn newey_west_maxlags(n: usize) -> usize {
    (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize
}

/// Newey-West (1994) nonparametric plug-in bandwidth `S_T` for the LRV of
/// a mean-zero univariate series.
///
/// Procedure (Newey & West 1994, Review of Economic Studies, Table II C),
/// for a kernel with characteristic exponent `q` and constant `c_gamma`
/// ([`Kernel::andrews_q`], [`Kernel::andrews_constant`]):
///
/// ```text
/// m      = floor(4 * (n/100)^r),  r = 2/9 (Bartlett), 4/25 (Parzen), 2/25 (QS)
/// s^(0)  = gamma_hat(0) + 2 * sum_{j=1}^{m} gamma_hat(j)
/// s^(q)  = 2 * sum_{j=1}^{m} j^q * gamma_hat(j)
/// S_T    = c_gamma * ((s^(q)/s^(0))^2)^(1/(2q+1)) * n^(1/(2q+1))
/// ```
///
/// Autocovariances are biased (`1/n`) and about zero. The returned value
/// is the continuous Andrews-style scale (identical to R
/// `sandwich::bwNeweyWest(..., prewhite = 0)`); convert with
/// [`Kernel::bandwidth_from_scale`] before calling [`lrv`], or floor it to
/// an integer `maxlags` for the R `NeweyWest` reading.
///
/// # Errors
///
/// [`HacError::UnsupportedKernel`] for the truncated kernel (no published
/// rule), [`HacError::SeriesTooShort`] if `n < 4`, [`HacError::NonFinite`]
/// on NaN/inf input, [`HacError::ConstantSeries`] if the pilot estimate
/// `s^(0)` is exactly zero.
pub fn newey_west_bandwidth(x: &[f64], kernel: Kernel) -> Result<f64, HacError> {
    const WHAT: &str = "Newey-West (1994) plug-in bandwidth";
    let rate = match kernel {
        Kernel::Bartlett => 2.0 / 9.0,
        Kernel::Parzen => 4.0 / 25.0,
        Kernel::QuadraticSpectral => 2.0 / 25.0,
        Kernel::Truncated => {
            return Err(HacError::UnsupportedKernel {
                what: WHAT,
                kernel: kernel.name(),
            })
        }
    };
    check_min_len(x, 4, WHAT)?;
    check_finite(x, WHAT)?;

    let n = x.len();
    let nf = n as f64;
    let m = ((4.0 * (nf / 100.0).powf(rate)).floor() as usize).min(n - 1);

    let q = kernel.andrews_q();
    let gamma0 = autocov_zero(x, 0);
    let mut s0 = gamma0;
    let mut sq = 0.0;
    for j in 1..=m {
        let g = autocov_zero(x, j);
        s0 += 2.0 * g;
        sq += 2.0 * (j as f64).powf(q) * g;
    }
    if s0 == 0.0 {
        return Err(HacError::ConstantSeries { what: WHAT });
    }

    let exponent = 1.0 / (2.0 * q + 1.0);
    let ratio_sq = (sq / s0) * (sq / s0);
    Ok(kernel.andrews_constant() * ratio_sq.powf(exponent) * nf.powf(exponent))
}

/// Andrews (1991) AR(1) parametric plug-in bandwidth `S_T*` for the LRV of
/// a mean-zero univariate series.
///
/// Fits an AR(1) through the origin by OLS,
/// `rho_hat = sum_t x_t x_{t-1} / sum_t x_{t-1}^2`, and plugs it into the
/// univariate (unit-weight) curvature coefficients of Andrews (1991,
/// eq. 6.4):
///
/// ```text
/// alpha(1) = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2)
/// alpha(2) = 4 rho^2 / (1 - rho)^4
/// S_T*     = c * (alpha(q) * n)^(1/(2q+1))
/// ```
///
/// with `(q, c)` from [`Kernel::andrews_q`] / [`Kernel::andrews_constant`]
/// (Andrews 1991, Table I; e.g. Bartlett: `1.1447 * (alpha(1) n)^(1/3)`).
/// The returned value is the continuous Andrews scale `S_T*`; convert with
/// [`Kernel::bandwidth_from_scale`] before calling [`lrv`]. Comparable to
/// R `sandwich::bwAndrews` on a univariate score.
///
/// # Errors
///
/// [`HacError::SeriesTooShort`] if `n < 4`, [`HacError::NonFinite`] on
/// NaN/inf input, [`HacError::ConstantSeries`] if the series is
/// identically zero, [`HacError::NumericalBreakdown`] if `rho_hat` is at a
/// unit root (`|1 -+ rho| < 1e-12`), where the plug-in formula diverges.
pub fn andrews_bandwidth_ar1(x: &[f64], kernel: Kernel) -> Result<f64, HacError> {
    const WHAT: &str = "Andrews (1991) AR(1) plug-in bandwidth";
    check_min_len(x, 4, WHAT)?;
    check_finite(x, WHAT)?;

    let rho = ar1_coefficient(x, WHAT)?;
    if (1.0 - rho).abs() < 1e-12 || (1.0 + rho).abs() < 1e-12 {
        return Err(HacError::NumericalBreakdown { what: WHAT });
    }

    let q = kernel.andrews_q();
    let alpha = if q == 1.0 {
        let d = (1.0 - rho) * (1.0 + rho);
        4.0 * rho * rho / (d * d)
    } else {
        let d = 1.0 - rho;
        4.0 * rho * rho / (d * d * d * d)
    };
    let exponent = 1.0 / (2.0 * q + 1.0);
    Ok(kernel.andrews_constant() * (alpha * x.len() as f64).powf(exponent))
}

/// OLS AR(1) coefficient through the origin.
fn ar1_coefficient(x: &[f64], what: &'static str) -> Result<f64, HacError> {
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..x.len() {
        num += x[t] * x[t - 1];
        den += x[t - 1] * x[t - 1];
    }
    if den <= 0.0 {
        return Err(HacError::ConstantSeries { what });
    }
    Ok(num / den)
}

/// Result of the Andrews-Monahan (1992) prewhitened long-run variance.
#[derive(Debug, Clone, PartialEq)]
pub struct PrewhitenedLrv {
    /// The recolored long-run variance estimate
    /// `omega_hat = omega_hat_e / (1 - rho_hat)^2`.
    pub value: f64,
    /// The (capped) AR(1) prewhitening coefficient actually used.
    pub rho: f64,
    /// The kernel bandwidth applied to the whitened residuals (either the
    /// caller's, or the Andrews AR(1) plug-in choice when `None` was
    /// passed).
    pub bandwidth: f64,
}

/// AR(1)-prewhitened kernel long-run variance (Andrews & Monahan 1992).
///
/// Prewhitening filters most of the persistence out before the kernel
/// smoothing and then recolors, which sharply reduces the bias of kernel
/// LRV estimators on persistent series:
///
/// ```text
/// rho_hat = sum_t x_t x_{t-1} / sum_t x_{t-1}^2,   capped to |rho| <= 0.97
/// e_t     = x_t - rho_hat * x_{t-1},               t = 2..n
/// omega   = lrv(e; kernel, bandwidth) / (1 - rho_hat)^2
/// ```
///
/// The `|rho| <= 0.97` cap is the Andrews-Monahan (1992, Econometrica,
/// Section 3) eigenvalue adjustment specialized to the scalar AR(1) case:
/// it keeps the recoloring factor `1/(1 - rho)^2` bounded when the fitted
/// root is (near-)unit. Pass `bandwidth = None` to select the bandwidth on
/// the *whitened* residuals via [`andrews_bandwidth_ar1`] (mapped through
/// [`Kernel::bandwidth_from_scale`]), which is the procedure Andrews and
/// Monahan recommend; or supply an explicit bandwidth in the [`lrv`]
/// convention.
///
/// The series must be mean zero (demeaned/residuals/scores), as for
/// [`lrv`]. Matrix (VAR(1)) prewhitening for vector scores is deferred to
/// the shared linalg layer.
///
/// # Errors
///
/// [`HacError::SeriesTooShort`] if `n < 4` (the whitened series needs at
/// least 3 observations), plus the [`lrv`] /
/// [`andrews_bandwidth_ar1`] error conditions on the whitened residuals.
pub fn lrv_prewhitened_ar1(
    x: &[f64],
    kernel: Kernel,
    bandwidth: Option<f64>,
) -> Result<PrewhitenedLrv, HacError> {
    // TODO(phase0): matrix (VAR(1)) prewhitening for vector scores once the
    // shared linalg crate provides the eigenvalue adjustment; the scalar
    // AR(1) case below is the univariate specialization.
    const WHAT: &str = "Andrews-Monahan (1992) prewhitened long-run variance";
    check_min_len(x, 4, WHAT)?;
    check_finite(x, WHAT)?;
    if let Some(b) = bandwidth {
        check_bandwidth(b)?;
    }

    let rho = ar1_coefficient(x, WHAT)?.clamp(-0.97, 0.97);
    let whitened: Vec<f64> = (1..x.len()).map(|t| x[t] - rho * x[t - 1]).collect();

    let bw = match bandwidth {
        Some(b) => b,
        None => kernel.bandwidth_from_scale(andrews_bandwidth_ar1(&whitened, kernel)?),
    };
    let omega_e = lrv(&whitened, kernel, bw)?;
    let recolor = 1.0 - rho;
    Ok(PrewhitenedLrv {
        value: omega_e / (recolor * recolor),
        rho,
        bandwidth: bw,
    })
}
