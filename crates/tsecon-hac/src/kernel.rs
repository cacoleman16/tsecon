//! HAC kernels (lag windows) and their Andrews (1991) plug-in constants.
//!
//! A kernel HAC / long-run-variance estimator weights the sample
//! autocovariance at lag `j` by `k(x_j)`, where `k` is one of the kernel
//! functions below and `x_j` is the lag scaled by a bandwidth. Two
//! bandwidth conventions coexist in the literature and this crate keeps
//! both explicit:
//!
//! * **Lag-truncation ("maxlags") convention** — used by Newey-West (1987)
//!   and statsmodels `cov_type="HAC"`: for Bartlett and Parzen the scaled
//!   lag is `x_j = j / (bandwidth + 1)`, so weights are exactly zero for
//!   `j > bandwidth` when `bandwidth` is an integer, and `bandwidth` reads
//!   as "the number of lags included". The truncated kernel likewise
//!   includes lags `j <= bandwidth`.
//! * **Continuous-scale convention** — Andrews (1991) weights lag `j` by
//!   `k(j / S_T)` with a real-valued bandwidth `S_T`. The quadratic
//!   spectral kernel never truncates, so its `bandwidth` argument *is*
//!   `S_T`. For Bartlett/Parzen, [`Kernel::bandwidth_from_scale`] converts
//!   an Andrews-style `S_T` (e.g. from [`crate::andrews_bandwidth_ar1`] or
//!   [`crate::newey_west_bandwidth`]) into this crate's lag-truncation
//!   `bandwidth` so the Andrews weighting `k(j / S_T)` is reproduced
//!   exactly.
//!
//! References: Newey & West (1987, Econometrica), Andrews (1991,
//! Econometrica, Table I), Gallant (1987) for Parzen, White (1984) for the
//! truncated kernel.

use core::f64::consts::PI;

/// Kernel (lag-window) choices for HAC / long-run variance estimation.
///
/// Weights are produced by [`Kernel::weight`]; the Andrews (1991) plug-in
/// constants used by the automatic bandwidth selectors are exposed via
/// [`Kernel::andrews_constant`] and [`Kernel::andrews_q`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Bartlett (triangular) kernel, `k(x) = 1 - |x|` for `|x| <= 1` —
    /// the Newey-West (1987) kernel. Guarantees a positive semi-definite
    /// estimate. Lag-truncation convention: `w_j = 1 - j/(bandwidth + 1)`.
    Bartlett,
    /// Parzen kernel (Parzen 1957; Gallant 1987):
    /// `k(x) = 1 - 6x^2 + 6|x|^3` for `|x| <= 1/2`,
    /// `k(x) = 2(1 - |x|)^3` for `1/2 < |x| <= 1`, else 0.
    /// Positive semi-definite. Lag-truncation convention:
    /// `x = j/(bandwidth + 1)`.
    Parzen,
    /// Quadratic spectral kernel (Andrews 1991, eq. 2.7):
    /// `k(x) = 25/(12 pi^2 x^2) [ sin(6 pi x/5)/(6 pi x/5) - cos(6 pi x/5) ]`,
    /// `k(0) = 1`. Positive semi-definite and mean-squared-error optimal in
    /// the Andrews (1991) class; never truncates, so every lag receives
    /// some weight. Continuous convention: `x = j/bandwidth` with
    /// `bandwidth = S_T`.
    QuadraticSpectral,
    /// Truncated (uniform) kernel, `k(x) = 1` for `|x| <= 1` (White 1984):
    /// unit weight on lags `j <= bandwidth`, zero beyond. **Not** positive
    /// semi-definite — sandwich variances can come out negative; provided
    /// for pedagogy and comparability, not as a default.
    Truncated,
}

impl Kernel {
    /// Weight applied to the lag-`j` autocovariance at the given bandwidth.
    ///
    /// `weight(0, b) = 1` for every kernel. Negative bandwidths are clamped
    /// to zero here (fallible entry points reject them with
    /// [`crate::HacError::InvalidBandwidth`] first). At `bandwidth = 0`
    /// every kernel puts zero weight on all `j >= 1`, so the estimator
    /// degenerates to the lag-0 autocovariance.
    ///
    /// Conventions (see the module docs): Bartlett/Parzen use
    /// `x = j/(bandwidth + 1)` (lag-truncation, statsmodels `maxlags`),
    /// the truncated kernel includes `j <= bandwidth`, and quadratic
    /// spectral uses `x = j/bandwidth` (Andrews' `S_T`).
    #[must_use]
    pub fn weight(self, j: usize, bandwidth: f64) -> f64 {
        if j == 0 {
            return 1.0;
        }
        let bandwidth = bandwidth.max(0.0);
        let jf = j as f64;
        match self {
            Kernel::Bartlett => (1.0 - jf / (bandwidth + 1.0)).max(0.0),
            Kernel::Parzen => {
                let x = jf / (bandwidth + 1.0);
                if x <= 0.5 {
                    1.0 - 6.0 * x * x + 6.0 * x * x * x
                } else if x <= 1.0 {
                    let one_minus = 1.0 - x;
                    2.0 * one_minus * one_minus * one_minus
                } else {
                    0.0
                }
            }
            Kernel::QuadraticSpectral => {
                if bandwidth == 0.0 {
                    return 0.0;
                }
                let d = 6.0 * PI * (jf / bandwidth) / 5.0;
                if d.abs() < 1e-3 {
                    // Series expansion around 0 to avoid 0/0 cancellation:
                    // k = 1 - d^2/10 + d^4/280 + O(d^6).
                    1.0 - d * d / 10.0 + d.powi(4) / 280.0
                } else {
                    3.0 / (d * d) * (d.sin() / d - d.cos())
                }
            }
            Kernel::Truncated => {
                if jf <= bandwidth {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Whether the kernel's weights are exactly zero beyond a finite lag
    /// (true for Bartlett/Parzen/truncated, false for quadratic spectral).
    /// Lag loops over `weight(j, b)` may stop at the first zero weight for
    /// truncating kernels.
    #[must_use]
    pub fn truncates(self) -> bool {
        !matches!(self, Kernel::QuadraticSpectral)
    }

    /// The constant `c` in Andrews' (1991, Table I) optimal bandwidth
    /// `S_T* = c * (alpha(q) * T)^(1/(2q+1))`: 1.1447 (Bartlett), 2.6614
    /// (Parzen), 1.3221 (quadratic spectral), 0.6611 (truncated).
    #[must_use]
    pub fn andrews_constant(self) -> f64 {
        match self {
            Kernel::Bartlett => 1.1447,
            Kernel::Parzen => 2.6614,
            Kernel::QuadraticSpectral => 1.3221,
            Kernel::Truncated => 0.6611,
        }
    }

    /// The characteristic exponent `q` of the kernel (Andrews 1991):
    /// 1 for Bartlett, 2 for Parzen/quadratic spectral/truncated. It sets
    /// the optimal growth rate `S_T ~ T^(1/(2q+1))` and which curvature
    /// coefficient `alpha(q)` the plug-in bandwidth uses.
    #[must_use]
    pub fn andrews_q(self) -> f64 {
        match self {
            Kernel::Bartlett => 1.0,
            _ => 2.0,
        }
    }

    /// Convert an Andrews-style continuous bandwidth `S_T` (which weights
    /// lag `j` by `k(j / S_T)`) into the `bandwidth` argument expected by
    /// [`Kernel::weight`] / [`crate::lrv`].
    ///
    /// For Bartlett/Parzen (lag-truncation convention, `x = j/(b + 1)`)
    /// this is `max(S_T - 1, 0)`; for quadratic spectral and truncated the
    /// scale passes through unchanged. Feeding
    /// `kernel.bandwidth_from_scale(s_t)` to [`crate::lrv`] therefore
    /// reproduces Andrews' `k(j / S_T)` weighting exactly. (R's
    /// `sandwich::NeweyWest` instead floors `S_T` to an integer `maxlags`;
    /// both readings are in common use — this crate keeps the exact one.)
    #[must_use]
    pub fn bandwidth_from_scale(self, scale: f64) -> f64 {
        match self {
            Kernel::Bartlett | Kernel::Parzen => (scale - 1.0).max(0.0),
            Kernel::QuadraticSpectral | Kernel::Truncated => scale.max(0.0),
        }
    }

    /// Human-readable kernel name (used in error messages).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Kernel::Bartlett => "Bartlett",
            Kernel::Parzen => "Parzen",
            Kernel::QuadraticSpectral => "quadratic spectral",
            Kernel::Truncated => "truncated",
        }
    }
}
