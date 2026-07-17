//! Weighted MIDAS via nonlinear least squares.
//!
//! When the frequency mismatch is large, the unrestricted design of
//! [`crate::umidas`] has too many free lag coefficients; the classical MIDAS
//! remedy restricts them to a low-dimensional weight function and estimates
//! the handful of remaining parameters by nonlinear least squares (Ghysels,
//! Santa-Clara & Valkanov 2004; Ghysels, Sinko & Valkanov 2007). The model is
//!
//! ```text
//! y_t = alpha + beta * sum_{k=1}^{K} w_k(psi) x_{t,k} + eps_t,
//! ```
//!
//! where `w_k(psi)` are the normalized [`crate::exp_almon_weights`] or
//! [`crate::beta_weights`] and `psi` their two hyperparameters. Because the
//! weights sum to one, `beta` is the aggregate slope on a proper weighted
//! average of the high-frequency lags — directly comparable to the sum of the
//! U-MIDAS coefficients. The four parameters `(alpha, beta, psi_1, psi_2)` are
//! fit by minimizing the residual sum of squares through the library's single
//! optimizer, [`tsecon_optim`] (derivative-free Nelder-Mead with restarts —
//! MIDAS NLS objectives are mildly multimodal, so a warm linear start plus
//! restarts is the robust default; R `midasr`'s fragile single-start NLS is a
//! known pain point).
//!
//! The Beta shape parameters must stay positive, so the optimizer works on
//! `log psi` for the Beta scheme (exponential-Almon hyperparameters are
//! unconstrained). `alpha` and `beta` are warm-started from an OLS fit of `y`
//! on the start-weighted aggregate, which puts the search on the right scale.

use tsecon_hac::ols;
use tsecon_optim::{minimize, FnObjective, Method, NelderMeadOptions};

use crate::error::MidasError;
use crate::weights::{beta_weights, exp_almon_weights};

/// The parametric weight scheme used by [`weighted_midas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightScheme {
    /// Exponential Almon, `w_k proportional to exp(psi_1 k + psi_2 k^2)`;
    /// hyperparameters unconstrained.
    ExpAlmon,
    /// Two-parameter Beta, `w_k proportional to x_k^{psi_1 - 1}
    /// (1 - x_k)^{psi_2 - 1}`; hyperparameters strictly positive.
    Beta,
}

impl WeightScheme {
    /// Default hyperparameters `(psi_1, psi_2)` in natural space: a gently
    /// decaying profile for each scheme (exponential-Almon `(0, -0.1)`; Beta
    /// `(1, 3)`).
    fn default_start(self) -> [f64; 2] {
        match self {
            WeightScheme::ExpAlmon => [0.0, -0.1],
            WeightScheme::Beta => [1.0, 3.0],
        }
    }

    /// Evaluate the normalized weights at natural-space hyperparameters.
    fn weights(self, psi: [f64; 2], k: usize) -> Result<Vec<f64>, MidasError> {
        match self {
            WeightScheme::ExpAlmon => exp_almon_weights(psi[0], psi[1], k),
            WeightScheme::Beta => beta_weights(psi[0], psi[1], k),
        }
    }

    /// Map optimizer working coordinates to natural-space hyperparameters
    /// (identity for exponential-Almon; `exp` for the positive Beta shapes).
    fn to_natural(self, z: [f64; 2]) -> [f64; 2] {
        match self {
            WeightScheme::ExpAlmon => z,
            WeightScheme::Beta => [z[0].exp(), z[1].exp()],
        }
    }

    /// Map natural-space hyperparameters to optimizer working coordinates
    /// (inverse of [`to_natural`](WeightScheme::to_natural)).
    fn to_working(self, psi: [f64; 2]) -> [f64; 2] {
        match self {
            WeightScheme::ExpAlmon => psi,
            WeightScheme::Beta => [psi[0].ln(), psi[1].ln()],
        }
    }
}

/// A fitted weighted-MIDAS regression.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedMidasFit {
    /// The weight scheme that was fit.
    pub scheme: WeightScheme,
    /// Estimated intercept `alpha`.
    pub intercept: f64,
    /// Estimated aggregate slope `beta`.
    pub slope: f64,
    /// Estimated weight hyperparameters `(psi_1, psi_2)` in natural space.
    pub weight_params: [f64; 2],
    /// The normalized fitted weights `w_k(psi_hat)`, length `K`.
    pub weights: Vec<f64>,
    /// Fitted values, length `nobs`.
    pub fitted: Vec<f64>,
    /// Residuals `y_t - fitted_t`, length `nobs`.
    pub residuals: Vec<f64>,
    /// Residual sum of squares at the optimum.
    pub ssr: f64,
    /// Centered coefficient of determination `1 - SSR / sum_t (y_t - ybar)^2`.
    pub rsquared: f64,
    /// Whether the optimizer's convergence test was satisfied.
    pub converged: bool,
    /// Number of optimizer iterations.
    pub iterations: usize,
}

/// Fit a weighted MIDAS regression by nonlinear least squares.
///
/// `hf_lags` are the `K` most-recent-first stacked high-frequency lag columns,
/// each aligned to `y`. `weight_start` optionally overrides the starting
/// hyperparameters `(psi_1, psi_2)` in **natural** space (positive for Beta);
/// `None` uses [`WeightScheme::default_start`]. Returns the intercept, slope,
/// hyperparameters, fitted weights, and fit diagnostics.
///
/// The optimizer always returns its best point; inspect
/// [`WeightedMidasFit::converged`]. A non-converged multimodal fit is honest
/// output, not an error — rerun with a different `weight_start` if needed.
///
/// # Errors
///
/// * [`MidasError::InvalidLagCount`] if `hf_lags` is empty, or `K < 2` for the
///   Beta scheme;
/// * [`MidasError::DimensionMismatch`] if a lag column's length differs from
///   `y.len()`;
/// * [`MidasError::NonFinite`] if `y` or a lag column contains NaN/inf;
/// * [`MidasError::InvalidWeightParam`] if a Beta `weight_start` component is
///   not strictly positive, or any component is non-finite;
/// * [`MidasError::Optim`] / [`MidasError::Ols`] surfaced by the shared
///   optimizer or the warm-start OLS.
pub fn weighted_midas(
    y: &[f64],
    hf_lags: &[Vec<f64>],
    scheme: WeightScheme,
    weight_start: Option<[f64; 2]>,
) -> Result<WeightedMidasFit, MidasError> {
    let k = hf_lags.len();
    if k == 0 {
        return Err(MidasError::InvalidLagCount {
            what: "weighted MIDAS",
            k,
            needed: 1,
        });
    }
    if scheme == WeightScheme::Beta && k < 2 {
        return Err(MidasError::InvalidLagCount {
            what: "weighted MIDAS (Beta scheme)",
            k,
            needed: 2,
        });
    }
    let n = y.len();
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(MidasError::NonFinite {
                what: "weighted MIDAS target",
                index: i,
                value: v,
            });
        }
    }
    for (j, col) in hf_lags.iter().enumerate() {
        if col.len() != n {
            return Err(MidasError::DimensionMismatch {
                what: "weighted MIDAS lag column vs target",
                expected: n,
                got: col.len(),
            });
        }
        for (i, &v) in col.iter().enumerate() {
            if !v.is_finite() {
                return Err(MidasError::NonFinite {
                    what: "weighted MIDAS lag column",
                    index: i * hf_lags.len() + j,
                    value: v,
                });
            }
        }
    }

    // Starting hyperparameters (natural space), validated by evaluating the
    // start weights — this also rejects non-positive Beta shapes up front.
    let psi_start = weight_start.unwrap_or_else(|| scheme.default_start());
    let w_start = scheme.weights(psi_start, k)?;

    // Warm-start alpha, beta from an OLS of y on the start-weighted aggregate.
    let agg_start = aggregate(hf_lags, &w_start, n);
    let (alpha0, beta0) = linear_warm_start(y, &agg_start)?;

    let z_psi = scheme.to_working(psi_start);
    let z0 = [alpha0, beta0, z_psi[0], z_psi[1]];

    // SSR objective in working space; infeasible weights map to +infinity so
    // Nelder-Mead simply rejects the trial (no panics, no errors mid-search).
    let mut objective = FnObjective::new(|z: &[f64]| {
        let alpha = z[0];
        let beta = z[1];
        let psi = scheme.to_natural([z[2], z[3]]);
        let w = match scheme.weights(psi, k) {
            Ok(w) => w,
            Err(_) => return f64::INFINITY,
        };
        let mut ssr = 0.0;
        for t in 0..n {
            let mut agg = 0.0;
            for (col, &wk) in hf_lags.iter().zip(w.iter()) {
                agg += wk * col[t];
            }
            let resid = y[t] - (alpha + beta * agg);
            ssr += resid * resid;
        }
        ssr
    });

    let opts = NelderMeadOptions {
        max_iter: Some(4000),
        max_fevals: Some(8000),
        restarts: 2,
        initial_step: 0.1,
        ..NelderMeadOptions::default()
    };
    let res = minimize(&mut objective, &z0, &Method::NelderMead(opts))?;

    // Reconstruct the fit at the optimum.
    let psi_hat = scheme.to_natural([res.x[2], res.x[3]]);
    let weights = scheme.weights(psi_hat, k)?;
    let alpha = res.x[0];
    let beta = res.x[1];
    let agg = aggregate(hf_lags, &weights, n);
    let fitted: Vec<f64> = agg.iter().map(|a| alpha + beta * a).collect();
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yt, ft)| yt - ft)
        .collect();
    let ssr: f64 = residuals.iter().map(|u| u * u).sum();

    let rsquared = if n == 0 {
        f64::NAN
    } else {
        let ybar = y.iter().sum::<f64>() / n as f64;
        let tss: f64 = y.iter().map(|v| (v - ybar).powi(2)).sum();
        if tss > 0.0 {
            1.0 - ssr / tss
        } else {
            f64::NAN
        }
    };

    Ok(WeightedMidasFit {
        scheme,
        intercept: alpha,
        slope: beta,
        weight_params: psi_hat,
        weights,
        fitted,
        residuals,
        ssr,
        rsquared,
        converged: res.converged,
        iterations: res.iterations,
    })
}

/// The weighted aggregate regressor `a_t = sum_k w_k x_{t,k}`.
fn aggregate(hf_lags: &[Vec<f64>], w: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|t| {
            hf_lags
                .iter()
                .zip(w.iter())
                .map(|(col, &wk)| wk * col[t])
                .sum()
        })
        .collect()
}

/// OLS of `y` on `[const, agg]`, returning `(intercept, slope)` — the linear
/// warm start for the nonlinear search.
fn linear_warm_start(y: &[f64], agg: &[f64]) -> Result<(f64, f64), MidasError> {
    let cols = vec![vec![1.0; y.len()], agg.to_vec()];
    let fit = ols(y, &cols)?;
    Ok((fit.params[0], fit.params[1]))
}
