//! Linear quantile regression by iteratively reweighted least squares,
//! matching statsmodels `QuantReg(endog, exog).fit(q=tau)` (all defaults)
//! step for step.
//!
//! ## Point estimates
//!
//! The estimator minimizes the check loss
//! `sum_t rho_tau(y_t - x_t' b)`, `rho_tau(u) = u (tau - 1{u < 0})`, by the
//! Schnabel-Koenker IRLS iteration statsmodels uses: starting from the OLS
//! fit (an unweighted first step), each iteration solves the weighted least
//! squares problem with weights `1 / s_t` where
//! `s_t = |tau * u_t|` for `u_t < 0` and `|(1 - tau) * u_t|` otherwise, the
//! residual having first been floored at `|u_t| >= 1e-6` (statsmodels'
//! `.000001` smoothing floor). Convergence is declared when the maximum
//! absolute coefficient change drops to `1e-6` (`p_tol`), with an iteration
//! cap of 1000 (`max_iter`). Each weighted solve is delegated to
//! [`tsecon_hac::ols`] on scaled data — this crate never reimplements least
//! squares.
//!
//! ## Standard errors
//!
//! `vcov="robust"` in statsmodels: the Powell (1991) kernel sandwich
//! `(X'X)^{-1} (X' D X) (X'X)^{-1}` with
//! `d_t = (tau / f(0))^2` for `u_t > 0` and `((1-tau) / f(0))^2` otherwise,
//! where `f(0)` is an Epanechnikov kernel density estimate of the residuals
//! at zero. Its bandwidth is the Hall-Sheather (1988) rule mapped through
//! the Stata-12 recipe statsmodels transcribes:
//! `h = min(std(y), IQR(e)/1.34) * (Phi^{-1}(tau + h_HS) - Phi^{-1}(tau - h_HS))`
//! with `h_HS = n^{-1/3} z_{0.975}^{2/3} (1.5 phi(z_tau)^2 / (2 z_tau^2 + 1))^{1/3}`.

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::special::inv_norm_cdf;

use crate::error::QuantileError;

/// statsmodels `max_iter` default.
pub(crate) const MAX_ITER: usize = 1000;
/// statsmodels `p_tol` default: max absolute coefficient change.
pub(crate) const P_TOL: f64 = 1e-6;
/// statsmodels' residual smoothing floor (`.000001` in the source).
const RESID_FLOOR: f64 = 1e-6;

/// One fitted quantile regression; produced by [`quantile_regression`].
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileFit {
    /// The quantile level this fit targets.
    pub tau: f64,
    /// Coefficient estimates, in the order the design columns were passed.
    pub params: Vec<f64>,
    /// Powell kernel-sandwich standard errors (statsmodels `vcov="robust"`).
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`.
    pub tvalues: Vec<f64>,
    /// Parameter covariance matrix, `k x k` row-major.
    pub cov: Vec<f64>,
    /// IRLS iterations used.
    pub iterations: usize,
    /// Whether the coefficient change dropped below `p_tol` before the
    /// iteration cap (statsmodels warns instead of erroring; we report).
    pub converged: bool,
    /// The kernel bandwidth `h` used for the density-at-zero estimate.
    pub bandwidth: f64,
    /// The sparsity `1 / f_hat(0)` (Koenker's `s(tau)`).
    pub sparsity: f64,
}

/// Fit linear quantile regressions of `y` on the design columns `x_cols`
/// at each level in `taus`.
///
/// `x_cols` are the columns of the design matrix; include the constant
/// column explicitly (statsmodels exog convention). Returns one
/// [`QuantileFit`] per tau, in the order the taus were passed. Matches
/// statsmodels `QuantReg(y, X).fit(q=tau)` — coefficients to ~1e-6 (the
/// shared IRLS stopping tolerance) and robust `bse` at the same order.
///
/// # Errors
///
/// [`QuantileError::NoTaus`], [`QuantileError::InvalidTau`],
/// [`QuantileError::EmptyInput`], [`QuantileError::DimensionMismatch`],
/// [`QuantileError::NonFinite`], [`QuantileError::DegreesOfFreedom`],
/// [`QuantileError::Singular`] on a collinear design, and
/// [`QuantileError::DegenerateBandwidth`] when the Powell bandwidth cannot
/// be formed (tau too extreme for the sample size, or degenerate residuals).
pub fn quantile_regression(
    y: &[f64],
    x_cols: &[Vec<f64>],
    taus: &[f64],
) -> Result<Vec<QuantileFit>, QuantileError> {
    validate_taus(taus)?;
    validate_design(y, x_cols)?;
    taus.iter().map(|&tau| fit_one(y, x_cols, tau)).collect()
}

/// Shared tau validation: at least one, each strictly inside `(0, 1)`.
pub(crate) fn validate_taus(taus: &[f64]) -> Result<(), QuantileError> {
    if taus.is_empty() {
        return Err(QuantileError::NoTaus);
    }
    for &tau in taus {
        if !(tau > 0.0 && tau < 1.0) {
            return Err(QuantileError::InvalidTau { tau });
        }
    }
    Ok(())
}

/// Shared design validation: shapes, finiteness, degrees of freedom.
pub(crate) fn validate_design(y: &[f64], x_cols: &[Vec<f64>]) -> Result<(), QuantileError> {
    if y.is_empty() {
        return Err(QuantileError::EmptyInput { what: "y" });
    }
    if x_cols.is_empty() {
        return Err(QuantileError::EmptyInput {
            what: "design columns",
        });
    }
    let n = y.len();
    for col in x_cols {
        if col.len() != n {
            return Err(QuantileError::DimensionMismatch {
                what: "design column vs y",
                expected: n,
                got: col.len(),
            });
        }
    }
    check_finite(y, "y")?;
    for col in x_cols {
        check_finite(col, "design column")?;
    }
    if n <= x_cols.len() {
        return Err(QuantileError::DegreesOfFreedom { n, k: x_cols.len() });
    }
    Ok(())
}

/// Non-finite guard shared by all entry points.
pub(crate) fn check_finite(x: &[f64], what: &'static str) -> Result<(), QuantileError> {
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(QuantileError::NonFinite { what, index: i });
        }
    }
    Ok(())
}

/// One tau: the IRLS loop plus the Powell sandwich.
pub(crate) fn fit_one(
    y: &[f64],
    x_cols: &[Vec<f64>],
    tau: f64,
) -> Result<QuantileFit, QuantileError> {
    let n = y.len();
    let k = x_cols.len();

    // --- IRLS (statsmodels protocol, transcribed) -------------------------
    // xstar = exog initially (all weights one -> first step is plain OLS);
    // the initial beta of ones only seeds the convergence check.
    let mut beta = vec![1.0_f64; k];
    let mut weights: Vec<f64> = vec![1.0; n];
    let mut diff = f64::INFINITY;
    let mut n_iter = 0usize;
    while n_iter < MAX_ITER && diff > P_TOL {
        n_iter += 1;
        let beta_new = wls_step(y, x_cols, &weights)?;
        // Residuals on the original scale, floored and check-scaled into
        // the next iteration's inverse weights.
        for t in 0..n {
            let mut r = y[t] - row_dot(x_cols, &beta_new, t);
            if r.abs() < RESID_FLOOR {
                r = if r >= 0.0 { RESID_FLOOR } else { -RESID_FLOOR };
            }
            let scaled = if r < 0.0 { tau * r } else { (1.0 - tau) * r };
            weights[t] = scaled.abs();
        }
        diff = beta
            .iter()
            .zip(beta_new.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        beta = beta_new;
    }
    let converged = diff <= P_TOL;

    // --- Powell kernel sandwich (statsmodels vcov="robust") ---------------
    let resid: Vec<f64> = (0..n).map(|t| y[t] - row_dot(x_cols, &beta, t)).collect();

    let iqre = percentile(&resid, 0.75) - percentile(&resid, 0.25);
    let h_hs = hall_sheather(n, tau)?;
    if tau + h_hs >= 1.0 || tau - h_hs <= 0.0 {
        return Err(QuantileError::DegenerateBandwidth {
            tau,
            n,
            what: "the Hall-Sheather offset pushes tau +/- h outside (0, 1)",
        });
    }
    let spread = (inv_norm_cdf(tau + h_hs)? - inv_norm_cdf(tau - h_hs)?).abs();
    let bandwidth = std_pop(y).min(iqre / 1.34) * spread;
    if bandwidth <= 0.0 || !bandwidth.is_finite() {
        return Err(QuantileError::DegenerateBandwidth {
            tau,
            n,
            what: "the residual scale (min of std(y) and IQR/1.34) collapsed to zero",
        });
    }
    // Epanechnikov density-at-zero estimate.
    let mut kernel_sum = 0.0_f64;
    for &e in &resid {
        let u = e / bandwidth;
        if u.abs() <= 1.0 {
            kernel_sum += 0.75 * (1.0 - u * u);
        }
    }
    let fhat0 = kernel_sum / (n as f64 * bandwidth);
    if fhat0 <= 0.0 || !fhat0.is_finite() {
        return Err(QuantileError::DegenerateBandwidth {
            tau,
            n,
            what: "the kernel density estimate of the residuals at zero vanished",
        });
    }

    // Bread: (X'X)^{-1} via an SPD Cholesky on the ORIGINAL design.
    let xtx = Mat::from_fn(k, k, |i, j| {
        x_cols[i]
            .iter()
            .zip(x_cols[j].iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
    });
    let xtx_inv = xtx
        .llt(Side::Lower)
        .map_err(|_| QuantileError::Singular {
            what: "quantile-regression design cross-moment X'X",
        })?
        .inverse();

    // Meat: X' D X with d_t = (tau/f)^2 for e_t > 0, ((1-tau)/f)^2 otherwise.
    let dpos = (tau / fhat0) * (tau / fhat0);
    let dneg = ((1.0 - tau) / fhat0) * ((1.0 - tau) / fhat0);
    let mut xtdx = vec![0.0_f64; k * k];
    for (t, &e) in resid.iter().enumerate() {
        let d = if e > 0.0 { dpos } else { dneg };
        for i in 0..k {
            let xi = x_cols[i][t];
            for j in 0..=i {
                xtdx[i * k + j] += d * xi * x_cols[j][t];
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            xtdx[j * k + i] = xtdx[i * k + j];
        }
    }

    // cov = bread * meat * bread.
    let mut tmp = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += xtx_inv[(i, l)] * xtdx[l * k + j];
            }
            tmp[i * k + j] = acc;
        }
    }
    let mut cov = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += tmp[i * k + l] * xtx_inv[(l, j)];
            }
            cov[i * k + j] = acc;
        }
    }

    let mut bse = Vec::with_capacity(k);
    for i in 0..k {
        let v = cov[i * k + i];
        if v < 0.0 || !v.is_finite() {
            return Err(QuantileError::Singular {
                what: "Powell sandwich covariance diagonal",
            });
        }
        bse.push(v.sqrt());
    }
    let tvalues = beta.iter().zip(bse.iter()).map(|(p, s)| p / s).collect();

    Ok(QuantileFit {
        tau,
        params: beta,
        bse,
        tvalues,
        cov,
        iterations: n_iter,
        converged,
        bandwidth,
        sparsity: 1.0 / fhat0,
    })
}

/// One weighted least-squares step: solve
/// `(X' W X) b = X' W y`, `W = diag(1 / s_t)`, by scaling both sides with
/// `1 / sqrt(s_t)` and delegating to [`tsecon_hac::ols`].
fn wls_step(y: &[f64], x_cols: &[Vec<f64>], s: &[f64]) -> Result<Vec<f64>, QuantileError> {
    let n = y.len();
    let scale: Vec<f64> = s.iter().map(|&w| 1.0 / w.sqrt()).collect();
    let ys: Vec<f64> = (0..n).map(|t| y[t] * scale[t]).collect();
    let xs: Vec<Vec<f64>> = x_cols
        .iter()
        .map(|col| (0..n).map(|t| col[t] * scale[t]).collect())
        .collect();
    Ok(tsecon_hac::ols(&ys, &xs)?.params)
}

/// `x_t' b` for row `t` of a column-stored design.
fn row_dot(x_cols: &[Vec<f64>], beta: &[f64], t: usize) -> f64 {
    x_cols
        .iter()
        .zip(beta.iter())
        .map(|(col, b)| col[t] * b)
        .sum()
}

/// Population (ddof = 0) standard deviation, matching `np.std`.
fn std_pop(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt()
}

/// Linear-interpolation percentile of `x` at fraction `p in [0, 1]`,
/// matching `scipy.stats.scoreatpercentile` (the 'fraction' method).
fn percentile(x: &[f64], p: f64) -> f64 {
    let mut s = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let pos = p * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if lo + 1 < s.len() {
        s[lo] + frac * (s[lo + 1] - s[lo])
    } else {
        s[lo]
    }
}

/// Hall-Sheather (1988) bandwidth for the density-at-zero estimate,
/// exactly statsmodels' `hall_sheather(n, q)` with `alpha = 0.05`.
fn hall_sheather(n: usize, tau: f64) -> Result<f64, QuantileError> {
    let z = inv_norm_cdf(tau)?;
    let phi_z = (-0.5 * z * z).exp() / (2.0 * core::f64::consts::PI).sqrt();
    let num = 1.5 * phi_z * phi_z;
    let den = 2.0 * z * z + 1.0;
    let z975 = inv_norm_cdf(0.975)?;
    Ok((n as f64).powf(-1.0 / 3.0) * z975.powf(2.0 / 3.0) * (num / den).powf(1.0 / 3.0))
}
