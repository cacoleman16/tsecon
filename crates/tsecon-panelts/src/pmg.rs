//! The Pesaran, Shin & Smith (1999) pooled-mean-group (PMG) ARDL estimator.
//!
//! PMG sits between the fully heterogeneous [mean group](crate::mg) (every
//! coefficient free per unit) and a fully pooled dynamic panel (every
//! coefficient common): it **pools the long-run coefficients** across units by
//! maximum likelihood while leaving the **error-correction speed and the
//! short-run dynamics free** per unit. This is the natural estimator when
//! economic theory implies a common long-run relationship (a common
//! cointegrating vector) but the speed of adjustment back to it, and the
//! short-run response, legitimately differ across units.
//!
//! ## The ARDL(1,1) error-correction model
//!
//! For each unit `i`, start from the autoregressive-distributed-lag ARDL(1,1)
//!
//! ```text
//! y_it = mu_i + lambda_i y_{i,t-1} + delta_i0' x_it + delta_i1' x_{i,t-1} + e_it .
//! ```
//!
//! Subtracting `y_{i,t-1}` and adding/subtracting `delta_i0' x_{i,t-1}` gives
//! the Pesaran-Shin-Smith error-correction reparameterization
//!
//! ```text
//! Δy_it = phi_i ( y_{i,t-1} - theta' x_{i,t-1} ) + delta_i0' Δx_it + mu_i + e_it,
//! ```
//!
//! where
//!
//! ```text
//! phi_i  = lambda_i - 1                         (error-correction speed, < 0),
//! theta  = (delta_i0 + delta_i1) / (1 - lambda_i)   (long-run coefficient).
//! ```
//!
//! The **PMG restriction** is that the long-run vector `theta` is *common*
//! across units, while `phi_i`, the short-run slope `delta_i0`, and the
//! intercept `mu_i` stay unit-specific. The error-correction term
//! `xi_{i,t-1}(theta) = y_{i,t-1} - theta' x_{i,t-1}` is the deviation from the
//! common long-run equilibrium; `phi_i` measures how fast unit `i` closes it.
//!
//! ## Estimation: the PSS concentrated-ML back-substitution
//!
//! Assume `e_it ~ N(0, sigma_i^2)` independently. Collect for each unit the
//! `T_i` usable rows (one period is lost to the lag, one to the difference),
//! the short-run regressors `W_i = [const, Δx_i]`, the differenced response
//! `Δy_i`, the lagged level `y_{i,-1}`, and the lagged regressors `X_{i,-1}`.
//! By Frisch-Waugh-Lovell, concentrate the short-run block out by partialling
//! every quantity on `W_i` (least-squares residuals, marked with a tilde):
//! `Δỹ_i`, `ỹ_{i,-1}`, `X̃_{i,-1}`. The model in the partialled space is
//!
//! ```text
//! Δỹ_i = phi_i ( ỹ_{i,-1} - X̃_{i,-1} theta ) + ẽ_i .
//! ```
//!
//! The concentrated (over `W_i`, `phi_i`, `sigma_i^2`) log-likelihood in the
//! single common parameter `theta` is
//!
//! ```text
//! l(theta) = -1/2 sum_i T_i [ log(2 pi) + 1 + log sigma_i^2(theta) ],
//! ```
//!
//! with, given `theta`, per unit
//!
//! ```text
//! xi~_i    = ỹ_{i,-1} - X̃_{i,-1} theta,
//! phi_i    = (xi~_i' xi~_i)^{-1} xi~_i' Δỹ_i,
//! sigma_i^2 = (1/T_i) || Δỹ_i - phi_i xi~_i ||^2 .
//! ```
//!
//! The first-order condition for `theta`, given `{phi_i, sigma_i^2}`, is the
//! feasible-GLS pooled update (PSS 1999, appendix)
//!
//! ```text
//! A theta = b,
//!   A = sum_i (phi_i^2 / sigma_i^2) X̃_{i,-1}' X̃_{i,-1},
//!   b = - sum_i (phi_i / sigma_i^2) X̃_{i,-1}' ( Δỹ_i - phi_i ỹ_{i,-1} ).
//! ```
//!
//! Iterating "given `theta` -> `{phi_i, sigma_i^2}`; given those -> `theta`"
//! is the PSS back-substitution and maximizes the concentrated likelihood. At
//! the fixed point the long-run covariance is the inverse information block
//!
//! ```text
//! Var(theta) = A^{-1},   SE(theta_k) = sqrt( [A^{-1}]_kk ),
//! ```
//!
//! `phi_bar = (1/N) sum_i phi_i` is the average adjustment speed, and the
//! reported log-likelihood is `l` evaluated at the converged estimates.
//!
//! ## Scope
//!
//! This module implements the standard **ARDL(1,1)** PMG solidly. General
//! ARDL(p, q) lag orders (extra `Δy` and `Δx` lags in the short-run block) are
//! a documented `TODO`: the only change is a wider `W_i`, but the public input
//! shape and a lag-order argument are deferred to keep this deliverable
//! focused.
//!
//! The per-unit OLS residuals (the partialling) come from [`tsecon_hac::ols`];
//! the pooled `k x k` solve and inverse come from `faer` via
//! [`tsecon_linalg`]. Nothing here reimplements least squares or a factorization.

use tsecon_hac::ols;
use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};

use crate::error::PanelTsError;
use crate::mg::{validate_units, PanelUnit};

/// Maximum number of back-substitution iterations before declaring failure.
const MAX_ITER: usize = 1000;
/// Convergence tolerance on the max-abs change in `theta` between iterations.
const TOL: f64 = 1e-12;

/// A fitted pooled-mean-group (PMG) estimate for an ARDL(1,1) panel.
///
/// `theta` and `theta_se` are indexed by long-run regressor `k` in the order
/// the columns were supplied in [`PanelUnit::x`]. `phi` is indexed by unit.
#[derive(Debug, Clone, PartialEq)]
pub struct PooledMeanGroup {
    /// Pooled long-run coefficient vector `theta` (common across units),
    /// length `k`.
    pub theta: Vec<f64>,
    /// Standard errors of `theta`, `sqrt(diag(A^{-1}))`, length `k`.
    pub theta_se: Vec<f64>,
    /// Average error-correction speed `phi_bar = mean_i phi_i` (negative under
    /// stable adjustment to the common long run).
    pub phi_bar: f64,
    /// Per-unit error-correction speeds `phi_i`, length `N`.
    pub phi: Vec<f64>,
    /// Per-unit innovation variances `sigma_i^2 = RSS_i / T_i`, length `N`.
    pub sigma2: Vec<f64>,
    /// Maximized concentrated log-likelihood at the converged estimates.
    pub loglik: f64,
    /// Number of back-substitution iterations run to convergence.
    pub iterations: usize,
    /// Number of units `N`.
    pub n_units: usize,
    /// Number of long-run regressors `k`.
    pub k: usize,
}

/// Per-unit data prepared for the PMG iteration: everything already partialled
/// on the short-run block `W_i = [const, Δx_i]`.
struct PreparedUnit {
    /// `Δỹ_i`: differenced response, residualized on `W_i`. Length `T_i`.
    dy: Vec<f64>,
    /// `ỹ_{i,-1}`: lagged level, residualized on `W_i`. Length `T_i`.
    ylag: Vec<f64>,
    /// `X̃_{i,-1}`: `k` lagged-regressor columns, each residualized on `W_i`.
    xlag: Vec<Vec<f64>>,
    /// Effective number of rows `T_i`.
    t: usize,
}

/// Build the ARDL(1,1) error-correction rows for one unit and partial the
/// error-correction pieces (`Δy`, `y_{-1}`, each `x_{-1}` column) on the
/// short-run block `W = [const, Δx]` via OLS residuals (Frisch-Waugh-Lovell).
fn prepare_unit(unit: &PanelUnit, i: usize, k: usize) -> Result<PreparedUnit, PanelTsError> {
    let t_raw = unit.t();
    // Rows are indexed by t = 1 .. t_raw-1 (0-based), so T = t_raw - 1. The
    // partialling design W = [const, Δx] has 1 + k columns; the subsequent
    // error-correction regression needs T > 1 + k. Require T >= k + 2, i.e.
    // t_raw >= k + 3, and additionally t_raw >= 2 to form a lag/difference.
    let needed = k + 3;
    if t_raw < needed {
        return Err(PanelTsError::PmgTooFewPeriods {
            unit: i,
            got: t_raw,
            needed,
        });
    }
    let t = t_raw - 1;

    let mut dy = vec![0.0_f64; t];
    let mut ylag = vec![0.0_f64; t];
    let mut dx = vec![vec![0.0_f64; t]; k];
    let mut xlag = vec![vec![0.0_f64; t]; k];
    for row in 0..t {
        let tt = row + 1; // actual time index (0-based) in the raw series
        dy[row] = unit.y[tt] - unit.y[tt - 1];
        ylag[row] = unit.y[tt - 1];
        for j in 0..k {
            dx[j][row] = unit.x[j][tt] - unit.x[j][tt - 1];
            xlag[j][row] = unit.x[j][tt - 1];
        }
    }

    // Short-run partialling design W = [const, Δx_0, ..., Δx_{k-1}].
    let mut w = Vec::with_capacity(k + 1);
    w.push(vec![1.0_f64; t]);
    w.extend(dx);

    let residualize = |target: &[f64]| -> Result<Vec<f64>, PanelTsError> {
        let fit = ols(target, &w).map_err(|source| PanelTsError::Ols { unit: i, source })?;
        Ok(fit.residuals)
    };

    let dy_t = residualize(&dy)?;
    let ylag_t = residualize(&ylag)?;
    let mut xlag_t = Vec::with_capacity(k);
    for col in &xlag {
        xlag_t.push(residualize(col)?);
    }

    Ok(PreparedUnit {
        dy: dy_t,
        ylag: ylag_t,
        xlag: xlag_t,
        t,
    })
}

/// Given the current `theta`, compute per-unit `phi_i` and `sigma_i^2` from the
/// partialled data (the "given theta -> {phi, sigma2}" half of the iteration).
fn phi_sigma_given_theta(units: &[PreparedUnit], theta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut phi = Vec::with_capacity(units.len());
    let mut sigma2 = Vec::with_capacity(units.len());
    for u in units {
        // xi~ = ylag~ - X~ theta.
        let mut num = 0.0_f64; // xi' Δỹ
        let mut den = 0.0_f64; // xi' xi
        let mut xi = vec![0.0_f64; u.t];
        for row in 0..u.t {
            let mut x = u.ylag[row];
            for (j, col) in u.xlag.iter().enumerate() {
                x -= theta[j] * col[row];
            }
            xi[row] = x;
            num += x * u.dy[row];
            den += x * x;
        }
        let phi_i = if den > 0.0 { num / den } else { 0.0 };
        let rss: f64 =
            u.dy.iter()
                .zip(xi.iter())
                .map(|(&dyv, &xiv)| {
                    let r = dyv - phi_i * xiv;
                    r * r
                })
                .sum();
        phi.push(phi_i);
        sigma2.push(rss / u.t as f64);
    }
    (phi, sigma2)
}

/// Assemble the pooled `k x k` system `A theta = b` from the partialled data
/// and the current `{phi_i, sigma_i^2}` (the "given {phi, sigma2} -> theta"
/// half). Returns `(A, b)` in `faer` form.
fn pooled_system(
    units: &[PreparedUnit],
    phi: &[f64],
    sigma2: &[f64],
    k: usize,
) -> (Mat<f64>, Mat<f64>) {
    let mut a = vec![0.0_f64; k * k];
    let mut b = vec![0.0_f64; k];
    for ((u, &phi_i), &s2) in units.iter().zip(phi.iter()).zip(sigma2.iter()) {
        let wgt_a = phi_i * phi_i / s2; // phi^2 / sigma^2
        let wgt_b = phi_i / s2; // phi   / sigma^2
        for r in 0..k {
            let xr = &u.xlag[r];
            // b_r -= (phi/sigma^2) * X~_r' d,  d = Δỹ - phi * ỹ_{-1}
            let br: f64 = xr
                .iter()
                .zip(u.dy.iter())
                .zip(u.ylag.iter())
                .map(|((&xv, &dyv), &ylv)| xv * (dyv - phi_i * ylv))
                .sum();
            b[r] -= wgt_b * br;
            // A_rc += (phi^2/sigma^2) * X~_r' X~_c
            for c in 0..k {
                let xc = &u.xlag[c];
                let mut arc = 0.0_f64;
                for row in 0..u.t {
                    arc += xr[row] * xc[row];
                }
                a[r * k + c] += wgt_a * arc;
            }
        }
    }
    let a_mat = Mat::from_fn(k, k, |i, j| a[i * k + j]);
    let b_mat = Mat::from_fn(k, 1, |i, _| b[i]);
    (a_mat, b_mat)
}

/// The Pesaran, Shin & Smith (1999) pooled-mean-group ARDL(1,1) estimator.
///
/// Fits the error-correction ARDL(1,1) panel with a **common long-run
/// coefficient** `theta` (pooled by maximum likelihood) and **free** per-unit
/// adjustment speeds `phi_i`, short-run slopes, and intercepts, via the PSS
/// concentrated-likelihood back-substitution (see the module docs for the
/// exact equations). The panel may be *unbalanced* (different `T_i`).
///
/// Returns the pooled `theta` with its standard error, the average adjustment
/// speed `phi_bar`, the per-unit `phi_i`, the per-unit `sigma_i^2`, and the
/// maximized log-likelihood.
///
/// # Errors
///
/// [`PanelTsError::TooFewUnits`] for `N < 2`; [`PanelTsError::NoRegressors`],
/// [`PanelTsError::InconsistentRegressors`], or [`PanelTsError::RaggedUnit`]
/// for malformed units; [`PanelTsError::PmgTooFewPeriods`] if a unit is too
/// short for the ARDL(1,1) reparameterization; [`PanelTsError::Ols`] wrapping a
/// per-unit partialling OLS failure (e.g. collinear `Δx`);
/// [`PanelTsError::PmgSingularLongRun`] if the pooled long-run cross-product is
/// not positive definite; [`PanelTsError::PmgNotConverged`] if the iteration
/// does not converge within the internal budget.
pub fn pmg(units: &[PanelUnit]) -> Result<PooledMeanGroup, PanelTsError> {
    let k = validate_units(units)?;
    let n = units.len();

    let prepared: Vec<PreparedUnit> = units
        .iter()
        .enumerate()
        .map(|(i, u)| prepare_unit(u, i, k))
        .collect::<Result<_, _>>()?;

    // Deterministic start theta = 0 (identical in the NumPy golden), then
    // iterate the PSS back-substitution to the concentrated-ML fixed point.
    let mut theta = vec![0.0_f64; k];
    let mut iterations = 0;
    let mut converged = false;

    for iter in 1..=MAX_ITER {
        let (phi_new, sigma2_new) = phi_sigma_given_theta(&prepared, &theta);
        let (a_mat, b_mat) = pooled_system(&prepared, &phi_new, &sigma2_new, k);
        let a_inv = a_mat
            .llt(Side::Lower)
            .map_err(|_| PanelTsError::PmgSingularLongRun)?
            .inverse();
        let sol = &a_inv * &b_mat;
        let theta_new: Vec<f64> = (0..k).map(|i| sol[(i, 0)]).collect();

        let delta = theta_new
            .iter()
            .zip(theta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        theta = theta_new;
        iterations = iter;

        if delta < TOL {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(PanelTsError::PmgNotConverged { iters: MAX_ITER });
    }

    // Compute phi / sigma2 at the converged theta so all reported quantities
    // are mutually consistent with it.
    let (phi, sigma2) = phi_sigma_given_theta(&prepared, &theta);

    // Long-run covariance Var(theta) = A^{-1} at the converged estimates.
    let (a_mat, _b) = pooled_system(&prepared, &phi, &sigma2, k);
    let a_inv = a_mat
        .llt(Side::Lower)
        .map_err(|_| PanelTsError::PmgSingularLongRun)?
        .inverse();
    let theta_se: Vec<f64> = (0..k).map(|i| a_inv[(i, i)].max(0.0).sqrt()).collect();

    let phi_bar = phi.iter().sum::<f64>() / n as f64;

    // Concentrated log-likelihood at the converged estimates:
    // l = -1/2 sum_i T_i [ log(2 pi) + 1 + log sigma_i^2 ].
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    let mut loglik = 0.0_f64;
    for (u, &s2) in prepared.iter().zip(sigma2.iter()) {
        loglik += -0.5 * u.t as f64 * (ln_2pi + 1.0 + s2.ln());
    }

    Ok(PooledMeanGroup {
        theta,
        theta_se,
        phi_bar,
        phi,
        sigma2,
        loglik,
        iterations,
        n_units: n,
        k,
    })
}
