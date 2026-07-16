//! Equation-by-equation OLS estimation of the reduced-form VAR.

use tsecon_linalg::faer::linalg::solvers::{DenseSolveCore, SolveLstsq};
use tsecon_linalg::faer::{Mat, MatRef, Side};

use crate::error::VarError;
use crate::results::{ln_det_spd, VarResults};
use crate::spec::{Trend, VarSpec};

/// Core estimator shared by [`VarSpec::fit`] and
/// [`crate::select_order`].
///
/// `offset` drops that many leading rows of `endog` before anything else
/// (statsmodels' `select_order` uses `offset = maxlags - p` so every
/// candidate order is fitted on the same effective sample).
///
/// Multivariate LS (Lütkepohl 2005, section 3.2): with `Y` the
/// `T x k` matrix of current values and `Z` the `T x m` design
/// (`m = n_trend + k p` — deterministic terms first, then lag 1 of every
/// variable, then lag 2, ...), the estimator is `B = (Z'Z)^{-1} Z'Y`,
/// computed here by Householder QR least squares.
pub(crate) fn estimate(
    endog: MatRef<'_, f64>,
    lags: usize,
    trend: Trend,
    offset: usize,
) -> Result<VarResults, VarError> {
    let k = endog.ncols();
    if k == 0 {
        return Err(VarError::Dimension {
            what: "endog must have at least one column",
            expected: 1,
            got: 0,
        });
    }
    if lags == 0 && trend == Trend::None {
        return Err(VarError::InvalidArgument {
            what: "lags = 0 with Trend::None leaves no regressors; \
                   include a constant or at least one lag",
        });
    }
    for j in 0..k {
        for i in 0..endog.nrows() {
            if !endog[(i, j)].is_finite() {
                return Err(VarError::NonFinite { what: "endog" });
            }
        }
    }
    if endog.nrows() < offset {
        return Err(VarError::InsufficientObservations {
            needed: offset,
            got: endog.nrows(),
        });
    }
    let n = endog.nrows() - offset;
    let y_all = endog.submatrix(offset, 0, n, k);

    let n_trend = trend.n_terms();
    let m = n_trend + k * lags;
    // OLS needs T = n - lags > m for a positive df_resid.
    if n < lags + m + 1 {
        return Err(VarError::InsufficientObservations {
            needed: offset + lags + m + 1,
            got: endog.nrows(),
        });
    }
    let t_eff = n - lags;

    // Y: current values; Z: [trend, y_{t-1}, ..., y_{t-p}] rows.
    let y = y_all.submatrix(lags, 0, t_eff, k).to_owned();
    let mut z = Mat::<f64>::zeros(t_eff, m);
    for t in 0..t_eff {
        if n_trend == 1 {
            z[(t, 0)] = 1.0;
        }
        for lag in 1..=lags {
            for j in 0..k {
                z[(t, n_trend + (lag - 1) * k + j)] = y_all[(lags + t - lag, j)];
            }
        }
    }

    // B = argmin ||Y - Z B||_F via Householder QR (numerically matches
    // statsmodels' lstsq on these well-conditioned macro designs).
    let params = z.qr().solve_lstsq(&y);
    let resid = &y - &z * &params;

    // Residual covariances: df-adjusted (divisor T - m, statsmodels
    // sigma_u) and ML (divisor T, enters loglik and the criteria).
    let df_resid = t_eff - m;
    let rtr = resid.transpose() * &resid;
    let sigma_u = Mat::from_fn(k, k, |i, j| rtr[(i, j)] / df_resid as f64);
    let sigma_u_mle = Mat::from_fn(k, k, |i, j| rtr[(i, j)] / t_eff as f64);

    let zz = z.transpose() * &z;
    let zz_inv = zz
        .llt(Side::Lower)
        .map_err(|_| VarError::NotPositiveDefinite {
            what: "Z'Z (regressors are collinear)",
        })?
        .inverse();

    // Gaussian log-likelihood at the ML covariance (Lütkepohl 2005,
    // eq. 3.4.5; the quadratic form collapses to T k at the optimum):
    // llf = -(T k / 2) ln(2 pi) - (T / 2) ln det(Sigma_mle) - T k / 2.
    let ld = ln_det_spd(sigma_u_mle.as_ref(), "sigma_u_mle")?;
    let (tf, kf) = (t_eff as f64, k as f64);
    let llf = -0.5 * tf * kf * (1.0 + core::f64::consts::TAU.ln()) - 0.5 * tf * ld;

    // Information criteria, statsmodels conventions (Lütkepohl 2005,
    // section 4.3): free parameter count f = p k^2 + k n_trend.
    let free = (lags * k * k + k * n_trend) as f64;
    let aic = ld + 2.0 / tf * free;
    let bic = ld + tf.ln() / tf * free;
    let hqic = ld + 2.0 * tf.ln().ln() / tf * free;
    let fpe = ((tf + m as f64) / (tf - m as f64)).powi(k as i32) * ld.exp();

    let intercept: Vec<f64> = if n_trend == 1 {
        (0..k).map(|j| params[(0, j)]).collect()
    } else {
        vec![0.0; k]
    };
    let coefs: Vec<Mat<f64>> = (0..lags)
        .map(|lag| Mat::from_fn(k, k, |r, c| params[(n_trend + lag * k + c, r)]))
        .collect();

    Ok(VarResults {
        spec: VarSpec { lags, trend },
        neqs: k,
        nobs: t_eff,
        df_model: m,
        df_resid,
        endog: y_all.to_owned(),
        params,
        intercept,
        coefs,
        resid,
        sigma_u,
        sigma_u_mle,
        zz_inv,
        llf,
        aic,
        bic,
        hqic,
        fpe,
    })
}
