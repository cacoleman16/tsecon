//! Equation-by-equation OLS estimation of the reduced-form VAR.

use tsecon_linalg::faer::linalg::solvers::{DenseSolveCore, Llt, SolveLstsq};
use tsecon_linalg::faer::{Mat, MatRef, Side};

use crate::error::{NonFiniteCell, VarError};
use crate::results::{ln_det_spd, VarResults};
use crate::spec::{Trend, VarSpec};

/// What [`VarSpec::fit`]'s `NotPositiveDefinite` names when the residual
/// covariance fails its Cholesky factorization.
pub(crate) const SIGMA_U_WHAT: &str = "Sigma_u, the residual covariance";

/// What [`VarSpec::fit`]'s `NotPositiveDefinite` names when the regressor
/// cross-product fails its Cholesky factorization.
pub(crate) const ZZ_WHAT: &str = "Z'Z, the VAR regressor cross-product";

/// The validated OLS problem of a VAR(lags) on `endog`: the estimation
/// sample, the current values `Y` and the design `Z`.
struct Design<'a> {
    y_all: MatRef<'a, f64>,
    y: Mat<f64>,
    z: Mat<f64>,
    k: usize,
    n_trend: usize,
    t_eff: usize,
    m: usize,
}

/// Validates the inputs and builds `Y` and `Z` — the part of [`estimate`]
/// the coefficients-only refit [`lag_coefs`] shares, so both see the same
/// design bit for bit.
fn design(
    endog: MatRef<'_, f64>,
    lags: usize,
    trend: Trend,
    offset: usize,
) -> Result<Design<'_>, VarError> {
    let k = endog.ncols();
    if k == 0 {
        return Err(VarError::Dimension {
            what: "the data matrix has no columns; pass a 2-D array shaped \
                   (n_obs, n_series) with observations in rows, oldest first",
            expected: 1,
            got: 0,
        });
    }
    if lags == 0 && trend == Trend::None {
        return Err(VarError::InvalidArgument {
            what: "lags = 0 with trend = \"n\" leaves no regressors at all; \
                   pass trend = \"c\" for an intercept-only model, or lags >= 1",
        });
    }
    for j in 0..k {
        for i in 0..endog.nrows() {
            if !endog[(i, j)].is_finite() {
                return Err(VarError::NonFinite {
                    what: "the data matrix",
                    at: Some(NonFiniteCell {
                        row: i,
                        col: j,
                        why: "the VAR estimator has no missing-value handling, so a single \
                              gap would corrupt every coefficient — drop or impute those \
                              rows first (pandas: df.dropna())",
                    }),
                });
            }
        }
    }
    let n_trend = trend.n_terms();
    if endog.nrows() < offset {
        return Err(VarError::InsufficientObservations {
            needed: offset,
            got: endog.nrows(),
            lags,
            neqs: k,
            n_trend,
        });
    }
    let n = endog.nrows() - offset;
    let y_all = endog.submatrix(offset, 0, n, k);

    let m = n_trend + k * lags;
    // OLS needs T = n - lags > m for a positive df_resid.
    if n < lags + m + 1 {
        return Err(VarError::InsufficientObservations {
            needed: offset + lags + m + 1,
            got: endog.nrows(),
            lags,
            neqs: k,
            n_trend,
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
    Ok(Design {
        y_all,
        y,
        z,
        k,
        n_trend,
        t_eff,
        m,
    })
}

/// The lag matrices `[A_1, ..., A_p]` of a stacked `params` matrix.
fn lag_matrices(params: &Mat<f64>, n_trend: usize, k: usize, lags: usize) -> Vec<Mat<f64>> {
    (0..lags)
        .map(|lag| Mat::from_fn(k, k, |r, c| params[(n_trend + lag * k + c, r)]))
        .collect()
}

/// Refuses a singular regressor cross-product `Z'Z`.
fn zz_cholesky(z: &Mat<f64>) -> Result<Llt<f64>, VarError> {
    let zz = z.transpose() * z;
    zz.llt(Side::Lower)
        .map_err(|_| VarError::NotPositiveDefinite { what: ZZ_WHAT })
}

/// The OLS lag matrices alone — bit-identical to [`VarSpec::fit`]'s
/// `coefs` (same design, same Householder QR), and refusing a singular
/// `Z'Z` the same way, but without the residual covariance, its
/// determinant or the information criteria. The Kilian (1998) inner bias
/// bootstrap needs only the slopes, so a pseudo-sample whose resampled
/// residuals happen to make `Sigma_u` singular does not stop it.
pub(crate) fn lag_coefs(
    endog: MatRef<'_, f64>,
    lags: usize,
    trend: Trend,
) -> Result<Vec<Mat<f64>>, VarError> {
    let d = design(endog, lags, trend, 0)?;
    let params = d.z.qr().solve_lstsq(&d.y);
    zz_cholesky(&d.z)?;
    Ok(lag_matrices(&params, d.n_trend, d.k, lags))
}

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
    let Design {
        y_all,
        y,
        z,
        k,
        n_trend,
        t_eff,
        m,
    } = design(endog, lags, trend, offset)?;
    // With df_resid = T - m < k the T x k residual matrix has rank at most
    // df_resid, so Sigma_u is singular whatever the data: say so by sample
    // size before the fit, instead of letting its Cholesky blame a column.
    if t_eff - m < k {
        return Err(VarError::SingularResidualCovariance {
            needed: offset + lags + m + k,
            got: endog.nrows(),
            lags,
            neqs: k,
            n_trend,
            df_resid: t_eff - m,
        });
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

    let zz_inv = zz_cholesky(&z)?.inverse();

    // Gaussian log-likelihood at the ML covariance (Lütkepohl 2005,
    // eq. 3.4.5; the quadratic form collapses to T k at the optimum):
    // llf = -(T k / 2) ln(2 pi) - (T / 2) ln det(Sigma_mle) - T k / 2.
    let ld = ln_det_spd(sigma_u_mle.as_ref(), SIGMA_U_WHAT)?;
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
    let coefs = lag_matrices(&params, n_trend, k, lags);

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
