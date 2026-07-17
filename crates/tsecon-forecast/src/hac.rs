//! Bartlett long-run-variance helpers shared by the predictive-ability
//! tests (Clark-West, Giacomini-White).
//!
//! Both tests need the heteroskedasticity-and-autocorrelation-consistent
//! (HAC) long-run variance of a mean-adjusted differential series. Per the
//! library ownership map (ROADMAP §5) the *scalar* estimator is delegated to
//! the single HAC engine, [`tsecon_hac::lrv`] with [`Kernel::Bartlett`], so
//! that identical `(kernel, lag)` settings can never produce different
//! p-values in different modules. Concretely, for a series `x` with sample
//! mean `xbar` and `L = lags`,
//!
//! ```text
//! omega_hat = gamma_hat(0) + 2 * sum_{k=1}^{L} (1 - k/(L+1)) * gamma_hat(k),
//! gamma_hat(k) = (1/n) * sum_{t=k}^{n-1} (x_t - xbar)(x_{t-k} - xbar)
//! ```
//!
//! i.e. biased (`1/n`) autocovariances of the **demeaned** series with the
//! Bartlett (Newey-West 1987) weights `1 - k/(L+1)`, which vanish for
//! `k > L`. This is exactly the recipe the golden CW/GW fixtures document.
//!
//! The *matrix* estimator [`bartlett_hac_matrix`] is the vector
//! generalization needed for the general q-dimensional Giacomini-White
//! conditional Wald statistic; it reduces bit-for-bit to the scalar
//! [`bartlett_lrv`] when `q == 1`.

use crate::error::ForecastError;
use tsecon_hac::{lrv, Kernel};

/// Scalar Bartlett long-run variance of a series about its **sample mean**.
///
/// Demeans `x` and delegates the kernel sum to [`tsecon_hac::lrv`] with
/// [`Kernel::Bartlett`] and lag-truncation bandwidth `lags` (so the weights
/// are `1 - k/(lags + 1)`, zero for `k > lags`). Returns the estimate
/// `omega_hat` documented in the [module docs](self).
///
/// # Errors
///
/// [`ForecastError::InvalidLrvLags`] if `lags >= n` (no room for the
/// autocovariance at that lag), or a wrapped [`ForecastError::Hac`] from the
/// underlying engine (non-finite input, `n < 2`).
pub(crate) fn bartlett_lrv(
    x: &[f64],
    lags: usize,
    what: &'static str,
) -> Result<f64, ForecastError> {
    let n = x.len();
    if lags >= n {
        return Err(ForecastError::InvalidLrvLags { what, lags, n });
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let demeaned: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    Ok(lrv(&demeaned, Kernel::Bartlett, lags as f64)?)
}

/// Matrix Bartlett HAC estimate of the long-run variance of the rows of `z`.
///
/// `z` is an `n`-by-`q` matrix stored row-major (`z[t]` is the length-`q`
/// vector at time `t`). Each column is demeaned by its own sample mean, then
///
/// ```text
/// Shat = Gamma_hat(0) + sum_{k=1}^{L} (1 - k/(L+1)) (Gamma_hat(k) + Gamma_hat(k)'),
/// Gamma_hat(k) = (1/n) * sum_{t=k}^{n-1} z_tilde_t z_tilde_{t-k}'
/// ```
///
/// with `z_tilde` the column-demeaned rows and `L = lags`. The result is the
/// symmetric `q`-by-`q` matrix (row-major). For `q == 1` this equals
/// [`bartlett_lrv`] bit-for-bit (same product order, and `w*(g + g)` is
/// `2*w*g` exactly in IEEE-754).
///
/// # Errors
///
/// [`ForecastError::InvalidLrvLags`] if `lags >= n`.
pub(crate) fn bartlett_hac_matrix(
    z: &[Vec<f64>],
    q: usize,
    lags: usize,
    what: &'static str,
) -> Result<Vec<Vec<f64>>, ForecastError> {
    let n = z.len();
    if lags >= n {
        return Err(ForecastError::InvalidLrvLags { what, lags, n });
    }
    let nf = n as f64;
    // Column means, then the demeaned rows.
    let mut mean = vec![0.0; q];
    for row in z {
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= nf;
    }
    let zt: Vec<Vec<f64>> = z
        .iter()
        .map(|row| row.iter().zip(mean.iter()).map(|(&v, &m)| v - m).collect())
        .collect();

    // Gamma(k) accumulated in the same t-order the scalar engine uses, so the
    // q == 1 diagonal matches tsecon_hac::lrv exactly.
    let gamma = |k: usize| -> Vec<Vec<f64>> {
        let mut g = vec![vec![0.0; q]; q];
        for t in k..n {
            let (a, b) = (&zt[t], &zt[t - k]);
            for i in 0..q {
                for j in 0..q {
                    g[i][j] += a[i] * b[j];
                }
            }
        }
        for row in &mut g {
            for v in row {
                *v /= nf;
            }
        }
        g
    };

    let mut shat = gamma(0);
    for k in 1..=lags {
        let w = 1.0 - k as f64 / (lags as f64 + 1.0);
        let gk = gamma(k);
        for i in 0..q {
            for j in 0..q {
                // gk + gk' then times w == 2*w*gk_sym; add symmetric part.
                shat[i][j] += w * (gk[i][j] + gk[j][i]);
            }
        }
    }
    Ok(shat)
}

/// The Wald quadratic form `n * zbar' Shat^{-1} zbar` via a Cholesky solve.
///
/// `shat` is a symmetric positive-definite `q`-by-`q` matrix (row-major),
/// `zbar` a length-`q` vector. Factors `Shat = L L'`, solves `L w = zbar` by
/// forward substitution, and returns `n * (w · w) = n * zbar' Shat^{-1}
/// zbar`. A non-positive pivot (a singular or indefinite `Shat`, e.g. from
/// collinear test functions) is a clear error rather than a NaN statistic.
///
/// # Errors
///
/// [`ForecastError::SingularWaldCovariance`] if the Cholesky factorization
/// hits a non-positive pivot.
pub(crate) fn wald_statistic(
    shat: &[Vec<f64>],
    zbar: &[f64],
    n: usize,
) -> Result<f64, ForecastError> {
    let q = zbar.len();
    // Lower-triangular Cholesky factor L with Shat = L L'.
    let mut l = vec![vec![0.0; q]; q];
    for i in 0..q {
        for j in 0..=i {
            let mut sum = shat[i][j];
            for (lik, ljk) in l[i].iter().zip(l[j].iter()).take(j) {
                sum -= lik * ljk;
            }
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return Err(ForecastError::SingularWaldCovariance { q });
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Forward-substitution solve L w = zbar; accumulate w · w.
    let mut ww = 0.0;
    let mut w = vec![0.0; q];
    for i in 0..q {
        let mut sum = zbar[i];
        for (lik, wk) in l[i].iter().zip(w.iter()).take(i) {
            sum -= lik * wk;
        }
        w[i] = sum / l[i][i];
        ww += w[i] * w[i];
    }
    Ok(n as f64 * ww)
}
