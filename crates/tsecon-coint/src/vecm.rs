//! Johansen maximum-likelihood estimation of the vector error-correction
//! model at a fixed cointegration rank, and the mapping back to the level
//! VAR companion form.
//!
//! The model (no deterministic terms, statsmodels `deterministic = "n"`) is
//!
//! ```text
//! Delta y_t = alpha beta' y_{t-1}
//!           + sum_{i=1}^{k_ar_diff} Gamma_i Delta y_{t-i} + u_t,
//! ```
//!
//! with `beta` (`k x r`) the cointegrating vectors, `alpha` (`k x r`) the
//! error-correction loadings, and `Gamma_i` the short-run dynamics. The
//! reduced-rank maximum-likelihood estimator (Johansen 1988; Lütkepohl
//! 2005, section 7.2) partials the lagged differences out of `Delta y_t`
//! and `y_{t-1}`, solves the canonical-correlation eigenproblem
//! [`crate::linalg::reduced_rank_eig`], takes the eigenvectors of the `r`
//! largest eigenvalues as `beta`, and recovers `alpha`, `Gamma`, and the
//! residual covariance by least squares.
//!
//! `beta` is normalized exactly as statsmodels does — the leading `r x r`
//! block is the identity (`beta[:r, :r] = I`), which fixes the otherwise
//! arbitrary rotation of the cointegrating space. The golden fixture
//! `fixtures/coint.json` (`vecm_rank1` block) arbitrates `alpha`, `beta`,
//! `gamma`, and the log-likelihood.

use tsecon_linalg::companion_from_var;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::CointError;
use crate::linalg::{
    check_finite, inv_general, inv_spd, ln_det_spd, partial_out, reduced_rank_eig,
};

/// Result of a rank-`r` Johansen maximum-likelihood VECM fit.
///
/// Estimator conventions match statsmodels 0.14.6 `VECM(..., coint_rank =
/// r, deterministic = "n").fit()` exactly.
#[derive(Debug, Clone)]
pub struct VecmResult {
    /// Number of series `k`.
    pub neqs: usize,
    /// Effective sample size `T` (rows after `p = k_ar_diff + 1`
    /// presample rows).
    pub nobs: usize,
    /// Number of lagged differences `k_ar_diff = p - 1`.
    pub k_ar_diff: usize,
    /// Cointegration rank `r`.
    pub coint_rank: usize,
    /// Error-correction loadings `alpha` (`k x r`).
    pub alpha: Mat<f64>,
    /// Cointegrating vectors `beta` (`k x r`), normalized so the leading
    /// `r x r` block is the identity.
    pub beta: Mat<f64>,
    /// Short-run coefficients `Gamma = [Gamma_1, ..., Gamma_{k_ar_diff}]`
    /// stacked horizontally (`k x k*k_ar_diff`); `gamma[(eq, i*k + var)]`
    /// is the effect of `Delta` variable `var` at lag `i + 1` on equation
    /// `eq`.
    pub gamma: Mat<f64>,
    /// Maximum-likelihood residual covariance `U'U / T` (`k x k`).
    pub sigma_u: Mat<f64>,
    /// The Johansen eigenvalues `lambda_1 > ... > lambda_k` from the
    /// canonical-correlation problem (decreasing).
    pub eig: Vec<f64>,
    /// Gaussian log-likelihood at the maximum (Lütkepohl 2005, eq. 7.2.20).
    pub llf: f64,
}

impl VecmResult {
    /// The long-run impact matrix `Pi = alpha beta'` (`k x k`).
    pub fn pi(&self) -> Mat<f64> {
        &self.alpha * self.beta.transpose()
    }

    /// The short-run matrix `Gamma_i` (`k x k`), for `i = 1 ..= k_ar_diff`.
    ///
    /// # Errors
    ///
    /// [`CointError::InvalidArgument`] if `i` is `0` or exceeds
    /// `k_ar_diff`.
    pub fn gamma_lag(&self, i: usize) -> Result<Mat<f64>, CointError> {
        if i == 0 || i > self.k_ar_diff {
            return Err(CointError::InvalidArgument {
                what: "gamma_lag index must satisfy 1 <= i <= k_ar_diff",
            });
        }
        let k = self.neqs;
        let base = (i - 1) * k;
        Ok(Mat::from_fn(k, k, |r, c| self.gamma[(r, base + c)]))
    }

    /// The coefficient matrices `[A_1, ..., A_p]` (`p = k_ar_diff + 1`) of
    /// the equivalent level VAR `y_t = sum_j A_j y_{t-j} + u_t`.
    ///
    /// The mapping is (Lütkepohl 2005, eq. 6.3.2, inverted)
    ///
    /// ```text
    /// A_1 = I + Pi + Gamma_1
    /// A_i = Gamma_i - Gamma_{i-1}          (2 <= i <= k_ar_diff)
    /// A_p = -Gamma_{k_ar_diff}
    /// ```
    ///
    /// with the obvious degeneracies when `k_ar_diff = 0` (`A_1 = I + Pi`).
    /// This is the utility the impulse-response layer consumes: feed the
    /// returned matrices to [`companion_from_var`] or to the VAR analysis
    /// crate.
    pub fn var_coefs(&self) -> Vec<Mat<f64>> {
        let k = self.neqs;
        let p = self.k_ar_diff + 1;
        let pi = self.pi();
        let ident = Mat::from_fn(k, k, |i, j| if i == j { 1.0 } else { 0.0 });
        // Gamma_i, with Gamma_0 and Gamma_{k_ar_diff+1} treated as zero.
        let gamma_block = |i: usize| -> Mat<f64> {
            if i == 0 || i > self.k_ar_diff {
                Mat::<f64>::zeros(k, k)
            } else {
                let base = (i - 1) * k;
                Mat::from_fn(k, k, |r, c| self.gamma[(r, base + c)])
            }
        };
        let mut coefs = Vec::with_capacity(p);
        for j in 1..=p {
            let a = if j == 1 {
                &(&ident + &pi) + &gamma_block(1)
            } else {
                &gamma_block(j) - &gamma_block(j - 1)
            };
            coefs.push(a);
        }
        coefs
    }

    /// The `kp x kp` companion matrix of the equivalent level VAR
    /// (Lütkepohl 2005, eq. 2.1.8), for downstream stability checks and
    /// impulse responses.
    ///
    /// # Errors
    ///
    /// [`CointError::Linalg`] if the companion assembly rejects the
    /// coefficient matrices (never on a well-formed fit).
    pub fn companion(&self) -> Result<Mat<f64>, CointError> {
        let coefs = self.var_coefs();
        let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
        Ok(companion_from_var(&refs)?)
    }
}

/// Estimates the VECM at cointegration rank `coint_rank` by Johansen
/// maximum likelihood, on `endog` (a `T x k` matrix, oldest row first)
/// with `k_ar_diff` lagged differences and no deterministic terms.
///
/// # Errors
///
/// * [`CointError::Dimension`] if `endog` has no columns;
/// * [`CointError::InvalidRank`] if `coint_rank` is outside `0 ..= k`;
/// * [`CointError::NonFinite`] if `endog` contains a NaN or infinity;
/// * [`CointError::InsufficientObservations`] if the effective sample is
///   too small;
/// * [`CointError::NotPositiveDefinite`] / [`CointError::Singular`] /
///   [`CointError::Linalg`] on a degenerate design or a failed
///   factorization.
pub fn fit_vecm(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    coint_rank: usize,
) -> Result<VecmResult, CointError> {
    let k = endog.ncols();
    if k == 0 {
        return Err(CointError::Dimension {
            what: "endog must have at least one column",
            expected: 1,
            got: 0,
        });
    }
    if coint_rank > k {
        return Err(CointError::InvalidRank {
            rank: coint_rank,
            neqs: k,
        });
    }
    check_finite(endog, "endog")?;
    let n = endog.nrows();
    let p = k_ar_diff + 1;
    if n <= p {
        return Err(CointError::InsufficientObservations {
            needed: p + 1,
            got: n,
        });
    }
    let t = n - p;
    let n_short = k * k_ar_diff;
    if t <= n_short + k {
        return Err(CointError::InsufficientObservations {
            needed: n_short + k + 1,
            got: t,
        });
    }

    // Sample matrices (statsmodels _endog_matrices, deterministic = "n"),
    // in T x (.) layout. Effective row i corresponds to level index p + i.
    let delta_y0 = Mat::from_fn(t, k, |i, j| endog[(p + i, j)] - endog[(p + i - 1, j)]);
    let y_lag1 = Mat::from_fn(t, k, |i, j| endog[(p + i - 1, j)]);
    let delta_x = Mat::from_fn(t, n_short, |i, col| {
        let lag = col / k + 1; // 1 ..= k_ar_diff
        let var = col % k;
        endog[(p + i - lag, var)] - endog[(p + i - lag - 1, var)]
    });

    // Auxiliary-regression residuals.
    let r0 = partial_out(delta_y0.as_ref(), delta_x.as_ref());
    let r1 = partial_out(y_lag1.as_ref(), delta_x.as_ref());

    let tf = t as f64;
    let s00 = Mat::from_fn(k, k, |i, j| dot_cols(r0.as_ref(), r0.as_ref(), i, j) / tf);
    let s01 = Mat::from_fn(k, k, |i, j| dot_cols(r0.as_ref(), r1.as_ref(), i, j) / tf);
    let s11 = Mat::from_fn(k, k, |i, j| dot_cols(r1.as_ref(), r1.as_ref(), i, j) / tf);

    let (eig, evec) = reduced_rank_eig(s00.as_ref(), s01.as_ref(), s11.as_ref())?;

    let r = coint_rank;
    // beta: the r eigenvectors of the largest eigenvalues, normalized so
    // that beta[:r, :r] = I (statsmodels normalization).
    let mut beta_raw = Mat::from_fn(k, r, |i, j| evec[(i, j)]);
    if r > 0 {
        let top = Mat::from_fn(r, r, |i, j| beta_raw[(i, j)]);
        let top_inv = inv_general(top.as_ref(), "beta[:r, :r]")?;
        beta_raw = &beta_raw * &top_inv;
    }
    let beta = beta_raw;

    // alpha = S_01 beta (beta' S_11 beta)^{-1}.
    let alpha = if r == 0 {
        Mat::<f64>::zeros(k, 0)
    } else {
        let bsb = beta.transpose() * &s11 * &beta;
        let bsb_inv = inv_general(bsb.as_ref(), "beta' S_11 beta")?;
        &s01 * &beta * &bsb_inv
    };

    // Pi = alpha beta'; Gamma from regressing the error-corrected
    // differences on the lagged differences.
    let pi = &alpha * beta.transpose();
    // W = Delta y0 - y_lag1 Pi'  (T x k).
    let w = &delta_y0 - &y_lag1 * pi.transpose();
    let gamma = if n_short == 0 {
        Mat::<f64>::zeros(k, 0)
    } else {
        let dxtdx = delta_x.transpose() * &delta_x;
        let dxtdx_inv = inv_spd(dxtdx.as_ref(), "Delta X' Delta X")?;
        // gamma = W' Delta X (Delta X' Delta X)^{-1}  (k x n_short).
        &(w.transpose() * &delta_x) * &dxtdx_inv
    };

    // Full residuals and ML covariance.
    let resid = if n_short == 0 {
        w.clone()
    } else {
        &w - &delta_x * gamma.transpose()
    };
    let sigma_u = Mat::from_fn(k, k, |i, j| {
        dot_cols(resid.as_ref(), resid.as_ref(), i, j) / tf
    });

    // Concentrated log-likelihood (Lütkepohl 2005, eq. 7.2.20;
    // statsmodels VECMResults.llf):
    // llf = -kT/2 ln(2pi) - T/2 (ln|S_00| + sum_{i<r} ln(1 - lambda_i)) - kT/2.
    let ln_det_s00 = ln_det_spd(s00.as_ref(), "S_00")?;
    let mut sum_ln = 0.0;
    for &lam in eig.iter().take(r) {
        sum_ln += (1.0 - lam).ln();
    }
    let kf = k as f64;
    let llf = -kf * tf / 2.0 * core::f64::consts::TAU.ln()
        - tf / 2.0 * (ln_det_s00 + sum_ln)
        - kf * tf / 2.0;

    Ok(VecmResult {
        neqs: k,
        nobs: t,
        k_ar_diff,
        coint_rank: r,
        alpha,
        beta,
        gamma,
        sigma_u,
        eig,
        llf,
    })
}

/// Inner product of column `a` of `x` with column `b` of `y`.
fn dot_cols(x: MatRef<'_, f64>, y: MatRef<'_, f64>, a: usize, b: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..x.nrows() {
        s += x[(i, a)] * y[(i, b)];
    }
    s
}
