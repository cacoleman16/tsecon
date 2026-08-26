//! Johansen maximum-likelihood estimation of the vector error-correction
//! model at a fixed cointegration rank, and the mapping back to the level
//! VAR companion form.
//!
//! The model is
//!
//! ```text
//! Delta y_t = alpha beta' y_{t-1}
//!           + sum_{i=1}^{k_ar_diff} Gamma_i Delta y_{t-i} + C d_t + u_t,
//! ```
//!
//! with `beta` (`k x r`) the cointegrating vectors, `alpha` (`k x r`) the
//! error-correction loadings, `Gamma_i` the short-run dynamics, and
//! `C d_t` the deterministic terms chosen by [`VecmDeterministic`]:
//! either none at all (statsmodels `deterministic = "n"`, the
//! [`fit_vecm`] default) or an unrestricted constant outside the
//! cointegration relation (statsmodels `deterministic = "co"` — the case
//! [`crate::johansen`]'s `det_order = 0` convention assumes). The
//! reduced-rank maximum-likelihood estimator (Johansen 1988; Lütkepohl
//! 2005, section 7.2) partials the lagged differences (and any
//! deterministic terms) out of `Delta y_t` and `y_{t-1}`, solves the
//! canonical-correlation eigenproblem
//! [`crate::linalg::reduced_rank_eig`], takes the eigenvectors of the `r`
//! largest eigenvalues as `beta`, and recovers `alpha`, `Gamma`, the
//! deterministic coefficients, and the residual covariance by least
//! squares.
//!
//! The two deterministic cases answer *different models*: on drifting
//! data the no-deterministic fit must absorb the drift and the mean of
//! the equilibrium error into `alpha beta' y_{t-1}`, which rotates `beta`
//! away from the constant-adjusted cointegrating space the Johansen rank
//! test (`det_order = 0`) works in. Fit with
//! [`VecmDeterministic::Constant`] when the rank came from
//! [`crate::johansen`].
//!
//! `beta` is normalized exactly as statsmodels does — the leading `r x r`
//! block is the identity (`beta[:r, :r] = I`), which fixes the otherwise
//! arbitrary rotation of the cointegrating space. The golden fixtures
//! `fixtures/coint.json` (`vecm_rank1` block, `deterministic = "n"`) and
//! `fixtures/vecm_deterministic.json` (both cases on drifting data)
//! arbitrate `alpha`, `beta`, `gamma`, `det_coef`, and the
//! log-likelihood.

use tsecon_linalg::companion_from_var;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::CointError;
use crate::linalg::{
    check_finite, inv_general, inv_spd, ln_det_spd, partial_out, reduced_rank_eig,
};

/// The deterministic-term specification of a VECM fit.
///
/// Named after the statsmodels `VECM(..., deterministic = ...)` string it
/// reproduces. The two cases supported today are the two ends of the
/// classic replication trap: no deterministic terms at all, and the
/// unrestricted constant the Johansen rank test ([`crate::johansen`],
/// statsmodels `coint_johansen(det_order = 0)`) assumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VecmDeterministic {
    /// No deterministic terms — statsmodels `deterministic = "n"`. The
    /// historical (and current) default of [`fit_vecm`].
    #[default]
    None,
    /// An unrestricted constant outside the cointegration relation —
    /// statsmodels `deterministic = "co"`. The short-run equation gains
    /// an intercept (returned in [`VecmResult::det_coef`]), and the
    /// reduced-rank step partials the constant out alongside the lagged
    /// differences, which makes the estimated cointegrating space match
    /// the one [`crate::johansen`] (`det_order = 0`) tests.
    Constant,
}

impl VecmDeterministic {
    /// Number of deterministic regressors in the short-run equation.
    fn n_det(self) -> usize {
        match self {
            VecmDeterministic::None => 0,
            VecmDeterministic::Constant => 1,
        }
    }
}

/// Result of a rank-`r` Johansen maximum-likelihood VECM fit.
///
/// Estimator conventions match statsmodels 0.14.6 `VECM(..., coint_rank =
/// r, deterministic = d).fit()` exactly, for the supported `d`
/// ([`VecmDeterministic`]).
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
    /// The deterministic-term specification the model was fit under.
    pub deterministic: VecmDeterministic,
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
    /// Coefficients of the deterministic terms outside the cointegration
    /// relation (statsmodels `det_coef`): `k x 0` under
    /// [`VecmDeterministic::None`], the `k x 1` intercept of each
    /// short-run equation under [`VecmDeterministic::Constant`].
    pub det_coef: Mat<f64>,
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
    /// crate. Only the autoregressive part is returned: under
    /// [`VecmDeterministic::Constant`] the VECM intercept carries over to
    /// the level VAR unchanged (`nu = det_coef`) and does not enter the
    /// `A_j`.
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
/// with `k_ar_diff` lagged differences and **no deterministic terms**
/// (statsmodels `deterministic = "n"`).
///
/// This is [`fit_vecm_det`] with [`VecmDeterministic::None`], kept as the
/// historical default. Note the Johansen rank test ([`crate::johansen`])
/// assumes an unrestricted constant instead — to estimate the same model
/// the test ranks, call [`fit_vecm_det`] with
/// [`VecmDeterministic::Constant`].
///
/// # Errors
///
/// As [`fit_vecm_det`].
pub fn fit_vecm(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    coint_rank: usize,
) -> Result<VecmResult, CointError> {
    fit_vecm_det(endog, k_ar_diff, coint_rank, VecmDeterministic::None)
}

/// Estimates the VECM at cointegration rank `coint_rank` by Johansen
/// maximum likelihood, on `endog` (a `T x k` matrix, oldest row first)
/// with `k_ar_diff` lagged differences and the deterministic terms chosen
/// by `deterministic`.
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
pub fn fit_vecm_det(
    endog: MatRef<'_, f64>,
    k_ar_diff: usize,
    coint_rank: usize,
    deterministic: VecmDeterministic,
) -> Result<VecmResult, CointError> {
    let k = endog.ncols();
    if k == 0 {
        return Err(CointError::Dimension {
            what: "the data matrix has no columns; pass a 2-D array shaped \
                   (n_obs, n_series) with observations in rows, oldest first",
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
    check_finite(endog, "the data matrix")?;
    let n = endog.nrows();
    let p = k_ar_diff + 1;
    if n <= p {
        return Err(CointError::InsufficientObservations {
            needed: k * k_ar_diff + k + 1,
            got: 0,
            nobs: n,
            neqs: k,
            k_ar_diff,
        });
    }
    let t = n - p;
    let n_short = k * k_ar_diff;
    let n_det = deterministic.n_det();
    let n_reg = n_short + n_det;
    if t <= n_reg + k {
        return Err(CointError::InsufficientObservations {
            needed: n_reg + k + 1,
            got: t,
            nobs: n,
            neqs: k,
            k_ar_diff,
        });
    }

    // Sample matrices (statsmodels _endog_matrices), in T x (.) layout.
    // Effective row i corresponds to level index p + i. The short-run
    // regressor block stacks the lagged differences first and then the
    // deterministic terms (a column of ones for the unrestricted
    // constant), exactly as statsmodels stacks delta_x.
    let delta_y0 = Mat::from_fn(t, k, |i, j| endog[(p + i, j)] - endog[(p + i - 1, j)]);
    let y_lag1 = Mat::from_fn(t, k, |i, j| endog[(p + i - 1, j)]);
    let delta_x = Mat::from_fn(t, n_reg, |i, col| {
        if col >= n_short {
            return 1.0; // the unrestricted constant
        }
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
        let top_inv = inv_general(
            top.as_ref(),
            "beta[:r, :r], the block used to normalize the cointegrating vectors; \
             coint_rank exceeds the number of independent cointegrating relations \
             in this sample — lower coint_rank (johansen() reports the rank it \
             selects at 5%)",
        )?;
        beta_raw = &beta_raw * &top_inv;
    }
    let beta = beta_raw;

    // alpha = S_01 beta (beta' S_11 beta)^{-1}.
    let alpha = if r == 0 {
        Mat::<f64>::zeros(k, 0)
    } else {
        let bsb = beta.transpose() * &s11 * &beta;
        let bsb_inv = inv_general(
            bsb.as_ref(),
            "beta' S_11 beta, the cointegrating-space second moment; two series are \
             collinear, so the cointegrating space is not identified — drop the \
             redundant series",
        )?;
        &s01 * &beta * &bsb_inv
    };

    // Pi = alpha beta'; Gamma (and the deterministic coefficients) from
    // regressing the error-corrected differences on the short-run block.
    let pi = &alpha * beta.transpose();
    // W = Delta y0 - y_lag1 Pi'  (T x k).
    let w = &delta_y0 - &y_lag1 * pi.transpose();
    let coef = if n_reg == 0 {
        Mat::<f64>::zeros(k, 0)
    } else {
        let dxtdx = delta_x.transpose() * &delta_x;
        let dxtdx_inv = inv_spd(
            dxtdx.as_ref(),
            "Delta X' Delta X, the short-run regressor cross-product",
        )?;
        // coef = W' Delta X (Delta X' Delta X)^{-1}  (k x n_reg).
        &(w.transpose() * &delta_x) * &dxtdx_inv
    };
    // Split as statsmodels does: lagged-difference columns first, then
    // the deterministic terms.
    let gamma = Mat::from_fn(k, n_short, |i, j| coef[(i, j)]);
    let det_coef = Mat::from_fn(k, n_det, |i, j| coef[(i, n_short + j)]);

    // Full residuals and ML covariance.
    let resid = if n_reg == 0 {
        w.clone()
    } else {
        &w - &delta_x * coef.transpose()
    };
    let sigma_u = Mat::from_fn(k, k, |i, j| {
        dot_cols(resid.as_ref(), resid.as_ref(), i, j) / tf
    });

    // Concentrated log-likelihood (Lütkepohl 2005, eq. 7.2.20;
    // statsmodels VECMResults.llf):
    // llf = -kT/2 ln(2pi) - T/2 (ln|S_00| + sum_{i<r} ln(1 - lambda_i)) - kT/2.
    let ln_det_s00 = ln_det_spd(
        s00.as_ref(),
        "S_00, the second-moment matrix of the differenced residuals",
    )?;
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
        deterministic,
        alpha,
        beta,
        gamma,
        det_coef,
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
