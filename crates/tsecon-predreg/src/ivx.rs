//! The IVX estimator and its Wald predictability test (Kostakis, Magdalinos &
//! Stamatogiannis 2015, *Review of Financial Studies* 28:1506-1553).
//!
//! IVX instruments the persistent predictor `x_t` with a self-generated,
//! "mildly integrated" process whose persistence sits just inside the unit
//! circle. The resulting Wald statistic for `H0: beta = 0` is asymptotically
//! chi-square UNIFORMLY over the persistence of `x` — stationary,
//! local-to-unity, or exact unit root — which is the entire reason IVX
//! exists. (Validated by the crate's Monte-Carlo size test across
//! `rho in {0.9, 0.95, 0.99, 1.0}`.)
//!
//! ## The instrument (predetermination is the crux)
//!
//! ```text
//! Rz  = 1 + cz / n^alpha            (defaults cz = -1, alpha = 0.95)
//! Dx_k = x[k+1] - x[k]              (k = 0 .. n-2; carries innovation e_{k+1})
//! z_0 = 0,  z_t = Rz * z_{t-1} + Dx_{t-1}
//!      = sum_{k=0}^{t-1} Rz^{t-1-k} Dx_k
//! ```
//!
//! `z_t` accumulates predictor innovations only up to time `t`, so it is
//! predetermined with respect to `u_{t+1}` (which is correlated with the
//! *future* innovation `e_{t+1}`). That predetermination is what delivers
//! valid inference under endogeneity.
//!
//! ## Estimator and Wald statistic
//!
//! ```text
//! beta_ivx = sum_t z_t (b_t - bbar) / sum_t z_t (a_t - abar)
//! num = sum_t z_t (b_t - bbar)
//! Szz = sum_t z_t^2                 (RAW instrument second moment: z_t is a
//!                                    mean-zero mildly-integrated process, so
//!                                    it is NOT demeaned in the normaliser)
//! s2u = sum_t u_hat_t^2 / N         (OLS predictive-regression residual var.)
//! W   = num^2 / (s2u * Szz)   ~  chi-square(1),   p = chi2_sf(W, 1).
//! ```

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::chi2_sf;

use crate::align::{align_pair, check_finite, mean};
use crate::error::PredRegError;
use crate::ols::ols_predictive;

/// Configuration of the IVX self-generated instrument.
///
/// The instrument persistence is `Rz = 1 + cz / n^alpha`. The KMS (2015)
/// defaults are `cz = -1`, `alpha = 0.95`; `alpha` must lie in `(0, 1)` for
/// the "mildly integrated" asymptotics to hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvxConfig {
    /// The (negative) localizing constant `cz` in `Rz = 1 + cz / n^alpha`.
    pub cz: f64,
    /// The exponent `alpha in (0, 1)` in `Rz = 1 + cz / n^alpha`.
    pub alpha: f64,
}

impl Default for IvxConfig {
    fn default() -> Self {
        Self {
            cz: -1.0,
            alpha: 0.95,
        }
    }
}

impl IvxConfig {
    fn validate(&self) -> Result<(), PredRegError> {
        if !self.cz.is_finite() || self.cz >= 0.0 {
            return Err(PredRegError::InvalidArgument {
                what: "IVX cz must be a finite negative constant \
                       (Rz = 1 + cz/n^alpha sits just inside the unit circle)",
            });
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(PredRegError::InvalidArgument {
                what: "IVX alpha must lie in the open interval (0, 1)",
            });
        }
        Ok(())
    }

    /// The instrument persistence `Rz = 1 + cz / n^alpha` for sample size `n`.
    #[must_use]
    pub fn rz(&self, n: usize) -> f64 {
        1.0 + self.cz / (n as f64).powf(self.alpha)
    }
}

/// Build the IVX instrument path for a length-`n` predictor `x`.
///
/// Returns `z` of length `N = n - 1`, aligned to the predictor `a_t = x[t]`.
fn instrument(x: &[f64], rz: f64) -> Vec<f64> {
    let n = x.len();
    let mut z = Vec::with_capacity(n - 1);
    let mut acc = 0.0_f64;
    for k in 0..n - 1 {
        z.push(acc); // z_t uses Dx[0..t-1] only
        acc = rz * acc + (x[k + 1] - x[k]);
    }
    z
}

/// Result of the scalar IVX predictive regression and its Wald test.
#[derive(Debug, Clone, PartialEq)]
pub struct IvxResult {
    /// The IVX predictive slope `beta_ivx`.
    pub beta_ivx: f64,
    /// The IVX-Wald statistic for `H0: beta = 0`, asymptotically
    /// chi-square(1) uniformly over the persistence of `x`.
    pub wald: f64,
    /// The chi-square(1) p-value `chi2_sf(wald, 1)`.
    pub pvalue: f64,
    /// The self-generated instrument path `z_t`, length `N = n - 1`.
    pub instrument: Vec<f64>,
    /// The instrument persistence `Rz = 1 + cz / n^alpha`.
    pub rz: f64,
    /// The residual variance `sum u_hat^2 / N` used in the Wald normaliser.
    pub sigma2_u: f64,
    /// Number of aligned observations `N = n - 1`.
    pub nobs: usize,
}

/// Estimate the scalar IVX predictive regression of `r_{t+1}` on `x_t` and its
/// Wald predictability test.
///
/// # Errors
///
/// Propagates the input validation of [`ols_predictive`];
/// [`PredRegError::InvalidArgument`] for an out-of-range `config`;
/// [`PredRegError::Singular`] if the instrument-by-predictor cross moment is
/// zero (a degenerate, non-varying predictor); and [`PredRegError::Stats`]
/// from the chi-square p-value.
pub fn ivx(r: &[f64], x: &[f64], config: IvxConfig) -> Result<IvxResult, PredRegError> {
    config.validate()?;
    let (a, b) = align_pair(r, x)?;
    let ols_fit = ols_predictive(r, x)?;
    let big_n = ols_fit.nobs;

    let rz = config.rz(x.len());
    let z = instrument(x, rz);

    let abar = mean(a);
    let bbar = mean(b);
    let num: f64 = z.iter().zip(b).map(|(zt, bt)| zt * (bt - bbar)).sum();
    let den: f64 = z.iter().zip(a).map(|(zt, at)| zt * (at - abar)).sum();
    if den == 0.0 || !den.is_finite() {
        return Err(PredRegError::Singular {
            what: "IVX instrument-by-predictor cross moment sum_t z_t (a_t - abar)",
        });
    }
    let beta_ivx = num / den;

    let szz: f64 = z.iter().map(|zt| zt * zt).sum();
    let sigma2_u = ols_fit.residuals.iter().map(|e| e * e).sum::<f64>() / big_n as f64;
    if szz <= 0.0 || sigma2_u <= 0.0 {
        return Err(PredRegError::Singular {
            what: "IVX Wald normaliser s2u * sum_t z_t^2 is non-positive",
        });
    }
    let wald = num * num / (sigma2_u * szz);
    let pvalue = chi2_sf(wald, 1.0)?;

    Ok(IvxResult {
        beta_ivx,
        wald,
        pvalue,
        instrument: z,
        rz,
        sigma2_u,
        nobs: big_n,
    })
}

/// Result of the multivariate IVX predictive regression and its joint Wald
/// test.
#[derive(Debug, Clone, PartialEq)]
pub struct IvxMultiResult {
    /// The IVX predictive slopes, one per predictor column.
    pub beta_ivx: Vec<f64>,
    /// The joint IVX-Wald statistic for `H0: beta = 0` (all slopes zero),
    /// asymptotically chi-square(`q`), `q = ` number of predictors.
    pub wald: f64,
    /// The chi-square(`q`) p-value.
    pub pvalue: f64,
    /// The instrument persistence `Rz = 1 + cz / n^alpha` (shared scalar).
    pub rz: f64,
    /// The residual variance `sum u_hat^2 / N` used in the Wald normaliser.
    pub sigma2_u: f64,
    /// Number of predictors `q`.
    pub nregressors: usize,
    /// Number of aligned observations `N = n - 1`.
    pub nobs: usize,
}

/// Estimate the multivariate IVX predictive regression of `r_{t+1}` on a
/// vector predictor `x_t = (x_{1,t}, .., x_{q,t})` and its joint Wald test.
///
/// Each predictor gets its own instrument built with the shared scalar `Rz`
/// (the KMS matrix instrument specialises to `Rz I_q`). With demeaned
/// regressors `a_j` and instruments `z_i`,
///
/// ```text
/// A[i][j] = sum_t z_{i,t} (a_{j,t} - abar_j)      (q x q)
/// c[i]    = sum_t z_{i,t} (b_t - bbar)            (q)
/// beta    = A^{-1} c
/// M[i][j] = s2u * sum_t z_{i,t} z_{j,t}           (q x q, SPD)
/// W       = c' M^{-1} c   ~  chi-square(q).
/// ```
///
/// # Errors
///
/// [`PredRegError::EmptyInput`] with no predictor columns;
/// [`PredRegError::DimensionMismatch`] / [`PredRegError::NonFinite`] /
/// [`PredRegError::DegreesOfFreedom`] on malformed input;
/// [`PredRegError::InvalidArgument`] for an out-of-range `config`;
/// [`PredRegError::Singular`] if `A` or `M` is singular (collinear or
/// degenerate predictors); and [`PredRegError::Hac`] / [`PredRegError::Stats`]
/// from the OLS and chi-square layers.
pub fn ivx_multi(
    r: &[f64],
    x_cols: &[Vec<f64>],
    config: IvxConfig,
) -> Result<IvxMultiResult, PredRegError> {
    config.validate()?;
    let q = x_cols.len();
    if q == 0 {
        return Err(PredRegError::EmptyInput {
            what: "predictor columns x_cols",
        });
    }
    if r.is_empty() {
        return Err(PredRegError::EmptyInput { what: "response r" });
    }
    let n = r.len();
    for col in x_cols {
        if col.len() != n {
            return Err(PredRegError::DimensionMismatch {
                what: "predictor column vs response r",
                expected: n,
                got: col.len(),
            });
        }
        check_finite(col, "predictor column")?;
    }
    check_finite(r, "response r")?;
    let big_n = n.saturating_sub(1);
    let k = q + 1; // predictors + intercept
    if big_n <= k {
        return Err(PredRegError::DegreesOfFreedom { n: big_n, k });
    }

    // Aligned target and predictors.
    let b = &r[1..];
    let bbar = mean(b);
    let a_cols: Vec<&[f64]> = x_cols.iter().map(|c| &c[..n - 1]).collect();
    let abar: Vec<f64> = a_cols.iter().map(|a| mean(a)).collect();

    // OLS predictive residuals for the joint residual variance.
    let mut design: Vec<Vec<f64>> = Vec::with_capacity(k);
    design.push(vec![1.0_f64; big_n]);
    for a in &a_cols {
        design.push(a.to_vec());
    }
    let ols_fit = tsecon_hac::ols(b, &design)?;
    let sigma2_u = ols_fit.residuals.iter().map(|e| e * e).sum::<f64>() / big_n as f64;

    // Instruments (shared Rz).
    let rz = config.rz(n);
    let z_cols: Vec<Vec<f64>> = x_cols.iter().map(|c| instrument(c, rz)).collect();

    // A[i][j] = sum_t z_i (a_j - abar_j); c[i] = sum_t z_i (b - bbar);
    // Szz[i][j] = sum_t z_i z_j.
    let mut a_mat = vec![0.0_f64; q * q];
    let mut szz = vec![0.0_f64; q * q];
    let mut c = vec![0.0_f64; q];
    for i in 0..q {
        for t in 0..big_n {
            c[i] += z_cols[i][t] * (b[t] - bbar);
        }
        for j in 0..q {
            let mut aij = 0.0;
            let mut sij = 0.0;
            for t in 0..big_n {
                aij += z_cols[i][t] * (a_cols[j][t] - abar[j]);
                sij += z_cols[i][t] * z_cols[j][t];
            }
            a_mat[i * q + j] = aij;
            szz[i * q + j] = sigma2_u * sij;
        }
    }

    // beta = A^{-1} c  (A is general, not symmetric).
    let a_faer = Mat::from_fn(q, q, |i, j| a_mat[i * q + j]);
    let a_inv = a_faer.partial_piv_lu().inverse();
    let mut beta = vec![0.0_f64; q];
    for i in 0..q {
        let mut acc = 0.0;
        for j in 0..q {
            acc += a_inv[(i, j)] * c[j];
        }
        beta[i] = acc;
    }
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(PredRegError::Singular {
            what: "IVX instrument-by-predictor cross-moment matrix A",
        });
    }

    // W = c' M^{-1} c, M = sigma2_u * (z' z) SPD.
    let m_faer = Mat::from_fn(q, q, |i, j| szz[i * q + j]);
    let m_inv = m_faer
        .llt(Side::Lower)
        .map_err(|_| PredRegError::Singular {
            what: "IVX instrument second-moment matrix M = s2u * (z'z)",
        })?
        .inverse();
    let mut wald = 0.0_f64;
    for i in 0..q {
        for j in 0..q {
            wald += c[i] * m_inv[(i, j)] * c[j];
        }
    }
    if !wald.is_finite() || wald < 0.0 {
        return Err(PredRegError::Singular {
            what: "IVX joint Wald quadratic form c' M^{-1} c",
        });
    }
    let pvalue = chi2_sf(wald, q as f64)?;

    Ok(IvxMultiResult {
        beta_ivx: beta,
        wald,
        pvalue,
        rz,
        sigma2_u,
        nregressors: q,
        nobs: big_n,
    })
}
