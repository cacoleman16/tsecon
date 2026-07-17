//! OLS with nonrobust, heteroskedasticity-robust (HC0/HC1), and HAC
//! (kernel sandwich) standard errors, matching statsmodels
//! `OLS(...).fit(cov_type=...)` conventions exactly.
//!
//! The point estimates solve the normal equations `(X'X) b = X'y` with a
//! dense Cholesky factorization plus one step of iterative refinement
//! (mirroring the private helper in `tsecon-diag`). The design matrix is
//! passed as explicit columns — include the constant column yourself,
//! statsmodels-style.
//!
//! // TODO(phase0): delegate the solve to shared linalg (tsecon-linalg QR)
//! // once the workspace-wide linear algebra layer stabilizes.
//!
//! Sandwich covariances are `(X'X)^{-1} S (X'X)^{-1}` with the meat `S`
//! built from scores `s_t = x_t u_t`:
//!
//! * nonrobust: `sigma2_hat (X'X)^{-1}`, `sigma2_hat = RSS/(n - k)`;
//! * HC0 (White 1980): `S = sum_t u_t^2 x_t x_t'`;
//! * HC1 (MacKinnon & White 1985): HC0 scaled by `n/(n - k)`;
//! * HAC (Newey & West 1987): `S = Gamma_0 + sum_{j>=1} w_j (Gamma_j +
//!   Gamma_j')` with `Gamma_j = sum_{t>j} s_t s_{t-j}'` and kernel weights
//!   `w_j` from [`Kernel::weight`]; optionally scaled by `n/(n - k)`
//!   (statsmodels `use_correction=True`).

use crate::error::HacError;
use crate::kernel::Kernel;
use crate::validate::{check_bandwidth, check_finite};

/// Standard-error flavor for [`OlsFit::inference`], mirroring statsmodels
/// `cov_type=`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeType {
    /// Classical spherical-errors covariance `sigma2_hat (X'X)^{-1}` with
    /// `sigma2_hat = RSS/(n - k)` (statsmodels `cov_type="nonrobust"`).
    NonRobust,
    /// White (1980) heteroskedasticity-robust covariance
    /// (statsmodels `cov_type="HC0"`).
    Hc0,
    /// HC0 with the `n/(n - k)` degrees-of-freedom inflation
    /// (MacKinnon & White 1985; statsmodels `cov_type="HC1"`).
    Hc1,
    /// Kernel HAC sandwich covariance (statsmodels `cov_type="HAC"` with
    /// `cov_kwds={"maxlags": bandwidth, "use_correction": ...}` when the
    /// kernel is [`Kernel::Bartlett`]).
    Hac {
        /// Kernel weighting the score autocovariances.
        kernel: Kernel,
        /// Bandwidth in the [`Kernel::weight`] convention (for
        /// Bartlett/Parzen/truncated this is the lag-truncation `maxlags`).
        bandwidth: f64,
        /// Apply the small-sample `n/(n - k)` correction to the covariance
        /// (statsmodels `use_correction`, default `true` there).
        use_correction: bool,
    },
}

/// Standard errors, t-statistics, and the full parameter covariance for
/// one [`SeType`].
#[derive(Debug, Clone, PartialEq)]
pub struct OlsInference {
    /// Parameter covariance matrix, `k x k` row-major.
    pub cov: Vec<f64>,
    /// Standard errors `sqrt(diag(cov))`, one per parameter.
    pub bse: Vec<f64>,
    /// t-statistics `params / bse`, one per parameter.
    pub tvalues: Vec<f64>,
}

/// A fitted OLS regression; produced by [`ols`].
#[derive(Debug, Clone, PartialEq)]
pub struct OlsFit {
    /// Coefficient estimates, in the order the design columns were passed.
    pub params: Vec<f64>,
    /// Residuals `u_t = y_t - x_t' b`, length `n`.
    pub residuals: Vec<f64>,
    /// Number of observations `n`.
    pub nobs: usize,
    /// Number of regressors `k` (including the constant if supplied).
    pub nparams: usize,
    /// `(X'X)^{-1}`, `k x k` row-major — the sandwich "bread".
    xtx_inv: Vec<f64>,
    /// The design columns (owned copy), needed for the sandwich meats.
    x_cols: Vec<Vec<f64>>,
}

/// Fit `y = X b + u` by ordinary least squares.
///
/// `x_cols` are the columns of the design matrix; include the constant
/// column explicitly (statsmodels exog convention — the golden fixture
/// assembles `[const, x1, x2]`). Solved via Cholesky normal equations with
/// one step of iterative refinement; at this crate's problem sizes that
/// matches statsmodels' pinv-based solve to near machine precision.
///
/// # Errors
///
/// [`HacError::EmptyDesign`] with no columns,
/// [`HacError::DimensionMismatch`] if a column's length differs from
/// `y.len()`, [`HacError::DegreesOfFreedom`] unless `n > k`,
/// [`HacError::NonFinite`] on NaN/inf input, and
/// [`HacError::SingularDesign`] when `X'X` is not numerically positive
/// definite (collinear columns).
pub fn ols(y: &[f64], x_cols: &[Vec<f64>]) -> Result<OlsFit, HacError> {
    let n = y.len();
    let k = x_cols.len();
    if k == 0 {
        return Err(HacError::EmptyDesign);
    }
    for (column, col) in x_cols.iter().enumerate() {
        if col.len() != n {
            return Err(HacError::DimensionMismatch {
                what: "OLS design column",
                column,
                expected: n,
                got: col.len(),
            });
        }
    }
    if n <= k {
        return Err(HacError::DegreesOfFreedom { n, k });
    }
    check_finite(y, "OLS response")?;
    for col in x_cols {
        check_finite(col, "OLS design column")?;
    }

    // Normal equations: xtx = X'X (row-major k x k), xty = X'y.
    let mut xtx = vec![0.0_f64; k * k];
    let mut xty = vec![0.0_f64; k];
    for i in 0..k {
        for j in 0..=i {
            let mut acc = 0.0;
            for (&xi, &xj) in x_cols[i].iter().zip(x_cols[j].iter()) {
                acc += xi * xj;
            }
            xtx[i * k + j] = acc;
            xtx[j * k + i] = acc;
        }
        let mut acc = 0.0;
        for (&xi, &yt) in x_cols[i].iter().zip(y.iter()) {
            acc += xi * yt;
        }
        xty[i] = acc;
    }

    let chol = cholesky(&xtx, k).ok_or(HacError::SingularDesign {
        what: "OLS normal equations",
    })?;
    let params = solve_refined(&xtx, &chol, k, &xty);

    // (X'X)^{-1} column by column, each with one refinement step.
    let mut xtx_inv = vec![0.0_f64; k * k];
    let mut unit = vec![0.0_f64; k];
    for j in 0..k {
        unit[j] = 1.0;
        let col = solve_refined(&xtx, &chol, k, &unit);
        for i in 0..k {
            xtx_inv[i * k + j] = col[i];
        }
        unit[j] = 0.0;
    }

    let residuals: Vec<f64> = (0..n)
        .map(|t| {
            let mut fit = 0.0;
            for (b, col) in params.iter().zip(x_cols.iter()) {
                fit += b * col[t];
            }
            y[t] - fit
        })
        .collect();

    Ok(OlsFit {
        params,
        residuals,
        nobs: n,
        nparams: k,
        xtx_inv,
        x_cols: x_cols.to_vec(),
    })
}

impl OlsFit {
    /// Standard errors, t-statistics, and parameter covariance under the
    /// requested [`SeType`] (see the module docs for the exact formulas
    /// and references).
    ///
    /// # Errors
    ///
    /// [`HacError::InvalidBandwidth`] for a negative/non-finite HAC
    /// bandwidth; [`HacError::NumericalBreakdown`] if a covariance
    /// diagonal comes out negative, which can only happen with a
    /// non-positive-semi-definite kernel ([`Kernel::Truncated`]).
    pub fn inference(&self, se_type: SeType) -> Result<OlsInference, HacError> {
        let n = self.nobs;
        let k = self.nparams;
        let nf = n as f64;
        let dof_scale = nf / (n - k) as f64;

        let cov = match se_type {
            SeType::NonRobust => {
                let rss: f64 = self.residuals.iter().map(|u| u * u).sum();
                let sigma2 = rss / (n - k) as f64;
                self.xtx_inv.iter().map(|v| sigma2 * v).collect()
            }
            SeType::Hc0 => self.sandwich(&self.hc_meat()),
            SeType::Hc1 => {
                let mut cov = self.sandwich(&self.hc_meat());
                for v in &mut cov {
                    *v *= dof_scale;
                }
                cov
            }
            SeType::Hac {
                kernel,
                bandwidth,
                use_correction,
            } => {
                check_bandwidth(bandwidth)?;
                let mut cov = self.sandwich(&self.hac_meat(kernel, bandwidth));
                if use_correction {
                    for v in &mut cov {
                        *v *= dof_scale;
                    }
                }
                cov
            }
        };

        let mut bse = Vec::with_capacity(k);
        for i in 0..k {
            let v = cov[i * k + i];
            if v < 0.0 {
                return Err(HacError::NumericalBreakdown {
                    what: "sandwich covariance diagonal (non-PSD kernel?)",
                });
            }
            bse.push(v.sqrt());
        }
        let tvalues = self
            .params
            .iter()
            .zip(bse.iter())
            .map(|(p, s)| p / s)
            .collect();
        Ok(OlsInference { cov, bse, tvalues })
    }

    /// White meat `S = sum_t u_t^2 x_t x_t'` (un-normalized, as in
    /// statsmodels: the bread's `(X'X)^{-1}` factors absorb the scaling).
    fn hc_meat(&self) -> Vec<f64> {
        let k = self.nparams;
        let mut s = vec![0.0_f64; k * k];
        for (t, &u) in self.residuals.iter().enumerate() {
            let u2 = u * u;
            for i in 0..k {
                let xi = self.x_cols[i][t];
                for j in 0..=i {
                    s[i * k + j] += u2 * xi * self.x_cols[j][t];
                }
            }
        }
        for i in 0..k {
            for j in 0..i {
                s[j * k + i] = s[i * k + j];
            }
        }
        s
    }

    /// HAC meat `S = Gamma_0 + sum_{j>=1} w_j (Gamma_j + Gamma_j')` from
    /// scores `s_t = x_t u_t`, exactly statsmodels'
    /// `sandwich_covariance.S_hac_simple`.
    fn hac_meat(&self, kernel: Kernel, bandwidth: f64) -> Vec<f64> {
        let n = self.nobs;
        let k = self.nparams;
        // Scores, row-major n x k.
        let mut scores = vec![0.0_f64; n * k];
        for t in 0..n {
            let u = self.residuals[t];
            for i in 0..k {
                scores[t * k + i] = self.x_cols[i][t] * u;
            }
        }

        let mut s = vec![0.0_f64; k * k];
        for lag in 0..n {
            let w = kernel.weight(lag, bandwidth);
            if lag > 0 && w == 0.0 && kernel.truncates() {
                break;
            }
            // Gamma_lag[i][j] = sum_{t=lag}^{n-1} s_t[i] s_{t-lag}[j].
            for t in lag..n {
                let row_t = &scores[t * k..(t + 1) * k];
                let row_l = &scores[(t - lag) * k..(t - lag + 1) * k];
                for i in 0..k {
                    for j in 0..k {
                        let g = row_t[i] * row_l[j];
                        if lag == 0 {
                            s[i * k + j] += g;
                        } else {
                            // w * (Gamma + Gamma'): add to both (i,j) and (j,i).
                            s[i * k + j] += w * g;
                            s[j * k + i] += w * g;
                        }
                    }
                }
            }
        }
        s
    }

    /// Sandwich `(X'X)^{-1} S (X'X)^{-1}` for a symmetric `S`.
    fn sandwich(&self, meat: &[f64]) -> Vec<f64> {
        let k = self.nparams;
        let bread = &self.xtx_inv;
        // tmp = meat * bread.
        let mut tmp = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for l in 0..k {
                    acc += meat[i * k + l] * bread[l * k + j];
                }
                tmp[i * k + j] = acc;
            }
        }
        // cov = bread * tmp.
        let mut cov = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for l in 0..k {
                    acc += bread[i * k + l] * tmp[l * k + j];
                }
                cov[i * k + j] = acc;
            }
        }
        cov
    }
}

/// Dense Cholesky factorization `S = L L'` of a symmetric positive-definite
/// `p x p` matrix in row-major storage. Returns the lower factor, or `None`
/// if a pivot is non-positive (or negligibly small relative to the original
/// diagonal), i.e. the matrix is not numerically positive definite.
fn cholesky(s: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut acc = s[i * p + j];
            for m in 0..j {
                acc -= l[i * p + m] * l[j * p + m];
            }
            if i == j {
                // Relative pivot tolerance: guards against collinear columns.
                let tol = s[i * p + i].abs() * 1e-13;
                if acc <= tol.max(f64::MIN_POSITIVE) {
                    return None;
                }
                l[i * p + i] = acc.sqrt();
            } else {
                l[i * p + j] = acc / l[j * p + j];
            }
        }
    }
    Some(l)
}

/// Solve `L L' x = b` given the lower Cholesky factor `L` (row-major).
fn chol_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    // Forward substitution: L z = b.
    let mut z = vec![0.0_f64; p];
    for i in 0..p {
        let mut acc = b[i];
        for m in 0..i {
            acc -= l[i * p + m] * z[m];
        }
        z[i] = acc / l[i * p + i];
    }
    // Back substitution: L' x = z.
    let mut x = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut acc = z[i];
        for m in (i + 1)..p {
            acc -= l[m * p + i] * x[m];
        }
        x[i] = acc / l[i * p + i];
    }
    x
}

/// Cholesky solve with one step of iterative refinement in working
/// precision — cheap, and recovers a couple of digits when the design
/// columns are correlated.
fn solve_refined(s: &[f64], chol: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut x = chol_solve(chol, p, b);
    let mut resid = b.to_vec();
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += s[i * p + j] * x[j];
        }
        resid[i] -= acc;
    }
    let correction = chol_solve(chol, p, &resid);
    for (xi, d) in x.iter_mut().zip(correction.iter()) {
        *xi += d;
    }
    x
}
