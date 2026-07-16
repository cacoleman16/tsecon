//! Private ordinary-least-squares helper for the regression-based
//! diagnostics ([`crate::pacf_ols`], [`crate::arch_lm`]).
//!
//! Solves the intercept-augmented regression `y = a + X b + e` by centering
//! the columns (which partials out the intercept exactly and drastically
//! improves the conditioning of the normal equations) and solving the
//! centered normal equations `(Xc'Xc) b = Xc'y_c` with a Cholesky
//! factorization plus one step of iterative refinement. At the problem
//! sizes of this crate (a handful of lag columns) this matches a QR/SVD
//! solve to near machine precision.
//!
//! // TODO(phase0): delegate to shared linalg (tsecon-linalg QR) later.

use crate::error::DiagError;

/// Result of an intercept-augmented OLS fit.
pub(crate) struct OlsFit {
    /// Slope coefficients, in the order the columns were supplied.
    pub(crate) slopes: Vec<f64>,
    /// Centered coefficient of determination `1 - RSS/TSS`.
    pub(crate) r_squared: f64,
}

/// Fit `y = a + X b + e` by OLS, where `cols` are the columns of `X`.
///
/// All columns must have the same length as `y`. Requires
/// `n >= p + 2` observations so at least one residual degree of freedom
/// remains.
pub(crate) fn ols_with_intercept(
    cols: &[Vec<f64>],
    y: &[f64],
    what: &'static str,
) -> Result<OlsFit, DiagError> {
    let n = y.len();
    let p = cols.len();
    if n < p + 2 {
        return Err(DiagError::SeriesTooShort {
            what,
            n,
            needed: p + 2,
        });
    }
    // Compile-time-impossible for the callers in this crate, which always
    // build equal-length columns; keep a debug check only.
    debug_assert!(cols.iter().all(|c| c.len() == n));

    let nf = n as f64;
    let y_mean = y.iter().sum::<f64>() / nf;
    let x_means: Vec<f64> = cols.iter().map(|c| c.iter().sum::<f64>() / nf).collect();

    // Centered cross products: s = Xc'Xc (dense symmetric, row-major p x p),
    // rhs = Xc'y_c.
    let mut s = vec![0.0_f64; p * p];
    let mut rhs = vec![0.0_f64; p];
    for i in 0..p {
        for j in 0..=i {
            let mut acc = 0.0;
            for (&xi, &xj) in cols[i].iter().zip(cols[j].iter()) {
                acc += (xi - x_means[i]) * (xj - x_means[j]);
            }
            s[i * p + j] = acc;
            s[j * p + i] = acc;
        }
        let mut acc = 0.0;
        for (&xi, &yt) in cols[i].iter().zip(y.iter()) {
            acc += (xi - x_means[i]) * (yt - y_mean);
        }
        rhs[i] = acc;
    }

    let chol = cholesky(&s, p).ok_or(DiagError::SingularDesign { what })?;
    let mut beta = chol_solve(&chol, p, &rhs);

    // One step of iterative refinement in working precision: cheap and
    // recovers a couple of digits when the lag columns are correlated.
    let mut resid = rhs.clone();
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += s[i * p + j] * beta[j];
        }
        resid[i] -= acc;
    }
    let correction = chol_solve(&chol, p, &resid);
    for (b, d) in beta.iter_mut().zip(correction.iter()) {
        *b += d;
    }

    // Centered R^2 = 1 - RSS/TSS with TSS = sum (y - ybar)^2.
    let mut rss = 0.0;
    let mut tss = 0.0;
    for t in 0..n {
        let yc = y[t] - y_mean;
        let mut fit = 0.0;
        for j in 0..p {
            fit += beta[j] * (cols[j][t] - x_means[j]);
        }
        rss += (yc - fit) * (yc - fit);
        tss += yc * yc;
    }
    if tss <= 0.0 {
        return Err(DiagError::ConstantSeries { what });
    }
    // OLS with an intercept guarantees 0 <= RSS <= TSS in exact arithmetic;
    // clamp away last-ulp rounding excursions only.
    let r_squared = (1.0 - rss / tss).clamp(0.0, 1.0);

    Ok(OlsFit {
        slopes: beta,
        r_squared,
    })
}

/// Result of a general OLS fit (no implicit intercept; deterministics are
/// passed as explicit columns) with classical nonrobust standard errors,
/// as needed by the unit-root regressions in [`crate::adf`].
pub(crate) struct OlsDetailed {
    /// Coefficient t-ratios `b_j / se(b_j)` with
    /// `se(b_j) = sqrt(s^2 [(X'X)^{-1}]_{jj})` and `s^2 = RSS / (n - k)`,
    /// in the order the columns were supplied.
    pub(crate) t_values: Vec<f64>,
    /// Residual sum of squares.
    pub(crate) ssr: f64,
    /// Number of observations.
    pub(crate) nobs: usize,
    /// Number of estimated coefficients (columns of the design).
    pub(crate) nparams: usize,
}

impl OlsDetailed {
    /// Gaussian log-likelihood at the OLS estimate, statsmodels convention:
    /// `llf = -n/2 [ln(2 pi) + ln(RSS/n) + 1]`.
    fn llf(&self) -> f64 {
        let n = self.nobs as f64;
        -0.5 * n * ((2.0 * core::f64::consts::PI).ln() + (self.ssr / n).ln() + 1.0)
    }

    /// Akaike information criterion, statsmodels OLS convention
    /// `AIC = -2 llf + 2 k` with `k` the number of estimated coefficients.
    pub(crate) fn aic(&self) -> f64 {
        -2.0 * self.llf() + 2.0 * self.nparams as f64
    }

    /// Bayesian information criterion, statsmodels OLS convention
    /// `BIC = -2 llf + ln(n) k`.
    pub(crate) fn bic(&self) -> f64 {
        -2.0 * self.llf() + (self.nobs as f64).ln() * self.nparams as f64
    }
}

/// Fit `y = X b + e` by OLS via a Householder QR factorization of `X`,
/// where `cols` are the columns of `X` (pass a column of ones explicitly
/// for an intercept). Returns t-ratios, RSS, and the fit dimensions.
///
/// QR is used instead of the centered normal equations above because the
/// ADF designs mix levels, differences, and deterministic trends whose
/// cross-product matrix can be poorly conditioned; Householder QR keeps
/// the error proportional to `cond(X)`, not `cond(X)^2` (Golub & Van Loan
/// 2013, ch. 5).
///
/// // TODO(phase0): delegate to shared linalg (tsecon-linalg QR) later.
pub(crate) fn ols_detailed(
    cols: &[Vec<f64>],
    y: &[f64],
    what: &'static str,
) -> Result<OlsDetailed, DiagError> {
    let n = y.len();
    let k = cols.len();
    debug_assert!(k >= 1, "ols_detailed requires at least one regressor");
    debug_assert!(cols.iter().all(|c| c.len() == n));
    if n < k + 1 {
        return Err(DiagError::SeriesTooShort {
            what,
            n,
            needed: k + 1,
        });
    }

    // Working copies: `a` is transformed in place into the Householder
    // vectors (on and below the diagonal) and the strict upper triangle of
    // R; `rdiag` holds the diagonal of R; `qty` accumulates Q'y.
    let mut a: Vec<Vec<f64>> = cols.to_vec();
    let mut qty: Vec<f64> = y.to_vec();
    let mut rdiag = vec![0.0_f64; k];

    for j in 0..k {
        // Householder reflector annihilating a[j][j+1..n]. The full column
        // norm is preserved by the earlier orthogonal transforms, so it
        // serves as the scale for the relative rank tolerance.
        let sub: f64 = a[j][j..].iter().map(|&v| v * v).sum();
        let head: f64 = a[j][..j].iter().map(|&v| v * v).sum();
        let norm = sub.sqrt();
        let tol = ((head + sub).sqrt() * 1e-13).max(f64::MIN_POSITIVE);
        if norm.is_nan() || norm <= tol {
            return Err(DiagError::SingularDesign { what });
        }
        // v = x + sign(x_j) ||x|| e_1 avoids cancellation in v_1.
        let alpha = if a[j][j] >= 0.0 { -norm } else { norm };
        a[j][j] -= alpha;
        rdiag[j] = alpha;
        let vtv: f64 = a[j][j..].iter().map(|&v| v * v).sum();

        // Apply H = I - 2 v v' / (v'v) to the remaining columns and to y.
        let (left, right) = a.split_at_mut(j + 1);
        let v = &left[j][j..];
        for col in right.iter_mut() {
            let dot: f64 = v.iter().zip(&col[j..]).map(|(&vi, &ci)| vi * ci).sum();
            let f = 2.0 * dot / vtv;
            for (vi, ci) in v.iter().zip(col[j..].iter_mut()) {
                *ci -= f * vi;
            }
        }
        let dot: f64 = v.iter().zip(&qty[j..]).map(|(&vi, &qi)| vi * qi).sum();
        let f = 2.0 * dot / vtv;
        for (vi, qi) in v.iter().zip(qty[j..].iter_mut()) {
            *qi -= f * vi;
        }
    }

    // Back substitution: R b = (Q'y)[0..k]. The strict upper triangle of R
    // lives at a[m][j] for m > j.
    let mut beta = vec![0.0_f64; k];
    for j in (0..k).rev() {
        let mut acc = qty[j];
        for (m, bm) in beta.iter().enumerate().skip(j + 1) {
            acc -= a[m][j] * bm;
        }
        beta[j] = acc / rdiag[j];
    }

    // RSS from explicit residuals against the original columns (matches the
    // statsmodels definition resid'resid).
    let mut ssr = 0.0;
    for i in 0..n {
        let mut fit = 0.0;
        for (bj, col) in beta.iter().zip(cols.iter()) {
            fit += bj * col[i];
        }
        let e = y[i] - fit;
        ssr += e * e;
    }

    let sigma2 = ssr / (n - k) as f64;
    if !(sigma2 > 0.0 && sigma2.is_finite()) {
        return Err(DiagError::NumericalBreakdown { what });
    }

    // diag[(X'X)^{-1}] = squared row norms of R^{-1}, accumulated column by
    // column: solve R x = e_c (support 0..=c) and add x_j^2 into slot j.
    let mut xtx_inv_diag = vec![0.0_f64; k];
    let mut x = vec![0.0_f64; k];
    for c in 0..k {
        x[c] = 1.0 / rdiag[c];
        for j in (0..c).rev() {
            let mut acc = 0.0;
            for (l, xl) in x.iter().enumerate().take(c + 1).skip(j + 1) {
                acc += a[l][j] * xl;
            }
            x[j] = -acc / rdiag[j];
        }
        for (dj, &xj) in xtx_inv_diag.iter_mut().zip(x.iter()).take(c + 1) {
            *dj += xj * xj;
        }
    }

    let t_values = beta
        .iter()
        .zip(&xtx_inv_diag)
        .map(|(&b, &d)| b / (sigma2 * d).sqrt())
        .collect();

    Ok(OlsDetailed {
        t_values,
        ssr,
        nobs: n,
        nparams: k,
    })
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
            for k in 0..j {
                acc -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                // Relative pivot tolerance: guards against collinear lags.
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
        for k in 0..i {
            acc -= l[i * p + k] * z[k];
        }
        z[i] = acc / l[i * p + i];
    }
    // Back substitution: L' x = z.
    let mut x = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut acc = z[i];
        for k in (i + 1)..p {
            acc -= l[k * p + i] * x[k];
        }
        x[i] = acc / l[i * p + i];
    }
    x
}
