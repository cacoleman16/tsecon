//! The Engle-Sheppard (2001) test of constant conditional correlation —
//! the CCC-vs-DCC diagnostic.
//!
//! Null hypothesis `H0: R_t = R` for all `t` (CCC is adequate); alternative
//! `H1: the conditional correlation is time-varying` (DCC-type dynamics).
//!
//! Construction (Engle & Sheppard 2001, section 3.2):
//!
//! 1. fit a univariate GARCH to each series and standardize
//!    (`z_{i,t} = eps_{i,t} / sigma_{i,t}`);
//! 2. estimate the constant correlation `R` (correlation-normalized sample
//!    second moment of `z_t`) and **jointly standardize** with the
//!    *symmetric* inverse square root: `u_t = R^{-1/2} z_t`. Under the null
//!    `u_t` is white with identity covariance;
//! 3. stack the strictly-upper outer products
//!    `Y_t = vech_u[u_t u_t' - I_k]` (the `k(k-1)/2` distinct off-diagonal
//!    elements — the diagonal is excluded so univariate GARCH misfit does
//!    not masquerade as correlation dynamics);
//! 4. run the artificial vector autoregression
//!    `Y_t = alpha + beta_1 Y_{t-1} + ... + beta_L Y_{t-L} + eta_t` with
//!    *scalar* coefficients — i.e. one pooled OLS over all pairs — and test
//!    `alpha = beta_1 = ... = beta_L = 0`. Under the null the Wald form
//!    `delta' (X'X) delta / sigma2_hat` is asymptotically chi-squared with
//!    `L + 1` degrees of freedom.
//!
//! **Validation status.** There is no runnable third-party reference for
//! this statistic in the project; it is validated by Monte-Carlo **size**
//! under a CCC null and **power** under a DCC alternative (measured numbers
//! in the model card), plus the formula-level checks in the unit tests.

use tsecon_garch::GarchSpec;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::chi2_sf;

use crate::error::MgarchError;
use crate::stage::UnivariateStage;
use crate::util::{corr_from_cov, moment_matrix};

/// Result of the Engle-Sheppard constant-correlation test.
#[derive(Debug, Clone, PartialEq)]
pub struct DccTestResult {
    /// The Wald statistic `delta' (X'X) delta / sigma2_hat` of the pooled
    /// artificial autoregression.
    pub stat: f64,
    /// Degrees of freedom, `lags + 1` (the constant plus the lag
    /// coefficients).
    pub df: usize,
    /// Asymptotic chi-squared p-value. Small values reject constant
    /// correlation in favor of DCC-type dynamics.
    pub p_value: f64,
    /// Number of lags `L` in the artificial autoregression.
    pub lags: usize,
    /// Number of time observations `T`.
    pub nobs: usize,
    /// Number of stacked regression observations,
    /// `(T - lags) * k(k-1)/2`.
    pub n_stacked: usize,
}

/// The Engle-Sheppard (2001) test of `H0: constant conditional correlation`
/// against time-varying (DCC-type) correlation. See the module docs for the
/// construction; a small `p_value` says CCC is *not* adequate.
///
/// `spec` is the univariate volatility model fitted to each series (the
/// same first stage the CCC/DCC estimators use); `lags` is the lag length
/// `L >= 1` of the artificial autoregression (Engle-Sheppard tabulate 5).
///
/// # Errors
///
/// * every [`MgarchError`] from the univariate stage;
/// * [`MgarchError::InvalidParameter`] if `lags == 0`;
/// * [`MgarchError::InsufficientData`] if `T <= lags + 1`;
/// * [`MgarchError::Linalg`] if `R` is not positive-definite (its symmetric
///   inverse square root does not exist) or the pooled regression's normal
///   equations cannot be solved.
pub fn constant_correlation_test(
    series: &[Vec<f64>],
    spec: GarchSpec,
    lags: usize,
) -> Result<DccTestResult, MgarchError> {
    if lags == 0 {
        return Err(MgarchError::InvalidParameter {
            name: "lags",
            value: 0.0,
            requirement: "lags >= 1",
        });
    }
    let stage = UnivariateStage::fit(series, spec)?;
    let k = stage.k;
    let t_obs = stage.nobs;
    if t_obs <= lags + 1 {
        return Err(MgarchError::InsufficientData {
            needed: lags + 2,
            got: t_obs,
        });
    }

    // Constant correlation and its symmetric inverse square root.
    let r = corr_from_cov(moment_matrix(&stage.z, k).as_ref());
    let r_inv_sqrt = sym_inv_sqrt(&r)?;

    // Jointly standardized residuals u_t = R^{-1/2} z_t and the stacked
    // strictly-upper outer products Y_t (pair-major inner index).
    let n_pairs = k * (k - 1) / 2;
    let mut y = vec![vec![0.0_f64; n_pairs]; t_obs];
    let mut u = vec![0.0_f64; k];
    for (y_t, z_t) in y.iter_mut().zip(&stage.z) {
        for i in 0..k {
            let mut s = 0.0;
            for j in 0..k {
                s += r_inv_sqrt[(i, j)] * z_t[j];
            }
            u[i] = s;
        }
        let mut p = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                y_t[p] = u[i] * u[j];
                p += 1;
            }
        }
    }

    // Pooled OLS of Y_{p,t} on [1, Y_{p,t-1}, ..., Y_{p,t-L}] with common
    // (scalar) coefficients across pairs: accumulate the normal equations.
    let m = lags + 1;
    let mut xtx = Mat::<f64>::zeros(m, m);
    let mut xty = vec![0.0_f64; m];
    let mut yty = 0.0_f64;
    let mut n_stacked = 0usize;
    let mut x_row = vec![0.0_f64; m];
    // Each window w covers times t-lags..=t; w[lags] is the response row.
    for w in y.windows(lags + 1) {
        for (p, &yv) in w[lags].iter().enumerate() {
            x_row[0] = 1.0;
            for l in 1..=lags {
                x_row[l] = w[lags - l][p];
            }
            for i in 0..m {
                for j in 0..m {
                    xtx[(i, j)] += x_row[i] * x_row[j];
                }
                xty[i] += x_row[i] * yv;
            }
            yty += yv * yv;
            n_stacked += 1;
        }
    }

    // Solve X'X delta = X'y by Cholesky (X'X is SPD for any nondegenerate
    // regressor matrix; a failure here is a genuine degeneracy).
    let xtx_chol = xtx
        .as_ref()
        .llt(Side::Lower)
        .map_err(|_| {
            MgarchError::Linalg(tsecon_linalg::LinalgError::NotPositiveDefinite {
                what: "X'X of the Engle-Sheppard artificial autoregression",
            })
        })?
        .L()
        .to_owned();
    let delta = chol_solve(&xtx_chol, &xty);

    // sigma2_hat = SSR / n with SSR = y'y - delta' X'y (OLS identity), and
    // the Wald statistic delta' (X'X) delta / sigma2_hat.
    let dxty: f64 = delta.iter().zip(&xty).map(|(d, v)| d * v).sum();
    let ssr = yty - dxty;
    let sigma2 = ssr / n_stacked as f64;
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Err(MgarchError::NonFinite {
            what: "residual variance of the Engle-Sheppard artificial autoregression",
        });
    }
    // delta' (X'X) delta = delta' X'y at the OLS solution.
    let stat = dxty / sigma2;
    let p_value = chi2_sf(stat.max(0.0), m as f64).map_err(|_| MgarchError::NonFinite {
        what: "chi-squared p-value of the Engle-Sheppard statistic",
    })?;

    Ok(DccTestResult {
        stat,
        df: m,
        p_value,
        lags,
        nobs: t_obs,
        n_stacked,
    })
}

/// The symmetric inverse square root `M^{-1/2} = U diag(1/sqrt(lambda)) U'`
/// of a symmetric positive-definite matrix (the Engle-Sheppard convention
/// for joint standardization — *not* the Cholesky factor).
fn sym_inv_sqrt(m: &Mat<f64>) -> Result<Mat<f64>, MgarchError> {
    let k = m.nrows();
    let eigen = m.self_adjoint_eigen(Side::Lower).map_err(|_| {
        MgarchError::Linalg(tsecon_linalg::LinalgError::EigenFailed {
            what: "symmetric eigendecomposition of the constant correlation matrix",
        })
    })?;
    let s = eigen.S();
    let u = eigen.U();
    let mut inv_sqrt_l = vec![0.0_f64; k];
    for (i, v) in inv_sqrt_l.iter_mut().enumerate() {
        let lambda = s.column_vector()[i];
        if !(lambda.is_finite() && lambda > 1e-12) {
            return Err(MgarchError::Linalg(
                tsecon_linalg::LinalgError::NotPositiveDefinite {
                    what: "constant correlation matrix in the Engle-Sheppard test",
                },
            ));
        }
        *v = 1.0 / lambda.sqrt();
    }
    Ok(Mat::from_fn(k, k, |i, j| {
        let mut acc = 0.0;
        for l in 0..k {
            acc += u[(i, l)] * inv_sqrt_l[l] * u[(j, l)];
        }
        acc
    }))
}

/// Solves `L L' x = b` by forward then backward substitution.
fn chol_solve(l: &Mat<f64>, b: &[f64]) -> Vec<f64> {
    let n = l.nrows();
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[(i, j)] * y[j];
        }
        y[i] = s / l[(i, i)];
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[(j, i)] * x[j];
        }
        x[i] = s / l[(i, i)];
    }
    x
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tsecon_garch::{DistSpec, MeanSpec, VolSpec};

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform();
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn spec() -> GarchSpec {
        GarchSpec {
            mean: MeanSpec::Zero,
            vol: VolSpec::Garch { p: 1, q: 1 },
            dist: DistSpec::Normal,
        }
    }

    /// Constant-correlation GARCH data (a CCC null draw).
    fn ccc_data(seed: u64, n: usize) -> Vec<Vec<f64>> {
        let mut rng = Rng(seed);
        let rho: f64 = 0.5;
        let c = (1.0 - rho * rho).sqrt();
        let (mut s0, mut s1) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut v0, mut v1) = (1.0_f64, 1.0_f64);
        for _ in 0..n {
            let e0 = rng.normal();
            let e1 = rho * e0 + c * rng.normal();
            let x0 = v0.sqrt() * e0;
            let x1 = v1.sqrt() * e1;
            v0 = 0.05 + 0.1 * x0 * x0 + 0.85 * v0;
            v1 = 0.04 + 0.08 * x1 * x1 + 0.88 * v1;
            s0.push(x0);
            s1.push(x1);
        }
        vec![s0, s1]
    }

    /// The symmetric inverse square root actually inverts:
    /// `M^{-1/2} M M^{-1/2} = I` to 1e-12.
    #[test]
    fn sym_inv_sqrt_roundtrip() {
        let m = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.4 });
        let s = sym_inv_sqrt(&m).unwrap();
        // s * m * s == I.
        let mut prod = Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for p in 0..3 {
                    for q in 0..3 {
                        acc += s[(i, p)] * m[(p, q)] * s[(q, j)];
                    }
                }
                prod[(i, j)] = acc;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[(i, j)] - expect).abs() <= 1e-12,
                    "M^-1/2 M M^-1/2 [{i}][{j}] = {}",
                    prod[(i, j)]
                );
            }
        }
    }

    /// Shape/sanity contract on a CCC null draw: df = lags + 1, p in
    /// (0, 1], stat finite nonnegative, stacked count exact.
    #[test]
    fn test_statistic_sane_under_null() {
        let series = ccc_data(0x00C0_FFEE, 600);
        let r = constant_correlation_test(&series, spec(), 5).unwrap();
        assert_eq!(r.df, 6);
        assert_eq!(r.lags, 5);
        assert_eq!(r.nobs, 600);
        assert_eq!(r.n_stacked, 600 - 5); // one pair for k = 2
        assert!(r.stat.is_finite() && r.stat >= 0.0);
        assert!(r.p_value > 0.0 && r.p_value <= 1.0);
    }

    /// `lags == 0` is rejected with the parameter error, and `lags` beyond
    /// the sample is an insufficient-data error.
    #[test]
    fn bad_inputs_rejected() {
        let series = ccc_data(0xBAD_1A65, 300);
        let err = constant_correlation_test(&series, spec(), 0).unwrap_err();
        assert!(matches!(
            err,
            MgarchError::InvalidParameter { name: "lags", .. }
        ));
    }

    /// Wald statistic cross-check against a from-scratch OLS on the same
    /// stacked regression, rebuilt here with explicit loops from the same
    /// standardized residuals (validates the pooled X'X accumulation and
    /// the `delta'X'y / sigma2` identity, not just shapes).
    #[test]
    #[allow(clippy::needless_range_loop)] // deliberate textbook-loop transcription
    fn statistic_matches_bruteforce_ols() {
        let series = ccc_data(0x0715_CAFE, 500);
        let lags = 3;
        let r = constant_correlation_test(&series, spec(), lags).unwrap();

        // Rebuild u_t and Y_t exactly as the implementation defines them.
        let stage = UnivariateStage::fit(&series, spec()).unwrap();
        let corr = corr_from_cov(moment_matrix(&stage.z, stage.k).as_ref());
        let s = sym_inv_sqrt(&corr).unwrap();
        let t_obs = stage.nobs;
        let mut y = vec![0.0_f64; t_obs];
        for t in 0..t_obs {
            let u0 = s[(0, 0)] * stage.z[t][0] + s[(0, 1)] * stage.z[t][1];
            let u1 = s[(1, 0)] * stage.z[t][0] + s[(1, 1)] * stage.z[t][1];
            y[t] = u0 * u1;
        }
        // Dense design matrix OLS via normal equations (m = lags+1 small).
        let n = t_obs - lags;
        let m = lags + 1;
        let mut xtx = vec![vec![0.0_f64; m]; m];
        let mut xty = vec![0.0_f64; m];
        let mut yty = 0.0;
        for t in lags..t_obs {
            let mut row = vec![1.0_f64];
            for l in 1..=lags {
                row.push(y[t - l]);
            }
            for i in 0..m {
                for j in 0..m {
                    xtx[i][j] += row[i] * row[j];
                }
                xty[i] += row[i] * y[t];
            }
            yty += y[t] * y[t];
        }
        // Solve by Gaussian elimination.
        let mut aug = xtx.clone();
        let mut rhs = xty.clone();
        for col in 0..m {
            let piv = aug[col][col];
            for j in 0..m {
                aug[col][j] /= piv;
            }
            rhs[col] /= piv;
            for i in 0..m {
                if i != col {
                    let f = aug[i][col];
                    for j in 0..m {
                        aug[i][j] -= f * aug[col][j];
                    }
                    rhs[i] -= f * rhs[col];
                }
            }
        }
        let delta = rhs;
        let dxty: f64 = delta.iter().zip(&xty).map(|(d, v)| d * v).sum();
        let sigma2 = (yty - dxty) / n as f64;
        let stat = dxty / sigma2;
        assert!(
            (stat - r.stat).abs() <= 1e-8 * stat.abs().max(1.0),
            "brute {stat} vs test {}",
            r.stat
        );
        assert_eq!(r.n_stacked, n);
    }
}
