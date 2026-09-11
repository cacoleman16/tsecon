//! Residual diagnostics of a fitted VAR: the multivariate Portmanteau
//! statistics (Hosking 1980; Lütkepohl 2005, section 4.4.3), the
//! multivariate Jarque-Bera normality test with its skewness and kurtosis
//! components (Lütkepohl 2005, section 4.5; Kilian-Demiroglu 2000), and the
//! companion-form stability roots. Every convention follows statsmodels'
//! `VARResults.test_whiteness` / `test_normality` / `roots` / `is_stable`,
//! which the golden fixture `fixtures/var_diag.json` arbitrates.
//!
//! **Portmanteau.** With `C_i = T^{-1} Σ_{t=i+1}^{T} ũ_t ũ_{t-i}'` the
//! residual autocovariances of the *column-centred* residuals `ũ`,
//!
//! ```text
//! Q_h  = T   Σ_{i=1}^{h} tr(C_i' C_0^{-1} C_i C_0^{-1})              (unadjusted)
//! Q̄_h  = T²  Σ_{i=1}^{h} tr(C_i' C_0^{-1} C_i C_0^{-1}) / (T - i)    (adjusted)
//! ```
//!
//! both `χ²(k² (h - p))` under the null of white residuals (`h > p`). The
//! adjusted statistic is the small-sample correction that keeps the size
//! closer to nominal when `h` is a sizeable fraction of `T`.
//!
//! **Normality.** Centre the residuals, take `Σ̃ = ũ'ũ / T`, orthogonalise with
//! the lower Cholesky factor `P` of `Σ̃` (`w_t = P^{-1} ũ_t`), and form the
//! per-component skewness `b1_i = T^{-1} Σ_t w_{it}^3` and excess kurtosis
//! `b2_i = T^{-1} Σ_t w_{it}^4 - 3`; then
//!
//! ```text
//! λ_s = T b1'b1 / 6 ~ χ²(k),   λ_k = T b2'b2 / 24 ~ χ²(k),   λ = λ_s + λ_k ~ χ²(2k).
//! ```
//!
//! The Cholesky orthogonalisation is the statsmodels convention (it is also
//! Lütkepohl's; JMulTi's Doornik-Hansen variant uses the symmetric square
//! root and different transforms, and is *not* provided here — no runnable
//! reference for it exists in the fixture environment). The Cholesky choice
//! makes `λ_s` and `λ_k` depend on the column ordering; the ordering is the
//! user's, as with orthogonalised impulse responses.
//!
//! **Stability.** [`VarResults::roots_moduli`] (the statsmodels `roots`
//! moduli, reciprocal characteristic roots, descending) and
//! [`VarResults::is_stable`] (every companion eigenvalue strictly inside the
//! unit circle) are re-exposed together with the eigenvalue moduli
//! themselves.

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, Side};
use tsecon_stats::chi2_sf;

use crate::error::VarError;
use crate::results::{chol_lower, VarResults};

/// Multivariate Portmanteau test of residual autocorrelation up to `nlags`.
#[derive(Debug, Clone, PartialEq)]
pub struct PortmanteauTest {
    /// Unadjusted statistic `Q_h` (statsmodels `test_whiteness(adjusted=False)`).
    pub statistic: f64,
    /// Small-sample adjusted statistic `Q̄_h` (`adjusted=True`).
    pub adjusted: f64,
    /// Degrees of freedom `k² (nlags - p)`.
    pub df: usize,
    /// `P(χ²(df) > statistic)`.
    pub pvalue: f64,
    /// `P(χ²(df) > adjusted)`.
    pub adjusted_pvalue: f64,
    /// The lag count `h` the test was run at.
    pub nlags: usize,
}

/// Multivariate Jarque-Bera normality test with its two components.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalityTest {
    /// Omnibus statistic `λ = λ_s + λ_k` (statsmodels `test_normality`).
    pub statistic: f64,
    /// `P(χ²(2k) > statistic)`.
    pub pvalue: f64,
    /// Degrees of freedom of the omnibus test, `2k`.
    pub df: usize,
    /// Skewness statistic `λ_s = T b1'b1 / 6`.
    pub skewness: f64,
    /// `P(χ²(k) > skewness)`.
    pub skewness_pvalue: f64,
    /// Kurtosis statistic `λ_k = T b2'b2 / 24`.
    pub kurtosis: f64,
    /// `P(χ²(k) > kurtosis)`.
    pub kurtosis_pvalue: f64,
    /// Per-component third moments `b1` of the Cholesky-orthogonalised
    /// residuals (length `k`).
    pub skewness_components: Vec<f64>,
    /// Per-component excess fourth moments `b2` (length `k`).
    pub kurtosis_components: Vec<f64>,
}

/// The residual-diagnostics bundle of [`VarResults::diagnostics`].
#[derive(Debug, Clone, PartialEq)]
pub struct VarDiagnostics {
    /// Portmanteau test at the requested `nlags`.
    pub portmanteau: PortmanteauTest,
    /// Multivariate Jarque-Bera test.
    pub normality: NormalityTest,
    /// Moduli of the reciprocal characteristic roots, descending
    /// ([`VarResults::roots_moduli`]); stable iff the last one exceeds 1.
    pub roots: Vec<f64>,
    /// Moduli of the companion eigenvalues, descending (the first is the
    /// spectral radius); stable iff the first is below 1.
    pub eigenvalue_moduli: Vec<f64>,
    /// [`VarResults::is_stable`].
    pub is_stable: bool,
    /// Effective sample size `T` the residual tests use.
    pub nobs: usize,
    /// Number of series `k`.
    pub neqs: usize,
    /// Lag order `p`.
    pub lags: usize,
}

impl VarResults {
    /// Column-centred residuals `ũ = U - mean(U)` (`T × k`).
    fn centred_resid(&self) -> Mat<f64> {
        let (t, k) = (self.resid.nrows(), self.resid.ncols());
        let mut mean = vec![0.0; k];
        for j in 0..k {
            for i in 0..t {
                mean[j] += self.resid[(i, j)];
            }
            mean[j] /= t as f64;
        }
        Mat::from_fn(t, k, |i, j| self.resid[(i, j)] - mean[j])
    }

    /// Multivariate Portmanteau test of residual autocorrelation up to lag
    /// `nlags` — both the unadjusted and the small-sample adjusted
    /// statistics (module docs for the formulas; statsmodels
    /// `test_whiteness(nlags, adjusted=False/True)`).
    ///
    /// # Errors
    ///
    /// * [`VarError::InvalidParameter`] naming `nlags` if `nlags <= p` (the
    ///   `χ²` has `k² (nlags - p)` degrees of freedom, which must be
    ///   positive) or `nlags >= T`;
    /// * [`VarError::NotPositiveDefinite`] if the lag-0 residual covariance
    ///   is singular.
    pub fn portmanteau_test(&self, nlags: usize) -> Result<PortmanteauTest, VarError> {
        let p = self.spec.lags;
        let t = self.nobs;
        let k = self.neqs;
        if nlags <= p {
            return Err(VarError::InvalidParameter {
                name: "nlags",
                value: nlags as f64,
                requirement: "strictly more lags than the VAR order p: the Portmanteau \
                              statistic is chi-squared with k^2 (nlags - p) degrees of \
                              freedom, which must be positive (statsmodels test_whiteness \
                              refuses the same); pass nlags > lags",
            });
        }
        if nlags >= t {
            return Err(VarError::InvalidParameter {
                name: "nlags",
                value: nlags as f64,
                requirement: "fewer lags than the effective sample size T = nobs: the lag-h \
                              residual autocovariance needs T - h > 0 overlapping pairs, \
                              and the adjusted statistic divides by T - h",
            });
        }
        let u = self.centred_resid();
        let tf = t as f64;
        let acov = |lag: usize| -> Mat<f64> {
            Mat::from_fn(k, k, |a, b| {
                let mut acc = 0.0;
                for s in lag..t {
                    acc += u[(s, a)] * u[(s - lag, b)];
                }
                acc / tf
            })
        };
        let c0 = acov(0);
        let c0_inv = c0
            .llt(Side::Lower)
            .map_err(|_| VarError::NotPositiveDefinite {
                what: "C_0, the lag-0 residual covariance of the Portmanteau test",
            })?
            .inverse();
        let mut statistic = 0.0;
        let mut adjusted = 0.0;
        for i in 1..=nlags {
            let ci = acov(i);
            // tr(C_i' C_0^{-1} C_i C_0^{-1}) = sum over the entries of
            // (C_0^{-1} C_i C_0^{-1}) .* C_i.
            let mid = &c0_inv * &ci * &c0_inv;
            let mut term = 0.0;
            for a in 0..k {
                for b in 0..k {
                    term += ci[(a, b)] * mid[(a, b)];
                }
            }
            statistic += term;
            adjusted += term / (t - i) as f64;
        }
        statistic *= tf;
        adjusted *= tf * tf;
        let df = k * k * (nlags - p);
        Ok(PortmanteauTest {
            statistic,
            adjusted,
            df,
            pvalue: chi2_sf(statistic, df as f64)?,
            adjusted_pvalue: chi2_sf(adjusted, df as f64)?,
            nlags,
        })
    }

    /// Multivariate Jarque-Bera normality test of the residuals, with the
    /// Cholesky orthogonalisation statsmodels `test_normality` uses (module
    /// docs), plus its skewness and kurtosis components.
    ///
    /// # Errors
    ///
    /// [`VarError::NotPositiveDefinite`] if the residual covariance is
    /// singular (impossible for a successful fit).
    pub fn normality_test(&self) -> Result<NormalityTest, VarError> {
        let t = self.nobs;
        let k = self.neqs;
        let tf = t as f64;
        let u = self.centred_resid();
        let sig = Mat::from_fn(k, k, |a, b| {
            let mut acc = 0.0;
            for s in 0..t {
                acc += u[(s, a)] * u[(s, b)];
            }
            acc / tf
        });
        let p_chol = chol_lower(
            sig.as_ref(),
            "the centred residual covariance of the normality test",
        )?;
        let mut b1 = vec![0.0; k];
        let mut b2 = vec![0.0; k];
        let mut w = vec![0.0; k];
        for s in 0..t {
            // Forward substitution: P w = u_s.
            for i in 0..k {
                let mut acc = u[(s, i)];
                for l in 0..i {
                    acc -= p_chol[(i, l)] * w[l];
                }
                w[i] = acc / p_chol[(i, i)];
            }
            for i in 0..k {
                let w2 = w[i] * w[i];
                b1[i] += w2 * w[i];
                b2[i] += w2 * w2;
            }
        }
        for i in 0..k {
            b1[i] /= tf;
            b2[i] = b2[i] / tf - 3.0;
        }
        let skewness = tf * b1.iter().map(|x| x * x).sum::<f64>() / 6.0;
        let kurtosis = tf * b2.iter().map(|x| x * x).sum::<f64>() / 24.0;
        let statistic = skewness + kurtosis;
        Ok(NormalityTest {
            statistic,
            pvalue: chi2_sf(statistic, 2.0 * k as f64)?,
            df: 2 * k,
            skewness,
            skewness_pvalue: chi2_sf(skewness, k as f64)?,
            kurtosis,
            kurtosis_pvalue: chi2_sf(kurtosis, k as f64)?,
            skewness_components: b1,
            kurtosis_components: b2,
        })
    }

    /// The residual-diagnostics bundle: [`VarResults::portmanteau_test`] at
    /// `nlags`, [`VarResults::normality_test`], and the stability roots.
    ///
    /// # Errors
    ///
    /// Anything the three components can return.
    pub fn diagnostics(&self, nlags: usize) -> Result<VarDiagnostics, VarError> {
        let portmanteau = self.portmanteau_test(nlags)?;
        let normality = self.normality_test()?;
        let roots = self.roots_moduli()?;
        let mut eigenvalue_moduli: Vec<f64> = roots
            .iter()
            .map(|r| if r.is_infinite() { 0.0 } else { r.recip() })
            .collect();
        eigenvalue_moduli.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        let is_stable = self.is_stable()?;
        Ok(VarDiagnostics {
            portmanteau,
            normality,
            roots,
            eigenvalue_moduli,
            is_stable,
            nobs: self.nobs,
            neqs: self.neqs,
            lags: self.spec.lags,
        })
    }
}
