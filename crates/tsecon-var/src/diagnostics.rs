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
//! **Normality — three statistics on one centred residual matrix.** Centre
//! the residuals (`ũ = U - 1 mean(U)`, a sequential column mean) and take the
//! ML covariance `S = ũ'ũ / T` (divisor `T`, no degrees-of-freedom
//! correction).
//!
//! 1. *Cholesky* (statsmodels `test_normality`; Lütkepohl 2005, section 4.5;
//!    JMulTi's "Lütkepohl (1993)" block): orthogonalise with the lower
//!    Cholesky factor `P` of `S` (`w_t = P^{-1} ũ_t`), form the per-component
//!    skewness `b1_i = T^{-1} Σ_t w_{it}^3` and excess kurtosis
//!    `b2_i = T^{-1} Σ_t w_{it}^4 - 3`; then
//!
//!    ```text
//!    λ_s = T b1'b1 / 6 ~ χ²(k),   λ_k = T b2'b2 / 24 ~ χ²(k),   λ = λ_s + λ_k ~ χ²(2k).
//!    ```
//!
//!    The Cholesky choice makes `λ_s` and `λ_k` depend on the column
//!    ordering; the ordering is the user's, as with orthogonalised impulse
//!    responses. [`normality_test_cholesky`], [`VarResults::normality_test`].
//! 2. *Symmetric root* (JMulTi's block labelled "Doornik & Hansen (1994)"):
//!    identical to (1) except `w_t = S^{-1/2} ũ_t` with the unique symmetric
//!    inverse square root `S^{-1/2} = H Λ^{-1/2} H'` of `S = H Λ H'` — plain
//!    `b1`, excess `b2`, NO further transforms. All 16 JMulTi outputs bundled
//!    with statsmodels 0.15 are reproduced to their printed precision
//!    (`fixtures/var_diag.json`, `jmulti`). Invariant to column order and
//!    sign flips but NOT to per-column rescaling: it depends on units.
//!    [`normality_test_symmetric`], [`VarResults::normality_test_symmetric`].
//! 3. *Doornik-Hansen (2008) omnibus*: with `V = diag(S)` and the correlation
//!    matrix `C = V^{-1/2} S V^{-1/2} = H Λ H'`, transform
//!    `r_t = H Λ^{-1/2} H' V^{-1/2} ũ_t`; per component take the plain
//!    moments `m_q = T^{-1} Σ_t r_{tj}^q`, `√b1 = m3 / m2^{3/2}`,
//!    `b2 = m4 / m2²`, and map them to approximately standard normal `z1`
//!    (D'Agostino 1970's skewness transform) and `z2` (the Wilson-Hilferty
//!    cube root of Shenton-Bowman's gamma approximation to the kurtosis given
//!    the skewness), with `n = T`:
//!
//!    ```text
//!    β  = 3(n²+27n−70)(n+1)(n+3) / ((n−2)(n+5)(n+7)(n+9))
//!    ω² = −1 + √(2(β−1)),   δ = 1/√(ln √ω²)
//!    y  = √b1 √((ω²−1)(n+1)(n+3) / (12(n−2))),   z1 = δ asinh(y)
//!    d  = (n−3)(n+1)(n²+15n−4)
//!    a  = (n−2)(n+5)(n+7)(n²+27n−70) / (6d)
//!    c  = (n−7)(n+5)(n+7)(n²+2n−5) / (6d)
//!    κ  = (n+5)(n+7)(n³+37n²+11n−313) / (12d)
//!    α  = a + c b1,   χ = 2κ(b2 − 1 − b1)
//!    z2 = (∛(χ/(2α)) − 1 + 1/(9α)) √(9α)
//!    E  = Σ z1² + Σ z2² ~ χ²(2k)   (each sum ~ χ²(k))
//!    ```
//!
//!    The skewness transform degenerates at `n = 7` (`β = 3`, `ω² = 1`), so
//!    the statistic needs `n >= 8` (scipy's `skewtest` refuses below 8 too).
//!    `E` is invariant to column permutation, positive per-column rescaling,
//!    sign flips (which flip the sign of `z1_j` exactly) and shifts. No
//!    independent implementation of the full statistic is installable here;
//!    the grade is a documented formula plus three legs: `z1` equals
//!    `scipy.stats.skewtest` on the transformed columns, the kurtosis
//!    coefficients `(a, c, κ)` satisfy the exact mean-matching identity
//!    `2κ(E b2 − 1 − E b1) = 2(a + c E b1)` (with `E b1 = 6(n−2)/((n+1)(n+3))`,
//!    `E b2 = 3(n−1)/(n+1)`, checked in rational arithmetic by the fixture
//!    generator and at 1e-13 here), and the common denominator `d` (which
//!    cancels from that identity) is guarded by the null Monte Carlo of `z2`.
//!    [`doornik_hansen_test`], [`VarResults::doornik_hansen_test`].
//!
//! JMulTi's "Doornik & Hansen (1994)" label therefore names statistic (2),
//! not the transformed test (3) of the published Doornik-Hansen (2008)
//! paper; the two are kept apart by name everywhere in this library.
//!
//! **Stability.** [`VarResults::roots_moduli`] (the statsmodels `roots`
//! moduli, reciprocal characteristic roots, descending) and
//! [`VarResults::is_stable`] (every companion eigenvalue strictly inside the
//! unit circle) are re-exposed together with the eigenvalue moduli
//! themselves.

use tsecon_linalg::faer::linalg::solvers::DenseSolveCore;
use tsecon_linalg::faer::{Mat, MatRef, Side};
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

/// Multivariate Jarque-Bera normality test with its two components — the
/// Cholesky-orthogonalised form ([`normality_test_cholesky`]) or the
/// symmetric-root form ([`normality_test_symmetric`]); the formulas differ
/// only in the orthogonalisation (module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct NormalityTest {
    /// Omnibus statistic `λ = λ_s + λ_k` (statsmodels `test_normality` for
    /// the Cholesky form, JMulTi's "Doornik & Hansen (1994)" joint statistic
    /// for the symmetric form).
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
    /// Per-component third moments `b1` of the orthogonalised residuals
    /// (length `k`).
    pub skewness_components: Vec<f64>,
    /// Per-component excess fourth moments `b2` (length `k`).
    pub kurtosis_components: Vec<f64>,
}

/// The Doornik-Hansen (2008) omnibus normality test (module docs, item 3).
#[derive(Debug, Clone, PartialEq)]
pub struct DoornikHansenTest {
    /// Omnibus statistic `E = z1'z1 + z2'z2`.
    pub statistic: f64,
    /// `P(χ²(2k) > statistic)`.
    pub pvalue: f64,
    /// Degrees of freedom of the omnibus test, `2k`.
    pub df: usize,
    /// Skewness part `z1'z1`.
    pub skewness: f64,
    /// `P(χ²(k) > skewness)`.
    pub skewness_pvalue: f64,
    /// Kurtosis part `z2'z2`.
    pub kurtosis: f64,
    /// `P(χ²(k) > kurtosis)`.
    pub kurtosis_pvalue: f64,
    /// Per-component transformed skewness `z1` (length `k`).
    pub z_skewness: Vec<f64>,
    /// Per-component transformed kurtosis `z2` (length `k`).
    pub z_kurtosis: Vec<f64>,
    /// Per-component sample skewness `√b1 = m3 / m2^{3/2}` of the
    /// transformed residuals (length `k`).
    pub sqrt_b1: Vec<f64>,
    /// Per-component sample kurtosis `b2 = m4 / m2²` (NOT excess; length
    /// `k`).
    pub b2: Vec<f64>,
}

/// The residual-diagnostics bundle of [`VarResults::diagnostics`].
#[derive(Debug, Clone, PartialEq)]
pub struct VarDiagnostics {
    /// Portmanteau test at the requested `nlags`.
    pub portmanteau: PortmanteauTest,
    /// Multivariate Jarque-Bera test, Cholesky orthogonalisation (statsmodels).
    pub normality: NormalityTest,
    /// Multivariate Jarque-Bera test, symmetric-root orthogonalisation
    /// (JMulTi's "Doornik & Hansen (1994)" block). `None` when the residual
    /// covariance is numerically singular for the symmetric root (smallest
    /// eigenvalue `<= 1e-12 x` its trace — typically series on wildly
    /// different scales); the bundle never refuses on its account.
    pub normality_symmetric: Option<NormalityTest>,
    /// Doornik-Hansen (2008) omnibus test. `None` when `nobs < 8` (the
    /// skewness transform degenerates at `n = 7`) or when the residual
    /// correlation matrix is numerically singular; the bundle never refuses
    /// on its account.
    pub doornik_hansen: Option<DoornikHansenTest>,
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

/// Relative eigenvalue floor of the symmetric roots: an eigenvalue at or
/// below `SYM_ROOT_FLOOR x trace` is treated as zero.
const SYM_ROOT_FLOOR: f64 = 1e-12;

/// Column-centred copy `m - 1 mean(m)` of a `T x k` matrix, with the column
/// mean accumulated sequentially in `t`.
fn centred(m: MatRef<'_, f64>) -> Mat<f64> {
    let (t, k) = (m.nrows(), m.ncols());
    let mut mean = vec![0.0; k];
    for (j, mj) in mean.iter_mut().enumerate() {
        for i in 0..t {
            *mj += m[(i, j)];
        }
        *mj /= t as f64;
    }
    Mat::from_fn(t, k, |i, j| m[(i, j)] - mean[j])
}

/// ML covariance `u'u / T` of a centred `T x k` matrix (sequential sums; the
/// result is exactly symmetric).
fn ml_cov(u: &Mat<f64>) -> Mat<f64> {
    let (t, k) = (u.nrows(), u.ncols());
    let tf = t as f64;
    Mat::from_fn(k, k, |a, b| {
        let mut acc = 0.0;
        for s in 0..t {
            acc += u[(s, a)] * u[(s, b)];
        }
        acc / tf
    })
}

/// Eigenvalues (ascending) and eigenvectors of a symmetric matrix.
fn sym_eigen(m: &Mat<f64>) -> Result<(Vec<f64>, Mat<f64>), VarError> {
    let eigen = m.self_adjoint_eigen(Side::Lower).map_err(|_| {
        VarError::Linalg(tsecon_linalg::LinalgError::EigenFailed {
            what: "symmetric eigendecomposition in the residual normality test",
        })
    })?;
    let lambda: Vec<f64> = eigen.S().column_vector().iter().copied().collect();
    Ok((lambda, eigen.U().to_owned()))
}

/// Whether every eigenvalue is finite and above `SYM_ROOT_FLOOR x trace`.
fn eigen_floor_ok(m: &Mat<f64>, lambda: &[f64]) -> bool {
    let trace: f64 = (0..m.nrows()).map(|i| m[(i, i)]).sum();
    trace.is_finite()
        && lambda
            .iter()
            .all(|&l| l.is_finite() && l > SYM_ROOT_FLOOR * trace)
}

/// The symmetric inverse square root `M^{-1/2} = H diag(λ^{-1/2}) H'` of a
/// symmetric positive-definite matrix (unique, so the eigenvector sign and
/// order do not matter) — the crates/tsecon-mgarch `estest.rs` pattern.
/// Refuses an eigenvalue at or below `1e-12 x trace`, or a non-finite one,
/// with [`VarError::NotPositiveDefinite`] naming `what`.
fn sym_inv_sqrt(m: &Mat<f64>, what: &'static str) -> Result<Mat<f64>, VarError> {
    let (lambda, h) = sym_eigen(m)?;
    if !eigen_floor_ok(m, &lambda) {
        return Err(VarError::NotPositiveDefinite { what });
    }
    let k = m.nrows();
    let inv_sqrt: Vec<f64> = lambda.iter().map(|l| 1.0 / l.sqrt()).collect();
    Ok(Mat::from_fn(k, k, |i, j| {
        let mut acc = 0.0;
        for l in 0..k {
            acc += h[(i, l)] * inv_sqrt[l] * h[(j, l)];
        }
        acc
    }))
}

/// The multivariate Jarque-Bera statistics from orthogonalised rows:
/// `fill(s, w)` writes the `k` orthogonalised components of row `s`.
fn jb_from_rows(
    t: usize,
    k: usize,
    mut fill: impl FnMut(usize, &mut [f64]),
) -> Result<NormalityTest, VarError> {
    let tf = t as f64;
    let mut b1 = vec![0.0; k];
    let mut b2 = vec![0.0; k];
    let mut w = vec![0.0; k];
    for s in 0..t {
        fill(s, &mut w);
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

/// Cholesky-orthogonalised multivariate Jarque-Bera of CENTRED residuals —
/// the body `VarResults::normality_test` has always had (same loops, same
/// summation order).
fn jb_cholesky(u: &Mat<f64>, what: &'static str) -> Result<NormalityTest, VarError> {
    let (t, k) = (u.nrows(), u.ncols());
    let sig = ml_cov(u);
    let p_chol = chol_lower(sig.as_ref(), what)?;
    jb_from_rows(t, k, |s, w| {
        // Forward substitution: P w = u_s.
        for i in 0..k {
            let mut acc = u[(s, i)];
            for l in 0..i {
                acc -= p_chol[(i, l)] * w[l];
            }
            w[i] = acc / p_chol[(i, i)];
        }
    })
}

/// Symmetric-root multivariate Jarque-Bera of CENTRED residuals.
fn jb_symmetric(u: &Mat<f64>, what: &'static str) -> Result<NormalityTest, VarError> {
    let (t, k) = (u.nrows(), u.ncols());
    let root = sym_inv_sqrt(&ml_cov(u), what)?;
    jb_from_rows(t, k, |s, w| {
        for (i, wi) in w.iter_mut().enumerate() {
            let mut acc = 0.0;
            for l in 0..k {
                acc += root[(i, l)] * u[(s, l)];
            }
            *wi = acc;
        }
    })
}

/// D'Agostino's `β` (the kurtosis of `√b1` under normality) at sample size
/// `n`; exactly 3 at `n = 7`, where the skewness transform degenerates.
fn dh_beta(n: f64) -> f64 {
    3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0))
}

/// The skewness-transform coefficients `(δ, scale)`: `z1 = δ asinh(scale √b1)`.
fn dh_skew_coeffs(n: f64) -> (f64, f64) {
    let beta = dh_beta(n);
    let omega2 = -1.0 + (2.0 * (beta - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * omega2.ln()).sqrt();
    let scale = ((omega2 - 1.0) * (n + 1.0) * (n + 3.0) / (12.0 * (n - 2.0))).sqrt();
    (delta, scale)
}

/// The kurtosis-transform coefficients `(a, c, κ)` (module docs).
fn dh_kurt_coeffs(n: f64) -> (f64, f64, f64) {
    let d = (n - 3.0) * (n + 1.0) * (n * n + 15.0 * n - 4.0);
    let a = (n - 2.0) * (n + 5.0) * (n + 7.0) * (n * n + 27.0 * n - 70.0) / (6.0 * d);
    let c = (n - 7.0) * (n + 5.0) * (n + 7.0) * (n * n + 2.0 * n - 5.0) / (6.0 * d);
    let kappa = (n + 5.0) * (n + 7.0) * (n * n * n + 37.0 * n * n + 11.0 * n - 313.0) / (12.0 * d);
    (a, c, kappa)
}

/// Doornik-Hansen (2008) of CENTRED residuals with `n >= 8` rows.
fn dh_centred(u: &Mat<f64>) -> Result<DoornikHansenTest, VarError> {
    let (t, k) = (u.nrows(), u.ncols());
    let s = ml_cov(u);
    let dinv: Vec<f64> = (0..k).map(|j| 1.0 / s[(j, j)].sqrt()).collect();
    let corr = Mat::from_fn(k, k, |a, b| s[(a, b)] * dinv[a] * dinv[b]);
    let root = sym_inv_sqrt(
        &corr,
        "the correlation matrix of resid, the residual matrix of the Doornik-Hansen test \
         (its smallest eigenvalue is <= 1e-12 x its trace)",
    )?;
    let mut m2 = vec![0.0; k];
    let mut m3 = vec![0.0; k];
    let mut m4 = vec![0.0; k];
    let mut y = vec![0.0; k];
    for row in 0..t {
        for (l, yl) in y.iter_mut().enumerate() {
            *yl = u[(row, l)] * dinv[l];
        }
        for i in 0..k {
            let mut r = 0.0;
            for (l, &yl) in y.iter().enumerate() {
                r += root[(i, l)] * yl;
            }
            let r2 = r * r;
            m2[i] += r2;
            m3[i] += r2 * r;
            m4[i] += r2 * r2;
        }
    }
    let n = t as f64;
    let (delta, scale) = dh_skew_coeffs(n);
    let (a, c, kappa) = dh_kurt_coeffs(n);
    let mut sqrt_b1 = Vec::with_capacity(k);
    let mut b2v = Vec::with_capacity(k);
    let mut z1 = Vec::with_capacity(k);
    let mut z2 = Vec::with_capacity(k);
    for i in 0..k {
        let (q2, q3, q4) = (m2[i] / n, m3[i] / n, m4[i] / n);
        let rb1 = q3 / (q2 * q2.sqrt());
        let b2 = q4 / (q2 * q2);
        let b1 = rb1 * rb1;
        z1.push(delta * (scale * rb1).asinh());
        let alpha = a + c * b1;
        let chi = 2.0 * kappa * (b2 - 1.0 - b1);
        z2.push(((chi / (2.0 * alpha)).cbrt() - 1.0 + 1.0 / (9.0 * alpha)) * (9.0 * alpha).sqrt());
        sqrt_b1.push(rb1);
        b2v.push(b2);
    }
    let skewness: f64 = z1.iter().map(|z| z * z).sum();
    let kurtosis: f64 = z2.iter().map(|z| z * z).sum();
    let statistic = skewness + kurtosis;
    if !statistic.is_finite() {
        return Err(VarError::NonFinite {
            what: "the Doornik-Hansen statistic (the transformed moments overflowed)",
            at: None,
        });
    }
    Ok(DoornikHansenTest {
        statistic,
        pvalue: chi2_sf(statistic, 2.0 * k as f64)?,
        df: 2 * k,
        skewness,
        skewness_pvalue: chi2_sf(skewness, k as f64)?,
        kurtosis,
        kurtosis_pvalue: chi2_sf(kurtosis, k as f64)?,
        z_skewness: z1,
        z_kurtosis: z2,
        sqrt_b1,
        b2: b2v,
    })
}

/// Minimum number of rows of the Doornik-Hansen (2008) test.
pub const DOORNIK_HANSEN_MIN_NOBS: usize = 8;

/// Shared validation of a residual matrix handed to a free normality
/// function; returns the centred matrix.
///
/// Refuses (each naming `resid`): no columns; fewer than `min_rows` rows or
/// no more rows than columns; a non-finite entry; a constant column; and a
/// correlation matrix whose smallest eigenvalue is `<= 1e-12 x` its trace
/// (collinear columns, e.g. a duplicate).
fn validate_resid(resid: MatRef<'_, f64>, min_rows: usize) -> Result<Mat<f64>, VarError> {
    let (t, k) = (resid.nrows(), resid.ncols());
    if k == 0 {
        return Err(VarError::InvalidParameter {
            name: "resid",
            value: 0.0,
            requirement: "at least one column: pass the T x k residual matrix (a 1-D \
                          series is one column)",
        });
    }
    if t < min_rows {
        return Err(VarError::InvalidParameter {
            name: "resid",
            value: t as f64,
            requirement: if min_rows >= DOORNIK_HANSEN_MIN_NOBS {
                "at least 8 rows: the Doornik-Hansen (2008) skewness transform \
                 degenerates at T = 7 (beta = 3, omega^2 = 1) and is undefined below \
                 (scipy.stats.skewtest refuses below 8 too); pass more rows (the value \
                 shown is the row count)"
            } else {
                "at least 2 rows: a residual covariance needs two observations; pass \
                 more rows (the value shown is the row count)"
            },
        });
    }
    if t <= k {
        return Err(VarError::InvalidParameter {
            name: "resid",
            value: t as f64,
            requirement: "more rows than columns: the k x k covariance of k residual \
                          series is singular with T <= k observations; pass more rows \
                          or fewer columns (the value shown is the row count)",
        });
    }
    for j in 0..k {
        for i in 0..t {
            if !resid[(i, j)].is_finite() {
                return Err(VarError::NonFinite {
                    what: "resid (the residual matrix of the normality test)",
                    at: Some((i, j)),
                });
            }
        }
    }
    let u = centred(resid);
    for j in 0..k {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut raw2, mut cen2) = (0.0, 0.0);
        for i in 0..t {
            let x = resid[(i, j)];
            lo = lo.min(x);
            hi = hi.max(x);
            raw2 += x * x;
            cen2 += u[(i, j)] * u[(i, j)];
        }
        if !(raw2.is_finite() && cen2.is_finite()) {
            return Err(VarError::NonFinite {
                what: "the second moments of resid (entries so large that their squares \
                       overflow; rescale resid)",
                at: None,
            });
        }
        // Exactly constant, or varying only at the rounding level of its
        // own magnitude (standard deviation below 1e-12 x its RMS); the sums
        // are finite here, so the comparison is total.
        if lo == hi || cen2 <= 1e-24 * raw2 {
            return Err(VarError::InvalidArgument {
                what: "resid has a constant column (zero sample variance after \
                       centring — or entries so small that their squares underflow): \
                       every normality statistic standardises each column, so a \
                       constant column has no skewness or kurtosis to test — drop that \
                       column from resid (or rescale it)",
            });
        }
    }
    // Collinear columns: the scale-free check on the correlation matrix.
    let s = ml_cov(&u);
    let dinv: Vec<f64> = (0..k).map(|j| 1.0 / s[(j, j)].sqrt()).collect();
    let corr = Mat::from_fn(k, k, |a, b| s[(a, b)] * dinv[a] * dinv[b]);
    let (lambda, _) = sym_eigen(&corr)?;
    if !eigen_floor_ok(&corr, &lambda) {
        return Err(VarError::NotPositiveDefinite {
            what: "the correlation matrix of resid, the residual matrix of the normality \
                   test (its smallest eigenvalue is <= 1e-12 x its trace)",
        });
    }
    Ok(u)
}

/// The Cholesky-orthogonalised multivariate Jarque-Bera test of a residual
/// matrix (`T x k`, column-centred here; module docs, item 1) — statsmodels
/// `VARResults.test_normality` and JMulTi's "Lütkepohl (1993)" block.
/// [`VarResults::normality_test`] is this function on the fit's residuals.
///
/// # Errors
///
/// [`VarError::InvalidParameter`] naming `resid` (no columns, fewer than 2
/// rows, or `T <= k`), [`VarError::NonFinite`] (a NaN or infinity),
/// [`VarError::InvalidArgument`] (a constant column) and
/// [`VarError::NotPositiveDefinite`] (collinear columns: the correlation
/// matrix's smallest eigenvalue `<= 1e-12 x` its trace).
pub fn normality_test_cholesky(resid: MatRef<'_, f64>) -> Result<NormalityTest, VarError> {
    let u = validate_resid(resid, 2)?;
    jb_cholesky(
        &u,
        "the covariance of resid, the residual matrix of the normality test",
    )
}

/// The symmetric-root multivariate Jarque-Bera test of a residual matrix
/// (module docs, item 2) — the statistic JMulTi prints under "Doornik &
/// Hansen (1994)": the Cholesky form with `S^{-1/2}` in place of `P^{-1}`
/// and no further transforms. It depends on the units of the columns.
///
/// # Errors
///
/// As [`normality_test_cholesky`], plus [`VarError::InvalidArgument`] when
/// the covariance's smallest eigenvalue is `<= 1e-12 x` its trace although
/// the correlation matrix is fine — columns on wildly different scales,
/// for which the unit-dependent symmetric root cannot be computed reliably.
pub fn normality_test_symmetric(resid: MatRef<'_, f64>) -> Result<NormalityTest, VarError> {
    let u = validate_resid(resid, 2)?;
    jb_symmetric(&u, "").map_err(|e| match e {
        VarError::NotPositiveDefinite { .. } => VarError::InvalidArgument {
            what: "resid: the columns are on wildly different scales (the smallest \
                   eigenvalue of their covariance is <= 1e-12 x its trace while the \
                   correlation matrix is well conditioned), so the symmetric-root \
                   (JMulTi) statistics, which depend on units, cannot be computed \
                   reliably — rescale the columns (e.g. divide each by its standard \
                   deviation); the Cholesky and Doornik-Hansen (2008) statistics are \
                   scale-invariant",
        },
        other => other,
    })
}

/// The Doornik-Hansen (2008) omnibus normality test of a residual matrix
/// (module docs, item 3). A single column (`k = 1`) is the univariate
/// Doornik-Hansen test.
///
/// # Errors
///
/// As [`normality_test_cholesky`], with `T >= 8` rows required
/// ([`VarError::InvalidParameter`] naming `resid`).
pub fn doornik_hansen_test(resid: MatRef<'_, f64>) -> Result<DoornikHansenTest, VarError> {
    let u = validate_resid(resid, DOORNIK_HANSEN_MIN_NOBS)?;
    dh_centred(&u)
}

impl VarResults {
    /// Column-centred residuals `ũ = U - mean(U)` (`T × k`).
    fn centred_resid(&self) -> Mat<f64> {
        centred(self.resid.as_ref())
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
    /// docs), plus its skewness and kurtosis components — the shared body of
    /// [`normality_test_cholesky`] on the fit's residuals.
    ///
    /// # Errors
    ///
    /// [`VarError::NotPositiveDefinite`] if the residual covariance is
    /// singular (impossible for a successful fit).
    pub fn normality_test(&self) -> Result<NormalityTest, VarError> {
        jb_cholesky(
            &self.centred_resid(),
            "the centred residual covariance of the normality test",
        )
    }

    /// The symmetric-root multivariate Jarque-Bera test of the residuals
    /// (JMulTi's "Doornik & Hansen (1994)" block):
    /// [`normality_test_symmetric`] on the fit's residuals.
    ///
    /// # Errors
    ///
    /// As [`normality_test_symmetric`].
    pub fn normality_test_symmetric(&self) -> Result<NormalityTest, VarError> {
        normality_test_symmetric(self.resid.as_ref())
    }

    /// The Doornik-Hansen (2008) omnibus test of the residuals:
    /// [`doornik_hansen_test`] on the fit's residuals (`nobs >= 8`).
    ///
    /// # Errors
    ///
    /// As [`doornik_hansen_test`] — in particular
    /// [`VarError::InvalidParameter`] naming `resid` when `nobs < 8`.
    pub fn doornik_hansen_test(&self) -> Result<DoornikHansenTest, VarError> {
        doornik_hansen_test(self.resid.as_ref())
    }

    /// The residual-diagnostics bundle: [`VarResults::portmanteau_test`] at
    /// `nlags`, [`VarResults::normality_test`], the symmetric-root and
    /// Doornik-Hansen (2008) normality tests, and the stability roots.
    ///
    /// The two added normality tests are `Option`s that never make the
    /// bundle refuse: [`VarDiagnostics::doornik_hansen`] is `None` when
    /// `nobs < 8`, and either is `None` when its matrix root is numerically
    /// singular (see the field docs).
    ///
    /// # Errors
    ///
    /// Anything the Portmanteau test, the Cholesky normality test or the
    /// stability roots can return.
    pub fn diagnostics(&self, nlags: usize) -> Result<VarDiagnostics, VarError> {
        let portmanteau = self.portmanteau_test(nlags)?;
        let normality = self.normality_test()?;
        let normality_symmetric = self.normality_test_symmetric().ok();
        let doornik_hansen = if self.nobs >= DOORNIK_HANSEN_MIN_NOBS {
            self.doornik_hansen_test().ok()
        } else {
            None
        };
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
            normality_symmetric,
            doornik_hansen,
            roots,
            eigenvalue_moduli,
            is_stable,
            nobs: self.nobs,
            neqs: self.neqs,
            lags: self.spec.lags,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// The exact mean-matching identity of the gamma approximation behind
    /// the kurtosis transform: with `E b1 = 6(n-2)/((n+1)(n+3))` and
    /// `E b2 = 3(n-1)/(n+1)`, `2κ(E b2 - 1 - E b1) = 2(a + c E b1)`. It holds
    /// exactly in rationals (the fixture generator checks it with
    /// `fractions.Fraction`); here the f64 coefficients must satisfy it to
    /// 1e-13 relative, which catches a transcription typo in `a`, `c` or `κ`
    /// (the common denominator `d` cancels, so it is guarded by the null
    /// Monte Carlo of `z2` in tests/var_cf_properties.rs instead).
    #[test]
    fn kurtosis_coefficients_satisfy_the_exact_mean_matching_identity() {
        for n in [8usize, 9, 10, 20, 50, 200, 1000] {
            let nf = n as f64;
            let (a, c, kappa) = dh_kurt_coeffs(nf);
            let eb1 = 6.0 * (nf - 2.0) / ((nf + 1.0) * (nf + 3.0));
            let eb2 = 3.0 * (nf - 1.0) / (nf + 1.0);
            let lhs = 2.0 * kappa * (eb2 - 1.0 - eb1);
            let rhs = 2.0 * (a + c * eb1);
            assert!(
                (lhs - rhs).abs() <= 1e-13 * rhs.abs(),
                "n = {n}: 2 kappa (E b2 - 1 - E b1) = {lhs:e} but 2 (a + c E b1) = {rhs:e}"
            );
        }
    }

    /// D'Agostino's `β` is exactly 3 at `n = 7`, where `ω² = 1` and the
    /// skewness transform's `δ = 1/√(ln ω)` is infinite — the reason the
    /// test needs `n >= 8`. At `n = 8` every coefficient is finite.
    #[test]
    fn the_skewness_transform_degenerates_exactly_at_n_7() {
        assert_eq!(dh_beta(7.0), 3.0);
        let (delta, _) = dh_skew_coeffs(7.0);
        assert!(delta.is_infinite());
        let (delta, scale) = dh_skew_coeffs(8.0);
        assert!(delta.is_finite() && scale.is_finite() && scale > 0.0);
        let (a, c, kappa) = dh_kurt_coeffs(8.0);
        assert!(a > 0.0 && c > 0.0 && kappa > 0.0);
    }

    #[test]
    fn seven_rows_are_refused_naming_resid_and_eight_are_accepted() {
        let x = [0.3, -1.2, 0.8, 2.1, -0.4, 0.05, -0.9, 1.4];
        let seven = Mat::from_fn(7, 1, |i, _| x[i]);
        let e = doornik_hansen_test(seven.as_ref()).unwrap_err();
        assert!(
            matches!(e, VarError::InvalidParameter { name: "resid", .. }),
            "{e:?}"
        );
        let text = e.to_string();
        assert!(text.contains("resid") && text.contains("8 rows"), "{text}");
        let eight = Mat::from_fn(8, 1, |i, _| x[i]);
        let r = doornik_hansen_test(eight.as_ref()).unwrap();
        assert!(r.statistic.is_finite() && r.df == 2);
        // The Jarque-Bera forms only need two rows and T > k.
        assert!(normality_test_cholesky(seven.as_ref()).is_ok());
        assert!(normality_test_symmetric(seven.as_ref()).is_ok());
    }
}
