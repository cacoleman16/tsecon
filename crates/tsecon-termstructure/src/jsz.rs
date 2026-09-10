//! The JSZ canonical Gaussian dynamic term-structure model (Joslin,
//! Singleton & Zhu 2011).
//!
//! Joslin, Singleton & Zhu (2011, *Review of Financial Studies* 24(3)) show
//! that every `N`-factor Gaussian dynamic term-structure model (GDTSM) is
//! observationally equivalent to a **canonical** one whose risk-neutral
//! (`Q`) dynamics carry only `N + 1` free parameters, and that — once the
//! pricing factors are rotated onto `N` observed yield portfolios — the
//! physical (`P`) dynamics can be **concentrated out of the likelihood by
//! OLS**. The rotation indeterminacy that plagued earlier canonical forms
//! (Dai-Singleton 2000) disappears, the numerical search shrinks to a handful
//! of well-identified `Q` parameters, and the maximum-likelihood estimator
//! becomes nearly self-checking. This module implements that estimator.
//!
//! ## The canonical model
//!
//! A latent state `X_t` (`N` x 1) follows a Gaussian VAR(1) under the
//! risk-neutral measure,
//!
//! ```text
//! X_{t+1} = K0^Q + K1^Q X_t + Sigma_X eps^Q_{t+1},   eps^Q ~ N(0, I),
//! K0^Q = (k_inf, 0, ..., 0)',    K1^Q = J(lambda^Q),    r_t = iota' X_t,
//! ```
//!
//! with `J(lambda^Q)` the real Jordan form of the ordered eigenvalues
//! `lambda_1 >= lambda_2 >= ... >= lambda_N` (diagonal when they are
//! distinct; equal neighbours form a Jordan block with ones on the
//! superdiagonal) and `r_t` the one-period short rate. No-arbitrage makes
//! every log bond price affine, `ln P_t^(n) = A_n + B_n' X_t`, with the
//! Riccati recursions (seeded at the zero-maturity bond, `A_0 = 0`, `B_0 =
//! 0`)
//!
//! ```text
//! A_{n+1} = A_n + K0^Q' B_n + 1/2 B_n' Sigma_X Sigma_X' B_n,
//! B_{n+1} = K1^Q' B_n - iota,
//! ```
//!
//! and per-period yields `y_t^(n) = -(A_n + B_n' X_t) / n`. The `Q`
//! parameters are `(k_inf, lambda^Q)` plus the innovation covariance; the
//! `P` dynamics `X_{t+1} = K0^P + K1^P X_t + Sigma_X eps^P` are unrestricted.
//!
//! ## The rotation to observed portfolios, and why the likelihood factors
//!
//! Let `W` (`N` x `M`) map the `M` observed yields to `N` portfolios
//! `P_t = W y_t` — by default the first `N` principal-component loadings of
//! the panel, or any user-supplied full-row-rank matrix. Because yields are
//! affine in `X_t`, so are the portfolios, `P_t = W A_X + (W B_X) X_t`, and
//! the model can be rewritten with `P_t` itself as the state. JSZ assume the
//! portfolios are priced **without error** and the remaining `M - N`
//! directions of the yield cross-section with iid Gaussian error `sigma_e`.
//! The density of the panel then factors,
//!
//! ```text
//! f(y_t | y_{t-1}) = f^P(P_t | P_{t-1}; K0^P_P, K1^P_P, Sigma_P)
//!                  x f^Q(y_t | P_t; k_inf, lambda^Q, Sigma_P, sigma_e),
//! ```
//!
//! and `(K0^P_P, K1^P_P)` appear only in the first factor — a Gaussian
//! VAR(1) whose maximum-likelihood estimate is **OLS**, whatever `Sigma_P`
//! is. They are therefore concentrated out exactly (the P-measure VAR this
//! module reports is bit-for-bit the equation-by-equation OLS statsmodels'
//! `VAR(1)` computes). Two more parameters concentrate out analytically at
//! every evaluation: the pricing errors are affine in `k_inf`, so its
//! profile maximizer is a one-dimensional least-squares coefficient, and
//! `sigma_e` is the root-mean-square pricing error. The numerical search
//! therefore runs only over `lambda^Q` (`N` values, ordered through
//! log-gaps) and the Cholesky factor of `Sigma_P` (`N (N + 1) / 2`) —
//! `Sigma_P` enters both factors, through the VAR density and through the
//! convexity term of `A_n`, so it cannot be profiled in closed form (JSZ note
//! its `Q`-side dependence is weak, which is why the OLS covariance is an
//! excellent start).
//!
//! ## Two representations of the latent state
//!
//! When two `Q`-eigenvalues approach each other, the columns of `B_X` in the
//! diagonal canonical basis become collinear and the rotation matrix `W B_X`
//! ill-conditioned — yet the *model* is perfectly regular there (the AFNS
//! case `lambda^Q = (1, rho, rho)` lives exactly on that boundary). The fit
//! therefore evaluates the recursions in an equivalent **Jordan-chain**
//! representation: `K1^Q` upper bidiagonal with `lambda^Q` on the diagonal
//! and ones on the *whole* superdiagonal. Its loadings are divided
//! differences of the powers `lambda_k^j`, continuous through coincidences,
//! and they coincide with the literal Jordan form whenever eigenvalues tie.
//! The two representations are related by a state rotation `C` with
//! `C e_1 = e_1`, so `k_inf` keeps its meaning, the intercept `A_n` is
//! identical, and everything reported in the portfolio rotation is
//! representation-free. [`JszFit::b_x`] and [`jsz_loadings`] report the
//! *literal* canonical form (diagonal for distinct eigenvalues, Jordan
//! blocks for exact ties).
//!
//! ## Units and outputs
//!
//! Inputs are annualized continuously-compounded zero-coupon yields in
//! decimal, at integer maturities in periods; `periods_per_year` converts to
//! the per-period quantities the recursions price. The `Q` parameters
//! `lambda_q` and `k_inf_q` are per-period (JSZ's convention); portfolio-
//! rotation quantities (`mu_p`, `phi_p`, `sigma`, `k0_q_p`, `k1_q_p`,
//! `a_p`, `b_p`) are in the annualized units of `P_t = W y_t`; `sigma_e`
//! is in annualized-yield units. The term-premium decomposition follows
//! [`crate::acm_term_premium`] exactly: **risk-neutral** yields re-run the
//! recursion with the `P`-measure VAR `(mu_p, phi_p)` in place of the `Q`
//! dynamics (convexity kept), and `term_premium = fitted - risk_neutral`.
//! The market prices of risk in ACM's units are then `lambda0 = mu_p -
//! k0_q_p` and `lambda1 = phi_p - k1_q_p`.
//!
//! ## Reading the fit (and the two traps)
//!
//! - **The likelihood is flat in the market prices of risk.** `lambda^Q`
//!   and `k_inf` are pinned by the cross-section (hundreds of pricing
//!   equations per date) and are estimated to many digits; `(mu_p, phi_p)`
//!   come from a `T`-observation VAR of highly persistent factors and are
//!   not. The difference `phi_p - k1_q_p` — the state-dependent price of
//!   risk — inherits the VAR's imprecision, which is why `mu_p_se` /
//!   `phi_p_se` (the OLS standard errors, statsmodels' `VAR` convention) are
//!   reported: read the term premium's *level* with those in mind. No
//!   standard errors are reported for the `Q` parameters — an honest
//!   asymptotic covariance of a profile likelihood near a unit root is not
//!   something a numerical Hessian delivers reliably, and a number that
//!   looks precise but is not would be worse than none.
//! - **The level factor is nearly a unit root under `Q`.** `lambda_1` is
//!   left unbounded: the recursions are evaluated by the recurrence itself
//!   (no `1 / (1 - lambda)` closed forms), so nothing degrades at or across
//!   `lambda_1 = 1`, and `k_inf` stays identified there (it becomes the
//!   drift of a unit-root level: the intercept grows linearly in maturity).
//!   An estimate above 1 means explosive risk-neutral dynamics — report it,
//!   do not hide it, and check the sample.
//! - **`W` is a normalization within its row space, an assumption across
//!   row spaces.** Any two bases of the same portfolio space (`W` and
//!   `G W`) give identical `llf`, fitted yields, `lambda_q`, `k_inf_q` and
//!   `sigma_e` (the fit runs internally in a canonical orthonormal basis of
//!   that space and maps back exactly). Portfolios spanning a *different*
//!   space are a different statement about which yields are priced exactly
//!   and give a different — usually very close — fit.
//!
//! ## References
//!
//! - Joslin, S., Singleton, K. J., & Zhu, H. (2011). "A New Perspective on
//!   Gaussian Dynamic Term Structure Models." *Review of Financial Studies*,
//!   24(3), 926-970.
//! - Dai, Q., & Singleton, K. J. (2000). "Specification Analysis of Affine
//!   Term Structure Models." *Journal of Finance*, 55(5), 1943-1978.
//! - Christensen, J. H. E., Diebold, F. X., & Rudebusch, G. D. (2011). "The
//!   affine arbitrage-free class of Nelson-Siegel term structure models."
//!   *Journal of Econometrics*, 164(1), 4-20. (The AFNS special case
//!   `lambda^Q = (1, e^{-lambda}, e^{-lambda})`.)
//! - Adrian, T., Crump, R. K., & Moench, E. (2013). "Pricing the Term
//!   Structure with Linear Regressions." *Journal of Financial Economics*,
//!   110(1), 110-138. (The term-premium convention shared with
//!   [`crate::acm_term_premium`].)

use crate::error::TermStructureError;
use crate::fit::map_ols_err;
use tsecon_hac::{ols, SeType};
use tsecon_linalg::faer::Mat;
use tsecon_optim::{minimize, FnObjective, Method, NelderMeadOptions, OptimizeResult};

/// Row-major dense matrix.
type Matrix = Vec<Vec<f64>>;

// ---------------------------------------------------------------------------
// Small dense helpers (N x N with N = n_factors, a handful at most).
// ---------------------------------------------------------------------------

fn zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

fn identity(n: usize) -> Matrix {
    let mut m = zeros(n, n);
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

fn transpose(a: &[Vec<f64>]) -> Matrix {
    if a.is_empty() {
        return Vec::new();
    }
    let (r, c) = (a.len(), a[0].len());
    let mut t = zeros(c, r);
    for i in 0..r {
        for j in 0..c {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
    let r = a.len();
    let inner = if r == 0 { 0 } else { a[0].len() };
    let c = if b.is_empty() { 0 } else { b[0].len() };
    let mut out = zeros(r, c);
    for i in 0..r {
        for k in 0..inner {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..c {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn mat_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&x, &y)| x * y).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Inverse of a small square matrix by Gauss-Jordan elimination with partial
/// pivoting; `None` when a pivot falls below `1e-13` of the largest entry
/// (numerically singular).
fn invert(a: &[Vec<f64>]) -> Option<Matrix> {
    let n = a.len();
    let mut work: Matrix = a.iter().map(|row| row.to_vec()).collect();
    let mut inv = identity(n);
    let scale = a
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0f64, |acc, &v| acc.max(v.abs()));
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if work[row][col].abs() > work[pivot][col].abs() {
                pivot = row;
            }
        }
        let p = work[pivot][col];
        if !p.is_finite() || p.abs() <= 1e-13 * scale {
            return None;
        }
        work.swap(col, pivot);
        inv.swap(col, pivot);
        let inv_p = 1.0 / p;
        for j in 0..n {
            work[col][j] *= inv_p;
            inv[col][j] *= inv_p;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = work[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..n {
                work[row][j] -= factor * work[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }
    Some(inv)
}

/// Lower Cholesky factor of a symmetric positive-definite matrix, or `None`.
fn cholesky(a: &[Vec<f64>]) -> Option<Matrix> {
    let n = a.len();
    let mut l = zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let cross: f64 = l[i][..j]
                .iter()
                .zip(l[j][..j].iter())
                .map(|(x, y)| x * y)
                .sum();
            let s = a[i][j] - cross;
            if i == j {
                if !s.is_finite() || s <= 0.0 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

fn quad_form(m: &[Vec<f64>], v: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (i, row) in m.iter().enumerate() {
        for (j, &mij) in row.iter().enumerate() {
            acc += v[i] * mij * v[j];
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Strictly ascending positive integer maturities.
fn check_maturities(maturities: &[usize]) -> Result<(), TermStructureError> {
    if maturities.is_empty() {
        return Err(TermStructureError::EmptyMaturities);
    }
    for (i, &m) in maturities.iter().enumerate() {
        if m == 0 {
            return Err(TermStructureError::InvalidMaturity {
                index: i,
                value: 0.0,
            });
        }
        if i > 0 && m <= maturities[i - 1] {
            return Err(TermStructureError::MaturitiesNotAscending { index: i });
        }
    }
    Ok(())
}

/// Finite, non-increasing Q-eigenvalues.
fn check_lambda_q(lambda_q: &[f64]) -> Result<(), TermStructureError> {
    for (index, &v) in lambda_q.iter().enumerate() {
        if !v.is_finite() {
            return Err(TermStructureError::InvalidQEigenvalue { index, value: v });
        }
        if index > 0 && v > lambda_q[index - 1] {
            return Err(TermStructureError::QEigenvaluesNotOrdered { index });
        }
    }
    Ok(())
}

fn check_square(m: &[Vec<f64>], n: usize, what: &'static str) -> Result<(), TermStructureError> {
    if m.len() != n {
        return Err(TermStructureError::DimensionMismatch {
            what,
            expected: n,
            got: m.len(),
        });
    }
    for (i, row) in m.iter().enumerate() {
        if row.len() != n {
            return Err(TermStructureError::DimensionMismatch {
                what,
                expected: n,
                got: row.len(),
            });
        }
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(TermStructureError::NonFinite {
                    what,
                    index: i * n + j,
                    value: v,
                });
            }
        }
    }
    Ok(())
}

fn check_ppy(periods_per_year: f64) -> Result<(), TermStructureError> {
    if !periods_per_year.is_finite() || periods_per_year <= 0.0 {
        return Err(TermStructureError::InvalidPeriodsPerYear {
            value: periods_per_year,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The Riccati recursions
// ---------------------------------------------------------------------------

/// Superdiagonal of `K1^Q`: all ones in the Jordan-chain representation,
/// ones only at exact ties in the literal canonical form.
fn superdiagonal(lambda_q: &[f64], chain: bool) -> Vec<f64> {
    (1..lambda_q.len())
        .map(|k| {
            if chain || lambda_q[k] == lambda_q[k - 1] {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// `K1^Q` as a dense matrix (diagonal `lambda_q`, superdiagonal as above).
fn k1_matrix(lambda_q: &[f64], chain: bool) -> Matrix {
    let n = lambda_q.len();
    let s = superdiagonal(lambda_q, chain);
    let mut k1 = zeros(n, n);
    for k in 0..n {
        k1[k][k] = lambda_q[k];
        if k > 0 {
            k1[k - 1][k] = s[k - 1];
        }
    }
    k1
}

/// The loading recursion `B_{n+1} = K1^Q' B_n - iota` for `n = 0..=n_max`
/// (row `n` is `B_n`, `B_0 = 0`), plus the `k_inf` accumulator
/// `alpha_{n+1} = alpha_n + B_n[0]`, so that `A_n = k_inf alpha_n +
/// gamma_n` with `gamma` the convexity sum of [`convexity`].
fn loading_recursion(lambda_q: &[f64], chain: bool, n_max: usize) -> (Matrix, Vec<f64>) {
    let n = lambda_q.len();
    let s = superdiagonal(lambda_q, chain);
    let mut b = zeros(n_max + 1, n);
    let mut alpha = vec![0.0f64; n_max + 1];
    for m in 0..n_max {
        for k in 0..n {
            let mut v = lambda_q[k] * b[m][k];
            if k > 0 {
                v += s[k - 1] * b[m][k - 1];
            }
            b[m + 1][k] = v - 1.0;
        }
        alpha[m + 1] = alpha[m] + b[m][0];
    }
    (b, alpha)
}

/// The convexity part of the intercept recursion,
/// `gamma_{n+1} = gamma_n + 1/2 B_n' Sigma_X B_n` (`gamma_0 = 0`), where
/// `sigma_x` is the per-period innovation **covariance** of the state.
fn convexity(b: &[Vec<f64>], sigma_x: &[Vec<f64>]) -> Vec<f64> {
    let mut gamma = vec![0.0f64; b.len()];
    for m in 0..b.len() - 1 {
        gamma[m + 1] = gamma[m] + 0.5 * quad_form(sigma_x, &b[m]);
    }
    gamma
}

/// The JSZ canonical loadings at fixed parameters (see [`jsz_loadings`]).
#[derive(Debug, Clone, PartialEq)]
pub struct JszLoadings {
    /// The input maturities (periods).
    pub maturities: Vec<usize>,
    /// Yield intercepts `a_n = -A_n / n` per maturity, times
    /// `periods_per_year`.
    pub a: Vec<f64>,
    /// Yield loadings `b_n = -B_n' / n` per maturity (`M x N`,
    /// dimensionless).
    pub b: Matrix,
    /// `K0^Q = (k_inf, 0, ..., 0)'`.
    pub k0_q: Vec<f64>,
    /// `K1^Q = J(lambda_q)`: diagonal, with a `1` on the superdiagonal
    /// wherever two consecutive eigenvalues are exactly equal (a Jordan
    /// block).
    pub k1_q: Matrix,
}

/// The JSZ canonical bond-loading recursions at given parameters.
///
/// Evaluates the Riccati recursions of the [module docs](self) for the
/// literal canonical form `K0^Q = (k_inf_q, 0, ..., 0)'`, `K1^Q =
/// J(lambda_q)` (diagonal for distinct eigenvalues, Jordan blocks with ones
/// on the superdiagonal for exactly equal neighbours — the AFNS case
/// `(1, rho, rho)` is one such block), `r_t = iota' X_t`, and per-period
/// state innovation covariance `sigma_x` (`N x N`, in the canonical basis),
/// and returns the per-maturity yield coefficients `a_n = -A_n / n` and
/// `b_n = -B_n' / n` so that `y_t^(n) = a_n + b_n X_t`.
///
/// `periods_per_year` only rescales the intercept `a` (multiplies it), so
/// that with `12.0` the intercepts are in annualized-yield units for a
/// per-period state; pass `1.0` for pure per-period output. `lambda_q` and
/// `k_inf_q` are always per-period.
///
/// # Errors
///
/// [`TermStructureError::EmptyMaturities`], [`TermStructureError::InvalidMaturity`]
/// (a zero maturity), [`TermStructureError::MaturitiesNotAscending`],
/// [`TermStructureError::InvalidQEigenvalue`] (non-finite or empty
/// `lambda_q`), [`TermStructureError::QEigenvaluesNotOrdered`],
/// [`TermStructureError::NonFinite`] (`k_inf_q` or `sigma_x`),
/// [`TermStructureError::DimensionMismatch`] (`sigma_x` not `N x N`), and
/// [`TermStructureError::InvalidPeriodsPerYear`].
///
/// # Example
///
/// ```
/// use tsecon_termstructure::jsz_loadings;
/// // The AFNS pattern: a unit-root level and a Jordan block at rho.
/// let rho = (-0.0609f64).exp();
/// let sigma_x = vec![vec![0.0; 3]; 3];
/// let l = jsz_loadings(&[1.0, rho, rho], 0.0, &sigma_x, &[1, 12, 60, 120], 1.0).unwrap();
/// assert_eq!(l.k1_q[1][2], 1.0); // the Jordan block
/// assert_eq!(l.k1_q[0][1], 0.0); // distinct from the level
/// // The level loading is flat at 1, the one-period yield is the short rate.
/// assert!(l.b.iter().all(|row| (row[0] - 1.0).abs() < 1e-12));
/// assert_eq!(l.a[0], 0.0);
/// ```
pub fn jsz_loadings(
    lambda_q: &[f64],
    k_inf_q: f64,
    sigma_x: &[Vec<f64>],
    maturities: &[usize],
    periods_per_year: f64,
) -> Result<JszLoadings, TermStructureError> {
    check_maturities(maturities)?;
    if lambda_q.is_empty() {
        return Err(TermStructureError::InvalidQEigenvalue {
            index: 0,
            value: f64::NAN,
        });
    }
    check_lambda_q(lambda_q)?;
    if !k_inf_q.is_finite() {
        return Err(TermStructureError::NonFinite {
            what: "k_inf_q",
            index: 0,
            value: k_inf_q,
        });
    }
    let n = lambda_q.len();
    check_square(sigma_x, n, "JSZ sigma_x (must be n_factors x n_factors)")?;
    check_ppy(periods_per_year)?;

    let n_max = maturities[maturities.len() - 1];
    let (b_rec, alpha) = loading_recursion(lambda_q, false, n_max);
    let gamma = convexity(&b_rec, sigma_x);
    let a: Vec<f64> = maturities
        .iter()
        .map(|&m| -(k_inf_q * alpha[m] + gamma[m]) / m as f64 * periods_per_year)
        .collect();
    let b: Matrix = maturities
        .iter()
        .map(|&m| b_rec[m].iter().map(|&v| -v / m as f64).collect())
        .collect();
    let mut k0_q = vec![0.0f64; n];
    k0_q[0] = k_inf_q;
    Ok(JszLoadings {
        maturities: maturities.to_vec(),
        a,
        b,
        k0_q,
        k1_q: k1_matrix(lambda_q, false),
    })
}

// ---------------------------------------------------------------------------
// The portfolio rotation and the P-measure VAR
// ---------------------------------------------------------------------------

/// Sign convention for a basis vector: its largest-magnitude entry positive.
fn fix_sign(v: &mut [f64]) {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x.abs() > v[best].abs() {
            best = i;
        }
    }
    if v[best] < 0.0 {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

/// The default `W`: the first `N` principal-component loadings of the
/// demeaned yield panel (orthonormal rows, sign-fixed by [`fix_sign`]).
fn pca_weights(yields: &[Vec<f64>], n_factors: usize) -> Result<Matrix, TermStructureError> {
    let t = yields.len();
    let m = yields[0].len();
    let means: Vec<f64> = (0..m)
        .map(|j| yields.iter().map(|r| r[j]).sum::<f64>() / t as f64)
        .collect();
    let demeaned = Mat::from_fn(t, m, |i, j| yields[i][j] - means[j]);
    let svd = demeaned
        .thin_svd()
        .map_err(|_| TermStructureError::SingularDesign {
            what: "JSZ principal-component portfolio weights (SVD did not converge)",
        })?;
    let v = svd.V();
    let mut w = zeros(n_factors, m);
    for (c, row) in w.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = v[(j, c)];
        }
        fix_sign(row);
    }
    Ok(w)
}

/// A canonical orthonormal basis of the row space of `w`: orthonormalize,
/// then align with the principal axes of the projected, demeaned panel and
/// fix signs. Depends on `w` only through its row space, so any two bases
/// of the same portfolio space yield the same matrix (to rounding).
fn canonical_basis(w: &[Vec<f64>], yields: &[Vec<f64>]) -> Result<Matrix, TermStructureError> {
    let n = w.len();
    let m = w[0].len();
    let t = yields.len();
    // Orthonormal basis U (M x N) of rowspace(W) from the thin SVD of W'.
    let wt = Mat::from_fn(m, n, |j, i| w[i][j]);
    let svd = wt
        .thin_svd()
        .map_err(|_| TermStructureError::InvalidWeights {
            reason: "the SVD of w did not converge",
            rows: n,
            cols: m,
        })?;
    let s = svd.S();
    let sv = s.column_vector();
    let s_max = (0..n).map(|i| sv[i]).fold(0.0f64, f64::max);
    let s_min = (0..n).map(|i| sv[i]).fold(f64::INFINITY, f64::min);
    if !s_min.is_finite() || s_min <= 1e-10 * s_max {
        return Err(TermStructureError::InvalidWeights {
            reason: "the rows of w are linearly dependent (rank-deficient)",
            rows: n,
            cols: m,
        });
    }
    let u = svd.U();
    // Coordinates of the demeaned panel in the U basis: Z = Y_d U (T x N).
    let means: Vec<f64> = (0..m)
        .map(|j| yields.iter().map(|r| r[j]).sum::<f64>() / t as f64)
        .collect();
    let z = Mat::from_fn(t, n, |i, c| {
        (0..m)
            .map(|j| (yields[i][j] - means[j]) * u[(j, c)])
            .sum::<f64>()
    });
    let zsvd = z
        .thin_svd()
        .map_err(|_| TermStructureError::SingularDesign {
            what: "JSZ canonical portfolio basis (SVD did not converge)",
        })?;
    let v = zsvd.V();
    // W_c = (U V)'.
    let mut wc = zeros(n, m);
    for (c, row) in wc.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = (0..n).map(|k| u[(j, k)] * v[(k, c)]).sum();
        }
        fix_sign(row);
    }
    Ok(wc)
}

/// The P-measure VAR(1) of a portfolio panel by equation-by-equation OLS.
struct PVar {
    mu: Vec<f64>,
    phi: Matrix,
    mu_se: Vec<f64>,
    phi_se: Matrix,
    /// Residual cross-product `U'U` (`N x N`).
    s: Matrix,
}

fn p_var(p: &[Vec<f64>]) -> Result<PVar, TermStructureError> {
    let t = p.len();
    let n = p[0].len();
    let t_v = t - 1;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    cols.push(vec![1.0; t_v]);
    cols.extend((0..n).map(|c| p.iter().take(t_v).map(|r| r[c]).collect::<Vec<f64>>()));
    let mut mu = vec![0.0; n];
    let mut phi = zeros(n, n);
    let mut mu_se = vec![0.0; n];
    let mut phi_se = zeros(n, n);
    let mut resid = zeros(t_v, n);
    for eq in 0..n {
        let y: Vec<f64> = (1..t).map(|i| p[i][eq]).collect();
        let fit = ols(&y, &cols).map_err(|e| map_ols_err(e, "JSZ portfolio VAR(1)"))?;
        mu[eq] = fit.params[0];
        phi[eq].copy_from_slice(&fit.params[1..]);
        for (i, &r) in fit.residuals.iter().enumerate() {
            resid[i][eq] = r;
        }
        let inf = fit
            .inference(SeType::NonRobust)
            .map_err(|e| map_ols_err(e, "JSZ portfolio VAR(1) standard errors"))?;
        mu_se[eq] = inf.bse[0];
        phi_se[eq].copy_from_slice(&inf.bse[1..]);
    }
    let mut s = zeros(n, n);
    for row in &resid {
        for i in 0..n {
            for j in 0..n {
                s[i][j] += row[i] * row[j];
            }
        }
    }
    Ok(PVar {
        mu,
        phi,
        mu_se,
        phi_se,
        s,
    })
}

/// Everything about the data that the likelihood needs, precomputed once.
struct Problem<'a> {
    yields: &'a [Vec<f64>],
    maturities: &'a [usize],
    ppy: f64,
    t: usize,
    m: usize,
    n: usize,
    n_max: usize,
    /// Canonical orthonormal portfolio basis (`N x M`).
    w_c: Matrix,
    /// Portfolios in that basis (`T x N`, annualized).
    p_c: Matrix,
    y_mean: Vec<f64>,
    var_c: PVar,
}

/// One evaluation of the concentrated likelihood at `(lambda_q, sigma_c)`
/// (`sigma_c` the annualized portfolio-innovation covariance in the
/// canonical basis).
struct Eval {
    llf: f64,
    k_inf: f64,
    sigma_e: f64,
    /// Annualized intercepts `a_p` (`M`) in the canonical rotation.
    a_pc: Vec<f64>,
    /// Loadings `b_p` (`M x N`) in the canonical rotation.
    b_pc: Matrix,
    /// `D = W_c B_X` and its inverse.
    d: Matrix,
    d_inv: Matrix,
    /// Per-period `c = W_c A_X`.
    c_pp: Vec<f64>,
    /// Chain-representation `B_n` table and the intercept pieces.
    b_rec: Matrix,
    alpha: Vec<f64>,
    gamma: Vec<f64>,
}

impl Problem<'_> {
    /// Evaluate at `(lambda_q, sigma_c)`. `k_inf`/`sigma_e` given -> plain
    /// log-likelihood; `None` -> profiled out (their maximizers are used).
    fn evaluate(
        &self,
        lambda_q: &[f64],
        sigma_c: &[Vec<f64>],
        fixed: Option<(f64, f64)>,
    ) -> Option<Eval> {
        let (t, m, n, ppy) = (self.t, self.m, self.n, self.ppy);
        let tf = t as f64;
        // --- P part: Gaussian VAR density at the OLS coefficients ---------
        let l = cholesky(sigma_c)?;
        let sigma_inv = invert(sigma_c)?;
        let log_det: f64 = 2.0 * l.iter().enumerate().map(|(i, r)| r[i].ln()).sum::<f64>();
        let trace: f64 = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| sigma_inv[i][j] * self.var_c.s[i][j])
                    .sum::<f64>()
            })
            .sum();
        let t_v = (t - 1) as f64;
        let llf_p = -0.5 * t_v * n as f64 * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * t_v * log_det
            - 0.5 * trace;
        // --- Q part ----------------------------------------------------------
        let (b_rec, alpha) = loading_recursion(lambda_q, true, self.n_max);
        let b_x: Matrix = self
            .maturities
            .iter()
            .map(|&mat| b_rec[mat].iter().map(|&v| -v / mat as f64).collect())
            .collect();
        let d = mat_mul(&self.w_c, &b_x);
        let d_inv = invert(&d)?;
        // Sigma_X (per period) = D^{-1} (Sigma_c / ppy^2) D^{-1}'.
        let sigma_pp: Matrix = sigma_c
            .iter()
            .map(|r| r.iter().map(|&v| v / (ppy * ppy)).collect())
            .collect();
        let sigma_x = mat_mul(&mat_mul(&d_inv, &sigma_pp), &transpose(&d_inv));
        let gamma = convexity(&b_rec, &sigma_x);
        let a0: Vec<f64> = self
            .maturities
            .iter()
            .map(|&mat| -gamma[mat] / mat as f64 * ppy)
            .collect();
        let a1: Vec<f64> = self
            .maturities
            .iter()
            .map(|&mat| -alpha[mat] / mat as f64 * ppy)
            .collect();
        let b_pc = mat_mul(&b_x, &d_inv);
        // Pi x = x - b_p (W_c x).
        let project = |x: &[f64]| -> Vec<f64> {
            let wx = mat_vec(&self.w_c, x);
            let bwx = mat_vec(&b_pc, &wx);
            x.iter().zip(bwx.iter()).map(|(&a, &b)| a - b).collect()
        };
        let ybar_minus_a0: Vec<f64> = (0..m).map(|j| self.y_mean[j] - a0[j]).collect();
        let u_bar = project(&ybar_minus_a0);
        let v = project(&a1);
        let vv = dot(&v, &v);
        let pi_a0 = project(&a0);
        let k_inf = match fixed {
            Some((k, _)) => k,
            None => {
                if !vv.is_finite() || vv <= 0.0 {
                    return None;
                }
                dot(&u_bar, &v) / vv
            }
        };
        let a_pc: Vec<f64> = (0..m).map(|j| pi_a0[j] + k_inf * v[j]).collect();
        let mut sse = 0.0f64;
        for (row, p_row) in self.yields.iter().zip(self.p_c.iter()) {
            for j in 0..m {
                let fit = a_pc[j] + dot(&b_pc[j], p_row);
                let e = row[j] - fit;
                sse += e * e;
            }
        }
        let dfq = tf * (m - n) as f64;
        let (sigma_e, llf_q) = match fixed {
            Some((_, se)) => (
                se,
                -0.5 * dfq * (2.0 * std::f64::consts::PI * se * se).ln() - sse / (2.0 * se * se),
            ),
            None => {
                let s2 = sse / dfq;
                if !s2.is_finite() || s2 <= 0.0 {
                    return None;
                }
                (
                    s2.sqrt(),
                    -0.5 * dfq * ((2.0 * std::f64::consts::PI * s2).ln() + 1.0),
                )
            }
        };
        let llf = llf_p + llf_q;
        if !llf.is_finite() {
            return None;
        }
        // Per-period c = W_c A_X with A_X = -(k_inf alpha + gamma)/n.
        let a_x_pp: Vec<f64> = self
            .maturities
            .iter()
            .map(|&mat| -(k_inf * alpha[mat] + gamma[mat]) / mat as f64)
            .collect();
        let c_pp = mat_vec(&self.w_c, &a_x_pp);
        Some(Eval {
            llf,
            k_inf,
            sigma_e,
            a_pc,
            b_pc,
            d,
            d_inv,
            c_pp,
            b_rec,
            alpha,
            gamma,
        })
    }
}

// ---------------------------------------------------------------------------
// Parameter packing: theta = [lambda_1, ln(gap_2..N), L~ (lower triangle,
// diagonal as logs)], with L = scale * L~ the Cholesky factor of Sigma_c.
// ---------------------------------------------------------------------------

#[allow(clippy::needless_range_loop)]
fn pack(lambda_q: &[f64], sigma_c: &[Vec<f64>], scale: f64) -> Option<Vec<f64>> {
    let n = lambda_q.len();
    let mut th = Vec::with_capacity(n + n * (n + 1) / 2);
    th.push(lambda_q[0]);
    for k in 1..n {
        let gap = lambda_q[k - 1] - lambda_q[k];
        if !gap.is_finite() || gap <= 0.0 {
            return None;
        }
        th.push(gap.ln());
    }
    let l = cholesky(sigma_c)?;
    for i in 0..n {
        for j in 0..=i {
            let v = l[i][j] / scale;
            th.push(if i == j { v.ln() } else { v });
        }
    }
    Some(th)
}

#[allow(clippy::needless_range_loop)]
fn unpack(th: &[f64], n: usize, scale: f64) -> (Vec<f64>, Matrix) {
    let mut lambda_q = vec![0.0; n];
    lambda_q[0] = th[0];
    for k in 1..n {
        lambda_q[k] = lambda_q[k - 1] - th[k].exp();
    }
    let mut l = zeros(n, n);
    let mut p = n;
    for i in 0..n {
        for j in 0..=i {
            l[i][j] = if i == j { th[p].exp() } else { th[p] } * scale;
            p += 1;
        }
    }
    let sigma_c = mat_mul(&l, &transpose(&l));
    (lambda_q, sigma_c)
}

// ---------------------------------------------------------------------------
// The fit
// ---------------------------------------------------------------------------

/// A fitted JSZ canonical Gaussian affine term-structure model.
///
/// Produced by [`fit_jsz`]. Matrix fields are row-major `Vec<Vec<f64>>`;
/// `T` is the number of dates, `M` the number of maturities, `N =
/// n_factors`. See the [module docs](self) for units: `lambda_q` and
/// `k_inf_q` per period, everything in the portfolio rotation in the
/// annualized units of `P_t = w y_t`, yields annualized decimal.
#[derive(Debug, Clone, PartialEq)]
pub struct JszFit {
    /// The input maturity grid (periods).
    pub maturities: Vec<usize>,
    /// The number of pricing factors / portfolios `N`.
    pub n_factors: usize,
    /// Compounding periods per year.
    pub periods_per_year: f64,
    /// The portfolio weights `W` (`N x M`) actually used: the principal-
    /// component loadings (orthonormal rows, largest entry positive) or the
    /// user's matrix.
    pub w: Matrix,
    /// The pricing factors `P_t = W y_t` (`T x N`, annualized).
    pub factors: Matrix,
    /// The ordered risk-neutral eigenvalues `lambda^Q` (per period).
    pub lambda_q: Vec<f64>,
    /// The canonical drift `k_inf^Q` of the first (most persistent) latent
    /// state, per period, in short-rate units.
    pub k_inf_q: f64,
    /// Maximum-likelihood innovation covariance `Sigma_P` of the portfolio
    /// VAR (`N x N`, annualized units).
    pub sigma: Matrix,
    /// Standard deviation of the iid pricing error on the `M - N` yield
    /// directions orthogonal to the portfolios (annualized-yield units).
    pub sigma_e: f64,
    /// P-measure VAR(1) intercept `mu_P` (`N`): OLS of `P_t` on `[1,
    /// P_{t-1}]`, exactly statsmodels `VAR(1)`.
    pub mu_p: Vec<f64>,
    /// P-measure VAR(1) feedback `Phi_P` (`N x N`, row `i` = equation `i`).
    pub phi_p: Matrix,
    /// OLS standard errors of `mu_p` (statsmodels `VAR` convention:
    /// classical, residual degrees of freedom `T - 1 - (N + 1)`).
    pub mu_p_se: Vec<f64>,
    /// OLS standard errors of `phi_p` (`N x N`).
    pub phi_p_se: Matrix,
    /// The OLS residual covariance `U'U / (T - 1)` (statsmodels
    /// `sigma_u_mle`); differs from `sigma` only through the convexity
    /// term's weak dependence on `Sigma_P`.
    pub sigma_ols: Matrix,
    /// Risk-neutral VAR(1) intercept in the portfolio rotation, annualized:
    /// `P_{t+1} = k0_q_p + k1_q_p P_t + ...` under `Q`.
    pub k0_q_p: Vec<f64>,
    /// Risk-neutral feedback in the portfolio rotation (`N x N`).
    pub k1_q_p: Matrix,
    /// Constant market price of risk in ACM's units: `mu_p - k0_q_p`.
    pub lambda0: Vec<f64>,
    /// State-dependent market price of risk: `phi_p - k1_q_p` (`N x N`).
    pub lambda1: Matrix,
    /// Yield intercepts in the portfolio rotation (`M`, annualized):
    /// `fitted_t = a_p + b_p P_t`.
    pub a_p: Vec<f64>,
    /// Yield loadings on the portfolios (`M x N`); `W b_p = I` and
    /// `W a_p = 0`, so the portfolios are priced exactly.
    pub b_p: Matrix,
    /// Yield intercepts for the latent canonical state (`M`, annualized).
    pub a_x: Vec<f64>,
    /// Yield loadings on the literal canonical latent state (`M x N`):
    /// `b_x[i][k] = (1/n_i) sum_{j<n_i} lambda_k^j` for distinct eigenvalues.
    pub b_x: Matrix,
    /// Model-implied yields (`T x M`, annualized decimal).
    pub fitted: Matrix,
    /// Risk-neutral yields (`T x M`): the recursion re-run with the
    /// P-measure VAR — the expected-short-rate (plus convexity) component,
    /// as in [`crate::acm_term_premium`].
    pub risk_neutral: Matrix,
    /// The term premium `fitted - risk_neutral` (`T x M`).
    pub term_premium: Matrix,
    /// Root-mean-square pricing error per maturity (`M`, annualized).
    pub rmse: Vec<f64>,
    /// The maximized log-likelihood of the yield panel (conditional on the
    /// first cross-section's portfolios): the JSZ factorization `llk_P +
    /// llk_Q` evaluated in an orthonormal portfolio basis, which makes it
    /// invariant to the basis `W` of the portfolio space.
    pub llf: f64,
    /// Whether the final optimizer stage met its convergence test.
    pub converged: bool,
    /// Total optimizer iterations (quasi-Newton stage plus the simplex
    /// polish).
    pub n_iter: usize,
}

/// Validate the panel and the factor count; returns `(T, M)`.
fn check_panel(
    yields: &[Vec<f64>],
    maturities: &[usize],
    n_factors: usize,
    periods_per_year: f64,
) -> Result<(usize, usize), TermStructureError> {
    check_maturities(maturities)?;
    let m = maturities.len();
    if n_factors == 0 || n_factors >= m {
        return Err(TermStructureError::InvalidFactorCount {
            requested: n_factors,
            max: m.saturating_sub(1),
        });
    }
    check_ppy(periods_per_year)?;
    let t = yields.len();
    let needed = 2 * n_factors + 3;
    if t < needed {
        return Err(TermStructureError::PanelTooShort {
            what: "JSZ yield panel (the portfolio VAR(1) needs residual degrees \
                   of freedom)",
            dates: t,
            needed,
        });
    }
    for (i, row) in yields.iter().enumerate() {
        if row.len() != m {
            return Err(TermStructureError::DimensionMismatch {
                what: "JSZ yield panel row vs maturities",
                expected: m,
                got: row.len(),
            });
        }
        for (j, &y) in row.iter().enumerate() {
            if !y.is_finite() {
                return Err(TermStructureError::NonFinite {
                    what: "JSZ yield panel",
                    index: i * m + j,
                    value: y,
                });
            }
        }
    }
    Ok((t, m))
}

/// Resolve the portfolio weights: validate a user matrix or build the PCA
/// default.
fn resolve_weights(
    yields: &[Vec<f64>],
    n_factors: usize,
    m: usize,
    w: Option<&[Vec<f64>]>,
) -> Result<Matrix, TermStructureError> {
    match w {
        None => pca_weights(yields, n_factors),
        Some(w) => {
            if w.len() != n_factors {
                return Err(TermStructureError::InvalidWeights {
                    reason: "wrong number of rows (expected n_factors)",
                    rows: w.len(),
                    cols: m,
                });
            }
            for row in w {
                if row.len() != m {
                    return Err(TermStructureError::InvalidWeights {
                        reason: "wrong number of columns (expected n_maturities)",
                        rows: n_factors,
                        cols: row.len(),
                    });
                }
                if row.iter().any(|v| !v.is_finite()) {
                    return Err(TermStructureError::InvalidWeights {
                        reason: "a non-finite entry",
                        rows: n_factors,
                        cols: m,
                    });
                }
            }
            Ok(w.to_vec())
        }
    }
}

/// Build the precomputed problem for a validated panel and weight matrix.
fn build_problem<'a>(
    yields: &'a [Vec<f64>],
    maturities: &'a [usize],
    n_factors: usize,
    ppy: f64,
    w: &[Vec<f64>],
) -> Result<Problem<'a>, TermStructureError> {
    let t = yields.len();
    let m = maturities.len();
    let w_c = canonical_basis(w, yields)?;
    let p_c: Matrix = yields.iter().map(|row| mat_vec(&w_c, row)).collect();
    let y_mean: Vec<f64> = (0..m)
        .map(|j| yields.iter().map(|r| r[j]).sum::<f64>() / t as f64)
        .collect();
    let var_c = p_var(&p_c)?;
    Ok(Problem {
        yields,
        maturities,
        ppy,
        t,
        m,
        n: n_factors,
        n_max: maturities[m - 1],
        w_c,
        p_c,
        y_mean,
        var_c,
    })
}

/// JSZ's recommended start for `lambda^Q`: the eigenvalues of the OLS
/// feedback matrix (real parts, sorted descending, separated by at least
/// `1e-3`).
fn starting_lambda(phi: &[Vec<f64>]) -> Result<Vec<f64>, TermStructureError> {
    let n = phi.len();
    let a = Mat::from_fn(n, n, |i, j| phi[i][j]);
    let eigs = a
        .eigenvalues()
        .map_err(|_| TermStructureError::OptimizationFailed {
            reason: "the eigenvalues of the OLS feedback matrix (the JSZ starting \
                     values for lambda_q) could not be computed",
        })?;
    let mut lam: Vec<f64> = eigs.iter().map(|c| c.re).collect();
    lam.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    for v in lam.iter_mut() {
        *v = v.clamp(-0.5, 1.05);
    }
    for k in 1..n {
        if lam[k] > lam[k - 1] - 1e-3 {
            lam[k] = lam[k - 1] - 1e-3;
        }
    }
    Ok(lam)
}

/// The generic affine recursion in the portfolio rotation (per-period
/// units): `A_{n+1} = A_n + k0' B_n + 1/2 B_n' Sigma B_n - rho0`,
/// `B_{n+1} = k1' B_n - rho1`, returning the per-maturity yield
/// coefficients (annualized intercepts).
fn rotated_yield_coefficients(
    k0: &[f64],
    k1: &[Vec<f64>],
    sigma_pp: &[Vec<f64>],
    rho0: f64,
    rho1: &[f64],
    maturities: &[usize],
    ppy: f64,
) -> (Vec<f64>, Matrix) {
    let n = k0.len();
    let n_max = maturities[maturities.len() - 1];
    let k1t = transpose(k1);
    let mut a = vec![0.0f64; n_max + 1];
    let mut b = zeros(n_max + 1, n);
    for mth in 0..n_max {
        a[mth + 1] = a[mth] + dot(k0, &b[mth]) + 0.5 * quad_form(sigma_pp, &b[mth]) - rho0;
        let kb = mat_vec(&k1t, &b[mth]);
        for k in 0..n {
            b[mth + 1][k] = kb[k] - rho1[k];
        }
    }
    let ay: Vec<f64> = maturities
        .iter()
        .map(|&mat| -a[mat] / mat as f64 * ppy)
        .collect();
    let by: Matrix = maturities
        .iter()
        .map(|&mat| b[mat].iter().map(|&v| -v / mat as f64).collect())
        .collect();
    (ay, by)
}

/// Estimate the JSZ (2011) canonical Gaussian affine term-structure model
/// by maximum likelihood.
///
/// The estimator of the [module docs](self): portfolios `P_t = W y_t` priced
/// exactly, the remaining `M - N` yield directions with iid error; the
/// P-measure VAR(1) of the portfolios concentrated out by OLS; `k_inf^Q`
/// and `sigma_e` profiled analytically; a quasi-Newton search over
/// `(lambda^Q, chol(Sigma_P))` started from JSZ's recommendation (the
/// eigenvalues of the OLS feedback matrix and the OLS residual covariance),
/// polished by an adaptive Nelder-Mead simplex, the better of the two
/// reported. The optimization runs in a canonical orthonormal basis of the
/// portfolio space and is mapped back to `W` exactly, so the fit depends on
/// `W` only through its row space.
///
/// # Arguments
///
/// - `yields`: `T x M` panel of **annualized, continuously-compounded
///   zero-coupon log yields in decimal** (0.05 = 5%), one row per date.
/// - `maturities`: the `M` maturities in integer **periods**, strictly
///   ascending (need not contain 1).
/// - `n_factors`: the number of pricing factors `N`, `1 <= N < M` (three
///   is JSZ's baseline).
/// - `periods_per_year`: 12 for monthly, 4 for quarterly.
/// - `w`: optional `N x M` portfolio weights; `None` selects the first `N`
///   principal-component loadings of the panel.
///
/// # Errors
///
/// [`TermStructureError::EmptyMaturities`], [`TermStructureError::InvalidMaturity`]
/// (a zero maturity), [`TermStructureError::MaturitiesNotAscending`]
/// (unsorted or duplicated), [`TermStructureError::InvalidFactorCount`]
/// (`n_factors = 0` or `>= M`), [`TermStructureError::InvalidPeriodsPerYear`],
/// [`TermStructureError::PanelTooShort`] (fewer than `2 N + 3` dates),
/// [`TermStructureError::DimensionMismatch`] (a ragged panel row),
/// [`TermStructureError::NonFinite`] (NaN/inf yields),
/// [`TermStructureError::InvalidWeights`] (a malformed or rank-deficient
/// `w`), [`TermStructureError::SingularDesign`] (a degenerate panel — e.g.
/// constant yields — whose portfolios carry no variation), and
/// [`TermStructureError::OptimizationFailed`] if the likelihood cannot be
/// evaluated at the starting point.
///
/// # Example
///
/// ```
/// use tsecon_termstructure::fit_jsz;
///
/// // A toy monthly panel built from two persistent curve factors.
/// let maturities = [1usize, 3, 6, 12, 24, 36, 60, 84, 120];
/// let mut level = 0.04f64;
/// let mut slope = -0.01f64;
/// let yields: Vec<Vec<f64>> = (0..120)
///     .map(|t| {
///         level += 0.0004 * ((t as f64) * 0.7).sin() - 0.02 * (level - 0.04);
///         slope += 0.0005 * ((t as f64) * 1.3).cos() - 0.10 * (slope + 0.01);
///         maturities
///             .iter()
///             .map(|&n| {
///                 let x = n as f64 / 12.0;
///                 level + slope * (1.0 - (-0.5 * x).exp()) / (0.5 * x)
///                     + 1e-5 * ((t * 7 + n) as f64).sin()
///             })
///             .collect()
///     })
///     .collect();
///
/// let fit = fit_jsz(&yields, &maturities, 2, 12.0, None).unwrap();
/// assert_eq!(fit.lambda_q.len(), 2);
/// assert!(fit.lambda_q[0] >= fit.lambda_q[1]);
/// // The two portfolios are priced exactly: W * fitted = W * observed.
/// for t in 0..yields.len() {
///     for k in 0..2 {
///         let wy: f64 = (0..9).map(|j| fit.w[k][j] * yields[t][j]).sum();
///         let wf: f64 = (0..9).map(|j| fit.w[k][j] * fit.fitted[t][j]).sum();
///         assert!((wy - wf).abs() < 1e-10);
///     }
/// }
/// // fitted = risk_neutral + term_premium exactly.
/// assert!((fit.fitted[5][8] - fit.risk_neutral[5][8] - fit.term_premium[5][8]).abs() < 1e-14);
/// ```
pub fn fit_jsz(
    yields: &[Vec<f64>],
    maturities: &[usize],
    n_factors: usize,
    periods_per_year: f64,
    w: Option<&[Vec<f64>]>,
) -> Result<JszFit, TermStructureError> {
    let (t, m) = check_panel(yields, maturities, n_factors, periods_per_year)?;
    let n = n_factors;
    let ppy = periods_per_year;
    let w_user = resolve_weights(yields, n, m, w)?;
    let prob = build_problem(yields, maturities, n, ppy, &w_user)?;

    // --- starting values --------------------------------------------------------
    let lambda0 = starting_lambda(&prob.var_c.phi)?;
    let t_v = (t - 1) as f64;
    let sigma_ols_c: Matrix = prob
        .var_c
        .s
        .iter()
        .map(|r| r.iter().map(|&v| v / t_v).collect())
        .collect();
    let scale = ((0..n).map(|i| sigma_ols_c[i][i]).sum::<f64>() / n as f64).sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(TermStructureError::SingularDesign {
            what: "JSZ portfolio VAR(1) (the portfolios carry no innovation \
                   variance — is the yield panel constant or exactly affine in \
                   its lags?)",
        });
    }
    let theta0 = pack(&lambda0, &sigma_ols_c, scale).ok_or(TermStructureError::SingularDesign {
        what: "JSZ portfolio VAR(1) residual covariance (not positive definite)",
    })?;
    let llf0 = prob
        .evaluate(&lambda0, &sigma_ols_c, None)
        .map(|e| e.llf)
        .ok_or(TermStructureError::OptimizationFailed {
            reason: "the JSZ likelihood could not be evaluated at the starting values \
                     (OLS eigenvalues and residual covariance)",
        })?;

    // --- the concentrated objective, centered at the start ---------------------
    let mut objective = FnObjective::new(|th: &[f64]| {
        let (lam, sig) = unpack(th, n, scale);
        match prob.evaluate(&lam, &sig, None) {
            Some(e) => -(e.llf - llf0),
            None => f64::INFINITY,
        }
    });
    let stage1: Option<OptimizeResult> = minimize(&mut objective, &theta0, &Method::bfgs())
        .ok()
        .filter(|r| r.f.is_finite());
    let polish_from: Vec<f64> = stage1
        .as_ref()
        .map_or(theta0.as_slice(), |r| r.x.as_slice())
        .to_vec();
    let nm = Method::NelderMead(NelderMeadOptions {
        restarts: 1,
        ..NelderMeadOptions::default()
    });
    let stage2: Option<OptimizeResult> = minimize(&mut objective, &polish_from, &nm)
        .ok()
        .filter(|r| r.f.is_finite());
    let n_iter =
        stage1.as_ref().map_or(0, |r| r.iterations) + stage2.as_ref().map_or(0, |r| r.iterations);
    let best = match (stage1, stage2) {
        (Some(a), Some(b)) => {
            if b.f <= a.f {
                b
            } else {
                a
            }
        }
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => {
            return Err(TermStructureError::OptimizationFailed {
                reason: "neither the quasi-Newton search nor the simplex polish \
                         produced a finite JSZ likelihood",
            })
        }
    };
    let converged = best.converged;
    let (lambda_q, sigma_c) = unpack(&best.x, n, scale);
    let ev =
        prob.evaluate(&lambda_q, &sigma_c, None)
            .ok_or(TermStructureError::OptimizationFailed {
                reason: "the JSZ likelihood is not finite at the optimizer's final point",
            })?;

    // --- objects in the canonical rotation (per period) --------------------------
    // Q dynamics of P_c: K1_c = D K1_X D^{-1}; K0_c = c + D K0_X - K1_c c.
    let k1_x = k1_matrix(&lambda_q, true);
    let k1_c = mat_mul(&mat_mul(&ev.d, &k1_x), &ev.d_inv);
    let mut k0_x = vec![0.0f64; n];
    k0_x[0] = ev.k_inf;
    let dk0 = mat_vec(&ev.d, &k0_x);
    let k1c_c = mat_vec(&k1_c, &ev.c_pp);
    let k0_c_pp: Vec<f64> = (0..n).map(|i| ev.c_pp[i] + dk0[i] - k1c_c[i]).collect();
    // Short rate in the P_c rotation: r = rho0 + rho1' P_pp.
    let iota = vec![1.0f64; n];
    let rho1 = mat_vec(&transpose(&ev.d_inv), &iota);
    let rho0 = -dot(&rho1, &ev.c_pp);
    let sigma_c_pp: Matrix = sigma_c
        .iter()
        .map(|r| r.iter().map(|&v| v / (ppy * ppy)).collect())
        .collect();
    // Risk-neutral yields: the same recursion under the P-measure VAR.
    let mu_c_pp: Vec<f64> = prob.var_c.mu.iter().map(|&v| v / ppy).collect();
    let (a_rn, b_rn) = rotated_yield_coefficients(
        &mu_c_pp,
        &prob.var_c.phi,
        &sigma_c_pp,
        rho0,
        &rho1,
        maturities,
        ppy,
    );

    // --- fitted / risk-neutral / term premium / rmse ---------------------------
    let mut fitted = zeros(t, m);
    let mut risk_neutral = zeros(t, m);
    let mut term_premium = zeros(t, m);
    let mut rmse = vec![0.0f64; m];
    for (i, (row, p_row)) in yields.iter().zip(prob.p_c.iter()).enumerate() {
        for j in 0..m {
            let f = ev.a_pc[j] + dot(&ev.b_pc[j], p_row);
            let rn = a_rn[j] + dot(&b_rn[j], p_row);
            fitted[i][j] = f;
            risk_neutral[i][j] = rn;
            term_premium[i][j] = f - rn;
            rmse[j] += (row[j] - f) * (row[j] - f);
        }
    }
    for r in rmse.iter_mut() {
        *r = (*r / t as f64).sqrt();
    }

    // --- map to the user's rotation: P_user = G P_c, G = W W_c' -------------------
    let g = mat_mul(&w_user, &transpose(&prob.w_c));
    let g_inv = invert(&g).ok_or(TermStructureError::InvalidWeights {
        reason: "the rows of w are linearly dependent (rank-deficient)",
        rows: n,
        cols: m,
    })?;
    let factors: Matrix = yields.iter().map(|row| mat_vec(&w_user, row)).collect();
    let var_u = p_var(&factors)?;
    let sigma = mat_mul(&mat_mul(&g, &sigma_c), &transpose(&g));
    let sigma_ols: Matrix = var_u
        .s
        .iter()
        .map(|r| r.iter().map(|&v| v / t_v).collect())
        .collect();
    let k0_q_p: Vec<f64> = mat_vec(&g, &k0_c_pp).iter().map(|&v| v * ppy).collect();
    let k1_q_p = mat_mul(&mat_mul(&g, &k1_c), &g_inv);
    let lambda0_p: Vec<f64> = (0..n).map(|i| var_u.mu[i] - k0_q_p[i]).collect();
    let lambda1_p: Matrix = (0..n)
        .map(|i| (0..n).map(|j| var_u.phi[i][j] - k1_q_p[i][j]).collect())
        .collect();
    let b_p = mat_mul(&ev.b_pc, &g_inv);
    let a_p = ev.a_pc.clone();

    // --- literal canonical latent loadings -----------------------------------------
    let (b_lit, _) = loading_recursion(&lambda_q, false, prob.n_max);
    let b_x: Matrix = maturities
        .iter()
        .map(|&mat| b_lit[mat].iter().map(|&v| -v / mat as f64).collect())
        .collect();
    let a_x: Vec<f64> = maturities
        .iter()
        .map(|&mat| -(ev.k_inf * ev.alpha[mat] + ev.gamma[mat]) / mat as f64 * ppy)
        .collect();
    let _ = &ev.b_rec;

    Ok(JszFit {
        maturities: maturities.to_vec(),
        n_factors: n,
        periods_per_year: ppy,
        w: w_user,
        factors,
        lambda_q,
        k_inf_q: ev.k_inf,
        sigma,
        sigma_e: ev.sigma_e,
        mu_p: var_u.mu,
        phi_p: var_u.phi,
        mu_p_se: var_u.mu_se,
        phi_p_se: var_u.phi_se,
        sigma_ols,
        k0_q_p,
        k1_q_p,
        lambda0: lambda0_p,
        lambda1: lambda1_p,
        a_p,
        b_p,
        a_x,
        b_x,
        fitted,
        risk_neutral,
        term_premium,
        rmse,
        llf: ev.llf,
        converged,
        n_iter,
    })
}

/// The JSZ log-likelihood of a yield panel at **given** `Q` parameters,
/// with the P-measure VAR concentrated out by OLS.
///
/// Evaluates `llk_P + llk_Q` of the [module docs](self) at `lambda_q`
/// (per period, ordered), `k_inf_q` (per period), `sigma_p` (the annualized
/// `N x N` portfolio-innovation covariance in the rotation defined by `w`)
/// and `sigma_e` (annualized), with `(mu_P, Phi_P)` at their OLS values.
/// Like [`JszFit::llf`] it is the log-density of the yield panel
/// conditional on the first cross-section's portfolios, invariant to the
/// basis of the portfolio space; for an orthonormal `w` it equals the JSZ
/// replication code's `llkP + llkQ` convention term for term. Useful for
/// profile-likelihood plots and as the validation target of the fit.
///
/// # Errors
///
/// The validation errors of [`fit_jsz`] and [`jsz_loadings`], plus
/// [`TermStructureError::NonFinite`] for a non-positive or non-finite
/// `sigma_e`, [`TermStructureError::DimensionMismatch`] for a `sigma_p`
/// that is not `N x N`, and [`TermStructureError::SingularDesign`] if
/// `sigma_p` is not positive definite or the rotation `W B_X` is singular
/// at these eigenvalues.
#[allow(clippy::too_many_arguments)]
pub fn jsz_loglik(
    yields: &[Vec<f64>],
    maturities: &[usize],
    n_factors: usize,
    periods_per_year: f64,
    w: Option<&[Vec<f64>]>,
    lambda_q: &[f64],
    k_inf_q: f64,
    sigma_p: &[Vec<f64>],
    sigma_e: f64,
) -> Result<f64, TermStructureError> {
    let (_t, m) = check_panel(yields, maturities, n_factors, periods_per_year)?;
    let n = n_factors;
    if lambda_q.len() != n {
        return Err(TermStructureError::DimensionMismatch {
            what: "JSZ lambda_q vs n_factors",
            expected: n,
            got: lambda_q.len(),
        });
    }
    check_lambda_q(lambda_q)?;
    if !k_inf_q.is_finite() {
        return Err(TermStructureError::NonFinite {
            what: "k_inf_q",
            index: 0,
            value: k_inf_q,
        });
    }
    check_square(sigma_p, n, "JSZ sigma_p (must be n_factors x n_factors)")?;
    if !sigma_e.is_finite() || sigma_e <= 0.0 {
        return Err(TermStructureError::NonFinite {
            what: "sigma_e (must be a finite positive pricing-error standard deviation)",
            index: 0,
            value: sigma_e,
        });
    }
    let w_user = resolve_weights(yields, n, m, w)?;
    let prob = build_problem(yields, maturities, n, periods_per_year, &w_user)?;
    // Sigma_c = G^{-1} Sigma_user G^{-T}, G = W W_c'.
    let g = mat_mul(&w_user, &transpose(&prob.w_c));
    let g_inv = invert(&g).ok_or(TermStructureError::InvalidWeights {
        reason: "the rows of w are linearly dependent (rank-deficient)",
        rows: n,
        cols: m,
    })?;
    let sigma_c = mat_mul(&mat_mul(&g_inv, sigma_p), &transpose(&g_inv));
    prob.evaluate(lambda_q, &sigma_c, Some((k_inf_q, sigma_e)))
        .map(|e| e.llf)
        .ok_or(TermStructureError::SingularDesign {
            what: "JSZ likelihood evaluation (sigma_p not positive definite, or the \
                   rotation W B_X singular at these eigenvalues)",
        })
}
