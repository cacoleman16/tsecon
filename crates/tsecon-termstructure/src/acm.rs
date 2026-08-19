//! The ACM regression-based term premium (Adrian, Crump & Moench 2013).
//!
//! Adrian, Crump & Moench (2013, *J. Financial Economics*; FRBNY Staff Report
//! 340) estimate a Gaussian affine term-structure model **entirely by linear
//! regressions** — no numerical likelihood maximization, no Kalman filter —
//! which is why their decomposition of long yields into expected short rates
//! and a term premium became the practitioner standard (the NY Fed publishes
//! the resulting "ACM term premium" series daily). This module implements the
//! full three-step estimator and the affine pricing recursions.
//!
//! ## The model
//!
//! `K` pricing factors `X_t` (principal components of the yield panel) follow
//! a VAR(1) under the physical measure,
//!
//! ```text
//! X_{t+1} = mu + Phi X_t + v_{t+1},        v ~ (0, Sigma),
//! ```
//!
//! the one-period short rate is affine, `r_t = delta0 + delta1' X_t`, and the
//! market prices of risk are affine in the factors,
//! `lambda_t = lambda0 + lambda1 X_t`. No-arbitrage then makes every log bond
//! price affine, `ln P_t^(n) = A_n + B_n' X_t`.
//!
//! ## The three regression steps
//!
//! 1. **Factor VAR.** OLS of `X_{t+1}` on `[1, X_t]` gives `mu`, `Phi`, the
//!    innovations `v-hat`, and `Sigma = v'v / (T - 2)` (the innovations have
//!    exact zero mean because the VAR includes a constant).
//! 2. **Excess-return regressions.** With per-period log prices
//!    `p_t(n) = -n y_t(n) / periods_per_year` and short rate `r_t = -p_t(1)`,
//!    the one-period holding excess return of an `n`-period bond is
//!
//!    ```text
//!    rx_{t+1}(n) = p_{t+1}(n-1) - p_t(n) - r_t.
//!    ```
//!
//!    Each is regressed on a constant, the **lagged** factors, and the
//!    **contemporaneous** VAR innovations:
//!
//!    ```text
//!    rx_{t+1} = a + c' X_t + beta' v_{t+1} + e_{t+1},
//!    ```
//!
//!    giving `a` (N), `c` (N x K), `beta` (N x K) and the pooled pricing-error
//!    variance `sigma^2 = sum(e^2) / (N (T-1))`.
//! 3. **Prices of risk.** Let `B*` (N x K^2) have rows
//!    `vec(beta_i beta_i')'`. Expected-return restrictions of the model imply
//!    the cross-sectional regressions
//!
//!    ```text
//!    lambda0 = (beta'beta)^{-1} beta' (a + 1/2 (B* vec(Sigma) + sigma^2 1_N)),
//!    lambda1 = (beta'beta)^{-1} beta' c,
//!    ```
//!
//!    — the `1/2(...)` term is the Jensen convexity adjustment that converts
//!    average log excess returns into risk compensation.
//!
//! ## The affine recursions
//!
//! With `delta0`, `delta1` from an OLS of `r_t` on `[1, X_t]`, log-price
//! coefficients follow (the one-period bond is priced without error, so the
//! recursion is seeded at `n = 1` and the convexity terms enter from `n = 2` —
//! exactly the NY Fed production code's form):
//!
//! ```text
//! A_1 = -delta0,   B_1 = -delta1,
//! A_n = A_{n-1} + B_{n-1}'(mu - lambda0)
//!               + 1/2 (B_{n-1}' Sigma B_{n-1} + sigma^2) - delta0,
//! B_n = (Phi - lambda1)' B_{n-1} - delta1.
//! ```
//!
//! **Risk-neutral** coefficients use the same recursion with
//! `lambda0 = 0, lambda1 = 0` (convexity kept): they price the curve as if
//! investors demanded no compensation for duration risk, so the risk-neutral
//! yield is the average expected future short rate (plus convexity). Fitted
//! and risk-neutral yields (annualized decimal) and the term premium are
//!
//! ```text
//! y-hat_t(n)  = -(A_n + B_n' X_t) * periods_per_year / n,
//! y-rn_t(n)   = -(A_n^RN + B_n^RN' X_t) * periods_per_year / n,
//! tp_t(n)     = y-hat_t(n) - y-rn_t(n).
//! ```
//!
//! ## Reading the decomposition (and common traps)
//!
//! - **Units are load-bearing.** Yields must be *annualized,
//!   continuously-compounded zero-coupon log yields in decimal* (0.05, not
//!   5.0). The convexity terms are quadratic while everything else is linear,
//!   so feeding percent instead of decimal does **not** just rescale the
//!   answer — it misprices the Jensen terms by a factor of 100. Divide
//!   percent yields by 100 first.
//! - **Maturities are integer periods** (months for monthly data) and the
//!   grid must contain 1 (the short rate) and, for every excess-return
//!   maturity `n`, its neighbour `n - 1`. ACM interpolate the GSW curve to
//!   monthly maturities 1..120 first; a Nelson-Siegel fit
//!   ([`crate::fit_nelson_siegel`]) is one way to do that interpolation.
//! - **The term premium is estimation-sample sensitive.** The prices of risk
//!   are estimated means of excess returns; re-estimating on a subsample
//!   moves the *level* of the premium by tens of basis points while the
//!   *shape* is stable (on the 1961-2014 GSW panel this implementation
//!   matches the NY Fed's published 10-year ACM premium with correlation
//!   0.985 and RMSE 0.31pp; on 1983-2014 alone the correlation is unchanged
//!   but the level sits ~1.1pp higher). Compare premia only across models
//!   estimated on the same sample.
//! - **AFNS vs ACM.** The arbitrage-free Nelson-Siegel ([`crate::fit_afns`])
//!   *restricts* the factor loadings to the Nelson-Siegel shapes and adds a
//!   deterministic convexity adjustment — use it to fit/interpolate a curve
//!   consistently with no-arbitrage. ACM leaves the loadings free (estimated
//!   PCs), prices the *time series* of returns, and is the tool when the
//!   object of interest is the **term premium** — the split of a long yield
//!   into expected short rates and risk compensation.
//!
//! ## References
//!
//! - Adrian, T., Crump, R. K., & Moench, E. (2013). "Pricing the Term
//!   Structure with Linear Regressions." *Journal of Financial Economics*,
//!   110(1), 110-138. (FRBNY Staff Report 340.)
//! - Gürkaynak, R. S., Sack, B., & Wright, J. H. (2007). "The U.S. Treasury
//!   Yield Curve: 1961 to the Present." *Journal of Monetary Economics*,
//!   54(8), 2291-2304. (The zero-coupon panel the NY Fed series is built on.)

use crate::error::TermStructureError;
use crate::fit::{centered_rsquared, map_ols_err};
use tsecon_hac::ols;
use tsecon_linalg::faer::Mat;

/// A fitted ACM (Adrian-Crump-Moench 2013) regression-based affine term
/// structure: factors, VAR dynamics, prices of risk, affine pricing
/// coefficients, and the fitted / risk-neutral / term-premium decomposition.
///
/// Produced by [`acm_term_premium`]. Matrix fields are row-major
/// `Vec<Vec<f64>>`; `T` is the number of dates, `M` the number of input
/// maturities, `K = n_factors`, `N` the number of excess-return maturities.
/// Yields are annualized decimals, matching the input units.
#[derive(Debug, Clone, PartialEq)]
pub struct AcmFit {
    /// The input maturity grid, in periods (e.g. months).
    pub maturities: Vec<usize>,
    /// The number of principal-component pricing factors `K`.
    pub n_factors: usize,
    /// Compounding periods per year used to convert annualized yields to the
    /// per-period log yields the recursions price.
    pub periods_per_year: f64,
    /// The pricing factors `X_t` (`T x K`): principal components of the
    /// demeaned yield panel, scaled to unit sample standard deviation, signs
    /// fixed so each loading column sums positive. All model outputs are
    /// invariant to this (or any other invertible linear) normalization.
    pub factors: Vec<Vec<f64>>,
    /// The factor loadings (`M x K`): `factors = (yields - mean) *
    /// factor_loadings` column by column.
    pub factor_loadings: Vec<Vec<f64>>,
    /// VAR(1) intercept `mu` (`K`).
    pub mu: Vec<f64>,
    /// VAR(1) transition `Phi` (`K x K`, row `i` = equation `i`).
    pub phi: Vec<Vec<f64>>,
    /// Innovation covariance `Sigma = v'v / (T - 2)` (`K x K`).
    pub sigma: Vec<Vec<f64>>,
    /// The excess-return maturities `n` (each has `n - 1` in the grid).
    pub rx_maturities: Vec<usize>,
    /// Excess-return regression intercepts `a` (`N`).
    pub rx_a: Vec<f64>,
    /// Coefficients `beta` on the contemporaneous VAR innovations (`N x K`):
    /// the return exposures to factor risk that identify the prices of risk.
    pub rx_beta: Vec<Vec<f64>>,
    /// Coefficients `c` on the lagged factors (`N x K`): the predictable
    /// component of excess returns.
    pub rx_c: Vec<Vec<f64>>,
    /// Pooled return-pricing-error variance
    /// `sigma^2 = sum(e^2) / (N (T-1))`.
    pub sigma2: f64,
    /// Constant price of risk `lambda0` (`K`), in per-period units.
    pub lambda0: Vec<f64>,
    /// State-dependent price of risk `lambda1` (`K x K`).
    pub lambda1: Vec<Vec<f64>>,
    /// Short-rate intercept `delta0` (per-period decimal).
    pub delta0: f64,
    /// Short-rate factor loadings `delta1` (`K`).
    pub delta1: Vec<f64>,
    /// Log-price intercepts `A_n` (`n_max`, entry `n - 1` is maturity `n`).
    pub price_a: Vec<f64>,
    /// Log-price factor loadings `B_n` (`n_max x K`).
    pub price_b: Vec<Vec<f64>>,
    /// Risk-neutral log-price intercepts `A_n^RN` (`n_max`).
    pub price_a_rn: Vec<f64>,
    /// Risk-neutral log-price loadings `B_n^RN` (`n_max x K`).
    pub price_b_rn: Vec<Vec<f64>>,
    /// Model-implied (fitted) yields (`T x M`, annualized decimal).
    pub fitted: Vec<Vec<f64>>,
    /// Risk-neutral yields (`T x M`, annualized decimal): the expected-short-
    /// rate (plus convexity) component of the curve.
    pub risk_neutral: Vec<Vec<f64>>,
    /// The term premium (`T x M`, annualized decimal):
    /// `fitted - risk_neutral`.
    pub term_premium: Vec<Vec<f64>>,
    /// Centered R^2 of each VAR equation (`K`).
    pub var_rsquared: Vec<f64>,
    /// Centered R^2 of each excess-return regression (`N`); ACM report high
    /// values because the contemporaneous innovations absorb most return
    /// variation.
    pub rx_rsquared: Vec<f64>,
    /// Centered R^2 of the short-rate equation.
    pub short_rate_rsquared: f64,
    /// Centered R^2 of the fitted vs observed yields per maturity (`M`) — the
    /// cross-sectional fit diagnostic.
    pub yield_rsquared: Vec<f64>,
}

/// Column index of each maturity, or the position of maturity `n` in the grid.
fn position(maturities: &[usize], n: usize) -> Option<usize> {
    maturities.binary_search(&n).ok()
}

/// Validate the maturity grid: non-empty, strictly ascending positive
/// integers, starting at the one-period maturity.
fn check_acm_maturities(maturities: &[usize]) -> Result<(), TermStructureError> {
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
    if maturities[0] != 1 {
        return Err(TermStructureError::MissingShortMaturity {
            shortest: maturities[0],
        });
    }
    Ok(())
}

/// Multiply the transpose of a `K x K` matrix (row-major) by a `K`-vector:
/// `out = m' v`.
fn mat_t_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let k = v.len();
    (0..k)
        .map(|j| (0..k).map(|i| m[i][j] * v[i]).sum())
        .collect()
}

/// Quadratic form `v' m v` for a `K x K` row-major matrix.
fn quad_form(m: &[Vec<f64>], v: &[f64]) -> f64 {
    let k = v.len();
    let mut acc = 0.0;
    for i in 0..k {
        for j in 0..k {
            acc += v[i] * m[i][j] * v[j];
        }
    }
    acc
}

/// The affine log-price recursion (module docs): seeded at the one-period
/// bond, convexity from `n = 2`. `lambda0 = None` / `lambda1 = None` runs the
/// risk-neutral recursion.
#[allow(clippy::too_many_arguments)]
fn price_recursion(
    n_max: usize,
    k: usize,
    mu: &[f64],
    phi: &[Vec<f64>],
    sigma: &[Vec<f64>],
    sigma2: f64,
    delta0: f64,
    delta1: &[f64],
    lambda0: Option<&[f64]>,
    lambda1: Option<&[Vec<f64>]>,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Risk-adjusted drift mu - lambda0 and transition Phi - lambda1.
    let drift: Vec<f64> = match lambda0 {
        Some(l0) => mu.iter().zip(l0.iter()).map(|(&m, &l)| m - l).collect(),
        None => mu.to_vec(),
    };
    let trans: Vec<Vec<f64>> = match lambda1 {
        Some(l1) => (0..k)
            .map(|i| (0..k).map(|j| phi[i][j] - l1[i][j]).collect())
            .collect(),
        None => phi.to_vec(),
    };

    let mut a = vec![0.0f64; n_max];
    let mut b = vec![vec![0.0f64; k]; n_max];
    a[0] = -delta0;
    for (bj, &dj) in b[0].iter_mut().zip(delta1.iter()) {
        *bj = -dj;
    }
    for n in 1..n_max {
        let b_prev = b[n - 1].clone();
        let drift_term: f64 = b_prev.iter().zip(drift.iter()).map(|(&x, &d)| x * d).sum();
        a[n] = a[n - 1] + drift_term + 0.5 * (quad_form(sigma, &b_prev) + sigma2) - delta0;
        let tb = mat_t_vec(&trans, &b_prev);
        for j in 0..k {
            b[n][j] = tb[j] - delta1[j];
        }
    }
    (a, b)
}

/// Estimate the ACM (Adrian-Crump-Moench 2013) regression-based affine term
/// structure and decompose fitted yields into risk-neutral yields and the
/// term premium.
///
/// The estimator is the exact three-step pipeline in the [module docs](self):
/// principal-component factors, a factor VAR(1), excess-return regressions on
/// lagged factors and contemporaneous innovations, the convexity-adjusted
/// `lambda0` / `lambda1` cross-sectional OLS, and the affine log-price
/// recursions run twice — with and without the prices of risk.
///
/// # Arguments
///
/// - `yields`: `T x M` panel of **annualized, continuously-compounded
///   zero-coupon log yields in decimal** (0.05 = 5%; divide percent by 100),
///   one row per date, oldest first.
/// - `maturities`: the `M` maturities in integer **periods** (months for
///   monthly data), strictly ascending, containing `1`. Excess returns are
///   built at every `n >= 2` whose neighbour `n - 1` is also in the grid, so
///   either supply a contiguous grid (1, 2, ..., n_max) or pairs around the
///   return maturities you want (ACM use 1..120 monthly; the vendored GSW
///   fixture uses the pairs {1, 2, 5, 6, 11, 12, 23, 24, ...}).
/// - `n_factors`: the number of principal-component pricing factors `K`
///   (ACM's baseline: 5).
/// - `periods_per_year`: compounding periods per year (12 = monthly).
///
/// # Errors
///
/// [`TermStructureError::EmptyMaturities`], [`TermStructureError::InvalidMaturity`]
/// (a zero maturity), [`TermStructureError::MaturitiesNotAscending`],
/// [`TermStructureError::MissingShortMaturity`] (no one-period maturity),
/// [`TermStructureError::InvalidFactorCount`] (`n_factors = 0` or
/// `>= M`), [`TermStructureError::InvalidPeriodsPerYear`],
/// [`TermStructureError::DimensionMismatch`] (a panel row of the wrong
/// length), [`TermStructureError::PanelTooShort`] (fewer than `2 n_factors +
/// 3` dates — the excess-return regression on `[1, X, v]` needs residual
/// degrees of freedom), [`TermStructureError::Underdetermined`] (fewer than
/// `n_factors + 1` excess-return maturities, so `beta'beta` cannot identify
/// the prices of risk), [`TermStructureError::NonFinite`] (NaN/inf yields),
/// and [`TermStructureError::SingularDesign`] (a degenerate panel — e.g.
/// constant yields — whose factors carry no variation).
///
/// # Example
///
/// ```
/// use tsecon_termstructure::acm_term_premium;
///
/// // A toy monthly panel: 30 dates, maturities 1..8 months, yields in
/// // decimal built from two slow-moving curves plus tiny noise.
/// let maturities: Vec<usize> = (1..=8).collect();
/// let yields: Vec<Vec<f64>> = (0..30)
///     .map(|t| {
///         let level = 0.03 + 0.002 * (t as f64 / 10.0).sin();
///         let slope = 0.01 + 0.001 * (t as f64 / 7.0).cos();
///         maturities
///             .iter()
///             .map(|&n| {
///                 let x = n as f64 / 8.0;
///                 level + slope * x + 1e-5 * ((t * 8 + n) as f64).sin()
///             })
///             .collect()
///     })
///     .collect();
///
/// let fit = acm_term_premium(&yields, &maturities, 2, 12.0).unwrap();
/// assert_eq!(fit.term_premium.len(), 30);
/// assert_eq!(fit.term_premium[0].len(), 8);
/// // The decomposition is exact: fitted = risk-neutral + term premium.
/// for t in 0..30 {
///     for j in 0..8 {
///         let sum = fit.risk_neutral[t][j] + fit.term_premium[t][j];
///         assert!((fit.fitted[t][j] - sum).abs() < 1e-12);
///     }
/// }
/// ```
pub fn acm_term_premium(
    yields: &[Vec<f64>],
    maturities: &[usize],
    n_factors: usize,
    periods_per_year: f64,
) -> Result<AcmFit, TermStructureError> {
    check_acm_maturities(maturities)?;
    let m_count = maturities.len();
    let k = n_factors;
    if k == 0 || k >= m_count {
        return Err(TermStructureError::InvalidFactorCount {
            requested: k,
            max: m_count.saturating_sub(1),
        });
    }
    if !periods_per_year.is_finite() || periods_per_year <= 0.0 {
        return Err(TermStructureError::InvalidPeriodsPerYear {
            value: periods_per_year,
        });
    }
    let t_len = yields.len();
    let needed = 2 * k + 3;
    if t_len < needed {
        return Err(TermStructureError::PanelTooShort {
            what: "ACM yield panel (the excess-return regression on [1, X, v] \
                   needs residual degrees of freedom)",
            dates: t_len,
            needed,
        });
    }
    for (t, row) in yields.iter().enumerate() {
        if row.len() != m_count {
            return Err(TermStructureError::DimensionMismatch {
                what: "ACM yield panel row vs maturities",
                expected: m_count,
                got: row.len(),
            });
        }
        for (j, &y) in row.iter().enumerate() {
            if !y.is_finite() {
                return Err(TermStructureError::NonFinite {
                    what: "ACM yield panel",
                    index: t * m_count + j,
                    value: y,
                });
            }
        }
    }

    // Excess-return maturities: every n >= 2 with n - 1 in the grid. Keep the
    // (n, buy-column, sell-column) triples so the return construction below
    // never re-searches the grid.
    let mut rx_pairs: Vec<(usize, usize, usize)> = Vec::new();
    for (j_buy, &n) in maturities.iter().enumerate() {
        if n >= 2 {
            if let Some(j_sell) = position(maturities, n - 1) {
                rx_pairs.push((n, j_buy, j_sell));
            }
        }
    }
    let rx_maturities: Vec<usize> = rx_pairs.iter().map(|&(n, _, _)| n).collect();
    let n_rx = rx_maturities.len();
    if n_rx < k + 1 {
        return Err(TermStructureError::Underdetermined {
            what: "ACM price-of-risk cross-section (excess-return maturities \
                   n with n - 1 in the grid)",
            maturities: n_rx,
            factors: k,
        });
    }

    // ---- factors: PCs of the demeaned panel -------------------------------
    let col_means: Vec<f64> = (0..m_count)
        .map(|j| yields.iter().map(|row| row[j]).sum::<f64>() / t_len as f64)
        .collect();
    let demeaned = Mat::from_fn(t_len, m_count, |t, j| yields[t][j] - col_means[j]);
    let svd = demeaned
        .thin_svd()
        .map_err(|_| TermStructureError::SingularDesign {
            what: "ACM principal-component factors (SVD did not converge)",
        })?;
    let v = svd.V();

    // Loadings = first K right singular vectors; factors = demeaned * W.
    let mut loadings: Vec<Vec<f64>> = (0..m_count)
        .map(|j| (0..k).map(|c| v[(j, c)]).collect())
        .collect();
    let mut factors: Vec<Vec<f64>> = (0..t_len)
        .map(|t| {
            (0..k)
                .map(|c| {
                    (0..m_count)
                        .map(|j| demeaned[(t, j)] * loadings[j][c])
                        .sum()
                })
                .collect()
        })
        .collect();

    // Scale each factor to unit sample std (ddof = 1) and fix signs so each
    // loading column sums positive. Outputs are invariant to both choices.
    for c in 0..k {
        let mean: f64 = factors.iter().map(|f| f[c]).sum::<f64>() / t_len as f64;
        let ss: f64 = factors.iter().map(|f| (f[c] - mean).powi(2)).sum();
        let sd = (ss / (t_len as f64 - 1.0)).sqrt();
        if sd <= 0.0 {
            return Err(TermStructureError::SingularDesign {
                what: "ACM principal-component factors (a factor has zero \
                       variance — is the yield panel constant?)",
            });
        }
        let loading_sum: f64 = loadings.iter().map(|l| l[c]).sum();
        let flip = if loading_sum < 0.0 { -1.0 } else { 1.0 };
        for f in factors.iter_mut() {
            f[c] *= flip / sd;
        }
        for l in loadings.iter_mut() {
            l[c] *= flip / sd;
        }
    }

    // ---- step 1: VAR(1) with intercept ------------------------------------
    let t_v = t_len - 1; // innovation rows
    let ones = vec![1.0f64; t_v];
    let mut var_cols: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
    var_cols.push(ones.clone());
    var_cols.extend((0..k).map(|c| factors.iter().take(t_v).map(|f| f[c]).collect::<Vec<f64>>()));
    let mut mu = vec![0.0f64; k];
    let mut phi = vec![vec![0.0f64; k]; k];
    let mut innovations = vec![vec![0.0f64; k]; t_v];
    let mut var_rsquared = vec![0.0f64; k];
    for eq in 0..k {
        let y_eq: Vec<f64> = (1..t_len).map(|t| factors[t][eq]).collect();
        let fit = ols(&y_eq, &var_cols).map_err(|e| map_ols_err(e, "ACM factor VAR(1)"))?;
        mu[eq] = fit.params[0];
        phi[eq].copy_from_slice(&fit.params[1..]);
        for (t, &res) in fit.residuals.iter().enumerate() {
            innovations[t][eq] = res;
        }
        var_rsquared[eq] = centered_rsquared(&y_eq, &fit.residuals);
    }
    // Sigma = v'v / (T - 2): the ddof-1 covariance (v has exact zero mean).
    let sigma: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| {
                    innovations.iter().map(|row| row[i] * row[j]).sum::<f64>() / (t_v as f64 - 1.0)
                })
                .collect()
        })
        .collect();

    // ---- excess returns ----------------------------------------------------
    // Per-period log prices p_t(n) = -n y_t(n) / ppy; short rate r_t = -p_t(1).
    let short_col = 0; // maturities[0] == 1 by validation
    let short_rate: Vec<f64> = (0..t_len)
        .map(|t| yields[t][short_col] / periods_per_year)
        .collect();
    let log_price =
        |t: usize, j: usize| -> f64 { -(maturities[j] as f64) * yields[t][j] / periods_per_year };
    // rx[t][i]: return from t to t+1 of the bond bought at rx_maturities[i].
    let rx: Vec<Vec<f64>> = (0..t_v)
        .map(|t| {
            rx_pairs
                .iter()
                .map(|&(_, j_buy, j_sell)| {
                    log_price(t + 1, j_sell) - log_price(t, j_buy) - short_rate[t]
                })
                .collect()
        })
        .collect();

    // ---- step 2: rx on [1, X_{t-1}, v_t] -----------------------------------
    let mut rx_cols: Vec<Vec<f64>> = Vec::with_capacity(1 + 2 * k);
    rx_cols.push(ones);
    rx_cols.extend((0..k).map(|c| factors.iter().take(t_v).map(|f| f[c]).collect::<Vec<f64>>()));
    rx_cols.extend((0..k).map(|c| innovations.iter().map(|v| v[c]).collect::<Vec<f64>>()));
    let mut rx_a = vec![0.0f64; n_rx];
    let mut rx_c = vec![vec![0.0f64; k]; n_rx];
    let mut rx_beta = vec![vec![0.0f64; k]; n_rx];
    let mut rx_rsquared = vec![0.0f64; n_rx];
    let mut sse = 0.0f64;
    for i in 0..n_rx {
        let y_i: Vec<f64> = (0..t_v).map(|t| rx[t][i]).collect();
        let fit =
            ols(&y_i, &rx_cols).map_err(|e| map_ols_err(e, "ACM excess-return regression"))?;
        rx_a[i] = fit.params[0];
        rx_c[i].copy_from_slice(&fit.params[1..=k]);
        rx_beta[i].copy_from_slice(&fit.params[k + 1..]);
        sse += fit.residuals.iter().map(|r| r * r).sum::<f64>();
        rx_rsquared[i] = centered_rsquared(&y_i, &fit.residuals);
    }
    let sigma2 = sse / (n_rx as f64 * t_v as f64);

    // ---- step 3: prices of risk --------------------------------------------
    // Convexity-adjusted intercepts a* = a + 1/2 (B* vec(Sigma) + sigma^2).
    let a_adj: Vec<f64> = (0..n_rx)
        .map(|i| rx_a[i] + 0.5 * (quad_form(&sigma, &rx_beta[i]) + sigma2))
        .collect();
    // lambda = (beta'beta)^{-1} beta' [a*, c]: OLS of each column on beta.
    let beta_cols: Vec<Vec<f64>> = (0..k)
        .map(|c| (0..n_rx).map(|i| rx_beta[i][c]).collect())
        .collect();
    let lam_fit = ols(&a_adj, &beta_cols)
        .map_err(|e| map_ols_err(e, "ACM price-of-risk cross-section (lambda0)"))?;
    let lambda0 = lam_fit.params;
    let mut lambda1 = vec![vec![0.0f64; k]; k];
    for j in 0..k {
        let col_j: Vec<f64> = (0..n_rx).map(|i| rx_c[i][j]).collect();
        let fit = ols(&col_j, &beta_cols)
            .map_err(|e| map_ols_err(e, "ACM price-of-risk cross-section (lambda1)"))?;
        for (row, &param) in lambda1.iter_mut().zip(fit.params.iter()) {
            row[j] = param;
        }
    }

    // ---- short-rate equation ------------------------------------------------
    let mut sr_cols: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
    sr_cols.push(vec![1.0f64; t_len]);
    sr_cols.extend((0..k).map(|c| factors.iter().map(|f| f[c]).collect::<Vec<f64>>()));
    let sr_fit =
        ols(&short_rate, &sr_cols).map_err(|e| map_ols_err(e, "ACM short-rate equation"))?;
    let delta0 = sr_fit.params[0];
    let delta1: Vec<f64> = sr_fit.params[1..].to_vec();
    let short_rate_rsquared = centered_rsquared(&short_rate, &sr_fit.residuals);

    // ---- affine recursions ----------------------------------------------------
    let n_max = maturities[m_count - 1];
    let (price_a, price_b) = price_recursion(
        n_max,
        k,
        &mu,
        &phi,
        &sigma,
        sigma2,
        delta0,
        &delta1,
        Some(&lambda0),
        Some(&lambda1),
    );
    let (price_a_rn, price_b_rn) = price_recursion(
        n_max, k, &mu, &phi, &sigma, sigma2, delta0, &delta1, None, None,
    );

    // ---- fitted / risk-neutral / term premium ---------------------------------
    let yield_from = |a: &[f64], b: &[Vec<f64>], t: usize, n: usize| -> f64 {
        let bx: f64 = b[n - 1]
            .iter()
            .zip(factors[t].iter())
            .map(|(&bj, &xj)| bj * xj)
            .sum();
        -(a[n - 1] + bx) * periods_per_year / n as f64
    };
    let mut fitted = vec![vec![0.0f64; m_count]; t_len];
    let mut risk_neutral = vec![vec![0.0f64; m_count]; t_len];
    let mut term_premium = vec![vec![0.0f64; m_count]; t_len];
    for t in 0..t_len {
        for (j, &n) in maturities.iter().enumerate() {
            let f = yield_from(&price_a, &price_b, t, n);
            let rn = yield_from(&price_a_rn, &price_b_rn, t, n);
            fitted[t][j] = f;
            risk_neutral[t][j] = rn;
            term_premium[t][j] = f - rn;
        }
    }
    let yield_rsquared: Vec<f64> = (0..m_count)
        .map(|j| {
            let obs: Vec<f64> = (0..t_len).map(|t| yields[t][j]).collect();
            let resid: Vec<f64> = (0..t_len).map(|t| yields[t][j] - fitted[t][j]).collect();
            centered_rsquared(&obs, &resid)
        })
        .collect();

    Ok(AcmFit {
        maturities: maturities.to_vec(),
        n_factors: k,
        periods_per_year,
        factors,
        factor_loadings: loadings,
        mu,
        phi,
        sigma,
        rx_maturities,
        rx_a,
        rx_beta,
        rx_c,
        sigma2,
        lambda0,
        lambda1,
        delta0,
        delta1,
        price_a,
        price_b,
        price_a_rn,
        price_b_rn,
        fitted,
        risk_neutral,
        term_premium,
        var_rsquared,
        rx_rsquared,
        short_rate_rsquared,
        yield_rsquared,
    })
}
