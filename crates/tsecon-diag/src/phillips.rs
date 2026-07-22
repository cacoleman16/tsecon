//! Phillips-Perron unit-root test and Phillips-Ouliaris residual
//! cointegration test.
//!
//! Both are *semiparametric* tests: they run a simple Dickey-Fuller-style
//! levels regression (no lag augmentation) and then correct the statistic
//! for serial correlation with a nonparametric Bartlett long-run-variance
//! (LRV) estimate of the residuals, rather than by adding lagged
//! differences as the ADF test does. The single LRV engine is
//! [`tsecon_hac::lrv`] (Newey-West 1987 weights `1 - j/(bandwidth + 1)`,
//! biased `1/n` autocovariances about zero), shared library-wide.
//!
//! * [`phillips_perron`] — Phillips & Perron (1988) `Z-tau` (t-based) and
//!   `Z-alpha` (normalized-bias) statistics for a single series. Matches
//!   `arch.unitroot.PhillipsPerron` (arch 8.0.0). `Z-tau` p-values/critical
//!   values reuse the N = 1 MacKinnon tau surfaces of [`crate::mackinnon`];
//!   `Z-alpha` uses the ADF-z surfaces of [`crate::mackinnon_ext`].
//!
//! * [`phillips_ouliaris`] — Phillips & Ouliaris (1990) residual-based test
//!   for no cointegration among `N = 1 + m` I(1) series. Matches the
//!   `Zt` / `Za` statistics of `arch.unitroot.cointegration.phillips_ouliaris`.
//!   `Zt` p-values/critical values use the MacKinnon (1994, 2010)
//!   cointegration surfaces of [`crate::mackinnon_ext`], indexed by `N`,
//!   the same route `statsmodels.tsa.stattools.coint` takes. `Za` is
//!   statistic-only (no MacKinnon z-surface exists for `N > 1`).
//!
//! References: Phillips & Perron (1988), Biometrika 75(2); Phillips &
//! Ouliaris (1990), Econometrica 58(1); MacKinnon (1994, 2010).

use tsecon_hac::{lrv, newey_west_maxlags, Kernel};

use crate::error::DiagError;
use crate::mackinnon::{mackinnon_crit, mackinnon_p, AdfCriticalValues};
use crate::mackinnon_ext::{
    mackinnon_coint_crit, mackinnon_coint_p, mackinnon_z_crit, mackinnon_z_p,
};
use crate::unitroot::AdfRegression;
use crate::validate::check_series;

// ------------------------------------------------------- OLS helper

/// Result of a plain OLS fit `y = X b + e` (no implicit intercept;
/// deterministics are explicit columns) with classical nonrobust standard
/// errors, exposing the pieces the Phillips tests need: coefficients, their
/// standard errors, the residual vector, and the residual sum of squares.
struct OlsQr {
    /// Coefficients `b`, in the order the columns were supplied.
    params: Vec<f64>,
    /// Standard errors `se(b_j) = sqrt(s^2 [(X'X)^{-1}]_{jj})`,
    /// `s^2 = SSR / (n - k)`.
    bse: Vec<f64>,
    /// Residual vector `e = y - X b` (length `n`).
    resid: Vec<f64>,
    /// Residual sum of squares `e'e`.
    ssr: f64,
}

/// Fit `y = X b + e` by Householder QR (columns of `X` in `cols`; pass a
/// column of ones for an intercept). Mirrors the ADF crate's `ols_detailed`
/// but returns the residual vector and coefficient standard errors, which
/// the nonparametric LRV correction requires. Householder QR keeps the
/// error proportional to `cond(X)` rather than `cond(X)^2` — the
/// levels/deterministic designs here can be poorly conditioned.
fn ols_qr(cols: &[Vec<f64>], y: &[f64], what: &'static str) -> Result<OlsQr, DiagError> {
    let n = y.len();
    let k = cols.len();
    debug_assert!(cols.iter().all(|c| c.len() == n));
    if k == 0 || n < k + 1 {
        return Err(DiagError::SeriesTooShort {
            what,
            n,
            needed: k + 1,
        });
    }

    let mut a: Vec<Vec<f64>> = cols.to_vec();
    let mut qty: Vec<f64> = y.to_vec();
    let mut rdiag = vec![0.0_f64; k];

    for j in 0..k {
        let sub: f64 = a[j][j..].iter().map(|&v| v * v).sum();
        let head: f64 = a[j][..j].iter().map(|&v| v * v).sum();
        let norm = sub.sqrt();
        let tol = ((head + sub).sqrt() * 1e-13).max(f64::MIN_POSITIVE);
        if norm.is_nan() || norm <= tol {
            return Err(DiagError::SingularDesign { what });
        }
        let alpha = if a[j][j] >= 0.0 { -norm } else { norm };
        a[j][j] -= alpha;
        rdiag[j] = alpha;
        let vtv: f64 = a[j][j..].iter().map(|&v| v * v).sum();

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

    // Back substitution R b = (Q'y)[0..k].
    let mut beta = vec![0.0_f64; k];
    for j in (0..k).rev() {
        let mut acc = qty[j];
        for (m, bm) in beta.iter().enumerate().skip(j + 1) {
            acc -= a[m][j] * bm;
        }
        beta[j] = acc / rdiag[j];
    }

    // Residuals and SSR against the original columns.
    let mut resid = vec![0.0_f64; n];
    let mut ssr = 0.0;
    for (i, ri) in resid.iter_mut().enumerate() {
        let mut fit = 0.0;
        for (bj, col) in beta.iter().zip(cols.iter()) {
            fit += bj * col[i];
        }
        let e = y[i] - fit;
        *ri = e;
        ssr += e * e;
    }

    let sigma2 = ssr / (n - k) as f64;
    if !(sigma2 > 0.0 && sigma2.is_finite()) {
        return Err(DiagError::NumericalBreakdown { what });
    }

    // diag[(X'X)^{-1}] = squared row norms of R^{-1}.
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
    let bse = xtx_inv_diag.iter().map(|&d| (sigma2 * d).sqrt()).collect();

    Ok(OlsQr {
        params: beta,
        bse,
        resid,
        ssr,
    })
}

/// Bartlett long-run variance via the shared HAC engine, translating any
/// HAC-side failure (which cannot arise for the finite, adequately long
/// residual series built here) into a diagnostic breakdown.
fn bartlett_lrv(resid: &[f64], bandwidth: usize, what: &'static str) -> Result<f64, DiagError> {
    lrv(resid, Kernel::Bartlett, bandwidth as f64)
        .map_err(|_| DiagError::NumericalBreakdown { what })
}

// ----------------------------------------------------- Phillips-Perron

/// Which Phillips-Perron statistic is the primary result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PpTestType {
    /// The t-based `Z-tau` statistic (statsmodels/arch `"tau"`), the
    /// conventional default, with MacKinnon ADF-t p-values.
    Tau,
    /// The normalized-bias `Z-alpha` statistic (`"rho"`), with MacKinnon
    /// ADF-z p-values.
    Rho,
}

/// Result of the Phillips-Perron unit-root test.
///
/// The null hypothesis is a unit root; the alternative is stationarity, so
/// (as for the ADF test) small p-values speak *for* stationarity. Both the
/// `Z-tau` and `Z-alpha` statistics are always reported; `stat`,
/// `p_value`, and `crit` refer to the one selected by `test_type`.
#[derive(Debug, Clone, PartialEq)]
pub struct PpResult {
    /// The t-based `Z-tau` statistic.
    pub ztau: f64,
    /// The normalized-bias `Z-alpha` statistic.
    pub zalpha: f64,
    /// The selected statistic (`ztau` for `Tau`, `zalpha` for `Rho`).
    pub stat: f64,
    /// MacKinnon p-value of the selected statistic (ADF-t surface for
    /// `Tau`, ADF-z surface for `Rho`).
    pub p_value: f64,
    /// The Bartlett LRV bandwidth used.
    pub lags: usize,
    /// Effective observations in the levels regression (`n - 1`).
    pub nobs: usize,
    /// MacKinnon (2010) finite-sample critical values of the selected
    /// statistic at `nobs`.
    pub crit: AdfCriticalValues,
    /// The deterministic specification that was tested.
    pub regression: AdfRegression,
    /// Which statistic was selected.
    pub test_type: PpTestType,
}

/// Phillips-Perron unit-root test (Phillips & Perron 1988), matching
/// `arch.unitroot.PhillipsPerron`.
///
/// On the `T = n - 1` usable rows the Dickey-Fuller *levels* regression
///
/// ```text
/// y_t = rho y_{t-1} + (deterministics) + u_t
/// ```
///
/// is fit by OLS (no lagged differences). With `gamma0 = SSR/T`, the
/// Bartlett long-run variance `lam2 = lrv(u; bandwidth)`, `s^2 = SSR/(T-k)`,
/// and `sigma = se(rho)`, the two Phillips-Perron statistics are
///
/// ```text
/// Z_tau   = sqrt(gamma0/lam2) (rho-1)/sigma - 0.5 (lam2-gamma0)/lam * (T sigma / s)
/// Z_alpha = T (rho-1) - 0.5 (T^2 sigma^2 / s^2) (lam2 - gamma0)
/// ```
///
/// with `H0: rho = 1` (unit root). `regression` selects the deterministics
/// (`"n"`, `"c"`, `"ct"`). `lags = None` uses the default Bartlett
/// bandwidth `ceil(12 (n/100)^{1/4})` (on the full length `n`, matching
/// arch). P-values and critical values are the MacKinnon response surfaces
/// for the selected statistic.
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if the series contains NaN or infinities.
/// * [`DiagError::SeriesTooShort`] if fewer than `2 (ntrend + 1)`
///   observations are supplied.
/// * [`DiagError::ConstantSeries`] if the series is constant.
/// * [`DiagError::InvalidLags`] if the bandwidth exceeds `nobs = n - 1`.
/// * [`DiagError::SingularDesign`] / [`DiagError::NumericalBreakdown`] for
///   (near-)deterministic series whose levels regression is degenerate.
pub fn phillips_perron(
    y: &[f64],
    regression: AdfRegression,
    test_type: PpTestType,
    lags: Option<usize>,
) -> Result<PpResult, DiagError> {
    let ntrend = match regression {
        AdfRegression::NoConstant => 0,
        AdfRegression::Constant => 1,
        AdfRegression::ConstantTrend => 2,
    };
    let n = check_series(y, 2 * (ntrend + 1), "phillips_perron")?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(DiagError::ConstantSeries {
            what: "phillips_perron",
        });
    }

    let t = n - 1; // rows in the levels regression
                   // Design: lhs = y_t (t = 1..n-1); columns = [y_{t-1}, deterministics].
    let lhs: Vec<f64> = y[1..].to_vec();
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(ntrend + 1);
    cols.push(y[..n - 1].to_vec());
    if ntrend >= 1 {
        cols.push(vec![1.0; t]);
    }
    if ntrend >= 2 {
        cols.push((1..=t).map(|i| i as f64).collect());
    }
    let k = cols.len();

    // Default Bartlett bandwidth uses the full pre-differencing length n.
    let l = match lags {
        Some(l) => l,
        None => (12.0 * (n as f64 / 100.0).powf(0.25)).ceil() as usize,
    };
    if l > t {
        return Err(DiagError::InvalidLags {
            what: "phillips_perron",
            nlags: l,
            n,
            requirement: "bandwidth <= n - 1 (the LRV window cannot exceed \
                          the levels-regression sample)",
        });
    }

    let fit = ols_qr(&cols, &lhs, "phillips_perron")?;
    let rho = fit.params[0];
    let sigma = fit.bse[0];
    if !(sigma > 0.0 && sigma.is_finite()) {
        return Err(DiagError::NumericalBreakdown {
            what: "phillips_perron",
        });
    }

    let tf = t as f64;
    let ssr = fit.ssr;
    let s2 = ssr / (tf - k as f64);
    let s = s2.sqrt();
    let gamma0 = ssr / tf;
    let lam2 = bartlett_lrv(&fit.resid, l, "phillips_perron")?;
    if !(lam2 > 0.0 && lam2.is_finite()) {
        return Err(DiagError::NumericalBreakdown {
            what: "phillips_perron",
        });
    }
    let lam = lam2.sqrt();
    let sigma2 = sigma * sigma;

    let ztau = (gamma0 / lam2).sqrt() * ((rho - 1.0) / sigma)
        - 0.5 * ((lam2 - gamma0) / lam) * (tf * sigma / s);
    let zalpha = tf * (rho - 1.0) - 0.5 * (tf * tf * sigma2 / s2) * (lam2 - gamma0);

    let (stat, p_value, crit) = match test_type {
        PpTestType::Tau => (
            ztau,
            mackinnon_p(ztau, regression),
            mackinnon_crit(regression, Some(t)),
        ),
        PpTestType::Rho => (
            zalpha,
            mackinnon_z_p(zalpha, regression),
            mackinnon_z_crit(regression, t),
        ),
    };

    Ok(PpResult {
        ztau,
        zalpha,
        stat,
        p_value,
        lags: l,
        nobs: t,
        crit,
        regression,
        test_type,
    })
}

// --------------------------------------------------- Phillips-Ouliaris

/// Deterministic terms included in the Phillips-Ouliaris cointegrating
/// regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoTrend {
    /// No deterministic terms (statsmodels/arch `"n"`).
    None,
    /// Constant only (`"c"`, the conventional default).
    Constant,
    /// Constant and linear trend (`"ct"`).
    ConstantTrend,
}

impl PoTrend {
    /// Number of deterministic columns appended to the cointegrating
    /// regression.
    fn ntrend(self) -> usize {
        match self {
            PoTrend::None => 0,
            PoTrend::Constant => 1,
            PoTrend::ConstantTrend => 2,
        }
    }
}

/// Which Phillips-Ouliaris statistic is the primary result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoTestType {
    /// The normalized-bias `Za` statistic (statistic-only: no MacKinnon
    /// z-surface exists for `N > 1`, so its p-value/critical values are
    /// unavailable).
    Za,
    /// The t-based `Zt` statistic (the default), with MacKinnon
    /// cointegration p-values/critical values.
    Zt,
}

/// Result of the Phillips-Ouliaris residual cointegration test.
///
/// The null hypothesis is *no cointegration*; the alternative is
/// cointegration, so small p-values speak *for* cointegration.
#[derive(Debug, Clone, PartialEq)]
pub struct PoResult {
    /// The selected statistic (`Za` or `Zt`).
    pub stat: f64,
    /// MacKinnon cointegration p-value of `Zt` (NaN for `Za`, and for
    /// `Zt` when `N > 6`).
    pub p_value: f64,
    /// MacKinnon (2010) cointegration critical values of `Zt` at
    /// `nobs = T - 1` (`None` for `Za`, for `N > 12`, and for the
    /// no-constant case with `N > 1`).
    pub crit: Option<AdfCriticalValues>,
    /// The Bartlett LRV bandwidth used.
    pub lags: usize,
    /// Number of observations `T` (no lag loss in the cross-section
    /// regression).
    pub nobs: usize,
    /// The stochastic dimension `N = 1 + (number of regressors in `x`)`.
    pub n_vars: usize,
    /// The deterministic specification that was tested.
    pub trend: PoTrend,
    /// Which statistic was selected.
    pub test_type: PoTestType,
}

/// Phillips-Ouliaris residual cointegration test (Phillips & Ouliaris
/// 1990), matching `arch.unitroot.cointegration.phillips_ouliaris`.
///
/// A cross-section OLS of `y` on `[x, deterministics]` (no lag loss) yields
/// residuals `u`; an AR(1) through the origin `alpha = (u_{-1}.u_1)/
/// (u_{-1}.u_{-1})` gives filtered residuals `k_t = u_t - alpha u_{t-1}`.
/// With `u2 = sum u_{t-1}^2`, `k_scale = (T-1)/T`, `gamma0_k` the residual
/// variance of `k`, `omega2 = lrv(k; bandwidth)`, and
/// `lambda1 = (omega2 - gamma0_k)/2`,
///
/// ```text
/// z  = (alpha - 1) - T (k_scale lambda1) / u2
/// Za = T z
/// Zt = z / sqrt( k_scale omega2 / u2 )
/// ```
///
/// `x` supplies the `m` stochastic regressors *as-is* (do not add your own
/// constant — deterministics come from `trend`); `N = m + 1`.
/// `bandwidth = None` uses the Newey-West rule of thumb
/// `floor(4 ((T-1)/100)^{2/9})`. `Zt` p-values/critical values use the
/// MacKinnon cointegration surfaces indexed by `N` (the `statsmodels`
/// `coint` route); `Za` is statistic-only.
///
/// # Errors
///
/// * [`DiagError::NonFinite`] if `y` or `x` contains NaN or infinities.
/// * [`DiagError::SeriesTooShort`] if `x` has no columns (`N < 2`), if the
///   `x` columns do not match `y` in length, or if too few observations
///   remain for the regression.
/// * [`DiagError::SingularDesign`] if the cross-section design is collinear.
/// * [`DiagError::NumericalBreakdown`] for a (near-)perfect fit whose
///   residuals are degenerate.
pub fn phillips_ouliaris(
    y: &[f64],
    x: &[Vec<f64>],
    trend: PoTrend,
    test_type: PoTestType,
    bandwidth: Option<usize>,
) -> Result<PoResult, DiagError> {
    let m = x.len();
    // N = m + 1 must be >= 2, i.e. at least one stochastic regressor.
    if m == 0 {
        return Err(DiagError::SeriesTooShort {
            what: "phillips_ouliaris",
            n: 0,
            needed: 1,
        });
    }
    let n_vars = m + 1;
    let ntrend = trend.ntrend();

    let t = check_series(y, ntrend + m + 2, "phillips_ouliaris")?;
    for col in x {
        if col.len() != t {
            return Err(DiagError::SeriesTooShort {
                what: "phillips_ouliaris",
                n: col.len(),
                needed: t,
            });
        }
        for (index, &value) in col.iter().enumerate() {
            if !value.is_finite() {
                return Err(DiagError::NonFinite { index, value });
            }
        }
    }

    // Cross-section design: [x columns, deterministics appended].
    let mut cols: Vec<Vec<f64>> = x.to_vec();
    if ntrend >= 1 {
        cols.push(vec![1.0; t]);
    }
    if ntrend >= 2 {
        cols.push((1..=t).map(|i| i as f64).collect());
    }

    let fit = ols_qr(&cols, y, "phillips_ouliaris")?;
    let u = &fit.resid;

    // AR(1) through the origin on the residuals.
    let mut num = 0.0;
    let mut u2 = 0.0;
    for w in u.windows(2) {
        num += w[1] * w[0];
        u2 += w[0] * w[0];
    }
    if !(u2 > 0.0 && u2.is_finite()) {
        return Err(DiagError::NumericalBreakdown {
            what: "phillips_ouliaris",
        });
    }
    let alpha = num / u2;
    let filtered: Vec<f64> = u.windows(2).map(|w| w[1] - alpha * w[0]).collect();
    let tk = filtered.len(); // = T - 1

    let l = bandwidth.unwrap_or_else(|| newey_west_maxlags(tk));

    let tf = t as f64;
    let k_scale = (tf - 1.0) / tf;
    let gamma0_k = filtered.iter().map(|&v| v * v).sum::<f64>() / tk as f64;
    let omega2 = bartlett_lrv(&filtered, l, "phillips_ouliaris")?;
    let lambda1 = (omega2 - gamma0_k) / 2.0;

    let z = (alpha - 1.0) - tf * (k_scale * lambda1) / u2;
    let za = tf * z;
    let denom = ((k_scale * omega2) / u2).sqrt();
    let zt = z / denom;

    let stat = match test_type {
        PoTestType::Za => za,
        PoTestType::Zt => zt,
    };
    let (p_value, crit) = match test_type {
        PoTestType::Zt => (
            mackinnon_coint_p(zt, trend, n_vars),
            mackinnon_coint_crit(trend, n_vars, t - 1),
        ),
        PoTestType::Za => (f64::NAN, None),
    };

    Ok(PoResult {
        stat,
        p_value,
        crit,
        lags: l,
        nobs: t,
        n_vars,
        trend,
        test_type,
    })
}
