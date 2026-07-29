//! The Engle-Granger (1987) two-step residual-based cointegration test — a
//! simpler single-equation alternative to the Johansen system approach.
//!
//! Step 1 runs the static cointegrating regression of the first series on
//! the others (plus deterministics) by OLS. Step 2 tests the residuals for
//! a unit root with the augmented Dickey-Fuller regression: if the
//! residuals are stationary the series are cointegrated. The residual ADF
//! machinery is delegated to [`tsecon_diag::adf`], which is already
//! validated against statsmodels.
//!
//! Convention note (important): under the null of *no* cointegration the
//! step-1 residuals are a spurious regression, so the step-2 statistic does
//! **not** follow the standard Dickey-Fuller distribution — it needs the
//! Engle-Granger / Phillips-Ouliaris critical values, which depend on the
//! number of series `N` and the deterministic terms. This module therefore
//! reports the MacKinnon (1994) cointegration p-value and the MacKinnon
//! (2010) finite-sample cointegration critical values on
//! [`EngleGrangerResult`] itself; the `p_value`/`crit` carried inside the
//! inner [`tsecon_diag::AdfResult`] are the standard-ADF (`N = 1`) values
//! and must never be used to decide cointegration.
//!
//! The whole routine reproduces `statsmodels.tsa.stattools.coint(y0, y1,
//! trend, maxlag, autolag)` — statistic, p-value, and critical values — with
//! two deliberate improvements:
//!
//! * for `N > 6` statsmodels raises `IndexError` (its p-value tables stop
//!   at `N = 6`) while this function returns a `NaN` p-value alongside the
//!   critical values, which the 2010 surfaces do cover up to `N = 12`;
//! * for an (almost) perfectly collinear step-1 regression statsmodels
//!   emits a `CollinearityWarning` and returns `(-inf, 0.0, crit)`; this
//!   function returns a [`CointError::Singular`] that names the problem
//!   instead of a decision that looks like overwhelming evidence.
//!
//! References: Engle & Granger (1987), Econometrica 55(2); MacKinnon
//! (1994), JBES 12(2); MacKinnon (2010), Queen's University working paper
//! 1227.

use tsecon_diag::{
    adf, mackinnon_coint_crit, mackinnon_coint_p, AdfCriticalValues, AdfLagSelection,
    AdfRegression, AdfResult, PoTrend,
};
use tsecon_linalg::faer::linalg::solvers::SolveLstsq;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::CointError;
use crate::linalg::check_finite;

/// statsmodels' collinearity guard on the step-1 fit: `rsquared < 1 - 100 *
/// sqrt(eps)`, with `sqrt(eps) = 1.4901161193847656e-8` for f64.
const EG_MAX_RSQUARED: f64 = 1.0 - 100.0 * 1.490_116_119_384_765_6e-8;

/// Deterministic terms in the step-1 cointegrating regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngleGrangerTrend {
    /// No deterministic terms (statsmodels `coint(trend = "n")`). The 2010
    /// critical-value surfaces do not cover this case for `N > 1`, so
    /// `crit` is `None` — exactly as `statsmodels.coint` reports it.
    None,
    /// Constant only (statsmodels `coint(trend = "c")`, the default).
    Constant,
    /// Constant and a linear time trend (statsmodels `trend = "ct"`).
    ConstantTrend,
}

impl EngleGrangerTrend {
    fn n_det(self) -> usize {
        match self {
            EngleGrangerTrend::None => 0,
            EngleGrangerTrend::Constant => 1,
            EngleGrangerTrend::ConstantTrend => 2,
        }
    }

    /// Whether the step-1 design carries an intercept, which decides
    /// whether statsmodels' `rsquared` is centered or uncentered.
    fn has_constant(self) -> bool {
        !matches!(self, EngleGrangerTrend::None)
    }

    /// The equivalent deterministic code for the shared MacKinnon
    /// cointegration surfaces in `tsecon-diag`.
    fn po_trend(self) -> PoTrend {
        match self {
            EngleGrangerTrend::None => PoTrend::None,
            EngleGrangerTrend::Constant => PoTrend::Constant,
            EngleGrangerTrend::ConstantTrend => PoTrend::ConstantTrend,
        }
    }
}

/// Result of the Engle-Granger two-step procedure.
///
/// The null hypothesis is *no cointegration*; the alternative is
/// cointegration, so small p-values speak *for* cointegration (and a
/// statistic below the critical value rejects the null).
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
    /// The Engle-Granger test statistic: the augmented Dickey-Fuller `tau`
    /// on the step-1 residuals. More negative is stronger evidence of
    /// cointegration.
    pub stat: f64,
    /// MacKinnon (1994) cointegration p-value of `stat` at `n_vars`,
    /// evaluated on the surface for `trend`. `NaN` when `n_vars > 6`: the
    /// published p-value tables stop there.
    pub p_value: f64,
    /// MacKinnon (2010) finite-sample cointegration critical values at the
    /// 1% / 5% / 10% levels, evaluated at `nobs - 1` (the `-1` matches
    /// `statsmodels.coint`, which follows Stata's `egranger`). `None` when
    /// no published 2010 surface exists: `trend = None`, or `n_vars > 12`.
    pub crit: Option<AdfCriticalValues>,
    /// The number of series `N = k` (regressand plus regressors) that
    /// indexes the MacKinnon surfaces.
    pub n_vars: usize,
    /// Observations `T` in the step-1 regression (the full sample; the
    /// residual ADF loses rows on top of this — see `resid_adf.nobs`).
    pub nobs: usize,
    /// The deterministic specification that was tested.
    pub trend: EngleGrangerTrend,
    /// Coefficients of the step-1 cointegrating regression of series 0 on
    /// the deterministics and the remaining series, in design order
    /// `[deterministics..., series_1, ..., series_{k-1}]`. The implied
    /// cointegrating vector on the levels is `[1, -coef_on_series_1, ...]`.
    pub coint_coefs: Vec<f64>,
    /// Step-1 OLS residuals (length `T`), the series tested for a unit root.
    pub resid: Vec<f64>,
    /// The residual augmented Dickey-Fuller result: its `statistic` is the
    /// Engle-Granger statistic and its `used_lag`/`nobs` describe the
    /// step-2 regression. **Its `p_value`/`crit` are standard-ADF
    /// (`N = 1`) values and are not valid for a cointegration decision —
    /// use the `p_value`/`crit` on this struct instead.**
    pub resid_adf: AdfResult,
}

impl EngleGrangerResult {
    /// The Engle-Granger test statistic: the augmented Dickey-Fuller
    /// `tau` on the step-1 residuals. More negative is stronger evidence
    /// of cointegration. (Alias of the `stat` field.)
    pub fn statistic(&self) -> f64 {
        self.stat
    }
}

/// Runs the Engle-Granger two-step test on `endog` (a `T x k` matrix,
/// oldest row first; series 0 is the regressand). The residual ADF uses no
/// deterministic term (statsmodels `regression = "n"`) because the step-1
/// deterministics already absorb the level.
///
/// Reproduces `statsmodels.tsa.stattools.coint(endog[:, 0], endog[:, 1:],
/// trend = trend, autolag = lags)`: `lags = AdfLagSelection::Aic(None)` is
/// the statsmodels default (`autolag = "aic"`, `maxlag = None`), and
/// `AdfLagSelection::Fixed(p)` is `autolag = None, maxlag = p`.
///
/// # Errors
///
/// * [`CointError::Dimension`] if `endog` has fewer than two columns;
/// * [`CointError::NonFinite`] if `endog` contains a NaN or infinity;
/// * [`CointError::Singular`] if the step-1 design is collinear, or if the
///   step-1 fit is (almost) perfect — the point at which statsmodels warns
///   that the test is unreliable and returns `-inf`;
/// * [`CointError::Diag`] if the residual ADF step fails (too few
///   observations, degenerate residuals, ...).
pub fn engle_granger(
    endog: MatRef<'_, f64>,
    trend: EngleGrangerTrend,
    lags: AdfLagSelection,
) -> Result<EngleGrangerResult, CointError> {
    let k = endog.ncols();
    if k < 2 {
        return Err(CointError::Dimension {
            what: "Engle-Granger tests for a relation *between* series, so it needs at \
                   least two columns; pass a 2-D array shaped (n_obs, n_series)",
            expected: 2,
            got: k,
        });
    }
    check_finite(endog, "the data matrix")?;
    let n = endog.nrows();
    let n_det = trend.n_det();
    let n_reg = n_det + (k - 1);

    // Step 1: design [deterministics, series 1..k-1], response series 0.
    // The trend column runs 1..=n, matching statsmodels `add_trend`; the
    // deterministics are placed first here rather than appended (a column
    // permutation, so the fit and the residuals are unchanged).
    let mut x = Mat::<f64>::zeros(n, n_reg);
    for i in 0..n {
        let mut col = 0;
        if n_det >= 1 {
            x[(i, col)] = 1.0;
            col += 1;
        }
        if n_det >= 2 {
            x[(i, col)] = (i + 1) as f64;
            col += 1;
        }
        for j in 1..k {
            x[(i, col)] = endog[(i, j)];
            col += 1;
        }
    }
    let y = Mat::from_fn(n, 1, |i, _| endog[(i, 0)]);
    let b = x.qr().solve_lstsq(&y);
    check_finite(b.as_ref(), "cointegrating regression coefficients").map_err(|_| {
        CointError::Singular {
            what: "the Engle-Granger step-1 design; the regressors are collinear, so \
                   the cointegrating regression has no unique solution — drop the \
                   redundant series",
        }
    })?;
    let fitted = &x * &b;
    let resid: Vec<f64> = (0..n).map(|i| y[(i, 0)] - fitted[(i, 0)]).collect();
    let coint_coefs: Vec<f64> = (0..n_reg).map(|r| b[(r, 0)]).collect();

    // statsmodels' collinearity guard on the step-1 fit: an (almost)
    // perfect R^2 means the residual has no signal left and the test is
    // meaningless. statsmodels warns and returns -inf / p = 0, which reads
    // like overwhelming evidence of cointegration; say so instead.
    let ssr: f64 = resid.iter().map(|&e| e * e).sum();
    let tss: f64 = if trend.has_constant() {
        let mean = (0..n).map(|i| y[(i, 0)]).sum::<f64>() / n as f64;
        (0..n).map(|i| (y[(i, 0)] - mean).powi(2)).sum()
    } else {
        (0..n).map(|i| y[(i, 0)] * y[(i, 0)]).sum()
    };
    // NaN (a constant regressand makes `tss` zero) counts as degenerate.
    let rsquared = 1.0 - ssr / tss;
    if rsquared.is_nan() || rsquared >= EG_MAX_RSQUARED {
        return Err(CointError::Singular {
            what: "the Engle-Granger step-1 regression fits (almost) perfectly, so \
                   the series are collinear rather than cointegrated and the \
                   residual test carries no information — drop the redundant \
                   series (statsmodels instead warns and returns p = 0, which \
                   reads like overwhelming evidence of cointegration)",
        });
    }

    // Step 2: unit-root test on the residuals, no deterministic term.
    let resid_adf = adf(&resid, AdfRegression::NoConstant, lags)?;
    let stat = resid_adf.statistic;

    // The residual statistic lives on the MacKinnon cointegration surfaces
    // indexed by N = k, not on the standard (N = 1) ADF surfaces. Critical
    // values are evaluated at T - 1, as statsmodels does.
    let po = trend.po_trend();
    let p_value = mackinnon_coint_p(stat, po, k);
    let crit = mackinnon_coint_crit(po, k, n.saturating_sub(1));

    Ok(EngleGrangerResult {
        stat,
        p_value,
        crit,
        n_vars: k,
        nobs: n,
        trend,
        coint_coefs,
        resid,
        resid_adf,
    })
}
