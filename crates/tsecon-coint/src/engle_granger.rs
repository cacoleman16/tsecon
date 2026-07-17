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
//! Engle-Granger / Phillips-Ouliaris critical values (which depend on the
//! number of regressors and the deterministic terms; MacKinnon 2010). Only
//! the ADF *statistic* returned here is meaningful for a cointegration
//! decision; the `p_value` and `crit` fields carried inside the inner
//! [`tsecon_diag::AdfResult`] are the standard-ADF values and must not be
//! used to decide cointegration.
//!
//! `// TODO(phase0)`: ship the Engle-Granger / Phillips-Ouliaris
//! response-surface critical values and p-values so this test can return a
//! self-contained decision. The single-equation statistic and the
//! cointegrating vector are the load-bearing outputs today.

use tsecon_diag::{adf, AdfLagSelection, AdfRegression, AdfResult};
use tsecon_linalg::faer::linalg::solvers::SolveLstsq;
use tsecon_linalg::faer::{Mat, MatRef};

use crate::error::CointError;
use crate::linalg::check_finite;

/// Deterministic terms in the step-1 cointegrating regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngleGrangerTrend {
    /// Constant only (statsmodels `coint(trend = "c")`, the default).
    Constant,
    /// Constant and a linear time trend (statsmodels `trend = "ct"`).
    ConstantTrend,
}

impl EngleGrangerTrend {
    fn n_det(self) -> usize {
        match self {
            EngleGrangerTrend::Constant => 1,
            EngleGrangerTrend::ConstantTrend => 2,
        }
    }
}

/// Result of the Engle-Granger two-step procedure.
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
    /// Coefficients of the step-1 cointegrating regression of series 0 on
    /// the deterministics and the remaining series, in design order
    /// `[deterministics..., series_1, ..., series_{k-1}]`. The implied
    /// cointegrating vector on the levels is `[1, -coef_on_series_1, ...]`.
    pub coint_coefs: Vec<f64>,
    /// Step-1 OLS residuals (length `T`), the series tested for a unit root.
    pub resid: Vec<f64>,
    /// The residual augmented Dickey-Fuller result. **Only** its
    /// `statistic` is valid for a cointegration decision (see the module
    /// note); its `p_value`/`crit` are standard-ADF, not Engle-Granger.
    pub resid_adf: AdfResult,
}

impl EngleGrangerResult {
    /// The Engle-Granger test statistic: the augmented Dickey-Fuller
    /// `tau` on the step-1 residuals. More negative is stronger evidence
    /// of cointegration.
    pub fn statistic(&self) -> f64 {
        self.resid_adf.statistic
    }
}

/// Runs the Engle-Granger two-step test on `endog` (a `T x k` matrix,
/// oldest row first; series 0 is the regressand). The residual ADF uses no
/// deterministic term (statsmodels `regression = "n"`) because the step-1
/// residuals are mean zero by construction.
///
/// # Errors
///
/// * [`CointError::Dimension`] if `endog` has fewer than two columns;
/// * [`CointError::NonFinite`] if `endog` contains a NaN or infinity;
/// * [`CointError::Singular`] if the step-1 design is collinear;
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
            what: "Engle-Granger needs at least two series",
            expected: 2,
            got: k,
        });
    }
    check_finite(endog, "endog")?;
    let n = endog.nrows();
    let n_det = trend.n_det();
    let n_reg = n_det + (k - 1);

    // Step 1: design [deterministics, series 1..k-1], response series 0.
    let mut x = Mat::<f64>::zeros(n, n_reg);
    for i in 0..n {
        let mut col = 0;
        x[(i, col)] = 1.0;
        col += 1;
        if trend == EngleGrangerTrend::ConstantTrend {
            x[(i, col)] = i as f64;
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
            what: "Engle-Granger step-1 design",
        }
    })?;
    let fitted = &x * &b;
    let resid: Vec<f64> = (0..n).map(|i| y[(i, 0)] - fitted[(i, 0)]).collect();
    let coint_coefs: Vec<f64> = (0..n_reg).map(|r| b[(r, 0)]).collect();

    // Step 2: unit-root test on the residuals, no deterministic term.
    let resid_adf = adf(&resid, AdfRegression::NoConstant, lags)?;

    Ok(EngleGrangerResult {
        coint_coefs,
        resid,
        resid_adf,
    })
}
