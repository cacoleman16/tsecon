//! Python bindings for the 0.10.0 additions to `tsecon-var`:
//! `var_conditional_forecast` (hard-path conditional forecasts, the
//! Doan-Litterman-Sims / Waggoner-Zha closed form), `var_diagnostics`
//! (multivariate Portmanteau, multivariate Jarque-Bera, stability roots)
//! and `var_select_order` (the lag-order selection that was validated in
//! Rust since 0.1 but never bound). Registered into `_core` through
//! [`register`].

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{mat_to_vec2, parse_trend, to_py, var_results};

/// Conditional (hard-path) forecast of a VAR(p): the forecast of every
/// series when some cells of the future path are PINNED to given values —
/// "inflation follows this path for four quarters; what happens to output?"
///
/// `conditions` is a nested list with one row per horizon and one entry per
/// series (the data's column order): a number pins that `(horizon, series)`
/// cell, `None` (or `NaN`) leaves it free — so a NumPy array with NaN for the
/// free cells, or a pandas DataFrame with missing entries, works too. Rows
/// beyond `len(conditions)` up to `steps` are free, so a short list
/// conditions the near horizons only; `steps` defaults to `len(conditions)`.
/// At least one cell must be pinned (the all-free case is `var_forecast`).
///
/// Method (Doan-Litterman-Sims 1984; Waggoner-Zha 1999): with the coefficients
/// treated as known and Gaussian innovations, the conditional path is the
/// Gaussian conditioning of the joint forecast-error distribution on the
/// pinned cells, equivalently the unconditional path plus the response to the
/// MINIMUM-NORM sequence of future shocks that delivers the conditions. It is
/// also exactly what a Kalman smoother returns when the free future cells are
/// set missing (Bańbura-Giannone-Lenza 2015) — the second, independent golden
/// leg in fixtures/var_cf.json (statsmodels `VARMAX(...).smooth`).
///
/// Returned keys, every path a `steps x k` nested list (row `h` = horizon
/// `h + 1`): `point` (the conditional mean path; pinned cells hold their
/// condition exactly), `unconditional` (the plain iterated forecast — equal to
/// `var_forecast(...)["point"]` bitwise), `cov` (`[h][i][j]`: the conditional
/// forecast-error covariance at each horizon, `k x k`; pinned cells have a
/// zero row and column), `se` (`sqrt(diag cov)`, exactly 0 at pinned cells),
/// `unconditional_se` (the plain forecast's se, to read the variance
/// reduction), `lower`/`upper` (`point -/+ z_{1-alpha/2} se`, innovation
/// uncertainty only, coefficients treated as known — the `var_forecast`
/// convention), `shocks` (the implied reduced-form innovations `u*`, row `s`
/// = period `T+s+1`), `orth_shocks` (the same shocks orthogonalised by the
/// lower Cholesky factor of `sigma_u` in the data's column order — the
/// Waggoner-Zha structural shocks under a recursive ordering; ONLY this key
/// depends on the ordering), `constrained` (`steps x k` booleans),
/// `n_constrained`, `mahalanobis` (`r' (B Sigma B')^{-1} r`, the squared
/// Sigma-norm of the implied shocks: how far the conditions sit from the
/// unconditional forecast in the model's own metric), `mahalanobis_pvalue`
/// (its `chi2(n_constrained)` tail probability — small means the model must be
/// pushed hard to deliver the scenario), `steps`, `alpha`.
///
/// Validation: the closed form transcribed in NumPy (documented formula) and
/// statsmodels `VARMAX(...NaN future...).smooth(params)` (independent
/// package) — path, covariance and implied shocks — both pinned at 1e-10 /
/// 1e-8 in fixtures/var_cf.json on a seeded VAR(2) and a transformed US
/// macro system; the unconditional case equals `var_forecast` bitwise;
/// coverage of the free series under the DGP measured by seeded Monte Carlo
/// (see the model card).
///
/// Further arguments, with defaults: `lags` (2), `trend` ("c"), `steps`
/// (None: `len(conditions)`), `alpha` (0.05).
#[pyfunction]
#[pyo3(signature = (data, conditions, lags = 2, trend = "c", steps = None, alpha = 0.05))]
fn var_conditional_forecast<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    conditions: Vec<Vec<Option<f64>>>,
    lags: usize,
    trend: &str,
    steps: Option<usize>,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = var_results(&data, lags, trend)?;
    let steps = match steps {
        Some(s) => s,
        None => {
            if conditions.is_empty() {
                return Err(PyValueError::new_err(
                    "conditions is empty and steps is None: pass conditions as a nested \
                     list with one row per forecast horizon (None or NaN for a free cell, \
                     a number for a pinned one), or pass steps explicitly",
                ));
            }
            conditions.len()
        }
    };
    let cf = r
        .conditional_forecast(steps, &conditions, alpha)
        .map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("point", mat_to_vec2(&cf.point))?;
    d.set_item("unconditional", mat_to_vec2(&cf.unconditional))?;
    let cov: Vec<Vec<Vec<f64>>> = cf.cov.iter().map(mat_to_vec2).collect();
    d.set_item("cov", cov)?;
    d.set_item("se", mat_to_vec2(&cf.se))?;
    d.set_item("unconditional_se", mat_to_vec2(&cf.unconditional_se))?;
    d.set_item("lower", mat_to_vec2(&cf.lower))?;
    d.set_item("upper", mat_to_vec2(&cf.upper))?;
    d.set_item("shocks", mat_to_vec2(&cf.shocks))?;
    d.set_item("orth_shocks", mat_to_vec2(&cf.orth_shocks))?;
    d.set_item("constrained", cf.constrained.clone())?;
    d.set_item("n_constrained", cf.n_constrained)?;
    d.set_item("mahalanobis", cf.mahalanobis)?;
    d.set_item("mahalanobis_pvalue", cf.mahalanobis_pvalue)?;
    d.set_item("steps", cf.steps)?;
    d.set_item("alpha", cf.alpha)?;
    Ok(d)
}

/// Residual diagnostics of a fitted VAR(p) — the three checks every VAR
/// write-up reports: residual autocorrelation, residual normality, and
/// stability.
///
/// PORTMANTEAU (Hosking 1980; Lütkepohl 2005, section 4.4.3): with `C_i` the
/// lag-`i` autocovariances of the column-centred residuals, the unadjusted
/// `Q = T sum_{i=1}^{nlags} tr(C_i' C_0^{-1} C_i C_0^{-1})` and the
/// small-sample adjusted `T^2 sum_i tr(...) / (T - i)`, both `chi2(k^2
/// (nlags - p))` under white residuals; `nlags` must exceed the lag order
/// `lags`. Report the adjusted statistic: it corrects the unadjusted one's
/// small-sample undersize (the crate's seeded Monte Carlo at T = 200, k = 2,
/// nlags = 8, 1000 replications measures 4.6% for Q and 5.4% adjusted at a
/// nominal 5%, MC se 0.7 points; the gap widens as nlags grows relative to
/// T). Matches statsmodels `VARResults.test_whiteness(nlags,
/// adjusted=False/True)`.
///
/// NORMALITY (multivariate Jarque-Bera; Lütkepohl 2005, section 4.5): the
/// centred residuals are orthogonalised by the LOWER CHOLESKY factor of their
/// ML covariance — the statsmodels `test_normality` convention, so the
/// skewness/kurtosis components depend on the column order, as
/// orthogonalised impulse responses do — and `skewness = T b1'b1 / 6`,
/// `kurtosis = T b2'b2 / 24` (each `chi2(k)`) add up to the omnibus
/// `jarque_bera` (`chi2(2k)`). The Doornik-Hansen variant (symmetric square
/// root, transformed moments) is NOT provided: no runnable reference exists
/// in the fixture environment.
///
/// STABILITY: `roots` are the moduli of the reciprocal characteristic roots
/// in DESCENDING order (statsmodels `VARResults.roots`; stable iff the LAST
/// one exceeds 1), `eigenvalue_moduli` the companion eigenvalue moduli in
/// descending order (the first is the spectral radius; stable iff it is
/// below 1), and `is_stable` the verdict.
///
/// Returned keys: `portmanteau`, `portmanteau_adjusted`, `portmanteau_df`,
/// `portmanteau_pvalue`, `portmanteau_adjusted_pvalue`, `nlags`,
/// `jarque_bera`, `jarque_bera_pvalue`, `jarque_bera_df`, `skewness`,
/// `skewness_pvalue`, `kurtosis`, `kurtosis_pvalue`, `skewness_components`
/// (the per-series third moments `b1`), `kurtosis_components` (the
/// per-series excess fourth moments `b2`), `roots`, `eigenvalue_moduli`,
/// `is_stable`, `nobs` (the effective sample size `T` the residual tests
/// use), `k`, `lags`.
///
/// Validation: statsmodels `test_whiteness` / `test_normality` / `roots` /
/// `is_stable` pinned at 1e-10 (p-values also on the log scale) in
/// fixtures/var_diag.json on seeded VAR(2)/VAR(3)/underfit VAR(1) systems,
/// a Student-t-driven VAR(2) and a transformed US macro system; size and
/// power measured by seeded Monte Carlo (see the model card).
///
/// Further arguments, with defaults: `lags` (2), `trend` ("c"), `nlags`
/// (10).
#[pyfunction]
#[pyo3(signature = (data, lags = 2, trend = "c", nlags = 10))]
fn var_diagnostics<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    lags: usize,
    trend: &str,
    nlags: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let r = var_results(&data, lags, trend)?;
    let dg = r.diagnostics(nlags).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("portmanteau", dg.portmanteau.statistic)?;
    d.set_item("portmanteau_adjusted", dg.portmanteau.adjusted)?;
    d.set_item("portmanteau_df", dg.portmanteau.df)?;
    d.set_item("portmanteau_pvalue", dg.portmanteau.pvalue)?;
    d.set_item(
        "portmanteau_adjusted_pvalue",
        dg.portmanteau.adjusted_pvalue,
    )?;
    d.set_item("nlags", dg.portmanteau.nlags)?;
    d.set_item("jarque_bera", dg.normality.statistic)?;
    d.set_item("jarque_bera_pvalue", dg.normality.pvalue)?;
    d.set_item("jarque_bera_df", dg.normality.df)?;
    d.set_item("skewness", dg.normality.skewness)?;
    d.set_item("skewness_pvalue", dg.normality.skewness_pvalue)?;
    d.set_item("kurtosis", dg.normality.kurtosis)?;
    d.set_item("kurtosis_pvalue", dg.normality.kurtosis_pvalue)?;
    d.set_item(
        "skewness_components",
        dg.normality.skewness_components.clone(),
    )?;
    d.set_item(
        "kurtosis_components",
        dg.normality.kurtosis_components.clone(),
    )?;
    d.set_item("roots", dg.roots.clone())?;
    d.set_item("eigenvalue_moduli", dg.eigenvalue_moduli.clone())?;
    d.set_item("is_stable", dg.is_stable)?;
    d.set_item("nobs", dg.nobs)?;
    d.set_item("k", dg.neqs)?;
    d.set_item("lags", dg.lags)?;
    Ok(d)
}

/// VAR lag-order selection by information criteria on a COMMON sample
/// (statsmodels `VAR.select_order` conventions; Lütkepohl 2005, section
/// 4.3): every candidate order `p` is fitted after dropping the first
/// `max_lags - p` rows, so all candidates share the same `n - max_lags`
/// effective observations and their criteria are comparable — which is why
/// this is not the same as reading `aic` off `var_fit` at each `lags`. The
/// candidates run from `p = 0` (an intercept-only baseline) with `trend="c"`
/// and from `p = 1` with `trend="n"`; ties go to the smaller order.
///
/// Returned keys: `aic`, `bic`, `hqic`, `fpe` (the selected order under each
/// criterion), `candidates` (the candidate orders, ascending), `aic_values`,
/// `bic_values`, `hqic_values`, `fpe_values` (one value per candidate),
/// `max_lags`, `trend`.
///
/// Validation: statsmodels `VAR.select_order` picks (fixtures/var.json,
/// since 0.1) and the full criterion tables (fixtures/var_diag.json) at 1e-8.
///
/// Further arguments, with defaults: `max_lags` (8), `trend` ("c").
#[pyfunction]
#[pyo3(signature = (data, max_lags = 8, trend = "c"))]
fn var_select_order<'py>(
    py: Python<'py>,
    data: numpy::PyReadonlyArray2<'py, f64>,
    max_lags: usize,
    trend: &str,
) -> PyResult<Bound<'py, PyDict>> {
    use tsecon_var::tsecon_linalg::faer::Mat;
    if max_lags == 0 {
        return Err(PyValueError::new_err(
            "max_lags = 0: the lag-order search needs at least one candidate order; pass \
             max_lags >= 1 (8 is a common quarterly choice)",
        ));
    }
    let tr = parse_trend(trend)?;
    let a = data.as_array();
    // `select_order` reserves one candidate per order up front, so a
    // `max_lags` typo would ask the allocator for terabytes before the
    // per-candidate estimation could refuse it. A candidate order at or
    // beyond the sample length can never be estimated anyway (it leaves
    // `n - max_lags <= 0` common observations), so refuse it here, naming
    // the parameter and the row count.
    if max_lags >= a.nrows() {
        return Err(PyValueError::new_err(format!(
            "max_lags = {max_lags} with only {} rows of data: every candidate order is \
             fitted on the common sample of n - max_lags = {} observations, so max_lags \
             must be smaller than the number of rows (and in practice far smaller — it \
             also has to leave more observations than k * max_lags + 1 coefficients per \
             equation); pass a smaller max_lags",
            a.nrows(),
            a.nrows() as i128 - max_lags as i128,
        )));
    }
    let m = Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
    let sel = tsecon_var::select_order(m.as_ref(), max_lags, tr).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("aic", sel.aic)?;
    d.set_item("bic", sel.bic)?;
    d.set_item("hqic", sel.hqic)?;
    d.set_item("fpe", sel.fpe)?;
    let cands: Vec<usize> = sel.candidates.iter().map(|c| c.lags).collect();
    d.set_item("candidates", cands)?;
    d.set_item(
        "aic_values",
        sel.candidates.iter().map(|c| c.aic).collect::<Vec<f64>>(),
    )?;
    d.set_item(
        "bic_values",
        sel.candidates.iter().map(|c| c.bic).collect::<Vec<f64>>(),
    )?;
    d.set_item(
        "hqic_values",
        sel.candidates.iter().map(|c| c.hqic).collect::<Vec<f64>>(),
    )?;
    d.set_item(
        "fpe_values",
        sel.candidates.iter().map(|c| c.fpe).collect::<Vec<f64>>(),
    )?;
    d.set_item("max_lags", max_lags)?;
    d.set_item("trend", trend)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(var_conditional_forecast, m)?)?;
    m.add_function(wrap_pyfunction!(var_diagnostics, m)?)?;
    m.add_function(wrap_pyfunction!(var_select_order, m)?)?;
    Ok(())
}
