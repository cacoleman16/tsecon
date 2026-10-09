//! Python bindings for the residual-normality family of `tsecon-var`
//! (0.11.0): the `multivariate_normality` entry point on a residual matrix,
//! and the shared key builders `var_diagnostics` uses for the symmetric-root
//! Jarque-Bera (JMulTi's "Doornik & Hansen (1994)" block) and the
//! Doornik-Hansen (2008) omnibus test. Registered into `_core` through
//! [`register`].

use numpy::ndarray::Ix2;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tsecon_var::tsecon_linalg::faer::Mat;
use tsecon_var::{DoornikHansenTest, NormalityTest};

use crate::to_py;

/// The `*_symmetric` keys: the symmetric-root Jarque-Bera, or `None` per key
/// when it was not computed.
pub(crate) fn set_symmetric_keys(d: &Bound<'_, PyDict>, n: Option<&NormalityTest>) -> PyResult<()> {
    d.set_item("jarque_bera_symmetric", n.map(|n| n.statistic))?;
    d.set_item("jarque_bera_symmetric_pvalue", n.map(|n| n.pvalue))?;
    d.set_item("skewness_symmetric", n.map(|n| n.skewness))?;
    d.set_item("skewness_symmetric_pvalue", n.map(|n| n.skewness_pvalue))?;
    d.set_item("kurtosis_symmetric", n.map(|n| n.kurtosis))?;
    d.set_item("kurtosis_symmetric_pvalue", n.map(|n| n.kurtosis_pvalue))?;
    d.set_item(
        "skewness_symmetric_components",
        n.map(|n| n.skewness_components.clone()),
    )?;
    d.set_item(
        "kurtosis_symmetric_components",
        n.map(|n| n.kurtosis_components.clone()),
    )?;
    Ok(())
}

/// The `doornik_hansen*` keys, or `None` per key when the test was not
/// computed (fewer than 8 observations).
pub(crate) fn set_doornik_hansen_keys(
    d: &Bound<'_, PyDict>,
    t: Option<&DoornikHansenTest>,
) -> PyResult<()> {
    d.set_item("doornik_hansen", t.map(|t| t.statistic))?;
    d.set_item("doornik_hansen_pvalue", t.map(|t| t.pvalue))?;
    d.set_item("doornik_hansen_df", t.map(|t| t.df))?;
    d.set_item("doornik_hansen_skewness", t.map(|t| t.skewness))?;
    d.set_item(
        "doornik_hansen_skewness_pvalue",
        t.map(|t| t.skewness_pvalue),
    )?;
    d.set_item("doornik_hansen_kurtosis", t.map(|t| t.kurtosis))?;
    d.set_item(
        "doornik_hansen_kurtosis_pvalue",
        t.map(|t| t.kurtosis_pvalue),
    )?;
    d.set_item("doornik_hansen_z_skewness", t.map(|t| t.z_skewness.clone()))?;
    d.set_item("doornik_hansen_z_kurtosis", t.map(|t| t.z_kurtosis.clone()))?;
    Ok(())
}

/// The Cholesky (statsmodels) keys, named as in `var_diagnostics`.
fn set_cholesky_keys(d: &Bound<'_, PyDict>, n: &NormalityTest) -> PyResult<()> {
    d.set_item("jarque_bera", n.statistic)?;
    d.set_item("jarque_bera_pvalue", n.pvalue)?;
    d.set_item("jarque_bera_df", n.df)?;
    d.set_item("skewness", n.skewness)?;
    d.set_item("skewness_pvalue", n.skewness_pvalue)?;
    d.set_item("kurtosis", n.kurtosis)?;
    d.set_item("kurtosis_pvalue", n.kurtosis_pvalue)?;
    d.set_item("skewness_components", n.skewness_components.clone())?;
    d.set_item("kurtosis_components", n.kurtosis_components.clone())?;
    Ok(())
}

/// Residual normality tests on a residual matrix — the three statistics
/// `var_diagnostics` reports, for residuals from ANY multivariate model
/// (a VECM, a structural VAR, a factor model) or for one series.
///
/// `resid` is a `T x k` residual matrix, or a 1-D length-`T` series taken as
/// `k = 1`; it is column-centred here, and the covariance uses the ML divisor
/// `T`. Three statistics on the same centred residuals:
///
/// * CHOLESKY (the statsmodels `VARResults.test_normality` convention, and
///   JMulTi's block labelled "Lütkepohl (1993)"): orthogonalise with the lower
///   Cholesky factor of the residual covariance, then `skewness = T b1'b1 / 6`
///   and `kurtosis = T b2'b2 / 24` (each `chi2(k)`) add up to `jarque_bera`
///   (`chi2(2k)`, `jarque_bera_df = 2k`). Depends on the column order.
/// * SYMMETRIC ROOT (what JMulTi prints under "Doornik & Hansen (1994)"): the
///   same with the symmetric inverse square root `S^{-1/2}` instead of the
///   Cholesky factor and NO further transforms; invariant to the column order
///   and to sign flips but NOT to rescaling a column — it depends on units.
///   All 16 JMulTi outputs bundled with statsmodels 0.15 are reproduced to
///   their printed four decimals.
/// * DOORNIK-HANSEN (2008), the recommended test: the correlation-matrix
///   symmetric root, then D'Agostino's skewness transform and the
///   Shenton-Bowman / Wilson-Hilferty kurtosis transform per component, so the
///   z-scores are close to N(0, 1) in small samples; `doornik_hansen =
///   z1'z1 + z2'z2 ~ chi2(2k)`. Invariant to column order, rescaling, sign
///   flips and shifts. With a 1-D input it is the UNIVARIATE Doornik-Hansen
///   omnibus, better sized in small samples than the asymptotic
///   Jarque-Bera: the crate's seeded Monte Carlo (n = 50, 20 000 Gaussian
///   samples) rejects at 4.89% against 3.73% for `jarque_bera` at a nominal
///   5%.
///
/// At `k = 1` the three Jarque-Bera forms coincide and equal
/// `tsecon.jarque_bera(x)["statistic"]`, and `doornik_hansen_z_skewness[0]`
/// equals `scipy.stats.skewtest(x).statistic`.
///
/// Returned keys: the Cholesky block `jarque_bera`, `jarque_bera_pvalue`,
/// `jarque_bera_df`, `skewness`, `skewness_pvalue`, `kurtosis`,
/// `kurtosis_pvalue`, `skewness_components` (per-component `b1`),
/// `kurtosis_components` (per-component excess `b2`); the symmetric block
/// `jarque_bera_symmetric`, `jarque_bera_symmetric_pvalue`,
/// `skewness_symmetric`, `skewness_symmetric_pvalue`, `kurtosis_symmetric`,
/// `kurtosis_symmetric_pvalue`, `skewness_symmetric_components`,
/// `kurtosis_symmetric_components` (its df is `jarque_bera_df`); the
/// Doornik-Hansen block `doornik_hansen`, `doornik_hansen_pvalue`,
/// `doornik_hansen_df` (`2k`), `doornik_hansen_skewness` (`z1'z1`, `chi2(k)`),
/// `doornik_hansen_skewness_pvalue`, `doornik_hansen_kurtosis` (`z2'z2`,
/// `chi2(k)`), `doornik_hansen_kurtosis_pvalue`, `doornik_hansen_z_skewness`
/// and `doornik_hansen_z_kurtosis` (the per-component z-scores); `nobs`
/// (`T`), `k`.
///
/// Refusals (ValueError naming `resid`): fewer than 8 rows (the
/// Doornik-Hansen skewness transform degenerates at T = 7), `T <= k`, a 0-D
/// or 3-D (or higher) array, zero columns, a NaN or infinity, a constant
/// column, collinear columns (a correlation-matrix eigenvalue `<= 1e-12 x`
/// its trace), and columns on scales so different that the unit-dependent
/// symmetric root cannot be computed reliably.
///
/// Validation: JMulTi (independent program) — the symmetric block against
/// its "Doornik & Hansen (1994)" block, the Cholesky block against its
/// "Lütkepohl (1993)" block and the 1-D case against its per-variable
/// Jarque-Bera rows, at the printed 5e-5 on 16 residual matrices; the
/// Doornik-Hansen (2008) formulas transcribed in fixtures/var_diag.json at
/// 1e-10 (z1 matched to `scipy.stats.skewtest`; the kurtosis coefficients
/// checked by an exact identity).
#[pyfunction]
#[pyo3(signature = (resid))]
fn multivariate_normality<'py>(
    py: Python<'py>,
    resid: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let v = resid.as_array();
    let m = match v.ndim() {
        1 => Mat::from_fn(v.len(), 1, |i, _| v[[i]]),
        2 => {
            let v2 = v.into_dimensionality::<Ix2>().map_err(to_py)?;
            Mat::from_fn(v2.nrows(), v2.ncols(), |i, j| v2[(i, j)])
        }
        d => {
            return Err(PyValueError::new_err(format!(
                "resid must be a 2-D T x k residual matrix (one column per series) or a \
                 1-D length-T series (taken as k = 1); got {d} dimensions"
            )))
        }
    };
    let dh = tsecon_var::doornik_hansen_test(m.as_ref()).map_err(to_py)?;
    let chol = tsecon_var::normality_test_cholesky(m.as_ref()).map_err(to_py)?;
    let sym = tsecon_var::normality_test_symmetric(m.as_ref()).map_err(to_py)?;
    let d = PyDict::new(py);
    set_cholesky_keys(&d, &chol)?;
    set_symmetric_keys(&d, Some(&sym))?;
    set_doornik_hansen_keys(&d, Some(&dh))?;
    d.set_item("nobs", m.nrows())?;
    d.set_item("k", m.ncols())?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multivariate_normality, m)?)?;
    Ok(())
}
