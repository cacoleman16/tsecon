//! Python bindings for the serial-correlation tests of `tsecon-spectest`:
//! `breusch_godfrey_test` (the Breusch-Godfrey LM test, statsmodels
//! `acorr_breusch_godfrey` convention) and `durbin_watson_test` (the
//! Durbin-Watson statistic with its exact Imhof p-value). Both take the
//! `(y, x)` regression of the specification-test family — `x` the `T x k`
//! design INCLUDING the constant column. Registered into `_core` through
//! [`register`].

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tsecon_spectest::{DwAlternative, SpecTestError};

use crate::{to_py, vec1};

/// Boundary validation of the `(y, x)` regression shared by both tests, with
/// refusals that name the offending argument. Returns `y` and the columns of
/// `x`.
fn regression_inputs(
    y: &PyReadonlyArray1<'_, f64>,
    x: &PyReadonlyArray2<'_, f64>,
) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let ys = vec1(y);
    let a = x.as_array();
    let (rows, cols) = a.dim();
    if ys.is_empty() {
        return Err(PyValueError::new_err(
            "y is empty: pass the regression's dependent variable, one value per \
             observation",
        ));
    }
    if cols == 0 {
        return Err(PyValueError::new_err(
            "x has no columns: pass the T x k design of the regression, including the \
             constant column (np.ones(T)) as one of its columns",
        ));
    }
    if rows != ys.len() {
        return Err(PyValueError::new_err(format!(
            "x has {rows} rows but y has {} observations: x must be the T x k design \
             with one row per observation of y",
            ys.len()
        )));
    }
    if let Some(i) = ys.iter().position(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "y contains a non-finite value (NaN or inf) at index {i}; the test does not \
             skip missing values — drop the incomplete observations from y and x first"
        )));
    }
    if let Some((i, j)) = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| (i, j)))
        .find(|&(i, j)| !a[(i, j)].is_finite())
    {
        return Err(PyValueError::new_err(format!(
            "x contains a non-finite value (NaN or inf) at row {i}, column {j}; the test \
             does not skip missing values — drop the incomplete observations from y and x \
             first"
        )));
    }
    let x_cols = (0..cols).map(|j| a.column(j).to_vec()).collect();
    Ok((ys, x_cols))
}

/// Rebuild a crate error into a `ValueError` that names the argument at
/// fault (`y`, `x`, or `nlags`); `k_x` is the number of columns of `x`.
fn named_error(e: SpecTestError, k_x: usize) -> PyErr {
    match e {
        SpecTestError::DegreesOfFreedom { what, n, k } if what.contains("auxiliary") => {
            PyValueError::new_err(format!(
                "nlags is too long for the sample: the Breusch-Godfrey auxiliary regression \
                 has k_aux = {k} regressors (the {k_x} columns of x, the nlags lagged \
                 residuals, and an added constant when x spans none) but y has only \
                 n = {n} observations; it needs n > k_aux — pass a smaller nlags"
            ))
        }
        SpecTestError::DegreesOfFreedom { what, n, .. } if what.contains("Durbin-Watson") => {
            PyValueError::new_err(format!(
                "y is too short for the Durbin-Watson test: n = {n} observations with the \
                 {k_x} columns of x leave fewer than two residual degrees of freedom (it \
                 needs len(y) >= x.shape[1] + 2; with n = k + 1 the statistic is a \
                 constant)"
            ))
        }
        SpecTestError::DegreesOfFreedom { n, .. } => PyValueError::new_err(format!(
            "y is too short for x: n = {n} observations with k = {k_x} columns of x leave \
             no residual degrees of freedom (the regression needs len(y) > x.shape[1])"
        )),
        SpecTestError::SingularDesign { what } if what.contains("auxiliary") => {
            PyValueError::new_err(
                "the Breusch-Godfrey auxiliary design [x, lagged residuals] is singular \
                 (collinear columns): a lagged-residual column is a linear combination of \
                 the columns of x — pass a smaller nlags, or check x for regressors that \
                 replicate the residual lags",
            )
        }
        SpecTestError::SingularDesign { .. } => PyValueError::new_err(
            "x is rank deficient (collinear columns): the OLS normal equations have no \
             unique solution — drop a redundant column of x (e.g. a second constant or a \
             full set of dummies alongside the constant)",
        ),
        SpecTestError::DegenerateResponse { what } => PyValueError::new_err(format!(
            "y leaves degenerate residuals on x, so there is no serial correlation to test \
             ({what})"
        )),
        SpecTestError::SampleTooLong { n, max, .. } => PyValueError::new_err(format!(
            "y has {n} observations, beyond the {max} the exact Durbin-Watson p-value \
             supports: the exact null distribution eigen-decomposes a T x T matrix, which \
             is cubic in T (measured in the model card); for a long sample use \
             breusch_godfrey_test, whose cost is linear in T and whose chi-square p-value \
             is accurate at that length"
        )),
        SpecTestError::NumericalBreakdown { what } => PyValueError::new_err(format!(
            "the exact Durbin-Watson p-value could not be computed for this y and x \
             (numerical breakdown in the {what}); check x for nearly collinear columns"
        )),
        other => to_py(other),
    }
}

/// Breusch-Godfrey LM test for serial correlation of order up to `nlags` in
/// the errors of the OLS regression of `y` on `x` (`T x k`, INCLUDING the
/// constant column, as for the other specification tests).
///
/// The residuals `u` are regressed on `x` and their own lags
/// `u_{t-1}..u_{t-nlags}`; `statistic` is `LM = T * R^2` of that auxiliary
/// regression, `chi2(nlags)` under no serial correlation, and `fstat` the
/// F-form of the same null (the lag coefficients jointly zero),
/// `F(nlags, T - k_aux)`. Unlike Durbin-Watson it allows lagged dependent
/// variables among the regressors and tests every order up to `nlags`.
///
/// Convention (statsmodels `acorr_breusch_godfrey`, pinned at 1e-10): the
/// lagged residuals are padded with zeros, so all `T` observations stay in
/// the auxiliary regression; `nlags=None` uses `min(10, T // 5)`; when `x`
/// spans no constant (no constant column and no implicit one such as
/// complementary dummies) a column of ones is added to the auxiliary
/// regression (`constant_added`), so `k_aux = k + nlags + 1`, otherwise
/// `k + nlags`. With `constant_added` the LM form also counts what that
/// constant explains and over-rejects (measured in the model card): read the
/// F-form (`f_pvalue`) there.
///
/// Returned keys: `constant_added`, `df`, `f_df_den`, `f_df_num`, `f_pvalue`,
/// `fstat`, `nlags`, `nobs`, `pvalue`, `statistic`.
/// `statistic`/`df`/`pvalue` are the LM form (`df = nlags`), `fstat` with
/// `f_df_num` (`= nlags`) and `f_df_den` (`= T - k_aux`) and `f_pvalue` the
/// F-form, `nlags` the order used (the default resolved), `nobs` = `T`.
///
/// Further arguments, with defaults: `nlags` (None: `min(10, T // 5)`).
#[pyfunction]
#[pyo3(signature = (y, x, nlags = None))]
fn breusch_godfrey_test<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    nlags: Option<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let nlags = match nlags {
        None => None,
        Some(p) if p < 1 => {
            return Err(PyValueError::new_err(format!(
                "nlags must be a positive integer (the highest residual lag in the \
                 auxiliary regression; None uses min(10, T // 5)); got {p}"
            )))
        }
        Some(p) => Some(usize::try_from(p).map_err(|_| {
            PyValueError::new_err(format!("nlags = {p} does not fit this platform's usize"))
        })?),
    };
    let (ys, cols) = regression_inputs(&y, &x)?;
    let r = tsecon_spectest::breusch_godfrey_test(&ys, &cols, nlags)
        .map_err(|e| named_error(e, cols.len()))?;
    let d = PyDict::new(py);
    d.set_item("statistic", r.statistic)?;
    d.set_item("df", r.df)?;
    d.set_item("pvalue", r.pvalue)?;
    d.set_item("fstat", r.fstat)?;
    d.set_item("f_df_num", r.f_df_num)?;
    d.set_item("f_df_den", r.f_df_den)?;
    d.set_item("f_pvalue", r.f_pvalue)?;
    d.set_item("nlags", r.nlags)?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("constant_added", r.constant_added)?;
    Ok(d)
}

/// Durbin-Watson test for first-order serial correlation in the errors of
/// the OLS regression of `y` on `x` (`T x k`, INCLUDING the constant
/// column), with the EXACT p-value under normal errors.
///
/// `statistic` is `d = sum (e_t - e_{t-1})^2 / sum e_t^2` on the OLS
/// residuals (statsmodels `durbin_watson`, pinned at 1e-10): about
/// `2 (1 - r_1)`, below 2 for positive and above 2 for negative
/// autocorrelation. `pvalue` is exact, not a bounds-table verdict: under
/// normal iid errors and fixed regressors `P(d <= c) = P(sum_i (nu_i - c)
/// z_i^2 <= 0)` with `nu_i` the eigenvalues of `M A M` (Durbin-Watson 1950),
/// evaluated by Imhof's (1961) inversion formula to about 1e-13 absolute
/// (so a p-value printed as ~1e-14 means "below 1e-13"). `alternative` is
/// `"positive"` (rho > 0, the classical test: `P(d <= d_obs)`),
/// `"negative"` (`P(d >= d_obs)`) or `"two_sided"`
/// (`min(1, 2 min(...))`).
///
/// Validity: exact for Gaussian errors with strictly exogenous `x`; with
/// non-normal errors it is approximate (seeded Monte Carlo in the model
/// card); with a lagged dependent variable in `x` it is INVALID — use
/// `breusch_godfrey_test`. COST: the eigendecomposition behind the p-value is
/// cubic in `T`, so the p-value is computed for `T <= 4000` (measured
/// timings in the model card); for a longer sample `pvalue` is `None` and the
/// statistic, which is linear in `T`, is still returned. Use
/// `breusch_godfrey_test` for a test at that length.
///
/// Returned keys: `alternative`, `df_resid`, `nobs`, `pvalue`, `statistic`.
/// `df_resid` is `T - k`, `nobs` is `T`, `alternative` echoes the option,
/// `pvalue` is `None` when `T > 4000`.
///
/// Validation: the exact p-value is a documented-formula golden (LAPACK
/// eigenvalues + QUADPACK, fixtures/serial_corr.json, 1e-10), the null
/// distribution is checked by seeded Monte Carlo, and the `nu_i` obey the
/// Durbin-Watson bounds (interlacing) theorem.
///
/// Further arguments, with defaults: `alternative` ("positive").
#[pyfunction]
#[pyo3(signature = (y, x, alternative = "positive"))]
fn durbin_watson_test<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    alternative: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let alt = match alternative {
        "positive" => DwAlternative::Positive,
        "negative" => DwAlternative::Negative,
        "two_sided" => DwAlternative::TwoSided,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown alternative {other:?}; expected \"positive\" (rho > 0, the \
                 classical one-sided test), \"negative\" (rho < 0) or \"two_sided\""
            )))
        }
    };
    let (ys, cols) = regression_inputs(&y, &x)?;
    let d = PyDict::new(py);
    if ys.len() > tsecon_spectest::DW_EXACT_MAX_NOBS {
        // The exact p-value is cubic in T; the statistic is not.
        let (statistic, nobs, df_resid) =
            tsecon_spectest::durbin_watson_regression_statistic(&ys, &cols)
                .map_err(|e| named_error(e, cols.len()))?;
        d.set_item("statistic", statistic)?;
        d.set_item("pvalue", py.None())?;
        d.set_item("alternative", alt.name())?;
        d.set_item("nobs", nobs)?;
        d.set_item("df_resid", df_resid)?;
        return Ok(d);
    }
    let r = tsecon_spectest::durbin_watson_test(&ys, &cols, alt)
        .map_err(|e| named_error(e, cols.len()))?;
    d.set_item("statistic", r.statistic)?;
    d.set_item("pvalue", r.pvalue)?;
    d.set_item("alternative", r.alternative.name())?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("df_resid", r.df_resid)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(breusch_godfrey_test, m)?)?;
    m.add_function(wrap_pyfunction!(durbin_watson_test, m)?)?;
    Ok(())
}
