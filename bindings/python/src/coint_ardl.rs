//! Python binding for the ARDL / unrestricted error-correction model and
//! the Pesaran-Shin-Smith (2001) bounds test of `tsecon-coint`
//! (`ardl_bounds_test`). Registered into `_core` through [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use tsecon_coint::{ArdlOptions, DolsIc, PssCase};

use crate::{to_py, vec1};

/// Marks an `order` given as one int (broadcast to every regressor once
/// the column count is known). A list entry can never equal it: the
/// converter refuses entries at or beyond [`ORDER_CAP`].
const BROADCAST: usize = usize::MAX;

/// Orders at or beyond 2**48 are refused by name (the coercion layer
/// draws the same line for every count argument).
const ORDER_CAP: i64 = 1 << 48;

fn x_to_faer(x: &numpy::PyReadonlyArray2<'_, f64>) -> tsecon_coint::tsecon_linalg::faer::Mat<f64> {
    let a = x.as_array();
    tsecon_coint::tsecon_linalg::faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
}

fn nested<'py>(py: Python<'py>, m: &[Vec<f64>]) -> PyResult<Bound<'py, PyList>> {
    let outer = PyList::empty(py);
    for row in m {
        outer.append(PyList::new(py, row.iter().copied())?)?;
    }
    Ok(outer)
}

/// One integer order: an int (or NumPy integer), not a bool or a float,
/// in `0 ..= 2**48`; `what` names it in the refusal.
fn one_order(item: &Bound<'_, PyAny>, what: &str) -> PyResult<usize> {
    let shown = item
        .repr()
        .map(|r| r.to_string())
        .unwrap_or_else(|_| "?".to_string());
    if item.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(format!(
            "{what} = {shown} is a bool, not an order: pass an integer q_j >= 1"
        )));
    }
    if item.is_instance_of::<PyFloat>() || item.is_instance_of::<PyString>() {
        return Err(PyValueError::new_err(format!(
            "{what} = {shown} is not an integer: an ARDL order is an integer number of lags \
             (q_j >= 1), e.g. order = [1, 3, 2] or order = 2"
        )));
    }
    let v: i64 = match item.extract() {
        Ok(v) => v,
        // a Python int that does not fit i64 is far beyond any lag count
        Err(_) if item.is_instance_of::<PyInt>() => {
            return Err(PyValueError::new_err(format!(
                "{what} = {shown} is at or beyond 2**48 — no sample has that many lags; \
                 pass a small integer order"
            )))
        }
        Err(_) => {
            return Err(PyValueError::new_err(format!(
                "{what} = {shown} is not an integer order: pass integers q_j >= 1, e.g. \
                 order = [1, 3, 2] or order = 2"
            )))
        }
    };
    if v < 0 {
        return Err(PyValueError::new_err(format!(
            "{what} = {v} is negative: an ARDL order counts lags, so pass q_j >= 1"
        )));
    }
    if v >= ORDER_CAP {
        return Err(PyValueError::new_err(format!(
            "{what} = {v} is at or beyond 2**48 — no sample has that many lags; pass a \
             small integer order"
        )));
    }
    Ok(v as usize)
}

/// `order`: None, one int (broadcast), or a sequence of ints (a list, a
/// tuple, a NumPy integer array, a pandas Series). An int is returned as
/// `[BROADCAST, q]` and expanded once the column count is known.
fn order_arg(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<usize>>> {
    if obj.is_none() {
        return Ok(None);
    }
    // a Python int / bool / float, or a NumPy integer scalar (indexable,
    // no length): one order for every regressor (refused by name if not
    // a nonnegative int)
    let scalar = obj.is_instance_of::<PyInt>()
        || obj.is_instance_of::<PyFloat>()
        || (obj.extract::<i64>().is_ok() && obj.len().is_err());
    if scalar {
        let q = one_order(obj, "order")?;
        return Ok(Some(vec![BROADCAST, q]));
    }
    if obj.is_instance_of::<PyString>() {
        return Err(PyValueError::new_err(format!(
            "order = {} is a string: pass an int (one order for every regressor) or a list of \
             ints, one per column of x",
            obj.repr()?
        )));
    }
    let iter = obj.try_iter().map_err(|_| {
        PyValueError::new_err(format!(
            "order = {} is neither an int nor a sequence of ints: pass e.g. order = 2 or \
             order = [1, 3, 2] (one per column of x)",
            obj.repr().map(|r| r.to_string()).unwrap_or_default()
        ))
    })?;
    let mut out = Vec::new();
    for (j, item) in iter.enumerate() {
        out.push(one_order(&item?, &format!("order[{j}]"))?);
    }
    Ok(Some(out))
}

/// ARDL model and Pesaran-Shin-Smith (2001) bounds test for a level
/// relationship between `y` and the regressors `x`, valid whether the
/// regressors are I(0), I(1) or mutually cointegrated.
///
/// `y` (length T) is the regressand and `x` the (T, k) regressors, k >= 1
/// (do NOT add a constant: the deterministics come from `case`). The
/// model is the ARDL(p, q_1..q_k) in its unrestricted error-correction
/// form (statsmodels `UECM`): `D.y_t = d_t'delta + pi y_{t-1} + sum_j
/// theta_j x_{j,t-1} + sum_{i<p} phi_i D.y_{t-i} + sum_j sum_{l<q_j}
/// beta_{j,l} D.x_{j,t-l} + e_t`, OLS over rows t = h..T-1 with h =
/// max(p, max q_j), p >= 1 and every q_j >= 1. `case` (default 3) picks the
/// deterministics and what the test restricts: 1 none; 2 a constant,
/// tested with the levels; 3 an unrestricted constant; 4 a constant and a
/// trend, the trend tested; 5 an unrestricted constant and trend. The
/// trend is t + 1 over the FULL sample (statsmodels' convention).
///
/// The bounds test is the Wald F on the level block (`y.L1`, `x.L1`, plus
/// the constant in case 2 or the trend in case 4) divided by the number
/// of restrictions. Its critical values and p-values are the asymptotic
/// PSS response surface (statsmodels' 32-million-draw tables) for k
/// regressors, I(0) (lower) and I(1) (upper) bounds — looked up at k =
/// `n_x`; statsmodels 0.15's `bounds_test` uses k = n_x + 1 (its own
/// simulator, run at the key it passes, reproduces the n_x row), so its
/// critical values and p-values are wrong and tsecon does not inherit
/// them. `verdict` reads the p-values at `alpha` (default 0.05):
/// "level_relationship" if `p_value_upper` < alpha,
/// "no_level_relationship" if `p_value_lower` >= alpha, else
/// "inconclusive" ("unavailable" when n_x > 10, where no table exists and
/// `crit_note` says so). Because the surface p at a tabulated critical
/// value is off its nominal level by up to 0.001, the verdict can differ
/// from comparing `statistic` with `crit_lower`/`crit_upper` in that
/// sliver; the p-values are the definition. The t statistic on y.L1 is
/// returned without critical values: a rejecting F with a t near zero is
/// PSS's degenerate case (the x levels matter but y does not
/// error-correct) — check it. Asymptotic bounds only: with T below about
/// 80 they can mislead (no Narayan / Kripfganz-Schneider finite-sample
/// surfaces, no t bounds, no HAC version yet).
///
/// Orders: `lags` (p >= 1) and `order` (an int for every regressor, or
/// one q_j >= 1 per column; q_j = 0 is a planned follow-up) fix them; a
/// None is searched by minimising `ic` — "bic" (default), "aic" or "hqic"
/// on statsmodels' scale -2 llf + c (m + 1) (NOT `dols`'s ln(RSS/n)
/// scale) — over p in 1..`max_lag` and every q_j in 1..`max_order`
/// (defaults 4) on the common sample h* = max(P, Q), the winner refit on
/// its own sample. This is statsmodels' `ardl_select_order` restricted to
/// the orders a UECM can represent (its unrestricted pick can have q_j = 0
/// or drop a regressor, which its own `UECM.from_ardl` then refuses).
/// Sentinels: `ic` with both `lags` and `order` fixed RAISES, `max_lag`
/// with `lags` fixed RAISES, `max_order` with `order` fixed RAISES (each
/// would be inert); `lags` = 0 RAISES (statsmodels silently builds the
/// lags = 1 design); a grid of more than 100 000 candidate models
/// (P x Q^k) RAISES naming `max_lag`/`max_order` — fix `order` to search
/// only the lags (measured: 65 536 models at seven regressors, T = 300,
/// about 17 s); so does a design of more than 2^26 entries (512 MiB).
///
/// Fit statistics use the regression's own row count `nobs` = T - h;
/// statsmodels' ARDL/UECM report T - p whenever max q_j > p, which shifts
/// their llf, information criteria, df_resid and t-test p-values (their
/// standard errors are right).
///
/// Returned keys: `statistic` (the PSS F), `t_statistic` (t on y.L1, no
/// critical values), `case`, `trend` ("n"/"c"/"ct"), `n_x` (the table
/// key k), `n_restrictions`, `crit_levels` ([0.90, 0.95, 0.99, 0.999]),
/// `crit_lower` (I(0) bounds), `crit_upper` (I(1) bounds),
/// `p_value_lower`, `p_value_upper`, `verdict`, `alpha`; the UECM:
/// `params`, `se`, `tvalues`, `pvalues` (Student t on `df_resid`), `cov`,
/// `param_names` ([const, trend, y.L1, x1.L1..xk.L1, D.y.L1..D.y.L{p-1},
/// D.x1.L0..D.x1.L{q1-1}, ...]), `adjustment` (the coefficient on y.L1),
/// `n_det`, `n_params`; the long run (statsmodels `ci_params` times -1,
/// without y): `long_run`, `long_run_se` (delta method),
/// `long_run_tvalues`, `long_run_pvalues` (normal), `long_run_names`
/// (const, trend, x1..xk); the fit: `resid` (length `nobs`, D.y - Z b),
/// `nobs`, `n_total` (T), `hold_back` (h), `df_resid`, `ssr`, `llf`,
/// `aic`, `bic`, `hqic` (-2 llf + c (m + 1) on the model's own sample);
/// the orders: `lags`, `order` (list of q_j), `selected` (False when both
/// were fixed), `ic`, `ic_value` (the minimised criterion on the common
/// sample; NaN when both were fixed), `max_lag`, `max_order` (the caps
/// used; the fixed values when fixed), `n_models` (candidates compared);
/// `crit_note` (None, or why the bounds are NaN).
///
/// Validation: statsmodels 0.15 `UECM` / `ARDL` / `ardl_select_order` at
/// 1e-10 on estimates, F (two routes, plus an exact-rational referee), t,
/// long run and selection; the critical values and p-values are
/// statsmodels' tabulated surface at the corrected key (p-values atol
/// 1e-14 / rtol 1e-9); seeded Monte Carlo null size 0.048 / 0.047 at 5%
/// (fixtures/ardl.json, model card).
///
/// Further arguments, with defaults: `lags` (None), `order` (None), `case`
/// (3), `ic` (None: "bic"), `max_lag` (None: 4), `max_order` (None: 4),
/// `alpha` (0.05).
#[pyfunction]
#[pyo3(signature = (y, x, lags = None, order = None, case = 3, ic = None, max_lag = None, max_order = None, alpha = 0.05))]
#[allow(clippy::too_many_arguments)]
fn ardl_bounds_test<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: numpy::PyReadonlyArray2<'py, f64>,
    lags: Option<usize>,
    #[pyo3(from_py_with = order_arg)] order: Option<Vec<usize>>,
    case: usize,
    ic: Option<&str>,
    max_lag: Option<usize>,
    max_order: Option<usize>,
    alpha: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let pss_case = PssCase::from_index(case).ok_or_else(|| {
        PyValueError::new_err(format!(
            "case = {case} is not a Pesaran-Shin-Smith case: pass 1 (no deterministics), 2 \
             (a constant restricted to the level relation), 3 (an unrestricted constant, the \
             default), 4 (an unrestricted constant and a restricted trend) or 5 (an \
             unrestricted constant and trend)"
        ))
    })?;
    let kx = x.as_array().ncols();
    let order = order.map(|v| {
        if v.len() == 2 && v[0] == BROADCAST {
            vec![v[1]; kx]
        } else {
            v
        }
    });
    let both_fixed = lags.is_some() && order.is_some();
    let ic_rule = match ic {
        None => DolsIc::Bic,
        Some(code) => {
            let rule = DolsIc::parse(code).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown ic {code:?}; expected \"aic\", \"bic\" or \"hqic\""
                ))
            })?;
            if both_fixed {
                return Err(PyValueError::new_err(format!(
                    "ic = {code:?} was given but both lags = {} and order are fixed: the \
                     criterion only acts when an order is searched, so it would be inert; drop \
                     ic or leave lags/order as None",
                    lags.unwrap_or(0)
                )));
            }
            rule
        }
    };
    let opts = ArdlOptions {
        lags,
        order,
        case: pss_case,
        ic: ic_rule,
        max_lag,
        max_order,
        alpha,
    };
    let xm = x_to_faer(&x);
    let yv = vec1(&y);
    let r = tsecon_coint::ardl_bounds_test(&yv, xm.as_ref(), &opts).map_err(to_py)?;
    let d = PyDict::new(py);
    // bounds test
    d.set_item("statistic", r.statistic)?;
    d.set_item("t_statistic", r.t_statistic)?;
    d.set_item("case", r.case.index())?;
    d.set_item("trend", r.case.trend_code())?;
    d.set_item("n_x", r.n_x)?;
    d.set_item("n_restrictions", r.n_restrictions)?;
    d.set_item("crit_levels", r.crit_levels.to_vec().into_pyarray(py))?;
    d.set_item("crit_lower", r.crit_lower.to_vec().into_pyarray(py))?;
    d.set_item("crit_upper", r.crit_upper.to_vec().into_pyarray(py))?;
    d.set_item("p_value_lower", r.p_value_lower)?;
    d.set_item("p_value_upper", r.p_value_upper)?;
    d.set_item("verdict", r.verdict.code())?;
    d.set_item("alpha", r.alpha)?;
    // UECM
    d.set_item("params", r.params.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    d.set_item("tvalues", r.tvalues.into_pyarray(py))?;
    d.set_item("pvalues", r.pvalues.into_pyarray(py))?;
    d.set_item("cov", nested(py, &r.cov)?)?;
    d.set_item("param_names", PyList::new(py, r.param_names)?)?;
    d.set_item("adjustment", r.adjustment)?;
    d.set_item("n_det", r.n_det)?;
    d.set_item("n_params", r.n_params)?;
    // long run
    d.set_item("long_run", r.long_run.into_pyarray(py))?;
    d.set_item("long_run_se", r.long_run_se.into_pyarray(py))?;
    d.set_item("long_run_tvalues", r.long_run_tvalues.into_pyarray(py))?;
    d.set_item("long_run_pvalues", r.long_run_pvalues.into_pyarray(py))?;
    d.set_item("long_run_names", PyList::new(py, r.long_run_names)?)?;
    // fit
    d.set_item("resid", r.resid.into_pyarray(py))?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("n_total", r.n_total)?;
    d.set_item("hold_back", r.hold_back)?;
    d.set_item("df_resid", r.df_resid)?;
    d.set_item("ssr", r.ssr)?;
    d.set_item("llf", r.llf)?;
    d.set_item("aic", r.aic)?;
    d.set_item("bic", r.bic)?;
    d.set_item("hqic", r.hqic)?;
    // orders
    d.set_item("lags", r.lags)?;
    d.set_item("order", PyList::new(py, r.order)?)?;
    d.set_item("selected", r.selected)?;
    d.set_item("ic", r.ic.code())?;
    d.set_item("ic_value", r.ic_value)?;
    d.set_item("max_lag", r.max_lag)?;
    d.set_item("max_order", r.max_order)?;
    d.set_item("n_models", r.n_models)?;
    d.set_item("crit_note", r.crit_note)?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ardl_bounds_test, m)?)?;
    Ok(())
}
