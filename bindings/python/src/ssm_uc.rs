//! Python bindings for the structural-model layer of `tsecon-ssm`:
//! `unobserved_components` (Harvey's structural time-series models by
//! exact-diffuse MLE) and `tvp_regression` (random-walk-coefficient
//! regression with the pile-up check). Registered into `_core` through
//! [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tsecon_ssm::{FreqSeasonalSpec, TrendSpec, TvpOptions, UcComponent, UcOptions, UcSpec};

use crate::{to_py, vec1};

/// Columns of a 2-D array (accepts non-contiguous input).
fn columns(a: &PyReadonlyArray2<'_, f64>) -> Vec<Vec<f64>> {
    let m = a.as_array();
    (0..m.ncols()).map(|j| m.column(j).to_vec()).collect()
}

fn nested<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for r in rows {
        out.append(PyList::new(py, r.iter().copied())?)?;
    }
    Ok(out)
}

fn set_component<'py>(
    py: Python<'py>,
    d: &Bound<'py, PyDict>,
    name: &str,
    c: Option<&UcComponent>,
) -> PyResult<()> {
    match c {
        None => {
            d.set_item(name, py.None())?;
            d.set_item(format!("{name}_var"), py.None())?;
            d.set_item(format!("filtered_{name}"), py.None())?;
            d.set_item(format!("filtered_{name}_var"), py.None())?;
        }
        Some(c) => {
            d.set_item(name, c.smoothed.clone().into_pyarray(py))?;
            d.set_item(
                format!("{name}_var"),
                c.smoothed_var.clone().into_pyarray(py),
            )?;
            d.set_item(
                format!("filtered_{name}"),
                c.filtered.clone().into_pyarray(py),
            )?;
            d.set_item(
                format!("filtered_{name}_var"),
                c.filtered_var.clone().into_pyarray(py),
            )?;
        }
    }
    Ok(())
}

/// Harvey's structural time-series ("unobserved components") models by
/// exact-diffuse maximum likelihood — level/trend, dummy and trigonometric
/// seasonals, a (damped) stochastic cycle, and regressors:
///
///     y_t = mu_t + gamma_t + c_t + beta' x_t + eps_t
///
/// with every state initialized exactly diffuse (Koopman 1997), NaN in
/// `y` treated as missing, and the components assembled exactly as
/// statsmodels' `UnobservedComponents(..., use_exact_diffuse=True)`
/// enumerates them (the log-likelihoods are directly comparable).
///
/// `level` picks the level/trend block by its statsmodels name (long or
/// short form): "irregular"/"ntrend", "fixed intercept", "deterministic
/// constant"/"dconstant", "local level"/"llevel" (default), "random
/// walk"/"rwalk", "fixed slope", "deterministic trend"/"dtrend", "local
/// linear deterministic trend"/"lldtrend", "random walk with
/// drift"/"rwdrift", "local linear trend"/"lltrend", "smooth
/// trend"/"strend", "random trend"/"rtrend". `seasonal=s` adds an `s-1`-
/// state dummy seasonal (`stochastic_seasonal`, default True, gives it a
/// variance; False makes it fixed dummies). `freq_seasonal=[p, ...]`
/// adds trigonometric seasonals with `freq_seasonal_harmonics` harmonics
/// each (default `floor(p/2)`) and `stochastic_freq_seasonal` flags
/// (default True each). `cycle=True` adds the stochastic cycle:
/// `damped_cycle` (default False) estimates a damping in (0, 1),
/// `stochastic_cycle` (default False) gives it a variance, and
/// `cycle_period_bounds=[min, max]` confines its frequency to
/// `(2 pi/max, 2 pi/min)`. Its default is `[2, len(y)]`, and an infinite
/// `max` is read the same way: under exact-diffuse initialization the
/// log-likelihood of a stochastic cycle DIVERGES as the frequency goes to
/// zero (the second cycle state becomes weakly observable and its diffuse
/// resolution contributes `-ln(lambda)`), so an unbounded period is not a
/// safe search region — and a cycle longer than the sample is not
/// identified in any case. statsmodels leaves that bound at infinity.
/// `exog` (T x k) enters with
/// time-invariant coefficients estimated jointly (statsmodels
/// `mle_regression=True`); `forecast_steps=h` returns h-step forecasts
/// and needs `forecast_exog` (h x k) when `exog` is given.
/// `fixed_params` (statsmodels order: `sigma2.irregular`, the state
/// variances in component order, `frequency.cycle`, `damping.cycle`,
/// `beta.x1`...) evaluates the model there instead of estimating.
/// `n_starts` (default 3) is the deterministic start ladder. Counts that
/// size an allocation are bounded and refuse rather than abort: `seasonal`
/// and each `freq_seasonal` period at most `len(y)` (they cost states, and
/// a period the sample never completes is not identified), `forecast_steps`
/// at most 100000, `n_starts` at most 64.
///
/// Options that act only under a component RAISE when passed without it:
/// `stochastic_seasonal` without `seasonal`; `freq_seasonal_harmonics` /
/// `stochastic_freq_seasonal` without `freq_seasonal`; `damped_cycle` /
/// `stochastic_cycle` / `cycle_period_bounds` without `cycle=True`;
/// `forecast_exog` without `exog` or without `forecast_steps`.
///
/// Estimation: BFGS + Nelder-Mead on the exact prediction-error
/// log-likelihood in statsmodels' square-root/logistic working space,
/// scale-adaptive (y standardized, mapped back exactly). A variance whose
/// estimate cannot be told from zero (zeroing it costs < 1e-4
/// log-likelihood — the pile-up) is flagged in `at_boundary` with a NaN
/// standard error; `se` are observed-information (numerical Hessian)
/// CONDITIONAL on the flagged parameters sitting exactly at their
/// boundary, i.e. the information matrix is inverted over the free
/// parameters only. statsmodels' `cov_type="approx"` inverts the full
/// matrix instead, including the boundary directions where it is
/// indefinite, so the two agree exactly when nothing is flagged and
/// differ by definition when something is.
/// `aic`/`bic` use statsmodels' `k_params + k_diffuse` degrees of freedom.
/// The component keys without a prefix are the SMOOTHED (two-sided)
/// paths; `filtered_*` are the one-sided ones. Variances inside the
/// diffuse period are the finite part; `std_resid` is NaN there and at
/// missing periods.
///
/// Validation (honest grade): fixed-parameter log-likelihood, filtered
/// states and variances, smoothed states, residuals, forecasts and
/// forecast variances pinned at 1e-8 against statsmodels for 26 component
/// combinations and NaN-inserted series (fixtures/uc.json); the SMOOTHED
/// variances at 1e-8 too, except inside the diffuse period of the
/// hardest combinations, where the exact-diffuse smoother is
/// ill-conditioned and the tolerance is the distance between
/// statsmodels' own two smoother implementations (up to 4.5e-3 there,
/// against filters that agree to 2.9e-11); the MLE
/// pinned to the better of statsmodels' own fit and a SciPy re-
/// optimization of the identical criterion (two optimizers); the
/// Durbin-Koopman (2012) Nile local level reproduced to the book's printed
/// precision; the Harvey-Durbin (1986) UK seat-belt BSM re-estimated to
/// that same optimum, with its slope and seasonal variance pile-ups
/// flagged (the series is fetched from Rdatasets when the fixture is
/// generated and never redistributed, so only its derived optimum is
/// stored); parameter recovery, forecast-interval coverage and scale
/// invariance measured by seeded Monte Carlo (see the model card).
///
/// Returned keys: `trend_specification`, `param_names`, `params`, `se`,
/// `at_boundary`, `loglik`, `aic`, `bic`, `nobs`, `nobs_observed`,
/// `nobs_diffuse`, `k_states`, `k_diffuse`, `k_params`, `estimated`,
/// `converged`, `n_iter`, `n_fevals`, `state_names`, `filtered_state`,
/// `filtered_state_var`, `smoothed_state`, `smoothed_state_var` (nested
/// lists, nobs x k_states), `level`, `level_var`, `filtered_level`,
/// `filtered_level_var`, `slope`, `slope_var`, `filtered_slope`,
/// `filtered_slope_var`, `seasonal`, `seasonal_var`, `filtered_seasonal`,
/// `filtered_seasonal_var`, `cycle`, `cycle_var`, `filtered_cycle`,
/// `filtered_cycle_var` (None when the component is absent),
/// `freq_seasonal`, `freq_seasonal_var`, `filtered_freq_seasonal`,
/// `filtered_freq_seasonal_var` (lists with one array per block), `fitted`,
/// `resid`, `std_resid`, `forecast`, `forecast_var`.
#[pyfunction]
#[pyo3(signature = (y, level = "llevel", seasonal = None, stochastic_seasonal = None, freq_seasonal = None, freq_seasonal_harmonics = None, stochastic_freq_seasonal = None, cycle = false, damped_cycle = None, stochastic_cycle = None, cycle_period_bounds = None, exog = None, forecast_steps = 0, forecast_exog = None, fixed_params = None, n_starts = 3))]
#[allow(clippy::too_many_arguments)]
fn unobserved_components<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    level: &str,
    seasonal: Option<usize>,
    stochastic_seasonal: Option<bool>,
    freq_seasonal: Option<Vec<f64>>,
    freq_seasonal_harmonics: Option<Vec<usize>>,
    stochastic_freq_seasonal: Option<Vec<bool>>,
    cycle: bool,
    damped_cycle: Option<bool>,
    stochastic_cycle: Option<bool>,
    cycle_period_bounds: Option<Vec<f64>>,
    exog: Option<PyReadonlyArray2<'py, f64>>,
    forecast_steps: usize,
    forecast_exog: Option<PyReadonlyArray2<'py, f64>>,
    fixed_params: Option<Vec<f64>>,
    n_starts: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if seasonal.is_none() && stochastic_seasonal.is_some() {
        return Err(PyValueError::new_err(
            "stochastic_seasonal was given but seasonal is None: it only switches the \
             dummy seasonal's variance on or off, so without seasonal it is inert; pass \
             seasonal=<period> or drop stochastic_seasonal",
        ));
    }
    if freq_seasonal.is_none() {
        if freq_seasonal_harmonics.is_some() {
            return Err(PyValueError::new_err(
                "freq_seasonal_harmonics was given but freq_seasonal is None: it only sets \
                 the harmonics of the trigonometric seasonal blocks, so without \
                 freq_seasonal it is inert; pass freq_seasonal=[period, ...] or drop it",
            ));
        }
        if stochastic_freq_seasonal.is_some() {
            return Err(PyValueError::new_err(
                "stochastic_freq_seasonal was given but freq_seasonal is None: it only \
                 switches the trigonometric seasonal variances, so without freq_seasonal \
                 it is inert; pass freq_seasonal=[period, ...] or drop it",
            ));
        }
    }
    if !cycle {
        if damped_cycle.is_some() {
            return Err(PyValueError::new_err(
                "damped_cycle was given but cycle=False: it only adds a damping factor to \
                 the cycle, so without cycle=True it is inert; pass cycle=True or drop it",
            ));
        }
        if stochastic_cycle.is_some() {
            return Err(PyValueError::new_err(
                "stochastic_cycle was given but cycle=False: it only switches the cycle \
                 variance on, so without cycle=True it is inert; pass cycle=True or drop it",
            ));
        }
        if cycle_period_bounds.is_some() {
            return Err(PyValueError::new_err(
                "cycle_period_bounds was given but cycle=False: it only bounds the cycle \
                 frequency, so without cycle=True it is inert; pass cycle=True or drop it",
            ));
        }
    }
    let trend = TrendSpec::parse(level).map_err(to_py)?;
    let mut freq_specs = Vec::new();
    if let Some(periods) = &freq_seasonal {
        if let Some(h) = &freq_seasonal_harmonics {
            if h.len() != periods.len() {
                return Err(PyValueError::new_err(format!(
                    "freq_seasonal_harmonics has {} entries but freq_seasonal has {} periods; \
                     give one harmonics count per period",
                    h.len(),
                    periods.len()
                )));
            }
        }
        if let Some(s) = &stochastic_freq_seasonal {
            if s.len() != periods.len() {
                return Err(PyValueError::new_err(format!(
                    "stochastic_freq_seasonal has {} entries but freq_seasonal has {} \
                     periods; give one flag per period",
                    s.len(),
                    periods.len()
                )));
            }
        }
        for (i, &p) in periods.iter().enumerate() {
            let mut f = FreqSeasonalSpec::new(p);
            if let Some(h) = &freq_seasonal_harmonics {
                f.harmonics = h[i];
            }
            if let Some(s) = &stochastic_freq_seasonal {
                f.stochastic = s[i];
            }
            freq_specs.push(f);
        }
    }
    let bounds = match &cycle_period_bounds {
        None => (2.0, f64::INFINITY),
        Some(b) if b.len() == 2 => (b[0], b[1]),
        Some(b) => {
            return Err(PyValueError::new_err(format!(
                "cycle_period_bounds must be [min, max] (2 entries), got {} entries",
                b.len()
            )))
        }
    };
    let spec = UcSpec {
        trend,
        seasonal,
        stochastic_seasonal: stochastic_seasonal.unwrap_or(true),
        freq_seasonal: freq_specs,
        cycle,
        damped_cycle: damped_cycle.unwrap_or(false),
        stochastic_cycle: stochastic_cycle.unwrap_or(false),
        cycle_period_bounds: bounds,
        exog: exog.as_ref().map(columns).unwrap_or_default(),
    };
    let opts = UcOptions {
        forecast_steps,
        forecast_exog: forecast_exog.as_ref().map(columns).unwrap_or_default(),
        fixed_params,
        n_starts,
    };
    let ys = vec1(&y);
    let r = tsecon_ssm::unobserved_components(&ys, &spec, &opts).map_err(to_py)?;

    let d = PyDict::new(py);
    d.set_item("trend_specification", r.trend_specification)?;
    d.set_item("param_names", PyList::new(py, r.param_names)?)?;
    d.set_item("params", r.params.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    d.set_item("at_boundary", PyList::new(py, r.at_boundary)?)?;
    d.set_item("loglik", r.loglik)?;
    d.set_item("aic", r.aic)?;
    d.set_item("bic", r.bic)?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("nobs_observed", r.nobs_observed)?;
    d.set_item("nobs_diffuse", r.nobs_diffuse)?;
    d.set_item("k_states", r.k_states)?;
    d.set_item("k_diffuse", r.k_diffuse)?;
    d.set_item("k_params", r.k_params)?;
    d.set_item("estimated", r.estimated)?;
    d.set_item("converged", r.converged)?;
    d.set_item("n_iter", r.n_iter)?;
    d.set_item("n_fevals", r.n_fevals)?;
    d.set_item("state_names", PyList::new(py, r.state_names)?)?;
    d.set_item("filtered_state", nested(py, &r.filtered_state)?)?;
    d.set_item("filtered_state_var", nested(py, &r.filtered_state_var)?)?;
    d.set_item("smoothed_state", nested(py, &r.smoothed_state)?)?;
    d.set_item("smoothed_state_var", nested(py, &r.smoothed_state_var)?)?;
    set_component(py, &d, "level", r.level.as_ref())?;
    set_component(py, &d, "slope", r.slope.as_ref())?;
    set_component(py, &d, "seasonal", r.seasonal.as_ref())?;
    set_component(py, &d, "cycle", r.cycle.as_ref())?;
    let fs = PyList::empty(py);
    let fs_var = PyList::empty(py);
    let ffs = PyList::empty(py);
    let ffs_var = PyList::empty(py);
    for c in &r.freq_seasonal {
        fs.append(c.smoothed.clone().into_pyarray(py))?;
        fs_var.append(c.smoothed_var.clone().into_pyarray(py))?;
        ffs.append(c.filtered.clone().into_pyarray(py))?;
        ffs_var.append(c.filtered_var.clone().into_pyarray(py))?;
    }
    d.set_item("freq_seasonal", fs)?;
    d.set_item("freq_seasonal_var", fs_var)?;
    d.set_item("filtered_freq_seasonal", ffs)?;
    d.set_item("filtered_freq_seasonal_var", ffs_var)?;
    d.set_item("fitted", r.fitted.into_pyarray(py))?;
    d.set_item("resid", r.resid.into_pyarray(py))?;
    d.set_item("std_resid", r.std_resid.into_pyarray(py))?;
    d.set_item("forecast", r.forecast.into_pyarray(py))?;
    d.set_item("forecast_var", r.forecast_var.into_pyarray(py))?;
    Ok(d)
}

/// Regression with random-walk (time-varying) coefficients by
/// exact-diffuse maximum likelihood, with the pile-up check:
///
///     y_t = x_t' beta_t + eps_t,   beta_{t+1} = beta_t + eta_t,
///     eps_t ~ N(0, sigma2_eps),    eta_t ~ N(0, diag(sigma2_beta)),
///     beta_1 diffuse.
///
/// `x` is T x k (`constant=True`, the default, prepends a random-walk
/// intercept). NaN in `y` is a missing period; `x` must be finite. The
/// observation variance and the k coefficient-innovation variances are
/// estimated by BFGS + Nelder-Mead on the exact prediction-error
/// log-likelihood (square-root working space, scale-adaptive, `n_starts`
/// deterministic starts, default 3); `fixed_params=[sigma2_eps,
/// sigma2_beta_1, ..., sigma2_beta_k]` evaluates the filter there instead
/// — with every state variance 0 it is recursive least squares
/// (statsmodels `RecursiveLS`), the expanding-window OLS path.
///
/// The pile-up problem (Shephard-Harvey 1990; Stock-Watson 1998): a
/// random-walk-coefficient variance whose MLE cannot be told from zero
/// (zeroing it costs < 1e-4 log-likelihood) is flagged in `pile_up` (and
/// `at_boundary`) and gets a NaN standard error, so a coefficient the
/// data cannot show moving is not reported as moving by a tiny amount.
/// `se` are observed-information (numerical Hessian) conditional on the
/// flagged variances being exactly zero; `aic`/`bic` use
/// statsmodels' `k_params + k_diffuse` degrees of freedom; `std_resid` is
/// NaN inside the diffuse period (the first k informative observations)
/// and at missing periods.
///
/// Validation (honest grade): fixed-parameter log-likelihood, filtered
/// and smoothed coefficient paths and variances, residuals pinned at 1e-8
/// against a statsmodels `MLEModel` transcription of the documented
/// state-space form and against `RecursiveLS` in the zero-variance limit
/// (its filtered coefficients and concentrated log-likelihood); the MLE
/// pinned to the better of statsmodels' fit and a SciPy re-optimization
/// of the same criterion (two optimizers) with the true-zero variance's
/// pile-up flagged; the pile-up frequency on constant versus moving
/// coefficients measured by seeded Monte Carlo (model card).
///
/// Returned keys: `coef_names`, `k`, `param_names`, `params`, `se`,
/// `at_boundary`, `sigma2_eps`, `sigma2_beta`, `pile_up`, `loglik`,
/// `aic`, `bic`, `nobs`, `nobs_observed`, `nobs_diffuse`, `k_params`,
/// `estimated`, `converged`, `n_iter`, `n_fevals`, `beta_filtered`,
/// `beta_filtered_var`, `beta_smoothed`, `beta_smoothed_var` (nested
/// lists, nobs x k), `fitted`, `resid`, `std_resid`.
#[pyfunction]
#[pyo3(signature = (y, x, constant = true, fixed_params = None, n_starts = 3))]
fn tvp_regression<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    constant: bool,
    fixed_params: Option<Vec<f64>>,
    n_starts: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let ys = vec1(&y);
    let cols = columns(&x);
    let opts = TvpOptions {
        constant,
        fixed_params,
        n_starts,
    };
    let r = tsecon_ssm::tvp_regression(&ys, &cols, &opts).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("coef_names", PyList::new(py, r.coef_names)?)?;
    d.set_item("k", r.k)?;
    d.set_item("param_names", PyList::new(py, r.param_names)?)?;
    d.set_item("params", r.params.into_pyarray(py))?;
    d.set_item("se", r.se.into_pyarray(py))?;
    d.set_item("at_boundary", PyList::new(py, r.at_boundary)?)?;
    d.set_item("sigma2_eps", r.sigma2_eps)?;
    d.set_item("sigma2_beta", r.sigma2_beta.into_pyarray(py))?;
    d.set_item("pile_up", PyList::new(py, r.pile_up)?)?;
    d.set_item("loglik", r.loglik)?;
    d.set_item("aic", r.aic)?;
    d.set_item("bic", r.bic)?;
    d.set_item("nobs", r.nobs)?;
    d.set_item("nobs_observed", r.nobs_observed)?;
    d.set_item("nobs_diffuse", r.nobs_diffuse)?;
    d.set_item("k_params", r.k_params)?;
    d.set_item("estimated", r.estimated)?;
    d.set_item("converged", r.converged)?;
    d.set_item("n_iter", r.n_iter)?;
    d.set_item("n_fevals", r.n_fevals)?;
    d.set_item("beta_filtered", nested(py, &r.beta_filtered)?)?;
    d.set_item("beta_filtered_var", nested(py, &r.beta_filtered_var)?)?;
    d.set_item("beta_smoothed", nested(py, &r.beta_smoothed)?)?;
    d.set_item("beta_smoothed_var", nested(py, &r.beta_smoothed_var)?)?;
    d.set_item("fitted", r.fitted.into_pyarray(py))?;
    d.set_item("resid", r.resid.into_pyarray(py))?;
    d.set_item("std_resid", r.std_resid.into_pyarray(py))?;
    Ok(d)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unobserved_components, m)?)?;
    m.add_function(wrap_pyfunction!(tvp_regression, m)?)?;
    Ok(())
}
