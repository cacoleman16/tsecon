//! Python bindings for `tsecon-ets`: `ets_fit` (one member of the
//! innovations state-space exponential-smoothing taxonomy, Hyndman et al.
//! 2008) and `auto_ets` (the information-criterion search over the
//! taxonomy behind R's `forecast::ets`). Registered into `_core` through
//! [`register`].

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tsecon_ets::{
    AutoEtsOptions, Component, ErrorType, EtsFit, EtsForecast, EtsParams, EtsSpec, EtsStates,
    FitOptions, Ic, Initialization, Optimizer,
};

use crate::{to_py, vec1};

const DEFAULT_N_SIM: usize = 5000;
const DEFAULT_LEVEL: f64 = 0.95;

fn error_type(s: &str) -> PyResult<ErrorType> {
    match s {
        "add" | "additive" => Ok(ErrorType::Additive),
        "mul" | "multiplicative" => Ok(ErrorType::Multiplicative),
        other => Err(PyValueError::new_err(format!(
            "error = {other:?} is invalid: pass \"add\" (additive innovations, y = mu + e) or \
             \"mul\" (multiplicative, y = mu (1 + e); needs y > 0)"
        ))),
    }
}

fn component(name: &str, s: Option<&str>) -> PyResult<Component> {
    match s {
        None => Ok(Component::None),
        Some("add") | Some("additive") => Ok(Component::Additive),
        Some("mul") | Some("multiplicative") => Ok(Component::Multiplicative),
        Some(other) => Err(PyValueError::new_err(format!(
            "{name} = {other:?} is invalid: pass None (no {name} component), \"add\" or \"mul\""
        ))),
    }
}

fn component_name(c: Component) -> Option<&'static str> {
    match c {
        Component::None => None,
        Component::Additive => Some("add"),
        Component::Multiplicative => Some("mul"),
    }
}

fn optimizer(s: Option<&str>) -> PyResult<Optimizer> {
    match s {
        None | Some("auto") => Ok(Optimizer::TwoStage),
        Some("nelder_mead") => Ok(Optimizer::NelderMead),
        Some("bfgs") => Ok(Optimizer::Bfgs),
        Some("lbfgs") => Ok(Optimizer::Lbfgs),
        Some(other) => Err(PyValueError::new_err(format!(
            "optimizer = {other:?} is invalid: pass \"auto\" (Nelder-Mead then a BFGS polish, \
             the better kept), \"nelder_mead\", \"bfgs\" or \"lbfgs\""
        ))),
    }
}

fn ic_of(s: &str) -> PyResult<Ic> {
    match s {
        "aicc" => Ok(Ic::Aicc),
        "aic" => Ok(Ic::Aic),
        "bic" => Ok(Ic::Bic),
        other => Err(PyValueError::new_err(format!(
            "ic = {other:?} is invalid: pass \"aicc\" (the default, Hyndman-Khandakar 2008), \
             \"aic\" or \"bic\""
        ))),
    }
}

/// Validates the forecast options against the horizon and (when known)
/// the spec's class; returns `(level, n_sim, seed)` with the effective
/// defaults.
fn forecast_options(
    horizon: usize,
    level: Option<f64>,
    n_sim: Option<usize>,
    seed: Option<u64>,
    class1: Option<bool>,
) -> PyResult<(f64, usize, u64)> {
    if horizon == 0 {
        if level.is_some() {
            return Err(PyValueError::new_err(
                "level was given but horizon=0 requests no forecast, so the interval level is \
                 inert; pass horizon=h (e.g. 12) or drop level",
            ));
        }
        if n_sim.is_some() {
            return Err(PyValueError::new_err(
                "n_sim was given but horizon=0 requests no forecast, so the simulation size is \
                 inert; pass horizon=h or drop n_sim",
            ));
        }
        if seed.is_some() {
            return Err(PyValueError::new_err(
                "seed was given but horizon=0 requests no forecast, so it is inert; pass \
                 horizon=h or drop seed",
            ));
        }
    } else if class1 == Some(true) {
        if n_sim.is_some() {
            return Err(PyValueError::new_err(
                "n_sim was given but this is a class-1 model (additive error, additive or no \
                 trend and seasonal), whose prediction intervals are the exact closed forms of \
                 Hyndman et al. (2008, Table 6.1) — no simulation runs, so n_sim is inert; drop \
                 it, or use a multiplicative component for simulated intervals",
            ));
        }
        if seed.is_some() {
            return Err(PyValueError::new_err(
                "seed was given but this is a class-1 model whose prediction intervals are exact \
                 closed forms — nothing is simulated, so seed is inert; drop it",
            ));
        }
    }
    Ok((
        level.unwrap_or(DEFAULT_LEVEL),
        n_sim.unwrap_or(DEFAULT_N_SIM),
        seed.unwrap_or(0),
    ))
}

/// Writes a fit (and its optional forecast) into a result dict.
fn fit_to_dict<'py>(
    py: Python<'py>,
    fit: &EtsFit,
    fc: Option<&EtsForecast>,
    horizon: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = &fit.spec;
    let d = PyDict::new(py);
    d.set_item("spec", spec.name())?;
    d.set_item("short_name", spec.short_name())?;
    d.set_item(
        "error",
        match spec.error {
            ErrorType::Additive => "add",
            ErrorType::Multiplicative => "mul",
        },
    )?;
    d.set_item("trend", component_name(spec.trend))?;
    d.set_item("damped", spec.damped)?;
    d.set_item("seasonal", component_name(spec.seasonal))?;
    d.set_item(
        "seasonal_periods",
        spec.has_seasonal().then_some(spec.seasonal_periods),
    )?;
    d.set_item("alpha", fit.params.alpha)?;
    d.set_item("beta", fit.params.beta)?;
    d.set_item("gamma", fit.params.gamma)?;
    d.set_item("phi", fit.params.phi)?;
    d.set_item("params", fit.params.to_vec().into_pyarray(py))?;
    d.set_item("param_names", spec.smoothing_names())?;
    d.set_item("initial_level", fit.initial_state.level)?;
    d.set_item("initial_trend", fit.initial_state.trend)?;
    d.set_item(
        "initial_seasonal",
        fit.initial_state
            .seasonal
            .clone()
            .map(|s| s.into_pyarray(py)),
    )?;
    d.set_item(
        "initial_states",
        fit.initial_state.to_vec().into_pyarray(py),
    )?;
    d.set_item("initial_state_names", spec.initial_state_names())?;
    d.set_item("initialization", fit.initialization)?;
    d.set_item("fitted", fit.fitted.clone().into_pyarray(py))?;
    d.set_item("resid", fit.resid.clone().into_pyarray(py))?;
    d.set_item("level_path", fit.level.clone().into_pyarray(py))?;
    d.set_item("trend_path", fit.trend.clone().map(|v| v.into_pyarray(py)))?;
    d.set_item(
        "seasonal_path",
        fit.seasonal.clone().map(|v| v.into_pyarray(py)),
    )?;
    d.set_item("final_level", fit.final_state.level)?;
    d.set_item("final_trend", fit.final_state.trend)?;
    d.set_item(
        "final_seasonal",
        fit.final_state.seasonal.clone().map(|s| s.into_pyarray(py)),
    )?;
    d.set_item("final_states", fit.final_state.to_vec().into_pyarray(py))?;
    d.set_item("loglik", fit.loglik)?;
    d.set_item("sigma2", fit.sigma2)?;
    d.set_item("nobs", fit.nobs)?;
    d.set_item("k_params", fit.k_params)?;
    d.set_item("aic", fit.aic)?;
    d.set_item("aicc", fit.aicc)?;
    d.set_item("bic", fit.bic)?;
    d.set_item("converged", fit.converged)?;
    d.set_item("n_iterations", fit.n_iterations)?;
    d.set_item("n_fevals", fit.n_fevals)?;
    d.set_item("optimizer", fit.optimizer)?;
    d.set_item("class1", spec.is_class1())?;
    d.set_item("horizon", horizon)?;
    match fc {
        Some(f) => {
            d.set_item("forecast", f.mean.clone().into_pyarray(py))?;
            d.set_item("forecast_variance", f.variance.clone().into_pyarray(py))?;
            d.set_item("forecast_lower", f.lower.clone().into_pyarray(py))?;
            d.set_item("forecast_upper", f.upper.clone().into_pyarray(py))?;
            d.set_item("interval_level", f.level)?;
            d.set_item("interval_method", f.method.name())?;
            d.set_item("n_sim", f.n_sim)?;
            d.set_item("seed", f.seed)?;
        }
        None => {
            for key in [
                "forecast",
                "forecast_variance",
                "forecast_lower",
                "forecast_upper",
                "interval_level",
                "interval_method",
                "n_sim",
                "seed",
            ] {
                d.set_item(key, py.None())?;
            }
        }
    }
    Ok(d)
}

/// Innovations state-space exponential smoothing — one member of the
/// ETS(Error, Trend, Seasonal) taxonomy of Hyndman, Koehler, Snyder &
/// Grose (2002) / Hyndman et al. (2008), fitted by maximum likelihood.
///
/// `error` is "add" or "mul"; `trend` and `seasonal` are None, "add" or
/// "mul"; `damped=True` damps the trend (estimates `phi`; refused without
/// a trend, where it would be inert); `seasonal_periods` is the period m
/// (12 monthly, 4 quarterly), required with a seasonal component and
/// refused without one. ETS(A,N,N) is simple exponential smoothing,
/// (A,A,N) Holt, (A,Ad,N) the damped trend, (A,A,A) / (M,A,M) the
/// additive / multiplicative Holt-Winters. Any multiplicative component
/// needs strictly positive `y` (refused otherwise, naming the offending
/// observation). NaN is refused: the innovations form conditions every
/// state on the observed error and has no missing-value mechanism
/// (R's ets refuses NaN too) — interpolate first, or use a Kalman-filter
/// model.
///
/// The smoothing parameters are Hyndman's `alpha`, `beta`, `gamma`, `phi`
/// (not beta* = beta/alpha or gamma* = gamma/(1 - alpha)), searched in the
/// traditional box 0 < alpha < 1, 0 < beta < alpha, 0 < gamma < 1 - alpha,
/// 0.8 <= phi <= 0.98 (R's and statsmodels' default bounds).
/// `initialization`: "estimated" (default: the initial states are free
/// parameters, started from the heuristic; the seasonal indices are
/// normalised to sum to zero / average one, R's convention, and count
/// m - 1 free parameters), "heuristic" (the Hyndman 2008 section 2.6.1
/// heuristic — a centred moving average over the first cycles and a
/// linear regression of its first ten values — held fixed; needs at
/// least 10 observations and, with a seasonal, 2m and 10 + 2 floor(m/2)),
/// or "known" (fixed at `initial_states`). `initial_states` is
/// `[level, trend?, seasonal[0], ..., seasonal[m-1]]` where
/// `seasonal[j]` is the index in force for observation j; it is required
/// with "known" and refused with the other two. `smoothing_params`
/// (`[alpha, beta?, gamma?, phi?]`, the components present, in that
/// order) evaluates the model at FIXED parameters with no optimisation —
/// statsmodels' `smooth(params)` — and needs initialization "heuristic"
/// or "known" (with "estimated" nothing would be estimated: refused);
/// `optimizer` and `max_iter` are then inert and refused if passed.
/// `optimizer` is "auto" (Nelder-Mead then a BFGS polish, the better
/// kept; the effective default), "nelder_mead", "bfgs" or "lbfgs";
/// `max_iter` caps each stage's iterations.
///
/// `horizon=h` adds h-step forecasts with `level` (0.95 when omitted)
/// prediction intervals: for the class-1 models — additive error with
/// additive or no trend and seasonal — the exact Gaussian intervals from
/// the closed-form variances of Hyndman et al. (2008, Table 6.1)
/// (`interval_method="exact"`); for every other model `n_sim` (5000 when
/// omitted) seeded innovation paths through the fitted recursion, the
/// bounds being empirical quantiles (`"simulated"`; `seed` 0 when
/// omitted). `level`, `n_sim` and `seed` are refused with `horizon=0`,
/// and `n_sim` / `seed` are refused for a class-1 model, where nothing is
/// simulated. The point forecast is always the zero-innovation path (R's
/// and statsmodels' convention).
///
/// Returned keys: `spec` (e.g. "ETS(A,Ad,N)"), `short_name` ("AAdN"),
/// `error`, `trend`, `damped`, `seasonal`, `seasonal_periods` (None
/// without a seasonal), `alpha`, `beta`, `gamma`, `phi` (None when the
/// component is absent), `params` and `param_names` (the packed
/// smoothing vector), `initial_level`, `initial_trend`,
/// `initial_seasonal` (None when absent), `initial_states` and
/// `initial_state_names` (packed), `initialization`, `fitted`
/// (one-step-ahead), `resid` (`y - fitted`, or `(y - fitted) / fitted`
/// under multiplicative errors), `level_path`, `trend_path`,
/// `seasonal_path` (the states after each update; None when absent),
/// `final_level`, `final_trend`, `final_seasonal` (the forecast anchor;
/// `final_seasonal[j]` is the index for forecast step j), `final_states`,
/// `loglik` (the concentrated Gaussian log-likelihood, statsmodels'
/// convention; R's `ets` omits the constant -(n/2)(ln(2 pi / n) + 1)),
/// `sigma2` (`mean(resid^2)`), `nobs`, `k_params` (smoothing parameters +
/// free initial states under "estimated" + sigma2), `aic`, `aicc`, `bic`,
/// `converged`, `n_iterations`, `n_fevals`, `optimizer` ("none" at fixed
/// parameters), `class1`, `horizon`, and — None when `horizon=0` —
/// `forecast`, `forecast_variance`, `forecast_lower`, `forecast_upper`,
/// `interval_level`, `interval_method`, `n_sim`, `seed`.
///
/// Validation (fixtures/ets.json): the twenty models without a
/// multiplicative seasonal are pinned at fixed parameters to statsmodels
/// `ETSModel` (log-likelihood, fitted values, states, forecasts,
/// simulations) at 1e-10, the six class-1 forecast variances to
/// statsmodels' exact `get_prediction` and to the Table 6.1 closed forms
/// at 1e-10 / 1e-12, the heuristic initialisation to
/// `holtwinters.ExponentialSmoothing` at 1e-10, and the maximum likelihood
/// to statsmodels' L-BFGS-B fit (match-or-beat, two optimizers); the ten
/// multiplicative-seasonal models are pinned at 1e-12 to an independent
/// transcription of the published recursion and their simulator to
/// statsmodels `simulate` at 1e-10 — statsmodels' own smoother uses the
/// classical Holt-Winters seasonal update there (a stated, measured
/// convention gap). Interval coverage and parameter recovery are measured
/// by seeded Monte Carlo and quoted on the model card.
///
/// Further arguments, with defaults: `error` ("add"), `trend` (None),
/// `damped` (False), `seasonal` (None), `seasonal_periods` (None),
/// `initialization` ("estimated"), `horizon` (0), `level` (None: 0.95
/// when forecasting), `n_sim` (None: 5000 when simulating), `seed` (None:
/// 0 when simulating), `optimizer` (None: "auto"), `smoothing_params`
/// (None: estimated), `initial_states` (None), `max_iter` (None).
#[pyfunction]
#[pyo3(signature = (y, error = "add", trend = None, damped = false, seasonal = None, seasonal_periods = None, initialization = "estimated", horizon = 0, level = None, n_sim = None, seed = None, optimizer = None, smoothing_params = None, initial_states = None, max_iter = None))]
#[allow(clippy::too_many_arguments)]
fn ets_fit<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    error: &str,
    trend: Option<&str>,
    damped: bool,
    seasonal: Option<&str>,
    seasonal_periods: Option<usize>,
    initialization: &str,
    horizon: usize,
    level: Option<f64>,
    n_sim: Option<usize>,
    seed: Option<u64>,
    optimizer: Option<&str>,
    smoothing_params: Option<Vec<f64>>,
    initial_states: Option<Vec<f64>>,
    max_iter: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = EtsSpec::new(
        error_type(error)?,
        component("trend", trend)?,
        damped,
        component("seasonal", seasonal)?,
        seasonal_periods,
    )
    .map_err(to_py)?;
    let ys = vec1(&y);
    let (lvl, ns, sd) = forecast_options(horizon, level, n_sim, seed, Some(spec.is_class1()))?;
    let init = match (initialization, initial_states) {
        ("known", Some(st)) => {
            Initialization::Known(EtsStates::from_slice(&spec, &st).map_err(to_py)?)
        }
        ("known", None) => {
            return Err(PyValueError::new_err(format!(
                "initialization=\"known\" needs initial_states = [level, trend?, seasonal[0..m)] \
                 ({} values for this spec: {}); pass them, or use \"estimated\" / \"heuristic\"",
                spec.n_initial_states(),
                spec.initial_state_names().join(", ")
            )))
        }
        ("estimated", None) => Initialization::Estimated,
        ("heuristic", None) => Initialization::Heuristic,
        ("estimated", Some(_)) | ("heuristic", Some(_)) => {
            return Err(PyValueError::new_err(format!(
                "initial_states was given but initialization={initialization:?} computes the \
                 initial states itself, so the values would be inert; pass \
                 initialization=\"known\" to use them, or drop initial_states"
            )))
        }
        (other, _) => {
            return Err(PyValueError::new_err(format!(
                "initialization = {other:?} is invalid: pass \"estimated\" (initial states \
                 estimated with the smoothing parameters), \"heuristic\" (Hyndman 2008 \
                 heuristic, held fixed) or \"known\" (initial_states as given)"
            )))
        }
    };
    let fit = match smoothing_params {
        Some(p) => {
            if optimizer.is_some() {
                return Err(PyValueError::new_err(
                    "optimizer was given but smoothing_params fixes the smoothing parameters, \
                     so no optimisation runs and the choice is inert; drop optimizer or drop \
                     smoothing_params",
                ));
            }
            if max_iter.is_some() {
                return Err(PyValueError::new_err(
                    "max_iter was given but smoothing_params fixes the smoothing parameters, \
                     so no optimisation runs and the budget is inert; drop max_iter or drop \
                     smoothing_params",
                ));
            }
            let params = EtsParams::from_slice(&spec, &p).map_err(to_py)?;
            let states = match init {
                Initialization::Known(st) => st,
                Initialization::Heuristic => {
                    tsecon_ets::heuristic_initial_states(&spec, &ys).map_err(to_py)?
                }
                Initialization::Estimated => {
                    return Err(PyValueError::new_err(
                        "smoothing_params fixes the smoothing parameters, but \
                         initialization=\"estimated\" asks to estimate the initial states \
                         jointly with them — that combination is not offered; pass \
                         initialization=\"heuristic\" (Hyndman 2008 initial states) or \
                         \"known\" with initial_states, or drop smoothing_params to estimate \
                         everything",
                    ))
                }
            };
            let mut f = tsecon_ets::ets_at(&spec, &ys, &params, &states).map_err(to_py)?;
            f.initialization = if initialization == "heuristic" {
                "heuristic"
            } else {
                "known"
            };
            f
        }
        None => {
            let opts = FitOptions {
                initialization: init,
                optimizer: optimizer_of(optimizer)?,
                max_iter,
            };
            tsecon_ets::ets_fit(&spec, &ys, &opts).map_err(to_py)?
        }
    };
    let fc = if horizon > 0 {
        Some(tsecon_ets::forecast(&fit, horizon, lvl, ns, sd).map_err(to_py)?)
    } else {
        None
    };
    fit_to_dict(py, &fit, fc.as_ref(), horizon)
}

fn optimizer_of(s: Option<&str>) -> PyResult<Optimizer> {
    optimizer(s)
}

/// Automatic ETS model selection — the candidate-set search of Hyndman
/// et al. (2008, section 7.2) as R's `forecast::ets` runs it: every
/// admissible member of the taxonomy is fitted by maximum likelihood
/// (`ets_fit`) and the one with the smallest information criterion
/// (`ic`: "aicc" default, "aic", "bic") is returned, fitted, with the
/// ranked candidate table.
///
/// Candidates: error "add"/"mul", trend None/"add" (plus "mul" with
/// `allow_multiplicative_trend=True`; R's default excludes it), damped
/// and undamped trends (`damped=None`; `True`/`False` restricts to one),
/// seasonal None/"add"/"mul" when `seasonal_periods` >= 2 (None: non-
/// seasonal candidates only). Multiplicative errors and seasonals are
/// tried only when every `y` > 0. `restrict=True` (R's default) drops the
/// combinations with infinite forecast variance or a mis-scaled error —
/// additive error with any multiplicative component, and (M,M,A) — so a
/// positive seasonal series has 15 candidates (6 additive-error, 9
/// multiplicative-error), a non-positive one 6, a non-seasonal positive
/// series 5. `initialization` is "estimated" or "heuristic" for every
/// candidate; `optimizer` and the forecast options (`horizon`, `level`,
/// `n_sim`, `seed`) are those of `ets_fit` — `n_sim` and `seed` act only
/// if the selected model is not class 1 (the winner is not known in
/// advance, so they are accepted regardless; with `horizon=0` they are
/// refused as inert). NaN is refused.
///
/// Returned keys: every key of `ets_fit` for the selected model (its very
/// fit from the search, not a refit — refitting reproduces it exactly),
/// plus `ic`, `ic_value`, `candidates` — a list of dicts with `spec`,
/// `short_name`, `loglik`, `aic`, `aicc`, `bic`, `ic_value`, `k_params`,
/// `converged`, `status` ("ok" or "error") and `error` (the message when
/// a candidate failed; failures never abort the search) ranked by the
/// criterion, failures last — `n_candidates` and `n_fitted`. Read the
/// table: candidates within ~2 of the best criterion are near-ties the
/// data do not distinguish.
///
/// Validation (honest grade, as for `auto_arima`): every candidate's
/// likelihood is the golden-pinned `ets_fit` likelihood; the candidate
/// set reproduces R's `forecast::ets` enumeration exactly
/// (fixtures/ets.json); the selection loop itself has no runnable
/// third-party reference (the M3 forecast-competition parity of the
/// method is R-only), so it is graded by seeded Monte-Carlo recovery of
/// the generating component form, quoted on the model card.
///
/// Further arguments, with defaults: `seasonal_periods` (None), `ic`
/// ("aicc"), `allow_multiplicative_trend` (False), `restrict` (True),
/// `damped` (None: both), `initialization` ("estimated"), `horizon` (0),
/// `level` (None: 0.95), `n_sim` (None: 5000), `seed` (None: 0),
/// `optimizer` (None: "auto").
#[pyfunction]
#[pyo3(signature = (y, seasonal_periods = None, ic = "aicc", allow_multiplicative_trend = false, restrict = true, damped = None, initialization = "estimated", horizon = 0, level = None, n_sim = None, seed = None, optimizer = None))]
#[allow(clippy::too_many_arguments)]
fn auto_ets<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    seasonal_periods: Option<usize>,
    ic: &str,
    allow_multiplicative_trend: bool,
    restrict: bool,
    damped: Option<bool>,
    initialization: &str,
    horizon: usize,
    level: Option<f64>,
    n_sim: Option<usize>,
    seed: Option<u64>,
    optimizer: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let ys = vec1(&y);
    let (lvl, ns, sd) = forecast_options(horizon, level, n_sim, seed, None)?;
    let init = match initialization {
        "estimated" => Initialization::Estimated,
        "heuristic" => Initialization::Heuristic,
        other => {
            return Err(PyValueError::new_err(format!(
                "initialization = {other:?} is invalid for auto_ets: pass \"estimated\" or \
                 \"heuristic\" (known initial states cannot be shared across candidates with \
                 different state vectors)"
            )))
        }
    };
    let opts = AutoEtsOptions {
        ic: ic_of(ic)?,
        seasonal_periods,
        allow_multiplicative_trend,
        restrict,
        damped,
        initialization: init,
        optimizer: optimizer_of(optimizer)?,
    };
    let r = tsecon_ets::auto_ets(&ys, &opts).map_err(to_py)?;
    let fc = if horizon > 0 {
        Some(tsecon_ets::forecast(&r.best, horizon, lvl, ns, sd).map_err(to_py)?)
    } else {
        None
    };
    let d = fit_to_dict(py, &r.best, fc.as_ref(), horizon)?;
    d.set_item("ic", r.ic.name())?;
    d.set_item("ic_value", r.candidates[0].ic_value)?;
    let cands = PyList::empty(py);
    for c in &r.candidates {
        let cd = PyDict::new(py);
        cd.set_item("spec", c.spec.name())?;
        cd.set_item("short_name", &c.short_name)?;
        cd.set_item("loglik", c.loglik)?;
        cd.set_item("aic", c.aic)?;
        cd.set_item("aicc", c.aicc)?;
        cd.set_item("bic", c.bic)?;
        cd.set_item("ic_value", c.ic_value)?;
        cd.set_item("k_params", c.k_params)?;
        cd.set_item("converged", c.converged)?;
        cd.set_item("status", c.status)?;
        cd.set_item("error", c.error.as_deref())?;
        cands.append(cd)?;
    }
    d.set_item("candidates", cands)?;
    d.set_item("n_candidates", r.n_candidates)?;
    d.set_item("n_fitted", r.n_fitted)?;
    Ok(d)
}

/// Adds the module's functions to the `_core` extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ets_fit, m)?)?;
    m.add_function(wrap_pyfunction!(auto_ets, m)?)?;
    Ok(())
}
