//! Python bindings for the static-panel menu of `tsecon-panel`: random
//! effects (`panel_re`), the between estimator (`panel_between`), first
//! differences (`panel_fd`) and Fama-MacBeth (`panel_fama_macbeth`).
//! Registered into `_core` through [`register`].
//!
//! The four share `panel_fe`'s inputs (an `N x T` outcome, `k x N x T`
//! regressors, an optional `N x T` 0/1 mask parsed by the shared
//! [`crate::panel_data`]) and their own covariance vocabulary
//! ([`static_se`]): the house words `"hc1"` (alias `"robust"`) and `"hac"`
//! (alias `"newey_west"`) are accepted here and nowhere in the panel-FE
//! family, whose `panel_se` is untouched.

use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tsecon_panel::{
    BetweenConfig, FamaMacBethSe, PanelData, RandomEffectsConfig, StaticInference, StaticSeType,
};
use tsecon_var::tsecon_linalg::faer::Mat;

use crate::{panel_data, to_py};

/// The kernel truncation used when the caller passes `bandwidth=None`
/// under a kernel `se_type` (Driscoll-Kraay, or the Newey-West HAC of the
/// Fama-MacBeth `beta_t` series): lags 1..=4 — the same default as
/// `panel_fe`'s Driscoll-Kraay.
const STATIC_DEFAULT_BANDWIDTH: f64 = 4.0;

/// Which covariance menu a static-panel callable offers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Menu {
    /// Random effects and first differences: nonrobust, hc1, cluster,
    /// Driscoll-Kraay.
    Pooled,
    /// The between estimator: nonrobust, hc1, cluster (no time dimension).
    Between,
}

/// Parses `se_type` (and the kernel `bandwidth`, which acts ONLY under
/// Driscoll-Kraay and is refused, by name, under every other choice) for
/// the random-effects / between / first-difference menu.
fn static_se(
    caller: &str,
    se_type: &str,
    bandwidth: Option<f64>,
    menu: Menu,
) -> PyResult<StaticSeType> {
    let expected = match menu {
        Menu::Pooled => {
            "\"nonrobust\", \"hc1\" (alias \"robust\"), \"cluster\" (alias \
             \"cluster_entity\") or \"driscoll_kraay\" (alias \"dk\")"
        }
        Menu::Between => {
            "\"nonrobust\", \"hc1\" (alias \"robust\") or \"cluster\" (alias \
             \"cluster_entity\"; numerically HC1 here — one row per entity)"
        }
    };
    let no_kernel = |se: StaticSeType| -> PyResult<StaticSeType> {
        if let Some(bw) = bandwidth {
            return Err(PyValueError::new_err(format!(
                "{caller}: bandwidth={bw} has no effect under se_type={se_type:?}: \
                 bandwidth is the Driscoll-Kraay kernel truncation lag, and {se_type} \
                 standard errors use no kernel, so there is nothing for it to \
                 truncate and the argument would be silently absorbed. Pass \
                 se_type=\"driscoll_kraay\" for the bandwidth to act, or drop \
                 bandwidth to keep {se_type} standard errors."
            )));
        }
        Ok(se)
    };
    match se_type {
        "nonrobust" => no_kernel(StaticSeType::NonRobust),
        "hc1" | "robust" => no_kernel(StaticSeType::Hc1),
        "cluster" | "cluster_entity" => no_kernel(StaticSeType::ClusterEntity),
        "driscoll_kraay" | "dk" if menu == Menu::Pooled => Ok(StaticSeType::DriscollKraay {
            bandwidth: bandwidth.unwrap_or(STATIC_DEFAULT_BANDWIDTH),
        }),
        "driscoll_kraay" | "dk" => Err(PyValueError::new_err(format!(
            "{caller}: se_type={se_type:?} is not available for the between estimator: \
             it regresses one row per entity (the entity means), so there is no time \
             dimension to aggregate scores over; se_type must be {expected}"
        ))),
        other => Err(PyValueError::new_err(format!(
            "{caller}: unknown se_type {other:?}; se_type must be {expected}"
        ))),
    }
}

/// Parses `se_type` / `bandwidth` for `panel_fama_macbeth`.
fn fama_macbeth_se(se_type: &str, bandwidth: Option<f64>) -> PyResult<FamaMacBethSe> {
    match se_type {
        "fama_macbeth" => {
            if let Some(bw) = bandwidth {
                return Err(PyValueError::new_err(format!(
                    "panel_fama_macbeth: bandwidth={bw} has no effect under \
                     se_type=\"fama_macbeth\": bandwidth is the Newey-West (Bartlett) \
                     truncation lag of the HAC on the beta_t series, and the \
                     Fama-MacBeth standard error uses no kernel, so the argument would \
                     be silently absorbed. Pass se_type=\"hac\" for the bandwidth to \
                     act, or drop bandwidth."
                )));
            }
            Ok(FamaMacBethSe::Standard)
        }
        "hac" | "newey_west" => Ok(FamaMacBethSe::Hac {
            bandwidth: bandwidth.unwrap_or(STATIC_DEFAULT_BANDWIDTH),
        }),
        other => Err(PyValueError::new_err(format!(
            "panel_fama_macbeth: unknown se_type {other:?}; se_type must be \
             \"fama_macbeth\" (sd(beta_t)/sqrt(T_v)) or \"hac\" (alias \"newey_west\"; \
             Bartlett HAC on the beta_t series)"
        ))),
    }
}

/// The panel container from the Python inputs (`x{c}` regressor names).
fn build(
    caller: &str,
    outcome: &PyReadonlyArray2<'_, f64>,
    regressors: &PyReadonlyArray3<'_, f64>,
    mask: Option<&PyReadonlyArray2<'_, f64>>,
) -> PyResult<PanelData> {
    let o = outcome.as_array();
    let outcome_m = Mat::from_fn(o.nrows(), o.ncols(), |i, j| o[(i, j)]);
    let r = regressors.as_array();
    let (k, n, t) = (r.shape()[0], r.shape()[1], r.shape()[2]);
    let regs: Vec<(String, Mat<f64>)> = (0..k)
        .map(|c| (format!("x{c}"), Mat::from_fn(n, t, |i, j| r[[c, i, j]])))
        .collect();
    if k == 0 && (n, t) != (o.nrows(), o.ncols()) {
        return Err(PyValueError::new_err(format!(
            "{caller}: regressors has shape (0, {n}, {t}) but outcome is {} x {}; pass \
             np.empty((0, N, T)) with the outcome's N and T for the intercept-only model",
            o.nrows(),
            o.ncols()
        )));
    }
    panel_data(caller, outcome_m, regs, mask)
}

fn cov_lists(cov: &Mat<f64>) -> Vec<Vec<f64>> {
    let k = cov.nrows();
    (0..k)
        .map(|a| (0..k).map(|b| cov[(a, b)]).collect())
        .collect()
}

/// The keys every static-panel result shares.
fn common<'py>(
    py: Python<'py>,
    params: Vec<f64>,
    inf: StaticInference,
    names: Vec<String>,
    se_type: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("params", params.into_pyarray(py))?;
    d.set_item("bse", inf.bse.into_pyarray(py))?;
    d.set_item("tvalues", inf.tvalues.into_pyarray(py))?;
    d.set_item("cov", cov_lists(&inf.cov))?;
    d.set_item("names", names)?;
    d.set_item("se_type", se_type)?;
    Ok(d)
}

/// Random-effects panel regression — feasible GLS with Swamy-Arora-type
/// variance components and a per-entity quasi-demeaning weight `theta_i`,
/// matching linearmodels `RandomEffects` (version 7.0).
///
/// `outcome` is `N x T`; `regressors` is `k x N x T` (stack a list of
/// `N x T` arrays with `np.stack`; `np.empty((0, N, T))` with
/// `constant=True` is the intercept-only variance-components model, whose
/// `rho` is the intraclass correlation). `constant` (True) adds the
/// intercept, named `const` and listed first. `mask` (default None = a
/// balanced panel) is an `N x T` array of 0/1 (False/True) flags, 1 where
/// the entity is observed in that period; cells outside the mask are
/// ignored and may hold NaN. The model assumes the entity effect is
/// uncorrelated with every regressor (`E[u_i | X] = 0`); time-invariant
/// regressors are estimable. Steps: the within residuals give
/// `sigma2_eps = e'e / (n - K - N + 1)` (`K` counting the constant; the
/// linearmodels denominator, also without a constant); the between
/// regression of the entity means gives
/// `sigma2_effects = max(0, u'u/(N - K) - sigma2_eps / Tbar)` with the
/// HARMONIC mean `Tbar`; `theta_i = 1 - sqrt(sigma2_eps / (T_i
/// sigma2_effects + sigma2_eps))`; then OLS on `y - theta_i ybar_i` and
/// `x - theta_i xbar_i`. `small_sample=True` uses linearmodels'
/// small-sample variance components when the entities' cell counts differ
/// (a no-op when every entity has the same count, including a panel whose
/// mask drops a whole period); it is linearmodels' formula, NOT Stata's
/// Baltagi-Chang `sa` or plm's unbalanced `swar`, and it is refused when the
/// between design is rank-deficient (time dummies on an unbalanced panel),
/// where linearmodels raises LinAlgError.
///
/// `se_type`: "nonrobust", "hc1" (alias "robust": the `n/(n - K)` White
/// sandwich), "cluster" (by entity, the default; alias "cluster_entity") or
/// "driscoll_kraay" (alias "dk"; Bartlett over CALENDAR periods — a period
/// nobody observes contributes a zero score). `bandwidth` is the
/// Driscoll-Kraay lag truncation and acts ONLY under
/// `se_type="driscoll_kraay"` (4.0 when omitted there); passing it
/// explicitly with any other `se_type` raises, and so does `bandwidth + 1`
/// above the number of periods that carry scores (T on a balanced panel;
/// past it every standard error only shrinks, like
/// `sqrt(T / (bandwidth + 1))`). The Mundlak check: pass each
/// time-varying regressor's entity mean as an extra regressor and the slopes
/// on the time-varying regressors equal `panel_fe`'s exactly; a Wald test of
/// the mean coefficients on the returned `cov` (cluster) is the robust
/// Hausman test.
///
/// Returned keys: `params`, `bse`, `tvalues` (length K), `cov` (K x K nested
/// lists), `names`, `nobs` (observed cells), `n_entities` (entities with an
/// observation), `df_resid` (`nobs - K`), `sigma2_eps`, `sigma2_effects`
/// (clamped at 0), `rho` (`sigma2_effects / (sigma2_effects + sigma2_eps)`),
/// `theta` (length N in layout order, NaN for an entity with no observed
/// cell), and the stamped `se_type`, `constant`, `small_sample`. Validated
/// against linearmodels RandomEffects at 1e-10 on the Arellano-Bond EmplUK
/// panel and seeded panels (fixtures/panel_static.json).
#[pyfunction]
#[pyo3(signature = (outcome, regressors, se_type = "cluster", bandwidth = None, constant = true, small_sample = false, mask = None))]
#[allow(clippy::too_many_arguments)]
fn panel_re<'py>(
    py: Python<'py>,
    outcome: PyReadonlyArray2<'py, f64>,
    regressors: PyReadonlyArray3<'py, f64>,
    se_type: &str,
    bandwidth: Option<f64>,
    constant: bool,
    small_sample: bool,
    mask: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let se = static_se("panel_re", se_type, bandwidth, Menu::Pooled)?;
    let data = build("panel_re", &outcome, &regressors, mask.as_ref())?;
    let fit = tsecon_panel::panel_random_effects(
        &data,
        RandomEffectsConfig {
            constant,
            small_sample,
        },
    )
    .map_err(to_py)?;
    let inf = fit.inference(se).map_err(to_py)?;
    let d = common(py, fit.params.clone(), inf, fit.names.clone(), se_type)?;
    d.set_item("nobs", fit.nobs)?;
    d.set_item("n_entities", fit.n_entities)?;
    d.set_item("df_resid", fit.df_resid)?;
    d.set_item("sigma2_eps", fit.sigma2_eps)?;
    d.set_item("sigma2_effects", fit.sigma2_effects)?;
    d.set_item("rho", fit.rho)?;
    d.set_item("theta", fit.theta.clone().into_pyarray(py))?;
    d.set_item("constant", constant)?;
    d.set_item("small_sample", small_sample)?;
    Ok(d)
}

/// The between estimator — OLS of the entity means of the outcome on the
/// entity means of the regressors, matching linearmodels `BetweenOLS`
/// (version 7.0).
///
/// `outcome` is `N x T`; `regressors` is `k x N x T` (`np.empty((0, N, T))`
/// with `constant=True` is the intercept-only model). The means run over
/// each entity's observed cells (`mask`, default None = a balanced panel:
/// an `N x T` array of 0/1 flags; masked-out cells are ignored and may hold
/// NaN). `constant` (True) adds the intercept, `const`, first.
/// `reweight=True` multiplies each entity's row by `sqrt(T_i / mean T_i)`
/// (weighted least squares by cell count; Stata `xtreg, be wls`); a no-op
/// when every entity has the same count. `se_type`: "nonrobust", "hc1"
/// (the default; alias "robust": `N/(N - K)` times the White sandwich) or
/// "cluster" (alias "cluster_entity") — clustering by entity with one row
/// per entity IS the White sandwich, so "cluster" returns the HC1 numbers
/// (linearmodels agrees); "driscoll_kraay" is refused because the between
/// regression has no time dimension. There is no `bandwidth`.
///
/// Returned keys: `params`, `bse`, `tvalues`, `cov` (K x K nested lists),
/// `names`, `nobs` (entities with an observation — the rows of the between
/// regression), `n_entities`, `df_resid` (`nobs - K`), and the stamped
/// `se_type`, `constant`, `reweight`. Validated against linearmodels
/// BetweenOLS at 1e-10 (fixtures/panel_static.json), and against
/// statsmodels OLS / HC1 on the means.
#[pyfunction]
#[pyo3(signature = (outcome, regressors, se_type = "hc1", constant = true, reweight = false, mask = None))]
fn panel_between<'py>(
    py: Python<'py>,
    outcome: PyReadonlyArray2<'py, f64>,
    regressors: PyReadonlyArray3<'py, f64>,
    se_type: &str,
    constant: bool,
    reweight: bool,
    mask: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let se = static_se("panel_between", se_type, None, Menu::Between)?;
    let data = build("panel_between", &outcome, &regressors, mask.as_ref())?;
    let fit =
        tsecon_panel::panel_between(&data, BetweenConfig { constant, reweight }).map_err(to_py)?;
    let inf = fit.inference(se).map_err(to_py)?;
    let d = common(py, fit.params.clone(), inf, fit.names.clone(), se_type)?;
    d.set_item("nobs", fit.nobs)?;
    d.set_item("n_entities", fit.n_entities)?;
    d.set_item("df_resid", fit.df_resid)?;
    d.set_item("constant", constant)?;
    d.set_item("reweight", reweight)?;
    Ok(d)
}

/// The first-difference estimator — OLS of `y_it - y_i,t-1` on
/// `x_it - x_i,t-1`, matching linearmodels `FirstDifferenceOLS` (version
/// 7.0) with `constant=False` (the default).
///
/// `outcome` is `N x T`; `regressors` is `k x N x T` (`k >= 1`). A
/// difference row `(i, t)` exists when entity `i` is observed in period
/// `t` AND in `t - 1` — CALENDAR adjacency: an internal gap drops the
/// differences that would bridge it, and a period nobody observes is never
/// bridged (linearmodels differences across such a hole; see the panel
/// model card). `mask` (default None = a balanced panel) is an `N x T` array
/// of 0/1 flags; masked-out cells are ignored and may hold NaN.
/// `constant=True` adds the intercept (`const`, first) that plm
/// `model="fd"` and Stata `regress D.y D.x` include by default — a linear
/// trend in levels; linearmodels refuses a constant in first differences.
/// A regressor that never changes between consecutive observed periods
/// (time-invariant) is refused — use `panel_re` or `panel_between` for it.
/// `se_type`: "nonrobust", "hc1" (alias "robust"), "cluster" (by entity,
/// the default — differencing induces MA(1) errors; alias
/// "cluster_entity") or "driscoll_kraay" (alias "dk"; scores indexed by
/// the later period `t`, calendar lags). `bandwidth` is the Driscoll-Kraay
/// lag truncation and acts ONLY under `se_type="driscoll_kraay"` (4.0 when
/// omitted there); passing it explicitly with any other `se_type` raises,
/// and so does `bandwidth + 1` above the number of periods that carry
/// differences (T - 1 on a balanced panel; past it every standard error
/// only shrinks). "cluster" needs differences from at least two entities and
/// "driscoll_kraay" differences in at least two periods — one group's score
/// sum is zero by the normal equations — so a single entity or a two-period
/// panel is refused under them (use "hc1").
///
/// Returned keys: `params`, `bse`, `tvalues`, `cov` (K x K nested lists),
/// `names`, `nobs` (difference rows), `n_entities` (entities contributing a
/// difference), `df_resid` (`nobs - K`), and the stamped `se_type`,
/// `constant`. Validated against linearmodels FirstDifferenceOLS (and
/// PooledOLS on the hand-differenced data for `constant=True`) at 1e-10
/// (fixtures/panel_static.json).
#[pyfunction]
#[pyo3(signature = (outcome, regressors, se_type = "cluster", bandwidth = None, constant = false, mask = None))]
fn panel_fd<'py>(
    py: Python<'py>,
    outcome: PyReadonlyArray2<'py, f64>,
    regressors: PyReadonlyArray3<'py, f64>,
    se_type: &str,
    bandwidth: Option<f64>,
    constant: bool,
    mask: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let se = static_se("panel_fd", se_type, bandwidth, Menu::Pooled)?;
    let data = build("panel_fd", &outcome, &regressors, mask.as_ref())?;
    let fit = tsecon_panel::panel_first_difference(&data, constant).map_err(to_py)?;
    let inf = fit.inference(se).map_err(to_py)?;
    let d = common(py, fit.params.clone(), inf, fit.names.clone(), se_type)?;
    d.set_item("nobs", fit.nobs)?;
    d.set_item("n_entities", fit.n_entities)?;
    d.set_item("df_resid", fit.df_resid)?;
    d.set_item("constant", constant)?;
    Ok(d)
}

/// Fama-MacBeth (1973) two-pass regression — one cross-sectional OLS per
/// period, then the time-series mean of the per-period coefficients,
/// matching linearmodels `FamaMacBeth` (version 7.0).
///
/// `outcome` is `N x T`; `regressors` is `k x N x T` (`np.empty((0, N, T))`
/// with `constant=True` averages the per-period means). `constant` (True)
/// adds the intercept, `const`, first. `mask` (default None = a balanced
/// panel) is an `N x T` array of 0/1 flags; masked-out cells are ignored
/// and may hold NaN. A period is skipped (a NaN row of
/// `params_per_period`) when fewer than K entities are observed in it or
/// its cross-section is rank-deficient (numpy singular-value criterion;
/// linearmodels uses LAPACK gelsy — the two agree away from the numerical
/// boundary). At least 2 usable periods are required. `se_type`:
/// "fama_macbeth" (the default: `sd(beta_t, ddof=1) / sqrt(T_v)` over the
/// `T_v` periods used) or "hac" (alias "newey_west": a Bartlett HAC of the
/// `beta_t` series over the COMPRESSED sequence of periods used, divided by
/// `T_v (T_v - 1)`). `bandwidth` is that HAC's lag truncation and acts ONLY
/// under `se_type="hac"` (4.0 when omitted there; linearmodels' own
/// default is an automatic rule); passing it under "fama_macbeth" raises,
/// and `bandwidth + 1 > T_v` is refused. CAVEATS: the standard error treats
/// the `beta_t` as independent, so persistent firm effects make it far too
/// small (Petersen 2009; measured on the panel model card) and the HAC does
/// not repair it — the correlation does not decay with the lag; use
/// `panel_re` or `panel_fe` with entity clustering there. There is no
/// Shanken errors-in-variables correction for generated regressors.
///
/// Returned keys: `params`, `bse`, `tvalues`, `cov` (K x K nested lists),
/// `names`, `nobs` (every observed cell, skipped periods included),
/// `params_per_period` (T x K nested lists, NaN rows for skipped periods),
/// `periods_used` (length-T bools), `n_periods_used`, `bandwidth` (the
/// effective float under "hac", None otherwise), and the stamped `se_type`,
/// `constant`. Validated against linearmodels FamaMacBeth at 1e-10
/// (fixtures/panel_static.json).
#[pyfunction]
#[pyo3(signature = (outcome, regressors, se_type = "fama_macbeth", bandwidth = None, constant = true, mask = None))]
fn panel_fama_macbeth<'py>(
    py: Python<'py>,
    outcome: PyReadonlyArray2<'py, f64>,
    regressors: PyReadonlyArray3<'py, f64>,
    se_type: &str,
    bandwidth: Option<f64>,
    constant: bool,
    mask: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let se = fama_macbeth_se(se_type, bandwidth)?;
    let data = build("panel_fama_macbeth", &outcome, &regressors, mask.as_ref())?;
    let fit = tsecon_panel::fama_macbeth(&data, constant).map_err(to_py)?;
    let inf = fit.inference(se).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("params", fit.params.clone().into_pyarray(py))?;
    d.set_item("bse", inf.bse.into_pyarray(py))?;
    d.set_item("tvalues", inf.tvalues.into_pyarray(py))?;
    d.set_item("cov", cov_lists(&inf.cov))?;
    d.set_item("names", fit.names.clone())?;
    d.set_item("nobs", fit.nobs)?;
    d.set_item("params_per_period", fit.params_per_period.clone())?;
    d.set_item("periods_used", fit.periods_used.clone())?;
    d.set_item("n_periods_used", fit.n_periods_used)?;
    let bw = match se {
        FamaMacBethSe::Hac { bandwidth } => Some(bandwidth),
        FamaMacBethSe::Standard => None,
    };
    d.set_item("bandwidth", bw)?;
    d.set_item("se_type", se_type)?;
    d.set_item("constant", constant)?;
    Ok(d)
}

/// Adds the module's functions to the `_core` extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(panel_re, m)?)?;
    m.add_function(wrap_pyfunction!(panel_between, m)?)?;
    m.add_function(wrap_pyfunction!(panel_fd, m)?)?;
    m.add_function(wrap_pyfunction!(panel_fama_macbeth, m)?)?;
    Ok(())
}
