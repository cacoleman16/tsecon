#!/usr/bin/env python
"""Golden fixtures for the 0.11.0 follow-ups of ``unobserved_components``
(``tsecon-ssm``) -> ``fixtures/uc_extensions.json``:

* ``cycle_init="stationary"`` -- the damped stochastic cycle initialized
  from its unconditional (Lyapunov) distribution instead of exactly diffuse;
* ``autoregressive=p`` -- an AR(p) component (statsmodels' own
  ``autoregressive=`` keyword, row-companion form, always initialized from
  its stationary distribution);
* ``mle_regression=False`` -- the regression coefficients as exactly diffuse
  constant states with design row ``x_t'`` (statsmodels' own
  ``mle_regression=False``), the regression-coefficient component included.

Reference implementation (this venv; versions pinned in ``_meta``):

* ``statsmodels.tsa.statespace.structural.UnobservedComponents`` with
  ``use_exact_diffuse=True`` -- an INDEPENDENT-PACKAGE golden, evaluated at
  FIXED parameters with ``.smooth(params)`` and ``.get_forecast(h, ...)``:
  the log-likelihood, AIC/BIC (statsmodels' ``df_model = k_params +
  k_diffuse_states``), ``nobs_diffuse``, filtered and smoothed states with
  their variances, one-step predictions and errors, standardized residuals,
  forecasts and forecast variances, and the component paths.
* The STATIONARY CYCLE is not a statsmodels keyword: statsmodels always
  initializes the cycle diffuse. The reference is the subclass
  ``StationaryCycleUC`` below, whose ``initialize_default`` calls the parent
  and then rebuilds ``statsmodels.tsa.statespace.initialization.Initialization``
  from contiguous blocks -- "diffuse" for every state except the cycle pair,
  which is "stationary". That is statsmodels' own block Lyapunov solve and
  its own exact-diffuse filter and smoother, so this leg is graded as
  statsmodels machinery under a documented initialization, plus the closed
  form ``P_* = sigma2_cycle / (1 - rho^2) I_2`` (pinned in the Rust crate's
  own unit test, since the public fit does not expose ``P_1``). The cycle is
  located through the PUBLIC ``state_names.index("cycle")``, never the
  private ``_k_seasonal_states`` / ``_k_freq_seas_states``, so a statsmodels
  upgrade cannot silently move it.
* ``smoother_spread`` is recorded per fixed block exactly as in ``uc.json``:
  the largest relative disagreement between statsmodels' OWN univariate and
  conventional smoothers on the same model, used by the tests as the
  smoothed-variance tolerance inside the diffuse period (it is not assumed
  to be zero).
* The MLE, one criterion, two optimizers, through the ``uc.json``
  generator's own helpers (imported, not copied: ``generate_uc_fixtures``
  keeps its work behind ``if __name__ == "__main__"``): statsmodels' L-BFGS
  and a SciPy Nelder-Mead + L-BFGS-B polish; the tests pin the BETTER. The
  pile-up flags come from the same documented rule, and ``se_conditional``
  is statsmodels' complex-step Hessian (complex-step aware through its
  Lyapunov solve) inverted over the free parameters, where the free set
  mirrors tsecon's identification rule: when the variance of a STATIONARY
  block is flagged at zero, that block's transition parameters (the
  stationary cycle's frequency and damping, an AR block's ``ar.L*``) are
  dropped from it too -- the block is identically zero at variance 0, so
  they carry no information. AR coefficients are never flagged themselves
  (the stationarity region is open), which the base ``boundary_flags``
  already implements (only ``sigma2.*``, ``frequency.cycle`` and
  ``damping.cycle`` are ever flagged). The complex-step Hessian never
  leaves the stationarity region, so the reference needs no counterpart of
  tsecon's AR probe-step guard.
* CLOSED FORMS for the state regression (de Jong 1991; Durbin & Koopman
  2012, secs. 5.7 and 7.2): with white noise only, the diffuse-prior
  coefficient states are GLS = OLS, ``beta_hat = (X'X)^-1 X'y``,
  ``Var = sigma2 (X'X)^-1``, ``l = -[n ln 2 pi + n ln sigma2 + RSS/sigma2 +
  ln det(X'X / sigma2)] / 2``, forecast ``x' beta_hat`` with variance
  ``sigma2 (1 + x'(X'X)^-1 x)``; the cross-mode identity ``l_state(theta) =
  l_mle(theta, beta_gls) + ln det(P_beta) / 2`` (marginal against profile
  likelihood) on six specifications; and the regressor-units identity:
  rescaling a column by ``c`` shifts the state-form likelihood by exactly
  ``-ln |c|`` and leaves the MLE form unchanged.
* The ``lambda -> 0`` sweep: statsmodels' log-likelihood at the GDP
  stationary-cycle optimum with only the frequency moved, under both
  initializations. The stationary one is flat; the diffuse one keeps rising
  toward the singularity documented in ``uc.json``'s generator.

Data: the bundled public-domain Nile series and US macrodata transforms
(``infl``, ``100 log realgdp`` and its first difference), the seeded
simulated series of ``uc.json`` (reconstructed here from its seed and
asserted equal), and a second seeded series (``default_rng(20261008)``: a
local level plus two regressors, a NaN-inserted copy, six forecast rows).
Only these public-domain or seeded series and derived numbers are stored.

This generator NEVER imports tsecon.

Run:  .venv/bin/python fixtures/generate_uc_extensions_fixtures.py
"""
from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.structural import UnobservedComponents

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import generate_uc_fixtures as base  # noqa: E402  (its main() is __main__-guarded)

warnings.simplefilter("ignore")

OUT = HERE / "uc_extensions.json"
lst = base.lst


class StationaryCycleUC(UnobservedComponents):
    """statsmodels' UC with the cycle pair (and any AR block) initialized
    STATIONARY and every other state exactly diffuse, from contiguous blocks
    of statsmodels' own ``Initialization``. The cycle is located through the
    public ``state_names``."""

    def initialize_default(self, approximate_diffuse_variance=None):
        super().initialize_default(approximate_diffuse_variance)
        k = self.k_states
        names = list(self.state_names)
        flags = np.ones(k, bool)
        off = names.index("cycle")
        flags[off:off + 2] = False
        if self.ar_order:
            a = names.index("ar.L1")
            flags[a:a + self.ar_order] = False
        init = Initialization(k)
        i = 0
        while i < k:
            j = i
            while j < k and flags[j] == flags[i]:
                j += 1
            init.set((i, j), "diffuse" if flags[i] else "stationary")
            i = j
        self.ssm.initialization = init


def components_ext(res):
    """The uc.json component record plus the AR component (statsmodels'
    ``results.autoregressive``: the first AR state)."""
    comp = base.components(res)
    ar = res.autoregressive
    if ar is not None:
        comp["autoregressive"] = dict(
            filtered=lst(ar.filtered), filtered_var=lst(ar.filtered_cov),
            smoothed=lst(ar.smoothed), smoothed_var=lst(ar.smoothed_cov),
        )
    return comp


def fixed_block(mod, params, h=8, forecast_exog=None, components=False, full_states=True):
    """The uc.json fixed block plus the initialization record. With
    ``full_states=False`` the state matrices are dropped (size budget) --
    only where the component paths already pin every state."""
    block = base.fixed_block(mod, np.asarray(params, float), h=h, forecast_exog=forecast_exog)
    if not full_states:
        for k in ("filtered_state", "filtered_state_var", "smoothed_state", "smoothed_state_var"):
            del block[k]
    res = mod.smooth(np.asarray(params, float))
    block["k_diffuse"] = int(res.k_diffuse_states)
    block["df_model"] = int(res.df_model)
    block["initial_state_cov_diag"] = lst(np.diag(res.filter_results.initial_state_cov))
    pinf = res.filter_results.initial_diffuse_state_cov     # None when nothing is diffuse
    block["initial_diffuse_state_cov_diag"] = lst(np.zeros(mod.k_states) if pinf is None else np.diag(pinf))
    if components:
        block["components"] = components_ext(res)
    return block


def tied_free_set(mod, flags, stationary_cycle):
    """tsecon's free set for the standard errors: not flagged, and not a
    transition parameter of a stationary block whose variance is flagged."""
    names = list(mod.param_names)
    out = [not f for f in flags]
    if "sigma2.ar" in names and flags[names.index("sigma2.ar")]:
        for i, n in enumerate(names):
            if n.startswith("ar.L"):
                out[i] = False
    if stationary_cycle and "sigma2.cycle" in names and flags[names.index("sigma2.cycle")]:
        for n in ("frequency.cycle", "damping.cycle"):
            if n in names:
                out[names.index(n)] = False
    return out


def conditional_se_tied(mod, params, flags, stationary_cycle):
    params = np.asarray(params, float)
    hess = np.asarray(mod.hessian(params, transformed=True, approx_complex_step=True)) * mod.nobs
    keep = tied_free_set(mod, flags, stationary_cycle)
    free = [i for i, k in enumerate(keep) if k]
    se = np.full(len(params), np.nan)
    if not free:
        return se
    try:
        cov = np.linalg.inv(-hess[np.ix_(free, free)])
    except np.linalg.LinAlgError:
        return se
    d = np.diag(cov)
    for k, i in enumerate(free):
        se[i] = np.sqrt(d[k]) if np.isfinite(d[k]) and d[k] > 0 else np.nan
    return se


def mle_case(mod, stationary_cycle=False, extra_starts=()):
    r = base.two_optimizer_mle(mod, extra_starts=extra_starts)
    best = np.asarray(r["best"]["params"])
    flags = r["best"]["at_boundary"]
    r["best"]["se_conditional"] = lst(conditional_se_tied(mod, best, flags, stationary_cycle))
    res = mod.smooth(best)
    r["best"]["nobs_diffuse"] = int(res.nobs_diffuse)
    r["best"]["k_diffuse"] = int(res.k_diffuse_states)
    r["best"]["aic"] = float(res.aic)
    r["best"]["bic"] = float(res.bic)
    return r


def sim_series():
    """The uc.json simulated series, reconstructed from its seed."""
    rng = np.random.default_rng(20260911)
    T = 120
    level = np.cumsum(rng.normal(0.0, 0.4, T)) + 0.02 * np.arange(T)
    season = np.tile([1.2, -0.4, -1.1, 0.3], T // 4)[:T]
    cyc = np.zeros(T)
    c, cs = 1.0, 0.0
    lam, rho = 2 * np.pi / 10.0, 0.9
    for t in range(T):
        cyc[t] = c
        c, cs = (rho * (np.cos(lam) * c + np.sin(lam) * cs) + rng.normal(0, 0.3),
                 rho * (-np.sin(lam) * c + np.cos(lam) * cs) + rng.normal(0, 0.3))
    X = np.column_stack([rng.normal(size=T), rng.uniform(-2, 2, size=T)])
    y = level + season + cyc + X @ np.array([0.5, -0.3]) + rng.normal(0.0, 0.9, T)
    Xf = np.column_stack([rng.normal(size=8), rng.uniform(-2, 2, size=8)])
    uc = json.loads((HERE / "uc.json").read_text())
    assert np.array_equal(y, np.asarray(uc["sim"]["y"])), "sim reconstruction drifted from uc.json"
    assert np.array_equal(X, np.asarray(uc["sim"]["x"]))
    assert np.array_equal(Xf, np.asarray(uc["sim"]["x_forecast"]))
    return y, X, Xf


def main():
    fx = {"_meta": dict(
        statsmodels=statsmodels.__version__, numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), generator="fixtures/generate_uc_extensions_fixtures.py",
        note=__doc__.strip(),
    )}
    nile = sm.datasets.nile.load_pandas().data["volume"].values.astype(float)
    md = sm.datasets.macrodata.load_pandas().data
    lgdp = 100.0 * np.log(md["realgdp"].values.astype(float))
    dlgdp = np.diff(lgdp)
    infl = md["infl"].values.astype(float)
    y, X, Xf = sim_series()
    fx["data"] = dict(nile=lst(nile), lgdp=lst(lgdp), dlgdp=lst(dlgdp), infl=lst(infl),
                      sim_y=lst(y), sim_x=lst(X), sim_x_forecast=lst(Xf))
    series = dict(nile=nile, lgdp=lgdp, dlgdp=dlgdp, infl=infl, sim=y)

    # ---------------------------------------------------- (a) AR component
    ar_cases = []
    for name, data, spec, p, h in [
        ("nile_llevel_ar1", "nile", dict(level="llevel", autoregressive=1), [8000.0, 1400.0, 6000.0, 0.5], 5),
        ("nile_llevel_ar2", "nile", dict(level="llevel", autoregressive=2), [8000.0, 1400.0, 6000.0, 0.5, -0.2], 5),
        ("nile_rwalk_ar1", "nile", dict(level="rwalk", autoregressive=1), [1400.0, 9000.0, 0.3], 5),
        ("infl_llevel_ar2", "infl", dict(level="llevel", autoregressive=2), [1.0, 0.1, 3.0, 0.4, 0.2], 5),
        ("lgdp_lltrend_ar2", "lgdp", dict(level="lltrend", autoregressive=2),
         [0.01, 0.3, 0.001, 0.6, 1.3, -0.4], 5),
        # Every state stationary: k_diffuse 0, nobs_diffuse 0.
        ("dlgdp_ntrend_ar3", "dlgdp", dict(level="ntrend", autoregressive=3), [0.2, 0.5, 0.3, 0.15, -0.05], 3),
        # The AR block as the only stochastic component.
        ("sim_fixed_intercept_ar1", "sim", dict(level="fixed intercept", autoregressive=1), [0.8, 0.5], 4),
        ("sim_llevel_seasonal4_cycle_ar1", "sim",
         dict(level="llevel", seasonal=4, cycle=True, damped_cycle=True, stochastic_cycle=True, autoregressive=1),
         [0.8, 0.3, 0.1, 0.2, 0.3, 0.6, 0.85, 0.5], 4),
        ("sim_lltrend_seasonal4_ar2_exog", "sim",
         dict(level="lltrend", seasonal=4, autoregressive=2, exog="sim_x"),
         [0.8, 0.3, 0.02, 0.1, 0.3, 0.5, -0.2, 0.5, -0.3], 4),
    ]:
        kw = {k: (X if k == "exog" else v) for k, v in spec.items()}
        mod = UnobservedComponents(series[data], use_exact_diffuse=True, **kw)
        fexog = Xf[:h] if "exog" in spec else None
        # Two states, both pinned by the level and AR components.
        full = name not in ("nile_llevel_ar1", "nile_rwalk_ar1")
        ar_cases.append(dict(name=name, data=data, spec=spec,
                             **fixed_block(mod, p, h=h, forecast_exog=fexog, components=True, full_states=full)))
    fx["autoregressive"] = ar_cases

    ar_mle = []
    for name, data, spec in [
        ("nile_llevel_ar1", "nile", dict(level="llevel", autoregressive=1)),
        ("infl_llevel_ar2", "infl", dict(level="llevel", autoregressive=2)),
        ("lgdp_lltrend_ar2", "lgdp", dict(level="lltrend", autoregressive=2)),
    ]:
        mod = UnobservedComponents(series[data], use_exact_diffuse=True, **spec)
        ar_mle.append(dict(name=name, data=data, spec=spec, **mle_case(mod)))
    fx["autoregressive_mle"] = ar_mle

    # Refusal evidence, NOT a golden: statsmodels has no stationarity check
    # and returns finite numbers where tsecon refuses (a unit or explosive
    # AR root has no stationary initial distribution). Recorded so the
    # documentation can quote them; the tests pin tsecon's refusal only.
    m = UnobservedComponents(nile, "llevel", autoregressive=1, use_exact_diffuse=True)
    fx["autoregressive_nonstationary_statsmodels"] = [
        dict(params=[8000.0, 1400.0, 6000.0, phi], statsmodels_loglike=float(m.loglike(np.array([8000.0, 1400.0, 6000.0, phi]))))
        for phi in (1.2, 1.0, 0.9999)
    ]

    # ------------------------------------- (c) regressors in the state vector
    # rng 20261008: a local level plus two regressors, six forecast rows, and
    # a NaN-inserted copy (t = 5, 6, 50, 119: the forecast origin missing).
    rng = np.random.default_rng(20261008)
    T2 = 120
    X2 = np.column_stack([rng.normal(size=T2), rng.uniform(-2, 2, size=T2)])
    y2 = np.cumsum(rng.normal(0, 0.4, T2)) + X2 @ [0.5, -0.3] + rng.normal(0, 0.9, T2)
    Xf2 = np.column_stack([rng.normal(size=6), rng.uniform(-2, 2, size=6)])
    y2m = y2.copy()
    y2m[[5, 6, 50, 119]] = np.nan
    fx["data"].update(sim2_y=lst(y2), sim2_y_missing=lst(y2m), sim2_x=lst(X2), sim2_x_forecast=lst(Xf2))
    series.update(sim2=y2, sim2_missing=y2m)

    def regression_components(res):
        rc = res.regression_coefficients
        k = rc.filtered.shape[0]
        return dict(
            filtered=lst(rc.filtered.T), smoothed=lst(rc.smoothed.T),
            filtered_var=lst(np.array([np.diag(rc.filtered_cov[:, :, t]) for t in range(res.nobs)])),
            smoothed_var=lst(np.array([np.diag(rc.smoothed_cov[:, :, t]) for t in range(res.nobs)])),
            smoothed_cov_last=lst(rc.smoothed_cov[:, :, -1].reshape(k, k)),
        )

    state_reg = []
    for name, data, spec, p, h, cls in [
        ("sim2_llevel", "sim2", dict(level="llevel"), [0.8, 0.3], 4, UnobservedComponents),
        ("sim2_ntrend", "sim2", dict(level="ntrend"), [0.8], 6, UnobservedComponents),
        ("sim2_lltrend_ar2_missing", "sim2_missing", dict(level="lltrend", autoregressive=2),
         [0.5, 0.2, 0.01, 0.4, 0.5, -0.2], 3, UnobservedComponents),
        # Everything at once: smooth trend, a 12-period trigonometric
        # seasonal with two harmonics (not the period-4 Nyquist pair), a
        # stationary damped cycle, an AR(1), and the regressors in the state.
        ("sim_strend_freq12h2_statcycle_ar1", "sim",
         dict(level="strend", freq_seasonal=[dict(period=12, harmonics=2)], cycle=True, damped_cycle=True,
              stochastic_cycle=True, autoregressive=1),
         [0.8, 0.02, 0.05, 0.2, 0.3, 0.6, 0.85, 0.5], 4, StationaryCycleUC),
    ]:
        Xd = X if data == "sim" else X2
        Xfd = Xf if data == "sim" else Xf2
        mod = cls(series[data], exog=Xd, mle_regression=False, use_exact_diffuse=True, **spec)
        block = fixed_block(mod, p, h=h, forecast_exog=Xfd[:h], components=True)
        block["components"]["regression_coefficients"] = regression_components(mod.smooth(np.asarray(p, float)))
        out_spec = dict(spec, exog="sim_x" if data == "sim" else "sim2_x", mle_regression=False)
        if cls is StationaryCycleUC:
            out_spec["cycle_init"] = "stationary"
        state_reg.append(dict(name=name, data=data, spec=out_spec, **block))
    fx["state_regression"] = state_reg

    # Closed form (de Jong 1991; Durbin & Koopman 2012, secs. 5.7 and 7.2):
    # white noise plus regressors in the state is diffuse-prior GLS = OLS.
    s2 = 0.8
    beta = np.linalg.solve(X2.T @ X2, X2.T @ y2)
    rss = float(np.sum((y2 - X2 @ beta) ** 2))
    xtx_inv = np.linalg.inv(X2.T @ X2)
    fx["state_regression_closed_form"] = dict(
        sigma2=s2, beta_ols=lst(beta), cov=lst(s2 * xtx_inv),
        loglike=float(-0.5 * (T2 * np.log(2 * np.pi) + T2 * np.log(s2) + rss / s2
                              + np.linalg.slogdet(X2.T @ X2 / s2)[1])),
        forecast=lst(Xf2 @ beta), forecast_var=lst(s2 * (1.0 + np.einsum("ij,jk,ik->i", Xf2, xtx_inv, Xf2))),
    )

    # The cross-mode identity: the state form is the marginal (diffuse)
    # likelihood, the MLE form the profile one, and at the GLS beta
    # l_state(theta) = l_mle(theta, beta_gls) + 1/2 ln det P_beta.
    identity = []
    for lvl, p in (("llevel", [0.8, 0.3]), ("lltrend", [0.8, 0.3, 0.01]), ("ntrend", [0.8])):
        for ar in (None, 2):
            pp = p + ([0.5, 0.4, -0.2] if ar else [])
            ms = UnobservedComponents(y2, lvl, exog=X2, autoregressive=ar, mle_regression=False,
                                      use_exact_diffuse=True)
            rs = ms.smooth(np.array(pp))
            bhat = rs.smoothed_state[-2:, -1]
            pb = rs.smoothed_state_cov[-2:, -2:, -1]
            mm = UnobservedComponents(y2, lvl, exog=X2, autoregressive=ar, use_exact_diffuse=True)
            rm = mm.smooth(np.r_[pp, bhat])
            identity.append(dict(level=lvl, autoregressive=ar, params=lst(pp), loglike_state=float(rs.llf),
                                 beta_gls=lst(bhat), loglike_mle_at_gls=float(rm.llf),
                                 half_logdet_p_beta=float(0.5 * np.linalg.slogdet(pb)[1])))
    fx["state_regression_identity"] = identity

    # The regressor-units identity: rescaling column 1 by 100 shifts the
    # state-form likelihood by exactly -ln 100 and leaves the MLE form alone.
    X2c = X2.copy()
    X2c[:, 0] *= 100.0
    fx["state_regression_units"] = dict(
        scale=100.0, params=[0.8, 0.3],
        loglike_state=float(UnobservedComponents(y2, "llevel", exog=X2, mle_regression=False,
                                                 use_exact_diffuse=True).loglike(np.array([0.8, 0.3]))),
        loglike_state_scaled=float(UnobservedComponents(y2, "llevel", exog=X2c, mle_regression=False,
                                                        use_exact_diffuse=True).loglike(np.array([0.8, 0.3]))),
        loglike_mle=float(UnobservedComponents(y2, "llevel", exog=X2, use_exact_diffuse=True)
                          .loglike(np.array([0.8, 0.3, 0.5, -0.3]))),
        loglike_mle_scaled=float(UnobservedComponents(y2, "llevel", exog=X2c, use_exact_diffuse=True)
                                 .loglike(np.array([0.8, 0.3, 0.005, -0.3]))),
    )

    mod = UnobservedComponents(y2, "llevel", exog=X2, mle_regression=False, use_exact_diffuse=True)
    sr_mle = mle_case(mod)
    res = mod.smooth(np.asarray(sr_mle["best"]["params"]))
    sr_mle["best"]["regression_coefficients"] = lst(res.smoothed_state[-2:, -1])
    sr_mle["best"]["regression_coefficients_cov"] = lst(res.smoothed_state_cov[-2:, -2:, -1])
    fx["state_regression_mle"] = [dict(name="sim2_llevel", data="sim2",
                                       spec=dict(level="llevel", exog="sim2_x", mle_regression=False), **sr_mle)]

    # ------------------------------------------------- (b) stationary cycle
    stat = []
    damped = dict(cycle=True, damped_cycle=True, stochastic_cycle=True)

    spec = dict(level="llevel", **damped)
    p = [0.3, 0.5, 2.0, 2 * np.pi / 30, 0.9]
    mod = StationaryCycleUC(lgdp, use_exact_diffuse=True, **spec)
    block = fixed_block(mod, p, h=8, components=True)
    diffuse = UnobservedComponents(lgdp, use_exact_diffuse=True, **spec).smooth(np.asarray(p))
    block["diffuse_cycle"] = dict(loglike=float(diffuse.llf), aic=float(diffuse.aic),
                                  df_model=int(diffuse.df_model), nobs_diffuse=int(diffuse.nobs_diffuse))
    block["closed_form_cycle_p_star"] = 2.0 / (1.0 - 0.9 ** 2)
    stat.append(dict(name="lgdp_llevel", data="lgdp", spec=dict(spec, cycle_init="stationary"), **block))

    spec = dict(level="llevel", **damped)
    mod = StationaryCycleUC(nile, use_exact_diffuse=True, **spec)
    stat.append(dict(name="nile_llevel", data="nile", spec=dict(spec, cycle_init="stationary"),
                     **fixed_block(mod, [10000.0, 1000.0, 3000.0, 2 * np.pi / 12, 0.7], h=6, components=True)))

    spec = dict(level="lltrend", seasonal=4, **damped)
    mod = StationaryCycleUC(y, exog=X, use_exact_diffuse=True, **spec)
    stat.append(dict(name="sim_lltrend_seasonal4_exog", data="sim",
                     spec=dict(spec, exog="sim_x", cycle_init="stationary"),
                     **fixed_block(mod, base.fixed_params_for(mod), h=8, forecast_exog=Xf)))

    # Stationary cycle and AR(1) side by side: one block-diagonal Lyapunov
    # system, P_* = sigma2/(1-rho^2) on the cycle and sigma2_ar/(1-phi^2).
    spec = dict(level="strend", autoregressive=1, **damped)
    mod = StationaryCycleUC(lgdp, use_exact_diffuse=True, **spec)
    block = fixed_block(mod, [0.01, 0.002, 0.5, 0.2, 2 * np.pi / 24, 0.9, 0.5], h=6, components=True)
    block["closed_form_p_star"] = dict(cycle=0.5 / (1 - 0.81), ar=0.2 / (1 - 0.25))
    stat.append(dict(name="lgdp_strend_ar1", data="lgdp", spec=dict(spec, cycle_init="stationary"), **block))
    fx["stationary_cycle"] = stat

    # MLE: GDP smooth trend + stationary cycle, periods 6..32 quarters.
    spec = dict(level="strend", cycle_period_bounds=[6.0, 32.0], **damped)
    mod = StationaryCycleUC(lgdp, use_exact_diffuse=True, **dict(spec, cycle_period_bounds=(6.0, 32.0)))
    gdp = mle_case(mod, stationary_cycle=True)
    fx["stationary_cycle_mle"] = [dict(name="lgdp_strend", data="lgdp", spec=dict(spec, cycle_init="stationary"),
                                       **gdp)]

    # lambda -> 0 at the GDP optimum, frequency moved only.
    best = np.asarray(gdp["best"]["params"])
    sweep = []
    for lam in (1e-2, 1e-4, 1e-6):
        pp = best.copy()
        pp[list(mod.param_names).index("frequency.cycle")] = lam
        s_ll = StationaryCycleUC(lgdp, use_exact_diffuse=True, **dict(spec, cycle_period_bounds=(2.0, 1e9))).loglike(pp)
        d_ll = UnobservedComponents(lgdp, use_exact_diffuse=True,
                                    **dict(spec, cycle_period_bounds=(2.0, 1e9))).loglike(pp)
        sweep.append(dict(frequency=lam, params=lst(pp), stationary=float(s_ll), diffuse=float(d_ll)))
    fx["stationary_cycle_lambda_sweep"] = sweep

    OUT.write_text(json.dumps(fx, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size, "bytes")
    for b in ar_cases + state_reg + stat:
        print(b["name"], "llf %.12f nobs_diffuse %d k_diffuse %d spread %.2e"
              % (b["loglike"], b["nobs_diffuse"], b["k_diffuse"], b["smoother_spread"]))
    print("gdp mle", gdp["statsmodels"]["llf"], gdp["scipy"]["llf"], gdp["best"])
    cf = fx["state_regression_closed_form"]
    print("state regression closed form llf %.12f beta %s" % (cf["loglike"], cf["beta_ols"]))
    for c in fx["state_regression_identity"]:
        print("identity", c["level"], c["autoregressive"], "%.12f %.12f" % (
            c["loglike_state"] - c["loglike_mle_at_gls"], c["half_logdet_p_beta"]))
    u = fx["state_regression_units"]
    print("units: state %.10f -> %.10f (shift %.10f), mle %.10f -> %.10f" % (
        u["loglike_state"], u["loglike_state_scaled"], u["loglike_state"] - u["loglike_state_scaled"],
        u["loglike_mle"], u["loglike_mle_scaled"]))
    print("state regression mle", sr_mle["statsmodels"]["llf"], sr_mle["scipy"]["llf"], sr_mle["best"]["params"],
          sr_mle["best"]["at_boundary"])
    for c in fx["autoregressive_nonstationary_statsmodels"]:
        print("statsmodels at a non-stationary AR", c["params"][-1], c["statsmodels_loglike"])
    for c in ar_mle:
        print("ar mle", c["name"], c["statsmodels"]["llf"], c["scipy"]["llf"], c["best"]["params"],
              c["best"]["at_boundary"], c["best"]["se_conditional"])
    for s in sweep:
        print("sweep", s["frequency"], s["stationary"], s["diffuse"])


if __name__ == "__main__":
    main()
