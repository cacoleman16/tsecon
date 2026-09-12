#!/usr/bin/env python
"""Golden fixtures for the structural time-series ("unobserved components")
and time-varying-parameter regression estimators of ``tsecon-ssm``
(``unobserved_components``, ``tvp_regression``) -> ``fixtures/uc.json``.

Reference implementations (this venv):

* ``statsmodels.tsa.statespace.structural.UnobservedComponents`` with
  ``use_exact_diffuse=True`` -- an INDEPENDENT-PACKAGE golden. At FIXED
  parameters: ``loglike``, filtered and smoothed states with their variances,
  one-step predictions and prediction errors, standardized residuals,
  h-step forecasts with variances, AIC/BIC (its ``df_model = k_params +
  k_diffuse_states`` convention), and the component paths (level, trend,
  seasonal, frequency-domain seasonal, cycle) for every component
  combination the Rust estimator supports, plus NaN-inserted series.
* The smoothed state variances additionally carry ``smoother_spread``: the
  largest relative disagreement between statsmodels' OWN univariate and
  conventional smoothers on the same model at the same parameters. It is
  zero (bit-identical) on most specifications and rises to ~5e-3 inside the
  diffuse period of the eight-diffuse-state combination, which is where the
  exact-diffuse smoother recursion is ill-conditioned. The Rust and Python
  tests use it as the tolerance there -- tsecon must be at least as close to
  the univariate reference as statsmodels' own second path is -- instead of
  a guessed number.
* The MLE, one criterion, two optimizers: statsmodels' own ``fit`` (L-BFGS
  in its square-root / logistic working space) and a SciPy Nelder-Mead +
  L-BFGS-B polish of the identical ``loglike`` in the identical working
  space. The Rust optimum is pinned to the BETTER of the two at optimizer
  tolerance -- never to one optimizer's stopping point. Standard errors are
  recorded twice: ``se_approx`` is statsmodels' own ``cov_type="approx"``
  (the full Hessian inverted), and ``se_conditional`` is the same Hessian
  inverted over the NON-boundary parameters only, which is what tsecon
  reports and what the tests pin. The pile-up flags themselves
  (``at_boundary``) are recomputed here from the documented rule, so the
  tests compare tsecon's flags with an independent implementation of the
  same criterion rather than with itself.
* TVP regression: the custom ``statsmodels.tsa.statespace.MLEModel``
  transcribed below (the documented state-space form: per-period design
  row ``x_t'``, identity transition and selection, diagonal state
  covariance, exact-diffuse initialization), and
  ``statsmodels.regression.recursive_ls.RecursiveLS`` as the
  zero-state-variance limit (its filtered coefficients must equal the TVP
  filter with every state variance fixed at 0, and its concentrated
  log-likelihood the TVP log-likelihood at ``sigma2_eps = scale``).

Data: the bundled public-domain Nile series (``sm.datasets.nile``; the
Durbin-Koopman 2012 local-level example), seeded simulated series, and the
UK ``Seatbelts`` monthly data (Harvey & Durbin 1986; R ``datasets``,
fetched through ``sm.datasets.get_rdataset`` AT GENERATION TIME and again
by the Python test -- the series is NOT redistributed: R's ``datasets``
package is GPL-licensed, so the fixture holds only the derived optimum
(parameters, log-likelihood, standard errors, pile-up values), never the
data or anything that reconstructs it). No published number that is not
reproduced here is quoted anywhere.

This generator NEVER imports tsecon.

Run:  .venv/bin/python fixtures/generate_uc_fixtures.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.simplefilter("ignore")

HERE = Path(__file__).parent
OUT = HERE / "uc.json"

FIXED_VALUES = {
    "sigma2.irregular": 0.8,
    "sigma2.level": 0.3,
    "sigma2.trend": 0.02,
    "sigma2.seasonal": 0.1,
    "sigma2.cycle": 0.2,
    "frequency.cycle": 0.6,
    "damping.cycle": 0.85,
    "beta.x1": 0.5,
    "beta.x2": -0.3,
}


def lst(a):
    return np.asarray(a, dtype=float).tolist()


# statsmodels' own default upper period bound is infinity when the series
# carries no frequency information; tsecon's is the sample length (see the
# cycle note in crates/tsecon-ssm/src/uc.rs). The band only matters for
# ESTIMATION -- a fixed-parameter evaluation does not consult it -- but the
# fixed cycle cases below declare it explicitly anyway so the two sides are
# provably comparing the same model.
def fixed_params_for(mod):
    out = []
    for name in mod.param_names:
        if name.startswith("sigma2.freq_seasonal"):
            out.append(0.05)
        else:
            out.append(FIXED_VALUES[name])
    return np.array(out)


def components(res):
    comp = {}
    for name in ("level", "trend", "seasonal", "cycle"):
        try:
            c = getattr(res, name)
        except Exception:
            c = None
        if c is None:
            continue
        comp[name] = dict(
            filtered=lst(c.filtered), filtered_var=lst(c.filtered_cov),
            smoothed=lst(c.smoothed), smoothed_var=lst(c.smoothed_cov),
        )
    fs = res.freq_seasonal
    if fs is not None:
        comp["freq_seasonal"] = [
            dict(filtered=lst(c.filtered), filtered_var=lst(c.filtered_cov),
                 smoothed=lst(c.smoothed), smoothed_var=lst(c.smoothed_cov))
            for c in fs
        ]
    return comp


# Component paths are deterministic sums of state entries; they are stored
# for the cases that pin the assembly convention (which state, which
# variance sum) and omitted elsewhere to keep the fixture compact.
COMPONENT_CASES = {
    "nile", "llevel", "lltrend_seasonal4", "llevel_freq12h2", "llevel_freq_two_blocks",
    "lltrend_seasonal4_cycle", "strend_freq12h2_cycle_exog", "seatbelts",
}


def smoother_spread(mod, params):
    """statsmodels' own two smoother paths on the same model and parameters.

    The exact-diffuse *smoother* is the ill-conditioned part of this family:
    on a model with many diffuse states statsmodels' univariate
    (Koopman-Durbin sequential) and conventional (matrix) smoothers disagree
    with EACH OTHER by orders of magnitude more than their filters do, and
    only inside the diffuse period. Recording that internal spread turns
    "how close must tsecon be?" into a measured quantity instead of a
    guessed tolerance: the Rust test requires tsecon to be at least as close
    to the univariate path as the conventional path is. Returns the largest
    relative disagreement of the smoothed state variances over the whole
    sample (0.0 when the two paths agree bit for bit).
    """
    def diag_var(univariate):
        mod.ssm.filter_univariate = univariate
        r = mod.smooth(params)
        return np.array([np.diag(r.smoothed_state_cov[:, :, t]) for t in range(mod.nobs)])

    a, b = diag_var(True), diag_var(False)
    mod.ssm.filter_univariate = True          # restore the reference path
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def fixed_block(mod, params, h=8, forecast_exog=None, name=None):
    res = mod.smooth(params)
    fc = res.get_forecast(h, exog=forecast_exog) if h > 0 else None
    m = mod.k_states
    block = dict(
        params=lst(params),
        param_names=list(mod.param_names),
        k_states=int(m),
        nobs_diffuse=int(res.nobs_diffuse),
        loglike=float(res.llf),
        aic=float(res.aic),
        bic=float(res.bic),
        filtered_state=lst(res.filtered_state.T),
        filtered_state_var=lst(np.array([np.diag(res.filtered_state_cov[:, :, t]) for t in range(mod.nobs)])),
        smoothed_state=lst(res.smoothed_state.T),
        smoothed_state_var=lst(np.array([np.diag(res.smoothed_state_cov[:, :, t]) for t in range(mod.nobs)])),
        fitted=lst(res.forecasts[0]),
        resid=lst(res.forecasts_error[0]),
        std_resid=lst(res.standardized_forecasts_error[0]),
        smoother_spread=smoother_spread(mod, params),
    )
    if name in COMPONENT_CASES:
        block["components"] = components(res)
    if fc is not None:
        block["forecast"] = lst(fc.predicted_mean)
        block["forecast_var"] = lst(fc.var_pred_mean)
    return block


BOUNDARY_LL_TOL = 1e-4          # the crate's pile-up criterion, transcribed
BOUNDED_EDGE_FRACTION = 1e-6


def boundary_flags(mod, params):
    """The documented pile-up rule, implemented here independently.

    A variance is AT THE BOUNDARY when setting it to exactly zero -- every
    other parameter held at its estimate -- costs less than
    ``BOUNDARY_LL_TOL`` of log-likelihood: the likelihood cannot tell the
    estimate from zero (Shephard & Harvey 1990). A bounded parameter (the
    cycle frequency and damping) is flagged when it sits within
    ``BOUNDED_EDGE_FRACTION`` of its interval width from either end.
    """
    params = np.asarray(params, float)
    ll = mod.loglike(params)
    flags = []
    for i, name in enumerate(mod.param_names):
        if name.startswith("sigma2"):
            p0 = params.copy()
            p0[i] = 0.0
            try:
                ll0 = mod.loglike(p0)
            except Exception:
                ll0 = -np.inf
            flags.append(bool(np.isfinite(ll0) and (ll - ll0) < BOUNDARY_LL_TOL))
        elif name == "frequency.cycle":
            lo, hi = mod.cycle_frequency_bound
            edge = BOUNDED_EDGE_FRACTION * (hi - lo)
            flags.append(bool(params[i] - lo < edge or hi - params[i] < edge))
        elif name == "damping.cycle":
            edge = BOUNDED_EDGE_FRACTION
            flags.append(bool(params[i] < edge or 1.0 - params[i] < edge))
        else:
            flags.append(False)
    return flags


def conditional_se(mod, params, flags):
    """Standard errors CONDITIONAL on the flagged parameters being exactly
    at their boundary: the inverse of the observed-information block over
    the free parameters only.

    This is the quantity tsecon reports, and it is not statsmodels'
    ``cov_type="approx"``: statsmodels inverts the FULL Hessian, including
    the boundary directions, where it is indefinite (on the Seatbelts BSM
    its own ``bse`` for ``sigma2.trend`` comes back NaN from a negative
    variance, and the numbers it reports for the other parameters are read
    off that same indefinite inverse). Inverting a submatrix is not the
    submatrix of an inverse, so the two disagree by a factor of 26 on
    ``sigma2.level`` there -- a difference of definition, not of accuracy.
    Both are recorded; the tests pin tsecon against this one and the
    fixture keeps ``se_approx`` so the difference stays visible. With no
    parameter flagged the two are identical.
    """
    params = np.asarray(params, float)
    hess = np.asarray(mod.hessian(params, transformed=True, approx_complex_step=True)) * mod.nobs
    free = [i for i, b in enumerate(flags) if not b]
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


def two_optimizer_mle(mod, extra_starts=()):
    """statsmodels' own fit and a SciPy NM + L-BFGS-B polish of the same
    criterion in the same working space; returns both and the better."""
    sm_res = mod.fit(method="lbfgs", maxiter=5000, disp=0, pgtol=1e-9, factr=10.0)
    sm_params = np.asarray(sm_res.params)

    def negll(u):
        try:
            v = -mod.loglike(mod.transform_params(u))
        except Exception:
            return np.inf
        return v if np.isfinite(v) else np.inf

    best = None
    starts = [mod.untransform_params(sm_params), np.asarray(mod.untransform_params(mod.start_params))]
    starts += [np.asarray(s) for s in extra_starts]
    for u0 in starts:
        r1 = minimize(negll, u0, method="Nelder-Mead",
                      options=dict(xatol=1e-10, fatol=1e-12, maxiter=40000, maxfev=40000))
        r2 = minimize(negll, r1.x, method="L-BFGS-B", options=dict(ftol=1e-15, gtol=1e-10, maxiter=5000))
        for r in (r1, r2):
            if best is None or r.fun < best.fun:
                best = r
    scipy_params = mod.transform_params(best.x)
    scipy_llf = -best.fun
    if scipy_llf >= sm_res.llf:
        best_params, best_llf = scipy_params, scipy_llf
    else:
        best_params, best_llf = sm_params, float(sm_res.llf)
    bse = np.asarray(mod.smooth(best_params, cov_type="approx").bse)
    flags = boundary_flags(mod, best_params)
    se_cond = conditional_se(mod, best_params, flags)
    return dict(
        param_names=list(mod.param_names),
        statsmodels=dict(params=lst(sm_params), llf=float(sm_res.llf)),
        scipy=dict(params=lst(scipy_params), llf=float(scipy_llf)),
        best=dict(
            params=lst(best_params),
            llf=float(best_llf),
            se_approx=lst(bse),
            se_conditional=lst(se_cond),
            at_boundary=flags,
        ),
    )


# ----------------------------------------------------------------- TVP model
class TVPRegression(MLEModel):
    """y_t = x_t' beta_t + eps_t, beta_{t+1} = beta_t + eta_t, beta_1 diffuse.

    Parameters: [sigma2_eps, sigma2_beta_1, ..., sigma2_beta_k], optimized
    as squares of unconstrained values."""

    def __init__(self, endog, exog):
        exog = np.asarray(exog, dtype=float)
        k = exog.shape[1]
        super().__init__(endog, k_states=k, k_posdef=k, initialization="diffuse")
        self.k_exog = k
        self.ssm["design"] = exog.T[np.newaxis, :, :]
        self.ssm["transition"] = np.eye(k)
        self.ssm["selection"] = np.eye(k)

    @property
    def param_names(self):
        return ["sigma2.irregular"] + [f"sigma2.beta{i + 1}" for i in range(self.k_exog)]

    @property
    def start_params(self):
        return np.r_[1.0, np.full(self.k_exog, 0.01)]

    def transform_params(self, unconstrained):
        return np.asarray(unconstrained) ** 2

    def untransform_params(self, constrained):
        return np.asarray(constrained) ** 0.5

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self.ssm["obs_cov", 0, 0] = params[0]
        self.ssm["state_cov"] = np.diag(params[1:])


def tvp_fixed_block(mod, params):
    res = mod.smooth(params)
    k = mod.k_states
    T = mod.nobs
    return dict(
        params=lst(params),
        loglike=float(res.llf),
        aic=float(res.aic),
        bic=float(res.bic),
        nobs_diffuse=int(res.nobs_diffuse),
        beta_filtered=lst(res.filtered_state.T),
        beta_filtered_var=lst(np.array([np.diag(res.filtered_state_cov[:, :, t]) for t in range(T)])),
        beta_smoothed=lst(res.smoothed_state.T),
        beta_smoothed_var=lst(np.array([np.diag(res.smoothed_state_cov[:, :, t]) for t in range(T)])),
        fitted=lst(res.forecasts[0]),
        resid=lst(res.forecasts_error[0]),
        std_resid=lst(res.standardized_forecasts_error[0]),
        smoother_spread=smoother_spread(mod, params),
    )


# ---------------------------------------------------------------- seatbelts
def load_seatbelts():
    """R ``datasets::Seatbelts`` (Harvey & Durbin 1986; UK Department of
    Transport monthly series 1969-1984) fetched live; never written to disk."""
    df = sm.datasets.get_rdataset("Seatbelts", "datasets").data
    return {c: np.asarray(df[c], dtype=float) for c in df.columns}


def main():
    fx = {"meta": dict(statsmodels=statsmodels.__version__, numpy=np.__version__, scipy=scipy.__version__,
                       fixed_values=FIXED_VALUES, note=__doc__.strip())}

    # ------------------------------------------------------------ Nile (DK)
    nile = sm.datasets.nile.load_pandas().data["volume"].values.astype(float)
    dk = np.array([15099.0, 1469.1])  # DK (2012) Table/Fig. 2.x local level MLE, as printed
    mod = UnobservedComponents(nile, "llevel", use_exact_diffuse=True)
    fx["nile"] = dict(
        y=lst(nile),
        dk_params=lst(dk),
        fixed=fixed_block(mod, dk, h=10, name="nile"),
        mle=two_optimizer_mle(mod),
    )

    # ------------------------------------------------------ simulated series
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
    # Missing pattern: an interior run, two isolated holes, and a missing
    # TAIL (so the forecast origin itself is unobserved) -- 12 of 120.
    y_missing = y.copy()
    y_missing[10:15] = np.nan
    y_missing[40] = np.nan
    y_missing[77] = np.nan
    y_missing[115:120] = np.nan
    assert np.isnan(y_missing).sum() == 12
    fx["sim"] = dict(y=lst(y), y_missing=lst(y_missing), x=lst(X), x_forecast=lst(Xf))

    specs = [
        ("llevel", dict(level="llevel")),
        ("lltrend", dict(level="lltrend")),
        ("rwdrift", dict(level="rwdrift")),
        ("dtrend", dict(level="dtrend")),
        ("strend", dict(level="strend")),
        ("rtrend", dict(level="rtrend")),
        ("dconstant", dict(level="dconstant")),
        ("ntrend", dict(level="ntrend")),
        ("rwalk", dict(level="rwalk")),
        ("lldtrend", dict(level="lldtrend")),
        ("fixed_slope_seasonal4", dict(level="fixed slope", seasonal=4)),
        ("fixed_intercept_cycle", dict(level="fixed intercept", cycle=True, stochastic_cycle=True)),
        ("lltrend_seasonal4", dict(level="lltrend", seasonal=4)),
        ("lltrend_seasonal4_det", dict(level="lltrend", seasonal=4, stochastic_seasonal=False)),
        ("llevel_seasonal12", dict(level="llevel", seasonal=12)),
        ("llevel_freq12h2", dict(level="llevel", freq_seasonal=[dict(period=12, harmonics=2)])),
        ("lltrend_freq4", dict(level="lltrend", freq_seasonal=[dict(period=4)])),
        ("llevel_freq6h1_det", dict(level="llevel", freq_seasonal=[dict(period=6, harmonics=1)],
                                   stochastic_freq_seasonal=[False])),
        ("llevel_freq_two_blocks", dict(level="llevel", freq_seasonal=[dict(period=4, harmonics=1),
                                                                        dict(period=12, harmonics=3)])),
        ("llevel_cycle", dict(level="llevel", cycle=True)),
        ("llevel_cycle_damped_stoch", dict(level="llevel", cycle=True, damped_cycle=True, stochastic_cycle=True)),
        ("llevel_cycle_damped_det", dict(level="llevel", cycle=True, damped_cycle=True)),
        ("lltrend_seasonal4_cycle", dict(level="lltrend", seasonal=4, cycle=True, damped_cycle=True,
                                         stochastic_cycle=True)),
        ("llevel_exog", dict(level="llevel", exog=X)),
        ("lltrend_seasonal4_exog", dict(level="lltrend", seasonal=4, exog=X)),
        ("strend_freq12h2_cycle_exog", dict(level="strend", freq_seasonal=[dict(period=12, harmonics=2)],
                                            cycle=True, damped_cycle=True, stochastic_cycle=True, exog=X)),
    ]
    cases = []
    for name, spec in specs:
        mod = UnobservedComponents(y, use_exact_diffuse=True, **spec)
        params = fixed_params_for(mod)
        fexog = Xf if "exog" in spec else None
        spec_out = {k: (v if k != "exog" else "sim.x") for k, v in spec.items()}
        cases.append(dict(name=name, spec=spec_out, **fixed_block(mod, params, h=8, forecast_exog=fexog, name=name)))
    fx["cases"] = cases

    missing = []
    for name, spec in [
        ("lltrend_seasonal4", dict(level="lltrend", seasonal=4)),
        ("llevel_cycle_damped_stoch_exog", dict(level="llevel", cycle=True, damped_cycle=True,
                                                stochastic_cycle=True, exog=X)),
        ("llevel_freq12h2", dict(level="llevel", freq_seasonal=[dict(period=12, harmonics=2)])),
    ]:
        mod = UnobservedComponents(y_missing, use_exact_diffuse=True, **spec)
        params = fixed_params_for(mod)
        fexog = Xf if "exog" in spec else None
        spec_out = {k: (v if k != "exog" else "sim.x") for k, v in spec.items()}
        missing.append(dict(name=name, spec=spec_out, **fixed_block(mod, params, h=8, forecast_exog=fexog)))
    fx["missing"] = missing

    mle_cases = []
    for name, spec in [
        ("llevel", dict(level="llevel")),
        ("lltrend_seasonal4", dict(level="lltrend", seasonal=4)),
        # The cycle MLE case carries EXPLICIT period bounds, and so must
        # any honest comparison of optimizers on this criterion. With the
        # period unbounded above, the exact-diffuse log-likelihood of a
        # stochastic cycle is unbounded above too (as lambda -> 0+ the
        # second cycle state becomes weakly observable and its diffuse
        # resolution contributes -(ln 2 pi + ln F_inf)/2 -> +inf), so
        # "which optimizer got higher" measures who walked further into a
        # singularity. statsmodels does not meet it because its diffuse
        # tolerance is an absolute 1e-10 on F_inf; tsecon's is relative and
        # does. Bounds of 6-20 are the business-cycle range the simulated
        # series was built with (period 10).
        ("llevel_cycle_damped_stoch", dict(level="llevel", cycle=True, damped_cycle=True,
                                           stochastic_cycle=True, cycle_period_bounds=(6.0, 20.0))),
        ("llevel_exog", dict(level="llevel", exog=X)),
        ("strend", dict(level="strend")),
    ]:
        mod = UnobservedComponents(y, use_exact_diffuse=True, **spec)
        spec_out = {k: (list(v) if k == "cycle_period_bounds" else v if k != "exog" else "sim.x")
                    for k, v in spec.items()}
        mle_cases.append(dict(name=name, spec=spec_out, **two_optimizer_mle(mod)))
    fx["mle_cases"] = mle_cases

    # ------------------------------------------------------------ seatbelts
    sb = load_seatbelts()
    ys = np.log(sb["drivers"])
    Xs = np.column_stack([np.log(sb["PetrolPrice"]), np.log(sb["kms"]), sb["law"]])
    mod = UnobservedComponents(ys, level="lltrend", seasonal=12, exog=Xs, use_exact_diffuse=True)
    mle = two_optimizer_mle(mod)
    fx["seatbelts"] = dict(
        source="R datasets::Seatbelts via sm.datasets.get_rdataset('Seatbelts', 'datasets') -- not stored",
        transform="y = log(drivers); exog = [log(PetrolPrice), log(kms), law]",
        spec=dict(level="lltrend", seasonal=12),
        nobs=int(mod.nobs), k_states=int(mod.k_states),
        nobs_diffuse=int(mod.smooth(np.asarray(mle["best"]["params"])).nobs_diffuse),
        mle=mle,
    )

    # ------------------------------------------------------------------ TVP
    rng = np.random.default_rng(777)
    Tt = 200
    Xt = np.column_stack([np.ones(Tt), rng.normal(size=Tt), rng.uniform(-1, 1, size=Tt)])
    true_q = np.array([0.01, 0.0025, 0.0])
    beta = np.zeros((Tt, 3))
    beta[0] = [1.0, 0.5, -0.5]
    for t in range(1, Tt):
        beta[t] = beta[t - 1] + rng.normal(0, np.sqrt(true_q))
    yt = np.sum(Xt * beta, axis=1) + rng.normal(0, 1.0, Tt)
    yt_missing = yt.copy()
    yt_missing[20:25] = np.nan
    yt_missing[100] = np.nan
    tvp = TVPRegression(yt, Xt)
    fixed = np.array([1.0, 0.01, 0.0025, 0.001])
    rls = RecursiveLS(yt, Xt).fit()
    rls_block = dict(
        llf=float(rls.llf), scale=float(rls.scale), params=lst(rls.params),
        filtered_coefficients=lst(rls.recursive_coefficients.filtered.T),
        nobs_diffuse=int(rls.nobs_diffuse),
    )
    tvp_mle = two_optimizer_mle(tvp, extra_starts=[np.sqrt([1.0, 1e-3, 1e-3, 1e-3])])
    fx["tvp"] = dict(
        y=lst(yt), x=lst(Xt[:, 1:]), y_missing=lst(yt_missing), truth=dict(sigma2_eps=1.0, sigma2_beta=lst(true_q)),
        fixed=tvp_fixed_block(tvp, fixed),
        rls=rls_block,
        rls_limit=tvp_fixed_block(tvp, np.array([rls.scale, 0.0, 0.0, 0.0])),
        mle=tvp_mle,
        missing=tvp_fixed_block(TVPRegression(yt_missing, Xt), fixed),
    )

    OUT.write_text(json.dumps(fx, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size, "bytes")
    print("nile mle", fx["nile"]["mle"]["best"])
    print("seatbelts mle", mle["best"])
    print("tvp mle", tvp_mle["best"], "rls", rls_block["llf"], rls_block["scale"])
    for c in mle_cases:
        print(c["name"], c["statsmodels"]["llf"], c["scipy"]["llf"])


if __name__ == "__main__":
    main()
