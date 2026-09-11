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
* The MLE, one criterion, two optimizers: statsmodels' own ``fit`` (L-BFGS
  in its square-root / logistic working space) and a SciPy Nelder-Mead +
  L-BFGS-B polish of the identical ``loglike`` in the identical working
  space. The Rust optimum is pinned to the BETTER of the two at optimizer
  tolerance -- never to one optimizer's stopping point -- and the
  ``cov_type="approx"`` standard errors are recorded.
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
    )
    if name in COMPONENT_CASES:
        block["components"] = components(res)
    if fc is not None:
        block["forecast"] = lst(fc.predicted_mean)
        block["forecast_var"] = lst(fc.var_pred_mean)
    return block


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
    return dict(
        param_names=list(mod.param_names),
        statsmodels=dict(params=lst(sm_params), llf=float(sm_res.llf)),
        scipy=dict(params=lst(scipy_params), llf=float(scipy_llf)),
        best=dict(params=lst(best_params), llf=float(best_llf), se_approx=lst(bse)),
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
    y_missing = y.copy()
    y_missing[10:15] = np.nan
    y_missing[40] = np.nan
    y_missing[77] = np.nan
    y_missing[120:125] = np.nan
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
        ("llevel_cycle_damped_stoch", dict(level="llevel", cycle=True, damped_cycle=True, stochastic_cycle=True)),
        ("llevel_exog", dict(level="llevel", exog=X)),
        ("strend", dict(level="strend")),
    ]:
        mod = UnobservedComponents(y, use_exact_diffuse=True, **spec)
        spec_out = {k: (v if k != "exog" else "sim.x") for k, v in spec.items()}
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
