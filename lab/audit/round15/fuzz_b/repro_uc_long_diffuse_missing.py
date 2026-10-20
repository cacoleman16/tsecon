"""unobserved_components: a monthly basic structural model with two missing values in the
same calendar month of the first two years (t = 2 and t = 14) -- the diffuse period then
legitimately lasts 27 observations -- gets nobs_diffuse = len(y) and a wrong log-likelihood,
at fixed parameters and at the MLE. Same failure under the 0.11 options (AR component,
regressors in the state).

Expected = statsmodels UnobservedComponents(use_exact_diffuse=True), confirmed by an
extended-precision de Jong augmented Kalman filter (augkf.py next to this file).
Run from this directory with the audit venv's python.
"""
import os, sys, warnings
import numpy as np
import tsecon
from statsmodels.tsa.statespace.structural import UnobservedComponents as UC

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augkf import augmented_loglike  # noqa: E402

pat = np.array([1.0, 0.5, -0.5, -1.0, 0.2, 0.0, 0.3, -0.2, 0.8, -0.6, 0.1, -0.6])
rng = np.random.default_rng(1)
T = 120
y = np.cumsum(rng.normal(0, 0.3, T)) + np.tile(pat, 10) + rng.normal(0, 1, T)
y[[2, 14]] = np.nan

def show(label, spec, params, exog=None):
    mod = UC(y, exog=exog, use_exact_diffuse=True, **spec)
    sm = mod.smooth(np.array(params))
    aug = augmented_loglike(mod, np.array(params), y)[0]
    kw = dict(spec)
    if exog is not None:
        kw["exog"] = exog
    r = tsecon.unobserved_components(y, fixed_params=list(params), **kw)
    print(f"{label}\n  expected: loglik {sm.llf:.6f} (statsmodels), {aug:.6f} (augmented KF); nobs_diffuse {sm.nobs_diffuse}"
          f"\n  actual:   loglik {r['loglik']:.6f}; nobs_diffuse {r['nobs_diffuse']}; "
          f"std_resid all NaN: {bool(np.all(np.isnan(np.array(r['std_resid'], float))))}")

show("local level + fixed seasonal 12, params [1, .1]",
     dict(level="local level", seasonal=12, stochastic_seasonal=False), [1.0, 0.1])
show("local level + stochastic seasonal 12, params [1, .1, .05]",
     dict(level="local level", seasonal=12), [1.0, 0.1, 0.05])
show("0.11: local level + seasonal 12 + autoregressive=1, params [1, .1, .05, .5, .5]",
     dict(level="local level", seasonal=12, autoregressive=1), [1.0, 0.1, 0.05, 0.5, 0.5])
x = np.random.default_rng(5).normal(size=(T, 1))
show("0.11: local level + seasonal 12 + 1 regressor in the state (mle_regression=False), params [1, .1, .05]",
     dict(level="local level", seasonal=12, mle_regression=False), [1.0, 0.1, 0.05], exog=x)

# the MLE: tsecon's optimum, re-evaluated by statsmodels, vs statsmodels' own fit
spec = dict(level="local level", seasonal=12)
fit = tsecon.unobserved_components(y, **spec)
mod = UC(y, use_exact_diffuse=True, **spec)
smfit = mod.fit(disp=False, maxiter=500)
print(f"MLE: tsecon loglik {fit['loglik']:.6f} (nobs_diffuse {fit['nobs_diffuse']}) at {np.round(fit['params'], 5).tolist()}"
      f"\n     true diffuse loglik at tsecon's params (statsmodels): {mod.loglike(np.array(fit['params'])):.6f}"
      f"\n     statsmodels' own MLE: {smfit.llf:.6f} at {np.round(smfit.params, 5).tolist()}")
