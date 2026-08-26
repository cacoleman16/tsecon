"""Sharpen the theta use_test probe: iid data (seasonality pre-test should
decline), then measure tsecon vs statsmodels default."""
import numpy as np
import pandas as pd
import tsecon
from statsmodels.tsa.forecasting.theta import ThetaModel


def sm_theta(y, period, steps, use_test):
    tm = ThetaModel(pd.Series(y), period=period, deseasonalize=True, use_test=use_test)
    return tm.fit().forecast(steps).to_numpy()


div_default = 0
div_notest = 0
worst_default = 0.0
worst_notest = 0.0
reps = 30
for k in range(reps):
    rng = np.random.default_rng(9100 + k)
    y = 100 + rng.normal(0, 3, 120)  # iid positive: no seasonality, no trend persistence
    a = tsecon.theta_forecast(y, steps=4, period=12)
    b = sm_theta(y, 12, 4, use_test=True)
    c = sm_theta(y, 12, 4, use_test=False)
    dd = np.max(np.abs(a - b) / np.abs(b))
    dn = np.max(np.abs(a - c) / np.abs(c))
    worst_default = max(worst_default, dd)
    worst_notest = max(worst_notest, dn)
    if dd > 1e-6:
        div_default += 1
    if dn > 1e-6:
        div_notest += 1
print(f"iid data, period=12, {reps} draws:")
print(f"  vs statsmodels DEFAULT (use_test=True): {div_default}/{reps} diverge, worst rel {worst_default:.3e}")
print(f"  vs use_test=False:                     {div_notest}/{reps} diverge, worst rel {worst_notest:.3e}")

# where does the residual use_test=False mismatch come from? alpha at boundary?
rng = np.random.default_rng(80809)
y_ns = 50 + np.cumsum(rng.normal(0, 1, 144))
tm = ThetaModel(pd.Series(y_ns), period=12, deseasonalize=True, use_test=False).fit()
print("statsmodels alpha/b0:", tm.params["alpha"], tm.params["b0"])
