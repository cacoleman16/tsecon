"""Refutation attempt for the small tsecon-vs-statsmodels(use_test=False)
theta divergences: are they optimizer slack on a flat SES objective?

Method: rebuild the exact model (statsmodels' own deseasonalization, SES with
l0=x0, b0 OLS slope), profile the one-step SSE in alpha, and locate both
packages' implied alphas on that profile.
"""
import numpy as np
import pandas as pd
import tsecon
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import optimize


def ses_sse(x, alpha):
    l = x[0]
    sse = 0.0
    for t in range(1, len(x)):
        e = x[t] - l
        sse += e * e
        l = alpha * x[t] + (1 - alpha) * l
    return sse


def ses_level(x, alpha):
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return l


def my_forecast(y, period, steps, alpha):
    s = pd.Series(y)
    dec = seasonal_decompose(s, model="multiplicative", period=period)
    factors = dec.seasonal[:period].to_numpy()
    x = (s / dec.seasonal).to_numpy()
    T = len(x)
    tt = np.arange(T)
    b0 = np.polyfit(tt, x, 1)[0]
    l_T = ses_level(x, alpha)
    out = []
    for h in range(1, steps + 1):
        drift = 0.5 * b0 * (h - 1 + 1.0 / alpha - (1 - alpha) ** T / alpha)
        val = l_T + drift
        out.append(val * factors[(T + h - 1) % period])
    return np.array(out)


worst = None
for k in range(30):
    rng = np.random.default_rng(9100 + k)
    y = 100 + rng.normal(0, 3, 120)
    a = tsecon.theta_forecast(y, steps=4, period=12)
    tm = ThetaModel(pd.Series(y), period=12, deseasonalize=True, use_test=False).fit()
    c = tm.forecast(4).to_numpy()
    d = np.max(np.abs(a - c) / np.abs(c))
    if worst is None or d > worst[0]:
        worst = (d, k, y, a, c, tm.params["alpha"])

d, k, y, a, c, alpha_sm = worst
print(f"worst draw k={k}: rel diff {d:.3e}, statsmodels alpha={alpha_sm:.8f}")

s = pd.Series(y)
dec = seasonal_decompose(s, model="multiplicative", period=12)
x = (s / dec.seasonal).to_numpy()

# profile the SSE
r = optimize.minimize_scalar(lambda al: ses_sse(x, al), bounds=(1e-9, 1.0), method="bounded",
                             options={"xatol": 1e-12})
alpha_star = r.x
print(f"grid-precision optimal alpha = {alpha_star:.10f}, SSE = {r.fun:.10f}")
print(f"SSE at statsmodels alpha     = {ses_sse(x, alpha_sm):.10f}")

# invert tsecon's forecast to its implied alpha (steps=1 pins l_T + drift(1))
from scipy.optimize import brentq
def implied_gap(al):
    return my_forecast(y, 12, 4, al)[0] - a[0]
try:
    lo, hi = 1e-6, 0.999999
    alpha_ts = brentq(implied_gap, lo, hi, xtol=1e-12)
    print(f"tsecon implied alpha         = {alpha_ts:.10f}, SSE = {ses_sse(x, alpha_ts):.10f}")
    print(f"SSE(tsecon) - SSE(sm)        = {ses_sse(x, alpha_ts) - ses_sse(x, alpha_sm):.3e}")
    print(f"SSE flatness: SSE(alpha*±0.01) - SSE(alpha*) = "
          f"{ses_sse(x, alpha_star + 0.01) - r.fun:.3e} / {ses_sse(x, max(alpha_star - 0.01, 1e-9)) - r.fun:.3e}")
    # my own reconstruction at each alpha vs both packages
    print("my forecast at alpha_sm vs statsmodels:",
          np.max(np.abs(my_forecast(y, 12, 4, alpha_sm) - c) / np.abs(c)))
    print("my forecast at alpha_ts vs tsecon:",
          np.max(np.abs(my_forecast(y, 12, 4, alpha_ts) - a) / np.abs(a)))
except ValueError as e:
    print("inversion failed:", e, "-> check bracket")
    print("gap at ends:", implied_gap(1e-6), implied_gap(0.999999))
