"""Post-fix checker probe: the round-8 fixes are doc-only; prove the rebuilt
module's numerics are bit-identical to the released 0.4.0 build on the touched
surfaces (copula, theta, acm).

Run once with each interpreter; compares JSON dumps.
"""
import json
import sys

import numpy as np
import tsecon

rng = np.random.default_rng(80812)
n = 300
z = rng.multivariate_normal([0, 0], [[1, 0.55], [0.55, 1]], size=n)
u = tsecon.pseudo_obs(z)
out = {}
f = tsecon.copula_fit(u, family="t")
out["copula_t"] = [f["rho"], f["nu"], f["loglik"]]
f2 = tsecon.copula_fit(u, family="frank", method="tau")
out["copula_frank_tau"] = [f2["theta"], f2["tau"]]

t = np.arange(120)
y = 50 + 0.2 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 120)
out["theta"] = list(tsecon.theta_forecast(y, steps=6, period=12))

T, mats = 240, np.arange(1, 61)
L = 0.04 + 0.001 * np.cumsum(rng.standard_normal(T)) * 0.01
yy = np.clip(L[:, None] + 0.0005 * rng.standard_normal((T, len(mats))), 1e-4, None)
r = tsecon.acm_term_premium(yy, mats, n_factors=3)
out["acm"] = [float(np.asarray(r["term_premium"])[100, 30]), float(r["delta0"])]

print(json.dumps(out, sort_keys=True))
