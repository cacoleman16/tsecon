"""Panel golden fixtures from Kevin Sheppard's linearmodels package.

Run with the project venv:  .venv/bin/python fixtures/generate_panel_fixtures.py

Simulated balanced panel with entity fixed effects and a known dynamic
response to an observed common shock, so a panel local projection has a
known true IRF. PanelOLS pins the within (FE) estimator with three
covariance estimators: clustered by entity, Driscoll-Kraay (kernel), and
nonrobust.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import linearmodels
from linearmodels.panel import PanelOLS

OUT = Path(__file__).parent
META = {"linearmodels": linearmodels.__version__, "numpy": np.__version__,
        "python": platform.python_version()}

rng = np.random.default_rng(88)
N, T = 30, 120  # entities x periods

# Common observed shock with known dynamic effect: y responds with IRF
# psi_h = 0.8 * 0.6^h; entity fixed effects; AR-ish idiosyncratic noise.
shock = rng.standard_normal(T)
alpha = rng.normal(0, 2.0, N)
psi = 0.8 * 0.6 ** np.arange(8)
y = np.empty((N, T))
for i in range(N):
    u = np.empty(T)
    u[0] = rng.standard_normal()
    for t in range(1, T):
        u[t] = 0.3 * u[t - 1] + rng.standard_normal()
    resp = np.convolve(shock, psi)[:T]
    y[i] = alpha[i] + resp + u + 0.3 * rng.standard_normal(T)

# One PanelOLS golden: regress y_{i,t} on shock_t and shock_{t-1} with entity FE.
rows = []
for i in range(N):
    for t in range(1, T):
        rows.append((i, t, y[i, t], shock[t], shock[t - 1]))
df = pd.DataFrame(rows, columns=["entity", "time", "y", "s0", "s1"]).set_index(["entity", "time"])

res = {}
mod = PanelOLS(df["y"], df[["s0", "s1"]], entity_effects=True)
for name, kw in [
    ("nonrobust", {"cov_type": "unadjusted"}),
    ("cluster_entity", {"cov_type": "clustered", "cluster_entity": True}),
    ("driscoll_kraay", {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 4}),
]:
    r = mod.fit(**kw)
    res[name] = {
        "params": {k: float(v) for k, v in r.params.items()},
        "bse": {k: float(v) for k, v in r.std_errors.items()},
        "tstats": {k: float(v) for k, v in r.tstats.items()},
    }
    res[name]["nobs"] = int(r.nobs)

out = {
    "_meta": META,
    "panel": {
        "n_entities": N, "n_periods": T,
        "y": [[round(float(v), 6) for v in row] for row in y],
        "shock": [round(float(v), 6) for v in shock],
        "true_irf_psi": [round(float(v), 6) for v in psi],
        "note": "y[i,t] = alpha_i + sum_h psi_h shock_{t-h} + AR(0.3) noise + meas. noise",
    },
    "panel_ols_fe_s0_s1_drop_t0": res,
}
path = OUT / "panel.json"
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)")
