"""Precompute every scenario for the interactive demo page.

All numbers come from the real tsecon Rust core — the page only switches
between and renders them. Run: .venv/bin/python docs/demo/generate_demo_data.py
"""
import json
from pathlib import Path

import numpy as np
import tsecon

OUT = Path(__file__).parent / "demo_data.json"
FIX = Path(__file__).parents[2] / "fixtures"
rng = np.random.default_rng(20260716)
r4 = lambda a: [round(float(x), 4) for x in np.asarray(a).ravel()]


def simulate_arma(phi=0.0, theta=0.0, n=240, seed=1):
    r = np.random.default_rng(seed)
    e = r.standard_normal(n + 60)
    y = np.empty(n + 60)
    y[0] = 0.0
    for t in range(1, n + 60):
        y[t] = phi * y[t - 1] + e[t] + theta * e[t - 1]
    return y[60:]


# 1 · Identify: process explorer
identify = []
for label, phi, theta in [("AR(1) φ=-0.8", -0.8, 0), ("AR(1) φ=0.3", 0.3, 0),
                          ("AR(1) φ=0.6", 0.6, 0), ("AR(1) φ=0.9", 0.9, 0),
                          ("MA(1) θ=0.8", 0.0, 0.8), ("White noise", 0.0, 0.0)]:
    y = simulate_arma(phi, theta, seed=int(10 + phi * 10 + theta * 100))
    identify.append({
        "label": label,
        "series": r4(y[:160]),
        "acf": r4(tsecon.acf(y, nlags=20)["acf"]),
        "pacf": r4(tsecon.pacf(y, nlags=20)),
        "band": round(1.96 / np.sqrt(len(y)), 4),
    })

# 2 · Stationarity workflow
stationarity = []
base_rw = np.cumsum(0.08 + rng.standard_normal(300))
for label, y in [("Stationary AR(1)", simulate_arma(0.6, 0, 300, seed=41)),
                 ("Random walk + drift", base_rw),
                 ("Same walk, differenced", np.diff(base_rw)),
                 ("White noise", np.random.default_rng(7).standard_normal(300))]:
    rep = tsecon.check_stationarity(y)
    stationarity.append({
        "label": label, "series": r4(y[:220]),
        "quadrant": rep["quadrant"], "recommendation": rep["recommendation"],
        "adf_p": round(rep["adf_p_value"], 4), "kpss_p": round(rep["kpss_p_value"], 4),
    })

# 3 · Kalman gap-bridging
level = np.cumsum(rng.standard_normal(160) * 0.7) + 20
obs = level + rng.standard_normal(160) * 2.2
kalman = {"truth": r4(level), "observed": r4(obs), "gaps": []}
for width in [0, 15, 30, 45]:
    y = obs.copy()
    s, e = 80 - width // 2, 80 + (width + 1) // 2
    if width:
        y[s:e] = np.nan
    r = tsecon.local_level_smooth(y, sigma2_eps=2.2**2, sigma2_eta=0.7**2)
    kalman["gaps"].append({
        "width": width, "start": int(s) if width else None, "end": int(e) if width else None,
        "smoothed": r4(r["smoothed_state"]),
        "band": r4(1.96 * np.sqrt(np.asarray(r["smoothed_state_var"]))),
        "loglik": round(r["loglik"], 2),
    })

# 4 · VAR impulse responses (same DGP as the gallery)
n = 400
e3 = rng.standard_normal((n + 100, 3))
yv = np.zeros((n + 100, 3))
A1 = np.array([[0.5, 0, 0], [0.35, 0.45, -0.15], [0.15, 0.25, 0.6]])
A2 = np.array([[0.1, 0, 0], [0.1, 0.1, -0.05], [0, 0.1, 0.1]])
for t in range(2, n + 100):
    yv[t] = A1 @ yv[t - 1] + A2 @ yv[t - 2] + e3[t]
vdata = yv[100:]
var_block = {
    "names": ["Demand", "Output", "Policy rate"],
    "orth": [[[round(v, 4) for v in row] for row in m] for m in tsecon.var_irf(vdata, lags=2, horizon=16, orth=True)],
    "nonorth": [[[round(v, 4) for v in row] for row in m] for m in tsecon.var_irf(vdata, lags=2, horizon=16, orth=False)],
    "fevd_output": [[round(v, 4) for v in row] for row in np.array(tsecon.var_fevd(vdata, lags=2, horizon=16))[1]],
}

# 5 · Bayesian shrinkage: tightness dial
macro = np.array(json.loads((FIX / "bvar_niw.json").read_text())["data"])
bayes = {"names": ["GDP growth", "Consumption", "Investment"], "lams": []}
for lam in [0.05, 0.2, 1.0]:
    fit = tsecon.bvar_fit(macro, lags=2, lambda1=lam)
    draws = np.array(tsecon.bvar_irf_draws(macro, lags=2, horizon=12, n_draws=600, seed=42, lambda1=lam))
    q = np.quantile(draws[:, :, :, 0], [0.05, 0.5, 0.95], axis=0)  # responses to GDP shock
    bayes["lams"].append({
        "lambda1": lam, "lml": round(fit["log_marginal_likelihood"], 1),
        "q05": [r4(q[0, :, i]) for i in range(3)],
        "q50": [r4(q[1, :, i]) for i in range(3)],
        "q95": [r4(q[2, :, i]) for i in range(3)],
    })

# 6 · GARCH: fitted vol + forecast, normal vs t
ret = np.array(json.loads((FIX / "garch.json").read_text())["returns"])[:1200]
garch = {"returns": r4(ret), "fits": []}
for dist in ["normal", "t"]:
    g = tsecon.garch_fit(ret, vol="garch", mean="zero", dist=dist, forecast_horizon=80)
    garch["fits"].append({
        "dist": dist,
        "params": {k: round(float(v), 4) for k, v in zip(g["param_names"], g["params"])},
        "se": {k: round(float(v), 4) for k, v in zip(g["param_names"], g["se_robust"])},
        "vol": r4(g["conditional_volatility"]),
        "fc_vol": r4(np.sqrt(g["variance_forecast"])),
        "loglik": round(g["loglik"], 2), "aic": round(g["aic"], 1),
    })

# 7 · ARIMA fan chart
g1 = np.empty(220 + 60)
g1[0] = 0.0
ee = np.random.default_rng(5).standard_normal(280)
for t in range(1, 280):
    g1[t] = 0.275 + 0.45 * g1[t - 1] + ee[t] * 0.9 + 0.35 * ee[t - 1]
lvl = 100 + np.cumsum(g1[60:])
fit = tsecon.arima_fit(lvl, p=1, d=1, q=1, constant=True, forecast_steps=20)
arima = {
    "history": r4(lvl[-90:]),
    "mean": r4(fit["forecast_mean"]), "se": r4(fit["forecast_se"]),
    "params": {k: round(float(v), 4) for k, v in zip(fit["param_names"], fit["params"])},
    "loglik": round(fit["loglik"], 2),
}

# Philox parity statement (verified live at generation time)
ours = tsecon.philox_uniforms(42, 5)
theirs = np.random.Generator(np.random.Philox(42)).random(5)
assert ours.tobytes() == theirs.tobytes()

data = {
    "meta": {"generated_by": "tsecon 0.0.1 (Rust core)", "date": "2026-07-16",
             "note": "every number precomputed by the real library; the page only renders",
             "philox_first5": [f"{x:.17f}" for x in ours]},
    "identify": identify, "stationarity": stationarity, "kalman": kalman,
    "var": var_block, "bayes": bayes, "garch": garch, "arima": arima,
}
OUT.write_text(json.dumps(data, separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
