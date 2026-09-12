"""Bitwise snapshot of the BALANCED-panel surfaces of ``panel_fe``,
``panel_distributed_lag``, ``panel_lp``, ``lp_did`` and ``mean_group_var``.

Like ``generate_backtest_string_snapshot.py`` — and unlike every other
generator in this directory — this one deliberately CALLS TSECON: it is a
self-snapshot, not a third-party golden. It was captured from the 0.9.0
build (commit ``dde820d``) immediately BEFORE the observation mask for
unbalanced panels landed in the panel crate (0.10.0). Its purpose is
regression, not validation: ``bindings/python/tests/test_panel_unbalanced.py``
asserts, float-hex for float-hex, that every balanced-panel call is
bit-identical after ``PanelData`` learned to carry a mask and the within
estimator, the covariances and the lag designs learned to skip masked
cells. Do NOT regenerate it against a build that already contains the mask
unless you intend to re-baseline after an intentional behavioural change
to the balanced paths (which would belong in the CHANGELOG).

Run from the repo root with a venv holding the tsecon build to snapshot:
    python fixtures/generate_panel_balanced_snapshot.py

The inputs are seeded NumPy draws stored alongside the outputs (so the test
needs no DGP code): a balanced N = 8 x T = 40 panel with entity effects, a
common shock with a known dynamic response, a weather-like regressor for
the distributed-lag calls, and a staggered absorbing treatment for LP-DiD.
"""
import json
from pathlib import Path

import numpy as np
import tsecon

rng = np.random.default_rng(20260911)
N, T = 8, 40
alpha = rng.normal(0.0, 1.0, N)
delta = rng.normal(0.0, 0.5, T)
shock = rng.standard_normal(T)
psi = 0.8 * 0.6 ** np.arange(8)
temp = rng.normal(20.0, 4.0, N)[:, None] + rng.standard_normal((N, T))
y = np.empty((N, T))
for i in range(N):
    u = np.empty(T)
    u[0] = rng.standard_normal()
    for t in range(1, T):
        u[t] = 0.4 * u[t - 1] + rng.standard_normal()
    y[i] = alpha[i] + delta + np.convolve(shock, psi)[:T] + 0.3 * temp[i] - 0.0075 * temp[i] ** 2 + u
x2 = rng.standard_normal((N, T))
# staggered absorbing adoption: entities 0..4 switch on at period 10 + 4 i,
# entities 5..7 are never treated (so clean controls exist at every horizon)
treat = np.zeros((N, T))
for i in range(5):
    treat[i, 10 + 4 * i:] = 1.0
entities = [rng.standard_normal((30 + 5 * i, 2)) for i in range(4)]


def hexes(seq):
    return [float(v).hex() for v in np.asarray(seq, dtype=float).ravel()]


out = {
    "_note": (
        "Self-snapshot of the balanced-panel surfaces, captured from the 0.9.0 "
        "build (dde820d) before the observation mask landed. Floats are hex "
        "(float.hex()) so the comparison is bitwise; inputs are stored at full "
        "precision as plain lists."
    ),
    "inputs": {
        "y": y.tolist(),
        "temp": temp.tolist(),
        "x2": x2.tolist(),
        "shock": shock.tolist(),
        "treat": treat.tolist(),
        "entities": [e.tolist() for e in entities],
    },
    "panel_fe": {},
    "panel_distributed_lag": {},
    "panel_lp": {},
    "lp_did": {},
    "mean_group_var": {},
}

X = np.array([temp, x2])
for se_type, kw in [("nonrobust", {}), ("cluster", {}), ("driscoll_kraay", {"bandwidth": 3.0})]:
    r = tsecon.panel_fe(y, X, se_type=se_type, **kw)
    out["panel_fe"][se_type] = {
        "kwargs": {"se_type": se_type, **kw},
        "params": hexes(r["params"]),
        "bse": hexes(r["bse"]),
        "tvalues": hexes(r["tvalues"]),
    }

dl_cases = {
    "two_way_L2_quad": dict(lags=2, powers=2, entity_effects=True, time_effects=True, entity_trends=False),
    "entity_L1": dict(lags=1, powers=1, entity_effects=True, time_effects=False, entity_trends=False),
    "time_L0": dict(lags=0, powers=1, entity_effects=False, time_effects=True, entity_trends=False),
    "trends_L1": dict(lags=1, powers=1, entity_effects=True, time_effects=False, entity_trends=True),
    "trends_time_L1_quad": dict(lags=1, powers=2, entity_effects=True, time_effects=True, entity_trends=True),
}
for name, cfg in dl_cases.items():
    for se_type, kw in [("nonrobust", {}), ("cluster", {}), ("driscoll_kraay", {"bandwidth": 4.0})]:
        call = dict(cfg, se_type=se_type, **kw)
        if cfg["powers"] == 2:
            call["eval_points"] = np.array([15.0, 20.0, 25.0])
        r = tsecon.panel_distributed_lag(y, np.array([temp]), **call)
        rec = {
            "kwargs": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in call.items()},
            "params": hexes(r["params"]),
            "bse": hexes(r["bse"]),
            "cov": hexes(np.asarray(r["cov"])),
            "cumulative_effect": hexes(r["cumulative_effect"]),
            "cumulative_se": hexes(r["cumulative_se"]),
            "nobs": int(r["nobs"]),
            "df_resid": int(r["df_resid"]),
        }
        if cfg["powers"] == 2:
            rec["marginal_effect"] = hexes(r["marginal_effect"])
            rec["marginal_se"] = hexes(r["marginal_se"])
            rec["turning_point"] = hexes(r["turning_point"])
            rec["turning_point_se"] = hexes(r["turning_point_se"])
        out["panel_distributed_lag"][f"{name}/{se_type}"] = rec

lp_cases = {
    "dk": dict(horizon=4, n_lag_controls=2, se_type="driscoll_kraay"),
    "cluster_cumulative": dict(horizon=3, n_lag_controls=1, se_type="cluster", cumulative=True),
    "nonrobust_nolags": dict(horizon=2, n_lag_controls=0, se_type="nonrobust"),
    "jackknife": dict(horizon=3, n_lag_controls=1, se_type="driscoll_kraay", jackknife=True),
    "spj": dict(horizon=3, n_lag_controls=1, se_type="cluster", bias_correction="spj"),
}
for name, cfg in lp_cases.items():
    r = tsecon.panel_lp(y, shock, **cfg)
    out["panel_lp"][name] = {
        "kwargs": cfg,
        "irf": hexes(r["irf"]),
        "se": hexes(r["se"]),
        "nobs": [int(v) for v in r["nobs"]],
    }

did_cases = {
    "vw": dict(pre_window=3, post_window=4),
    "ew_pooled": dict(pre_window=3, post_window=4, reweight=True, pooled=True),
    "never_treated": dict(pre_window=2, post_window=3, never_treated_only=True, pooled=True),
}
for name, cfg in did_cases.items():
    r = tsecon.lp_did(y, treat, **cfg)
    rec = {
        "kwargs": cfg,
        "coef": hexes(r["coef"]),
        "se": hexes(r["se"]),
        "nobs": [int(v) for v in r["nobs"]],
        "n_switchers": [int(v) for v in r["n_switchers"]],
    }
    if cfg.get("pooled"):
        rec["pooled_post"] = hexes([r["pooled_post_att"], r["pooled_post_se"]])
        rec["pooled_pre"] = hexes([r["pooled_pre_att"], r["pooled_pre_se"]])
    out["lp_did"][name] = rec

r = tsecon.mean_group_var(entities, lags=1, horizon=4)
out["mean_group_var"]["lag1"] = {
    "kwargs": {"lags": 1, "horizon": 4},
    "intercept": hexes(r["intercept"]),
    "coefs": hexes(np.asarray(r["coefs"])),
    "orth_irfs": hexes(np.asarray(r["orth_irfs"])),
    "irf_path_se": hexes(r["irf_path_se"]),
}

path = Path(__file__).parent / "panel_balanced_snapshot.json"
path.write_text(json.dumps(out, indent=1) + "\n")
print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB); tsecon {tsecon.__version__} from {tsecon.__file__}")
