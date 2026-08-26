"""Bitwise snapshot of the STRING-forecaster surfaces of ``backtest``,
``conformal_forecast``, and ``conformal_backtest``.

Unlike every other generator in this directory, this one deliberately CALLS
TSECON: it is not a third-party golden but a self-snapshot, captured from the
build immediately BEFORE the Python-callable forecaster plumbing landed
(0.6.0-dev, field-report item 9). Its purpose is regression, not validation:
``test_backtest_callable.py`` asserts, float-hex for float-hex, that the
pre-existing string-forecaster paths are bit-identical after the ``forecaster``
/ ``base`` arguments learned to accept callables. Do NOT regenerate it against
a build that already contains the callable plumbing unless you intend to
re-baseline (e.g. after an intentional behavioral change to the string paths,
which would belong in the CHANGELOG).

Run from the repo root with a venv holding the tsecon build to snapshot:
    python fixtures/generate_backtest_string_snapshot.py
"""
import json
from pathlib import Path

import numpy as np
import tsecon

rng = np.random.default_rng(20260826)
y = np.cumsum(rng.standard_normal(90)) + 50.0

out = {
    "_note": (
        "Self-snapshot of the string-forecaster paths, captured pre-callable-"
        "plumbing (0.6.0-dev). Floats are hex (float.hex()) so the comparison "
        "is bitwise. Series: default_rng(20260826), cumsum of 90 standard "
        "normals + 50."
    )
}


def hexes(seq):
    return [float(v).hex() for v in np.asarray(seq, dtype=float).ravel()]


# --- backtest: one config per string forecaster, exercising both windows
# and an infrequent refit cadence.
bt_cfgs = {
    "naive": dict(window="expanding", train=20, horizon=3, refit_every=1),
    "drift": dict(window="rolling", train=25, horizon=2, refit_every=3),
    "mean": dict(window="expanding", train=30, horizon=1, refit_every=1),
    "seasonal_naive": dict(window="rolling", train=24, horizon=4, refit_every=2, period=12),
    "theta": dict(window="expanding", train=40, horizon=2, refit_every=1, period=1),
}
out["backtest"] = {}
for name, cfg in bt_cfgs.items():
    r = tsecon.backtest(y, forecaster=name, **cfg)
    out["backtest"][name] = {
        "config": cfg,
        "origins": [int(t) for t in r["origins"]],
        "forecasts": [hexes(f) for f in r["forecasts"]],
        "targets": [hexes(t) for t in r["targets"]],
        "accuracy_rmse": hexes([row["rmse"] for row in r["accuracy"]]),
        "accuracy_mase": hexes([row["mase"] for row in r["accuracy"]]),
    }

# --- conformal_forecast: split (theta base, asymmetric), aci (ar base),
# enbpi (seeded).
cf = tsecon.conformal_forecast(
    y, horizon=3, method="split", base="theta", alpha=0.1, mode="asymmetric"
)
out["conformal_forecast_split"] = {
    "mean": hexes(cf["mean"]),
    "lower": hexes(cf["lower"]),
    "upper": hexes(cf["upper"]),
    "n_calib": int(cf["n_calib"]),
}
ca = tsecon.conformal_forecast(
    y, horizon=2, method="aci", base="ar", lags=2, alpha=0.1, gamma=0.01
)
out["conformal_forecast_aci"] = {
    "mean": hexes(ca["mean"]),
    "lower": hexes(ca["lower"]),
    "upper": hexes(ca["upper"]),
    "alpha_final": hexes(ca["alpha_final"]),
}
ce = tsecon.conformal_forecast(
    y, horizon=2, method="enbpi", base="ar", lags=2, n_boot=10, seed=7
)
out["conformal_forecast_enbpi"] = {
    "mean": hexes(ce["mean"]),
    "lower": hexes(ce["lower"]),
    "upper": hexes(ce["upper"]),
    "beta": hexes([ce["beta"]]),
}

# --- conformal_backtest: split with drift base.
cb = tsecon.conformal_backtest(
    y, horizon=2, method="split", base="drift", alpha=0.2, calib=15, n_eval=12
)
out["conformal_backtest_split"] = {
    "realized_coverage": hexes(cb["realized_coverage"]),
    "mean_h1": hexes(cb["mean"][0]),
    "lower_h1": hexes(cb["lower"][0]),
    "upper_h1": hexes(cb["upper"][0]),
}

path = Path(__file__).parent / "backtest_string_snapshot.json"
path.write_text(json.dumps(out, indent=1) + "\n")
print(f"wrote {path}")
