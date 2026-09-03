"""Sweep G — complexity cliffs: time every callable at T in {200, 800, 3200}.

Each (function, T) runs in a fresh subprocess with a wall-clock cap so a
runaway call cannot take the sweep down. The log-log slope over the three
sizes is fitted by least squares; the flags are
  (a) slope > LINEAR_FLAG for a function on the LINEAR_EXPECTED list,
  (b) any call over 5 s at T=3200,
  (c) non-monotone timings (t(800) < t(200) by > 2x, or t(3200) < t(800)).
A second pass re-times every flagged cell once to refute noise.

Run:  .venv-wt/bin/python lab/audit/round11/sweep_g_timing.py [--only name,...]
Out:  lab/audit/round11/out/sweep_g.log, sweep_g.json
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, log  # noqa: E402
from registry import NAMES  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SIZES = (200, 800, 3200)
CAP_S = 180
LINEAR_FLAG = 1.35
LINEAR_EXPECTED = {
    "acf", "pacf", "ljung_box", "jarque_bera", "arch_lm", "hp_filter", "bk_filter",
    "cf_filter", "hamilton_filter", "bn_filter", "periodogram", "welch", "coherence",
    "bootstrap_indices", "philox_uniforms", "cv_splits", "long_run_variance",
    "realized_measures", "realized_quarticity", "tripower_quarticity", "bns_jump_test",
    "realized_range", "pseudo_obs", "accuracy", "dm_test", "gw_test", "cw_test", "ols",
    "theta_forecast", "stl", "mstl", "seasonal_strength", "local_level_smooth",
    "ar_loglik", "optimal_block_length", "var_backtest", "mcmc_diagnostics",
    "frac_diff", "frac_integrate", "ridge", "lasso", "elastic_net", "var_fit",
    "var_irf", "var_fevd", "var_forecast", "var_granger", "johansen", "vecm",
    "engle_granger", "adf", "kpss", "phillips_perron", "ou_fit", "spread_zscore",
    "heteroskedasticity_test", "reset_test", "chow_test", "cusum_test", "umidas",
    "har_rv", "connectedness", "forecast_efficiency", "cg_regression",
    "long_memory_d", "jarque_bera", "gpd_fit", "gev_fit", "copula_fit",
    "factor_model", "functional_pca", "lp", "lp_iv", "lp_multiplier", "lp_state",
    "panel_fe", "panel_lp", "quantile_regression", "iv_gmm", "recession_probit",
    "predictive_regression", "ivx_test", "structural_fevd", "long_run_svar",
    "historical_decomposition", "max_share_svar", "proxy_svar", "proxy_first_stage",
    "sup_f_test", "check_stationarity", "ndiffs", "nsdiffs", "box_cox_lambda",
}

CHILD = r"""
import sys, time, json
sys.path.insert(0, %r)
import numpy as np
import tsecon
from registry import build
name, T = sys.argv[1], int(sys.argv[2])
args, kwargs = build(name, T=T, seed=0)
if len(sys.argv) > 3 and sys.argv[3] == "defaults":
    # keep only the required positional arguments
    import inspect
    fn = getattr(tsecon._core, name, None) or getattr(tsecon, name)
    try:
        sig = inspect.signature(fn)
        n_req = sum(1 for p in sig.parameters.values() if p.default is p.empty)
    except Exception:
        n_req = len(args)
    args, kwargs = args[:n_req], {}
fn = getattr(tsecon, name)
t0 = time.perf_counter()
fn(*args, **kwargs)
print(json.dumps({"t": time.perf_counter() - t0}))
"""


def time_one(name, T, mode="canonical"):
    cmd = [sys.executable, "-c", CHILD % HERE, name, str(T)] + (["defaults"] if mode == "defaults" else [])
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=CAP_S, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"t": None, "status": f"TIMEOUT>{CAP_S}s"}
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        return {"t": None, "status": "ERROR: " + (tail[-1] if tail else "?")[:200], "wall": wall}
    try:
        t = json.loads(p.stdout.strip().splitlines()[-1])["t"]
    except Exception:  # noqa: BLE001
        return {"t": None, "status": "PARSE: " + p.stdout[-200:], "wall": wall}
    return {"t": t, "status": "ok", "wall": wall}


def slope(ts):
    xs = [math.log(s) for s in SIZES]
    ys = [math.log(max(t, 1e-6)) for t in ts]
    mx, my = sum(xs) / 3, sum(ys) / 3
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    names = only or NAMES
    fh = open(os.path.join(OUT, "sweep_g.log"), "a" if only else "w")
    report = {}
    for name in names:
        rec = {"canonical": {}, "flags": []}
        for T in SIZES:
            rec["canonical"][T] = time_one(name, T)
        ts = [rec["canonical"][T]["t"] for T in SIZES]
        if all(t is not None for t in ts):
            rec["slope"] = slope(ts)
            if name in LINEAR_EXPECTED and rec["slope"] > LINEAR_FLAG:
                rec["flags"].append(f"superlinear slope {rec['slope']:.2f} (expected ~1)")
            if ts[2] > 5.0:
                rec["flags"].append(f"{ts[2]:.1f}s at T=3200 (canonical kwargs)")
            if ts[1] < ts[0] / 2 or ts[2] < ts[1]:
                rec["flags"].append("non-monotone")
        else:
            rec["slope"] = None
            rec["flags"].append("incomplete: " + "; ".join(f"T={T}:{rec['canonical'][T]['status']}" for T in SIZES if rec["canonical"][T]["t"] is None))
        # defaults-only pass at T=3200
        rec["defaults_3200"] = time_one(name, 3200, "defaults")
        d = rec["defaults_3200"]
        if d["t"] is not None and d["t"] > 5.0:
            rec["flags"].append(f"{d['t']:.1f}s at T=3200 with DEFAULT arguments")
        elif d["t"] is None and d["status"].startswith("TIMEOUT"):
            rec["flags"].append(f"defaults at T=3200: {d['status']}")
        # refute noise: re-time flagged cells once
        if any("superlinear" in f or "non-monotone" in f for f in rec["flags"]):
            rec["retime"] = {T: time_one(name, T) for T in SIZES}
            ts2 = [rec["retime"][T]["t"] for T in SIZES]
            if all(t is not None for t in ts2):
                rec["slope_retime"] = slope(ts2)
        fmt = lambda v: "   -   " if v is None else f"{v:7.3f}"  # noqa: E731
        log(fh, f"{name:26s} " + " ".join(fmt(t) for t in ts) + f"  slope={rec['slope'] if rec['slope'] is None else round(rec['slope'], 2)!s:>5}"
            f"  dflt3200={fmt(d['t'])}" + (f"  retime_slope={rec.get('slope_retime'):.2f}" if rec.get("slope_retime") is not None else "")
            + ("  FLAGS: " + "; ".join(rec["flags"]) if rec["flags"] else ""))
        report[name] = rec
        json.dump(report, open(os.path.join(OUT, "sweep_g%s.json" % ("_only" if only else "")), "w"), indent=1)
    fh.close()


if __name__ == "__main__":
    main()
