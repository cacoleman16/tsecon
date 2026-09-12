"""Sweep G — complexity cliffs and resource caps for the 0.10.0 surface.

Part 1 (timing): each callable at three sizes in fresh subprocesses, log-log
slope, every flagged cell re-timed once to refute noise; a defaults-only pass
(required arguments only) at the largest size.

Part 2 (caps): every count argument that sizes an allocation driven to 2^47
(just under the count seal), 2^31, 10^6 and to the documented cap +/- 1, each
in a child under a 4 GB virtual-memory cap with a deadline, recording
refusal / ok / abort / hang and the resident-set delta — so a count the caps
let through is measured, not guessed. The wave fixed five allocator aborts;
this is the re-sweep.

Run:  .venv/bin/python lab/audit/round14/sweep_g_timing.py [--caps-only]
Out:  lab/audit/round14/out/sweep_g.txt, sweep_g.json
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, OUT, log  # noqa: E402
from parent import run_cells  # noqa: E402
from registry import MASKED, NEW14  # noqa: E402

SIZES = (200, 800, 3200)
CAP_S = 900

CHILD = r"""
import sys, time, json, inspect
sys.path.insert(0, %r)
import numpy as np, tsecon
from registry import build
name, T, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
args, kwargs = build(name, T=T, seed=0)
base = name.split("@")[0]
if mode == "defaults":
    fn = getattr(tsecon._core, base)
    n_req = sum(1 for p in inspect.signature(fn).parameters.values() if p.default is p.empty)
    keep = {"panel_distributed_lag": ("lags",), "panel_distributed_lag@mask": ("lags", "mask"),
            "panel_fe@mask": ("mask",), "panel_lp@mask": ("mask",), "lp_did@mask": ("mask",)}
    args = args[:n_req]
    kwargs = {k: kwargs[k] for k in keep.get(name, ()) if k in kwargs}
fn = getattr(tsecon, base)
t0 = time.perf_counter(); fn(*args, **kwargs)
print(json.dumps({"t": time.perf_counter() - t0}))
"""


def time_one(name, T, mode="canonical"):
    cmd = [sys.executable, "-c", CHILD % HERE, name, str(T), mode]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=CAP_S, cwd=HERE)
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT>{CAP_S}s"
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        return None, "ERROR: " + (tail[-1] if tail else "?")[:200]
    return json.loads(p.stdout.strip().splitlines()[-1])["t"], "ok"


def slope(ts):
    xs = [math.log(s) for s in SIZES]
    ys = [math.log(max(t, 1e-6)) for t in ts]
    mx, my = sum(xs) / 3, sum(ys) / 3
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


# Part 2: (function, kw) -> list of (label, value)
HUGE = [("2^47", 2**47), ("2^31", 2**31), ("1e6", 10**6)]
CAPS = {
    "unobserved_components": {
        "seasonal": HUGE + [("cap+1", 201), ("cap", 200)],
        "freq_seasonal": [(l, [v]) for l, v in HUGE] + [("cap+1", [201.0]), ("cap", [200.0])],
        "freq_seasonal_harmonics": [(l, [v]) for l, v in HUGE],
        "forecast_steps": HUGE + [("cap+1", 100_001), ("cap", 100_000), ("1e4", 10_000)],
        "n_starts": HUGE + [("cap+1", 65), ("cap", 64)],
    },
    "tvp_regression": {"n_starts": HUGE + [("cap+1", 65), ("cap", 64)]},
    "ets_fit": {
        "seasonal_periods": HUGE + [("2e5", 200_000)],
        "horizon": HUGE + [("cap+1", 1_000_001), ("cap", 1_000_000), ("2e5", 200_000)],
        "n_sim": HUGE + [("2^28", 2**28), ("2^26", 2**26)],
        "max_iter": HUGE,
    },
    "auto_ets": {
        "seasonal_periods": HUGE,
        "horizon": HUGE + [("cap+1", 1_000_001), ("2e5", 200_000)],
        "n_sim": HUGE + [("2^26", 2**26)],
    },
    "var_conditional_forecast": {"lags": HUGE, "steps": HUGE + [("2e5", 200_000), ("1e4", 10_000)]},
    "var_diagnostics": {"lags": HUGE, "nlags": HUGE + [("2e5", 200_000)]},
    "var_select_order": {"max_lags": HUGE + [("2e5", 200_000), ("1e4", 10_000)]},
    "spa_test": {"reps": HUGE + [("2^24", 2**24), ("1e5", 100_000)], "block_size": HUGE},
    "model_confidence_set": {"reps": HUGE + [("2^24", 2**24)], "block_size": HUGE},
    "stepm_test": {"reps": HUGE + [("2^24", 2**24)], "block_size": HUGE},
    "fmols": {"bandwidth": [(l, float(v)) for l, v in HUGE]},
    "ccr": {"bandwidth": [(l, float(v)) for l, v in HUGE]},
    "dols": {
        "lags": HUGE, "leads": HUGE,
        "max_lag": HUGE + [("1e4", 10_000)], "max_lead": HUGE + [("1e4", 10_000)],
    },
    "panel_lp@mask": {"horizon": HUGE + [("1e4", 10_000)]},
    "panel_distributed_lag@mask": {"lags": HUGE, "powers": HUGE},
    "lp_did@mask": {"pre_window": HUGE, "post_window": HUGE},
}
# the count arguments that must be passed with a companion (a searched count)
POS1 = {"var_conditional_forecast": None}


def main():
    caps_only = "--caps-only" in sys.argv
    timing_only = "--timing-only" in sys.argv
    fh = open(os.path.join(OUT, "sweep_g_caps.txt" if caps_only else "sweep_g.txt"), "w")
    report = {"timing": {}, "caps": {}}
    scope = NEW14 + MASKED
    if not caps_only:
        log(fh, f"{'function':30s} {'T=200':>8s} {'T=800':>8s} {'T=3200':>8s}  slope  dflt@3200  flags")
    for name in ([] if caps_only else scope):
        rec = {"canonical": {}, "flags": []}
        ts = []
        for T in SIZES:
            t, st = time_one(name, T)
            rec["canonical"][T] = {"t": t, "status": st}
            ts.append(t)
        if all(t is not None for t in ts):
            rec["slope"] = slope(ts)
            if rec["slope"] > 1.35:
                rec["flags"].append(f"superlinear slope {rec['slope']:.2f}")
            if ts[2] > 5.0:
                rec["flags"].append(f"{ts[2]:.1f}s at T=3200")
            if ts[1] < ts[0] / 2 or ts[2] < ts[1]:
                rec["flags"].append("non-monotone")
        else:
            rec["slope"] = None
            rec["flags"].append("incomplete: " + "; ".join(
                f"T={T}:{rec['canonical'][T]['status']}" for T in SIZES if rec["canonical"][T]["t"] is None))
        d, st = time_one(name, 3200, "defaults")
        rec["defaults_3200"] = {"t": d, "status": st}
        if d is not None and d > 5.0:
            rec["flags"].append(f"{d:.1f}s at T=3200 with DEFAULT arguments")
        elif d is None:
            rec["flags"].append(f"defaults@3200: {st}")
        if any("superlinear" in f or "non-monotone" in f for f in rec["flags"]):
            ts2 = [time_one(name, T)[0] for T in SIZES]
            rec["retime"] = ts2
            if all(t is not None for t in ts2):
                rec["slope_retime"] = slope(ts2)
        fmt = lambda v: "   -    " if v is None else f"{v:8.3f}"  # noqa: E731
        log(fh, f"{name:30s} " + " ".join(fmt(t) for t in ts)
                + f"  {rec['slope'] if rec['slope'] is None else round(rec['slope'], 2)!s:>5}  {fmt(d)}  "
                + ("; ".join(rec["flags"])
                   + (f" (retime slope {rec['slope_retime']:.2f})" if rec.get("slope_retime") is not None else "")))
        report["timing"][name] = rec
    log(fh, "\n== caps: counts just under the seal, in the allocation band, and at the "
            "documented cap (child: 4 GB RLIMIT_AS, 60 s)")
    n_cells = n_bad = 0
    for name, spec in ({} if timing_only else CAPS).items():
        muts = []
        for kw, vals in spec.items():
            for label, v in vals:
                muts.append({"id": f"{kw}={label}", "slot": ["kw", kw], "variant": ["value", v], "deadline": 60.0})
        t0 = time.time()
        recs = run_cells(name, muts, rlimit_gb=4.0, deadline=60.0)
        report["caps"][name] = recs
        for r in recs:
            n_cells += 1
            msg = (r.get("msg") or r.get("detail") or "")[:150]
            if r["outcome"] not in ("refusal", "ok"):
                n_bad += 1
            log(fh, f"  {name:26s} {r['id']:26s} {r['outcome']:12s} t={r.get('seconds', '-')!s:>7} "
                    f"rss+={r.get('rss_mb', '-')!s:>8}MB  {msg}")
        log(fh, f"  ({name}: {len(recs)} cells in {time.time() - t0:.0f} s)")
    log(fh, f"\ncap cells: {n_cells}; not refusal/ok: {n_bad}")
    json.dump(report, open(os.path.join(OUT, "sweep_g_caps.json" if caps_only else "sweep_g.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
