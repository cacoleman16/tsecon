"""Per-replication outcomes of test_auto_arima.py::test_recovery_small_mc_nonseasonal.

The test asserts `within >= 0.5 * reps` per DGP with reps = 12 and does not
print per-replication results, so the effect of a smaller `reps` cannot be
read from the suite log. This replays the identical loop (same seeds, same
simulator, same call) and prints every (truth, selected order) pair, then the
within-one counts at reps = 6 and reps = 12 — the evidence behind the
proposal in finding 4. Nothing here changes the test.

Run:  .venv-wt/bin/python lab/audit/repo/tests/mc_trim_probe.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "bindings", "python", "tests"))

import tsecon  # noqa: E402
from test_auto_arima import simulate_arma  # noqa: E402

cases = [
    ((1, 0, 0), dict(ar=[0.6])),
    ((0, 0, 1), dict(ma=[0.6])),
    ((1, 0, 1), dict(ar=[0.5], ma=[0.4])),
]
reps = 12
t0 = time.perf_counter()
for truth, kw in cases:
    hits = []
    for rep in range(reps):
        rng = np.random.default_rng(1000 + 17 * rep)
        y = simulate_arma(rng, 300, **kw)
        t = time.perf_counter()
        r = tsecon.auto_arima(y)
        dt = time.perf_counter() - t
        p, d, q = r["order"]
        within = d == truth[1] and abs(p - truth[0]) <= 1 and abs(q - truth[2]) <= 1
        hits.append(within)
        print(f"truth {truth} rep {rep:2d}: selected {(p, d, q)}  within-one={within}  n_models={r['n_models']}  {dt:.1f}s", flush=True)
    print(f"truth {truth}: within-one {sum(hits[:6])}/6 at reps=6 (bar 3), {sum(hits)}/12 at reps=12 (bar 6)\n", flush=True)
print(f"total {time.perf_counter() - t0:.0f}s")
