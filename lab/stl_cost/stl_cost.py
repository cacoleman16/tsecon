"""Wall-clock cost of `stl`/`mstl` against the LOESS work count, the numbers
behind the STL iteration budget (audit round 16, A8).

One STL inner pass costs `U = (n/sj + 1 + 3P) min(ns, ceil(n/P)) +
(n/lj + 1) min(nl, n) + (n/tj + 1) min(nt, n) + 12 n` LOESS evaluation units
(`tsecon_filters::stl_pass_work`); a call runs `inner_iter * (outer_iter + 1)`
passes (times `iterate` rounds and the periods for MSTL). A request of more
than 32 passes per fit (the robust default's count) whose units exceed 2^32 is
refused before any smoothing, naming the count and its largest admissible
value. This script times passes across (n, period, windows, jumps), reports
nanoseconds per unit, runs the largest admissible `inner_iter` on the
96-observation monthly series of the audit, and times the refusals of
`inner_iter`, `outer_iter` and `iterate` at 2**31.

Run:  python lab/stl_cost/stl_cost.py [rates|cap|refusals]   (the built wheel
      on the path; each part writes out/stl_cost_<part>.txt)
"""
from __future__ import annotations

import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

import tsecon

OUT = Path(__file__).resolve().parent / "out"
BUDGET = 1 << 32


def units(n, cfg):
    p = cfg["period"]
    k = -(-n // p)
    pts = lambda j: n // j + 1  # noqa: E731
    return ((pts(cfg["seasonal_jump"]) + 3 * p) * min(cfg["seasonal"], k)
            + pts(cfg["low_pass_jump"]) * min(cfg["low_pass"], n)
            + pts(cfg["trend_jump"]) * min(cfg["trend"], n) + 12 * n)


def series(n, period, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return 10 + np.sin(2 * np.pi * t / period) + 0.3 * rng.standard_normal(n) + 0.002 * t


def header():
    return (f"tsecon {tsecon.__version__}, python {sys.version.split()[0]}, "
            f"{platform.machine()}, {os.cpu_count()} logical CPUs (shared machine)")


def rates():
    lines = [header(), "", "inner passes timed at inner_iter = 4, outer_iter = 0", "",
             f"{'n':>7s} {'P':>5s} {'(ns, nt, nl)':>18s} {'jumps':>9s} {'units':>10s} "
             f"{'ms/pass':>9s} {'ns/unit':>8s}"]
    cells = [(96, 12, {}), (1000, 12, {}), (10_000, 12, {}), (100_000, 24, {}),
             (10_000, 168, {}), (5_000, 500, {}), (20_000, 2000, {}),
             (20_000, 2000, {"seasonal_jump": 5, "trend_jump": 50, "low_pass_jump": 50}),
             (2_000, 12, {"seasonal": 1001})]
    for n, p, kw in cells:
        y = series(n, p)
        reps = max(1, 200_000 // n)
        t0 = time.perf_counter()
        for _ in range(reps):
            r = tsecon.stl(y, period=p, inner_iter=4, outer_iter=0, **kw)
        dt = (time.perf_counter() - t0) / reps / 4
        c = r["config"]
        u = units(n, c)
        jumps = f"{c['seasonal_jump']}/{c['trend_jump']}/{c['low_pass_jump']}"
        wins = f"({c['seasonal']}, {c['trend']}, {c['low_pass']})"
        lines.append(f"{n:7d} {p:5d} {wins:>18s} {jumps:>9s} {u:10.3e} {dt * 1e3:9.3f} "
                     f"{dt / u * 1e9:8.2f}")
        print(lines[-1], flush=True)
    return lines


def cap():
    y = series(96, 12, seed=4)
    cfg = tsecon.stl(y, period=12)["config"]
    u = units(96, cfg)
    best = BUDGET // u
    lines = [header(), "", f"n = 96, period = 12, windows (7, 23, 13): {u} units per pass",
             f"largest admissible inner_iter at outer_iter = 0: 2^32 // {u} = {best}"]
    t0 = time.perf_counter()
    tsecon.stl(y, period=12, inner_iter=best, outer_iter=0)
    dt = time.perf_counter() - t0
    lines.append(f"stl(inner_iter={best}, outer_iter=0) ran in {dt:.1f} s "
                 f"({dt / (best * u) * 1e9:.2f} ns per unit)")
    print("\n".join(lines), flush=True)
    return lines


def refusals():
    y = series(96, 12, seed=4)
    lines = [header(), ""]
    t = np.arange(240)
    y2 = 5 + np.sin(2 * np.pi * t / 7) + 0.5 * np.cos(2 * np.pi * t / 30) + 0.01 * t
    calls = [
        ("stl(inner_iter=2**31, outer_iter=0)",
         lambda: tsecon.stl(y, period=12, inner_iter=2**31, outer_iter=0)),
        ("stl(inner_iter=2, outer_iter=2**31)",
         lambda: tsecon.stl(y, period=12, inner_iter=2, outer_iter=2**31)),
        ("mstl(periods=[12], inner_iter=2**31, outer_iter=0)",
         lambda: tsecon.mstl(y, periods=[12], windows=[7], inner_iter=2**31, outer_iter=0)),
        ("mstl(periods=[7, 30], iterate=2**31)",
         lambda: tsecon.mstl(y2, periods=[7, 30], iterate=2**31)),
    ]
    for label, call in calls:
        t0 = time.perf_counter()
        try:
            call()
            raise AssertionError(f"{label} was not refused")
        except ValueError as exc:
            msg = str(exc)
        dt = time.perf_counter() - t0
        remedy = re.search(r"pass (\w+ <= \d+)$", msg).group(1)
        lines.append(f"{label}: refused in {dt * 1e3:.2f} ms, quoting {remedy}")
        print(lines[-1], flush=True)
    return lines


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "rates"
    lines = {"rates": rates, "cap": cap, "refusals": refusals}[part]()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stl_cost_{part}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
