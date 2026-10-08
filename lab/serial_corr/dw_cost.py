"""Wall-clock cost of `durbin_watson_test` (exact p-value: one dense T x T
symmetric eigendecomposition, cubic in T) against `breusch_godfrey_test`
(linear in T), the timings quoted in the specification-tests model card.
Median of 3 calls per size, on the release wheel.

Run:  python lab/serial_corr/dw_cost.py   (needs the built tsecon wheel)
Out:  lab/serial_corr/out/dw_cost.txt
"""
from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

import tsecon

OUT = Path(__file__).resolve().parent / "out" / "dw_cost.txt"


def timed(fn, *args, **kw):
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        fn(*args, **kw)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    rng = np.random.default_rng(0)
    lines = [f"tsecon {tsecon.__version__}, python {sys.version.split()[0]}, "
             f"{platform.machine()}, {os.cpu_count()} logical CPUs (shared machine)", ""]
    lines.append(f"{'T':>6s} {'durbin_watson_test':>20s} {'breusch_godfrey_test':>22s}")
    for n in (100, 250, 500, 1000, 2000, 3000, 4000):
        x = np.column_stack([np.ones(n), rng.standard_normal(n), np.arange(n) / n])
        y = x @ np.array([1.0, 0.5, -0.5]) + rng.standard_normal(n)
        t_dw = timed(tsecon.durbin_watson_test, y, x)
        t_bg = timed(tsecon.breusch_godfrey_test, y, x)
        lines.append(f"{n:6d} {t_dw:19.3f}s {t_bg:21.4f}s")
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
