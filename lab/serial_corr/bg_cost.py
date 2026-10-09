"""Wall-clock cost of `breusch_godfrey_test` against the size of its
auxiliary regression, the numbers behind its memory and work budgets.

The auxiliary design is T x k_aux (k_aux = k + nlags, plus one when a
constant is added). Forming its normal equations costs T k_aux^2 / 2
multiply-adds and factoring and inverting them about 6 k_aux^3, so the
work count is W = T k_aux^2 / 2 + 6 k_aux^3; the crate refuses an order
with W above 2^34 or T k_aux above 2^26 entries (512 MiB), naming nlags
and the largest order that fits. This script times the test across the
(T, nlags) plane, reports seconds per unit of W, runs the largest
admissible order at two sample sizes (one where the work budget binds and
one where the cubic term dominates), and times the refusal of an order
past the budgets. One call per cell (the cells are long), on the release
wheel.

Run:  python lab/serial_corr/bg_cost.py   (needs the built tsecon wheel)
Out:  lab/serial_corr/out/bg_cost.txt
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

OUT = Path(__file__).resolve().parent / "out" / "bg_cost.txt"
CELLS, WORK = 1 << 26, 1 << 34


def work(n, k_aux):
    return n * k_aux * k_aux // 2 + 6 * k_aux ** 3


def largest_order(n, k_base):
    """The largest nlags both budgets and n > k_aux admit (the documented rule)."""
    best = 0
    for p in range(1, n - k_base):
        k = k_base + p
        if n * k > CELLS or work(n, k) > WORK:
            break
        best = p
    return best


def design(rng, n):
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    return x @ np.array([1.0, 0.5, -0.5]) + rng.standard_normal(n), x


def timed(n, nlags, rng):
    y, x = design(rng, n)
    t0 = time.perf_counter()
    tsecon.breusch_godfrey_test(y, x, nlags=nlags)
    return time.perf_counter() - t0


def main():
    rng = np.random.default_rng(0)
    lines = [f"tsecon {tsecon.__version__}, python {sys.version.split()[0]}, "
             f"{platform.machine()}, {os.cpu_count()} logical CPUs (shared machine)", "",
             "x = [1, z1, z2] (k = 3, constant column: k_aux = 3 + nlags)", "",
             f"{'T':>7s} {'nlags':>6s} {'k_aux':>6s} {'W':>10s} {'seconds':>9s} {'ns per W':>9s}"]
    cells = [(1_000, 10), (10_000, 10), (100_000, 10), (1_000_000, 10),
             (20_000, 100), (20_000, 250), (100_000, 100), (5_000, 1_000), (3_000, 1_000)]
    for n, k_base in ((100_000, 3), (3_000, 3)):
        cells.append((n, largest_order(n, k_base)))
    for n, p in cells:
        k = 3 + p
        s = timed(n, p, rng)
        lines.append(f"{n:7d} {p:6d} {k:6d} {work(n, k):10.2e} {s:9.3f} {s / work(n, k) * 1e9:9.2f}")
        print(lines[-1], flush=True)
    lines.append("")
    lines.append("largest admissible nlags (the documented rule; the refusal message quotes it):")
    for n in (1_000, 3_000, 20_000, 100_000, 1_000_000, 5_000_000):
        lines.append(f"  T = {n:9d}: nlags <= {largest_order(n, 3)}")
    lines.append(f"  default order (10) with k = 3 fits up to T = {CELLS // 13}")
    # The refusal: before anything is allocated, so it is fast and the
    # message quotes the same largest order.
    y, x = design(rng, 100_000)
    t0 = time.perf_counter()
    try:
        tsecon.breusch_godfrey_test(y, x, nlags=20_000)
        raise AssertionError("not refused")
    except ValueError as exc:
        msg = str(exc)
    dt = time.perf_counter() - t0
    quoted = int(re.search(r"nlags <= (\d+)", msg).group(1))
    assert quoted == largest_order(100_000, 3), (quoted, msg)
    lines.append("")
    lines.append(f"T = 100000, nlags = 20000 (a 16 GB design): refused in {dt:.3f} s, "
                 f"quoting nlags <= {quoted}")
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
