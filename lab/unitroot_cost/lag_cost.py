"""Wall-clock cost of the unit-root and ARCH-LM lag regressions against their
work counts, the numbers behind the shared lag budget of `tsecon-diag`
(audit round 16: the pre-existing `arch_lm`/`adf` allocation aborts).

Work counts (multiply-adds, `tsecon_diag::budget`): a Householder QR of a
`rows x k` design costs `rows k^2`; an ADF lag search over `0..=L` from `s`
fixed columns costs `rows (s^2 + ... + (s+L)^2) + (n - 1)(s+L)^2`; the
ARCH-LM normal equations cost `rows p^2 / 2 + p^3`; the Zivot-Andrews
break-date scan refits one `rows x k` design at each of `c` candidate breaks,
`c rows k^2`. An order is refused when
its design exceeds 2^26 entries or its work 2^35 multiply-adds; orders up to
the default (Schwert's `ceil(12 (n/100)^(1/4))`) are exempt from the work
budget. This script times `adf` (default searches and an explicit search at
the work cap), `arch_lm` (at a work-bound cap), the `zivot_andrews` scan (at
a work-bound fixed lag), and the refusals of the orders that used to abort
the interpreter.

Run:  python lab/unitroot_cost/lag_cost.py [adf|adf_cap|arch_cap|za|refusals|thresholds]
      (the built wheel on the path; each part writes out/lag_cost_<part>.txt)
"""
from __future__ import annotations

import math
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

import tsecon

OUT = Path(__file__).resolve().parent / "out"
CELLS, WORK = 1 << 26, 1 << 35


def schwert(n):
    return math.ceil(12 * (n / 100) ** 0.25)


def adf_work(n, s, lag, search=True):
    rows, k = n - 1 - lag, s + lag
    if not search:
        return rows * k * k
    return rows * sum(j * j for j in range(s, k + 1)) + (n - 1) * k * k


def arch_work(n, p):
    return (n - p) * p * p // 2 + p ** 3


def za_work(n, s, cands, lag):
    rows, k = n - 1 - lag, s + lag
    return cands * rows * k * k


def walk(n, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def header():
    return (f"tsecon {tsecon.__version__}, python {sys.version.split()[0]}, "
            f"{platform.machine()}, {os.cpu_count()} logical CPUs (shared machine)")


def timed(call):
    t0 = time.perf_counter()
    call()
    return time.perf_counter() - t0


def part_adf():
    lines = [header(), "", "adf default search (regression 'c', AIC over 0..=maxlag)", "",
             f"{'n':>7s} {'maxlag':>6s} {'work':>10s} {'seconds':>8s} {'ns/madd':>8s}"]
    for n in (1_000, 10_000, 30_000, 100_000):
        y = walk(n)
        m = min(schwert(n), n // 2 - 2)
        w = adf_work(n, 2, m)
        s = timed(lambda: tsecon.adf(y))
        lines.append(f"{n:7d} {m:6d} {w:10.2e} {s:8.2f} {s / w * 1e9:8.2f}")
        print(lines[-1], flush=True)
    return lines


def largest(n, cost, lo, hi, free=0):
    best = None
    a, b = lo, hi
    def ok(p):
        c, w = cost(p)
        return c <= CELLS and (p <= free or w <= WORK)
    if not ok(a):
        return None
    while a < b:
        mid = (a + b + 1) // 2
        if ok(mid):
            a = mid
        else:
            b = mid - 1
    return a


def part_adf_cap():
    n = 20_000
    y = walk(n, 7)
    cost = lambda p: ((n - 1 - p) * (3 + p), adf_work(n, 3, p))  # noqa: E731
    best = largest(n, cost, 0, n // 2 - 3, free=schwert(n))
    s = timed(lambda: tsecon.adf(y, regression="ct", maxlag=best))
    w = adf_work(n, 3, best)
    lines = [header(), "",
             f"adf(n = {n}, regression 'ct', AIC search): largest admissible maxlag = {best}",
             f"  work {w:.2e} multiply-adds, ran in {s:.1f} s ({s / w * 1e9:.2f} ns/madd)"]
    print("\n".join(lines), flush=True)
    return lines


def part_arch_cap():
    n = 30_000
    e = np.random.default_rng(3).standard_normal(n)
    cost = lambda p: ((n - p) * p, arch_work(n, p))  # noqa: E731
    best = largest(n, cost, 1, (n - 2) // 2)
    s = timed(lambda: tsecon.arch_lm(e, nlags=best))
    w = arch_work(n, best)
    lines = [header(), "",
             f"arch_lm(n = {n}): largest admissible nlags = {best} (the work budget binds)",
             f"  work {w:.2e} multiply-adds, ran in {s:.1f} s ({s / w * 1e9:.2f} ns/madd)"]
    print("\n".join(lines), flush=True)
    return lines


def part_za():
    """The Zivot-Andrews break-date scan: the largest admissible fixed lag
    where the work budget binds (n = 3000, regression "c": 4 base columns,
    2100 candidate breaks), run once; and a fixed lag inside the trimmed
    window whose design no machine holds (n = 1e5), refused."""
    n = 3000
    trimcnt = int(n * 0.15)
    cands = n - 2 * trimcnt
    y = walk(n, 11)
    cost = lambda p: ((n - 1 - p) * (4 + p), za_work(n, 4, cands, p))  # noqa: E731
    best = largest(n, cost, 0, trimcnt - 1, free=schwert(n))
    s = timed(lambda: tsecon.zivot_andrews(y, autolag=None, lags=best))
    w = za_work(n, 4, cands, best)
    lines = [header(), "",
             f"zivot_andrews(n = {n}, regression 'c', fixed lags): largest admissible "
             f"lags = {best} (the work budget binds; int(n*trim) - 1 = {trimcnt - 1})",
             f"  work {w:.2e} multiply-adds, ran in {s:.1f} s ({s / w * 1e9:.2f} ns/madd)"]
    big = walk(100_000, 13)
    t0 = time.perf_counter()
    try:
        tsecon.zivot_andrews(big, autolag=None, lags=14_000)
        raise AssertionError("zivot_andrews(lags=14000) was not refused")
    except ValueError as exc:
        msg = str(exc)
    dt = time.perf_counter() - t0
    remedy = re.search(r"pass (\w+ <= \d+)", msg).group(1)
    lines.append(f"zivot_andrews(n = 1e5, lags = 14000, autolag=None): refused in "
                 f"{dt * 1e3:.1f} ms, quoting {remedy}")
    print("\n".join(lines), flush=True)
    return lines


def part_refusals():
    lines = [header(), ""]
    big = walk(1_000_000, 5)
    e = np.random.default_rng(3).standard_normal(1_000_000)
    calls = [
        ("arch_lm(n = 1e6, nlags = 400000)", lambda: tsecon.arch_lm(e, nlags=400_000)),
        ("adf(n = 1e6, maxlag = 400000, autolag=None)",
         lambda: tsecon.adf(big, maxlag=400_000, autolag=None)),
        ("adf(n = 1e6, maxlag = 100000) (AIC search)", lambda: tsecon.adf(big, maxlag=100_000)),
        ("dfgls(n = 1e6, lags = 300000)", lambda: tsecon.dfgls(big, lags=300_000)),
        ("zivot_andrews(n = 1e6, max_lags = 400000)",
         lambda: tsecon.zivot_andrews(big, max_lags=400_000)),
    ]
    for label, call in calls:
        t0 = time.perf_counter()
        try:
            call()
            raise AssertionError(f"{label} was not refused")
        except ValueError as exc:
            msg = str(exc)
        dt = time.perf_counter() - t0
        remedy = re.search(r"pass (\w+ <= \d+)", msg).group(1)
        lines.append(f"{label}: refused in {dt * 1e3:.1f} ms, quoting {remedy}")
        print(lines[-1], flush=True)
    return lines


def part_thresholds():
    """The longest series whose DEFAULT order still fits the memory budget
    (default orders are exempt from the work budget), from the documented
    size formulas — no tsecon call."""
    def bisect(ok):
        lo, hi = 100, 10**9
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    lines = [header(), "", "longest n whose default order fits 2^26 design entries:"]
    for ntrend, reg in ((0, "n"), (1, "c"), (2, "ct")):
        def ok(n, nt=ntrend):
            m = min(schwert(n), n // 2 - nt - 1)
            return (n - 1 - m) * (nt + 1 + m) <= CELLS
        lines.append(f"  adf(regression={reg!r}): n <= {bisect(ok)}")
    lines.append(f"  dfgls / ng_perron: n <= {bisect(lambda n: (n - 1 - schwert(n)) * (1 + schwert(n)) <= CELLS)}")
    lines.append(f"  arch_lm(nlags=4): n <= {bisect(lambda n: (n - 4) * 4 <= CELLS)}")
    print("\n".join(lines), flush=True)
    return lines


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "adf"
    fn = {"adf": part_adf, "adf_cap": part_adf_cap, "arch_cap": part_arch_cap,
          "za": part_za, "refusals": part_refusals, "thresholds": part_thresholds}[part]
    lines = fn()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"lag_cost_{part}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
