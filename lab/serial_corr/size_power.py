"""Seeded Monte Carlo: size and power of `breusch_godfrey_test` and
`durbin_watson_test` (the numbers quoted in the specification-tests model
card, docs/reference/model-cards/specification-tests.md).

DGP: y_t = 1 + 0.5 x1_t - 0.5 x2_t + u_t with x1 an AR(1)(0.5) regressor and
x2 iid N(0, 1), both drawn ONCE per sample size and held fixed across
replications (the classical fixed-regressor setting the exact Durbin-Watson
distribution assumes). Errors:

  * size  — u iid N(0, 1), and u iid Student-t(5) (scaled to unit variance)
            to show the exact DW p-value's sensitivity to non-normality;
  * power — u AR(1) with rho = 0.2 and 0.4 (stationary start).

Two further columns add a lagged dependent variable: y_t = 0.5 y_{t-1} +
1 + 0.5 x1_t + u_t, regressed on [1, y_{t-1}, x1_t], with u iid N(0, 1)
(size) and AR(1) rho = 0.4 (power) — the case where Durbin-Watson is
invalid and Breusch-Godfrey is not.

Rejection rates at the 5% level, R = 2000 replications per cell (Monte-Carlo
standard error 0.49 percentage points at a true 5%).

Run:  python lab/serial_corr/size_power.py   (needs the built tsecon wheel)
Out:  lab/serial_corr/out/size_power.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import tsecon

R = 2000
SIZES = (50, 100, 200)
OUT = Path(__file__).resolve().parent / "out" / "size_power.txt"


def ar1(rng, n, rho):
    e = rng.standard_normal(n)
    u = np.empty(n)
    u[0] = e[0] / np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    return u


def errors(rng, n, kind):
    if kind == "iid normal":
        return rng.standard_normal(n)
    if kind == "iid t(5)":
        return rng.standard_t(5, n) / np.sqrt(5.0 / 3.0)
    rho = float(kind.split("=")[1])
    return ar1(rng, n, rho)


def tests_on(y, x):
    """p-values of every test variant on one sample."""
    n = len(y)
    out = {}
    for p in (1, 4):
        r = tsecon.breusch_godfrey_test(y, x, nlags=p)
        out[f"BG LM p={p}"] = r["pvalue"]
        out[f"BG F  p={p}"] = r["f_pvalue"]
    r = tsecon.breusch_godfrey_test(y, x)
    out[f"BG LM default (p={min(10, n // 5)})"] = r["pvalue"]
    out[f"BG F  default (p={min(10, n // 5)})"] = r["f_pvalue"]
    out["DW exact (rho>0)"] = tsecon.durbin_watson_test(y, x)["pvalue"]
    return out


def run_cell(n, kind, seed):
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), ar1(rng, n, 0.5), rng.standard_normal(n)])
    beta = np.array([1.0, 0.5, -0.5])
    rej = {}
    for _ in range(R):
        y = x @ beta + errors(rng, n, kind)
        for name, p in tests_on(y, x).items():
            rej[name] = rej.get(name, 0) + (p < 0.05)
    return {k: v / R for k, v in rej.items()}


def run_ldv(n, kind, seed):
    rng = np.random.default_rng(seed)
    x1 = ar1(rng, n + 1, 0.5)
    rej = {}
    for _ in range(R):
        u = errors(rng, n + 1, kind)
        y = np.empty(n + 1)
        y[0] = 2.0 + u[0]
        for t in range(1, n + 1):
            y[t] = 1.0 + 0.5 * y[t - 1] + 0.5 * x1[t] + u[t]
        x = np.column_stack([np.ones(n), y[:-1], x1[1:]])
        for name, p in tests_on(y[1:], x).items():
            rej[name] = rej.get(name, 0) + (p < 0.05)
    return {k: v / R for k, v in rej.items()}


def main():
    t0 = time.perf_counter()
    lines = [
        f"Rejection rates at the 5% level, R = {R} replications per cell "
        f"(MC se 0.49 pp at 5%). tsecon {tsecon.__version__}, numpy {np.__version__}, "
        f"python {sys.version.split()[0]}.",
        "",
    ]
    kinds = ["iid normal", "iid t(5)", "AR(1) rho=0.2", "AR(1) rho=0.4"]
    for i, n in enumerate(SIZES):
        cells = {k: run_cell(n, k, 20261000 + 10 * i + j) for j, k in enumerate(kinds)}
        cells["LDV, iid normal"] = run_ldv(n, "iid normal", 20262000 + i)
        cells["LDV, rho=0.4"] = run_ldv(n, "AR(1) rho=0.4", 20263000 + i)
        names = list(cells["iid normal"])
        lines.append(f"T = {n}")
        header = f"  {'test':28s}" + "".join(f"{k:>18s}" for k in cells)
        lines.append(header)
        for name in names:
            lines.append(f"  {name:28s}" + "".join(f"{cells[k][name]:18.3f}" for k in cells))
        lines.append("")
    lines.append(f"elapsed {time.perf_counter() - t0:.0f} s")
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
