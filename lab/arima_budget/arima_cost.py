"""Wall-clock cost of `arima_fit` against the size of its state-space form,
the numbers behind the ARIMA estimation budgets (audit round 16).

The exact likelihood runs a Kalman filter on the ARMA state of dimension
m = max(p + s*P, q + s*Q + 1). Every step forms T P T' twice (P_* and the
diffuse P_inf; about 4 m^3 multiply-adds) and stores three m x m
covariances, and the stationary initialization (Lyapunov doubling plus the
spectral-radius check) costs at most about 200 m^3. So one evaluation of the
likelihood on n_eff usable observations costs E = (4 n_eff + 200) m^3 and
holds 3 (n_eff + 1) m^2 doubles. Whatever the optimizer does, a fit then
evaluates the likelihood at least 2 k (k + 2) + 1 times (one central-
difference gradient, 2k + 1, and the four-point observed-information
Hessian behind `bse`, 2 k (k + 1)), with k the free parameters; CSS
evaluates its recursion (n_eff (p* + q* + 1) per evaluation, p* = p + s*P,
q* = q + s*Q) that many times and runs the exact filter once more for the
residuals. The crate refuses a specification whose count W of that
guaranteed work exceeds 2^37 multiply-adds, or whose filter storage exceeds
2^28 doubles (2 GiB), naming p, q and seasonal and quoting the largest state
dimension that fits.

This script times arima_fit across the (n, m, k) plane, one capped child
process per cell (ulimit -v 4 GB, timeout), and reports seconds per unit of
W. With --refusals it also checks that specifications past the budgets are
refused before anything is allocated (needs a wheel with the budget).

Run:  python lab/arima_budget/arima_cost.py [--refusals]
Out:  lab/arima_budget/out/arima_cost.txt (or arima_refusals.txt)
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK, VALUES = 1 << 37, 1 << 28

CHILD = r'''
import json, sys, time
import numpy as np, tsecon
n, p, q, P, Q, s, method = json.loads(sys.argv[1])
r = np.random.default_rng(0)
e = r.standard_normal(n + 100)
y = np.zeros(n + 100)
for t in range(1, n + 100):
    y[t] = 0.5 * y[t - 1] + e[t]
y = y[100:]
if s:
    y = y + np.sin(2 * np.pi * np.arange(n) / s)
kw = dict(p=p, q=q, method=method)
if s:
    kw["seasonal"] = (P, 0, Q, s)
t0 = time.perf_counter()
try:
    tsecon.arima_fit(y, **kw)
    out = "ok"
except Exception as ex:
    out = type(ex).__name__ + ": " + str(ex)
print(json.dumps({"t": time.perf_counter() - t0, "out": out}))
'''


def state_dim(p, q, P, Q, s):
    return max(p + s * P, q + s * Q + 1)


def k_params(p, q, P, Q):
    return 1 + p + q + P + Q + 1  # constant (arima_fit default) + sigma2


def work(n, p, q, P, Q, s, method):
    """The documented guaranteed-work count W (n_eff = n: d = D = 0)."""
    m, k = state_dim(p, q, P, Q, s), k_params(p, q, P, Q)
    evals = 2 * k * (k + 2) + 1
    exact = (4 * n + 200) * m ** 3
    if method == "mle":
        return evals * exact
    return evals * n * (p + s * P + q + s * Q + 1) + exact


def values(n, p, q, P, Q, s):
    return 3 * (n + 1) * state_dim(p, q, P, Q, s) ** 2


def run(cell, timeout):
    pr = subprocess.run(
        ["bash", "-c", f'ulimit -v 4000000; timeout {timeout} "{sys.executable}" -c "$0" "$1"',
         CHILD, json.dumps(cell)],
        capture_output=True, text=True)
    line = [l for l in pr.stdout.splitlines() if l.startswith("{")]
    if line:
        return json.loads(line[-1])
    return {"t": None, "out": f"rc={pr.returncode} {pr.stderr.strip()[-160:]}"}


def header(title):
    import tsecon
    return [f"tsecon {tsecon.__version__}, python {sys.version.split()[0]}, "
            f"{platform.machine()}, {os.cpu_count()} logical CPUs (shared machine)", "", title, "",
            f"{'method':>6s} {'n':>5s} {'p':>3s} {'q':>3s} {'P':>2s} {'Q':>2s} {'s':>4s} "
            f"{'m':>4s} {'k':>4s} {'W':>9s} {'values':>9s} {'seconds':>8s} {'ns per W':>9s}  outcome"]


def row(cell, res):
    n, p, q, P, Q, s, method = cell
    w, v = work(*cell), values(n, p, q, P, Q, s)
    t = res["t"]
    ts = f"{t:8.2f}" if t is not None else f"{'-':>8s}"
    per = f"{t / w * 1e9:9.3f}" if t is not None and res["out"] == "ok" else f"{'-':>9s}"
    return (f"{method:>6s} {n:5d} {p:3d} {q:3d} {P:2d} {Q:2d} {s:4d} "
            f"{state_dim(p, q, P, Q, s):4d} {k_params(p, q, P, Q):4d} {w:9.2e} {v:9.2e} "
            f"{ts} {per}  {res['out'][:150]}")


def calibrate():
    cells = [
        (200, 5, 0, 0, 0, 0, "mle"),
        (200, 10, 0, 0, 0, 0, "mle"),
        (200, 20, 0, 0, 0, 0, "mle"),
        (200, 30, 0, 0, 0, 0, "mle"),
        (520, 1, 0, 1, 0, 52, "mle"),
        (520, 1, 1, 1, 1, 52, "mle"),
        (400, 1, 0, 1, 0, 100, "mle"),
        (200, 0, 50, 0, 0, 0, "css"),
    ]
    lines = header("valid fits, d = D = 0, constant included (one call per cell)")
    for c in cells:
        lines.append(row(c, run(c, 300)))
        print(lines[-1], flush=True)
    return lines


def refusals():
    cells = [
        (200, 1, 0, 50, 0, 4, "mle"),      # the H16-14 refuter's 40-second timeout
        (200, 1, 0, 1, 0, 199, "mle"),
        (2000, 1000, 0, 0, 0, 0, "mle"),   # 48 GB of filter storage per evaluation
        (2000, 0, 900, 0, 0, 0, "css"),
    ]
    lines = header("specifications past the budgets (each must be refused, fast)")
    for c in cells:
        res = run(c, 60)
        lines.append(row(c, res))
        print(lines[-1], flush=True)
        assert res["out"].startswith("ValueError") and "seasonal" in res["out"], res
        assert res["t"] is not None and res["t"] < 1.0, res
    return lines


def main():
    if "--refusals" in sys.argv:
        lines, name = refusals(), "arima_refusals.txt"
    else:
        lines, name = calibrate(), "arima_cost.txt"
    out = HERE / "out" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
