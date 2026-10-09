"""Measured peak memory per horizon (or per draw) of the VAR-family callables,
against the count their size budget charges.

Each case runs in a fresh child process: it builds the data, makes a small
warm-up call (so allocator arenas, the thread pool and imports are paid for),
records the peak RSS, makes the measured call, and reports the growth of the
peak RSS divided by the size argument. Linux only (ru_maxrss in KiB).

Usage: python footprint.py [case ...]
"""

from __future__ import annotations

import subprocess
import sys

PRE = r"""
import numpy as np, resource, time, tsecon
def var_data(t, k, seed):
    rng = np.random.default_rng(seed)
    a = 0.5 * np.eye(k) + 0.1 * np.ones((k, k)) / k
    y = np.zeros((t, k))
    for i in range(1, t):
        y[i] = a @ y[i - 1] + rng.standard_normal(k)
    return y
def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
"""

# name -> (setup, warm-up call, measured call, size of the measured argument)
CASES = {
    "var_irf_k2": ("X = var_data(200, 2, 1)", "tsecon.var_irf(X, lags=2, horizon=50)",
                   "tsecon.var_irf(X, lags=2, horizon=SIZE)", 400_000),
    "var_irf_k3": ("X = var_data(200, 3, 1)", "tsecon.var_irf(X, lags=2, horizon=50)",
                   "tsecon.var_irf(X, lags=2, horizon=SIZE)", 300_000),
    "var_irf_k8": ("X = var_data(400, 8, 1)", "tsecon.var_irf(X, lags=1, horizon=50)",
                   "tsecon.var_irf(X, lags=1, horizon=SIZE)", 40_000),
    "var_irf_k9": ("X = var_data(400, 9, 1)", "tsecon.var_irf(X, lags=1, horizon=50)",
                   "tsecon.var_irf(X, lags=1, horizon=SIZE)", 30_000),
    "var_irf_k20": ("X = var_data(600, 20, 1)", "tsecon.var_irf(X, lags=1, horizon=10)",
                    "tsecon.var_irf(X, lags=1, horizon=SIZE)", 5_000),
    "var_irf_cum_k2": ("X = var_data(200, 2, 1)", "tsecon.var_irf(X, lags=2, horizon=50, cumulative=True)",
                       "tsecon.var_irf(X, lags=2, horizon=SIZE, cumulative=True)", 400_000),
    "var_fevd_k2": ("X = var_data(200, 2, 1)", "tsecon.var_fevd(X, lags=2, horizon=50)",
                    "tsecon.var_fevd(X, lags=2, horizon=SIZE)", 400_000),
    "var_fevd_k8": ("X = var_data(400, 8, 1)", "tsecon.var_fevd(X, lags=1, horizon=50)",
                    "tsecon.var_fevd(X, lags=1, horizon=SIZE)", 40_000),
    "long_run_svar_k2": ("X = var_data(200, 2, 1)", "tsecon.long_run_svar(X, lags=2, horizon=50)",
                         "tsecon.long_run_svar(X, lags=2, horizon=SIZE)", 300_000),
    "long_run_svar_k8": ("X = var_data(400, 8, 1)", "tsecon.long_run_svar(X, lags=1, horizon=50)",
                         "tsecon.long_run_svar(X, lags=1, horizon=SIZE)", 30_000),
    "connectedness_k2": ("X = var_data(200, 2, 1)", "tsecon.connectedness(X, lags=2, horizon=50)",
                         "tsecon.connectedness(X, lags=2, horizon=SIZE)", 400_000),
    "mean_group_var_n4": ("E = [var_data(120, 2, 10 + i) for i in range(4)]",
                          "tsecon.mean_group_var(E, lags=1, horizon=50)",
                          "tsecon.mean_group_var(E, lags=1, horizon=SIZE)", 200_000),
    "fvar_scenario": ("rng = np.random.default_rng(7); C = rng.standard_normal((150, 12)).cumsum(axis=1); "
                      "Y = rng.standard_normal(150); D = np.linspace(0.1, 0.3, 12)",
                      "tsecon.fvar_scenario(Y, C, D, n_factors=2, lags=1, horizon=50)",
                      "tsecon.fvar_scenario(Y, C, D, n_factors=2, lags=1, horizon=SIZE)", 300_000),
    "vib_boot_h12_k2": ("X = var_data(200, 2, 1)",
                        "tsecon.var_irf_bands(X, lags=2, horizon=12, method='bootstrap', n_boot=1000)",
                        "tsecon.var_irf_bands(X, lags=2, horizon=12, method='bootstrap', n_boot=SIZE)", 200_000),
    "vib_boot_h0_k2": ("X = var_data(200, 2, 1)",
                       "tsecon.var_irf_bands(X, lags=2, horizon=0, method='bootstrap', n_boot=1000)",
                       "tsecon.var_irf_bands(X, lags=2, horizon=0, method='bootstrap', n_boot=SIZE)", 400_000),
    "lrsb_boot_h12_k2": ("X = var_data(200, 2, 1)",
                         "tsecon.long_run_svar_bands(X, lags=2, horizon=12, n_boot=1000)",
                         "tsecon.long_run_svar_bands(X, lags=2, horizon=12, n_boot=SIZE)", 200_000),
    "lrsb_boot_h0_k2": ("X = var_data(200, 2, 1)",
                        "tsecon.long_run_svar_bands(X, lags=2, horizon=0, n_boot=1000)",
                        "tsecon.long_run_svar_bands(X, lags=2, horizon=0, n_boot=SIZE)", 400_000),
    "lrsb_boot_h12_k2_draws": ("X = var_data(200, 2, 1)",
                               "tsecon.long_run_svar_bands(X, lags=2, horizon=12, n_boot=1000, return_draws=True)",
                               "tsecon.long_run_svar_bands(X, lags=2, horizon=12, n_boot=SIZE, return_draws=True)",
                               200_000),
    "lrsb_boot_h40_k3_supt": ("X = var_data(300, 3, 1)",
                              "tsecon.long_run_svar_bands(X, lags=1, horizon=40, n_boot=1000, band='sup-t')",
                              "tsecon.long_run_svar_bands(X, lags=1, horizon=40, n_boot=SIZE, band='sup-t')",
                              40_000),
}

BODY = r"""
{setup}
{warm}
p0 = peak()
t0 = time.perf_counter()
SIZE = {size}
r = {call}
dt = time.perf_counter() - t0
grown = peak() - p0
print(f"{{grown / SIZE:10.1f}} bytes per unit   peak growth {{grown / 2**20:8.1f}} MiB   {{dt:6.2f}} s")
"""


def main() -> None:
    names = sys.argv[1:] or list(CASES)
    for name in names:
        setup, warm, call, size = CASES[name]
        code = PRE + BODY.format(setup=setup, warm=warm, call=call, size=size)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        out = p.stdout.strip() or ("FAILED " + p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "")
        print(f"{name:26s} size={size:<8d} {out}", flush=True)


if __name__ == "__main__":
    main()
