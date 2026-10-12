"""Cost of `tsecon.adl_midas`, behind the design cap and the work budget in
bindings/python/src/midas_adl.rs and the timings quoted in its docstring and
the stub.

The budget counts nobs * k^2 * (L + 2) work units (nobs = n - P fitted rows,
k = 1 + P + K coefficients, L = min(maxlags, nobs - 1) HAC lags, 0 under
"nonrobust"): one pass over the k^2 score cross-products per HAC lag, plus the
OLS fit and the sandwich. Each case is timed as the best of three on white
noise; the ns-per-unit column is what the 2^33 cap is calibrated against. The
last block shows the refusals at and past the caps (no time is spent there).

Run:  python lab/audit/wave11/adl_midas_cost.py > lab/audit/wave11/out/adl_midas_cost.txt
"""
import platform
import time

import numpy as np

import tsecon

CAP = 2**33
CASES = [  # (n, P, K, maxlags, se_type)
    (160, 2, 6, None, "hac"),
    (20000, 1, 6, None, "hac"),
    (100000, 1, 50, None, "hac"),
    (20000, 1, 6, 2000, "hac"),
    (5000, 1, 6, 4999, "hac"),
    (20000, 150, 6, None, "hac"),
    (4000, 600, 6, None, "nonrobust"),
    (4000, 400, 6, None, "hac"),
    (100000, 1, 6, 1340, "hac"),
]
REFUSED = [  # past a cap: refused before any column is built
    (100000, 49000, 1, None, "nonrobust"),  # design 51000 x 49002 cells
    (100000, 1, 6, 10**9, "hac"),  # maxlags: 99998 lags of 64 terms
    (20000, 2000, 6, None, "nonrobust"),  # ar_lags: k = 2007
]


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"# tsecon {tsecon.__version__}, Python {platform.python_version()}, {platform.machine()}")
    print("#      n      P    K      L         units  x cap   seconds  ns/unit")
    for n, p, k_hf, maxlags, se in CASES:
        y = rng.standard_normal(n)
        hf = rng.standard_normal((n, k_hf))
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            res = tsecon.adl_midas(y, hf, ar_lags=p, se_type=se, maxlags=maxlags)
            best = min(best, time.perf_counter() - t0)
        nobs = n - p
        k = 1 + p + k_hf
        lags = 0 if res["maxlags"] is None else min(res["maxlags"], nobs - 1)
        units = nobs * k * k * (lags + 2)
        print(
            f"{n:8d} {p:6d} {k_hf:4d} {lags:6d} {units:13.2e} {units / CAP:6.2f} "
            f"{best:9.3f} {best / units * 1e9:8.3f}"
        )
    print("# refused past the caps (time to the refusal):")
    for n, p, k_hf, maxlags, se in REFUSED:
        y = rng.standard_normal(n)
        hf = rng.standard_normal((n, k_hf))
        t0 = time.perf_counter()
        try:
            tsecon.adl_midas(y, hf, ar_lags=p, se_type=se, maxlags=maxlags)
            msg = "NOT REFUSED"
        except ValueError as e:
            msg = str(e)[:110]
        print(f"{n:8d} {p:6d} {k_hf:4d} {str(maxlags):>10} {time.perf_counter() - t0:7.3f}s  {msg}")


if __name__ == "__main__":
    main()
