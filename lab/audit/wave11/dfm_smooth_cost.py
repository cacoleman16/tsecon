"""Cost of `tsecon.dfm_smooth`, behind the work budget in
bindings/python/src/nowcast_smooth.rs and the timings quoted in its docstring,
the stub and the nowcasting model card.

The budget counts T * (2 m^3 + 20 N m^2 + N^2 + 1000 N + 12000) + 16 m^3 work
units (m = r p states, N series, T periods), about a nanosecond each: a few
m x m triple products per period, an O(m^2) update per observed series on the
way forward and back, the engine's per-period pass over the dense N x N
observation covariance, the per-series / per-period overhead, and once per
call the stationary initialization (the companion eigenvalue check and the
Lyapunov doubling, whose iteration count grows with the factor persistence:
the T = 1 cases below, at AR coefficient 0.5, 0.99999 and the largest double
below one, where the doubling runs its ~58 steps). The per-period
coefficients were chosen from a non-negative least-squares fit to the timings
of the first ten cases (--fit prints this run's fit; it moves with the
machine's load, so the ns-per-unit column is the number that matters); the
one-off coefficient prices the edge-of-stationarity case. Each case is timed
as the best of three on a stationary model with unit idiosyncratic
variances; the ns-per-unit column is what the 2^34 cap is calibrated against.
The last block shows refusals past the caps.

Run:  python lab/audit/wave11/dfm_smooth_cost.py --fit > lab/audit/wave11/out/dfm_smooth_cost.txt
"""
import platform
import sys
import time

import numpy as np

import tsecon

CAP = 2**34
EDGE = float(np.nextafter(1.0, 0.0))  # the largest stationary AR coefficient
CASES = [  # (T, N, r, p, AR coefficient on the first lag)
    (500, 50, 2, 4, 0.5),
    (2000, 200, 3, 4, 0.5),
    (1000, 10, 4, 10, 0.5),
    (60000, 30, 2, 5, 0.5),
    (30000, 60, 4, 4, 0.5),
    (4000, 1000, 1, 2, 0.5),
    (200, 5, 10, 25, 0.5),
    (1000, 2000, 1, 1, 0.5),
    (200000, 1, 1, 1, 0.5),
    (100000, 30, 1, 2, 0.5),
    (8000, 1000, 1, 2, 0.5),
    (1, 2, 8, 120, 0.5),
    (1, 2, 8, 120, 0.99999),
    (1, 2, 1, 960, EDGE),  # one factor: its root is exact, so it stays < 1
]
N_FIT = 10
REFUSED = [  # (T, N, r, p): past a cap, refused before anything is built
    (1, 2, 8, 125),  # 18 m^3 + ... > 2^34 at m = 1000
    (100, 9000, 1, 1),  # N^2 > 2^26
    (9000, 1000, 1, 2),  # T (N^2 + 1000 N + ...) > 2^34
]


def units_of(t_len, n, m):
    return t_len * (2 * m**3 + 20 * n * m * m + n * n + 1000 * n + 12000) + 16 * m**3


def fit(rows):
    """Non-negative least squares of the ns per period on the five terms."""
    from scipy.optimize import nnls

    a = np.array([[m**3, n * m * m, n * n, n, 1.0] for t, n, m, _ in rows])
    b = np.array([s / t * 1e9 for t, n, m, s in rows])
    sc = a.max(axis=0)
    x, _ = nnls(a / sc, b)
    return x / sc


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"# tsecon {tsecon.__version__}, Python {platform.python_version()}, {platform.machine()}")
    print("#      T      N    m    1-phi      units  x cap   seconds  ns/unit")
    timed = []
    for t_len, n, r, p, phi in CASES:
        lam = rng.uniform(0.5, 1.0, (n, r))
        ar = np.zeros((r, r * p))
        ar[:, :r] = phi * np.eye(r)
        y = rng.standard_normal((t_len, n))
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            tsecon.dfm_smooth(y, loadings=lam, factor_ar=ar, idiosyncratic=np.ones(n))
            best = min(best, time.perf_counter() - t0)
        m = r * p
        units = units_of(t_len, n, m)
        timed.append((t_len, n, m, best))
        print(
            f"{t_len:8d} {n:6d} {m:4d} {1 - phi:8.1e} {units:10.2e} {units / CAP:6.2f} "
            f"{best:9.3f} {best / units * 1e9:8.3f}"
        )
    if "--fit" in sys.argv:
        x = fit(timed[:N_FIT])
        print(
            "# NNLS fit, ns per period (first %d cases): %.2f m^3 + %.2f N m^2 + %.2f N^2 "
            "+ %.0f N + %.0f" % (N_FIT, *x)
        )
    print("# refused past the caps (time to the refusal):")
    for t_len, n, r, p in REFUSED:
        lam = np.ones((n, r))
        ar = np.zeros((r, r * p))
        ar[:, :r] = 0.5 * np.eye(r)
        y = np.zeros((t_len, n))
        t0 = time.perf_counter()
        try:
            tsecon.dfm_smooth(y, loadings=lam, factor_ar=ar, idiosyncratic=np.ones(n))
            msg = "NOT REFUSED"
        except ValueError as e:
            msg = str(e)[:110]
        print(f"{t_len:8d} {n:6d} {r * p:4d} {time.perf_counter() - t0:7.3f}s  {msg}")


if __name__ == "__main__":
    main()
