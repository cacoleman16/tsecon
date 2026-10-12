"""Cost of `multivariate_normality` and the source of its work budget.

Times the call on standard-normal T x k residual matrices with the installed
wheel, fits time ~ c1 * T k^2 + c2 * k^3 by non-negative least squares on the
relative error, and shows the refusal past the budget
(tsecon_var::NORMALITY_WORK_BUDGET = 2^34 units of `5 T k^2 + 20 k^3`, the
fitted coefficients rounded up). Before the round-16 fix the call validated
its input three times and had no budget: 0.07 s at 300 x 100, 3.8 s at
1200 x 400 and 98 s at 1500 x 1000 on the same machine.

    python lab/audit/round16/var_numerics/normality_cost.py
"""

import time

import numpy as np
from scipy.optimize import nnls

import tsecon

GRID = [
    (200, 100),
    (1000, 100),
    (400, 200),
    (3000, 200),
    (800, 400),
    (2000, 400),
    (5000, 300),
    (20000, 100),
    (1500, 600),
    (1000, 800),
]


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    print("    T     k   time (s)   5 T k^2 + 20 k^3")
    for t, k in GRID:
        x = rng.standard_normal((t, k))
        t0 = time.perf_counter()
        tsecon.multivariate_normality(x)
        dt = time.perf_counter() - t0
        rows.append((t, k, dt))
        print(f"{t:5d} {k:5d} {dt:10.3f}   {5 * t * k * k + 20 * k**3:.2e}")
    d = np.asarray(rows, dtype=float)
    a = np.c_[d[:, 0] * d[:, 1] ** 2, d[:, 1] ** 3]
    c, _ = nnls(a / d[:, 2:3], np.ones(len(d)))
    print(f"fit: {c[0] * 1e9:.1f} ns per T k^2, {c[1] * 1e9:.1f} ns per k^3")
    budget = 2**34
    print(f"the budget 2^34 units = {budget * 1e-9:.1f} s at 1 ns per unit")
    x = np.broadcast_to(np.float64(0.0), (1500, 1000))
    t0 = time.perf_counter()
    try:
        tsecon.multivariate_normality(x)
    except ValueError as e:
        print(f"1500 x 1000 refused in {time.perf_counter() - t0:.4f} s: {e}")


if __name__ == "__main__":
    main()
