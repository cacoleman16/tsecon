"""Cost of tsecon.variance_ratio: the robust variance is O(n q).

Reproduces the COST sentence of the `variance_ratio` docstring and the
"Horizons near n" failure mode of the model card
(docs/reference/model-cards/unit-root-cointegration-tests.md). Wall-clock,
best of three, on one Gaussian random walk of 20,001 levels.

    python scripts/variance_ratio_cost.py
"""
from __future__ import annotations

import time

import numpy as np

import tsecon


def best_of_three(fn) -> float:
    out = []
    for _ in range(3):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out)


def main() -> None:
    y = np.cumsum(np.random.default_rng(0).standard_normal(20_001))
    print(f"tsecon {tsecon.__version__}; n = {y.size} levels")
    for q in (16, 1_000, 10_000):
        rob = best_of_three(lambda: tsecon.variance_ratio(y, q))
        hom = best_of_three(lambda: tsecon.variance_ratio(y, q, robust=False))
        print(f"  q = {q:>6}: robust {rob:.4f} s   robust=False {hom:.4f} s")


if __name__ == "__main__":
    main()
