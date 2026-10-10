"""Measure the narrative weight loop's cost per work unit:
units = n_draws * n_weight_draws * (T_eff n^2 + (h + 3) n^3), the count
NARRATIVE_WEIGHT_WORK caps (as if every draw were accepted)."""
import time

import numpy as np
import tsecon


def var(T, n, seed, a=0.5, c=0.0):
    r = np.random.default_rng(seed)
    A = a * np.eye(n) + c
    X = np.zeros((T, n))
    for t in range(1, T):
        X[t] = A @ X[t - 1] + r.standard_normal(n)
    return X


SR = [(0, 0, 0, "+"), (1, 0, 0, "+")]
lags, h = 1, 4
# (n, T, data, narrative period, n_draws, n_weight_draws)
CASES = [
    (3, 160, var(160, 3, 101, 0.45, 0.08), 12, 20000, 200),
    (4, 400, var(400, 4, 0), 10, 20, 20000),
    (4, 400, var(400, 4, 0), 10, 100, 10000),
]
for n, T, X, period, D, K in CASES:
    NR = [{"type": "shock_sign", "shock": 0, "period": period, "sign": "+"}]
    t0 = time.time()
    out = tsecon.narrative_svar(X, SR, NR, lags=lags, horizon=h, n_draws=D, max_tries=200, n_weight_draws=K)
    dt = time.time() - t0
    units = D * K * ((T - lags) * n * n + (h + 3) * n**3)
    acc = out["diagnostics"]["accepted"]
    print(
        f"n={n} T={T} D={D} K={K}: {dt:.2f}s accepted={acc} units={units:.3e} "
        f"ns/unit={dt / units * 1e9:.3f}",
        flush=True,
    )
