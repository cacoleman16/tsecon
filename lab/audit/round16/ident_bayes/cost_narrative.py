"""Measure the narrative weight loop's cost per work unit:
units = n_draws * n_weight_draws * (T_eff n^2 + (h + 3) n^3)."""
import time, numpy as np, tsecon
r = np.random.default_rng(0)
for n, T in ((3, 200), (4, 400)):
    X = np.zeros((T, n))
    for t in range(1, T):
        X[t] = 0.5 * X[t - 1] + r.standard_normal(n)
    SR = [(0, 0, 0, "+"), (1, 0, 0, "+")]
    NR = [{"type": "shock_sign", "shock": 0, "period": 10, "sign": "+"}]
    lags, h = 1, 4
    for D, K in ((20, 5000), (20, 20000), (100, 10000)):
        t0 = time.time()
        tsecon.narrative_svar(X, SR, NR, lags=lags, horizon=h, n_draws=D, max_tries=200, n_weight_draws=K)
        dt = time.time() - t0
        units = D * K * ((T - lags) * n * n + (h + 3) * n ** 3)
        print(f"n={n} T={T} D={D} K={K}: {dt:.2f}s units={units:.3e} ns/unit={dt / units * 1e9:.3f}", flush=True)
