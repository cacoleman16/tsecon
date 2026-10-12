import numpy as np, tsecon, time, sys
def var_data(t, k, seed):
    rng = np.random.default_rng(seed)
    a = 0.5 * np.eye(k) + 0.1 * np.ones((k, k)) / k
    y = np.zeros((t, k))
    for i in range(1, t):
        y[i] = a @ y[i - 1] + rng.standard_normal(k)
    return y
for k, p, h in [(2, 8, 500), (2, 8, 1000), (2, 8, 2047), (3, 2, 900), (10, 4, 60), (20, 4, 14), (20, 4, 28)]:
    X = var_data(max(400, 10 * k * p), k, 1)
    t = time.perf_counter(); tsecon.var_irf_bands(X, lags=p, horizon=h)
    w = h * (h + 1) // 2 * k * k * p * k * k
    print(f"k={k} lags={p} horizon={h}: {time.perf_counter() - t:.2f} s, jacobian work {w:.3g} = 2^{np.log2(w):.1f}", flush=True)
