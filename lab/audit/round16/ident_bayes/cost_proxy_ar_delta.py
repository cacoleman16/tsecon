import time, numpy as np, tsecon
r = np.random.default_rng(0)
T = 300
X = np.zeros((T, 3))
for t in range(1, T):
    X[t] = 0.5 * X[t-1] + r.standard_normal(3)
proxy = r.standard_normal(T) + X[:, 0]
for lags in (2, 4):
    for H in (500, 1000, 2000):
        t0 = time.time()
        tsecon.proxy_ar_sets(X, proxy, lags=lags, horizon=H)
        dt = time.time() - t0
        n, p = 3, lags
        np_, dim = n * p, n * n * p
        work = H * (H + 1) / 2 * np_ * n * (n + 1) + H * (n * dim**2 + n * n * dim + np_**3)
        print(f"lags={lags} H={H}: {dt:.2f}s  work={work:.3e}  ns/unit={dt/work*1e9:.3f}", flush=True)
