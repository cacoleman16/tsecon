import numpy as np

def var_data(T=200, k=2, seed=1):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.2, 0.3]]) if k == 2 else 0.3*np.eye(k)
    X = np.zeros((T, k))
    for t in range(1, T):
        X[t] = A @ X[t-1] + rng.standard_normal(k)
    return X

def dfm_panel(T=120, N=6, seed=2):
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.7*f[t-1] + rng.standard_normal()
    L = rng.uniform(0.5, 1.5, N)
    X = np.outer(f, L) + 0.5*rng.standard_normal((T, N))
    X[-1, :2] = np.nan
    X[-2, 0] = np.nan
    return X

def midas_data(n=150, K=3, seed=3):
    rng = np.random.default_rng(seed)
    hf = rng.standard_normal((n, K))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4*y[t-1] + hf[t] @ np.array([0.5, 0.3, 0.1][:K]) + rng.standard_normal()
    return y, hf

def arma_data(n=300, seed=4):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n+1)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6*y[t-1] + e[t+1] + 0.3*e[t] + 0.5
    return y

def describe(v, depth=0):
    import numpy as np
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape}{v.dtype}"
    if isinstance(v, list):
        if not v: return "list[]"
        return f"list[{len(v)}]<{describe(v[0], depth+1)}>"
    if isinstance(v, tuple):
        return f"tuple[{len(v)}]"
    if isinstance(v, dict):
        return "dict{" + ",".join(f"{k}:{describe(x)}" for k, x in v.items()) + "}"
    return type(v).__name__
