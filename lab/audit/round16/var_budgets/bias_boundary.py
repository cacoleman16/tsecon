"""The largest n_boot the bootstrap budget admits with bias_correct=True must
run within the counted memory (2^28 values, 2 GiB).

Usage, memory-capped as the audit runs it:
    bash -c 'ulimit -v 4000000; timeout 300 python bias_boundary.py K vib|lrsb'
K is the number of variables (a VAR(1) at horizon 12 on 300 observations).
"""
import re, resource, sys, time
import numpy as np, tsecon

def var_data(t, k, seed):
    rng = np.random.default_rng(seed)
    a = 0.5 * np.eye(k) + 0.1 * np.ones((k, k)) / k
    y = np.zeros((t, k))
    for i in range(1, t):
        y[i] = a @ y[i - 1] + rng.standard_normal(k)
    return y

k = int(sys.argv[1]); fn = sys.argv[2]
X = var_data(300, k, 1)
kw = dict(lags=1, horizon=12, bias_correct=True)
call = (lambda n: tsecon.var_irf_bands(X, method="bootstrap", n_boot=n, **kw)) if fn == "vib" else \
       (lambda n: tsecon.long_run_svar_bands(X, n_boot=n, **kw))
try:
    call(10**9)
except ValueError as e:
    n_max = int(re.search(r"largest n_boot that fits at the other arguments is (\d+)", str(e)).group(1))
print(f"{fn} k={k}: largest admitted n_boot = {n_max}", flush=True)
t0 = time.perf_counter()
call(n_max)
print(f"ran in {time.perf_counter() - t0:.1f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB", flush=True)
