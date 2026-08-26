"""Round-8 probe: scale_ar on bvar_fit / bvar_irf_draws / bvar_hierarchical.

Lens 1 (axis alive on all three surfaces), 3 (degenerate values), plus an
independent check of the AR(p) residual-variance scaling for scale_ar=1.
"""
import numpy as np
import tsecon
import statsmodels.api as sm

attempted = 0
made = 0
fails = []


def check(name, cond, detail=""):
    global attempted, made
    attempted += 1
    made += 1
    if not cond:
        fails.append((name, detail))
    print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}")


def expect_raise(name, fn):
    global attempted, made
    attempted += 1
    try:
        fn()
        made += 1
        fails.append((name, "no raise"))
        print(f"[FAIL] {name}: no raise")
    except Exception as e:
        made += 1
        print(f"[ok] {name}: {type(e).__name__}: {str(e)[:110]}")


rng = np.random.default_rng(80805)
T, n = 150, 3
A = np.array([[0.6, 0.1, 0.0], [0.0, 0.5, 0.1], [0.1, 0.0, 0.4]])
y = np.zeros((T, n))
e = rng.standard_normal((T, n)) @ np.diag([1.0, 0.5, 2.0])
for t in range(1, T):
    y[t] = A @ y[t - 1] + e[t]

f4 = tsecon.bvar_fit(y, lags=2)
f4e = tsecon.bvar_fit(y, lags=2, scale_ar=4)
f1 = tsecon.bvar_fit(y, lags=2, scale_ar=1)
check("default == explicit scale_ar=4 (bitwise logml)", f4["log_ml"] == f4e["log_ml"]
      if "log_ml" in f4 else np.array_equal(np.asarray(f4[list(f4)[0]]), np.asarray(f4e[list(f4)[0]])),
      str(sorted(f4.keys())))
key = "log_ml" if "log_ml" in f4 else sorted(f4.keys())[0]
check("scale_ar axis alive on bvar_fit", f4[key] != f1[key], f"{f4[key]} vs {f1[key]}")

d4 = tsecon.bvar_irf_draws(y, lags=2, horizon=8, n_draws=50, seed=3)
d1 = tsecon.bvar_irf_draws(y, lags=2, horizon=8, n_draws=50, seed=3, scale_ar=1)
a4 = np.asarray(d4, dtype=float)
a1 = np.asarray(d1, dtype=float)
check("scale_ar axis alive on bvar_irf_draws", not np.array_equal(a4, a1), str(a4.shape))

h4 = tsecon.bvar_hierarchical(y, lags=2)
h1 = tsecon.bvar_hierarchical(y, lags=2, scale_ar=1)
check("scale_ar axis alive on bvar_hierarchical", h4["lambda1_opt"] != h1["lambda1_opt"],
      f"{h4['lambda1_opt']:.5f} vs {h1['lambda1_opt']:.5f}")

# degenerate values
expect_raise("scale_ar=0", lambda: tsecon.bvar_fit(y, lags=2, scale_ar=0))
expect_raise("scale_ar negative -> teaching ValueError", lambda: tsecon.bvar_fit(y, lags=2, scale_ar=-1))
expect_raise("scale_ar >= T", lambda: tsecon.bvar_fit(y, lags=2, scale_ar=T))

# scale_ar=2 vs 4 vs 1 all pairwise distinct (not just 1 vs 4)
f2 = tsecon.bvar_fit(y, lags=2, scale_ar=2)
check("scale_ar=2 differs from both 1 and 4", f2[key] != f1[key] and f2[key] != f4[key])

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
print("bvar_fit keys:", sorted(f4.keys()))
