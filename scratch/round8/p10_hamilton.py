"""Round-8 probe: hamilton_filter vs an independent OLS reference.

Hamilton (2018): regress y_{t} on [1, y_{t-h}, y_{t-h-1}, ..., y_{t-h-p+1}];
cycle = residual, trend = fitted. Also lens 1/2/3.
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
        print(f"[ok] {name}: {type(e).__name__}: {str(e)[:90]}")


rng = np.random.default_rng(80810)
T = 200
y = 100 + np.cumsum(0.5 + rng.normal(0, 1, T))

h, p = 8, 4
r = tsecon.hamilton_filter(y, h=h, p=p)
print("keys:", sorted(r.keys()), "| first_index:", r["first_index"],
      "| beta:", np.round(np.asarray(r["beta"]), 4))

# reference: rows t = h+p-1 .. T-1
rows = np.arange(h + p - 1, T)
X = np.column_stack([np.ones(len(rows))] + [y[rows - h - k] for k in range(p)])
ols = sm.OLS(y[rows], X).fit()
trend_ref = ols.fittedvalues
cycle_ref = ols.resid

tr = np.asarray(r["trend"])
cy = np.asarray(r["cycle"])
check("first_index == h+p-1", r["first_index"] == h + p - 1, str(r["first_index"]))
check("trend/cycle lengths == T - (h+p-1)", len(tr) == len(rows) and len(cy) == len(rows),
      f"{len(tr)} vs {len(rows)}")
check("beta == statsmodels OLS params (1e-9)",
      np.allclose(np.asarray(r["beta"]), ols.params, rtol=1e-9, atol=1e-12),
      f"{np.asarray(r['beta'])} vs {ols.params.to_numpy() if hasattr(ols.params,'to_numpy') else ols.params}")
check("trend == fitted (1e-9)", np.allclose(tr, trend_ref, rtol=1e-9),
      f"max {np.max(np.abs(tr - trend_ref)):.2e}")
check("cycle == residuals (1e-9)", np.allclose(cy, cycle_ref, rtol=1e-7, atol=1e-9),
      f"max {np.max(np.abs(cy - cycle_ref)):.2e}")
check("trend + cycle == y (identity)", np.allclose(tr + cy, y[rows], rtol=0, atol=1e-9))

# lens 1: h, p axes
r2 = tsecon.hamilton_filter(y, h=4, p=4)
check("h axis alive", r2["first_index"] != r["first_index"] or not np.allclose(
    np.asarray(r2["cycle"])[:10], cy[:10]))
r3 = tsecon.hamilton_filter(y, h=8, p=2)
check("p axis alive", len(np.asarray(r3["beta"])) == 3, str(len(np.asarray(r3["beta"]))))

# lens 2: scale equivariance
r_s = tsecon.hamilton_filter(y * 1e-8, h=h, p=p)
check("cycle equivariant in scale", np.allclose(np.asarray(r_s["cycle"]), cy * 1e-8, rtol=1e-6),
      f"max rel {np.max(np.abs(np.asarray(r_s['cycle']) / 1e-8 - cy)):.2e}")

# lens 3
expect_raise("too short", lambda: tsecon.hamilton_filter(y[: h + p], h=h, p=p))
expect_raise("p=0", lambda: tsecon.hamilton_filter(y, h=8, p=0))
expect_raise("h=0", lambda: tsecon.hamilton_filter(y, h=0, p=4))
expect_raise("NaN", lambda: tsecon.hamilton_filter(np.where(np.arange(T) == 5, np.nan, y), h=h, p=p))
# constant series: collinear regressors
try:
    rc = tsecon.hamilton_filter(np.full(60, 3.0), h=8, p=4)
    print("[note] constant series -> beta:", np.asarray(rc["beta"]), "cycle sd:",
          np.std(np.asarray(rc["cycle"])))
except Exception as e:
    print("[note] constant series raises:", type(e).__name__, str(e)[:90])

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
