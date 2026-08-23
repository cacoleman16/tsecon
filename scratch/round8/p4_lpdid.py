"""Round-8 probe: lp_did.

Independent NumPy reference for the absorbing case (clean-control long-
difference regression with period FE, entity-clustered fixest-convention SEs),
plus lens 1/2/3 sweeps.
"""
import numpy as np
import tsecon

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
        print(f"[ok] {name}: {type(e).__name__}: {str(e)[:100]}")


# ---------- DGP: staggered absorbing adoption, heterogeneous cohorts ----------
rng = np.random.default_rng(80804)
N, T = 60, 30
adopt = np.full(N, 10**9)
adopt[:15] = 10
adopt[15:30] = 15
adopt[30:45] = 20
# 15 never treated
D = np.zeros((N, T))
for i in range(N):
    if adopt[i] < T:
        D[i, adopt[i]:] = 1.0
alpha = rng.normal(0, 1, N)[:, None]
delta = rng.normal(0, 0.5, T)[None, :]
eps = rng.normal(0, 0.4, (N, T))
# homogeneous dynamic effect: 1.0 at h=0 growing 0.2/period
effect = np.zeros((N, T))
for i in range(N):
    if adopt[i] < T:
        for t in range(adopt[i], T):
            effect[i, t] = 1.0 + 0.2 * (t - adopt[i])
y = alpha + delta + effect + eps

res = tsecon.lp_did(y, D, pre_window=4, post_window=6)

H = np.asarray(res["horizons"])
check("horizons run -4..6", np.array_equal(H, np.arange(-4, 7)), str(H))
i_m1 = int(np.where(H == -1)[0][0])
check("h=-1 coef exact zero", res["coef"][i_m1] == 0.0)
check("h=-1 se exact zero", res["se"][i_m1] == 0.0)

# truth: coef[h] ~ 1 + 0.2h for h>=0; pre-trends ~ 0
for h in range(0, 7):
    ih = int(np.where(H == h)[0][0])
    truth = 1.0 + 0.2 * h
    check(f"h={h} coef near truth {truth:.1f}", abs(res["coef"][ih] - truth) < 4 * res["se"][ih] + 0.15,
          f"{res['coef'][ih]:.3f} (se {res['se'][ih]:.3f})")
for h in [-4, -3, -2]:
    ih = int(np.where(H == h)[0][0])
    check(f"h={h} pre-trend near 0", abs(res["coef"][ih]) < 4 * res["se"][ih] + 0.1,
          f"{res['coef'][ih]:.3f}")


# ---------- independent reference implementation (absorbing) ----------
def ref_lpdid_h(y, D, h):
    """Long-difference clean-control regression at post horizon h (absorbing).

    Returns (coef, se_fixest, nobs, n_switchers)."""
    N, T = y.shape
    rows = []  # (i, t, dy, treat)
    for t in range(1, T):
        if t + h > T - 1:
            continue
        for i in range(N):
            switch = D[i, t] == 1 and D[i, t - 1] == 0
            clean = D[i, t + h] == 0
            if switch or clean:
                rows.append((i, t, y[i, t + h] - y[i, t - 1], 1.0 if switch else 0.0))
    rows = np.array(rows)
    ent = rows[:, 0].astype(int)
    per = rows[:, 1].astype(int)
    dy = rows[:, 2]
    tr = rows[:, 3]
    periods = np.unique(per)
    X = np.zeros((len(rows), 1 + len(periods)))
    X[:, 0] = tr
    for k, p in enumerate(periods):
        X[per == p, 1 + k] = 1.0
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ dy)
    e = dy - X @ beta
    n, K = X.shape
    G = len(np.unique(ent))
    meat = np.zeros((K, K))
    for g in np.unique(ent):
        Xg = X[ent == g]
        eg = e[ent == g]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    XtXi = np.linalg.inv(XtX)
    V = XtXi @ meat @ XtXi * ((n - 1) / (n - K)) * (G / (G - 1))
    return beta[0], np.sqrt(V[0, 0]), n, int(tr.sum())


def ref_lpdid_pre(y, D, j):
    """Pre horizon h=-j (j>=2): controls need D[i,t]==0 only."""
    N, T = y.shape
    rows = []
    for t in range(1, T):
        if t - j < 0:
            continue
        for i in range(N):
            switch = D[i, t] == 1 and D[i, t - 1] == 0
            clean = D[i, t] == 0
            if switch or clean:
                rows.append((i, t, y[i, t - j] - y[i, t - 1], 1.0 if switch else 0.0))
    rows = np.array(rows)
    ent = rows[:, 0].astype(int)
    per = rows[:, 1].astype(int)
    dy = rows[:, 2]
    tr = rows[:, 3]
    periods = np.unique(per)
    X = np.zeros((len(rows), 1 + len(periods)))
    X[:, 0] = tr
    for k, p in enumerate(periods):
        X[per == p, 1 + k] = 1.0
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ dy)
    e = dy - X @ beta
    n, K = X.shape
    G = len(np.unique(ent))
    meat = np.zeros((K, K))
    for g in np.unique(ent):
        Xg = X[ent == g]
        eg = e[ent == g]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    XtXi = np.linalg.inv(XtX)
    V = XtXi @ meat @ XtXi * ((n - 1) / (n - K)) * (G / (G - 1))
    return beta[0], np.sqrt(V[0, 0]), n, int(tr.sum())


for h in [0, 3, 6]:
    ih = int(np.where(H == h)[0][0])
    b, s, n_, ns_ = ref_lpdid_h(y, D, h)
    check(f"h={h} coef == independent reference (1e-10)", abs(res["coef"][ih] - b) < 1e-10,
          f"{res['coef'][ih]:.12f} vs {b:.12f}")
    check(f"h={h} se == independent fixest-convention reference (1e-10)",
          abs(res["se"][ih] - s) < 1e-10, f"{res['se'][ih]:.12f} vs {s:.12f}")
    check(f"h={h} nobs matches", res["nobs"][ih] == n_, f"{res['nobs'][ih]} vs {n_}")
    check(f"h={h} n_switchers matches", res["n_switchers"][ih] == ns_,
          f"{res['n_switchers'][ih]} vs {ns_}")

for j in [2, 4]:
    ih = int(np.where(H == -j)[0][0])
    b, s, n_, ns_ = ref_lpdid_pre(y, D, j)
    check(f"h=-{j} coef == independent reference (1e-10)", abs(res["coef"][ih] - b) < 1e-10,
          f"{res['coef'][ih]:.12f} vs {b:.12f}")
    check(f"h=-{j} se == independent reference (1e-10)", abs(res["se"][ih] - s) < 1e-10,
          f"{res['se'][ih]:.12f} vs {s:.12f}")

# ---------- lens 1: axes ----------
res_rw = tsecon.lp_did(y, D, pre_window=4, post_window=6, reweight=True)
check("reweight axis alive", not np.allclose(res_rw["coef"], res["coef"]))
res_nt = tsecon.lp_did(y, D, pre_window=4, post_window=6, never_treated_only=True)
check("never_treated_only axis alive", not np.allclose(res_nt["coef"], res["coef"]))
res_p = tsecon.lp_did(y, D, pre_window=4, post_window=6, pooled=True)
check("pooled adds the documented keys",
      all(k in res_p for k in ["pooled_post_att", "pooled_post_se", "pooled_post_nobs",
                               "pooled_post_n_switchers", "pooled_pre_att", "pooled_pre_se"]))
check("pooled keys absent when pooled=False", "pooled_post_att" not in res)
check("pooled event-study unchanged (same coef)", np.allclose(res_p["coef"], res["coef"], atol=0, rtol=0))
# stamped options
for k, v in [("absorbing", True), ("nonabsorbing_lag", 0), ("reweight", False),
             ("pooled", False), ("never_treated_only", False)]:
    check(f"stamp {k} == {v}", res[k] == v, str(res.get(k)))
print("[note] se_type stamp:", res.get("se_type"))

# nonabsorbing path
D_rev = D.copy()
D_rev[3, 20:] = 0.0  # unit 3 exits at 20
res_na = tsecon.lp_did(y, D_rev, pre_window=2, post_window=3, absorbing=False, nonabsorbing_lag=3)
check("nonabsorbing path runs", np.isfinite(res_na["coef"][-1]))
res_na2 = tsecon.lp_did(y, D_rev, pre_window=2, post_window=3, absorbing=False, nonabsorbing_lag=6)
check("nonabsorbing_lag axis alive", not np.allclose(res_na["coef"], res_na2["coef"]))

# ---------- lens 2: scale equivariance ----------
res_s = tsecon.lp_did(y * 1e6, D, pre_window=4, post_window=6)
check("coef equivariant in y scale", np.allclose(np.asarray(res_s["coef"]), np.asarray(res["coef"]) * 1e6, rtol=1e-9))
check("se equivariant in y scale", np.allclose(np.asarray(res_s["se"]), np.asarray(res["se"]) * 1e6, rtol=1e-9))

# ---------- lens 3: degenerate ----------
expect_raise("reverting treatment under absorbing", lambda: tsecon.lp_did(y, D_rev, pre_window=2, post_window=3))
expect_raise("non-0/1 treatment", lambda: tsecon.lp_did(y, D * 0.5, pre_window=2, post_window=3))
expect_raise("no switchers (all zero D)", lambda: tsecon.lp_did(y, np.zeros_like(D), pre_window=2, post_window=3))
expect_raise("all treated from t=0 (no switch observed)", lambda: tsecon.lp_did(y, np.ones_like(D), pre_window=2, post_window=3))
expect_raise("NaN outcome", lambda: tsecon.lp_did(np.where(np.arange(T) == 2, np.nan, y), D, pre_window=2, post_window=3))
expect_raise("post_window too long for T", lambda: tsecon.lp_did(y, D, pre_window=2, post_window=40))
expect_raise("shape mismatch", lambda: tsecon.lp_did(y[:, :20], D, pre_window=2, post_window=3))

# clean-control condition does real work: naive all-controls comparison biased
# on heterogeneous-cohort DGP (the CHANGELOG's 56.5% claim direction)
eff_het = np.zeros((N, T))
for i in range(N):
    if adopt[i] < T:
        for t in range(adopt[i], T):
            eff_het[i, t] = (3.0 if adopt[i] == 10 else 0.5) * (1 + 0.3 * (t - adopt[i]))
y_het = alpha + delta + eff_het + eps
r_het = tsecon.lp_did(y_het, D, pre_window=2, post_window=6)
r_het_rw = tsecon.lp_did(y_het, D, pre_window=2, post_window=6, reweight=True)
print(f"[note] heterogeneous cohorts: VW ATT h=6 {r_het['coef'][-1]:.3f}, EW {r_het_rw['coef'][-1]:.3f}")
check("reweight moves ATT under heterogeneity", abs(r_het["coef"][-1] - r_het_rw["coef"][-1]) > 0.05)

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
