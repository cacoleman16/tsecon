"""Round-8 probe: the NU_GAUSSIAN_RIDGE deterministic converged flag.

Attacks per the round brief:
  (a) genuine interior optima near nu = 1e3 — would the forced-False flag
      misreport a real optimum?
  (b) ridge rides that stop below the threshold with converged=True — the
      platform-dependence the fix claims to have removed;
  (c) Laplace/Gaussian densities unaffected.
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


RIDGE = 1e3

# ---------- (c) Gaussian / Laplace unaffected ----------
rng = np.random.default_rng(80806)
y_clean = 5.0 + np.cumsum(0.05 * rng.standard_normal(400)) + 0.5 * rng.standard_normal(400)
for dens in ["gaussian", "laplace"]:
    r = tsecon.dcs_local_level(y_clean, density=dens)
    check(f"dcs {dens}: converged True on clean data", r["converged"] is True,
          f"converged={r['converged']}")

# ---------- t on clean Gaussian data: ridge -> deterministic False ----------
ridge_cases = []
below_thresh_converged = []
for seed in range(20):
    rr = np.random.default_rng(1000 + seed)
    y = 2.0 + np.cumsum(0.03 * rr.standard_normal(300)) + 0.4 * rr.standard_normal(300)
    r = tsecon.dcs_local_level(y, density="t")
    ridge_cases.append((seed, r["nu"], r["converged"]))
    if r["nu"] > RIDGE and r["converged"]:
        fails.append((f"dcs t seed {seed}: nu={r['nu']:.3g} > 1e3 but converged=True", ""))
    if r["nu"] <= RIDGE and r["converged"]:
        below_thresh_converged.append((seed, r["nu"]))
attempted += 20
made += 20
n_ridge = sum(1 for _, nu, _ in ridge_cases if nu > RIDGE)
print(f"[note] dcs t on 20 clean-Gaussian series: {n_ridge} fits with nu > 1e3 "
      f"(all must have converged=False), "
      f"{len(below_thresh_converged)} converged=True fits at nu <= 1e3: {below_thresh_converged}")
check("dcs t: every nu>1e3 fit reports converged=False",
      all((not c) for _, nu, c in ridge_cases if nu > RIDGE))
# (b): converged=True stops below the threshold on clean Gaussian data would
# reintroduce the platform lottery. Report incidence and where nu landed.
for seed, nu in below_thresh_converged:
    print(f"    [suspect] seed {seed}: converged=True at nu={nu:.4f} on clean Gaussian data")

# distribution of nu at the ridge: how far do the rides go?
print("[note] nu landings:", ", ".join(f"{nu:.3g}({'T' if c else 'F'})" for _, nu, c in ridge_cases))

# ---------- t data with genuine interior optimum ----------
ok_int = 0
for seed in range(10):
    rr = np.random.default_rng(2000 + seed)
    lvl = np.cumsum(0.05 * rr.standard_normal(400))
    y = lvl + 0.5 * rr.standard_t(5, 400)
    r = tsecon.dcs_local_level(y, density="t")
    if r["converged"] and 2.5 < r["nu"] < 30:
        ok_int += 1
attempted += 10
made += 10
check("dcs t: interior optima (true nu=5) certified converged (>=8/10)", ok_int >= 8, f"{ok_int}/10")

# ---------- gas_volatility student_t ----------
gas_ridge = []
gas_below = []
for seed in range(20):
    rr = np.random.default_rng(3000 + seed)
    ret = 0.01 * rr.standard_normal(500)  # Gaussian returns, no GARCH
    r = tsecon.gas_volatility(ret, density="student_t")
    gas_ridge.append((seed, r["nu"], r["converged"]))
    if r["nu"] > RIDGE and r["converged"]:
        fails.append((f"gas seed {seed}: nu={r['nu']:.3g} > 1e3 but converged=True", ""))
    if r["nu"] <= RIDGE and r["converged"]:
        gas_below.append((seed, round(r["nu"], 2)))
attempted += 20
made += 20
n_ridge_g = sum(1 for _, nu, _ in gas_ridge if nu > RIDGE)
print(f"[note] gas_volatility t on 20 Gaussian-return series: {n_ridge_g} fits with nu > 1e3, "
      f"{len(gas_below)} converged=True at nu <= 1e3: {gas_below}")
check("gas t: every nu>1e3 fit reports converged=False",
      all((not c) for _, nu, c in gas_ridge if nu > RIDGE))
print("[note] gas nu landings:", ", ".join(f"{nu:.3g}({'T' if c else 'F'})" for _, nu, c in gas_ridge))

# interior: t returns with GARCH-type variance
ok_int_g = 0
nus = []
for seed in range(10):
    rr = np.random.default_rng(4000 + seed)
    T = 750
    f = np.zeros(T)
    yv = np.zeros(T)
    f[0] = 1.0
    for t in range(T):
        eps = rr.standard_t(6) / np.sqrt(6 / 4)  # unit-variance t6
        yv[t] = np.sqrt(f[t]) * eps
        if t + 1 < T:
            f[t + 1] = 0.1 + 0.1 * (yv[t] ** 2 - f[t]) + 0.85 * f[t] + 0.1 * 0  # GARCH-like
    r = tsecon.gas_volatility(yv, density="student_t")
    nus.append(round(r["nu"], 1))
    if r["converged"] and r["nu"] < 100:
        ok_int_g += 1
attempted += 10
made += 10
check("gas t: interior optima (true nu=6) certified converged (>=7/10)", ok_int_g >= 7,
      f"{ok_int_g}/10, nus={nus}")

# ---------- (a) near-threshold data: t with true nu ~ 1e3 ----------
# Statistically ~indistinguishable from Gaussian; the question is only whether
# the fit can land in (say) [500, 1e3] with converged=True (platform lottery
# territory) or just above 1e3 with a *genuine* interior optimum.
landings = []
for seed in range(12):
    rr = np.random.default_rng(5000 + seed)
    lvl = np.cumsum(0.05 * rr.standard_normal(500))
    y = lvl + 0.5 * rr.standard_t(800, 500)
    r = tsecon.dcs_local_level(y, density="t")
    landings.append((round(r["nu"], 1), r["converged"]))
attempted += 12
made += 12
print("[note] true-nu=800 landings (nu, converged):", landings)
viol = [(nu, c) for nu, c in landings if nu > RIDGE and c]
check("near-threshold: no converged=True above the ridge", len(viol) == 0, str(viol))
sus = [(nu, c) for nu, c in landings if 200 < nu <= RIDGE and c]
print(f"[note] converged=True landings in (200, 1000]: {sus} "
      "(these would be certified 'interior optima' the threshold narrowly spares)")

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
