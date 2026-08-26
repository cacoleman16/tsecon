"""Round-8 probe: bns_jump_test + the realized-measure trio.

Independent transcription of the Huang-Tauchen ratio statistic; MC null size;
jump power; scale invariance; degenerate inputs.
"""
import numpy as np
import tsecon
from scipy.special import gammaln
from scipy import stats

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


def my_bns(r):
    r = np.asarray(r)
    n = len(r)
    rv = np.sum(r ** 2)
    bv = (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    mu43 = 2 ** (2 / 3) * np.exp(gammaln(7 / 6) - gammaln(0.5))
    tq = n * mu43 ** -3 * np.sum(np.abs(r[2:]) ** (4 / 3) * np.abs(r[1:-1]) ** (4 / 3) * np.abs(r[:-2]) ** (4 / 3))
    theta = np.pi ** 2 / 4 + np.pi - 5
    return np.sqrt(n) * ((rv - bv) / rv) / np.sqrt(theta * max(1.0, tq / bv ** 2))


rng = np.random.default_rng(80811)
r = 0.01 * rng.standard_normal(390)
z = tsecon.bns_jump_test(r)["ratio"]
check("ratio == independent transcription (1e-12)", abs(z - my_bns(r)) < 1e-12,
      f"{z:.12f} vs {my_bns(r):.12f}")

# measure trio against the documented forms
m = tsecon.realized_measures(r)
rv_ref = np.sum(r ** 2)
bv_ref = (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
check("realized_measures rv", abs(m["rv"] - rv_ref) < 1e-15)
check("realized_measures bipower", abs(m["bipower"] - bv_ref) < 1e-15)
check("jump = max(rv-bv, 0)", m["jump"] == max(rv_ref - bv_ref, 0.0))
rq = tsecon.realized_quarticity(r)
check("realized_quarticity == (n/3) sum r^4", abs(rq - len(r) / 3 * np.sum(r ** 4)) < 1e-18)
tq = tsecon.tripower_quarticity(r)
mu43 = 2 ** (2 / 3) * np.exp(gammaln(7 / 6) - gammaln(0.5))
tq_ref = len(r) * mu43 ** -3 * np.sum(np.abs(r[2:]) ** (4 / 3) * np.abs(r[1:-1]) ** (4 / 3) * np.abs(r[:-2]) ** (4 / 3))
check("tripower_quarticity == documented form", abs(tq - tq_ref) < 1e-15 * max(1, tq_ref),
      f"{tq:.6e} vs {tq_ref:.6e}")

# scale invariance of the ratio (z is dimensionless)
z_s = tsecon.bns_jump_test(r * 1e6)["ratio"]
check("ratio scale-invariant over 1e6", abs(z_s - z) < 1e-9, f"{z_s:.12f} vs {z:.12f}")

# MC size under the null (constant-vol Gaussian, n=390 like a 1-min day)
reps = 3000
zs = np.empty(reps)
for k in range(reps):
    rr = np.random.default_rng(50_000 + k)
    zs[k] = tsecon.bns_jump_test(0.01 * rr.standard_normal(390))["ratio"]
size_5 = np.mean(zs > 1.645)
size_1 = np.mean(zs > 2.326)
print(f"[note] null MC (n=390, {reps} reps): mean {zs.mean():.3f}, sd {zs.std():.3f}, "
      f"P(z>1.645) = {size_5:.4f} (nominal 0.05), P(z>2.326) = {size_1:.4f} (nominal 0.01)")
check("null size at 5% within [0.02, 0.09]", 0.02 <= size_5 <= 0.09, f"{size_5:.4f}")

# power: inject one large jump
hits = 0
for k in range(300):
    rr = np.random.default_rng(70_000 + k)
    rj = 0.01 * rr.standard_normal(390)
    rj[200] += 0.06  # a 6-sigma-of-daily-vol jump
    if tsecon.bns_jump_test(rj)["ratio"] > 1.645:
        hits += 1
print(f"[note] power vs one 6x jump: {hits}/300")
check("power > 0.8 against a large jump", hits / 300 > 0.8, f"{hits/300:.3f}")

# degenerates
expect_raise("fewer than 3 returns", lambda: tsecon.bns_jump_test(np.array([0.01, -0.02])))
expect_raise("all zeros", lambda: tsecon.bns_jump_test(np.zeros(100)))
expect_raise("NaN", lambda: tsecon.bns_jump_test(np.array([0.01, np.nan, 0.02, 0.01])))
# one zero return inside: BV fine
z_z = tsecon.bns_jump_test(np.concatenate([r[:100], [0.0], r[100:]]))["ratio"]
check("interior zero return tolerated", np.isfinite(z_z), str(z_z))

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
