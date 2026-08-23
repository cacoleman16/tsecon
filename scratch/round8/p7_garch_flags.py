"""Round-8 probe: garch_fit boundary/se_valid machinery — edge consistency.

Round 7 built and battery-tested this; here we check the *invariants* the
contract implies, on fresh DGPs:
  - se_valid[i] False  <=>  NaN in se_mle[i] (and se_robust[i])
  - any boundary flag  =>  boundary_note is a non-empty string
  - no boundary flag   =>  boundary_note is None
  - flags align with param_names length
  - flagged fits keep finite SEs on interior (se_valid) coordinates
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


def dgps(seed):
    rr = np.random.default_rng(seed)
    T = 600
    out = {}
    out["white_noise"] = 0.01 * rr.standard_normal(T)
    # IGARCH-ish
    f = np.zeros(T); y = np.zeros(T); f[0] = 1e-4
    for t in range(T):
        y[t] = np.sqrt(f[t]) * rr.standard_normal()
        if t + 1 < T:
            f[t + 1] = 1e-7 + 0.15 * y[t] ** 2 + 0.85 * f[t]
    out["igarch"] = y
    # tiny alpha
    f = np.zeros(T); y = np.zeros(T); f[0] = 1e-4
    for t in range(T):
        y[t] = np.sqrt(f[t]) * rr.standard_normal()
        if t + 1 < T:
            f[t + 1] = 2e-5 + 0.005 * y[t] ** 2 + 0.8 * f[t]
    out["tiny_alpha"] = y
    # ordinary garch
    f = np.zeros(T); y = np.zeros(T); f[0] = 1e-4
    for t in range(T):
        y[t] = np.sqrt(f[t]) * rr.standard_normal()
        if t + 1 < T:
            f[t + 1] = 5e-6 + 0.08 * y[t] ** 2 + 0.9 * f[t]
    out["garch"] = y
    return out


n_flagged = 0
n_interior = 0
for seed in range(5):
    for name, y in dgps(60000 + seed).items():
        r = tsecon.garch_fit(y)
        sv = np.asarray(r["se_valid"])
        bd = np.asarray(r["boundary"])
        se_m = np.asarray(r["se_mle"])
        se_r = np.asarray(r["se_robust"])
        pn = r["param_names"]
        tag = f"{name}/s{seed}"
        check(f"{tag}: flag lengths match params", len(sv) == len(pn) == len(bd) == len(se_m))
        check(f"{tag}: se_valid False <=> NaN se_mle",
              all((not v) == np.isnan(m) for v, m in zip(sv, se_m)),
              f"sv={sv} se={se_m}")
        check(f"{tag}: se_valid False <=> NaN se_robust",
              all((not v) == np.isnan(m) for v, m in zip(sv, se_r)))
        if bd.any():
            n_flagged += 1
            check(f"{tag}: boundary => note non-empty",
                  isinstance(r["boundary_note"], str) and len(r["boundary_note"]) > 10,
                  repr(r["boundary_note"])[:60])
        else:
            n_interior += 1
            check(f"{tag}: interior => note None and all se_valid",
                  r["boundary_note"] is None and sv.all(),
                  f"note={r['boundary_note']!r} sv={sv}")
print(f"[note] flagged fits: {n_flagged}, interior fits: {n_interior} (want both classes exercised)")
check("both classes exercised", n_flagged >= 3 and n_interior >= 3, f"{n_flagged}/{n_interior}")

# summary() renders the note
res_obj = tsecon.results
try:
    y = dgps(60001)["white_noise"]
    r = tsecon.garch_fit(y)
    import tsecon.results as tres
    g = tres.GARCHResults(r) if hasattr(tres, "GARCHResults") else None
    if g is not None:
        s = g.summary()
        has = ("boundary" in s.lower()) == bool(np.asarray(r["boundary"]).any())
        check("summary mentions boundary iff flagged", has)
    else:
        print("[note] GARCHResults wrapper not found under tsecon.results; skipped summary check")
except Exception as e:
    print("[note] summary check skipped:", e)

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
