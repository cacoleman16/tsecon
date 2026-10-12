"""REPRO (finding VR-size): the variance_ratio size ranges quoted in CHANGELOG [Unreleased] and the
unit-root-cointegration-tests card come from 600 replications (MC se ~0.009) and do not reproduce.
Same DGPs as scripts/mc_variance_ratio_size.py (n=500, q=2/4/8/16, GARCH(1,1) .05/.15/.80), fresh seed.
Runtime ~10 s."""
import numpy as np, tsecon, math
REPS, N, QS = 20000, 500, [2, 4, 8, 16]
rng = np.random.default_rng(77001)
def garch():
    e = np.empty(N - 1); s2 = 1.0; z = rng.standard_normal(N - 1)
    for t in range(N - 1):
        e[t] = math.sqrt(s2) * z[t]; s2 = 0.05 + 0.15 * e[t] ** 2 + 0.80 * s2
    return e
claims = {("iid", True): ([.042, .043, .042, .063], .028), ("iid", False): ([.045, .038, .043, .062], .023),
          ("garch", True): ([.042, .052, .037, .038], .040), ("garch", False): ([.113, .113, .093, .085], .093)}
for dgp in ("iid", "garch"):
    rej = {True: np.zeros(4), False: np.zeros(4)}; cd = {True: 0, False: 0}
    for _ in range(REPS):
        y = np.concatenate([[0.0], np.cumsum(rng.standard_normal(N - 1) if dgp == "iid" else garch())])
        for rob in (True, False):
            r = tsecon.variance_ratio(y, QS, robust=rob); rej[rob] += np.asarray(r["pvalue"]) < .05; cd[rob] += r["cd_pvalue"] < .05
    for rob in (True, False):
        m = rej[rob] / REPS; mc = cd[rob] / REPS; cq, ccd = claims[(dgp, rob)]
        se = np.sqrt(m * (1 - m) / REPS); sec = math.sqrt(mc * (1 - mc) / REPS)
        print(f"{dgp:5s} {'robust z*' if rob else 'homosk z '}: per-q claimed {cq} measured {np.round(m, 4).tolist()} "
              f"(z = {np.round((np.array(cq) - m) / se, 1).tolist()});  Chow-Denning claimed {ccd} measured {mc:.4f} (z = {(ccd - mc) / sec:.1f})")
print("CHANGELOG: 'robust z* holds 3.7-5.2% size under GARCH ... against 8.5-11.3% for the homoskedastic z;"
      " the Chow-Denning bound is conservative (2.3-4.0% ...)'")
