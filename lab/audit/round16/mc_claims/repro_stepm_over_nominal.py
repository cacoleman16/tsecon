"""Repro: the forecasting card says that under dependence at n = 200 StepM's FWER
"runs slightly over a nominal 0.10 (0.118-0.120 in arch on lfc, mixed and stepdown)"
(arch 1000 reps, seeds 880000 + r). On 1500 FRESH panels of the same designs, arch's
own StepM and tsecon's un-studentized StepM sit AT nominal (0.095-0.106)."""
import math, warnings, numpy as np, tsecon
from arch.bootstrap import StepM
warnings.filterwarnings("ignore")
MC = 1500
DES = {"lfc": [1.0] * 6, "mixed": [1.0, 0.6, 0.8, 1.0, 1.0, 1.5], "stepdown": [1.0, 0.5, 0.88, 1.0, 1.0, 1.0]}
CLAIM = {"lfc": 0.119, "mixed": 0.118, "stepdown": 0.120}
def ar1(rng, n, rho=0.5):
    e = rng.standard_normal(n); x = np.empty(n); p = 0.0; s = math.sqrt(1 - rho * rho)
    for t in range(n):
        p = rho * p + s * e[t]; x[t] = p
    return x
def panel(rng, scales, n=200):
    u = ar1(rng, n)
    return np.column_stack([(0.7 * u + s * ar1(rng, n)) ** 2 for s in scales])
for lab, sc in DES.items():
    tn = np.array([sc[0] ** 2 - s ** 2 <= 0 for s in sc[1:]])
    fa = ft = 0
    for r in range(MC):
        p = panel(np.random.default_rng(0xA16F0000 + 7919 * r), sc)
        b, m = p[:, 0], p[:, 1:]
        sm = StepM(b, m, size=0.10, block_size=8, reps=300, bootstrap="stationary", studentize=False, seed=0xA2C4 + r)
        try:
            sm.compute(); sa = list(sm.superior_models)
        except ValueError:
            sa = list(range(5))
        st = tsecon.stepm_test(b, m, size=0.10, block_size=8, reps=300, bootstrap="stationary",
                               studentize=False, seed=0x5EED + r)["superior_models"]
        fa += bool(tn[list(sa)].any()) if len(sa) else 0
        ft += bool(tn[list(st)].any()) if len(st) else 0
    pa, pt = fa / MC, ft / MC
    se = math.sqrt(pa * (1 - pa) / MC)
    print(f"{lab:9s} size 0.10: arch FWER {pa:.4f} (MC se {se:.4f}), tsecon {pt:.4f}; card's arch claim "
          f"{CLAIM[lab]} -> {(CLAIM[lab] - pa) / se:+.1f} fresh se; nominal 0.10 -> {(0.10 - pa) / se:+.1f} se")
