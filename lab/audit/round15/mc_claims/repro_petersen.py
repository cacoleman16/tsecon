"""REPRO (finding Petersen-table): the panel card / CHANGELOG Petersen (2009) table is quoted to three
decimals from 1000 replications per cell (MC se ~0.007 at 0.95, ~0.015 at 0.6); two cells do not
reproduce. Design copied from bindings/python/tests/test_panel_static.py::_petersen_coverage, fresh seed.
Runtime ~25 s."""
import math, numpy as np, tsecon
R, n = 20000, 100
claims = {(10, True): (0.627, 0.499, 0.956), (10, False): (0.930, 0.805, 0.949)}
for (t, firm), cl in claims.items():
    rng = np.random.default_rng([424242, t, int(firm)]); hits = np.zeros(3)
    for _ in range(R):
        fx = rng.standard_normal((n, 1)) if firm else 0.0; fe = rng.standard_normal((n, 1)) if firm else 0.0
        x = fx + rng.standard_normal((n, t)); y = x + fe + rng.standard_normal((n, t))
        rs = (tsecon.panel_fama_macbeth(y, x[None]), tsecon.panel_fama_macbeth(y, x[None], se_type="hac", bandwidth=4.0), tsecon.panel_re(y, x[None]))
        for j, r in enumerate(rs): hits[j] += abs(r["params"][1] - 1.0) <= 1.959964 * r["bse"][1]
    c = hits / R; se = np.sqrt(c * (1 - c) / R)
    for j, nm in enumerate(("panel_fama_macbeth", "fama_macbeth hac(4)", "panel_re cluster")):
        print(f"T={t} firm={firm} {nm:20s}: claimed {cl[j]:.3f}  measured {c[j]:.4f} (se {se[j]:.4f})  z = {(cl[j]-c[j])/se[j]:+.1f}")
