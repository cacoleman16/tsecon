"""The runnable examples of the structural-identification model card's
`long_run_svar_bands` and `historical_decomposition(identification="impact")`
sections, verbatim, and their printed output.

    python lab/bq_bands/card_examples.py > lab/bq_bands/out/card_examples.txt
"""
print("# --- long_run_svar_bands ---")
import numpy as np, tsecon

# Blanchard-Quah bivariate: output growth (dy) and unemployment (u).
rng = np.random.default_rng(0)
T = 400
es = rng.standard_normal(T)   # supply (permanent)
ed = rng.standard_normal(T)   # demand (transitory)
dy = np.zeros(T); u = np.zeros(T)
for t in range(2, T):
    dy[t] = 0.2 * dy[t - 1] + es[t] + 0.5 * ed[t] - 0.5 * ed[t - 1]
    u[t] = 0.6 * u[t - 1] - 0.3 * es[t] + 0.7 * ed[t]
data = np.column_stack([dy, u])

kw = dict(lags=4, horizon=20, impact_signs=[(0, 1)], cumulative=True)
efron = tsecon.long_run_svar_bands(data, n_boot=999, seed=1, **kw)
kilian = tsecon.long_run_svar_bands(data, n_boot=999, seed=1, bias_correct=True, **kw)
delta = tsecon.long_run_svar_bands(data, method="asymptotic", **kw)

for name, r in (("Efron", efron), ("Efron+Kilian", kilian), ("delta", delta)):
    lo, up = np.asarray(r["lower"]), np.asarray(r["upper"])
    cells = ", ".join(f"h={h}: [{lo[h, 0, 1]:+.3f}, {up[h, 0, 1]:+.3f}]" for h in (0, 4, 20))
    print(f"{name:>12}  output level <- demand, 90%: {cells}")
print("imposed long-run zero, every arm: se", [r["long_run_se"][0][1] for r in (efron, kilian, delta)],
      "band", [(r["long_run_lower"][0][1], r["long_run_upper"][0][1]) for r in (efron, kilian, delta)])
print("min_root", round(efron["min_root"], 3), " n_unstable_draws", efron["n_unstable_draws"],
      " sign_anchor_mismatches", efron["sign_anchor_mismatches"])

print()
print("# --- historical_decomposition(identification='impact') ---")
B = tsecon.long_run_svar(data, lags=4, impact_signs=[(0, 1)])["impact"]   # a nested list
hd = tsecon.historical_decomposition(data, lags=4, identification="impact", impact=B)
contrib, base = np.asarray(hd["hd"]), np.asarray(hd["baseline"])
print("adding-up  max|y - baseline - sum_j hd|:",
      np.max(np.abs(data[4:] - (base + contrib.sum(axis=2)))))
print("shock_covariance:\n", np.round(np.asarray(hd["shock_covariance"]), 6))
T_eff, m = data.shape[0] - 4, 1 + 2 * 4
print("(T_eff - m) / T_eff =", round((T_eff - m) / T_eff, 6))
