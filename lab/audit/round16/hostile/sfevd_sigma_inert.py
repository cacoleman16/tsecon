"""structural_fevd(impact=..., sigma=...): when impact is given, sigma is
never read -- not even validated -- so sigma='mle' is silently inert and
sigma='bogus' is silently accepted. Neighbouring batch-2 callables refuse
inert keywords (historical_decomposition's sampling controls,
long_run_svar_bands' n_boot/seed/band_seed...), and structural_fevd refuses
the same bogus sigma when impact is None."""
import numpy as np, tsecon
rng = np.random.default_rng(1); X = np.zeros((200, 2))
for t in range(1, 200): X[t] = 0.5 * X[t-1] + rng.standard_normal(2)
A0 = tsecon.long_run_svar(X)["impact"]          # a nested list: the batch-2 array-like path
a = tsecon.structural_fevd(X, impact=A0, sigma="dfadj")
b = tsecon.structural_fevd(X, impact=A0, sigma="mle")
print("impact given, sigma='mle' vs 'dfadj': outputs identical (inert):", a == b)
try:
    tsecon.structural_fevd(X, impact=A0, sigma="bogus"); print("impact given, sigma='bogus': ACCEPTED silently")
except ValueError as e:
    print("impact given, sigma='bogus': refused:", e)
try:
    tsecon.structural_fevd(X, sigma="bogus"); print("impact None, sigma='bogus': accepted")
except ValueError as e:
    print("impact None, sigma='bogus': refused:", e)
