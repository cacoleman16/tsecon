"""normalize="long_run" is documented as making "the long-run diagonal
positive" (long_run_svar card and docstring; long_run_svar_bands' normalize
refusal: 'expected "long_run" (positive long-run diagonal)').  Under a permuted
`restrictions` pattern (which both callables accept) the long-run diagonal
is NOT positive: it can be negative, and an imposed zero can sit on it.  What
the rule actually signs is each shock's pivot LR[m_s, s] (m_s = the number of
long-run zeros of shock s), i.e. the first variable the shock may move in the
long run.  normalize="impact" by contrast really does sign the impact
diagonal."""
import numpy as np
import tsecon

Y = np.random.default_rng(3).standard_normal((300, 3))
zeros = [(0, 0), (1, 0), (0, 2)]   # shock 0: 2 zeros, shock 2: 1 zero, shock 1: none
r = tsecon.long_run_svar(Y, lags=1, restrictions=zeros, normalize="long_run")
LR = np.asarray(r["long_run"])
print("long_run (normalize='long_run'):\n", np.round(LR, 4))
print("long-run diagonal:", np.round(np.diag(LR), 4), "-> all positive?", bool(np.all(np.diag(LR) > 0)))
m = {0: 2, 1: 0, 2: 1}
print("pivots LR[m_s, s]:", [round(float(LR[m[s], s]), 4) for s in range(3)], "(these are the entries actually made positive)")
b = tsecon.long_run_svar_bands(Y, lags=1, restrictions=zeros, normalize="long_run", method="asymptotic")
print("long_run_svar_bands long_run_point diagonal:", np.round(np.diag(np.asarray(b["long_run_point"])), 4))
ri = tsecon.long_run_svar(Y, lags=1, restrictions=zeros, normalize="impact")
print("impact diagonal under normalize='impact':", np.round(np.diag(np.asarray(ri["impact"])), 4))
try:
    tsecon.long_run_svar_bands(Y, normalize="bogus")
except ValueError as e:
    print("refusal text:", e)
