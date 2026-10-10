"""var_conditional_forecast(soft="signal"): an sd whose square underflows to 0
silently turns the soft cell into a HARD (pinned) one, while the output still
reports it as soft (n_soft, condition_sd, mode="signal").  The overflow side
(sd > ~1.3e154) is refused naming condition_sd[h][j]; the underflow side is not.
Docstring: a signal-mode soft cell "is NOT pinned and its se is below sd",
"A soft cell far tighter than its forecast se keeps a correct se (sd 1e-9 gives se 1.0e-9)".
"""
import numpy as np, tsecon
rng = np.random.default_rng(7)
T, k = 120, 3
Y = np.zeros((T, k)); E = rng.standard_normal((T, k))
for t in range(2, T): Y[t] = 0.5 * Y[t-1] - 0.2 * Y[t-2] + E[t]
cond = [[0.8, None, None]]
for sd in [1e-150, 1e-160, 1e-162, 1e-170, 1e-200]:
    r = tsecon.var_conditional_forecast(Y, cond, lags=2, steps=1, condition_sd=[[sd, None, None]], soft="signal")
    print(f"sd={sd:.0e}: se[0][0]={r['se'][0][0]:.3e} (expected ~{sd:.0e}, below sd but > 0)  "
          f"point==value exactly: {r['point'][0][0] == 0.8}  n_soft={r['n_soft']} mode={r['mode']} condition_sd={r['condition_sd'][0][0]:.0e}")
