"""Cost of the joint forecast-error covariance behind var_forecast's
simultaneous band: its cross-horizon blocks are steps (steps^2 - 1) / 6
products of k x k matrices, CUBIC in steps. Prints the wall time of a
Sidak band (no simulation) on a bivariate VAR(1) and the work count the
2^31 budget checks (each product counted at k^3 + 32)."""
import math
import time

import numpy as np
import tsecon

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 2))
k = 2
for steps in (100, 200, 400, 685):
    t = time.perf_counter()
    tsecon.var_forecast(X, lags=1, steps=steps, band="sidak")
    work = steps**3 // 6 * (k**3 + 32)
    print(f"steps={steps}: {time.perf_counter() - t:.2f} s, work {work:.3g} = 2^{math.log2(work):.1f}", flush=True)
try:
    tsecon.var_forecast(X, lags=1, steps=686, band="sidak")
except ValueError as e:
    print("steps=686:", str(e)[-80:])
