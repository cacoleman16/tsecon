"""A bootstrap pseudo-sample whose VAR refit has a singular residual
covariance aborts the whole long_run_svar_bands call with the full-sample,
data-blaming message ("two columns are exact linear combinations - a
duplicated series ... Drop the redundant column and refit") and no
replication index.  The data are fine: long_run_svar and the asymptotic arm
succeed on the same data, and the docstring promises that a failing
replication "is reported, by index, as an error"."""
import numpy as np
import tsecon

k, p = 2, 1
T = 6                                   # T_eff - m = 5 - 3 = 2 = k (admissible)
Y = np.random.default_rng(1).standard_normal((T, k))
print("long_run_svar impact:", np.round(np.asarray(tsecon.long_run_svar(Y, lags=p, horizon=3)["impact"]), 4).tolist())
print("var_fit sigma_u:", np.round(np.asarray(tsecon.var_fit(Y, lags=p)["sigma_u"]), 4).tolist())
print("asymptotic arm max se:", float(np.max(tsecon.long_run_svar_bands(Y, lags=p, horizon=3, method="asymptotic")["se"])))
try:
    tsecon.long_run_svar_bands(Y, lags=p, horizon=3, n_boot=50, seed=1)
    print("bootstrap arm: OK")
except Exception as e:
    print("bootstrap arm:", type(e).__name__, e)
    print("names a replication index:", "replication" in str(e))
