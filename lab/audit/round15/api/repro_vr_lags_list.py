"""variance_ratio with a sequence of horizons: the stub docstring and the model
card say every per-horizon key (stub: explicitly including `lags`) is a 1-D array;
`lags` comes back as a Python list, so array idioms on it break."""
import numpy as np, tsecon
rng = np.random.default_rng(1)
y = np.cumsum(rng.standard_normal(500))
r = tsecon.variance_ratio(y, lags=[2, 4, 8, 16])
for k in ("lags", "vr", "stat", "pvalue", "nobs", "n_dropped"):
    print(f"{k:10s} {type(r[k]).__name__}")
mask = r["pvalue"] < 0.99
print("vr[mask]  :", r["vr"][mask])
try:
    print("lags[mask]:", r["lags"][mask])
except TypeError as e:
    print("lags[mask]: TypeError:", e)
print("lags * 2  :", r["lags"] * 2, " (list repetition, not [4, 8, 16, 32])")
