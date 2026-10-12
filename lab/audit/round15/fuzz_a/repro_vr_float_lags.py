"""variance_ratio: a float horizon is refused as a scalar but accepted inside
a sequence (docstring: 'a list, tuple, INTEGER array or pandas Series').
lags=4.0 raises; lags=[4.0] and lags=np.array([2.0, 4.0]) are accepted."""
import numpy as np, tsecon
y = np.cumsum(np.random.default_rng(2).standard_normal(100))
for L in (4.0, [4.0], np.array([2.0, 4.0]), [2, 4.0]):
    try:
        o = tsecon.variance_ratio(y, lags=L)
        print(f"lags={L!r:24} accepted -> lags echo {o['lags']}, vr {o['vr']}")
    except ValueError as ex:
        print(f"lags={L!r:24} REFUSED: {str(ex)[:90]}")
