"""ardl_bounds_test refuses order='3' (a str) by name, but a bytes object is
iterated as a sequence of ints: order=b'3' silently becomes order=[51]
(ord('3')), where variance_ratio refuses lags=b'3' as a string."""
import numpy as np, tsecon
rng = np.random.default_rng(0)
x = np.cumsum(rng.standard_normal((300, 1)), axis=0)
y = 0.5 * x[:, 0] + rng.standard_normal(300)
try:
    tsecon.ardl_bounds_test(y, x, lags=1, order="3")
except ValueError as e:
    print("order='3'  -> refused:", str(e)[:80])
r = tsecon.ardl_bounds_test(y, x, lags=1, order=b"3")
print("order=b'3' -> ACCEPTED, order used =", r["order"], ", nobs =", r["nobs"], ", n_params =", r["n_params"])
try:
    tsecon.variance_ratio(np.cumsum(rng.standard_normal(300)), lags=b"3")
except ValueError as e:
    print("variance_ratio(lags=b'3') -> refused:", str(e)[:80])
