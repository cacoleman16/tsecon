"""mypy --strict on calls the docstrings document. variance_ratio's docstring
says `lags` may be 'a list, tuple, integer array or pandas Series', and
ardl_bounds_test's `order` stub admits an integer array; variance_ratio's stub
(`int | Sequence[int] | None`) rejects the documented integer array."""
import numpy as np
import tsecon

y: np.ndarray = np.cumsum(np.ones(300))
x: np.ndarray = np.ones((300, 2))
tsecon.variance_ratio(y, lags=[2, 4, 8])                 # accepted by mypy
tsecon.variance_ratio(y, lags=np.array([2, 4, 8]))       # documented; mypy error
tsecon.ardl_bounds_test(y, x, order=np.array([1, 2]))    # same idea; accepted by mypy
