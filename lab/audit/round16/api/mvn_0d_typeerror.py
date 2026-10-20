# multivariate_normality docstring/stub: "Refusals (ValueError naming `resid`): ... not 1-D or 2-D".
# A 3-D array is a ValueError, but a 0-D input (Python float, numpy scalar, 0-d array) raises TypeError,
# so `except ValueError` (what the docstring promises) does not catch it.
import numpy as np, tsecon
for v in (np.zeros((3, 10, 2)), 1.5, np.float64(1.5), np.array(1.5)):
    try:
        tsecon.multivariate_normality(v)
    except ValueError as e:
        print(f"{type(v).__name__:8s} ndim={np.ndim(v)}: ValueError (as documented)")
    except TypeError as e:
        print(f"{type(v).__name__:8s} ndim={np.ndim(v)}: TypeError (documented: ValueError) -> {str(e)[:90]}")
