"""multivariate_normality refuses the WHOLE call (including the scale-invariant,
recommended Doornik-Hansen and Cholesky statistics) when only the unit-dependent
symmetric block is unreliable; var_diagnostics on the same residuals returns DH
and sets only the *_symmetric keys to None."""
import numpy as np, tsecon
from statsmodels.tsa.api import VAR
from dhref import doornik_hansen
rng = np.random.default_rng(1)
Y = np.cumsum(rng.standard_normal((200, 2)), 0) * 0 + rng.standard_normal((200, 2))
Y = Y * np.array([1e4, 1e-3])   # e.g. one series in dollars, one as a fraction
d = tsecon.var_diagnostics(Y, lags=1, nlags=4)
print("var_diagnostics: doornik_hansen =", d["doornik_hansen"], " jarque_bera_symmetric =", d["jarque_bera_symmetric"])
U = VAR(Y).fit(1).resid
print("reference DH on the same residuals:", doornik_hansen(U)["doornik_hansen"])
try:
    r = tsecon.multivariate_normality(U)
    print("multivariate_normality: doornik_hansen =", r["doornik_hansen"])
except ValueError as e:
    print("multivariate_normality REFUSED:", str(e)[:160], "...")

# The docstring: DH is "Invariant to column order, rescaling, sign flips and shifts".
X = rng.standard_exponential((100, 2)) @ np.array([[1.0, 0.4], [0.2, 1.0]])
a = tsecon.multivariate_normality(X)
print("multivariate_normality(X): doornik_hansen =", a["doornik_hansen"])
try:
    b = tsecon.multivariate_normality(X * np.array([1e4, 1e-3]))
    print("rescaled:", b["doornik_hansen"])
except ValueError as e:
    print("multivariate_normality(X * [1e4, 1e-3]) REFUSED; reference DH on the rescaled data =",
          doornik_hansen(X * np.array([1e4, 1e-3]))["doornik_hansen"])
