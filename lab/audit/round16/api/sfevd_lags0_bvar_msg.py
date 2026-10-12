# structural_fevd(lags=0) is refused as "a BVAR needs at least one lag", but structural_fevd fits an OLS VAR
# (its siblings say "the decomposition needs a VAR" / "a VAR(0) has no dynamics").
import sys, numpy as np, tsecon
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import var_data
X = var_data(T=150, k=2, seed=3)
try:
    tsecon.structural_fevd(X, lags=0)
except ValueError as e:
    print("structural_fevd:", e)
try:
    tsecon.historical_decomposition(X, lags=0)
except ValueError as e:
    print("historical_decomposition:", e)
