# long_run_svar(lags=0) is refused with the engine's internal argument name ("coefs"), not `lags`;
# its batch-2 sibling long_run_svar_bands and the batch-2-fixed historical_decomposition name `lags`.
import sys, numpy as np, tsecon
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import var_data
X = var_data(T=150, k=2, seed=3)
for name, f in (('long_run_svar', lambda: tsecon.long_run_svar(X, lags=0)),
                ('long_run_svar_bands', lambda: tsecon.long_run_svar_bands(X, lags=0)),
                ('historical_decomposition', lambda: tsecon.historical_decomposition(X, lags=0)),
                ('structural_fevd', lambda: tsecon.structural_fevd(X, lags=0))):
    try:
        f(); print(name, 'accepted')
    except ValueError as e:
        print(f"{name:26s} names 'lags': {'lags' in str(e)!s:5s} -> {e}")
