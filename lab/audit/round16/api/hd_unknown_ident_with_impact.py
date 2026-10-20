# historical_decomposition: an unknown `identification` spelling combined with `impact=` is blamed on
# `impact` (and described as a mode that "identifies its own shocks") instead of being refused as an
# unknown identification.
import sys, numpy as np, tsecon
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import var_data
X = var_data(T=80, k=2, seed=3)
imp = tsecon.long_run_svar(X, lags=2)['impact']
for ident in ("Impact", "impacts", "long_run", "chol"):
    print(f"--- identification={ident!r}, impact=A0")
    try:
        tsecon.historical_decomposition(X, lags=2, identification=ident, impact=imp)
    except Exception as e:
        print("   actual:", type(e).__name__, e)
    print(f"--- identification={ident!r}, no impact")
    try:
        tsecon.historical_decomposition(X, lags=2, identification=ident)
    except Exception as e:
        print("   expected-style:", type(e).__name__, e)
