# structural_fevd: `sigma` is inert once `impact` is given, and is not even validated then.
# Without impact an unknown sigma is refused by name; with impact any string is accepted silently.
# historical_decomposition (same `impact` array-likes, same batch) refuses inert keywords instead.
import sys, numpy as np, tsecon
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import var_data
X = var_data(T=120, k=2, seed=3)
imp = tsecon.long_run_svar(X, lags=2)['impact']
try:
    tsecon.structural_fevd(X, lags=2, horizon=4, sigma='bogus')
    print("no impact, sigma='bogus': accepted")
except Exception as e:
    print("no impact, sigma='bogus': refused ->", type(e).__name__, e)
a = tsecon.structural_fevd(X, lags=2, horizon=4, impact=imp)
for s in ('mle', 'bogus', ''):
    try:
        b = tsecon.structural_fevd(X, lags=2, horizon=4, impact=imp, sigma=s)
        print(f"impact given, sigma={s!r}: ACCEPTED silently; output identical to default: {a == b}")
    except Exception as e:
        print(f"impact given, sigma={s!r}: refused ->", e)
print("expected: sigma refused as inert (or at least validated) when impact is given, as historical_decomposition refuses inert keywords")
