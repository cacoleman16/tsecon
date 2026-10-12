# var_irf_bands silently ignores keywords that its engine-sharing sibling long_run_svar_bands refuses
# as inert: n_boot / seed under method="asymptotic", band_scope under band="pointwise".
# long_run_svar_bands' docstring discloses only the band_seed/band_n_sim difference.
import sys, numpy as np, tsecon
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import var_data
X = var_data(T=150, k=2, seed=3)
base = tsecon.var_irf_bands(X, lags=2, horizon=4, method='asymptotic')
for kw in (dict(n_boot=5000), dict(seed=7), dict(band_scope='all')):
    r = tsecon.var_irf_bands(X, lags=2, horizon=4, method='asymptotic', **kw)
    print(f"var_irf_bands(method='asymptotic', {kw}) accepted; output identical to default: {r == base}; echoed n_boot={r['n_boot']}")
    try:
        tsecon.long_run_svar_bands(X, lags=2, horizon=4, method='asymptotic', **kw)
        print('   long_run_svar_bands accepted')
    except ValueError as e:
        print('   long_run_svar_bands refuses:', str(e)[:110])
bb = tsecon.var_irf_bands(X, lags=2, horizon=4, method='bootstrap', n_boot=40)
r = tsecon.var_irf_bands(X, lags=2, horizon=4, method='bootstrap', n_boot=40, band_scope='shock')
print("var_irf_bands(method='bootstrap', band='pointwise', band_scope='shock') identical to default:", r == bb)
doc = tsecon.long_run_svar_bands.__doc__
print("lrsb docstring mentions var_irf_bands' silent n_boot/seed on the asymptotic arm:", 'var_irf_bands` accepts' in doc and 'n_boot' in doc.split('var_irf_bands` accepts')[1][:80])
