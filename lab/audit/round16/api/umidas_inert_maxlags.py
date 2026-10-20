# umidas silently ignores maxlags under se_type="nonrobust"; its new sibling adl_midas (batch 2) refuses
# the same call by name ("maxlags = 3 has no effect under se_type=\"nonrobust\""). adl_midas(ar_lags=0)
# is refused with "that model is umidas", sending users to the callable that swallows the keyword.
import sys, numpy as np, tsecon
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import midas_data
y, hf = midas_data(n=150, K=3)
a = tsecon.umidas(y, hf, se_type='nonrobust')
b = tsecon.umidas(y, hf, se_type='nonrobust', maxlags=3)
c = tsecon.umidas(y, hf, se_type='nonrobust', maxlags=10**6)
print('umidas nonrobust, maxlags=3 accepted; bse identical to no maxlags:', np.array_equal(a['bse'], b['bse']))
print('umidas nonrobust, maxlags=10**6 accepted; bse identical:', np.array_equal(a['bse'], c['bse']))
try:
    tsecon.adl_midas(y, hf, se_type='nonrobust', maxlags=3)
except ValueError as e:
    print('adl_midas refuses:', e)
try:
    tsecon.adl_midas(y, hf, ar_lags=0)
except ValueError as e:
    print('adl_midas(ar_lags=0):', e)
