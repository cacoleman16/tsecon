"""help(tsecon.ardl_bounds_test) / the stub say the surface p at a tabulated
critical value 'is off its nominal level by up to 0.001'. The model card (How to
read the output) says up to 0.0013 (90%), 0.0010 (95%), 0.0003 (99%), 0.0001
(99.9%), and ardl_properties.rs asserts < 0.0013 / 0.0011 / 0.0003 / 0.0001.
Recomputed on statsmodels' surface (which tsecon's pss.rs transcribes bit for
bit; checked here on the macrodata example) the 90% offset is 0.00128."""
import numpy as np, statsmodels.api as sm, tsecon, warnings
warnings.simplefilter("ignore")
from statsmodels.tsa.ardl import pss_critical_values as P
from statsmodels.tsa.ardl.model import _pss_pvalue
md = sm.datasets.macrodata.load_pandas().data
y = 100 * np.log(md["realcons"].values)
x = np.column_stack([100 * np.log(md["realgdp"].values), md["realint"].values])
r = tsecon.ardl_bounds_test(y, x)
print("tsecon p_lower/p_upper:", r["p_value_lower"], r["p_value_upper"])
print("statsmodels surface at k=n_x=2, case 3:", _pss_pvalue(r["statistic"], 2, 3, False), _pss_pvalue(r["statistic"], 2, 3, True))
levels = [0.10, 0.05, 0.01, 0.001]
worst = [0.0] * 4
for (k, case, i1), cv in P.crit_vals.items():
    for i, c in enumerate(cv):
        worst[i] = max(worst[i], abs(_pss_pvalue(c, k, case, i1) - levels[i]))
print("max |surface p - nominal| at the tabulated crit, 90/95/99/99.9%:", [f"{w:.5f}" for w in worst])
doc = " ".join(tsecon.ardl_bounds_test.__doc__.split())
print("docstring:", doc[doc.find("Because the surface"):doc.find("Because the surface") + 110])
