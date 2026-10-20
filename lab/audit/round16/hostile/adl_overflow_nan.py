"""adl_midas refuses non-finite INPUTS but returns non-finite OUTPUTS without
a word when finite inputs overflow: the lagged-y block puts y into the
design, so the default HAC meat (x_t u_t)^2 ~ y^4 overflows at |y| ~ 1e77 and
every bse/tvalue comes back NaN; at larger scales params, rsquared and
residuals are NaN as well, and some scales are refused as 'collinear columns'
(X'X numerically singular), which they are not. dfm_smooth and
multivariate_normality refuse their overflow cases by name; adl_midas has no
such guard (statsmodels OLS/HAC also returns NaN here)."""
import numpy as np, tsecon
rng = np.random.default_rng(12345)
y = rng.standard_normal(120).cumsum() * 0.1 + rng.standard_normal(120)
HF = rng.standard_normal((120, 3))
def show(label, f):
    try:
        r = f()
        print(f"{label}: RETURNED  params finite={np.isfinite(r['params']).all()}  "
              f"bse finite={np.isfinite(r['bse']).all()}  rsquared={r['rsquared']}")
    except ValueError as e:
        print(f"{label}: refused: {str(e)[:120]}")
for s in [1e76, 1e77, 1e100, 1e154, 1e155, 1e300]:
    show(f"adl_midas(y*{s:.0e}, hf_lags)", lambda: tsecon.adl_midas(y * s, HF))
for s in [1e154, 1e155, 1e300]:
    show(f"adl_midas(y, hf_lags*{s:.0e})", lambda: tsecon.adl_midas(y, HF * s))
