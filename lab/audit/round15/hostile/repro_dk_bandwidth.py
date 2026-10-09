"""panel_re / panel_fd (Driscoll-Kraay) accept any finite bandwidth, including bandwidth >= T,
and the standard errors silently collapse toward 0 (exactly 0 at 1e300) because the Bartlett
weights tend to 1 and the DK sum telescopes to (sum_t s_t)(sum_t s_t)' = 0.

Expected: refuse bandwidth + 1 > T (number of periods), as linearmodels 7.0 does ("Length of w
is larger than the number of elements in z") and as tsecon's own panel_fama_macbeth does
("needs bandwidth + 1 <= T_v").
Actual: accepted; slope t-statistic goes from ~14 (bw = 9) to ~4400 (bw = 1e6) to inf (bw = 1e300).
"""
import warnings
import numpy as np, pandas as pd, tsecon
warnings.simplefilter("ignore")
from linearmodels.panel import RandomEffects

rng = np.random.default_rng(3)
N, T = 50, 10
X = rng.standard_normal((1, N, T))
Y = 0.5 * X[0] + rng.standard_normal((N, T))
idx = pd.MultiIndex.from_product([range(N), range(T)], names=["e", "t"])
df = pd.DataFrame({"y": Y.ravel(), "x0": X[0].ravel(), "const": 1.0}, index=idx)

for bw in [9.0, 20.0, 1e6, 1e300]:
    re = tsecon.panel_re(Y, X, se_type="driscoll_kraay", bandwidth=bw)
    fd = tsecon.panel_fd(Y, X, se_type="driscoll_kraay", bandwidth=bw)
    try:
        lm = RandomEffects(df["y"], df[["const", "x0"]]).fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)
        lm_s = f"slope se={lm.std_errors['x0']:.4e}"
    except Exception as e:
        lm_s = f"REFUSED ({type(e).__name__}: {str(e)[:70]})"
    print(f"bandwidth={bw:g} (T={T}): panel_re slope se={re['bse'][-1]:.4e} t={re['tvalues'][-1]:.4g} | "
          f"panel_fd slope se={fd['bse'][-1]:.4e} t={fd['tvalues'][-1]:.4g} | linearmodels RE: {lm_s}")
try:
    tsecon.panel_fama_macbeth(Y, X, se_type="hac", bandwidth=20.0)
except ValueError as e:
    print("sibling panel_fama_macbeth(se_type='hac', bandwidth=20) refuses:", str(e)[:120])
