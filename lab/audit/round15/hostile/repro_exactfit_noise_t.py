"""panel_fama_macbeth and panel_between return finite, ordinary-looking t-statistics computed
from pure rounding noise when the outcome is fit exactly, where panel_re refuses the exact fit
by name and linearmodels' FamaMacBeth raises.

Case A: outcome constant within every period (a market-wide variable regressed on firm
characteristics): every cross-section is an exact fit, the per-period slopes are exactly 0 in
exact arithmetic (here ~1e-18) and so is their dispersion, so the slope t-stat is 0/0.
Case B: an entity-demeaned outcome passed to panel_between: every entity mean is 0 (to
rounding), so the between regression is of zeros on the regressor means.

Expected: a named refusal (as panel_re: "outcome: the within residuals vanish ..."), or NaN.
Actual:   slope t = -1.20 / -0.53 (FM) and -0.28 / 0.92 / 0.76 (between): order-1 noise that reads as
          an ordinary insignificant result.
"""
import numpy as np, tsecon
rng = np.random.default_rng(7)
N, T = 50, 30
X = rng.standard_normal((2, N, T))
Ya = np.repeat(rng.standard_normal(T)[None, :], N, axis=0)
r = tsecon.panel_fama_macbeth(Ya, X)
print("A panel_fama_macbeth slopes", r["params"][1:], "bse", r["bse"][1:], "t", r["tvalues"][1:])
try:
    import pandas as pd
    from linearmodels.panel import FamaMacBeth
    idx = pd.MultiIndex.from_product([range(N), range(T)], names=["e", "t"])
    df = pd.DataFrame({"y": Ya.ravel(), "x0": X[0].ravel(), "x1": X[1].ravel(), "const": 1.0}, index=idx)
    FamaMacBeth(df["y"], df[["const", "x0", "x1"]]).fit()
    print("  linearmodels FamaMacBeth: returned")
except Exception as e:
    print("  linearmodels FamaMacBeth: raised", type(e).__name__, e)
Y0 = rng.standard_normal((N, T))
Yb = Y0 - Y0.mean(axis=1, keepdims=True)
r = tsecon.panel_between(Yb, X)
print("B panel_between params", r["params"], "bse", r["bse"], "t", r["tvalues"])
Yc = 1.0 + 2.0 * X[0] - 0.5 * X[1]          # exact linear fit: panel_re refuses it by name
try:
    tsecon.panel_re(Yc, X)
except ValueError as e:
    print("panel_re on an exact fit refuses:", str(e)[:110])
