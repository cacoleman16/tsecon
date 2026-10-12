"""ardl_bounds_test case 2 (restricted constant): the PSS F loses accuracy with a
level offset far faster than statsmodels' F on the same data.

In case 2 the model contains a constant, so F is EXACTLY invariant to adding a
constant to y (D.y is unchanged and the constant absorbs -pi*c in both the
unrestricted and restricted regressions).  Reference values: (a) tsecon on the
centred data, (b) the restricted-vs-unrestricted SSR difference on the raw
data, (c) statsmodels UECM.bounds_test on the raw data, (d) the Wald form
evaluated with np.linalg.solve on tsecon's OWN returned params/cov.
"""
import numpy as np, tsecon, warnings
from statsmodels.tsa.ardl import UECM
warnings.filterwarnings("ignore")

rng = np.random.default_rng(69)
T, k, p, q = 100, 2, 3, 2
_ = (rng.choice([100]), rng.integers(1, 4), rng.integers(1, 4), rng.integers(1, 4))  # keep the search's draw order
x = np.cumsum(rng.standard_normal((T, k)), axis=0)
y0 = 0.5 * x.sum(1) + rng.standard_normal(T)

def ssr_diff_F(y, x):
    mod = UECM(y, p, x, {j: q for j in range(k)}, trend="c")
    Z = np.asarray(mod._x); dy = np.asarray(mod._y)
    rows = [0] + list(range(1, 2 + k))                 # const, y.L1, x1.L1, x2.L1
    keep = [j for j in range(Z.shape[1]) if j not in rows]
    ru = dy - Z @ np.linalg.lstsq(Z, dy, rcond=None)[0]
    rr = dy - Z[:, keep] @ np.linalg.lstsq(Z[:, keep], dy, rcond=None)[0]
    return ((rr @ rr - ru @ ru) / len(rows)) / ((ru @ ru) / (len(dy) - Z.shape[1])), mod

print(f"{'level L':>8} {'tsecon F':>14} {'tsecon(centred)':>16} {'SSR-diff F':>14} {'statsmodels F':>14} {'Wald solve on tsecon cov':>25} {'tsecon rel err':>15} {'sm rel err':>11}")
for L in [0.0, 1e3, 1e4, 1e5, 1e6]:
    y = 1e-2 * y0 + L            # a level L whose increments have sd ~ 0.015
    Fc = tsecon.ardl_bounds_test(y - y.mean(), x, lags=p, order=q, case=2)["statistic"]
    Fx, mod = ssr_diff_F(y, x)
    Fs = float(mod.fit().bounds_test(case=2).statistic)
    try:
        o = tsecon.ardl_bounds_test(y, x, lags=p, order=q, case=2)
        Ft = o["statistic"]
        b = np.asarray(o["params"]); V = np.asarray(o["cov"]); r = [0, 1, 2, 3]
        Fw = b[r] @ np.linalg.solve(V[np.ix_(r, r)], b[r]) / 4
        print(f"{L:8.0e} {Ft:14.8f} {Fc:16.8f} {Fx:14.8f} {Fs:14.8f} {Fw:25.8f} {abs(Ft-Fc)/Fc:15.2e} {abs(Fs-Fc)/Fc:11.2e}")
    except ValueError as ex:
        print(f"{L:8.0e} REFUSED: {str(ex)[:150]}...  (centred F = {Fc:.8f}, statsmodels F = {Fs:.6f})")
