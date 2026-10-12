"""adl_midas solves the OLS normal equations by Cholesky (tsecon-hac ols.rs),
squaring the design's condition number. The lagged-y columns that adl_midas
adds are nearly collinear with the constant whenever the target is a level
with a small spread around a large mean (an index near 100, a log level), so
the estimates lose digits far beyond the documented 1e-10 agreement with
statsmodels OLS (which uses a pinv/QR solve).  Exact reference: 50-digit
mpmath normal equations on the same float64 design.
"""
import sys, os, numpy as np, tsecon, warnings
import statsmodels.api as sm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylib"))
import mpmath as mp
mp.mp.dps = 50
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n, K, P = 200, 3, 2
x = rng.standard_normal((n, K))
for level, spread in [(0.0, 1.0), (100.0, 1.0), (100.0, 0.1), (100.0, 0.01), (1000.0, 0.01)]:
    u = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(2, n):
        u[t] = 0.6 * u[t-1] - 0.2 * u[t-2] + 0.3 * x[t, 0] + e[t]
    y = level + spread * u
    r = tsecon.adl_midas(y, x, ar_lags=P, se_type="nonrobust")
    X = np.column_stack([np.ones(n - P), y[P-1:n-1], y[P-2:n-2], x[P:]])
    yy = y[P:]
    f = sm.OLS(yy, X).fit()
    Xm = mp.matrix(X.tolist()); ym = mp.matrix(yy.tolist())
    b = mp.lu_solve(Xm.T * Xm, Xm.T * ym)
    bex = np.array([float(v) for v in b])
    rel = lambda a: float(np.max(np.abs(np.asarray(a) - bex) / np.abs(bex)))
    print(f"y = {level:6.0f} + {spread:5.2f}*AR(2): cond(X)={np.linalg.cond(X):.1e}  "
          f"params rel.err tsecon {rel(r['params']):.1e}  statsmodels {rel(f.params):.1e}  "
          f"|tsecon - statsmodels| bse rel {np.max(np.abs(np.asarray(r['bse']) - f.bse) / f.bse):.1e}")

# A realistic target: quarterly log GDP level (~9.5, growth 0.5%/q), monthly indicators as hf_lags.
rng = np.random.default_rng(3)
n = 120
for P in [1, 2, 4]:
    x = rng.standard_normal((n, 3)) * 0.01
    y = 9.5 + np.cumsum(0.005 + 0.3 * x[:, 0] + 0.008 * rng.standard_normal(n))
    r = tsecon.adl_midas(y, x, ar_lags=P)
    X = np.column_stack([np.ones(n - P)] + [y[P - p:n - p] for p in range(1, P + 1)] + [x[P:]])
    yy = y[P:]
    Xm = mp.matrix(X.tolist()); ym = mp.matrix(yy.tolist())
    bex = np.array([float(v) for v in mp.lu_solve(Xm.T * Xm, Xm.T * ym)])
    f = sm.OLS(yy, X).fit()
    print(f"log-GDP level, ar_lags={P}: params rel.err tsecon {np.max(np.abs(np.asarray(r['params']) - bex) / np.abs(bex)):.1e}"
          f"  statsmodels {np.max(np.abs(f.params - bex) / np.abs(bex)):.1e}   (documented: 1e-10 vs statsmodels)")
