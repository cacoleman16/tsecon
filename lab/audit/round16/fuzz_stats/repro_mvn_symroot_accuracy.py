"""multivariate_normality: the symmetric-root (JMulTi) block loses accuracy
on columns of very different units while it still accepts the input.

Reference: the same formula in 60-digit mpmath on the identical float64
input (exact answer for that input) and numpy.linalg.eigh in float64.
"""
import numpy as np, tsecon
from dhref import mjb
from mpref import mp_all
rng = np.random.default_rng(0)
n = 100
E = rng.exponential(size=(n, 2))
X0 = E @ np.array([[1.0, 0.3], [0.5, 1.0]]).T
print(f"{'scale ratio':>12} {'lam_min/tr':>10} {'tsecon relerr':>14} {'numpy relerr':>13} {'tsecon p relerr':>15}")
from scipy import stats
for logr in [0, 2, 3, 4, 5, 5.5, 5.8]:
    X = X0 * np.array([10.0 ** (logr / 2), 10.0 ** (-logr / 2)])
    ev = np.linalg.eigvalsh(np.cov(X.T, bias=True))
    try:
        r = tsecon.multivariate_normality(X)
    except ValueError as e:
        print(f"{10**logr:12.0e} {ev[0]/ev.sum():10.1e}  refused"); continue
    m = mp_all(X)["sym"]; s = mjb(X, "sym")
    exact_p = stats.chi2.sf(m[2], 4)
    print(f"{10**logr:12.0e} {ev[0]/ev.sum():10.1e} {abs(r['jarque_bera_symmetric']-m[2])/m[2]:14.2e} {abs(s['jb']-m[2])/m[2]:13.2e} {abs(r['jarque_bera_symmetric_pvalue']-exact_p)/exact_p:15.2e}")

# The same engine behind var_diagnostics' *_symmetric keys (series in units 1e5 apart):
from statsmodels.tsa.api import VAR
rng = np.random.default_rng(3)
Y = np.zeros((300, 2)); E = rng.exponential(size=(300, 2)) - 1
for t in range(1, 300): Y[t] = 0.5 * Y[t - 1] + E[t] @ np.array([[1.0, 0.3], [0.5, 1.0]]).T
for ratio in [1e3, 1e4, 1e5]:
    Ys = Y * np.array([np.sqrt(ratio), 1 / np.sqrt(ratio)])
    d = tsecon.var_diagnostics(Ys, lags=1, nlags=4)
    U = VAR(Ys).fit(1).resid
    ex = mp_all(U)["sym"][2]
    print(f"var_diagnostics units ratio {ratio:.0e}: jarque_bera_symmetric rel.err {abs(d['jarque_bera_symmetric'] - ex) / ex:.1e}  (numpy eigh {abs(mjb(U, 'sym')['jb'] - ex) / ex:.1e})")
