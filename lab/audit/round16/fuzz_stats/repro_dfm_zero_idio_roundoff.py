"""dfm_smooth accepts idiosyncratic variances of exactly 0 (documented ">= 0").
One factor, three series; series 1 and 2 have zero idiosyncratic variance and
are observed together, so series 1 pins the factor exactly and series 2 is
redundant (its data equal loading * factor up to roundoff).  Its prediction
variance F is then roundoff dust, and the univariate filter's
numerically-singular skip test does not catch it: `loglik` gains
-0.5*(ln F_dust + v^2/F_dust) ~ +20 points.  The smoothed factor is unchanged.
Reference: the same model with the redundant cell set to NaN (the Gaussian
density on its support) -- the value tsecon itself returns for other draws.
"""
import numpy as np, tsecon, warnings
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
warnings.filterwarnings("ignore")
z = np.array([[0.03666137558761732, 0.5527422219007221, -0.04042486581736605],
              [np.nan, np.nan, np.nan]])
lam = np.array([[-0.43272912833939525], [0.7196862105052038], [-0.05263433357820673]])
A = [[-0.8642299294783886, -0.8484056219079968]]
Q = [[0.13302995088165312]]
idio = [0.8355556816599208, 0.0, 0.0]
print("series 2 / loading vs series 1 / loading (the same factor):", z[0, 2] / lam[2, 0], z[0, 1] / lam[1, 0])
full = tsecon.dfm_smooth(z, lam, A, idio, factor_cov=Q)
zr = z.copy(); zr[0, 2] = np.nan
red = tsecon.dfm_smooth(zr, lam, A, idio, factor_cov=Q)
print(f"tsecon loglik, all cells:              {full['loglik']:.6f}")
print(f"tsecon loglik, redundant cell dropped: {red['loglik']:.6f}   (expected the same value)")
print("smoothed factor at t=0:", full["smoothed_factors"][0][0], red["smoothed_factors"][0][0])
# statsmodels, identity-normalised factor (g = f / sqrt(Q))
q = np.sqrt(Q[0][0]); p2 = np.r_[(lam * q).ravel(), idio, A[0][0], A[0][1]]
for name, zz in [("all cells", z), ("redundant dropped", zr)]:
    print(f"statsmodels DynamicFactor loglik, {name}: {DynamicFactor(zz, k_factors=1, factor_order=2, error_order=0).smooth(p2, cov_type='none').llf:.6f}")
