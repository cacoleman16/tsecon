"""breusch_godfrey_test: a design whose 'constant' column carries a tiny
perturbation (1 + 1e-8 * noise, e.g. a ratio that should be exactly 1, or
complementary dummies with noise) falls in a band (~2.5e-7 .. 1.6e-10) where
tsecon decides x spans no constant, adds a column of ones (constant_added),
and then refuses the near-collinear auxiliary design with a message that
blames the LAGGED RESIDUALS and tells the user to pass a smaller nlags.
Smaller nlags cannot help (nlags=1 is refused too); the collinear column is
the constant tsecon itself added.  statsmodels returns numbers (it adds the
constant too), and durbin_watson_test accepts the same x.
"""
import numpy as np, tsecon, warnings
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
warnings.filterwarnings("ignore")
rng = np.random.default_rng(5); T = 60
z = rng.standard_normal(T); y = rng.standard_normal(T); noise = rng.standard_normal(T)
for eps in [1e-6, 1e-8, 1e-12]:
    X = np.column_stack([1 + eps * noise, z])
    r = sm.OLS(y, X).fit()
    s = acorr_breusch_godfrey(r, nlags=1, result_object=False)
    print(f"eps={eps:g}: cond(X)={np.linalg.cond(X):.1f}  statsmodels: k_constant={r.k_constant} LM={s[0]:.6f} F={s[2]:.6f}")
    for nl in (3, 1):
        try:
            o = tsecon.breusch_godfrey_test(y, X, nlags=nl)
            print(f"   tsecon nlags={nl}: constant_added={o['constant_added']} LM={o['statistic']:.6f} F={o['fstat']:.6f}")
        except ValueError as ex:
            print(f"   tsecon nlags={nl}: REFUSED: {ex}")
    dw = tsecon.durbin_watson_test(y, X)
    print(f"   durbin_watson_test on the same x: d={dw['statistic']:.6f} p={dw['pvalue']:.4f}")
