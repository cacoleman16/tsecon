"""var_forecast / var_fit on a short sample of independent N(0,1) series: the residual
covariance is singular only because nobs - m < k (too few observations for the lag order),
yet the refusal blames 'a duplicated series, a scaled copy ... or a constant column' and
names neither `lags` nor the sample size. statsmodels fits the same VAR."""
import numpy as np, tsecon
from statsmodels.tsa.api import VAR
y = np.random.default_rng(0).normal(size=(7, 4))   # 4 independent series, T = 7
res = VAR(y).fit(1, trend="n")
print("statsmodels: nobs", res.nobs, "df_resid", res.df_resid, "rank(Sigma_u) =", np.linalg.matrix_rank(res.sigma_u), "of 4")
print("expected: a refusal naming lags / the sample size (nobs - m = 6 - 4 = 2 < k = 4)")
try:
    tsecon.var_forecast(y, lags=1, steps=2, trend="n")
except ValueError as ex:
    print("actual  :", ex)
