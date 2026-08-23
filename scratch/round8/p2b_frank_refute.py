import numpy as np, tsecon
from scipy import optimize
from statsmodels.distributions.copula.api import FrankCopula

rng = np.random.default_rng(80802)
n = 600
z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
tp = z / np.sqrt(rng.chisquare(4, size=n) / 4)[:, None]
u = tsecon.pseudo_obs(np.column_stack([tp[:, 0], tp[:, 1]]))


def negll(th):
    v = -np.sum(FrankCopula(theta=th).logpdf(u))
    return np.inf if np.isnan(v) else v


r = optimize.minimize_scalar(negll, bounds=(0.01, 30.0), method="bounded", options={"xatol": 1e-10})
print("scipy frank optimum:", r.x, "loglik:", -r.fun)
f = tsecon.copula_fit(u, family="frank")
print("tsecon frank:", f["theta"], "loglik:", f["loglik"])
print("param diff:", abs(f["theta"] - r.x), "loglik diff:", f["loglik"] - (-r.fun))
rneg = optimize.minimize_scalar(negll, bounds=(-30.0, -0.01), method="bounded", options={"xatol": 1e-8})
print("negative-side best:", rneg.x, -rneg.fun)
