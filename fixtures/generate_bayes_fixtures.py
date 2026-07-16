"""Bayesian golden fixtures: conjugate NIW-BVAR posterior (analytic, formulas
documented below) and ArviZ convergence diagnostics.

Run with the project venv (arviz installed there):
    .venv/bin/python fixtures/generate_bayes_fixtures.py

The NIW posterior is closed-form; the fixture is self-authored from the
standard conjugate updating equations (e.g. Giannone-Lenza-Primiceri 2015,
appendix; Karlsson 2013 Handbook chapter, section 2):

  VAR(p): y_t' = x_t' B + u_t',  x_t = [1, y_{t-1}', ..., y_{t-p}']',
          U ~ MN(0, I_T (x) Sigma)   [rows iid N(0, Sigma)]
  Prior:  Sigma ~ IW(S0, v0),  vec(B) | Sigma ~ N(vec(B0), Sigma (x) Omega0)
  Posterior:
    Obar = (Omega0^-1 + X'X)^-1
    Bbar = Obar (Omega0^-1 B0 + X'Y)
    Sbar = S0 + Y'Y + B0' Omega0^-1 B0 - Bbar' Obar^-1 Bbar
    vbar = v0 + T
  Log marginal likelihood (matrix-variate-t normalization):
    ln p(Y) = -(n T / 2) ln pi + (n/2)(ln|Obar| - ln|Omega0|)
              + (v0/2) ln|S0| - (vbar/2) ln|Sbar|
              + ln Gamma_n(vbar/2) - ln Gamma_n(v0/2)
  where Gamma_n is the multivariate gamma function.

The Minnesota prior moments used to build (B0, Omega0, S0, v0) follow the
standard construction: own-lag prior mean delta_i on lag 1 (0 here, data are
growth rates), prior variance of the coefficient on lag l of variable j in
equation i = (lambda1^2 / l^(2*lambda3)) * (sigma_i^2/sigma_j^2 scaling via
S0), Omega0 diagonal in the (1 + n*p) regressors with intercept variance
lambda0^2; sigma_j^2 from univariate AR(4) residual variances (the common
convention). Because Omega0 is shared across equations (Kronecker form), the
cross-variable scaling lives in S0 = diag(sigma_1^2..sigma_n^2), v0 = n + 2.
"""
import json
import platform
from pathlib import Path

import numpy as np
from scipy.special import multigammaln

OUT = Path(__file__).parent
META = {"numpy": np.__version__, "python": platform.python_version(), "authored": "closed-form NIW updating, see docstring"}


def ar_resid_var(y, p=4):
    n = len(y)
    X = np.column_stack([np.ones(n - p)] + [y[p - j - 1 : n - j - 1] for j in range(p)])
    yy = y[p:]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    r = yy - X @ b
    return float(r @ r / (len(yy) - X.shape[1]))


def gen_bvar():
    var_fx = json.loads((OUT / "var.json").read_text())
    data = np.array(var_fx["data_100dlog_gdp_cons_inv"])
    n = data.shape[1]
    p = 2
    lam0, lam1, lam3 = 100.0, 0.2, 1.0

    T = data.shape[0] - p
    Y = data[p:]
    X = np.column_stack([np.ones(T)] + [data[p - l : -l] for l in range(1, p + 1)])
    k = X.shape[1]

    sig2 = np.array([ar_resid_var(data[:, j]) for j in range(n)])

    B0 = np.zeros((k, n))  # growth-rate data: own-lag prior mean 0
    omega_diag = np.empty(k)
    omega_diag[0] = lam0**2
    for l in range(1, p + 1):
        for j in range(n):
            omega_diag[1 + (l - 1) * n + j] = (lam1**2) / (l ** (2 * lam3)) / sig2[j]
    Omega0 = np.diag(omega_diag)
    S0 = np.diag(sig2)
    v0 = n + 2.0

    Oinv = np.diag(1.0 / omega_diag)
    Obar = np.linalg.inv(Oinv + X.T @ X)
    Bbar = Obar @ (Oinv @ B0 + X.T @ Y)
    Sbar = S0 + Y.T @ Y + B0.T @ Oinv @ B0 - Bbar.T @ (Oinv + X.T @ X) @ Bbar
    vbar = v0 + T

    sign, logdet_Obar = np.linalg.slogdet(Obar)
    _, logdet_Omega0 = np.linalg.slogdet(Omega0)
    _, logdet_S0 = np.linalg.slogdet(S0)
    _, logdet_Sbar = np.linalg.slogdet(Sbar)
    lml = (
        -(n * T / 2) * np.log(np.pi)
        + (n / 2) * (logdet_Obar - logdet_Omega0)
        + (v0 / 2) * logdet_S0
        - (vbar / 2) * logdet_Sbar
        + multigammaln(vbar / 2, n)
        - multigammaln(v0 / 2, n)
    )

    # Posterior mean of Sigma (IW mean) and of B; plus the posterior
    # predictive one-step moments at the sample end for cross-checking draws.
    sigma_mean = Sbar / (vbar - n - 1)

    dump = {
        "_meta": META,
        "data": data.tolist(),
        "spec": {"p": p, "lambda0": lam0, "lambda1": lam1, "lambda3": lam3,
                 "ar_resid_var_lag4": sig2.tolist(), "v0": v0,
                 "regressor_order": "intercept first, then lag 1 block (vars in data order), then lag 2"},
        "prior": {"omega0_diag": omega_diag.tolist(), "s0_diag": sig2.tolist()},
        "posterior": {
            "b_bar": Bbar.tolist(),
            "omega_bar": Obar.tolist(),
            "s_bar": Sbar.tolist(),
            "v_bar": vbar,
            "sigma_posterior_mean": sigma_mean.tolist(),
            "log_marginal_likelihood": float(lml),
        },
    }
    path = OUT / "bvar_niw.json"
    path.write_text(json.dumps(dump, indent=1))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def gen_convergence():
    import arviz as az

    rng = np.random.default_rng(77)
    # Four chains, well mixed: AR(0.3) around 0.
    def chain(seed, rho, mu):
        r = np.random.default_rng(seed)
        x = np.empty(1000)
        x[0] = mu
        for t in range(1, 1000):
            x[t] = mu + rho * (x[t - 1] - mu) + r.standard_normal() * np.sqrt(1 - rho**2)
        return x

    good = np.stack([chain(s, 0.3, 0.0) for s in [1, 2, 3, 4]])
    bad = np.stack([chain(s, 0.95, m) for s, m in [(5, 0.0), (6, 0.0), (7, 1.5), (8, 1.5)]])

    out = {"_meta": {**META, "arviz": az.__version__}}
    for name, chains in [("good", good), ("bad", bad)]:
        idata = az.convert_to_dataset(chains)
        out[name] = {
            "chains": chains.tolist(),
            "rhat_rank": float(np.asarray(az.rhat(idata, method="rank")["x"].values).item()),
            "ess_bulk": float(np.asarray(az.ess(idata, method="bulk")["x"].values).item()),
            "ess_tail": float(np.asarray(az.ess(idata, method="tail")["x"].values).item()),
        }
    path = OUT / "convergence.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    gen_bvar()
    gen_convergence()
