"""Golden fixtures for the hierarchical (empirical-Bayes / ML-II) Minnesota
BVAR, à la Giannone, Lenza & Primiceri (2015, REStat "Prior Selection for
Vector Autoregressions").

The whole method is a low-dimensional maximization of the SAME closed-form
matrix-variate-t log marginal likelihood the conjugate NIW posterior already
computes; there is no new likelihood algebra. This reference therefore
re-implements that closed form independently in NumPy/SciPy (it NEVER imports
tsecon) and maximizes it with `scipy.optimize.minimize_scalar`, exactly the
quantity the Rust core optimizes.

Closed form (Kadiyala & Karlsson 1997, eq. 3.6), with the crate's stacked
regression layout y_t' = x_t' B + u_t', x_t = [1, y_{t-1}', ..., y_{t-p}']':

  Obar = (Omega0^-1 + X'X)^-1
  Bbar = Obar (Omega0^-1 B0 + X'Y)
  Sbar = S0 + Y'Y + B0' Omega0^-1 B0 - Bbar' (Omega0^-1 + X'X) Bbar
  vbar = v0 + T
  ln p(Y | lambda) = -(n T / 2) ln pi + (n/2)(ln|Obar| - ln|Omega0|)
                     + (v0/2) ln|S0| - (vbar/2) ln|Sbar|
                     + ln Gamma_n(vbar/2) - ln Gamma_n(v0/2)

Minnesota-NIW prior (lambda-dependent piece is Omega0 only): Sigma ~ IW(S0,v0),
vec(B)|Sigma ~ N(vec(B0), Sigma (x) Omega0), Omega0 = diag(omega),
omega[0] = lambda0^2, omega[1+(l-1)n+j] = lambda1^2 / (l^(2 lambda3) sigma_j^2),
S0 = diag(sigma_j^2), v0 = n + 2, B0 own-first-lag = delta. sigma_j^2 are
lambda-independent univariate AR(4) OLS residual variances.

Run with the project venv:
    .venv/bin/python fixtures/generate_bvar_hierarchical_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import multigammaln

OUT = Path(__file__).parent
META = {
    "numpy": np.__version__,
    "scipy": __import__("scipy").__version__,
    "python": platform.python_version(),
    "authored": "independent closed-form NIW marginal likelihood, maximized by "
    "scipy.optimize.minimize_scalar; GLP-2015 ML-II lambda selection",
}

# Rust-core default search box and grid (must match HierarchicalConfig::default).
LAMBDA1_LO = 1e-4
LAMBDA1_HI = 10.0
N_GRID = 25
LAMBDA1_INIT = 0.2
FIXED_BATTERY = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def ar_resid_var(y, p=4):
    """OLS AR(p)-with-intercept residual variance, denominator T_eff - (p+1)."""
    n = len(y)
    X = np.column_stack([np.ones(n - p)] + [y[p - j - 1 : n - j - 1] for j in range(p)])
    yy = y[p:]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    r = yy - X @ b
    return float(r @ r / (len(yy) - X.shape[1]))


def build_design(data, p):
    T = data.shape[0] - p
    Y = data[p:]
    X = np.column_stack([np.ones(T)] + [data[p - l : -l] for l in range(1, p + 1)])
    return Y, X, T


def posterior(data, p, lam0, lam1, lam3, delta, sig2):
    """Closed-form NIW posterior moments and the log marginal likelihood."""
    n = data.shape[1]
    Y, X, T = build_design(data, p)
    k = X.shape[1]

    B0 = np.zeros((k, n))
    for j in range(n):
        B0[1 + j, j] = delta  # own first lag

    omega_diag = np.empty(k)
    omega_diag[0] = lam0**2
    for l in range(1, p + 1):
        for j in range(n):
            omega_diag[1 + (l - 1) * n + j] = (lam1**2) / (l ** (2 * lam3)) / sig2[j]

    Omega0 = np.diag(omega_diag)
    S0 = np.diag(sig2)
    v0 = n + 2.0

    Oinv = np.diag(1.0 / omega_diag)
    K = Oinv + X.T @ X
    Obar = np.linalg.inv(K)
    Bbar = Obar @ (Oinv @ B0 + X.T @ Y)
    Sbar = S0 + Y.T @ Y + B0.T @ Oinv @ B0 - Bbar.T @ K @ Bbar
    vbar = v0 + T

    _, logdet_Obar = np.linalg.slogdet(Obar)
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
    sigma_mean = Sbar / (vbar - n - 1)
    return float(lml), Bbar, sigma_mean


def ml_only(data, p, lam0, lam1, lam3, delta, sig2):
    return posterior(data, p, lam0, lam1, lam3, delta, sig2)[0]


def simulate_var1(seed=20260722, T=200, burn=100):
    """A stationary bivariate VAR(1) with companion spectral radius 0.6
    (eigenvalues 0.6 and 0.4 of A = [[0.5, 0.1], [0.1, 0.5]]).

    NumPy RNG only — this generator never touches tsecon; the Rust test
    consumes the stored sample, so no cross-language RNG contract is needed.
    """
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.1, 0.5]])
    assert abs(max(abs(np.linalg.eigvals(A))) - 0.6) < 1e-12
    n = 2
    L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]]))
    y = np.zeros((T + burn, n))
    for t in range(1, T + burn):
        y[t] = A @ y[t - 1] + L @ rng.standard_normal(n)
    return y[burn:]


def gen():
    var_fx = json.loads((OUT / "var.json").read_text(encoding="utf-8"))
    data = np.array(var_fx["data_100dlog_gdp_cons_inv"])
    n = data.shape[1]
    p = 2
    lam0, lam3, delta = 100.0, 1.0, 0.0

    sig2 = np.array([ar_resid_var(data[:, j]) for j in range(n)])

    def ml(lam1):
        return ml_only(data, p, lam0, lam1, lam3, delta, sig2)

    # 25-point natural-log-spaced grid over the Rust default box [1e-4, 10],
    # matching HierarchicalConfig::default's pre-scan exactly.
    grid = np.exp(np.linspace(np.log(LAMBDA1_LO), np.log(LAMBDA1_HI), N_GRID))
    grid_log_ml = np.array([ml(l) for l in grid])

    # ML-II optimum over the same box.
    res = minimize_scalar(
        lambda l: -ml(l),
        method="bounded",
        bounds=(LAMBDA1_LO, LAMBDA1_HI),
        options={"xatol": 1e-10},
    )
    lambda1_star = float(res.x)
    log_ml_star, Bbar_star, sigma_mean_star = posterior(
        data, p, lam0, lambda1_star, lam3, delta, sig2
    )

    fixed_lambda_lml = {f"{l}": ml(l) for l in FIXED_BATTERY}

    # Simulated stationary VAR(1) for the interior-and-dominant recovery test.
    sim = simulate_var1()
    sim_sig2 = np.array([ar_resid_var(sim[:, j]) for j in range(sim.shape[1])])
    sim_grid = np.exp(np.linspace(np.log(LAMBDA1_LO), np.log(LAMBDA1_HI), N_GRID))
    sim_grid_log_ml = np.array(
        [ml_only(sim, 1, lam0, l, lam3, delta, sim_sig2) for l in sim_grid]
    )
    sim_res = minimize_scalar(
        lambda l: -ml_only(sim, 1, lam0, l, lam3, delta, sim_sig2),
        method="bounded",
        bounds=(LAMBDA1_LO, LAMBDA1_HI),
        options={"xatol": 1e-10},
    )

    dump = {
        "_meta": META,
        "data": data.tolist(),
        "spec": {
            "p": p,
            "lambda0": lam0,
            "lambda3": lam3,
            "delta": delta,
            "lambda1_lo": LAMBDA1_LO,
            "lambda1_hi": LAMBDA1_HI,
            "lambda1_init": LAMBDA1_INIT,
            "n_grid": N_GRID,
            "ar_resid_var_lag4": sig2.tolist(),
        },
        "grid_lambda1_25": grid.tolist(),
        "grid_log_ml_25": grid_log_ml.tolist(),
        "lambda1_star": lambda1_star,
        "log_ml_star": log_ml_star,
        "b_bar_star": Bbar_star.tolist(),
        "sigma_mean_star": sigma_mean_star.tolist(),
        "lambda1_init_log_ml": ml(LAMBDA1_INIT),
        "fixed_lambda_lml": fixed_lambda_lml,
        "sim_var1": {
            "p": 1,
            "data": sim.tolist(),
            "ar_resid_var_lag4": sim_sig2.tolist(),
            "lambda1_star": float(sim_res.x),
            "log_ml_star": float(-sim_res.fun),
        },
    }
    path = OUT / "bvar_hierarchical.json"
    path.write_text(json.dumps(dump, indent=1), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    print(f"main dataset: lambda1_star = {lambda1_star:.6f}, log_ml_star = {log_ml_star:.6f}")
    print(f"  grid argmax lambda1 = {grid[int(np.argmax(grid_log_ml))]:.6f}")
    print(f"  fixed battery lml   = {fixed_lambda_lml}")
    print(f"sim VAR(1): lambda1_star = {float(sim_res.x):.6f}, interior = {LAMBDA1_LO < sim_res.x < LAMBDA1_HI}")


if __name__ == "__main__":
    gen()
