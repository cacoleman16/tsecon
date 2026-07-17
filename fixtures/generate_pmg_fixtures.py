"""Golden fixture for tsecon-panelts::pmg — the Pesaran, Shin & Smith (1999)
pooled-mean-group (PMG) ARDL(1,1) estimator.

Run with the project venv:

    .venv/bin/python fixtures/generate_pmg_fixtures.py

============================================================================
WHAT KIND OF GOLDEN IS THIS
============================================================================
This is a **DOCUMENTED-FORMULA golden**, NOT an external-package golden. There
is no `pmg` in statsmodels / linearmodels to call, so this generator
reimplements the *same* PSS concentrated-maximum-likelihood back-substitution
estimator independently in NumPy, with the estimating equations written out in
full below (Pesaran, Shin & Smith 1999, "Pooled Mean Group Estimation of
Dynamic Heterogeneous Panels", JASA 94(446):621-634). The Rust crate implements
the identical estimator through a different numerical path (per-unit OLS via
Cholesky normal equations in tsecon-hac, the pooled k×k solve via faer's
Cholesky), and must reproduce the pooled long-run theta and the average
adjustment speed phi_bar to ~1e-8.

Because both sides are the SAME estimator, agreement is a cross-implementation
consistency check (NumPy lstsq/SVD residualization + explicit GLS update vs.
Rust Cholesky), not a check against an independent authority. That is stated
plainly and is the honest description of the validation surface. The estimator
itself is additionally *property*-validated on the Rust side: on data simulated
with a known common long-run theta0 it recovers theta0 within Monte-Carlo bands
and pools far more tightly than a free mean-group of the per-unit long runs.

============================================================================
THE MODEL AND THE ESTIMATOR (PSS 1999)
============================================================================
For each unit i the ARDL(1,1) is

    y_it = mu_i + lambda_i y_{i,t-1} + delta_i0' x_it + delta_i1' x_{i,t-1} + e_it,

reparameterized in error-correction form as

    Δy_it = phi_i ( y_{i,t-1} - theta' x_{i,t-1} ) + delta_i0' Δx_it + mu_i + e_it,
    phi_i = lambda_i - 1,   theta = (delta_i0 + delta_i1)/(1 - lambda_i).

PMG pools the LONG-RUN theta (common across i) by ML while phi_i, delta_i0, mu_i
stay free. With e_it ~ N(0, sigma_i^2), concentrate the short-run block
W_i = [const, Δx_i] out by partialling every quantity on W_i (least-squares
residuals, tilde), giving  Δỹ_i = phi_i ( ỹ_{i,-1} - X̃_{i,-1} theta ) + ẽ_i.
The concentrated log-likelihood is

    l(theta) = -1/2 sum_i T_i [ log(2 pi) + 1 + log sigma_i^2(theta) ],

and, given theta,

    xi~_i     = ỹ_{i,-1} - X̃_{i,-1} theta
    phi_i     = (xi~_i' xi~_i)^{-1} xi~_i' Δỹ_i
    sigma_i^2 = || Δỹ_i - phi_i xi~_i ||^2 / T_i.

Given {phi_i, sigma_i^2}, the pooled feasible-GLS update for theta solves

    A theta = b,
    A = sum_i (phi_i^2 / sigma_i^2) X̃_{i,-1}' X̃_{i,-1},
    b = - sum_i (phi_i / sigma_i^2) X̃_{i,-1}' ( Δỹ_i - phi_i ỹ_{i,-1} ).

Iterate "theta -> {phi, sigma2} -> theta" from theta = 0 (the identical start
used by the Rust crate) to convergence. At the fixed point

    Var(theta) = A^{-1},   SE(theta_k) = sqrt([A^{-1}]_kk),
    phi_bar    = mean_i phi_i.

============================================================================
DGP — genuine common-long-run ARDL(1,1)
============================================================================
Data are simulated FROM an ARDL(1,1) with a COMMON long-run theta0: pick
lambda_i in (0.2, 0.7) (stationary), free short-run delta_i0, and set
delta_i1 = theta0 (1 - lambda_i) - delta_i0 so that the long run
(delta_i0 + delta_i1)/(1 - lambda_i) == theta0 for every unit. Intercepts mu_i
and adjustment speeds phi_i = lambda_i - 1 are heterogeneous. PMG should recover
theta0; the stored `theta0` lets the reader see it directly.
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
META = {"numpy": np.__version__, "python": platform.python_version()}

rng = np.random.default_rng(20260717)

N, T_RAW, K = 30, 90, 2          # units, raw periods, long-run regressors
theta0 = np.array([1.50, -0.80])  # TRUE common long-run coefficient

MAX_ITER = 1000
TOL = 1e-12


def simulate_unit():
    """Simulate one ARDL(1,1) series with common long run theta0."""
    lam = rng.uniform(0.2, 0.7)                       # stationary AR root
    mu = rng.normal(0.5, 1.0)                         # unit intercept
    d0 = rng.normal([0.6, -0.3], [0.25, 0.25])        # free short-run delta_i0
    d1 = theta0 * (1.0 - lam) - d0                    # pins the common long run
    burn = 50
    tt = T_RAW + burn
    # stationary AR(1) regressors, unit-specific mean/persistence
    x = np.empty((tt, K))
    rho = rng.uniform(0.3, 0.6, K)
    xmean = rng.normal(0.0, 1.0, K)
    x[0] = xmean
    for t in range(1, tt):
        x[t] = xmean * (1 - rho) + rho * x[t - 1] + rng.normal(0.0, 1.0, K)
    y = np.empty(tt)
    y[0] = mu / (1 - lam)
    for t in range(1, tt):
        y[t] = (mu + lam * y[t - 1] + d0 @ x[t] + d1 @ x[t - 1]
                + rng.normal(0.0, 0.5))
    return y[burn:], x[burn:]


def ols_resid(target, design):
    """Least-squares residuals of `target` on `design` (SVD via lstsq)."""
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ beta


def prepare(y, x):
    """ARDL(1,1) EC rows partialled on W = [const, Δx]. Returns dỹ, ỹlag, X̃lag."""
    t_raw = len(y)
    dy = y[1:] - y[:-1]                 # Δy, length T
    ylag = y[:-1]                       # y_{-1}
    dx = x[1:] - x[:-1]                 # Δx, (T, K)
    xlag = x[:-1]                       # x_{-1}, (T, K)
    T = t_raw - 1
    W = np.column_stack([np.ones(T), dx])
    dy_t = ols_resid(dy, W)
    ylag_t = ols_resid(ylag, W)
    xlag_t = np.column_stack([ols_resid(xlag[:, j], W) for j in range(K)])
    return dy_t, ylag_t, xlag_t, T


def phi_sigma_given_theta(units, theta):
    phi, sig2 = [], []
    for dy, ylag, xlag, T in units:
        xi = ylag - xlag @ theta
        den = xi @ xi
        num = xi @ dy
        phi_i = num / den if den > 0 else 0.0
        r = dy - phi_i * xi
        phi.append(phi_i)
        sig2.append((r @ r) / T)
    return np.array(phi), np.array(sig2)


def pooled_system(units, phi, sig2):
    A = np.zeros((K, K))
    b = np.zeros(K)
    for (dy, ylag, xlag, T), phi_i, s2 in zip(units, phi, sig2):
        A += (phi_i * phi_i / s2) * (xlag.T @ xlag)
        d = dy - phi_i * ylag
        b -= (phi_i / s2) * (xlag.T @ d)
    return A, b


# --- simulate the panel -------------------------------------------------------
Y, X = [], []
for _ in range(N):
    y, x = simulate_unit()
    Y.append(y)
    X.append(x)

units = [prepare(Y[i], X[i]) for i in range(N)]

# --- PSS back-substitution from theta = 0 ------------------------------------
theta = np.zeros(K)
iterations = 0
converged = False
for it in range(1, MAX_ITER + 1):
    phi, sig2 = phi_sigma_given_theta(units, theta)
    A, b = pooled_system(units, phi, sig2)
    theta_new = np.linalg.solve(A, b)
    delta = np.max(np.abs(theta_new - theta))
    theta = theta_new
    iterations = it
    if delta < TOL:
        converged = True
        break
assert converged, "PMG iteration did not converge"

phi, sig2 = phi_sigma_given_theta(units, theta)
A, _ = pooled_system(units, phi, sig2)
A_inv = np.linalg.inv(A)
theta_se = np.sqrt(np.diag(A_inv))
phi_bar = float(phi.mean())
loglik = float(np.sum([-0.5 * T * (np.log(2 * np.pi) + 1 + np.log(s2))
                       for (_, _, _, T), s2 in zip(units, sig2)]))

# --- free mean-group of the per-unit long runs (for the pooling contrast) -----
# Per-unit unrestricted EC OLS: Δy on [const, y_{-1}, x_{-1}, Δx]; long-run
# theta_i = -coef(x_{-1}) / coef(y_{-1}).
free_theta = []
for i in range(N):
    y, x = Y[i], X[i]
    T = len(y) - 1
    design = np.column_stack([
        np.ones(T), y[:-1], x[:-1], x[1:] - x[:-1],
    ])
    beta, *_ = np.linalg.lstsq(design, y[1:] - y[:-1], rcond=None)
    phi_i = beta[1]
    coef_xlag = beta[2:2 + K]
    free_theta.append(-coef_xlag / phi_i)
free_theta = np.array(free_theta)          # (N, K)
free_mg = free_theta.mean(axis=0)
free_mg_sd = free_theta.std(axis=0, ddof=1)

print(f"true theta0    : {theta0}")
print(f"PMG   theta     : {theta}   (SE {theta_se})")
print(f"free MG theta   : {free_mg}   (cross-unit sd {free_mg_sd})")
print(f"phi_bar         : {phi_bar:.6f}")
print(f"loglik          : {loglik:.6f}   iters {iterations}")

out = {
    "_meta": META,
    "_doc": "documented-formula golden: NumPy reimplementation of the PSS 1999 "
            "PMG concentrated-ML back-substitution; NOT an external package.",
    "design": {"N": N, "T_raw": T_RAW, "K": K},
    "theta0": [float(v) for v in theta0],
    "y": [[float(v) for v in row] for row in Y],        # N x T_raw
    "x": [[[float(v) for v in x_i[:, k]] for x_i in X] for k in range(K)],  # K x N x T_raw
    "pmg": {
        "theta": [float(v) for v in theta],
        "theta_se": [float(v) for v in theta_se],
        "phi_bar": phi_bar,
        "phi": [float(v) for v in phi],
        "sigma2": [float(v) for v in sig2],
        "loglik": loglik,
        "iterations": iterations,
    },
    "free_mg": {
        "theta": [float(v) for v in free_mg],
        "cross_unit_sd": [float(v) for v in free_mg_sd],
    },
}

path = OUT / "pmg.json"
path.write_text(json.dumps(out))
print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
