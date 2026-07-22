"""Golden fixtures for the historical decomposition (`historical_decomposition`,
Kilian & Lütkepohl 2017, ch. 4) and its structural-shock / baseline core.

VALIDATION STRATEGY
===================
Nothing here imports tsecon. A self-contained ~60-line NumPy reference fits a
fixed VAR(2) by OLS on a fixed, seeded dataset, Cholesky-identifies (Q = I),
and computes — from an INDEPENDENT implementation — the structural shocks
`E`, the structural IRF `Theta_s`, the decomposition tensor `HD[t][i][j]`, and
the deterministic/initial-condition baseline. Reproducing these in Rust
(faer residuals + forward-substitution orthogonalization + companion-power MA)
is a genuine cross-implementation check.

THE ADDING-UP ANCHOR
--------------------
`HD[t][i][j] = sum_{s=0}^t Theta_s[i,j] E[t-s,j]` together with
`y_{p+t,i} = baseline[t,i] + sum_j HD[t][i][j]` is a deterministic linear-
algebra closed form. The stored `adding_up_residual` (max abs violation) is
checked below to be < 1e-10, and the Rust golden reproduces `HD`, `E`,
`Theta`, and `baseline` to rtol=1e-8/atol=1e-10.

CONVENTIONS (must match the Rust crate exactly)
-----------------------------------------------
* Regressors: Y[t,j] = data[p+t,j]; X[:,0] = 1; X[:,1+(l-1)*n+v] = data[p+t-l,v].
* Coefficients b (k x n, k = 1 + n*p): b[0,i] = intercept of equation i;
  b[1+(l-1)*n+v, i] = A_l[i,v] (coeff of y_{t-l,v} in equation i).
* Sigma = U'U / T_eff (MLE); P = lower Cholesky (positive diagonal).
* W = U P^{-T} (rows w_t = P^{-1} u_t); E = W Q; Theta_s = Psi_s P Q; Q = I.
"""

import json
import os

import numpy as np


def simulate_var(coefs, sigma, n_obs, seed, burn=200):
    """Simulate a stable VAR(p) with Cholesky-correlated Gaussian shocks."""
    n = coefs[0].shape[0]
    p = len(coefs)
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(sigma)
    total = n_obs + burn + p
    y = np.zeros((total, n))
    for t in range(p, total):
        val = chol @ rng.standard_normal(n)
        for i in range(1, p + 1):
            val = val + coefs[i - 1] @ y[t - i]
        y[t] = val
    return y[burn:]  # keep p presample rows + n_obs usable rows


def build_regressors(data, p):
    n = data.shape[1]
    t_eff = data.shape[0] - p
    k = 1 + n * p
    y = np.zeros((t_eff, n))
    x = np.zeros((t_eff, k))
    for t in range(t_eff):
        y[t] = data[p + t]
        x[t, 0] = 1.0
        for col in range(1, k):
            l = (col - 1) // n + 1
            v = (col - 1) % n
            x[t, col] = data[p + t - l, v]
    return y, x


def ols(y, x):
    """B = (X'X)^{-1} X'Y (k x n)."""
    return np.linalg.solve(x.T @ x, x.T @ y)


def coefs_from_b(b, n, p):
    """A_l[i,v] = b[1+(l-1)*n+v, i]; intercept c[i] = b[0,i]."""
    coefs = []
    for l in range(1, p + 1):
        a = np.zeros((n, n))
        for i in range(n):
            for v in range(n):
                a[i, v] = b[1 + (l - 1) * n + v, i]
        coefs.append(a)
    intercept = np.array([b[0, i] for i in range(n)])
    return coefs, intercept


def ma_weights(coefs, horizon):
    """Psi_0..Psi_horizon from lag matrices [A_1, ..., A_p]."""
    n = coefs[0].shape[0]
    p = len(coefs)
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            acc = acc + psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return psi


def baseline_path(coefs, intercept, data, p):
    n = data.shape[1]
    t_eff = data.shape[0] - p
    base = np.zeros((t_eff, n))
    for t in range(t_eff):
        val = intercept.copy()
        for l in range(1, p + 1):
            lag_eff = t - l
            yhat = data[p + lag_eff] if lag_eff < 0 else base[lag_eff]
            val = val + coefs[l - 1] @ yhat
        base[t] = val
    return base


def hist_decomp(theta, e):
    """HD[t][i][j] = sum_{s=0}^t Theta_s[i,j] E[t-s,j]."""
    t_eff = e.shape[0]
    n = e.shape[1]
    hd = np.zeros((t_eff, n, n))
    for t in range(t_eff):
        for s in range(t + 1):
            hd[t] += theta[s] * e[t - s][np.newaxis, :]
    return hd


def as_rows(mat):
    return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]


def pack_case(coefs_true, sigma_true, n_obs, seed):
    n = coefs_true[0].shape[0]
    p = len(coefs_true)
    data = simulate_var(coefs_true, sigma_true, n_obs=n_obs, seed=seed)
    y, x = build_regressors(data, p)
    t_eff = y.shape[0]
    b = ols(y, x)                       # k x n
    u = y - x @ b                       # residuals
    sigma = (u.T @ u) / t_eff           # MLE covariance
    p_chol = np.linalg.cholesky(sigma)  # lower, positive diagonal
    # W = U P^{-T}: rows w_t = P^{-1} u_t  ->  W.T = solve(P, U.T).
    w = np.linalg.solve(p_chol, u.T).T
    e = w.copy()                        # Q = I
    horizon = t_eff - 1
    coefs_b, intercept = coefs_from_b(b, n, p)
    psi = ma_weights(coefs_b, horizon)
    theta = [ps @ p_chol for ps in psi]
    hd = hist_decomp(theta, e)
    baseline = baseline_path(coefs_b, intercept, data, p)

    # Internal (tsecon-free) cross-checks.
    # Round-trip: u_t = P Q eps_t = P w_t.
    recon_u = (p_chol @ w.T).T
    assert np.allclose(recon_u, u, atol=1e-12), "residual round-trip failed"
    # Adding-up identity.
    resid = np.zeros((t_eff, n))
    for t in range(t_eff):
        resid[t] = data[p + t] - baseline[t] - hd[t].sum(axis=1)
    adding_up_residual = float(np.max(np.abs(resid)))
    assert adding_up_residual < 1e-10, f"adding-up residual too large: {adding_up_residual}"

    case = {
        "p": p,
        "n": n,
        "horizon": horizon,
        "data": as_rows(data),
        "b": as_rows(b),
        "sigma": as_rows(sigma),
        "chol": as_rows(p_chol),
        "shocks": as_rows(e),
        "theta": [as_rows(t) for t in theta],
        "hd": [as_rows(hd[t]) for t in range(t_eff)],
        "baseline": as_rows(baseline),
        "adding_up_residual": adding_up_residual,
    }
    return case


def main():
    fixtures = {}

    # -- hd_4var_p2: the primary adding-up golden -----------------------------
    a1 = np.array(
        [
            [0.30, 0.08, 0.00, 0.05],
            [0.05, 0.25, 0.10, 0.00],
            [0.00, 0.06, 0.28, 0.07],
            [0.04, 0.00, 0.05, 0.22],
        ]
    )
    a2 = np.array(
        [
            [0.10, 0.00, 0.02, 0.00],
            [0.00, 0.08, 0.00, 0.03],
            [0.02, 0.00, 0.09, 0.00],
            [0.00, 0.01, 0.00, 0.07],
        ]
    )
    # Positive-definite Sigma = L L' with L lower-triangular.
    lmat = np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.40, 0.90, 0.00, 0.00],
            [0.20, 0.30, 0.80, 0.00],
            [0.10, 0.15, 0.25, 0.70],
        ]
    )
    sigma4 = lmat @ lmat.T
    fixtures["hd_4var_p2"] = pack_case([a1, a2], sigma4, n_obs=120, seed=20260722)

    # -- hd_3var_p1: a smaller companion, single lag --------------------------
    b1 = np.array([[0.50, 0.10, 0.00], [0.00, 0.40, 0.10], [0.10, 0.00, 0.30]])
    l3 = np.array([[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [0.2, 0.3, 0.7]])
    sig3 = l3 @ l3.T
    fixtures["hd_3var_p1"] = pack_case([b1], sig3, n_obs=90, seed=20260723)

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "historical_decomposition_chol.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fixtures, fh, indent=2)
        fh.write("\n")
    print("wrote", path)
    for name, case in fixtures.items():
        print(f"  {name}: T_eff={len(case['baseline'])} n={case['n']} "
              f"adding_up_residual={case['adding_up_residual']:.2e}")


if __name__ == "__main__":
    main()
