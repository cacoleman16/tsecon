"""Golden fixtures for the zero + sign restricted Bayesian SVAR
(`zero_sign_svar`, Rubio-Ramirez-Waggoner-Zha 2010).

VALIDATION STRATEGY
===================
Nothing here imports tsecon. The stored `theta[h]` are produced by an
INDEPENDENT NumPy path: the reduced-form MA weights `Psi_h` from the pure
companion-power recursion and the lower Cholesky factor `L = chol(Sigma)`
(positive diagonal), combined as `Theta_h = Psi_h @ L`. So reproducing them in
Rust (faer companion power + jittered lower Cholesky, then the RWZ null-space
recursion) is a genuine cross-implementation check.

THE RECURSIVE / CHOLESKY ANCHOR
-------------------------------
Impose `Theta_0[(i, j)] = 0` for every `i < j` (the strict upper triangle of
the impact matrix), no sign restrictions, positive-diagonal normalization.
Then the RWZ column recursion has a one-dimensional null space at every step,
so `Q` is determined up to column signs; positive-diagonal normalization fixes
`Q = I`, and `Theta_h = Psi_h L` DETERMINISTICALLY (seed-independent,
weight-free). That equals `var_irf(orth=True)`, an independent Cholesky path,
so the Python full-pipeline cross-check is genuine too.

Each case stores the raw reduced form (`b` in the `1 + n*p` regressor layout,
`sigma`), `p`, `horizon`, the expected `theta[h][i][j]`, and a simulated
`data` matrix (so an end-to-end binding test can run `zero_sign_svar(data,
...)` through the Minnesota-NIW posterior and, on the recursive pattern, match
`var_irf(data, orth=True)`).
"""

import json
import os

import numpy as np


def ma_weights(coefs, horizon):
    """Psi_0..Psi_horizon from lag matrices coefs = [A_1, ..., A_p]."""
    n = coefs[0].shape[0]
    p = len(coefs)
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            acc = acc + psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return psi


def cholesky_irf(coefs, sigma, horizon):
    """Theta_h = Psi_h @ chol_lower(Sigma), the recursive structural IRF."""
    lower = np.linalg.cholesky(sigma)  # lower, positive diagonal
    psi = ma_weights(coefs, horizon)
    return [ps @ lower for ps in psi], lower


def as_rows(mat):
    return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]


def b_layout(coefs):
    """Pack lag matrices into the crate `1 + n*p` regressor-by-equation `b`.

    b[0, :]              = intercept (zero here)
    b[1 + (l-1)*n + v, i] = coefficient of y_{t-l, v} in equation i = A_l[i, v].
    """
    n = coefs[0].shape[0]
    p = len(coefs)
    k = 1 + n * p
    b = np.zeros((k, n))
    for l in range(1, p + 1):
        a = coefs[l - 1]
        for i in range(n):
            for v in range(n):
                b[1 + (l - 1) * n + v, i] = a[i, v]
    return b


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
    return y[burn + p:]


def pack_case(coefs, sigma, horizon, seed):
    theta, lower = cholesky_irf(coefs, sigma, horizon)
    data = simulate_var(coefs, sigma, n_obs=300, seed=seed)
    return {
        "p": len(coefs),
        "horizon": horizon,
        "b": as_rows(b_layout(coefs)),
        "sigma": as_rows(sigma),
        "chol": as_rows(lower),
        "theta": [as_rows(t) for t in theta],
        "data": as_rows(data),
    }, theta, lower


def main():
    fixtures = {}

    # -- recursive_3var_p1 ---------------------------------------------------
    a1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    sig3 = np.array([[1.0, 0.4, 0.2], [0.4, 0.97, 0.33], [0.2, 0.33, 0.62]])
    fixtures["recursive_3var_p1"], theta1, lower1 = pack_case([a1], sig3, 12, seed=20260722)

    # -- recursive_3var_p2: exercises the companion-power recursion ----------
    b1 = np.array([[0.40, 0.10, 0.00], [0.05, 0.30, 0.10], [0.00, 0.10, 0.20]])
    b2 = np.array([[0.10, 0.00, 0.00], [0.00, 0.10, 0.00], [0.00, 0.00, 0.10]])
    fixtures["recursive_3var_p2"], _, _ = pack_case([b1, b2], sig3, 12, seed=20260723)

    # -- sanity assertions (no tsecon involved) ------------------------------
    # P1: theta[0] == L (impact IRF is the Cholesky factor).
    assert np.allclose(np.array(fixtures["recursive_3var_p1"]["theta"][0]), lower1, atol=1e-14)
    # P2: L is lower-triangular with positive diagonal.
    assert abs(lower1[0, 1]) < 1e-15 and abs(lower1[0, 2]) < 1e-15 and abs(lower1[1, 2]) < 1e-15
    assert all(lower1[i, i] > 0 for i in range(3))
    # P3: L L' == Sigma.
    assert np.allclose(lower1 @ lower1.T, sig3, atol=1e-12)
    # P4: strict-upper-triangle impact zeros hold (they ARE the restriction).
    for i in range(3):
        for j in range(3):
            if i < j:
                assert abs(theta1[0][i, j]) < 1e-14

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "zero_sign_svar.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fixtures, fh, indent=2)
        fh.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
