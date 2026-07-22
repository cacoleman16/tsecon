"""Golden fixtures for the Blanchard-Quah (1989) long-run SVAR.

VALIDATION STRATEGY
===================
Nothing here imports tsecon. Every stored number is produced by an
INDEPENDENT NumPy path that transcribes the documented closed form, so
reproducing it in Rust (faer LU inverse + faer lower Cholesky) is a genuine
cross-implementation check.

THE CLOSED FORM (exact analog of R `vars::BQ`, Pfaff 2008)
---------------------------------------------------------
Reduced-form VAR(p):  y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,
E[u_t u_t'] = Sigma_u.  MA(inf): Psi_0 = I,
Psi_h = sum_{i=1..min(h,p)} Psi_{h-i} A_i.

Frequency-zero multiplier   C1 = (I - A_1 - ... - A_p)^{-1}   (= sum_h Psi_h).
Let D = I - sum A_i (= C1^{-1}).  Structural shocks eps (cov = I),
u = B eps, so B B' = Sigma_u.  The long-run (cumulative) structural impact
LR = C1 B is imposed to be lower-triangular with positive diagonal:

    LR = chol_lower( C1 Sigma_u C1' )          (numpy.linalg.cholesky)
    B  = C1^{-1} LR = D @ LR                    (structural impact matrix)
    Theta_h = Psi_h @ B                         (structural IRF, Theta_0 = B)

This is bit-for-bit `vars::BQ`, whose R source computes
Amat = I - sum A_i = D, LR = t(chol(solve(Amat) %*% Sigma %*% t(solve(Amat)))),
B = Amat %*% LR.  R was not available in the build environment, so the R
cross-check here is by ALGEBRA EQUIVALENCE (the two expressions are
identical) rather than by execution; the NumPy path below is otherwise fully
independent of the Rust implementation under test.

Each structural column is identified only up to sign; the default pins it by
the positive-diagonal Cholesky on LR (matches vars::BQ, no re-signing). The
"impact"-normalized variant instead flips column j whenever B[j, j] < 0.

CASES
-----
* algebra_2var / algebra_3var : hard-coded reduced forms (A_i, Sigma_u).
  These isolate the identification map from any estimation step.
* estimated_2var : a stable VAR(2) simulated from a seeded RNG, OLS-fitted in
  pure NumPy with the statsmodels/tsecon regressor layout
  ([1, y_{t-1}, ..., y_{t-p}], df-adjusted Sigma_u = U'U/(T-m), m = 1 + k p).
  The raw data matrix is stored so an end-to-end binding test can run
  long_run_svar(data, ...) through the OLS path; the reduced form and its
  expected identification are stored so the crate test can check the map on a
  realistic (non-toy) reduced form.
"""

import json
import os

import numpy as np


def ma_weights(coefs, horizon):
    """Psi_0..Psi_horizon from lag matrices coefs = [A_1, ..., A_p]."""
    k = coefs[0].shape[0]
    p = len(coefs)
    psi = [np.eye(k)]
    for h in range(1, horizon + 1):
        acc = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            acc = acc + psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return psi


def long_run_svar_closed_form(coefs, sigma_u, horizon, normalize_impact=False):
    """The documented BQ closed form, NumPy-only."""
    k = coefs[0].shape[0]
    d = np.eye(k)
    for a in coefs:
        d = d - a
    c1 = np.linalg.inv(d)
    m = c1 @ sigma_u @ c1.T
    m = 0.5 * (m + m.T)  # symmetrize, mirroring the Rust path
    lr = np.linalg.cholesky(m)  # lower, positive diagonal
    b = d @ lr
    if normalize_impact:
        for j in range(k):
            if b[j, j] < 0.0:
                b[:, j] = -b[:, j]
                lr[:, j] = -lr[:, j]
    psi = ma_weights(coefs, horizon)
    theta = [ps @ b for ps in psi]
    return {
        "impact": b,
        "long_run": lr,
        "long_run_multiplier": c1,
        "irf": theta,
    }


def as_rows(mat):
    return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]


def pack_case(coefs, sigma_u, horizon, data=None, normalize_impact=False):
    res = long_run_svar_closed_form(coefs, sigma_u, horizon, normalize_impact)
    out = {
        "horizon": horizon,
        "normalize_impact": normalize_impact,
        "coefs": [as_rows(a) for a in coefs],
        "sigma_u": as_rows(sigma_u),
        "impact": as_rows(res["impact"]),
        "long_run": as_rows(res["long_run"]),
        "long_run_multiplier": as_rows(res["long_run_multiplier"]),
        "irf": [as_rows(t) for t in res["irf"]],
    }
    if data is not None:
        out["data"] = as_rows(data)
        out["lags"] = len(coefs)
    return out


def ols_var(data, p):
    """OLS VAR(p) with a constant, statsmodels/tsecon layout.

    Returns (coefs=[A_1..A_p], sigma_u) with the df-adjusted covariance
    U'U/(T-m), m = 1 + k p.
    """
    n, k = data.shape
    rows_y = []
    rows_z = []
    for t in range(p, n):
        z = [1.0]
        for lag in range(1, p + 1):
            z.extend(data[t - lag].tolist())
        rows_z.append(z)
        rows_y.append(data[t].tolist())
    y = np.asarray(rows_y)  # (T_eff, k)
    z = np.asarray(rows_z)  # (T_eff, 1 + k p)
    teff = y.shape[0]
    m = 1 + k * p
    beta = np.linalg.solve(z.T @ z, z.T @ y)  # (1 + k p, k)
    resid = y - z @ beta
    sigma_u = resid.T @ resid / (teff - m)
    coefs = []
    for lag in range(1, p + 1):
        block = beta[1 + (lag - 1) * k : 1 + lag * k, :]  # (k, k): [regressor, equation]
        coefs.append(block.T.copy())  # A_i[r, c] = block[c, r]
    return coefs, sigma_u


def simulate_var(coefs, sigma_u, n, seed, burn=200):
    """Simulate a stable VAR(p) with a Cholesky-correlated Gaussian shock."""
    k = coefs[0].shape[0]
    p = len(coefs)
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(sigma_u)
    total = n + burn + p
    y = np.zeros((total, k))
    for t in range(p, total):
        val = chol @ rng.standard_normal(k)
        for i in range(1, p + 1):
            val = val + coefs[i - 1] @ y[t - i]
        y[t] = val
    return y[burn + p :]


def main():
    fixtures = {}

    # -- algebra_2var: hard-coded reduced form -------------------------------
    a1 = np.array([[0.5, 0.1], [0.2, 0.3]])
    a2 = np.array([[0.10, 0.00], [0.05, 0.10]])
    sig2 = np.array([[1.0, 0.3], [0.3, 0.5]])
    fixtures["algebra_2var"] = pack_case([a1, a2], sig2, 12)

    # -- flip_2var: a reduced form whose default impact B has a negative -----
    # diagonal (A[0,0] > 1 makes D[0,0] < 0), so normalize="impact" genuinely
    # flips column 0. Stored twice (default + impact-normalized) off the SAME
    # reduced form to test both conventions and their relationship.
    f1 = np.array([[1.30, 0.10], [0.20, 0.30]])
    sigf = np.array([[1.0, 0.3], [0.3, 0.5]])
    fixtures["flip_2var"] = pack_case([f1], sigf, 12)
    fixtures["flip_2var_impact"] = pack_case([f1], sigf, 12, normalize_impact=True)

    # -- algebra_3var: hard-coded reduced form -------------------------------
    b1 = np.array(
        [[0.40, 0.10, 0.00], [0.05, 0.30, 0.10], [0.00, 0.10, 0.20]]
    )
    b2 = np.array(
        [[0.10, 0.00, 0.00], [0.00, 0.10, 0.00], [0.00, 0.00, 0.10]]
    )
    sig3 = np.array(
        [[1.00, 0.20, 0.10], [0.20, 0.80, 0.15], [0.10, 0.15, 0.60]]
    )
    fixtures["algebra_3var"] = pack_case([b1, b2], sig3, 12)

    # -- estimated_2var: simulate, OLS-fit, then identify --------------------
    data = simulate_var([a1, a2], sig2, n=400, seed=20260722)
    coefs_hat, sigma_hat = ols_var(data, p=2)
    fixtures["estimated_2var"] = pack_case(coefs_hat, sigma_hat, 12, data=data)

    # -- sanity assertions on the primary case (no tsecon involved) ----------
    res = long_run_svar_closed_form([a1, a2], sig2, 12)
    b = res["impact"]
    lr = res["long_run"]
    c1 = res["long_run_multiplier"]
    # P1: LR strictly-upper-triangle zeros.
    assert abs(lr[0, 1]) < 1e-14
    # P2: B B' reconstructs Sigma_u.
    assert np.allclose(b @ b.T, sig2, atol=1e-12)
    # P3: C1 B == LR and LR LR' == C1 Sigma C1'.
    assert np.allclose(c1 @ b, lr, atol=1e-12)
    assert np.allclose(lr @ lr.T, c1 @ sig2 @ c1.T, atol=1e-12)
    # P4: irf[0] == impact.
    assert np.allclose(res["irf"][0], b, atol=1e-14)

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "long_run_svar.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fixtures, fh, indent=2)
        fh.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
