"""Golden fixtures for hetero_svar -- identification through
heteroskedasticity, known-regime case (Rigobon 2003; Lanne-Lutkepohl 2008).

VALIDATION STRATEGY
===================
Nothing here imports tsecon. Every stored number is an INDEPENDENT
numpy/scipy reference for the exact estimator the Rust crate implements,
so reproducing these numbers in Rust is a genuine cross-implementation
check.

The estimator, reproduced here from scratch:

  1. Simulate a stable VAR(1), k = 3, with a KNOWN constant impact matrix
     B_true and TWO exogenous variance regimes: regime 1 has structural
     shock covariance I, regime 2 has diag(Lambda_true) with distinct
     entries (0.5, 2.0, 5.0). u_t = B_true e_t, y_t = c + A_1 y_{t-1} + u_t.

  2. Fit ONE pooled reduced-form VAR(1) with a constant by OLS (normal
     equations); form residuals; drop the first p labels so residual row r
     aligns to label[p + r]; split residuals by regime.

  3. Decomposition covariances (ML divisor, raw residuals -- mean ~ 0):
        Sigma_s = U_s' U_s / n_s.
     Recover (B, Lambda) from the generalized symmetric-definite pencil
     (Sigma_2, Sigma_1) via scipy.linalg.eigh(a=Sigma_2, b=Sigma_1):
     eigenvalues w ascending, eigenvectors V normalized so V' Sigma_1 V = I,
     hence B_ref = inv(V.T) and Lambda = w (the variance ratios). Apply the
     max-abs column-sign canonicalization.
        This is the same LAPACK sygv Cholesky-whitening the Rust route uses,
     so agreement is exact algebra, not a re-derivation.

  4. Structural IRF: Psi_0 = I, Psi_h = sum_i Psi_{h-i} A_i (reduced-form MA
     weights from the OLS coefficients); Theta_h = Psi_h @ B_ref.

  5. Box's M test of Sigma_1 = Sigma_2 (Bartlett-corrected), with the
     UNBIASED, mean-subtracted within-regime covariances (divisor nu_s =
     n_s - 1) and scipy.stats.chi2.sf.

THE DGP
-------
k = 3, T1 = T2 = 4000 residual-aligned observations after a 200-row
burn-in, stability asserted in file. The data are DERIVED from a seeded RNG
(numpy.random.default_rng); nothing is a redistributed dataset.

WHAT IS STORED (lags p = 1, horizon H = 8, trend = "c", base_regime = 0)
------------------------------------------------------------------------
  data                  : (T1+T2) x 3 estimation sample
  regime_labels         : length T1+T2 integer labels (0, then 1)
  sigma1, sigma2        : 3x3 ML within-regime covariances (decomposition)
  s1_boxm, s2_boxm      : 3x3 unbiased mean-subtracted covariances (Box's M)
  n1, n2                : residual counts per regime
  B                     : 3x3 reference impact matrix (canonicalized)
  variance_ratios       : length-3 eigenvalues (ascending)
  min_ratio_gap         : min_{i<j} |lambda_i - lambda_j|
  ratio_dist_from_unity : |lambda_j - 1|
  psi                   : (H+1) x 3 x 3 reduced-form MA weights
  structural_irf        : (H+1) x 3 x 3, Theta_h = Psi_h @ B
  box_m                 : {statistic, dof, pvalue, distinct_regimes}
  identified            : bool
  mc_b_true_canonical   : 3x3 true B, columns reordered to ascending
                          variance ratio and sign-canonicalized (MC target)
  mc_variance_ratios_true : sorted(Lambda_true) (MC target)

Array [h][i][j] is the response of variable i to shock j at horizon h, the
same layout as tsecon.var_irf and statsmodels.
"""

import json
import os

import numpy as np
import scipy.linalg as sla
from scipy.stats import chi2

SEED = 20260722
OUT = os.path.join(os.path.dirname(__file__), "hetero_svar.json")

N = 3
T1 = 4000
T2 = 4000
BURN = 200
LAGS = 1
HORIZON = 8

A1 = np.array(
    [[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]], dtype=float
)
C = np.array([0.2, -0.1, 0.05], dtype=float)
B_TRUE = np.array(
    [[1.0, 0.5, -0.3], [0.4, 1.0, 0.2], [-0.2, 0.3, 1.0]], dtype=float
)
LAMBDA_TRUE = np.array([0.5, 2.0, 5.0], dtype=float)


def canon_maxabs(b):
    """Max-|entry| per column made positive; ties -> smallest row index."""
    out = b.copy()
    for j in range(out.shape[1]):
        istar = int(np.argmax(np.abs(out[:, j])))  # argmax: first max on ties
        if out[istar, j] < 0.0:
            out[:, j] = -out[:, j]
    return out


def simulate(rng):
    """Simulate the two-regime VAR(1); return (data, labels)."""
    total = T1 + T2
    y = np.zeros(N)
    data = np.zeros((total, N))
    labels = np.zeros(total, dtype=int)
    for t in range(BURN + total):
        idx = t - BURN
        # regime is by observation index (post burn-in); regime 2 has scaled
        # shock variances. During burn-in use regime-1 variances.
        in_regime2 = idx >= T1
        sd = np.sqrt(LAMBDA_TRUE) if in_regime2 else np.ones(N)
        e = rng.standard_normal(N) * sd
        u = B_TRUE @ e
        y = C + A1 @ y + u
        if idx >= 0:
            data[idx] = y
            labels[idx] = 1 if in_regime2 else 0
    return data, labels


def ols_var1(data):
    """OLS VAR(1) with a constant; return (A1_hat, residuals U)."""
    y_full = data[LAGS:]  # (T-1) x N targets
    lagged = data[:-LAGS]  # (T-1) x N
    t_eff = y_full.shape[0]
    x = np.column_stack([np.ones(t_eff), lagged])  # (T-1) x (1+N)
    # Normal equations: coef (1+N) x N, coef[0]=const, coef[1:,r]=A1[r,c].
    coef, *_ = np.linalg.lstsq(x, y_full, rcond=None)
    a1_hat = coef[1:, :].T.copy()
    resid = y_full - x @ coef
    return a1_hat, resid


def ma_rep(coefs, horizon):
    """Reduced-form MA weights Psi_0..Psi_horizon; Psi_0 = I."""
    p = len(coefs)
    k = coefs[0].shape[0]
    psi = [np.eye(k)]
    for h in range(1, horizon + 1):
        acc = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            acc = acc + psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return psi


def box_m(cov_list, n_list):
    """Bartlett-corrected Box's M test; covs use divisor nu_s = n_s - 1."""
    g = len(cov_list)
    d = cov_list[0].shape[0]
    nus = [n - 1 for n in n_list]
    sum_nu = float(sum(nus))
    pooled = sum(nu * s for nu, s in zip(nus, cov_list)) / sum_nu
    _, ld_pooled = np.linalg.slogdet(pooled)
    m_stat = sum_nu * ld_pooled
    for nu, s in zip(nus, cov_list):
        _, ld = np.linalg.slogdet(s)
        m_stat -= nu * ld
    c1 = ((2 * d * d + 3 * d - 1) / (6.0 * (d + 1) * (g - 1))) * (
        sum(1.0 / nu for nu in nus) - 1.0 / sum_nu
    )
    statistic = (1.0 - c1) * m_stat
    dof = (g - 1) * d * (d + 1) // 2
    pvalue = float(chi2.sf(statistic, dof))
    return float(statistic), int(dof), pvalue


def main():
    rng = np.random.default_rng(SEED)
    data, labels = simulate(rng)

    a1_hat, resid = ols_var1(data)
    resid_labels = labels[LAGS:]  # residual row r <-> label[p + r]

    u1 = resid[resid_labels == 0]
    u2 = resid[resid_labels == 1]
    n1, n2 = u1.shape[0], u2.shape[0]

    # Decomposition covariances: ML divisor, raw residuals (mean ~ 0).
    sigma1 = (u1.T @ u1) / n1
    sigma2 = (u2.T @ u2) / n2

    # Box's M covariances: unbiased divisor, mean-subtracted.
    s1_boxm = np.cov(u1, rowvar=False, bias=False)
    s2_boxm = np.cov(u2, rowvar=False, bias=False)

    # Generalized symmetric-definite pencil (Sigma_2, Sigma_1).
    w, v = sla.eigh(sigma2, sigma1)  # w ascending, v' Sigma_1 v = I
    b_ref = canon_maxabs(np.linalg.inv(v.T))
    variance_ratios = w.copy()
    gaps = [
        abs(variance_ratios[i] - variance_ratios[j])
        for i in range(N)
        for j in range(i + 1, N)
    ]
    min_ratio_gap = float(min(gaps))
    ratio_dist_from_unity = [float(abs(l - 1.0)) for l in variance_ratios]

    # Structural IRF.
    psi = ma_rep([a1_hat], HORIZON)
    theta = [p @ b_ref for p in psi]

    # Box's M.
    bm_stat, bm_dof, bm_p = box_m([s1_boxm, s2_boxm], [n1, n2])
    distinct = bool(bm_p < 0.05)

    # identified heuristic.
    max_lam = float(max(abs(l) for l in variance_ratios))
    tol = 1e-6 * max(1.0, max_lam)
    identified = bool(
        min_ratio_gap > tol
        and min(abs(l - 1.0) for l in variance_ratios) > tol
    )

    # MC recovery target: true B reordered to ascending Lambda, canonicalized.
    order = np.argsort(LAMBDA_TRUE)
    b_true_ord = B_TRUE[:, order]
    mc_b_true_canonical = canon_maxabs(b_true_ord)
    mc_variance_ratios_true = [float(x) for x in np.sort(LAMBDA_TRUE)]

    # Stability check (companion of a VAR(1) is just A1).
    roots = np.abs(np.linalg.eigvals(a1_hat))
    assert roots.max() < 1.0, f"unstable VAR: max root {roots.max()}"

    fixture = {
        "_meta": {
            "seed": SEED,
            "n": N,
            "lags": LAGS,
            "horizon": HORIZON,
            "trend": "c",
            "base_regime": 0,
            "sign_normalization": "max",
            "description": "hetero_svar golden: numpy/scipy generalized-eig "
            "reference for Rigobon (2003) known-regime heteroskedasticity ID",
        },
        "data": data.tolist(),
        "regime_labels": [int(x) for x in labels],
        "sigma1": sigma1.tolist(),
        "sigma2": sigma2.tolist(),
        "s1_boxm": s1_boxm.tolist(),
        "s2_boxm": s2_boxm.tolist(),
        "n1": int(n1),
        "n2": int(n2),
        "B": b_ref.tolist(),
        "variance_ratios": [float(x) for x in variance_ratios],
        "min_ratio_gap": min_ratio_gap,
        "ratio_dist_from_unity": ratio_dist_from_unity,
        "psi": [p.tolist() for p in psi],
        "structural_irf": [t.tolist() for t in theta],
        "box_m": {
            "statistic": bm_stat,
            "dof": bm_dof,
            "pvalue": bm_p,
            "distinct_regimes": distinct,
        },
        "identified": identified,
        "mc_b_true_canonical": mc_b_true_canonical.tolist(),
        "mc_variance_ratios_true": mc_variance_ratios_true,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    print(
        f"n1={n1} n2={n2} ratios={variance_ratios} gap={min_ratio_gap:.4f} "
        f"boxM={bm_stat:.2f} p={bm_p:.2e} identified={identified}"
    )
    # Sanity: reference recovers the truth (MC consistency, large T).
    lam_err = np.max(np.abs(np.array(variance_ratios) - np.sort(LAMBDA_TRUE)))
    b_err = np.max(np.abs(b_ref - mc_b_true_canonical))
    print(f"MC recovery: max|lambda-truth|={lam_err:.4e} max|B-truth|={b_err:.4e}")


if __name__ == "__main__":
    main()
