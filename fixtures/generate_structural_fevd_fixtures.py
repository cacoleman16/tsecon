"""Golden fixtures for structural_fevd (tsecon-ident): forecast-error variance
decomposition for an ARBITRARY structural impact matrix A0 (= P Q).

METHOD
======
tsecon-var::fevd is recursive-Cholesky ONLY (impact hardcoded to P = chol(Sigma)).
structural_fevd closes the gap: given the reduced form plus ANY admissible
structural impact matrix A0 (columns = one-standard-deviation structural impulse
vectors, A0 A0' = Sigma; from sign / zero / proxy / max-share / long-run
identification, any of which yields a non-triangular A0 = P Q), it returns the
per-(variable, shock, horizon) forecast-error-variance shares.

Structural MA weights:  Theta_s = Psi_s A0   (Psi_0 = I, Psi_s the reduced-form
MA weights;  Theta_0 = A0).  Share of structural shock j in the (h+1)-step
forecast-error variance of variable i:

  omega_{ij}(h) = [ sum_{s=0..h} Theta_s[i,j]^2 ]
                / [ sum_{m=0..n-1} sum_{s=0..h} Theta_s[i,m]^2 ].

KEY INVARIANTS (pinned by the crate test):
  * sum_j omega_{ij}(h) = 1 exactly, for every (i, h) and ANY A0.
  * The denominator (the i-th diagonal of the (h+1)-step forecast MSE) is
    ROTATION-INVARIANT: A0 A0' = P Q Q' P' = Sigma, so replacing P by A0 = P Q
    changes only the split across shocks j, never the total.  Hence the
    per-(i, h) MSE diagonal computed from Theta^chol equals the one from
    Theta^general.
  * Column sign flips of A0 leave omega unchanged (the responses enter squared).
  * When A0 = P (Q = I) this reduces EXACTLY to Lutkepohl (2005, eq. 2.3.37),
    i.e. statsmodels VARResults.fevd and tsecon-var var_fevd.

VALIDATION STRATEGY
===================
The Cholesky reference is statsmodels VARResults.fevd -- a fully INDEPENDENT
implementation -- so reproducing it in Rust is a genuine cross-implementation
check (not circular). The general-A0 shares (no external reference exists) are an
independent NumPy running-sum computation and are additionally pinned by the
exact algebraic invariants above. Data are DERIVED from a seeded NumPy structural
DGP; this file NEVER imports tsecon.

Run with the project venv:
    .venv/bin/python fixtures/generate_structural_fevd_fixtures.py
"""

import json

import numpy as np
import scipy
import statsmodels
from statsmodels.tsa.api import VAR

OUT = "fixtures/structural_fevd.json"

# --------------------------------------------------------------------------- #
# Structural DGP: n=3 VAR(2), a clearly stable, non-diagonal system so every
# variable's FEV is genuinely shared across shocks (no degenerate rows).
# --------------------------------------------------------------------------- #
A1_TRUE = np.array([[0.50, 0.10, 0.00], [0.05, 0.40, 0.10], [0.00, 0.05, 0.30]])
A2_TRUE = np.array([[-0.10, 0.00, 0.05], [0.00, -0.05, 0.00], [0.05, 0.00, -0.10]])
# Structural impact for the DGP (lower-triangular; recursive ordering).
P_TRUE = np.array([[1.00, 0.00, 0.00], [0.40, 0.90, 0.00], [0.20, 0.30, 0.70]])
N = 3
LAGS = 2
HORIZON = 10  # FEVD for horizons 0..HORIZON (HORIZON+1 periods)
SEED = 20260722
T = 200


def simulate(seed, t, a1, a2, p, n, burn=200):
    rng = np.random.default_rng(seed)
    m = t + burn
    eps = rng.standard_normal((m, n))
    y = np.zeros((m, n))
    for tt in range(2, m):
        y[tt] = a1 @ y[tt - 1] + a2 @ y[tt - 2] + p @ eps[tt]
    return y[burn:]


def packed_coefs(coefs, intercept, n, p):
    """Pack statsmodels coefs/intercept into the crate regressor layout:
    b is (1 + n*p) x n; row 0 = intercept; the lag-l block occupies rows
    1+(l-1)*n .. 1+l*n with b[1+(l-1)*n + j, i] = A_l[i, j] (coefficient of
    y_{t-l, j} in equation i).  This is exactly what cholesky_irf / structural_ma
    read (crates/tsecon-ident/src/summary.rs)."""
    b = np.zeros((1 + n * p, n))
    b[0, :] = intercept
    for l in range(1, p + 1):
        al = coefs[l - 1]  # A_l, shape (n, n): A_l[i, j]
        for i in range(n):
            for j in range(n):
                b[1 + (l - 1) * n + j, i] = al[i, j]
    return b


def haar_orthogonal(seed, n):
    """A random orthogonal Q via QR of a standard-normal matrix with the
    Stewart/Mezzadri R-diagonal sign fix (Haar-uniform on O(n))."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diag(r))
    d[d == 0.0] = 1.0
    return q * d  # column-scale by sign(diag(R))


def structural_ma(psi, a0):
    """Theta_s = Psi_s @ A0 for the stored reduced-form MA weights psi
    (list of (H+1) n x n arrays, psi[0] = I)."""
    return [ps @ a0 for ps in psi]


def fevd_from_theta(theta, n, horizon):
    """Independent NumPy running-sum FEVD.
    Returns (omega, mse_diag):
      omega[h][i][j]  = share of shock j in variable i's (h+1)-step FE variance;
      mse_diag[h][i]  = the i-th (h+1)-step forecast MSE diagonal (denominator).
    """
    omega = []
    mse_diag = []
    cum = np.zeros((n, n))
    for h in range(horizon + 1):
        cum = cum + theta[h] ** 2  # elementwise; accumulate squares
        share_h = np.zeros((n, n))
        diag_h = np.zeros(n)
        for i in range(n):
            mse_i = float(cum[i, :].sum())
            diag_h[i] = mse_i
            if mse_i <= 0.0:
                raise ValueError(f"non-positive MSE diagonal at (h={h}, i={i})")
            share_h[i, :] = cum[i, :] / mse_i
        omega.append(share_h)
        mse_diag.append(diag_h)
    return omega, mse_diag


def main():
    data = simulate(SEED, T, A1_TRUE, A2_TRUE, P_TRUE, N)

    # Independent reduced form via statsmodels (df-adjusted sigma_u == the
    # tsecon VarResults.sigma_u convention: SSE / (nobs - (1 + n*p))).
    res = VAR(data).fit(maxlags=LAGS, trend="c")
    coefs = res.coefs  # (p, n, n), A_l[i, j]
    intercept = res.intercept  # (n,)
    sigma = np.asarray(res.sigma_u)  # (n, n), df-adjusted
    b = packed_coefs(coefs, intercept, N, LAGS)
    p_chol = np.linalg.cholesky(sigma)  # lower Cholesky (== faer/numpy lower)

    # Reduced-form MA weights Psi_s and the orthogonalized MA Theta^chol = Psi_s P.
    psi = res.ma_rep(HORIZON)  # (H+1, n, n), psi[0] = I
    theta_chol = structural_ma(psi, p_chol)

    # ------- Cholesky-case FEVD: cross-check statsmodels vs our running sum ----
    fevd_chol, mse_chol = fevd_from_theta(theta_chol, N, HORIZON)
    sm = res.fevd(HORIZON + 1).decomp  # (neqs, periods, neqs): sm[i, h, j]
    sm_reshaped = np.transpose(sm, (1, 0, 2))  # -> [h][i][j]
    max_sm_gap = float(np.max(np.abs(np.array(fevd_chol) - sm_reshaped)))
    assert max_sm_gap < 1e-12, f"running-sum FEVD disagrees with statsmodels: {max_sm_gap}"

    # ------- General A0 = P Q (non-triangular): the gap var_fevd cannot cover --
    q_rot = haar_orthogonal(SEED + 7, N)
    a0_general = p_chol @ q_rot  # A0 A0' = P Q Q' P' = Sigma
    theta_general = structural_ma(psi, a0_general)
    fevd_general, mse_general = fevd_from_theta(theta_general, N, HORIZON)

    # Rotation invariance of the denominator (sanity self-check in the generator).
    max_mse_gap = float(np.max(np.abs(np.array(mse_chol) - np.array(mse_general))))
    assert max_mse_gap < 1e-10, f"MSE diagonal not rotation-invariant: {max_mse_gap}"
    # Rows sum to 1.
    row_gap = float(np.max(np.abs(np.array(fevd_general).sum(axis=2) - 1.0)))
    assert row_gap < 1e-12, f"general FEVD rows do not sum to 1: {row_gap}"

    fixture = {
        "_meta": {
            "description": "Golden fixtures for structural_fevd (FEVD for a general "
            "structural impact A0 = P Q; the gap tsecon-var::fevd, recursive-Cholesky "
            "only, leaves). Cholesky case cross-checked against statsmodels "
            "VARResults.fevd; general-A0 case pinned by exact algebraic invariants.",
            "references": {
                "reduced_form": "statsmodels VAR(data).fit(2, 'c'); sigma_u df-adjusted "
                "(SSE/(nobs-(1+n*p)), matching tsecon VarResults.sigma_u)",
                "orth_ma": "Theta_s = Psi_s P, Psi_s = res.ma_rep, P = numpy.linalg.cholesky",
                "fevd_chol": "statsmodels VARResults.fevd(periods).decomp (independent), "
                "verified == the NumPy running-sum FEVD to < 1e-12",
                "fevd_general": "NumPy running-sum FEVD of Theta_s = Psi_s (P Q); Q Haar "
                "orthogonal (Stewart/Mezzadri sign fix); pinned by row-sum + "
                "rotation-invariant-denominator + sign-flip invariants",
            },
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "seed": SEED,
        },
        "n": N,
        "lags": LAGS,
        "horizon": HORIZON,
        "data": data.tolist(),  # T x n (for an end-to-end binding test)
        "reg_coefs": b.tolist(),  # (1 + n*p) x n packed OLS coefficients
        "sigma": sigma.tolist(),  # n x n df-adjusted innovation covariance
        "chol": p_chol.tolist(),  # n x n lower Cholesky of sigma (= default A0)
        # Cholesky case (A0 = P):
        "theta_chol": [t.tolist() for t in theta_chol],  # (H+1) x n x n
        "fevd_chol": [f.tolist() for f in fevd_chol],  # (H+1) x n x n
        "fevd_statsmodels": sm_reshaped.tolist(),  # (H+1) x n x n (== fevd_chol)
        "mse_diag_chol": [m.tolist() for m in mse_chol],  # (H+1) x n
        # General case (A0 = P Q, non-triangular):
        "q_rot": q_rot.tolist(),  # n x n orthogonal
        "impact_general": a0_general.tolist(),  # n x n = P Q
        "theta_general": [t.tolist() for t in theta_general],  # (H+1) x n x n
        "fevd_general": [f.tolist() for f in fevd_general],  # (H+1) x n x n
        "mse_diag_general": [m.tolist() for m in mse_general],  # (H+1) x n
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    print(f"  statsmodels vs running-sum FEVD max gap: {max_sm_gap:.2e}")
    print(f"  denominator rotation-invariance max gap:  {max_mse_gap:.2e}")
    print(f"  general FEVD row-sum max deviation:       {row_gap:.2e}")
    print(f"  sample share fevd_chol[H][0]:    {np.round(fevd_chol[HORIZON][0], 4)}")
    print(f"  sample share fevd_general[H][0]: {np.round(fevd_general[HORIZON][0], 4)}")


if __name__ == "__main__":
    main()
