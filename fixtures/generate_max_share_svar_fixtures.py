"""Golden fixtures for max_share_svar (tsecon-ident): the max-share / maximum-FEV
structural shock.

METHOD
======
Max-share identification (Uhlig 2004 penalty-free eigenvalue variant;
Francis-Owyang-Roush-DiCecio 2014 finite-horizon main-business-cycle shock;
Barsky-Sims 2011 news shock). Identify the single structural shock whose
contribution to the forecast-error variance (FEV) of a TARGET variable,
accumulated over a horizon window [h0, h1], is maximal. This is a CLOSED-FORM
identification: the identified impact direction is the leading eigenvector of a
small symmetric PSD matrix built from the cumulated orthogonalized MA (IRF)
coefficients. No RNG, no rejection sampling, no iteration.

VALIDATION STRATEGY
===================
Every number this file writes is produced by an INDEPENDENT reference --
numpy.linalg.lstsq for the OLS reduced form, numpy.linalg.cholesky for the
orthogonalization, and numpy.linalg.eigh for the eigenproblem -- never by the
tsecon Rust crate, so reproducing these numbers in Rust is a genuine
cross-implementation check. The data are DERIVED from a seeded numpy structural
DGP (no redistributed datasets); this file NEVER imports tsecon.

THE ALGEBRA (mirrored bit-for-bit by crates/tsecon-ident/src/max_share.rs)
--------------------------------------------------------------------------
Reduced-form VAR(p):  y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,
E[u_t u_t'] = Sigma.  MA(inf):  Psi_0 = I, Psi_s = sum_i Psi_{s-i} A_i.
P = lower Cholesky of Sigma.  Orthogonalized IRF Theta_s = Psi_s P (Theta_0 = P).
Let r_s = Theta_s[target, :]' (the target ROW as a k-vector).

  weighting="window"     (Uhlig / Francis):  A = sum_{s=h0..h1} r_s r_s'.
  weighting="cumulative" (Barsky-Sims):      A = sum_{h=h0..h1} C_h / tr(C_h),
                                             C_h = sum_{s=0..h} r_s r_s'.

Objective  max_{q'q=1} q' A q  =>  q* = leading eigenvector of A (largest
eigenvalue lambda_max).  Identified impact b* = P q* = Theta_0 q*; structural
IRF irf[s] = Theta_s q*.  The shock is unit-variance ((P q*)'Sigma^{-1}(P q*)=1).

share_window = lambda_max / tr(A).  For "window" tr(A) = D, the total windowed
FEV of the target, so the share is an exact accumulated-FEV fraction.  For
"cumulative" tr(A) = (h1 - h0 + 1) identically, so the share is the window-MEAN
cumulative share.

exclude_impact=True (Barsky-Sims news): require e_target' Theta_0 q = 0, i.e.
c'q = 0 with c = r_0 (the target row of P).  Project: N (k x (k-1)) spans
null(c') via the eigenvalue-1 subspace of the projector I - c c'/(c'c); solve the
(k-1)x(k-1) eigenproblem on N'AN; q* = N z*.  The reported eigenvalues are those
of N'AN (basis-independent: any orthonormal N spanning null(c') gives the same
q* up to sign and the same eigenvalues).

SIGN NORMALIZATION (q* is defined only up to sign; MUST be pinned identically on
both sides or the golden is unreproducible).  sign="cumsum": flip q* (and irf) so
sum_{s=h0..h1} (Theta_s q*)[target] >= 0.

fev_share[h] = (sum_{s=0..h} (Theta_s q*)[target]^2) /
               (sum_{s=0..h} sum_j Theta_s[target, j]^2),   h = 0..horizon.

Run with the project venv:
    .venv/bin/python fixtures/generate_max_share_svar_fixtures.py
"""

import json

import numpy as np
import scipy

OUT = "fixtures/max_share_svar.json"

# --------------------------------------------------------------------------- #
# Structural DGP: k=3 VAR(2). Variable 2 is a persistent "driver" whose
# structural shock, propagated through the dynamics, comes to dominate the
# target's windowed forecast-error variance -- giving a clear spectral gap.
# --------------------------------------------------------------------------- #
A1 = np.array(
    [[0.30, 0.00, 0.55], [0.00, 0.20, 0.00], [0.00, 0.00, 0.90]]
)
A2 = np.array(
    [[-0.10, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, -0.05]]
)
# Structural impact matrix (lower-triangular => recursive/Cholesky ordering).
P_TRUE = np.array([[1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.3, 1.2]])
K = 3
SEED = 20260722
T = 500


def simulate(seed, t, a1, a2, p, k, burn=200):
    rng = np.random.default_rng(seed)
    n = t + burn
    eps = rng.standard_normal((n, k))
    y = np.zeros((n, k))
    for tt in range(2, n):
        y[tt] = a1 @ y[tt - 1] + a2 @ y[tt - 2] + p @ eps[tt]
    return y[burn:]


def ols_var(data, p):
    """Independent OLS reduced form. df-adjusted Sigma matching VarResults.sigma_u.

    Returns the packed regressor-coefficient matrix B (rows = 1 + k*p in the
    order [const, lag1 block, ..., lagp block]; cols = k) and Sigma.
    """
    t, k = data.shape
    design = np.column_stack(
        [np.ones(t - p)] + [data[p - l : t - l] for l in range(1, p + 1)]
    )
    yt = data[p:]
    b, *_ = np.linalg.lstsq(design, yt, rcond=None)
    u = yt - design @ b
    nobs = t - p
    n_params = 1 + k * p
    sigma = u.T @ u / (nobs - n_params)
    return b, sigma


def theta_series(b, sigma, p, horizon):
    """Orthogonalized MA coefficients Theta_s = Psi_s P, s = 0..horizon."""
    k = sigma.shape[0]
    # A_l[i, j] = coefficient of y_{t-l, j} in equation i.
    a_mats = [b[1 + l * k : 1 + (l + 1) * k, :].T for l in range(p)]
    npp = k * p
    comp = np.zeros((npp, npp))
    for l in range(p):
        comp[:k, l * k : (l + 1) * k] = a_mats[l]
    if p > 1:
        comp[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    p_chol = np.linalg.cholesky(sigma)
    thetas = []
    f_pow = np.eye(npp)
    for _h in range(horizon + 1):
        thetas.append(f_pow[:k, :k] @ p_chol)
        f_pow = comp @ f_pow
    return thetas, p_chol


def objective_matrix(thetas, target, h0, h1, weighting):
    """Symmetric PSD objective matrix A (window or cumulative)."""
    k = thetas[0].shape[1]
    if weighting == "window":
        a = np.zeros((k, k))
        for s in range(h0, h1 + 1):
            r = thetas[s][target, :]
            a += np.outer(r, r)
        return a
    if weighting == "cumulative":
        a = np.zeros((k, k))
        c_h = np.zeros((k, k))
        d_h = 0.0
        for h in range(h1 + 1):
            r = thetas[h][target, :]
            c_h = c_h + np.outer(r, r)
            d_h = d_h + float(r @ r)
            if h >= h0:
                if d_h <= 0.0:
                    raise ValueError("cumulative denominator d_h must be positive")
                a += c_h / d_h
        return a
    raise ValueError(f"unknown weighting {weighting!r}")


def null_basis(c):
    """Orthonormal basis N (k x (k-1)) of null(c') via the projector's
    eigenvalue-1 subspace (matches the Rust construction)."""
    k = c.shape[0]
    pperp = np.eye(k) - np.outer(c, c) / (c @ c)
    _w, v = np.linalg.eigh(pperp)  # ascending: [0, 1, ..., 1]
    return v[:, 1:]  # eigenvalue-1 columns


def solve_case(thetas, target, h0, h1, horizon, weighting, exclude_impact, sign):
    k = thetas[0].shape[1]
    a = objective_matrix(thetas, target, h0, h1, weighting)
    trace_a = float(np.trace(a))
    if exclude_impact:
        c = thetas[0][target, :].copy()  # target row of P
        nbasis = null_basis(c)
        a_red = nbasis.T @ a @ nbasis
        w, v = np.linalg.eigh(a_red)  # ascending
        z = v[:, -1]
        q = nbasis @ z
        eigenvalues = w  # k-1 eigenvalues of N'AN, ascending
        lambda_max = float(w[-1])
    else:
        w, v = np.linalg.eigh(a)  # ascending
        q = v[:, -1]
        eigenvalues = w  # k eigenvalues of A, ascending
        lambda_max = float(w[-1])

    # Structural IRF and impact.
    irf = [thetas[s] @ q for s in range(horizon + 1)]  # each length k

    # Sign normalization.
    if sign == "cumsum":
        s_target = sum(irf[s][target] for s in range(h0, h1 + 1))
        if s_target < 0.0:
            q = -q
            irf = [-v for v in irf]
    elif sign == "impact":
        if exclude_impact:
            raise ValueError("sign='impact' invalid with exclude_impact=True")
        if irf[0][target] < 0.0:
            q = -q
            irf = [-v for v in irf]
    elif sign == "none":
        pass
    else:
        raise ValueError(f"unknown sign {sign!r}")

    impact = irf[0].copy()

    # share_window.
    if weighting == "window":
        share_window = lambda_max / trace_a
    else:  # cumulative: trace_a == (h1 - h0 + 1) identically
        share_window = lambda_max / trace_a

    # fev_share profile.
    fev_share = []
    num = 0.0
    den = 0.0
    for h in range(horizon + 1):
        num += float(irf[h][target] ** 2)
        r = thetas[h][target, :]
        den += float(r @ r)
        fev_share.append(num / den)

    return {
        "irf": [v.tolist() for v in irf],
        "impact": impact.tolist(),
        "q": q.tolist(),
        "share_window": float(share_window),
        "fev_share": [float(x) for x in fev_share],
        "eigenvalues": [float(x) for x in eigenvalues],
    }


def make_case(name, data, b, sigma, thetas, p_chol, params):
    exp = solve_case(
        thetas,
        params["target"],
        params["h0"],
        params["h1"],
        params["horizon"],
        params["weighting"],
        params["exclude_impact"],
        params["sign"],
    )
    return {"name": name, "params": params, "expected": exp}


def main():
    data = simulate(SEED, T, A1, A2, P_TRUE, K)
    horizon = 40
    b, sigma = ols_var(data, 2)
    thetas, p_chol = theta_series(b, sigma, 2, horizon)

    def params(target, h0, h1, weighting, exclude_impact, sign):
        return {
            "lags": 2,
            "target": target,
            "h0": h0,
            "h1": h1,
            "horizon": horizon,
            "trend": "c",
            "weighting": weighting,
            "exclude_impact": exclude_impact,
            "sign": sign,
        }

    cases = [
        make_case(
            "window_main",
            data, b, sigma, thetas, p_chol,
            params(0, 0, 40, "window", False, "cumsum"),
        ),
        make_case(
            "news_target0_exclude_impact",
            data, b, sigma, thetas, p_chol,
            params(0, 0, 40, "window", True, "cumsum"),
        ),
        make_case(
            "cumulative_target0",
            data, b, sigma, thetas, p_chol,
            params(0, 0, 40, "cumulative", False, "cumsum"),
        ),
        make_case(
            "exclude_impact_target1_general",
            data, b, sigma, thetas, p_chol,
            params(1, 0, 40, "window", True, "cumsum"),
        ),
    ]

    fixture = {
        "_meta": {
            "description": "Golden fixtures for max_share_svar (max-share / maximum-FEV "
            "structural shock; Uhlig 2004 / Francis et al 2014 / Barsky-Sims 2011).",
            "references": {
                "reduced_form": "numpy.linalg.lstsq OLS; Sigma = U'U/(nobs - (1+k*p)) "
                "(df-adjusted, matching VarResults.sigma_u)",
                "orth_ma": "Theta_s = Psi_s P, P = numpy.linalg.cholesky(Sigma) "
                "(== VarResults.orth_ma_rep)",
                "eigen": "numpy.linalg.eigh of the symmetric PSD objective matrix; "
                "leading eigenvector; sign pinned by the shared cumsum rule",
                "exclude_impact": "projection onto null(c'), c = target row of P; "
                "eigenproblem on N'AN (basis-independent q* and eigenvalues)",
            },
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "seed": SEED,
        },
        # Shared inputs (one dataset; params vary per case).
        "k": K,
        "horizon": horizon,
        "data": data.tolist(),  # T x k  (for the end-to-end binding test)
        "reg_coefs": b.tolist(),  # (1 + k*p) x k packed OLS coefficients
        "sigma": sigma.tolist(),  # k x k df-adjusted innovation covariance
        "theta": [t.tolist() for t in thetas],  # (horizon+1) x k x k orth MA
        "cases": [c for c in cases],
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    for c in cases:
        e = c["expected"]
        p = c["params"]
        print(
            f"  {c['name']}: target={p['target']} [{p['h0']},{p['h1']}] "
            f"{p['weighting']} excl={p['exclude_impact']} "
            f"share={e['share_window']:.6f} n_eig={len(e['eigenvalues'])} "
            f"top2={e['eigenvalues'][-2:]}"
        )


if __name__ == "__main__":
    main()
