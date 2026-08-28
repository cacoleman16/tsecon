"""Golden fixtures for the Hansen-Seo (2002) threshold VECM
(`threshold_vecm`) and sup-LM threshold-cointegration test
(`hansen_seo_test`).

Reference: NO third-party Hansen-Seo implementation runs in this container
(R itself installed from apt, but CRAN is unreachable through the egress
proxy, so `tsDyn` — the natural `TVECM`/`TVECM.HStest` reference — cannot
be installed; no Python package implements the estimator). The reference is
therefore built HERE, in NumPy, as an independent transcription of the
published algorithm — Hansen & Seo (2002, Journal of Econometrics 110(2),
293-318), "Testing for two-regime threshold cointegration in vector
error-correction models":

    Delta y_t = A_1' X_{t-1}(b) 1{w_{t-1}(b) <= g} +
                A_2' X_{t-1}(b) 1{w_{t-1}(b) >  g} + u_t,
    X_{t-1}(b) = (1, w_{t-1}(b), Dy_{t-1}', ..., Dy_{t-l}')',
    w_t(b) = b' y_t,  b = (1, b_2, ..., b_k)'.

Transcribed conventions (the Rust implementation states the same rules):

  * usable sample     level rows t = l+1 .. T-1 (0-indexed); n = T - l - 1;
                      m = 2 + k*l regressors, columns [const, ect, lags].
  * gamma grid        tie-grouped order statistics gamma of {w_{t-1}} with
                      min_regime <= #{w <= gamma} <= n - min_regime,
                      min_regime = max(m + 1, ceil(trim * n)) (trim is
                      Hansen-Seo's pi_0; they suggest 0.05), subsampled to
                      at most n_grid_gamma evenly spaced candidates by the
                      exact integer rule
                      idx_j = (2 j (c-1) + (G-1)) // (2 (G-1)).
  * beta grid         (bivariate only) n_grid_beta points spanning the
                      linear rank-1 Johansen ML estimate (unrestricted
                      constant) +/- beta_span first-order standard errors
                      se = [ (a' Su^-1 a) * (R1'R1)_{22} ]^{-1/2}, R1 the
                      lagged levels partialled of [1, lagged differences]
                      — a search-region scale, not an inference formula.
  * per cell          OLS in each regime; criterion ln det SigmaHat with
                      SigmaHat = (E_1 + E_2)/n; FIRST candidate attaining
                      the minimum wins (np.argmin), and across beta values
                      the first strictly smaller criterion wins, iterating
                      the grid in ascending order.
  * reported fit      refit at (b^, g^): per-regime OLS coefficients (rows
                      = equations) with EICKER-WHITE standard errors (no
                      dof correction — the form Hansen-Seo report), pooled
                      ML Sigma, ln det Sigma, and the Gaussian
                      log-likelihood -(nk/2)(ln 2pi + 1) - (n/2) ln det.
  * sup-LM            (their eq. 10-12) with beta FIXED at the null
                      (linear Johansen ML) estimate: per candidate g,
                      x_it = X_t d_it, M_i = X_i'X_i, A^_i = M_i^-1 X_i'U~
                      (U~ the null OLS residuals),
                      Omega_i = sum_i (u~u~') kron (xx'),
                      V_i = (I kron M_i^-1) Omega_i (I kron M_i^-1),
                      LM(g) = vec(A^_1-A^_2)' (V_1+V_2)^-1 vec(...),
                      vec stacking equations as the OUTER index.

Grade (honest): documented-ALGORITHM transcription validated against an
independent NumPy implementation (this file) — NOT a third-party golden.
Fixed-beta cases pin the concentrated-LS machinery at 1e-10; cases that
estimate beta go through the Johansen eigensolver (scipy here, faer in the
crate) and are pinned at 1e-8. The bootstrap p-value is deliberately NOT
pinned — it is a Monte Carlo quantity checked by property (null size ~
nominal) in `tvecm_properties.rs`.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  .venv-wt/bin/python fixtures/generate_tvecm_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.linalg

OUT = Path(__file__).resolve().parent / "tvecm.json"


# ------------------------------------------------------------------ series

def sim_tvecm(seed, T, gamma, low, high, beta2=-1.0, burn=100):
    """Bivariate threshold-cointegrated DGP: the equilibrium error
    w_t = y1 + beta2*y2 is a two-regime TAR (coef vectors [const, rho]),
    y2 is a random walk, y1 = w - beta2*... i.e. y1 = w - beta2*y2."""
    rng = np.random.default_rng(seed)
    n = T + burn
    w = np.zeros(n)
    for t in range(1, n):
        c = low if w[t - 1] <= gamma else high
        w[t] = c[0] + c[1] * w[t - 1] + rng.standard_normal()
    y2 = np.cumsum(0.5 * rng.standard_normal(n))
    y1 = w - beta2 * y2
    return np.column_stack([y1, y2])[burn:]


def sim_linear_vecm(seed, T, rho=0.6, burn=100):
    rng = np.random.default_rng(seed)
    n = T + burn
    w = np.zeros(n)
    for t in range(1, n):
        w[t] = rho * w[t - 1] + rng.standard_normal()
    y2 = np.cumsum(0.5 * rng.standard_normal(n))
    y1 = w + y2
    return np.column_stack([y1, y2])[burn:]


def sim_tvecm3(seed, T, gamma, low, high, burn=100):
    """Trivariate fixed-beta DGP: w = y1 - 0.5 y2 - 0.5 y3 a two-regime
    TAR, y2 and y3 independent random walks."""
    rng = np.random.default_rng(seed)
    n = T + burn
    w = np.zeros(n)
    for t in range(1, n):
        c = low if w[t - 1] <= gamma else high
        w[t] = c[0] + c[1] * w[t - 1] + rng.standard_normal()
    y2 = np.cumsum(0.5 * rng.standard_normal(n))
    y3 = np.cumsum(0.5 * rng.standard_normal(n))
    y1 = w + 0.5 * y2 + 0.5 * y3
    return np.column_stack([y1, y2, y3])[burn:]


def make_series():
    return {
        # Strong threshold cointegration: the equilibrium error is pushed
        # back across gamma = 0 from both sides, so both regimes stay
        # well-populated and (beta, gamma) are sharply identified.
        "tv_strong": sim_tvecm(20260828, 300, 0.0,
                               low=[1.0, 0.7], high=[-1.0, 0.3]),
        # Linear cointegration — null data for the test's behavior.
        "tv_linear": sim_linear_vecm(11, 250),
        # Trivariate, beta = (1, -0.5, -0.5) fixed.
        "tv3": sim_tvecm3(7, 320, 0.2, low=[0.6, 0.75], high=[-0.6, 0.2]),
    }


# ------------------------------------- the transcription (see module doc)

def build_design(y, l, beta):
    """X = [1, w_{t-1}, Dy_{t-1}, ..., Dy_{t-l}], Y = Dy_t, w = w_{t-1}."""
    T, k = y.shape
    p = l + 1
    n = T - p
    dy = np.diff(y, axis=0)                     # T-1 x k, row t-1 = Dy_t
    Y = dy[p - 1:]                              # n x k
    w = y[p - 1:T - 1] @ beta                   # n, w_{t-1}
    cols = [np.ones((n, 1)), w[:, None]]
    for lag in range(1, l + 1):
        cols.append(dy[p - 1 - lag:T - 1 - lag])
    X = np.hstack(cols)
    return X, Y, w


def even_indices(count, n_grid):
    if count <= n_grid:
        return list(range(count))
    out = []
    for j in range(n_grid):
        idx = (2 * j * (count - 1) + (n_grid - 1)) // (2 * (n_grid - 1))
        if not out or out[-1] != idx:
            out.append(idx)
    return out


def candidate_grid(w, m, trim, n_grid):
    n = w.size
    min_regime = max(m + 1, math.ceil(trim * n))
    ws = np.sort(w)
    gammas = np.unique(ws)
    nlow = np.searchsorted(ws, gammas, side="right")
    ok = (nlow >= min_regime) & (nlow <= n - min_regime)
    gammas, nlow = gammas[ok], nlow[ok]
    keep = even_indices(gammas.size, n_grid)
    return gammas[keep], nlow[keep], int(min_regime)


def ols(X, Y):
    B = np.linalg.lstsq(X, Y, rcond=None)[0]    # m x k
    U = Y - X @ B
    return B, U


def logdet_profile(X, Y, w, gammas):
    n = Y.shape[0]
    path = np.empty(gammas.size)
    for i, g in enumerate(gammas):
        lo = w <= g
        _, U1 = ols(X[lo], Y[lo])
        _, U2 = ols(X[~lo], Y[~lo])
        sig = (U1.T @ U1 + U2.T @ U2) / n
        path[i] = np.linalg.slogdet(sig)[1]
    return path


def johansen_rank1(y, l):
    """Linear rank-1 Johansen ML with unrestricted constant: returns the
    normalized beta (beta[0] = 1), the OLS-form log-likelihood, and the
    first-order se of each free beta coefficient (module docstring)."""
    T, k = y.shape
    p = l + 1
    n = T - p
    dy = np.diff(y, axis=0)
    Y = dy[p - 1:]
    ylag = y[p - 1:T - 1]
    cols = [np.ones((n, 1))]
    for lag in range(1, l + 1):
        cols.append(dy[p - 1 - lag:T - 1 - lag])
    Xb = np.hstack(cols)                        # [1, lagged differences]

    def resid(A, B):
        return A - B @ np.linalg.lstsq(B, A, rcond=None)[0]

    R0 = resid(Y, Xb)
    R1 = resid(ylag, Xb)
    S00 = R0.T @ R0 / n
    S01 = R0.T @ R1 / n
    S11 = R1.T @ R1 / n
    Bmat = S01.T @ np.linalg.solve(S00, S01)
    lam, vec = scipy.linalg.eigh(Bmat, S11)
    v = vec[:, np.argmax(lam)]
    beta = v / v[0]

    # OLS at beta for residuals, alpha (= coefficient on the ect), llf.
    X, Y2, w = build_design(y, l, beta)
    Bols, U = ols(X, Y2)
    sigma_u = U.T @ U / n
    llf = -n * k / 2 * (math.log(2 * math.pi) + 1) \
        - n / 2 * np.linalg.slogdet(sigma_u)[1]
    alpha = Bols[1]                             # k, loading on the ect
    q = alpha @ np.linalg.solve(sigma_u, alpha)
    M = R1.T @ R1                               # unnormalized moment
    se = np.full(k, np.nan)
    for j in range(1, k):
        info = q * M[j, j]
        se[j] = 1.0 / math.sqrt(info) if info > 0 else np.nan
    return beta, float(llf), se


def white_se(X, Y):
    """Per-equation Eicker-White SEs of the OLS of Y on X (no dof
    correction), rows = equations."""
    B, U = ols(X, Y)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.empty((Y.shape[1], X.shape[1]))
    for j in range(Y.shape[1]):
        meat = (X * (U[:, j] ** 2)[:, None]).T @ X
        se[j] = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    return B.T, se, U                           # coefs rows = equations


def fit_tvecm(y, l, trim, n_grid_gamma, beta=None,
              n_grid_beta=1, beta_span=0.0):
    T, k = y.shape
    m = 2 + k * l
    n = T - l - 1
    if beta is not None:
        beta = np.asarray(beta, dtype=float)
        beta = beta / beta[0]
        beta_cands = [beta]
        beta_linear = beta
        X0, Y0, _ = build_design(y, l, beta)
        _, U0 = ols(X0, Y0)
        sig0 = U0.T @ U0 / n
        llf_linear = -n * k / 2 * (math.log(2 * math.pi) + 1) \
            - n / 2 * np.linalg.slogdet(sig0)[1]
        beta_grid = []
    else:
        bl, llf_linear, se = johansen_rank1(y, l)
        beta_linear = bl
        center = bl[1]
        if n_grid_beta == 1 or beta_span == 0.0:
            grid = [center]
        else:
            grid = list(np.linspace(center - beta_span * se[1],
                                    center + beta_span * se[1],
                                    n_grid_beta))
        beta_cands = [np.array([1.0, b2]) for b2 in grid]
        beta_grid = [float(b) for b in grid]

    best = None
    for bi, b in enumerate(beta_cands):
        X, Y, w = build_design(y, l, b)
        gammas, _, min_regime = candidate_grid(w, m, trim, n_grid_gamma)
        path = logdet_profile(X, Y, w, gammas)
        i = int(np.argmin(path))
        if best is None or path[i] < best["val"]:
            best = dict(bi=bi, i=i, val=float(path[i]),
                        gammas=gammas, min_regime=min_regime)
    b = beta_cands[best["bi"]]
    gamma = float(best["gammas"][best["i"]])
    X, Y, w = build_design(y, l, b)
    lo = w <= gamma
    Bl, sel, U1 = white_se(X[lo], Y[lo])
    Bh, seh, U2 = white_se(X[~lo], Y[~lo])
    sigma = (U1.T @ U1 + U2.T @ U2) / n
    logdet = float(np.linalg.slogdet(sigma)[1])
    llf = -n * k / 2 * (math.log(2 * math.pi) + 1) - n / 2 * logdet
    return {
        "beta": [float(v) for v in b],
        "threshold": gamma,
        "coefs_low": Bl.tolist(),
        "coefs_high": Bh.tolist(),
        "se_low": sel.tolist(),
        "se_high": seh.tolist(),
        "n_low": int(lo.sum()),
        "n_high": int((~lo).sum()),
        "nobs": int(n),
        "frac_low": float(lo.sum() / n),
        "sigma": sigma.tolist(),
        "log_det_sigma": logdet,
        "llf": float(llf),
        "llf_linear": float(llf_linear),
        "beta_linear": [float(v) for v in beta_linear],
        "beta_grid": beta_grid,
        "min_regime": best["min_regime"],
        "n_regressors": int(m),
    }


def lm_stat_path(y, l, trim, n_grid, beta=None):
    T, k = y.shape
    m = 2 + k * l
    if beta is not None:
        beta = np.asarray(beta, dtype=float)
        beta = beta / beta[0]
    else:
        beta, _, _ = johansen_rank1(y, l)
    X, Y, w = build_design(y, l, beta)
    gammas, _, min_regime = candidate_grid(w, m, trim, n_grid)
    _, U = ols(X, Y)                            # null residuals
    n = Y.shape[0]
    mk = m * k
    # z_t = u_t kron x_t, index j*m + a (equations outer).
    Z = np.einsum("tj,ta->tja", U, X).reshape(n, mk)
    path = np.empty(gammas.size)
    for i, g in enumerate(gammas):
        lo = w <= g
        M1 = X[lo].T @ X[lo]
        M2 = X[~lo].T @ X[~lo]
        A1 = np.linalg.solve(M1, X[lo].T @ U[lo])    # m x k
        A2 = np.linalg.solve(M2, X[~lo].T @ U[~lo])
        d = (A1 - A2).T.ravel()
        O1 = Z[lo].T @ Z[lo]
        O2 = Z[~lo].T @ Z[~lo]
        K1 = np.kron(np.eye(k), np.linalg.inv(M1))
        K2 = np.kron(np.eye(k), np.linalg.inv(M2))
        V = K1 @ O1 @ K1 + K2 @ O2 @ K2
        path[i] = d @ np.linalg.solve(V, d)
    i = int(np.argmax(path))
    return {
        "stat": float(path[i]),
        "threshold": float(gammas[i]),
        "beta": [float(v) for v in beta],
        "nobs": int(n),
        "min_regime": int(min_regime),
        "n_regressors": int(m),
        "thresholds": gammas.tolist(),
        "lm_path": path.tolist(),
    }


# ------------------------------------------------------------------- main

def main():
    series = make_series()

    fit_cases = []
    for name, l, trim, ngg, beta, ngb, span in [
        # Fixed beta: pins the concentrated scan at 1e-10.
        ("tv_strong", 1, 0.05, 300, [1.0, -1.0], 1, 0.0),
        # Estimated beta (grid): pins the Johansen anchor, the se formula,
        # and the joint grid at 1e-8; n_grid_gamma=50 exercises the
        # even-subsample rule.
        ("tv_strong", 1, 0.05, 50, None, 21, 8.0),
        # Trivariate fixed beta, two lagged differences, coarser trim.
        ("tv3", 2, 0.10, 300, [1.0, -0.5, -0.5], 1, 0.0),
        # Null (linear) data with a fixed beta: behavior pinned.
        ("tv_linear", 1, 0.05, 300, [1.0, -1.0], 1, 0.0),
    ]:
        case = fit_tvecm(series[name], l, trim, ngg, beta, ngb, span)
        case.update(series=name, k_ar_diff=l, trim=trim,
                    n_grid_gamma=ngg, beta_fixed=beta,
                    n_grid_beta=ngb, beta_span=span,
                    tol=1e-10 if beta is not None else 1e-8)
        fit_cases.append(case)

    test_cases = []
    for name, l, trim, ng, beta in [
        ("tv_strong", 1, 0.05, 50, None),
        ("tv_strong", 1, 0.05, 40, [1.0, -1.0]),
        ("tv_linear", 1, 0.05, 50, None),
        ("tv3", 1, 0.10, 40, [1.0, -0.5, -0.5]),
    ]:
        case = lm_stat_path(series[name], l, trim, ng, beta)
        case.update(series=name, k_ar_diff=l, trim=trim, n_grid=ng,
                    beta_fixed=beta, tol=1e-10 if beta is not None else 1e-8)
        test_cases.append(case)

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "note": (
                "Hansen-Seo (2002) threshold VECM and sup-LM: documented-"
                "algorithm transcription validated against this independent "
                "NumPy implementation — no third-party reference runs in "
                "this container (R installed but CRAN unreachable through "
                "the egress proxy, so no tsDyn). The bootstrap p-value is "
                "checked by Monte Carlo property tests, not pinned here."
            ),
        },
        "series": {kk: vv.tolist() for kk, vv in series.items()},
        "fit": fit_cases,
        "test": test_cases,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); "
          f"{len(fit_cases)} fit cases, {len(test_cases)} sup-LM cases")
    for c in fit_cases:
        print(f"  fit  {c['series']:10s} l={c['k_ar_diff']} "
              f"beta2={c['beta'][1]: .4f} gamma={c['threshold']: .4f} "
              f"logdet={c['log_det_sigma']:.6f} "
              f"n=({c['n_low']},{c['n_high']})")
    for c in test_cases:
        print(f"  supLM {c['series']:10s} l={c['k_ar_diff']} "
              f"stat={c['stat']:.4f} gamma={c['threshold']: .4f}")


if __name__ == "__main__":
    main()
