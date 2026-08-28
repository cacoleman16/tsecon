"""Golden fixtures for the two-regime threshold VAR (`threshold_var`) and
its robust sup-Wald (score-form) linearity test (`threshold_var_test`).

Reference: NO third-party TVAR implementation runs in this container (R
installed from apt, but CRAN is unreachable through the egress proxy, so
`tsDyn` — the natural `TVAR`/`TVAR.LRtest` reference — cannot be
installed; no Python package implements the estimator). The reference is
therefore built HERE, in NumPy, as an independent transcription of the
documented algorithm — the multivariate SETAR (Tong 1983; Tsay 1998; Lo &
Zivot 2001):

    y_t = A_1' X_t 1{z_t <= gamma} + A_2' X_t 1{z_t > gamma} + u_t,
    X_t = (1?, y_{t-1}', ..., y_{t-p}')',
    z_t = y_{threshold_index, t-d}.

Transcribed conventions (the Rust implementation states the same rules):

  * usable sample     t = start .. T-1 (0-indexed), start = max(p, max d
                      searched); n = T - start; m = k*p + constant.
  * candidate grid    tie-grouped order statistics gamma of {z_t} with
                      min_regime <= #{z <= gamma} <= n - min_regime,
                      min_regime = max(m + 1, ceil(trim * n)). The FIT
                      scans the whole feasible grid (the SETAR
                      convention); the TEST subsamples it to at most
                      n_grid evenly spaced candidates by the exact integer
                      rule idx_j = (2 j (c-1) + (G-1)) // (2 (G-1))
                      (the Hansen-Seo convention).
  * per candidate     OLS in each regime; criterion ln det SigmaHat,
                      SigmaHat = (E_1 + E_2)/n (the concentrated Gaussian
                      MLE); the FIRST candidate attaining the minimum
                      wins, and when several delays are searched the first
                      delay with a strictly smaller criterion wins,
                      iterating in the given order.
  * reported fit      refit at (d^, g^): per-regime OLS coefficients (rows
                      = equations) with CLASSICAL nonrobust SEs
                      sqrt(SSR_jr/(n_r - m) * diag[(X_r'X_r)^{-1}]) (the
                      SETAR convention), per-regime ML Sigma_r = E_r/n_r,
                      pooled Sigma, llf = -(nk/2)(ln 2pi + 1) -
                      (n/2) ln det Sigma, and AIC/BIC = n ln det Sigma +
                      penalty * q with q = 2*k*m + 1 (both coefficient
                      blocks plus the threshold; covariance excluded —
                      the multivariate analogue of the SETAR convention).
  * test statistic    the coefficient-difference quadratic form with
                      Eicker-White covariance at the NULL residuals (the
                      multivariate Hansen-Seo sup-LM, transcribed
                      identically to generate_tvecm_fixtures.py): per
                      candidate g, M_i = X_i'X_i, A^_i = M_i^-1 X_i'U~,
                      Omega_i = sum_i (u~u~') kron (xx'),
                      V_i = (I kron M_i^-1) Omega_i (I kron M_i^-1),
                      W(g) = vec(A^_1-A^_2)' (V_1+V_2)^-1 vec(...),
                      vec stacking equations as the OUTER index; sup over
                      the (subsampled) grid.

Grade (honest): documented-ALGORITHM transcription validated against an
independent NumPy implementation (this file) — NOT a third-party golden;
pinned at 1e-10. Statistical correctness (test size, threshold/coefficient
recovery) is established by the crate's seeded Monte Carlo property tests
(`tvar_properties.rs`), whose numbers are quoted in the model card. The
bootstrap p-value is deliberately NOT pinned — it is a Monte Carlo
quantity checked by property (null size ~ nominal).

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  .venv-wt/bin/python fixtures/generate_tvar_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "tvar.json"


# ------------------------------------------------------------------ series

def sim_tvar(seed, T, d, gamma, c_low, a_low, c_high, a_high, tv=0,
             burn=100):
    """Two-regime TVAR(1) with regime by y[tv, t-d] <= gamma."""
    rng = np.random.default_rng(seed)
    k = len(c_low)
    n = T + burn + d
    y = np.zeros((n, k))
    for t in range(max(1, d), n):
        if y[t - d, tv] <= gamma:
            c, a = np.asarray(c_low), np.asarray(a_low)
        else:
            c, a = np.asarray(c_high), np.asarray(a_high)
        y[t] = c + a @ y[t - 1] + rng.standard_normal(k)
    return y[burn + d:]


def sim_var(seed, T, c, a, burn=100):
    rng = np.random.default_rng(seed)
    k = len(c)
    n = T + burn
    y = np.zeros((n, k))
    c, a = np.asarray(c), np.asarray(a)
    for t in range(1, n):
        y[t] = c + a @ y[t - 1] + rng.standard_normal(k)
    return y[burn:]


def make_series():
    return {
        # Strongly separated two-regime VAR(1), regime by y0_{t-1} <= 0:
        # the regime intercepts push series 0 back across the threshold.
        "tvar_strong": sim_tvar(
            20260828, 300, 1, 0.0,
            c_low=[1.0, 0.3], a_low=[[0.5, 0.1], [0.2, 0.4]],
            c_high=[-1.0, -0.3], a_high=[[0.1, 0.0], [-0.1, 0.5]]),
        # Regime switching on the SECOND lag of series 0, gamma = 0.3.
        "tvar_d2": sim_tvar(
            7, 320, 2, 0.3,
            c_low=[0.6, 0.2], a_low=[[0.5, 0.0], [0.1, 0.4]],
            c_high=[-0.6, -0.2], a_high=[[0.1, 0.1], [0.0, 0.5]]),
        # Linear VAR(1) — null data (no threshold exists).
        "tvar_linear": sim_var(
            11, 250, c=[0.0, 0.0], a=[[0.5, 0.1], [0.1, 0.4]]),
    }


# ------------------------------------- the transcription (see module doc)

def build_design(y, p, tv, delay, start, constant):
    T, k = y.shape
    n = T - start
    cols = []
    if constant:
        cols.append(np.ones((n, 1)))
    for lag in range(1, p + 1):
        cols.append(y[start - lag:T - lag])
    X = np.hstack(cols)
    Y = y[start:]
    z = y[start - delay:T - delay, tv]
    return X, Y, z


def even_indices(count, n_grid):
    if count <= n_grid:
        return list(range(count))
    out = []
    for j in range(n_grid):
        idx = (2 * j * (count - 1) + (n_grid - 1)) // (2 * (n_grid - 1))
        if not out or out[-1] != idx:
            out.append(idx)
    return out


def candidate_grid(z, m, trim, n_grid=None):
    n = z.size
    min_regime = max(m + 1, math.ceil(trim * n))
    zs = np.sort(z)
    gammas = np.unique(zs)
    nlow = np.searchsorted(zs, gammas, side="right")
    ok = (nlow >= min_regime) & (nlow <= n - min_regime)
    gammas = gammas[ok]
    if n_grid is not None:
        gammas = gammas[even_indices(gammas.size, n_grid)]
    return gammas, int(min_regime)


def ols(X, Y):
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    U = Y - X @ B
    return B, U


def logdet_profile(X, Y, z, gammas):
    n = Y.shape[0]
    path = np.empty(gammas.size)
    for i, g in enumerate(gammas):
        lo = z <= g
        _, U1 = ols(X[lo], Y[lo])
        _, U2 = ols(X[~lo], Y[~lo])
        sig = (U1.T @ U1 + U2.T @ U2) / n
        path[i] = np.linalg.slogdet(sig)[1]
    return path


def fit_tvar(y, p, tv, delays, trim, constant):
    T, k = y.shape
    start = max(p, max(delays))
    m = k * p + int(constant)
    best = None
    for d in delays:
        X, Y, z = build_design(y, p, tv, d, start, constant)
        gammas, min_regime = candidate_grid(z, m, trim)
        path = logdet_profile(X, Y, z, gammas)
        i = int(np.argmin(path))
        if best is None or path[i] < best["val"]:
            best = dict(d=d, i=i, val=float(path[i]), gammas=gammas,
                        path=path, min_regime=min_regime)
    d = best["d"]
    X, Y, z = build_design(y, p, tv, d, start, constant)
    gamma = float(best["gammas"][best["i"]])
    lo = z <= gamma
    n = Y.shape[0]
    n1, n2 = int(lo.sum()), int((~lo).sum())
    B1, U1 = ols(X[lo], Y[lo])
    B2, U2 = ols(X[~lo], Y[~lo])
    E1, E2 = U1.T @ U1, U2.T @ U2
    sigma = (E1 + E2) / n
    logdet = float(np.linalg.slogdet(sigma)[1])

    def classical_se(Xr, E, nr):
        diag = np.diag(np.linalg.inv(Xr.T @ Xr))
        return np.sqrt(np.outer(np.diag(E) / (nr - m), diag))

    llf = -n * k / 2 * (math.log(2 * math.pi) + 1) - n / 2 * logdet
    q = 2 * k * m + 1
    return {
        "threshold": gamma,
        "delay": int(d),
        "threshold_index": int(tv),
        "coefs_low": B1.T.tolist(),
        "coefs_high": B2.T.tolist(),
        "se_low": classical_se(X[lo], E1, n1).tolist(),
        "se_high": classical_se(X[~lo], E2, n2).tolist(),
        "n_low": n1,
        "n_high": n2,
        "nobs": int(n),
        "sigma": sigma.tolist(),
        "sigma_low": (E1 / n1).tolist(),
        "sigma_high": (E2 / n2).tolist(),
        "log_det_sigma": logdet,
        "llf": float(llf),
        "aic": float(n * logdet + 2 * q),
        "bic": float(n * logdet + q * math.log(n)),
        "thresholds": best["gammas"].tolist(),
        "logdet_path": best["path"].tolist(),
        "min_regime": best["min_regime"],
        "n_regressors": int(m),
    }


def wald_stat_path(y, p, tv, delay, trim, constant, n_grid):
    T, k = y.shape
    start = max(p, delay)
    m = k * p + int(constant)
    X, Y, z = build_design(y, p, tv, delay, start, constant)
    gammas, min_regime = candidate_grid(z, m, trim, n_grid)
    _, U = ols(X, Y)                            # null (linear VAR) residuals
    n = Y.shape[0]
    mk = m * k
    Z = np.einsum("tj,ta->tja", U, X).reshape(n, mk)
    path = np.empty(gammas.size)
    for i, g in enumerate(gammas):
        lo = z <= g
        M1 = X[lo].T @ X[lo]
        M2 = X[~lo].T @ X[~lo]
        A1 = np.linalg.solve(M1, X[lo].T @ U[lo])
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
        "delay": int(delay),
        "threshold_index": int(tv),
        "nobs": int(n),
        "min_regime": int(min_regime),
        "n_regressors": int(m),
        "thresholds": gammas.tolist(),
        "wald_path": path.tolist(),
    }


# ------------------------------------------------------------------- main

def main():
    series = make_series()

    fit_cases = []
    for name, p, tv, delays, trim, constant in [
        ("tvar_strong", 1, 0, [1], 0.10, True),
        ("tvar_strong", 2, 0, [1], 0.10, True),    # over-specified p
        ("tvar_d2", 1, 0, [1, 2, 3], 0.15, True),  # delay search -> pick 2
        ("tvar_linear", 1, 0, [1], 0.10, False),   # null data, no constant
        ("tvar_strong", 1, 1, [1], 0.15, True),    # threshold on series 1
    ]:
        case = fit_tvar(series[name], p, tv, delays, trim, constant)
        case.update(series=name, p=p, delays=delays, trim=trim,
                    constant=constant)
        fit_cases.append(case)

    test_cases = []
    for name, p, tv, delay, trim, n_grid in [
        ("tvar_strong", 1, 0, 1, 0.10, 50),        # subsampled grid
        ("tvar_linear", 1, 0, 1, 0.10, 300),       # full feasible grid
        ("tvar_d2", 1, 0, 2, 0.15, 40),
    ]:
        case = wald_stat_path(series[name], p, tv, delay, trim, True,
                              n_grid)
        case.update(series=name, p=p, trim=trim, n_grid=n_grid,
                    constant=True)
        test_cases.append(case)

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "note": (
                "Two-regime threshold VAR and robust sup-Wald linearity "
                "statistic: documented-algorithm transcription validated "
                "against this independent NumPy implementation — no "
                "third-party TVAR runs in this container (R installed but "
                "CRAN unreachable through the egress proxy, so no tsDyn). "
                "The bootstrap p-value is checked by Monte Carlo property "
                "tests, not pinned here."
            ),
        },
        "series": {kk: vv.tolist() for kk, vv in series.items()},
        "fit": fit_cases,
        "test": test_cases,
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); "
          f"{len(fit_cases)} fit cases, {len(test_cases)} sup-Wald cases")
    for c in fit_cases:
        print(f"  fit  {c['series']:12s} p={c['p']} d={c['delay']} "
              f"tv={c['threshold_index']} gamma={c['threshold']: .4f} "
              f"logdet={c['log_det_sigma']:.6f} "
              f"n=({c['n_low']},{c['n_high']})")
    for c in test_cases:
        print(f"  supW {c['series']:12s} p={c['p']} d={c['delay']} "
              f"W={c['stat']:.4f} gamma={c['threshold']: .4f}")


if __name__ == "__main__":
    main()
