"""Golden fixtures for the two-regime SETAR fit (`setar`) and the Hansen
(1996) sup-F linearity statistic (`setar_test`).

Reference: NO third-party SETAR implementation exists in this venv (no R
`tsDyn`, no statsmodels threshold AR), so the reference is built HERE, in
NumPy, as an independent transcription of the published algorithm — Tong &
Lim (1980) concentrated least squares in Hansen's (1997, Studies in
Nonlinear Dynamics & Econometrics 2(1)) notation:

    y_t = x_t' beta_1 1{z_t <= gamma} + x_t' beta_2 1{z_t > gamma} + e_t,
    x_t = (1?, y_{t-1}, ..., y_{t-p})',   z_t = y_{t-d}.

Transcribed conventions (the Rust implementation states the same rules):

  * usable sample     t = start .. T-1 (0-indexed), start = max(p, max d
                      searched); n = T - start; k = p + constant.
  * candidate grid    the UNIQUE order statistics gamma of {z_t} with
                      min_regime <= #{z <= gamma} <= n - min_regime, where
                      min_regime = max(k + 1, ceil(trim * n)) — Hansen's
                      trimming plus estimability of both regimes with one
                      residual degree of freedom.
  * per candidate     OLS in each regime (np.linalg.lstsq), pooled
                      SSR(gamma) = SSR_1 + SSR_2; the FIRST candidate
                      attaining the minimum wins (np.argmin), and when
                      several delays are searched the first delay with a
                      strictly smaller SSR wins, iterating in given order.
  * reported fit      refit at (d^, gamma^): coefficients, classical SEs
                      sqrt(SSR_j/(n_j - k) * diag[(X_j'X_j)^{-1}]), pooled
                      sigma2 = SSR/(n - 2k), per-regime sigma2_j =
                      SSR_j/(n_j - k), AIC = n ln(SSR/n) + 2 m and BIC =
                      n ln(SSR/n) + m ln(n) with m = 2k + 1 (both
                      coefficient blocks plus the threshold).
  * sup-F             n (S0 - S1) / S1 (Hansen 1997 F12), S0 = SSR of the
                      linear AR(p) on the same usable sample, S1 = the
                      concentrated SETAR SSR; per-candidate path
                      F(gamma) = n (S0 - S(gamma)) / S(gamma).

Grade (honest): documented-ALGORITHM transcription validated against an
independent NumPy implementation (this file) — NOT a third-party golden.
What the pin proves: the Rust concentrated-LS machinery (grid, trimming,
tie grouping, per-regime OLS, SEs, ICs, sup-F) agrees with a direct dense
NumPy implementation of the same published rules at 1e-10. Statistical
correctness (threshold recovery, test size/power) is established separately
by the crate's seeded Monte Carlo property tests, whose numbers are quoted
in the model card. The bootstrap p-value is deliberately NOT pinned here —
it is a Monte Carlo quantity checked by property (null size ~ nominal).

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_setar_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "setar.json"


# ------------------------------------------------------------------ series

def sim_setar(seed, T, d, gamma, low, high, sigma=1.0, burn=100):
    """Simulate a two-regime SETAR: coef vectors are [const, phi_1, ...]."""
    rng = np.random.default_rng(seed)
    p = len(low) - 1
    m = max(p, d)
    y = np.zeros(T + burn + m)
    e = sigma * rng.standard_normal(T + burn + m)
    for t in range(m, T + burn + m):
        c = low if y[t - d] <= gamma else high
        y[t] = c[0] + sum(c[1 + l] * y[t - 1 - l] for l in range(p)) + e[t]
    return y[burn + m:]


def sim_ar(seed, T, coefs, sigma=1.0, burn=100):
    """Simulate a linear AR: coefs = [const, phi_1, ...]."""
    rng = np.random.default_rng(seed)
    p = len(coefs) - 1
    y = np.zeros(T + burn + p)
    e = sigma * rng.standard_normal(T + burn + p)
    for t in range(p, T + burn + p):
        y[t] = coefs[0] + sum(coefs[1 + l] * y[t - 1 - l] for l in range(p)) + e[t]
    return y[burn + p:]


def make_series():
    return {
        # Strongly separated SETAR(1), d=1, gamma=0. The regime intercepts
        # push the process back across the threshold, so both regimes stay
        # well-populated and the threshold is sharply identified.
        "setar_strong": sim_setar(20260817, 300, 1, 0.0,
                                  low=[1.0, 0.6], high=[-1.0, 0.2]),
        # SETAR(1) switching on the SECOND lag, gamma=0.3.
        "setar_d2": sim_setar(7, 320, 2, 0.3,
                              low=[0.5, 0.6], high=[-0.5, 0.1]),
        # Linear AR(1) — null data (pins behavior when no threshold exists).
        "linear_ar1": sim_ar(11, 200, [0.0, 0.5]),
        # Linear AR(2) — for the p=2 and no-constant cases.
        "linear_ar2": sim_ar(13, 250, [0.2, 0.4, 0.25]),
    }


# ------------------------------------- the transcription (see module doc)

def design(y, p, delay, start, constant):
    T = y.size
    n = T - start
    cols = []
    if constant:
        cols.append(np.ones(n))
    for lag in range(1, p + 1):
        cols.append(y[start - lag:T - lag])
    X = np.column_stack(cols)
    resp = y[start:]
    z = y[start - delay:T - delay]
    return X, resp, z


def ols(X, yy):
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    r = yy - X @ b
    return b, float(r @ r)


def candidate_grid(z, k, trim):
    n = z.size
    min_regime = max(k + 1, math.ceil(trim * n))
    zs = np.sort(z)
    gammas = np.unique(zs)
    nlow = np.searchsorted(zs, gammas, side="right")
    ok = (nlow >= min_regime) & (nlow <= n - min_regime)
    return gammas[ok], int(min_regime)


def ssr_profile(X, yy, z, gammas):
    path = np.empty(gammas.size)
    for i, g in enumerate(gammas):
        lo = z <= g
        _, s1 = ols(X[lo], yy[lo])
        _, s2 = ols(X[~lo], yy[~lo])
        path[i] = s1 + s2
    return path


def fit_setar(y, p, delays, trim, constant):
    start = max(p, max(delays))
    k = p + int(constant)
    best = None
    for d in delays:
        X, yy, z = design(y, p, d, start, constant)
        gammas, min_regime = candidate_grid(z, k, trim)
        path = ssr_profile(X, yy, z, gammas)
        i = int(np.argmin(path))
        if best is None or path[i] < best["ssr_scan"]:
            best = dict(d=d, X=X, yy=yy, z=z, gammas=gammas, path=path,
                        i=i, ssr_scan=float(path[i]), min_regime=min_regime)
    b = best
    gamma = float(b["gammas"][b["i"]])
    lo = b["z"] <= gamma
    Xl, yl, Xh, yh = b["X"][lo], b["yy"][lo], b["X"][~lo], b["yy"][~lo]
    bl, s_lo = ols(Xl, yl)
    bh, s_hi = ols(Xh, yh)
    n_low, n_high = int(lo.sum()), int((~lo).sum())
    se = lambda Xj, sj, nj: np.sqrt(np.diag(np.linalg.inv(Xj.T @ Xj)) * sj / (nj - k))
    n = b["yy"].size
    ssr = s_lo + s_hi
    m_params = 2 * k + 1
    return {
        "threshold": gamma,
        "delay": int(b["d"]),
        "coefs_low": bl.tolist(),
        "coefs_high": bh.tolist(),
        "se_low": se(Xl, s_lo, n_low).tolist(),
        "se_high": se(Xh, s_hi, n_high).tolist(),
        "n_low": n_low,
        "n_high": n_high,
        "nobs": int(n),
        "min_regime": b["min_regime"],
        "k": int(k),
        "ssr": float(ssr),
        "sigma2": float(ssr / (n - 2 * k)),
        "sigma2_low": float(s_lo / (n_low - k)),
        "sigma2_high": float(s_hi / (n_high - k)),
        "aic": float(n * math.log(ssr / n) + 2 * m_params),
        "bic": float(n * math.log(ssr / n) + m_params * math.log(n)),
        "thresholds": b["gammas"].tolist(),
        "ssr_path": b["path"].tolist(),
    }


def supf_setar(y, p, delay, trim, constant=True):
    start = max(p, delay)
    k = p + int(constant)
    X, yy, z = design(y, p, delay, start, constant)
    gammas, _ = candidate_grid(z, k, trim)
    path = ssr_profile(X, yy, z, gammas)
    i = int(np.argmin(path))
    _, s0 = ols(X, yy)
    n = yy.size
    s1 = float(path[i])
    return {
        "stat": float(n * (s0 - s1) / s1),
        "threshold": float(gammas[i]),
        "delay": int(delay),
        "nobs": int(n),
        "ssr_linear": float(s0),
        "ssr_setar": s1,
        "thresholds": gammas.tolist(),
        "f_path": (n * (s0 - path) / path).tolist(),
    }


# ------------------------------------------------------------------- main

def main():
    series = make_series()

    fit_cases = []
    for name, p, delays, trim, constant in [
        ("setar_strong", 1, [1], 0.15, True),
        ("setar_strong", 2, [1], 0.15, True),   # over-specified p
        ("setar_strong", 1, [1], 0.10, True),   # wider grid
        ("setar_d2", 1, [1, 2, 3], 0.15, True), # delay search -> must pick 2
        ("linear_ar1", 1, [1], 0.15, True),     # null data, behavior pinned
        ("linear_ar2", 2, [1, 2], 0.15, False), # p=2, no constant
    ]:
        case = fit_setar(series[name], p, delays, trim, constant)
        case.update(series=name, p=p, delays=delays, trim=trim,
                    constant=constant)
        fit_cases.append(case)

    test_cases = []
    for name, p, delay, trim in [
        ("setar_strong", 1, 1, 0.15),
        ("linear_ar1", 1, 1, 0.15),
        ("linear_ar2", 2, 1, 0.15),
        ("setar_d2", 1, 2, 0.10),
    ]:
        case = supf_setar(series[name], p, delay, trim)
        case.update(series=name, p=p, trim=trim)
        test_cases.append(case)

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "note": (
                "SETAR concentrated LS and Hansen sup-F: documented-"
                "algorithm transcription (Tong-Lim 1980; Hansen 1996, 1997) "
                "validated against this independent NumPy implementation — "
                "no third-party SETAR exists in this venv (no R tsDyn). "
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
          f"{len(fit_cases)} fit cases, {len(test_cases)} sup-F cases")
    for c in fit_cases:
        print(f"  fit  {c['series']:13s} p={c['p']} d={c['delay']} "
              f"gamma={c['threshold']: .4f} ssr={c['ssr']:.4f} "
              f"n=({c['n_low']},{c['n_high']})")
    for c in test_cases:
        print(f"  supF {c['series']:13s} p={c['p']} d={c['delay']} "
              f"F={c['stat']:.4f} gamma={c['threshold']: .4f}")


if __name__ == "__main__":
    main()
