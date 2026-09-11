"""Golden fixtures for the conditional (hard-path) VAR forecast
`var_conditional_forecast` (crate `tsecon-var`, module `conditional.rs`).

Two exact reference legs, graded honestly:

  * CLOSED FORM (documented formula, transcribed here in NumPy) — the
    Doan-Litterman-Sims (1984) / Waggoner-Zha (1999) conditional forecast.
    With `Psi_h` the MA coefficients of the fitted VAR(p) (statsmodels
    `VARResults.coefs`, `irf.irfs` asserted equal at generation), `Sigma`
    the df-adjusted residual covariance `sigma_u` (the one `var_forecast`
    uses), `yhat` the unconditional path `VARResults.forecast(y[-p:], H)`,
    and `R` the (H k) x (H k) block-lower-triangular matrix with block
    `(h, s) = Psi_{h-s}` for `s <= h`:

        B      = the rows of R belonging to the constrained (h, j) cells
        r      = condition - yhat at those cells
        S      = I_H kron Sigma
        u*     = S B' (B S B')^{-1} r                 implied reduced-form shocks
        y_cond = yhat + R u*                          conditional mean path
        V      = R (S - S B' (B S B')^{-1} B S) R'    conditional covariance;
                 cov[h] = the h-th k x k diagonal block
        eps*   = (I_H kron P)^{-1} u*, P = chol_lower(Sigma)   structural shocks
        maha   = r' (B S B')^{-1} r  (= eps*' eps*, asserted),  p = chi2.sf(maha, m)

  * KALMAN CONDITIONING (independent package) — the Banbura-Giannone-Lenza
    (2015) route: statsmodels `VARMAX(endog_with_future, order=(p, 0),
    trend=..., error_cov_type="unstructured").smooth(params)` with the
    future rows appended as NaN except the conditioned cells and `params`
    set BY NAME to the OLS intercept, lag coefficients and the lower
    Cholesky factor of `sigma_u` (`mod.ssm["state_cov"]` asserted equal to
    `sigma_u`). `smoothed_state[:k, T:T+H]` is the conditional path,
    `smoothed_state_cov[:k, :k, T+h]` the conditional covariance at horizon
    h + 1, and `smoothed_state_disturbance[:, T-1:T+H-1]` the implied
    shocks (statsmodels indexes the disturbance eta_t by the state it
    produces, alpha_{t+1} = T alpha_t + R eta_t, so the shock that drives
    y_{T+h} sits one index earlier than the state).
    The two legs are asserted to agree at 1e-10 before anything is stored
    (measured ~1e-15) and both are stored; the all-NaN future is asserted
    to reproduce `VARResults.forecast` and `VARResults.mse` at 1e-12.

Data: a seeded simulated VAR(2) (k = 3, non-diagonal innovation
covariance, stored) and a three-variable US macro system built from
statsmodels' bundled `macrodata` by transformation only — 100 x dlog real
GDP (quarterly growth, percent), 400 x dlog CPI (annualised inflation,
percent) and the first difference of the 3-month T-bill rate — so the
fixture stores derived series, not the dataset. The macro scenario is a
policy staple: inflation held at its last observed value for four
quarters while the T-bill rate is on hold (zero change) for two.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden tests parse to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_var_cf_fixtures.py
"""

from __future__ import annotations

import json
import math
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

OUT = Path(__file__).resolve().parent / "var_cf.json"


# ------------------------------------------------------------------ series

def sim_var(seed, T, c, a_list, chol, burn=100):
    """VAR(p) with intercept c, lag matrices a_list, innovation covariance
    chol @ chol.T (the girf.json simulator)."""
    rng = np.random.default_rng(seed)
    k = len(c)
    p = len(a_list)
    n = T + burn + p
    y = np.zeros((n, k))
    c = np.asarray(c)
    a_list = [np.asarray(a) for a in a_list]
    chol = np.asarray(chol)
    for t in range(p, n):
        v = c.copy()
        for i, a in enumerate(a_list, start=1):
            v = v + a @ y[t - i]
        y[t] = v + chol @ rng.standard_normal(k)
    return y[burn + p:]


def macro_system():
    """[100 dlog realgdp, 400 dlog cpi, diff tbilrate], 1959Q2-2009Q3
    (T = 202) — transformations of statsmodels' bundled macrodata."""
    md = sm.datasets.macrodata.load_pandas().data
    g = 100.0 * np.diff(np.log(md["realgdp"].to_numpy(dtype=float)))
    infl = 400.0 * np.diff(np.log(md["cpi"].to_numpy(dtype=float)))
    dtb = np.diff(md["tbilrate"].to_numpy(dtype=float))
    return np.column_stack([g, infl, dtb])


# --------------------------------------------------------- closed-form leg

def ma_rep(coefs, H):
    p = len(coefs)
    k = coefs[0].shape[0]
    phi = [np.eye(k)]
    for h in range(1, H + 1):
        acc = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            acc += phi[h - i] @ coefs[i - 1]
        phi.append(acc)
    return phi


def closed_form(res, H, cond, alpha):
    """cond: H x k array, NaN = free. Returns the dict of stored quantities."""
    k = res.neqs
    p = res.k_ar
    coefs = [np.asarray(res.coefs[i]) for i in range(p)]
    sigma = np.asarray(res.sigma_u)
    yhat = np.asarray(res.forecast(res.endog[-p:], H))
    psi = ma_rep(coefs, H - 1)
    if p >= 1:
        assert np.allclose(np.asarray(res.irf(H - 1).irfs), np.asarray(psi), atol=1e-12)
    n = H * k
    R = np.zeros((n, n))
    for h in range(H):
        for s in range(h + 1):
            R[h * k:(h + 1) * k, s * k:(s + 1) * k] = psi[h - s]
    S = np.kron(np.eye(H), sigma)
    Omega = R @ S @ R.T
    idx = [h * k + j for h in range(H) for j in range(k) if np.isfinite(cond[h, j])]
    m = len(idx)
    r = np.array([cond[i // k, i % k] - yhat[i // k, i % k] for i in idx])
    B = R[idx, :]
    W = S @ B.T
    Occ = B @ W
    lam = np.linalg.solve(Occ, r)
    u = W @ lam
    ycond = (yhat.ravel() + R @ u).reshape(H, k)
    G = R @ W
    covfull = Omega - G @ np.linalg.solve(Occ, G.T)
    covs = np.array([covfull[h * k:(h + 1) * k, h * k:(h + 1) * k] for h in range(H)])
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))
    P = np.linalg.cholesky(sigma)
    eps = np.linalg.solve(np.kron(np.eye(H), P), u).reshape(H, k)
    maha = float(r @ lam)
    assert abs(maha - float((eps ** 2).sum())) <= 1e-10 * max(1.0, maha)
    # Constrained cells reproduce their conditions to roundoff; the crate
    # pins them bitwise, so store the pinned values.
    mask = np.isfinite(cond)
    assert np.max(np.abs(ycond[mask] - cond[mask])) < 1e-10
    ycond = np.where(mask, cond, ycond)
    # The conditional variance of a constrained cell is zero to roundoff;
    # the crate zeroes those rows/columns, so store them zeroed.
    for h in range(H):
        for j in range(k):
            if mask[h, j]:
                covs[h, j, :] = 0.0
                covs[h, :, j] = 0.0
    se = np.sqrt(np.maximum(np.einsum("hjj->hj", covs), 0.0))
    mse = np.asarray(res.mse(H))
    unc_se = np.sqrt(np.einsum("hjj->hj", mse))
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "unconditional": yhat.tolist(),
        "point": ycond.tolist(),
        "cov": covs.tolist(),
        "se": se.tolist(),
        "unconditional_se": unc_se.tolist(),
        "lower": (ycond - z * se).tolist(),
        "upper": (ycond + z * se).tolist(),
        "shocks": u.reshape(H, k).tolist(),
        "orth_shocks": eps.tolist(),
        "n_constrained": m,
        "mahalanobis": maha,
        "mahalanobis_pvalue": float(stats.chi2.sf(maha, m)),
        "constrained": mask.tolist(),
    }


# ------------------------------------------------------- Kalman (BGL) leg

def varmax_params_by_name(mod, res, trend):
    P = np.linalg.cholesky(np.asarray(res.sigma_u))
    params = np.zeros(len(mod.param_names))
    for i, nm in enumerate(mod.param_names):
        if nm.startswith("intercept."):
            j = int(nm.split(".y")[1]) - 1
            params[i] = res.intercept[j] if trend == "c" else 0.0
        elif nm.startswith("L"):
            lag_s, src, eq = nm.split(".")  # 'L1.y2.y1': y2 at lag 1 in equation y1
            params[i] = res.coefs[int(lag_s[1:]) - 1][int(eq[1:]) - 1, int(src[1:]) - 1]
        elif nm.startswith("sqrt.var."):
            j = int(nm.split(".y")[1]) - 1
            params[i] = P[j, j]
        elif nm.startswith("sqrt.cov."):
            a, b = nm[len("sqrt.cov."):].split(".")
            ia, ib = int(a[1:]) - 1, int(b[1:]) - 1
            params[i] = P[max(ia, ib), min(ia, ib)]
        else:
            raise RuntimeError(f"unexpected VARMAX parameter {nm}")
    return params


def kalman_leg(y, res, H, cond, trend):
    k = res.neqs
    p = res.k_ar
    T = y.shape[0]
    mod = VARMAX(np.vstack([y, cond]), order=(p, 0), trend=trend,
                 error_cov_type="unstructured")
    params = varmax_params_by_name(mod, res, trend)
    sres = mod.smooth(params)
    sc = np.asarray(mod.ssm["state_cov"])
    sc = sc if sc.ndim == 2 else sc[:, :, 0]
    assert np.max(np.abs(sc[:k, :k] - np.asarray(res.sigma_u))) < 1e-12
    path = np.asarray(sres.smoothed_state[:k, T:T + H].T)
    cov = np.transpose(np.asarray(sres.smoothed_state_cov[:k, :k, T:T + H]), (2, 0, 1))
    # eta_t enters alpha_{t+1}: the shock driving y_{T+h} is at index T+h-1.
    shocks = np.asarray(sres.smoothed_state_disturbance[:k, T - 1:T + H - 1].T)
    return path, cov, shocks


def build_case(name, y, p, trend, steps, cond_rows, alpha=0.05):
    """cond_rows: list of rows (each a list with None for free), possibly
    shorter than `steps`; padded with free rows for the arithmetic."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(y).fit(p, trend=trend)
    k = res.neqs
    cond = np.full((steps, k), np.nan)
    for h, row in enumerate(cond_rows):
        for j, v in enumerate(row):
            if v is not None:
                cond[h, j] = v
    cf = closed_form(res, steps, cond, alpha)
    path, cov, shocks = kalman_leg(y, res, steps, cond, trend)
    dev_path = float(np.max(np.abs(path - np.asarray(cf["point"]))))
    dev_shocks = float(np.max(np.abs(shocks - np.asarray(cf["shocks"]))))
    covs_cf = np.asarray(cf["cov"])
    dev_cov = float(np.max(np.abs(cov - covs_cf)))
    assert dev_path < 1e-10 and dev_cov < 1e-10 and dev_shocks < 1e-10, (dev_path, dev_cov, dev_shocks)
    # Unconditional: an all-NaN future reproduces VARResults.forecast / mse.
    path0, cov0, _ = kalman_leg(y, res, steps, np.full((steps, k), np.nan), trend)
    assert np.max(np.abs(path0 - np.asarray(cf["unconditional"]))) < 1e-12
    assert np.max(np.abs(cov0 - np.asarray(res.mse(steps)))) < 1e-12
    case = {
        "name": name,
        "p": p,
        "trend": trend,
        "steps": steps,
        "alpha": alpha,
        "conditions": cond_rows,
        "bgl_point": path.tolist(),
        "bgl_cov": cov.tolist(),
        "bgl_shocks": shocks.tolist(),
        "bgl_max_abs_dev": {"point": dev_path, "cov": dev_cov, "shocks": dev_shocks},
    }
    case.update(cf)
    return case


# ------------------------------------------------------------------- main

def main():
    chol = [[0.8, 0.0, 0.0], [0.3, 0.6, 0.0], [-0.2, 0.25, 0.5]]
    y = sim_var(
        20260910, 300, c=[0.5, -0.2, 0.1],
        a_list=[[[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]],
                [[0.1, 0.0, 0.05], [0.0, 0.1, 0.0], [0.05, 0.0, 0.1]]],
        chol=chol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res2 = VAR(y).fit(2, trend="c")
    yhat2 = np.asarray(res2.forecast(y[-2:], 6))

    N = None
    cases = [
        # Inflation-style path on series 1 for four horizons, one cell each
        # on series 2 and 0 later.
        build_case("var2c_path", y, 2, "c", 8, [
            [N, 0.3, N], [N, 0.2, N], [N, 0.1, N], [N, 0.0, N],
            [N, N, N], [N, N, N], [N, N, N], [1.0, N, N],
        ]),
        # Fewer condition rows than steps: the tail horizons are free.
        build_case("var2c_short_rows", y, 2, "c", 8, [
            [0.9, N, N], [0.8, N, N], [0.7, N, -0.4],
        ]),
        # VAR(3) without a constant, two series conditioned at scattered
        # horizons over a longer horizon.
        build_case("var3n_scattered", y, 3, "n", 10, [
            [N, N, 0.2], [N, -0.3, N], [N, N, N], [N, N, 0.1], [N, 0.4, N],
            [N, N, N], [N, N, N], [N, N, N], [0.5, N, N], [N, N, 0.0],
        ]),
        # First horizon fully pinned (the one-step-ahead 'nowcast is known'
        # case) plus one later cell.
        build_case("var1c_full_h1", y, 1, "c", 5, [
            [0.6, -0.1, 0.2], [N, N, N], [N, N, N], [N, 0.05, N],
        ], alpha=0.10),
        # Conditioning at exactly the unconditional forecast leaves the
        # path unchanged (u* = 0) but still shrinks the variance.
        build_case("var2c_at_unconditional", y, 2, "c", 6, [
            [N, N, N], [N, float(yhat2[1, 1]), N],
        ]),
    ]

    macro = macro_system()
    m_last = macro[-1]
    cases.append(build_case("macro_var2c_rates_on_hold", macro, 2, "c", 8, [
        [N, float(m_last[1]), 0.0], [N, float(m_last[1]), 0.0],
        [N, float(m_last[1]), N], [N, float(m_last[1]), N],
    ]))

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
            "note": (
                "Conditional (hard-path) VAR forecasts: closed form "
                "(Doan-Litterman-Sims / Waggoner-Zha, NumPy transcription of the "
                "documented formula, `point`/`cov`/`shocks`/...) and the "
                "Banbura-Giannone-Lenza Kalman-conditioning route "
                "(statsmodels VARMAX(...NaN future...).smooth(params), "
                "`bgl_point`/`bgl_cov`/`bgl_shocks`), asserted equal at 1e-10 "
                "at generation. Macro series are transformations of "
                "statsmodels' macrodata (100 dlog realgdp, 400 dlog cpi, "
                "diff tbilrate)."
            ),
        },
        "series": {"sim_var2_k3": y.tolist(), "macro_growth_infl_dtbill": macro.tolist()},
        "cases": cases,
    }
    for c in cases:
        c["series"] = "macro_growth_infl_dtbill" if c["name"].startswith("macro") else "sim_var2_k3"
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(cases)} cases")
    for c in cases:
        d = c["bgl_max_abs_dev"]
        print(f"  {c['name']:28s} p={c['p']} {c['trend']} H={c['steps']} m={c['n_constrained']} "
              f"maha={c['mahalanobis']:.4f} p={c['mahalanobis_pvalue']:.4f} "
              f"bgl dev path={d['point']:.1e} cov={d['cov']:.1e} shocks={d['shocks']:.1e}")


if __name__ == "__main__":
    main()
