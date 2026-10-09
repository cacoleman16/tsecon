"""Golden fixtures for `dfm_smooth`, the fixed-parameter dynamic-factor Kalman
smoother (crate `tsecon-nowcast`: `smooth_fixed` and `smooth_standardized`).

WHAT IS PINNED
--------------
For r factors following a VAR(p) and white-noise idiosyncratic errors,

    y_t = Lambda F_t + e_t,                e_t ~ N(0, diag(idiosyncratic))
    F_t = A_1 F_{t-1} + ... + A_p F_{t-p} + eta_t,   eta_t ~ N(0, factor_cov)

in the stacked state alpha_t = [F_t', ..., F_{t-p+1}']' (m = r p): Z = [Lambda|0],
T = [A_1 ... A_p] over identity sub-blocks, R = [I_r; 0], Q = factor_cov,
H = diag(idiosyncratic), stationary (Lyapunov) initialization. At FIXED
parameters the Kalman filter/smoother output is a deterministic function of the
panel, so it is pinned against an independent package:

  * Case `dynamic_factor_r2_p2` — statsmodels `DynamicFactor(Y, k_factors=2,
    factor_order=2, error_order=0).smooth(params)` on a ragged panel (NaN at the
    edge and one interior gap): `llf`, `smoothed_state` (T x m) and
    `smoothed_state_cov` (T x m x m). statsmodels normalizes Q = I. Parameter
    order (asserted against `param_names`): the loadings row-major
    (loading.f1.y1, loading.f2.y1, ...), then sigma2.y1.., then the factor VAR
    per equation, row i of [A_1 | A_2].
  * Case `kalman_smoother_q` — Q != I, which DynamicFactor cannot express:
    statsmodels' generic `KalmanSmoother` with the same system matrices and
    `initialize_stationary()`, on the same ragged panel. A canary asserts the
    generic smoother with Q = I reproduces the DynamicFactor case, so both legs
    are one reference.
  * Case `standardized_r1_p3` — the standardization and the edge nowcast that
    `dfm_smooth(..., center=, scale=)` adds: a raw panel X, moments c and s, the
    standardized panel z = (X - c) / s computed elementwise in IEEE double (the
    same two operations the crate performs, so z is bit-identical), the generic
    KalmanSmoother on z (r = 1, p = 3, Q = 0.8), the smoothed factor at the last
    period (edge_factor) and nowcast = c + s * (Lambda @ edge_factor). The
    log-likelihood is that of z (no Jacobian term).

statsmodels' filter is the multivariate (or its own univariate) Kalman filter
and tsecon's is the univariate sequential filter; for diagonal H they agree to
roundoff, not bitwise (the agreement achieved is recorded by the Rust test).

Data: seeded NumPy `default_rng` draws through the DGPs below (no dataset).
NaN (missing) cells are written as JSON null.

This generator NEVER imports tsecon. Doubles are written with json's shortest
round-trip repr, which the Rust golden tests parse to identical bits
(serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_dfm_smooth_fixtures.py
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas
import scipy
import statsmodels
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

HERE = Path(__file__).resolve().parent
OUT = HERE / "dfm_smooth.json"
SEED = 20261009


def nan_to_none(a: np.ndarray) -> list:
    """Nested list with NaN -> None (JSON null)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return [None if np.isnan(v) else float(v) for v in a]
    return [nan_to_none(row) for row in a]


def system(lam: np.ndarray, ar: np.ndarray, q: np.ndarray, idio: np.ndarray):
    n, r = lam.shape
    m = ar.shape[1]
    t_mat = np.zeros((m, m))
    t_mat[:r] = ar
    t_mat[r:, : m - r] = np.eye(m - r)
    z_mat = np.zeros((n, m))
    z_mat[:, :r] = lam
    r_mat = np.zeros((m, r))
    r_mat[:r] = np.eye(r)
    return z_mat, t_mat, r_mat, q, np.diag(idio)


def generic_smooth(y: np.ndarray, lam, ar, q, idio):
    n, r = lam.shape
    m = ar.shape[1]
    z_mat, t_mat, r_mat, q_mat, h_mat = system(lam, ar, q, idio)
    ks = KalmanSmoother(k_endog=n, k_states=m, k_posdef=r)
    ks.bind(np.ascontiguousarray(y))
    ks["design"] = z_mat
    ks["obs_cov"] = h_mat
    ks["transition"] = t_mat
    ks["selection"] = r_mat
    ks["state_cov"] = q_mat
    ks.initialize_stationary()
    out = ks.smooth()
    return (
        float(np.sum(out.llf_obs)),
        np.asarray(out.smoothed_state).T.copy(),
        np.moveaxis(np.asarray(out.smoothed_state_cov), 2, 0).copy(),
    )


def simulate(rng, n, t, lam, ar, q, idio, burn=200):
    r = lam.shape[1]
    p = ar.shape[1] // r
    chol = np.linalg.cholesky(q)
    f = np.zeros((t + burn, r))
    for s in range(p, t + burn):
        acc = chol @ rng.standard_normal(r)
        for lag in range(p):
            acc = acc + ar[:, lag * r:(lag + 1) * r] @ f[s - 1 - lag]
        f[s] = acc
    f = f[burn:]
    return f @ lam.T + rng.standard_normal((t, n)) * np.sqrt(idio)


def case_dynamic_factor(rng):
    n, t, r, p = 6, 150, 2, 2
    a1 = np.array([[0.5, 0.1], [0.0, 0.4]])
    a2 = np.array([[0.2, 0.0], [0.05, 0.1]])
    ar = np.c_[a1, a2]
    lam = rng.normal(size=(n, r))
    idio = rng.uniform(0.2, 0.6, n)
    y = simulate(rng, n, t, lam, ar, np.eye(r), idio)
    # Ragged edge: the last period half released, one series a period behind,
    # plus one interior gap (a skipped release).
    y[-1, :3] = np.nan
    y[-2, 0] = np.nan
    y[70, 4] = np.nan

    mod = DynamicFactor(y, k_factors=r, factor_order=p, error_order=0)
    names = list(mod.param_names)
    expected_names = (
        [f"loading.f{k + 1}.y{i + 1}" for i in range(n) for k in range(r)]
        + [f"sigma2.y{i + 1}" for i in range(n)]
        + [f"L{lag + 1}.f{j + 1}.f{i + 1}" for i in range(r) for lag in range(p) for j in range(r)]
    )
    assert names == expected_names, names
    params = np.r_[lam.ravel(), idio, ar.ravel()]
    res = mod.smooth(params, transformed=True)
    llf = float(res.llf)
    state = np.asarray(res.smoothed_state).T.copy()
    state_cov = np.moveaxis(np.asarray(res.smoothed_state_cov), 2, 0).copy()
    # The DynamicFactor transition is exactly the documented companion layout.
    assert np.array_equal(np.asarray(mod.ssm["transition"])[:r], ar)

    # Canary: the generic KalmanSmoother at Q = I is the same reference.
    llf_g, state_g, cov_g = generic_smooth(y, lam, ar, np.eye(r), idio)
    canary = {
        "llf_abs_diff": abs(llf_g - llf),
        "state_max_abs_diff": float(np.max(np.abs(state_g - state))),
        "state_cov_max_abs_diff": float(np.max(np.abs(cov_g - state_cov))),
    }
    assert canary["llf_abs_diff"] < 1e-9 and canary["state_max_abs_diff"] < 1e-10, canary
    return {
        "name": "dynamic_factor_r2_p2",
        "source": "statsmodels DynamicFactor(k_factors=2, factor_order=2, error_order=0).smooth(params)",
        "n_factors": r,
        "factor_order": p,
        "panel": nan_to_none(y),
        "loadings": lam.tolist(),
        "factor_ar": ar.tolist(),
        "factor_cov": np.eye(r).tolist(),
        "idiosyncratic": idio.tolist(),
        "param_names": names,
        "loglik": llf,
        "smoothed_state": state.tolist(),
        "smoothed_state_cov": state_cov.tolist(),
        "generic_vs_dynamic_factor": canary,
    }, y, lam, ar, idio


def case_generic_q(y, lam, ar, idio):
    q = np.array([[1.3, 0.4], [0.4, 0.7]])
    llf, state, cov = generic_smooth(y, lam, ar, q, idio)
    return {
        "name": "kalman_smoother_q",
        "source": "statsmodels KalmanSmoother (generic), initialize_stationary(), Q != I",
        "n_factors": lam.shape[1],
        "factor_order": ar.shape[1] // lam.shape[1],
        "panel": nan_to_none(y),
        "loadings": lam.tolist(),
        "factor_ar": ar.tolist(),
        "factor_cov": q.tolist(),
        "idiosyncratic": idio.tolist(),
        "loglik": llf,
        "smoothed_state": state.tolist(),
        "smoothed_state_cov": cov.tolist(),
    }


def case_standardized(rng):
    n, t, r, p = 5, 120, 1, 3
    ar = np.array([[0.6, -0.2, 0.15]])
    q = np.array([[0.8]])
    lam = rng.uniform(0.4, 1.2, size=(n, r))
    idio = rng.uniform(0.3, 0.7, n)
    z_true = simulate(rng, n, t, lam, ar, q, idio)
    center = np.array([2.5, -1.0, 103.0, 0.04, 12.0])
    scale = np.array([0.7, 3.1, 12.5, 0.011, 1.9])
    x = center + scale * z_true
    # A staggered ragged edge: the series are released one to three periods late.
    x[-1, [0, 2, 3]] = np.nan
    x[-2, [0, 3]] = np.nan
    x[-3, 3] = np.nan
    z = (x - center) / scale  # the crate's exact operations, elementwise
    llf, state, cov = generic_smooth(z, lam, ar, q, idio)
    edge = state[-1, :r]
    nowcast = center + scale * (lam @ edge)
    return {
        "name": "standardized_r1_p3",
        "source": "statsmodels KalmanSmoother (generic) on z = (X - center) / scale; "
                  "nowcast = center + scale * (loadings @ smoothed_factor[T-1])",
        "n_factors": r,
        "factor_order": p,
        "panel": nan_to_none(x),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "loadings": lam.tolist(),
        "factor_ar": ar.tolist(),
        "factor_cov": q.tolist(),
        "idiosyncratic": idio.tolist(),
        "loglik": llf,
        "smoothed_state": state.tolist(),
        "smoothed_state_cov": cov.tolist(),
        "edge_factor": edge.tolist(),
        "nowcast": nowcast.tolist(),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    a, y, lam, ar, idio = case_dynamic_factor(rng)
    b = case_generic_q(y, lam, ar, idio)
    c = case_standardized(rng)
    fixture = {
        "_meta": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "pandas": pandas.__version__,
            "seed": SEED,
            "note": (
                "Fixed-parameter DFM Kalman smoother: statsmodels DynamicFactor.smooth "
                "(Q = I) and the generic KalmanSmoother (Q != I; standardized panel) at "
                "given parameters, stationary initialization. smoothed_state is T x m and "
                "smoothed_state_cov T x m x m (statsmodels' arrays moved to time-first). "
                "NaN cells of the panels are null. loglik is that of the panel the "
                "smoother saw (the standardized z for the standardized case)."
            ),
        },
        "cases": [a, b, c],
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for case in fixture["cases"]:
        print(f"  {case['name']}: loglik = {case['loglik']!r}")
    print("  generic vs DynamicFactor canary:", a["generic_vs_dynamic_factor"])
    print("  standardized nowcast:", c["nowcast"])


if __name__ == "__main__":
    main()
