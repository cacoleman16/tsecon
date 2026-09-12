"""Canonical valid inputs for all 179 public callables (audit round 13).

Extends the security sweep's registry (``lab/audit/repo/security/registry_ml.py``
— the round-11 registry plus the 0.8.0 machine-learning wave, 173 entries) with
the six callables that landed in 0.9.0, so every round-13 sweep reaches
179/179:

    from registry import build, NAMES, NEW
    args, kwargs = build("var_girf", T=200, seed=0)
    tsecon.var_girf(*args, **kwargs)

``NEW`` is the six-name scope of this round. Draw counts and start counts are
deliberately small so the sweeps finish; the seed-taking calls put the seed in
``kwargs`` so ``reseed`` can move it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(path, modname):
    """Load a sibling registry under `modname` (round 11's is also called
    `registry`, so it is swapped into sys.modules only while the security
    registry — which imports it by that name — is being executed)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_self = sys.modules.get(__name__)
_r11 = _load(os.path.join(HERE, "..", "round11", "registry.py"), "registry")
_ml = _load(os.path.join(HERE, "..", "repo", "security", "registry_ml.py"), "registry_ml")
if _self is not None:
    sys.modules[__name__] = _self
sys.path[:] = [p for p in sys.path if not (p.rstrip('/').endswith('round11') or p.rstrip('/').endswith('security'))]
R, _rng, _setar, reg, reseed, stable_var, yield_panel = (
    _r11.R, _r11._rng, _r11._setar, _r11.reg, _r11.reseed, _r11.stable_var, _r11.yield_panel,
)

NEW = (
    "setar_threshold_ci",
    "var_girf",
    "threshold_var_girf",
    "jsz_fit",
    "jsz_loadings",
    "panel_distributed_lag",
)


def tvar(T, s):
    """Bivariate TVAR(1): persistent below 0, transient above (guide 13's DGP,
    100 burn-in rows discarded)."""
    rng = _rng(s)
    a_low = np.array([[0.8, 0.1], [0.1, 0.7]])
    a_high = np.array([[0.2, 0.0], [0.0, 0.3]])
    y = np.zeros((T + 100, 2))
    for t in range(1, T + 100):
        a = a_low if y[t - 1, 0] <= 0.0 else a_high
        y[t] = a @ y[t - 1] + 0.6 * rng.standard_normal(2)
    return y[100:]


def dl_panel(N, T, s):
    """Country-year panel with a known quadratic weather response (guide 14's
    DGP, shrunk): returns ``(growth N x T, regressors 1 x N x T)``."""
    rng = _rng(s)
    mu = rng.normal(20.0, 5.0, N)
    temp = mu[:, None] + rng.standard_normal((N, T))
    growth = (
        rng.normal(0.0, 1.0, N)[:, None]
        + rng.normal(0.0, 0.5, T)[None, :]
        + 0.30 * temp
        - 0.0075 * temp**2
        + rng.standard_normal((N, T))
    )
    return growth, temp[None]


JSZ_MATS = [1, 3, 6, 12, 24, 60, 120]


def jsz_params():
    return np.array([0.998, 0.96, 0.90]), 1e-4, 1e-6 * np.eye(3)


reg("setar_threshold_ci")(
    lambda T, s: ((_setar(T, s), 1), {"slope_level": 0.95, "null_threshold": 0.0})
)
reg("var_girf")(lambda T, s: ((stable_var(T, 3, s), 2), {"horizon": 6}))
reg("threshold_var_girf")(
    lambda T, s: ((tvar(T, s), 1), {"horizon": 6, "n_draws": 20, "seed": s})
)
reg("jsz_fit")(
    lambda T, s: ((yield_panel(T, s)[0], list(range(1, 13))), {"n_starts": 2, "seed": s})
)
reg("jsz_loadings")(
    lambda T, s: ((*jsz_params(), list(JSZ_MATS)), {"periods_per_year": 12.0})
)
reg("panel_distributed_lag")(
    lambda T, s: (dl_panel(6, T, s), {"lags": 1, "powers": 2, "eval_points": [10.0, 20.0, 30.0]})
)

NAMES = sorted(R)


def build(name, T=200, seed=0):
    args, kwargs = R[name](T, seed)
    return list(args), dict(kwargs)


if __name__ == "__main__":
    import tsecon

    public = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
    missing = sorted(set(public) - set(NAMES))
    extra = sorted(set(NAMES) - set(public))
    print(f"public={len(public)} registry={len(NAMES)} missing={missing} extra={extra}")
    for n in NEW:
        a, k = build(n)
        r = getattr(tsecon, n)(*a, **k)
        print(f"{n}: reached, {len(r)} keys")
