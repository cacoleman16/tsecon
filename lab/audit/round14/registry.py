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
balanced_panel, lpdid_panel = _r11.balanced_panel, _r11.lpdid_panel

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

# --------------------------------------------------------------------------- #
# round 14: the thirteen callables of the 0.10.0 wave, plus the `mask=` cells
# --------------------------------------------------------------------------- #

NEW14 = (
    "unobserved_components",
    "tvp_regression",
    "ets_fit",
    "auto_ets",
    "var_conditional_forecast",
    "var_diagnostics",
    "var_select_order",
    "spa_test",
    "model_confidence_set",
    "stepm_test",
    "fmols",
    "dols",
    "ccr",
)

MASKED = (
    "panel_fe@mask",
    "panel_distributed_lag@mask",
    "panel_lp@mask",
    "lp_did@mask",
)


def uc_series(T, s):
    """Local linear trend + quarterly seasonal + noise (the UC card's shape)."""
    rng = _rng(s)
    t = np.arange(T)
    return 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 4) + np.cumsum(rng.standard_normal(T)) * 0.2


def tvp_data(T, s, k=2):
    """y with random-walk coefficients on k regressors."""
    rng = _rng(s)
    x = rng.standard_normal((T, k))
    beta = np.cumsum(rng.standard_normal((T, k)) * 0.05, axis=0)
    y = (x * beta).sum(axis=1) + rng.standard_normal(T) * 0.5
    return y, x


def ets_series(T, s, period=4):
    """Strictly positive trending seasonal series (multiplicative candidates are
    admissible, so `auto_ets` reaches its full candidate set)."""
    rng = _rng(s)
    t = np.arange(T)
    return 10.0 + 0.02 * t + 2.0 * np.sin(2 * np.pi * t / period) + rng.standard_normal(T) * 0.3


def losses(T, s, m=4):
    """A benchmark loss series and a T x m panel of competitor losses."""
    rng = _rng(s)
    bench = (rng.standard_normal(T) + 0.15) ** 2
    scale = np.linspace(0.95, 1.05, m)
    model = (rng.standard_normal((T, m)) * scale) ** 2
    return bench, model


def loss_panel(T, s, m=5):
    rng = _rng(s)
    return (rng.standard_normal((T, m)) * np.linspace(1.0, 1.06, m)) ** 2


def coint_system(T, s, k=2):
    """One cointegrating vector on k I(1) regressors."""
    rng = _rng(s)
    x = np.cumsum(rng.standard_normal((T, k)), axis=0)
    y = 1.5 * x[:, 0] - 0.5 * x[:, 1] + rng.standard_normal(T)
    return y, x


def cf_conditions(steps=4):
    """A hard path on series 1 for two horizons, one pin on series 0 later."""
    out = [[None] * 3 for _ in range(steps)]
    out[0][1] = 0.5
    out[1][1] = 0.6
    out[steps - 1][0] = 0.1
    return out


def ragged_mask(N, T, s):
    """0/1 observation flags: entity 0 enters late, entity 1 leaves early, one
    interior gap; every entity keeps most of its window."""
    rng = _rng(s + 31)
    m = np.ones((N, T))
    m[0, : max(1, T // 10)] = 0.0
    m[1, -max(1, T // 10) :] = 0.0
    if N > 2 and T > 20:
        m[2, T // 2 : T // 2 + 3] = 0.0
    del rng
    return m


reg("unobserved_components")(
    lambda T, s: ((uc_series(T, s),), {"level": "lltrend", "seasonal": 4, "n_starts": 1})
)
reg("tvp_regression")(lambda T, s: (tvp_data(T, s), {"n_starts": 1}))
reg("ets_fit")(
    lambda T, s: (
        (ets_series(T, s),),
        {
            "error": "add",
            "trend": "add",
            "seasonal": "add",
            "seasonal_periods": 4,
            "horizon": 4,
        },
    )
)
reg("auto_ets")(lambda T, s: ((ets_series(T, s),), {"seasonal_periods": 4, "horizon": 4}))


def ets_mul_series(T, s, period=4):
    """A strictly positive series whose noise SCALES with the level, so the
    `auto_ets` search picks a multiplicative-error model — the configuration on
    which `n_sim`/`seed` are live (a class-1 winner simulates nothing)."""
    rng = _rng(s)
    t = np.arange(T)
    return (10.0 + 0.05 * t) * (1.0 + 0.3 * np.sin(2 * np.pi * t / period)) * np.exp(
        rng.standard_normal(T) * 0.05
    )


# a seed-probe cell only: the search must land on a non-class-1 model
reg("auto_ets@mul")(
    lambda T, s: ((ets_mul_series(T, s),), {"seasonal_periods": 4, "horizon": 4, "n_sim": 200})
)
reg("var_conditional_forecast")(
    lambda T, s: ((stable_var(T, 3, s), cf_conditions(4)), {"lags": 2})
)
reg("var_diagnostics")(lambda T, s: ((stable_var(T, 3, s),), {"lags": 2, "nlags": 10}))
reg("var_select_order")(lambda T, s: ((stable_var(T, 3, s),), {"max_lags": 4}))
reg("spa_test")(lambda T, s: (losses(T, s), {"reps": 200, "seed": s}))
reg("model_confidence_set")(lambda T, s: ((loss_panel(T, s),), {"reps": 200, "seed": s}))
reg("stepm_test")(lambda T, s: (losses(T, s), {"reps": 200, "seed": s}))
reg("fmols")(lambda T, s: (coint_system(T, s), {}))
reg("dols")(lambda T, s: (coint_system(T, s), {"lags": 1, "leads": 1}))
reg("ccr")(lambda T, s: (coint_system(T, s), {}))

reg("panel_fe@mask")(
    lambda T, s: (
        (
            balanced_panel(6, T, s),
            np.array([balanced_panel(6, T, s + 1), balanced_panel(6, T, s + 2)]),
        ),
        {"mask": ragged_mask(6, T, s)},
    )
)
reg("panel_distributed_lag@mask")(
    lambda T, s: (
        dl_panel(6, T, s),
        {"lags": 1, "powers": 2, "eval_points": [10.0, 20.0, 30.0], "mask": ragged_mask(6, T, s)},
    )
)
reg("panel_lp@mask")(
    lambda T, s: (
        (balanced_panel(6, T, s), _rng(s + 9).standard_normal(T)),
        {"horizon": 6, "mask": ragged_mask(6, T, s)},
    )
)
reg("lp_did@mask")(
    lambda T, s: (
        lpdid_panel(12, T, s),
        {"pre_window": 2, "post_window": 3, "mask": np.ones((12, T))},
    )
)


NAMES = sorted(R)


def build(name, T=200, seed=0):
    args, kwargs = R[name](T, seed)
    return list(args), dict(kwargs)


if __name__ == "__main__":
    import tsecon

    public = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
    real = [n for n in NAMES if "@" not in n]
    missing = sorted(set(public) - set(real))
    extra = sorted(set(real) - set(public))
    print(f"public={len(public)} registry={len(real)} missing={missing} extra={extra}")
    for n in NEW14 + MASKED:
        a, k = build(n)
        r = getattr(tsecon, n.split("@")[0])(*a, **k)
        print(f"{n}: reached, {len(r)} keys")
