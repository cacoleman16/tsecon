"""Canonical valid inputs for all 173 public callables (repo audit, security sweep).

Extends the round-11 registry (``lab/audit/round11/registry.py``, 162 entries
frozen at the 0.7.0 surface) with the eleven machine-learning-wave callables
that landed in 0.8.0, so every sweep in this directory reaches 173/173.

    from registry_ml import build, NAMES
    args, kwargs = build("random_forest", T=200, seed=0)
    tsecon.random_forest(*args, **kwargs)

Draw counts are deliberately tiny; the point of a canonical call is to be a
*valid* starting point that the adversarial mutations then corrupt one
argument at a time.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "round11"))

from registry import R, ar1, design, reg, yx  # noqa: E402
from registry import build as _build  # noqa: E402


def _xy(T, s, k=3):
    """A (T, k) design without a constant and a linear-signal outcome."""
    X = design(T, k, s, const=False)
    y = yx(T, s, k)[0]
    return X, y


def _pds(T, s, k=5):
    """Outcome, treatment, and controls for a partialling-out lasso."""
    rng = np.random.default_rng(s)
    X = rng.standard_normal((T, k))
    d = X @ np.r_[1.0, 0.5, np.zeros(k - 2)] + rng.standard_normal(T)
    y = 0.5 * d + X @ np.r_[0.8, 0.0, 0.3, np.zeros(k - 3)] + rng.standard_normal(T)
    return y, d, X


reg("boosting")(lambda T, s: (_xy(T, s), {"n_steps": 20}))
reg("echo_state_network")(lambda T, s: (_xy(T, s), {"reservoir_size": 20, "washout": 5, "seed": s}))
reg("group_lasso")(lambda T, s: ((*_xy(T, s, 4), [0, 0, 1, 1], 0.1), {}))
reg("kernel_regression")(lambda T, s: (_xy(T, s, 1), {"bandwidth": 0.5}))
reg("kernel_ridge")(lambda T, s: (_xy(T, s, 2), {}))
reg("l1_trend_filter")(lambda T, s: ((ar1(T, s), 1.0), {}))
reg("mlp_regression")(lambda T, s: (_xy(T, s, 2), {"hidden": [4], "max_epochs": 20, "seed": s}))
reg("pds_lasso")(lambda T, s: (_pds(T, s), {}))
reg("post_lasso")(lambda T, s: ((*_xy(T, s), 0.1), {}))
reg("random_forest")(lambda T, s: (_xy(T, s), {"n_trees": 10, "seed": s}))
reg("regression_tree")(lambda T, s: (_xy(T, s), {"max_depth": 3}))

NAMES = sorted(R)


def build(name, T=200, seed=0):
    return _build(name, T=T, seed=seed)


if __name__ == "__main__":
    import tsecon

    public = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
    missing = sorted(set(public) - set(NAMES))
    extra = sorted(set(NAMES) - set(public))
    print(f"public={len(public)} registry={len(NAMES)} missing={missing} extra={extra}")
    bad = 0
    for n in NAMES:
        args, kwargs = build(n)
        try:
            getattr(tsecon, n)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"CANONICAL FAILS {n}: {type(exc).__name__}: {str(exc)[:160]}")
    print(f"canonical calls: {len(NAMES) - bad}/{len(NAMES)} succeed")
