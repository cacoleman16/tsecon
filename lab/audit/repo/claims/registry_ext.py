"""Round-11 registry plus canonical inputs for the eleven 0.8.0 ML callables.

``lab/audit/round11/registry.py`` covers the 162 pre-wave functions; this
module re-exports it and registers the machine-learning wave so every one of
the 173 public callables has one small seeded input.

    from registry_ext import build, NAMES
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "round11"))
from registry import R, build as _build, reg  # noqa: E402


def _rng(seed):
    return np.random.default_rng(seed)


def _xy(T, s, p=4, ar=0.0):
    rng = _rng(s)
    x = rng.standard_normal((T, p))
    y = np.sin(2.0 * x[:, 0]) + (0.5 * x[:, 1] if p > 1 else 0.0) + 0.3 * rng.standard_normal(T)
    return x, y


reg("kernel_ridge")(lambda T, s: ((_xy(T, s, 2)[0], _xy(T, s, 2)[1]), {"alpha": 1.0, "kernel": "rbf", "x_test": _xy(T, s + 1, 2)[0][:5]}))
reg("kernel_regression")(lambda T, s: ((_xy(T, s, 1)[0][:, 0], _xy(T, s, 1)[1]), {"bandwidth": 0.5, "x_test": _xy(T, s + 1, 1)[0][:5, 0]}))
reg("group_lasso")(lambda T, s: ((_xy(T, s, 6)[0], _xy(T, s, 6)[1], [0, 0, 1, 1, 2, 2], 0.05), {}))
reg("post_lasso")(lambda T, s: ((_xy(T, s, 6)[0], _xy(T, s, 6)[1], 0.05), {}))


def _pds(T, s):
    rng = _rng(s)
    x = rng.standard_normal((T, 8))
    d = x[:, 0] - x[:, 1] + rng.standard_normal(T)
    y = 1.0 * d + 0.5 * x[:, 0] + 0.5 * x[:, 1] + rng.standard_normal(T)
    return y, d, x


reg("pds_lasso")(lambda T, s: (_pds(T, s), {}))
reg("regression_tree")(lambda T, s: ((_xy(T, s)[0], _xy(T, s)[1]), {"max_depth": 3, "x_test": _xy(T, s + 1)[0][:5]}))
reg("random_forest")(lambda T, s: ((_xy(T, s)[0], _xy(T, s)[1]), {"n_trees": 20, "seed": s, "x_test": _xy(T, s + 1)[0][:5]}))
reg("l1_trend_filter")(lambda T, s: ((np.cumsum(_rng(s).standard_normal(T)), 5.0), {}))
reg("boosting")(lambda T, s: ((_xy(T, s)[0], _xy(T, s)[1]), {"n_steps": 50, "x_test": _xy(T, s + 1)[0][:5]}))
reg("mlp_regression")(lambda T, s: ((_xy(T, s, 2)[0], _xy(T, s, 2)[1]), {"hidden": 8, "max_epochs": 30, "n_seeds": 1, "seed": s, "x_test": _xy(T, s + 1, 2)[0][:5]}))
reg("echo_state_network")(lambda T, s: ((_xy(T, s, 2)[0], _xy(T, s, 2)[1]), {"reservoir_size": 30, "washout": 10, "seed": s, "x_test": _xy(T, s + 1, 2)[0][:5]}))

NAMES = sorted(R)


def build(name, T=200, seed=0):
    return _build(name, T, seed)


if __name__ == "__main__":
    import tsecon

    public = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
    print("public", len(public), "registry", len(NAMES))
    print("missing", sorted(set(public) - set(NAMES)))
    print("extra", sorted(set(NAMES) - set(public)))
