"""Canonical inputs for all 173 public callables (repo-wide API audit).

Re-exports the round-11 registry (162 callables at 0.7.0) and adds one
seeded, tiny builder for each of the eleven 0.8.0 machine-learning
callables, so every sweep in this directory drives the same call per
function:

    from registry import build, NAMES, FAMILY
    args, kwargs = build("random_forest", T=200, seed=0)

``FAMILY`` maps every callable to its model-card family, parsed from the
table in docs/reference/README.md (the surface a user reads), falling back
to the stub's ``# ---- section`` heading for the callables the table does
not list.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
R11 = os.path.join(REPO, "lab", "audit", "round11")

# Load the round-11 registry under its own module name (this file is also
# called registry.py, so a plain import would resolve to itself).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("registry_round11", os.path.join(R11, "registry.py"))
_r11 = importlib.util.module_from_spec(_spec)
sys.modules["registry_round11"] = _r11
_spec.loader.exec_module(_r11)
R, design, reg, yx, ar1, rw, _rng = _r11.R, _r11.design, _r11.reg, _r11.yx, _r11.ar1, _r11.rw, _r11._rng

PYI = os.path.join(REPO, "bindings", "python", "python", "tsecon", "__init__.pyi")
CARDS_README = os.path.join(REPO, "docs", "reference", "README.md")


def _ml_xy(T, s, p=4):
    x = design(T, p, s, const=False)
    rng = _rng(s + 100)
    beta = np.array([1.0, -0.5, 0.0, 0.0][:p])
    y = x @ beta + 0.5 * np.sin(x[:, 0]) + 0.5 * rng.standard_normal(T)
    return x, y


def _pds(T, s):
    rng = _rng(s)
    x = rng.standard_normal((T, 5))
    d = x @ np.array([0.5, 0.0, 0.3, 0.0, 0.0]) + rng.standard_normal(T)
    y = 1.0 * d + x @ np.array([0.7, 0.0, 0.0, 0.4, 0.0]) + rng.standard_normal(T)
    return y - y.mean(), d - d.mean(), (x - x.mean(0)) / x.std(0)


# ----- 0.8.0 machine-learning wave (eleven callables)
reg("kernel_ridge")(lambda T, s: ((*_ml_xy(T, s, 2),), {"alpha": 1.0}))
reg("kernel_regression")(lambda T, s: ((_ml_xy(T, s, 1)[0][:, 0], _ml_xy(T, s, 1)[1]), {"bandwidth": 0.5, "bandwidth_method": "fixed"}))
reg("group_lasso")(lambda T, s: ((*_ml_xy(T, s, 4), [0, 0, 1, 1], 0.1), {}))
reg("post_lasso")(lambda T, s: ((*_ml_xy(T, s, 4), 0.1), {}))
reg("pds_lasso")(lambda T, s: (_pds(T, s), {}))
reg("l1_trend_filter")(lambda T, s: ((rw(T, s), 10.0), {}))
reg("boosting")(lambda T, s: ((*_ml_xy(T, s, 4),), {"n_steps": 50}))
reg("mlp_regression")(lambda T, s: ((*_ml_xy(T, s, 2),), {"max_epochs": 50, "n_seeds": 2, "seed": s}))
reg("echo_state_network")(lambda T, s: ((*_ml_xy(T, s, 2),), {"reservoir_size": 50, "washout": 20, "seed": s}))
reg("regression_tree")(lambda T, s: ((*_ml_xy(T, s, 2),), {}))
reg("random_forest")(lambda T, s: ((*_ml_xy(T, s, 2),), {"n_trees": 20, "seed": s}))

NAMES = sorted(R)


def build(name, T=200, seed=0):
    args, kwargs = R[name](T, seed)
    return list(args), dict(kwargs)


def reseed(name, T, seed, kwargs):
    for key in ("seed", "band_seed", "rf_seed"):
        if key in kwargs:
            kwargs = {**kwargs, key: seed}
    return kwargs


# --------------------------------------------------------------------------- #
# family (model card) membership
# --------------------------------------------------------------------------- #
def _families_from_readme():
    out = {}
    if not os.path.exists(CARDS_README):
        return out
    for line in open(CARDS_README, encoding="utf-8"):
        m = re.match(r"\|\s*\[([^\]]+)\]\(model-cards/([^)]+)\)\s*\|(.*)\|\s*$", line)
        if not m:
            continue
        family, card, cell = m.group(1), m.group(2), m.group(3)
        for fn in re.findall(r"`([a-z_][a-z0-9_]*)`", cell):
            out.setdefault(fn, (family, card))
    return out


def _sections_from_stub():
    out = {}
    section = "General"
    for line in open(PYI, encoding="utf-8"):
        m = re.match(r"#\s*-+\s*(.+?)\s*$", line)
        if m:
            section = m.group(1).strip()
            continue
        m = re.match(r"def (\w+)\(", line)
        if m:
            out[m.group(1)] = section
    return out


_README = _families_from_readme()
_STUB = _sections_from_stub()
FAMILY = {}
for _n in NAMES:
    if _n in _README:
        FAMILY[_n] = {"family": _README[_n][0], "card": _README[_n][1], "source": "readme"}
    else:
        FAMILY[_n] = {"family": _STUB.get(_n, "General"), "card": None, "source": "stub-section"}


if __name__ == "__main__":
    import tsecon

    public = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
    missing = sorted(set(public) - set(NAMES))
    extra = sorted(set(NAMES) - set(public))
    print(f"public={len(public)} registry={len(NAMES)} missing={missing} extra={extra}")
    nocard = sorted(n for n in NAMES if FAMILY[n]["source"] != "readme")
    print(f"not in README family table ({len(nocard)}): {nocard}")
