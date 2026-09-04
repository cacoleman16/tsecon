"""Shared helpers for the round-11 sweeps: walk results, compare bit-for-bit."""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PYI = os.path.join(REPO, "bindings", "python", "python", "tsecon", "__init__.pyi")
CARDS = os.path.join(REPO, "docs", "reference", "model-cards")
API_MD = os.path.join(REPO, "docs", "reference", "api.md")


def walk(obj, path="$"):
    """Yield (path, leaf) for every scalar leaf; arrays yield per-element? No:
    arrays yield ONE leaf (the array) so shape checks stay cheap."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, np.ndarray):
        yield path, obj
    elif isinstance(obj, (list, tuple)):
        # a nested numeric list is an array in disguise; try to lift it
        try:
            arr = np.asarray(obj, dtype=float)
            if arr.ndim >= 1 and arr.dtype.kind == "f":
                yield path, arr
                return
        except (ValueError, TypeError):
            pass
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def nonfinite_paths(obj):
    out = []
    for p, leaf in walk(obj):
        if isinstance(leaf, np.ndarray):
            if leaf.dtype.kind == "f" and not np.isfinite(leaf).all():
                n_nan = int(np.isnan(leaf).sum())
                n_inf = int(np.isinf(leaf).sum())
                out.append((p, f"array{leaf.shape} nan={n_nan} inf={n_inf}"))
        elif isinstance(leaf, float):
            if math.isnan(leaf):
                out.append((p, "nan"))
            elif math.isinf(leaf):
                out.append((p, "inf"))
    return out


def shapes(obj):
    return {p: tuple(leaf.shape) for p, leaf in walk(obj) if isinstance(leaf, np.ndarray)}


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def json_roundtrip(obj):
    """json.dumps with an ndarray default; returns (ok, message, decoded)."""
    try:
        s = json.dumps(obj, default=_default, allow_nan=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"json.dumps raised {type(exc).__name__}: {exc}", None
    try:
        back = json.loads(s)
    except Exception as exc:  # noqa: BLE001
        return False, f"json.loads raised {type(exc).__name__}: {exc}", None
    return True, "", back


def pickle_roundtrip(obj):
    try:
        back = pickle.loads(pickle.dumps(obj))
    except Exception as exc:  # noqa: BLE001
        return False, f"pickle raised {type(exc).__name__}: {exc}", None
    return True, "", back


def bits_equal(a, b, path="$"):
    """Bit-for-bit structural equality (NaN == NaN, -0.0 != 0.0 as bits)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False, f"{path}: keys differ {sorted(set(a) ^ set(b))}"
        for k in a:
            ok, why = bits_equal(a[k], b[k], f"{path}.{k}")
            if not ok:
                return ok, why
        return True, ""
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        try:
            aa = np.asarray(a, dtype=float)
            bb = np.asarray(b, dtype=float)
            if aa.shape != bb.shape:
                return False, f"{path}: shape {aa.shape} vs {bb.shape}"
            if aa.dtype.kind == "f" and aa.shape != ():
                if aa.view(np.uint64).tobytes() != bb.view(np.uint64).tobytes():
                    d = np.nanmax(np.abs(aa - bb)) if aa.size else 0.0
                    return False, f"{path}: bits differ (max|d|={d:.3e})"
                return True, ""
        except (ValueError, TypeError):
            pass
        if len(a) != len(b):
            return False, f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, why = bits_equal(x, y, f"{path}[{i}]")
            if not ok:
                return ok, why
        return True, ""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True, ""
        if a != b:
            return False, f"{path}: {a!r} vs {b!r}"
        return True, ""
    if a != b:
        return False, f"{path}: {a!r} vs {b!r}"
    return True, ""


def doc_tokens(doc):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", doc or ""))


def top_keys(obj):
    return set(obj.keys()) if isinstance(obj, dict) else set()


def card_for(name):
    """Return the concatenated text of every model card mentioning `name`."""
    texts = []
    for fn in sorted(os.listdir(CARDS)):
        if not fn.endswith(".md"):
            continue
        t = open(os.path.join(CARDS, fn), encoding="utf-8").read()
        if re.search(rf"\b{re.escape(name)}\b", t):
            texts.append((fn, t))
    return texts


def log(fh, *parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    fh.write(line + "\n")
    fh.flush()
