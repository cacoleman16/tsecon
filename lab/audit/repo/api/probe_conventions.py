"""Convention compliance, probed not read (repo-wide API audit, item 4).

For every public callable with a registry entry, five malformed calls built
from its canonical call:

  (a) nan      — a NaN written into the first data array;
  (b) empty    — the first data array replaced by an empty one of the same rank;
  (c) ndim     — the first data array with the wrong number of dimensions
                 (a 1-D series passed as an (n, 1) column; a 2-D panel as its
                 first column);
  (d) string   — "nonsense_xyz" for the first string-valued parameter;
  (e) negative — -1 for the first integer-typed (count) parameter.

Each probe records the exception class (or ``silent`` when the call returns,
with whether the return carries non-finite values), whether the message
names the offending argument, whether it names the function, and whether it
states a fix (the teaching-error contract: "the message names the problem"
— docs/reference/testing.md Tier 3), or whether a Rust panic escapes as
``PanicException``. Every function runs in its own subprocess with a
timeout, so a hang or an abort is recorded rather than killing the sweep.

Run:  .venv-wt/bin/python lab/audit/repo/api/probe_conventions.py
Out:  lab/audit/repo/api/out/conventions.json, conventions.log
"""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

FIX_WORDS = re.compile(
    r"\b(expected|must|use |pass |at least|accepted|one of|try |reshape|instead|should|"
    r"requires?|required|ravel|wrap|provide|supply|choose|set |increase|reduce|drop|"
    r"remove|fill|impute|interpolate|see |needs?)\b",
    re.I,
)
ACCEPTED_LIST = re.compile(r"(\"[a-z_\-]+\"|'[a-z_\-]+')(\s*,\s*(\"[a-z_\-]+\"|'[a-z_\-]+'))+|one of|accepted|expected", re.I)


def _is_data(v):
    return isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.size > 0


def _first_data(args, kwargs, names):
    """(where, key, name, value) of the first float ndarray argument."""
    for i, a in enumerate(args):
        if _is_data(a):
            return "args", i, (names[i] if i < len(names) else f"arg{i}"), a
        if isinstance(a, list) and a and all(isinstance(e, (int, float)) for e in a):
            return "args", i, (names[i] if i < len(names) else f"arg{i}"), np.asarray(a, dtype=float)
        if isinstance(a, list) and a and all(isinstance(e, np.ndarray) for e in a):
            # ragged panel: poison the first unit
            return "args-list", i, (names[i] if i < len(names) else f"arg{i}"), a
    for k, v in kwargs.items():
        if _is_data(v):
            return "kwargs", k, k, v
    return None


def _set(args, kwargs, where, key, value):
    args = list(args)
    kwargs = dict(kwargs)
    if where == "args":
        args[key] = value
    elif where == "args-list":
        args[key] = value
    else:
        kwargs[key] = value
    return args, kwargs


def classify(exc, fn_name, param):
    if exc is None:
        return {"exc": "silent"}
    msg = str(exc)
    rec = {
        "exc": type(exc).__name__,
        "module": type(exc).__module__,
        "msg": msg[:400],
        "names_arg": bool(param) and re.search(rf"(?<![A-Za-z0-9_]){re.escape(param)}(?![A-Za-z0-9_])", msg) is not None,
        "names_fn": re.search(rf"\b{re.escape(fn_name)}\b", msg) is not None,
        "has_fix": FIX_WORDS.search(msg) is not None,
        "lists_accepted": ACCEPTED_LIST.search(msg) is not None,
        "panic": type(exc).__name__ == "PanicException",
    }
    return rec


def _nonfinite(res):
    stack = [res]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif isinstance(v, np.ndarray):
            if v.dtype.kind == "f" and v.size and not np.isfinite(v).all():
                return True
        elif isinstance(v, float) and not np.isfinite(v):
            return True
    return False


def run_one(name):
    import tsecon
    from registry import build

    fn = getattr(tsecon, name)
    try:
        names = list(inspect.signature(fn).parameters)
        params = list(inspect.signature(fn).parameters.values())
    except (ValueError, TypeError):
        names, params = [], []
    out = {"name": name}
    args, kwargs = build(name, T=200, seed=0)

    def attempt(label, a, k, param):
        t0 = time.perf_counter()
        try:
            res = fn(*a, **k)
            rec = {"exc": "silent", "nonfinite": _nonfinite(res)}
        except BaseException as exc:  # noqa: BLE001  PanicException is a BaseException
            rec = classify(exc, name, param)
        rec["param"] = param
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        out[label] = rec

    # stub types (for picking the int parameter)
    from probe_surface import stub_table

    stub = stub_table().get(name, {"params": []})
    stub_types = {p["name"]: (p["type"] or "") for p in stub["params"]}

    fd = _first_data(args, kwargs, names)
    if fd is None:
        out["nan"] = out["empty"] = out["ndim"] = {"exc": "n/a", "param": None}
    else:
        where, key, pname, val = fd
        if where == "args-list":
            unit = val[0].copy()
            unit.flat[len(unit.flat) // 2] = np.nan
            a, k = _set(args, kwargs, where, key, [unit] + list(val[1:]))
            attempt("nan", a, k, pname)
            a, k = _set(args, kwargs, where, key, [np.empty((0,) + val[0].shape[1:])] + list(val[1:]))
            attempt("empty", a, k, pname)
            bad = val[0].reshape(-1, 1) if val[0].ndim == 1 else val[0][:, 0]
            a, k = _set(args, kwargs, where, key, [bad] + list(val[1:]))
            attempt("ndim", a, k, pname)
        else:
            v = val.copy()
            v.flat[v.size // 2] = np.nan
            a, k = _set(args, kwargs, where, key, v)
            attempt("nan", a, k, pname)
            a, k = _set(args, kwargs, where, key, np.empty((0,) + val.shape[1:]))
            attempt("empty", a, k, pname)
            bad = val.reshape(-1, 1) if val.ndim == 1 else val[:, 0].copy()
            a, k = _set(args, kwargs, where, key, bad)
            attempt("ndim", a, k, pname)

    # (d) first string-valued parameter: runtime str default, else stub str type
    sparam = None
    for p in params:
        if isinstance(p.default, str):
            sparam = p.name
            break
    if sparam is None:
        for p in params:
            t = stub_types.get(p.name, "")
            if re.match(r"^str\b", t) and p.name not in ("summary",):
                sparam = p.name
                break
    if sparam is None:
        out["string"] = {"exc": "n/a", "param": None}
    else:
        idx = names.index(sparam)
        if idx < len(args):
            a, k = _set(args, kwargs, "args", idx, "nonsense_xyz")
        else:
            a, k = _set(args, kwargs, "kwargs", sparam, "nonsense_xyz")
        attempt("string", a, k, sparam)

    # (e) first integer-typed (count) parameter
    iparam = None
    for p in params:
        t = stub_types.get(p.name, "")
        if isinstance(p.default, bool):
            continue
        if isinstance(p.default, int) or re.match(r"^int\b", t) or re.match(r"^int \| None", t) or t == "int | None":
            if p.name in ("caused", "causing", "groups", "delays", "periods", "windows", "maturities", "seasonal", "slow_indices", "importance_groups", "order"):
                # sequence-of-int or (p,d,q) spec parameters: not a count
                if t.startswith("int") or isinstance(p.default, int):
                    pass
                else:
                    continue
            iparam = p.name
            break
    if iparam is None:
        out["negative"] = {"exc": "n/a", "param": None}
    else:
        idx = names.index(iparam)
        if idx < len(args):
            a, k = _set(args, kwargs, "args", idx, -1)
        else:
            a, k = _set(args, kwargs, "kwargs", iparam, -1)
        attempt("negative", a, k, iparam)
    return out


def main():
    from registry import NAMES

    names = NAMES if len(sys.argv) == 1 else sys.argv[1:]
    results = {}
    log = open(os.path.join(OUT, "conventions.log"), "w")
    for name in names:
        cmd = [sys.executable, os.path.abspath(__file__), "--one", name]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=HERE)
            if p.returncode != 0:
                rec = {"name": name, "crash": p.stderr[-800:], "returncode": p.returncode}
            else:
                rec = json.loads(p.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            rec = {"name": name, "crash": "TIMEOUT 300s"}
        results[name] = rec
        line = f"[{name}] " + " ".join(
            f"{lab}={rec.get(lab, {}).get('exc', '?')}" for lab in ("nan", "empty", "ndim", "string", "negative")
        ) + (f" CRASH {rec['crash'][:200]!r}" if "crash" in rec else "")
        print(line)
        log.write(line + "\n")
        log.flush()
    json.dump(results, open(os.path.join(OUT, "conventions.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--one":
        rec = run_one(sys.argv[2])
        print(json.dumps(rec, default=str))
    else:
        main()
