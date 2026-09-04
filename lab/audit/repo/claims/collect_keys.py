"""Call every public callable on its registry input and record the returned
keys (top level, and one level of nesting), plus the parameter list.

Writes ``out/returned_keys.json``; the names sweep reads it.

Run:  .venv-wt/bin/python lab/audit/repo/claims/collect_keys.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import OUT, log, public_callables, signature_params  # noqa: E402
from registry_ext import NAMES, build  # noqa: E402


def keyset(obj):
    top, nested = [], []
    if isinstance(obj, dict):
        top = [str(k) for k in obj]
        for v in obj.values():
            if isinstance(v, dict):
                nested.extend(str(k) for k in v)
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
                nested.extend(str(k) for k in v[0])
    return sorted(set(top)), sorted(set(nested) - set(top))


def main():
    fh = open(os.path.join(OUT, "collect_keys.log"), "w")
    out = {}
    public = public_callables()
    missing = sorted(set(public) - set(NAMES))
    if missing:
        log(fh, "REGISTRY MISSING:", missing)
    for name in public:
        rec = {"params": signature_params(name), "top": [], "nested": [], "error": None}
        if name in NAMES:
            try:
                a, k = build(name, T=200, seed=0)
                res = getattr(tsecon, name)(*a, **k)
                rec["top"], rec["nested"] = keyset(res)
                rec["type"] = type(res).__name__
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                log(fh, f"[{name}] call failed: {rec['error']}")
        else:
            rec["error"] = "not in registry"
        out[name] = rec
    json.dump(out, open(os.path.join(OUT, "returned_keys.json"), "w"), indent=1)
    n_ok = sum(1 for r in out.values() if r["error"] is None)
    log(fh, f"public={len(public)} called_ok={n_ok} failed={len(public) - n_ok}")
    fh.close()


if __name__ == "__main__":
    main()
