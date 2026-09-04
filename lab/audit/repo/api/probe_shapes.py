"""Multivariate input-shape convention (item 2, last bullet).

For every callable whose canonical call carries a 2-D array or a list of
arrays, record the actual shape passed (registry) and what the runtime
docstring says about orientation — the `(T, k)` / `(n, k)` /
`(observations, series)` family, the `(N, T)` / `(n_units, T)` panel
family, or a list of per-unit arrays — so the doc can state which functions
take which and where the orientation flips.

Run:  .venv-wt/bin/python lab/audit/repo/api/probe_shapes.py
Out:  lab/audit/repo/api/out/shapes.json
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from registry import NAMES, build  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

TK = re.compile(r"\((?:T|n|n_obs|nobs|N|t)\s*,\s*(?:k|K|p|n_vars|m|M|n_series|N|d|q|n_features|J|n_cols|n_x|k_x|n_curves)\)|\(observations?,\s*(?:series|variables|columns?)\)|\((?:n|T)\s*,\s*\d\)|\(n_obs\s*,\s*n_\w+\)|\(T\s*,\s*n\w*\)")
NT = re.compile(r"\((?:N|n_units|n|units|G|n_groups|n_entities|n_chains|chains|n_forecasters)\s*,\s*(?:T|t|n_obs|T_i|periods|n_periods|n_draws)\)|\(units?,\s*(?:time|periods)\)|\(entities,\s*time\)")
LIST = re.compile(r"\b(list|sequence) of (?:per-unit |per-entity |per-country |unit |entity )?(?:\d-D |1-D |2-D )?(?:arrays|series|matrices|vectors|panels)\b|\bone (?:array|matrix|series) per (?:unit|entity|country|forecaster|group|chain)\b|\bragged\b", re.I)


def main():
    out = {}
    for name in NAMES:
        fn = getattr(tsecon, name)
        doc = fn.__doc__ or ""
        try:
            pnames = list(inspect.signature(fn).parameters)
        except (ValueError, TypeError):
            pnames = []
        args, kwargs = build(name, T=200, seed=0)
        multi = []
        for i, a in enumerate(args):
            label = pnames[i] if i < len(pnames) else f"arg{i}"
            if isinstance(a, np.ndarray) and a.ndim >= 2:
                multi.append({"param": label, "shape": list(a.shape), "kind": "ndarray"})
            elif isinstance(a, list) and a and all(isinstance(e, np.ndarray) for e in a):
                multi.append({"param": label, "shape": [len(a), list(a[0].shape)], "kind": "list-of-arrays"})
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                multi.append({"param": k, "shape": list(v.shape), "kind": "ndarray"})
        if not multi:
            continue
        out[name] = {
            "inputs": multi,
            "doc_says_T_k": TK.search(doc) is not None,
            "doc_says_N_T": NT.search(doc) is not None,
            "doc_says_list": LIST.search(doc) is not None,
            "doc_snippets": sorted({m.group(0) for m in TK.finditer(doc)} | {m.group(0) for m in NT.finditer(doc)} | {m.group(0) for m in LIST.finditer(doc)})[:8],
        }
    json.dump(out, open(os.path.join(OUT, "shapes.json"), "w"), indent=1, sort_keys=True)
    for name, r in out.items():
        print(f"[{name}] {r['inputs']} T,k={r['doc_says_T_k']} N,T={r['doc_says_N_T']} list={r['doc_says_list']} {r['doc_snippets']}")
    print(len(out), "functions with a 2-D or list-of-arrays input")


if __name__ == "__main__":
    main()
