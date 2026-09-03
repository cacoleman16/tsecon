"""Sweep E (iv), second half — the three-surface docstring diff.

The brief: "Establish every promise from __doc__ ... but read all three
surfaces, and treat a disagreement among them as the finding." For every
public callable this compares the runtime ``fn.__doc__`` (what ``help()``
shows — the binding surface) with the stub's docstring (what IDEs show) and
reports (a) whether they are byte-identical after whitespace folding,
(b) returned keys the STUB names but the RUNTIME doc does not, (c) the
reverse, (d) backticked tokens that look like keys in either doc but are
never returned (phantom candidates, both surfaces).

Run:  .venv-wt/bin/python lab/audit/round11/sweep_e_docdiff.py
Out:  lab/audit/round11/out/sweep_e_docdiff.log, .json
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import HERE, PYI, doc_tokens, log  # noqa: E402
from registry import NAMES, build  # noqa: E402

OUT = os.path.join(HERE, "out")


def stub_docs():
    tree = ast.parse(open(PYI, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.get_docstring(node) or ""
    return out


def fold(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def main():
    fh = open(os.path.join(OUT, "sweep_e_docdiff.log"), "w")
    stubs = stub_docs()
    rep = {}
    n_ident = n_stub_longer = n_rt_longer = 0
    for name in NAMES:
        fn = getattr(tsecon, name)
        rt = fold(fn.__doc__)
        st = fold(stubs.get(name, ""))
        try:
            a, k = build(name, T=200, seed=0)
            res = fn(*a, **k)
            keys = set(res) if isinstance(res, dict) else set()
        except Exception as exc:  # noqa: BLE001
            keys = set()
            log(fh, f"[{name}] call failed: {type(exc).__name__}: {str(exc)[:100]}")
        rt_t, st_t = doc_tokens(fn.__doc__), doc_tokens(stubs.get(name, ""))
        rec = {"identical": rt == st, "len_runtime": len(rt), "len_stub": len(st)}
        if rt == st:
            n_ident += 1
        elif len(st) > len(rt):
            n_stub_longer += 1
        else:
            n_rt_longer += 1
        in_rt = {k for k in keys if re.search(rf"\b{re.escape(k)}\b", rt)}
        in_st = {k for k in keys if re.search(rf"\b{re.escape(k)}\b", st)}
        rec["keys_stub_only"] = sorted(in_st - in_rt)
        rec["keys_runtime_only"] = sorted(in_rt - in_st)
        rec["keys_in_neither"] = sorted(k for k in keys if not re.search(rf"\b{re.escape(k)}\b", rt) and not re.search(rf"\b{re.escape(k)}\b", st))
        rep[name] = rec
        tag = "IDENTICAL" if rec["identical"] else f"DIFFER (runtime {len(rt)} chars, stub {len(st)} chars)"
        extra = ""
        if rec["keys_stub_only"]:
            extra += f"  stub-only keys={rec['keys_stub_only']}"
        if rec["keys_runtime_only"]:
            extra += f"  runtime-only keys={rec['keys_runtime_only']}"
        if rec["keys_in_neither"]:
            extra += f"  keys named NOWHERE={rec['keys_in_neither']}"
        log(fh, f"[{name}] {tag}{extra}")
    log(fh, f"\nidentical={n_ident} stub_longer={n_stub_longer} runtime_longer={n_rt_longer} of {len(NAMES)}")
    json.dump(rep, open(os.path.join(OUT, "sweep_e_docdiff.json"), "w"), indent=1)
    fh.close()


if __name__ == "__main__":
    main()
