"""Build the master surface table: lab/audit/repo/api/surface.json.

For each of the public callables: runtime signature (name, kind, default),
stub annotation per parameter and the stub return annotation, the runtime
docstring's first line, the stub docstring's first line, the family (model
card), and — on the canonical registry call — the returned key set with a
value-kind per key (scalar / 1-D / 2-D / list / dict / str / bool / none),
one level of nested dicts included.

Run:  .venv-wt/bin/python lab/audit/repo/api/probe_surface.py
Out:  lab/audit/repo/api/surface.json
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import FAMILY, NAMES, PYI, build  # noqa: E402

import tsecon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "surface.json")


def kind_of(v, depth=0):
    """Classify a returned value; nested dicts carry their own key table."""
    if v is None:
        return "none", None
    if isinstance(v, (bool, np.bool_)):
        return "bool", None
    if isinstance(v, (int, np.integer)):
        return "int", None
    if isinstance(v, (float, np.floating)):
        return "float", None
    if isinstance(v, str):
        return "str", None
    if isinstance(v, np.ndarray):
        return f"{v.ndim}-D[{v.dtype.kind}]", None
    if isinstance(v, dict):
        nested = None
        if depth < 2:
            nested = {str(k): kind_of(x, depth + 1)[0] for k, x in v.items()}
        return "dict", nested
    if isinstance(v, tuple):
        return "tuple[" + ",".join(kind_of(e, depth + 1)[0] for e in v) + "]", None
    if isinstance(v, list):
        if not v:
            return "list[]", None
        if all(isinstance(e, (int, float, np.integer, np.floating)) and not isinstance(e, bool) for e in v):
            return "list[num]", None
        if all(isinstance(e, np.ndarray) for e in v):
            return "list[array]", None
        if all(isinstance(e, list) for e in v):
            return "list[list]", None
        if all(isinstance(e, dict) for e in v):
            return "list[dict]", None
        if all(isinstance(e, str) for e in v):
            return "list[str]", None
        if all(isinstance(e, (bool, np.bool_)) for e in v):
            return "list[bool]", None
        return "list[mixed]", None
    return type(v).__name__, None


def stub_table():
    """name -> {params: [(name, annotation, has_default)], returns: annotation}."""
    src = open(PYI, encoding="utf-8").read()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        a = node.args
        params = []
        pos = a.posonlyargs + a.args
        n_def = len(a.defaults)
        for i, arg in enumerate(pos):
            has_def = i >= len(pos) - n_def
            params.append({
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else None,
                "has_default": has_def,
                "kind": "positional",
            })
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            params.append({
                "name": arg.arg,
                "type": ast.unparse(arg.annotation) if arg.annotation else None,
                "has_default": d is not None,
                "kind": "keyword_only",
            })
        doc = ast.get_docstring(node) or ""
        out[node.name] = {
            "params": params,
            "returns": ast.unparse(node.returns) if node.returns else None,
            "doc": doc,
        }
    return out


def json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return f"<array{v.shape}>"
    return repr(v)


def first_line(doc):
    for line in (doc or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def main():
    stub = stub_table()
    public = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
    table = {}
    n_called = 0
    for name in public:
        fn = getattr(tsecon, name)
        rec = {"family": FAMILY.get(name, {}).get("family"), "card": FAMILY.get(name, {}).get("card")}
        # runtime signature
        try:
            sig = inspect.signature(fn)
            rec["params"] = [
                {
                    "name": p.name,
                    "kind": str(p.kind).lower().replace("parameter.", ""),
                    "default": None if p.default is inspect.Parameter.empty else json.loads(json.dumps(p.default, default=json_default)),
                    "has_default": p.default is not inspect.Parameter.empty,
                }
                for p in sig.parameters.values()
            ]
        except (ValueError, TypeError) as exc:
            rec["params"] = None
            rec["signature_error"] = str(exc)
        # stub
        s = stub.get(name)
        if s:
            types = {p["name"]: p["type"] for p in s["params"]}
            for p in rec.get("params") or []:
                p["stub_type"] = types.get(p["name"])
            rec["stub_params"] = [p["name"] for p in s["params"]]
            rec["stub_returns"] = s["returns"]
            rec["stub_doc_first_line"] = first_line(s["doc"])
            rec["stub_doc_len"] = len(s["doc"])
        else:
            rec["stub_params"] = None
        rec["doc_first_line"] = first_line(fn.__doc__)
        rec["doc_len"] = len(fn.__doc__ or "")
        # canonical call
        if name not in NAMES:
            rec["called"] = False
            rec["error"] = "no registry entry"
            table[name] = rec
            continue
        args, kwargs = build(name, T=200, seed=0)
        rec["canonical_kwargs"] = json.loads(json.dumps(kwargs, default=json_default))
        t0 = time.perf_counter()
        try:
            res = fn(*args, **kwargs)
            rec["called"] = True
            n_called += 1
        except Exception as exc:  # noqa: BLE001
            rec["called"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()[-800:]
            table[name] = rec
            print(f"[{name}] CALL FAILED {rec['error']}")
            continue
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        top, nested = kind_of(res)
        rec["return_kind"] = top
        if isinstance(res, dict):
            rec["keys"] = {}
            rec["nested"] = {}
            for k, v in res.items():
                kk, nn = kind_of(v)
                rec["keys"][str(k)] = kk
                if nn is not None:
                    rec["nested"][str(k)] = nn
        elif isinstance(res, tuple):
            rec["keys"] = None
            rec["tuple_kinds"] = [kind_of(e)[0] for e in res]
        else:
            rec["keys"] = None
        table[name] = rec
    json.dump(table, open(OUT, "w"), indent=1, sort_keys=True)
    kinds = {}
    for n, r in table.items():
        kinds[r.get("return_kind", "n/a")] = kinds.get(r.get("return_kind", "n/a"), 0) + 1
    print(f"public={len(public)} called={n_called} return kinds={kinds}")
    print("failed:", [n for n, r in table.items() if not r.get("called")])


if __name__ == "__main__":
    main()
