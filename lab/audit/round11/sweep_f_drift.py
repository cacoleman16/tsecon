"""Sweep F — signature / stub / docstring drift.

(a) inspect.signature(compiled) vs the .pyi stub: names, order, has-default;
(b) defaults the docstring prose states ("defaults to X", "(default X)",
    "X by default", "default: X") vs the runtime default;
(c) keyword names used in call snippets across docstrings, model cards and
    api.md that the function does not accept;
(d) accepted-value lists in docstrings (`param` is "a", "b" or "c"): probe
    each listed string on the canonical input and record refusals.

Run:  .venv-wt/bin/python lab/audit/round11/sweep_f_drift.py
Out:  lab/audit/round11/out/sweep_f.log, sweep_f.json
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import API_MD, CARDS, HERE, PYI, log  # noqa: E402
from registry import NAMES, build  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def stub_signatures():
    tree = ast.parse(open(PYI, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            a = node.args
            params = []
            pos = a.posonlyargs + a.args
            n_def = len(a.defaults)
            for i, p in enumerate(pos):
                has_def = i >= len(pos) - n_def
                params.append((p.arg, "pos", has_def))
            if a.vararg:
                params.append((a.vararg.arg, "var", False))
            for p, d in zip(a.kwonlyargs, a.kw_defaults):
                params.append((p.arg, "kw", d is not None))
            out[node.name] = params
    return out


def runtime_signature(name):
    fn = getattr(tsecon._core, name, None) or getattr(tsecon, name)
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = []
    for p in sig.parameters.values():
        kind = "var" if p.kind is p.VAR_POSITIONAL else ("kw" if p.kind is p.KEYWORD_ONLY else "pos")
        params.append((p.name, kind, p.default is not p.empty, p.default))
    return params


DEFAULT_PATTERNS = [
    # `name` ... defaults to X   |  `name` (default X)  | `name` default X
    r"`(?P<n>[A-Za-z_][A-Za-z_0-9]*)`[^`.;]{0,80}?\bdefaults? (?:to |is |= |: )?(?P<v>\"[^\"]*\"|'[^']*'|[-+]?\d+(?:\.\d+)?(?:e-?\d+)?|None|True|False)",
    r"`(?P<n>[A-Za-z_][A-Za-z_0-9]*)`[^`.;]{0,60}?\(default:? (?P<v>\"[^\"]*\"|'[^']*'|[-+]?\d+(?:\.\d+)?(?:e-?\d+)?|None|True|False)",
    # X (default) right after a param name: `name` ... "x" (default)
    r"`(?P<n>[A-Za-z_][A-Za-z_0-9]*)`[^`.;]{0,40}?(?P<v>\"[^\"]*\")\s*\((?:the )?default\)",
    # name=X (default)  |  name=X by default
    r"\b(?P<n>[A-Za-z_][A-Za-z_0-9]*)=(?P<v>\"[^\"]*\"|'[^']*'|[-+]?\d+(?:\.\d+)?(?:e-?\d+)?|None|True|False)\b[^.]{0,20}?\b(?:the )?default\b",
    # bare "name defaults to X" (no backticks)
    r"\b(?P<n>[a-z][a-z0-9_]{2,})\b defaults? to (?P<v>\"[^\"]*\"|'[^']*'|[-+]?\d+(?:\.\d+)?(?:e-?\d+)?|None|True|False)",
    # "default X" right after a str value list: `name`: "a" (constant, default)
    r"`(?P<n>[A-Za-z_][A-Za-z_0-9]*)`:? (?P<v>\"[^\"]*\") \([^)]*default\)",
]


def parse_value(v):
    v = v.strip()
    if v[0] in "\"'":
        return v[1:-1]
    if v in ("None", "True", "False"):
        return {"None": None, "True": True, "False": False}[v]
    try:
        return int(v)
    except ValueError:
        return float(v)


def doc_defaults(doc):
    flat = re.sub(r"\s+", " ", doc or "")
    found = []
    for pat in DEFAULT_PATTERNS:
        for m in re.finditer(pat, flat):
            try:
                found.append((m.group("n"), parse_value(m.group("v")), m.group(0)[:100]))
            except (ValueError, IndexError):
                pass
    return found


def kwargs_in_calls(text, name):
    """Keyword names used in `name(...)` call snippets inside `text`."""
    out = set()
    for m in re.finditer(rf"\b{re.escape(name)}\(", text):
        i = m.end()
        depth = 1
        j = i
        while j < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        body = text[i : j - 1]
        out |= set(re.findall(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\s*=(?!=)", body))
    return out


VALUE_LIST = re.compile(
    r"`?(?P<n>[A-Za-z_][A-Za-z_0-9]*)`?(?: is| takes|:| =|\s*\()?\s*"
    r"(?P<vals>(?:\"[A-Za-z_0-9+\-/. ]+\"(?:\s*(?:,|/|\|| or |, or )\s*)?){2,})"
)


def value_lists(doc):
    flat = re.sub(r"\s+", " ", doc or "")
    out = {}
    for m in VALUE_LIST.finditer(flat):
        vals = re.findall(r"\"([A-Za-z_0-9+\-/. ]+)\"", m.group("vals"))
        out.setdefault(m.group("n"), set()).update(vals)
    return out


def main():
    fh = open(os.path.join(OUT, "sweep_f.log"), "w")
    stubs = stub_signatures()
    report = {}
    cards = {fn: open(os.path.join(CARDS, fn), encoding="utf-8").read() for fn in os.listdir(CARDS) if fn.endswith(".md")}
    api = open(API_MD, encoding="utf-8").read()
    for name in NAMES:
        rec = {}
        fn = getattr(tsecon, name)
        rt = runtime_signature(name)
        st = stubs.get(name)
        rec["runtime"] = None if rt is None else [(p[0], p[1], p[2]) for p in rt]
        rec["stub"] = st
        if rt is None:
            log(fh, f"[{name}] no runtime signature")
        elif st is None:
            log(fh, f"[{name}] no stub entry")
        else:
            rt_view = [(p[0], p[1], p[2]) for p in rt]
            if [p[0] for p in rt_view] != [p[0] for p in st]:
                log(fh, f"[{name}] NAME/ORDER DRIFT runtime={[p[0] for p in rt_view]} stub={[p[0] for p in st]}")
                rec["drift"] = "names"
            else:
                for a, b in zip(rt_view, st):
                    if a[2] != b[2]:
                        log(fh, f"[{name}] DEFAULT-PRESENCE DRIFT on {a[0]}: runtime has_default={a[2]} stub={b[2]}")
                        rec.setdefault("drift", "has_default")
                    if a[1] != b[1]:
                        log(fh, f"[{name}] KIND DRIFT on {a[0]}: runtime {a[1]} stub {b[1]}")
                        rec.setdefault("drift", "kind")
        # (b) docstring-stated defaults
        rdef = {p[0]: p[3] for p in (rt or []) if p[2]}
        pnames = {p[0] for p in (rt or [])}
        stated = doc_defaults(fn.__doc__)
        mism = []
        for n, v, ctx in stated:
            if n not in pnames:
                continue
            if n in rdef:
                actual = rdef[n]
                same = actual == v or (isinstance(actual, float) and isinstance(v, (int, float)) and abs(actual - v) < 1e-12)
                if not same:
                    mism.append((n, v, actual, ctx))
                    log(fh, f"[{name}] DOC DEFAULT MISMATCH {n}: doc says {v!r}, runtime {actual!r}  <- {ctx!r}")
            else:
                mism.append((n, v, "<required>", ctx))
                log(fh, f"[{name}] DOC DEFAULT for a REQUIRED param {n}: doc says {v!r}  <- {ctx!r}")
        rec["doc_default_mismatch"] = mism
        # (c) kwargs in call snippets
        bad = {}
        for src, text in [("__doc__", fn.__doc__ or ""), ("api.md", api), *cards.items()]:
            used = kwargs_in_calls(text, name)
            unknown = sorted(u for u in used if u not in pnames)
            if unknown:
                bad[src] = unknown
                log(fh, f"[{name}] UNKNOWN KWARGS in {src} call snippet: {unknown}")
        rec["unknown_kwargs"] = bad
        # (d) accepted-value lists
        vl = value_lists(fn.__doc__)
        probes = {}
        for pname, vals in vl.items():
            if pname not in pnames:
                continue
            for v in sorted(vals):
                try:
                    args, kwargs = build(name, T=200, seed=0)
                    kwargs = {**kwargs, pname: v}
                    fn(*args, **kwargs)
                    probes[f"{pname}={v}"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    probes[f"{pname}={v}"] = f"{type(exc).__name__}: {msg[:160]}"
                    if re.search(r"unknown|unsupported|must be one of|invalid|not (?:a )?recogni|expected one of|got '", msg, re.I) and re.search(re.escape(v), msg):
                        log(fh, f"[{name}] LISTED VALUE REFUSED {pname}={v!r}: {msg[:140]}")
        rec["value_probes"] = probes
        report[name] = rec
    json.dump(report, open(os.path.join(OUT, "sweep_f.json"), "w"), indent=1, default=str)
    fh.close()


if __name__ == "__main__":
    main()
