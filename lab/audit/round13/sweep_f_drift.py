"""Sweep F — signature / stub / docstring drift for the six 0.9.0 callables.

(a) inspect.signature(compiled) vs the .pyi stub: names, order, has-default
    (all 179, cheap), and the runtime default VALUE vs the stub annotation;
(b) defaults the prose states — __doc__, stub docstring, the card's
    `| Argument | Default |` rows, and the guide's signature bullets — vs the
    runtime default;
(c) keyword names used in `fn(...)` call snippets across __doc__, the six card
    sections, guide chapters 13-16 and api.md that the function does not accept;
(d) every string option value a docstring lists, actually passed on the
    canonical input;
(e) inert-keyword refusal: an option documented to act in one mode only must
    raise when passed in another (and a documented no-op is recorded, not
    counted).

Run:  .venv/bin/python lab/audit/round13/sweep_f_drift.py
Out:  lab/audit/round13/out/sweep_f.txt, sweep_f.json
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import API_MD, CARDS, GUIDE_CHAPTERS, OUT, PYI, card_section, log  # noqa: E402
from registry import NAMES, NEW, build  # noqa: E402


def stub_params():
    tree = ast.parse(open(PYI, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            a = node.args
            pos = a.posonlyargs + a.args
            n_def = len(a.defaults)
            params = [(p.arg, i >= len(pos) - n_def, ast.unparse(p.annotation) if p.annotation else "") for i, p in enumerate(pos)]
            params += [(p.arg, d is not None, ast.unparse(p.annotation) if p.annotation else "") for p, d in zip(a.kwonlyargs, a.kw_defaults)]
            out[node.name] = params
    return out


def runtime_params(name):
    fn = getattr(tsecon._core, name, None) or getattr(tsecon, name)
    return [(p.name, p.default is not p.empty, p.default) for p in inspect.signature(fn).parameters.values()]


LITERAL = r"\"[^\"]*\"|'[^']*'|[-+]?\d+(?:\.\d+)?(?:e-?\d+)?|None|True|False|\([^)]*\)"
DOC_DEFAULT = [
    # "`name` (VALUE" — the 'Further arguments, with defaults' convention
    re.compile(r"`(?P<n>[A-Za-z_][A-Za-z_0-9]*)`\s*\((?P<v>" + LITERAL + r")(?:[;:,)]|\s)"),
    re.compile(r"`(?P<n>[A-Za-z_][A-Za-z_0-9]*)`[^`.;]{0,60}?\bdefaults? (?:to |is |= |: )?(?P<v>" + LITERAL + ")"),
    re.compile(r"`(?P<n>[A-Za-z_][A-Za-z_0-9]*)`[^`.;]{0,60}?\(default:? (?P<v>" + LITERAL + ")"),
    re.compile(r"\b(?P<n>[A-Za-z_][A-Za-z_0-9]*)=(?P<v>" + LITERAL + r")\b[^.]{0,20}?\b(?:the )?default\b"),
]


def parse_value(v):
    v = v.strip()
    if v[0] in "\"'":
        return v[1:-1]
    if v in ("None", "True", "False"):
        return {"None": None, "True": True, "False": False}[v]
    if v.startswith("("):
        try:
            return ast.literal_eval(v)
        except Exception:  # noqa: BLE001
            raise ValueError(v)
    try:
        return int(v)
    except ValueError:
        return float(v)


def stated_defaults(text, pnames):
    flat = re.sub(r"\s+", " ", text or "")
    found = []
    for pat in DOC_DEFAULT:
        for m in pat.finditer(flat):
            n = m.group("n")
            if n not in pnames:
                continue
            try:
                found.append((n, parse_value(m.group("v")), m.group(0)[:90]))
            except (ValueError, IndexError):
                pass
    return found


def same(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return abs(a - b) < 1e-12
    if isinstance(b, float) and isinstance(a, (int, float)) and not isinstance(a, bool):
        return abs(a - b) < 1e-12
    return a == b and type(a) is type(b) or (a is None and b is None)


def kwargs_in_calls(text, name):
    out = set()
    for m in re.finditer(rf"(?<!def )\b{re.escape(name)}\(", text):
        i, depth, j = m.end(), 1, m.end()
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
        out.setdefault(m.group("n"), set()).update(re.findall(r"\"([A-Za-z_0-9+\-/. ]+)\"", m.group("vals")))
    return out


def card_default_rows(name):
    """(arg, default-cell) rows of the `| Argument | Default |` tables that name `name`."""
    rows = []
    for fn in os.listdir(CARDS):
        if not fn.endswith(".md"):
            continue
        lines = open(os.path.join(CARDS, fn), encoding="utf-8").read().splitlines()
        header, current = None, None
        for line in lines:
            if not line.startswith("|"):
                header = None
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = [c.lower() for c in cells]
                continue
            if all(re.fullmatch(r"-+", c) for c in cells if c):
                continue
            if "argument" in header and "default" in header:
                col = {h: j for j, h in enumerate(header)}
                if "call" in col and cells[col["call"]]:
                    current = (re.findall(r"`([a-z_][a-z_0-9]*)`", cells[col["call"]]) or [None])[0]
                if current != name:
                    continue
                for a in re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", cells[col["argument"]]):
                    rows.append((fn, a, cells[col["default"]]))
    return rows


# (e) the inert-keyword contract: (kwargs that put the function in the mode
# where the keyword is documented NOT to act) -> must raise
INERT = {
    "setar_threshold_ci": [({"slope_level": None, "slope_region_level": 0.9}, "slope_region_level without slope_level")],
    "jsz_fit": [({"n_starts": 1, "seed": 3}, "seed with n_starts=1")],
    "panel_distributed_lag": [
        ({"bandwidth": 4.0}, "bandwidth under se_type=cluster"),
        ({"bandwidth": 4.0, "se_type": "nonrobust"}, "bandwidth under se_type=nonrobust"),
        ({"powers": 1, "eval_points": [10.0]}, "eval_points under powers=1"),
    ],
}
# documented no-ops (the prose says the argument does not change the output):
# recorded, compared, not counted as findings when the promise holds
DOCUMENTED_NOOP = {
    "var_girf": [("n_draws", {"n_draws": 8}), ("seed", {"seed": 99}), ("antithetic", {"antithetic": False})],
}


def main():
    fh = open(os.path.join(OUT, "sweep_f.txt"), "w")
    stubs = stub_params()
    report = {"a": {}, "b": {}, "c": {}, "d": {}, "e": {}}
    n_cand = 0
    # (a) every callable: names / order / has-default; plus the six: default values
    n_ok = 0
    for name in NAMES:
        rt, st = runtime_params(name), stubs.get(name)
        if st is None:
            log(fh, f"[{name}] (a) NO STUB ENTRY")
            n_cand += 1
            continue
        if [p[0] for p in rt] != [p[0] for p in st] or [p[1] for p in rt] != [p[1] for p in st]:
            log(fh, f"[{name}] (a) SIGNATURE DRIFT runtime={rt} stub={st}")
            n_cand += 1
        else:
            n_ok += 1
    log(fh, f"(a) signature vs stub: {n_ok}/{len(NAMES)} agree on names, order and has-default")
    for name in NEW:
        rt = runtime_params(name)
        report["a"][name] = {p[0]: repr(p[2]) for p in rt if p[1]}
        ell = [p[0] for p in rt if p[1] and p[2] is Ellipsis]
        if ell:
            log(fh, f"[{name}] (a) runtime default renders as Ellipsis for {ell} (round-11 OPEN-1 class)")
        for pname, has_def, ann in stubs[name]:
            d = dict((p[0], p[2]) for p in rt if p[1]).get(pname, inspect.Parameter.empty)
            if has_def and d is None and "None" not in ann:
                log(fh, f"[{name}] (a) stub annotates `{pname}: {ann}` but the runtime default is None")
                n_cand += 1
    # (b) prose defaults on four surfaces
    guide_text = "".join(open(g, encoding="utf-8").read() for g in GUIDE_CHAPTERS)
    stub_src = open(PYI, encoding="utf-8").read()
    tree = ast.parse(stub_src)
    stub_doc = {n.name: ast.get_docstring(n) or "" for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in NEW:
        fn = getattr(tsecon, name)
        rt = {p[0]: (p[1], p[2]) for p in runtime_params(name)}
        pnames = set(rt)
        found = []
        for sname, text in (("__doc__", fn.__doc__), ("stub", stub_doc[name]), ("card", card_section(name)), ("guide", guide_text)):
            for n, v, ctx in stated_defaults(text, pnames):
                has, actual = rt[n]
                if not has:
                    log(fh, f"[{name}] (b) {sname} states a default for REQUIRED `{n}`: {v!r} <- {ctx!r}")
                    found.append((sname, n, v, "required"))
                    n_cand += 1
                elif not same(actual, v):
                    log(fh, f"[{name}] (b) {sname} says `{n}` default {v!r}; runtime {actual!r} <- {ctx!r}")
                    found.append((sname, n, v, repr(actual)))
                    n_cand += 1
        for cardfn, a, cell in card_default_rows(name):
            if a not in rt:
                log(fh, f"[{name}] (b) {cardfn} table names `{a}`, not a parameter")
                n_cand += 1
                continue
            has, actual = rt[a]
            raw = cell.strip().strip("`")
            try:
                v = ast.literal_eval(raw)
            except Exception:  # noqa: BLE001
                continue
            if has and not same(actual, v):
                log(fh, f"[{name}] (b) {cardfn} table: `{a}` default {v!r}; runtime {actual!r}")
                found.append(("card-table", a, v, repr(actual)))
                n_cand += 1
        # guide signature bullets: `tsecon.name(... k=v ...)`
        for m in re.finditer(rf"`tsecon\.{name}\(([^`]*)\)`", guide_text):
            for k, v in re.findall(r"(\w+)=(\"[^\"]*\"|None|True|False|[-+]?\d+(?:\.\d+)?)", m.group(1)):
                if k in rt and rt[k][0] and not same(rt[k][1], parse_value(v)):
                    log(fh, f"[{name}] (b) guide bullet says {k}={v}; runtime default {rt[k][1]!r}")
                    found.append(("guide-bullet", k, v, repr(rt[k][1])))
                    n_cand += 1
        report["b"][name] = found
    # (c) kwargs in call snippets
    api = open(API_MD, encoding="utf-8").read()
    for name in NEW:
        pnames = {p[0] for p in runtime_params(name)}
        bad = {}
        srcs = [("__doc__", getattr(tsecon, name).__doc__ or ""), ("card", card_section(name)), ("api.md", api)]
        srcs += [(os.path.basename(g), open(g, encoding="utf-8").read()) for g in GUIDE_CHAPTERS]
        n_snip = 0
        for sname, text in srcs:
            n_snip += len(re.findall(rf"\b{name}\(", text))
            unknown = sorted(u for u in kwargs_in_calls(text, name) if u not in pnames)
            if unknown:
                bad[sname] = unknown
                log(fh, f"[{name}] (c) unknown kwargs in {sname} snippet: {unknown}")
                n_cand += 1
        report["c"][name] = {"snippets": n_snip, "unknown": bad}
        log(fh, f"[{name}] (c) {n_snip} call snippets scanned")
    # (d) listed string values
    for name in NEW:
        fn = getattr(tsecon, name)
        pnames = {p[0] for p in runtime_params(name)}
        probes = {}
        flat = re.sub(r"\s+", " ", fn.__doc__ or "")
        str_params = {p[0] for p in runtime_params(name) if p[1] and isinstance(p[2], str)}
        listed = {}
        for pname in str_params:
            for m in re.finditer(rf"`{pname}`", flat):
                window = re.split(r"`(?:" + "|".join(re.escape(q) for q in str_params if q != pname) + r")`", flat[m.end(): m.end() + 260])[0]
                listed.setdefault(pname, set()).update(re.findall(r'"([a-z_]+)"', window))
        for pname, vals in listed.items():
            for v in sorted(vals):
                args, kwargs = build(name, T=200, seed=0)
                try:
                    fn(*args, **{**kwargs, pname: v})
                    probes[f"{pname}={v}"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    probes[f"{pname}={v}"] = f"{type(exc).__name__}: {str(exc)[:140]}"
                    log(fh, f"[{name}] (d) LISTED VALUE REFUSED {pname}={v!r}: {str(exc)[:120]}")
                    n_cand += 1
        report["d"][name] = probes
        log(fh, f"[{name}] (d) {len(probes)} listed values probed: {probes}")
    # (e) inert keywords
    for name, cases in INERT.items():
        fn = getattr(tsecon, name)
        for extra, label in cases:
            args, kwargs = build(name, T=200, seed=0)
            try:
                fn(*args, **{**kwargs, **extra})
                log(fh, f"[{name}] (e) INERT KEYWORD ACCEPTED: {label}")
                report["e"][f"{name}:{label}"] = "accepted"
                n_cand += 1
            except ValueError as exc:
                report["e"][f"{name}:{label}"] = f"raises: {str(exc)[:100]}"
    for name, cases in DOCUMENTED_NOOP.items():
        fn = getattr(tsecon, name)
        args, kwargs = build(name, T=200, seed=0)
        base = fn(*args, **kwargs)
        for pname, extra in cases:
            other = fn(*args, **{**kwargs, **extra})
            diff = max(abs(np.asarray(base["girf"]) - np.asarray(other["girf"])).max() for _ in [0])
            report["e"][f"{name}:{pname}"] = f"documented no-op; max|d girf|={diff:.1e}"
            log(fh, f"[{name}] (e) documented no-op `{pname}`: max|d girf| = {diff:.1e} (promise: identical)")
    log(fh, f"\ncandidates raised: {n_cand}")
    json.dump(report, open(os.path.join(OUT, "sweep_f.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
