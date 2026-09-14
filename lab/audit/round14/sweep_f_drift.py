"""Sweep F — signature / stub / docstring drift for the 0.10.0 surface.

(a) inspect.signature(compiled) vs the .pyi stub: names, order, has-default
    (all 192, cheap), the runtime default VALUE vs the stub annotation, and a
    library-wide re-check that no default renders as `Ellipsis`;
(b) defaults the prose states — __doc__, stub docstring, the card's
    `| Argument | Default |` rows, and the guide's signature bullets — vs the
    runtime default;
(c) keyword names used in `fn(...)` call snippets across __doc__, the card
    sections, the guide chapters the wave touched, the Blanchard-Quah page and
    api.md that the function does not accept;
(d) every string option value a docstring lists, actually passed on the
    canonical input;
(e) inert-keyword refusal: an option documented to act in one mode only must
    raise when passed in another (and a documented no-op is recorded, not
    counted).

Run:  .venv/bin/python lab/audit/round14/sweep_f_drift.py
Out:  lab/audit/round14/out/sweep_f.txt, sweep_f.json
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
from common import API_MD, CARDS, GUIDE_CHAPTERS14 as GUIDE_CHAPTERS, OUT, PYI, card_section, log  # noqa: E402
from registry import MASKED, NAMES, NEW14, build  # noqa: E402

SCOPE = NEW14 + ("panel_fe", "panel_distributed_lag", "panel_lp", "lp_did")


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
import numpy as _np  # noqa: E402

def _ragged(n, t):
    m = _np.ones((n, t))
    m[0, : t // 10] = 0.0
    m[1, -(t // 10) :] = 0.0
    return m


_M = _ragged(12, 200)

INERT = {
    "unobserved_components": [
        ({"seasonal": None, "stochastic_seasonal": True}, "stochastic_seasonal without seasonal"),
        ({"freq_seasonal_harmonics": [2]}, "freq_seasonal_harmonics without freq_seasonal"),
        ({"stochastic_freq_seasonal": [True]}, "stochastic_freq_seasonal without freq_seasonal"),
        ({"damped_cycle": True}, "damped_cycle without cycle"),
        ({"stochastic_cycle": True}, "stochastic_cycle without cycle"),
        ({"cycle_period_bounds": [6.0, 32.0]}, "cycle_period_bounds without cycle"),
        ({"forecast_exog": _np.ones((3, 1))}, "forecast_exog without exog"),
    ],
    "ets_fit": [
        ({"trend": None, "damped": True}, "damped without trend"),
        ({"seasonal": None, "seasonal_periods": 4}, "seasonal_periods without seasonal"),
        ({"initial_states": [1.0, 0.1, 0.0, 0.0, 0.0, 0.0]}, "initial_states under initialization=estimated"),
        ({"smoothing_params": [0.3, 0.1, 0.1]}, "smoothing_params under initialization=estimated"),
        ({"initialization": "heuristic", "smoothing_params": [0.3, 0.1, 0.1], "optimizer": "bfgs"},
         "optimizer with smoothing_params"),
        ({"initialization": "heuristic", "smoothing_params": [0.3, 0.1, 0.1], "max_iter": 10},
         "max_iter with smoothing_params"),
        ({"horizon": 0, "level": 0.9}, "level with horizon=0"),
        ({"horizon": 0, "n_sim": 100}, "n_sim with horizon=0"),
        ({"horizon": 0, "seed": 3}, "seed with horizon=0"),
        ({"n_sim": 100}, "n_sim on a class-1 model"),
        ({"seed": 3}, "seed on a class-1 model"),
    ],
    "auto_ets": [
        ({"horizon": 0, "n_sim": 100}, "n_sim with horizon=0"),
        ({"horizon": 0, "seed": 3}, "seed with horizon=0"),
        ({"horizon": 0, "level": 0.9}, "level with horizon=0"),
    ],
    "fmols": [
        ({"bandwidth": 4.0, "bandwidth_rule": "andrews"}, "bandwidth_rule with an explicit bandwidth"),
        ({"trend": "c", "diff": True}, "diff with a trend-free x_trend"),
    ],
    "ccr": [
        ({"bandwidth": 4.0, "bandwidth_rule": "andrews"}, "bandwidth_rule with an explicit bandwidth"),
        ({"trend": "c", "diff": True}, "diff with a trend-free x_trend"),
    ],
    "dols": [
        ({"lags": 1, "leads": 1, "ic": "aic"}, "ic with lags and leads fixed"),
        ({"lags": 1, "max_lag": 4}, "max_lag with lags fixed"),
        ({"leads": 1, "max_lead": 4}, "max_lead with leads fixed"),
        ({"lags": 1, "leads": 1, "common": True}, "common with lags and leads fixed"),
        ({"bandwidth": 4.0, "bandwidth_rule": "andrews"}, "bandwidth_rule with an explicit bandwidth"),
    ],
    "panel_fe": [({"bandwidth": 4.0}, "bandwidth under se_type=cluster")],
    "panel_lp": [
        ({"mask": _ragged(6, 200), "jackknife": True}, "jackknife on an unbalanced panel"),
        ({"mask": _ragged(6, 200), "bias_correction": "dj"}, "bias_correction=dj on an unbalanced panel"),
        ({"mask": _ragged(6, 200), "bias_correction": "spj"}, "bias_correction=spj on an unbalanced panel"),
    ],
    "lp_did": [({"mask": _M}, "an unbalanced mask")],
}
# documented no-ops (the prose says the argument does not change the output):
# recorded, compared, not counted as findings when the promise holds
DOCUMENTED_NOOP = {}
# mask identities the docstrings promise: an all-ones mask == no mask, bitwise
MASK_IDENTITY = ("panel_fe", "panel_distributed_lag", "panel_lp", "lp_did")



# (d2) the value lists the docstrings state, transcribed by hand from the
# runtime __doc__ (not regex-derived) and passed one at a time on the
# canonical input. `None` entries are the documented "component absent" value.
LISTED_VALUES = {
    "unobserved_components": {
        "level": ["irregular", "ntrend", "fixed intercept", "deterministic constant", "dconstant",
                  "local level", "llevel", "random walk", "rwalk", "fixed slope",
                  "deterministic trend", "dtrend", "local linear deterministic trend", "lldtrend",
                  "random walk with drift", "rwdrift", "local linear trend", "lltrend",
                  "smooth trend", "strend", "random trend", "rtrend"],
    },
    "ets_fit": {
        "error": ["add", "mul"],
        "trend": [None, "add", "mul"],
        "initialization": ["estimated", "heuristic"],
        "optimizer": ["auto", "nelder_mead", "bfgs", "lbfgs"],
    },
    "auto_ets": {
        "ic": ["aicc", "aic", "bic"],
        "initialization": ["estimated", "heuristic"],
        "optimizer": ["auto", "nelder_mead", "bfgs", "lbfgs"],
    },
    "var_conditional_forecast": {"trend": ["c", "n"]},
    "var_diagnostics": {"trend": ["c", "n"]},
    "var_select_order": {"trend": ["c", "n"]},
    "spa_test": {"bootstrap": ["stationary", "circular", "moving_block"]},
    "stepm_test": {"bootstrap": ["stationary", "circular", "moving_block"]},
    "model_confidence_set": {
        "bootstrap": ["stationary", "circular", "moving_block"],
        "method": ["R", "max"],
    },
    "fmols": {
        "trend": ["n", "c", "ct", "ctt"],
        "kernel": ["bartlett", "parzen", "quadratic-spectral"],
        "bandwidth_rule": ["newey-west", "andrews"],
    },
    "ccr": {
        "trend": ["n", "c", "ct", "ctt"],
        "kernel": ["bartlett", "parzen", "quadratic-spectral"],
        "bandwidth_rule": ["newey-west", "andrews"],
    },
    "dols": {
        "trend": ["n", "c", "ct", "ctt"],
        "kernel": ["bartlett", "parzen", "quadratic-spectral"],
        "cov_type": ["unadjusted", "robust"],
    },
    "panel_fe": {"se_type": ["nonrobust", "cluster", "driscoll_kraay"]},
    "panel_distributed_lag": {"se_type": ["nonrobust", "cluster", "driscoll_kraay"]},
    "panel_lp": {
        "se_type": ["nonrobust", "cluster", "driscoll_kraay"],
        "bias_correction": ["none", "dj", "spj"],
        "band": [None, "sidak", "bonferroni"],
    },
}
# values that need a companion keyword to be admissible
VALUE_COMPANION = {
    ("dols", "ic"): {"lags": None, "leads": None},
    ("fmols", "bandwidth_rule"): {"bandwidth": None},
    ("ccr", "bandwidth_rule"): {"bandwidth": None},
    ("ets_fit", "optimizer"): {"initialization": "estimated"},
    ("auto_ets", "optimizer"): {},
    ("unobserved_components", "level"): {"seasonal": 4},
}


def main():
    fh = open(os.path.join(OUT, "sweep_f.txt"), "w")
    stubs = stub_params()
    report = {"a": {}, "b": {}, "c": {}, "d": {}, "e": {}}
    n_cand = 0
    # (a) every callable: names / order / has-default; plus the six: default values
    n_ok = 0
    real_names = [n for n in NAMES if "@" not in n]
    for name in real_names:
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
    log(fh, f"(a) signature vs stub: {n_ok}/{len(real_names)} agree on names, order and has-default")
    ell_all = [(n, p[0]) for n in real_names for p in runtime_params(n) if p[1] and p[2] is Ellipsis]
    log(fh, f"(a) library-wide Ellipsis defaults: {len(ell_all)} {ell_all}")
    if ell_all:
        n_cand += len(ell_all)
    for name in SCOPE:
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
    for name in SCOPE:
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
    for name in SCOPE:
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
    for name in SCOPE:
        fn = getattr(tsecon, name)
        pnames = {p[0] for p in runtime_params(name)}
        probes = {}
        flat = re.sub(r"\s+", " ", fn.__doc__ or "")
        str_params = {p[0] for p in runtime_params(name) if isinstance(p[2], str)}
        listed = {}
        for pname in str_params:
            # the window runs to the NEXT backticked parameter name of this
            # function (round 13's window stopped only at another string-valued
            # parameter, so `optimizer`'s values bled into `initialization`'s)
            others = [q for q in pnames if q != pname]
            stop = r"`(?:" + "|".join(re.escape(q) for q in sorted(others, key=len, reverse=True)) + r")`"
            for m in re.finditer(rf"`{pname}`", flat):
                window = re.split(stop, flat[m.end(): m.end() + 320])[0]
                listed.setdefault(pname, set()).update(
                    v for v in re.findall(r'"([a-z][a-z_0-9 \-]*)"', window) if v.strip()
                )
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
    # (d2) the hand-transcribed value lists
    n_val = n_val_ok = 0
    for name, table in LISTED_VALUES.items():
        fn = getattr(tsecon, name)
        for pname, vals in table.items():
            companion = VALUE_COMPANION.get((name, pname), {})
            for v in vals:
                args, kwargs = build(name, T=200, seed=0)
                kwargs.update(companion)
                kwargs[pname] = v
                if name in ("ets_fit", "auto_ets") and pname == "trend" and v is None:
                    kwargs.pop("damped", None)
                n_val += 1
                try:
                    fn(*args, **kwargs)
                    n_val_ok += 1
                    report["d"].setdefault(name + ":table", {})[f"{pname}={v}"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    report["d"].setdefault(name + ":table", {})[f"{pname}={v}"] = f"{type(exc).__name__}: {str(exc)[:140]}"
                    log(fh, f"[{name}] (d2) DOCUMENTED VALUE REFUSED {pname}={v!r}: {str(exc)[:140]}")
                    n_cand += 1
    log(fh, f"(d2) hand-transcribed value lists: {n_val_ok}/{n_val} accepted")
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
            except Exception as exc:  # noqa: BLE001
                report["e"][f"{name}:{label}"] = f"{type(exc).__name__}: {str(exc)[:140]}"
                if not isinstance(exc, ValueError):
                    log(fh, f"[{name}] (e) inert `{label}` raised {type(exc).__name__} (not ValueError): {str(exc)[:120]}")
    # (f) the documented mask identity: an all-ones mask is bit-identical to no mask
    for name in MASK_IDENTITY:
        fn = getattr(tsecon, name)
        args, kwargs = build(name, T=200, seed=0)
        base = fn(*args, **kwargs)
        n, t = np.asarray(args[0]).shape[-2:]
        ones = fn(*args, **{**kwargs, "mask": np.ones((n, t))})
        worst, where = 0.0, ""
        for k, v in base.items():
            try:
                a, b = np.asarray(v, dtype=float), np.asarray(ones[k], dtype=float)
            except (TypeError, ValueError):
                if v != ones[k]:
                    worst, where = float("inf"), k
                continue
            if a.shape != b.shape:
                worst, where = float("inf"), f"{k} shape {a.shape} vs {b.shape}"
                continue
            if a.size:
                d = float(np.nanmax(np.abs(a - b))) if a.dtype.kind == "f" else float((a != b).sum())
                if d > worst:
                    worst, where = d, k
        report["e"][f"{name}:mask=ones"] = f"max|d|={worst:.3g} at {where}"
        verdict = "BITWISE IDENTICAL" if worst == 0.0 else "DIFFERS"
        log(fh, f"[{name}] (f) mask of all ones vs no mask: {verdict} (max|d|={worst:.3g} at {where or '-'})")
        if worst != 0.0:
            n_cand += 1
    log(fh, f"\ncandidates raised: {n_cand}")
    json.dump(report, open(os.path.join(OUT, "sweep_f.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
