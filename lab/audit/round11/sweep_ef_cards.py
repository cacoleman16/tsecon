"""Model-card cross-checks shared by sweeps E and F.

(F-cards) every `| Call | Argument | Default |` / `| Argument | Default |` table
row in docs/reference/model-cards is resolved to a (function, argument,
default) triple and compared with inspect.signature of the compiled function:
an argument the function does not accept, or a default that differs, is a
candidate.

(E-cards) every "**`fn`** -> {"k1", "k2", ...}" output list in the cards is
compared with the keys the canonical call actually returns: a listed key that
is never returned is a phantom candidate; a returned key absent from a
complete list (no "...") is an undocumented-in-card candidate.

Run:  .venv-wt/bin/python lab/audit/round11/sweep_ef_cards.py
Out:  lab/audit/round11/out/sweep_ef_cards.log, sweep_ef_cards.json
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
from common import CARDS, HERE, log  # noqa: E402
from registry import NAMES, build  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
PUBLIC = set(NAMES)


def sig_of(name):
    fn = getattr(tsecon._core, name, None) or getattr(tsecon, name)
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def parse_default(cell):
    """Turn a card default cell into a Python value, or None if not literal."""
    cell = cell.strip().strip("`").strip()
    if cell in ("", "—", "-", "–", "required", "(required)"):
        return ("required", None)
    if cell.startswith("`") and cell.endswith("`"):
        cell = cell[1:-1]
    # normalise "1e-8" / "None" / '"c"' / "0.05"
    try:
        return ("value", ast.literal_eval(cell))
    except Exception:  # noqa: BLE001
        return ("text", cell)


def same(a, b):
    if a is b:
        return True
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a))
    if isinstance(b, float) and isinstance(a, (int, float)):
        return abs(a - b) <= 1e-12 * max(1.0, abs(b))
    return a == b


def main():
    fh = open(os.path.join(OUT, "sweep_ef_cards.log"), "w")
    report = {"defaults": [], "keys": []}
    results = {}
    for name in NAMES:
        try:
            a, k = build(name, T=200, seed=0)
            results[name] = getattr(tsecon, name)(*a, **k)
        except Exception as exc:  # noqa: BLE001
            results[name] = exc
    n_rows = n_keys = 0
    for fn in sorted(os.listdir(CARDS)):
        if not fn.endswith(".md"):
            continue
        text = open(os.path.join(CARDS, fn), encoding="utf-8").read()
        lines = text.splitlines()
        current_fn = None       # function from the last Call cell / heading
        header = None
        for i, line in enumerate(lines):
            if line.startswith("#"):
                names = [t for t in re.findall(r"`([a-z_][a-z_0-9]*)`", line) if t in PUBLIC]
                if names:
                    current_fn = names[0]
                header = None
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if header is None:
                    header = [c.lower() for c in cells]
                    continue
                if set(cells) <= {"", "-", "---", "----", "-----", "------", "-------", "--------", "---------", "----------"} or all(re.fullmatch(r"-+", c) for c in cells if c):
                    continue
                if "argument" in header and "default" in header:
                    col = {h: j for j, h in enumerate(header)}
                    if "call" in col and cells[col["call"]]:
                        cand = re.findall(r"`([a-z_][a-z_0-9]*)`", cells[col["call"]])
                        cand = [c for c in cand if c in PUBLIC]
                        if cand:
                            current_fn = cand[0]
                    arg = cells[col["argument"]] if col["argument"] < len(cells) else ""
                    dflt = cells[col["default"]] if col["default"] < len(cells) else ""
                    args = re.findall(r"`([A-Za-z_*][A-Za-z_0-9*]*)`", arg) or [arg.strip("`")]
                    if current_fn is None:
                        continue
                    sig = sig_of(current_fn)
                    if sig is None:
                        continue
                    for a in args:
                        n_rows += 1
                        if "*" in a:
                            continue
                        if a not in sig.parameters:
                            msg = f"{fn}:{i+1} `{current_fn}` table names argument `{a}` the function does not accept"
                            log(fh, "[CARD-ARG] " + msg)
                            report["defaults"].append({"card": fn, "line": i + 1, "fn": current_fn, "arg": a, "issue": "unknown argument"})
                            continue
                        kind, val = parse_default(dflt)
                        p = sig.parameters[a]
                        if kind == "value":
                            if p.default is p.empty:
                                msg = f"{fn}:{i+1} `{current_fn}.{a}` card default {val!r} but the parameter is REQUIRED"
                                log(fh, "[CARD-DEFAULT] " + msg)
                                report["defaults"].append({"card": fn, "line": i + 1, "fn": current_fn, "arg": a, "card": val, "runtime": "required"})
                            elif not same(p.default, val):
                                msg = f"{fn}:{i+1} `{current_fn}.{a}` card default {val!r} != runtime {p.default!r}"
                                log(fh, "[CARD-DEFAULT] " + msg)
                                report["defaults"].append({"card": fn, "line": i + 1, "fn": current_fn, "arg": a, "card_default": val, "runtime": p.default})
                        elif kind == "required" and p.default is not p.empty:
                            msg = f"{fn}:{i+1} `{current_fn}.{a}` card says required, runtime default {p.default!r}"
                            log(fh, "[CARD-DEFAULT] " + msg)
                            report["defaults"].append({"card": fn, "line": i + 1, "fn": current_fn, "arg": a, "card_default": "required", "runtime": p.default})
                continue
            header = None
        # (E-cards) output key lists
        for m in re.finditer(r"\*\*`([a-z_][a-z_0-9]*)`\*\*\s*(?:→|->)\s*`?\{([^}]*)\}`?", text):
            f, body = m.group(1), m.group(2)
            if f not in PUBLIC:
                continue
            listed = re.findall(r"\"([A-Za-z_][A-Za-z_0-9]*)\"", body)
            complete = "..." not in body and "…" not in body
            n_keys += 1
            res = results.get(f)
            if not isinstance(res, dict):
                continue
            keys = set(res)
            phantom = [k for k in listed if k not in keys]
            missing = sorted(keys - set(listed)) if complete else []
            if phantom:
                log(fh, f"[CARD-KEYS] {fn}: `{f}` lists keys never returned on the canonical call: {phantom}")
                report["keys"].append({"card": fn, "fn": f, "phantom": phantom})
            if missing:
                log(fh, f"[CARD-KEYS] {fn}: `{f}` complete list omits returned keys: {missing}")
                report["keys"].append({"card": fn, "fn": f, "omitted": missing})
    log(fh, f"checked {n_rows} card default rows and {n_keys} card key lists")
    json.dump(report, open(os.path.join(OUT, "sweep_ef_cards.json"), "w"), indent=1, default=str)
    fh.close()


if __name__ == "__main__":
    main()
