"""Apply the documentation fixes of the repo-wide API audit (reproducible).

Reads surface.json (compiled signature + returned keys on the canonical call)
and out/docstrings.json (what each surface fails to name) and inserts, into
the ``///`` doc comment of the pyfunction in bindings/python/src/lib.rs AND
into the matching stub docstring in __init__.pyi:

  * ``Returned keys: `a`, `b`, ...`` — for every function with a returned
    key that the surface does not backtick (33 functions named none of
    their keys on either surface; 11 more mentioned them bare; the
    round-3/11 tripwire rule is that every returned key is backticked in
    ``__doc__``);
  * ``Further arguments, with defaults: `lags` (2), `trend` ("c"), ...`` —
    for every option parameter the surface never names, with the default
    read from the compiled signature (so the line is correct by
    construction; the nine parameters whose default renders as ``...`` are
    skipped);
  * the NaN-as-missing sentence on ``ar_loglik`` (the SSM crate documents
    "NaN in `y` means missing" and the probe confirmed the observation is
    skipped, but the binding said nothing).

Idempotent: a surface already carrying the marker text is left alone.

Run:  .venv-wt/bin/python lab/audit/repo/api/apply_doc_fixes.py
"""
from __future__ import annotations

import json
import os
import re
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
LIB = os.path.join(REPO, "bindings", "python", "src", "lib.rs")
PYI = os.path.join(REPO, "bindings", "python", "python", "tsecon", "__init__.pyi")
INSPECT = os.path.join(REPO, "bindings", "python", "python", "tsecon", "_inspect.py")
S = json.load(open(os.path.join(HERE, "surface.json")))
D = json.load(open(os.path.join(HERE, "out", "docstrings.json")))

SKIP = {"summarize"}  # summarize's "keys" are the probe's own input
PURE_PY = {"check_series": INSPECT}  # pure-Python entry points: edit the .py docstring
# Parameters whose compiled default renders as `...` (non-literal defaults)
# cannot be covered by the generated line; these are written by hand.
MANUAL = {
    "ivx_test": [
        "`cz` (-1.0) and `alpha` (0.95) tune the IVX instrument's persistence "
        "`rho_z = 1 + cz / n^alpha` (Kostakis-Magdalinos-Stamatogiannis 2015), "
        "exactly as in `predictive_regression`; `alpha` here is not a "
        "significance level (no level is passed; `pvalue` is returned)."
    ],
    "narrative_svar": [
        "`sign_restrictions` (default: none) is the `sign_restricted_svar` list "
        "of `(variable, shock, horizon, sign)` tuples with `sign` in "
        "{\"+\", \"-\"}."
    ],
    "historical_decomposition": [
        "`restrictions` (default: none; used under identification=\"sign\") is "
        "the `sign_restricted_svar` list of `(variable, shock, horizon, sign)` "
        "tuples with `sign` in {\"+\", \"-\"}."
    ],
}
AR_LOGLIK_NOTE = (
    "NaN entries in `y` are treated as missing observations: the Kalman "
    "filter skips their update and the log-likelihood sums the remaining "
    "innovations (with `coeffs=[0]` it equals the log-likelihood of the "
    "series with those entries deleted). Infinite entries are rejected."
)


def fmt_default(v):
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, float):
        return repr(v)
    return str(v)


def keys_line(name):
    keys = list((S[name].get("keys") or {}).keys())
    if not keys:
        return None
    return "Returned keys: " + ", ".join(f"`{k}`" for k in keys) + "."


def params_line(name, missing):
    by = {p["name"]: p for p in S[name].get("params") or []}
    parts = []
    for m in missing:
        p = by.get(m)
        if p is None or not p.get("has_default") or p.get("default") == "Ellipsis":
            continue
        parts.append(f"`{m}` ({fmt_default(p['default'])})")
    if not parts:
        return None
    return "Further arguments, with defaults: " + ", ".join(parts) + "."


# --------------------------------------------------------------------------- #
# lib.rs
# --------------------------------------------------------------------------- #
def insert_rs(lines, name, paragraphs):
    """Append `paragraphs` (list of str) to the /// block above `fn name`."""
    pat = re.compile(rf"^fn {re.escape(name)}(<'py>)?\(")
    idx = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if idx is None:
        return False, "fn not found"
    j = idx - 1
    while j >= 0 and not lines[j].startswith("///"):
        if lines[j].startswith("fn ") or lines[j].strip() == "}":
            return False, "no /// block"
        j -= 1
    if j < 0:
        return False, "no /// block"
    # the function's own contiguous /// block only (the previous function's
    # block would otherwise mask a missing line)
    b0 = j
    while b0 - 1 >= 0 and lines[b0 - 1].startswith("///"):
        b0 -= 1
    block = "\n".join(lines[b0:j + 1])
    new = []
    for para in paragraphs:
        if para.split(":")[0] in block or para[:40] in block:
            continue
        new.append("///")
        new.extend("/// " + w for w in textwrap.wrap(para, width=74))
    if not new:
        return False, "already present"
    lines[j + 1:j + 1] = new
    return True, f"inserted {len(new)} lines"


# --------------------------------------------------------------------------- #
# __init__.pyi
# --------------------------------------------------------------------------- #
def insert_pyi(lines, name, paragraphs):
    idx = next((i for i, ln in enumerate(lines) if ln.startswith(f"def {name}(")), None)
    if idx is None:
        return False, "def not found"
    k = idx
    while not lines[k].rstrip().endswith(":"):
        k += 1
    d0 = k + 1
    if not lines[d0].lstrip().startswith('"""'):
        return False, "no docstring"
    ind = " " * (len(lines[d0]) - len(lines[d0].lstrip()))
    body = lines[d0].strip()
    if body.endswith('"""') and len(body) > 6:
        # single-line docstring -> multi-line
        text = body[3:-3].strip()
        lines[d0:d0 + 1] = [ind + '"""' + text, ind + '"""']
        d1 = d0 + 1
    else:
        d1 = d0 + 1
        while '"""' not in lines[d1]:
            d1 += 1
        if lines[d1].strip() != '"""':
            # closing quotes share a line with text: split them
            text = lines[d1].split('"""')[0].rstrip()
            lines[d1:d1 + 1] = [text, ind + '"""']
            d1 += 1
    block = "\n".join(lines[d0:d1])
    new = []
    for para in paragraphs:
        if para.split(":")[0] in block or para[:40] in block:
            continue
        # the stub's own bare "Keys: a, b, c" line already names the keys; the
        # backtick rule binds the runtime docstring, not the stub
        if para.startswith("Returned keys:") and re.search(r"\bKeys:", block):
            continue
        new.append("")
        new.extend(ind + w for w in textwrap.wrap(para, width=76 - len(ind)))
    if not new:
        return False, "already present"
    lines[d1:d1] = new
    return True, f"inserted {len(new)} lines"


def main():
    rs = open(LIB, encoding="utf-8").read().split("\n")
    py = open(PYI, encoding="utf-8").read().split("\n")
    n_rs = n_py = 0
    for name in sorted(S):
        if name in SKIP:
            continue
        d = D[name]
        rs_paras, py_paras = [], []
        if d["keys_unbackticked_runtime"]:
            kl = keys_line(name)
            if kl:
                rs_paras.append(kl)
        if d["keys_unbackticked_stub"]:
            kl = keys_line(name)
            if kl:
                py_paras.append(kl)
        pl = params_line(name, d["params_missing_runtime"])
        if pl:
            rs_paras.append(pl)
        pl = params_line(name, d["params_missing_stub"])
        if pl:
            py_paras.append(pl)
        if name == "ar_loglik":
            rs_paras.append(AR_LOGLIK_NOTE)
            py_paras.append(AR_LOGLIK_NOTE)
        for para in MANUAL.get(name, []):
            rs_paras.append(para)
            py_paras.append(para)
        if rs_paras and name in PURE_PY:
            src = open(PURE_PY[name], encoding="utf-8").read().split("\n")
            ok, why = insert_pyi(src, name, rs_paras)
            print(f"[{os.path.basename(PURE_PY[name])}] {name}: {why}")
            if ok:
                open(PURE_PY[name], "w", encoding="utf-8").write("\n".join(src))
        elif rs_paras:
            ok, why = insert_rs(rs, name, rs_paras)
            print(f"[lib.rs] {name}: {why}")
            n_rs += ok
        if py_paras:
            ok, why = insert_pyi(py, name, py_paras)
            print(f"[pyi]    {name}: {why}")
            n_py += ok
    open(LIB, "w", encoding="utf-8").write("\n".join(rs))
    open(PYI, "w", encoding="utf-8").write("\n".join(py))
    print(f"edited {n_rs} functions in lib.rs, {n_py} in the stub")


if __name__ == "__main__":
    main()
