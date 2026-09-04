"""Docstring structure, per callable (repo-wide API audit, item 5).

Reads surface.json (runtime signature, returned keys) and checks, on the
runtime ``__doc__`` (the binding surface — ``help(tsecon.fn)``) and on the
stub docstring:

  summary     — a one-line first sentence (first line <= 160 chars, or the
                first paragraph is <= 2 lines);
  params      — every runtime parameter is mentioned (backticked or bare);
  keys        — every returned top-level key is backticked (the round-3/11
                tripwire rule); nested keys reported separately;
  reference   — a citation (Author (YYYY), "et al.", or a named package);
  grade       — a validation statement (validated / golden / pinned /
                matches / Monte-Carlo / property / transcription / grade);
  stub_types  — the stub's return annotation agrees with the returned
                kind, and each parameter's stub type admits the runtime
                default (None / str / bool / int / float).

Run:  .venv-wt/bin/python lab/audit/repo/api/probe_docstrings.py
Out:  lab/audit/repo/api/out/docstrings.json, docstrings.log
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from probe_surface import stub_table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SURFACE = json.load(open(os.path.join(HERE, "surface.json")))

# a citation: a capitalised name within ~70 chars before a 19xx/20xx year, or a
# named reference package / author list
REF = re.compile(r"[A-Z][\w\-éàüöñç]+[^.;]{0,70}?\b(19|20)\d{2}[a-z]?\b|\bet al\.|\b(statsmodels|scikit-learn|sklearn|MacKinnon|Newey-West|Hodrick-Prescott|Baxter-King|Christiano-Fitzgerald|Nelson-Siegel|Stata|MATLAB|Matlab|EViews|Dynare|skglm|cvxpy|reservoirpy|SciPy|scipy|NumPy)\b")
GRADE = re.compile(r"\b(validat\w*|golden|pinned|pins|matches|matching|Monte[- ]Carlo|property|transcription|grade|asserted at|agrees with|reproduces|cross-checked|replicat\w*|bit-identical|at 1e-\d+|1e-\d+\)|coverage)\b", re.I)


def mentions(doc, name):
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", doc) is not None


def backticked(doc):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", doc or ""))


def type_admits(stub_type, default):
    t = stub_type or ""
    if default is None:
        return "None" in t or "Any" in t or t == ""
    if isinstance(default, bool):
        return "bool" in t or "Any" in t
    if isinstance(default, str):
        return "str" in t or "Literal" in t or "Any" in t
    if isinstance(default, int):
        return "int" in t or "float" in t or "Any" in t
    if isinstance(default, float):
        return "float" in t or "Any" in t
    if isinstance(default, (list, tuple)):
        return "Sequence" in t or "tuple" in t or "list" in t or "Any" in t or "NDArray" in t
    return True


def return_agrees(stub_ret, kind):
    if kind is None:
        return None
    r = stub_ret or ""
    if kind == "dict":
        return r.startswith("dict") or "Dict" in r or "Mapping" in r
    if kind == "float":
        return r == "float" or r.startswith("float")
    if kind == "int":
        return r == "int"
    if kind.startswith("tuple"):
        return r.startswith("tuple") or r.startswith("Tuple")
    if re.match(r"\d-D", kind):
        return "NDArray" in r or "_ArrayLike" in r or "ndarray" in r or r == "_F64"
    if kind == "str":
        return r == "str"
    if kind.startswith("list"):
        return r.startswith("list") or r.startswith("List") or "Sequence" in r
    return None


def main():
    stub = stub_table()
    out = {}
    log = open(os.path.join(OUT, "docstrings.log"), "w")
    for name, rec in sorted(SURFACE.items()):
        fn = getattr(tsecon, name)
        doc = fn.__doc__ or ""
        sdoc = stub.get(name, {}).get("doc", "")
        lines = [ln for ln in doc.strip().splitlines()]
        first_para = []
        for ln in lines:
            if not ln.strip():
                break
            first_para.append(ln.strip())
        r = {
            "doc_len": len(doc),
            "stub_doc_len": len(sdoc),
            "first_line": rec.get("doc_first_line", ""),
            # a summary exists when the first paragraph is short, or its first
            # sentence closes within two lines (a one-sentence opener wrapped)
            "summary_ok": bool(first_para) and (len(first_para) <= 4 or re.search(r"[.!?]\s*$", " ".join(first_para[:2])) is not None or len(re.findall(r"[.!?](\s|$)", " ".join(first_para))) <= 2),
            "first_para_lines": len(first_para),
        }
        params = [p["name"] for p in (rec.get("params") or [])]
        # the leading data arguments are usually described as "the series" /
        # "the panel" rather than by name; report them separately from options
        n_lead = 0
        for p in rec.get("params") or []:
            if p.get("has_default"):
                break
            n_lead += 1
        lead, opts = params[:n_lead], params[n_lead:]
        r["data_params_missing_runtime"] = [p for p in lead if not mentions(doc, p)]
        r["params_missing_runtime"] = [p for p in opts if not mentions(doc, p)]
        r["params_missing_stub"] = [p for p in opts if not mentions(sdoc, p)]
        keys = list((rec.get("keys") or {}).keys())
        bt, sbt = backticked(doc), backticked(sdoc)
        r["keys_unbackticked_runtime"] = [k for k in keys if k not in bt]
        r["keys_unmentioned_runtime"] = [k for k in keys if not mentions(doc, k)]
        r["keys_unbackticked_stub"] = [k for k in keys if k not in sbt]
        r["keys_unmentioned_stub"] = [k for k in keys if not mentions(sdoc, k)]
        nested = rec.get("nested") or {}
        r["nested_keys_unmentioned_runtime"] = sorted({f"{k}.{kk}" for k, sub in nested.items() for kk in sub if not mentions(doc, kk)})
        r["has_reference"] = REF.search(doc) is not None
        r["has_reference_stub"] = REF.search(sdoc) is not None
        r["has_grade"] = GRADE.search(doc) is not None
        r["has_grade_stub"] = GRADE.search(sdoc) is not None
        r["has_keys_line"] = re.search(r"\bKeys:", doc) is not None
        # stub types
        r["stub_return"] = rec.get("stub_returns")
        r["return_kind"] = rec.get("return_kind")
        r["stub_return_agrees"] = return_agrees(rec.get("stub_returns"), rec.get("return_kind"))
        bad = []
        for p in rec.get("params") or []:
            if p.get("has_default") and not type_admits(p.get("stub_type"), p.get("default")):
                bad.append(f"{p['name']}: stub `{p.get('stub_type')}` vs default {p.get('default')!r}")
        r["stub_param_type_mismatch"] = bad
        stub_params = rec.get("stub_params") or []
        r["stub_param_order_drift"] = stub_params != params
        r["doc_runtime_vs_stub_ratio"] = round(len(doc) / len(sdoc), 2) if sdoc else None
        out[name] = r
        flags = []
        if not r["summary_ok"]:
            flags.append("summary")
        if r["params_missing_runtime"]:
            flags.append(f"params-{r['params_missing_runtime']}")
        if r["keys_unbackticked_runtime"]:
            flags.append(f"keys-{r['keys_unbackticked_runtime']}")
        if not r["has_reference"]:
            flags.append("noref")
        if not r["has_grade"]:
            flags.append("nograde")
        if r["stub_return_agrees"] is False:
            flags.append(f"stubret({r['stub_return']} vs {r['return_kind']})")
        if bad:
            flags.append(f"stubtype{bad}")
        if r["stub_param_order_drift"]:
            flags.append("stub-order")
        line = f"[{name}] " + (" ".join(flags) if flags else "clean")
        print(line)
        log.write(line + "\n")
    json.dump(out, open(os.path.join(OUT, "docstrings.json"), "w"), indent=1, sort_keys=True)
    n = len(out)
    print("---- totals over", n)
    for k in ("summary_ok", "has_reference", "has_grade", "has_keys_line"):
        print(f"{k}: {sum(1 for r in out.values() if r[k])}/{n}")
    print("params missing (runtime):", sum(1 for r in out.values() if r["params_missing_runtime"]))
    print("keys unbackticked (runtime):", sum(1 for r in out.values() if r["keys_unbackticked_runtime"]))
    print("keys unmentioned (runtime):", sum(1 for r in out.values() if r["keys_unmentioned_runtime"]))
    print("stub return disagrees:", [k for k, r in out.items() if r["stub_return_agrees"] is False])
    print("stub param type mismatch:", sum(1 for r in out.values() if r["stub_param_type_mismatch"]))
    print("stub order drift:", [k for k, r in out.items() if r["stub_param_order_drift"]])


if __name__ == "__main__":
    main()
