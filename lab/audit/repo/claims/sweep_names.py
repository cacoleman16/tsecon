"""Names sweep: every backticked identifier and every ``tsecon.<name>`` /
``<name>(...)`` mention in the audited doc set, classified against the
installed wheel.

Classes
  A  public callable that exists
  B  keyword argument that exists on a function the paragraph is about
  C  returned key that exists on a function the paragraph is about
  R  results-layer name (class, method) that exists in tsecon.results
  X  Rust-side symbol (crate, ``pub fn``/struct/enum) mentioned as such
  W  exists somewhere in the library, but not on the paragraph's function(s)
  P  phantom candidate — matches nothing above (hand review follows)

Call forms ``name(kw=...)`` are checked kwarg-by-kwarg against ``name``'s
signature; ``obj["key"]`` forms are checked against the paragraph's functions'
returned keys. Migration tables are read column-aware: the other library's
column is not a tsecon claim.

Run:  .venv-wt/bin/python lab/audit/repo/claims/sweep_names.py
Out:  out/sweep_names.json, out/sweep_names.log, out/sweep_names_review.md
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import DOC_FILES, OUT, REPO, load_keys, log, paragraphs, public_callables, split_blocks  # noqa: E402

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BACKTICK = re.compile(r"`([^`\n]+)`")
TSECON_DOT = re.compile(r"\btsecon\.(?:results\.)?([A-Za-z_][A-Za-z0-9_]*)")
CALL = re.compile(r"^(?:tsecon\.)?(?:results\.)?([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*(?:\[.*)?$", re.S)
KWARG = re.compile(r"(?<![A-Za-z0-9_.\"'])([a-z_][a-z0-9_]*)\s*=(?!=)")
SUBSCRIPT = re.compile(r"\[\s*[\"']([A-Za-z0-9_%]+)[\"']\s*\]")
ATTR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)(\(.*\))?$", re.S)
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")

OTHERLIB = re.compile(
    r"statsmodels|scipy|numpy|pandas|sklearn|scikit|\barch\b|linearmodels|\bR'?s\b|\bR\b|Stata|"
    r"::|mapie|cvxpy|skglm|reservoirpy|fixest|tsDyn|urca|pmdarima|glmnet|mboost|hdm|pdslasso|"
    r"matplotlib|ArviZ|arviz|Dynare|MATLAB|Matlab|forecast::|vars::|svars::|lpirfs|BVAR\b|midasr|KFAS|"
    r"plm|panelvar|rmgarch|rugarch|tseries|strucchange|quantreg|np::|boot\b|dlm\b",
)

STOP = set(
    """
    True False None int float str list dict tuple set bool print len range sum abs min max import from def
    return lambda for if else in not and or is np pd plt json os sys self cls args kwargs numpy pandas scipy
    tsecon results python rust cargo pytest pip maturin mkdocs venv main test tests fixtures docs crates
    y x z u v w t T n N k K p q d h H i j m r s b c e f g a l M S L P Q D B G A I F R V W X Y Z Ω Σ
    fit res out irf data df rng ret rets panel yields y0 y1 y2 x0 x1 x2 xs ys spec model shock series
    lower upper level names name key keys value values idx index row col cols rows shape size mean std var
    array arrays ndarray DataFrame Series float64 float32 int64 int32 uint64 NaN nan inf Inf pi
    ValueError RuntimeError KeyError TypeError AttributeError Exception NotImplementedError PanicException
    ConvergenceWarning FileNotFoundError ImportError
    json pickle repr help dir callable isinstance getattr hasattr setattr type object property staticmethod
    git github pypi ci CI abi3 py39 cdylib pyo3 PyO3 faer rayon rustfft rustup cargo clippy rustfmt
    """.split()
)


def rust_symbols():
    """Every identifier declared ``fn``/``struct``/``enum``/``trait``/``mod``/``type``
    in the Rust workspace, plus crate names. Cached per run."""
    syms = set()
    for path in glob.glob(os.path.join(REPO, "crates", "*", "src", "**", "*.rs"), recursive=True) + glob.glob(
        os.path.join(REPO, "bindings", "python", "src", "**", "*.rs"), recursive=True
    ):
        txt = open(path, encoding="utf-8", errors="replace").read()
        syms.update(re.findall(r"\b(?:fn|struct|enum|trait|mod|type|const|static)\s+([A-Za-z_][A-Za-z0-9_]*)", txt))
        syms.update(re.findall(r"^\s+([A-Z][A-Za-z0-9_]*)\s*[\{\(,]", txt, re.M))  # enum variants
    for d in glob.glob(os.path.join(REPO, "crates", "*")):
        syms.add(os.path.basename(d))
        syms.add(os.path.basename(d).replace("-", "_"))
    syms.add("tsecon-python")
    syms.add("_core")
    return syms


def results_names():
    import tsecon.results as R

    names = set(n for n in dir(R) if not n.startswith("_"))
    for n in list(names):
        obj = getattr(R, n)
        if isinstance(obj, type):
            names.update(a for a in dir(obj) if not a.startswith("_"))
    names.update(["summary", "plot", "to_dict", "fit", "irf", "response", "conf_int", "peak", "plot_irf",
                  "persistence", "plot_volatility", "forecast_frame", "plot_forecast", "significant",
                  "impulse_response", "plot_diagnostics", "coefficient_frame", "to_pandas", "params_named"])
    return names


def context_functions(text, public):
    found = set()
    for tok in BACKTICK.findall(text):
        tok = tok.strip()
        m = CALL.match(tok)
        base = m.group(1) if m else tok.replace("tsecon.", "").replace("results.", "")
        if base in public:
            found.add(base)
    found.update(m for m in TSECON_DOT.findall(text) if m in public)
    return found


def cells_for(relpath, text):
    """Column-aware table handling for the migration pages."""
    if not text.lstrip().startswith("|"):
        return text
    cells = [c for c in text.strip().strip("|").split("|")]
    if relpath.startswith("docs/migration/rosetta"):
        return "|".join(cells[1:2])
    if relpath.startswith("docs/migration/"):
        return "|".join(cells[1:])
    return text


def main():
    fh = open(os.path.join(OUT, "sweep_names.log"), "w")
    public = set(public_callables())
    keys = load_keys()
    params = {n: set(r["params"]) for n, r in keys.items()}
    kset = {n: set(r["top"]) | set(r["nested"]) for n, r in keys.items()}
    all_params = set().union(*params.values())
    all_keys = set().union(*kset.values())
    rsyms = rust_symbols()
    rnames = results_names()

    records = []
    for rel in DOC_FILES:
        text = open(os.path.join(REPO, rel), encoding="utf-8").read()
        heading_fns = set()
        prev_ctx = []
        for kind, lang, start, body in split_blocks(text):
            if kind == "code":
                # code blocks: only the explicit tsecon.<name>( calls and their kwargs
                for m in re.finditer(r"tsecon\.(?:results\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", body):
                    fn, argtxt = m.group(1), m.group(2)
                    line = start + body[: m.start()].count("\n") + 1
                    if fn not in public and fn not in rnames:
                        records.append(dict(file=rel, line=line, token=fn, cls="P", where="code-call", ctx=[], note="tsecon.<name>( in a code block"))
                        continue
                    if fn in public:
                        records.append(dict(file=rel, line=line, token=fn, cls="A", where="code-call", ctx=[fn], note=""))
                        for kw in KWARG.findall(argtxt):
                            cls = "B" if kw in params.get(fn, set()) else ("W" if kw in all_params else "P")
                            records.append(dict(file=rel, line=line, token=kw, cls=cls, where="code-kwarg", ctx=[fn], note=f"kwarg of {fn}(...)"))
                continue
            for pstart, para in paragraphs(body, start):
                hm = HEADING.match(para)
                if hm:
                    heading_fns = context_functions(hm.group(1), public)
                ptxt = cells_for(rel, para)
                ctx = context_functions(ptxt, public)
                if not ctx:
                    ctx = set(heading_fns)
                if not ctx:
                    for back in prev_ctx[-2:][::-1]:
                        if back:
                            ctx = set(back)
                            break
                otherlib = bool(OTHERLIB.search(ptxt))
                # bare tsecon.<name> mentions outside backticks
                for m in TSECON_DOT.finditer(ptxt):
                    nm = m.group(1)
                    cls = "A" if nm in public else ("R" if nm in rnames else "P")
                    records.append(dict(file=rel, line=pstart, token=nm, cls=cls, where="tsecon.dot", ctx=sorted(ctx), note="", otherlib=otherlib))
                for raw in BACKTICK.findall(ptxt):
                    tok = raw.strip()
                    if not tok or "/" in tok or tok.endswith((".md", ".py", ".rs", ".json", ".csv", ".toml", ".yml", ".txt", ".R")):
                        continue
                    base_ctx = sorted(ctx)
                    # call form
                    m = CALL.match(tok)
                    if m:
                        fn, argtxt = m.group(1), m.group(2)
                        if fn in public:
                            records.append(dict(file=rel, line=pstart, token=fn, cls="A", where="call", ctx=base_ctx, note="", otherlib=otherlib))
                            for kw in KWARG.findall(argtxt):
                                if kw in STOP:
                                    continue
                                cls = "B" if kw in params.get(fn, set()) else ("W" if kw in all_params else "P")
                                records.append(dict(file=rel, line=pstart, token=kw, cls=cls, where="kwarg", ctx=[fn], note=f"kwarg of {fn}(...)", otherlib=otherlib))
                            for key in SUBSCRIPT.findall(tok):
                                cls = "C" if key in kset.get(fn, set()) else ("W" if key in all_keys else "P")
                                records.append(dict(file=rel, line=pstart, token=key, cls=cls, where="key", ctx=[fn], note=f"key of {fn}(...)", otherlib=otherlib))
                            continue
                        if fn in rnames:
                            records.append(dict(file=rel, line=pstart, token=fn, cls="R", where="call", ctx=base_ctx, note="", otherlib=otherlib))
                            continue
                        if fn in rsyms and not otherlib:
                            records.append(dict(file=rel, line=pstart, token=fn, cls="X", where="call", ctx=base_ctx, note="", otherlib=otherlib))
                            continue
                        if fn not in STOP:
                            records.append(dict(file=rel, line=pstart, token=fn, cls="P", where="call", ctx=base_ctx, note="call form", otherlib=otherlib))
                        continue
                    # subscript form fit["key"]["k2"]
                    keys_here = SUBSCRIPT.findall(tok)
                    if keys_here:
                        for key in keys_here:
                            hit = [f for f in ctx if key in kset.get(f, set())]
                            cls = "C" if hit else ("W" if key in all_keys else "P")
                            records.append(dict(file=rel, line=pstart, token=key, cls=cls, where="subscript", ctx=base_ctx, note="", otherlib=otherlib))
                        continue
                    # kw=value form
                    km = re.match(r"^([a-z_][a-z0-9_]*)\s*=\s*(.+)$", tok)
                    if km:
                        kw = km.group(1)
                        hit = [f for f in ctx if kw in params.get(f, set())]
                        cls = "B" if hit else ("W" if kw in all_params else "P")
                        records.append(dict(file=rel, line=pstart, token=kw, cls=cls, where="kw=", ctx=base_ctx, note="", otherlib=otherlib))
                        continue
                    # attribute form obj.attr / obj.method()
                    am = ATTR.match(tok)
                    if am:
                        head, attr = am.group(1), am.group(2)
                        if head == "tsecon" or head == "results":
                            nm = attr
                            cls = "A" if nm in public else ("R" if nm in rnames else "P")
                            records.append(dict(file=rel, line=pstart, token=nm, cls=cls, where="tsecon.dot", ctx=base_ctx, note="", otherlib=otherlib))
                        elif attr in rnames:
                            records.append(dict(file=rel, line=pstart, token=attr, cls="R", where="attr", ctx=base_ctx, note="", otherlib=otherlib))
                        elif head in ("np", "pd", "plt", "scipy", "sm", "stats", "signal", "linalg", "random", "tsa", "api"):
                            pass
                        elif not otherlib and attr not in STOP and head not in STOP:
                            records.append(dict(file=rel, line=pstart, token=tok, cls="P", where="attr", ctx=base_ctx, note="attribute form", otherlib=otherlib))
                        continue
                    tok2 = tok.replace("tsecon.", "").replace("results.", "")
                    if not IDENT.match(tok2):
                        continue
                    if tok2 in STOP or len(tok2) == 1:
                        continue
                    if tok2 in public:
                        cls = "A"
                    elif tok2 in rnames:
                        cls = "R"
                    elif any(tok2 in params.get(f, set()) for f in ctx):
                        cls = "B"
                    elif any(tok2 in kset.get(f, set()) for f in ctx):
                        cls = "C"
                    elif tok2 in rsyms:
                        cls = "X"
                    elif tok2 in all_params or tok2 in all_keys:
                        cls = "W"
                    else:
                        cls = "P"
                    records.append(dict(file=rel, line=pstart, token=tok2, cls=cls, where="backtick", ctx=base_ctx, note="", otherlib=otherlib))
                prev_ctx.append(ctx)

    json.dump(records, open(os.path.join(OUT, "sweep_names.json"), "w"), indent=0)
    from collections import Counter

    cnt = Counter(r["cls"] for r in records)
    log(fh, "records:", len(records), dict(sorted(cnt.items())))
    per_file = Counter((r["file"], r["cls"]) for r in records)
    # review sheet: P and W, grouped by file, de-duplicated by (file, token)
    seen = set()
    with open(os.path.join(OUT, "sweep_names_review.md"), "w") as rv:
        for cls in ("P", "W"):
            rv.write(f"\n## class {cls}\n\n| file:line | token | where | ctx | otherlib | note |\n|---|---|---|---|---|---|\n")
            for r in records:
                if r["cls"] != cls:
                    continue
                k = (r["file"], r["token"], r["where"])
                if k in seen:
                    continue
                seen.add(k)
                rv.write(f"| {r['file']}:{r['line']} | `{r['token']}` | {r['where']} | {','.join(r['ctx'][:4])} | {'y' if r.get('otherlib') else ''} | {r['note']} |\n")
    for f in DOC_FILES:
        row = {c: per_file.get((f, c), 0) for c in "ABCRXWP"}
        if row["P"] or row["W"]:
            log(fh, f"{f}: {row}")
    log(fh, "unique P tokens:", len({(r['file'], r['token']) for r in records if r['cls'] == 'P'}))
    log(fh, "unique W tokens:", len({(r['file'], r['token']) for r in records if r['cls'] == 'W'}))
    fh.close()


if __name__ == "__main__":
    main()
