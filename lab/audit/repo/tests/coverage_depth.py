"""Coverage depth of the public surface (repo audit, tests sweep, item 6).

Extends test_exercise_gap.py's notion (every public callable is exercised
through `tsecon.<name>(`) with depth: for every public callable, how many
distinct test functions call it, and at what strength the strongest of those
tests asserts:

  reference   a numeric assertion in a test that also reads a value loaded
              from fixtures/*.json, or calls a reference library
              (statsmodels / scipy.stats / arch / sklearn / linearmodels /
              mapie) in the same function, or compares against a value the
              test computes itself from NumPy (closed form)
  numeric     a numeric assertion, but nothing in the function reads a
              fixture or a reference library (a property / sign / bound)
  structural  keys, shapes, finiteness, types only
  raises      only pytest.raises
  none        exercised, but no assertion at all

"reference" vs "numeric" is a static heuristic: a test comparing against a
hand-typed literal (e.g. `== pytest.approx(0.6348)`) is classed numeric even
when the literal came from a paper. The report is a screening list; every
row it flags was read before being reported.

Run:  .venv-wt/bin/python lab/audit/repo/tests/coverage_depth.py
Out:  lab/audit/repo/tests/out/coverage_depth.json
"""
from __future__ import annotations

import ast
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assertion_scan import ModuleScan, TESTS, REPO, _dotted, _name_of  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
REF_PREFIXES = ("sm.", "statsmodels", "scipy.stats", "stats.", "arch.", "arch_model", "sklearn", "linearmodels", "mapie", "PhillipsPerron", "SARIMAX", "ARIMA", "VAR(", "coint_johansen", "adfuller", "kpss")

RANK = {"none": 0, "raises": 1, "structural": 2, "numeric": 3, "reference": 4}


def module_fixture_names(tree: ast.Module) -> set[str]:
    """Module-level names bound to something loaded from JSON / a fixture file."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            src = ast.dump(node.value)
            if any(k in src for k in ("json", "load", "fixture", "FIXTURE", "Fixture", "read_text", "csv")):
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("test"):
            src = ast.dump(node)
            if "json" in src or "fixture" in src.lower() or "csv" in src:
                names.add(node.name)
    return names


def ref_alias_names(tree: ast.Module) -> set[str]:
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in {"statsmodels", "scipy", "arch", "sklearn", "linearmodels", "mapie", "pandas"}:
                    out.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in {"statsmodels", "scipy", "arch", "sklearn", "linearmodels", "mapie"}:
            for a in node.names:
                out.add(a.asname or a.name)
    return out


def main():
    import tsecon  # noqa: F401  (installed extension)

    public = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
    per_fn = defaultdict(list)  # name -> list of (file, test, strength)

    for f in sorted(os.listdir(TESTS)):
        if not (f.startswith("test_") and f.endswith(".py")):
            continue
        m = ModuleScan(os.path.join(TESTS, f))
        fixnames = module_fixture_names(m.tree)
        refaliases = ref_alias_names(m.tree)
        # index test functions by (cls, name) for body lookup
        bodies = {}
        for node in m.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                bodies[(None, node.name)] = node
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test"):
                        bodies[(node.name, sub.name)] = sub
        # helper functions that call tsecon.<x> on behalf of tests
        helper_calls = {}
        for hname, hfn in m.helpers.items():
            helper_calls[hname] = {_dotted(n.func).split(".")[-1] for n in ast.walk(hfn) if isinstance(n, ast.Call) and _dotted(n.func).startswith("tsecon.")}
        for t in m.tests:
            fn = bodies[(t["cls"], t["name"])]
            called = set(c.split(".")[-1] for c in t["calls"])
            # calls routed through a same-module helper (e.g. `_fit()` wrapping tsecon.garch_fit)
            for n in ast.walk(fn):
                if isinstance(n, ast.Call):
                    hn = _name_of(n.func)
                    if hn in helper_calls:
                        called |= helper_calls[hn]
            # also a module-level constant computed once from tsecon (RES = tsecon.x(...)) and read by the test
            names_used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            for node in m.tree.body:
                if isinstance(node, ast.Assign):
                    tgt = {n.id for tt in node.targets for n in ast.walk(tt) if isinstance(n, ast.Name)}
                    if tgt & names_used:
                        for n in ast.walk(node.value):
                            if isinstance(n, ast.Call) and _dotted(n.func).startswith("tsecon."):
                                called.add(_dotted(n.func).split(".")[-1])
            called &= set(public)
            if not called:
                continue
            if t["kind"] == "none":
                strength = "none"
            elif t["kind"] == "raises-only":
                strength = "raises"
            elif t["kind"] == "weak-only":
                strength = "structural"
            else:
                uses_fixture = bool(names_used & fixnames)
                uses_ref = bool(names_used & refaliases) or any(
                    _dotted(n.func).startswith(REF_PREFIXES) for n in ast.walk(fn) if isinstance(n, ast.Call)
                ) or any(
                    # `ref = pytest.importorskip("arch.unitroot")` bound inside the test
                    _dotted(n.func) == "pytest.importorskip" and n.args and isinstance(n.args[0], ast.Constant)
                    and str(n.args[0].value).split(".")[0] in {"statsmodels", "scipy", "arch", "sklearn", "linearmodels", "mapie"}
                    for n in ast.walk(fn) if isinstance(n, ast.Call)
                )
                # closed form: the test computes an expectation with numpy and compares
                has_np_math = any(
                    isinstance(n, ast.Call) and _dotted(n.func).startswith("np.") and _dotted(n.func).split(".")[-1] not in {"asarray", "array", "isfinite", "all", "any", "testing", "column_stack", "cumsum", "random", "default_rng", "zeros", "ones", "arange", "linspace", "empty", "isnan"}
                    for n in ast.walk(fn)
                )
                strength = "reference" if (uses_fixture or uses_ref or has_np_math) else "numeric"
            for c in called:
                per_fn[c].append((f, (t["cls"] + "::" if t["cls"] else "") + t["name"], strength))

    rows = []
    for name in public:
        tests = per_fn.get(name, [])
        best = max((RANK[s] for _, _, s in tests), default=-1)
        best_name = {v: k for k, v in RANK.items()}.get(best, "UNEXERCISED")
        counts = defaultdict(int)
        for _, _, s in tests:
            counts[s] += 1
        rows.append({"name": name, "n_tests": len(tests), "best": best_name, "counts": dict(counts), "tests": tests})

    json.dump(rows, open(os.path.join(OUT, "coverage_depth.json"), "w"), indent=1)
    print(f"public callables: {len(public)}")
    hist = defaultdict(int)
    for r in rows:
        hist[r["best"]] += 1
    print("strongest assertion per callable:", dict(hist))
    print()
    print("== callables with only smoke coverage (best in none/raises/structural) ==")
    for r in rows:
        if RANK.get(r["best"], -1) <= 2:
            print(f"  {r['name']:28s} n_tests={r['n_tests']:3d} best={r['best']:10s} " + "; ".join(f"{f}::{t}" for f, t, _ in r["tests"][:4]))
    print()
    print("== callables with numeric-but-no-reference coverage (property / sign / bound only) ==")
    for r in rows:
        if r["best"] == "numeric":
            print(f"  {r['name']:28s} n_tests={r['n_tests']:3d} " + "; ".join(f"{f}::{t}" for f, t, _ in r["tests"][:3]))
    print()
    print("== thinnest coverage (n_tests <= 2), any strength ==")
    for r in sorted(rows, key=lambda r: r["n_tests"]):
        if r["n_tests"] <= 2:
            print(f"  {r['name']:28s} n_tests={r['n_tests']} best={r['best']}")


if __name__ == "__main__":
    sys.exit(main())
