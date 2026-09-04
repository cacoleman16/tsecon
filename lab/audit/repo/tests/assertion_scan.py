"""Assertion audit of the Python binding suite (repo audit, tests sweep, item 3).

For every test function in bindings/python/tests/*.py this walks the AST and
classifies its assertions:

  none        no `assert`, no `pytest.raises`/`warns`, no `np.testing.assert_*`,
              no call to a same-module helper that asserts, no `pytest.fail`
              -> the test cannot fail except by an exception in the callee
  weak-only   every assertion is structural: `is not None`, `isinstance`,
              `.shape ==`, `len(...) ==`, `k in res`, `set(...) <=`,
              `np.isfinite(...).all()`, `type(...)`, `== True/False`
  numeric     at least one assertion compares a number (approx / allclose /
              isclose / a comparison with a float or int literal or a
              fixture value / `== other_result`)

It also reports, per file, every tolerance literal (`rtol=`, `atol=`, `abs=`,
`rel=`, and the right-hand side of `< 1e-x`-style comparisons) so the
looser-than-documented check in `tolerance_vs_matrix.py` can consume them, and
finds duplicated test bodies (identical body AST, different name or file).

Run:  .venv-wt/bin/python lab/audit/repo/tests/assertion_scan.py
Out:  lab/audit/repo/tests/out/assertion_scan.json (machine)
      stdout: human summary
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TESTS = os.path.join(REPO, "bindings", "python", "tests")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

ASSERT_CALL_NAMES = {
    "raises", "warns", "deprecated_call", "fail", "approx",
    "assert_allclose", "assert_array_almost_equal", "assert_array_equal",
    "assert_almost_equal", "assert_equal", "assert_array_less",
    "assert_", "assert_approx_equal", "assert_string_equal",
    "assert_frame_equal", "assert_series_equal", "assert_index_equal",
}
WEAK_FUNCS = {"isinstance", "len", "type", "set", "sorted", "list", "tuple", "dict", "hasattr", "callable"}
WEAK_ATTRS = {"shape", "ndim", "dtype", "size", "keys", "columns", "index", "names"}


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _has_numeric_literal(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return True
    return False


def _is_weak_assert(test: ast.expr) -> bool:
    """A structural assertion: shape / key / type / finiteness / not-None."""
    # `x is not None`, `x is None`
    if isinstance(test, ast.Compare):
        ops = test.ops
        if all(isinstance(op, (ast.Is, ast.IsNot)) for op in ops):
            return True
        if all(isinstance(op, (ast.In, ast.NotIn)) for op in ops):
            return True
        # `set(res) <= KEYS`, `KEYS <= set(res)`, `set(res) == {...}`
        sides = [test.left] + list(test.comparators)
        if any(isinstance(s, ast.Call) and _name_of(s.func) in {"set", "sorted", "list", "tuple", "type", "len"} for s in sides):
            # len(x) == 3 is structural; len(x) == len(y) too
            return True
        if any(isinstance(s, (ast.Set, ast.Dict)) for s in sides):
            return True
        if any(isinstance(s, ast.Attribute) and s.attr in WEAK_ATTRS for s in sides):
            return True
        # `x.shape[0] == 5`
        if any(isinstance(s, ast.Subscript) and isinstance(s.value, ast.Attribute) and s.value.attr in WEAK_ATTRS for s in sides):
            return True
        # `res["d"] == True` / `is True` compare against booleans
        if all(isinstance(s, ast.Constant) and isinstance(s.value, bool) for s in test.comparators):
            return True
        return False
    if isinstance(test, ast.Call):
        fn = _name_of(test.func)
        if fn in {"isinstance", "hasattr", "callable", "issubclass"}:
            return True
        # np.isfinite(x).all(), np.all(np.isfinite(x)), np.isfinite(x)
        d = _dotted(test.func)
        if fn == "all" and isinstance(test.func, ast.Attribute):
            inner = test.func.value
            if isinstance(inner, ast.Call) and _name_of(inner.func) in {"isfinite", "isreal"}:
                return True
        if fn in {"all", "any"} and test.args:
            a = test.args[0]
            if isinstance(a, ast.Call) and _name_of(a.func) in {"isfinite", "isreal", "isnan"}:
                return True
        if _name_of(test.func) in {"isfinite"}:
            return True
        return False
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _is_weak_assert(test.operand)
    if isinstance(test, ast.BoolOp):
        return all(_is_weak_assert(v) for v in test.values)
    if isinstance(test, ast.Name):
        # `assert ok` — unknown, treat as strong (it may encode a numeric check)
        return False
    return False


def module_fixture_names(tree: ast.Module) -> set[str]:
    """Module-level names bound to something loaded from JSON / a fixture file,
    plus module-level helper fns whose body reads one (a `_case(name)` accessor)."""
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
    # second pass: names assigned FROM a fixture name (CASE = FX["arma11"])
    changed = True
    while changed:
        changed = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                if used & names:
                    for t in node.targets:
                        for n in ast.walk(t):
                            if isinstance(n, ast.Name) and n.id not in names:
                                names.add(n.id)
                                changed = True
    return names


class ModuleScan:
    def __init__(self, path: str):
        self.path = path
        self.src = open(path, encoding="utf-8").read()
        self.tree = ast.parse(self.src, filename=path)
        self.helpers: dict[str, ast.FunctionDef] = {}
        self.helper_asserts: dict[str, bool] = {}
        self.tests: list[dict] = []
        self.fixture_names = module_fixture_names(self.tree)
        self._collect_helpers()
        self._collect_tests()

    # -- helpers -------------------------------------------------------------
    def _collect_helpers(self):
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("test"):
                self.helpers[node.name] = node
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith("test"):
                        self.helpers[sub.name] = sub
        # fixed point: a helper asserts if it holds an assert or calls one that does
        changed = True
        self.helper_asserts = {k: False for k in self.helpers}
        while changed:
            changed = False
            for name, fn in self.helpers.items():
                if self.helper_asserts[name]:
                    continue
                info = self._scan_body(fn, resolve_helpers=True)
                if info["n_assert"] > 0:
                    self.helper_asserts[name] = True
                    changed = True

    def _scan_body(self, fn: ast.AST, resolve_helpers: bool) -> dict:
        n_assert = 0
        n_weak = 0
        n_strong = 0
        n_raises = 0
        n_helper = 0
        tolerances = []
        # names bound inside this function from a fixture (case = FX["x"]; for case in FX["cases"])
        local_fix = set(self.fixture_names)
        changed = True
        while changed:
            changed = False
            for n in ast.walk(fn):
                if isinstance(n, (ast.Assign, ast.For, ast.comprehension)):
                    src_node = n.value if isinstance(n, ast.Assign) else n.iter
                    used = {x.id for x in ast.walk(src_node) if isinstance(x, ast.Name)}
                    if used & local_fix:
                        tgt = n.targets if isinstance(n, ast.Assign) else [n.target]
                        for t in tgt:
                            for x in ast.walk(t):
                                if isinstance(x, ast.Name) and x.id not in local_fix:
                                    local_fix.add(x.id)
                                    changed = True
        # parametrize(...) argument names bound to fixture cases
        for d in getattr(fn, "decorator_list", []):
            if isinstance(d, ast.Call) and _dotted(d.func).endswith("parametrize") and len(d.args) >= 2:
                used = {x.id for x in ast.walk(d.args[1]) if isinstance(x, ast.Name)}
                if used & local_fix and isinstance(d.args[0], ast.Constant):
                    for nm in str(d.args[0].value).split(","):
                        local_fix.add(nm.strip())

        def _refs_fixture(stmt: ast.AST) -> bool:
            return any(isinstance(x, ast.Name) and x.id in local_fix for x in ast.walk(stmt))

        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                n_assert += 1
                if _is_weak_assert(node.test):
                    n_weak += 1
                else:
                    n_strong += 1
                # right-hand side of `< 1e-x` counts as a tolerance literal
                if isinstance(node.test, ast.Compare):
                    for op, comp in zip(node.test.ops, node.test.comparators):
                        if isinstance(op, (ast.Lt, ast.LtE)) and isinstance(comp, ast.Constant) and isinstance(comp.value, float):
                            tolerances.append(("lt", comp.value, node.lineno, _refs_fixture(node)))
            elif isinstance(node, ast.Raise) and node.exc is not None and _name_of(
                node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            ) == "AssertionError":
                # `raise AssertionError(...)` inside a helper is an assertion
                n_assert += 1
                n_strong += 1
            elif isinstance(node, ast.Call):
                fname = _name_of(node.func)
                if fname in ASSERT_CALL_NAMES:
                    n_assert += 1
                    if fname in {"raises", "warns", "deprecated_call", "fail"}:
                        n_raises += 1
                    else:
                        n_strong += 1
                    for kw in node.keywords:
                        if kw.arg in {"rtol", "atol", "abs", "rel", "decimal"} and isinstance(kw.value, ast.Constant):
                            tolerances.append((kw.arg, kw.value.value, node.lineno, _refs_fixture(node)))
                elif fname.startswith("assert"):
                    # unittest-style self.assertEqual etc., or a local assert_* helper
                    n_assert += 1
                    n_strong += 1
                elif resolve_helpers and fname in self.helper_asserts and self.helper_asserts[fname]:
                    n_assert += 1
                    n_helper += 1
                    n_strong += 1
                if fname in {"approx", "isclose", "allclose"}:
                    for kw in node.keywords:
                        if kw.arg in {"rtol", "atol", "abs", "rel"} and isinstance(kw.value, ast.Constant):
                            tolerances.append((kw.arg, kw.value.value, node.lineno, _refs_fixture(node)))
        return {
            "n_assert": n_assert,
            "n_weak": n_weak,
            "n_strong": n_strong,
            "n_raises": n_raises,
            "n_helper": n_helper,
            "tolerances": tolerances,
        }

    # -- tests ---------------------------------------------------------------
    def _collect_tests(self):
        def visit(fn: ast.AST, cls: str | None):
            info = self._scan_body(fn, resolve_helpers=True)
            body = fn.body
            # strip docstring for the duplicate hash
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            dump = "\n".join(ast.dump(b, annotate_fields=False) for b in body)
            h = hashlib.sha1(dump.encode()).hexdigest()[:12]
            kind = "none"
            if info["n_assert"] > 0:
                if info["n_strong"] == 0 and info["n_raises"] == 0 and info["n_weak"] > 0:
                    kind = "weak-only"
                elif info["n_strong"] == 0 and info["n_raises"] > 0:
                    kind = "raises-only"
                else:
                    kind = "numeric"
            calls = sorted({_dotted(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call) and _dotted(n.func).startswith("tsecon.")})
            self.tests.append({
                "file": os.path.basename(self.path),
                "cls": cls,
                "name": fn.name,
                "line": fn.lineno,
                "nlines": (fn.end_lineno or fn.lineno) - fn.lineno + 1,
                "kind": kind,
                "body_hash": h,
                "calls": calls,
                "skipmark": any(_dotted(d.func if isinstance(d, ast.Call) else d).endswith(("skip", "skipif", "xfail")) for d in fn.decorator_list),
                **{k: v for k, v in info.items() if k != "tolerances"},
                "tolerances": info["tolerances"],
            })

        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                visit(node, None)
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith("test"):
                        visit(sub, node.name)


def main() -> int:
    files = sorted(f for f in os.listdir(TESTS) if f.startswith("test_") and f.endswith(".py"))
    all_tests = []
    per_file_tol = {}
    for f in files:
        m = ModuleScan(os.path.join(TESTS, f))
        all_tests.extend(m.tests)
        tol = defaultdict(list)
        for t in m.tests:
            for kind, val, line, fixref in t["tolerances"]:
                tol[kind].append((val, line, fixref))
        # helper-level tolerances too (a `_close(a, b, rtol=...)` helper)
        for name, fn in m.helpers.items():
            info = m._scan_body(fn, resolve_helpers=False)
            for kind, val, line, fixref in info["tolerances"]:
                tol[kind].append((val, line, fixref))
        per_file_tol[f] = {k: sorted(v, key=lambda x: -float(x[0]) if isinstance(x[0], (int, float)) else 0) for k, v in tol.items()}

    by_kind = defaultdict(list)
    for t in all_tests:
        by_kind[t["kind"]].append(t)

    # duplicates: identical body hash across different (file, name)
    by_hash = defaultdict(list)
    for t in all_tests:
        if t["nlines"] >= 3:
            by_hash[t["body_hash"]].append(t)
    dups = [v for v in by_hash.values() if len(v) > 1]

    print(f"test functions scanned: {len(all_tests)} in {len(files)} files")
    for k in ("none", "weak-only", "raises-only", "numeric"):
        print(f"  {k:12s} {len(by_kind[k])}")
    print()
    print("== tests with NO assertion of any kind (cannot fail except via exception) ==")
    for t in by_kind["none"]:
        print(f"  {t['file']}:{t['line']}  {t['cls'] + '::' if t['cls'] else ''}{t['name']}   calls={','.join(c.split('.')[-1] for c in t['calls'])}")
    print()
    print("== tests whose only assertions are structural (shape / key / type / finite / not-None) ==")
    for t in by_kind["weak-only"]:
        print(f"  {t['file']}:{t['line']}  {t['cls'] + '::' if t['cls'] else ''}{t['name']}   n_weak={t['n_weak']} calls={','.join(c.split('.')[-1] for c in t['calls'])}")
    print()
    print(f"== duplicated test bodies ({len(dups)} groups) ==")
    for grp in dups:
        print("  " + "  |  ".join(f"{t['file']}:{t['line']} {t['name']}" for t in grp))

    json.dump(
        {"tests": all_tests, "per_file_tolerances": per_file_tol, "duplicates": dups},
        open(os.path.join(OUT, "assertion_scan.json"), "w"),
        indent=1,
        default=str,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
