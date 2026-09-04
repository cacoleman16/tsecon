"""Skip / xfail / #[ignore] inventory (repo audit, tests sweep, item 4).

Lists every `pytest.skip`, `pytest.importorskip`, `pytest.mark.skip[if]`,
`pytest.mark.xfail` in bindings/python/tests with its stated reason and the
module it gates, plus every `#[ignore ...]` in crates/*/tests and crates/*/src
with the reason attribute and the test it attaches to. For each importorskip
the script also says whether the module is importable in the current venv, so
a "skip because extra X is absent" that never fires in the full-extras venv is
visible, and one that does fire (a gap hidden in this venv) is flagged.

Run:  .venv-wt/bin/python lab/audit/repo/tests/skips_scan.py
"""
from __future__ import annotations

import ast
import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TESTS = os.path.join(REPO, "bindings", "python", "tests")
CRATES = os.path.join(REPO, "crates")


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _const(node):
    if isinstance(node, ast.Constant):
        return node.value
    return ast.unparse(node) if hasattr(ast, "unparse") else "<expr>"


def python_skips():
    rows = []
    for f in sorted(os.listdir(TESTS)):
        if not f.endswith(".py"):
            continue
        path = os.path.join(TESTS, f)
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        # module-level importorskip => whole-file collection gate
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                d = _dotted(node.func)
                if d in {"pytest.importorskip"}:
                    mod = _const(node.args[0]) if node.args else "?"
                    rows.append(("importorskip", f, node.lineno, mod, ""))
                elif d in {"pytest.skip"}:
                    reason = _const(node.args[0]) if node.args else ""
                    for kw in node.keywords:
                        if kw.arg == "reason":
                            reason = _const(kw.value)
                    rows.append(("skip", f, node.lineno, "", str(reason)))
                elif d in {"pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail"}:
                    cond = _const(node.args[0]) if node.args else ""
                    reason = ""
                    for kw in node.keywords:
                        if kw.arg == "reason":
                            reason = _const(kw.value)
                    rows.append((d.split(".")[-1], f, node.lineno, str(cond), str(reason)))
            elif isinstance(node, ast.Attribute) and _dotted(node) in {"pytest.mark.skip", "pytest.mark.xfail"}:
                rows.append((node.attr, f, node.lineno, "", "(bare mark)"))
    return rows


def rust_ignores():
    rows = []
    pat = re.compile(r"#\[ignore(?:\s*=\s*\"([^\"]*)\")?\]")
    for root, _dirs, files in os.walk(CRATES):
        for fn in files:
            if not fn.endswith(".rs"):
                continue
            path = os.path.join(root, fn)
            lines = open(path, encoding="utf-8").read().splitlines()
            for i, line in enumerate(lines):
                m = pat.search(line)
                if not m:
                    continue
                reason = m.group(1) or ""
                # find the fn name in the next few lines
                name = "?"
                for j in range(i + 1, min(i + 6, len(lines))):
                    mm = re.search(r"fn\s+(\w+)", lines[j])
                    if mm:
                        name = mm.group(1)
                        break
                # a comment reason on the preceding lines, if the attribute has none
                if not reason:
                    for j in range(i - 1, max(i - 4, -1), -1):
                        if lines[j].strip().startswith("//"):
                            reason = "(comment) " + lines[j].strip().lstrip("/ ")
                            break
                rows.append((os.path.relpath(path, REPO), i + 1, name, reason))
    return rows


def main():
    rows = python_skips()
    print(f"== Python skip-family sites: {len(rows)} ==")
    seen_mods = {}
    for kind, f, line, arg, reason in rows:
        extra = ""
        if kind == "importorskip":
            if arg not in seen_mods:
                try:
                    importlib.import_module(arg)
                    seen_mods[arg] = "importable"
                except Exception as e:  # noqa: BLE001
                    seen_mods[arg] = f"NOT importable ({type(e).__name__})"
            extra = f"  [{seen_mods[arg]}]"
        print(f"  {kind:13s} {f}:{line}  {arg!s:24s} {reason}{extra}")
    print()
    r = rust_ignores()
    print(f"== Rust #[ignore] sites: {len(r)} ==")
    for path, line, name, reason in r:
        print(f"  {path}:{line}  {name}  -- {reason}")


if __name__ == "__main__":
    sys.exit(main())
