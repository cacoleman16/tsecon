"""CI guard: the type stub must describe exactly the runtime function surface.

If a binding is added/removed without updating tsecon.pyi, this fails —
keeping the stub (and every user's autocomplete) honest.
"""
import re
from pathlib import Path

import tsecon

ROOT = Path(__file__).parents[1]


def _runtime_funcs():
    return {
        n for n in dir(tsecon)
        if not n.startswith("_") and callable(getattr(tsecon, n))
    }


def _stub_funcs():
    # UTF-8 explicitly: the stub's docstrings contain em-dashes, which the
    # Windows default (cp1252) decodes to mush rather than failing loudly.
    text = (ROOT / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    return set(re.findall(r"^def (\w+)\(", text, re.MULTILINE))


def test_stub_matches_runtime():
    runtime, stub = _runtime_funcs(), _stub_funcs()
    missing = runtime - stub
    extra = stub - runtime
    assert not missing, f"functions missing from tsecon.pyi: {sorted(missing)}"
    assert not extra, f"tsecon.pyi documents non-existent functions: {sorted(extra)}"


def test_stub_parses_as_python():
    """The stub must be valid Python, or every type checker discards it whole.

    Audit round 13: two docstrings appended by parallel slices lost their
    closing triple quote in the merge (the append-point hunk class recorded in
    round 11's integrator notes), so the stub raised ``SyntaxError`` at
    ``ast.parse`` and mypy reported "errors prevented further checking" for
    any file importing tsecon — while ``test_stub_matches_runtime`` stayed
    green, because it only regexes ``def`` names, and the regex-based api.md
    generator pasted the next function's signature into the previous
    function's prose. ``ast.parse`` is what mypy/pyright do first.
    """
    import ast

    text = (ROOT / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    tree = ast.parse(text)  # raises SyntaxError with the line number on drift
    docs = {n.name: ast.get_docstring(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    # Every stub entry carries a docstring, and none has swallowed a neighbour
    # (a "def " or section comment inside a docstring is the unterminated case).
    for name, doc in docs.items():
        assert doc, f"{name}: stub entry has no docstring"
        assert "\ndef " not in doc and "# ---" not in doc, f"{name}: docstring swallowed the next entry"
    assert set(docs) == _runtime_funcs()


def test_py_typed_marker_present():
    assert (ROOT / "python" / "tsecon" / "py.typed").exists()


def test_api_reference_not_stale():
    """The generated API reference must match a fresh generation from the stub.

    docs/reference/api.md is generated from the stub by docs/gen_api_reference.py.
    The stub-vs-module guard above keeps the stub honest, but nothing stopped
    api.md from drifting when new functions were added. Regenerating must leave
    the committed file byte-identical, so a forgotten regeneration fails CI
    instead of silently shipping a stale reference.
    """
    import subprocess
    import sys

    import pytest

    repo = ROOT.parent.parent
    gen_path = repo / "docs" / "gen_api_reference.py"
    out_path = repo / "docs" / "reference" / "api.md"
    if not gen_path.exists() or not out_path.exists():
        pytest.skip("docs tree not present in this checkout")

    # Read explicitly as UTF-8: the docs contain em-dashes, and on Windows the
    # platform default (cp1252) would mangle them and make this compare garbage
    # against garbage. The generator is pinned to UTF-8 for the same reason.
    before = out_path.read_text(encoding="utf-8")
    subprocess.run(
        [sys.executable, str(gen_path)], cwd=str(repo), check=True, capture_output=True
    )
    after = out_path.read_text(encoding="utf-8")
    assert before == after, (
        "docs/reference/api.md is stale — run "
        "`python docs/gen_api_reference.py` and commit the result"
    )
