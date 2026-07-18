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
    text = (ROOT / "python" / "tsecon" / "__init__.pyi").read_text()
    return set(re.findall(r"^def (\w+)\(", text, re.MULTILINE))


def test_stub_matches_runtime():
    runtime, stub = _runtime_funcs(), _stub_funcs()
    missing = runtime - stub
    extra = stub - runtime
    assert not missing, f"functions missing from tsecon.pyi: {sorted(missing)}"
    assert not extra, f"tsecon.pyi documents non-existent functions: {sorted(extra)}"


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

    before = out_path.read_text()
    subprocess.run(
        [sys.executable, str(gen_path)], cwd=str(repo), check=True, capture_output=True
    )
    after = out_path.read_text()
    assert before == after, (
        "docs/reference/api.md is stale — run "
        "`python docs/gen_api_reference.py` and commit the result"
    )
