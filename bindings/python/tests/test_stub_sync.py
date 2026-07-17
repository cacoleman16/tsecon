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
    text = (ROOT / "tsecon.pyi").read_text()
    return set(re.findall(r"^def (\w+)\(", text, re.MULTILINE))


def test_stub_matches_runtime():
    runtime, stub = _runtime_funcs(), _stub_funcs()
    missing = runtime - stub
    extra = stub - runtime
    assert not missing, f"functions missing from tsecon.pyi: {sorted(missing)}"
    assert not extra, f"tsecon.pyi documents non-existent functions: {sorted(extra)}"


def test_py_typed_marker_present():
    assert (ROOT / "py.typed").exists()
