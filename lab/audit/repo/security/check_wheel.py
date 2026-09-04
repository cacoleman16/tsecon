#!/usr/bin/env python
"""Wheel-contents check (repo audit, security): does the built wheel ship
exactly the package and nothing else?

    maturin build --release -o dist          # in bindings/python
    python lab/audit/repo/security/check_wheel.py bindings/python/dist/tsecon-*.whl

Expected members: the ``tsecon`` package (``__init__.py``, ``__init__.pyi``,
``py.typed``, ``_coerce.py``, ``_inspect.py``, ``results/*.py``), the compiled
``_core`` extension, and the ``.dist-info`` directory (METADATA, WHEEL, RECORD,
and the two license texts under ``licenses/``). Anything else — tests,
fixtures, notebooks, ``.pyc``, the JOSS paper, lab material, scratch files —
is reported as UNEXPECTED. Also verifies the claims the repository makes about
the wheel: dual license texts present; METADATA's ``License-File`` entries;
whether third-party license notices are present (THIRD-PARTY-LICENSES.md says
they are "reproduced in released wheels").
"""
from __future__ import annotations

import re
import sys
import zipfile

EXPECTED_PY = {
    "tsecon/__init__.py",
    "tsecon/__init__.pyi",
    "tsecon/py.typed",
    "tsecon/_coerce.py",
    "tsecon/_inspect.py",
}
EXPECTED_RESULTS_RE = re.compile(r"^tsecon/results/[A-Za-z_]+\.py$")
EXPECTED_EXT_RE = re.compile(r"^tsecon/_core\.(abi3|cpython[^.]*)\.(so|pyd|dylib)$")
EXPECTED_DISTINFO_RE = re.compile(r"^tsecon-[0-9][^/]*\.dist-info/(METADATA|WHEEL|RECORD|licenses/LICENSE-MIT|licenses/LICENSE-APACHE)$")
SUSPICIOUS_RE = re.compile(r"(^|/)(tests?|fixtures|notebooks|paper|lab|scratch|benchmarks|docs|target|\.venv[^/]*|__pycache__)(/|$)|\.pyc$|\.ipynb$|\.log$|\.env$|\.pypirc$")


def main(path):
    zf = zipfile.ZipFile(path)
    names = [i.filename for i in zf.infolist()]
    total = sum(i.file_size for i in zf.infolist())
    print(f"{path}: {len(names)} members, {total / 1e6:.1f} MB uncompressed")
    unexpected, suspicious, ext, distinfo = [], [], [], []
    for n in names:
        if n in EXPECTED_PY or EXPECTED_RESULTS_RE.match(n):
            continue
        if EXPECTED_EXT_RE.match(n):
            ext.append(n)
            continue
        if EXPECTED_DISTINFO_RE.match(n):
            distinfo.append(n)
            continue
        unexpected.append(n)
        if SUSPICIOUS_RE.search(n):
            suspicious.append(n)
    for n in names:
        print("   ", n, f"({zf.getinfo(n).file_size} B)")
    print(f"extension modules: {ext}")
    print(f"dist-info members: {sorted(distinfo)}")
    print(f"UNEXPECTED members: {unexpected or 'none'}")
    print(f"SUSPICIOUS members (tests/fixtures/pyc/...): {suspicious or 'none'}")

    meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
    meta = zf.read(meta_name).decode()
    lic_files = re.findall(r"^License-File: (.*)$", meta, re.M)
    lic_expr = re.findall(r"^License(?:-Expression)?: (.*)$", meta, re.M)
    print(f"METADATA License: {lic_expr}; License-File: {lic_files}")
    print(f"METADATA Requires-Dist: {re.findall(r'^Requires-Dist: (.*)$', meta, re.M)}")
    print(f"METADATA Requires-Python: {re.findall(r'^Requires-Python: (.*)$', meta, re.M)}")
    has_mit = any(n.endswith("licenses/LICENSE-MIT") for n in names)
    has_apache = any(n.endswith("licenses/LICENSE-APACHE") for n in names)
    third = [n for n in names if re.search(r"(?i)third|notice|about", n)]
    print(f"dual license texts shipped: MIT={has_mit} Apache={has_apache}")
    print(f"third-party license notices in wheel: {third or 'NONE'}")
    missing = sorted(EXPECTED_PY - set(names))
    print(f"missing expected package files: {missing or 'none'}")
    ok = not unexpected and has_mit and has_apache and not missing and len(ext) == 1
    print("VERDICT:", "clean" if ok else "see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
