"""Does the shipped type stub parse as Python, and what did the regex-based
api.md generator do with it? (Round 13, sweep F — run before and after the fix.)

Run:  .venv/bin/python lab/audit/round13/probe_stub_parse.py
"""
from __future__ import annotations

import ast
import re
import sys

from common import API_MD, PYI

src = open(PYI, encoding="utf-8").read()
try:
    ast.parse(src)
    print("ast.parse(stub): OK")
except SyntaxError as exc:
    print(f"ast.parse(stub): SyntaxError line {exc.lineno}: {exc.msg}")
lines = src.splitlines()
state, open_at, breaches = "code", None, []
for i, line in enumerate(lines, 1):
    n = line.count('"""')
    if state == "doc" and (re.match(r"^def \w+\(", line) or line.startswith("# ---")):
        breaches.append((open_at, i, line[:70]))
    if n % 2 == 1:
        state, open_at = ("doc", i) if state == "code" else ("code", None)
print(f'triple-quote tokens: {src.count(chr(34) * 3)}; unterminated docstrings: {len(breaches)}')
for o, i, l in breaches:
    print(f"  docstring opened at line {o} still open at line {i}: {l!r}")
api = open(API_MD, encoding="utf-8").read().splitlines()
for i, l in enumerate(api, 1):
    if l.startswith("# ---") or (l.startswith("def ") and not api[i - 2].startswith("```")):
        print(f"api.md:{i}: stub source leaked into prose: {l[:80]!r}")
sys.exit(0)
