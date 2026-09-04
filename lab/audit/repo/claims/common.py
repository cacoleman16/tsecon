"""Shared helpers for the repository claims audit (audit round 12, claims sweep).

The doc set is the list the brief names: every page a newcomer or a referee
reads, minus the generated ``docs/reference/api.md``. Every sweep in this
directory imports ``DOC_FILES`` from here so the set is stated once.

Run everything from the repository root with the worktree venv::

    .venv-wt/bin/python lab/audit/repo/claims/sweep_names.py
"""
from __future__ import annotations

import glob
import inspect
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
ROUND11 = os.path.join(REPO, "lab", "audit", "round11")
PYI = os.path.join(REPO, "bindings", "python", "python", "tsecon", "__init__.pyi")


def doc_files():
    """The audited doc set, repo-relative, in a stable order."""
    fixed = [
        "README.md",
        "ROADMAP.md",
        "CONTRIBUTING.md",
        "paper/paper.md",
        "docs/index.md",
        "docs/quickstart.md",
        "docs/which-model-when.md",
    ]
    globs = [
        "docs/guide/*.md",
        "docs/cookbook/*.md",
        "docs/reference/*.md",
        "docs/reference/model-cards/*.md",
        "docs/migration/*.md",
        "docs/examples/*.md",
    ]
    out = list(fixed)
    for g in globs:
        out.extend(sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, g))))
    return [p for p in out if not p.endswith("docs/reference/api.md")]


DOC_FILES = doc_files()

# --------------------------------------------------------------------------- #
# markdown parsing
# --------------------------------------------------------------------------- #
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")


def split_blocks(text):
    """Yield ('code', lang, start_line, body) and ('prose', None, start_line, body).

    Line numbers are 1-based and point at the fence line for code blocks.
    """
    lines = text.split("\n")
    i = 0
    prose_start = 1
    prose = []
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m:
            if prose:
                yield ("prose", None, prose_start, "\n".join(prose))
                prose = []
            fence, lang = m.group(1), m.group(2).lower()
            start = i + 1
            body = []
            i += 1
            while i < len(lines) and not (lines[i].strip().startswith(fence[0] * 3) and lines[i].strip().strip("`~") == ""):
                body.append(lines[i])
                i += 1
            yield ("code", lang, start, "\n".join(body))
            i += 1
            prose_start = i + 1
        else:
            prose.append(lines[i])
            i += 1
    if prose:
        yield ("prose", None, prose_start, "\n".join(prose))


def paragraphs(prose, start_line):
    """Split prose into paragraphs; yield (start_line, text). Table rows are
    their own paragraphs so a migration table row is classified on its own."""
    buf, buf_start = [], start_line
    for k, line in enumerate(prose.split("\n")):
        ln = start_line + k
        if line.strip() == "" or line.lstrip().startswith("|"):
            if buf:
                yield buf_start, "\n".join(buf)
                buf = []
            if line.lstrip().startswith("|"):
                yield ln, line
            continue
        if not buf:
            buf_start = ln
        buf.append(line)
    if buf:
        yield buf_start, "\n".join(buf)


# --------------------------------------------------------------------------- #
# the public surface
# --------------------------------------------------------------------------- #
def public_callables():
    import tsecon

    return sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))


def stub_signatures():
    """name -> list of parameter names, parsed from the stub (the fallback when
    inspect.signature cannot see a compiled function's parameters)."""
    import ast

    tree = ast.parse(open(PYI, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            a = node.args
            names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
            if a.vararg:
                names.append(a.vararg.arg)
            if a.kwarg:
                names.append(a.kwarg.arg)
            out[node.name] = names
    return out


def signature_params(name):
    """Parameter names of a public callable: inspect.signature first, stub second."""
    import tsecon

    fn = getattr(tsecon, name)
    try:
        return [p for p in inspect.signature(fn).parameters]
    except (TypeError, ValueError):
        return stub_signatures().get(name, [])


def keys_cache_path():
    return os.path.join(OUT, "returned_keys.json")


def load_keys():
    """name -> {"top": [...], "nested": [...]} as produced by collect_keys.py."""
    p = keys_cache_path()
    if not os.path.exists(p):
        raise SystemExit("run collect_keys.py first (it needs the installed wheel)")
    return json.load(open(p))


def log(fh, *parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    fh.write(line + "\n")
    fh.flush()
