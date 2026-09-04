"""Blast radius of a rename: mentions per spelling across the repo (item 6).

For every parameter spelling and return-key spelling in a cluster, count the
functions using it (from surface.json) and the textual mentions across
docs/**/*.md, the stub, the Rust bindings, tests, notebooks, and the
examples, so a rename proposal can say "n functions, m docs mentions".

Run:  .venv-wt/bin/python lab/audit/repo/api/probe_blast.py name1 name2 ...
      (no arguments: every spelling in the concept clusters of clusters.json)
Out:  lab/audit/repo/api/out/blast.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
OUT = os.path.join(HERE, "out")
SURFACE = json.load(open(os.path.join(HERE, "surface.json")))

AREAS = {
    "docs": ["docs", "--glob", "*.md", "--glob", "!docs/reference/api.md", "--glob", "!docs/roadmap/**"],
    "api.md": ["docs/reference/api.md"],
    "stub": ["bindings/python/python/tsecon/__init__.pyi"],
    "bindings": ["bindings/python/src"],
    "tests": ["bindings/python/tests"],
    "examples+notebooks": ["docs/examples", "notebooks", "--glob", "*.py", "--glob", "*.ipynb"],
}


def count(word, area_args):
    # parameter usage: `word=` (call sites), backticked `word`, and bare word
    pat = rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])"
    cmd = ["rg", "-c", "--no-messages", "-P", "-e", pat, *area_args]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    total = 0
    files = 0
    for line in p.stdout.splitlines():
        try:
            n = int(line.rsplit(":", 1)[1])
        except (ValueError, IndexError):
            continue
        total += n
        files += 1
    return total, files


def main():
    words = sys.argv[1:]
    if not words:
        cl = json.load(open(os.path.join(OUT, "clusters.json")))
        words = sorted({w for c in cl["param_clusters"].values() for w in c} | {w for c in cl["key_clusters"].values() for w in c})
    fn_by_param = {}
    fn_by_key = {}
    for name, rec in SURFACE.items():
        for p in rec.get("params") or []:
            fn_by_param.setdefault(p["name"], []).append(name)
        for k in rec.get("keys") or {}:
            fn_by_key.setdefault(k, []).append(name)
    out = {}
    for w in words:
        rec = {"functions_as_param": len(fn_by_param.get(w, [])), "functions_as_key": len(fn_by_key.get(w, []))}
        for area, args in AREAS.items():
            n, f = count(w, args)
            rec[area] = {"mentions": n, "files": f}
        out[w] = rec
        print(f"{w:22s} param-fns={rec['functions_as_param']:3d} key-fns={rec['functions_as_key']:3d} " + " ".join(f"{a}={rec[a]['mentions']}" for a in AREAS))
    json.dump(out, open(os.path.join(OUT, "blast.json"), "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
