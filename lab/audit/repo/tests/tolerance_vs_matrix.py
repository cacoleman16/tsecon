"""Python-test tolerances vs the validation matrix (repo audit, tests sweep, 3c).

docs/reference/validation-matrix.md states, per family, the fixture and the
*asserted* tolerance. This joins every Python test file to the matrix rows
whose fixture it loads (by `*.json` literal) and compares the loosest
`rtol=` / `atol=` / `abs=` / `rel=` literal in that file's assertions against
the loosest number the matrix row quotes. A file whose loosest literal exceeds
the row's loosest documented tolerance is a candidate; each candidate is then
read by hand (the literal may belong to an assertion that is not against the
fixture at all, e.g. a property check or a Monte-Carlo bound).

Run:  .venv-wt/bin/python lab/audit/repo/tests/assertion_scan.py   # first
      .venv-wt/bin/python lab/audit/repo/tests/tolerance_vs_matrix.py
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TESTS = os.path.join(REPO, "bindings", "python", "tests")
MATRIX = os.path.join(REPO, "docs", "reference", "validation-matrix.md")

NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?e-\d+|0\.\d+|\d\.\d+e-\d+)(?![\w])")


def matrix_rows():
    rows = []
    for line in open(MATRIX, encoding="utf-8"):
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| Family"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        family, against, fixture, test, tol = cells[:5]
        fixtures = re.findall(r"`([\w\-]+\.json)`", fixture)
        nums = [float(x) for x in NUM.findall(tol)]
        rows.append({"family": family[:70], "fixtures": fixtures, "tol_text": tol, "tol_nums": nums})
    return rows


def main():
    scan = json.load(open(os.path.join(HERE, "out", "assertion_scan.json")))
    per_file = scan["per_file_tolerances"]
    rows = matrix_rows()
    print(f"matrix rows parsed: {len(rows)}; rows with a fixture: {sum(1 for r in rows if r['fixtures'])}")

    by_fixture = {}
    for r in rows:
        for fx in r["fixtures"]:
            by_fixture.setdefault(fx, []).append(r)

    print()
    print("== per test file: fixtures loaded, loosest literal in file, loosest documented in matrix row ==")
    candidates = []
    for f in sorted(os.listdir(TESTS)):
        if not f.startswith("test_"):
            continue
        src = open(os.path.join(TESTS, f), encoding="utf-8").read()
        fixtures = sorted(set(re.findall(r"[\"']([\w\-]+\.json)[\"']", src)))
        if not fixtures:
            continue
        tols = per_file.get(f, {})
        loosest = None
        loosest_where = ""
        # only literals in an assertion that reads a fixture-derived name count;
        # a `< 0.35` on a Monte-Carlo bound is not a golden tolerance
        for kind in ("rtol", "atol", "abs", "rel", "lt"):
            for val, line, fixref in tols.get(kind, []):
                if not fixref:
                    continue
                if isinstance(val, (int, float)) and (loosest is None or val > loosest):
                    loosest, loosest_where = float(val), f"{kind}={val} @ L{line}"
        for fx in fixtures:
            mrows = by_fixture.get(fx, [])
            if not mrows:
                print(f"  {f:44s} {fx:30s}  (no matrix row names this fixture)  file-loosest {loosest_where}")
                continue
            for r in mrows:
                doc_max = max(r["tol_nums"]) if r["tol_nums"] else None
                flag = ""
                if loosest is not None and doc_max is not None and loosest > doc_max * 1.0001:
                    flag = "  <-- LOOSER than documented"
                    candidates.append((f, fx, loosest_where, doc_max, r["family"]))
                print(f"  {f:44s} {fx:30s}  file-loosest {loosest_where:22s} matrix-loosest {doc_max}  [{r['family'][:50]}]{flag}")
    print()
    print(f"== candidates: {len(candidates)} (each to be read by hand) ==")
    for c in candidates:
        print("  %s  %s  file %s  vs matrix %s  (%s)" % c)


if __name__ == "__main__":
    sys.exit(main())
