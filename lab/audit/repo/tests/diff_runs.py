"""Compare two (or more) suite runs (repo audit, tests sweep, items 1 and 2).

  durations:  .venv-wt/bin/python lab/audit/repo/tests/diff_runs.py durations run_a
  diff:       .venv-wt/bin/python lab/audit/repo/tests/diff_runs.py diff run_a run_b [run_c ...]

`durations` prints the 20 slowest tests and the per-file wall time from the
junit XML of one run (the XML `time` is per test, so a file's total is the sum
over its tests; pytest's own --durations=40 output in the .log carries the
setup/call/teardown split).

`diff` compares the outcome of every test id across the runs (a test present
in one run and absent in another, or with a different outcome, is listed),
then diffs the captured stdout of passing tests between the logs, line by
line, after masking wall-clock-looking tokens (`0.12s`, `123 ms`,
`wall_seconds=`) so only printed *numbers* count as a difference.
"""
from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def load(label):
    tree = ET.parse(os.path.join(OUT, f"{label}.xml"))
    rows = {}
    for tc in tree.iter("testcase"):
        cls = tc.get("classname", "")
        name = tc.get("name", "")
        tid = f"{cls}::{name}"
        outcome = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                outcome = child.tag
            elif child.tag == "skipped":
                outcome = "xfail" if "xfail" in (child.get("type") or "") else "skipped"
        rows[tid] = (outcome, float(tc.get("time", "0")))
    return rows


def durations(label):
    rows = load(label)
    total = sum(t for _, t in rows.values())
    print(f"{label}: {len(rows)} tests, sum of test times {total:.1f} s")
    print()
    print("== 20 slowest tests ==")
    for tid, (o, t) in sorted(rows.items(), key=lambda kv: -kv[1][1])[:20]:
        print(f"  {t:7.1f} s  {tid}")
    print()
    per_file = defaultdict(float)
    n_file = defaultdict(int)
    for tid, (o, t) in rows.items():
        f = tid.split("::")[0].split(".")[-1] + ".py"
        per_file[f] += t
        n_file[f] += 1
    print("== per-file wall time (sum of test times) ==")
    for f, t in sorted(per_file.items(), key=lambda kv: -kv[1]):
        print(f"  {t:7.1f} s  {n_file[f]:4d} tests  {f}")


MASK = re.compile(r"\b\d+(?:\.\d+)?\s*(?:s|ms|sec|seconds)\b|wall_seconds=\S+|\b\d+\.\d+s\b")


def captured_stdout(label):
    """Map test id -> list of stdout lines from the -rA 'PASSED' sections."""
    path = os.path.join(OUT, f"{label}.log")
    out = defaultdict(list)
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"_{3,} (.+?) _{3,}$", line.rstrip())
        if m:
            cur = m.group(1)
            continue
        if line.startswith("PASSED ") or line.startswith("FAILED ") or line.startswith("SKIPPED ") or line.startswith("=") and "short test summary" in line:
            cur = None
            continue
        if cur is not None and not line.startswith("-" * 10):
            out[cur].append(MASK.sub("<t>", line.rstrip()))
    return out


def diff(labels):
    runs = {lb: load(lb) for lb in labels}
    ids = set()
    for r in runs.values():
        ids |= set(r)
    print(f"test ids across runs: {len(ids)}")
    for lb, r in runs.items():
        c = defaultdict(int)
        for o, _ in r.values():
            c[o] += 1
        print(f"  {lb}: {len(r)} tests  {dict(c)}")
    print()
    print("== outcome differences ==")
    n = 0
    for tid in sorted(ids):
        outs = [runs[lb].get(tid, ("ABSENT", 0))[0] for lb in labels]
        if len(set(outs)) > 1:
            n += 1
            print(f"  {tid}: " + " | ".join(f"{lb}={o}" for lb, o in zip(labels, outs)))
    print(f"  ({n} differences)")
    print()
    print("== non-passing tests in any run ==")
    for tid in sorted(ids):
        for lb in labels:
            o = runs[lb].get(tid, ("ABSENT", 0))[0]
            if o not in ("passed",):
                print(f"  {lb}: {o:8s} {tid}")
    print()
    print("== captured-stdout differences between runs (numbers only; timings masked) ==")
    caps = {lb: captured_stdout(lb) for lb in labels}
    base = labels[0]
    n = 0
    for tid in sorted(set().union(*[set(c) for c in caps.values()])):
        ref = caps[base].get(tid)
        for lb in labels[1:]:
            other = caps[lb].get(tid)
            if ref != other:
                n += 1
                print(f"  {tid}: differs between {base} and {lb}")
                if ref and other:
                    for a, b in zip(ref, other):
                        if a != b:
                            print(f"     {base}: {a[:160]}")
                            print(f"     {lb}: {b[:160]}")
                            break
    print(f"  ({n} tests with differing captured stdout)")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "durations":
        durations(sys.argv[2])
    elif mode == "diff":
        diff(sys.argv[2:])
    else:
        sys.exit("usage: diff_runs.py durations <run> | diff <run_a> <run_b> ...")
