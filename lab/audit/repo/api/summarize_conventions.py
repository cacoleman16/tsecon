"""Tabulate out/conventions.json into the compliance matrix (item 4).

Per probe (nan / empty / ndim / string / negative): exception class counts,
panic escapes, silent returns (and whether the silent return carried
non-finite values), and — for raising calls — whether the message names the
argument, names the function, and states a fix. A function is "fully
compliant" when every applicable probe raised a Python exception (not a
panic) whose message names the argument or states a fix; "partial" when at
least one applicable probe did; "non-compliant" otherwise.

Run:  .venv-wt/bin/python lab/audit/repo/api/summarize_conventions.py
Out:  lab/audit/repo/api/out/compliance.md (+ stdout)
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
C = json.load(open(os.path.join(OUT, "conventions.json")))
PROBES = ("nan", "empty", "ndim", "string", "negative")


def teaching(rec):
    """A raised Python exception whose message names the argument or a fix."""
    return rec.get("exc") not in ("silent", "n/a") and not rec.get("panic") and (rec.get("names_arg") or rec.get("has_fix"))


def main():
    lines = []
    exc_counts = {p: Counter() for p in PROBES}
    names_arg = {p: [0, 0] for p in PROBES}
    has_fix = {p: [0, 0] for p in PROBES}
    panics = []
    silent = defaultdict(list)
    crashes = []
    per_fn = {}
    for name, rec in sorted(C.items()):
        if "crash" in rec:
            crashes.append((name, rec["crash"][:200]))
            continue
        status = []
        for p in PROBES:
            r = rec.get(p, {})
            e = r.get("exc", "?")
            exc_counts[p][e] += 1
            if e == "n/a":
                status.append(None)
                continue
            if r.get("panic"):
                panics.append((name, p, r.get("param"), r.get("msg", "")[:200]))
            if e == "silent":
                silent[p].append((name, r.get("param"), r.get("nonfinite")))
                status.append(False)
                continue
            names_arg[p][1] += 1
            has_fix[p][1] += 1
            if r.get("names_arg"):
                names_arg[p][0] += 1
            if r.get("has_fix"):
                has_fix[p][0] += 1
            status.append(teaching(r))
        applicable = [s for s in status if s is not None]
        if not applicable:
            per_fn[name] = "n/a"
        elif all(applicable):
            per_fn[name] = "full"
        elif any(applicable):
            per_fn[name] = "partial"
        else:
            per_fn[name] = "none"
    tally = Counter(per_fn.values())
    lines.append("| probe | ValueError | TypeError | other Python exc | PanicException | silent return | n/a | names the argument | states a fix |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in PROBES:
        c = exc_counts[p]
        other = sum(v for k, v in c.items() if k not in ("ValueError", "TypeError", "PanicException", "silent", "n/a"))
        lines.append(
            f"| {p} | {c['ValueError']} | {c['TypeError']} | {other} | {c['PanicException']} | {c['silent']} | {c['n/a']} | "
            f"{names_arg[p][0]}/{names_arg[p][1]} | {has_fix[p][0]}/{has_fix[p][1]} |"
        )
    lines.append("")
    lines.append(f"Per-function verdict over {len(per_fn)} callables: full {tally['full']}, partial {tally['partial']}, none {tally['none']}, n/a {tally['n/a']}.")
    lines.append("")
    lines.append(f"Panic escapes: {len(panics)}")
    for t in panics:
        lines.append(f"- `{t[0]}` probe {t[1]} (param `{t[2]}`): `{t[3]}`")
    for p in PROBES:
        if silent[p]:
            lines.append("")
            lines.append(f"Silent returns on **{p}** ({len(silent[p])}): " + ", ".join(f"`{n}`" + (" (non-finite out)" if nf else " (finite out)") for n, _, nf in silent[p]))
    other_exc = Counter()
    for name, rec in C.items():
        for p in PROBES:
            e = rec.get(p, {}).get("exc")
            if e and e not in ("ValueError", "TypeError", "PanicException", "silent", "n/a"):
                other_exc[(e, name, p)] += 1
    if other_exc:
        lines.append("")
        lines.append("Other exception classes: " + ", ".join(f"`{e}` in `{n}`/{p}" for (e, n, p) in sorted(other_exc)))
    lines.append("")
    lines.append("Non-compliant or partial functions:")
    for name, v in sorted(per_fn.items()):
        if v in ("none", "partial"):
            rec = C[name]
            cells = []
            for p in PROBES:
                r = rec.get(p, {})
                e = r.get("exc")
                if e == "n/a":
                    cells.append(f"{p}=n/a")
                elif e == "silent":
                    cells.append(f"{p}=SILENT")
                elif r.get("panic"):
                    cells.append(f"{p}=PANIC")
                elif teaching(r):
                    cells.append(f"{p}=ok")
                else:
                    cells.append(f"{p}={e}(no arg/fix: {r.get('msg','')[:80]!r})")
            lines.append(f"- `{name}` [{v}]: " + "; ".join(cells))
    if crashes:
        lines.append("")
        lines.append("Subprocess crashes / timeouts: " + "; ".join(f"`{n}`: {c}" for n, c in crashes))
    text = "\n".join(lines)
    open(os.path.join(OUT, "compliance.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
