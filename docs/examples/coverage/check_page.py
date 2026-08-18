"""Cross-check the published interval-coverage page against the probe registry.

WHY THIS FILE EXISTS
--------------------
`run_all.py` harvests its consolidated tables from structured results, so
nothing is transcribed *inside the runner*. But the published page,
`docs/examples/interval-coverage.md`, carries a hand-committed copy of those
two tables — and audit round 2 (finding 5) caught that copy silently missing a
harvested row: the `ols se_type="hc3"` row, the one estimator `0.2.0` added
because of the audit. A transcription with no guard is exactly the failure
class the coverage tier exists to hunt.

This checker closes the gap without re-running any Monte Carlo. It parses
Table 1 and Table 2 out of the page and verifies, against `run_all.py`'s own
probe registry (`PROBE_BUILDERS`):

  * each table carries exactly one row per registered probe, keyed by
    (surface, interval/option, kind) — a dropped row, a duplicated row, and a
    row for a probe that no longer exists all fail;
  * the page's headline row-count claims match the registry;
  * the group headers ("N of M") and `docs/reference/testing.md`'s summary
    table are arithmetically consistent with the registry (the A/B split
    itself is a measured quantity; only the totals are checkable statically).

The coverage *numbers* in the page are still pinned by the modules' own
assertions and reproduced by `run_all.py`; this file only guarantees the page
cannot silently lose or invent a row again.

Run it directly (exits non-zero on any mismatch):

    python docs/examples/coverage/check_page.py

It also runs inside the Python binding suite
(`bindings/python/tests/test_coverage_page_sync.py`), so CI fails loudly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
PAGE = REPO / "docs" / "examples" / "interval-coverage.md"
TESTING = REPO / "docs" / "reference" / "testing.md"


def load_probes():
    sys.path.insert(0, str(HERE))
    try:
        import run_all
    finally:
        sys.path.pop(0)
    probes = []
    for build in run_all.PROBE_BUILDERS.values():
        probes.extend(build())
    return probes


def page_tables(text: str) -> dict[str, list[tuple[str, str, str]]]:
    """The (surface, option, kind) triples of each published table's rows."""
    tables: dict[str, list[tuple[str, str, str]]] = {}
    for header in ("### Table 1", "### Table 2"):
        i = text.index(header)
        rows = []
        for line in text[i:].splitlines():
            if line.startswith("| `tsecon"):
                cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
                if len(cells) < 4:
                    raise SystemExit(f"{header}: malformed row: {line!r}")
                surface = cells[0].replace("`", "")
                option = cells[1].replace("\\|", "|")
                rows.append((surface, option, cells[2]))
            elif rows and not line.startswith("|"):
                break
        tables[header] = rows
    return tables


def main() -> int:
    probes = load_probes()
    text = PAGE.read_text(encoding="utf-8")
    failures: list[str] = []

    want = [(p.surface, p.option, p.kind) for p in probes]
    want_keys = sorted(want)
    if len(set(want_keys)) != len(want_keys):
        failures.append("probe registry itself has duplicate (surface, option) keys")

    for header, rows in page_tables(text).items():
        got = sorted(rows)
        if got == want_keys:
            continue
        missing = [k for k in want_keys if k not in got]
        extra = [k for k in got if k not in want_keys]
        for k in missing:
            failures.append(f"{header}: registered probe has NO row: {k}")
        for k in extra:
            failures.append(f"{header}: row matches NO registered probe: {k}")
        if not missing and not extra:
            failures.append(f"{header}: duplicated rows: "
                            f"{[k for k in got if got.count(k) > 1]}")

    # headline count claims on the page
    n = len(probes)
    for claim in (rf"\*\*{n} interval-valued outputs across \d+ functions\*\*",
                  rf"## The headline: {n} measured interval-valued outputs"):
        if not re.search(claim, text):
            failures.append(f"page headline no longer matches the registry "
                            f"({n} probes): /{claim}/ not found")

    # group headers: the "of M" totals must equal the registry's counts
    n_freq = sum(1 for p in probes if p.kind in ("CI", "PRED") and p.stress)
    n_diag = sum(1 for p in probes if p.kind in ("CRED", "SET"))
    m = re.search(r"\*\*(\d+) of (\d+)\*\*\.\s+These are off nominal", text)
    a_count = None
    if m:
        a_count = int(m.group(1))
        if int(m.group(2)) != n_freq:
            failures.append(f"group A header says 'of {m.group(2)}', registry "
                            f"has {n_freq} frequentist probes")
    m = re.search(r"\*\*(\d+) of (\d+)\*\*\.\s+These behave", text)
    if m:
        if int(m.group(2)) != n_freq:
            failures.append(f"group B header says 'of {m.group(2)}', registry "
                            f"has {n_freq} frequentist probes")
        if a_count is not None and a_count + int(m.group(1)) != n_freq:
            failures.append(
                f"group A ({a_count}) + group B ({m.group(1)}) != {n_freq}")
    m = re.search(r"\*\*(\d+) rows\.\*\* Nothing here is a defect", text)
    if m and int(m.group(1)) != n_diag:
        failures.append(f"group C header says {m.group(1)}, registry has "
                        f"{n_diag} CRED/SET probes")

    # testing.md's summary of this page
    ttext = TESTING.read_text(encoding="utf-8")
    m = re.search(r"frequentist intervals measured \(CI \+ PRED\) \| (\d+)", ttext)
    if m and int(m.group(1)) != n_freq:
        failures.append(f"testing.md says {m.group(1)} frequentist intervals, "
                        f"registry has {n_freq}")
    if m:
        rows = re.findall(r"\| — .*? \| \*?\*?(\d+)\*?\*? \|", ttext[m.start():])
        if len(rows) >= 2 and int(rows[0]) + int(rows[1]) != n_freq:
            failures.append(f"testing.md favourable-miss ({rows[0]}) + "
                            f"stress-miss ({rows[1]}) != {n_freq}")

    if failures:
        print("interval-coverage page is OUT OF SYNC with the probe registry:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"interval-coverage page in sync: {n} probes x 2 tables, "
          f"{n_freq} frequentist + {n_diag} diagnostic + "
          f"{n - n_freq - n_diag} no-band")
    return 0


if __name__ == "__main__":
    sys.exit(main())
