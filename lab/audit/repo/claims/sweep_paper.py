"""JOSS draft check: every quantitative claim in paper/paper.md recomputed, every
function it names resolved against the wheel, and every ``@key`` citation
resolved against paper/paper.bib.

Run:  .venv-wt/bin/python lab/audit/repo/claims/sweep_paper.py
Out:  out/sweep_paper.log
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsecon  # noqa: E402
from common import OUT, REPO, log, public_callables  # noqa: E402


def main():
    fh = open(os.path.join(OUT, "sweep_paper.log"), "w")
    md = open(os.path.join(REPO, "paper", "paper.md"), encoding="utf-8").read()
    bib = open(os.path.join(REPO, "paper", "paper.bib"), encoding="utf-8").read()
    public = set(public_callables())

    # citations
    cited = sorted(set(re.findall(r"@([A-Za-z0-9_:-]+)", md)))
    entries = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    log(fh, "cited keys:", len(cited), "bib entries:", len(entries))
    for k in cited:
        log(fh, f"  {'OK ' if k in entries else 'MISSING'} @{k}")
    unused = sorted(entries - set(cited))
    log(fh, "bib entries never cited:", unused)

    # functions named
    names = sorted(set(re.findall(r"`([a-z_][a-z0-9_]*)`", md)))
    fn_like = [n for n in names if n in public]
    phantoms = [n for n in names if n not in public and "_" in n and not n.startswith("se") and n not in ("se_type", "seasonal_order")]
    log(fh, f"backticked snake_case names: {len(names)}; public callables among them: {len(fn_like)}")
    log(fh, "snake_case names that are NOT public callables (review):", phantoms)
    kw_like = [n for n in names if n not in public]
    log(fh, "all non-callable backticked names:", kw_like)

    # numbers
    n_pub = len(public)
    n_crates = len(glob.glob(os.path.join(REPO, "crates", "*", "Cargo.toml")))
    n_hac = len([p for p in glob.glob(os.path.join(REPO, "crates", "*", "Cargo.toml")) if re.search(r"^tsecon-hac", open(p).read(), re.M)])
    n_pyfiles = len(glob.glob(os.path.join(REPO, "bindings", "python", "tests", "test_*.py")))
    integ = subprocess.run("grep -c '#\\[test\\]' crates/*/tests/*.rs | awk -F: '{s+=$NF} END {print s}'", shell=True, cwd=REPO, capture_output=True, text=True).stdout.strip()
    unit = subprocess.run("grep -rc '#\\[test\\]' crates/*/src | awk -F: '{s+=$NF} END {print s}'", shell=True, cwd=REPO, capture_output=True, text=True).stdout.strip()
    ignored = subprocess.run("grep -rh '#\\[ignore' crates/*/tests/*.rs crates/*/src | grep -v '^\\s*//' | wc -l", shell=True, cwd=REPO, capture_output=True, text=True).stdout.strip()
    claims = {
        "173 functions": (173, n_pub),
        "43 Rust crates": (43, n_crates),
        "eighteen of the crates consume HAC": (18, n_hac),
        "1526 tests across 99 files (files)": (99, n_pyfiles),
        "1479 integration tests (static #[test] in crates/*/tests, incl. ignored)": (1479 + 10, int(integ)),
        "242 unit tests (static #[test] in src)": (242, int(unit)),
        "10 explicitly ignored (static #[ignore)": (10, int(ignored)),
    }
    for label, (stated, actual) in claims.items():
        log(fh, f"{'OK ' if stated == actual else 'DIFF'} {label}: stated {stated}, measured {actual}")
    log(fh, f"wheel version: {tsecon.__version__}; paper says 0.8.0: {'0.8.0' in md}")
    m = re.search(r"^date:\s*(.*)$", md, re.M)
    log(fh, "paper date field:", m.group(1) if m else None, "(the YAML comment says it predates the artifact)")
    fh.close()


if __name__ == "__main__":
    main()
