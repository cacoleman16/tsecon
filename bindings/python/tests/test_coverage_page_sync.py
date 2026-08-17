"""The published interval-coverage tables must match the runner's registry.

Audit round 2, finding 5: `run_all.py` harvests 40 rows and the published page
carried 39 — the `ols se_type="hc3"` row (the estimator 0.2.0 added because of
the audit) had been silently dropped in transcription, while
docs/reference/testing.md promised that could not happen. The Monte Carlo
numbers cannot be re-derived in a unit test, but the page's row *inventory*
can be checked statically, and that is precisely the failure that occurred.

This test just runs docs/examples/coverage/check_page.py, which parses
Tables 1 and 2 out of the page and compares them row-for-row against the
probe registry in run_all.py. It needs no tsecon import and no simulation.
Mutation-tested: deleting the hc3 row from Table 1 makes it fail with
"registered probe has NO row: ('tsecon.ols', 'se_type=\"hc3\"; ...')".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CHECKER = REPO / "docs" / "examples" / "coverage" / "check_page.py"
PAGE = REPO / "docs" / "examples" / "interval-coverage.md"


def test_coverage_page_matches_probe_registry():
    if not CHECKER.exists() or not PAGE.exists():
        pytest.skip("docs tree not present in this checkout")
    proc = subprocess.run(
        [sys.executable, str(CHECKER)], capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        "interval-coverage.md is out of sync with run_all.py's probe "
        f"registry:\n{proc.stdout}{proc.stderr}"
    )
