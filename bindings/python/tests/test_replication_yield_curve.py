"""Offline regression guard for the Estrella-Mishkin yield-curve replication.

Runs the replication's estimation logic against the committed FRED snapshot
(fixtures/yield_curve_recession.csv), so the published-result claim on the docs
page cannot silently rot. No network: the fixture is real FRED data captured
once and vendored (FRED data is redistributable with attribution).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "fixtures" / "yield_curve_recession.csv"
sys.path.insert(0, str(REPO / "docs" / "examples"))

replication = pytest.importorskip("replication_yield_curve_recession")


def test_term_spread_predicts_recessions():
    dates, spread, recession = replication.load_aligned(FIXTURE)
    fit, b, z = replication.run(dates, spread, recession)

    # The signature Estrella-Mishkin result: the spread coefficient is negative
    # and strongly significant — an inverting curve raises recession probability.
    assert b[1] < 0.0
    assert z[1] < -5.0
    assert fit["pseudo_r2"] > 0.1

    # Economic magnitude: an inverted (-1pp) curve implies a far higher
    # 12-month-ahead recession probability than a steep (+3pp) one.
    from math import erf

    def phi(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    p_inverted = phi(b[0] + b[1] * -1.0)
    p_steep = phi(b[0] + b[1] * 3.0)
    assert p_inverted > 0.3
    assert p_steep < 0.05
    assert p_inverted > 5 * p_steep


def test_fixture_is_the_real_monthly_panel():
    dates, spread, recession = replication.load_aligned(FIXTURE)
    assert len(dates) > 800  # decades of monthly data
    assert dates[0] == np.datetime64("1953-04-01")
    assert set(np.unique(recession)) == {0.0, 1.0}
    assert recession.sum() > 0  # spans real recessions
