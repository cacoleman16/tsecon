"""Offline regression guard for the Ramey-Zubairy government-spending replication.

Runs the replication's estimation against the committed RZ panel
(fixtures/ramey_zubairy.csv) so the published-result claim on the docs page
cannot silently rot. Fully offline — the data is vendored, the library ships no
loaders.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "examples"))

rz_repl = pytest.importorskip("replication_ramey_zubairy")


def _full_sample_multiplier():
    rz = rz_repl.load_ramey_zubairy()
    g, y, newsy = rz_repl.build_variables(rz)
    complete = ~np.isnan(g + y + newsy)
    r = rz_repl.integral_multiplier(g[complete], y[complete], newsy[complete])
    return rz, r


def test_dataset_is_the_full_committed_panel():
    rz = rz_repl.load_ramey_zubairy()
    assert len(rz["quarter"]) == 564
    assert {"news", "ngov", "rgdp", "pgdp", "rgdp_potcbo"} <= set(rz["names"])
    assert rz["quarter"][0] == 1875.0
    # early quarters predate the macro series -> nan, not zero
    assert np.isnan(rz["series"]["ngdp"][0])


def test_integral_multiplier_replicates_ramey_zubairy():
    _, r = _full_sample_multiplier()
    mult = np.asarray(r["multiplier"])
    # RZ's headline: the integral multiplier is below one across horizons.
    band = mult[[4, 8, 12, 16, 20]]
    assert np.all(band > 0.5)
    assert np.all(band < 0.8)          # inside the published 0.6-0.8 neighbourhood
    # a strong cumulated first stage, unlike the outcome-only trap (F ~ 1.7)
    assert np.asarray(r["first_stage_f"])[8] > 5.0


def test_multiplier_is_not_the_outcome_only_trap():
    """lp_iv(..., cumulative=True) cumulates only the outcome and is NOT a
    multiplier: on this data it runs away with the horizon. The guard keeps the
    replication pointed at lp_multiplier, not at the trap."""
    import tsecon

    rz = rz_repl.load_ramey_zubairy()
    g, y, newsy = rz_repl.build_variables(rz)
    ok = ~np.isnan(g + y + newsy)
    trap = np.asarray(
        tsecon.lp_iv(y[ok], g[ok], newsy[ok], horizons=20, n_lag_controls=4,
                     cumulative=True)["irf"]
    )
    good = np.asarray(rz_repl.integral_multiplier(g[ok], y[ok], newsy[ok])["multiplier"])
    assert trap[20] > 10 * good[20]    # the trap diverges; the multiplier does not
