"""Offline regression guard for the Uhlig (2005) monetary-policy replication.

Runs the replication's estimation against the committed Uhlig panel
(fixtures/uhlig2005.csv) so the published-result claims on the docs page
cannot silently rot. Fully offline — the data is vendored, the library ships
no loaders.

Draw budget: the docs script uses 2000 accepted draws; this test uses 300
with a fixed seed to keep runtime under ~2s. At 300 draws the 16/50/84
quantiles at the pinned horizons move by only ~0.03-0.05 across seeds
(measured over seeds 0, 1, 7, 42), so every tolerance below carries at
least a 2x margin over the observed seed-to-seed drift.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "examples"))

uh_repl = pytest.importorskip("replication_uhlig_monetary")

N_DRAWS = 300          # reduced from the script's 2000; seeded, see module docstring
KEY_HORIZONS = (6, 12, 24, 36, 48, 60)


@pytest.fixture(scope="module")
def uhlig_run():
    uh = uh_repl.load_uhlig()
    r = uh_repl.run_uhlig(uh["data"], n_draws=N_DRAWS, seed=0)
    return uh, r


def test_dataset_is_the_committed_uhlig_panel():
    uh = uh_repl.load_uhlig()
    data = uh["data"]
    assert data.shape == (468, 6)                       # 1965:1-2003:12, monthly
    assert uh["names"] == ["y", "yd", "p", "i", "rnb", "rt"]
    assert uh["dates"][0] == "1965-01" and uh["dates"][-1] == "2003-12"
    assert np.all(np.isfinite(data))
    # fed funds is in percent (levels), everything else is 100*log (levels)
    i = data[:, 3]
    assert 0.5 < i.min() and i.max() < 20.0
    for col in (0, 1, 2, 4, 5):
        assert data[:, col].min() > 200.0


def test_restrictions_are_uhligs_benchmark_set():
    restr = uh_repl.monetary_policy_restrictions()
    assert len(restr) == 24                             # 4 variables x horizons 0..5
    assert {h for (_, _, h, _) in restr} == set(range(6))       # K = 5
    assert {s for (_, s, _, _) in restr} == {uh_repl.SHOCK}     # one shock
    by_var = {v: sg for (v, _, _, sg) in restr}
    assert by_var == {1: "-", 2: "-", 4: "-", 3: "+"}   # yd, p, rnb down; ff up
    assert 0 not in by_var and 5 not in by_var          # GDP, total reserves free


def test_sampler_accepts_and_enforces_the_signs(uhlig_run):
    _, r = uhlig_run
    d = r["diagnostics"]
    assert d["accepted"] >= 0.9 * N_DRAWS               # observed: 300/300
    assert 0.02 < d["acceptance_rate"] < 0.20           # observed: ~0.05-0.07
    # every accepted draw satisfies the signs strictly, so the identified-set
    # envelope of each restricted cell sits strictly on the required side
    smin, smax = np.asarray(r["set_min"]), np.asarray(r["set_max"])
    j = uh_repl.SHOCK
    for h in range(6):
        assert smax[h, 1, j] < 0 and smax[h, 2, j] < 0 and smax[h, 4, j] < 0
        assert smin[h, 3, j] > 0


def test_no_price_puzzle(uhlig_run):
    """Uhlig's claim (a): the deflator does not rise — it falls only slowly.

    The restrictions force deflator <= 0 only for months 0..5; the pinned
    finding is that its 84% quantile stays negative through month 60
    (observed max -0.013), and the median decline is gradual but material.
    """
    _, r = uhlig_run
    d84 = uh_repl.quantile(r, "yd", 3)
    assert np.all(d84 < 0.0)                            # every horizon 0..60
    d50 = uh_repl.quantile(r, "yd", 2)
    assert -0.10 < d50[0] < 0.0                         # slow: small at impact
    assert -0.70 < d50[60] < -0.20                      # falling: ~-0.4% by h=60
    # the commodity-price response shows the same absence of a puzzle
    p84 = uh_repl.quantile(r, "p", 3)
    assert np.all(p84[: 48 + 1] < 0.0)


def test_gdp_response_is_ambiguous(uhlig_run):
    """Uhlig's claim (b): the 68% band on real GDP straddles zero, and 'with
    a 2/3 probability, a typical shock will move real GDP by up to 0.2
    percent'. Pinned for months 6..60 (in months 0..5 the band's lower edge
    grazes zero from below; the paper's ambiguity claim is about the medium
    run). Magnitude tolerance 0.35% vs the paper's ~0.2%: generous, but the
    band must be neither degenerate nor unbounded."""
    _, r = uhlig_run
    y16 = uh_repl.quantile(r, "y", 1)
    y84 = uh_repl.quantile(r, "y", 3)
    for h in range(6, 61):
        assert y16[h] < 0.0 < y84[h]                    # the straddle IS the finding
    assert np.max(np.abs(y16[6:])) < 0.35               # observed: <= 0.17
    assert np.max(y84[6:]) < 0.35                       # observed: <= 0.24
    # ... and the ambiguity is substantive, not a hairline band around zero
    half_width = np.maximum(-y16[6:], y84[6:])
    assert np.min(half_width) > 0.05                    # observed: >= 0.14


def test_identified_shock_looks_like_monetary_policy(uhlig_run):
    _, r = uhlig_run
    ff50 = uh_repl.quantile(r, "i", 2)
    assert 0.10 < ff50[0] < 0.35                        # ~+0.2pp impact (observed 0.19-0.23)
    assert ff50[0] > ff50[24]                           # and it decays
    rnb84 = uh_repl.quantile(r, "rnb", 3)
    assert np.all(rnb84[: 12 + 1] < 0.0)                # liquidity effect persists


def test_quantiles_are_stable_across_seeds(uhlig_run):
    """The docs-page table must not be a seed artifact: at 300 draws the
    16/50/84 quantiles for GDP, the deflator and the funds rate move by
    ~0.03-0.05 across seeds at the pinned horizons; 0.15 is a generous cap."""
    uh, r0 = uhlig_run
    r1 = uh_repl.run_uhlig(uh["data"], n_draws=N_DRAWS, seed=1)
    q0, q1 = np.asarray(r0["quantiles"]), np.asarray(r1["quantiles"])
    j = uh_repl.SHOCK
    for var in (0, 1, 3):                               # y, yd, i
        for h in KEY_HORIZONS:
            drift = np.abs(q0[h, var, j, 1:4] - q1[h, var, j, 1:4])
            assert np.max(drift) < 0.15


def test_seeded_run_is_bit_reproducible():
    uh = uh_repl.load_uhlig()
    kw = dict(n_draws=60, seed=11, horizon=12)
    a = uh_repl.run_uhlig(uh["data"], **kw)
    b = uh_repl.run_uhlig(uh["data"], **kw)
    assert np.array_equal(np.asarray(a["quantiles"]), np.asarray(b["quantiles"]))
    assert a["diagnostics"] == b["diagnostics"]
