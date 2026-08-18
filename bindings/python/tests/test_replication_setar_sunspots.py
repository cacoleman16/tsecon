"""Offline regression guard for the Hansen (1999) sunspot SETAR replication.

Runs the replication's estimation against the committed 1700-1988 Wolf
sunspot sample (fixtures/sunspots_tong.csv) so the published-result claims on
the docs page cannot silently rot. Fully offline — the data is vendored, the
library ships no loaders.

What is pinned, and how hard:

* published quantities, at published precision — Hansen reports the threshold
  as 7.4 (two digits) and the delay as 2; a threshold is identified only up to
  the gap between adjacent order statistics of the threshold variable, so the
  honest published-value assertion is "rounds to 7.4" plus "the raw-count gap
  is [21.2, 21.3)", not a 1e-10 equality with a two-digit table entry.
* the linearity verdict — the paper reports p ~ 0.03 for the sunspot series;
  the seeded homoskedastic residual bootstrap here must reject at 5%.
* tsecon's own values, tightly — threshold/SSR regression pins at 1e-10 so a
  refactor cannot drift the fit while the round-to-published checks still pass.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "examples"))

sun_repl = pytest.importorskip("replication_setar_sunspots")


def _transformed():
    years, counts = sun_repl.load_sunspots()
    return years, counts, sun_repl.transform(counts)


def test_dataset_is_the_committed_1700_1988_sample():
    years, counts, _ = _transformed()
    assert len(years) == 289
    assert years[0] == 1700 and years[-1] == 1988
    # endpoint and extreme values of the classic Wolf/Zurich annual series
    assert counts[0] == 5.0
    assert counts[-1] == 100.2
    assert counts.max() == 190.2 and years[counts.argmax()] == 1957
    assert np.all(counts >= 0.0)


def test_fixture_matches_the_statsmodels_bundled_series():
    """Provenance guard: the vendored CSV is exactly the statsmodels sunspot
    series truncated to 1700-1988 (the sample of Tong 1990, Appendix 3)."""
    sm_datasets = pytest.importorskip("statsmodels.api").datasets
    d = sm_datasets.sunspots.load_pandas().data
    yr = d["YEAR"].values.astype(int)
    keep = (yr >= 1700) & (yr <= 1988)
    years, counts, _ = _transformed()
    np.testing.assert_array_equal(years, yr[keep])
    np.testing.assert_array_equal(counts, d["SUNACTIVITY"].values[keep])


def test_setar2_replicates_hansen_threshold_and_delay():
    _, _, y = _transformed()
    r = sun_repl.fit_hansen_setar(y)

    # Published: delay d = 2 (the joint search over d in {1, 2} picks it).
    assert r["delay"] == 2
    # Published: threshold 7.4 on the 2(sqrt(1+N)-1) scale (two digits).
    assert round(r["threshold"], 1) == 7.4
    # The identified raw-count order-statistic gap is [21.2, 21.3): both ends
    # print as 7.4; any value in the gap is the same fitted model.
    assert sun_repl.to_raw(r["threshold"]) == pytest.approx(21.2, abs=1e-8)
    grid = np.asarray(r["thresholds"])
    nxt = grid[int(np.searchsorted(grid, r["threshold"])) + 1]
    assert sun_repl.to_raw(nxt) == pytest.approx(21.3, abs=1e-8)

    # Sample bookkeeping: 289 years minus 11 lags, both regimes well inside
    # the 10% trim.
    assert r["nobs"] == 278
    assert r["n_low"] == 86 and r["n_high"] == 192

    # tsecon regression pins (not published values): keep the fit from
    # drifting while the round-to-7.4 assertions above still pass.
    assert r["threshold"] == pytest.approx(7.423375191511797, rel=1e-10)
    assert r["ssr"] == pytest.approx(907.2374115912764, rel=1e-8)


def test_delay_2_also_wins_an_unrestricted_delay_search():
    """Hansen searched d in {1, 2}; the result is not an artifact of that
    restriction — d = 2 also wins over d = 1..11 on this data."""
    import tsecon

    _, _, y = _transformed()
    r = tsecon.setar(y, p=11, delays=list(range(1, 12)), trim=0.10)
    assert r["delay"] == 2
    assert round(r["threshold"], 1) == 7.4


def test_linearity_is_rejected_like_the_paper():
    _, _, y = _transformed()
    t2 = sun_repl.linearity_test(y, delay=2)
    t1 = sun_repl.linearity_test(y, delay=1)
    # The F12 profile peaks at the published delay...
    assert t2["stat"] > t1["stat"]
    assert t2["stat"] == pytest.approx(69.74744458766942, rel=1e-8)
    # ...and the seeded bootstrap rejects at 5%, as the paper's ~0.03 does.
    # (tsecon's is the homoskedastic residual bootstrap; the paper's exact
    # p-value depends on its bootstrap variant, so the pin is the verdict.)
    assert t2["p_value"] <= 0.05
    assert t1["p_value"] > 0.05  # the wrong delay does not reject


def test_information_criteria_order_like_the_tsdyn_replication():
    """tsDyn's executed replication of the same example has AIC preferring
    SETAR(2) over AR(11) and BIC the reverse; the same ordering must hold
    under tsecon's shared n*ln(SSR/n) + penalty convention."""
    _, _, y = _transformed()
    r = sun_repl.fit_hansen_setar(y)
    t = sun_repl.linearity_test(y, delay=2)
    n, s0 = t["nobs"], t["ssr_linear"]
    k_lin = 12
    aic_lin = n * np.log(s0 / n) + 2 * k_lin
    bic_lin = n * np.log(s0 / n) + k_lin * np.log(n)
    assert r["aic"] < aic_lin
    assert r["bic"] > bic_lin
