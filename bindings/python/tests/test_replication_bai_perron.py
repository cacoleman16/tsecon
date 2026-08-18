"""Offline regression guard for the Bai-Perron (2003) real-interest-rate replication.

Runs `bai_perron` against the committed RealInt series
(fixtures/realint_bai_perron.csv) and pins the published break dates EXACTLY,
the segment means at published rounding, and the CI/test-statistic behavior at
honest, construction-aware tolerances — so the docs page's claims cannot
silently rot. Fully offline: the data is vendored with attribution, the
library ships no loaders.

Published anchors (mean-shift model, trim 0.15; Bai & Perron 2003, JAE 18(1),
Section 4; corroborated by the Zeileis-Kleiber 2005 JAE validation study and
by Perron's own mbreaks R package run with the paper's settings):
  - global partitions: m=2 -> 1972:3, 1980:3;  m=3 -> 1966:4, 1972:3, 1980:3
  - BIC/LWZ select 2 breaks; the paper's HAC-robust sequential supF selects 3
    (tsecon's supF is classical, so its sequential procedure stops at 2 —
    the same count as the information criteria; asserted below)
  - two-break means 1.36, -1.80, 5.64; three-break means 1.82, 0.87, -1.80, 5.64
  - the 1980:3 break is sharply dated, the 1972:3 break is not

Cross-implementation anchors computed on this identical series (see the docs
page): R strucchange RSS path 1214.9219 / 644.9955 / 455.9502 / 445.1819 and
coefficients 1.355037 / -1.796138 / 5.642890; mbreaks classical supF sequence
89.245 / 52.204 / 7.414.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "examples"))

repl = pytest.importorskip("replication_bai_perron_realint")


@pytest.fixture(scope="module")
def fitted():
    quarter, rate = repl.load_realint()
    return quarter, rate, repl.run(rate)


def test_dataset_is_the_realint_series(fitted):
    quarter, rate, _ = fitted
    # 103 quarters, 1961Q1-1986Q3, values as distributed in strucchange/mbreaks
    assert len(rate) == 103
    assert quarter[0] == 1961.00
    assert quarter[-1] == 1986.50
    assert rate[0] == pytest.approx(1.99132, abs=1e-9)
    assert rate[-1] == pytest.approx(4.30529, abs=1e-9)
    assert repl.qlabel(0) == "1961Q1"
    assert repl.qlabel(102) == "1986Q3"


def test_break_dates_match_published_exactly(fitted):
    _, _, bp = fitted
    # The published partitions, as 0-indexed last-observation-of-regime dates:
    # 1972Q3 = obs 46, 1980Q3 = obs 78, 1966Q4 = obs 23.
    assert [int(d) for d in bp["break_dates_by_m"][0]] == [78]
    assert [int(d) for d in bp["break_dates_by_m"][1]] == [46, 78]
    assert [int(d) for d in bp["break_dates_by_m"][2]] == [23, 46, 78]
    assert [repl.qlabel(int(d)) for d in bp["break_dates"]] == ["1972Q3", "1980Q3"]
    assert [repl.qlabel(int(d)) for d in bp["break_dates_by_m"][2]] == [
        "1966Q4", "1972Q3", "1980Q3"]


def test_selection_is_two_breaks_with_classical_f(fitted):
    _, _, bp = fitted
    # tsecon's supF is classical (not the paper's HAC variant): the sequential
    # procedure at the published 5% critical values stops at 2 breaks — the
    # count BP's BIC and LWZ select. The paper's HAC sequence (57.91, 33.93,
    # 14.73) selects 3; that variant is out of tsecon's scope and NOT asserted.
    assert bp["n_breaks"] == 2
    seq = np.asarray(bp["sup_f_seq"])
    crit = np.asarray(bp["sup_f_crit"])
    # published Bai-Perron 5% critical values for q=1, trim=0.15 — verbatim
    assert crit[:3] == pytest.approx([8.58, 10.13, 11.14], abs=1e-9)
    # classical sequence matches Perron's own mbreaks with robust=0 (3 d.p.)
    assert seq[0] == pytest.approx(89.245, abs=2e-3)
    assert seq[1] == pytest.approx(52.204, abs=2e-3)
    assert seq[2] == pytest.approx(7.414, abs=2e-3)
    # and the selection is the sequence's own logic: reject, reject, stop
    assert seq[0] > crit[0] and seq[1] > crit[1] and seq[2] < crit[2]


def test_ssr_path_matches_strucchange(fitted):
    _, _, bp = fitted
    ssr = np.asarray(bp["ssr_path"])
    # R strucchange RSS for the globally optimal 0/1/2/3-break partitions
    assert ssr[:4] == pytest.approx(
        [1214.9219, 644.9955, 455.9502, 445.1819], abs=1e-3)
    # (m = 4, 5 use tsecon's h=16 admissible set vs the R packages' h=15 and
    # are legitimately different — not pinned.)


def test_segment_means_match_published(fitted):
    _, rate, bp = fitted
    params = np.asarray(bp["params"])[:, 0]
    # exact identity: per-regime OLS on an intercept is the segment average
    seg = [rate[0:47].mean(), rate[47:79].mean(), rate[79:103].mean()]
    assert params == pytest.approx(seg, abs=1e-10)
    # published two-break means, at half a printed rounding unit
    assert params == pytest.approx([1.36, -1.80, 5.64], abs=5e-3)
    # strucchange's coefficients on the identical series
    assert params == pytest.approx([1.355037, -1.796138, 5.642890], abs=1e-5)
    # published three-break means, from the m=3 partition the DP found
    m3 = [rate[0:24].mean(), rate[24:47].mean(), rate[47:79].mean(),
          rate[79:103].mean()]
    assert m3 == pytest.approx([1.82, 0.87, -1.80, 5.64], abs=5e-3)
    # SEs are NOT compared to the published HAC ones (different construction);
    # sanity only: positive and of the right order
    bse = np.asarray(bp["bse"])[:, 0]
    assert np.all(bse > 0.05) and np.all(bse < 1.0)


def test_break_date_confidence_intervals(fitted):
    _, _, bp = fitted
    lo90 = [int(v) for v in bp["ci_lower_90"]]
    hi90 = [int(v) for v in bp["ci_upper_90"]]
    lo95 = [int(v) for v in bp["ci_lower_95"]]
    hi95 = [int(v) for v in bp["ci_upper_95"]]

    # Regression guard on tsecon's own homogeneous classical Bai (1997) CIs
    # (the shipped construction — the paper's heterogeneity-robust HAC CIs are
    # a different estimator and are NOT asserted; see the model card and docs
    # page). 0-indexed observation bounds:
    assert (lo90, hi90) == ([41, 76], [51, 80])   # 1971Q2-1973Q4, 1980Q1-1981Q1
    assert (lo95, hi95) == ([39, 76], [53, 80])   # 1970Q4-1974Q2, 1980Q1-1981Q1

    # The published qualitative finding that DOES replicate across every
    # implementation: the 1980:3 Volcker break is sharply dated, the 1972:3
    # break is not.
    span_1972 = hi95[0] - lo95[0] + 1
    span_1980 = hi95[1] - lo95[1] + 1
    assert span_1980 <= 6
    assert span_1972 >= 2 * span_1980
    # each CI covers its point date, and 90% nests inside 95%
    assert lo95[0] <= 46 <= hi95[0] and lo95[1] <= 78 <= hi95[1]
    assert lo95[0] <= lo90[0] and hi90[0] <= hi95[0]
    # tsecon's 95% CI for the 1972Q3 break overlaps BP's published 95% CI
    # (1970:3-1972:4 = obs 38..47, three-break model, HAC-robust construction)
    assert max(lo95[0], 38) <= min(hi95[0], 47)


def test_supf_test_agrees_with_first_sequential_stage(fitted):
    """The Andrews sup-F entry point and bai_perron's first stage are the same
    statistic; on this data the one-break argmax is the Volcker break."""
    import tsecon

    _, rate, bp = fitted
    sf = tsecon.sup_f_test(rate, np.ones((len(rate), 1)), trim=0.15)
    assert sf["stat"] == pytest.approx(float(np.asarray(bp["sup_f_seq"])[0]),
                                       rel=1e-10)
    assert sf["p_value"] < 1e-6
    assert int(sf["break_date"]) == 78    # 1980Q3
