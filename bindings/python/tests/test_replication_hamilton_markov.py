"""Offline regression guard for the Hamilton (1989) Markov-switching replication.

Runs the replication's estimation against the committed GNP-growth series
(fixtures/hamilton_gnp.csv) so the published-result claims on the docs page
cannot silently rot. Fully offline - the data is vendored, the library ships
no loaders.

Tolerance rationale (stated per family, inherited by every assertion):

* Published values are Table I's printed digits: p = 0.9049 / q = 0.7550
  exactly as printed; the means at the two-decimal headline (+1.16 / -0.36);
  sigma^2 at the E-views/statsmodels re-estimation precision 0.5914 (Hamilton
  prints sigma = 0.769). Printed-digit rounding alone justifies ~0.005.
* tsecon fits by EM whose transition M-step conditions on the stationary
  initial distribution instead of differentiating it w.r.t. P, so its fixed
  point sits O(1/T) from the exact MLE: on this sample <= 0.016 on any
  parameter and 0.006 log-likelihood points (measured; see the doc page).
  Published-vs-tsecon tolerances are the sum of the two effects with modest
  headroom (0.02-0.03); tsecon-vs-statsmodels tolerances cover only the
  EM-vs-MLE gap (0.005-0.02). Achieved deviations are noted inline.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "examples"))

ham = pytest.importorskip("replication_hamilton_markov")


@pytest.fixture(scope="module")
def data():
    return ham.load_hamilton_gnp()


@pytest.fixture(scope="module")
def ts_fit(data):
    return ham.fit_tsecon(data["growth"])


def test_dataset_is_hamiltons_series(data):
    # 135 growth quarters 1951Q2-1984Q4 -> Hamilton's estimation sample
    # 1952Q2-1984Q4 (n = 131) after AR(4) conditioning.
    assert len(data["growth"]) == 135
    assert data["quarter"][0] == "1951Q2"
    assert data["quarter"][4] == "1952Q2"
    assert data["quarter"][-1] == "1984Q4"
    # exact first/last values of the vendored series (verbatim copy guard)
    assert data["growth"][0] == pytest.approx(2.59316421, abs=1e-12)
    assert data["growth"][-1] == pytest.approx(0.14802167, abs=1e-12)
    # the NBER indicator marks 7 recession episodes in the effective sample
    assert len(ham.nber_episodes(data["nber"][4:])) == 7


def test_tsecon_replicates_hamilton_table_1(ts_fit):
    assert ts_fit["converged"]
    # means (published two-decimal headline; achieved 0.005 / 0.017)
    assert ts_fit["mu_expansion"] == pytest.approx(1.16, abs=0.02)
    assert ts_fit["mu_contraction"] == pytest.approx(-0.36, abs=0.03)
    # transition persistences (Table I printed digits; achieved 0.003 / 0.008)
    assert ts_fit["p_expansion_stay"] == pytest.approx(0.9049, abs=0.02)
    assert ts_fit["p_contraction_stay"] == pytest.approx(0.7550, abs=0.02)
    # common innovation variance (achieved 0.003)
    assert ts_fit["sigma2"] == pytest.approx(0.5914, abs=0.02)
    # Hamilton's headline durations: ~10-quarter expansions, ~4-quarter recessions
    assert 9.0 < ts_fit["duration_expansion"] < 11.5
    assert 3.5 < ts_fit["duration_contraction"] < 5.0
    # EM fixed point vs the exact MLE -181.2634 (achieved gap 0.006)
    assert ts_fit["loglik"] == pytest.approx(-181.263, abs=0.05)


def test_smoothed_probabilities_recover_nber_dating(data, ts_fit):
    rec = data["nber"][4:].astype(bool)
    prob = ts_fit["prob_contraction"]
    assert prob.shape == rec.shape  # n = 131
    classified = prob > 0.5
    # doc-page claim: 120/131 = 91.6% agreement; guard just below it
    assert (classified == rec).mean() >= 0.90
    # every NBER recession is detected with high smoothed probability
    for s0, e0 in ham.nber_episodes(rec.astype(int)):
        assert prob[s0:e0 + 1].max() > 0.9  # achieved minimum 0.936 (1960-61)


def test_binding_exposes_the_common_ar_block():
    """Until this release the binding estimated the common AR(4) internally
    but never returned it (a guard here pinned that gap and instructed its
    own replacement). `ar` is the length-`order` common block, shared across
    regimes and hence label-free - no regime reordering applies to it."""
    import tsecon

    y = ham.load_hamilton_gnp()["growth"]
    r = tsecon.markov_switching_ar(y, k_regimes=2, order=4,
                                   switching_variance=False, max_iter=10)
    ar = np.asarray(r["ar"])
    assert ar.shape == (4,)
    assert np.all(np.isfinite(ar))


def test_tsecon_replicates_hamiltons_ar_coefficients(ts_fit):
    """The comparison the retired guard demanded: tsecon's common AR(4)
    against Hamilton (1989) Table I at the E-views/statsmodels re-estimation
    precision (0.014, -0.058, -0.247, -0.213). Hamilton's phis are
    notoriously optimizer-sensitive; the published-digit + EM-vs-MLE budget
    (0.02) from the module docstring applies. Achieved max |diff| 0.0048
    (phi_2), comfortably inside it."""
    assert np.allclose(ts_fit["ar"], ham.PUBLISHED["ar"], atol=0.02)


# ---------------------------------------------------------------- dual golden
sm = pytest.importorskip("statsmodels")


@pytest.fixture(scope="module")
def sm_fit(data):
    return ham.fit_statsmodels(data["growth"])


def test_statsmodels_reaches_the_eviews_benchmark(sm_fit):
    """Anchors the cross-check itself: statsmodels' MLE on the committed CSV
    must match the E-views SWITCHREG benchmark stored in statsmodels' own
    test suite - proving the fixture *is* Hamilton's series."""
    assert sm_fit["mu_contraction"] == pytest.approx(-0.358811, abs=1e-3)
    assert sm_fit["mu_expansion"] == pytest.approx(1.163516, abs=1e-3)
    assert sm_fit["p_contraction_stay"] == pytest.approx(0.754673, abs=1e-3)
    assert sm_fit["sigma2"] == pytest.approx(0.591364, abs=1e-3)
    assert np.allclose(sm_fit["ar"],
                       [0.013486, -0.057521, -0.246983, -0.212923], atol=1e-3)
    assert sm_fit["loglik"] == pytest.approx(-181.26339, abs=1e-3)


def test_tsecon_matches_statsmodels_on_identical_data(ts_fit, sm_fit):
    # EM-fixed-point vs exact-MLE tolerances (achieved: 0.016 worst case)
    assert ts_fit["mu_contraction"] == pytest.approx(sm_fit["mu_contraction"], abs=0.02)
    assert ts_fit["mu_expansion"] == pytest.approx(sm_fit["mu_expansion"], abs=0.02)
    assert ts_fit["p_contraction_stay"] == pytest.approx(
        sm_fit["p_contraction_stay"], abs=0.01)  # achieved 0.008
    assert ts_fit["p_expansion_stay"] == pytest.approx(
        sm_fit["p_expansion_stay"], abs=0.01)    # achieved 0.002
    assert ts_fit["sigma2"] == pytest.approx(sm_fit["sigma2"], abs=0.01)
    # common AR(4): achieved max |diff| 0.0043, phi_2 (EM-vs-MLE gap only)
    np.testing.assert_allclose(ts_fit["ar"], sm_fit["ar"], atol=0.02)
    assert ts_fit["loglik"] == pytest.approx(sm_fit["loglik"], abs=0.02)
    # smoothed recession paths: achieved max |diff| 0.033, corr 0.9998
    diff = np.max(np.abs(ts_fit["prob_contraction"] - sm_fit["prob_contraction"]))
    assert diff < 0.05
    # and the two packages call every quarter identically (achieved: exact)
    assert np.array_equal(ts_fit["prob_contraction"] > 0.5,
                          sm_fit["prob_contraction"] > 0.5)
