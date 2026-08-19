"""Offline regression guard for the Gertler-Karadi (2015) proxy-SVAR
replication.

Runs the replication's estimation against the committed GK panel
(fixtures/gertler_karadi.csv) so the published-result claims on the docs
page cannot silently rot. Fully offline -- the data is vendored, the library
ships no loaders.

What is pinned, and how tightly:

* the dataset vintage (shape, dates, availability windows, unit sanity);
* the paper's first-stage numbers VERBATIM -- classical F 21.55 and robust
  F 17.5 are printed values from the paper, so they are pinned at print
  precision (abs 0.05);
* the Figure-1 shape facts stated in the paper's text (signs, trough
  timing, reversion), at honest tolerances;
* the wild-vs-moving-block significance contrast (the Jentsch-Lunsford
  correction measured on this data) at a fixed seed;
* the post-1984 weakening: effective F falls, certified worst-case bias
  rises, AR sets widen, the output response loses all significance while
  the EBP impact response survives.

Draw budget: the docs script uses 2000 bootstrap draws; these tests use 500
with a fixed seed (the significance-horizon sets at 500/2000 draws differ
by at most a couple of horizons at the pinned edges, and every assertion
below carries margin over that drift).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "examples"))

gk_repl = pytest.importorskip("replication_gertler_karadi")
tsecon = pytest.importorskip("tsecon")

N_BOOT = 500


@pytest.fixture(scope="module")
def gk():
    return gk_repl.load_gk()


@pytest.fixture(scope="module")
def baseline(gk):
    proxy = gk_repl.baseline_proxy(gk)
    pr = tsecon.proxy_svar(gk["data"], proxy, lags=12, horizon=48,
                           norm_var=0, unit=0.2)
    return gk, proxy, pr


def test_dataset_is_the_committed_gk_panel(gk):
    data, dates = gk["data"], gk["dates"]
    assert data.shape == (396, 4)                       # 1979:7-2012:6, monthly
    assert dates[0] == "1979-07" and dates[-1] == "2012-06"
    assert np.all(np.isfinite(data))
    gs1, logcpi, logip, ebp = data.T
    assert 0.05 <= gs1.min() and gs1.max() < 18.0       # percent: ZLB 0.10 to Volcker ~16.7
    assert 420.0 < logcpi.min() and logcpi.max() < 550.0  # 100*log CPI
    assert 380.0 < logip.min() and logip.max() < 480.0    # 100*log IP
    assert -1.5 < ebp.min() and ebp.max() < 4.0         # percentage points
    # FF4 is available 1990:1-2012:6 (270 months), the baseline masks to 1991:1.
    ff4 = gk["instruments"]["ff4_tc"]
    finite = np.isfinite(ff4)
    assert finite.sum() == 270
    assert dates[int(np.argmax(finite))] == "1990-01"
    proxy = gk_repl.baseline_proxy(gk)
    assert np.isfinite(proxy).sum() == 258
    assert dates[int(np.argmax(np.isfinite(proxy)))] == "1991-01"


def test_first_stage_matches_the_published_numbers_verbatim(baseline):
    _, _, pr = baseline
    fs = pr["first_stage"]
    # The paper's printed first-stage values for FF4 on the one-year rate.
    assert fs["f_classical"] == pytest.approx(21.55, abs=0.05)
    assert fs["effective_f"] == pytest.approx(17.50, abs=0.05)
    assert fs["f_hc1"] == fs["effective_f"] == pr["first_stage_f"]
    assert fs["n_proxy"] == 258
    # The modern verdict: passes the folklore bar, fails the MOP tau=10% bar.
    assert not fs["weak_folklore"]
    assert fs["weak_mop_tau10"]
    assert fs["mop_cv_tau10"] == pytest.approx(23.109, abs=5e-3)
    assert fs["tau_bound"] == pytest.approx(0.155, abs=0.01)


def test_irf_matches_the_papers_stated_shapes(baseline):
    _, _, pr = baseline
    irf = np.asarray(pr["irf"])                          # (49, 4)
    gs1, cpi, ip, ebp = irf[:, 0], irf[:, 1], irf[:, 2], irf[:, 3]
    # Normalization: +20bp on the 1-year rate at h=0, exactly.
    assert gs1[0] == 0.2
    # "reverts back to trend after roughly a year"
    assert gs1[12] < 0.10 and gs1[18] < 0.02
    # IP: "drop ... begins after several months", trough "after roughly 18
    # months". Observed: negative from h=4, trough -0.43 at h=25 with an
    # 18..30 plateau.
    assert ip[0] > -0.10                                # no impact collapse
    assert np.all(ip[4:31] < 0.0)
    trough = int(ip.argmin())
    assert 18 <= trough <= 30
    assert -0.65 < ip[trough] < -0.25
    assert ip[18] - ip[trough] < 0.10                   # plateau, not a spike
    # CPI: "declines steadily, though this decline is not significant"
    assert np.all(cpi[:49] < 0.05)
    assert -0.30 < cpi[48] < -0.03
    # EBP: rises on impact ("+8bp" for the paper's ~1-sd surprise; +12bp for
    # the exact +20bp normalization here), stays positive for months.
    assert 0.05 < ebp[0] < 0.20
    assert np.all(ebp[:9] > 0.0)


def test_wild_reproduces_gk_but_valid_mbb_undoes_ip_significance(baseline):
    gk, proxy, _ = baseline
    kw = dict(lags=12, horizon=48, norm_var=0, unit=0.2, alpha=0.05,
              n_boot=N_BOOT, seed=0)
    wild = tsecon.proxy_svar_bands(gk["data"], proxy, bands="wild", **kw)
    mbb = tsecon.proxy_svar_bands(gk["data"], proxy, bands="moving_block", **kw)
    # The wild path must self-report its invalidity; the MBB must not.
    assert wild["asymptotically_valid"] is False
    assert mbb["asymptotically_valid"] is True
    assert wild["n_failed"] == 0 and mbb["n_failed"] == 0
    # Significance under the paper's own (Efron percentile) convention.
    wl, wu = np.asarray(wild["lower_efron"]), np.asarray(wild["upper_efron"])
    ml, mu = np.asarray(mbb["lower_efron"]), np.asarray(mbb["upper_efron"])
    ip_wild = {h for h in range(49) if wu[h, 2] < 0}
    ip_mbb = {h for h in range(49) if mu[h, 2] < 0}
    # Wild reproduces the paper's pattern: IP significant across the medium
    # run (observed 7..40 at 2000 draws), EBP significant on impact.
    assert set(range(10, 36)) <= ip_wild
    assert wl[0, 3] > 0
    # The valid bands undo most of it: observed 25..29 at 2000 draws.
    assert len(ip_mbb) < len(ip_wild) / 3
    assert ip_mbb <= set(range(15, 41))
    # ... because they are wider, not because they moved: median MBB/wild
    # width ratio on IP over h=1..48 (observed ~1.7).
    widths = (mu - ml)[1:, 2] / (wu - wl)[1:, 2]
    assert np.median(widths) > 1.2


def test_post84_identification_weakens_and_output_effects_dissolve(baseline):
    gk, proxy, pr = baseline
    data84, proxy84 = gk_repl.post84(gk)
    assert data84.shape[0] == 342                       # 1984:1-2012:6
    pr84 = tsecon.proxy_svar(data84, proxy84, lags=12, horizon=48,
                             norm_var=0, unit=0.2)
    f_full, f_84 = pr["first_stage"], pr84["first_stage"]
    # The effective F falls (observed 17.50 -> 13.82) and the certified
    # worst-case-bias bound rises (0.155 -> 0.233).
    assert f_84["effective_f"] < f_full["effective_f"] - 2.0
    assert 8.0 < f_84["effective_f"] < 17.0
    assert f_84["tau_bound"] > f_full["tau_bound"] + 0.04
    # AR sets: bounded in both samples (relevance clears the chi2 bar) ...
    kw = dict(lags=12, horizon=48, norm_var=0, unit=0.2, alpha=0.05)
    ar_full = tsecon.proxy_ar_sets(gk["data"], proxy, **kw)
    ar_84 = tsecon.proxy_ar_sets(data84, proxy84, **kw)
    assert ar_full["ar_bounded_all"] and ar_84["ar_bounded_all"]
    assert ar_full["level"] == 0.95 and ar_84["level"] == 0.95
    # ... but the post-84 IP sets are materially wider (observed 1.80x).
    wf = np.array([ar_full["cells"][h][2]["upper"] - ar_full["cells"][h][2]["lower"]
                   for h in range(1, 49)])
    w8 = np.array([ar_84["cells"][h][2]["upper"] - ar_84["cells"][h][2]["lower"]
                   for h in range(1, 49)])
    assert np.median(w8 / wf) > 1.4
    # The DTH conclusion: no IP set excludes zero post-1984 (nor, under this
    # most conservative object, in the full sample), while the EBP impact
    # response excludes zero in BOTH samples.
    assert not any(ar_84["cells"][h][2]["excludes_zero"] for h in range(49))
    assert ar_full["cells"][0][3]["excludes_zero"]
    assert ar_84["cells"][0][3]["excludes_zero"]


def test_seeded_bands_are_bit_reproducible(baseline):
    gk, proxy, _ = baseline
    kw = dict(lags=12, horizon=8, norm_var=0, unit=0.2, alpha=0.05,
              n_boot=100, seed=7)
    a = tsecon.proxy_svar_bands(gk["data"], proxy, **kw)
    b = tsecon.proxy_svar_bands(gk["data"], proxy, **kw)
    assert np.array_equal(np.asarray(a["lower"]), np.asarray(b["lower"]))
    assert np.array_equal(np.asarray(a["upper"]), np.asarray(b["upper"]))
