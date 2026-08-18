"""Python-boundary tests for panel_lp's split-panel jackknife
(bias_correction="spj", Mei-Sheng-Shi 2026).

The numeric authority is fixtures/panel_spj.json — an independent NumPy
transcription of the authors' pLP reference algebra (see the generator's
docstring for the honest validation grade: transcription + Monte Carlo,
not a stored-output match of the R package). Here we pin the Python
boundary against the same fixture at 1e-10 and exercise the argument
plumbing, metadata stamps, and refusals.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "fixtures" / "panel_spj.json").read_text()
)


def _run(case, **overrides):
    y = np.array(FIXTURE["y"])
    shock = np.array(FIXTURE["shock"])
    design = FIXTURE["design"]
    spec = FIXTURE["cases"][case]
    kwargs = dict(
        horizon=design["max_horizon"],
        n_lag_controls=design["shock_lags"],
        se_type="cluster" if spec["se_type"] == "cluster" else "driscoll_kraay",
        cumulative=spec["cumulative"],
        bias_correction="spj",
    )
    if spec["bandwidth"] is not None:
        kwargs["bandwidth"] = spec["bandwidth"]
    kwargs.update(overrides)
    return tsecon.panel_lp(y, shock, **kwargs)


@pytest.mark.parametrize(
    "case", ["spj_cluster", "spj_dk_bw2", "spj_cluster_cumulative"]
)
def test_spj_matches_the_transcription_fixture(case):
    r = _run(case)
    want = FIXTURE["cases"][case]["horizons"]
    assert len(r["irf"]) == len(want)
    for h, wh in enumerate(want):
        assert r["irf"][h] == pytest.approx(wh["beta_spj"][0], rel=1e-10), f"h={h}"
        assert r["se"][h] == pytest.approx(wh["se_spj"][0], rel=1e-10), f"h={h}"
        assert r["nobs"][h] == wh["nobs"]


def test_spj_metadata_is_stamped():
    r = _run("spj_cluster")
    assert r["bias_correction"] == "spj"
    assert r["jackknife"] is False
    assert r["se_type"] == "cluster"
    assert r["cumulative"] is False
    rc = _run("spj_cluster_cumulative")
    assert rc["cumulative"] is True


def test_dj_alias_and_flag_agree_and_conflict_raises():
    y = np.array(FIXTURE["y"])
    shock = np.array(FIXTURE["shock"])
    via_flag = tsecon.panel_lp(y, shock, horizon=2, n_lag_controls=1,
                               se_type="cluster", jackknife=True)
    via_enum = tsecon.panel_lp(y, shock, horizon=2, n_lag_controls=1,
                               se_type="cluster", bias_correction="dj")
    np.testing.assert_allclose(via_flag["irf"], via_enum["irf"], rtol=0, atol=0)
    assert via_flag["bias_correction"] == "dhaene_jochmans"
    assert via_flag["jackknife"] is True
    assert via_enum["jackknife"] is True
    # The two different corrections at once are ambiguous.
    with pytest.raises(ValueError, match="set exactly one"):
        tsecon.panel_lp(y, shock, horizon=2, n_lag_controls=1,
                        se_type="cluster", jackknife=True,
                        bias_correction="spj")


def test_spj_corrects_toward_the_truth_on_the_fixture_panel():
    # The fixture DGP has true irf beta * rho^h; the SPJ point estimates
    # must stay within 4 adjusted-score SEs of it at every horizon.
    r = _run("spj_cluster", se_type="driscoll_kraay", bandwidth=2.0)
    truth = FIXTURE["true_irf"]
    for h, t in enumerate(truth):
        assert abs(r["irf"][h] - t) < 4 * r["se"][h] + 0.05, f"h={h}"


def test_spj_refusals_teach():
    y = np.array(FIXTURE["y"])
    shock = np.array(FIXTURE["shock"])
    with pytest.raises(ValueError, match="no homoskedastic"):
        tsecon.panel_lp(y, shock, horizon=2, n_lag_controls=1,
                        se_type="nonrobust", bias_correction="spj")
    with pytest.raises(ValueError, match="bias_correction"):
        tsecon.panel_lp(y, shock, horizon=2, n_lag_controls=1,
                        bias_correction="typo")
    # T too short for the median split names the split, not a symptom.
    with pytest.raises(ValueError, match="split-panel jackknife"):
        tsecon.panel_lp(y[:, :8], shock[:8], horizon=3, n_lag_controls=1,
                        se_type="cluster", bias_correction="spj")


def test_uncorrected_route_is_untouched():
    # The default path must not pick up any correction silently.
    y = np.array(FIXTURE["y"])
    shock = np.array(FIXTURE["shock"])
    r = tsecon.panel_lp(y, shock, horizon=2, n_lag_controls=1, se_type="cluster")
    assert r["bias_correction"] == "none"
    assert r["jackknife"] is False
    # ... and matches the stored full-sample leg of the SPJ fixture.
    want = FIXTURE["cases"]["spj_cluster"]["horizons"]
    for h in range(3):
        assert r["irf"][h] == pytest.approx(want[h]["beta_full"][0], rel=1e-10)
