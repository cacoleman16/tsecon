"""Golden tests for the panel and Clark-West/Giacomini-White bindings."""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
PANEL = json.loads((FIX / "panel.json").read_text())
FE2 = json.loads((FIX / "forecast_eval2.json").read_text())


def test_panel_fe_matches_linearmodels():
    # Rebuild the fixture design: y[:,1:] on s0=shock[1:], s1=shock[:-1], entity FE.
    y = np.array(PANEL["panel"]["y"])           # N x T
    shock = np.array(PANEL["panel"]["shock"])   # T
    N, T = y.shape
    outcome = y[:, 1:]                           # N x (T-1)
    s0 = np.tile(shock[1:], (N, 1))             # N x (T-1)
    s1 = np.tile(shock[:-1], (N, 1))
    regressors = np.stack([s0, s1])             # 2 x N x (T-1)
    for key, se in [("nonrobust", "nonrobust"),
                    ("cluster_entity", "cluster"),
                    ("driscoll_kraay", "driscoll_kraay")]:
        r = tsecon.panel_fe(outcome, regressors, se_type=se, bandwidth=4.0)
        blk = PANEL["panel_ols_fe_s0_s1_drop_t0"][key]
        want_params = [blk["params"]["s0"], blk["params"]["s1"]]
        want_bse = [blk["bse"]["s0"], blk["bse"]["s1"]]
        np.testing.assert_allclose(r["params"], want_params, rtol=1e-6)
        np.testing.assert_allclose(r["bse"], want_bse, rtol=1e-6)


def test_panel_lp_recovers_known_irf():
    y = np.array(PANEL["panel"]["y"])
    shock = np.array(PANEL["panel"]["shock"])
    true_irf = np.array(PANEL["panel"]["true_irf_psi"])
    r = tsecon.panel_lp(y, shock, horizon=6, n_lag_controls=2,
                        se_type="driscoll_kraay", bandwidth=4.0)
    assert len(r["irf"]) == 7  # horizons 0..6
    for h in range(len(r["irf"])):
        assert abs(r["irf"][h] - true_irf[h]) < 4 * r["se"][h] + 0.05, f"h={h}"
    assert abs(r["irf"][0] - true_irf[0]) < 0.1


def test_cw_and_gw_match_fixture():
    ytrue = np.array(FE2["ytrue"]); yh1 = np.array(FE2["yhat1"]); yh2 = np.array(FE2["yhat2"])
    e1, e2 = ytrue - yh1, ytrue - yh2
    L = FE2["lrv_lags"]
    cw = tsecon.cw_test(e1, e2, yh1, yh2, lrv_lags=L)
    assert cw["cw_stat"] == pytest.approx(FE2["clark_west"]["stat"], rel=1e-9)
    assert cw["p_value"] == pytest.approx(FE2["clark_west"]["pvalue_one_sided"], rel=1e-8)
    gw = tsecon.gw_test(e1**2, e2**2, lrv_lags=L)
    assert gw["gw_stat"] == pytest.approx(FE2["giacomini_white_uncond"]["stat"], rel=1e-9)
    assert gw["df"] == 1


def test_absorbed_regressor_raises_not_a_t_statistic():
    """A regressor constant within every entity is annihilated by the
    within transformation; the old positive-definiteness guard returned a
    publishable cluster t for it unless the residue was bit-exactly zero
    (audit rounds 2-4, finding 1). linearmodels refuses this design, and
    the docstring promises its conventions -- so must we, at every
    se_type, for entity constants stored as ordinary doubles too."""
    rng = np.random.default_rng(20260813)
    n_ent, n_per = 20, 8
    live = rng.standard_normal((n_ent, n_per))
    outcome = 0.9 * live + rng.standard_normal((n_ent, n_per))
    for label, col in [
        ("log land area", np.log(7.3 * (np.arange(n_ent) + 2.0))),
        ("share in [0,1]", ((np.arange(n_ent) * 37 % 97) + 0.5) / 98.0),
        ("founding year", 1950.0 + 3.0 * np.arange(n_ent)),
    ]:
        dead = np.repeat(col[:, None], n_per, axis=1)
        regressors = np.stack([live, dead])
        for se in ("cluster", "nonrobust", "driscoll_kraay"):
            with pytest.raises(ValueError, match="constant within"):
                tsecon.panel_fe(outcome, regressors, se_type=se)
