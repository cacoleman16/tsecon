"""Golden and behavioural tests for `tsecon.panel_distributed_lag`.

Re-pins fixtures/panel_dl.json (linearmodels PanelOLS on the explicitly
lagged design, nine cases x three covariance estimators, plus the
documented delta-method transcription — see the generator header for the
honest grading) through the Python surface, checks the key set, the
inert-keyword refusals (`eval_points` under powers=1, `bandwidth` under a
non-kernel se_type), the bit-identity with `panel_fe` at L = 0, the error
messages, and the guide chapter's runnable example numbers.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
DL = json.loads((FIX / "panel_dl.json").read_text())
SE_TYPES = {
    "nonrobust": ("nonrobust", None),
    "cluster_entity": ("cluster", None),
    "driscoll_kraay": ("driscoll_kraay", None),
}
BASE_KEYS = {
    "params", "names", "bse", "tvalues", "cov", "lag_effects", "lag_se",
    "cumulative_effect", "cumulative_se", "cumulative_ci_low", "cumulative_ci_high",
    "eval_points", "marginal_effect", "marginal_se", "turning_point", "turning_point_se",
    "nobs", "n_entities", "n_periods_used", "lags", "powers", "df_resid", "se_type",
    "entity_effects", "time_effects", "entity_trends",
}


def _case_inputs(case):
    inputs = DL["inputs"]
    y = np.array(inputs[case["outcome"]])
    x = np.array([inputs[name] for name in case["regressors"]])
    return y, x


def _expected_names(case):
    """The binding labels regressors positionally (`x0`, `x1`, ...) where
    the fixture used `x`/`z`; the column order is the same."""
    names = []
    for j in range(len(case["regressors"])):
        for p in range(1, case["powers"] + 1):
            for l in range(case["lags"] + 1):
                names.append(f"x{j}_L{l}" if p == 1 else f"x{j}^2_L{l}")
    return names


def _call(case, key):
    y, x = _case_inputs(case)
    se_type, bw = SE_TYPES[key]
    kw = dict(
        lags=case["lags"], powers=case["powers"],
        entity_effects=case["entity_effects"], time_effects=case["time_effects"],
        entity_trends=case["entity_trends"], se_type=se_type,
    )
    if se_type == "driscoll_kraay":
        kw["bandwidth"] = float(case["bandwidth"])
    if isinstance(case["eval_points"], list):
        kw["eval_points"] = np.array(case["eval_points"])
    return tsecon.panel_distributed_lag(y, x, **kw)


@pytest.mark.parametrize("key", sorted(SE_TYPES))
@pytest.mark.parametrize("case", DL["cases"], ids=lambda c: c["name"])
def test_matches_linearmodels_panelols(case, key):
    r = _call(case, key)
    want = case["fits"][key]
    assert r["names"] == _expected_names(case)
    assert [n.replace("x_", "x0_").replace("z_", "x1_").replace("x^2", "x0^2") for n in case["names"]] == r["names"]
    assert r["nobs"] == want["nobs"]
    assert r["df_resid"] == want["df_resid"]
    assert r["n_entities"] == DL["n_entities"]
    assert r["n_periods_used"] == DL["n_periods"] - case["lags"]
    assert r["lags"] == case["lags"] and r["powers"] == case["powers"]
    np.testing.assert_allclose(r["params"], want["params"], rtol=1e-10)
    np.testing.assert_allclose(r["bse"], want["bse"], rtol=1e-10)
    np.testing.assert_allclose(r["tvalues"], want["tstats"], rtol=1e-10)
    cov = np.array(r["cov"])
    want_cov = np.array(want["cov"])
    scale = np.abs(want_cov).max()
    np.testing.assert_allclose(cov, want_cov, rtol=1e-10, atol=1e-10 * scale)
    assert cov.shape == (len(want["params"]),) * 2


@pytest.mark.parametrize("key", sorted(SE_TYPES))
@pytest.mark.parametrize(
    "case", [c for c in DL["cases"] if len(c["regressors"]) == 1], ids=lambda c: c["name"]
)
def test_delta_method_matches_transcription(case, key):
    r = _call(case, key)
    d = case["delta"][key]
    for name in ("cumulative_effect", "cumulative_se", "cumulative_ci_low", "cumulative_ci_high"):
        np.testing.assert_allclose(r[name][0], d[name], rtol=1e-10)
    # Half-width is the documented normal one.
    half = np.array(r["cumulative_ci_high"][0]) - np.array(r["cumulative_effect"][0])
    np.testing.assert_allclose(half, DL["z975"] * np.array(r["cumulative_se"][0]), rtol=1e-12)
    if case["powers"] == 2:
        np.testing.assert_allclose(r["eval_points"][0], d["eval_points"], rtol=1e-10)
        np.testing.assert_allclose(r["marginal_effect"][0], d["marginal_effect"], rtol=1e-10)
        np.testing.assert_allclose(r["marginal_se"][0], d["marginal_se"], rtol=1e-10)
        assert r["turning_point"][0] == pytest.approx(d["turning_point"], rel=1e-10)
        assert r["turning_point_se"][0] == pytest.approx(d["turning_point_se"], rel=1e-10)
    else:
        for name in ("eval_points", "marginal_effect", "marginal_se", "turning_point",
                     "turning_point_se"):
            assert r[name] is None


def test_worst_relative_error_against_linearmodels_is_reported():
    """The 1e-10 pin is a ceiling; record how far below it the fit lands."""
    worst = 0.0
    for case in DL["cases"]:
        for key in SE_TYPES:
            r = _call(case, key)
            want = case["fits"][key]
            for a, b in ((r["params"], want["params"]), (r["bse"], want["bse"])):
                worst = max(worst, float(np.max(np.abs(np.array(a) - np.array(b)) / np.abs(b))))
    assert worst < 1e-10
    print(f"\nworst relative error vs linearmodels (params, bse): {worst:.2e}")


def test_returned_keys_are_the_documented_set():
    case = DL["cases"][0]
    r = _call(case, "cluster_entity")
    assert set(r) == BASE_KEYS
    assert r["se_type"] == "cluster"
    assert r["entity_effects"] is True and r["time_effects"] is True
    assert r["entity_trends"] is False
    quad = next(c for c in DL["cases"] if c["powers"] == 2)
    rq = _call(quad, "cluster_entity")
    assert set(rq) == BASE_KEYS
    assert np.shape(rq["lag_effects"]) == (1, 2, quad["lags"] + 1)
    assert np.shape(rq["cumulative_effect"]) == (1, 2)
    assert np.shape(rq["marginal_effect"]) == (1, len(quad["eval_points"]))
    assert len(rq["turning_point"]) == 1


def test_lag_zero_without_time_effects_is_panel_fe_bit_identically():
    case = next(c for c in DL["cases"] if c["name"] == "linear_L1_two_regressors")
    y, x = _case_inputs(case)
    for se_type, bw in (("nonrobust", None), ("cluster", None), ("driscoll_kraay", 4.0)):
        kw = {} if bw is None else {"bandwidth": bw}
        fe = tsecon.panel_fe(y, x, se_type=se_type, **kw)
        dl = tsecon.panel_distributed_lag(y, x, lags=0, time_effects=False, se_type=se_type, **kw)
        assert dl["params"].tolist() == fe["params"].tolist()
        assert dl["bse"].tolist() == fe["bse"].tolist()
        assert dl["tvalues"].tolist() == fe["tvalues"].tolist()
        assert dl["names"] == ["x0_L0", "x1_L0"]
        assert dl["cumulative_effect"] == [[fe["params"][0]], [fe["params"][1]]]
        assert dl["cumulative_se"] == [[fe["bse"][0]], [fe["bse"][1]]]


def test_eval_points_is_refused_under_powers_one():
    case = DL["cases"][0]
    y, x = _case_inputs(case)
    with pytest.raises(ValueError, match="eval_points was given but powers=1 ignores it"):
        tsecon.panel_distributed_lag(y, x, lags=1, eval_points=[20.0])
    # Under powers=2 it acts and is echoed.
    r = tsecon.panel_distributed_lag(y, x, lags=1, powers=2, eval_points=[15.0, 25.0])
    assert r["eval_points"] == [[15.0, 25.0]]
    # And the default is the pooled regressor mean.
    r = tsecon.panel_distributed_lag(y, x, lags=1, powers=2)
    assert r["eval_points"][0][0] == pytest.approx(float(x[0].mean()), rel=1e-12)


def test_bandwidth_is_refused_under_non_kernel_se_types():
    case = DL["cases"][0]
    y, x = _case_inputs(case)
    for se_type in ("cluster", "nonrobust"):
        with pytest.raises(ValueError, match="bandwidth=8 has no effect under se_type"):
            tsecon.panel_distributed_lag(y, x, lags=1, se_type=se_type, bandwidth=8.0)
    with pytest.raises(ValueError, match="unknown se_type"):
        tsecon.panel_distributed_lag(y, x, lags=1, se_type="hc3")


def test_error_messages_teach():
    case = DL["cases"][0]
    y, x = _case_inputs(case)
    T = y.shape[1]
    with pytest.raises(ValueError, match="at least two periods"):
        tsecon.panel_distributed_lag(y, x, lags=T - 1)
    with pytest.raises(ValueError, match="lags must be a non-negative integer"):
        tsecon.panel_distributed_lag(y, x, lags=-1)
    with pytest.raises(ValueError, match="powers must be 1 .* or 2"):
        tsecon.panel_distributed_lag(y, x, lags=1, powers=3)
    with pytest.raises(ValueError, match="entity_trends=True requires entity_effects=True"):
        tsecon.panel_distributed_lag(y, x, lags=1, entity_effects=False, entity_trends=True)
    with pytest.raises(ValueError, match="leaves no fixed effects"):
        tsecon.panel_distributed_lag(y, x, lags=1, entity_effects=False, time_effects=False)
    bad = y.copy()
    bad[2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.panel_distributed_lag(bad, x, lags=1)
    # A regressor common to every entity is absorbed by the time effects.
    common = np.broadcast_to(np.sin(np.arange(T)), y.shape)[None]
    with pytest.raises(ValueError, match="absorb a regressor entirely"):
        tsecon.panel_distributed_lag(y, np.ascontiguousarray(common), lags=1)
    with pytest.raises(ValueError, match="no residual degrees of freedom"):
        tsecon.panel_distributed_lag(y[:2, :3], x[:, :2, :3], lags=1)
    with pytest.raises(ValueError, match="dimension mismatch"):
        tsecon.panel_distributed_lag(y, x[:, :, :-1], lags=1)


def _guide_panel():
    """The seeded simulated panel of the guide chapter's runnable example."""
    rng = np.random.default_rng(14)
    N, T = 40, 45
    mu = rng.normal(20.0, 5.0, N)
    temp = mu[:, None] + rng.standard_normal((N, T))
    growth = (rng.normal(0.0, 1.0, N)[:, None] + rng.normal(0.0, 0.5, T)[None, :]
              + 0.30 * temp - 0.0075 * temp ** 2 + rng.standard_normal((N, T)))
    return temp, growth


def test_guide_chapter_example_numbers():
    """The numbers printed in docs/guide/14-panel-time-series.md."""
    temp, growth = _guide_panel()
    r = tsecon.panel_distributed_lag(growth, temp[None], lags=2, powers=2, se_type="cluster",
                                     eval_points=[10.0, 20.0, 30.0])
    b1, b2 = r["cumulative_effect"][0]
    s1, s2 = r["cumulative_se"][0]
    tp, tp_se = r["turning_point"][0], r["turning_point_se"][0]
    me = np.array(r["marginal_effect"][0])
    mse = np.array(r["marginal_se"][0])
    assert r["nobs"] == 40 * 43 and r["names"][:3] == ["x0_L0", "x0_L1", "x0_L2"]
    # Pinned at the precision the guide prints them.
    assert np.round([b1, s1, b2, s2], 4).tolist() == [0.3179, 0.1159, -0.0074, 0.0026]
    assert np.round([tp, tp_se], 2).tolist() == [21.60, 2.69]
    assert np.round(me, 4).tolist() == [0.1707, 0.0236, -0.1235]
    assert np.round(mse, 4).tolist() == [0.0688, 0.0395, 0.0624]
    # The truth of the DGP: dy/dtemp = 0.30 - 0.015 temp, peak at 20.
    assert abs(b1 - 0.30) < 3 * s1 and abs(b2 + 0.0075) < 3 * s2 and abs(tp - 20.0) < 3 * tp_se
