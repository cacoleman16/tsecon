"""STL decomposition, seasonal strength, and nsdiffs through the Python
surface, re-pinned against fixtures/stl.json (statsmodels 0.14.6; see
fixtures/generate_stl_fixtures.py for the honest grading: STL arrays are a
strong third-party golden, strength/nsdiffs are documented-formula/rule
transcriptions on top of statsmodels components)."""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
FX = json.loads((FIXTURES / "stl.json").read_text())


def _params(case):
    kw = dict(case["kwargs"])
    kw.update(case["fit_kwargs"])
    return kw


# ------------------------------------------------------------------- STL
@pytest.mark.parametrize(
    "case",
    FX["stl"],
    ids=[f"{c['series']}-{c['config_name']}" for c in FX["stl"]],
)
def test_stl_matches_statsmodels(case):
    y = np.asarray(FX["series"][case["series"]], dtype=float)
    r = tsecon.stl(y, case["period"], **_params(case))
    for key in ("seasonal", "trend", "resid"):
        np.testing.assert_allclose(
            np.asarray(r[key]),
            np.asarray(case[key], dtype=float),
            rtol=0,
            atol=1e-8 * max(1.0, np.abs(np.asarray(case[key])).max()),
            err_msg=f"{case['series']}/{case['config_name']} {key}",
        )
    if case["weights"] is not None:
        np.testing.assert_allclose(
            np.asarray(r["weights"]),
            np.asarray(case["weights"], dtype=float),
            rtol=0,
            atol=1e-8,
        )
    else:
        assert np.all(np.asarray(r["weights"]) == 1.0)
    # Resolved config must equal statsmodels' STL(...).config.
    for key, want in case["resolved"].items():
        assert r["config"][key] == want, f"config[{key}]"


def test_stl_reconstruction_and_defaults():
    y = np.asarray(FX["series"]["co2"], dtype=float)
    r = tsecon.stl(y, 12)
    recon = np.asarray(r["seasonal"]) + np.asarray(r["trend"]) + np.asarray(r["resid"])
    np.testing.assert_allclose(recon, y, rtol=0, atol=1e-9)
    assert r["config"]["trend"] == 23  # ceil(1.5*12/(1-1.5/7)) -> odd
    assert r["config"]["low_pass"] == 13
    assert r["config"]["inner_iter"] == 5
    assert r["config"]["outer_iter"] == 0


def test_stl_pandas_and_list_inputs_coerce():
    y = FX["series"]["synthetic_m"]
    r_list = tsecon.stl(list(y), 12)
    r_arr = tsecon.stl(np.asarray(y), 12)
    np.testing.assert_array_equal(r_list["seasonal"], r_arr["seasonal"])


def test_stl_errors_teach():
    y = np.asarray(FX["series"]["synthetic_m"], dtype=float)
    with pytest.raises(ValueError, match="period"):
        tsecon.stl(y, 1)
    with pytest.raises(ValueError, match="two full cycles"):
        tsecon.stl(y[:20], 12)
    with pytest.raises(ValueError, match="seasonal"):
        tsecon.stl(y, 12, seasonal=4)
    with pytest.raises(ValueError, match="trend"):
        tsecon.stl(y, 12, trend=11)
    with pytest.raises(ValueError, match="inner_iter"):
        tsecon.stl(y, 12, inner_iter=0)


# -------------------------------------------------------------- strength
@pytest.mark.parametrize(
    "case", FX["strength"], ids=[c["series"] for c in FX["strength"]]
)
def test_seasonal_strength_matches_fixture(case):
    y = np.asarray(FX["series"][case["series"]], dtype=float)
    r = tsecon.seasonal_strength(y, case["period"])
    assert r["seasonal_strength"] == pytest.approx(
        case["seasonal_strength"], rel=1e-8, abs=1e-12
    )
    assert r["trend_strength"] == pytest.approx(
        case["trend_strength"], rel=1e-8, abs=1e-12
    )
    assert r["period"] == case["period"]


# --------------------------------------------------------------- nsdiffs
@pytest.mark.parametrize(
    "case",
    FX["nsdiffs"],
    ids=[f"{c['series']}-maxd{c['max_d']}" for c in FX["nsdiffs"]],
)
def test_nsdiffs_matches_fixture(case):
    y = np.asarray(FX["series"][case["series"]], dtype=float)
    r = tsecon.nsdiffs(y, case["period"], max_d=case["max_d"])
    assert r["d"] == case["d"]
    assert r["stop"] == case["stop"]
    assert r["threshold"] == FX["_meta"]["seas_threshold"]
    assert len(r["steps"]) == len(case["steps"])
    for got, want in zip(r["steps"], case["steps"]):
        assert got["d"] == want["d"]
        assert got["n"] == want["n"]
        assert got["seasonal_strength"] == pytest.approx(
            want["seasonal_strength"], rel=1e-8, abs=1e-12
        )
        assert got["needs_differencing"] == want["needs_differencing"]
    assert isinstance(r["interpretation"], str) and r["interpretation"]


def test_nsdiffs_errors_teach():
    y = np.asarray(FX["series"]["synthetic_m"], dtype=float)
    with pytest.raises(ValueError):
        tsecon.nsdiffs(y, 12, alpha=1.5)
    with pytest.raises(ValueError, match="two full cycles"):
        tsecon.nsdiffs(y[:20], 12)


def test_seasonal_strength_refuses_constant_series():
    """Audit round 6, finding 1: a flat line used to return a float-noise
    strength of ~0.64 — coincidentally at the nsdiffs threshold — instead of
    refusing. The refusal must hold at any scale and must teach."""
    for c in [0.0, 3.7, -2.5, 1e6]:
        with pytest.raises(ValueError, match="constant"):
            tsecon.seasonal_strength(np.full(120, c), 12)
    # Near-constant with visible variation still runs.
    y = np.full(120, 3.7) + 1e-9 * (np.arange(120) % 12)
    out = tsecon.seasonal_strength(y, 12)
    assert 0.0 <= out["seasonal_strength"] <= 1.0
