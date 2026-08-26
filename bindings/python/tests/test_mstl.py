"""MSTL through the Python surface, re-pinned against fixtures/mstl.json
(statsmodels 0.14.6 MSTL, elementwise — a strong third-party golden; see
fixtures/generate_mstl_fixtures.py for provenance). The degenerate
single-period case is additionally required to reproduce tsecon's own stl
bitwise — internal consistency between the two entry points, graded
separately from the third-party golden."""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
FX = json.loads((FIXTURES / "mstl.json").read_text())


def _call(case, y):
    periods = case["periods_arg"]
    if isinstance(periods, int):
        periods = [periods]
    kwargs = dict(case["stl_kwargs"])
    if case["windows_arg"] is not None:
        kwargs["windows"] = case["windows_arg"]
    return tsecon.mstl(y, periods, iterate=case["iterate"], **kwargs)


# ---------------------------------------------------------------- golden
@pytest.mark.parametrize(
    "case",
    FX["cases"],
    ids=[f"{c['series']}-{c['config_name']}" for c in FX["cases"]],
)
def test_mstl_matches_statsmodels(case):
    y = np.asarray(FX["series"][case["series"]], dtype=float)
    r = _call(case, y)

    assert r["periods"] == case["resolved_periods"]
    assert r["windows"] == case["resolved_windows"]
    assert r["dropped_periods"] == case["dropped_periods"]
    assert list(r["seasonal"].keys()) == [
        f"seasonal_{p}" for p in case["resolved_periods"]
    ], "seasonal dict keyed by ascending period"
    assert r["iterate"] == (
        1 if len(case["resolved_periods"]) == 1 else case["iterate"]
    )

    for p, expected in zip(case["resolved_periods"], case["seasonal"]):
        expected = np.asarray(expected, dtype=float)
        np.testing.assert_allclose(
            np.asarray(r["seasonal"][f"seasonal_{p}"]),
            expected,
            rtol=0,
            atol=1e-8 * max(1.0, np.abs(expected).max()),
            err_msg=f"{case['series']}/{case['config_name']} seasonal_{p}",
        )
    for key in ("trend", "resid"):
        expected = np.asarray(case[key], dtype=float)
        np.testing.assert_allclose(
            np.asarray(r[key]),
            expected,
            rtol=0,
            atol=1e-8 * max(1.0, np.abs(expected).max()),
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


def test_single_period_mstl_equals_own_stl_bitwise():
    """Internal consistency, graded separately from the statsmodels golden:
    one period means one STL pass with seasonal window 11 (the 7 + 4*1
    default rule), so tsecon.mstl must reproduce tsecon.stl to the bit."""
    y = np.asarray(FX["series"]["single"], dtype=float)
    m = tsecon.mstl(y, [12])
    s = tsecon.stl(y, 12, seasonal=11)
    assert m["periods"] == [12] and m["windows"] == [11] and m["iterate"] == 1
    np.testing.assert_array_equal(m["seasonal"]["seasonal_12"], s["seasonal"])
    np.testing.assert_array_equal(m["trend"], s["trend"])
    np.testing.assert_array_equal(m["resid"], s["resid"])
    np.testing.assert_array_equal(m["weights"], s["weights"])


# ------------------------------------------------------ result invariants
def test_components_sum_to_y_and_shapes():
    y = np.asarray(FX["series"]["two_seasonal"], dtype=float)
    r = tsecon.mstl(y, [24, 168])
    n = y.shape[0]
    assert all(
        np.asarray(r["seasonal"][k]).shape == (n,) for k in r["seasonal"]
    )
    assert np.asarray(r["trend"]).shape == (n,)
    assert np.asarray(r["resid"]).shape == (n,)
    assert np.asarray(r["weights"]).shape == (n,)
    recon = (
        sum(np.asarray(v) for v in r["seasonal"].values())
        + np.asarray(r["trend"])
        + np.asarray(r["resid"])
    )
    np.testing.assert_allclose(recon, y, rtol=0, atol=1e-9 * max(1.0, np.abs(y).max()))


def test_period_order_and_input_types_do_not_matter():
    y = FX["series"]["two_seasonal"]
    a = tsecon.mstl(np.asarray(y), [24, 168])
    b = tsecon.mstl(list(y), (168, 24))  # list y, unsorted tuple periods
    c = tsecon.mstl(np.asarray(y), np.array([168, 24]))  # numpy int periods
    for k in a["seasonal"]:
        np.testing.assert_array_equal(a["seasonal"][k], b["seasonal"][k])
        np.testing.assert_array_equal(a["seasonal"][k], c["seasonal"][k])
    np.testing.assert_array_equal(a["trend"], b["trend"])
    assert a["periods"] == b["periods"] == c["periods"] == [24, 168]


def test_period_at_half_n_is_dropped_like_statsmodels():
    """statsmodels warns and drops any period >= n/2; here the drop is
    reported in dropped_periods and the rest of the decomposition matches
    the fixture (the droppy golden case pins the numbers)."""
    y = np.asarray(FX["series"]["droppy"], dtype=float)  # n = 120
    r = tsecon.mstl(y, [12, 60])  # 60 >= 120/2
    assert r["dropped_periods"] == [60]
    assert r["periods"] == [12]
    only = tsecon.mstl(y, [12])
    np.testing.assert_array_equal(
        r["seasonal"]["seasonal_12"], only["seasonal"]["seasonal_12"]
    )
    # But every period dropped is a refusal, not a NameError crash.
    with pytest.raises(ValueError, match="half the series length"):
        tsecon.mstl(y[:24], [12])


# -------------------------------------------------------------- strength
def test_per_period_seasonal_strength_matches_formula():
    y = np.asarray(FX["series"]["two_seasonal"], dtype=float)
    r = tsecon.mstl(y, [24, 168])
    resid = np.asarray(r["resid"])
    vr = resid.var(ddof=1)
    for key, got in r["seasonal_strength"].items():
        s = np.asarray(r["seasonal"][key])
        want = max(0.0, 1.0 - vr / (s + resid).var(ddof=1))
        assert got == pytest.approx(want, rel=1e-12)
        assert 0.0 <= got <= 1.0


def test_constant_series_decomposes_but_reports_no_strength():
    """A constant series decomposes fine (statsmodels does too) but its
    variance-ratio strength would be float noise, so — matching the
    seasonal_strength function's refusal — no number is reported."""
    y = np.full(120, 3.7)
    r = tsecon.mstl(y, [12])
    np.testing.assert_allclose(r["seasonal"]["seasonal_12"], 0.0, atol=1e-10)
    np.testing.assert_allclose(r["trend"], 3.7, atol=1e-10)
    assert r["seasonal_strength"] is None
    # Near-constant with visible variation still reports strengths.
    y2 = np.full(120, 3.7) + 1e-9 * (np.arange(120) % 12)
    out = tsecon.mstl(y2, [12])
    assert out["seasonal_strength"] is not None


# ---------------------------------------------------------------- errors
def test_mstl_errors_teach():
    y = np.asarray(FX["series"]["two_seasonal"], dtype=float)
    with pytest.raises(ValueError, match="periods"):
        tsecon.mstl(y, [])
    with pytest.raises(ValueError, match="distinct"):
        tsecon.mstl(y, [24, 24])
    with pytest.raises(ValueError, match="positive integers"):
        tsecon.mstl(y, [24, -5])
    with pytest.raises(ValueError, match="periods"):
        tsecon.mstl(y, [1, 24])  # period 1 survives the drop rule, named
    with pytest.raises(ValueError, match="windows"):
        tsecon.mstl(y, [24, 168], windows=[11])  # length mismatch
    with pytest.raises(ValueError, match="windows"):
        tsecon.mstl(y, [24, 168], windows=[10, 15])  # even window
    with pytest.raises(ValueError, match="iterate"):
        tsecon.mstl(y, [24, 168], iterate=0)
    with pytest.raises(ValueError, match="trend"):
        tsecon.mstl(y, [24, 168], trend=101)  # <= period 168, STL's check
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.mstl(np.r_[y[:100], np.nan, y[101:]], [24])
    # A scalar periods is refused at the boundary: the API takes a sequence
    # (a single period is periods=[12]).
    with pytest.raises(TypeError, match="mstl"):
        tsecon.mstl(y, 24)
