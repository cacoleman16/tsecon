"""Python-level contract + fixture re-pin for the Zivot-Andrews one-break
unit-root test.

The heavy numerical validation lives in the crate golden
(crates/tsecon-diag/tests/zivot_andrews_golden.rs, against statsmodels
0.14.6 with an arch 8.0.0 cross-check). These tests re-pin the same
fixture THROUGH the Python surface — so the binding's argument plumbing is
covered, not just the Rust core — plus live parity against statsmodels
where this venv has it, the documented teaching errors, and the
break-localization contract.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURE = Path(__file__).parents[3] / "fixtures" / "zivot_andrews.json"


def _fixture():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


FX = _fixture() if FIXTURE.exists() else None
needs_fixture = pytest.mark.skipif(
    FX is None, reason="fixtures/zivot_andrews.json not present in this checkout"
)


def _call_from_case(series, case):
    y = np.asarray(series[case["series"]], dtype=float)
    kwargs = dict(regression=case["regression"], trim=case["trim"])
    if case["autolag"] is not None:
        kwargs["autolag"] = case["autolag"]
        if case["maxlag"] is not None:
            kwargs["max_lags"] = case["maxlag"]
    else:
        kwargs["autolag"] = None
        if case["lags"] is not None:
            kwargs["lags"] = case["lags"]
    return tsecon.zivot_andrews(y, **kwargs)


# --------------------------------------------------------------------------- #
# Fixture re-pin through the Python surface
# --------------------------------------------------------------------------- #
@needs_fixture
def test_fixture_cases_re_pin():
    for case in FX["za"]:
        got = _call_from_case(FX["series"], case)
        ctx = f"{case['series']}/{case['regression']}/trim={case['trim']}"
        assert got["stat"] == pytest.approx(case["stat"], rel=1e-10), ctx
        assert got["pvalue"] == pytest.approx(case["pvalue"], rel=1e-8, abs=1e-12), ctx
        assert got["break_index"] == case["bpidx"], ctx
        assert got["lags"] == case["baselag"], ctx
        assert got["nobs"] == case["nobs"], ctx
        for k in ("1%", "5%", "10%"):
            assert got["crit"][k] == pytest.approx(case["crit"][k], rel=1e-12), ctx
        assert got["regression"] == case["regression"], ctx
        assert got["trim"] == case["trim"], ctx


# --------------------------------------------------------------------------- #
# Live parity vs statsmodels (when available in the test venv)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regression", ["c", "t", "ct"])
def test_live_parity_vs_statsmodels(regression):
    stattools = pytest.importorskip("statsmodels.tsa.stattools")
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.standard_normal(150))
    y[80:] += 5.0
    ref = stattools.zivot_andrews(y, trim=0.15, regression=regression, autolag="aic")
    got = tsecon.zivot_andrews(y, regression=regression)
    assert got["stat"] == pytest.approx(ref[0], rel=1e-10)
    assert got["pvalue"] == pytest.approx(ref[1], rel=1e-8, abs=1e-12)
    assert got["lags"] == ref[3]
    assert got["break_index"] == ref[4]


# --------------------------------------------------------------------------- #
# Contract: break localization and argument handling
# --------------------------------------------------------------------------- #
def test_engineered_break_is_localized():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(200)
    y[100:] += 10.0  # last pre-break index = 99
    r = tsecon.zivot_andrews(y, regression="c")
    assert abs(r["break_index"] - 99) <= 1
    assert r["stat"] < r["crit"]["1%"]
    assert r["pvalue"] <= 0.01


def test_break_index_respects_trim_window():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(120))
    for trim in (0.05, 0.15, 0.25):
        r = tsecon.zivot_andrews(y, trim=trim, autolag=None, lags=1)
        trimcnt = int(len(y) * trim)
        assert trimcnt <= r["break_index"] <= len(y) - trimcnt - 1


def test_teaching_errors():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.standard_normal(100))
    with pytest.raises(ValueError, match="trim"):
        tsecon.zivot_andrews(y, trim=0.5)
    with pytest.raises(ValueError, match="autolag=None"):
        tsecon.zivot_andrews(y, lags=3)  # conflicts with default autolag="aic"
    with pytest.raises(ValueError, match="max_lags"):
        tsecon.zivot_andrews(y, autolag=None, max_lags=4)
    with pytest.raises(ValueError, match="regression"):
        tsecon.zivot_andrews(y, regression="nc")
    with pytest.raises(ValueError, match="autolag"):
        tsecon.zivot_andrews(y, autolag="hqic")
    with pytest.raises(ValueError, match="constant"):
        tsecon.zivot_andrews(np.full(100, 3.0))
    with pytest.raises(ValueError, match="non-finite"):
        bad = y.copy()
        bad[10] = np.nan
        tsecon.zivot_andrews(bad)
    with pytest.raises(ValueError):
        tsecon.zivot_andrews(y[:5])
    # Lag too large for the trim window: refused with an explanation.
    with pytest.raises(ValueError, match="trim"):
        tsecon.zivot_andrews(y, autolag=None, lags=15, trim=0.15)


def test_fixed_lag_and_bic_paths_run():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.standard_normal(150))
    fixed = tsecon.zivot_andrews(y, autolag=None, lags=4)
    assert fixed["lags"] == 4
    bic = tsecon.zivot_andrews(y, autolag="bic", max_lags=6)
    assert 0 <= bic["lags"] <= 6
    default = tsecon.zivot_andrews(y, autolag=None, trim=0.25)
    assert default["lags"] == int(12.0 * (len(y) / 100.0) ** 0.25)
