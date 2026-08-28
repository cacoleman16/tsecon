"""Tests for the pseudo-out-of-sample backtest engine binding.

There is no external golden here: the backtest orchestration is checked
analytically against numpy, since for the naive forecaster every quantity
is a closed form of the series.
"""
import numpy as np
import pytest
import tsecon


def test_naive_expanding_backtest_semantics():
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0], dtype=float)
    res = tsecon.backtest(
        y, window="expanding", train=3, horizon=1, forecaster="naive"
    )
    # Origins run from train-1 .. n-horizon-1 inclusive.
    n = len(y)
    expected_origins = list(range(3 - 1, n - 1))
    assert list(res["origins"]) == expected_origins
    assert res["n_origins"] == len(expected_origins)
    # Naive h=1 forecast at origin t is y[t]; target is y[t+1].
    fc = np.array(res["forecasts"][0])
    tg = np.array(res["targets"][0])
    np.testing.assert_allclose(fc, y[expected_origins])
    np.testing.assert_allclose(tg, y[np.array(expected_origins) + 1])
    # Accuracy ME matches numpy mean of (target - forecast).
    row = res["accuracy"][0]
    np.testing.assert_allclose(row["me"], np.mean(tg - fc))
    np.testing.assert_allclose(row["rmse"], np.sqrt(np.mean((tg - fc) ** 2)))
    assert row["name"] == "h=1"


def test_rolling_window_and_multi_horizon():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.standard_normal(60)) + 100.0
    res = tsecon.backtest(
        y, window="rolling", train=20, horizon=3, forecaster="drift"
    )
    assert res["horizon"] == 3
    assert len(res["forecasts"]) == 3
    assert len(res["accuracy"]) == 3
    # First rolling origin is width-1; targets exist through n-horizon.
    assert res["origins"][0] == 20 - 1
    assert res["origins"][-1] == len(y) - 3 - 1
    for h in range(3):
        assert len(res["forecasts"][h]) == res["n_origins"]


def test_theta_forecaster_runs():
    rng = np.random.default_rng(1)
    y = np.linspace(0, 10, 80) + rng.standard_normal(80) * 0.1
    res = tsecon.backtest(
        y, window="expanding", train=40, horizon=2, forecaster="theta", period=1
    )
    assert res["n_origins"] > 0
    assert np.all(np.isfinite(res["forecasts"][0]))


def test_unknown_forecaster_and_window_error():
    y = np.arange(30.0)
    with pytest.raises(ValueError):
        tsecon.backtest(y, forecaster="nope")
    with pytest.raises(ValueError):
        tsecon.backtest(y, window="sideways")


# --------------------------------------------------------------------------
# Audit round 10: period is refused where it cannot act
# --------------------------------------------------------------------------

def test_period_refused_for_nonseasonal_forecasters_and_callables():
    """period feeds only the seasonal built-ins (seasonal_naive/theta) and
    never reaches a callable; verified fully inert for naive/drift/mean and
    callables before the refusal landed (bit-identical to the default call,
    accuracy table included - MASE/RMSSE scale with insample_period, not
    period)."""
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.standard_normal(60)) + 50.0
    for fc in ("naive", "drift", "mean"):
        with pytest.raises(ValueError, match="period") as exc:
            tsecon.backtest(y, forecaster=fc, period=4)
        msg = str(exc.value)
        assert fc in msg and "insample_period" in msg and "seasonal_naive" in msg
    # The default forecaster is naive, so an explicit period alone raises too.
    with pytest.raises(ValueError, match="period"):
        tsecon.backtest(y, period=4)
    # Callables: the contract is forecaster(train, horizon) - no period slot.
    with pytest.raises(ValueError, match="period") as exc:
        tsecon.backtest(y, forecaster=lambda t, h: [t[-1]] * h, period=4)
    assert "callable" in str(exc.value) and "insample_period" in str(exc.value)
    # A typo'd forecaster still gets the unknown-forecaster error first.
    with pytest.raises(ValueError, match="unknown forecaster"):
        tsecon.backtest(y, forecaster="nope", period=4)


def test_period_sentinel_default_bit_identical_and_live_where_seasonal():
    rng = np.random.default_rng(9)
    t = np.arange(80.0)
    y = 10 + 0.05 * t + np.sin(t * 2 * np.pi / 4) + 0.2 * rng.standard_normal(80)
    # Sentinel resolution: omitted == the historical explicit period=1,
    # for a forecaster where period stays legal.
    a = tsecon.backtest(y, train=40, forecaster="theta")
    b = tsecon.backtest(y, train=40, forecaster="theta", period=1)
    np.testing.assert_array_equal(a["forecasts"][0], b["forecasts"][0])
    assert a["accuracy"][0] == b["accuracy"][0]
    # Live where documented: the seasonal forecasters move with period.
    s4 = tsecon.backtest(y, train=40, forecaster="seasonal_naive", period=4)
    s2 = tsecon.backtest(y, train=40, forecaster="seasonal_naive", period=2)
    assert not np.array_equal(s4["forecasts"][0], s2["forecasts"][0])
    t4 = tsecon.backtest(y, train=40, forecaster="theta", period=4)
    assert not np.array_equal(a["forecasts"][0], t4["forecasts"][0])
    # insample_period (the MASE/RMSSE scale) stays live for EVERY forecaster.
    n1 = tsecon.backtest(y, train=40, forecaster="naive", insample_period=1)
    n4 = tsecon.backtest(y, train=40, forecaster="naive", insample_period=4)
    np.testing.assert_array_equal(n1["forecasts"][0], n4["forecasts"][0])
    assert n1["accuracy"][0]["mase"] != n4["accuracy"][0]["mase"]


def test_constant_first_window_mase_error_names_the_real_cure():
    """The zero-scale refusal used to advise "an unscaled measure" no
    backtest parameter selects; it now names the honest remedies (a first
    training window that varies - train= - or insample_period)."""
    rng = np.random.default_rng(11)
    y = np.concatenate([np.full(20, 5.0), rng.standard_normal(30) + 5.0])
    with pytest.raises(ValueError, match="MASE") as exc:
        tsecon.backtest(y, train=20)
    msg = str(exc.value)
    assert "train=" in msg and "insample_period" in msg
    assert "use a different period or an unscaled measure" not in msg
    # And the named cure works: a longer first window varies, so the same
    # series backtests cleanly.
    assert tsecon.backtest(y, train=25)["n_origins"] > 0
