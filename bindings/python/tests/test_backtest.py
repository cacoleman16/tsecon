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
