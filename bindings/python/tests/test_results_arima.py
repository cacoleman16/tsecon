"""Tests for the ARIMA Results facade (`tsecon.results._arima`).

The load-bearing claim is that `ARIMAResults` is *additive*: it is a dict that
still holds exactly what `tsecon.arima_fit` returned, and only gains
`.summary()`, `.forecast_frame()` and `.plot_forecast()` on top. These tests
check the dict contract key by key against a raw call, then the rendering.
"""

from __future__ import annotations

import json
import pickle
from statistics import NormalDist

import numpy as np
import pytest

import tsecon
from tsecon.results._arima import ARIMAResults

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

P, D, Q = 1, 0, 1
STEPS = 12
ALPHA = 0.05


def _series(seed: int = 0, n: int = 240):
    """An ARMA(1,1) path, so the fitted model is the true one."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + e[t] + 0.3 * e[t - 1]
    return y


@pytest.fixture(scope="module")
def y():
    return _series()


@pytest.fixture(scope="module")
def raw(y):
    return tsecon.arima_fit(y, P, D, Q, forecast_steps=STEPS, conf_alpha=ALPHA)


@pytest.fixture(scope="module")
def res(y):
    return ARIMAResults.fit(y, P, D, Q, forecast_steps=STEPS, conf_alpha=ALPHA)


# --------------------------------------------------------------------------- #
# 1. backward compatibility: it IS the dict
# --------------------------------------------------------------------------- #
def test_is_a_dict(res):
    assert isinstance(res, dict)
    assert isinstance(res, ARIMAResults)


def test_every_raw_key_survives_unchanged(raw, res):
    assert set(res) == set(raw), "the facade must not add or drop dict keys"
    for key, expected in raw.items():
        got = res[key]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(got, expected)
        else:
            assert got == expected, key


def test_indexing_and_unpacking_still_work(res):
    assert res["params"].shape == (len(res["param_names"]),)
    spread = {**res}
    assert type(spread) is dict
    assert set(spread) == set(res)
    assert type(res.to_dict()) is dict


def test_json_round_trip_of_to_dict(res):
    plain = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in res.to_dict().items()
    }
    back = json.loads(json.dumps(plain))
    assert back["param_names"] == list(res["param_names"])
    assert back["aic"] == pytest.approx(res["aic"])
    np.testing.assert_allclose(back["forecast_mean"], res["forecast_mean"])


def test_pickle_round_trip_keeps_keys_and_attributes(res):
    back = pickle.loads(pickle.dumps(res))
    assert isinstance(back, ARIMAResults)
    assert set(back) == set(res)
    np.testing.assert_array_equal(back["params"], res["params"])
    assert back.order == (P, D, Q)
    np.testing.assert_array_equal(back.y, res.y)
    assert back.summary() == res.summary()


def test_input_series_is_an_attribute_not_a_key(res, y):
    assert "y" not in res and "series" not in res
    np.testing.assert_array_equal(res.y, y)


# --------------------------------------------------------------------------- #
# 2. summary content
# --------------------------------------------------------------------------- #
def test_summary_shows_order_fit_stats_params_and_nobs(res):
    text = res.summary()
    assert isinstance(text, str)

    assert f"ARIMA({P},{D},{Q})" in text
    assert f"{res['loglik']:.4f}" in text
    assert f"{res['aic']:.3f}" in text
    assert f"{res['bic']:.3f}" in text
    assert "No. Observations" in text and "240" in text
    assert f"Forecast steps {STEPS:>4d}" in text
    assert "Residual s.d." in text

    # every parameter, by name and by value, in the coefficient block
    for name, value in zip(res["param_names"], res["params"]):
        assert name in text
        assert f"{value:+.5f}" in text

    assert max(len(line) for line in text.splitlines()) <= 72


def test_repr_is_the_summary(res):
    assert repr(res) == res.summary()


# --------------------------------------------------------------------------- #
# 3. forecast_frame
# --------------------------------------------------------------------------- #
def test_forecast_frame_fields_and_stored_bands(res):
    frame = res.forecast_frame()
    assert frame.dtype.names == ("step", "mean", "se", "lower", "upper")
    assert len(frame) == STEPS
    np.testing.assert_array_equal(frame["step"], np.arange(1, STEPS + 1))
    np.testing.assert_allclose(frame["mean"], res["forecast_mean"])
    np.testing.assert_allclose(frame["se"], res["forecast_se"])
    # level 0.95 matches conf_alpha=0.05, so the compiled bands are reused
    np.testing.assert_allclose(frame["lower"], res["forecast_lower"])
    np.testing.assert_allclose(frame["upper"], res["forecast_upper"])
    assert np.all(frame["lower"] < frame["mean"])
    assert np.all(frame["upper"] > frame["mean"])


def test_forecast_frame_recomputes_other_levels(res):
    frame = res.forecast_frame(level=0.80)
    z = NormalDist().inv_cdf(0.90)
    np.testing.assert_allclose(frame["lower"], res["forecast_mean"] - z * res["forecast_se"])
    np.testing.assert_allclose(frame["upper"], res["forecast_mean"] + z * res["forecast_se"])
    # narrower than the 95% band
    assert np.all(frame["upper"] - frame["lower"] < res["forecast_upper"] - res["forecast_lower"])


def test_forecast_frame_rejects_impossible_level(res):
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        res.forecast_frame(level=1.0)


# --------------------------------------------------------------------------- #
# 4. plotting
# --------------------------------------------------------------------------- #
def test_plot_forecast_returns_a_fan_chart(res):
    from matplotlib.figure import Figure

    fig = res.plot_forecast()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]

    # observed history, forecast mean, and the forecast-origin rule
    assert len(ax.lines) == 3
    assert [ln.get_label() for ln in ax.lines[:2]] == ["observed", "forecast"]
    assert ax.lines[2].get_xdata()[0] == len(res.y) - 1

    # three nested bands (50/80/95)
    assert len(ax.collections) == 3
    band_labels = [c.get_label() for c in ax.collections]
    assert band_labels == ["95%", "80%", "50%"]

    # the forecast line starts at the last observation (the fan is anchored)
    fx, fy = ax.lines[1].get_data()
    assert len(fx) == STEPS + 1
    assert fy[0] == pytest.approx(res.y[-1])
    np.testing.assert_allclose(fy[1:], res["forecast_mean"])

    matplotlib.pyplot.close(fig)


def test_plot_forecast_windows_history_and_accepts_ax_and_path(res, tmp_path):
    plt = matplotlib.pyplot
    fig, ax = plt.subplots()
    out = tmp_path / "fan.png"
    got = res.plot_forecast(ax=ax, path=out)
    assert got is fig
    assert out.exists() and out.stat().st_size > 0

    # default window keeps max(40, 6h) trailing observations, on real indices
    hx, hy = ax.lines[0].get_data()
    assert len(hx) == max(40, 6 * STEPS)
    assert hx[-1] == len(res.y) - 1
    np.testing.assert_allclose(hy, res.y[-len(hx):])
    plt.close(fig)


def test_plot_forecast_full_history_and_explicit_y(res, y):
    plt = matplotlib.pyplot
    fig = res.plot_forecast(y=y, max_history=None)
    hx, _ = fig.axes[0].lines[0].get_data()
    assert len(hx) == len(y)
    plt.close(fig)


def test_plot_forecast_level_controls_the_outer_band(res):
    plt = matplotlib.pyplot
    fig = res.plot_forecast(level=0.80)
    ax = fig.axes[0]
    assert [c.get_label() for c in ax.collections] == ["80%", "50%"]
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. the no-forecast fit
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def res_nofc(y):
    return ARIMAResults.fit(y, P, D, Q)


def test_no_forecast_accessors(res_nofc):
    assert res_nofc.has_forecast is False
    assert res_nofc.forecast_steps == 0
    assert "forecast_mean" not in res_nofc
    assert "Forecast steps    0" in res_nofc.summary()


def test_no_forecast_plot_and_frame_teach_the_fix(res_nofc):
    for call in (res_nofc.plot_forecast, res_nofc.forecast_frame):
        with pytest.raises(ValueError) as exc:
            call()
        msg = str(exc.value)
        assert "forecast_steps" in msg
        assert f"ARIMAResults.fit(y, {P}, {D}, {Q}, forecast_steps=12)" in msg


# --------------------------------------------------------------------------- #
# 6. family-specific accessors
# --------------------------------------------------------------------------- #
def test_order_nobs_and_params_dict(res, y):
    assert res.order == (P, D, Q)
    assert res.nobs == len(y)
    pd_ = res.params_dict()
    assert list(pd_) == list(res["param_names"])
    assert pd_["ar.L1"] == pytest.approx(res["params"][1])
    assert set(pd_) >= {"ar.L1", "ma.L1", "sigma2"}


def test_differenced_fit_reports_the_right_order_and_nobs():
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.standard_normal(150)) + 20.0
    res = ARIMAResults.fit(y, 1, 1, 0, forecast_steps=6, conf_alpha=0.10)
    assert res.order == (1, 1, 0)
    assert res.nobs == 150
    assert len(res["residuals"]) == 149  # simple differencing loses d
    assert "ARIMA(1,1,0)" in res.summary()
    # conf_alpha=0.10 -> the stored bands are the 90% ones
    np.testing.assert_allclose(res.forecast_frame(level=0.90)["lower"], res["forecast_lower"])


def test_nobs_falls_back_to_residuals_when_series_absent(raw):
    detached = ARIMAResults(raw, order=(P, D, Q))
    assert detached.y is None
    assert detached.nobs == len(raw["residuals"])
    assert "ARIMA(1,0,1)" in detached.summary()
