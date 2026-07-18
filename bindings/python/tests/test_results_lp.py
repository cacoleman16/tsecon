"""Tests for the local-projection Results facade (`tsecon.results._lp`)."""

from __future__ import annotations

import json
import pickle
from statistics import NormalDist

import numpy as np
import pytest

import tsecon
from tsecon.results._lp import LPResults

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


HORIZONS = 8


def _data(seed: int = 0, T: int = 240):
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + shock[t] + 0.3 * shock[t - 1] + 0.2 * rng.standard_normal()
    return y, shock


@pytest.fixture(scope="module")
def data():
    return _data()


@pytest.fixture(scope="module")
def raw(data):
    y, shock = data
    return tsecon.lp(y, shock, horizons=HORIZONS)


@pytest.fixture(scope="module")
def res(data):
    y, shock = data
    return LPResults.fit(y, shock, horizons=HORIZONS)


# --------------------------------------------------------------------------- #
# 1. backward compatibility: the object IS the dict it always was
# --------------------------------------------------------------------------- #
def test_is_a_dict(res):
    assert isinstance(res, dict)


def test_every_original_key_present_and_equal(res, raw):
    assert set(res) == set(raw) == {"horizons", "irf", "se"}
    for key, value in raw.items():
        np.testing.assert_array_equal(res[key], value)


def test_dict_star_unpacking_still_works(res):
    def consume(*, horizons, irf, se):
        return len(horizons), len(irf), len(se)

    assert consume(**res) == (HORIZONS + 1,) * 3


def test_to_dict_is_a_plain_dict(res):
    plain = res.to_dict()
    assert type(plain) is dict
    assert set(plain) == set(res)


def test_json_round_trip(res):
    payload = {k: v.tolist() for k, v in res.to_dict().items()}
    back = json.loads(json.dumps(payload))
    np.testing.assert_allclose(back["irf"], res["irf"])
    assert back["horizons"] == list(range(HORIZONS + 1))


def test_pickle_round_trip(res):
    back = pickle.loads(pickle.dumps(res))
    assert isinstance(back, LPResults)
    for key in res:
        np.testing.assert_array_equal(back[key], res[key])
    # metadata used by the summary survives too
    assert back._nobs == res._nobs
    assert back.summary() == res.summary()


# --------------------------------------------------------------------------- #
# 2. summary contents
# --------------------------------------------------------------------------- #
def test_summary_is_a_str_naming_the_estimator(res):
    text = res.summary()
    assert isinstance(text, str)
    assert "Local projection" in text
    assert "lag-augmented, HAC standard errors" in text


def test_summary_cites_lag_augmented_inference(res):
    assert "Montiel Olea & Plagborg-Moller 2021" in res.summary()


def test_summary_has_the_per_horizon_table_header(res):
    text = res.summary()
    header = [ln for ln in text.splitlines() if ln.strip().startswith("h ")]
    assert header, text
    assert "IRF" in header[0]
    assert "std err" in header[0]
    assert "[95% conf. int.]" in header[0]


def test_summary_reports_every_horizon_with_its_numbers(res):
    text = res.summary()
    lo, hi = res.conf_int(0.95)
    for i, h in enumerate(res.horizons):
        row = [
            ln
            for ln in text.splitlines()
            if ln.strip().startswith(f"{h} ") or ln.strip() == f"{h}"
        ]
        assert row, f"no row for horizon {h}"
        line = row[0]
        assert f"{res.irf[i]:+.5f}" in line
        assert f"{res.se[i]:.5f}" in line
        assert f"[{lo[i]:+.5f}, {hi[i]:+.5f}]" in line


def test_summary_reports_the_peak(res):
    h, v = res.peak()
    assert f"peak  h={h} ({v:+.5f})" in res.summary()


def test_summary_lines_stay_narrow(res):
    assert max(len(ln) for ln in res.summary().splitlines()) <= 72


def test_summary_level_argument_changes_the_band_label(res):
    assert "[90% conf. int.]" in res.summary(level=0.90)


# --------------------------------------------------------------------------- #
# 3. repr is the summary
# --------------------------------------------------------------------------- #
def test_repr_equals_summary(res):
    assert repr(res) == res.summary()


# --------------------------------------------------------------------------- #
# 4. conf_int
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
def test_conf_int_matches_irf_plus_minus_z_se(res, level):
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    lo, hi = res.conf_int(level)
    np.testing.assert_allclose(lo, res.irf - z * res.se, rtol=0, atol=0)
    np.testing.assert_allclose(hi, res.irf + z * res.se, rtol=0, atol=0)


def test_conf_int_default_is_95_percent(res):
    np.testing.assert_array_equal(res.conf_int()[0], res.conf_int(0.95)[0])


def test_conf_int_z_is_the_familiar_1_96(res):
    lo, hi = res.conf_int(0.95)
    width = (hi - lo) / (2.0 * res.se)
    np.testing.assert_allclose(width, 1.959964, atol=1e-6)


def test_conf_int_rejects_bad_levels(res):
    for bad in (0.0, 1.0, -0.1, 95):
        with pytest.raises(ValueError):
            res.conf_int(bad)


# --------------------------------------------------------------------------- #
# 5. peak
# --------------------------------------------------------------------------- #
def test_peak_matches_argmax_abs(res):
    h, v = res.peak()
    i = int(np.argmax(np.abs(res.irf)))
    assert h == int(res.horizons[i])
    assert v == pytest.approx(float(res.irf[i]))


def test_peak_finds_a_planted_negative_trough():
    planted = LPResults(
        {
            "horizons": np.arange(5),
            "irf": np.array([0.1, 0.2, -0.9, 0.3, 0.4]),
            "se": np.full(5, 0.1),
        }
    )
    assert planted.peak() == (2, -0.9)


def test_peak_ties_go_to_the_earliest_horizon():
    tied = LPResults(
        {
            "horizons": np.arange(3),
            "irf": np.array([0.5, -0.5, 0.5]),
            "se": np.full(3, 0.1),
        }
    )
    assert tied.peak()[0] == 0


# --------------------------------------------------------------------------- #
# 6. plot
# --------------------------------------------------------------------------- #
def test_plot_irf_returns_a_figure_with_band_line_and_zero_line(res):
    from matplotlib.figure import Figure

    fig = res.plot_irf()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]

    # the shaded confidence band
    assert len(ax.collections) == 1, "expected exactly one fill_between band"
    band = ax.collections[0]
    ys = band.get_paths()[0].vertices[:, 1]
    lo, hi = res.conf_int(0.95)
    assert ys.min() == pytest.approx(lo.min())
    assert ys.max() == pytest.approx(hi.max())

    # the IRF itself is drawn, and there is a horizontal zero reference
    ydata = [np.asarray(ln.get_ydata(), dtype=float) for ln in ax.lines]
    assert any(y.size == res.irf.size and np.allclose(y, res.irf) for y in ydata)
    assert any(
        np.asarray(y, dtype=float).size == 2 and np.allclose(y, 0.0) for y in ydata
    )
    matplotlib.pyplot.close(fig)


def test_plot_irf_honours_the_level(res):
    fig = res.plot_irf(level=0.68)
    ys = fig.axes[0].collections[0].get_paths()[0].vertices[:, 1]
    lo, hi = res.conf_int(0.68)
    assert ys.min() == pytest.approx(lo.min())
    assert ys.max() == pytest.approx(hi.max())
    matplotlib.pyplot.close(fig)


def test_plot_irf_accepts_an_ax_and_saves_to_path(res, tmp_path):
    fig_in, ax = matplotlib.pyplot.subplots()
    out = tmp_path / "irf.png"
    fig = res.plot_irf(ax=ax, path=str(out))
    assert fig is fig_in
    assert out.exists() and out.stat().st_size > 0
    matplotlib.pyplot.close(fig)


def test_plot_irf_has_no_wide_side_margins(res):
    fig = res.plot_irf()
    lo, hi = fig.axes[0].get_xlim()
    assert (lo, hi) == (float(res.horizons[0]), float(res.horizons[-1]))
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# 7. kwargs pass-through and family accessors
# --------------------------------------------------------------------------- #
def test_fit_passes_kwargs_through_to_tsecon_lp(data):
    y, shock = data
    kw = dict(horizons=HORIZONS, n_lag_controls=4, se="hac", cumulative=True)
    res = LPResults.fit(y, shock, **kw)
    expected = tsecon.lp(y, shock, **kw)
    for key, value in expected.items():
        np.testing.assert_array_equal(res[key], value)
    text = res.summary()
    assert "cumulative" in text
    assert "HAC (Newey-West) standard errors" in text
    assert "Montiel Olea" not in text
    assert "lag controls  4" in text


def test_accessor_properties_mirror_the_dict(res):
    np.testing.assert_array_equal(res.irf, res["irf"])
    np.testing.assert_array_equal(res.se, res["se"])
    np.testing.assert_array_equal(res.horizons, np.asarray(res["horizons"]).astype(int))
    assert res.horizons.tolist() == list(range(HORIZONS + 1))
