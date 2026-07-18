"""VAR results facade: the dict/list contracts, the summary, and the IRF grid."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

import tsecon
from tsecon.results._var import CoefficientFrame, IRFArray, VARResults, var_irf

RAW_KEYS = {
    "params",
    "sigma_u",
    "llf",
    "aic",
    "bic",
    "hqic",
    "max_root",
    "min_root",
    "is_stable",
}


def stationary_data(n: int = 240, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a1 = np.array([[0.5, 0.10], [0.20, 0.40]])
    a2 = np.array([[0.10, -0.05], [0.00, 0.15]])
    y = np.zeros((n, 2))
    for t in range(2, n):
        y[t] = a1 @ y[t - 1] + a2 @ y[t - 2] + rng.normal(size=2) * 0.5
    return y


def explosive_data(n: int = 80, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = np.array([[1.25, 0.0], [0.0, 0.4]])
    z = np.zeros((n, 2))
    for t in range(1, n):
        z[t] = b @ z[t - 1] + rng.normal(size=2) * 0.1
    return z


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return stationary_data()


@pytest.fixture(scope="module")
def res(data) -> VARResults:
    return VARResults.fit(data, lags=2, trend="c", names=["gdp", "infl"])


# --------------------------------------------------------------------------- #
# 1. backward compatibility: it is still the dict var_fit always returned
# --------------------------------------------------------------------------- #
def test_is_a_dict_with_every_original_key_unchanged(data, res):
    raw = tsecon.var_fit(data, 2, "c")

    assert isinstance(res, dict)
    assert set(res) == RAW_KEYS == set(raw)
    for key in raw:
        got, want = res[key], raw[key]
        if isinstance(want, bool):
            assert got is want, key
        elif isinstance(want, float):
            assert got == pytest.approx(want), key
        else:
            assert np.asarray(got, dtype=float) == pytest.approx(np.asarray(want, dtype=float)), key
    # the values are the very objects var_fit produced, not reshaped copies
    assert np.asarray(res["params"]).shape == (1 + 2 * 2, 2)
    assert np.asarray(res["sigma_u"]).shape == (2, 2)


def test_data_and_labels_are_attributes_not_dict_keys(res):
    assert "data" not in res and "_data" not in res
    assert "names" not in res and "lags" not in res
    assert res.names == ["gdp", "infl"]
    assert (res.lags, res.neqs, res.trend) == (2, 2, "c")


def test_dict_conveniences_still_work(res):
    plain = res.to_dict()
    assert type(plain) is dict and set(plain) == RAW_KEYS
    text = json.dumps({k: (v if np.ndim(v) == 0 else np.asarray(v).tolist())
                       for k, v in plain.items()})
    assert json.loads(text)["is_stable"] is True

    unpacked = {**res}
    assert set(unpacked) == RAW_KEYS

    revived = pickle.loads(pickle.dumps(res))
    assert isinstance(revived, VARResults)
    assert set(revived) == RAW_KEYS
    assert revived["llf"] == pytest.approx(res["llf"])
    assert revived.names == res.names and revived.lags == res.lags


# --------------------------------------------------------------------------- #
# 2. summary
# --------------------------------------------------------------------------- #
def test_summary_shows_header_fit_stats_and_coefficient_matrix(res):
    text = res.summary()
    assert isinstance(text, str)

    assert "VAR(2) — 2 equations" in text
    assert "stable" in text and "UNSTABLE" not in text

    for key in ("llf", "aic", "bic", "hqic"):
        assert key in text
    assert f"{res['llf']:.3f}" in text
    assert f"{res['aic']:.4f}" in text
    assert f"{res['min_root']:.4f}" in text

    # row labels: deterministic term, then L<p>.<name>
    for label in ("const", "L1.gdp", "L1.infl", "L2.gdp", "L2.infl"):
        assert label in text
    # equation columns
    header = [ln for ln in text.splitlines() if ln.startswith("regressor")][0]
    assert header.split()[1:] == ["gdp", "infl"]

    params = np.asarray(res["params"], dtype=float)
    for row in params:
        for value in row:
            assert f"{value:+.5f}" in text

    assert max(len(ln) for ln in text.splitlines()) <= 72


def test_summary_says_unstable_for_an_explosive_var():
    unstable = VARResults.fit(explosive_data(), lags=1, trend="c")
    assert unstable["is_stable"] is False
    text = unstable.summary()
    assert "UNSTABLE" in text
    # the verdict comes from is_stable, not from max_root (which is > 1 here)
    assert unstable["max_root"] > 1.0
    assert unstable["min_root"] < 1.0

    stable_text = VARResults.fit(stationary_data(), lags=2).summary()
    assert "UNSTABLE" not in stable_text and "stable" in stable_text


def test_repr_is_the_summary(res):
    assert repr(res) == res.summary()


# --------------------------------------------------------------------------- #
# 3. accessors
# --------------------------------------------------------------------------- #
def test_coefficient_frame_labels_and_values(res):
    frame = res.coefficient_frame()
    assert isinstance(frame, CoefficientFrame)
    assert frame.rows == ["const", "L1.gdp", "L1.infl", "L2.gdp", "L2.infl"]
    assert frame.columns == ["gdp", "infl"]
    assert frame.values == pytest.approx(np.asarray(res["params"], dtype=float))
    assert frame.values.shape == (len(frame.rows), len(frame.columns))


def test_regressor_labels_track_lags_and_trend(data):
    no_const = VARResults.fit(data, lags=1, trend="n", names=["a", "b"])
    assert no_const.regressor_labels() == ["L1.a", "L1.b"]
    assert np.asarray(no_const["params"]).shape == (2, 2)


def test_default_names_and_name_validation(data):
    assert VARResults.fit(data, lags=1).names == ["y1", "y2"]
    with pytest.raises(ValueError, match="names has 3 entries"):
        VARResults.fit(data, lags=1, names=["a", "b", "c"])
    with pytest.raises(ValueError, match="2-D"):
        VARResults.fit(data[:, 0], lags=1)


def test_stable_property_mirrors_is_stable(res):
    assert res.stable is res["is_stable"] is True


# --------------------------------------------------------------------------- #
# 4. IRFArray is still a list
# --------------------------------------------------------------------------- #
def test_irf_array_matches_the_raw_nested_list(data):
    horizon = 8
    raw = tsecon.var_irf(data, lags=2, horizon=horizon, orth=True)
    irf = var_irf(data, lags=2, horizon=horizon, orth=True, names=["gdp", "infl"])

    assert isinstance(irf, list)
    assert len(irf) == len(raw) == horizon + 1
    assert np.array(irf).shape == (horizon + 1, 2, 2)
    assert np.array(irf) == pytest.approx(np.array(raw))
    assert irf[3][1][0] == pytest.approx(raw[3][1][0])
    assert list(irf) == raw
    assert np.array(irf[:2]).shape == (2, 2, 2)


def test_irf_response_paths_by_index_and_name(data):
    irf = var_irf(data, lags=2, horizon=6, names=["gdp", "infl"])
    path = irf.response(0, 1)
    assert path.shape == (7,)
    assert path == pytest.approx(irf.response("gdp", "infl"))
    assert path == pytest.approx(np.array(irf)[:, 0, 1])
    # orthogonalised: variable 0 cannot respond to shock 1 on impact
    assert path[0] == pytest.approx(0.0)
    # own impact response is the Cholesky diagonal, hence strictly positive
    assert irf.response("infl", "infl")[0] > 0
    with pytest.raises(KeyError):
        irf.response("nope", 0)

    assert irf.names == ["gdp", "infl"]
    assert (irf.neqs, irf.horizon, irf.orth) == (2, 6, True)
    assert repr(irf) == irf.summary()
    assert "orthogonalised" in repr(irf) and "horizons 0..6" in repr(irf)


def test_results_irf_convenience_uses_the_fitted_data(data, res):
    from_results = res.irf(horizon=5)
    assert isinstance(from_results, IRFArray)
    assert from_results.names == ["gdp", "infl"]
    assert np.array(from_results) == pytest.approx(
        np.array(tsecon.var_irf(data, lags=2, horizon=5, orth=True))
    )

    orphan = VARResults(tsecon.var_fit(data, 2, "c"))
    orphan.lags, orphan.trend, orphan.neqs = 2, "c", 2
    orphan.names = ["gdp", "infl"]
    with pytest.raises(ValueError, match="not built by VARResults.fit"):
        orphan.irf()


def test_irf_array_pickles_and_json_round_trips(data):
    irf = var_irf(data, lags=2, horizon=4, names=["gdp", "infl"])
    revived = pickle.loads(pickle.dumps(irf))
    assert isinstance(revived, list)
    assert np.array(revived) == pytest.approx(np.array(irf))
    assert json.loads(json.dumps(list(irf))) == list(irf)


# --------------------------------------------------------------------------- #
# 5. plotting
# --------------------------------------------------------------------------- #
def test_irf_plot_builds_a_k_by_k_grid(data, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    irf = var_irf(data, lags=2, horizon=10, names=["gdp", "infl"])
    fig = irf.plot()
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4  # k x k small multiples

        for i in range(2):
            for j in range(2):
                ax = fig.axes[i * 2 + j]
                # the always-drawn zero reference line, then the response line
                assert len(ax.lines) == 2
                assert ax.lines[0].get_ydata() == pytest.approx([0.0, 0.0])
                assert ax.lines[-1].get_ydata() == pytest.approx(irf.response(i, j))
                assert ax.lines[-1].get_xdata() == pytest.approx(np.arange(11))
                assert ax.get_title() == f"{irf.names[i]} ← shock {irf.names[j]}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    out = tmp_path / "irf.png"
    fig2 = irf.plot(path=out)
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig2)


def test_plot_import_error_names_the_plots_extra(monkeypatch, data):
    import builtins

    from tsecon.results import _plotting

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"tsecon\[plots\]"):
        _plotting.pyplot()
