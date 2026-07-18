"""Tests for :class:`tsecon.results._dsge.DSGEResults`.

The benchmark is the Cagan money-demand model, which has a closed form:

    A E_t[y_{t+1}] = B y_t + C z_{t+1},   y = (m_t, p_t),  z = money innovation
    A = [[1, 0], [0, a]],  B = [[rho, 0], [-1, 1]],  C = [[1], [0]]

with one predetermined variable (the money stock). Then

    G = 1 / (1 - a*rho),  P = rho,  Q = 1,
    eigenvalue moduli = {rho, 1/a}

so the price level jumps by G on impact and both series decay at rate rho.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

import tsecon
from tsecon.results._dsge import DSGEResults

A_COEF = 0.7  # semi-elasticity of money demand
RHO = 0.6  # persistence of the money-supply process

G_CLOSED = 1.0 / (1.0 - A_COEF * RHO)  # 1.7241379310344827
UNSTABLE_ROOT = 1.0 / A_COEF  # 1.4285714285714286


def cagan_model():
    a = np.array([[1.0, 0.0], [0.0, A_COEF]])
    b = np.array([[RHO, 0.0], [-1.0, 1.0]])
    c = np.array([[1.0], [0.0]])
    return a, b, c, 1


def two_block_model():
    """Two independent Cagan blocks — 2 states, 2 jumps, 2 shocks."""
    a = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.7, 0], [0, 0, 0, 0.5]], dtype=float
    )
    b = np.array(
        [[0.6, 0, 0, 0], [0, 0.3, 0, 0], [-1, 0, 1, 0], [0, -1, 0, 1]], dtype=float
    )
    c = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=float)
    return a, b, c, 2


@pytest.fixture(scope="module")
def raw():
    a, b, c, n = cagan_model()
    return tsecon.dsge_solve(a, b, c, n)


@pytest.fixture(scope="module")
def res():
    return DSGEResults.solve(*cagan_model())


@pytest.fixture(scope="module")
def res2():
    return DSGEResults.solve(*two_block_model())


def to_jsonable(d):
    return {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()
    }


# --------------------------------------------------------------------------- #
# 1. backward compatibility: it IS the dict the compiled function returned
# --------------------------------------------------------------------------- #
def test_is_a_dict(res):
    assert isinstance(res, dict)
    assert isinstance(res, DSGEResults)


def test_every_raw_key_present_and_equal(res, raw):
    assert set(res) == set(raw) == {"g", "p", "q", "eigenvalue_moduli", "verdict"}
    assert res["g"] == raw["g"]
    assert res["p"] == raw["p"]
    assert res["q"] == raw["q"]
    assert res["verdict"] == raw["verdict"]
    np.testing.assert_allclose(res["eigenvalue_moduli"], raw["eigenvalue_moduli"])


def test_key_values_match_the_closed_form(res):
    assert res["g"][0][0] == pytest.approx(G_CLOSED)
    assert res["p"][0][0] == pytest.approx(RHO)
    assert res["q"][0][0] == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.sort(res["eigenvalue_moduli"]), [RHO, UNSTABLE_ROOT], rtol=1e-10
    )
    assert res["verdict"].startswith("unique stable solution")


def test_dict_unpacking_and_to_dict(res):
    plain = res.to_dict()
    assert type(plain) is dict
    assert plain.keys() == res.keys()

    def consume(*, g, p, q, eigenvalue_moduli, verdict):
        return g, p, q, eigenvalue_moduli, verdict

    g, p, q, mod, verdict = consume(**res)
    assert g == res["g"] and p == res["p"] and q == res["q"]
    assert verdict == res["verdict"]
    assert len(mod) == 2


def test_json_round_trip(res):
    payload = json.dumps(to_jsonable(res.to_dict()))
    back = json.loads(payload)
    assert back["g"][0][0] == pytest.approx(G_CLOSED)
    assert back["verdict"] == res["verdict"]
    np.testing.assert_allclose(
        back["eigenvalue_moduli"], np.asarray(res["eigenvalue_moduli"])
    )


def test_pickle_round_trip(res):
    back = pickle.loads(pickle.dumps(res))
    assert isinstance(back, DSGEResults)
    assert back["g"] == res["g"]
    assert back["verdict"] == res["verdict"]
    np.testing.assert_allclose(back["eigenvalue_moduli"], res["eigenvalue_moduli"])
    assert back.summary() == res.summary()


# --------------------------------------------------------------------------- #
# 2 & 3. summary text and repr
# --------------------------------------------------------------------------- #
def test_summary_headline_is_the_verdict(res):
    text = res.summary()
    assert isinstance(text, str)
    assert "Linear RE model (Blanchard-Kahn): unique stable solution" in text
    # the full verdict, including the counts, appears too
    assert "1 unstable eigenvalue(s) = 1 jump variable(s)" in text.replace("\n         ", " ")
    assert "determinate yes" in text


def test_summary_splits_the_eigenvalues_with_counts(res):
    text = res.summary()
    assert "stable (<1) 1" in text
    assert "unstable (>1) 1" in text
    assert "0.60000" in text  # rho, inside the unit circle
    assert "1.42857" in text  # 1/a, outside it


def test_summary_prints_the_three_matrices(res):
    text = res.summary()
    assert "G  policy: jump = G . predetermined" in text
    assert "P  transition: k(t+1) = P . k(t) + Q . z" in text
    assert "Q  impact: shock loading on the state" in text
    assert "[1x1]" in text
    assert "+1.72414" in text  # G
    assert "+0.60000" in text  # P
    assert "+1.00000" in text  # Q


def test_summary_dimensions_line(res2):
    text = res2.summary()
    assert "predetermined 2" in text
    assert "jump 2" in text
    assert "shocks 2" in text
    assert "[2x2]" in text


def test_summary_stays_inside_72_columns(res, res2):
    for text in (res.summary(), res2.summary()):
        assert max(len(line) for line in text.splitlines()) <= 72


def test_repr_is_the_summary(res):
    assert repr(res) == res.summary()


# --------------------------------------------------------------------------- #
# 5. family accessors
# --------------------------------------------------------------------------- #
def test_matrix_accessors_are_2d_arrays(res):
    assert res.policy().shape == (1, 1)
    assert res.transition().shape == (1, 1)
    assert res.impact().shape == (1, 1)
    assert res.policy()[0, 0] == pytest.approx(G_CLOSED)
    assert res.transition()[0, 0] == pytest.approx(RHO)
    assert res.impact()[0, 0] == pytest.approx(1.0)


def test_dimension_properties(res, res2):
    assert (res.n_predetermined, res.n_jump, res.n_shocks) == (1, 1, 1)
    assert (res2.n_predetermined, res2.n_jump, res2.n_shocks) == (2, 2, 2)


def test_moduli_split_at_the_unit_circle(res, res2):
    np.testing.assert_allclose(res.stable_moduli(), [RHO], rtol=1e-10)
    np.testing.assert_allclose(res.unstable_moduli(), [UNSTABLE_ROOT], rtol=1e-10)
    # the split is a partition of everything the solver returned
    assert res.stable_moduli().size + res.unstable_moduli().size == res.moduli().size
    np.testing.assert_allclose(res2.stable_moduli(), [0.3, 0.6], rtol=1e-10)
    np.testing.assert_allclose(res2.unstable_moduli(), [1 / 0.7, 2.0], rtol=1e-10)


def test_is_determinate(res):
    assert res.is_determinate() is True
    # verdict drives the predicate, nothing else
    faked = DSGEResults(dict(res, verdict="indeterminate: 0 unstable < 1 jump"))
    assert faked.is_determinate() is False


# --------------------------------------------------------------------------- #
# impulse response: the closed-form saddle path
# --------------------------------------------------------------------------- #
def test_impulse_response_shape_and_keys(res):
    irf = res.impulse_response(horizon=10)
    assert set(irf) == {"horizon", "shock", "predetermined", "jump"}
    assert irf["horizon"] == 10
    assert irf["predetermined"].shape == (10, 1)
    assert irf["jump"].shape == (10, 1)
    np.testing.assert_allclose(irf["shock"], [1.0])


def test_impulse_response_decays_at_rho(res):
    irf = res.impulse_response(horizon=15)
    k = irf["predetermined"][:, 0]
    np.testing.assert_allclose(k, RHO ** np.arange(15), rtol=1e-10, atol=1e-14)
    # ratio of successive periods is exactly rho
    np.testing.assert_allclose(k[1:] / k[:-1], np.full(14, RHO), rtol=1e-10)


def test_jump_is_g_times_the_state_at_every_horizon(res):
    irf = res.impulse_response(horizon=15)
    k, x = irf["predetermined"][:, 0], irf["jump"][:, 0]
    np.testing.assert_allclose(x, G_CLOSED * k, rtol=1e-10, atol=1e-14)
    assert x[0] == pytest.approx(G_CLOSED)  # impact jump in the price level


def test_impulse_response_scales_with_the_shock(res):
    base = res.impulse_response(horizon=6)
    doubled = res.impulse_response(horizon=6, shock=[2.0])
    np.testing.assert_allclose(
        doubled["predetermined"], 2.0 * base["predetermined"], rtol=1e-12
    )
    np.testing.assert_allclose(doubled["jump"], 2.0 * base["jump"], rtol=1e-12)


def test_default_shock_hits_only_the_first_innovation(res2):
    irf = res2.impulse_response(horizon=5)
    np.testing.assert_allclose(irf["shock"], [1.0, 0.0])
    # second block is untouched by a shock to the first
    np.testing.assert_allclose(irf["predetermined"][:, 1], np.zeros(5), atol=1e-14)
    np.testing.assert_allclose(irf["predetermined"][:, 0], 0.6 ** np.arange(5))


def test_impulse_response_rejects_bad_arguments(res):
    with pytest.raises(ValueError, match="horizon"):
        res.impulse_response(horizon=0)
    with pytest.raises(ValueError, match="innovation"):
        res.impulse_response(shock=[1.0, 1.0])


# --------------------------------------------------------------------------- #
# 4. plotting
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def figure_module():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    return Figure


def test_plot_impulse_response_returns_a_figure(res, figure_module):
    fig = res.plot_impulse_response(horizon=20)
    assert isinstance(fig, figure_module)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # one line per series (state + jump) plus the zero reference line
    assert len(ax.get_lines()) == 2 + 1
    zero = [ln for ln in ax.get_lines() if np.allclose(ln.get_ydata(), 0.0)]
    assert zero, "impulse responses must show a zero reference line"
    state = [ln for ln in ax.get_lines() if len(ln.get_ydata()) == 20]
    assert len(state) == 2
    np.testing.assert_allclose(
        state[0].get_ydata(), RHO ** np.arange(20), rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(
        state[1].get_ydata(), G_CLOSED * RHO ** np.arange(20), rtol=1e-10, atol=1e-14
    )
    assert "impulse response" in ax.get_title(loc="left").lower()


def test_plot_labels_each_line_without_a_legend(res, figure_module):
    fig = res.plot_impulse_response(horizon=12, names=["money", "prices"])
    ax = fig.axes[0]
    assert ax.get_legend() is None
    texts = {t.get_text() for t in ax.texts}
    assert {"money", "prices"} <= texts
    # colliding end labels are pushed apart
    ys = sorted(t.xy[1] for t in ax.texts if t.get_text() in {"money", "prices"})
    assert ys[1] - ys[0] > 0.05


def test_plot_multi_series_and_bad_names(res2, figure_module):
    fig = res2.plot_impulse_response(horizon=15)
    assert len(fig.axes[0].get_lines()) == 4 + 1
    with pytest.raises(ValueError, match="names"):
        res2.plot_impulse_response(horizon=5, names=["only-one"])


def test_plot_accepts_an_axes_and_saves(res, figure_module, tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig, ax = plt.subplots()
    out = tmp_path / "irf.png"
    returned = res.plot_impulse_response(horizon=8, ax=ax, path=str(out))
    assert returned is fig
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)
