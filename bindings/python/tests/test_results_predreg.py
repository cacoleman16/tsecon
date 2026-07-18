"""Tests for the predictive-regression Results facade.

The DGP throughout is the Stambaugh setting the estimators exist for: a highly
persistent predictor (rho = 0.99) whose innovation is strongly negatively
correlated with the return innovation (corr = -0.9), and a **true beta of
zero**. That is the case where OLS is biased and its t-test over-rejects, so it
is the case whose reporting the summary is supposed to get right.
"""

from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pytest

import tsecon
from tsecon.results._predreg import (
    PERSISTENCE_WARN,
    IVXTestResults,
    PredictiveRegressionResults,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _stambaugh_dgp(seed: int, *, T: int = 500, rho: float = 0.99,
                   beta: float = 0.0, corr: float = -0.9):
    """Persistent, endogenous predictor with a known true slope."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T + 1, 2))
    e = z[:, 0]
    u = corr * e + math.sqrt(1.0 - corr**2) * z[:, 1]
    x = np.zeros(T + 1)
    for t in range(1, T + 1):
        x[t] = rho * x[t - 1] + e[t]
    return beta * x + u, x


@pytest.fixture(scope="module")
def data():
    return _stambaugh_dgp(7)


@pytest.fixture(scope="module")
def raw(data):
    r, x = data
    return tsecon.predictive_regression(r, x)


@pytest.fixture(scope="module")
def res(data):
    r, x = data
    return PredictiveRegressionResults.fit(r, x, name="dp")


# --------------------------------------------------------------------------- #
# 1. backward compatibility — the object IS the dict it always was
# --------------------------------------------------------------------------- #
def test_is_a_dict(res):
    assert isinstance(res, dict)


def test_every_original_key_survives_unchanged(res, raw):
    assert set(res) == set(raw) == {"ols", "stambaugh", "ivx", "nobs"}
    assert res["nobs"] == raw["nobs"]
    for block in ("ols", "stambaugh", "ivx"):
        assert set(res[block]) == set(raw[block])
        for key, value in raw[block].items():
            assert res[block][key] == pytest.approx(value)

    # the documented sub-keys themselves, spelled out so a rename breaks here
    assert set(raw["ols"]) == {"alpha", "beta", "se", "tstat"}
    assert set(raw["stambaugh"]) == {
        "beta_ols", "beta_corrected", "bias_term", "rho_ols", "se",
    }
    assert set(raw["ivx"]) == {"beta_ivx", "wald", "pvalue", "rz"}


def test_indexing_and_unpacking_still_work(res):
    assert res["ivx"]["pvalue"] == res.ivx["pvalue"]
    unpacked = {**res}
    assert type(unpacked) is dict
    assert unpacked["nobs"] == res["nobs"]


def test_to_dict_is_a_plain_dict(res):
    d = res.to_dict()
    assert type(d) is dict
    assert d == dict(res)


def test_json_round_trip(res):
    text = json.dumps(res.to_dict())
    back = json.loads(text)
    assert back["ivx"]["pvalue"] == pytest.approx(res["ivx"]["pvalue"])
    assert back["nobs"] == res["nobs"]


def test_pickle_round_trip(res):
    back = pickle.loads(pickle.dumps(res))
    assert isinstance(back, PredictiveRegressionResults)
    assert back == dict(res)
    assert back._name == "dp"          # the label survives too
    assert back.summary() == res.summary()


# --------------------------------------------------------------------------- #
# 2. summary content
# --------------------------------------------------------------------------- #
def test_summary_is_a_string_of_reasonable_width(res):
    s = res.summary()
    assert isinstance(s, str)
    assert max(len(line) for line in s.splitlines()) <= 72


def test_summary_shows_all_three_estimators_with_their_numbers(res):
    s = res.summary()
    for label in ("OLS", "Stambaugh", "IVX"):
        assert label in s

    assert f"{res['ols']['beta']:+.5f}" in s
    assert f"{res['ols']['se']:.5f}" in s
    assert f"t {res['ols']['tstat']:+.3f}" in s
    assert f"{res['stambaugh']['beta_corrected']:+.5f}" in s
    assert f"{res['ivx']['beta_ivx']:+.5f}" in s
    assert f"W {res['ivx']['wald']:.4f}" in s


def test_summary_headlines_the_ivx_pvalue(res):
    s = res.summary()
    p = f"{res['ivx']['pvalue']:.4f}"
    assert f"IVX p = {p}" in s.splitlines()[1]
    assert f"Report the IVX Wald p-value ({p})" in s


def test_summary_reports_diagnostics(res):
    s = res.summary()
    assert f"nobs {res['nobs']}" in s
    assert f"rho(dp) {res['stambaugh']['rho_ols']:.4f}" in s
    assert f"IVX rz {res['ivx']['rz']:.4f}" in s
    assert f"Stambaugh bias removed {res['stambaugh']['bias_term']:+.5f}" in s


def test_summary_uses_the_predictor_name(res):
    assert "b*dp(t)" in res.summary()


def test_summary_flags_the_persistent_predictor(res):
    """rho ~ 0.99 must trigger the unreliable-OLS-t warning."""
    assert res["stambaugh"]["rho_ols"] > PERSISTENCE_WARN
    s = res.summary()
    assert "WARNING" in s
    assert f"rho(dp) = {res['stambaugh']['rho_ols']:.4f}" in s
    assert "highly persistent" in s
    assert "OLS t-statistic is unreliable" in s


def test_summary_does_not_warn_for_a_calm_predictor():
    r, x = _stambaugh_dgp(3, rho=0.2)
    res = PredictiveRegressionResults.fit(r, x)
    assert res["stambaugh"]["rho_ols"] < PERSISTENCE_WARN
    assert "highly persistent" not in res.summary()


def test_summary_calls_out_ols_over_rejection_when_they_disagree():
    """Seed 15: true beta is 0, OLS says p<0.05, IVX correctly does not."""
    r, x = _stambaugh_dgp(15)
    res = PredictiveRegressionResults.fit(r, x)
    from statistics import NormalDist

    p_ols = 2.0 * (1.0 - NormalDist().cdf(abs(res["ols"]["tstat"])))
    assert p_ols < 0.05, "seed 15 is supposed to be an OLS false positive"
    assert not res.significant()

    s = res.summary()
    assert "OLS alone would have called this significant" in s
    assert "over-rejection IVX is designed to remove" in s


def test_summary_closing_line_reflects_the_ivx_verdict(res):
    assert "IVX does not reject b = 0" in res.summary()
    assert "IVX rejects b = 0" not in res.summary()


# --------------------------------------------------------------------------- #
# 3. repr
# --------------------------------------------------------------------------- #
def test_repr_equals_summary(res):
    assert repr(res) == res.summary()


# --------------------------------------------------------------------------- #
# 4. plot
# --------------------------------------------------------------------------- #
def test_plot_estimates_returns_a_figure(res):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = res.plot_estimates()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # one zero reference line + one interval and one dot per estimator
    assert len(ax.lines) == 7
    assert [t.get_text() for t in ax.get_yticklabels()] == ["IVX", "Stambaugh", "OLS"]
    matplotlib.pyplot.close(fig)


def test_plot_estimates_accepts_an_axes_and_a_path(res, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = tmp_path / "forest.png"
    returned = res.plot_estimates(ax=ax, path=str(out))
    assert returned is fig
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_plot_intervals_bracket_the_point_estimates(res):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    fig = res.plot_estimates(level=0.95)
    ax = fig.axes[0]
    lo, hi = res.conf_int(0.95)
    xs = [tuple(line.get_xdata()) for line in ax.lines]
    assert any(
        len(pair) == 2
        and pair[0] == pytest.approx(lo)
        and pair[1] == pytest.approx(hi)
        for pair in xs
    )
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# 5. family-specific accessors
# --------------------------------------------------------------------------- #
def test_significant_reads_the_ivx_pvalue_not_the_ols_t():
    """Seed 15 separates the two rules: OLS would reject, IVX would not."""
    r, x = _stambaugh_dgp(15)
    res = PredictiveRegressionResults.fit(r, x)
    assert abs(res["ols"]["tstat"]) > 1.96          # OLS would reject at 5%
    assert res["ivx"]["pvalue"] > 0.05              # IVX would not
    assert res.significant(0.05) is False           # the object follows IVX

    # and it tracks the IVX p-value as the level moves past it
    p = res["ivx"]["pvalue"]
    assert res.significant(min(0.99, p + 0.01)) is True
    assert res.significant(max(1e-6, p - 0.01)) is False


def test_significant_rejects_a_bad_level(res):
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            res.significant(bad)


def test_betas_returns_the_three_slopes(res):
    b = res.betas()
    assert set(b) == {"ols", "stambaugh", "ivx"}
    assert b["ols"] == pytest.approx(res["ols"]["beta"])
    assert b["stambaugh"] == pytest.approx(res["stambaugh"]["beta_corrected"])
    assert b["ivx"] == pytest.approx(res["ivx"]["beta_ivx"])


def test_stambaugh_correction_is_exactly_ols_minus_bias(res):
    s = res["stambaugh"]
    assert s["beta_ols"] == pytest.approx(res["ols"]["beta"])
    assert s["beta_corrected"] == pytest.approx(s["beta_ols"] - s["bias_term"])


def test_ivx_se_reproduces_the_wald_statistic(res):
    se = res.ivx_se()
    assert se > 0.0
    implied = (res["ivx"]["beta_ivx"] / se) ** 2
    assert implied == pytest.approx(res["ivx"]["wald"], rel=1e-9)


def test_conf_int_is_centred_and_widens_with_level(res):
    lo, hi = res.conf_int(0.95)
    beta = res["ivx"]["beta_ivx"]
    assert lo < beta < hi
    assert (lo + hi) / 2 == pytest.approx(beta)
    lo99, hi99 = res.conf_int(0.99)
    assert lo99 < lo and hi99 > hi


def test_conf_int_excludes_zero_iff_significant(res):
    lo, hi = res.conf_int(0.95)
    crosses_zero = lo <= 0.0 <= hi
    assert crosses_zero is (not res.significant(0.05))


def test_conf_int_rejects_a_bad_level(res):
    with pytest.raises(ValueError):
        res.conf_int(1.5)


def test_rho_and_is_persistent(res):
    assert res.rho() == pytest.approx(res["stambaugh"]["rho_ols"])
    assert res.is_persistent() is True
    assert res.is_persistent(threshold=0.999) is False


def test_fit_passes_keywords_through(data):
    r, x = data
    tuned = PredictiveRegressionResults.fit(r, x, cz=-0.5)
    baseline = PredictiveRegressionResults.fit(r, x)
    # cz tunes the IVX instrument, so rz must move but OLS must not
    assert tuned["ivx"]["rz"] != pytest.approx(baseline["ivx"]["rz"])
    assert tuned["ols"]["beta"] == pytest.approx(baseline["ols"]["beta"])


# --------------------------------------------------------------------------- #
# IVXTestResults — the joint test
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def joint(data):
    r, x = data
    xs = np.column_stack([x, np.roll(x, 3)])
    return IVXTestResults.fit(r, xs, names=["dp", "tbill"]), r, xs


def test_joint_is_a_dict_with_the_original_keys(joint):
    res, r, xs = joint
    raw = tsecon.ivx_test(r, xs)
    assert isinstance(res, dict)
    assert set(res) == set(raw)
    assert set(raw) == {"beta_ivx", "wald", "pvalue", "rz", "nobs", "nregressors"}
    assert res["wald"] == pytest.approx(raw["wald"])
    np.testing.assert_allclose(res["beta_ivx"], raw["beta_ivx"])


def test_joint_json_and_pickle_round_trip(joint):
    res, _, _ = joint
    d = res.to_dict()
    d["beta_ivx"] = d["beta_ivx"].tolist()
    back = json.loads(json.dumps(d))
    assert back["nregressors"] == 2

    unpickled = pickle.loads(pickle.dumps(res))
    assert isinstance(unpickled, IVXTestResults)
    assert unpickled["wald"] == pytest.approx(res["wald"])
    assert unpickled.names() == ["dp", "tbill"]


def test_joint_summary_content(joint):
    res, _, _ = joint
    s = res.summary()
    assert f"IVX p = {res['pvalue']:.4f}" in s
    assert f"Wald chi2(2) {res['wald']:.4f}" in s
    assert f"nobs {res['nobs']}" in s
    for name, beta in zip(["dp", "tbill"], res["beta_ivx"]):
        assert name in s
        assert f"{beta:+.5f}" in s
    assert "no joint evidence of predictability" in s
    assert max(len(line) for line in s.splitlines()) <= 72


def test_joint_repr_equals_summary(joint):
    res, _, _ = joint
    assert repr(res) == res.summary()


def test_joint_names_default_to_x1_xk(data):
    r, x = data
    xs = np.column_stack([x, np.roll(x, 3)])
    res = IVXTestResults.fit(r, xs)
    assert res.names() == ["x1", "x2"]
    assert "x1" in res.summary()


def test_joint_significant_reads_the_pvalue(joint):
    res, _, _ = joint
    p = res["pvalue"]
    assert res.significant(0.05) is (p < 0.05)
    assert res.significant(min(0.99, p + 0.01)) is True
    assert res.significant(max(1e-6, p - 0.01)) is False
    with pytest.raises(ValueError):
        res.significant(0.0)
