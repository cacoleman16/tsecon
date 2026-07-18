"""Tests for GARCHResults — the Results facade over ``tsecon.garch_fit``.

The load-bearing property is backward compatibility: the object *is* the dict
``garch_fit`` has always returned, so every original key must survive
untouched, alongside the new rendering and accessors.
"""

import json
import pickle

import numpy as np
import pytest

import tsecon
from tsecon.results._garch import GARCHResults

# DGP: a real GARCH(1,1). Pure white noise makes QMLE degenerate (the
# optimiser hits a NaN gradient), so the data must actually have volatility
# clustering for the fit to converge.
OMEGA, ALPHA, BETA = 0.02, 0.10, 0.87
N_OBS = 1200


def simulate_garch(seed: int = 11, n: int = N_OBS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = OMEGA / (1.0 - ALPHA - BETA)
    e = np.zeros(n)
    for t in range(n):
        if t > 0:
            h = OMEGA + ALPHA * e[t - 1] ** 2 + BETA * h
        e[t] = np.sqrt(h) * rng.standard_normal()
    return e


@pytest.fixture(scope="module")
def y():
    return simulate_garch()


@pytest.fixture(scope="module")
def res(y):
    return GARCHResults.fit(y, vol="garch", mean="zero", dist="normal")


@pytest.fixture(scope="module")
def raw(y):
    return tsecon.garch_fit(y, vol="garch", mean="zero", dist="normal")


def _to_jsonable(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# --------------------------------------------------------------------- #
# the DGP is recovered (so the summary numbers below are meaningful)
# --------------------------------------------------------------------- #
def test_fit_recovers_the_dgp(res):
    p = res.params_named()
    assert p["alpha[1]"] == pytest.approx(ALPHA, abs=0.05)
    assert p["beta[1]"] == pytest.approx(BETA, abs=0.05)
    assert p["omega"] == pytest.approx(OMEGA, abs=0.02)


# --------------------------------------------------------------------- #
# 1. backward compatibility: it is still the dict
# --------------------------------------------------------------------- #
def test_is_a_dict_with_every_original_key(res, raw):
    assert isinstance(res, dict)
    assert set(res) == set(raw)
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(res[key], value)
        else:
            assert res[key] == value

    # indexing and ** unpacking, exactly as before
    assert res["params"].shape == (3,)
    assert dict(**res).keys() == raw.keys()
    assert type(res.to_dict()) is dict  # a plain dict, not the subclass
    assert set(res.to_dict()) == set(raw)


def test_forecast_key_still_appears_when_requested(y):
    r = GARCHResults.fit(y, vol="garch", mean="zero", forecast_horizon=7)
    assert "variance_forecast" in r
    assert len(r["variance_forecast"]) == 7


def test_json_round_trip(res, raw):
    text = json.dumps(_to_jsonable(res.to_dict()))
    back = json.loads(text)
    assert back["param_names"] == list(raw["param_names"])
    assert back["loglik"] == pytest.approx(float(raw["loglik"]))
    assert back["params"] == pytest.approx(raw["params"].tolist())


def test_pickle_round_trip(res):
    back = pickle.loads(pickle.dumps(res))
    assert isinstance(back, GARCHResults)
    assert isinstance(back, dict)
    np.testing.assert_allclose(back["params"], res["params"])
    assert back["param_names"] == res["param_names"]
    # the recorded volatility spec survives, so persistence stays correct
    assert back.persistence() == pytest.approx(res.persistence())


# --------------------------------------------------------------------- #
# 2. summary content
# --------------------------------------------------------------------- #
def test_summary_names_the_model_and_the_robust_ses(res):
    s = res.summary()
    assert isinstance(s, str)
    assert "GARCH(1,1)" in s
    assert "zero mean" in s
    assert "Normal errors" in s
    assert "Bollerslev-Wooldridge robust" in s
    assert "QMLE" in s


def test_summary_shows_fit_statistics_with_the_real_numbers(res, raw):
    s = res.summary()
    assert f"No. obs {N_OBS}" in s
    assert f"Log-lik {float(raw['loglik']):.3f}" in s
    assert f"AIC {float(raw['aic']):.3f}" in s
    assert f"BIC {float(raw['bic']):.3f}" in s


def test_summary_param_block_carries_coef_robust_se_and_t(res, raw):
    s = res.summary()
    assert "robust SE" in s
    for i, name in enumerate(raw["param_names"]):
        assert name in s
        assert f"{raw['params'][i]:+.5f}" in s
        assert f"{raw['se_robust'][i]:.5f}" in s
    t = float(raw["params"][2] / raw["se_robust"][2])
    assert f"{t:+.2f}" in s


def test_summary_persistence_line(res):
    s = res.summary()
    assert f"Persistence  alpha + beta = {res.persistence():.5f}" in s
    assert "[near-IGARCH]" not in s  # 0.969 is persistent but not integrated


def test_summary_lines_stay_narrow(res):
    assert max(len(line) for line in res.summary().splitlines()) <= 72


# --------------------------------------------------------------------- #
# 3. repr is the summary
# --------------------------------------------------------------------- #
def test_repr_is_the_summary(res):
    assert repr(res) == res.summary()


# --------------------------------------------------------------------- #
# 4. plots
# --------------------------------------------------------------------- #
@pytest.fixture
def mpl():
    """matplotlib is an optional (`plots`) extra — skip if absent, and force a
    headless backend so these run on CI runners with no display."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    return matplotlib


def test_plot_volatility_returns_a_figure(res, mpl):
    from matplotlib.figure import Figure

    fig = res.plot_volatility()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # the sigma path plus the mean reference line
    assert len(ax.lines) == 2
    path = ax.lines[0].get_ydata()
    np.testing.assert_allclose(path, res["conditional_volatility"])
    assert ax.get_ylim()[0] == 0.0


def test_plot_volatility_accepts_an_ax_and_a_path(res, tmp_path, mpl):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = tmp_path / "vol.png"
    returned = res.plot_volatility(ax=ax, path=out)
    assert returned is fig
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_plot_diagnostics_has_two_panels(res, mpl):
    from matplotlib.figure import Figure

    fig = res.plot_diagnostics(lags=12)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    acf_ax = fig.axes[1]
    assert len(acf_ax.patches) == 12  # one bar per lag


# --------------------------------------------------------------------- #
# 5. family-specific accessors
# --------------------------------------------------------------------- #
def test_persistence_is_alpha_plus_beta(res):
    p = res.params_named()
    assert res.persistence() == pytest.approx(p["alpha[1]"] + p["beta[1]"])
    assert 0.9 < res.persistence() < 1.0


def test_conditional_volatility_is_a_standard_deviation(res, y):
    """It is sqrt(sigma2) at source, so plot_volatility must not sqrt again.

    Fitted on a rescaled series (std ~8.4, var ~71) the two candidates are far
    apart, which is what makes this decisive rather than suggestive.
    """
    sigma = res.volatility()
    assert sigma.shape == (N_OBS,)

    big = GARCHResults.fit(10.0 * y, vol="garch", mean="zero", dist="normal")
    s = big.volatility()
    assert s.mean() == pytest.approx((10.0 * y).std(), rel=0.15)
    assert not np.isclose(s.mean(), (10.0 * y).var(), rtol=0.5)
    # y_t / sigma_t is the standardized residual, i.e. unit-variance
    np.testing.assert_allclose(big["std_residuals"], (10.0 * y) / s, rtol=1e-10)
    assert big["std_residuals"].std() == pytest.approx(1.0, abs=0.05)


def test_tstats_are_params_over_robust_se(res):
    np.testing.assert_allclose(
        res.tstats(), np.asarray(res["params"]) / np.asarray(res["se_robust"])
    )


def test_near_igarch_is_flagged():
    fake = GARCHResults(
        {
            "params": np.array([0.01, 0.09, 0.905]),
            "param_names": ["omega", "alpha[1]", "beta[1]"],
            "loglik": -100.0,
            "aic": 206.0,
            "bic": 210.0,
            "se_mle": np.array([0.01, 0.01, 0.01]),
            "se_robust": np.array([0.01, 0.01, 0.01]),
            "conditional_volatility": np.ones(50),
            "std_residuals": np.zeros(50),
        }
    )
    assert fake.persistence() == pytest.approx(0.995)
    assert "[near-IGARCH]" in fake.summary()


def test_persistence_uses_names_not_positions(y):
    """A constant mean and t errors shift every position; names must win."""
    r = GARCHResults.fit(y, vol="garch", mean="constant", dist="t")
    names = list(r["param_names"])
    assert names[0] == "mu" and names[-1] == "nu"
    p = r.params_named()
    assert r.persistence() == pytest.approx(p["alpha[1]"] + p["beta[1]"])
    assert "constant mean" in r.summary()
    assert "Student-t errors" in r.summary()


def test_gjr_persistence_halves_the_leverage_term(y):
    r = GARCHResults.fit(y, vol="gjr", mean="zero", dist="normal")
    p = r.params_named()
    assert r.model_name() == "GJR-GARCH(1,1,1)"
    assert r.persistence() == pytest.approx(
        p["alpha[1]"] + 0.5 * p["gamma[1]"] + p["beta[1]"]
    )
    assert "alpha + gamma/2 + beta" in r.summary()


def test_egarch_persistence_is_beta_alone(y):
    r = GARCHResults.fit(y, vol="egarch", mean="zero", dist="normal")
    p = r.params_named()
    assert r.model_name() == "EGARCH(1,1,1)"
    assert r.persistence() == pytest.approx(p["beta[1]"])
    assert "beta (log-variance)" in r.summary()


def test_ambiguous_asymmetric_model_omits_persistence(y):
    """GJR and EGARCH share parameter names; without the recorded family the
    persistence formula is unknowable, so it is omitted rather than guessed."""
    fitted = GARCHResults.fit(y, vol="gjr")
    bare = GARCHResults(dict(fitted))  # rebuilt from the raw dict alone
    assert bare.persistence() is None
    assert "Persistence" not in bare.summary()
    # the summary still renders everything else
    assert "gamma[1]" in bare.summary()
