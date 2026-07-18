"""Degenerate and off-happy-path behaviour of the Results facade.

The per-family test files drive each object with well-behaved data. This one
drives the branches that only fire when something is *unusual*: a name list
that has to be inferred, a trend string the labeller does not recognise, a
model whose persistence is not identifiable, a matrix too wide to print, a
statistic that came back non-finite.

The reason those matter is uniform: none of them raises. Each one either picks
a label or returns a number, so a bug there produces a summary that is wrong
rather than a summary that is missing. Every assertion below is about the
object refusing to guess, or guessing correctly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tsecon
from tsecon.results import Results
from tsecon.results._base import fmt_row, param_table
from tsecon.results._dsge import DSGEResults, _matrix_lines, _value_lines
from tsecon.results._garch import GARCHResults
from tsecon.results._lp import LPResults
from tsecon.results._predreg import (
    IVXTestResults,
    PredictiveRegressionResults,
    _fnum,
    _naive_p,
)
from tsecon.results._var import IRFArray, VARResults, var_fit, var_irf


def _var_data(n: int = 240, k: int = 2, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = 0.4 * np.eye(k) + 0.05
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.normal(size=k) * 0.5
    return y


# --------------------------------------------------------------------------- #
# Results base class
# --------------------------------------------------------------------------- #
def test_base_results_default_summary_lists_its_keys():
    """A subclass that has not overridden summary() still renders — and repr()
    routes through it, which is what makes these objects usable in a REPL."""
    r = Results({"b": 2, "a": 1})
    assert r.summary() == "Results(a, b)"
    assert repr(r) == "Results(a, b)"
    assert r.to_dict() == {"a": 1, "b": 2}
    assert type(r.to_dict()) is dict


def test_param_table_without_se_or_tstats_still_aligns():
    lines = param_table(["alpha", "beta"], [0.1, -0.2])
    assert "std err" not in lines[0] and lines[0].split()[:2] == ["param", "coef"]
    assert "+0.10000" in lines[2] and "-0.20000" in lines[3]


def test_fmt_row_defaults_to_left_alignment():
    assert fmt_row(["a", "b"], [4, 4]) == "a     b"      # trailing pad stripped
    assert fmt_row(["a", "b"], [4, 4], ["r", "l"]) == "   a  b"


# --------------------------------------------------------------------------- #
# optional dependencies: the paths a user without the extras actually takes
# --------------------------------------------------------------------------- #
def test_plot_methods_without_matplotlib_name_the_extra_to_install(monkeypatch):
    """matplotlib is the `plots` extra. Its absence must produce a message that
    names the extra and points at the data-returning alternative — this is the
    first thing a core-wheel user hits, and it is marked `pragma: no cover`
    ("exercised via monkeypatch"), so this test is what makes that claim true."""
    import sys

    from tsecon.results import _plotting

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    with pytest.raises(ImportError) as excinfo:
        _plotting.pyplot()
    message = str(excinfo.value)
    assert "tsecon[plots]" in message
    assert "pip install matplotlib" in message
    assert "data-returning twin" in message


def test_to_pandas_without_pandas_names_the_fields_that_carry_the_numbers(monkeypatch):
    """pandas is not a dependency at all, so this error has to explain that the
    .rows/.columns/.values fields already hold everything the frame would."""
    import sys

    from tsecon.results._var import CoefficientFrame

    frame = CoefficientFrame(rows=["const"], columns=["y1"], values=np.zeros((1, 1)))
    monkeypatch.setitem(sys.modules, "pandas", None)

    with pytest.raises(ImportError, match="(?s)pip install pandas.*rows"):
        frame.to_pandas()


# --------------------------------------------------------------------------- #
# VAR: naming, which is where a silent mislabel would live
# --------------------------------------------------------------------------- #
def test_var_picks_up_dataframe_column_names():
    """Fitting a DataFrame must label equations with its columns; falling back
    to y1..yk here would attach the right numbers to the wrong series."""
    pd = pytest.importorskip("pandas")
    data = _var_data()
    df = pd.DataFrame(data, columns=["gdp", "infl"])

    fit = VARResults.fit(df, lags=2)
    assert fit.names == ["gdp", "infl"]
    assert "gdp" in fit.summary()
    frame = fit.coefficient_frame()
    assert frame.columns == ["gdp", "infl"]
    assert "L1.gdp" in frame.rows and "L2.infl" in frame.rows


def test_var_ignores_column_names_of_the_wrong_length():
    """A DataFrame whose columns cannot be aligned must fall back to y1..yk
    rather than truncate or recycle labels."""
    pd = pytest.importorskip("pandas")

    class OddColumns:
        """Array-like carrying a mismatched `.columns`, as a reshaped frame would."""

        def __init__(self, values):
            self._v = values
            self.columns = ["only_one"]

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self._v, dtype=dtype)

    fit = VARResults.fit(OddColumns(_var_data()), lags=1)
    assert fit.names == ["y1", "y2"]
    assert pd is not None


def test_var_explicit_names_of_the_wrong_length_are_rejected():
    with pytest.raises(ValueError, match="names has 3 entries"):
        VARResults.fit(_var_data(), lags=1, names=["a", "b", "c"])


def test_var_coefficient_frame_to_pandas_matches_the_raw_values():
    pd = pytest.importorskip("pandas")
    fit = VARResults.fit(_var_data(), lags=2, names=["gdp", "infl"])
    frame = fit.coefficient_frame()
    df = frame.to_pandas()

    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == frame.rows
    assert list(df.columns) == frame.columns
    np.testing.assert_allclose(df.to_numpy(), frame.values)


def test_var_trend_n_has_no_deterministic_rows():
    fit = VARResults.fit(_var_data(), lags=1, trend="n")
    rows = fit.coefficient_frame().rows
    assert "const" not in rows and "trend" not in rows


def test_var_deterministic_labels_stay_generic_for_an_unknown_trend():
    """The labeller only knows n/c/ct/ctt. Anything else must produce generic
    row names — mislabelling a deterministic row as 'const' would be worse."""
    from tsecon.results._var import _det_labels

    assert _det_labels(1, "c") == ["const"]
    assert _det_labels(2, "ct") == ["const", "trend"]
    assert _det_labels(0, "n") == []
    # Unknown string, or a known string whose length does not match the design:
    assert _det_labels(5, "quarterly_dummies") == [
        "const", "trend", "trend^2", "det4", "det5",
    ]
    assert _det_labels(2, "c") == ["const", "trend"]


def test_var_summary_blocks_wide_systems_without_dropping_equations():
    """More equations than fit one block: every variable must still appear."""
    fit = VARResults.fit(_var_data(n=400, k=7), lags=1)
    text = fit.summary()
    for name in fit.names:
        assert name in text


# --------------------------------------------------------------------------- #
# VAR: module-level helpers and IRF labelling
# --------------------------------------------------------------------------- #
def test_var_fit_helper_matches_the_classmethod():
    data = _var_data()
    a = var_fit(data, lags=2, names=["gdp", "infl"])
    b = VARResults.fit(data, lags=2, names=["gdp", "infl"])
    assert isinstance(a, VARResults)
    assert a.names == b.names
    np.testing.assert_allclose(np.asarray(a["params"]), np.asarray(b["params"]))


def test_var_irf_rejects_non_2d_data():
    with pytest.raises(ValueError, match="must be 2-D"):
        var_irf(np.arange(20.0), lags=1, horizon=4)


def test_irf_summary_reports_cumulative_and_reduced_form_honestly():
    """The label is the only thing telling a reader which object they hold."""
    data = _var_data()
    orth = var_irf(data, lags=1, horizon=6, orth=True, names=["gdp", "infl"])
    assert "orthogonalised" in orth.summary()
    assert "cumulative" not in orth.summary()

    cum = var_irf(data, lags=1, horizon=6, orth=False, cumulative=True,
                  names=["gdp", "infl"])
    assert "cumulative reduced-form" in cum.summary()
    assert repr(cum) == cum.summary()
    # Cumulating must actually change the numbers, not just the label.
    assert not np.allclose(cum.to_array(), var_irf(data, lags=1, horizon=6,
                                                   orth=False).to_array())


def test_irf_unknown_variable_name_is_a_keyerror_naming_the_options():
    irf = var_irf(_var_data(), lags=1, horizon=4, names=["gdp", "infl"])
    with pytest.raises(KeyError, match="gdp"):
        irf.response("nope", 0)


def test_empty_irf_array_refuses_to_plot():
    pytest.importorskip("matplotlib")
    empty = IRFArray([])
    assert empty.neqs == 0 and empty.horizon == 0
    with pytest.raises(ValueError, match="no impulse responses to plot"):
        empty.plot()


# --------------------------------------------------------------------------- #
# predictive regressions: non-finite handling and the verdict wording
# --------------------------------------------------------------------------- #
def test_naive_p_is_nan_for_a_non_finite_t():
    """A nan t-statistic must not be laundered into p = 0."""
    assert math.isnan(_naive_p(float("nan")))
    assert math.isnan(_naive_p(float("inf")))
    assert _naive_p(0.0) == pytest.approx(1.0)


def test_fnum_degrades_to_na_rather_than_printing_inf():
    assert _fnum(float("nan")) == "n/a"
    assert _fnum(float("inf"), ".4f") == "n/a"
    assert _fnum(1.5) == "+1.50000"


def test_ivx_se_is_nan_when_the_wald_statistic_is_degenerate():
    """se = |beta| / sqrt(wald) is undefined at wald <= 0 or beta = 0; a
    fabricated finite SE there would produce a confidence interval that lies."""
    def make(beta, wald):
        return PredictiveRegressionResults({
            "ols": {"alpha": 0.0, "beta": 0.0, "se": 1.0, "tstat": 0.0},
            "stambaugh": {"beta_ols": 0.0, "beta_corrected": 0.0, "se": 1.0,
                          "rho_ols": 0.5, "bias_term": 0.0},
            "ivx": {"beta_ivx": beta, "wald": wald, "pvalue": 1.0, "rz": 0.9},
            "nobs": 100,
        })

    for beta, wald in [(0.5, 0.0), (0.5, -1.0), (0.0, 4.0),
                       (float("nan"), 4.0), (0.5, float("inf"))]:
        assert math.isnan(make(beta, wald).ivx_se())

    ok = make(2.0, 4.0)
    assert ok.ivx_se() == pytest.approx(1.0)
    lo, hi = ok.conf_int(0.95)
    assert lo < 2.0 < hi


def test_predictive_regression_subdict_properties_are_the_dict_entries():
    res = PredictiveRegressionResults({
        "ols": {"alpha": 0.1, "beta": 0.2, "se": 0.3, "tstat": 0.66},
        "stambaugh": {"beta_ols": 0.2, "beta_corrected": 0.15, "se": 0.3,
                      "rho_ols": 0.99, "bias_term": 0.05},
        "ivx": {"beta_ivx": 0.12, "wald": 1.0, "pvalue": 0.3, "rz": 0.95},
        "nobs": 250,
    })
    assert res.ols is res["ols"]
    assert res.stambaugh is res["stambaugh"]
    assert res.ivx is res["ivx"]
    assert res.rho() == pytest.approx(0.99)


def test_summary_wording_flips_with_the_ivx_verdict():
    """The sentence a reader acts on. Both branches must exist and disagree."""
    base = {
        "ols": {"alpha": 0.0, "beta": 0.4, "se": 0.1, "tstat": 4.0},
        "stambaugh": {"beta_ols": 0.4, "beta_corrected": 0.3, "se": 0.1,
                      "rho_ols": 0.5, "bias_term": 0.1},
        "nobs": 300,
    }
    rejects = PredictiveRegressionResults(
        {**base, "ivx": {"beta_ivx": 0.4, "wald": 16.0, "pvalue": 0.001, "rz": 0.9}}
    )
    fails = PredictiveRegressionResults(
        {**base, "ivx": {"beta_ivx": 0.05, "wald": 0.25, "pvalue": 0.7, "rz": 0.9}}
    )
    assert rejects.significant(0.05) and "IVX rejects b = 0" in rejects.summary()
    assert not fails.significant(0.05)
    assert "does not reject b = 0" in fails.summary()


def test_joint_ivx_summary_wording_flips_too():
    def make(p):
        return IVXTestResults({
            "nregressors": 2, "wald": 10.0, "pvalue": p, "rz": 0.9,
            "beta_ivx": [0.1, -0.2], "nobs": 300,
        })

    assert "jointly predict" in make(0.001).summary()
    assert "no joint evidence" in make(0.6).summary()
    # Labels default to x1..xk and are overridable.
    assert make(0.6).names() == ["x1", "x2"]


def test_joint_ivx_summary_survives_a_non_finite_wald():
    """_fnum's n/a path, reached the way it actually would be in practice."""
    res = IVXTestResults({
        "nregressors": 1, "wald": float("nan"), "pvalue": float("nan"),
        "rz": float("nan"), "beta_ivx": [0.0], "nobs": 50,
    })
    text = res.summary()
    assert "n/a" in text
    assert not res.significant(0.05)      # a nan p-value must not reject


def test_forest_plot_omits_the_interval_for_a_degenerate_se():
    """A non-finite SE must drop the whisker but still draw the point, rather
    than crashing or drawing a bar of nonsense width."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    res = PredictiveRegressionResults({
        "ols": {"alpha": 0.0, "beta": 0.4, "se": 0.1, "tstat": 4.0},
        "stambaugh": {"beta_ols": 0.4, "beta_corrected": 0.3, "se": 0.1,
                      "rho_ols": 0.5, "bias_term": 0.1},
        # wald = 0 => ivx_se() is nan => the IVX row has no interval
        "ivx": {"beta_ivx": 0.4, "wald": 0.0, "pvalue": 0.5, "rz": 0.9},
        "nobs": 300,
    })
    fig = res.plot_estimates()
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["IVX", "Stambaugh", "OLS"]
    matplotlib.pyplot.close(fig)


# --------------------------------------------------------------------------- #
# DSGE: degenerate solutions and the matrix printer
# --------------------------------------------------------------------------- #
def _dsge_dict(q, moduli=(0.5, 2.0), verdict="unique stable solution"):
    return {
        "g": [[1.0]],
        "p": [[0.9]],
        "q": q,
        "eigenvalue_moduli": list(moduli),
        "verdict": verdict,
    }


def test_dsge_without_innovations_refuses_to_trace_an_irf():
    """Q with zero columns means there is no shock; iterating it would return
    an all-zero 'impulse response' that looks like a real result."""
    res = DSGEResults(_dsge_dict(np.zeros((1, 0))))
    assert res.n_shocks == 0
    with pytest.raises(ValueError, match="no innovations"):
        res.impulse_response(horizon=4)


def test_dsge_rejects_a_non_positive_horizon():
    res = DSGEResults(_dsge_dict([[1.0]]))
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        res.impulse_response(horizon=0)


def test_dsge_summary_says_none_when_a_side_of_the_unit_circle_is_empty():
    res = DSGEResults(_dsge_dict([[1.0]], moduli=(2.0, 3.0)))
    text = res.summary()
    assert res.stable_moduli().size == 0
    assert "stable    (none)" in text
    assert "3.00000" in text                       # the unstable side still prints


def test_value_lines_wraps_long_spectra():
    lines = _value_lines("stable", np.arange(12, dtype=float), per_line=5)
    assert len(lines) == 3
    assert lines[0].strip().startswith("stable")
    assert lines[1].strip().split()[0].startswith("5.")   # continuation has no label


def test_matrix_printer_elides_extra_columns_rather_than_wrapping():
    """A wide Q must be marked as elided; printing only the first six columns
    with no ellipsis would misrepresent the model's size."""
    mat = np.arange(20.0).reshape(2, 10)
    lines = _matrix_lines("Q", "impact", mat)
    assert "[2x10]" in lines[0]
    assert lines[1].rstrip().endswith("...")
    assert lines[2].rstrip().endswith("...")

    narrow = _matrix_lines("Q", "impact", np.zeros((1, 2)))
    assert "..." not in "".join(narrow)


# --------------------------------------------------------------------------- #
# GARCH: persistence is refused rather than guessed
# --------------------------------------------------------------------------- #
def _garch_dict(names, params):
    return {
        "params": list(params),
        "param_names": list(names),
        "loglik": -100.0, "aic": 208.0, "bic": 220.0,
        "se_mle": [0.1] * len(params), "se_robust": [0.1] * len(params),
        "conditional_volatility": np.ones(20),
        "std_residuals": np.zeros(20),
    }


def test_persistence_is_none_when_the_names_carry_no_alpha_beta_pair():
    """A wrong persistence is worse than no persistence: without both terms the
    accessor must return None, and the summary must not invent a number."""
    only_alpha = GARCHResults(_garch_dict(["omega", "alpha[1]"], [0.02, 0.1]))
    assert only_alpha.persistence() is None

    only_beta = GARCHResults(_garch_dict(["omega", "beta[1]"], [0.02, 0.9]))
    assert only_beta.persistence() is None

    neither = GARCHResults(_garch_dict(["omega", "mu"], [0.02, 0.0]))
    assert neither.persistence() is None
    assert "persistence" not in neither.summary().lower()


def test_persistence_is_none_for_an_asymmetric_model_of_unrecorded_family():
    """GJR and EGARCH share the alpha/gamma/beta naming, so a gamma term on an
    object built from a raw dict is ambiguous and must not be resolved."""
    raw = GARCHResults(
        _garch_dict(["omega", "alpha[1]", "gamma[1]", "beta[1]"], [0.02, 0.05, 0.08, 0.85])
    )
    assert raw._vol is None
    assert raw.persistence() is None
    assert "GARCH-family(1,1,1)" == raw.model_name()

    gjr = GARCHResults(
        _garch_dict(["omega", "alpha[1]", "gamma[1]", "beta[1]"], [0.02, 0.05, 0.08, 0.85])
    )
    gjr._vol = "gjr"
    assert gjr.persistence() == pytest.approx(0.05 + 0.5 * 0.08 + 0.85)


def test_egarch_persistence_is_beta_alone_and_none_without_it():
    eg = GARCHResults(_garch_dict(["omega", "alpha[1]", "beta[1]"], [0.0, 0.1, 0.95]))
    eg._vol = "egarch"
    assert eg.persistence() == pytest.approx(0.95)
    assert eg.model_name() == "EGARCH(1,0,1)"

    no_beta = GARCHResults(_garch_dict(["omega", "alpha[1]"], [0.0, 0.1]))
    no_beta._vol = "egarch"
    assert no_beta.persistence() is None


def test_garch_diagnostics_plot_writes_the_file_it_is_given(tmp_path):
    """`plot_volatility(path=...)` is covered elsewhere; the diagnostics panel
    has its own save branch, and a silently-unwritten figure is a real bug."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    rng = np.random.default_rng(4)
    res = GARCHResults(_garch_dict(["omega", "alpha[1]", "beta[1]"], [0.02, 0.1, 0.85]))
    res["std_residuals"] = rng.normal(size=200)
    res["conditional_volatility"] = np.abs(rng.normal(size=200)) + 0.1

    out = tmp_path / "diagnostics.png"
    fig = res.plot_diagnostics(path=out, lags=10)
    assert out.exists() and out.stat().st_size > 0
    assert len(fig.axes) == 2
    matplotlib.pyplot.close(fig)


def test_garch_acf_of_a_constant_series_is_zero_not_a_division_by_zero():
    """Zero sample variance is the degenerate case of an ACF; returning zeros
    is the documented answer, and nan would poison the diagnostic plot."""
    flat = GARCHResults._acf(np.full(50, 3.0), lags=5)
    np.testing.assert_array_equal(flat, np.zeros(5))

    varying = GARCHResults._acf(np.arange(50.0), lags=3)
    assert np.all(np.isfinite(varying))


# --------------------------------------------------------------------------- #
# LP: metadata that may legitimately be absent
# --------------------------------------------------------------------------- #
def test_lp_summary_renders_without_optional_metadata():
    """An LPResults built by hand (or unpickled from an older version) has no
    nobs or lag-control count; the summary must still render."""
    res = LPResults({
        "horizons": list(range(5)),
        "irf": [0.0, 0.5, 0.9, 0.4, 0.1],
        "se": [0.1] * 5,
    })
    text = res.summary()
    assert res._nobs is None and res._n_lag_controls is None
    assert "obs" not in text.split("peak")[0]
    assert "peak  h=2" in text

    full = LPResults.fit(np.random.default_rng(1).normal(size=200),
                         np.random.default_rng(2).normal(size=200),
                         horizons=4, n_lag_controls=2)
    stats = full.summary().split("\n")[3]
    assert "obs  200" in stats and "lag controls  2" in stats


def test_lp_summary_labels_an_unrecognised_se_kind_verbatim():
    res = LPResults({"horizons": [0, 1], "irf": [1.0, 0.5], "se": [0.1, 0.1]})
    res._se_kind = "bootstrap"
    assert "bootstrap standard errors" in res.summary()


# --------------------------------------------------------------------------- #
# ARIMA fan chart: the history-less and windowed branches
# --------------------------------------------------------------------------- #
@pytest.fixture
def arima_fit():
    rng = np.random.default_rng(3)
    e = rng.normal(size=400)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = 0.6 * y[t - 1] + e[t]
    return tsecon.results.ARIMAResults.fit(y, 1, 0, 0, forecast_steps=8), y


def test_fan_chart_without_history_starts_the_axis_at_period_one(arima_fit):
    """`ARIMAResults(raw)` keeps no series, so the fan must plot on its own —
    not crash reaching for a history that was never stored."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    fitted, _ = arima_fit
    bare = tsecon.results.ARIMAResults(dict(fitted))     # y dropped
    assert bare._y is None

    fig = bare.plot_forecast()
    ax = fig.axes[0]
    assert ax.get_xlim()[0] == pytest.approx(1.0)
    # No history means no vertical "forecast starts here" divider.
    assert not any(ln.get_linestyle() == (0, (3, 3)) for ln in ax.lines)
    matplotlib.pyplot.close(fig)


def test_fan_chart_max_history_window_is_honoured(arima_fit):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    fitted, y = arima_fit
    n = y.shape[0]

    windowed = fitted.plot_forecast(max_history=25)
    assert windowed.axes[0].get_xlim()[0] == pytest.approx(n - 25)
    matplotlib.pyplot.close(windowed)

    # A window wider than the sample is clamped, never negative.
    clamped = fitted.plot_forecast(max_history=10_000)
    assert clamped.axes[0].get_xlim()[0] == pytest.approx(0.0)
    matplotlib.pyplot.close(clamped)

    # ...and None means the whole series.
    whole = fitted.plot_forecast(max_history=None)
    assert whole.axes[0].get_xlim()[0] == pytest.approx(0.0)
    matplotlib.pyplot.close(whole)
