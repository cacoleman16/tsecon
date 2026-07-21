"""check_series results facade: the dict contract, the sectioned summary,
the multiple-testing footer, and the 2x2 diagnostic figure."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

import tsecon
from tsecon.results import CheckSeriesResults, check_series
from tsecon.results._check import CheckSeriesResults as _direct

#: Section headers the univariate summary must always render, whatever the
#: verdicts were — every family reports, even a clean one.
UNI_SECTIONS = [
    "Stationarity (ADF + KPSS)",
    "Serial correlation",
    "ARCH effects",
    "Normality (Jarque-Bera)",
    "Structural breaks",
    "Long memory (GPH)",
    "Seasonality",
    "Recommendations",
]

MULTI_SECTIONS = [
    "Per-series integration (ADF + KPSS)",
    "Cointegration (Johansen)",
    "VAR lag selection",
    "Stability",
    "Recommendations",
]


def random_walk(n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


def stationary_panel(n: int = 240, k: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.0], [0.0, 0.1, 0.3]])
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.normal(size=k) * 0.5
    return y


def expected_footer(res: dict) -> str:
    mt = res["multiple_testing"]
    return (
        f"{mt['n_tests']} hypothesis tests at alpha={float(mt['alpha']):g} "
        f"- expect ~{mt['expected_false_rejections']:.2f} false alarms "
        f"from the {mt['n_true_null']} true-null tests on a clean series."
    )


@pytest.fixture(scope="module")
def uni() -> CheckSeriesResults:
    return check_series(random_walk())


@pytest.fixture(scope="module")
def multi() -> CheckSeriesResults:
    return check_series(stationary_panel())


# --------------------------------------------------------------------------- #
# 1. backward compatibility: it is still the dict check_series always returned
# --------------------------------------------------------------------------- #
def test_is_a_dict_matching_raw_check_series_exactly(uni):
    raw = tsecon.check_series(random_walk())

    assert isinstance(uni, dict)
    assert set(uni) == set(raw)
    # the battery is fully deterministic, so equality is exact, not approx
    assert uni.to_dict() == raw
    assert type(uni.to_dict()) is dict
    assert uni["kind"] == "univariate"


def test_multivariate_dict_contract(multi):
    raw = tsecon.check_series(stationary_panel())
    assert isinstance(multi, dict)
    assert multi.to_dict() == raw
    assert multi["kind"] == "multivariate"
    assert multi["k"] == 3


def test_data_is_an_attribute_not_a_dict_key(uni, multi):
    assert "_data" not in uni and "data" not in uni
    assert "_data" not in multi and "data" not in multi
    assert uni._data.shape == (300,)
    assert multi._data.shape == (240, 3)


def test_dict_conveniences_json_pickle_unpacking(uni, multi):
    for res in (uni, multi):
        # the report is JSON-serializable as-is (check_series guarantees it)
        assert json.loads(json.dumps(res)) == res.to_dict()

        unpacked = {**res}
        assert set(unpacked) == set(res)

        revived = pickle.loads(pickle.dumps(res))
        assert isinstance(revived, CheckSeriesResults)
        assert revived.to_dict() == res.to_dict()
        assert revived.summary() == res.summary()


def test_module_constructor_and_exports():
    from tsecon import results

    assert results.check_series is check_series
    assert _direct is CheckSeriesResults
    assert "CheckSeriesResults" in results.__all__
    assert "check_series" in results.__all__

    res = results.check_series(random_walk(120, seed=9), alpha=0.10)
    assert type(res) is CheckSeriesResults
    assert res["alpha"] == pytest.approx(0.10)
    assert res.to_dict() == tsecon.check_series(random_walk(120, seed=9), alpha=0.10)


# --------------------------------------------------------------------------- #
# 2. the univariate summary
# --------------------------------------------------------------------------- #
def test_univariate_summary_header_descriptives_and_sections(uni):
    text = uni.summary()
    assert isinstance(text, str)

    assert "check_series — univariate" in text
    assert f"n={uni['n']}" in text and "alpha=0.05" in text

    d = uni["descriptives"]
    assert f"mean {d['mean']:+.4f}" in text
    assert f"sd {d['sd']:.4f}" in text
    assert f"outliers {uni['outliers']['count']}" in text

    for section in UNI_SECTIONS:
        assert section in text, section


def test_univariate_summary_quadrant_scale_and_family_numbers(uni):
    text = uni.summary()
    st = uni["stationarity"]
    assert f"quadrant {st['quadrant']} -> recommendation {st['recommendation']}" in text
    assert f"analysis scale: {uni['analysis_scale']['scale']}" in text
    # a random walk differences, and every downstream family says so
    assert uni["analysis_scale"]["scale"] == "first_difference"
    assert "Serial correlation — on first_difference" in text
    assert "ARCH effects — on first_difference" in text
    assert "Structural breaks — on first_difference" in text
    assert "Long memory (GPH) — on level" in text  # GPH stays on the level

    assert f"stat {st['adf_statistic']:+.4f}" in text
    assert f"stat {uni['arch_effects']['statistic']:.4f}" in text
    assert f"stat {uni['normality']['statistic']:.4f}" in text
    sf = uni["breaks"]["sup_f"]
    assert f"stat {sf['stat']:.4f}" in text
    assert f"break_date {sf['break_date']}" in text
    gph = uni["long_memory"]["gph"]
    assert f"d {gph['d']:+.4f}" in text


def test_univariate_summary_lists_every_recommendation(uni):
    text = uni.summary()
    recs = uni["recommendations"]
    # the random-walk DGP fires the persistence pair
    topics = [r["topic"] for r in recs]
    assert "unit_root" in topics and "persistent_regressor" in topics

    for i, rec in enumerate(recs, start=1):
        assert f"{i:>2}. {rec['topic']}" in text
        assert f"functions: {', '.join(rec['functions'])}" in text
        # findings and caveats are wrapped; their opening words must survive
        assert rec["finding"].split()[0] in text
        assert "caveat:" in text


def test_multiple_testing_footer_renders_verbatim(uni, multi):
    for res in (uni, multi):
        assert expected_footer(res) in res.summary()
    # the arithmetic shown is n_true_null * alpha, never a corrected number
    mt = uni["multiple_testing"]
    assert mt["expected_false_rejections"] == pytest.approx(
        mt["n_true_null"] * mt["alpha"]
    )


def test_repr_is_the_summary(uni, multi):
    assert repr(uni) == uni.summary()
    assert repr(multi) == multi.summary()


# --------------------------------------------------------------------------- #
# 3. the multivariate summary
# --------------------------------------------------------------------------- #
def test_multivariate_summary_sections_and_verdict_table(multi):
    text = multi.summary()
    assert "check_series — multivariate" in text
    assert f"n={multi['n']}" in text and f"k={multi['k']}" in text

    for section in MULTI_SECTIONS:
        assert section in text, section

    for entry in multi["per_series"]:
        assert f"y{entry['index'] + 1}" in text
        assert entry["verdict"] in text
        assert entry["recommendation"] in text

    sel = multi["var_lag_selection"]
    assert f"selected by BIC: {sel['selected_by_bic']}" in text
    assert f"{sel['bic'][0]:.4f}" in text
    stab = multi["stability"]
    assert f"min_root {stab['min_root']:.4f}" in text
    assert ("stable" in text) if stab["is_stable"] else ("UNSTABLE" in text)


def test_multivariate_summary_lists_every_recommendation(multi):
    text = multi.summary()
    topics = [r["topic"] for r in multi["recommendations"]]
    # the all-stationary panel routes to a levels VAR, LP, and connectedness
    assert "var" in topics and "single_shock_irf" in topics and "spillovers" in topics
    for i, rec in enumerate(multi["recommendations"], start=1):
        assert f"{i:>2}. {rec['topic']}" in text
        assert f"functions: {', '.join(rec['functions'])}" in text


def test_johansen_skip_reason_is_reported_for_a_stationary_panel(multi):
    # this panel is all-I(0), so the trace test's premise fails and the
    # summary must say so rather than silently omitting the family
    assert "skipped_reason" in multi["cointegration"]
    assert multi["cointegration"]["skipped_reason"].split(":")[0] in multi.summary()


# --------------------------------------------------------------------------- #
# 4. the 2x2 diagnostic figure
# --------------------------------------------------------------------------- #
def test_plot_diagnostics_univariate_panels(uni, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = uni.plot_diagnostics()
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 4

        ax_series, ax_acf, ax_pacf, ax_hist = fig.axes
        assert ax_series.get_title(loc="left") == "series — level"
        assert len(ax_series.lines) == 1
        assert ax_series.lines[0].get_ydata() == pytest.approx(uni._data)

        sc = uni["serial_correlation"]
        assert ax_acf.get_title(loc="left") == f"ACF — {sc['computed_on']}"
        assert ax_pacf.get_title(loc="left") == f"PACF — {sc['computed_on']}"
        # one bar per reported lag, and the two dashed band lines
        assert len(ax_acf.patches) == len(sc["acf"])
        assert len(ax_pacf.patches) == len(sc["pacf"])
        band = float(sc["conf_band"])
        band_levels = sorted(line.get_ydata()[0] for line in ax_acf.lines[:2])
        assert band_levels == pytest.approx([-band, band])

        assert ax_hist.get_title(loc="left").startswith("histogram")
        assert len(ax_hist.patches) > 0        # the histogram bars
        assert len(ax_hist.lines) == 1         # the normal overlay
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    out = tmp_path / "check_uni.png"
    fig2 = uni.plot_diagnostics(path=out)
    try:
        assert out.exists() and out.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig2)


def test_plot_diagnostics_multivariate_panels(multi):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    fig = multi.plot_diagnostics()
    try:
        assert len(fig.axes) == 4
        ax_series, ax_acf, ax_pacf, ax_hist = fig.axes
        k = multi["k"]
        assert len(ax_series.lines) == k       # one path per column
        # per-series correlograms plus the two band lines and the zero line
        assert len(ax_acf.lines) >= k + 2
        assert len(ax_pacf.lines) >= k + 2
        assert ax_hist.get_title(loc="left").startswith("pooled standardized")
        assert len(ax_hist.patches) > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plot_needs_the_original_data():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

    bare = CheckSeriesResults(tsecon.check_series(random_walk(100, seed=4)))
    assert bare.summary()  # the summary needs only the dict...
    with pytest.raises(ValueError, match="not built by CheckSeriesResults.run"):
        bare.plot_diagnostics()  # ...but the figure needs the series
