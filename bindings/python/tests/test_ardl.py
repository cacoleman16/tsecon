"""Golden and behavioral tests for ``tsecon.ardl_bounds_test`` — the ARDL /
unrestricted error-correction model and the Pesaran-Shin-Smith (2001)
bounds test for a level relationship.

Re-pins ``fixtures/ardl.json`` (statsmodels 0.15 ``UECM`` / ``ARDL`` /
``ardl_select_order`` / ``OLS`` and the PSS response surface at the
corrected key ``k = n_x``; see the generator header for the two
statsmodels defects documented rather than inherited) through the Python
surface, cross-checks statsmodels directly on fresh data when it is
importable, and exercises the ``order`` coercion, the sentinel refusals,
the teaching errors, determinism and the docstring contracts.
"""
import ast
import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
FX = json.loads((FIX / "ardl.json").read_text())
SYSTEMS = {k: (np.asarray(v["y"], float), np.asarray(v["x"], float)) for k, v in FX["systems"].items()}
CASES = FX["cases"]
STUB = Path(__file__).parents[1] / "python" / "tsecon" / "__init__.pyi"


def _nan(v):
    return float("nan") if v is None else v


def _arr(v):
    return np.array([_nan(e) for e in v], dtype=float)


def _kwargs(c):
    kw = dict(case=c["case"], alpha=c["alpha"])
    for name in ("lags", "order", "ic", "max_lag", "max_order"):
        v = c[f"{name}_arg"]
        if v is not None:
            kw[name] = v
    return kw


def _case_id(c):
    return "-".join(str(b) for b in (c["system"], f"case{c['case']}", c["lags_arg"], c["order_arg"],
                                      c["ic_arg"], c["max_lag_arg"], c["max_order_arg"]))


def _close(a, e, rtol, floor=0.0):
    a, e = np.asarray(a, float), np.asarray(e, float)
    assert a.shape == e.shape
    scale = np.maximum(np.abs(e), floor)
    nan = np.isnan(e)
    assert np.array_equal(np.isnan(a), nan)
    assert np.all(np.abs(a - e)[~nan] <= rtol * scale[~nan]), np.max(np.abs(a - e)[~nan] / scale[~nan])


def _close_vec(a, e, rtol=1e-10):
    e = _arr(e)
    big = np.max(np.abs(e)) if e.size else 0.0
    _close(a, e, rtol, 1e-3 * big)


def _close_p(a, e):
    a, e = np.asarray(a, float), _arr(np.atleast_1d(e))
    a = np.atleast_1d(a)
    nan = np.isnan(e)
    assert np.array_equal(np.isnan(a), nan)
    assert np.all(np.abs(a - e)[~nan] <= 1e-14 + 1e-9 * np.abs(e)[~nan])


# ------------------------------------------------------------ the golden

@pytest.mark.parametrize("c", CASES, ids=[_case_id(c) for c in CASES])
def test_matches_fixture(c):
    y, x = SYSTEMS[c["system"]]
    r = tsecon.ardl_bounds_test(y, x, **_kwargs(c))
    for key in ("lags", "order", "selected", "ic", "max_lag", "max_order", "n_models", "nobs",
                "n_total", "hold_back", "df_resid", "n_params", "n_det", "n_x", "n_restrictions",
                "case", "trend", "param_names", "long_run_names", "verdict", "alpha"):
        assert r[key] == c[key], key
    if c["ic_value"] is None:
        assert math.isnan(r["ic_value"])
    else:
        assert r["ic_value"] == pytest.approx(c["ic_value"], rel=1e-8, abs=1e-8)
    for key in ("params", "se", "tvalues", "long_run", "long_run_se", "long_run_tvalues"):
        _close_vec(r[key], c[key])
    _close_p(r["pvalues"], c["pvalues"])
    _close_p(r["long_run_pvalues"], c["long_run_pvalues"])
    cov = np.asarray(r["cov"])
    ecov = np.asarray(c["cov"], float)
    _close(cov, ecov, 1e-10, 1e-3 * np.max(np.abs(ecov)))
    e_resid = np.asarray(c["resid"], float)
    _close(r["resid"], e_resid, 1e-10, np.max(np.abs(e_resid)))
    for key in ("ssr", "llf", "aic", "bic", "hqic", "adjustment"):
        assert r[key] == pytest.approx(c[key], rel=1e-10), key
    # F: the exact-rational referee at 1e-10, statsmodels' UECM Wald at
    # 5e-10 (its own rounding reaches 1.5e-10), the levels f_test at 1e-8
    assert r["statistic"] == pytest.approx(c["statistic_exact"], rel=1e-10)
    assert r["statistic"] == pytest.approx(c["statistic"], rel=5e-10)
    assert r["statistic"] == pytest.approx(c["statistic_levels"], rel=1e-8)
    assert r["t_statistic"] == pytest.approx(c["t_statistic_exact"], rel=1e-10)
    assert r["t_statistic"] == pytest.approx(c["t_statistic"], rel=1e-10)
    np.testing.assert_allclose(r["crit_levels"], [0.90, 0.95, 0.99, 0.999])
    _close(r["crit_lower"], _arr(c["crit_lower"]), 1e-12)
    _close(r["crit_upper"], _arr(c["crit_upper"]), 1e-12)
    _close_p(r["p_value_lower"], c["p_value_lower"])
    _close_p(r["p_value_upper"], c["p_value_upper"])
    if r["n_x"] > 10:
        assert r["verdict"] == "unavailable" and "stops at 10" in r["crit_note"]
    else:
        assert r["crit_note"] is None


def test_fixture_covers_the_option_grid():
    assert len(CASES) >= 120
    assert {c["case"] for c in CASES} == {1, 2, 3, 4, 5}
    assert {c["ic_arg"] for c in CASES if c["selected"]} == {"aic", "bic", "hqic"}
    modes = {(c["lags_arg"] is None, c["order_arg"] is None) for c in CASES}
    assert modes == {(False, False), (True, True), (True, False), (False, True)}
    assert {c["verdict"] for c in CASES} == {"level_relationship", "no_level_relationship",
                                              "inconclusive", "unavailable"}
    assert sum(c["lags"] < max(c["order"]) for c in CASES) >= 20  # the nobs defect is exercised
    assert all(c["ic_margin"] > 1e-6 for c in CASES if c["selected"])


def test_the_documented_stories():
    # macrodata: BIC picks ARDL(1, 1, 1); F rejects at 5% while t(y.L1) is
    # -1.61 — degenerate case #1, the model card's headline.
    s = FX["stories"]["macrodata_bic_case3"]
    y, x = SYSTEMS["macrodata"]
    r = tsecon.ardl_bounds_test(y, x)
    assert (r["lags"], r["order"]) == (1, [1, 1])
    assert r["statistic"] == pytest.approx(4.891036357958964, rel=1e-10)
    assert r["t_statistic"] == pytest.approx(-1.611255354919633, rel=1e-10)
    assert r["p_value_upper"] == pytest.approx(s["p_value_upper"], rel=1e-9)
    assert r["verdict"] == "level_relationship"
    assert s["sm_p_value_upper"] == pytest.approx(0.0240, abs=1e-4)  # statsmodels' wrong key
    # Danish: BIC's unrestricted pick (3, 1, 0) has no UECM.
    d = FX["stories"]["danish_bic_case3"]
    y, x = SYSTEMS["danish"]
    r = tsecon.ardl_bounds_test(y, x)
    assert (r["lags"], r["order"]) == (3, [1, 1, 1])
    assert d["unrestricted"] == [3, 1, 0] and d["from_ardl"].startswith("raises")


# -------------------------------------------- statsmodels on fresh data

def _fresh(seed=4, T=180, k=2):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((T, k)), axis=0)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = (y[t - 1] - 0.3 * (y[t - 1] - 0.7 * x[t - 1, 0] + 0.2 * x[t - 1, 1])
                + 0.4 * (x[t, 0] - x[t - 1, 0]) + rng.standard_normal())
    return y, x


@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
def test_agrees_with_statsmodels_on_fresh_data(case):
    sm_ardl = pytest.importorskip("statsmodels.tsa.ardl")
    from scipy import stats
    from statsmodels.tsa.ardl import pss_critical_values as pss

    y, x = _fresh()
    trend = {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}[case]
    r = tsecon.ardl_bounds_test(y, x, lags=3, order=[1, 2], case=case)
    u = sm_ardl.UECM(y, 3, x, {0: 1, 1: 2}, trend=trend).fit()
    np.testing.assert_allclose(r["params"], np.asarray(u.params), rtol=1e-9)
    np.testing.assert_allclose(r["se"], np.asarray(u.bse), rtol=1e-9)
    bt = u.bounds_test(case=case)
    assert r["statistic"] == pytest.approx(bt.statistic, rel=1e-9)
    np.testing.assert_allclose(r["long_run"], -np.delete(np.asarray(u.ci_params), r["n_det"]), rtol=1e-9)
    # the p-values tsecon reports are statsmodels' surface at k = n_x ...
    for i1, key in ((False, "p_value_lower"), (True, "p_value_upper")):
        k = (2, case, i1)
        a = pss.small_p[k] if r["statistic"] > pss.stat_star[k] else pss.large_p[k]
        z = sum(c * np.log(r["statistic"]) ** i for i, c in enumerate(a))
        assert r[key] == pytest.approx(stats.norm.sf(z), rel=1e-9, abs=1e-14)
    np.testing.assert_array_equal(r["crit_upper"], pss.crit_vals[(2, case, True)])
    # ... and NOT the k = n_x + 1 row statsmodels' bounds_test reads
    np.testing.assert_array_equal(bt.critical_values["upper"].to_numpy(), pss.crit_vals[(3, case, True)])
    assert not np.allclose(r["crit_upper"], bt.critical_values["upper"].to_numpy())


@pytest.mark.parametrize("ic", ["aic", "bic", "hqic"])
def test_selection_agrees_with_statsmodels_admissible_subgrid(ic):
    sm_ardl = pytest.importorskip("statsmodels.tsa.ardl")
    y, x = _fresh(seed=9)
    r = tsecon.ardl_bounds_test(y, x, ic=ic, max_lag=3, max_order=3)
    sel = sm_ardl.ardl_select_order(y, 3, x, 3, trend="c", ic=ic)
    adm = sorted(
        (v, k[0], [k[1][j] for j in range(2)]) for v, k in getattr(sel, ic).items()
        if k[0] is not None and all(k[1][j] is not None and k[1][j] >= 1 for j in range(2))
    )
    assert len(adm) == r["n_models"] == 27
    assert (r["lags"], r["order"]) == (adm[0][1], adm[0][2])
    assert r["ic_value"] == pytest.approx(adm[0][0], rel=1e-9)


# ---------------------------------------------------- order coercion

def test_order_accepts_int_list_tuple_numpy_and_pandas():
    pd = pytest.importorskip("pandas")
    y, x = SYSTEMS["danish"]
    ref = tsecon.ardl_bounds_test(y, x, lags=2, order=[2, 2, 2])
    for order in (2, np.int64(2), [2, 2, 2], (2, 2, 2), np.array([2, 2, 2]),
                  np.array([2, 2, 2], dtype=np.int32), pd.Series([2, 2, 2])):
        r = tsecon.ardl_bounds_test(y, x, lags=2, order=order)
        assert r["order"] == [2, 2, 2]
        assert r["statistic"] == ref["statistic"]
    mixed = tsecon.ardl_bounds_test(y, x, lags=2, order=np.array([1, 3, 2]))
    assert mixed["order"] == [1, 3, 2]
    assert mixed["param_names"][-6:] == ["D.x1.L0", "D.x2.L0", "D.x2.L1", "D.x2.L2", "D.x3.L0", "D.x3.L1"]


@pytest.mark.parametrize("order, match", [
    ([1.0, 3.0, 2.0], r"order\[0\] = 1\.0 is not an integer"),
    (np.array([1.0, 3.0, 2.0]), r"order\[0\] = .*1\.0.* is not an integer"),
    (2.0, r"order = 2\.0 is not an integer"),
    (True, r"order = True is a bool"),
    ([1, True, 2], r"order\[1\] = True is a bool"),
    ([1, -1, 2], r"order\[1\] = -1 is negative"),
    (-2, r"order = -2 is negative"),
    ("2", r"order = '2' is a string"),
    ({0: 1, 1: 3, 2: 2}, r"order = \{0: 1, 1: 3, 2: 2\} is a mapping or set"),
    ([1, 0, 2], r"order\[1\] = 0 is not supported.*follow-up"),
    ([1, 2], r"order has 2 entries but x has 3 column"),
    ([[1, 2, 3]], r"order\[0\] = \[1, 2, 3\] is not an integer"),
])
def test_bad_orders_are_refused_by_name(order, match):
    y, x = SYSTEMS["danish"]
    with pytest.raises(ValueError, match=match):
        tsecon.ardl_bounds_test(y, x, lags=2, order=order)


def test_absurd_orders_are_refused_by_name():
    y, x = SYSTEMS["danish"]
    with pytest.raises(ValueError, match=r"order.*2\*\*48"):
        tsecon.ardl_bounds_test(y, x, lags=2, order=[1, 2**63, 2])
    with pytest.raises(ValueError, match=r"order.*2\*\*48"):
        tsecon._core.ardl_bounds_test(y, x, lags=2, order=[1, 2**63, 2])
    with pytest.raises(ValueError, match=r"order = 1000000 .* too large for T = 55|too large for T = 55"):
        tsecon.ardl_bounds_test(y, x, lags=2, order=10**6)


# ----------------------------------------------------------- refusals

def test_sentinels_and_teaching_errors_name_the_parameter():
    y, x = SYSTEMS["sim_k2"]
    with pytest.raises(ValueError, match=r'ic = "aic" was given but both lags = 2 and order are fixed'):
        tsecon.ardl_bounds_test(y, x, lags=2, order=1, ic="aic")
    with pytest.raises(ValueError, match=r"max_lag = 3 was given but lags = 2 is fixed"):
        tsecon.ardl_bounds_test(y, x, lags=2, max_lag=3)
    with pytest.raises(ValueError, match=r"max_order = 3 was given but order = \[1, 1\] is fixed"):
        tsecon.ardl_bounds_test(y, x, order=1, max_order=3)
    with pytest.raises(ValueError, match=r"lags = 0 is not a valid ARDL order.*silently"):
        tsecon.ardl_bounds_test(y, x, lags=0)
    with pytest.raises(ValueError, match=r"max_lag = 0"):
        tsecon.ardl_bounds_test(y, x, max_lag=0)
    with pytest.raises(ValueError, match=r"max_order = 0"):
        tsecon.ardl_bounds_test(y, x, max_order=0)
    for case in (0, 6):
        with pytest.raises(ValueError, match=rf"case = {case} is not a Pesaran-Shin-Smith case"):
            tsecon.ardl_bounds_test(y, x, case=case)
    with pytest.raises(ValueError, match=r"case=-1"):
        tsecon.ardl_bounds_test(y, x, case=-1)
    with pytest.raises(ValueError, match=r"lags=-1"):
        tsecon.ardl_bounds_test(y, x, lags=-1)
    with pytest.raises(ValueError, match=r"lags=\d+ is at or beyond 2\*\*48"):
        tsecon.ardl_bounds_test(y, x, lags=2**63)
    with pytest.raises(ValueError, match=r"lags = 1099511627776 .*too large for T = 160|too large for T = 160"):
        tsecon.ardl_bounds_test(y, x, lags=2**40, order=1)
    for a in (0.0, 1.0, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"alpha = .* is outside \(0, 1\)"):
            tsecon.ardl_bounds_test(y, x, alpha=a)
    with pytest.raises(ValueError, match=r'unknown ic "sic"'):
        tsecon.ardl_bounds_test(y, x, ic="sic")
    with pytest.raises(TypeError):
        tsecon.ardl_bounds_test(y, x, ic=3)


def test_grid_guard_names_the_caps_and_the_fix():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((300, 8)), axis=0)
    y = np.cumsum(rng.standard_normal(300))
    with pytest.raises(ValueError, match=r"262144 candidate models.*100000 guard.*max_order = 4 \(the default\).*Fix order"):
        tsecon.ardl_bounds_test(y, x)
    r = tsecon.ardl_bounds_test(y, x, order=1)  # lags-only search: 4 models
    assert r["n_models"] == 4 and r["order"] == [1] * 8


def test_malformed_inputs_are_refused_not_panics():
    y, x = SYSTEMS["sim_k2"]
    with pytest.raises(ValueError, match=r"x .*rows.*expected 160, got 159"):
        tsecon.ardl_bounds_test(y, x[:-1])
    yy = y.copy()
    yy[5] = np.nan
    with pytest.raises(ValueError, match=r"non-finite value .* index 5 of y"):
        tsecon.ardl_bounds_test(yy, x)
    xx = x.copy()
    xx[4, 1] = np.inf
    with pytest.raises(ValueError, match=r"non-finite value .* x at row 4, column 1"):
        tsecon.ardl_bounds_test(y, xx)
    with pytest.raises(ValueError, match=r"x must hold at least one regressor"):
        tsecon.ardl_bounds_test(y, np.empty((160, 0)))
    with pytest.raises(ValueError, match=r"too large for T = 0"):
        tsecon.ardl_bounds_test(np.empty(0), np.empty((0, 1)))
    with pytest.raises(ValueError, match=r"too large for T = 1"):
        tsecon.ardl_bounds_test(np.ones(1), np.ones((1, 1)), lags=1, order=1)
    with pytest.raises(ValueError, match=r"too large for T = 15"):
        tsecon.ardl_bounds_test(y[:15], x[:15])
    with pytest.raises(TypeError, match=r"2-D"):
        tsecon.ardl_bounds_test(y, x[:, 0])
    with pytest.raises(ValueError, match=r"numerically singular"):
        tsecon.ardl_bounds_test(y, np.column_stack([x[:, 0], x[:, 0]]), lags=1, order=1)
    # a constant y is named, not read as a singular design blaming x
    for kw in (dict(lags=1, order=1, case=1), dict(lags=1, order=1, case=4), {}):
        with pytest.raises(ValueError, match=r"y is constant.*check y"):
            tsecon.ardl_bounds_test(np.full(160, 2.0), x, **kw)
    with pytest.raises(ValueError, match=r"numerically singular.*constant column of x"):
        tsecon.ardl_bounds_test(y, np.full((160, 1), 3.0))


# ---------------------------------------------------------- behaviour

def test_eleven_regressors_have_no_bounds_and_say_so():
    y, x = SYSTEMS["sim_k11"]
    r = tsecon.ardl_bounds_test(y, x, lags=1, order=1)
    assert np.isfinite(r["statistic"]) and np.all(np.isnan(r["crit_upper"]))
    assert math.isnan(r["p_value_upper"]) and r["verdict"] == "unavailable"
    assert "n_x = 11" in r["crit_note"]


def test_verdict_reads_the_p_values_at_alpha():
    y, x = SYSTEMS["macrodata"]
    r = tsecon.ardl_bounds_test(y, x)
    pl, pu = r["p_value_lower"], r["p_value_upper"]
    assert pl < pu  # the I(0) bound is the smaller one
    for alpha in (0.01, 0.02, 0.03, 0.05, 0.10):
        v = tsecon.ardl_bounds_test(y, x, alpha=alpha)["verdict"]
        expected = ("level_relationship" if pu < alpha else
                    "no_level_relationship" if pl >= alpha else "inconclusive")
        assert v == expected, alpha
    assert tsecon.ardl_bounds_test(y, x, alpha=0.02)["verdict"] == "inconclusive"
    assert tsecon.ardl_bounds_test(y, x, alpha=0.01)["verdict"] == "no_level_relationship"


def test_determinism_and_pandas_inputs():
    pd = pytest.importorskip("pandas")
    y, x = SYSTEMS["sim_k2"]
    a = tsecon.ardl_bounds_test(y, x)
    b = tsecon.ardl_bounds_test(pd.Series(y), pd.DataFrame(x, columns=["a", "b"]))
    c = tsecon.ardl_bounds_test(list(y), np.asfortranarray(x))
    for key in a:
        for other in (b, c):
            va, vb = a[key], other[key]
            if isinstance(va, np.ndarray):
                assert np.array_equal(va, vb, equal_nan=True), key
            elif isinstance(va, float) and math.isnan(va):
                assert math.isnan(vb), key
            else:
                assert va == vb, key


def test_case_sets_the_deterministics_and_the_restrictions():
    y, x = SYSTEMS["sim_k2"]
    seen = {}
    for case in (1, 2, 3, 4, 5):
        r = tsecon.ardl_bounds_test(y, x, lags=1, order=1, case=case)
        seen[case] = r
        assert r["trend"] == {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}[case]
        assert r["n_restrictions"] == 3 + (case in (2, 4))
        assert r["long_run_names"] == ["const", "trend"][: r["n_det"]] + ["x1", "x2"]
    # cases 2/3 (and 4/5) are the same regression, tested differently
    np.testing.assert_array_equal(seen[2]["params"], seen[3]["params"])
    np.testing.assert_array_equal(seen[4]["params"], seen[5]["params"])
    assert seen[2]["statistic"] != seen[3]["statistic"]


# --------------------------------------------------------- contracts

def _tokens(fn):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))


def test_docstring_names_every_returned_key():
    y, x = SYSTEMS["sim_k2"]
    keys = set(tsecon.ardl_bounds_test(y, x).keys())
    missing = keys - _tokens(tsecon.ardl_bounds_test)
    assert not missing, missing
    assert len(keys) == 46


def test_stub_docstring_is_the_runtime_docstring():
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    stub = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ardl_bounds_test")
    assert ast.get_docstring(stub) == inspect.cleandoc(tsecon._core.ardl_bounds_test.__doc__)
    params = [a.arg for a in stub.args.args]
    assert params == list(inspect.signature(tsecon._core.ardl_bounds_test).parameters)


def test_no_ellipsis_defaults_and_documented_values():
    sig = inspect.signature(tsecon._core.ardl_bounds_test)
    for p in sig.parameters.values():
        assert p.default is not Ellipsis, p
    d = {k: v.default for k, v in sig.parameters.items()}
    assert d["lags"] is None and d["order"] is None and d["case"] == 3
    assert d["ic"] is None and d["max_lag"] is None and d["max_order"] is None
    assert d["alpha"] == 0.05
