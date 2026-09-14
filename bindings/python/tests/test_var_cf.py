"""Golden, parity and behavioural tests for the 0.10.0 VAR additions:
`tsecon.var_conditional_forecast` (Doan-Litterman-Sims / Waggoner-Zha
hard-path conditional forecasts), `tsecon.var_diagnostics` (multivariate
Portmanteau, multivariate Jarque-Bera, stability roots) and
`tsecon.var_select_order` (lag-order selection on a common sample).

Re-pins fixtures/var_cf.json and fixtures/var_diag.json through the Python
surface (the NumPy closed form at 1e-10 and the statsmodels VARMAX
Kalman-conditioning leg at 1e-8; statsmodels `test_whiteness` /
`test_normality` / `roots` / `select_order` at 1e-10 — see the generator
headers for the honest grading), then runs statsmodels DIRECTLY on the
bundled macrodata in levels (a system the fixtures do not store) and on a
fresh simulated VAR, checks the bitwise identity with `var_forecast`, the
NaN-array / pandas spellings of `conditions`, determinism, the teaching
errors, and the docstring / signature contracts.
"""
import inspect
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import tsecon

try:  # extras-gated: the live-parity tests need statsmodels
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.statespace.varmax import VARMAX

    HAVE_SM = True
except ImportError:  # pragma: no cover - extras absent
    HAVE_SM = False

FIX = Path(__file__).parents[3] / "fixtures"
CF = json.loads((FIX / "var_cf.json").read_text(encoding="utf-8"))
DG = json.loads((FIX / "var_diag.json").read_text(encoding="utf-8"))

needs_sm = pytest.mark.skipif(not HAVE_SM, reason="statsmodels not installed")


def _series(fx, name):
    return np.array(fx["series"][name])


def _cf_call(case, **override):
    kw = dict(
        conditions=case["conditions"],
        lags=case["p"],
        trend=case["trend"],
        steps=case["steps"],
        alpha=case["alpha"],
    )
    kw.update(override)
    return tsecon.var_conditional_forecast(_series(CF, case["series"]), **kw)


def _close(a, b, tol):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                               rtol=tol, atol=tol)


# ------------------------------------------------ conditional forecast: golden

@pytest.mark.parametrize("case", CF["cases"], ids=[c["name"] for c in CF["cases"]])
def test_conditional_forecast_repins_the_closed_form(case):
    r = _cf_call(case)
    assert r["steps"] == case["steps"] and r["alpha"] == case["alpha"]
    assert r["n_constrained"] == case["n_constrained"]
    assert r["constrained"] == case["constrained"]
    for key in ("point", "unconditional", "cov", "se", "unconditional_se", "lower",
                "upper", "shocks", "orth_shocks"):
        _close(r[key], case[key], 1e-10)
    assert r["mahalanobis"] == pytest.approx(case["mahalanobis"], rel=1e-10, abs=1e-12)
    assert r["mahalanobis_pvalue"] == pytest.approx(case["mahalanobis_pvalue"], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("case", CF["cases"], ids=[c["name"] for c in CF["cases"]])
def test_conditional_forecast_matches_the_varmax_kalman_leg(case):
    r = _cf_call(case)
    _close(r["point"], case["bgl_point"], 1e-8)
    _close(r["cov"], case["bgl_cov"], 1e-8)
    _close(r["shocks"], case["bgl_shocks"], 1e-8)


@pytest.mark.parametrize("case", CF["cases"], ids=[c["name"] for c in CF["cases"]])
def test_unconditional_path_is_var_forecast_bitwise(case):
    r = _cf_call(case)
    fc = tsecon.var_forecast(_series(CF, case["series"]), lags=case["p"],
                             steps=case["steps"], trend=case["trend"], alpha=case["alpha"])
    assert r["unconditional"] == fc["point"]
    # And its se is the marginal interval's se, recovered from the bounds.
    z = (np.asarray(fc["upper"]) - np.asarray(fc["point"]))
    _close(z / (np.asarray(fc["upper"]) - np.asarray(fc["lower"])) * 2.0, np.ones_like(z), 1e-12)


def test_pinned_cells_and_variance_reduction():
    case = next(c for c in CF["cases"] if c["name"] == "var2c_path")
    r = _cf_call(case)
    cond = case["conditions"]
    se = np.asarray(r["se"])
    unc = np.asarray(r["unconditional_se"])
    point = np.asarray(r["point"])
    for h, row in enumerate(cond):
        for j, v in enumerate(row):
            if v is not None:
                assert r["constrained"][h][j]
                assert point[h, j] == v and se[h, j] == 0.0
                assert r["lower"][h][j] == v == r["upper"][h][j]
                assert all(x == 0.0 for x in r["cov"][h][j])
            else:
                assert not r["constrained"][h][j]
                assert 0.0 < se[h, j] <= unc[h, j] * (1 + 1e-12)
    assert r["mahalanobis"] > 0.0 and 0.0 < r["mahalanobis_pvalue"] <= 1.0
    # Conditioning at the unconditional value moves nothing but the variance.
    case0 = next(c for c in CF["cases"] if c["name"] == "var2c_at_unconditional")
    r0 = _cf_call(case0)
    assert np.max(np.abs(np.asarray(r0["shocks"]))) < 1e-12
    assert r0["mahalanobis_pvalue"] == pytest.approx(1.0, abs=1e-12)
    _close(r0["point"], r0["unconditional"], 1e-12)
    assert np.asarray(r0["se"])[1, 0] < np.asarray(r0["unconditional_se"])[1, 0]


def test_nan_grid_dataframe_and_short_list_spellings_agree():
    case = next(c for c in CF["cases"] if c["name"] == "var2c_short_rows")
    y = _series(CF, case["series"])
    ref = _cf_call(case)
    # A NumPy float grid with NaN for the free cells, padded to `steps`.
    grid = np.full((case["steps"], 3), np.nan)
    for h, row in enumerate(case["conditions"]):
        for j, v in enumerate(row):
            if v is not None:
                grid[h, j] = v
    r_nan = tsecon.var_conditional_forecast(y, grid, lags=case["p"], trend=case["trend"])
    assert r_nan["point"] == ref["point"] and r_nan["se"] == ref["se"]
    assert r_nan["steps"] == case["steps"]  # steps defaults to len(conditions)
    # A pandas DataFrame with missing entries.
    pd = pytest.importorskip("pandas")
    r_pd = tsecon.var_conditional_forecast(
        pd.DataFrame(y), pd.DataFrame(grid), lags=case["p"], trend=case["trend"]
    )
    assert r_pd["point"] == ref["point"] and r_pd["shocks"] == ref["shocks"]
    # Integer pins in a nested list are fine.
    r_int = tsecon.var_conditional_forecast(
        y, [[1, None, None]], lags=case["p"], trend=case["trend"], steps=2
    )
    assert r_int["point"][0][0] == 1.0 and r_int["n_constrained"] == 1


def test_conditional_forecast_is_deterministic():
    case = CF["cases"][0]
    a, b = _cf_call(case), _cf_call(case)
    assert a == b


# ----------------------------------------------- conditional forecast: live

def _macro_levels():
    md = sm.datasets.macrodata.load_pandas().data
    g = 100.0 * np.diff(np.log(md["realgdp"].to_numpy(dtype=float)))
    return np.column_stack([g, md["infl"].to_numpy(dtype=float)[1:],
                            md["tbilrate"].to_numpy(dtype=float)[1:]])


def _varmax_smooth(y, res, H, cond, trend):
    """statsmodels VARMAX(...NaN future...).smooth(params) at the OLS fit."""
    k, p, T = res.neqs, res.k_ar, y.shape[0]
    mod = VARMAX(np.vstack([y, cond]), order=(p, 0), trend=trend, error_cov_type="unstructured")
    P = np.linalg.cholesky(np.asarray(res.sigma_u))
    params = np.zeros(len(mod.param_names))
    for i, nm in enumerate(mod.param_names):
        if nm.startswith("intercept."):
            params[i] = res.intercept[int(nm.split(".y")[1]) - 1]
        elif nm.startswith("L"):
            lag_s, src, eq = nm.split(".")
            params[i] = res.coefs[int(lag_s[1:]) - 1][int(eq[1:]) - 1, int(src[1:]) - 1]
        elif nm.startswith("sqrt.var."):
            j = int(nm.split(".y")[1]) - 1
            params[i] = P[j, j]
        else:
            a, b = nm[len("sqrt.cov."):].split(".")
            ia, ib = int(a[1:]) - 1, int(b[1:]) - 1
            params[i] = P[max(ia, ib), min(ia, ib)]
    sres = mod.smooth(params)
    path = np.asarray(sres.smoothed_state[:k, T:T + H].T)
    cov = np.transpose(np.asarray(sres.smoothed_state_cov[:k, :k, T:T + H]), (2, 0, 1))
    shocks = np.asarray(sres.smoothed_state_disturbance[:k, T - 1:T + H - 1].T)
    return path, cov, shocks


@needs_sm
def test_live_statsmodels_kalman_conditioning_parity_on_macrodata_levels():
    y = _macro_levels()  # growth, inflation, T-bill LEVEL: not a fixture system
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(y).fit(2, trend="c")
    H = 8
    last = y[-1]
    cond = np.full((H, 3), np.nan)
    cond[:3, 1] = last[1]   # inflation holds at its last value for 3 quarters
    cond[:2, 2] = last[2]   # the T-bill rate is on hold for 2 quarters
    r = tsecon.var_conditional_forecast(y, cond, lags=2, trend="c", steps=H)
    path, cov, shocks = _varmax_smooth(y, res, H, cond, "c")
    _close(r["point"], path, 1e-8)
    _close(r["cov"], cov, 1e-8)
    _close(r["shocks"], shocks, 1e-8)
    _close(r["unconditional"], res.forecast(y[-2:], H), 1e-8)
    _close(np.asarray(r["unconditional_se"]) ** 2,
           np.einsum("hjj->hj", np.asarray(res.mse(H))), 1e-8)
    assert r["n_constrained"] == 5


@needs_sm
def test_live_statsmodels_parity_on_a_fresh_var_without_constant():
    rng = np.random.default_rng(20260913)
    k, n = 2, 250
    A = np.array([[0.4, 0.15], [-0.1, 0.5]])
    L = np.array([[1.0, 0.0], [-0.6, 0.7]])
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + L @ rng.standard_normal(k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(y).fit(3, trend="n")
    H = 6
    cond = np.full((H, k), np.nan)
    cond[0, 0], cond[3, 1], cond[5, 0] = 0.5, -0.2, 0.0
    r = tsecon.var_conditional_forecast(y, cond, lags=3, trend="n")
    path, cov, shocks = _varmax_smooth(y, res, H, cond, "n")
    _close(r["point"], path, 1e-8)
    _close(r["cov"], cov, 1e-8)
    _close(r["shocks"], shocks, 1e-8)


# ------------------------------------------------------------- diagnostics

def _diag_blocks():
    out = []
    for case in DG["cases"]:
        for block in case["portmanteau"]:
            out.append((case, block))
    return out


@pytest.mark.parametrize("case,block", _diag_blocks(),
                         ids=[f"{c['name']}-nlags{b['nlags']}" for c, b in _diag_blocks()])
def test_diagnostics_repin_statsmodels(case, block):
    r = tsecon.var_diagnostics(_series(DG, case["series"]), lags=case["p"],
                               trend=case["trend"], nlags=block["nlags"])
    assert r["nlags"] == block["nlags"] and r["portmanteau_df"] == block["df"]
    assert r["nobs"] == case["nobs"] and r["k"] == 3 and r["lags"] == case["p"]
    assert r["portmanteau"] == pytest.approx(block["statistic"], rel=1e-10)
    assert r["portmanteau_adjusted"] == pytest.approx(block["adjusted"], rel=1e-10)
    assert r["portmanteau_pvalue"] == pytest.approx(block["pvalue"], rel=1e-6, abs=1e-12)
    assert r["portmanteau_adjusted_pvalue"] == pytest.approx(block["adjusted_pvalue"], rel=1e-6, abs=1e-12)
    n = case["normality"]
    assert r["jarque_bera_df"] == n["df"] == 6
    assert r["jarque_bera"] == pytest.approx(n["statistic"], rel=1e-10)
    assert r["jarque_bera_pvalue"] == pytest.approx(n["pvalue"], rel=1e-6, abs=1e-12)
    assert r["skewness"] == pytest.approx(n["skewness"], rel=1e-10)
    assert r["kurtosis"] == pytest.approx(n["kurtosis"], rel=1e-10)
    assert r["skewness_pvalue"] == pytest.approx(n["skewness_pvalue"], rel=1e-6, abs=1e-12)
    assert r["kurtosis_pvalue"] == pytest.approx(n["kurtosis_pvalue"], rel=1e-6, abs=1e-12)
    _close(r["skewness_components"], n["skewness_components"], 1e-10)
    _close(r["kurtosis_components"], n["kurtosis_components"], 1e-10)
    _close(r["roots"], case["roots"], 1e-8)
    _close(r["eigenvalue_moduli"], case["eigenvalue_moduli"], 1e-8)
    assert r["is_stable"] == case["is_stable"]
    assert r["is_stable"] == (r["roots"][-1] > 1.0) == (r["eigenvalue_moduli"][0] < 1.0)
    assert r["skewness"] + r["kurtosis"] == pytest.approx(r["jarque_bera"], rel=1e-12)


@needs_sm
def test_diagnostics_live_statsmodels_parity():
    y = _macro_levels()
    for p, trend, nlags in [(2, "c", 12), (1, "c", 6), (2, "n", 10)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = VAR(y).fit(p, trend=trend)
        r = tsecon.var_diagnostics(y, lags=p, trend=trend, nlags=nlags)
        un = res.test_whiteness(nlags=nlags, adjusted=False)
        ad = res.test_whiteness(nlags=nlags, adjusted=True)
        nm = res.test_normality()
        assert r["portmanteau"] == pytest.approx(float(un.test_statistic), rel=1e-10)
        assert r["portmanteau_adjusted"] == pytest.approx(float(ad.test_statistic), rel=1e-10)
        assert r["portmanteau_df"] == un.df
        assert r["portmanteau_pvalue"] == pytest.approx(float(un.pvalue), rel=1e-6, abs=1e-12)
        assert r["portmanteau_adjusted_pvalue"] == pytest.approx(float(ad.pvalue), rel=1e-6, abs=1e-12)
        assert r["jarque_bera"] == pytest.approx(float(nm.test_statistic), rel=1e-10)
        assert r["jarque_bera_pvalue"] == pytest.approx(float(nm.pvalue), rel=1e-6, abs=1e-12)
        _close(r["roots"], np.abs(np.asarray(res.roots)), 1e-8)
        assert r["is_stable"] == bool(res.is_stable())
        assert r["nobs"] == res.nobs
    # The var_fit stability summary and the diagnostics bundle agree.
    fit = tsecon.var_fit(y, lags=2, trend="c")
    r = tsecon.var_diagnostics(y, lags=2, trend="c", nlags=12)
    assert fit["is_stable"] == r["is_stable"]
    assert fit["min_root"] == pytest.approx(r["roots"][-1], rel=1e-12)
    assert fit["max_root"] == pytest.approx(r["roots"][0], rel=1e-12)


# ------------------------------------------------------------- select_order

@pytest.mark.parametrize("block", DG["select_order"],
                         ids=[f"{b['series']}-max{b['max_lags']}-{b['trend']}" for b in DG["select_order"]])
def test_select_order_repins_statsmodels_table(block):
    r = tsecon.var_select_order(_series(DG, block["series"]), max_lags=block["max_lags"],
                                trend=block["trend"])
    assert r["candidates"] == block["candidates"]
    assert r["max_lags"] == block["max_lags"] and r["trend"] == block["trend"]
    for crit in ("aic", "bic", "hqic", "fpe"):
        _close(r[f"{crit}_values"], block[f"{crit}_values"], 1e-8)
        assert r[crit] == block["selected"][crit]
        assert r[crit] == r["candidates"][int(np.argmin(r[f"{crit}_values"]))]


@needs_sm
def test_select_order_live_statsmodels_parity():
    y = _macro_levels()
    for maxlags, trend in [(8, "c"), (5, "n"), (1, "c")]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = VAR(y).select_order(maxlags=maxlags, trend=trend)
        r = tsecon.var_select_order(y, max_lags=maxlags, trend=trend)
        for crit in ("aic", "bic", "hqic", "fpe"):
            _close(r[f"{crit}_values"], sel.ics[crit], 1e-8)
            assert r[crit] == int(sel.selected_orders[crit])


# ---------------------------------------------------------------- refusals

def test_refusals_name_the_parameter():
    y = _series(CF, "sim_var2_k3")
    ok = [[None, 0.1, None]]
    tsecon.var_conditional_forecast(y, ok, lags=2)
    with pytest.raises(ValueError, match="every row of conditions"):
        tsecon.var_conditional_forecast(y, [[None, 0.1]], lags=2)
    with pytest.raises(ValueError, match="conditions has more rows than steps"):
        tsecon.var_conditional_forecast(y, ok + ok, lags=2, steps=1)
    with pytest.raises(ValueError, match="conditions constrains no cell"):
        tsecon.var_conditional_forecast(y, [[None, None, None]], lags=2)
    with pytest.raises(ValueError, match="conditions constrains no cell"):
        tsecon.var_conditional_forecast(y, np.full((3, 3), np.nan), lags=2)
    with pytest.raises(ValueError, match="conditions contains an infinite value"):
        tsecon.var_conditional_forecast(y, [[np.inf, None, None]], lags=2)
    with pytest.raises(ValueError, match="conditions is empty"):
        tsecon.var_conditional_forecast(y, [], lags=2)
    with pytest.raises(ValueError, match="conditions is empty"):
        tsecon.var_conditional_forecast(y, [], lags=2, steps=4)
    with pytest.raises(ValueError, match="steps = 0"):
        tsecon.var_conditional_forecast(y, ok, lags=2, steps=0)
    with pytest.raises(ValueError, match="alpha"):
        tsecon.var_conditional_forecast(y, ok, lags=2, alpha=1.0)
    with pytest.raises(ValueError, match="trend"):
        tsecon.var_conditional_forecast(y, ok, lags=2, trend="ct")
    # A `steps` typo is refused by the memory budget BEFORE anything is
    # allocated: 2**40 used to abort the allocator (SIGABRT), which no
    # Python-level `except` can catch.
    for bad in (10_000_000, 2**40, 2**47):
        with pytest.raises(ValueError, match="steps"):
            tsecon.var_conditional_forecast(y, ok, lags=2, steps=bad)
        with pytest.raises(ValueError, match="memory budget"):
            tsecon.var_conditional_forecast(y, ok, lags=2, steps=bad)
    with pytest.raises(ValueError, match="nlags"):
        tsecon.var_diagnostics(y, lags=2, nlags=2)
    with pytest.raises(ValueError, match="nlags"):
        tsecon.var_diagnostics(y, lags=2, nlags=0)
    for bad in (10_000, 2**40, 2**47):
        with pytest.raises(ValueError, match="nlags"):
            tsecon.var_diagnostics(y, lags=2, nlags=bad)
    # `lags` is shared with every other VAR entry point: the refusal comes
    # from the crate's sufficiency check, which states the row arithmetic but
    # (as of this wave) not the parameter name — that sweep belongs to the
    # hygiene slice. All this test claims here is that a huge `lags` is a
    # catchable ValueError and not an allocator abort.
    for bad in (10_000, 2**40, 2**47):
        with pytest.raises(ValueError):
            tsecon.var_diagnostics(y, lags=bad)
        with pytest.raises(ValueError):
            tsecon.var_conditional_forecast(y, ok, lags=bad)
    with pytest.raises(ValueError, match="max_lags = 0"):
        tsecon.var_select_order(y, max_lags=0)
    # `select_order` reserves one candidate per order, so a max_lags typo used
    # to abort the allocator before the per-candidate fit could refuse it.
    for bad in (len(y), len(y) + 1, 2**40, 2**47):
        with pytest.raises(ValueError, match="max_lags"):
            tsecon.var_select_order(y, max_lags=bad)
    with pytest.raises(ValueError, match="trend"):
        tsecon.var_select_order(y, max_lags=3, trend="x")
    with pytest.raises(ValueError, match="rows"):
        tsecon.var_select_order(y[:5], max_lags=8)


# ------------------------------------------------------ contract tripwires

def _doc_tokens(fn):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))


def test_docstrings_name_every_returned_key_and_signatures_carry_real_defaults():
    y = _series(CF, "sim_var2_k3")
    outs = {
        tsecon.var_conditional_forecast: tsecon.var_conditional_forecast(y, [[None, 0.1, None]]),
        tsecon.var_diagnostics: tsecon.var_diagnostics(y),
        tsecon.var_select_order: tsecon.var_select_order(y),
    }
    for fn, out in outs.items():
        missing = set(out) - _doc_tokens(fn)
        assert not missing, f"{fn.__name__}.__doc__ does not name returned keys: {sorted(missing)}"
        for name, prm in inspect.signature(fn).parameters.items():
            assert prm.default is not Ellipsis, f"{fn.__name__}({name}) has an Ellipsis default"
    sig = inspect.signature(tsecon.var_conditional_forecast)
    assert list(sig.parameters) == ["data", "conditions", "lags", "trend", "steps", "alpha"]
    assert sig.parameters["steps"].default is None
    assert list(inspect.signature(tsecon.var_diagnostics).parameters) == ["data", "lags", "trend", "nlags"]
    assert list(inspect.signature(tsecon.var_select_order).parameters) == ["data", "max_lags", "trend"]


# ---------------------------------------------------------- docs examples

def test_var_svar_card_example_prints_what_it_says(capsys):
    """The model-card example in docs/reference/model-cards/var-svar.md
    (conditional forecast / diagnostics / lag order) prints exactly the
    numbers the card shows."""
    rng = np.random.default_rng(0)
    k, n = 3, 400
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
    Y = np.zeros((n, k))
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)
    cond = np.full((8, k), np.nan)
    cond[:4, 1] = 0.5
    cond[1, 2] = 0.0
    cf = tsecon.var_conditional_forecast(Y, cond, lags=2, steps=8)
    print("pinned cells:", cf["n_constrained"], " plausibility p =", round(cf["mahalanobis_pvalue"], 3))
    print("series 0, h=1..4  conditional:", np.round(np.array(cf["point"])[:4, 0], 3))
    print("                unconditional:", np.round(np.array(cf["unconditional"])[:4, 0], 3))
    print("se ratio cond/uncond, series 0:",
          np.round(np.array(cf["se"])[:4, 0] / np.array(cf["unconditional_se"])[:4, 0], 3))
    d = tsecon.var_diagnostics(Y, lags=2, nlags=10)
    print("Portmanteau (adj) p =", round(d["portmanteau_adjusted_pvalue"], 3),
          " Jarque-Bera p =", round(d["jarque_bera_pvalue"], 3), " stable:", d["is_stable"])
    sel = tsecon.var_select_order(Y, max_lags=6)
    print("lag order by AIC/BIC/HQIC/FPE:", sel["aic"], sel["bic"], sel["hqic"], sel["fpe"])
    out = capsys.readouterr().out
    expected = (
        "pinned cells: 5  plausibility p = 0.294\n"
        "series 0, h=1..4  conditional: [-0.129 -0.015  0.012  0.006]\n"
        "                unconditional: [-0.218 -0.146 -0.113 -0.099]\n"
        "se ratio cond/uncond, series 0: [0.989 0.985 0.988 0.991]\n"
        "Portmanteau (adj) p = 0.468  Jarque-Bera p = 0.216  stable: True\n"
        "lag order by AIC/BIC/HQIC/FPE: 1 1 1 1\n"
    )
    assert out == expected
    card = (Path(__file__).parents[3] / "docs" / "reference" / "model-cards" / "var-svar.md").read_text(encoding="utf-8")
    for line in expected.splitlines():
        assert line in card, f"card does not show: {line!r}"


def _guide_chapter_7_data():
    """The three-column system docs/guide/07-multivariate.md builds at the top
    of the chapter: demand growth, output growth, policy rate."""
    rng = np.random.default_rng(42)
    T, burn = 400, 100
    A1 = np.array([[0.5, 0.0, -0.2], [0.3, 0.4, -0.1], [0.2, 0.1, 0.7]])
    shocks = rng.normal(size=(T + burn, 3)) * np.array([1.0, 0.8, 0.5])
    y = np.zeros((T + burn, 3))
    for t in range(1, T + burn):
        y[t] = A1 @ y[t - 1] + shocks[t]
    return y[burn:]


def test_guide_chapter_7_conditional_snippet_runs_as_written():
    """The guide's conditional-forecast snippet (docs/guide/07-multivariate.md)
    runs verbatim on the chapter's own `data`, and the paragraph under it says
    what the numbers say: output growth (column 1) is never pinned yet its se
    falls at every horizon, most at h = 2 and h = 3 (to ~90%)."""
    data = _guide_chapter_7_data()
    H = 8
    cond = np.full((H, data.shape[1]), np.nan)   # NaN = free; a number pins the cell
    cond[:4, 2] = data[-1, 2]                    # the policy rate on hold for four quarters
    cond[0, 0] = 0.5                             # plus a known demand-growth nowcast for h = 1
    cf = tsecon.var_conditional_forecast(data, cond, lags=1, steps=H)
    assert np.array(cf["point"]).shape == (H, 3) and np.array(cf["se"]).shape == (H, 3)
    assert cf["n_constrained"] == 5
    assert np.all(np.array(cf["point"])[:4, 2] == data[-1, 2])
    assert np.array(cf["point"])[0, 0] == 0.5
    assert np.all(np.array(cf["se"])[:4, 2] == 0.0) and np.array(cf["se"])[0, 0] == 0.0
    assert np.all(np.array(cf["se"]) <= np.array(cf["unconditional_se"]) * (1 + 1e-12))
    ratio = np.array(cf["se"])[:, 1] / np.array(cf["unconditional_se"])[:, 1]
    assert np.all(ratio < 1.0), ratio
    assert np.argmin(ratio) in (1, 2) and 0.88 < ratio.min() < 0.92, ratio
    assert 0.0 < cf["mahalanobis_pvalue"] <= 1.0
    # The snippet and its comments are the ones the chapter actually shows.
    guide = (Path(__file__).parents[3] / "docs" / "guide" / "07-multivariate.md").read_text(encoding="utf-8")
    for line in ("cond[:4, 2] = data[-1, 2]", "cond[0, 0] = 0.5",
                 'cf = tsecon.var_conditional_forecast(data, cond, lags=1, steps=H)'):
        assert line in guide, line
