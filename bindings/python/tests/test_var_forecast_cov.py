"""`var_forecast`'s forecast-error covariance (`cov`, always returned since
0.11) and the opt-in Lütkepohl parameter-uncertainty term
(`param_uncertainty=True`).

Re-pins fixtures/var_forecast_cov.json through the Python surface —
statsmodels 0.15.0 `VARResults.mse` and `forecast_cov(method="auto")` on the
var.json system, with the trend="n" golden built from `_omega_forc_cov`
because statsmodels' public "auto" drops the term without a constant — then
runs statsmodels live (importorskip) on a fresh simulated VAR, checks the
default path is untouched, the h = 1 identity, the band refusal that names
both keywords, the documented and bounded cubic cost, pandas input, and the
docstring/stub/card contracts.
"""
import ast
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import tsecon

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "fixtures"
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"
CARDS = REPO / "docs" / "reference" / "model-cards"
FC = json.loads((FIX / "var_forecast_cov.json").read_text(encoding="utf-8"))
DATA = np.asarray(json.loads((FIX / "var.json").read_text(encoding="utf-8"))["data_100dlog_gdp_cons_inv"])
Z975 = 1.959963984540054


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


def _cube_rel(a, e):
    a, e = np.asarray(a, float), np.asarray(e, float)
    scale = np.max(np.abs(e), axis=(1, 2), keepdims=True)
    return float(np.max(np.abs(a - e) / scale))


def _path_rel(a, e):
    a, e = np.asarray(a, float), np.asarray(e, float)
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), 1.0)))


def _sim_var(T=240, seed=5):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1, 0.0], [0.05, 0.4, 0.1], [0.1, 0.0, 0.3]])
    L = np.array([[1.0, 0.0, 0.0], [0.3, 0.8, 0.0], [-0.2, 0.25, 0.6]])
    y = np.zeros((T + 100, 3))
    for t in range(1, T + 100):
        y[t] = np.array([0.2, -0.1, 0.05]) + A @ y[t - 1] + L @ rng.standard_normal(3)
    return y[100:]


# --------------------------------------------------------------- goldens
@pytest.mark.parametrize("case", FC["cases"], ids=lambda c: c["name"])
def test_cov_and_bands_match_statsmodels(case):
    p, tr = case["lags"], case["trend"]
    for alpha_key, band in case["bands"].items():
        alpha = float(alpha_key)
        plain = tsecon.var_forecast(DATA, lags=p, steps=12, alpha=alpha, trend=tr)
        wide = tsecon.var_forecast(DATA, lags=p, steps=12, alpha=alpha, trend=tr,
                                   param_uncertainty=True)
        assert plain["param_uncertainty"] is False and wide["param_uncertainty"] is True
        assert _cube_rel(plain["cov"], case["mse"]) <= 1e-10
        assert _cube_rel(wide["cov"], case["cov_param"]) <= 1e-10
        assert _path_rel(plain["point"], case["point"]) <= 1e-10
        assert np.array_equal(np.asarray(plain["point"]), np.asarray(wide["point"]))
        assert _path_rel(plain["lower"], band["lower_mse"]) <= 1e-10
        assert _path_rel(plain["upper"], band["upper_mse"]) <= 1e-10
        assert _path_rel(wide["lower"], band["lower_param"]) <= 1e-10
        assert _path_rel(wide["upper"], band["upper_param"]) <= 1e-10
    # shorter horizons are the fixture's prefix
    for steps in (1, 8):
        wide = tsecon.var_forecast(DATA, lags=p, steps=steps, trend=tr, param_uncertainty=True)
        assert np.asarray(wide["cov"]).shape == (steps, 3, 3)
        assert _cube_rel(wide["cov"], case["cov_param"][:steps]) <= 1e-10


def test_cov_is_the_square_of_the_band_half_width():
    for flag in (False, True):
        for alpha, z in ((0.05, Z975), (0.32, 0.994457883209753)):
            out = tsecon.var_forecast(DATA, lags=2, steps=10, alpha=alpha, param_uncertainty=flag)
            se = np.sqrt(np.einsum("hjj->hj", np.asarray(out["cov"])))
            half = (np.asarray(out["upper"]) - np.asarray(out["point"])) / z
            np.testing.assert_allclose(se, half, rtol=1e-12)
            lo_half = (np.asarray(out["point"]) - np.asarray(out["lower"])) / z
            np.testing.assert_allclose(se, lo_half, rtol=1e-12)


def test_one_step_cov_is_one_plus_m_over_t_times_sigma_u():
    for p, tr in ((2, "c"), (2, "n"), (1, "c"), (4, "c"), (3, "n")):
        fit = tsecon.var_fit(DATA, lags=p, trend=tr)
        m = (1 if tr == "c" else 0) + 3 * p
        T = fit["nobs"]
        sig = np.asarray(fit["sigma_u"])
        wide = np.asarray(tsecon.var_forecast(DATA, lags=p, steps=3, trend=tr, param_uncertainty=True)["cov"])
        plain = np.asarray(tsecon.var_forecast(DATA, lags=p, steps=3, trend=tr)["cov"])
        assert np.max(np.abs(wide[0] - (1 + m / T) * sig)) <= 1e-13 * np.max(np.abs(sig))
        assert np.max(np.abs(plain[0] - sig)) <= 1e-13 * np.max(np.abs(sig))


def test_default_path_is_untouched_and_the_flag_only_widens():
    old = json.loads((FIX / "var.json").read_text(encoding="utf-8"))["forecast_8"]
    base = tsecon.var_forecast(DATA, lags=2, steps=8)
    assert _path_rel(base["lower"], old["lower95"]) <= 1e-8
    assert _path_rel(base["upper"], old["upper95"]) <= 1e-8
    explicit = tsecon.var_forecast(DATA, lags=2, steps=8, param_uncertainty=False)
    for key in ("point", "lower", "upper", "cov"):
        assert np.array_equal(np.asarray(base[key]), np.asarray(explicit[key])), key
    assert set(base) == {"point", "lower", "upper", "band", "cov", "param_uncertainty"}
    wide = tsecon.var_forecast(DATA, lags=2, steps=8, param_uncertainty=True)
    assert set(wide) == set(base)
    assert np.all(np.asarray(wide["lower"]) < np.asarray(base["lower"]))
    assert np.all(np.asarray(wide["upper"]) > np.asarray(base["upper"]))
    d = np.asarray(wide["cov"]) - np.asarray(base["cov"])
    for h in range(8):
        assert np.all(np.linalg.eigvalsh(0.5 * (d[h] + d[h].T)) > -1e-12 * np.trace(d[h]))
        c = np.asarray(wide["cov"])[h]
        assert np.max(np.abs(c - c.T)) <= 1e-14 * np.max(np.abs(c))


def test_simultaneous_bands_carry_the_plugin_cov():
    sim = tsecon.var_forecast(DATA, lags=2, steps=6, band="sidak")
    base = tsecon.var_forecast(DATA, lags=2, steps=6)
    assert np.array_equal(np.asarray(sim["cov"]), np.asarray(base["cov"]))
    assert sim["param_uncertainty"] is False
    se = np.sqrt(np.einsum("hjj->hj", np.asarray(sim["cov"])))
    assert np.array_equal(se, np.asarray(sim["se"]))


# ------------------------------------------------------- live statsmodels
def test_live_parity_with_statsmodels_on_a_fresh_var():
    pytest.importorskip("statsmodels.api")
    from statsmodels.tsa.api import VAR

    y = _sim_var()
    for p, tr in ((1, "c"), (3, "c"), (2, "n")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = VAR(y).fit(p, trend=tr)
            mse = np.asarray(res.mse(9))
            omega = np.asarray(res._omega_forc_cov(9))
            auto = np.asarray(res.forecast_cov(9, method="auto"))
        want = mse + omega / res.nobs
        if tr == "c":
            assert np.max(np.abs(auto - want)) <= 1e-12 * np.max(np.abs(want))
        else:
            # statsmodels' public "auto" silently drops the term without a
            # constant; tsecon does not
            assert np.array_equal(auto, mse)
        out = tsecon.var_forecast(y, lags=p, steps=9, trend=tr, param_uncertainty=True)
        assert _cube_rel(out["cov"], want) <= 1e-10
        assert _cube_rel(tsecon.var_forecast(y, lags=p, steps=9, trend=tr)["cov"], mse) <= 1e-10


def test_pandas_data_is_accepted():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(DATA, columns=["gdp", "cons", "inv"])
    a = tsecon.var_forecast(df, lags=2, steps=5, param_uncertainty=True)
    b = tsecon.var_forecast(DATA, lags=2, steps=5, param_uncertainty=True)
    for key in ("point", "lower", "upper", "cov"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]))


# ------------------------------------------------------------- refusals
@pytest.mark.parametrize("band", ["sup-t", "sidak", "bonferroni"])
def test_param_uncertainty_with_a_simultaneous_band_is_refused_naming_both(band):
    with pytest.raises(ValueError, match=r"param_uncertainty=True is only available with band=\"pointwise\"") as e:
        tsecon.var_forecast(DATA, lags=2, steps=4, band=band, param_uncertainty=True)
    assert f"band={band!r}".replace("'", '"') in str(e.value)


def test_non_bool_flag_and_bad_counts_are_refused_by_name():
    with pytest.raises(TypeError, match="param_uncertainty"):
        tsecon.var_forecast(DATA, lags=2, steps=4, param_uncertainty=1)
    with pytest.raises(ValueError, match=r"steps=-2"):
        tsecon.var_forecast(DATA, lags=2, steps=-2, param_uncertainty=True)
    with pytest.raises(ValueError, match=r"steps=9223372036854775808"):
        tsecon.var_forecast(DATA, lags=2, steps=2 ** 63, param_uncertainty=True)
    with pytest.raises(ValueError, match="steps = 0"):
        tsecon.var_forecast(DATA, lags=2, steps=0, param_uncertainty=True)
    bad = DATA.copy()
    bad[5, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.var_forecast(bad, lags=2, steps=4, param_uncertainty=True)


def test_the_cubic_cost_is_documented_budgeted_and_bounded():
    for text in (tsecon.var_forecast.__doc__, _stub_doc("var_forecast")):
        flat = re.sub(r"\s+", " ", text)
        assert "CUBIC in `steps`" in flat, flat[:300]
    card = re.sub(r"\s+", " ", (CARDS / "var-svar.md").read_text(encoding="utf-8"))
    assert "CUBIC in `steps`" in card
    t0 = time.perf_counter()
    out = tsecon.var_forecast(DATA, lags=2, steps=500, param_uncertainty=True)
    dt = time.perf_counter() - t0
    assert np.asarray(out["cov"]).shape == (500, 3, 3)
    assert dt < 30.0, dt  # measured well under a second in a release build
    for steps in (5000, 10 ** 6, 2 ** 40):
        with pytest.raises(ValueError, match=r"invalid steps = .*CUBIC in steps"):
            tsecon.var_forecast(DATA, lags=2, steps=steps, param_uncertainty=True)
    # without the flag a long horizon is not budgeted by the term
    assert len(tsecon.var_forecast(DATA, lags=2, steps=5000)["cov"]) == 5000


def test_a_wide_var_refusal_names_the_lag_driven_cost():
    """Audit round 15 (SWEEP-8): the refusal quoted only steps^3 (k^2 + 8) / 3
    and advised that "horizons are tens of periods" while the 2 steps m^3
    term (m = n_trend + k*lags) was what tripped the budget. It now quotes
    the whole count, says which terms dominate, and names `lags` when not
    even one step fits."""
    rng = np.random.default_rng(0)
    wide = rng.standard_normal((400, 30))  # m = 301: 197 steps fit (the lab script)
    with pytest.raises(ValueError, match=r"^invalid steps = 198: .*the m terms dominate") as e:
        tsecon.var_forecast(wide, lags=10, steps=198, param_uncertainty=True)
    msg = str(e.value)
    for frag in ("steps^2 k^3 / 2", "2 steps m^3", "m = n_trend + k*lags", "reduce lags"):
        assert frag in msg, msg
    assert "tens of periods" not in msg
    # a small VAR past its cap keeps the cubic-in-steps advice
    with pytest.raises(ValueError, match=r"^invalid steps = 5000: .*first term dominates: CUBIC in steps"):
        tsecon.var_forecast(DATA, lags=2, steps=5000, param_uncertainty=True)
    # m = 1 + 64 * 32 = 2049: 2 m^3 alone is over 2^34, so no horizon fits
    widest = rng.standard_normal((2200, 64))  # T - lags - m >= k keeps Sigma_u definite
    with pytest.raises(ValueError, match=r"^invalid lags = 32: .*even at steps = 1"):
        tsecon.var_forecast(widest, lags=32, steps=1, param_uncertainty=True)
    for text in (tsecon.var_forecast.__doc__, _stub_doc("var_forecast")):
        flat = re.sub(r"\s+", " ", text)
        assert "2 steps m^3" in flat and "m = n_trend + k*lags" in flat, flat[:200]


# ------------------------------------------------------ docs contracts
def test_every_returned_key_is_named_in_both_docstrings():
    keys = set(tsecon.var_forecast(DATA, lags=2, steps=3, param_uncertainty=True))
    keys |= set(tsecon.var_forecast(DATA, lags=2, steps=3, band="sup-t", band_n_sim=2000))
    for text in (tsecon.var_forecast.__doc__, _stub_doc("var_forecast")):
        tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", text))
        assert not keys - tokens, keys - tokens
        assert "Returned keys:" in text
        flat = re.sub(r"\s+", " ", text)
        assert "`param_uncertainty` (False)" in flat
        assert "PARTIAL repair" in flat and "STATIONARY" in flat
