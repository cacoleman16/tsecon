"""`tsecon.adl_midas` — the autoregressive-distributed-lag MIDAS regression
(0.11; crate tsecon-midas `adl_midas`).

Re-pins fixtures/adl_midas.json (statsmodels OLS / HAC on a design built
independently with pandas.shift), checks the effective-rows default bandwidth,
the hf1..hfK naming, a live statsmodels parity on a fresh design, pandas
inputs, every refusal by the keyword it names, and the docstring/stub
contracts.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon

REPO = Path(__file__).resolve().parents[3]
FIX = json.loads((REPO / "fixtures" / "adl_midas.json").read_text(encoding="utf-8"))
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"
KEYS = {"params", "param_names", "bse", "tvalues", "rsquared", "nobs", "residuals",
        "first_index", "se_type", "maxlags"}


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


def _tokens(text):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", text or ""))


def _data(case):
    return np.asarray(case["y"]), np.asarray(case["hf_lags"])


@pytest.mark.parametrize("case", FIX["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("label", ["nonrobust", "hac_default", "hac_8"])
def test_matches_statsmodels(case, label):
    y, hf = _data(case)
    fit = case["fits"][label]
    kw = dict(ar_lags=case["ar_lags"], se_type=fit["se_type"])
    if label == "hac_8":
        kw["maxlags"] = 8
    out = tsecon.adl_midas(y, hf, **kw)
    assert set(out) == KEYS
    np.testing.assert_allclose(out["params"], case["params"], rtol=1e-10)
    np.testing.assert_allclose(out["bse"], fit["bse"], rtol=1e-10)
    np.testing.assert_allclose(out["tvalues"], fit["tvalues"], rtol=1e-10)
    np.testing.assert_allclose(out["residuals"], case["residuals"], rtol=1e-10, atol=1e-12)
    assert out["rsquared"] == pytest.approx(case["rsquared"], rel=1e-10)
    assert out["param_names"] == case["param_names"]
    assert out["nobs"] == case["nobs"] == len(y) - case["ar_lags"]
    assert out["first_index"] == case["ar_lags"]
    assert out["se_type"] == fit["se_type"]
    assert out["maxlags"] == fit["maxlags"]


def test_default_bandwidth_uses_the_effective_rows():
    case = FIX["cases"][1]  # n = 275, P = 4: the rule on n - P gives 4, on n gives 5
    y, hf = _data(case)
    out = tsecon.adl_midas(y, hf, ar_lags=4)
    assert out["maxlags"] == case["default_maxlags"] == 4
    assert case["maxlags_rule_on_raw_n"] == 5
    explicit = tsecon.adl_midas(y, hf, ar_lags=4, maxlags=4)
    np.testing.assert_array_equal(out["bse"], explicit["bse"])
    five = tsecon.adl_midas(y, hf, ar_lags=4, maxlags=5)
    assert not np.array_equal(out["bse"], five["bse"])


def test_names_say_hf1_is_high_frequency_lag_zero():
    y, hf = _data(FIX["cases"][0])
    out = tsecon.adl_midas(y, hf, ar_lags=2)
    assert out["param_names"] == ["const", "y.L1", "y.L2", "hf1", "hf2", "hf3", "hf4", "hf5", "hf6"]
    assert not any(n.startswith("hf.L") for n in out["param_names"])


def test_equals_least_squares_on_the_documented_design():
    rng = np.random.default_rng(3)
    n, k, p = 90, 4, 3
    hf = rng.standard_normal((n, k))
    y = rng.standard_normal(n).cumsum() * 0.1 + hf[:, 0]
    out = tsecon.adl_midas(y, hf, ar_lags=p, se_type="nonrobust")
    X = np.column_stack([np.ones(n - p)] + [y[p - j:n - j] for j in range(1, p + 1)] + [hf[p:]])
    beta = np.linalg.lstsq(X, y[p:], rcond=None)[0]
    np.testing.assert_allclose(out["params"], beta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out["residuals"], y[p:] - X @ beta, rtol=1e-9, atol=1e-12)
    assert out["maxlags"] is None


def test_live_statsmodels_parity_on_a_fresh_design():
    sm = pytest.importorskip("statsmodels.api")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(17)
    n, k, p = 140, 5, 2
    hf = rng.standard_normal((n, k))
    y = np.zeros(n)
    for t in range(n):
        y[t] = 0.2 + (0.4 * y[t - 1] if t else 0.0) + hf[t] @ (0.5 ** np.arange(k)) + rng.standard_normal()
    df = pd.DataFrame({"y": y})
    for j in range(1, p + 1):
        df[f"l{j}"] = df["y"].shift(j)
    for c in range(k):
        df[f"x{c}"] = hf[:, c]
    df = df.iloc[p:]
    X = sm.add_constant(df.drop(columns="y")).to_numpy()
    res = sm.OLS(df["y"].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": True})
    out = tsecon.adl_midas(y, hf, ar_lags=p, maxlags=6)
    np.testing.assert_allclose(out["params"], res.params, rtol=1e-10)
    np.testing.assert_allclose(out["bse"], res.bse, rtol=1e-10)


def test_pandas_inputs_are_accepted():
    pd = pytest.importorskip("pandas")
    case = FIX["cases"][0]
    y, hf = _data(case)
    a = tsecon.adl_midas(y, hf, ar_lags=2)
    b = tsecon.adl_midas(pd.Series(y), pd.DataFrame(hf), ar_lags=2)
    for k in ("params", "bse", "residuals"):
        np.testing.assert_array_equal(a[k], b[k])


# --------------------------------------------------------------- refusals
def _good():
    return _data(FIX["cases"][0])


@pytest.mark.parametrize(
    "mutate,kw,needle",
    [
        (None, dict(ar_lags=0), "ar_lags = 0"),
        ("rows", {}, "hf_lags has 159 row"),
        ("nocols", {}, "hf_lags has zero columns"),
        (None, dict(se_type="nonrobust", maxlags=3), "maxlags = 3 has no effect"),
        (None, dict(ar_lags=80), "ar_lags = 80"),
        (None, dict(ar_lags=2**40), "ar_lags = 1099511627776"),
        ("nan_y", {}, "y contains a non-finite value (NaN) at index 7"),
        ("inf_hf", {}, "hf_lags contains a non-finite value (inf) at [3, 2]"),
        (None, dict(se_type="hc1"), 'unknown se_type "hc1"'),
        ("empty", {}, "y is empty"),
    ],
)
def test_every_refusal_names_the_keyword(mutate, kw, needle):
    y, hf = _good()
    y, hf = y.copy(), hf.copy()
    if mutate == "rows":
        hf = hf[:-1]
    elif mutate == "nocols":
        hf = hf[:, :0]
    elif mutate == "nan_y":
        y[7] = np.nan
    elif mutate == "inf_hf":
        hf[3, 2] = np.inf
    elif mutate == "empty":
        y, hf = y[:0], hf[:0]
    with pytest.raises(ValueError) as exc:
        tsecon.adl_midas(y, hf, **kw)
    assert needle in str(exc.value), str(exc.value)


def test_ar_lags_zero_points_to_umidas():
    y, hf = _good()
    with pytest.raises(ValueError, match="umidas"):
        tsecon.adl_midas(y, hf, ar_lags=0)


def test_negative_and_absurd_counts_are_refused_by_name():
    y, hf = _good()
    with pytest.raises(ValueError, match="ar_lags"):
        tsecon.adl_midas(y, hf, ar_lags=-1)
    with pytest.raises(ValueError, match="ar_lags"):
        tsecon.adl_midas(y, hf, ar_lags=2**63)
    with pytest.raises(ValueError, match="maxlags"):
        tsecon.adl_midas(y, hf, maxlags=-2)


def test_absurd_sizes_are_refused_before_any_column_is_built():
    rng = np.random.default_rng(0)
    n = 100_000
    y = rng.standard_normal(n)
    # ar_lags = 49 000 on 100 000 rows passes the row count but would build a
    # 51 000 x 49 002 design (20 GB): refused by the 2^26-cell cap.
    with pytest.raises(ValueError, match=r"ar_lags = 49000 .*beyond the 2\^26"):
        tsecon.adl_midas(y, rng.standard_normal((n, 1)), ar_lags=49_000, se_type="nonrobust")
    hf = rng.standard_normal((n, 6))
    # A bandwidth past the sample: 99 998 HAC lags of 64 cross-products over
    # 99 999 rows. The budget names maxlags and the largest value it admits.
    with pytest.raises(ValueError, match=r"maxlags = 1000000000 .*Pass maxlags <= 1340 "):
        tsecon.adl_midas(y, hf, maxlags=10**9)
    # Inside the design cap, past the work budget: k = 2007 coefficients.
    with pytest.raises(ValueError, match=r"ar_lags = 2000 .*budget of 2\^33"):
        tsecon.adl_midas(y[:20_000], hf[:20_000], ar_lags=2000, se_type="nonrobust")
    # On a short series the same request is cheap and runs: the Bartlett sum
    # stops at nobs - 1 lags, which is what the budget counts.
    out = tsecon.adl_midas(y[:2000], hf[:2000], maxlags=1999)
    assert out["maxlags"] == 1999 and np.all(np.isfinite(out["params"]))


def test_docs_state_the_cost_caps():
    for doc in (tsecon.adl_midas.__doc__, _stub_doc("adl_midas")):
        flat = re.sub(r"\s+", " ", doc)
        assert "2^26" in flat and "2^33" in flat


def test_one_dimensional_hf_lags_is_refused_naming_it():
    y, hf = _good()
    with pytest.raises((TypeError, ValueError), match="hf_lags"):
        tsecon.adl_midas(y, hf[:, 0])


def test_a_collinear_design_is_refused():
    y, hf = _good()
    dup = np.column_stack([hf, hf[:, 0]])
    # Named as this function's design (the crate's shared OLS error used to
    # read "U-MIDAS OLS fit failed"), with the inputs that cause it.
    for yy, hh in ((y, dup), (np.ones_like(y), hf)):
        with pytest.raises(ValueError) as exc:
            tsecon.adl_midas(yy, hh)
        msg = str(exc.value)
        assert msg.startswith("adl_midas: the OLS fit of y[ar_lags:] on [const, y.L1, hf1..hf"), msg
        assert "U-MIDAS" not in msg and "hf_lags" in msg and "constant y" in msg


# --------------------------------------------------------------- contracts
def test_every_returned_key_is_named_in_both_docstrings():
    y, hf = _good()
    for se in ("hac", "nonrobust"):
        keys = set(tsecon.adl_midas(y, hf, se_type=se))
        assert keys == KEYS
        assert not keys - _tokens(tsecon.adl_midas.__doc__)
        assert not keys - _tokens(_stub_doc("adl_midas"))


def test_docs_state_the_alignment_and_the_bandwidth_rule():
    for doc in (tsecon.adl_midas.__doc__, _stub_doc("adl_midas")):
        flat = re.sub(r"\s+", " ", doc)
        assert "LAG 0" in flat and "hf1" in flat
        assert "EFFECTIVE" in flat
