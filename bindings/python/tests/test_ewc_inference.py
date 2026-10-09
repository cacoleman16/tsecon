"""Equal-weighted cosine (EWC) HAR inference through the Python surface (0.11):
`tsecon.ewc_lrv`, `tsecon.ewc_default_b` and `tsecon.ols(se_type="ewc")`.

Re-pins fixtures/ewc.json — the Lazarus-Lewis-Stock-Watson (2018) closed form
evaluated with scipy.fft.dct (graded "documented closed form": no package
implements EWC; the generator's absence canary checks statsmodels, arch and
linearmodels) — then checks the Student-t_B p-values against SciPy, runs a
live statsmodels-plus-SciPy recomputation on a fresh design, the keyword
refusals (every one names the keyword), the coercion layer's handling of
negative and absurd counts, pandas inputs, and the docstring/stub contracts.
"""
import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
import tsecon

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "fixtures"
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"
EWC = json.loads((FIX / "ewc.json").read_text(encoding="utf-8"))
NILE = np.asarray(json.loads((FIX / "diagnostics.json").read_text(encoding="utf-8"))["nile"])


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


def _tokens(text):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", text or ""))


def _series(name):
    return NILE if name == "nile" else np.asarray(EWC["series"][name])


def _design(block):
    return np.asarray(block["y"]), np.column_stack([np.asarray(c) for c in block["x_cols"]])


def _rel(a, e):
    a, e = np.asarray(a, float), np.asarray(e, float)
    return float(np.max(np.abs(a - e) / np.abs(e)))


# --------------------------------------------------------------- goldens
@pytest.mark.parametrize("name", ["nile", "ar1_n250"])
def test_ewc_lrv_matches_the_dct_closed_form(name):
    block = EWC["lrv"][name]
    x = _series(name)
    assert len(x) == block["n"]
    z = x - x.mean()
    for b in block["b"]:
        want = block["ewc_demeaned"][str(b)]
        # demeaned internally: raw and pre-demeaned input agree with the
        # demeaned closed form
        assert _rel(tsecon.ewc_lrv(z, b=b), want) <= 1e-12
        assert _rel(tsecon.ewc_lrv(x, b=b), want) <= 1e-12
    # b=None is ewc_default_b(n), bitwise
    b0 = block["default_b"]
    assert tsecon.ewc_default_b(len(x)) == b0
    assert tsecon.ewc_lrv(x) == tsecon.ewc_lrv(x, b=b0)
    assert _rel(tsecon.ewc_lrv(x), block["ewc_demeaned_at_default_b"]) <= 1e-12


def test_ewc_default_b_is_the_llsw_rule_exactly():
    cases = EWC["default_b"]
    assert len(cases) >= 18
    for n, b in cases.items():
        got = tsecon.ewc_default_b(int(n))
        assert isinstance(got, int) and got == b, (n, got, b)
    # and the closed form itself, on a dense range
    for n in range(2, 3000):
        want = max(1, min(int(np.floor(0.4 * n ** (2 / 3) + 0.5)), n - 1))
        assert tsecon.ewc_default_b(n) == want, n


@pytest.mark.parametrize("name", ["k2_t250", "k3_t180"])
def test_ols_ewc_matches_the_closed_form(name):
    block = EWC["ols"][name]
    y, X = _design(block)
    for case in block["cases"]:
        b = case["b"]
        kw = {} if case["b_is_default"] else {"ewc_b": b}
        out = tsecon.ols(y, X, se_type="ewc", **kw)
        assert set(out) == {"bse", "params", "se_type", "tvalues", "ewc_b", "pvalues"}
        assert out["se_type"] == "ewc" and out["ewc_b"] == b
        assert _rel(out["params"], block["params"]) <= 1e-10
        assert _rel(out["bse"], case["bse"]) <= 1e-10
        assert _rel(out["tvalues"], case["tvalues"]) <= 1e-10
        p, want = np.asarray(out["pvalues"]), np.asarray(case["pvalues"])
        assert np.all(np.abs(p - want) <= 1e-9 * want + 1e-15), (p, want)


def test_ols_ewc_pvalues_are_student_t_with_ewc_b_dof():
    stats = pytest.importorskip("scipy.stats")
    y, X = _design(EWC["ols"]["k3_t180"])
    for b in (1, 3, 8, 40, 179):
        out = tsecon.ols(y, X, se_type="ewc", ewc_b=b)
        want = 2.0 * stats.t(df=b).sf(np.abs(out["tvalues"]))
        np.testing.assert_allclose(out["pvalues"], want, rtol=1e-9, atol=1e-15)
        # and NOT the normal: the t_B tail is heavier
        normal = 2.0 * stats.norm.sf(np.abs(out["tvalues"]))
        assert np.all(out["pvalues"] >= normal)


def test_live_parity_with_statsmodels_and_scipy_on_a_fresh_design():
    sm = pytest.importorskip("statsmodels.api")
    fft = pytest.importorskip("scipy.fft")
    rng = np.random.default_rng(91)
    T = 333
    x = np.cumsum(rng.standard_normal(T)) * 0.05 + rng.standard_normal(T)
    u = np.convolve(rng.standard_normal(T + 3), [1.0, 0.6, 0.3, 0.1], "valid")
    X = sm.add_constant(np.column_stack([x, rng.standard_normal(T)]))
    y = X @ np.array([0.2, -0.4, 0.1]) + u
    res = sm.OLS(y, X).fit()
    for b in (None, 5, 30):
        bb = tsecon.ewc_default_b(T) if b is None else b
        lam = fft.dct(X * res.resid[:, None], type=2, norm="ortho", axis=0)[1:bb + 1]
        xtxi = np.linalg.inv(X.T @ X)
        cov = xtxi @ (T * (lam.T @ lam / bb)) @ xtxi
        out = tsecon.ols(y, X, se_type="ewc", ewc_b=b)
        assert out["ewc_b"] == bb
        assert _rel(out["params"], res.params) <= 1e-10
        assert _rel(out["bse"], np.sqrt(np.diag(cov))) <= 1e-10
    # ewc_lrv on the same scores' slope column
    s = X[:, 1] * res.resid
    lam = fft.dct(s - s.mean(), type=2, norm="ortho")[1:13]
    assert _rel(tsecon.ewc_lrv(s, b=12), np.mean(lam ** 2)) <= 1e-12


# ----------------------------------------------------------- identities
def test_full_b_collapses_to_the_sample_variance_and_hc0():
    x = _series("ar1_n250")
    n = len(x)
    z = x - x.mean()
    assert _rel(tsecon.ewc_lrv(x, b=n - 1), z @ z / (n - 1)) <= 1e-12
    y, X = _design(EWC["ols"]["k2_t250"])
    n = len(y)
    hc0 = tsecon.ols(y, X, se_type="hc0")
    full = tsecon.ols(y, X, se_type="ewc", ewc_b=n - 1)
    assert _rel(full["bse"], hc0["bse"] * np.sqrt(n / (n - 1))) <= 1e-10


def test_explicit_default_b_and_explicit_use_correction_are_no_ops():
    y, X = _design(EWC["ols"]["k2_t250"])
    a = tsecon.ols(y, X, se_type="ewc")
    b = tsecon.ols(y, X, se_type="ewc", ewc_b=tsecon.ewc_default_b(len(y)), use_correction=True)
    for key in ("params", "bse", "tvalues", "pvalues"):
        assert np.array_equal(a[key], b[key]), key


def test_existing_se_types_are_unchanged_by_the_new_keyword():
    """No new keys and identical numbers on the pre-0.11 routes."""
    fx = json.loads((FIX / "hac.json").read_text(encoding="utf-8"))["regression"]
    y = np.asarray(fx["y"])
    X = np.column_stack([np.ones(len(y)), fx["x1"], fx["x2"]])
    for case in fx["hac_cases"]:
        out = tsecon.ols(y, X, se_type="hac", maxlags=case["maxlags"],
                         use_correction=case["use_correction"])
        assert set(out) == {"bse", "params", "se_type", "tvalues"}
        assert _rel(out["bse"], case["bse"]) <= 1e-10
    for se in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
        assert set(tsecon.ols(y, X, se_type=se)) == {"bse", "params", "se_type", "tvalues"}


def test_pandas_inputs_are_accepted():
    pd = pytest.importorskip("pandas")
    y, X = _design(EWC["ols"]["k3_t180"])
    a = tsecon.ols(y, X, se_type="ewc", ewc_b=8)
    b = tsecon.ols(pd.Series(y), pd.DataFrame(X, columns=["c", "x1", "x2"]), se_type="ewc", ewc_b=8)
    for key in ("params", "bse", "tvalues", "pvalues"):
        assert np.array_equal(a[key], b[key])
    x = _series("ar1_n250")
    assert tsecon.ewc_lrv(pd.Series(x), b=9) == tsecon.ewc_lrv(x, b=9)
    assert tsecon.ewc_lrv(list(x[:50])) == tsecon.ewc_lrv(x[:50])
    assert tsecon.ewc_lrv(np.asarray(x, dtype=np.float32)) == tsecon.ewc_lrv(
        np.asarray(x, dtype=np.float32).astype(np.float64))


# ------------------------------------------------------------- refusals
@pytest.mark.parametrize("b", [0, 250, 251, 10_000])
def test_ewc_lrv_refuses_b_outside_the_basis_by_name(b):
    x = _series("ar1_n250")
    with pytest.raises(ValueError, match=rf"ewc_lrv: b = {b} is outside \[1, n - 1\] = \[1, 249\]"):
        tsecon.ewc_lrv(x, b=b)


def test_negative_and_absurd_counts_are_refused_by_name():
    x = _series("ar1_n250")
    with pytest.raises(ValueError, match=r"b=-1"):
        tsecon.ewc_lrv(x, b=-1)
    with pytest.raises(ValueError, match=r"b=9223372036854775808"):
        tsecon.ewc_lrv(x, b=2 ** 63)
    with pytest.raises(ValueError, match=r"n=-1"):
        tsecon.ewc_default_b(-1)
    with pytest.raises(ValueError, match=r"n=9223372036854775808"):
        tsecon.ewc_default_b(2 ** 63)
    y, X = _design(EWC["ols"]["k2_t250"])
    with pytest.raises(ValueError, match=r"ewc_b=-3"):
        tsecon.ols(y, X, se_type="ewc", ewc_b=-3)
    with pytest.raises(ValueError, match=r"ewc_b=9223372036854775808"):
        tsecon.ols(y, X, se_type="ewc", ewc_b=2 ** 63)


@pytest.mark.parametrize("n", [0, 1])
def test_too_short_inputs_name_the_parameter(n):
    with pytest.raises(ValueError, match=rf"ewc_default_b: n = {n} observations"):
        tsecon.ewc_default_b(n)
    with pytest.raises(ValueError, match=rf"ewc_lrv: x has {n} observation"):
        tsecon.ewc_lrv(np.ones(n))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_x_is_refused_by_name(bad):
    x = np.array(_series("ar1_n250"))
    x[17] = bad
    with pytest.raises(ValueError, match=r"ewc_lrv: x contains a non-finite value .* at index 17"):
        tsecon.ewc_lrv(x)


def test_two_dimensional_x_is_refused_naming_x():
    with pytest.raises(TypeError, match=r"ewc_lrv: an array argument .*x: array\(10, 2\)"):
        tsecon.ewc_lrv(np.ones((10, 2)))


def test_ols_refuses_inert_combinations_by_name():
    y, X = _design(EWC["ols"]["k2_t250"])
    n = len(y)
    for se in ("hac", "nonrobust", "hc0", "hc1", "hc2", "hc3"):
        with pytest.raises(ValueError, match=r"ols: ewc_b = 8 only applies to se_type=\"ewc\""):
            tsecon.ols(y, X, se_type=se, ewc_b=8)
    with pytest.raises(ValueError, match=r"ols: maxlags = 4 has no effect under se_type=\"ewc\""):
        tsecon.ols(y, X, se_type="ewc", maxlags=4)
    with pytest.raises(ValueError, match=r"ols: use_correction=False has no effect under se_type=\"ewc\""):
        tsecon.ols(y, X, se_type="ewc", use_correction=False)
    for b in (0, n, n + 5):
        with pytest.raises(ValueError, match=rf"ols: ewc_b = {b} is outside \[1, n - 1\]"):
            tsecon.ols(y, X, se_type="ewc", ewc_b=b)
    with pytest.raises(ValueError, match=r"nonrobust/hc0/hc1/hc2/hc3/hac/ewc"):
        tsecon.ols(y, X, se_type="EWC")


def test_a_nan_t_statistic_keeps_a_nan_p_value():
    """Zero residual variation gives bse 0 and t = 0/0 on every route; the
    EWC p-value must stay NaN (not be clamped to 1 by `min`), and its key
    order must leave the pre-0.11 keys first."""
    _, X = _design(EWC["ols"]["k2_t250"])
    out = tsecon.ols(np.zeros(len(X)), X, se_type="ewc")
    assert np.all(np.isnan(out["tvalues"]))
    assert np.all(np.isnan(out["pvalues"]))
    assert list(out) == ["params", "bse", "tvalues", "se_type", "pvalues", "ewc_b"]
    hac = tsecon.ols(np.zeros(len(X)), X, se_type="hac")
    assert np.all(np.isnan(hac["tvalues"]))
    # an exact t of 0 is p = 1 exactly, not above it
    y, X = _design(EWC["ols"]["k3_t180"])
    assert np.all(tsecon.ols(y, X, se_type="ewc")["pvalues"] <= 1.0)


@pytest.mark.parametrize("x", [[1e308, -1e308, 1e308], [1.7e308, 1.7e308, 1.0, 2.0]])
def test_finite_input_that_overflows_is_refused_naming_x(x):
    with pytest.raises(ValueError, match=r"ewc_lrv: x overflows double precision"):
        tsecon.ewc_lrv(np.asarray(x))


def test_the_superlinear_cost_is_documented_and_bounded():
    """The cosines are evaluated directly: n * b of them, n^(5/3) at the
    default b. A request past the 2^35 work budget is refused before any work,
    naming the keyword and the largest b it admits."""
    for text in (tsecon.ewc_lrv.__doc__, _stub_doc("ewc_lrv"), tsecon.ols.__doc__, _stub_doc("ols")):
        flat = re.sub(r"\s+", " ", text)
        assert "SUPERLINEAR in `n`" in flat and "2^35" in flat, flat[:300]
    card = re.sub(r"\s+", " ", (REPO / "docs" / "reference" / "model-cards" / "robust-inference.md")
                  .read_text(encoding="utf-8"))
    assert "SUPERLINEAR in `n`" in card
    n = 10 ** 5
    x = np.random.default_rng(4).standard_normal(n)
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match=r"ewc_lrv: b = 99999 cosine frequencies .* Pass b <= 38177"):
        tsecon.ewc_lrv(x, b=n - 1)
    X = np.column_stack([np.ones(n), x])
    with pytest.raises(ValueError, match=r"ols: ewc_b = 99999 cosine frequencies .* Pass ewc_b <= 34359"):
        tsecon.ols(x, X, se_type="ewc", ewc_b=n - 1)
    # n = 10^6 at its default b (4000) is just past the budget
    with pytest.raises(ValueError, match=r"ewc_lrv: b = 4000 .* Pass b <= 3817"):
        tsecon.ewc_lrv(np.zeros(10 ** 6))
    assert time.perf_counter() - t0 < 5.0  # refused before the work
    # inside the budget the full basis runs (and is the sample variance)
    z = x[:5000] - x[:5000].mean()
    assert _rel(tsecon.ewc_lrv(x[:5000], b=4999), z @ z / 4999) <= 1e-12


def test_ols_ewc_still_refuses_malformed_data():
    y, X = _design(EWC["ols"]["k2_t250"])
    bad = y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.ols(bad, X, se_type="ewc")
    with pytest.raises(ValueError):
        tsecon.ols(y[:1], X[:1], se_type="ewc")
    with pytest.raises(ValueError):
        tsecon.ols(y[:-1], X, se_type="ewc")


# ------------------------------------------------------ docs contracts
def test_every_returned_key_is_named_in_both_docstrings():
    y, X = _design(EWC["ols"]["k2_t250"])
    keys = set(tsecon.ols(y, X, se_type="ewc")) | set(tsecon.ols(y, X))
    for text in (tsecon.ols.__doc__, _stub_doc("ols")):
        missing = keys - _tokens(text)
        assert not missing, missing
        assert "Returned keys:" in text
    for name in ("ewc_lrv", "ewc_default_b"):
        assert getattr(tsecon, name).__doc__
        assert _stub_doc(name)


def test_the_docs_state_the_t_b_contract_and_the_default():
    for text in (tsecon.ols.__doc__, _stub_doc("ols"), tsecon.ewc_lrv.__doc__, _stub_doc("ewc_lrv")):
        flat = re.sub(r"\s+", " ", text)
        assert re.search(r"student", flat, re.I) and re.search(r"not (the )?normal", flat, re.I), flat[:200]
    flat = re.sub(r"\s+", " ", tsecon.ols.__doc__)
    assert "`ewc_b` (None: `ewc_default_b(n)`)" in flat
    assert "`use_correction` (True)" in flat
