"""Golden, parity and behavioural tests for the serial-correlation tests:
`tsecon.breusch_godfrey_test` (Breusch-Godfrey LM test, LM and F forms) and
`tsecon.durbin_watson_test` (Durbin-Watson statistic with its exact Imhof
p-value).

Re-pins fixtures/serial_corr.json through the Python surface at 1e-10
(statsmodels `acorr_breusch_godfrey` and `durbin_watson`; the exact
Durbin-Watson p-value against the generator's documented-formula
transcription), runs statsmodels DIRECTLY on fresh seeded regressions the
fixture does not store, checks the exact null distribution against a NumPy
Monte Carlo and a closed form, and pins the input contract: pandas and
array-like inputs, every refusal naming the offending parameter, the
docstring / stub / signature contracts.
"""
from __future__ import annotations

import ast
import inspect
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

import tsecon

try:  # extras-gated: the live-parity tests need statsmodels
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    from statsmodels.stats.stattools import durbin_watson

    HAVE_SM = True
except ImportError:  # pragma: no cover - extras absent
    HAVE_SM = False

needs_sm = pytest.mark.skipif(not HAVE_SM, reason="statsmodels not installed")

ROOT = Path(__file__).parents[3]
FX = json.loads((ROOT / "fixtures" / "serial_corr.json").read_text(encoding="utf-8"))
CASES = FX["cases"]
CONDITIONING = FX["conditioning_cases"]

BG_KEYS = {
    "statistic", "df", "pvalue", "fstat", "f_df_num", "f_df_den", "f_pvalue",
    "nlags", "nobs", "constant_added",
}
DW_KEYS = {"statistic", "pvalue", "alternative", "nobs", "df_resid"}


def _xy(case):
    y = np.asarray(case["y"], dtype=float)
    x = np.column_stack([np.asarray(c, dtype=float) for c in case["columns"]])
    return y, x


def _close(actual, expected, what, tol=1e-10):
    err = abs(actual - expected)
    assert err < tol or err / max(abs(expected), 1.0) < tol, (
        f"{what}: {actual!r} vs {expected!r} (abs err {err:.3e})"
    )


def _ar1(rng, n, rho):
    e = rng.standard_normal(n)
    u = np.empty(n)
    u[0] = e[0] / np.sqrt(1.0 - rho * rho)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    return u


def _regression(seed=0, n=120, rho=0.5, const=True):
    rng = np.random.default_rng(seed)
    cols = [rng.standard_normal(n), np.cumsum(rng.standard_normal(n)) / np.sqrt(n)]
    if const:
        cols.insert(0, np.ones(n))
    x = np.column_stack(cols)
    y = x @ np.linspace(0.5, -0.5, x.shape[1]) + 0.3 + _ar1(rng, n, rho)
    return y, x


# --------------------------------------------------------------------------- #
# golden replay
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_breusch_godfrey_matches_the_statsmodels_golden(case):
    y, x = _xy(case)
    for ref in case["breusch_godfrey"]:
        r = tsecon.breusch_godfrey_test(y, x, nlags=ref["nlags_arg"])
        assert set(r) == BG_KEYS
        label = f"{case['name']} nlags={ref['nlags_arg']}"
        for key in ("nlags", "df", "f_df_num", "f_df_den"):
            assert r[key] == ref[key], f"{label} {key}"
        assert r["nobs"] == case["nobs"]
        assert r["constant_added"] is ref["constant_added"]
        for key in ("statistic", "pvalue", "fstat", "f_pvalue"):
            _close(r[key], ref[key], f"{label} {key}")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_durbin_watson_matches_the_golden(case):
    y, x = _xy(case)
    ref = case["durbin_watson"]
    for alt, key in (("positive", "pvalue_positive"), ("negative", "pvalue_negative"),
                     ("two_sided", "pvalue_two_sided")):
        r = tsecon.durbin_watson_test(y, x, alternative=alt)
        assert set(r) == DW_KEYS
        _close(r["statistic"], ref["statistic"], f"{case['name']} statistic")
        _close(r["pvalue"], ref[key], f"{case['name']} {key}")
        assert r["alternative"] == alt
        assert r["nobs"] == ref["nobs"] and r["df_resid"] == ref["df_resid"]


@pytest.mark.parametrize("case", CONDITIONING, ids=lambda c: c["name"])
def test_near_constant_designs_are_tested_against_exact_arithmetic(case):
    # x comes close to spanning a constant without spanning one (a column
    # 1 + 1e-8 noise; [z, 1e9 + t] with no constant column). Every order used
    # to be refused, blaming the lagged residuals. The reference is the
    # statsmodels convention in exact rational arithmetic; one-ulp changes of
    # x move it by up to `ulp_sensitivity` (<= 2.8e-9), hence 1e-8.
    y, x = _xy(case)
    for ref in case["breusch_godfrey"]:
        assert ref["ulp_sensitivity"] < 1e-8 / 3
        r = tsecon.breusch_godfrey_test(y, x, nlags=ref["nlags_arg"])
        assert r["constant_added"] is True
        assert r["f_df_den"] == ref["f_df_den"]
        for key in ("statistic", "pvalue", "fstat", "f_pvalue"):
            _close(r[key], ref[key], f"{case['name']} nlags={ref['nlags_arg']} {key}", tol=1e-8)


def test_near_constant_column_band_is_tested_up_to_the_documented_threshold():
    # The auditor's design: x = [1 + eps * noise, z]. From eps = 2.5e-7 down
    # to 1.6e-10 every order was refused; now the constant is added down to
    # the 1e-10 implicit-constant threshold, and x counts as spanning one
    # below it.
    rng = np.random.default_rng(5)
    n = 60
    z, y, noise = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    for nlags in (1, 3):
        base = tsecon.breusch_godfrey_test(y, np.column_stack([1 + 1e-5 * noise, z]), nlags=nlags)
        for eps in (3e-7, 1e-7, 1e-8, 1e-9, 3e-10):
            r = tsecon.breusch_godfrey_test(y, np.column_stack([1 + eps * noise, z]), nlags=nlags)
            assert r["constant_added"] is True, eps
            assert abs(r["statistic"] / base["statistic"] - 1) < 1e-5, (eps, r, base)
        for eps in (1e-11, 1e-12):
            r = tsecon.breusch_godfrey_test(y, np.column_stack([1 + eps * noise, z]), nlags=nlags)
            assert r["constant_added"] is False, eps


def test_golden_covers_the_promised_cases():
    names = {c["name"] for c in CASES}
    assert {"ar1_rho05_two_regressors", "iid_two_regressors", "ar1_rho08_const_only",
            "no_constant_design", "implicit_constant_dummies",
            "macro_consumption_on_income", "macro_inflation_on_unemployment"} <= names
    assert any(b["nlags_arg"] is None for c in CASES for b in c["breusch_godfrey"])
    assert any(b["constant_added"] for c in CASES for b in c["breusch_godfrey"])


# --------------------------------------------------------------------------- #
# live statsmodels parity on regressions the fixture does not store
# --------------------------------------------------------------------------- #
@needs_sm
@pytest.mark.parametrize("seed, n, rho, const, nlags", [
    (101, 60, 0.0, True, None), (102, 200, 0.4, True, 3), (103, 90, -0.5, True, 1),
    (104, 75, 0.3, False, 2), (105, 33, 0.6, True, None), (106, 400, 0.1, False, None),
])
def test_live_statsmodels_parity(seed, n, rho, const, nlags):
    y, x = _regression(seed, n, rho, const)
    res = sm.OLS(y, x).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lm, lmp, fv, fp = acorr_breusch_godfrey(res, nlags=nlags)[:4]
    r = tsecon.breusch_godfrey_test(y, x, nlags=nlags)
    _close(r["statistic"], lm, "lm")
    _close(r["pvalue"], lmp, "lm p")
    _close(r["fstat"], fv, "F")
    _close(r["f_pvalue"], fp, "F p")
    assert r["constant_added"] is (not const)
    d = tsecon.durbin_watson_test(y, x)
    _close(d["statistic"], float(durbin_watson(res.resid)), "dw")


# --------------------------------------------------------------------------- #
# the exact Durbin-Watson null distribution
# --------------------------------------------------------------------------- #
def test_exact_pvalue_matches_a_monte_carlo_of_the_null():
    """Fixed X, 40 000 null draws of d by NumPy: the empirical P(d <= d_obs)
    must sit within 4 Monte-Carlo standard errors of the exact p-value at
    several observed statistics."""
    rng = np.random.default_rng(20261008)
    n = 25
    x = np.column_stack([np.ones(n), rng.standard_normal(n), np.arange(n) / n])
    q, _ = np.linalg.qr(x)
    z = rng.standard_normal((40_000, n))
    e = z - (z @ q) @ q.T
    d_null = np.sum(np.diff(e, axis=1) ** 2, axis=1) / np.sum(e**2, axis=1)
    for rho in (0.6, 0.2, 0.0, -0.4):
        y = x @ np.array([1.0, 0.5, -1.0]) + _ar1(rng, n, rho) if rho else x[:, 0] + rng.standard_normal(n)
        r = tsecon.durbin_watson_test(y, x)
        emp = float(np.mean(d_null <= r["statistic"]))
        se = max(np.sqrt(r["pvalue"] * (1 - r["pvalue"]) / d_null.size), 1e-4)
        assert abs(emp - r["pvalue"]) < 4 * se, (rho, r["statistic"], r["pvalue"], emp)


def test_three_observation_mean_model_closed_form():
    # n = 3, x = ones: the null eigenvalues are 1 and 3, so
    # P(d <= c) = (2/pi) arctan(sqrt((c - 1) / (3 - c))).
    for y in ([0.3, -1.1, 0.9], [1.0, 2.0, 2.5]):
        r = tsecon.durbin_watson_test(np.array(y), np.ones((3, 1)))
        c = r["statistic"]
        exact = 2 / np.pi * np.arctan(np.sqrt((c - 1) / (3 - c)))
        assert r["pvalue"] == pytest.approx(exact, abs=1e-12)


def test_alternatives_are_consistent():
    y, x = _regression(7, 80, 0.5)
    pos = tsecon.durbin_watson_test(y, x)["pvalue"]
    neg = tsecon.durbin_watson_test(y, x, alternative="negative")["pvalue"]
    two = tsecon.durbin_watson_test(y, x, alternative="two_sided")["pvalue"]
    assert pos + neg == pytest.approx(1.0, abs=1e-12)
    assert two == pytest.approx(min(1.0, 2 * min(pos, neg)), abs=1e-15)
    assert pos < 0.01 < neg  # rho = 0.5: positive autocorrelation detected


# --------------------------------------------------------------------------- #
# behaviour
# --------------------------------------------------------------------------- #
def test_default_order_and_power():
    y, x = _regression(11, 120, 0.6)
    r = tsecon.breusch_godfrey_test(y, x)
    assert r["nlags"] == 10 and r["df"] == 10 and r["f_df_num"] == 10
    assert r["f_df_den"] == 120 - 3 - 10
    assert r["pvalue"] < 1e-4 and r["f_pvalue"] < 1e-4
    y0, x0 = _regression(12, 120, 0.0)
    r0 = tsecon.breusch_godfrey_test(y0, x0, nlags=4)
    assert r0["pvalue"] > 0.01
    assert tsecon.breusch_godfrey_test(y0[:20], x0[:20])["nlags"] == 4


def test_pandas_and_array_like_inputs_are_coerced():
    pd = pytest.importorskip("pandas")
    y, x = _regression(13, 90, 0.3)
    ref_bg = tsecon.breusch_godfrey_test(y, x, nlags=2)
    ref_dw = tsecon.durbin_watson_test(y, x)
    ys = pd.Series(y)
    xd = pd.DataFrame(x, columns=["const", "a", "b"])
    assert tsecon.breusch_godfrey_test(ys, xd, nlags=2) == ref_bg
    assert tsecon.durbin_watson_test(ys, xd) == ref_dw
    # float32 data and a Fortran-ordered design are converted at the boundary.
    assert tsecon.breusch_godfrey_test(y, np.asfortranarray(x), nlags=2) == ref_bg
    r32 = tsecon.breusch_godfrey_test(y.astype(np.float32), x, nlags=2)
    assert r32["statistic"] == pytest.approx(ref_bg["statistic"], rel=1e-5)
    # a plain list response
    assert tsecon.durbin_watson_test(list(y), x)["statistic"] == ref_dw["statistic"]


def test_long_sample_returns_the_statistic_without_the_exact_pvalue():
    # The exact p-value is cubic in T and is computed up to T = 4000; above
    # that the statistic (linear in T) is still returned, fast, with pvalue
    # None, and the Breusch-Godfrey test is the test at that length.
    rng = np.random.default_rng(0)
    n = 4001
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.standard_normal(n)
    t0 = time.perf_counter()
    r = tsecon.durbin_watson_test(y, x, alternative="two_sided")
    assert time.perf_counter() - t0 < 2.0
    assert r["pvalue"] is None
    assert (r["nobs"], r["df_resid"], r["alternative"]) == (n, n - 2, "two_sided")
    e = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    assert r["statistic"] == pytest.approx(np.sum(np.diff(e) ** 2) / np.sum(e**2), rel=1e-12)
    # one observation fewer is the exact path, with the same key set
    r_cap = tsecon.durbin_watson_test(y[:4000], x[:4000])
    assert set(r_cap) == set(r) and 0.0 <= r_cap["pvalue"] <= 1.0
    # the statistic path still refuses malformed data by name
    with pytest.raises(ValueError, match=r"\by\b"):
        tsecon.durbin_watson_test(np.full(n, np.nan), x)
    # the Breusch-Godfrey test has no cap
    assert tsecon.breusch_godfrey_test(y, x)["nobs"] == n


@pytest.mark.parametrize("scale", [1e-200, 1e-155, 1e154, 1e200])
def test_extreme_scales_change_nothing(scale):
    # Both tests are invariant to rescaling y or a column of x; squaring
    # data at 1e+-155 and beyond would overflow or underflow without the
    # exact power-of-two normalization (a false "zero residuals" or
    # "collinear" refusal, or a silently imprecise p-value).
    y, x = _regression(31, 70, 0.3)
    ref_bg = tsecon.breusch_godfrey_test(y, x, nlags=3)
    ref_dw = tsecon.durbin_watson_test(y, x)
    xs = x.copy()
    xs[:, 1] *= scale
    for yy, xx in ((y * scale, x), (y, xs)):
        bg = tsecon.breusch_godfrey_test(yy, xx, nlags=3)
        dw = tsecon.durbin_watson_test(yy, xx)
        for key in ("statistic", "fstat", "pvalue", "f_pvalue"):
            _close(bg[key], ref_bg[key], f"bg {key} at {scale:g}", tol=1e-12)
        for key in ("statistic", "pvalue"):
            _close(dw[key], ref_dw[key], f"dw {key} at {scale:g}", tol=1e-12)


def test_no_constant_design_reports_the_constant_and_its_lm_caveat():
    # Through the origin, y an exact linear combination of x and a constant:
    # the added auxiliary constant explains the residuals exactly (statsmodels
    # returns LM = T and a meaningless F here), so the test refuses.
    rng = np.random.default_rng(41)
    w = 2.0 + rng.standard_normal(50)
    with pytest.raises(ValueError, match=r"^y leaves degenerate residuals on x.*constant"):
        tsecon.breusch_godfrey_test(5.0 + 0.3 * w, w[:, None], nlags=2)
    # With noise the test runs and flags the added constant.
    r = tsecon.breusch_godfrey_test(5.0 + 0.3 * w + rng.standard_normal(50), w[:, None], nlags=2)
    assert r["constant_added"] is True and r["f_df_den"] == 50 - 1 - 1 - 2


# --------------------------------------------------------------------------- #
# refusals name the offending parameter
# --------------------------------------------------------------------------- #
def _names(msg, pname):
    return bool(re.search(rf"(?<![\w.]){re.escape(pname)}(?![\w])", msg or ""))


_Y, _X = _regression(21, 40, 0.3)
_X_COLLINEAR = np.column_stack([_X, _X[:, 1]])
_NAN_Y = _Y.copy()
_NAN_Y[3] = np.nan
_INF_X = _X.copy()
_INF_X[5, 1] = np.inf

REFUSALS = [
    # (function, parameter, bad value)
    *[(fn, "y", v) for fn in ("breusch_godfrey_test", "durbin_watson_test")
      for v in (np.empty(0), _Y[:1], _NAN_Y, np.where(np.arange(40) == 7, np.inf, _Y),
                _Y.reshape(-1, 1), "abc", None, 1.0, _Y[:-1])],
    *[(fn, "x", v) for fn in ("breusch_godfrey_test", "durbin_watson_test")
      for v in (_X[:, 1], _X[:-1], _INF_X, np.empty((40, 0)), _X_COLLINEAR, "abc", None, 1.0)],
    *[("breusch_godfrey_test", "nlags", v) for v in (0, -1, 37, 2**31, 2**47, 2**63, 1.5, "abc")],
    *[("durbin_watson_test", "alternative", v) for v in ("greater", "two-sided", 1, None)],
]


def _rid(cell):
    fn, p, v = cell
    tag = f"array{v.shape}" if isinstance(v, np.ndarray) else repr(v)[:16]
    return f"{fn}.{p}={tag}"


@pytest.mark.parametrize("cell", REFUSALS, ids=_rid)
def test_every_refusal_names_the_offending_parameter(cell):
    fn, pname, value = cell
    kwargs = {"y": _Y, "x": _X, pname: value}
    if pname == "y" and isinstance(value, np.ndarray) and value.ndim == 1 and value.size == 1:
        kwargs["x"] = _X[:1]  # a one-observation regression, aligned
    with pytest.raises((ValueError, TypeError)) as info:
        getattr(tsecon, fn)(**kwargs)
    msg = str(info.value)
    assert _names(msg, pname), f"{fn}: refusal does not name {pname!r}: {msg[:300]}"


def _largest_order(n, k_base, cells=1 << 26, work=1 << 34):
    """The documented budgets: T * k_aux <= 2^26 entries and
    T k_aux^2 / 2 + 6 k_aux^3 <= 2^34 multiply-adds, with T > k_aux."""
    best = 0
    for p in range(1, n - k_base):
        k = k_base + p
        if n * k > cells or n * k * k // 2 + 6 * k ** 3 > work:
            break
        best = p
    return best


def test_an_order_past_the_budgets_is_refused_by_name_before_anything_is_built():
    # T = 1e5 and nlags = 20000 is a valid order (T > k_aux) whose auxiliary
    # design is 2e9 doubles (16 GB): it used to abort the interpreter in the
    # allocator. Now it is refused at once, naming nlags and the largest order
    # the budgets admit; that order itself is what the lab script times.
    n = 100_000
    y = np.random.default_rng(8).standard_normal(n)
    x = np.ones((n, 1))
    t0 = time.perf_counter()
    with pytest.raises(ValueError) as info:
        tsecon.breusch_godfrey_test(y, x, nlags=20_000)
    assert time.perf_counter() - t0 < 2.0
    msg = str(info.value)
    assert msg.startswith("nlags = 20000 makes the Breusch-Godfrey auxiliary regression too large")
    assert f"pass nlags <= {_largest_order(n, 1)}," in msg, msg
    with pytest.raises(ValueError, match=r"^nlags = "):
        tsecon.breusch_godfrey_test(y, x, nlags=_largest_order(n, 1) + 1)


def test_degenerate_residuals_and_short_samples_are_named():
    x = _X
    y_exact = x @ np.array([1.0, 2.0, -1.0])
    for fn in (tsecon.breusch_godfrey_test, tsecon.durbin_watson_test):
        with pytest.raises(ValueError, match=r"\by\b.*\bx\b"):
            fn(y_exact, x)
    # n < 5: the default order would be zero.
    with pytest.raises(ValueError, match=r"nlags.*min\(10, n // 5\)"):
        tsecon.breusch_godfrey_test(_Y[:4], _X[:4])
    # n = k + 1: the Durbin-Watson statistic is a constant.
    with pytest.raises(ValueError, match=r"^y is too short"):
        tsecon.durbin_watson_test(_Y[:4], _X[:4])
    # the smallest admissible Durbin-Watson sample works
    assert 0.0 <= tsecon.durbin_watson_test(_Y[:5], _X[:5])["pvalue"] <= 1.0


# --------------------------------------------------------------------------- #
# documentation contracts
# --------------------------------------------------------------------------- #
def _tokens(doc):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", doc or ""))


def _returned_keys_line(doc):
    flat = re.sub(r"\s+", " ", doc or "")
    m = re.search(r"Returned keys: (.*?)\.(?: |$)", flat)
    assert m, "no 'Returned keys:' sentence"
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", m.group(1)))


def _stub_docs():
    stub = ROOT / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    return {n.name: ast.get_docstring(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("name, keys", [("breusch_godfrey_test", BG_KEYS),
                                        ("durbin_watson_test", DW_KEYS)])
def test_returned_keys_are_documented_on_both_surfaces(name, keys):
    fn = getattr(tsecon, name)
    stub = _stub_docs()[name]
    assert _returned_keys_line(fn.__doc__) == keys
    assert _returned_keys_line(stub) == keys
    assert keys <= _tokens(fn.__doc__) and keys <= _tokens(stub)


def test_signatures_render_real_defaults():
    bg = inspect.signature(tsecon.breusch_godfrey_test).parameters
    dw = inspect.signature(tsecon.durbin_watson_test).parameters
    assert list(bg) == ["y", "x", "nlags"] and bg["nlags"].default is None
    assert list(dw) == ["y", "x", "alternative"] and dw["alternative"].default == "positive"
