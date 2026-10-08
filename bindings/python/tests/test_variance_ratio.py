"""tsecon.variance_ratio — the Lo-MacKinlay variance-ratio test and the
Chow-Denning joint test.

* Golden re-pin: every fixture case (fixtures/variance_ratio.json, arch 8.0.0
  + closed forms) through the Python surface at the Rust golden tolerances —
  p-values against the accurate ``2 * norm.sf`` at 1e-9 relative with no
  absolute floor, including deep-tail cases where arch's own p is 0.0.
* Live arch parity on fresh data when arch is installed.
* The inert-keyword triple for ``debiased``/``robust`` under ``overlap=False``:
  explicit True raises naming the keyword and ``overlap``; the sentinel
  defaults are bit-identical to the explicit values they resolve to; and
  the flags are live where documented (``overlap=True``).
* The return contract: exact key set; scalar ``lags`` (int, NumPy integer,
  None) gives Python scalars, any sequence gives arrays aligned with the
  given order; pandas input; the rendered default ``lags=2``;
  ``tsecon.summarize`` renders both shapes.
* Refusals name the offending parameter (``\\blags\\b`` — a message saying only
  ``nlags`` would not pass) and never return garbage.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURE = Path(__file__).parents[3] / "fixtures" / "variance_ratio.json"
KEYS = {
    "lags", "vr", "stat", "pvalue", "stat_variance", "nobs", "n_dropped", "crit",
    "cd_stat", "cd_lag", "cd_pvalue", "cd_crit", "trend", "se_type", "debiased",
}
PER_HORIZON = ("vr", "stat", "pvalue", "stat_variance", "nobs", "n_dropped")


@pytest.fixture(scope="module")
def fx():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _y(fx, name):
    return np.asarray(fx["series"][name], dtype=float)


def _kw(c):
    return dict(trend=c["trend"], overlap=c["overlap"], debiased=c["debiased"],
                robust=c["robust"])


def _stat_close(a, e):
    a, e = np.asarray(a, float), np.asarray(e, float)
    assert np.all(np.abs(a - e) <= 1e-12 + 1e-10 * np.abs(e)), (a, e)


def _rw(n=300, seed=0):
    return np.concatenate([[0.0], np.cumsum(np.random.default_rng(seed).standard_normal(n - 1))])


def _ar_rw(n=400, phi=0.3, seed=2):
    rng = np.random.default_rng(seed)
    u = np.empty(n - 1)
    prev = 0.0
    for t in range(n - 1):
        prev = phi * prev + rng.standard_normal()
        u[t] = prev
    return np.concatenate([[0.0], np.cumsum(u)])


# --------------------------------------------------------------------------- #
# golden re-pin
# --------------------------------------------------------------------------- #
def test_every_single_horizon_case_matches_the_fixture(fx):
    assert len(fx["cases"]) == 250
    deep = 0
    for c in fx["cases"]:
        r = tsecon.variance_ratio(_y(fx, c["series"]), c["lags"], **_kw(c))
        ctx = (c["series"], c["lags"], c["trend"], c["overlap"], c["debiased"], c["robust"])
        assert r["vr"] == pytest.approx(c["vr"], rel=1e-12, abs=0), ctx
        _stat_close(r["stat"], c["stat"])
        # the accurate tail, NO absolute floor
        assert c["pvalue_sf"] > 0
        assert r["pvalue"] == pytest.approx(c["pvalue_sf"], rel=1e-9, abs=0), ctx
        assert abs(r["pvalue"] - c["pvalue_arch"]) <= 1e-12, ctx
        if c["pvalue_arch"] == 0.0:
            deep += 1
            assert r["pvalue"] > 0.0
        assert r["stat_variance"] == pytest.approx(c["stat_variance"], rel=1e-10, abs=0)
        assert (r["nobs"], r["n_dropped"]) == (c["nobs"], c["n_dropped"]), ctx
        assert r["lags"] == c["lags"]
        for k, v in c["crit"].items():
            assert r["crit"][k] == pytest.approx(v, rel=1e-12, abs=0)
        assert set(r["crit"]) == set(c["crit"])
        assert r["debiased"] is (c["overlap"] and c["debiased"])
        assert r["se_type"] == (
            "nonoverlapping" if not c["overlap"] else "robust" if c["robust"] else "homoskedastic"
        )
    assert deep >= 20


def test_every_chow_denning_case_matches_the_closed_form(fx):
    assert len(fx["chow_denning"]) == 45
    for c in fx["chow_denning"]:
        r = tsecon.variance_ratio(_y(fx, c["series"]), c["lags"], **_kw(c))
        np.testing.assert_allclose(r["vr"], c["vr"], rtol=1e-12, atol=0)
        _stat_close(r["stat"], c["stat"])
        np.testing.assert_allclose(r["pvalue"], c["pvalue_sf"], rtol=1e-9, atol=0)
        np.testing.assert_allclose(r["stat_variance"], c["stat_variance"], rtol=1e-10, atol=0)
        assert list(r["nobs"]) == c["nobs"] and list(r["n_dropped"]) == c["n_dropped"]
        assert r["lags"] == c["lags"]
        _stat_close(r["cd_stat"], c["cd_stat"])
        assert r["cd_lag"] == c["cd_lag"]
        assert r["cd_pvalue"] == pytest.approx(c["cd_pvalue"], rel=1e-9, abs=0)
        for (label, alpha) in (("1%", 0.01), ("5%", 0.05), ("10%", 0.10)):
            assert r["cd_crit"][label] == pytest.approx(c["cd_crit"][label], rel=1e-12, abs=0)
            # duality: the p-value and the critical value are the same bound
            assert (r["cd_pvalue"] < alpha) == (r["cd_stat"] > r["cd_crit"][label])


def test_one_q_sum_corners_are_refused_under_both_trends(fx):
    """arch returns vr = 0.0 (trend "c") or a single-squared-sum ratio
    (trend "n") here; tsecon refuses, naming lags."""
    assert len(fx["corners"]) == 8
    for c in fx["corners"]:
        kw = dict(trend=c["trend"], overlap=c["overlap"])
        if c["overlap"]:
            kw["debiased"] = False
        with pytest.raises(ValueError, match=r"\blags\b.*two"):
            tsecon.variance_ratio(np.asarray(c["y"]), c["lags"], **kw)


def test_headline_numbers_and_the_accurate_tail(fx):
    r = tsecon.variance_ratio(_y(fx, "log_realgdp"), 4)
    assert r["vr"] == pytest.approx(1.7619906809461274, rel=1e-12)
    assert r["pvalue"] == pytest.approx(6.933931871610465e-07, rel=1e-9)
    # arch's 2 - 2 Phi(|z|) value is ~1e-10 relative away: not what we print
    assert abs(r["pvalue"] / 6.933931870900523e-07 - 1) > 5e-11
    t = tsecon.variance_ratio(_y(fx, "tail"), 2)
    assert t["stat"] == pytest.approx(21.022655233561892, rel=1e-10)
    assert t["pvalue"] == pytest.approx(4.0701527078676616e-98, rel=1e-9)


def test_live_arch_parity_on_fresh_data():
    arch_ur = pytest.importorskip("arch.unitroot")
    import warnings

    from scipy.stats import norm

    rng = np.random.default_rng(20261008)
    y = np.cumsum(0.05 + rng.standard_normal(777)) / 7.0
    for q in (2, 3, 6, 11):
        for trend in ("n", "c"):
            for ov, deb, rob in ((True, True, True), (True, False, False), (False, False, False)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a = arch_ur.VarianceRatio(y, q, trend=trend, debiased=deb, robust=rob, overlap=ov)
                    a_vr, a_stat = float(a.vr), float(a.stat)
                r = tsecon.variance_ratio(y, q, trend=trend, overlap=ov, debiased=deb, robust=rob)
                assert r["vr"] == pytest.approx(a_vr, rel=1e-12)
                _stat_close(r["stat"], a_stat)
                assert r["pvalue"] == pytest.approx(2 * norm.sf(abs(a_stat)), rel=1e-9)


# --------------------------------------------------------------------------- #
# inert-keyword triple (round-10 convention)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ["debiased", "robust"])
def test_explicit_inert_flag_under_overlap_false_raises(flag):
    y = _rw()
    with pytest.raises(ValueError, match=rf"\b{flag}=True\b.*overlap=False"):
        tsecon.variance_ratio(y, 4, overlap=False, **{flag: True})


def test_sentinel_defaults_are_bit_identical_to_their_resolved_values():
    y = _ar_rw(phi=0.1)
    a = tsecon.variance_ratio(y, [2, 4, 8], overlap=False)
    b = tsecon.variance_ratio(y, [2, 4, 8], overlap=False, debiased=False, robust=False)
    for k in PER_HORIZON + ("cd_stat", "cd_pvalue"):
        assert np.array_equal(a[k], b[k]), k
    assert a["se_type"] == "nonoverlapping" and a["debiased"] is False
    c = tsecon.variance_ratio(y, [2, 4, 8])
    d = tsecon.variance_ratio(y, [2, 4, 8], debiased=True, robust=True, overlap=True)
    for k in PER_HORIZON + ("cd_stat", "cd_pvalue"):
        assert np.array_equal(c[k], d[k]), k
    assert c["se_type"] == "robust" and c["debiased"] is True


def test_flags_are_live_where_documented():
    y = _rw()
    on = tsecon.variance_ratio(y, 4)
    off = tsecon.variance_ratio(y, 4, debiased=False)
    assert on["vr"] != off["vr"] and off["debiased"] is False
    homo = tsecon.variance_ratio(y, 4, robust=False)
    assert homo["vr"] == on["vr"] and homo["stat_variance"] != on["stat_variance"]
    assert homo["se_type"] == "homoskedastic"
    assert homo["stat_variance"] == pytest.approx(2 * 7 * 3 / 12, rel=1e-15)


# --------------------------------------------------------------------------- #
# return contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lags", [None, 4, np.int64(4), np.array(4)])
def test_scalar_lags_give_python_scalars(lags):
    r = tsecon.variance_ratio(_rw(), lags) if lags is not None else tsecon.variance_ratio(_rw())
    assert set(r) == KEYS
    q = 2 if lags is None else 4
    assert r["lags"] == q and type(r["lags"]) is int
    for k in ("vr", "stat", "pvalue", "stat_variance", "cd_stat", "cd_pvalue"):
        assert type(r[k]) is float, k
    for k in ("nobs", "n_dropped", "cd_lag"):
        assert type(r[k]) is int, k
    assert r["cd_lag"] == q and r["cd_stat"] == abs(r["stat"]) and r["cd_pvalue"] == r["pvalue"]
    assert r["trend"] == "c"


@pytest.mark.parametrize(
    "lags",
    [[8, 2, 4], (8, 2, 4), np.array([8, 2, 4]), np.array([8.0, 2.0, 4.0])],
    ids=["list", "tuple", "int-array", "float-array"],
)
def test_sequence_lags_give_arrays_in_the_given_order(lags):
    y = _rw()
    r = tsecon.variance_ratio(y, lags)
    assert set(r) == KEYS
    assert r["lags"] == [8, 2, 4]
    for k in PER_HORIZON:
        assert isinstance(r[k], np.ndarray) and r[k].shape == (3,), k
    for i, q in enumerate((8, 2, 4)):
        single = tsecon.variance_ratio(y, q)
        for k in PER_HORIZON:
            assert r[k][i] == single[k], (k, q)
    i = int(np.argmax(np.abs(r["stat"])))
    assert r["cd_lag"] == r["lags"][i]


def test_pandas_lags_and_pandas_y():
    pd = pytest.importorskip("pandas")
    y = _rw()
    r = tsecon.variance_ratio(pd.Series(y, index=pd.date_range("2000", periods=y.size, freq="D")),
                              pd.Series([2, 4]))
    ref = tsecon.variance_ratio(y, [2, 4])
    assert r["lags"] == [2, 4]
    np.testing.assert_array_equal(r["stat"], ref["stat"])


def test_one_element_sequence_stays_an_array():
    y = _rw()
    r = tsecon.variance_ratio(y, [4])
    s = tsecon.variance_ratio(y, 4)
    assert r["lags"] == [4] and r["vr"].shape == (1,)
    assert r["vr"][0] == s["vr"] and r["cd_pvalue"] == s["pvalue"]


def test_signature_renders_the_effective_lags_default():
    sig = inspect.signature(tsecon.variance_ratio)
    assert sig.parameters["lags"].default == 2
    assert sig.parameters["trend"].default == "c"
    assert sig.parameters["debiased"].default is None
    assert sig.parameters["robust"].default is None
    assert sig.parameters["overlap"].default is True
    # the rendered default means exactly what it says
    y = _rw()
    assert tsecon.variance_ratio(y) == tsecon.variance_ratio(y, 2)


def test_docstring_names_every_returned_key():
    import re

    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.variance_ratio.__doc__ or ""))
    assert KEYS <= tokens, sorted(KEYS - tokens)


def test_summarize_renders_both_shapes():
    from tsecon.results import summarize

    y = _rw()
    for lags in (4, [2, 4, 8]):
        out = summarize(tsecon.variance_ratio(y, lags), title="variance_ratio")
        text = out.summary()
        for k in ("vr", "stat", "pvalue", "cd_pvalue"):
            assert k in text


def test_returns_passed_as_levels_give_the_documented_signature():
    """The card's misuse warning: iid returns passed as y test the
    differences of noise (an MA(1) with theta = -1), so vr(2) ~ 0.5."""
    r = tsecon.variance_ratio(np.random.default_rng(4).standard_normal(2000), 2)
    assert 0.45 < r["vr"] < 0.55 and r["stat"] < -8


def test_power_and_direction():
    m = tsecon.variance_ratio(_ar_rw(phi=0.3), 2)
    assert m["vr"] > 1 and m["stat"] > 0 and m["pvalue"] < 0.01
    mr = tsecon.variance_ratio(_ar_rw(phi=-0.3), 2)
    assert mr["vr"] < 1 and mr["stat"] < 0 and mr["pvalue"] < 0.01


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lags",
    [1, 0, -1, 2.5, 4.0, "4", True, [], [2, 2], [2, 1], [2.5, 4], (2, 4.5), np.array([[2, 4]]),
     {2, 4}, np.float64(4.0), [2, -3]],
    ids=lambda v: repr(v)[:20],
)
def test_bad_lags_are_refused_by_name(lags):
    with pytest.raises(ValueError, match=r"\blags\b"):
        tsecon.variance_ratio(_rw(), lags)


@pytest.mark.parametrize("lags", [299, 2**40, [2, 299], 2**63])
def test_horizons_too_long_for_the_sample_are_refused_by_name(lags):
    with pytest.raises(ValueError, match=r"\blags\b"):
        tsecon.variance_ratio(_rw(300), lags)


def test_two_q_sums_boundary():
    y = _rw(300)
    assert math.isfinite(tsecon.variance_ratio(y, 298)["stat"])
    with pytest.raises(ValueError, match=r"\blags = 299\b.*length 300"):
        tsecon.variance_ratio(y, 299)
    assert math.isfinite(tsecon.variance_ratio(y, 149, overlap=False)["stat"])
    with pytest.raises(ValueError, match=r"\blags = 150\b.*overlap=False"):
        tsecon.variance_ratio(y, 150, overlap=False)


def test_unknown_trend_is_refused_by_name():
    with pytest.raises(ValueError, match=r"\btrend\b.*\"ct\""):
        tsecon.variance_ratio(_rw(), 2, trend="ct")


@pytest.mark.parametrize(
    "y, match",
    [
        (np.r_[_rw(50), np.nan], r"\by\b contains a non-finite value \(NaN\) at index 50"),
        (np.r_[_rw(50), np.inf], r"\by\b contains a non-finite value \(inf\)"),
        (np.full(50, 3.0), r"increments of \by\b have \(numerically\) zero variance"),
        (np.arange(50.0), r"increments of \by\b have \(numerically\) zero variance"),
        (np.array([]), r"\by\b has 0 observations"),
        (np.array([1.0]), r"\by\b has 1 observations"),
    ],
    ids=["nan", "inf", "constant", "linear-trend", "empty", "one"],
)
def test_degenerate_data_is_refused_not_garbage(y, match):
    with pytest.raises(ValueError, match=match):
        tsecon.variance_ratio(y, 2)


def test_wrong_rank_y_is_refused():
    with pytest.raises((TypeError, ValueError), match=r"\by\b"):
        tsecon.variance_ratio(np.ones((50, 2)), 2)
