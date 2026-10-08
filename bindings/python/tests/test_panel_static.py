"""The static-panel menu through the Python surface: `panel_re`,
`panel_between`, `panel_fd` and `panel_fama_macbeth`.

* linearmodels parity: re-pins every case of fixtures/panel_static.json
  (linearmodels 7.0 RandomEffects / BetweenOLS / FirstDifferenceOLS /
  FamaMacBeth, and PooledOLS on the hand-differenced data for
  `panel_fd(constant=True)`; see the generator header) at 1e-10, with the
  outcome and mask passed as pandas DataFrames and the regressors stacked
  with `np.stack`;
* the Mundlak / correlated-random-effects recipe of the panel model card
  and the documented divergences from linearmodels (a period nobody
  observes);
* `panel_fe`'s additive 0.11.0 keys (`cov`, `names`, `nobs`, `n_entities`,
  `df_resid`) against fixtures/panel_unbalanced.json;
* refusals name their parameter; inert keyword combinations raise; the
  house covariance words (`"hc1"`, `"hac"`) stay out of the panel-FE family;
* the Fama-MacBeth caveat the card quotes (Petersen 2009) as a seeded
  Monte Carlo.
"""
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tsecon

REPO = Path(__file__).parents[3]
FIX = REPO / "fixtures"
STATIC = json.loads((FIX / "panel_static.json").read_text())
UNB = json.loads((FIX / "panel_unbalanced.json").read_text())
SE_RE = {"nonrobust": {}, "hc1": {}, "cluster": {}, "driscoll_kraay": {"bandwidth": 4.0}}
STATIC_FNS = ("panel_re", "panel_between", "panel_fd", "panel_fama_macbeth")


def _arr(rows):
    return np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)


def _empluk():
    ds = UNB["datasets"]["empluk"]
    mask = np.array(ds["mask"], dtype=float)
    return _arr(ds["log_emp"]), [_arr(ds["log_wage"]), _arr(ds["log_capital"])], mask


def _entity_means(x, mask):
    m = np.where(mask == 1, x, 0.0).sum(1) / mask.sum(1)
    return np.repeat(m[:, None], mask.shape[1], axis=1)


def _dataset(name):
    """(outcome, [regressors], mask) of a fixture dataset."""
    if name == "empluk":
        return _empluk()
    if name == "empluk_yeardummies":
        y, xs, mask = _empluk()
        n, t = mask.shape
        return y, xs + [np.tile((np.arange(t) == c).astype(float), (n, 1)) for c in range(1, t)], mask
    if name == "empluk_mundlak":
        y, xs, mask = _empluk()
        return y, xs + [_entity_means(xs[0], mask), _entity_means(xs[1], mask)], mask
    ds = STATIC["datasets"][name]
    return _arr(ds["outcome"]), [_arr(x) for x in ds["regressors"]], np.array(ds["mask"], dtype=float)


def _inputs(case):
    """pandas outcome and mask, np.stack'ed regressors (k x N x T)."""
    y, xs, mask = _dataset(case["dataset"])
    n, t = mask.shape
    x = np.stack(xs) if case["regressors"] == "all" else np.empty((0, n, t))
    return pd.DataFrame(y), x, pd.DataFrame(mask)


def _names(case):
    """The binding names regressors x0, x1, ...; the fixture uses the data's names."""
    k = len(case["names"]) - int(case["constant"])
    return (["const"] if case["constant"] else []) + [f"x{j}" for j in range(k)]


def _check(r, want, what):
    np.testing.assert_allclose(r["params"], want["params"], rtol=1e-10, err_msg=what)
    np.testing.assert_allclose(r["bse"], want["bse"], rtol=1e-10, err_msg=what)
    np.testing.assert_allclose(r["tvalues"], want["tvalues"], rtol=1e-10, err_msg=what)
    cov, wc = np.asarray(r["cov"]), np.asarray(want["cov"])
    # off-diagonals on the correlation scale (a covariance that is zero in
    # exact arithmetic carries ~1e-15 of summation-order noise)
    scale = np.sqrt(np.outer(np.abs(np.diag(wc)), np.abs(np.diag(wc))))
    assert np.max(np.abs(cov - wc) / np.maximum(np.abs(wc), scale)) < 1e-10, what


# --------------------------------------------------------------- parity

@pytest.mark.parametrize("key", sorted(SE_RE))
@pytest.mark.parametrize("case", STATIC["re_cases"], ids=lambda c: c["name"])
def test_panel_re_matches_linearmodels(case, key):
    y, x, mask = _inputs(case)
    r = tsecon.panel_re(y, x, se_type=key, constant=case["constant"], small_sample=case["small_sample"],
                        mask=mask, **SE_RE[key])
    _check(r, case["se"][key], f"{case['name']}/{key}")
    assert r["names"] == _names(case)
    assert (r["nobs"], r["n_entities"], r["df_resid"]) == (case["nobs"], case["n_entities"], case["df_resid"])
    assert r["se_type"] == key and r["constant"] is case["constant"] and r["small_sample"] is case["small_sample"]
    np.testing.assert_allclose(r["sigma2_eps"], case["sigma2_eps"], rtol=1e-10)
    if case["sigma2_effects"] == 0.0:
        assert r["sigma2_effects"] == 0.0 and r["rho"] == 0.0
    else:
        np.testing.assert_allclose(r["sigma2_effects"], case["sigma2_effects"], rtol=1e-10)
        np.testing.assert_allclose(r["rho"], case["rho"], rtol=1e-10)
    want = np.array([np.nan if v is None else v for v in case["theta"]])
    np.testing.assert_allclose(r["theta"], want, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("key", ["nonrobust", "hc1", "cluster"])
@pytest.mark.parametrize("case", STATIC["be_cases"], ids=lambda c: c["name"])
def test_panel_between_matches_linearmodels(case, key):
    y, x, mask = _inputs(case)
    r = tsecon.panel_between(y, x, se_type=key, constant=case["constant"], reweight=case["reweight"], mask=mask)
    _check(r, case["se"][key], f"{case['name']}/{key}")
    assert r["names"] == _names(case)
    assert (r["nobs"], r["n_entities"], r["df_resid"]) == (case["nobs"], case["n_entities"], case["df_resid"])
    assert r["reweight"] is case["reweight"]


@pytest.mark.parametrize("key", sorted(SE_RE))
@pytest.mark.parametrize("case", STATIC["fd_cases"], ids=lambda c: c["name"])
def test_panel_fd_matches_linearmodels(case, key):
    y, x, mask = _inputs(case)
    r = tsecon.panel_fd(y, x, se_type=key, constant=case["constant"], mask=mask, **SE_RE[key])
    _check(r, case["se"][key], f"{case['name']}/{key}")
    assert r["names"] == _names(case)
    assert (r["nobs"], r["n_entities"], r["df_resid"]) == (case["nobs"], case["n_entities"], case["df_resid"])
    assert r["constant"] is case["constant"]


@pytest.mark.parametrize("key", ["fama_macbeth", "hac"])
@pytest.mark.parametrize("case", STATIC["fm_cases"], ids=lambda c: c["name"])
def test_panel_fama_macbeth_matches_linearmodels(case, key):
    y, x, mask = _inputs(case)
    kw = {"bandwidth": 4.0} if key == "hac" else {}
    r = tsecon.panel_fama_macbeth(y, x, se_type=key, constant=case["constant"], mask=mask, **kw)
    _check(r, case["se"][key], f"{case['name']}/{key}")
    assert r["names"] == _names(case)
    assert r["nobs"] == case["nobs"] and r["n_periods_used"] == case["n_periods_used"]
    assert r["periods_used"] == case["periods_used"]
    want = np.array([[np.nan if v is None else v for v in row] for row in case["params_per_period"]])
    np.testing.assert_allclose(np.asarray(r["params_per_period"], dtype=float), want, rtol=1e-10)
    assert r["bandwidth"] == (4.0 if key == "hac" else None)


def test_the_defaults_are_the_documented_ones():
    case = STATIC["re_cases"][0]
    y, x, mask = _inputs(case)
    r = tsecon.panel_re(y, x, mask=mask)
    assert r["se_type"] == "cluster" and r["constant"] is True and r["small_sample"] is False
    _check(r, case["se"]["cluster"], "default")
    dk = tsecon.panel_re(y, x, se_type="driscoll_kraay", mask=mask)
    _check(dk, case["se"]["driscoll_kraay"], "dk default bandwidth 4")
    assert tsecon.panel_between(y, x, mask=mask)["se_type"] == "hc1"
    fd = tsecon.panel_fd(y, x, mask=mask)
    assert fd["se_type"] == "cluster" and fd["constant"] is False and fd["nobs"] == 891
    fm = tsecon.panel_fama_macbeth(y, x, mask=mask)
    assert fm["se_type"] == "fama_macbeth" and fm["bandwidth"] is None
    assert tsecon.panel_fama_macbeth(y, x, se_type="hac", mask=mask)["bandwidth"] == 4.0
    for fn in STATIC_FNS:
        sig = inspect.signature(getattr(tsecon, fn))
        assert all(p.default is not Ellipsis for p in sig.parameters.values()), fn
        assert sig.parameters["mask"].default is None


def test_aliases_give_the_same_numbers_and_stamp_what_was_passed():
    case = STATIC["re_cases"][2]
    y, x, mask = _inputs(case)
    a = tsecon.panel_re(y, x, se_type="hc1", mask=mask)
    b = tsecon.panel_re(y, x, se_type="robust", mask=mask)
    assert b["se_type"] == "robust" and np.array_equal(a["bse"], b["bse"])
    c = tsecon.panel_re(y, x, se_type="dk", mask=mask)
    d = tsecon.panel_re(y, x, se_type="driscoll_kraay", mask=mask)
    assert np.array_equal(c["bse"], d["bse"])
    e = tsecon.panel_fama_macbeth(y, x, se_type="newey_west", mask=mask)
    f = tsecon.panel_fama_macbeth(y, x, se_type="hac", mask=mask)
    assert e["se_type"] == "newey_west" and np.array_equal(e["bse"], f["bse"])
    # the between "cluster" is the HC1 sandwich (one row per entity)
    g = tsecon.panel_between(y, x, se_type="cluster", mask=mask)
    h = tsecon.panel_between(y, x, se_type="hc1", mask=mask)
    np.testing.assert_allclose(g["bse"], h["bse"], rtol=1e-12)


def test_an_all_ones_mask_is_bit_identical_to_no_mask():
    ds = STATIC["datasets"]["balanced"]
    y = _arr(ds["outcome"])
    x = np.stack([_arr(v) for v in ds["regressors"]])
    ones = np.ones(y.shape)
    calls = [
        lambda m: tsecon.panel_re(y, x, se_type="driscoll_kraay", mask=m),
        lambda m: tsecon.panel_re(y, x, small_sample=True, mask=m),
        lambda m: tsecon.panel_between(y, x, reweight=True, mask=m),
        lambda m: tsecon.panel_fd(y, x, se_type="hc1", constant=True, mask=m),
        lambda m: tsecon.panel_fama_macbeth(y, x, se_type="hac", bandwidth=2.0, mask=m),
    ]
    for call in calls:
        a, b, c = call(None), call(ones), call(ones.astype(bool))
        for k in ("params", "bse", "tvalues"):
            assert np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() == np.asarray(c[k]).tobytes()


def test_masked_out_nan_cells_are_ignored():
    ds = STATIC["datasets"]["ragged"]
    y = _arr(ds["outcome"])  # NaN outside the mask
    x = np.stack([_arr(v) for v in ds["regressors"]])
    mask = np.array(ds["mask"], dtype=float)
    a = tsecon.panel_re(y, x, mask=mask)
    b = tsecon.panel_re(np.where(mask == 1, y, 1e300), np.where(mask == 1, x, -5.0), mask=mask)
    assert np.asarray(a["params"]).tobytes() == np.asarray(b["params"]).tobytes()
    with pytest.raises(ValueError, match="mask"):
        tsecon.panel_re(y, x)  # NaN without a mask is refused, naming the mask


# --------------------------------------------------------------- Mundlak / divergences

def test_the_mundlak_recipe_of_the_card():
    """The card's FE-or-RE recipe, as written there: the entity means as
    extra regressors give panel_fe's slopes; the cluster Wald on their
    coefficients is the robust Hausman test."""
    y, xs, mask = _empluk()
    x = np.stack(xs)
    fe = tsecon.panel_fe(y, x, mask=mask)
    means = np.stack([_entity_means(v, mask) for v in xs])
    re_ = tsecon.panel_re(y, np.concatenate([x, means]), mask=mask)
    np.testing.assert_allclose(re_["params"][1:3], fe["params"], rtol=1e-12)
    b = np.asarray(re_["params"])[3:5]
    v = np.asarray(re_["cov"])[3:5, 3:5]
    wald = float(b @ np.linalg.solve(v, b))
    np.testing.assert_allclose(wald, STATIC["mundlak"]["cluster_wald_mean_coefficients"], rtol=1e-9)
    assert round(wald, 2) == 14.40
    # and the classic contrast can now be formed from tsecon output
    rei = tsecon.panel_re(y, x, mask=mask, se_type="nonrobust")
    fei = tsecon.panel_fe(y, x, mask=mask, se_type="nonrobust")
    d = np.asarray(fei["params"]) - np.asarray(rei["params"])[1:]
    vd = np.asarray(fei["cov"]) - np.asarray(rei["cov"])[1:, 1:]
    h = float(d @ np.linalg.solve(vd, d))
    assert h > 0 and np.isfinite(h)


def test_a_period_nobody_observes_is_not_bridged():
    dv = STATIC["divergences"]["empluk_without_1980"]
    y, xs, mask = _empluk()
    mask[:, dv["dropped_period"]] = 0.0
    fd = tsecon.panel_fd(y, np.stack(xs), mask=mask)
    assert fd["nobs"] == dv["fd_calendar_nobs"] == 611 != dv["fd_linearmodels_nobs"]
    np.testing.assert_allclose(fd["params"], dv["fd_calendar_params"], rtol=1e-10)
    re_ = tsecon.panel_re(y, np.stack(xs), se_type="driscoll_kraay", mask=mask)
    np.testing.assert_allclose(re_["bse"], dv["re_dk_calendar_bse"], rtol=1e-10)


def test_the_card_example_prints_what_it_says():
    """The worked example of the panel card's static-menu section, run from
    the repository root, prints exactly its 'Expected output' block."""
    import contextlib
    import io
    import os

    card = (REPO / "docs" / "reference" / "model-cards" / "panel.md").read_text(encoding="utf-8")
    section = card.split("### Worked example — EmplUK", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    expected = section.split("Expected output", 1)[1].split("```\n", 1)[1].split("```", 1)[0]
    buf = io.StringIO()
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "panel.md", "exec"), {})
    finally:
        os.chdir(cwd)
    assert buf.getvalue() == expected
    # the p-value the card quotes for the 2-df Wald (chi2(2) survival = exp(-x/2))
    wald = STATIC["mundlak"]["cluster_wald_mean_coefficients"]
    assert round(float(np.exp(-wald / 2)), 4) == 0.0007


def test_the_documented_divergences_are_the_fixtures():
    dv = STATIC["divergences"]
    assert dv["fd_first_appearance_time_axis"]["linearmodels_nobs"] == 808
    assert dv["fm_automatic_bandwidth_empluk"] == 2.0
    case = next(c for c in STATIC["re_cases"] if c["dataset"] == "empluk_yeardummies")
    assert STATIC["datasets"]["empluk_yeardummies"]["between_rank"] == 7 and len(case["names"]) == 11
    ds = STATIC["datasets"]["fm_rank_boundary"]
    fm = tsecon.panel_fama_macbeth(_arr(ds["outcome"]), np.stack([_arr(v) for v in ds["regressors"]]))
    assert fm["periods_used"] == [False] + dv["fm_rank_boundary"]["linearmodels_periods_used"][1:]
    assert dv["fm_rank_boundary"]["linearmodels_periods_used"][0] is True


# --------------------------------------------------------------- panel_fe keys

@pytest.mark.parametrize("key", ["nonrobust", "cluster_entity", "driscoll_kraay"])
def test_panel_fe_returns_cov_names_and_counts(key):
    case = next(c for c in UNB["fe_cases"] if c["name"] == "empluk/entity")
    y, xs, mask = _empluk()
    se_type, kw = {"nonrobust": ("nonrobust", {}), "cluster_entity": ("cluster", {}),
                   "driscoll_kraay": ("driscoll_kraay", {"bandwidth": 4.0})}[key]
    r = tsecon.panel_fe(y, np.stack(xs), se_type=se_type, mask=mask, **kw)
    want = case["fits"][key]
    np.testing.assert_allclose(np.asarray(r["cov"]), want["cov"], rtol=1e-10, atol=1e-14)
    assert r["nobs"] == want["nobs"] and r["df_resid"] == want["df_resid"]
    assert r["n_entities"] == 140 and r["names"] == ["x0", "x1"]
    np.testing.assert_allclose(np.sqrt(np.diag(r["cov"])), r["bse"], rtol=1e-14)


# --------------------------------------------------------------- refusals

def _ragged():
    ds = STATIC["datasets"]["ragged"]
    return _arr(ds["outcome"]), np.stack([_arr(v) for v in ds["regressors"]]), np.array(ds["mask"], dtype=float)


def test_refusals_name_the_parameter():
    y, x, mask = _ragged()
    n, t = mask.shape
    for fn in STATIC_FNS:
        call = getattr(tsecon, fn)
        with pytest.raises(ValueError, match="se_type"):
            call(y, x, se_type="bogus", mask=mask)
        with pytest.raises(ValueError, match="regressors"):
            call(y, np.empty((0, n, t)), constant=False, mask=mask)
        with pytest.raises(ValueError, match="regressors"):
            call(y, np.concatenate([x, np.ones((1, n, t))]), mask=mask) if fn != "panel_fd" else \
                call(y, np.concatenate([x, x[:1]]), mask=mask)
        with pytest.raises(ValueError, match="mask"):
            call(y, x, mask=np.full((n, t), 0.5))
        with pytest.raises(ValueError, match="regressors"):
            call(y, x[:, :-1], mask=mask)
        with pytest.raises((ValueError, TypeError), match="regressors"):
            call(y, x[0], mask=mask)  # wrong ndim
    # bandwidth outside a kernel se_type
    for fn, se in [("panel_re", "cluster"), ("panel_re", "hc1"), ("panel_fd", "nonrobust"),
                   ("panel_fama_macbeth", "fama_macbeth")]:
        with pytest.raises(ValueError, match="bandwidth"):
            getattr(tsecon, fn)(y, x, se_type=se, bandwidth=4.0, mask=mask)
    # bad bandwidths under a kernel se_type
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="bandwidth"):
            tsecon.panel_re(y, x, se_type="driscoll_kraay", bandwidth=bad, mask=mask)
        with pytest.raises(ValueError, match="bandwidth") as info:
            tsecon.panel_fama_macbeth(y, x, se_type="hac", bandwidth=bad, mask=mask)
        assert "Driscoll-Kraay" not in str(info.value)
    with pytest.raises(ValueError, match="bandwidth"):
        tsecon.panel_fama_macbeth(y, x, se_type="hac", bandwidth=2.0**63, mask=mask)
    with pytest.raises(ValueError, match="bandwidth"):
        tsecon.panel_fama_macbeth(y, x, se_type="hac", bandwidth=11.5, mask=mask)  # T_v = 12
    # the between estimator has no time dimension
    with pytest.raises(ValueError, match="se_type"):
        tsecon.panel_between(y, x, se_type="driscoll_kraay", mask=mask)
    with pytest.raises(TypeError, match="bandwidth"):
        tsecon.panel_between(y, x, bandwidth=4.0, mask=mask)
    # a time-invariant regressor in first differences
    z = np.repeat(np.arange(n, dtype=float)[:, None], t, axis=1)[None]
    with pytest.raises(ValueError, match="regressors") as info:
        tsecon.panel_fd(y, np.concatenate([x, z]), mask=mask)
    assert "panel_re" in str(info.value)
    tsecon.panel_re(y, np.concatenate([x, z]), mask=mask)  # ... which RE estimates
    # small_sample on a singular between moment matrix
    case = next(c for c in STATIC["re_cases"] if c["dataset"] == "empluk_yeardummies")
    yy, xx, mm = _inputs(case)
    with pytest.raises(ValueError, match="small_sample"):
        tsecon.panel_re(yy, xx, small_sample=True, mask=mm)
    # too few entities / periods
    with pytest.raises(ValueError, match="regressors"):
        tsecon.panel_re(y[:3], x[:, :3], mask=mask[:3])
    with pytest.raises(ValueError, match="outcome"):
        tsecon.panel_fd(y[:, :1], x[:, :, :1], mask=np.ones((n, 1)))
    with pytest.raises(ValueError, match="regressors"):
        tsecon.panel_fama_macbeth(y[:, :1], x[:, :, :1], mask=mask[:, :1])
    # intercept-only is legal with a constant (RE / BE / FM), refused for FD
    tsecon.panel_re(y, np.empty((0, n, t)), mask=mask)
    tsecon.panel_between(y, np.empty((0, n, t)), mask=mask)
    tsecon.panel_fama_macbeth(y, np.empty((0, n, t)), mask=mask)
    with pytest.raises(ValueError, match="regressors"):
        tsecon.panel_fd(y, np.empty((0, n, t)), constant=True, mask=mask)
    with pytest.raises(ValueError, match="regressors"):
        tsecon.panel_re(y, np.empty((0, n - 1, t)), mask=mask)


@pytest.mark.parametrize("bad", [np.empty((0, 0)), np.ones((1, 1)), "abc", None])
def test_malformed_outcomes_are_refused_by_name(bad):
    for fn in STATIC_FNS:
        with pytest.raises((ValueError, TypeError), match="outcome|regressors"):
            getattr(tsecon, fn)(bad, np.ones((1, 3, 4)))


def test_an_overflowing_scale_is_refused_rather_than_reported_with_zero_standard_errors():
    y, x, mask = _ragged()
    big = np.where(mask == 1, y * 1e300, np.nan)
    for call in (
        lambda: tsecon.panel_re(big, x, mask=mask),
        lambda: tsecon.panel_between(big, x, mask=mask),
        lambda: tsecon.panel_fd(big, x, se_type="hc1", mask=mask),
        lambda: tsecon.panel_fama_macbeth(big, x, mask=mask),
    ):
        with pytest.raises(ValueError, match="outcome"):
            call()


def test_magnitudes_outside_the_double_range_are_refused_by_name():
    """A regressor of magnitude ~1e200 (~1e-200) overflows (underflows) its
    sum of squares — refused naming `regressors`, not as a spurious rank or
    time-invariance failure; an outcome of magnitude 1e-170 would underflow
    every variance estimate to an exact 0 beside non-zero coefficients —
    refused naming `outcome`."""
    y, x, mask = _ragged()
    tiny_y = y * 1e-170
    for fn in STATIC_FNS:
        for scale in (1e200, 1e-200):
            bad_x = x.copy()
            bad_x[0] *= scale
            with pytest.raises(ValueError, match=r"^regressors: column 0 \(x0\) is of magnitude"):
                getattr(tsecon, fn)(y, bad_x, mask=mask)
        with pytest.raises(ValueError, match=r"^outcome: .*underflow"):
            getattr(tsecon, fn)(tiny_y, x, mask=mask)


def test_a_single_cluster_or_a_single_score_period_is_refused_by_name():
    """One entity's (or one period's) score sum is X'e = 0 by the normal
    equations, so the clustered (Driscoll-Kraay) covariance would be
    identically zero and the t-statistics ~1e15."""
    y, xs, mask = _dataset("balanced")
    assert mask.all()
    x = np.stack(xs)
    one = (y[:1, :], x[:, :1, :])
    with pytest.raises(ValueError, match=r"^se_type: the entity-clustered covariance"):
        tsecon.panel_fd(*one)
    assert np.all(tsecon.panel_fd(*one, se_type="hc1")["bse"] > 1e-3)
    two = (y[:, :2], x[:, :, :2])
    with pytest.raises(ValueError, match=r"^se_type: the Driscoll-Kraay covariance"):
        tsecon.panel_fd(*two, se_type="driscoll_kraay")
    assert np.all(tsecon.panel_fd(*two)["bse"] > 1e-3)


def _largest_parity_gap():
    """The largest relative gap of any parity entry above (params, bse,
    tvalues, the covariance on the correlation scale), measured as
    `_check` measures it."""
    def rel(got, want):
        got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
        return float(np.max(np.abs(got - want) / np.abs(want)))

    def gap(r, want):
        cov, wc = np.asarray(r["cov"]), np.asarray(want["cov"])
        scale = np.sqrt(np.outer(np.abs(np.diag(wc)), np.abs(np.diag(wc))))
        cov_gap = float(np.max(np.abs(cov - wc) / np.maximum(np.abs(wc), scale)))
        return max(cov_gap, *(rel(r[k], want[k]) for k in ("params", "bse", "tvalues")))

    worst = 0.0
    for case in STATIC["re_cases"]:
        y, x, mask = _inputs(case)
        for key, kw in SE_RE.items():
            r = tsecon.panel_re(y, x, se_type=key, constant=case["constant"],
                                small_sample=case["small_sample"], mask=mask, **kw)
            worst = max(worst, gap(r, case["se"][key]))
    for case in STATIC["be_cases"]:
        y, x, mask = _inputs(case)
        for key in ("nonrobust", "hc1", "cluster"):
            r = tsecon.panel_between(y, x, se_type=key, constant=case["constant"],
                                     reweight=case["reweight"], mask=mask)
            worst = max(worst, gap(r, case["se"][key]))
    for case in STATIC["fd_cases"]:
        y, x, mask = _inputs(case)
        for key, kw in SE_RE.items():
            r = tsecon.panel_fd(y, x, se_type=key, constant=case["constant"], mask=mask, **kw)
            worst = max(worst, gap(r, case["se"][key]))
    for case in STATIC["fm_cases"]:
        y, x, mask = _inputs(case)
        for key in ("fama_macbeth", "hac"):
            kw = {"bandwidth": 4.0} if key == "hac" else {}
            r = tsecon.panel_fama_macbeth(y, x, se_type=key, constant=case["constant"], mask=mask, **kw)
            worst = max(worst, gap(r, case["se"][key]))
    return worst


def test_the_largest_parity_gap_is_the_one_the_card_quotes():
    """The panel card, the validation matrix and the CHANGELOG quote the
    largest gap this measures: 7.8e-12, on the Mundlak design's
    Driscoll-Kraay covariance (the Rust golden, which sums the entity means
    sequentially where NumPy sums them pairwise, stays below 2e-12). The
    bound leaves room for other platforms' roundoff."""
    assert _largest_parity_gap() < 2e-11


def test_the_house_words_stay_out_of_the_panel_fe_family():
    y, x, mask = _ragged()
    for se in ("hc1", "robust", "hac", "newey_west"):
        with pytest.raises(ValueError, match="se_type"):
            tsecon.panel_fe(y, x, se_type=se, mask=mask)
        with pytest.raises(ValueError, match="se_type"):
            tsecon.panel_distributed_lag(y, x, lags=0, se_type=se, mask=mask)
    yy = np.random.default_rng(0).standard_normal((4, 40))
    with pytest.raises(ValueError, match="se_type"):
        tsecon.panel_lp(yy, yy[0], horizon=2, se_type="hc1")


def test_bool_arguments_pass_through():
    y, x, mask = _ragged()
    r = tsecon.panel_re(y, x, constant=np.bool_(False), small_sample=np.bool_(True), mask=mask)
    assert r["constant"] is False and r["small_sample"] is True and r["names"] == ["x0", "x1"]
    r = tsecon.panel_between(y, x, reweight=np.True_, mask=mask)
    assert r["reweight"] is True


# --------------------------------------------------------------- docs contracts

def _tokens(text):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", text or ""))


def _stub_doc(name):
    import ast
    text = (REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    tree = ast.parse(text)
    return next(ast.get_docstring(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_every_returned_key_is_named_in_both_docstrings():
    y, x, mask = _ragged()
    outs = {
        "panel_re": tsecon.panel_re(y, x, mask=mask),
        "panel_between": tsecon.panel_between(y, x, mask=mask),
        "panel_fd": tsecon.panel_fd(y, x, mask=mask),
        "panel_fama_macbeth": tsecon.panel_fama_macbeth(y, x, se_type="hac", mask=mask),
        "panel_fe": tsecon.panel_fe(y, x, mask=mask),
    }
    for fn, out in outs.items():
        runtime = _tokens(getattr(tsecon, fn).__doc__)
        stub = _tokens(_stub_doc(fn))
        assert not set(out) - runtime, f"{fn}.__doc__ misses {sorted(set(out) - runtime)}"
        assert not set(out) - stub, f"{fn} stub misses {sorted(set(out) - stub)}"
        rt_line = re.sub(r"\s+", " ", getattr(tsecon, fn).__doc__.split("Returned keys:")[1])
        st_line = re.sub(r"\s+", " ", _stub_doc(fn).split("Returned keys:")[1])
        assert re.findall(r"`(\w+)`", rt_line)[: len(out)] == re.findall(r"`(\w+)`", st_line)[: len(out)], fn


# --------------------------------------------------------------- the Petersen caveat

def _petersen_coverage(t, firm, reps=1000, n=100, seed=20261008):
    """Seeded Monte Carlo of Petersen's (2009) design: x and the error each
    carry a persistent firm effect (half of their variance) or none, true
    slope 1, N = 100 firms. Returns the 95% coverage of the Fama-MacBeth
    interval, of the Fama-MacBeth HAC interval (bandwidth 4), and of the
    entity-clustered random-effects interval on the same draws."""
    rng = np.random.default_rng(seed)
    hits = np.zeros(3)
    for _ in range(reps):
        fx = rng.standard_normal((n, 1)) if firm else 0.0
        fe = rng.standard_normal((n, 1)) if firm else 0.0
        x = fx + rng.standard_normal((n, t))
        y = 1.0 * x + fe + rng.standard_normal((n, t))
        fm = tsecon.panel_fama_macbeth(y, x[None])
        hac = tsecon.panel_fama_macbeth(y, x[None], se_type="hac", bandwidth=4.0)
        re_ = tsecon.panel_re(y, x[None])
        for j, r in enumerate((fm, hac, re_)):
            hits[j] += abs(r["params"][1] - 1.0) <= 1.959964 * r["bse"][1]
    return hits / reps


@pytest.mark.parametrize(
    "t, firm, quoted",
    [(10, True, (0.627, 0.499, 0.956)), (40, True, (0.398, 0.378, 0.946)), (10, False, (0.930, 0.805, 0.949))],
)
def test_fama_macbeth_understates_under_persistent_firm_effects(t, firm, quoted):
    """The panel card's Petersen (2009) table: with a persistent firm effect
    in x and in the error, the Fama-MacBeth 95% interval covers far below
    its nominal level — worse as T grows — and the Bartlett HAC on the
    beta_t series does not repair it (the correlation does not decay with
    the lag), while the entity-clustered random-effects interval stays
    nominal. Without the firm effect Fama-MacBeth is fine. 1000 seeded
    replications per cell; the tolerance allows two borderline draws to
    flip across platforms."""
    got = _petersen_coverage(t, firm)
    assert np.max(np.abs(got - np.array(quoted))) <= 0.002, got
