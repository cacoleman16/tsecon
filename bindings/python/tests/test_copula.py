"""Binding tests for tsecon-copula: static bivariate copulas.

Golden pins re-play fixtures/tsecon-copula.json through the Python surface —
the same statsmodels/scipy references the Rust golden pins (statsmodels
densities/cdfs at 1e-10, fit_corr_param tau inversions, scipy-polished MLE
of the statsmodels log-density at 1e-6 with loglik at 1e-10, central-4
observed-information SEs at 1e-4, kendalltau at 1e-15; see the generator
header for the honest grading — statsmodels exposes NO copula MLE, and its
StudentTCopula.dependence_tail has a documented precedence bug the fixture
records rather than pins). Structural tests cover the binding-level
contract: defaults, dict keys, named parameters, the (n, 2) shape rule,
error propagation, and the monotone-invariance property at the Python
level.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"
FX = json.loads((FIXTURES / "tsecon-copula.json").read_text())

FIT_KEYS_BASE = {
    "family", "method", "n", "params", "param_names", "se", "se_valid",
    "loglik", "aic", "bic", "tau", "tau_implied", "tail_lower", "tail_upper",
    "converged",
}


def _u(ds):
    return np.column_stack([np.asarray(ds["u1"]), np.asarray(ds["u2"])])


# --------------------------------------------------------------------------- #
# pseudo-observations — pinned to scipy rankdata(method="average")/(n+1)
# --------------------------------------------------------------------------- #
def test_pseudo_obs_matches_scipy_rankdata_with_ties():
    po = FX["pseudo_obs"]
    x = np.column_stack([po["x1"], po["x2"]])
    u = tsecon.pseudo_obs(x)
    np.testing.assert_allclose(u[:, 0], po["u1"], rtol=1e-15)
    np.testing.assert_allclose(u[:, 1], po["u2"], rtol=1e-15)
    assert np.all((u > 0) & (u < 1))


def test_pseudo_obs_is_increasing_invariant_and_fit_identical():
    # The point of the copula decomposition, asserted at the Python level:
    # exp() one margin, cube the other (both strictly increasing) —
    # bit-identical pseudo-obs and fit.
    ds = next(d for d in FX["fits"] if d["name"] == "gauss_rho07")
    u = _u(ds)
    x = np.column_stack([u[:, 0] * 1e4 - 5e3, u[:, 1] - 0.5])
    xt = np.column_stack([np.exp(x[:, 0] / 1e4), x[:, 1] ** 3])
    ua, ub = tsecon.pseudo_obs(x), tsecon.pseudo_obs(xt)
    np.testing.assert_array_equal(ua, ub)
    fa = tsecon.copula_fit(ua, family="gaussian")
    fb = tsecon.copula_fit(ub, family="gaussian")
    np.testing.assert_array_equal(np.asarray(fa["params"]), np.asarray(fb["params"]))
    assert fa["loglik"] == fb["loglik"] and fa["tau"] == fb["tau"]


def test_pseudo_obs_decreasing_transform_flips_dependence():
    # Audit round 8: the invariance is INCREASING-only. Negating one margin
    # reverses its ranks (u -> 1 - u absent ties) and flips the sign of the
    # fitted dependence; the docs must not promise invariance under "any
    # strictly monotone transform".
    ds = next(d for d in FX["fits"] if d["name"] == "gauss_rho07")
    u = _u(ds)
    x = np.column_stack([u[:, 0] * 1e4 - 5e3, u[:, 1] - 0.5])
    x_neg = np.column_stack([-x[:, 0], x[:, 1]])
    ua, un = tsecon.pseudo_obs(x), tsecon.pseudo_obs(x_neg)
    np.testing.assert_allclose(un[:, 0], 1.0 - ua[:, 0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(un[:, 1], ua[:, 1])
    # Kendall tau negates exactly under rank reversal; the tau-inversion
    # Gaussian rho = sin(pi*tau/2) therefore flips sign.
    fa = tsecon.copula_fit(ua, family="gaussian", method="tau")
    fn = tsecon.copula_fit(un, family="gaussian", method="tau")
    assert abs(fn["tau"] + fa["tau"]) < 1e-15
    assert abs(fn["rho"] + fa["rho"]) < 1e-12
    # And the doc surfaces carry the corrected claim.
    for fn_obj in (tsecon.pseudo_obs, tsecon.copula_fit):
        doc = fn_obj.__doc__ or ""
        assert "strictly monotone transform" not in doc, fn_obj.__name__
        assert "increasing" in doc, fn_obj.__name__


# --------------------------------------------------------------------------- #
# fits — pinned to statsmodels fit_corr_param / scipy-polished MLE
# --------------------------------------------------------------------------- #
CASES = [
    (ds, case)
    for ds in FX["fits"]
    for case in ds["cases"]
]


@pytest.mark.parametrize(
    "ds,case",
    CASES,
    ids=[f"{ds['name']}-{c['family']}-{c['method']}" for ds, c in CASES],
)
def test_copula_fit_matches_reference(ds, case):
    u = _u(ds)
    out = tsecon.copula_fit(u, family=case["family"], method=case["method"])
    assert out["family"] == case["family"] and out["method"] == case["method"]
    np.testing.assert_allclose(out["tau"], ds["tau"], rtol=1e-15)

    params = np.asarray(out["params"])
    ref = np.asarray(case["params"])
    if case["method"] == "mle":
        np.testing.assert_allclose(params, ref, rtol=1e-6, atol=1e-6)
        # The honest optimizer comparison: never worse, equal at 1e-10.
        assert out["loglik"] >= case["loglik"] - 1e-10
        np.testing.assert_allclose(out["loglik"], case["loglik"], rtol=0, atol=1e-10)
        assert out["converged"] and out["se_valid"]
        np.testing.assert_allclose(np.asarray(out["se"]), case["se"], rtol=1e-4)
    else:
        # Closed tau maps exactly; Frank via two exact root-finders; the
        # profiled t nu as two polished optimizers on the same 1-D profile.
        if case["family"] == "frank":
            np.testing.assert_allclose(params, ref, rtol=1e-8)
        elif case["family"] == "t":
            np.testing.assert_allclose(params[0], ref[0], rtol=1e-12)
            np.testing.assert_allclose(params[1], ref[1], rtol=1e-5)
        else:
            np.testing.assert_allclose(params, ref, rtol=1e-12)
        assert np.all(np.isnan(np.asarray(out["se"]))) and not out["se_valid"]
    np.testing.assert_allclose(out["aic"], case["aic"], rtol=1e-8, atol=1e-4)
    np.testing.assert_allclose(out["bic"], case["bic"], rtol=1e-8, atol=1e-4)
    np.testing.assert_allclose(out["tau_implied"], case["tau_implied"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        [out["tail_lower"], out["tail_upper"]], case["tail"], rtol=1e-4, atol=1e-6
    )
    # Named parameters mirror the stacked array.
    for i, name in enumerate(out["param_names"]):
        assert out[name] == params[i]
        assert (out[f"se_{name}"] == out["se"][i]) or (
            np.isnan(out[f"se_{name}"]) and np.isnan(out["se"][i])
        )


@pytest.mark.parametrize(
    "ds", FX["fits"], ids=[ds["name"] for ds in FX["fits"]]
)
def test_copula_select_crowns_the_generator_winner(ds):
    u = _u(ds)
    fams = sorted({c["family"] for c in ds["cases"]})
    sel = tsecon.copula_select(u, families=fams, method="mle")
    assert sel["best_aic"] == ds["best_aic"]
    assert sel["ranking_aic"][0] == sel["best_aic"]
    assert set(sel["fits"]) == set(fams)
    assert sel["best_aic"] in sel["verdict"]
    aics = [sel["fits"][f]["aic"] for f in sel["ranking_aic"]]
    assert aics == sorted(aics)


def test_real_data_case_reproduces_pseudo_obs_exactly():
    ds = next(d for d in FX["fits"] if d["name"] == "yield_diffs")
    x = np.column_stack([ds["x1"], ds["x2"]])  # rate diffs, with real ties
    u = tsecon.pseudo_obs(x)
    np.testing.assert_allclose(u[:, 0], ds["u1"], rtol=1e-15)
    np.testing.assert_allclose(u[:, 1], ds["u2"], rtol=1e-15)


def test_t_tail_dependence_uses_correct_form_not_statsmodels_bug():
    bug = FX["_meta"]["statsmodels_t_tail_bug"]
    ds = next(d for d in FX["fits"] if d["name"] == "t_rho05_nu4")
    out = tsecon.copula_fit(_u(ds), family="t", method="mle")
    # The fitted (rho, nu) are near (0.5, 4): the tail coefficient must be
    # in the correct closed form's neighborhood, far from the buggy value.
    assert abs(out["tail_lower"] - bug["correct_value"]) < 0.05
    assert abs(out["tail_lower"] - bug["statsmodels_value"]) > 0.05
    assert out["tail_lower"] == out["tail_upper"]  # symmetric tails


# --------------------------------------------------------------------------- #
# binding-level contract
# --------------------------------------------------------------------------- #
def _independent_u(n=200):
    u1 = (np.arange(n) + 0.5) / n
    u2 = np.modf((np.arange(n) + 1) * 0.6180339887498949)[0]
    return np.column_stack([u1, np.clip(u2, 1e-12, 1 - 1e-12)])


def test_fit_defaults_and_keys():
    u = _independent_u()
    out = tsecon.copula_fit(u)  # defaults: gaussian, mle
    assert out["family"] == "gaussian" and out["method"] == "mle"
    assert set(out) == FIT_KEYS_BASE | {"rho", "se_rho"}
    assert abs(out["rho"]) < 0.05 and abs(out["tau"]) < 0.03  # independence
    out = tsecon.copula_fit(u, family="t", method="tau")
    assert set(out) == FIT_KEYS_BASE | {"rho", "nu", "se_rho", "se_nu"}
    assert out["param_names"] == ["rho", "nu"]


def test_select_defaults_and_keys():
    ds = next(d for d in FX["fits"] if d["name"] == "clayton_th2")
    u = _u(ds)
    sel = tsecon.copula_select(u, families=["gaussian", "clayton", "frank"])
    assert set(sel) == {
        "fits", "skipped", "best_aic", "best_bic", "ranking_aic",
        "ranking_bic", "verdict",
    }
    assert sel["best_aic"] == "clayton" and sel["skipped"] == {}
    assert "AIC" in sel["verdict"]


def test_select_skips_positive_only_families_on_negative_dependence():
    ds = next(d for d in FX["fits"] if d["name"] == "frank_neg3")
    u = _u(ds)
    sel = tsecon.copula_select(
        u, families=["gaussian", "clayton", "gumbel", "frank"]
    )
    assert set(sel["skipped"]) == {"clayton", "gumbel"}
    assert "positive dependence" in sel["skipped"]["clayton"]
    assert sel["best_aic"] == "frank"
    assert "Skipped" in sel["verdict"]


def test_errors_teach():
    u = _independent_u(100)
    # Probability-scale contract: the boundary error routes to pseudo_obs.
    bad = u.copy()
    bad[3, 0] = 1.0
    with pytest.raises(ValueError, match="pseudo_obs"):
        tsecon.copula_fit(bad)
    # Bivariate-slice shape rule.
    with pytest.raises(ValueError, match="2 columns"):
        tsecon.copula_fit(np.random.default_rng(0).uniform(0.1, 0.9, (50, 3)))
    # Too few pairs.
    with pytest.raises(ValueError, match="at least 20"):
        tsecon.copula_fit(u[:10])
    # Constant column.
    cst = u.copy()
    cst[:, 0] = 0.5
    with pytest.raises(ValueError, match="constant"):
        tsecon.copula_fit(cst)
    # Perfect dependence.
    with pytest.raises(ValueError, match="monotone"):
        tsecon.copula_fit(np.column_stack([u[:, 0], u[:, 0]]))
    # Clayton on negative dependence teaches the alternatives.
    neg = np.column_stack([u[:, 0], np.clip(1 - u[:, 0] + 0.02 * u[:, 1], 1e-6, 1 - 1e-6)])
    with pytest.raises(ValueError, match="positive dependence"):
        tsecon.copula_fit(neg, family="clayton")
    # Unknown names.
    with pytest.raises(ValueError, match="unknown copula family"):
        tsecon.copula_fit(u, family="vine")
    with pytest.raises(ValueError, match="unknown fitting method"):
        tsecon.copula_fit(u, method="bayes")
    # Select menu rules.
    with pytest.raises(ValueError, match="at least one family"):
        tsecon.copula_select(u, families=[])
    with pytest.raises(ValueError, match="more than once"):
        tsecon.copula_select(u, families=["frank", "frank"])


def test_docstrings_name_every_returned_key():
    # The runtime __doc__ is the surface users read (audit rounds 3-4):
    # every returned key must appear backticked in it.
    import re

    def doc_tokens(fn):
        return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))

    u = _independent_u()
    keys = set(tsecon.copula_fit(u).keys())
    keys |= set(tsecon.copula_fit(u, family="t", method="tau").keys())
    missing = keys - doc_tokens(tsecon.copula_fit) - {
        # bare-word keys named in the "Keys:" line rather than backticked
        "family", "method", "n", "params", "param_names", "se", "converged",
        "aic", "bic", "loglik", "tau",
    }
    assert not missing, f"copula_fit.__doc__ misses keys: {sorted(missing)}"
    sel = tsecon.copula_select(u, families=["gaussian", "frank"])
    flat = (tsecon.copula_select.__doc__ or "")
    for k in sel.keys():
        assert k in flat, f"copula_select.__doc__ misses key {k}"
