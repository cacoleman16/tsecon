"""Binding tests for the structured-penalty and post-selection slice:
`group_lasso`, `post_lasso`, `pds_lasso`.

Re-pins fixtures/structured.json through the Python surface (skglm /
scikit-learn / statsmodels references — see the generator header for the
grades), re-evaluates the group-LASSO KKT certificate independently in
NumPy, checks the reductions to `lasso`, the honesty flag, the exact
returned key sets, pandas coercion (and that an integer `groups` array
survives coercion untouched), every teaching error, and that each runtime
docstring names every returned key.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
STRUCT = json.loads((FIX / "structured.json").read_text())

GL_KEYS = {
    "coef", "n_iter", "converged", "active_groups", "active_set", "objective",
    "kkt_violation", "max_rel_change", "alpha_max",
}
POST_KEYS = {"support", "coef_lasso", "coef_ols", "n_selected", "rss"}
PDS_KEYS = {
    "coef", "se", "t_stat", "p_value", "conf_int", "support_y", "support_d",
    "union_support", "n_controls_selected", "alpha_y", "alpha_d", "hac_lags_resolved",
}


def _design(name):
    d = STRUCT["group_lasso"]["designs"][name]
    return np.array(d["X"]), np.array(d["y"]), np.array(d["groups"], dtype=np.int64)


def _members(groups):
    labels = sorted(set(int(g) for g in groups))
    return labels, [np.flatnonzero(groups == lab) for lab in labels]


def _weights(spec, members):
    if spec == "sqrt_size":
        return np.array([np.sqrt(len(m)) for m in members])
    if spec == "none":
        return np.ones(len(members))
    return np.asarray(spec, dtype=float)


def _soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def kkt_residual(X, y, b, members, w, alpha, l1):
    """Independent NumPy evaluation of the subgradient KKT conditions."""
    n = X.shape[0]
    lam1, lam2 = alpha * l1, alpha * (1 - l1)
    grad = -X.T @ (y - X @ b) / n
    worst = 0.0
    for g, m in enumerate(members):
        bg, gg = b[m], grad[m]
        nb = np.linalg.norm(bg)
        if nb == 0.0:
            worst = max(worst, np.linalg.norm(_soft(-gg, lam1)) - lam2 * w[g])
        else:
            for k in range(len(m)):
                if bg[k] != 0.0:
                    worst = max(worst, abs(gg[k] + lam2 * w[g] * bg[k] / nb + lam1 * np.sign(bg[k])))
                else:
                    worst = max(worst, abs(gg[k]) - lam1)
    return max(worst, 0.0)


# --------------------------------------------------------------- group lasso

@pytest.mark.parametrize("case", STRUCT["group_lasso"]["cases"], ids=lambda c: c["name"])
def test_group_lasso_matches_reference_and_kkt_certificate(case):
    X, y, groups = _design(case["design"])
    r = tsecon.group_lasso(
        X, y, groups, alpha=case["alpha"], l1_ratio=case["l1_ratio"],
        group_weights=case["group_weights"], tol=1e-11, max_iter=100000,
    )
    assert set(r.keys()) == GL_KEYS
    assert r["converged"] is True
    # Cross-package golden: skglm (sklearn Lasso for l1_ratio = 1).
    np.testing.assert_allclose(r["coef"], case["coef"], atol=1e-8, rtol=0)
    # Independent optimality certificate: rigorous for a convex problem.
    labels, members = _members(groups)
    w = _weights(case["group_weights"], members)
    kkt = kkt_residual(X, y, r["coef"], members, w, case["alpha"], case["l1_ratio"])
    assert kkt <= 1e-8
    assert r["kkt_violation"] == pytest.approx(kkt, abs=1e-12)
    assert r["active_groups"] == case["active_groups"]
    assert r["active_set"] == [int(j) for j in np.flatnonzero(r["coef"] != 0)]
    assert r["objective"] == pytest.approx(case["objective"], rel=1e-10)
    assert r["alpha_max"] == pytest.approx(case["alpha_max"], rel=1e-12)
    assert r["max_rel_change"] <= 1e-11
    assert isinstance(r["n_iter"], int)


def test_group_lasso_default_tolerance_certificate():
    """At the shipped default tol=1e-8 the certificate is still tight: the
    KKT residual is bounded by tol * max_j|x_j'y|/n and lands far below
    1e-8 on the fixture (the achieved figure is asserted)."""
    X, y, groups = _design("blocks")
    labels, members = _members(groups)
    w = _weights("sqrt_size", members)
    worst = 0.0
    for alpha, l1 in [(0.05, 0.0), (0.08, 0.5), (0.1, 1.0)]:
        r = tsecon.group_lasso(X, y, groups, alpha=alpha, l1_ratio=l1)
        assert r["converged"] is True
        kkt = kkt_residual(X, y, r["coef"], members, w, alpha, l1)
        bound = 1e-8 * np.max(np.abs(X.T @ y)) / X.shape[0]
        assert kkt <= bound
        worst = max(worst, kkt)
    assert worst <= 1e-8


def test_group_lasso_reductions_to_lasso():
    X, y, groups = _design("blocks")
    p = X.shape[1]
    for alpha in (0.05, 0.1, 0.3):
        base = tsecon.lasso(X, y, alpha=alpha, tol=1e-11)["coef"]
        one = tsecon.group_lasso(X, y, groups, alpha=alpha, l1_ratio=1.0, tol=1e-11,
                                 max_iter=100000)["coef"]
        np.testing.assert_allclose(one, base, atol=1e-8, rtol=0)
        singles = tsecon.group_lasso(X, y, np.arange(p) * 3 - 7, alpha=alpha, l1_ratio=0.0,
                                     tol=1e-11, max_iter=100000)["coef"]
        np.testing.assert_allclose(singles, base, atol=1e-8, rtol=0)


def test_group_lasso_alpha_max_zeroes_the_fit():
    X, y, groups = _design("scattered")
    for l1 in (0.0, 0.5, 1.0):
        am = tsecon.group_lasso(X, y, groups, alpha=1.0, l1_ratio=l1)["alpha_max"]
        above = tsecon.group_lasso(X, y, groups, alpha=am * (1 + 1e-9), l1_ratio=l1)
        assert np.all(above["coef"] == 0.0)
        assert above["active_groups"] == [] and above["active_set"] == []
        below = tsecon.group_lasso(X, y, groups, alpha=am * (1 - 1e-3), l1_ratio=l1)
        assert np.any(below["coef"] != 0.0)


def test_group_lasso_converged_flag_fires():
    X, y, groups = _design("blocks")
    cut = tsecon.group_lasso(X, y, groups, alpha=0.02, l1_ratio=0.5, tol=1e-11, max_iter=1)
    full = tsecon.group_lasso(X, y, groups, alpha=0.02, l1_ratio=0.5, tol=1e-11, max_iter=100000)
    assert cut["converged"] is False and cut["n_iter"] == 1
    assert full["converged"] is True
    assert cut["kkt_violation"] > 100 * full["kkt_violation"]
    assert np.all(np.isfinite(cut["coef"]))


def test_group_lasso_custom_weights_and_group_weight_strings():
    X, y, groups = _design("blocks")
    a = tsecon.group_lasso(X, y, groups, alpha=0.1, group_weights="none")
    b = tsecon.group_lasso(X, y, groups, alpha=0.1, group_weights=np.ones(4))
    c = tsecon.group_lasso(X, y, groups, alpha=0.1, group_weights=[1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(a["coef"], b["coef"], atol=1e-12)
    np.testing.assert_allclose(a["coef"], c["coef"], atol=1e-12)
    d = tsecon.group_lasso(X, y, groups, alpha=0.1)  # default sqrt_size
    e = tsecon.group_lasso(X, y, groups, alpha=0.1, group_weights="sqrt_size")
    np.testing.assert_allclose(d["coef"], e["coef"], atol=1e-12)
    assert not np.allclose(a["coef"], d["coef"])


def test_group_lasso_pandas_and_integer_groups_survive_coercion():
    pd = pytest.importorskip("pandas")
    X, y, groups = _design("blocks")
    ref = tsecon.group_lasso(X, y, groups, alpha=0.1)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    r = tsecon.group_lasso(df, pd.Series(y), groups, alpha=0.1)
    np.testing.assert_allclose(r["coef"], ref["coef"], atol=1e-12)
    # Integer group labels are exempt from float64 coercion: int32, uint8
    # and a plain list all reach the boundary as integers.
    for g in (groups.astype(np.int32), groups.astype(np.uint8), groups.tolist()):
        r = tsecon.group_lasso(df, y, g, alpha=0.1)
        np.testing.assert_allclose(r["coef"], ref["coef"], atol=1e-12)
    # And a float64 label array is rejected at the boundary, as documented.
    with pytest.raises(TypeError):
        tsecon.group_lasso(X, y, groups.astype(np.float64), alpha=0.1)
    # Non-contiguous, negative labels are just names.
    relabeled = np.where(groups == 0, -3, np.where(groups == 1, 40, groups))
    r = tsecon.group_lasso(X, y, relabeled, alpha=0.1)
    np.testing.assert_allclose(r["coef"], ref["coef"], atol=1e-12)
    assert r["active_groups"] == sorted(
        {int(relabeled[j]) for j in range(len(relabeled)) if r["coef"][j] != 0}
    )


def test_group_lasso_teaching_errors():
    X, y, groups = _design("blocks")
    with pytest.raises(ValueError, match=r"groups.*expected 12, got 3"):
        tsecon.group_lasso(X, y, [0, 0, 1], alpha=0.1)
    with pytest.raises(ValueError, match=r"l1_ratio must lie in \[0, 1\]"):
        tsecon.group_lasso(X, y, groups, alpha=0.1, l1_ratio=1.5)
    with pytest.raises(ValueError, match=r"group_weights.*accepted values.*sqrt_size.*none"):
        tsecon.group_lasso(X, y, groups, alpha=0.1, group_weights="size")
    with pytest.raises(ValueError, match=r"group_weights.*expected 4, got 2"):
        tsecon.group_lasso(X, y, groups, alpha=0.1, group_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="group_weights must be finite and strictly positive"):
        tsecon.group_lasso(X, y, groups, alpha=0.1, group_weights=[1.0, -1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="alpha must be finite and non-negative"):
        tsecon.group_lasso(X, y, groups, alpha=-0.1)
    bad = X.copy()
    bad[3, 4] = np.nan
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in x"):
        tsecon.group_lasso(bad, y, groups, alpha=0.1)
    bad_y = y.copy()
    bad_y[0] = np.inf
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in y"):
        tsecon.group_lasso(X, bad_y, groups, alpha=0.1)
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        tsecon.group_lasso(X, y, groups, alpha=0.1, tol=0.0)


# ---------------------------------------------------------------- post lasso

@pytest.mark.parametrize("case", STRUCT["post_lasso"]["cases"], ids=lambda c: c["name"])
def test_post_lasso_matches_sklearn(case):
    X, y, _ = _design(case["design"])
    r = tsecon.post_lasso(X, y, alpha=case["alpha"], l1_ratio=case["l1_ratio"], tol=1e-11,
                          max_iter=100000)
    assert set(r.keys()) == POST_KEYS
    assert r["support"] == case["support"]
    assert r["n_selected"] == len(case["support"])
    np.testing.assert_allclose(r["coef_ols"], case["coef_ols"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(r["coef_lasso"], case["coef_lasso"], atol=1e-6, rtol=0)
    assert r["rss"] == pytest.approx(case["rss"], rel=1e-10)
    off = [j for j in range(X.shape[1]) if j not in case["support"]]
    assert np.all(r["coef_ols"][off] == 0.0)


def test_post_lasso_has_no_standard_errors_by_design():
    X, y, _ = _design("blocks")
    r = tsecon.post_lasso(X, y, alpha=0.1)
    assert not any(k.startswith("se") or "conf" in k or "p_value" in k for k in r)
    doc = tsecon.post_lasso.__doc__
    assert "No standard errors" in doc and "pds_lasso" in doc


def test_post_lasso_pandas_and_errors():
    pd = pytest.importorskip("pandas")
    X, y, _ = _design("blocks")
    ref = tsecon.post_lasso(X, y, alpha=0.1)
    r = tsecon.post_lasso(pd.DataFrame(X), pd.Series(y), alpha=0.1)
    np.testing.assert_allclose(r["coef_ols"], ref["coef_ols"], atol=1e-12)
    with pytest.raises(ValueError, match="l1_ratio must lie in"):
        tsecon.post_lasso(X, y, alpha=0.1, l1_ratio=2.0)
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in y"):
        tsecon.post_lasso(X, np.r_[y[:-1], np.nan], alpha=0.1)
    with pytest.raises(ValueError, match=r"^insufficient data: 5 observations, at least \d+ required"):
        tsecon.post_lasso(X[:5], y[:5], alpha=1e-9)


# ----------------------------------------------------------------------- pds

def _pds_data():
    d = STRUCT["pds"]
    return np.array(d["y"]), np.array(d["d"]), np.array(d["X"])


@pytest.mark.parametrize("case", STRUCT["pds"]["cases"], ids=lambda c: c["name"])
def test_pds_lasso_matches_statsmodels(case):
    y, d, X = _pds_data()
    rule = STRUCT["pds"]["newey_west_rule_maxlags"]
    hac_lags = None if case["hac_lags"] == rule else case["hac_lags"]
    r = tsecon.pds_lasso(y, d, X, alpha=case["alpha"], hac_lags=hac_lags, tol=1e-11,
                         max_iter=100000)
    assert set(r.keys()) == PDS_KEYS
    for key in ("coef", "se", "t_stat"):
        assert r[key] == pytest.approx(case[key], rel=1e-8)
    assert r["p_value"] == pytest.approx(case["p_value"], abs=1e-12)
    assert r["conf_int"] == pytest.approx(tuple(case["conf_int"]), rel=1e-8)
    assert r["support_y"] == case["support_y"]
    assert r["support_d"] == case["support_d"]
    assert r["union_support"] == case["union_support"]
    assert r["n_controls_selected"] == len(case["union_support"])
    assert r["hac_lags_resolved"] == case["hac_lags"]
    if case["alpha"] == "bic":
        assert r["alpha_y"] == pytest.approx(case["alpha_y"], rel=1e-10)
        assert r["alpha_d"] == pytest.approx(case["alpha_d"], rel=1e-10)
    else:
        assert r["alpha_y"] == case["alpha"] and r["alpha_d"] == case["alpha"]


def test_pds_lasso_defaults_and_alpha_forms():
    y, d, X = _pds_data()
    r = tsecon.pds_lasso(y, d, X)
    assert r["hac_lags_resolved"] == STRUCT["pds"]["newey_west_rule_maxlags"]
    r2 = tsecon.pds_lasso(y, d, X, alpha="bic", hac_lags=None)
    assert r2["coef"] == r["coef"] and r2["se"] == r["se"]
    # A float applied to both equations; an int is accepted as a float.
    r3 = tsecon.pds_lasso(y, d, X, alpha=0.05)
    assert r3["alpha_y"] == r3["alpha_d"] == 0.05
    r4 = tsecon.pds_lasso(y, d, X, alpha=1)
    assert r4["alpha_y"] == 1.0
    # hac_lags=0 is classical: narrower than HAC here, and t = coef / se.
    r5 = tsecon.pds_lasso(y, d, X, hac_lags=0)
    assert r5["hac_lags_resolved"] == 0
    assert r5["t_stat"] == pytest.approx(r5["coef"] / r5["se"])
    assert r5["conf_int"][0] < r5["coef"] < r5["conf_int"][1]
    # The union is the sorted union of the two supports.
    assert r["union_support"] == sorted(set(r["support_y"]) | set(r["support_d"]))


def test_pds_lasso_pandas_and_teaching_errors():
    pd = pytest.importorskip("pandas")
    y, d, X = _pds_data()
    ref = tsecon.pds_lasso(y, d, X)
    r = tsecon.pds_lasso(pd.Series(y), pd.Series(d), pd.DataFrame(X))
    assert r["coef"] == pytest.approx(ref["coef"], rel=1e-12)
    with pytest.raises(ValueError, match=r'alpha.*"aic".*accepted values.*"bic"'):
        tsecon.pds_lasso(y, d, X, alpha="aic")
    with pytest.raises(ValueError, match="alpha must be finite and non-negative"):
        tsecon.pds_lasso(y, d, X, alpha=-0.5)
    with pytest.raises(ValueError, match=r"hac_lags must be a non-negative integer \(got -1\).*hac_lags=0"):
        tsecon.pds_lasso(y, d, X, hac_lags=-1)
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in d"):
        tsecon.pds_lasso(y, np.r_[d[:-1], np.nan], X)
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in y"):
        tsecon.pds_lasso(np.r_[y[:-1], np.inf], d, X)
    with pytest.raises(ValueError, match=r"d length.*expected 200, got 10"):
        tsecon.pds_lasso(y, d[:10], X)
    with pytest.raises(ValueError, match=r"^insufficient data: 6 observations, at least \d+ required"):
        tsecon.pds_lasso(y[:6], d[:6], X[:6], alpha=1e-9, hac_lags=0)
    # A treatment identical to a control it selects makes the final OLS
    # design [d, x_0] singular: the HAC engine's error, not a panic.
    with pytest.raises(ValueError, match="HAC/OLS engine"):
        tsecon.pds_lasso(y, X[:, 0].copy(), X, alpha=1e-6, hac_lags=2)


# ------------------------------------------------------------------ docstrings

def test_docstrings_name_every_returned_key():
    def tokens(fn):
        return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))

    X, y, groups = _design("blocks")
    r = tsecon.group_lasso(X, y, groups, alpha=0.1)
    missing = set(r.keys()) - tokens(tsecon.group_lasso)
    assert not missing, f"group_lasso.__doc__ missing keys: {sorted(missing)}"
    r = tsecon.post_lasso(X, y, alpha=0.1)
    missing = set(r.keys()) - tokens(tsecon.post_lasso)
    assert not missing, f"post_lasso.__doc__ missing keys: {sorted(missing)}"
    yy, dd, XX = _pds_data()
    r = tsecon.pds_lasso(yy, dd, XX)
    missing = set(r.keys()) - tokens(tsecon.pds_lasso)
    assert not missing, f"pds_lasso.__doc__ missing keys: {sorted(missing)}"
    # The post-selection warning is on both surfaces that need it.
    assert "standard errors" in tsecon.post_lasso.__doc__
    assert "single selection" in tsecon.pds_lasso.__doc__.lower()
