"""Golden and behavioral tests for the JSZ canonical affine term-structure
bindings (``jsz_fit``, ``jsz_loadings``).

Re-pins ``fixtures/jsz.json`` (see ``fixtures/generate_jsz_fixtures.py`` for
the honest grading: the Riccati recursions are a documented-formula golden,
the portfolio VAR(1) an independent statsmodels golden, the likelihood a
documented-formula golden, the MLE a SciPy cross-optimizer target on a
simulated canonical model and on the real 1990-2007 GSW panel) through the
Python surface; checks the AFNS special case against ``afns_adjustment``
through the public surface; and exercises invariance to the portfolio basis,
exact pricing of the portfolios, determinism, and the teaching errors.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
JSZ = json.loads((FIX / "jsz.json").read_text())


def _sim_fit(**kw):
    sim = JSZ["sim"]
    return sim, tsecon.jsz_fit(
        np.asarray(sim["yields"], float), sim["maturities"],
        n_factors=sim["n_factors"], periods_per_year=sim["periods_per_year"],
        w=np.asarray(sim["truth"]["w"], float), **kw,
    )


# ---------------------------------------------------------------- loadings
@pytest.mark.parametrize("case", JSZ["loadings"], ids=lambda c: c["name"])
def test_jsz_loadings_match_the_documented_recursions(case):
    r = tsecon.jsz_loadings(
        np.asarray(case["lambda_q"]), case["k_inf_q"], np.asarray(case["sigma_x"]),
        case["maturities"], periods_per_year=case["periods_per_year"],
    )
    np.testing.assert_allclose(r["a_x"], case["a"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["b_x"], case["b"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(r["k0_q"], case["k0_q"])
    np.testing.assert_array_equal(r["k1_q"], case["k1_q"])
    assert list(r["maturities"]) == case["maturities"]


def test_jsz_afns_case_spans_nelson_siegel_and_converges_to_afns_adjustment():
    """The AFNS special case through the public surface: at lambda_q =
    (1, rho, rho) the JSZ loadings span the Nelson-Siegel loadings exactly,
    and the convexity intercept converges at first order in the period length
    to the CDR (2011) closed form that ``afns_adjustment`` implements."""
    afns = JSZ["afns"]
    lam, sig = afns["lambda_annual"], np.asarray(afns["sigma_annual"])
    tau = np.asarray(afns["tau_years"])
    ns = np.asarray(afns["ns_loadings"])
    cdr = tsecon.afns_adjustment(tau, sig, decay=lam)
    gaps = []
    for grid in afns["grids"]:
        ppy, rho, mats = grid["periods_per_year"], grid["rho"], grid["maturities"]
        l0 = tsecon.jsz_loadings(np.array([1.0, rho, rho]), 0.0, np.zeros((3, 3)), mats, ppy)
        assert np.asarray(l0["k1_q"])[1, 2] == 1.0          # the Jordan block
        span = np.asarray(l0["b_x"]) @ np.asarray(grid["rotation_c"])
        np.testing.assert_allclose(span, ns, rtol=0, atol=1e-10)
        l = tsecon.jsz_loadings(np.array([1.0, rho, rho]), 0.0, np.asarray(grid["sigma_x"]), mats, ppy)
        np.testing.assert_allclose(l["a_x"], grid["a_jsz"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(cdr, grid["a_cdr"], rtol=0, atol=1e-12)
        gap = float(np.abs(np.asarray(l["a_x"]) - cdr).max())
        assert gap <= grid["max_abs_gap"] * (1 + 1e-6)
        assert np.all(np.asarray(l["a_x"]) < 0)
        gaps.append(gap)
    for a, b in zip(gaps[:-1], gaps[1:]):
        assert 0.2 < b / a < 0.32                            # first order in dt
    assert gaps[0] < 1e-4 and gaps[-1] < 1e-6


# ---------------------------------------------------------------- sim panel
def test_jsz_fit_sim_portfolio_var_matches_statsmodels():
    sim, r = _sim_fit()
    var = sim["statsmodels_var"]
    params, stderr = np.asarray(var["params"]), np.asarray(var["stderr"])
    np.testing.assert_allclose(r["mu_p"], params[0], rtol=1e-9)
    np.testing.assert_allclose(np.asarray(r["phi_p"]), params[1:].T, rtol=1e-9)
    np.testing.assert_allclose(r["mu_p_se"], stderr[0], rtol=1e-9)
    np.testing.assert_allclose(np.asarray(r["phi_p_se"]), stderr[1:].T, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(r["sigma_ols"]), var["sigma_u_mle"], rtol=1e-9)


def test_jsz_fit_sim_mle_agrees_with_scipy_and_recovers_the_truth():
    sim, r = _sim_fit()
    truth, mle = sim["truth"], sim["mle"]
    assert r["converged"]
    assert r["llf"] >= mle["llf"] - 1e-4 and abs(r["llf"] - mle["llf"]) < 1e-2
    np.testing.assert_allclose(r["lambda_q"], mle["lambda_q"], rtol=0, atol=1e-5)
    assert r["k_inf_q"] == pytest.approx(mle["k_inf_q"], rel=1e-3)
    assert r["sigma_e"] == pytest.approx(mle["sigma_e"], rel=1e-4)
    # Recovery of the truth at the generator's measured tolerances.
    assert np.abs(np.asarray(r["lambda_q"]) - truth["lambda_q"]).max() < 2e-4
    assert abs(r["k_inf_q"] - truth["k_inf_q"]) / truth["k_inf_q"] < 0.03
    assert abs(r["sigma_e"] - truth["sigma_e"]) / truth["sigma_e"] < 0.02
    sig_true = np.asarray(truth["sigma"])
    assert np.abs(np.asarray(r["sigma"]) - sig_true).max() / np.abs(sig_true).max() < 0.2
    assert np.abs(np.asarray(r["b_p"]) - truth["b_p"]).max() < 2e-3
    assert np.abs(np.asarray(r["a_p"]) - truth["a_p"]).max() < 1e-4
    assert r["n_factors"] == 3 and r["n_starts"] == 5 and r["seed"] == 0
    assert r["n_iter"] > 0


def test_jsz_fit_prices_the_portfolios_exactly_and_the_decomposition_is_exact():
    sim, r = _sim_fit()
    y = np.asarray(sim["yields"])
    w, fitted = np.asarray(r["w"]), np.asarray(r["fitted"])
    np.testing.assert_allclose(fitted @ w.T, y @ w.T, rtol=0, atol=1e-13)
    np.testing.assert_allclose(fitted @ w.T, np.asarray(r["factors"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(fitted, np.asarray(r["risk_neutral"]) + np.asarray(r["term_premium"]),
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(w @ np.asarray(r["b_p"]), np.eye(3), rtol=0, atol=1e-10)
    np.testing.assert_allclose(w @ np.asarray(r["a_p"]), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["lambda0"], np.asarray(r["mu_p"]) - np.asarray(r["k0_q_p"]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(r["lambda1"], np.asarray(r["phi_p"]) - np.asarray(r["k1_q_p"]), rtol=0, atol=1e-15)
    rmse = np.sqrt(((y - fitted) ** 2).mean(axis=0))
    np.testing.assert_allclose(r["rmse"], rmse, rtol=1e-12)
    assert np.asarray(r["b_x"]).shape == (10, 3) and np.asarray(r["a_x"]).shape == (10,)


def test_jsz_fit_is_invariant_to_the_basis_of_the_portfolio_space():
    sim = JSZ["sim"]
    y = np.asarray(sim["yields"])
    w = np.asarray(sim["truth"]["w"])
    g = np.array([[2.0, 0.5, -0.3], [0.1, -1.5, 0.2], [0.4, 0.3, 3.0]])
    kw = dict(n_factors=3, periods_per_year=12.0)
    r1 = tsecon.jsz_fit(y, sim["maturities"], w=w, **kw)
    r2 = tsecon.jsz_fit(y, sim["maturities"], w=g @ w, **kw)
    assert r1["llf"] == pytest.approx(r2["llf"], rel=1e-8)
    np.testing.assert_allclose(r1["lambda_q"], r2["lambda_q"], rtol=0, atol=1e-8)
    assert r1["k_inf_q"] == pytest.approx(r2["k_inf_q"], rel=1e-7, abs=1e-8)
    assert r1["sigma_e"] == pytest.approx(r2["sigma_e"], rel=1e-8)
    np.testing.assert_allclose(r1["fitted"], r2["fitted"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(r1["term_premium"], r2["term_premium"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(r2["factors"]), np.asarray(r1["factors"]) @ g.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(r2["mu_p"], g @ np.asarray(r1["mu_p"]), rtol=0, atol=1e-10)
    # Sigma_P is the flattest direction: agreement to the optimizer's tolerance.
    s2 = g @ np.asarray(r1["sigma"]) @ g.T
    assert np.abs(np.asarray(r2["sigma"]) - s2).max() < 1e-6 * np.abs(s2).max()


def test_jsz_fit_is_deterministic_and_the_seed_only_moves_the_perturbed_starts():
    _, a = _sim_fit()
    _, b = _sim_fit()
    assert a["llf"] == b["llf"]
    np.testing.assert_array_equal(a["lambda_q"], b["lambda_q"])
    np.testing.assert_array_equal(a["fitted"], b["fitted"])
    _, c = _sim_fit(seed=7)
    assert c["seed"] == 7
    # A different seed reaches the same (unique, on this panel) optimum.
    np.testing.assert_allclose(c["lambda_q"], a["lambda_q"], rtol=0, atol=1e-6)
    assert c["llf"] == pytest.approx(a["llf"], abs=1e-6)
    # A single start is the JSZ start alone (no perturbations), same optimum here.
    _, d = _sim_fit(n_starts=1)
    assert d["n_starts"] == 1
    np.testing.assert_allclose(d["lambda_q"], a["lambda_q"], rtol=0, atol=1e-6)


# ---------------------------------------------------------------- GSW panel
def test_jsz_fit_gsw_reproduces_the_fixture_and_the_illustration():
    gsw = JSZ["gsw"]
    y = np.asarray(gsw["yields"])
    r = tsecon.jsz_fit(y, gsw["maturities"])          # all defaults: N = 3, monthly, PCA
    assert r["converged"]
    np.testing.assert_allclose(r["w"], gsw["w"], rtol=0, atol=1e-10)
    var = gsw["statsmodels_var"]
    np.testing.assert_allclose(r["mu_p"], np.asarray(var["params"])[0], rtol=1e-9)
    np.testing.assert_allclose(np.asarray(r["phi_p"]), np.asarray(var["params"])[1:].T, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(r["sigma_ols"]), var["sigma_u_mle"], rtol=1e-9)
    mle, ill = gsw["mle"], gsw["illustration"]
    assert r["llf"] >= mle["llf"] - 1e-4 and abs(r["llf"] - mle["llf"]) < 1e-2
    np.testing.assert_allclose(r["lambda_q"], mle["lambda_q"], rtol=0, atol=1e-5)
    assert r["sigma_e"] * 1e4 == pytest.approx(ill["sigma_e_bp"], abs=1e-3)
    np.testing.assert_allclose(np.asarray(r["rmse"]) * 1e4, ill["rmse_bp"], rtol=0, atol=1e-3)
    tp10 = np.asarray(r["term_premium"])[:, gsw["maturities"].index(120)]
    np.testing.assert_allclose(tp10, gsw["term_premium_120"], rtol=0, atol=1e-7)
    assert tp10.mean() * 100 == pytest.approx(ill["tp10_mean_pp"], abs=1e-4)
    # The multimodality on record: the SciPy starts found three basins and
    # the best one is where both optimizers agree.
    basins = sorted(set(np.round(mle["basin_llfs"], 1)), reverse=True)
    assert len(basins) >= 3 and basins[0] == pytest.approx(mle["llf"], abs=0.1)
    assert 0.99 < r["lambda_q"][0] < 1.0


# ---------------------------------------------------------------- refusals
def test_jsz_fit_rejects_invalid_inputs_with_teaching_errors():
    sim = JSZ["sim"]
    y = np.asarray(sim["yields"])[:60]
    mats = list(sim["maturities"])
    m = len(mats)
    with pytest.raises(ValueError, match="strictly ascending"):
        tsecon.jsz_fit(y, mats[:3] + [mats[2]] + mats[4:])
    with pytest.raises(ValueError, match="strictly ascending"):
        tsecon.jsz_fit(y, list(reversed(mats)))
    with pytest.raises(ValueError, match="n_factors"):
        tsecon.jsz_fit(y, mats, n_factors=m)
    with pytest.raises(ValueError, match="n_factors"):
        tsecon.jsz_fit(y, mats, n_factors=0)
    with pytest.raises(ValueError, match="needs at least"):
        tsecon.jsz_fit(y[:6], mats)
    bad = y.copy()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.jsz_fit(bad, mats)
    with pytest.raises(ValueError, match="periods_per_year"):
        tsecon.jsz_fit(y, mats, periods_per_year=0.0)
    with pytest.raises(ValueError, match="n_factors x n_maturities"):
        tsecon.jsz_fit(y, mats, w=np.ones((2, m)))
    w = np.asarray(sim["truth"]["w"])
    dependent = w.copy()
    dependent[2] = w[0] + w[1]
    with pytest.raises(ValueError, match="linearly dependent"):
        tsecon.jsz_fit(y, mats, w=dependent)
    with pytest.raises(ValueError, match="n_starts = 0"):
        tsecon.jsz_fit(y, mats, n_starts=0)
    with pytest.raises(ValueError, match="n_starts"):
        tsecon.jsz_fit(y, mats, n_starts=10_000)
    with pytest.raises(ValueError, match="seed was given but n_starts=1"):
        tsecon.jsz_fit(y, mats, n_starts=1, seed=3)
    with pytest.raises(ValueError, match="exceeds the supported maximum"):
        tsecon.jsz_fit(y, mats[:-1] + [10 ** 10])
    with pytest.raises(ValueError, match="descending"):
        tsecon.jsz_loadings(np.array([0.9, 0.95, 0.8]), 0.0, np.zeros((3, 3)), mats)
    with pytest.raises(ValueError, match="finite real"):
        tsecon.jsz_loadings(np.array([0.9, np.nan, 0.8]), 0.0, np.zeros((3, 3)), mats)
    with pytest.raises(ValueError, match="n_factors x n_factors"):
        tsecon.jsz_loadings(np.array([0.9, 0.8]), 0.0, np.zeros((3, 3)), mats)
    with pytest.raises(ValueError, match="periods_per_year"):
        tsecon.jsz_loadings(np.array([0.9, 0.8, 0.7]), 0.0, np.zeros((3, 3)), mats, periods_per_year=-1.0)


def test_jsz_integer_maturities_pass_through_coercion():
    """`maturities` is an integer spec (exempt from float64 coercion): lists,
    tuples and integer arrays all reach the estimator unchanged."""
    case = JSZ["loadings"][0]
    lam, sig = np.asarray(case["lambda_q"]), np.asarray(case["sigma_x"])
    a = tsecon.jsz_loadings(lam, case["k_inf_q"], sig, case["maturities"], periods_per_year=12.0)
    b = tsecon.jsz_loadings(lam, case["k_inf_q"], sig, tuple(case["maturities"]), periods_per_year=12.0)
    c = tsecon.jsz_loadings(lam, case["k_inf_q"], sig, np.asarray(case["maturities"]), periods_per_year=12.0)
    np.testing.assert_array_equal(a["a_x"], b["a_x"])
    np.testing.assert_array_equal(a["a_x"], c["a_x"])


@pytest.mark.parametrize("name", ["jsz_fit", "jsz_loadings"])
def test_jsz_docstrings_name_every_returned_key(name):
    import re
    fn = getattr(tsecon, name)
    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))
    if name == "jsz_fit":
        _, out = _sim_fit()
    else:
        case = JSZ["loadings"][0]
        out = tsecon.jsz_loadings(np.asarray(case["lambda_q"]), case["k_inf_q"],
                                  np.asarray(case["sigma_x"]), case["maturities"])
    missing = set(out.keys()) - tokens
    assert not missing, f"{name}.__doc__ does not name returned keys: {sorted(missing)}"
