"""Golden and behavioral tests for the Hansen-Seo threshold-VECM bindings.

Re-pins fixtures/tvecm.json (an independent NumPy transcription of the
Hansen-Seo 2002 grid estimator and sup-LM statistic — see the generator
header for the honest grading: no third-party reference was runnable in
the fixture container) through the Python surface, and checks the
bootstrap test's seeded determinism and its verdicts on threshold- vs.
linear-cointegrated data.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
TVECM = json.loads((FIX / "tvecm.json").read_text())


def _fit_kwargs(case):
    kw = dict(
        k_ar_diff=case["k_ar_diff"],
        trim=case["trim"],
        n_grid_gamma=case["n_grid_gamma"],
    )
    if case["beta_fixed"] is not None:
        # With beta fixed the beta grid search never runs, and explicit
        # n_grid_beta/beta_span are refused since the audit-10 fix (they
        # were verified inert here first — same fixture values, same
        # results with them omitted).
        kw["beta"] = case["beta_fixed"]
    else:
        kw["n_grid_beta"] = case["n_grid_beta"]
        kw["beta_span"] = case["beta_span"]
    return kw


@pytest.mark.parametrize("case", TVECM["fit"], ids=lambda c: (
    f"{c['series']}-l{c['k_ar_diff']}-t{c['trim']}-"
    f"{'fixed' if c['beta_fixed'] is not None else 'grid'}"
))
def test_tvecm_fit_matches_fixture(case):
    y = np.array(TVECM["series"][case["series"]])
    r = tsecon.threshold_vecm(y, **_fit_kwargs(case))
    tol = case["tol"]
    np.testing.assert_allclose(r["beta"], case["beta"], rtol=tol)
    assert r["threshold"] == pytest.approx(case["threshold"], rel=tol)
    np.testing.assert_allclose(r["params_low"], case["coefs_low"], rtol=tol)
    np.testing.assert_allclose(r["params_high"], case["coefs_high"], rtol=tol)
    np.testing.assert_allclose(r["bse_low"], case["se_low"], rtol=tol)
    np.testing.assert_allclose(r["bse_high"], case["se_high"], rtol=tol)
    assert r["n_low"] == case["n_low"]
    assert r["n_high"] == case["n_high"]
    assert r["nobs"] == case["nobs"]
    assert r["frac_low"] == pytest.approx(case["frac_low"], rel=tol)
    np.testing.assert_allclose(r["sigma"], case["sigma"], rtol=tol)
    assert r["log_det_sigma"] == pytest.approx(case["log_det_sigma"], rel=tol)
    assert r["llf"] == pytest.approx(case["llf"], rel=tol)
    # Eigen-based here vs OLS-based in the generator: analytically equal.
    assert r["llf_linear"] == pytest.approx(case["llf_linear"], rel=1e-8)
    np.testing.assert_allclose(r["beta_linear"], case["beta_linear"], rtol=tol)
    np.testing.assert_allclose(r["beta_grid"], case["beta_grid"], rtol=tol)
    assert r["min_regime"] == case["min_regime"]
    assert r["n_regressors"] == case["n_regressors"]
    assert r["k_ar_diff"] == case["k_ar_diff"]
    assert r["neqs"] == y.shape[1]
    # The ect series recovers the reported split.
    assert (np.asarray(r["ect"]) <= r["threshold"]).sum() == r["n_low"]


@pytest.mark.parametrize("case", TVECM["test"], ids=lambda c: (
    f"{c['series']}-l{c['k_ar_diff']}-g{c['n_grid']}-"
    f"{'fixed' if c['beta_fixed'] is not None else 'est'}"
))
def test_hansen_seo_statistic_matches_fixture(case):
    y = np.array(TVECM["series"][case["series"]])
    kw = {}
    if case["beta_fixed"] is not None:
        kw["beta"] = case["beta_fixed"]
    r = tsecon.hansen_seo_test(
        y, k_ar_diff=case["k_ar_diff"], trim=case["trim"],
        n_grid=case["n_grid"], n_boot=19, seed=0, **kw,
    )
    tol = case["tol"]
    assert r["stat"] == pytest.approx(case["stat"], rel=tol)
    assert r["threshold"] == pytest.approx(case["threshold"], rel=tol)
    np.testing.assert_allclose(r["beta"], case["beta"], rtol=tol)
    assert r["nobs"] == case["nobs"]
    assert r["min_regime"] == case["min_regime"]
    assert r["n_regressors"] == case["n_regressors"]
    np.testing.assert_allclose(r["thresholds"], case["thresholds"], rtol=tol)
    np.testing.assert_allclose(r["lm_path"], case["lm_path"], rtol=tol)
    assert len(r["boot_stats"]) == 19
    assert 0.0 < r["p_value"] <= 1.0


def test_bootstrap_is_seed_deterministic_and_verdicts_split():
    strong = np.array(TVECM["series"]["tv_strong"])
    linear = np.array(TVECM["series"]["tv_linear"])
    a = tsecon.hansen_seo_test(strong, n_grid=50, n_boot=199, seed=42)
    b = tsecon.hansen_seo_test(strong, n_grid=50, n_boot=199, seed=42)
    np.testing.assert_array_equal(a["boot_stats"], b["boot_stats"])
    assert a["p_value"] == b["p_value"]
    # Threshold cointegration is detected; linear cointegration is not.
    assert a["p_value"] <= 0.01
    c = tsecon.hansen_seo_test(linear, n_grid=50, n_boot=199, seed=42)
    assert c["p_value"] > 0.10


def test_two_regime_fit_nests_the_linear_fit():
    y = np.array(TVECM["series"]["tv_linear"])
    r = tsecon.threshold_vecm(y, beta=[1.0, -1.0])
    assert r["llf"] >= r["llf_linear"]
    assert r["n_low"] + r["n_high"] == r["nobs"]
    assert min(r["n_low"], r["n_high"]) >= r["min_regime"]


def test_pandas_input_accepted():
    pd = pytest.importorskip("pandas")
    y = np.array(TVECM["series"]["tv_strong"])
    df = pd.DataFrame(y, columns=["long_rate", "short_rate"])
    a = tsecon.threshold_vecm(df, beta=[1.0, -1.0])
    b = tsecon.threshold_vecm(y, beta=[1.0, -1.0])
    assert a["threshold"] == b["threshold"]
    np.testing.assert_array_equal(a["params_low"], b["params_low"])


def test_teaching_errors():
    y = np.array(TVECM["series"]["tv_strong"])
    y3 = np.array(TVECM["series"]["tv3"])
    with pytest.raises(ValueError, match="trim"):
        tsecon.threshold_vecm(y, trim=0.7)
    with pytest.raises(ValueError, match="at least two series"):
        tsecon.threshold_vecm(y[:, :1].reshape(-1, 1))
    with pytest.raises(ValueError, match="bivariate"):
        tsecon.threshold_vecm(y3)  # k = 3 without beta
    with pytest.raises(ValueError, match="beta"):
        tsecon.threshold_vecm(y, beta=[1.0, -1.0, 0.3])  # wrong length
    with pytest.raises(ValueError, match="nonzero"):
        tsecon.threshold_vecm(y, beta=[0.0, 1.0])
    with pytest.raises(ValueError, match="n_boot"):
        tsecon.hansen_seo_test(y, n_boot=0)
    with pytest.raises(ValueError, match="gamma grid"):
        tsecon.threshold_vecm(y, n_grid_gamma=1)
    with pytest.raises(ValueError, match="non-finite"):
        bad = y.copy()
        bad[5, 0] = np.nan
        tsecon.threshold_vecm(bad, beta=[1.0, -1.0])
    with pytest.raises(ValueError):
        tsecon.threshold_vecm(y[:8], beta=[1.0, -1.0])  # too short


def test_docstrings_name_every_returned_key():
    import re

    y = np.array(TVECM["series"]["tv_strong"])
    for fn, res in [
        (tsecon.threshold_vecm, tsecon.threshold_vecm(y, beta=[1.0, -1.0])),
        (tsecon.hansen_seo_test,
         tsecon.hansen_seo_test(y, n_grid=40, n_boot=9)),
    ]:
        tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__))
        missing = set(res.keys()) - tokens
        assert not missing, f"{fn.__name__} docstring misses {sorted(missing)}"


# --------------------------------------------------------------------------
# Audit round 10: the beta-grid kwargs are refused when beta is fixed
# --------------------------------------------------------------------------

def test_beta_grid_kwargs_refused_with_fixed_beta():
    """n_grid_beta/beta_span size the beta grid search, which a supplied
    beta= never runs (beta_grid comes back empty, as documented); explicit
    use together raises (verified bit-identical no-ops before the fix)."""
    y = np.array(TVECM["series"]["tv_strong"])
    for kwargs in (dict(n_grid_beta=5), dict(beta_span=2.0),
                   dict(n_grid_beta=5, beta_span=2.0)):
        with pytest.raises(ValueError, match="n_grid_beta/beta_span") as exc:
            tsecon.threshold_vecm(y, beta=[1.0, -1.0], **kwargs)
        assert "grid search" in str(exc.value) and "beta_grid" in str(exc.value)
    # The fixed-beta call still documents the never-ran search honestly.
    r = tsecon.threshold_vecm(y, beta=[1.0, -1.0])
    assert len(np.asarray(r["beta_grid"])) == 0


def test_beta_grid_kwargs_sentinel_defaults_and_live_when_estimated():
    y = np.array(TVECM["series"]["tv_strong"])
    # Sentinel resolution: omitted == the historical explicit 50/10.0.
    a = tsecon.threshold_vecm(y)
    b = tsecon.threshold_vecm(y, n_grid_beta=50, beta_span=10.0)
    assert a["threshold"] == b["threshold"] and a["llf"] == b["llf"]
    np.testing.assert_array_equal(a["beta"], b["beta"])
    np.testing.assert_array_equal(a["beta_grid"], b["beta_grid"])
    # Live where documented: the grid size/span move the searched grid.
    c = tsecon.threshold_vecm(y, n_grid_beta=7)
    assert len(np.asarray(c["beta_grid"])) == 7 != len(np.asarray(a["beta_grid"]))
    d = tsecon.threshold_vecm(y, beta_span=1.0)
    assert not np.array_equal(np.asarray(d["beta_grid"]), np.asarray(a["beta_grid"]))
