"""Binding tests for tsecon-evt: POT/GPD tails and GEV block maxima.

Golden pins re-play fixtures/tsecon-evt.json through the Python surface —
the same scipy genpareto/genextreme reference the Rust golden pins (polished
fits at 1e-6, loglik at 1e-10, observed-information SEs at 1e-4, VaR/ES and
return levels at 1e-5; see the generator header for the honest grading).
Structural tests cover the binding-level contract: defaults, dict keys,
error propagation, and the ES >= VaR ordering.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"
FX = json.loads((FIXTURES / "tsecon-evt.json").read_text())


# --------------------------------------------------------------------------- #
# GPD — pinned to polished scipy.stats.genpareto.fit(z, floc=0)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", FX["gpd"], ids=lambda c: c["name"])
def test_gpd_fit_matches_scipy(case):
    y = np.asarray(case["y"], float)
    kwargs = {"p_tail": case["p_tail"]}
    if case["threshold_arg"] is not None:
        kwargs["threshold"] = case["threshold_arg"]
    else:
        kwargs["quantile"] = case["quantile_arg"]
    out = tsecon.gpd_fit(y, **kwargs)

    assert out["n_exceed"] == case["n_exceed"]
    np.testing.assert_allclose(out["threshold"], case["threshold"], rtol=1e-12)
    np.testing.assert_allclose(
        out["threshold_quantile"], case["threshold_quantile"], rtol=1e-12
    )
    np.testing.assert_allclose(out["xi"], case["xi"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["beta"], case["beta"], rtol=1e-6, atol=1e-6)
    # Two optimizers, same maximum: loglik agreement at 1e-10 absolute,
    # and ours never worse.
    assert out["loglik"] >= case["loglik"] - 1e-10
    np.testing.assert_allclose(out["loglik"], case["loglik"], rtol=0, atol=1e-10)
    assert out["converged"] and out["se_valid"]
    np.testing.assert_allclose(out["se_xi"], case["se_xi"], rtol=1e-4)
    np.testing.assert_allclose(out["se_beta"], case["se_beta"], rtol=1e-4)
    np.testing.assert_allclose(np.asarray(out["var"]), case["var"], rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out["es"]), case["es"], rtol=1e-5)
    assert np.all(np.asarray(out["es"]) >= np.asarray(out["var"]))


# --------------------------------------------------------------------------- #
# GEV — pinned to polished scipy.stats.genextreme.fit(maxima)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", FX["gev"], ids=lambda c: c["name"])
def test_gev_fit_matches_scipy(case):
    y = np.asarray(case["y"], float)
    out = tsecon.gev_fit(
        y, block_size=case["block_size"], return_periods=case["return_periods"]
    )
    assert out["n_maxima"] == case["n_maxima"]
    assert out["block_size"] == case["block_size"]
    np.testing.assert_allclose(out["xi"], case["xi"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["mu"], case["mu"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["sigma"], case["sigma"], rtol=1e-6, atol=1e-6)
    assert out["loglik"] >= case["loglik"] - 1e-10
    np.testing.assert_allclose(out["loglik"], case["loglik"], rtol=0, atol=1e-10)
    assert out["converged"] and out["se_valid"]
    np.testing.assert_allclose(out["se_xi"], case["se_xi"], rtol=1e-4)
    np.testing.assert_allclose(out["se_mu"], case["se_mu"], rtol=1e-4)
    np.testing.assert_allclose(out["se_sigma"], case["se_sigma"], rtol=1e-4)
    np.testing.assert_allclose(
        np.asarray(out["return_levels"]), case["return_levels"], rtol=1e-5
    )


# --------------------------------------------------------------------------- #
# binding-level contract
# --------------------------------------------------------------------------- #
def _exp_grid(n):
    u = (np.arange(n) + 0.5) / n
    return -np.log1p(-u)


def test_gpd_defaults_and_keys():
    y = _exp_grid(1000)
    out = tsecon.gpd_fit(y)
    assert set(out) == {
        "threshold", "threshold_quantile", "n", "n_exceed", "exceed_rate",
        "xi", "beta", "se_xi", "se_beta", "se_valid", "loglik", "converged",
        "p_tail", "var", "es",
    }
    np.testing.assert_allclose(np.asarray(out["p_tail"]), [0.99, 0.995, 0.999])
    assert out["n"] == 1000 and out["threshold_quantile"] == 0.90
    assert abs(out["xi"]) < 0.05  # exponential data: xi near 0


def test_gev_defaults_and_keys():
    u = (np.arange(300) + 0.5) / 300.0
    maxima = -np.log(-np.log(u))  # exact Gumbel quantile grid
    out = tsecon.gev_fit(maxima)
    assert set(out) == {
        "xi", "mu", "sigma", "se_xi", "se_mu", "se_sigma", "se_valid",
        "loglik", "converged", "n_maxima", "block_size", "return_periods",
        "return_levels",
    }
    np.testing.assert_allclose(np.asarray(out["return_periods"]), [10.0, 50.0, 100.0])
    assert out["block_size"] is None
    assert abs(out["xi"]) < 0.05
    assert np.all(np.diff(np.asarray(out["return_levels"])) > 0)


def test_gpd_errors_teach():
    y = _exp_grid(60)  # 0.9 quantile -> 6 exceedances < 10
    with pytest.raises(ValueError, match="exceed the threshold"):
        tsecon.gpd_fit(y)
    with pytest.raises(ValueError, match="threshold quantile"):
        tsecon.gpd_fit(_exp_grid(500), quantile=1.5)
    with pytest.raises(ValueError, match="does not reach beyond"):
        tsecon.gpd_fit(_exp_grid(500), p_tail=[0.5])


def test_gev_errors_teach():
    with pytest.raises(ValueError, match="block maxima"):
        tsecon.gev_fit(_exp_grid(100), block_size=20)  # 5 maxima < 10
    with pytest.raises(ValueError, match="block_size"):
        tsecon.gev_fit(_exp_grid(100), block_size=0)
    with pytest.raises(ValueError, match="return period"):
        tsecon.gev_fit(_exp_grid(100), return_periods=[1.0])
