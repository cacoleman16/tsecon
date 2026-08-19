"""Binding tests for `dcs_local_level` (DCS robust local level,
tsecon-gas crate).

The tight numeric goldens live in the crate's Rust tests
(`crates/tsecon-gas/tests/dcs_golden.rs`); this file RE-PINS the
fitted-parameter golden through the Python boundary — the scipy MLE of the
identical criterion stored in `fixtures/tsecon-dcs.json` — and exercises
the robustness, nesting, and error surfaces end to end.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
DCS = json.loads((FIXTURES / "tsecon-dcs.json").read_text())


# ----------------------------------------------- gaussian golden re-pin
@pytest.mark.parametrize("case", DCS["gaussian_ss"], ids=lambda c: c["series"])
def test_gaussian_fit_repins_scipy_mle(case):
    """Two optimizers, one criterion: the binding's fit must land on the
    fixture's scipy L-BFGS-B + Nelder-Mead-polish MLE (params 1e-4
    relative, loglik 1e-8), on two seeded local levels and the Nile."""
    y = np.asarray(case["y"])
    mle = case["dcs_mle"]
    r = tsecon.dcs_local_level(y, density="gaussian")
    assert r["kappa"] == pytest.approx(mle["kappa"], rel=1e-4)
    assert r["scale"] == pytest.approx(mle["scale"], rel=1e-4)
    assert r["loglik"] == pytest.approx(mle["loglik"], rel=1e-8)
    assert r["converged"]
    assert r["density"] == "gaussian"
    # kappa is the steady-state Kalman gain of the UC-MLE fit up to the
    # finite-sample transient (mapping: kappa = p/(1+p),
    # p = (q + sqrt(q^2+4q))/2, q = s2_eta/s2_eps).
    assert abs(r["kappa"] - case["map"]["kappa"]) < 0.02
    # the mapping's inverse round-trips: q = kappa^2/(1-kappa)
    k = case["map"]["kappa"]
    assert case["map"]["q"] == pytest.approx(k * k / (1.0 - k), rel=1e-10)
    # result-surface invariants
    lvl, resid = np.asarray(r["level"]), np.asarray(r["resid"])
    assert lvl.shape == y.shape and resid.shape == y.shape
    np.testing.assert_allclose(resid, y - lvl, atol=1e-12)
    assert np.isfinite(r["next_level"])
    assert r["n_obs"] == len(y)
    assert r["aic"] == pytest.approx(2 * 2 - 2 * r["loglik"])
    assert r["bic"] == pytest.approx(2 * np.log(len(y)) - 2 * r["loglik"])
    assert r["kappa_se"] > 0 and r["scale_se"] > 0
    assert "nu" not in r  # gaussian carries no dof


# ----------------------------------------------------- robustness re-pin
def test_t_beats_gaussian_on_contaminated_fixture_series():
    """On the fixture-frozen contaminated local levels (8-sigma additive
    outliers, clean truth stored), the DCS-t one-step level RMSE beats the
    Gaussian control's — the graduation margin, one seed at a time."""
    for key, bound in [("sim_contam5", 0.90), ("sim_contam10", 0.85)]:
        case = DCS[key]
        y = np.asarray(case["y"])
        mu_true = np.asarray(case["mu_true"])
        rg = tsecon.dcs_local_level(y, density="gaussian")
        rt = tsecon.dcs_local_level(y, density="t")
        rmse_g = float(np.sqrt(np.mean((np.asarray(rg["level"]) - mu_true) ** 2)))
        rmse_t = float(np.sqrt(np.mean((np.asarray(rt["level"]) - mu_true) ** 2)))
        assert rmse_t < bound * rmse_g, (key, rmse_t, rmse_g)
        assert rt["nu"] > 2.0
        assert rt["converged"]


def test_t_nests_gaussian_on_clean_data_with_honest_flag():
    """On clean Gaussian data the t fit collapses onto the Gaussian filter
    (huge nu, near-identical path) and honestly reports converged=False —
    the nu boundary has no interior optimum to certify.

    The flag is deterministic, not an optimizer accident: whether the
    simplex happens to collapse on the flat nu ridge varies with the
    platform's libm (this exact assertion caught converged=True on
    Windows MSVC while Linux reported False), so past the crate's
    NU_GAUSSIAN_RIDGE threshold the fit forces the flag False everywhere.
    The loglik assertion pins the companion fix: the t normalizing
    constant used to cancel catastrophically at huge nu, reporting
    loglik = +54230 on this series against the -744 Gaussian limit it
    cannot exceed (measured Linux gap after the fix: 3.4e-13)."""
    case = DCS["gaussian_ss"][0]
    y = np.asarray(case["y"])
    rg = tsecon.dcs_local_level(y, density="gaussian")
    rt = tsecon.dcs_local_level(y, density="t")
    gap = np.sqrt(np.mean((np.asarray(rt["level"]) - np.asarray(rg["level"])) ** 2))
    assert gap < 0.05
    assert rt["nu"] > 30.0
    assert not rt["converged"]
    assert rg["converged"]
    assert rt["loglik"] == pytest.approx(rg["loglik"], abs=0.01)


def test_laplace_sign_filter_is_sane():
    case = DCS["sim_contam10"]
    y = np.asarray(case["y"])
    r = tsecon.dcs_local_level(y, density="laplace")
    assert r["density"] == "laplace"
    assert "nu" not in r and "nu_se" not in r
    assert r["kappa"] > 0 and r["scale"] > 0
    assert np.isfinite(r["loglik"])
    # the sign filter is robust too: its level stays near the clean truth
    mu_true = np.asarray(case["mu_true"])
    rmse_l = float(np.sqrt(np.mean((np.asarray(r["level"]) - mu_true) ** 2)))
    assert rmse_l < 0.6  # gaussian control measures ~0.46 on this seed


# ------------------------------------------------------------- errors
def test_rejects_unknown_density():
    with pytest.raises(ValueError, match="unknown density"):
        tsecon.dcs_local_level(np.random.default_rng(0).standard_normal(100),
                               density="cauchy")


def test_refuses_constant_series():
    with pytest.raises(ValueError, match="degenerate"):
        tsecon.dcs_local_level(np.full(100, 3.14))


def test_refuses_short_series():
    with pytest.raises(ValueError, match="insufficient data"):
        tsecon.dcs_local_level(np.arange(10.0) ** 0.5)


def test_refuses_nan():
    y = np.random.default_rng(1).standard_normal(100)
    y[13] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.dcs_local_level(y)
