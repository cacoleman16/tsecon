"""Golden + structural tests for the spec-test, AFNS, and DSGE bindings.

Covers the six functions wired from ``tsecon-spectest`` (E9), ``tsecon-termstructure``
AFNS (E2), and ``tsecon-dsge`` (E5):

    heteroskedasticity_test, reset_test, chow_test, cusum_test,
    afns_adjustment, dsge_solve

The spec-test and AFNS goldens are checked against offline references in
``fixtures/`` (statsmodels for the spec tests; the closed-form AFNS yield
adjustment for the term-structure factors). The DSGE check is structural: the
Cagan money-demand model has a known closed-form saddle-path solution.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def _design(case):
    """(n, k) design matrix; column 0 is the explicit intercept."""
    return np.column_stack([np.asarray(c, float) for c in case["columns"]])


# --------------------------------------------------------------------------- #
# White / Breusch-Pagan heteroskedasticity tests
# --------------------------------------------------------------------------- #
SPEC = _load("tsecon-spectest.json")


@pytest.mark.parametrize("case", SPEC["white_breusch_pagan"], ids=lambda c: c["name"])
def test_white_and_breusch_pagan(case):
    y = np.asarray(case["y"], float)
    x = _design(case)

    w = tsecon.heteroskedasticity_test(y, x, test="white")
    ref = case["white"]
    assert w["df"] == ref["df"]
    assert w["statistic"] == pytest.approx(ref["statistic"], rel=1e-8)
    assert w["pvalue"] == pytest.approx(ref["pvalue"], rel=1e-6, abs=1e-14)
    assert w["fstat"] == pytest.approx(ref["fstat"], rel=1e-8)

    bp = tsecon.heteroskedasticity_test(y, x, test="breusch_pagan")
    ref = case["breusch_pagan"]
    assert bp["df"] == ref["df"]
    assert bp["statistic"] == pytest.approx(ref["statistic"], rel=1e-8)
    assert bp["pvalue"] == pytest.approx(ref["pvalue"], rel=1e-6, abs=1e-14)

    # "bp" is an accepted alias for the same test.
    assert tsecon.heteroskedasticity_test(y, x, test="bp")["statistic"] == pytest.approx(
        bp["statistic"], rel=1e-12
    )


def test_heteroskedasticity_test_rejects_unknown():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.column_stack([np.ones(4), np.arange(4.0)])
    with pytest.raises(ValueError, match="unknown test"):
        tsecon.heteroskedasticity_test(y, x, test="goldfeld_quandt")


# --------------------------------------------------------------------------- #
# RESET functional-form test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", SPEC["reset"], ids=lambda c: c["name"])
def test_reset(case):
    y = np.asarray(case["y"], float)
    x = _design(case)
    r = tsecon.reset_test(y, x, max_power=3)
    ref = case["reset"]
    assert r["df_num"] == ref["df_num"]
    assert r["df_den"] == ref["df_den"]
    assert r["fstat"] == pytest.approx(ref["fstat"], rel=1e-7)
    assert r["pvalue"] == pytest.approx(ref["pvalue"], rel=1e-6, abs=1e-14)


# --------------------------------------------------------------------------- #
# Chow structural-break test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", SPEC["chow"], ids=lambda c: c["name"])
def test_chow(case):
    y = np.asarray(case["y"], float)
    x = _design(case)
    r = tsecon.chow_test(y, x, split=case["split"])
    ref = case["chow"]
    assert r["df_num"] == ref["df_num"]
    assert r["df_den"] == ref["df_den"]
    assert r["fstat"] == pytest.approx(ref["fstat"], rel=1e-7)
    assert r["pvalue"] == pytest.approx(ref["pvalue"], rel=1e-6, abs=1e-14)
    assert r["ssr_pooled"] == pytest.approx(ref["ssr_pooled"], rel=1e-8)
    assert r["ssr1"] == pytest.approx(ref["ssr1"], rel=1e-8)
    assert r["ssr2"] == pytest.approx(ref["ssr2"], rel=1e-8)


# --------------------------------------------------------------------------- #
# CUSUM parameter-stability test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", SPEC["cusum"], ids=lambda c: c["name"])
def test_cusum(case):
    y = np.asarray(case["y"], float)
    x = _design(case)
    r = tsecon.cusum_test(y, x)
    ref = case["cusum"]
    assert r["sigma"] == pytest.approx(ref["sigma"], rel=1e-8)
    np.testing.assert_allclose(r["path"], ref["path"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(r["bound_upper"], ref["bound_upper"], rtol=1e-8)
    np.testing.assert_allclose(r["bound_lower"], ref["bound_lower"], rtol=1e-8)


# --------------------------------------------------------------------------- #
# AFNS yield adjustment (closed form)
# --------------------------------------------------------------------------- #
AFNS = _load("afns.json")


@pytest.mark.parametrize(
    "case", AFNS["cases"], ids=[f"afns{i}" for i in range(len(AFNS["cases"]))]
)
def test_afns_adjustment(case):
    maturities = np.asarray(case["maturities"], float)
    sigma = np.asarray(case["sigma_diag"], float)
    out = tsecon.afns_adjustment(maturities, sigma, decay=case["lambda"])
    np.testing.assert_allclose(out, case["adjustment"], rtol=1e-9, atol=1e-14)
    # The adjustment is a downward level shift that deepens with maturity.
    assert np.all(out <= 0.0)
    assert np.all(np.diff(out) <= 1e-12)


def test_afns_adjustment_requires_three_factor_vols():
    with pytest.raises(ValueError, match="3 elements"):
        tsecon.afns_adjustment(np.array([1.0, 2.0]), np.array([0.01, 0.01]), decay=0.5)


# --------------------------------------------------------------------------- #
# DSGE: Cagan money-demand saddle path (structural)
# --------------------------------------------------------------------------- #
def test_dsge_cagan_saddle_path():
    # Cagan model  m_t = a * E_t m_{t+1} + rho * x_t  with a = 0.7, rho AR(1) = 0.6.
    # Written as  A E_t y_{t+1} = B y_t + C eps_t with y = (x, m), x predetermined.
    a, rho = 0.7, 0.6
    A = np.array([[1.0, 0.0], [0.0, a]])
    B = np.array([[rho, 0.0], [-1.0, 1.0]])
    C = np.array([[1.0], [0.0]])

    sol = tsecon.dsge_solve(A, B, C, n_predetermined=1)

    assert "unique stable solution" in sol["verdict"]
    # Policy loading on m to x:  m_t = x_t / (1 - a*rho) = 1.4286 x_t.
    g = np.asarray(sol["g"], float)
    assert g[0, 0] == pytest.approx(1.0 / (1.0 - a * rho), rel=1e-9)
    # Eigenvalue moduli: the stable root rho = 0.6 and the unstable root 1/a = 2.0.
    moduli = np.sort(np.asarray(sol["eigenvalue_moduli"], float))
    np.testing.assert_allclose(moduli, [rho, 1.0 / a], rtol=1e-9)


def test_dsge_singular_lead_matrix_is_reported():
    # A singular lead matrix A is a modeling error the solver must surface, not crash on.
    A = np.zeros((2, 2))
    B = np.eye(2)
    C = np.array([[1.0], [0.0]])
    with pytest.raises(Exception, match="(?i)singular|lead"):
        tsecon.dsge_solve(A, B, C, n_predetermined=1)
