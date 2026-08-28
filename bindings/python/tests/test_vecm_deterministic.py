"""Field item 12 (correctness trap): vecm's deterministic cases vs johansen.

The defect: ``vecm`` silently fit the no-deterministic case (statsmodels
``deterministic="n"``) while ``johansen`` documents and assumes the
unrestricted constant (``coint_johansen`` det_order=0) — a caller reading the
two against each other on drifting log levels got cointegrating vectors a
cosine of ~0.57 apart with no warning. The fix: ``vecm`` accepts
``deterministic="n"|"co"`` (default "n", unchanged), both docstrings name
their case and cross-reference each other, and ``deterministic="co"``
reconciles ``vecm`` with ``johansen`` exactly.

Goldens: ``fixtures/vecm_deterministic.json`` (statsmodels VECM under both
cases + coint_johansen on the same seeded drifting data).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
VD = json.loads((FIX / "vecm_deterministic.json").read_text())
# data is stored series-major (k lists of length T); transpose to T x k.
DATA = np.array(VD["data"]).T
K_AR_DIFF = VD["k_ar_diff"]
RANK = VD["coint_rank"]


def cosine(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_vecm_default_is_deterministic_n():
    """The default (no argument) is exactly deterministic="n" — the
    documented statsmodels case — so existing callers see no change."""
    r_default = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK)
    r_n = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK, deterministic="n")
    np.testing.assert_array_equal(r_default["beta"], r_n["beta"])
    np.testing.assert_array_equal(r_default["alpha"], r_n["alpha"])
    assert r_default["llf"] == r_n["llf"]
    # And it matches the statsmodels deterministic="n" golden.
    fx = VD["vecm_n"]
    np.testing.assert_allclose(r_default["alpha"], fx["alpha"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r_default["beta"], fx["beta"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r_default["gamma"], fx["gamma"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r_default["sigma_u"], fx["sigma_u"], rtol=1e-6, atol=1e-10)
    assert r_default["llf"] == pytest.approx(fx["llf"], rel=1e-6)
    # No deterministic terms -> det_coef has zero columns.
    assert np.asarray(r_default["det_coef"]).size == 0


def test_vecm_co_matches_statsmodels():
    """deterministic="co" (unrestricted constant) matches statsmodels
    VECM(..., deterministic="co") — alpha, beta, gamma, det_coef, sigma_u,
    llf — at the tolerance the existing vecm goldens use."""
    r = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK, deterministic="co")
    fx = VD["vecm_co"]
    np.testing.assert_allclose(r["alpha"], fx["alpha"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["beta"], fx["beta"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["gamma"], fx["gamma"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["det_coef"], fx["det_coef"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["sigma_u"], fx["sigma_u"], rtol=1e-6, atol=1e-10)
    assert r["llf"] == pytest.approx(fx["llf"], rel=1e-6)
    assert np.asarray(r["det_coef"]).shape == (3, 1)


def test_vecm_co_reconciles_with_johansen_and_n_diverges():
    """The reporter's scenario as a regression artifact: on seeded drifting
    log levels, vecm(deterministic="co") spans exactly the cointegrating
    space johansen (det_order=0, unrestricted constant) tests — cosine ~1 —
    while the deterministic="n" default diverges (cosine ~0.63 on this
    draw; the reporter measured ~0.57 on theirs). The divergence itself is
    pinned against the statsmodels-computed fixture value, so this is
    documented behavior, not an accident."""
    joh = tsecon.johansen(DATA, k_ar_diff=K_AR_DIFF)
    r_co = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK, deterministic="co")
    r_n = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK)

    # johansen's first eigenvector, normalized as the VECM normalizes beta.
    evec = np.asarray(joh["evec"])
    beta_joh = evec[:, 0] / evec[0, 0]
    beta_co = np.asarray(r_co["beta"])[:, 0]
    beta_n = np.asarray(r_n["beta"])[:, 0]

    # Matching cases reconcile: cosine ~1.
    assert abs(cosine(beta_co, beta_joh)) > 1 - 1e-10
    # Mismatched cases diverge, exactly as the fixture pins.
    cos_n_co = cosine(beta_n, beta_co)
    assert cos_n_co == pytest.approx(VD["beta_cosine_n_co"], rel=1e-6)
    assert cos_n_co < 0.8  # visibly different cointegrating vectors

    # And johansen's eigenvalues are the ones the "co" reduced-rank step
    # maximizes over (same eigenproblem), pinned against statsmodels.
    np.testing.assert_allclose(joh["eig"], VD["johansen"]["eig"], atol=1e-8)


def test_johansen_evec_matches_statsmodels_up_to_sign():
    """The newly exposed evec matches statsmodels coint_johansen's (both
    S_11-orthonormal; eigensolvers pick column signs arbitrarily)."""
    joh = tsecon.johansen(DATA, k_ar_diff=K_AR_DIFF)
    evec = np.asarray(joh["evec"])
    evec_fx = np.array(VD["johansen"]["evec"])
    assert evec.shape == evec_fx.shape
    for j in range(evec.shape[1]):
        sign = 1.0 if evec[:, j] @ evec_fx[:, j] >= 0 else -1.0
        np.testing.assert_allclose(sign * evec[:, j], evec_fx[:, j], rtol=1e-6, atol=1e-8)


def test_vecm_unknown_deterministic_rejected():
    """Unsupported cases are refused with an error naming what exists."""
    with pytest.raises(ValueError, match=r'"n".*"co"|unknown deterministic'):
        tsecon.vecm(DATA, k_ar_diff=1, coint_rank=1, deterministic="ci")
    with pytest.raises(ValueError):
        tsecon.vecm(DATA, k_ar_diff=1, coint_rank=1, deterministic="colo")


def test_docstrings_name_the_deterministic_cases():
    """The docstring floor: vecm names its default case and johansen's
    convention; johansen names its constant and points at "co"."""
    vdoc = tsecon.vecm.__doc__
    jdoc = tsecon.johansen.__doc__
    assert '"n"' in vdoc and '"co"' in vdoc
    assert "deterministic" in vdoc
    assert "johansen" in vdoc  # cross-reference
    assert "det_order=0" in jdoc or "det_order = 0" in jdoc
    assert "unrestricted constant" in jdoc.lower()
    assert 'deterministic="co"' in jdoc  # cross-reference back to vecm
