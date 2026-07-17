"""Tests for the penalized-ML path bindings: adaptive LASSO and the
elastic-net regularization path with AIC/BIC selection.

No external golden: adaptive-lasso oracle behavior and the path's
monotone structure are checked analytically on a sparse design.
"""
import numpy as np
import pytest
import tsecon


def _sparse_design(seed=0, n=200, p=8):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [3.0, -2.0, 1.5]  # only the first three are active
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return x, y, beta


def test_adaptive_lasso_recovers_support():
    x, y, beta = _sparse_design()
    fit = tsecon.adaptive_lasso(x, y, alpha=0.05, gamma=1.0)
    coef = np.array(fit["coef"])
    assert coef.shape == (8,)
    # The three true-active coefficients are clearly nonzero...
    assert np.all(np.abs(coef[:3]) > 0.5)
    # ...and the five true-zero coefficients are shrunk to (near) zero.
    assert np.all(np.abs(coef[3:]) < 0.1)
    assert fit["n_iter"] >= 1


def test_lasso_path_structure_and_selection():
    x, y, beta = _sparse_design()
    path = tsecon.lasso_path(x, y, n_lambdas=50)
    lambdas = np.array(path["lambdas"])
    coefs = np.array(path["coefs"])
    df = np.array(path["df"])
    assert coefs.shape == (50, 8)
    # Lambdas descend; the largest one zeros every coefficient.
    assert np.all(np.diff(lambdas) < 0)
    np.testing.assert_allclose(coefs[0], 0.0, atol=1e-8)
    assert df[0] == 0
    # Degrees of freedom (nonzero count) grow (weakly) as lambda shrinks.
    assert df[-1] >= df[0]
    # AIC/BIC selection returns valid path indices and recovers the support.
    assert 0 <= path["aic_best"] < 50
    assert 0 <= path["bic_best"] < 50
    bic_coef = coefs[path["bic_best"]]
    assert np.all(np.abs(bic_coef[:3]) > 0.3)


def test_lasso_path_rejects_ridge_only():
    # l1_ratio must be in (0, 1]; a pure-ridge path (0.0) is rejected.
    x, y, _ = _sparse_design()
    with pytest.raises(ValueError):
        tsecon.lasso_path(x, y, l1_ratio=0.0)
