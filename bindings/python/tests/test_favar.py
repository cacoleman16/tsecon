"""Golden + structural tests for the two-step FAVAR binding
(Bernanke-Boivin-Eliasz 2005, QJE).

The factor-extraction step is externally validated against the PCA golden
in fixtures/favar.json (the same fixture and sign-free comparison used by
test_depth.py::test_factor_model_matches_pca): the FAVAR's step-1 factors
must reproduce the numpy PCA principal components up to a joint sign flip.
The two-step assembly and the loadings-mapped IRFs are validated
structurally (dimensions, finiteness, the recursive policy IRF).
"""
import json
from pathlib import Path

import numpy as np
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
FAV = json.loads((FIXTURES / "favar.json").read_text())


def _panel():
    # fixture stores the standardized panel as a list of columns -> n x N.
    return np.array(FAV["X_standardized"]).T


def test_favar_factors_match_pca():
    xs = _panel()  # n x big_n, already standardized
    n = xs.shape[0]
    # An observed policy series ordered last in the VAR; a deterministic
    # pseudo-random draw keeps the FAVAR assembly well conditioned.
    rng = np.random.default_rng(0)
    policy = rng.standard_normal(n)

    res = tsecon.favar(xs, policy, n_factors=FAV["true_r"], lags=2, trend="c")

    # Step 2 bookkeeping: Y_t = [F_t, R_t] with R (policy) ordered last.
    assert res["n_factors"] == FAV["true_r"]
    assert res["n_endog"] == FAV["true_r"] + 1
    assert res["policy_index"] == FAV["true_r"]  # policy is the last equation

    # Step-1 factors reproduce the numpy PCA components up to sign.
    factors = np.array(res["factors"])
    assert factors.shape == (n, FAV["true_r"])
    np.testing.assert_allclose(np.abs(factors[:, 0]), FAV["pc1_abs"], atol=1e-4)
    np.testing.assert_allclose(np.abs(factors[:, 1]), FAV["pc2_abs"], atol=1e-4)


def test_favar_var_summary_shapes():
    xs = _panel()
    n = xs.shape[0]
    rng = np.random.default_rng(0)
    policy = rng.standard_normal(n)

    lags = 2
    res = tsecon.favar(xs, policy, n_factors=FAV["true_r"], lags=lags, trend="c")

    k = FAV["true_r"] + 1  # endogenous variables in the factor VAR
    # params is (n_trend + k*p) x k with a constant trend -> (1 + k*p) x k.
    params = np.array(res["params"])
    assert params.shape == (1 + k * lags, k)
    # sigma_u is the k x k reduced-form innovation covariance (symmetric PSD).
    sigma = np.array(res["sigma_u"])
    assert sigma.shape == (k, k)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(sigma) > 0)


def test_favar_policy_shock_irfs():
    xs = _panel()
    n, big_n = xs.shape
    rng = np.random.default_rng(0)
    policy = rng.standard_normal(n)

    horizon = 12
    res = tsecon.favar(
        xs, policy, n_factors=FAV["true_r"], lags=2, trend="c",
        horizon=horizon, orth=True,
    )

    # Panel IRF to the recursive policy shock: one row per observed series,
    # horizon + 1 columns, mapped through the observation equation X = L F.
    irf_panel = np.array(res["irf_panel"])
    assert irf_panel.shape == (big_n, horizon + 1)
    assert np.all(np.isfinite(irf_panel))

    # The policy variable's own response, in its own units.
    irf_policy = np.array(res["irf_policy"])
    assert irf_policy.shape == (horizon + 1,)
    assert np.all(np.isfinite(irf_policy))
    # Orthogonalized impact of the policy shock on itself is the Cholesky
    # diagonal for the last equation: strictly positive at h = 0.
    assert irf_policy[0] > 0.0


def test_favar_slow_fast_rotation():
    xs = _panel()
    n, big_n = xs.shape
    rng = np.random.default_rng(0)
    policy = rng.standard_normal(n)

    # Treat the first half of the panel columns as slow-moving series.
    slow = list(range(big_n // 2))
    res = tsecon.favar(
        xs, policy, n_factors=FAV["true_r"], lags=2, trend="c",
        slow_indices=slow,
    )
    assert res["n_factors"] == FAV["true_r"]
    assert res["policy_index"] == FAV["true_r"]
    factors = np.array(res["factors"])
    # Cleaned factors keep the (T x r) shape and stay finite; they differ
    # from the raw PCs (the contemporaneous policy component is purged).
    assert factors.shape == (n, FAV["true_r"])
    assert np.all(np.isfinite(factors))
