"""Nonlinear GMM (Hansen 1982) driver bound with a Python moment callback.

Mirrors the crate property test (crates/tsecon-gmm/tests/properties.rs::
nonlinear_gmm_recovers_mean_and_variance): the exactly identified
mean/variance moment system

    E[y - mu] = 0,   E[(y - mu)^2 - s2] = 0

has the closed-form solution (sample mean, biased 1/n variance), which the
derivative-free Nelder-Mead GMM driver must recover to ~1e-4 regardless of
the data. The moment function is written in pure Python here and called back
into from Rust.
"""
import numpy as np
import pytest
import tsecon


def _mean_var_moments(y):
    """theta -> n-by-2 matrix of per-observation moment contributions for y."""

    def moments(theta):
        mu = float(theta[0])
        s2 = float(theta[1])
        r = y - mu
        # list-of-lists (rows = observations, cols = the 2 moments)
        return np.column_stack([r, r * r - s2]).tolist()

    return moments


def test_nonlinear_gmm_recovers_mean_and_variance():
    rng = np.random.default_rng(90210)
    y = 2.0 + 1.5 * rng.standard_normal(500)

    fit = tsecon.gmm_nonlinear(_mean_var_moments(y), [0.0, 1.0])

    mean = y.mean()
    biased_var = y.var()  # numpy divides by n -> the biased (MLE) variance

    assert fit["converged"]
    assert fit["nmoments"] == 2
    assert fit["nparams"] == 2
    np.testing.assert_allclose(fit["params"], [mean, biased_var], atol=1e-4)
    # Exactly identified => the sample moments are driven to (near) zero.
    np.testing.assert_allclose(fit["gbar"], [0.0, 0.0], atol=1e-4)
    assert fit["objective"] < 1e-6
    assert fit["iterations"] >= 1
    assert fit["fevals"] >= 1


def test_nonlinear_gmm_explicit_identity_weight_matches_default():
    rng = np.random.default_rng(2024)
    y = -1.0 + 0.7 * rng.standard_normal(300)

    moments = _mean_var_moments(y)
    default_fit = tsecon.gmm_nonlinear(moments, [0.0, 1.0])
    # Flattened 2x2 identity (row-major) exercises the optional weight arg;
    # against the identity it must reproduce the default (identity) result.
    weighted_fit = tsecon.gmm_nonlinear(
        moments, [0.0, 1.0], weight=[1.0, 0.0, 0.0, 1.0]
    )

    np.testing.assert_allclose(
        weighted_fit["params"], default_fit["params"], atol=1e-6
    )
    np.testing.assert_allclose(
        default_fit["params"], [y.mean(), y.var()], atol=1e-4
    )


def test_nonlinear_gmm_propagates_callback_error():
    # A Python exception raised inside the moment function must surface as a
    # Python exception (not be swallowed by a crate-level shape error).
    rng = np.random.default_rng(7)
    y = rng.standard_normal(100)  # noqa: F841 - referenced by the closure intent

    def bad_moments(theta):
        raise ValueError("boom from the Python moment function")

    with pytest.raises(ValueError, match="boom from the Python moment function"):
        tsecon.gmm_nonlinear(bad_moments, [0.0, 1.0])
