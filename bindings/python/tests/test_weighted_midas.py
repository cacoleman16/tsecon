"""Property and self-consistency tests for the weighted-MIDAS binding.

`weighted_midas` fits the parsimonious MIDAS regression
``y_t = alpha + beta * sum_k w_k(psi) x_{t,k}`` by nonlinear least squares
(Ghysels, Santa-Clara & Valkanov 2004; Ghysels, Sinko & Valkanov 2007). The
weights ``w_k`` are normalized to sum to one, so ``beta`` is the aggregate
slope on a proper weighted average and is directly comparable to the sum of
the unrestricted U-MIDAS lag coefficients.

No golden fit exists in ``fixtures/midas.json`` (it stores the U-MIDAS OLS and
raw weight goldens only), so these tests combine a simulated exp-Almon DGP
(recovery of sign / scale / weight shape) with closed-form self-consistency
checks on the golden design.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
MIDAS = json.loads((FIX / "midas.json").read_text())


def test_weighted_midas_recovers_exp_almon_dgp():
    """Simulated exp-Almon DGP: the NLS fit recovers the intercept, the
    aggregate slope (sign and scale), and the generating weight profile."""
    rng = np.random.default_rng(0)
    K = MIDAS["K"]  # 6
    # The generating weights are the golden exp-Almon(0.1, -0.05, K=6) profile.
    w_true = np.asarray(MIDAS["weight_goldens"]["exp_almon_0.1_-0.05_K6"])
    assert len(w_true) == K

    nobs = 300
    X = rng.standard_normal((nobs, K))          # (nobs, K) high-frequency lags
    alpha_true, beta_true = 1.5, 2.0
    agg = X @ w_true                             # weighted aggregate regressor
    y = alpha_true + beta_true * agg + 0.05 * rng.standard_normal(nobs)

    r = tsecon.weighted_midas(y, X, scheme="exp_almon")

    w = np.asarray(r["weights"])
    assert w.shape == (K,)
    assert abs(w.sum() - 1.0) < 1e-8            # weights normalized to sum 1
    assert (w >= 0.0).all()                     # exp-Almon weights nonnegative
    np.testing.assert_allclose(w, w_true, atol=0.05)   # recovers the shape

    assert r["slope"] > 0.0                                 # sign
    assert r["slope"] == pytest.approx(beta_true, rel=0.1)  # scale
    assert r["intercept"] == pytest.approx(alpha_true, abs=0.1)
    assert r["rsquared"] > 0.98                             # near-perfect fit
    assert np.asarray(r["weight_params"]).shape == (2,)
    assert r["scheme"] == "exp_almon"

    fitted = np.asarray(r["fitted"])
    resid = np.asarray(r["residuals"])
    assert fitted.shape == (nobs,) and resid.shape == (nobs,)
    np.testing.assert_allclose(fitted + resid, y, atol=1e-9)  # y decomposition


def test_weighted_midas_fixture_self_consistent():
    """On the golden design the fit is internally coherent and is a proper
    restriction of the unrestricted U-MIDAS OLS.

    The fixture's U-MIDAS lag coefficients are all positive and gently
    decaying, so an exp-Almon weighted MIDAS fits it almost as well as the
    unrestricted regression, and its aggregate slope reproduces the sum of the
    unrestricted lag coefficients.
    """
    y = np.array(MIDAS["y"])
    X = np.array(MIDAS["X_stacked"]).T          # (nobs, K)
    nobs, K = X.shape

    r = tsecon.weighted_midas(y, X, scheme="exp_almon")

    w = np.asarray(r["weights"])
    assert w.shape == (K,)
    assert abs(w.sum() - 1.0) < 1e-8
    assert (w >= -1e-12).all()

    fitted = np.asarray(r["fitted"])
    resid = np.asarray(r["residuals"])
    assert fitted.shape == (nobs,) and resid.shape == (nobs,)
    np.testing.assert_allclose(fitted + resid, y, atol=1e-9)     # decomposition
    assert r["ssr"] == pytest.approx(float(resid @ resid), rel=1e-10)

    ybar = y.mean()
    tss = float(((y - ybar) ** 2).sum())
    assert r["rsquared"] == pytest.approx(1.0 - r["ssr"] / tss, rel=1e-9)

    # A restriction of U-MIDAS: its R^2 cannot exceed the unrestricted OLS R^2,
    # and (given the well-behaved decaying lag profile) is close to it.
    umidas_r2 = MIDAS["umidas_ols"]["rsquared"]
    assert 0.9 < r["rsquared"] <= umidas_r2 + 1e-6

    # Weights sum to 1, so the aggregate slope reproduces the sum of the
    # unrestricted U-MIDAS lag coefficients (params[0] is the intercept).
    lag_sum = float(np.sum(MIDAS["umidas_ols"]["params"][1:]))
    assert r["slope"] == pytest.approx(lag_sum, rel=0.25)


def test_weighted_midas_beta_scheme_and_start_override():
    """The Beta scheme and an explicit ``weight_start`` both run and return
    normalized, nonnegative weights with finite diagnostics."""
    y = np.array(MIDAS["y"])
    X = np.array(MIDAS["X_stacked"]).T
    K = X.shape[1]

    r = tsecon.weighted_midas(y, X, scheme="beta", weight_start=(2.0, 3.0))

    w = np.asarray(r["weights"])
    assert w.shape == (K,)
    assert abs(w.sum() - 1.0) < 1e-8
    assert (w >= -1e-12).all()                  # Beta weights nonnegative
    assert r["scheme"] == "beta"
    assert np.isfinite(r["ssr"]) and np.isfinite(r["rsquared"])
    assert np.asarray(r["weight_params"]).shape == (2,)
    assert (np.asarray(r["weight_params"]) > 0.0).all()   # Beta shapes positive


def test_weighted_midas_rejects_unknown_scheme():
    """An unrecognized weight scheme is a ValueError from the binding."""
    y = np.array(MIDAS["y"])
    X = np.array(MIDAS["X_stacked"]).T
    with pytest.raises(ValueError):
        tsecon.weighted_midas(y, X, scheme="almon")
