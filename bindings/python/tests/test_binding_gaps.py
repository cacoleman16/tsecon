"""Four adversarially-verified "computed in Rust, never bound" gaps (0.6).

Each section pins one surface that the sweep found fully computed in the
crates but absent from the Python results dict:

* ``dfm_nowcast`` — the fitted model itself (``loadings`` / ``factor_ar`` /
  ``factor_cov`` / ``idiosyncratic`` / ``center`` / ``scale``), on both
  estimation routes, with the exact factor-to-series mapping pinned:
  ``nowcast == center + scale * (loadings @ edge_factor)`` is literally how
  the crate computes the nowcast (``Nowcaster::destandardized_fit``).
* ``bvar_fit`` — the full NIW posterior (``omega_bar`` / ``s_bar`` /
  ``v_bar``), with the documented marginal-coefficient-sd one-liner
  validated against a seeded Monte Carlo of draws from the documented NIW.
* ``var_fit`` — ``resid`` / ``fitted`` / ``nobs`` / ``df_resid``, with the
  reconstruction ``data[lags:] - fitted == resid`` exact (bitwise).
* ``dcc_garch`` — the stage-1 remainder: per-series ``univariate`` GARCH
  results (bit-identical to a direct ``garch_fit`` under the same spec —
  one shared dict builder in the binding), the stacked ``std_residuals``
  that drive the correlation recursion, and the ADCC targeting matrix
  ``nbar``.
"""
import numpy as np
import pytest

import tsecon


# ==========================================================================
# Fix 1: dfm_nowcast returns the model
# ==========================================================================

def _factor_panel(T=200, N=8, r=2, seed=0, idio_sd=0.35):
    """A strongly factor-driven balanced panel: r AR(1) factors, fixed
    loadings, Gaussian idiosyncratic noise of sd ``idio_sd``."""
    rng = np.random.default_rng(seed)
    f = np.zeros((T, r))
    phi = np.array([0.8, 0.5][:r])
    for t in range(1, T):
        f[t] = phi * f[t - 1] + rng.standard_normal(r)
    lam = rng.uniform(0.5, 1.5, size=(N, r)) * np.sign(rng.standard_normal((N, r)))
    x = f @ lam.T + idio_sd * rng.standard_normal((T, N)) + rng.uniform(-2, 2, N)
    return x, f


def test_dfm_two_step_param_shapes():
    x, _ = _factor_panel()
    T, N = x.shape
    res = tsecon.dfm_nowcast(x, n_factors=2, factor_order=2)
    r, p = res["n_factors"], res["factor_order"]
    assert (r, p) == (2, 2)
    assert np.asarray(res["loadings"]).shape == (N, r)
    assert np.asarray(res["factor_ar"]).shape == (r, r * p)
    assert np.asarray(res["factor_cov"]).shape == (r, r)
    assert np.asarray(res["idiosyncratic"]).shape == (N,)
    assert np.asarray(res["center"]).shape == (N,)
    assert np.asarray(res["scale"]).shape == (N,)
    assert (np.asarray(res["idiosyncratic"]) >= 0).all()
    assert (np.asarray(res["scale"]) > 0).all()
    # factor_cov is a symmetric PSD covariance.
    Q = np.asarray(res["factor_cov"])
    np.testing.assert_allclose(Q, Q.T, atol=1e-12)
    assert np.linalg.eigvalsh(Q).min() >= -1e-12
    # center/scale are the training column moments (ddof=0) — the documented
    # standardization the parameters live on.
    np.testing.assert_allclose(res["center"], x.mean(axis=0), rtol=1e-10)
    np.testing.assert_allclose(res["scale"], x.std(axis=0), rtol=1e-10)


def test_dfm_nowcast_is_loadings_dot_edge_factor():
    """The exact factor-to-series mapping, straight from the crate
    (``destandardized_fit``): nowcast_i = center_i + scale_i * (Lambda_i . f).
    Pinned at 1e-10 relative (the binding computes the dot product with the
    same reals; only summation order can differ)."""
    x, _ = _factor_panel(seed=1)
    res = tsecon.dfm_nowcast(x, n_factors=2, factor_order=2)
    L = np.asarray(res["loadings"])
    reconstructed = np.asarray(res["center"]) + np.asarray(res["scale"]) * (
        L @ np.asarray(res["edge_factor"])
    )
    np.testing.assert_allclose(res["nowcast"], reconstructed, rtol=1e-10, atol=1e-12)


def test_dfm_common_component_reconstructs_balanced_fitted_values():
    """center + scale * (smoothed_factors @ loadings.T) is the model's fitted
    value for every cell of the balanced panel. Two pins:

    * the last fitted row IS the nowcast (balanced panel: the training pass
      and the nowcast pass smooth the same data, so the edge factor is the
      last smoothed factor) — 1e-8;
    * the fit explains the panel: per-series R^2 of the reconstruction is
      high for this strongly factor-driven sim (idiosyncratic sd 0.35 vs
      factor component of variance ~1), and the mean squared standardized
      residual is of the idiosyncratic-variance order the model itself
      reports (within a factor of 2 — the Kalman-smoothed factors are not
      the PCA factors the idiosyncratic variances were read from, so
      equality is not expected)."""
    x, _ = _factor_panel(seed=2)
    res = tsecon.dfm_nowcast(x, n_factors=2, factor_order=2)
    F = np.asarray(res["smoothed_factors"])          # (T, r)
    L = np.asarray(res["loadings"])                  # (N, r)
    center, scale = np.asarray(res["center"]), np.asarray(res["scale"])
    fitted = center + scale * (F @ L.T)              # (T, N) levels
    np.testing.assert_allclose(fitted[-1], res["nowcast"], rtol=1e-8)
    # Reconstruction quality, standardized scale.
    z = (x - center) / scale
    resid = z - F @ L.T
    mse = (resid**2).mean(axis=0)
    idio = np.asarray(res["idiosyncratic"])
    r2 = 1.0 - mse / (z**2).mean(axis=0)
    assert (r2 > 0.7).all(), f"common component explains too little: {r2}"
    ratio = mse / idio
    assert (ratio > 0.5).all() and (ratio < 2.0).all(), (
        f"smoothed-factor residual variance far from the model's own "
        f"idiosyncratic estimate: {ratio}"
    )


def test_dfm_mle_returns_same_param_surface():
    """The mle route returns the identical parameter surface, on its
    documented scale: scale is all ones (raw/centred fit), factor_cov is
    fixed to 1 for identification, and the nowcast identity still holds."""
    x, _ = _factor_panel(T=90, N=5, r=1, seed=6)
    res = tsecon.dfm_nowcast(x, n_factors=1, factor_order=1, method="mle")
    N = x.shape[1]
    for key, shape in [
        ("loadings", (N, 1)), ("factor_ar", (1, 1)), ("factor_cov", (1, 1)),
        ("idiosyncratic", (N,)), ("center", (N,)), ("scale", (N,)),
    ]:
        assert np.asarray(res[key]).shape == shape, key
    np.testing.assert_array_equal(res["scale"], np.ones(N))
    np.testing.assert_allclose(np.asarray(res["factor_cov"]), [[1.0]], atol=1e-12)
    np.testing.assert_allclose(res["center"], x.mean(axis=0), rtol=1e-10)
    L = np.asarray(res["loadings"])
    reconstructed = np.asarray(res["center"]) + np.asarray(res["scale"]) * (
        L @ np.asarray(res["edge_factor"])
    )
    np.testing.assert_allclose(res["nowcast"], reconstructed, rtol=1e-10, atol=1e-12)


def test_dfm_nowcast_docstring_names_every_returned_key():
    import re

    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.dfm_nowcast.__doc__ or ""))
    x, _ = _factor_panel(seed=3)
    keys = set(tsecon.dfm_nowcast(x, n_factors=1, factor_order=2).keys())
    missing = keys - tokens
    assert not missing, f"dfm_nowcast.__doc__ does not name returned keys: {sorted(missing)}"


# ==========================================================================
# Fix 2: bvar_fit returns the full NIW posterior
# ==========================================================================

def _var_data(T=150, K=3, seed=7):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.3, 0.2], [0.1, 0.0, 0.4]])[:K, :K]
    y = np.zeros((T, K))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(K)
    return y


def test_bvar_posterior_dimensions_and_dof():
    y = _var_data()
    T, K = y.shape
    p = 2
    r = tsecon.bvar_fit(y, lags=p)
    k = 1 + p * K
    O = np.asarray(r["omega_bar"])
    S = np.asarray(r["s_bar"])
    assert O.shape == (k, k)
    assert S.shape == (K, K)
    # vbar = v0 + T_eff with v0 = K + 2 and T_eff = T - p (crate convention).
    assert r["v_bar"] == float(K + 2 + (T - p))
    # Both scale matrices are symmetric positive definite.
    np.testing.assert_allclose(O, O.T, atol=1e-12)
    np.testing.assert_allclose(S, S.T, atol=1e-12)
    assert np.linalg.eigvalsh(O).min() > 0
    assert np.linalg.eigvalsh(S).min() > 0
    # sigma_posterior_mean is exactly s_bar / (v_bar - K - 1): the returned
    # pieces reproduce the already-shipped summary.
    np.testing.assert_allclose(
        np.asarray(r["sigma_posterior_mean"]), S / (r["v_bar"] - K - 1), rtol=1e-12
    )


def test_bvar_marginal_sd_formula_matches_niw_monte_carlo():
    """The documented one-liner

        sd = np.sqrt(np.outer(np.diag(omega_bar), np.diag(s_bar))
                     / (v_bar - K - 1))

    is validated against a seeded Monte Carlo from the documented NIW:
    Sigma ~ InvWishart(s_bar, v_bar) (scipy.stats.invwishart), then
    B | Sigma matrix-normal with row covariance omega_bar and column
    covariance Sigma — i.e. vec(B) ~ N(vec(Bbar), Sigma (x) Obar) with
    column-stacked vec, the binding's documented Kronecker order.

    n = 40_000 draws, seed 0; empirical per-coefficient sd within 5%
    relative of the formula (the MC se of an sd with ~t(v_bar - K + 1)
    tails at this n is well under 1%), empirical mean within 5% of an sd
    of Bbar."""
    from scipy.stats import invwishart

    y = _var_data(T=120, K=2, seed=11)
    K, p = 2, 1
    r = tsecon.bvar_fit(y, lags=p)
    O = np.asarray(r["omega_bar"])
    S = np.asarray(r["s_bar"])
    vbar = r["v_bar"]
    Bbar = np.asarray(r["posterior_mean_coefs"])          # (k, K)
    k = 1 + p * K
    sd_formula = np.sqrt(np.outer(np.diag(O), np.diag(S)) / (vbar - K - 1))

    rng = np.random.default_rng(0)
    n = 40_000
    sigmas = invwishart.rvs(df=vbar, scale=S, size=n, random_state=rng)
    Lo = np.linalg.cholesky(O)                            # (k, k)
    Ls = np.linalg.cholesky(sigmas)                       # (n, K, K)
    Z = rng.standard_normal((n, k, K))
    draws = Bbar + Lo @ Z @ np.swapaxes(Ls, 1, 2)         # (n, k, K)

    sd_mc = draws.std(axis=0, ddof=1)
    np.testing.assert_allclose(sd_mc, sd_formula, rtol=0.05)
    np.testing.assert_allclose(draws.mean(axis=0), Bbar, atol=0.05 * sd_formula.max())


def test_bvar_fit_docstring_names_every_returned_key():
    import re

    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.bvar_fit.__doc__ or ""))
    keys = set(tsecon.bvar_fit(_var_data(), lags=2).keys())
    missing = keys - tokens
    assert not missing, f"bvar_fit.__doc__ does not name returned keys: {sorted(missing)}"
    # The Kronecker order and the sd one-liner must be stated, not implied.
    flat = (tsecon.bvar_fit.__doc__ or "").replace("\n", " ")
    assert "np.kron(sigma, omega_bar)" in flat
    assert "np.outer(np.diag(omega_bar), np.diag(s_bar))" in flat


# ==========================================================================
# Fix 3: var_fit returns its residuals
# ==========================================================================

def test_var_resid_fitted_shapes_and_exact_reconstruction():
    y = _var_data(T=200, K=3, seed=5)
    lags = 2
    r = tsecon.var_fit(y, lags=lags)
    T_eff = y.shape[0] - lags
    resid = np.asarray(r["resid"])
    fitted = np.asarray(r["fitted"])
    assert resid.shape == (T_eff, 3)
    assert fitted.shape == (T_eff, 3)
    assert r["nobs"] == T_eff
    assert r["df_resid"] == T_eff - (1 + 3 * lags)
    # fitted is DEFINED as y[lags:] - resid (one IEEE subtraction per cell,
    # both here and in the binding), so this direction is exact by
    # construction ...
    np.testing.assert_array_equal(fitted, y[lags:] - resid)
    # ... and on this (macro-scaled) data the round trip is exact too:
    # y[lags:] - fitted == resid bitwise.
    np.testing.assert_array_equal(y[lags:] - fitted, resid)
    # Residuals average to ~0 per equation (the regression has a constant).
    np.testing.assert_allclose(resid.mean(axis=0), 0.0, atol=1e-10)
    # sigma_u is exactly resid'resid / df_resid (the crate's own divisor).
    np.testing.assert_allclose(
        np.asarray(r["sigma_u"]), resid.T @ resid / r["df_resid"], rtol=1e-10
    )


def test_var_resid_trend_n_counts():
    y = _var_data(T=150, K=2, seed=9)
    lags = 3
    r = tsecon.var_fit(y, lags=lags, trend="n")
    assert r["nobs"] == 150 - lags
    assert r["df_resid"] == (150 - lags) - 2 * lags   # no intercept regressor
    assert np.asarray(r["resid"]).shape == (150 - lags, 2)


def test_ljung_box_runs_on_returned_residual_column():
    y = _var_data(T=200, K=3, seed=5)
    r = tsecon.var_fit(y, lags=2)
    col = np.asarray(r["resid"])[:, 0]
    lb = tsecon.ljung_box(col, 10)
    assert np.all(np.isfinite(lb["lb_stat"]))
    assert np.all((np.asarray(lb["lb_pvalue"]) >= 0) & (np.asarray(lb["lb_pvalue"]) <= 1))


def test_var_fit_docstring_names_every_returned_key():
    import re

    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.var_fit.__doc__ or ""))
    keys = set(tsecon.var_fit(_var_data(), lags=2).keys())
    missing = keys - tokens
    assert not missing, f"var_fit.__doc__ does not name returned keys: {sorted(missing)}"


# ==========================================================================
# Fix 4: dcc_garch stage-1 remainder
# ==========================================================================

def _sim_garch_panel(T=700, k=2, seed=13):
    """k GARCH(1,1) series with constant cross-correlation 0.5."""
    rng = np.random.default_rng(seed)
    C = np.full((k, k), 0.5) + 0.5 * np.eye(k)
    Lc = np.linalg.cholesky(C)
    z = rng.standard_normal((T, k)) @ Lc.T
    y = np.empty((T, k))
    s2 = np.ones(k)
    for t in range(T):
        y[t] = np.sqrt(s2) * z[t]
        s2 = 0.05 + 0.09 * y[t] ** 2 + 0.87 * s2
    return y


GARCH_FIT_CORE_KEYS = {
    "params", "param_names", "params_named", "loglik", "aic", "bic",
    "se_mle", "se_robust", "se_valid", "boundary", "boundary_note",
    "converged", "conditional_volatility", "std_residuals",
}


def test_dcc_stage1_matches_direct_garch_fit_bitwise():
    """The per-series `univariate` dicts are garch_fit's dict, from the same
    shared builder — same keys, and every array bit-identical to calling
    garch_fit on that column under the same (default zero-mean Normal
    GARCH(1,1)) spec."""
    R = _sim_garch_panel()
    r = tsecon.dcc_garch(R)
    uni = r["univariate"]
    assert isinstance(uni, list) and len(uni) == R.shape[1]
    for i, u in enumerate(uni):
        direct = tsecon.garch_fit(R[:, i])
        assert set(u.keys()) == GARCH_FIT_CORE_KEYS
        assert u["param_names"] == direct["param_names"]
        for key in ["params", "se_mle", "se_robust", "conditional_volatility",
                    "std_residuals"]:
            np.testing.assert_array_equal(u[key], direct[key], err_msg=key)
        assert u["loglik"] == direct["loglik"]
        assert u["aic"] == direct["aic"] and u["bic"] == direct["bic"]
        assert u["params_named"] == direct["params_named"]
        np.testing.assert_array_equal(u["se_valid"], direct["se_valid"])
        np.testing.assert_array_equal(u["boundary"], direct["boundary"])
        assert u["boundary_note"] == direct["boundary_note"]
        assert u["converged"] == direct["converged"]


def test_dcc_stage1_threads_nondefault_univariate_spec():
    """The univariate threading (0.5 field fixes): a non-default stage-1 spec
    reaches both surfaces identically — mean="constant" adds a mu parameter
    and the stage params still match a direct garch_fit bitwise."""
    R = _sim_garch_panel(seed=17)
    r = tsecon.dcc_garch(R, mean="constant")
    for i, u in enumerate(r["univariate"]):
        direct = tsecon.garch_fit(R[:, i], mean="constant")
        assert u["param_names"] == direct["param_names"]
        assert u["param_names"][0] == "mu"
        np.testing.assert_array_equal(u["params"], direct["params"])
        np.testing.assert_array_equal(u["se_robust"], direct["se_robust"])
        assert u["loglik"] == direct["loglik"]


def test_dcc_std_residuals_are_returns_over_sigma_bitwise():
    """z[t][i] = eps_{i,t} / sigma_{i,t}; under the default mean="zero" spec
    eps IS the raw return, and sigma = sqrt(sigma2) (IEEE sqrt is exact), so
    the identity is bitwise. Timing is sigma2's own: entry t divides by the
    variance formed from information through t-1."""
    R = _sim_garch_panel(seed=19)
    r = tsecon.dcc_garch(R)
    z = np.asarray(r["std_residuals"])
    sigma2 = np.asarray(r["sigma2"])
    assert z.shape == R.shape
    assert sigma2.shape == R.shape
    np.testing.assert_array_equal(z, R / np.sqrt(sigma2))
    # And the stacked columns are exactly the per-series std_residuals.
    for i, u in enumerate(r["univariate"]):
        np.testing.assert_array_equal(z[:, i], np.asarray(u["std_residuals"]))


def test_dcc_nbar_only_under_adcc_and_is_the_documented_moment():
    R = _sim_garch_panel(seed=23)
    assert "nbar" not in tsecon.dcc_garch(R)
    assert "nbar" not in tsecon.dcc_garch(R, variant="cdcc")
    r = tsecon.dcc_garch(R, variant="adcc")
    k = R.shape[1]
    nbar = np.asarray(r["nbar"])
    assert nbar.shape == (k, k)
    # Nbar = (1/T) sum_t n_t n_t', n_t = min(z_t, 0), from the returned
    # standardized residuals themselves.
    n = np.minimum(np.asarray(r["std_residuals"]), 0.0)
    np.testing.assert_allclose(nbar, n.T @ n / n.shape[0], atol=1e-12)
