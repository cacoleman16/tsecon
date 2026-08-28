"""CCC forecasts + the in-sample covariance surface, and the configurable
univariate first stage (0.5 binding build-out).

Two adversarially-verified gaps closed here:

* ``ccc_garch`` gains ``forecast_horizon=`` (``covariance_forecast`` /
  ``variance_forecast`` — analytic and *exact* at every horizon, Bollerslev
  1990), and both ``ccc_garch`` and ``dcc_garch`` now return the in-sample
  conditional covariance path ``covariance`` (``H_t = D_t R_t D_t``,
  ``(T, k, k)``) plus the per-series conditional variance path ``sigma2``
  (``(T, k)``) — previously computed in Rust but never exposed.
* the univariate first stage of ``ccc_garch``/``dcc_garch``/``dcc_test`` is
  no longer hard-wired to a zero-mean Normal GARCH(1,1): the same
  ``vol``/``mean``/``p``/``o``/``q`` knobs as ``garch_fit`` thread through,
  plus ``univariate_dist`` — deliberately NOT named ``dist``, because
  ``dcc_garch``'s existing ``dist=`` selects the *second-stage correlation
  likelihood*, a different object.

Default calls remain bit-identical (asserted in-process below; also verified
bit-identical across builds against the 0.5.0 baseline with a 30-array
``f64``-bytes snapshot when this surface landed).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
MG = json.loads((FIX / "mgarch.json").read_text())
RETURNS = np.array(MG["returns"]).T              # (T, k) = (2400, 3)
SUB = np.ascontiguousarray(RETURNS[:900, :2])    # cheap bivariate slice


def _drd(sigma2, corr):
    """H = D R D with D = diag(sqrt(sigma2)), matching the Rust arithmetic
    (d[i] * r[i][j] * d[j], left to right) so equality can be exact."""
    d = np.sqrt(np.asarray(sigma2, dtype=float))
    r = np.asarray(corr, dtype=float)
    if r.ndim == 2:  # constant correlation broadcast over t
        return d[:, :, None] * r[None, :, :] * d[:, None, :]
    return d[:, :, None] * r * d[:, None, :]


# --------------------------------------------------------------------------
# Item A: the in-sample covariance surface
# --------------------------------------------------------------------------

def test_ccc_default_keys_and_shapes():
    r = tsecon.ccc_garch(SUB)
    assert set(r.keys()) == {"correlation", "loglik", "sigma2", "covariance"}
    T, k = SUB.shape
    assert np.asarray(r["sigma2"]).shape == (T, k)
    assert np.asarray(r["covariance"]).shape == (T, k, k)
    assert (np.asarray(r["sigma2"]) > 0).all()
    # No forecast keys unless asked (matching dcc_garch's opt-in contract).
    assert "covariance_forecast" not in r and "variance_forecast" not in r


def test_sigma2_is_the_perseries_garch_filter_and_shared_across_fits():
    """``sigma2[:, i]`` IS series i's own ``garch_fit`` conditional-variance
    filter (same spec, same data, same timing), and CCC and DCC expose the
    identical univariate stage bitwise."""
    c = tsecon.ccc_garch(SUB)
    d = tsecon.dcc_garch(SUB)
    sc = np.asarray(c["sigma2"])
    sd = np.asarray(d["sigma2"])
    np.testing.assert_array_equal(sc, sd)  # same stage computation, bitwise
    for i in range(SUB.shape[1]):
        cv = np.asarray(
            tsecon.garch_fit(np.ascontiguousarray(SUB[:, i]))["conditional_volatility"]
        )
        # conditional_volatility is sqrt(sigma2) at source; squaring costs
        # one rounding, hence the tight-but-not-bitwise tolerance.
        np.testing.assert_allclose(sc[:, i], cv**2, rtol=1e-12)


def test_ccc_covariance_identity_h_eq_drd():
    """The returned path satisfies H_t = D_t R D_t exactly (same arithmetic
    as the Rust side, so equality is bitwise)."""
    r = tsecon.ccc_garch(SUB)
    H = np.asarray(r["covariance"])
    Hb = _drd(r["sigma2"], r["correlation"])
    np.testing.assert_array_equal(H, Hb)


def test_dcc_covariance_identity_vs_returned_correlation_path():
    """DCC's H_t = D_t R_t D_t against the *returned* correlation path —
    the two surfaces must be one model, not two computations."""
    r = tsecon.dcc_garch(SUB)
    H = np.asarray(r["covariance"])
    T, k = SUB.shape
    assert H.shape == (T, k, k)
    Hb = _drd(r["sigma2"], np.asarray(r["correlation"]))
    np.testing.assert_array_equal(H, Hb)
    # Diagonal of H_t is the per-series conditional variance (R_t has unit
    # diagonal up to normalization rounding).
    np.testing.assert_allclose(
        np.diagonal(H, axis1=1, axis2=2), np.asarray(r["sigma2"]), rtol=1e-12
    )


def test_covariance_timing_last_insample_is_not_the_forecast():
    """Timing pin (the dcc_garch R_t convention, stated identically for H_t):
    ``covariance[t]`` conditions on information through t-1, so the last
    in-sample H_T and the one-step-ahead forecast H_{T+1} (which also uses
    the final residual) must differ — for CCC via D, for DCC via D and R."""
    c = tsecon.ccc_garch(SUB, forecast_horizon=1)
    assert not np.allclose(
        np.asarray(c["covariance"])[-1], np.asarray(c["covariance_forecast"])[0],
        atol=1e-12,
    )
    d = tsecon.dcc_garch(SUB, forecast_horizon=1)
    assert not np.allclose(
        np.asarray(d["covariance"])[-1], np.asarray(d["covariance_forecast"])[0],
        atol=1e-12,
    )
    # DCC's H_0 pairs the documented R_0 = corr(qbar) with sigma2[0].
    q = np.asarray(d["qbar"])
    dg = np.sqrt(np.diag(q))
    r0 = q / np.outer(dg, dg)
    np.fill_diagonal(r0, 1.0)
    np.testing.assert_allclose(
        np.asarray(d["covariance"])[0],
        _drd(np.asarray(d["sigma2"])[:1], r0)[0],
        rtol=1e-10,
    )


# --------------------------------------------------------------------------
# Item A: the CCC forecast surface
# --------------------------------------------------------------------------

def test_ccc_forecast_shapes_h1_exactness_and_analytic_identity():
    horizon = 8
    r = tsecon.ccc_garch(SUB, forecast_horizon=horizon)
    T, k = SUB.shape
    Hf = np.asarray(r["covariance_forecast"])
    Vf = np.asarray(r["variance_forecast"])
    assert Hf.shape == (horizon, k, k)
    assert Vf.shape == (horizon, k)
    assert (Vf > 0).all()

    # h = 1 equals the exact one-step: a horizon-1 call returns bitwise the
    # same first entry (nothing multi-step leaks backward into h = 1).
    r1 = tsecon.ccc_garch(SUB, forecast_horizon=1)
    np.testing.assert_array_equal(np.asarray(r1["covariance_forecast"])[0], Hf[0])
    np.testing.assert_array_equal(np.asarray(r1["variance_forecast"])[0], Vf[0])

    # The CCC forecast is analytic at EVERY horizon: H_{T+m} = D R D exactly,
    # with D from the per-series analytic variance forecasts (no h >= 2
    # approximation exists for CCC — R never moves).
    np.testing.assert_array_equal(Hf, _drd(Vf, r["correlation"]))

    # And those variance forecasts ARE each series' own garch_fit forecast
    # (identical code path, so identical bits).
    for i in range(k):
        vf_i = np.asarray(
            tsecon.garch_fit(
                np.ascontiguousarray(SUB[:, i]), forecast_horizon=horizon
            )["variance_forecast"]
        )
        np.testing.assert_array_equal(Vf[:, i], vf_i)


def test_ccc_forecast_horizon_zero_returns_no_forecast_keys():
    r = tsecon.ccc_garch(SUB, forecast_horizon=0)
    assert "covariance_forecast" not in r
    assert "variance_forecast" not in r


# --------------------------------------------------------------------------
# Item B: the univariate first stage threads through
# --------------------------------------------------------------------------

def test_default_call_equals_explicit_univariate_defaults_bitwise():
    """The new kwargs at their defaults are the SAME computation — bitwise —
    for all three entry points (the dispatch-does-not-perturb guarantee).

    `o` is passed as its sentinel default None (o=1 under vol="garch" is
    refused since the audit-10 fix — the garch_fit guard now covers the
    multivariate siblings; see test_o_refused_under_symmetric_garch)."""
    kw = dict(vol="garch", mean="zero", univariate_dist="normal", p=1, o=None, q=1)

    c0, c1 = tsecon.ccc_garch(SUB), tsecon.ccc_garch(SUB, **kw)
    assert c0["loglik"] == c1["loglik"]
    np.testing.assert_array_equal(np.asarray(c0["correlation"]), np.asarray(c1["correlation"]))
    np.testing.assert_array_equal(np.asarray(c0["sigma2"]), np.asarray(c1["sigma2"]))
    np.testing.assert_array_equal(np.asarray(c0["covariance"]), np.asarray(c1["covariance"]))

    d0, d1 = tsecon.dcc_garch(SUB), tsecon.dcc_garch(SUB, **kw)
    assert d0["a"] == d1["a"] and d0["b"] == d1["b"] and d0["loglik"] == d1["loglik"]
    np.testing.assert_array_equal(np.asarray(d0["correlation"]), np.asarray(d1["correlation"]))
    np.testing.assert_array_equal(np.asarray(d0["covariance"]), np.asarray(d1["covariance"]))

    t0, t1 = tsecon.dcc_test(SUB, lags=3), tsecon.dcc_test(SUB, lags=3, **kw)
    assert t0["stat"] == t1["stat"] and t0["p_value"] == t1["p_value"]


def test_gjr_univariate_stage_runs_and_changes_the_fitted_variances():
    """A GJR first stage is a different model: the fitted variance paths
    must move, and they must be exactly the per-series GJR garch_fit filter
    (proof the spec threads through rather than merely toggling a flag)."""
    base = tsecon.dcc_garch(SUB)
    gjr = tsecon.dcc_garch(SUB, vol="gjr")
    s_base = np.asarray(base["sigma2"])
    s_gjr = np.asarray(gjr["sigma2"])
    assert s_base.shape == s_gjr.shape
    assert not np.allclose(s_base, s_gjr, rtol=1e-6)
    assert gjr["loglik"] != base["loglik"]
    for i in range(SUB.shape[1]):
        cv = np.asarray(
            tsecon.garch_fit(np.ascontiguousarray(SUB[:, i]), vol="gjr")[
                "conditional_volatility"
            ]
        )
        np.testing.assert_allclose(s_gjr[:, i], cv**2, rtol=1e-12)


def test_student_t_univariate_stage_and_constant_mean_run():
    r = tsecon.ccc_garch(SUB, univariate_dist="t", mean="constant")
    assert np.isfinite(r["loglik"])
    assert np.asarray(r["covariance"]).shape == (SUB.shape[0], 2, 2)
    t = tsecon.dcc_test(SUB, lags=3, univariate_dist="t")
    assert np.isfinite(t["stat"]) and t["df"] == 4


def test_univariate_and_second_stage_dist_are_independent_knobs():
    """dist= (second stage) and univariate_dist= (first stage) must move
    different parts of the model: changing univariate_dist changes the
    stage (sigma2), changing dist alone does not."""
    base = tsecon.dcc_garch(SUB)
    second = tsecon.dcc_garch(SUB, dist="t")
    first = tsecon.dcc_garch(SUB, univariate_dist="t")
    np.testing.assert_array_equal(
        np.asarray(base["sigma2"]), np.asarray(second["sigma2"])
    )  # second stage cannot touch the univariate filter
    assert not np.allclose(
        np.asarray(base["sigma2"]), np.asarray(first["sigma2"]), rtol=1e-8
    )
    assert "nu" in second and "nu" not in first  # nu is second-stage-only


# --------------------------------------------------------------------------
# Error surfaces (teaching ValueErrors)
# --------------------------------------------------------------------------

def test_unknown_vol_teaches_choices():
    with pytest.raises(ValueError, match=r"garch/gjr/egarch"):
        tsecon.ccc_garch(SUB, vol="garch2")
    with pytest.raises(ValueError, match=r"garch/gjr/egarch"):
        tsecon.dcc_garch(SUB, vol="figarch")


def test_unknown_mean_teaches_choices():
    with pytest.raises(ValueError, match=r"zero/constant"):
        tsecon.dcc_test(SUB, mean="ar")


def test_unknown_univariate_dist_teaches_the_two_dist_kwargs_apart():
    """The predicted confusion: dist= vs univariate_dist=. The error must
    name the kwarg AND explain which stage each one configures."""
    for call in (
        lambda: tsecon.ccc_garch(SUB, univariate_dist="gaussian"),
        lambda: tsecon.dcc_garch(SUB, univariate_dist="gaussian"),
        lambda: tsecon.dcc_test(SUB, univariate_dist="gaussian"),
    ):
        with pytest.raises(ValueError, match=r"univariate_dist") as exc:
            call()
        assert "second-stage" in str(exc.value)


# --------------------------------------------------------------------------
# Docstring honesty (house tripwire: every returned key is named)
# --------------------------------------------------------------------------

def test_ccc_garch_docstring_names_every_returned_key():
    import re

    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.ccc_garch.__doc__ or ""))
    keys = set(tsecon.ccc_garch(SUB, forecast_horizon=2).keys())
    missing = keys - tokens
    assert not missing, f"ccc_garch.__doc__ does not name returned keys: {sorted(missing)}"
    flat = (tsecon.ccc_garch.__doc__ or "").replace("\n", " ")
    # The H_t timing must be stated, identically to dcc_garch's convention.
    assert "TIMING CONVENTION" in flat
    assert "covariance_forecast[0]" in flat


# --------------------------------------------------------------------------
# Audit round 10: the garch_fit `o` guard covers the multivariate siblings
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda **kw: tsecon.ccc_garch(SUB, **kw),
        lambda **kw: tsecon.dcc_garch(SUB, **kw),
        lambda **kw: tsecon.dcc_test(SUB, lags=3, **kw),
    ],
    ids=["ccc_garch", "dcc_garch", "dcc_test"],
)
def test_o_refused_under_symmetric_garch(call):
    """Explicit o > 0 under vol="garch" raises the garch_fit teaching error
    (before the fix it was silently dropped by the spec parser — verified
    bit-identical to the default call, i.e. a complete no-op)."""
    with pytest.raises(ValueError, match=r"o=1 has no effect") as exc:
        call(o=1)
    msg = str(exc.value)
    assert 'vol="gjr"' in msg and "arch_model" in msg  # names the remedy + trap


def test_o_zero_explicit_equals_default_under_garch():
    """o=0 says "no asymmetry term" out loud and stays legal under
    vol="garch" — bit-identical to the sentinel default (garch_fit parity)."""
    a = tsecon.dcc_test(SUB, lags=3)
    b = tsecon.dcc_test(SUB, lags=3, o=0)
    assert a["stat"] == b["stat"] and a["p_value"] == b["p_value"]


def test_o_still_live_under_gjr_first_stage():
    """Where o is documented to act (the asymmetric specs) it must still
    change the fit: GJR(1, o, 1) first stages with o=1 vs o=2 differ."""
    a = tsecon.dcc_test(SUB, lags=3, vol="gjr", o=1)
    b = tsecon.dcc_test(SUB, lags=3, vol="gjr", o=2)
    assert a["stat"] != b["stat"]
    # And the sentinel default under gjr is o=1, exactly as documented.
    d = tsecon.dcc_test(SUB, lags=3, vol="gjr")
    assert d["stat"] == a["stat"] and d["p_value"] == a["p_value"]
