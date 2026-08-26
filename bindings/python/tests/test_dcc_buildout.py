"""DCC build-out (0.5.0): variants (cDCC/ADCC), the Student-t second stage,
h-step correlation forecasts, the full in-sample correlation path, and the
Engle-Sheppard constant-correlation test.

VALIDATION STATUS (honest): there is no runnable third-party DCC reference
in this project — the assertions here are (i) a default-path regression pin
(the merge that introduced the variants was verified bit-identical in Rust
via f64::to_bits on every default CCC/DCC output; the pins below hold those
values at 1e-7 so a cross-platform libm ulp cannot fail CI while any real
behavioral change still does), (ii) internal nesting invariants (ADCC ⊇ DCC,
default call == explicit-default call bitwise in the same process), and
(iii) property checks against the literature formulas (Engle 2002; Aielli
2013; Cappiello-Engle-Sheppard 2006; Engle & Sheppard 2001). Monte-Carlo
size/power/recovery numbers live in the volatility model card, measured, not
here (single-realization gates would be flaky).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
MG = json.loads((FIX / "mgarch.json").read_text())
RETURNS = np.array(MG["returns"]).T          # (T, k) = (2400, 3)
SUB = np.ascontiguousarray(RETURNS[:1200, :2])  # cheap bivariate slice (mirrors the Rust tests)


def _corr(m):
    m = np.asarray(m, dtype=float)
    d = np.sqrt(np.diag(m))
    r = m / np.outer(d, d)
    np.fill_diagonal(r, 1.0)
    return r


def _sim_ccc_garch(n, rho=0.5, seed=7):
    """A CCC-GARCH(1,1) null draw (constant correlation rho), seeded."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 2))
    z0 = e[:, 0]
    z1 = rho * e[:, 0] + np.sqrt(1.0 - rho * rho) * e[:, 1]
    x = np.empty((n, 2))
    v0 = v1 = 1.0
    for t in range(n):
        x[t, 0] = np.sqrt(v0) * z0[t]
        x[t, 1] = np.sqrt(v1) * z1[t]
        v0 = 0.05 + 0.10 * x[t, 0] ** 2 + 0.85 * v0
        v1 = 0.04 + 0.08 * x[t, 1] ** 2 + 0.88 * v1
    return x


# --------------------------------------------------------------------------
# Default path: bit-identical + pinned
# --------------------------------------------------------------------------

def test_default_call_equals_explicit_defaults_bitwise():
    """`dcc_garch(x)` and `dcc_garch(x, variant="dcc", dist="normal",
    forecast_horizon=0)` must be the SAME computation — bitwise-equal
    outputs in the same process. This is the dispatch-does-not-perturb
    guarantee for the historical default surface."""
    r0 = tsecon.dcc_garch(RETURNS)
    r1 = tsecon.dcc_garch(RETURNS, variant="dcc", dist="normal", forecast_horizon=0)
    assert r0["a"] == r1["a"] and r0["b"] == r1["b"]
    assert r0["loglik"] == r1["loglik"]
    assert r0["converged"] == r1["converged"]
    np.testing.assert_array_equal(np.asarray(r0["qbar"]), np.asarray(r1["qbar"]))
    np.testing.assert_array_equal(
        np.asarray(r0["correlation_last"]), np.asarray(r1["correlation_last"])
    )
    np.testing.assert_array_equal(
        np.asarray(r0["correlation"]), np.asarray(r1["correlation"])
    )


def test_default_path_pinned_to_premerge_values():
    """Regression pin of the default fixture fit. These decimals were
    recorded from the 0.4.0 build (commit 9d5bed6) and verified bitwise
    unchanged after the variant merge (f64::to_bits comparison in Rust, this
    exact fixture). 1e-7 tolerance: tight enough that any change in the
    optimizer trajectory, starts, or objective arithmetic fails; loose
    enough that a cross-platform libm ulp does not."""
    r = tsecon.dcc_garch(RETURNS)
    assert r["a"] == pytest.approx(3.01296268585330602e-2, rel=1e-7)
    assert r["b"] == pytest.approx(9.39485974196103357e-1, rel=1e-7)
    assert r["loglik"] == pytest.approx(-1.15052815655409941e4, rel=1e-9)
    assert r["converged"] is True
    last = np.asarray(r["correlation_last"])
    assert last[0, 1] == pytest.approx(5.62003666800971136e-1, abs=1e-7)
    assert last[0, 2] == pytest.approx(2.07760662833610582e-1, abs=1e-7)
    assert last[1, 2] == pytest.approx(3.79787972779818850e-1, abs=1e-7)
    # CCC alongside (same pre-merge recording, same guarantee).
    c = tsecon.ccc_garch(RETURNS)
    assert c["loglik"] == pytest.approx(-1.15576347492020177e4, rel=1e-9)


def test_default_keys_are_additive():
    """The 0.4.0 keys survive untouched; every new key is additive and the
    opt-in keys stay absent from the default call. (0.5.0 added
    g/variant/dist/correlation; the 0.5 covariance build-out added the
    in-sample sigma2/covariance paths; the 0.6 stage-1 build-out added the
    per-series univariate results and the stacked std_residuals.)"""
    r = tsecon.dcc_garch(RETURNS)
    old = {"a", "b", "qbar", "loglik", "converged", "correlation_last"}
    assert old <= set(r.keys())
    assert set(r.keys()) == old | {
        "g", "variant", "dist", "correlation", "sigma2", "covariance",
        "univariate", "std_residuals",
    }
    assert r["g"] == 0.0            # structurally zero off-ADCC
    assert r["variant"] == "dcc" and r["dist"] == "normal"
    assert "nu" not in r            # Student-t only
    assert "nbar" not in r          # ADCC only
    assert "correlation_forecast" not in r and "covariance_forecast" not in r


# --------------------------------------------------------------------------
# The in-sample correlation path and its timing convention
# --------------------------------------------------------------------------

def test_correlation_path_shape_and_timing_convention():
    r = tsecon.dcc_garch(RETURNS)
    C = np.asarray(r["correlation"])
    T, k = RETURNS.shape
    assert C.shape == (T, k, k)
    # correlation_last IS the last path entry — same numbers, not a forecast.
    np.testing.assert_array_equal(C[-1], np.asarray(r["correlation_last"]))
    # Filter convention: R_0 conditions on no data (Q_0 = Qbar), so
    # C[0] = corr(Qbar) exactly (the recursion has consumed no residual yet).
    np.testing.assert_allclose(C[0], _corr(r["qbar"]), atol=1e-12)
    # Every R_t is a correlation matrix: unit diagonal, symmetric, PD
    # (spot-checked on a stride; the Rust suite checks the full path).
    for t in range(0, T, 97):
        np.testing.assert_allclose(np.diag(C[t]), 1.0, atol=1e-12)
        np.testing.assert_allclose(C[t], C[t].T, atol=1e-12)
        assert np.linalg.eigvalsh(C[t]).min() > 0


# --------------------------------------------------------------------------
# Forecast surface
# --------------------------------------------------------------------------

def test_forecast_shapes_h1_exactness_and_convergence():
    horizon = 100
    r = tsecon.dcc_garch(RETURNS, forecast_horizon=horizon)
    T, k = RETURNS.shape
    Rf = np.asarray(r["correlation_forecast"])
    Hf = np.asarray(r["covariance_forecast"])
    Vf = np.asarray(r["variance_forecast"])
    assert Rf.shape == (horizon, k, k)
    assert Hf.shape == (horizon, k, k)
    assert Vf.shape == (horizon, k)
    assert (Vf > 0).all()

    # h = 1 is the exact one-step recursion: a horizon-1 call returns
    # bitwise the same first entry (the h>=2 approximation must not leak
    # backward into h=1). The Rust suite additionally asserts forecast(1)
    # is bitwise the legacy exact one-step covariance.
    r1 = tsecon.dcc_garch(RETURNS, forecast_horizon=1)
    np.testing.assert_array_equal(np.asarray(r1["correlation_forecast"])[0], Rf[0])
    np.testing.assert_array_equal(np.asarray(r1["covariance_forecast"])[0], Hf[0])

    # The field-report item: correlation_last is NOT the one-step forecast.
    # R_{T+1} additionally uses the final residual z_T, so they must differ.
    assert not np.allclose(Rf[0], np.asarray(r["correlation_last"]), atol=1e-12)

    # Every forecast R is a proper correlation matrix, and H = D R D has
    # the variance forecasts on its diagonal.
    for h in (0, 9, horizon - 1):
        np.testing.assert_allclose(np.diag(Rf[h]), 1.0, atol=1e-14)
        assert np.linalg.eigvalsh(Rf[h]).min() > 0
        np.testing.assert_allclose(np.diag(Hf[h]), Vf[h], rtol=1e-12)

    # E[R_{T+h}] -> corr(Qbar) geometrically at rate a + b (the documented
    # Engle-Sheppard Q-recursion convention). 1.1x cushions the nonlinear
    # normalization, matching the Rust-side bound.
    rbar = _corr(r["qbar"])
    pers = r["a"] + r["b"]
    d1 = np.abs(Rf[0] - rbar).max()
    dH = np.abs(Rf[-1] - rbar).max()
    assert dH <= 1.1 * pers ** (horizon - 1) * d1 + 1e-12
    assert dH < d1


def test_forecast_horizon_zero_returns_no_forecast_keys():
    r = tsecon.dcc_garch(SUB, forecast_horizon=0)
    assert "correlation_forecast" not in r
    assert "covariance_forecast" not in r
    assert "variance_forecast" not in r


# --------------------------------------------------------------------------
# Variants: cDCC and ADCC run and nest
# --------------------------------------------------------------------------

def test_cdcc_runs_and_stays_near_dcc_on_symmetric_dgp():
    dcc = tsecon.dcc_garch(SUB)
    cdcc = tsecon.dcc_garch(SUB, variant="cdcc", forecast_horizon=5)
    assert cdcc["variant"] == "cdcc"
    assert cdcc["a"] >= 0 and cdcc["b"] >= 0 and cdcc["a"] + cdcc["b"] < 1
    # Aielli's S is a correlation matrix by construction: exactly-unit
    # diagonal (plain-DCC qbar has only approximately-unit diagonal).
    S = np.asarray(cdcc["qbar"])
    np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-12)
    # On this plain-DCC DGP the correction is small: parameters land near
    # the plain-DCC estimates and S near Qbar (the consistency correction
    # matters asymptotically, not on one finite symmetric sample).
    assert abs(cdcc["a"] - dcc["a"]) < 0.02
    assert abs(cdcc["b"] - dcc["b"]) < 0.05
    np.testing.assert_allclose(S, np.asarray(dcc["qbar"]), atol=0.02)
    assert np.asarray(cdcc["correlation_forecast"]).shape == (5, 2, 2)


def test_adcc_nests_dcc():
    dcc = tsecon.dcc_garch(SUB)
    adcc = tsecon.dcc_garch(SUB, variant="adcc")
    assert adcc["variant"] == "adcc"
    assert adcc["g"] >= 0.0
    # The ADCC feasible set contains every DCC point (g = 0), so the ADCC
    # optimum cannot be materially worse than DCC's (optimizer slack only).
    assert adcc["loglik"] >= dcc["loglik"] - 1e-2
    # Symmetric DGP: no real asymmetry to find (loose one-realization bar,
    # same as the Rust fixture test).
    assert adcc["g"] < 0.08
    # Stationarity/positivity: a + b + g bounded strictly below 1 is
    # necessary for the CES sufficient condition a + b + delta*g < 1
    # (delta <= 1-ish on this data); the estimator enforces the exact one.
    assert adcc["a"] + adcc["b"] < 1.0


def test_student_t_second_stage():
    gauss = tsecon.dcc_garch(SUB)
    t = tsecon.dcc_garch(SUB, dist="t")
    assert t["dist"] == "t"
    assert "nu" in t
    # Gaussian innovations: the t nests the normal as nu -> infinity, so
    # the estimated dof should be pushed up, and the dynamics should agree.
    assert t["nu"] > 10.0
    assert abs(t["a"] - gauss["a"]) < 0.02
    assert abs(t["b"] - gauss["b"]) < 0.05
    assert "nu" not in gauss


# --------------------------------------------------------------------------
# Engle-Sheppard constant-correlation test
# --------------------------------------------------------------------------

def test_dcc_test_rejects_on_dcc_fixture():
    """Power wiring: the fixture is a genuine DCC DGP (a=0.03, b=0.95,
    T=2400, k=3) — the diagnostic must reject constant correlation."""
    r = tsecon.dcc_test(RETURNS, lags=5)
    assert r["df"] == 6 and r["lags"] == 5
    assert r["nobs"] == 2400
    assert r["n_stacked"] == (2400 - 5) * 3
    assert np.isfinite(r["stat"]) and r["stat"] >= 0
    assert r["p_value"] < 0.05


def test_dcc_test_sane_on_seeded_ccc_null():
    """Size wiring on ONE seeded null draw: constant-correlation data must
    not be rejected wildly (this is a wiring check on a single seed — the
    measured 200-rep size Monte Carlo lives in the volatility model card)."""
    x = _sim_ccc_garch(600, rho=0.5, seed=7)
    r = tsecon.dcc_test(x, lags=5)
    assert r["df"] == 6
    assert r["n_stacked"] == 600 - 5      # one pair for k = 2
    assert 0.0 < r["p_value"] <= 1.0
    assert r["p_value"] > 0.05            # this seed's draw is comfortably null


def test_dcc_test_default_lags_is_five():
    x = _sim_ccc_garch(400, rho=0.3, seed=11)
    assert tsecon.dcc_test(x)["df"] == 6  # lags defaults to 5 (ES tabulate 5)


# --------------------------------------------------------------------------
# Error surfaces (teaching ValueErrors)
# --------------------------------------------------------------------------

def test_unknown_variant_names_the_choices():
    with pytest.raises(ValueError, match=r"cdcc"):
        tsecon.dcc_garch(SUB, variant="dcc2")


def test_unknown_dist_names_the_choices():
    with pytest.raises(ValueError, match=r"normal"):
        tsecon.dcc_garch(SUB, dist="gaussian")


def test_single_series_rejected():
    one = SUB[:, :1]
    with pytest.raises(ValueError, match=r"[Aa]t least"):
        tsecon.dcc_garch(one)
    with pytest.raises(ValueError, match=r"[Aa]t least"):
        tsecon.dcc_test(one)


def test_dcc_test_zero_lags_rejected():
    with pytest.raises(ValueError, match=r"lags"):
        tsecon.dcc_test(SUB, lags=0)


# --------------------------------------------------------------------------
# Docstring honesty (house tripwire pattern: every returned key is named)
# --------------------------------------------------------------------------

def test_dcc_garch_docstring_names_every_returned_key():
    import re

    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.dcc_garch.__doc__ or ""))
    keys = set(tsecon.dcc_garch(SUB, dist="t", forecast_horizon=2).keys())
    missing = keys - tokens
    assert not missing, f"dcc_garch.__doc__ does not name returned keys: {sorted(missing)}"
    # The timing convention must be spelled out (the 0.4.0 field report:
    # the minimal surface invited misreading correlation_last as stale).
    flat = (tsecon.dcc_garch.__doc__ or "").replace("\n", " ")
    assert "TIMING CONVENTION" in flat
    assert "correlation_forecast[0]" in flat


def test_dcc_test_docstring_names_every_returned_key():
    import re

    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.dcc_test.__doc__ or ""))
    keys = set(tsecon.dcc_test(_sim_ccc_garch(300, seed=3), lags=2).keys())
    missing = keys - tokens
    assert not missing, f"dcc_test.__doc__ does not name returned keys: {sorted(missing)}"
