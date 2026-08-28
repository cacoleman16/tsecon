"""Convergence-signaling surface tests for the four audited defects.

1. panel_pmg: the I(1) repair — relative stopping rule + unrestricted-ARDL
   restart. Pre-fix, the 20-seed battery below hard-failed 14/20 (I(1) x),
   0/20 (I(0) x), 16/20 (I(1) x scaled x100); the mechanism was divergence
   of the back-substitution from the pinned theta = 0 start. Post-fix all
   60 cells converge, and the pmg.json golden stops at the identical
   iterate (29), which is what keeps it bit-identical.
2. arima_fit: `converged` emitted, and the garch-pattern boundary flags —
   an over-differenced ARIMA(0,1,1) piles the MA root onto the unit circle
   where the full-vector observed information STILL inverts to a finite,
   confident-looking se with cov_ok=True; the flags now NaN that se.
3. quantile_lp / growth_at_risk: the IRLS engine's per-fit `converged`
   flag reaches the result dicts with the shape of the estimates.
4. dfm_nowcast(method="mle"): optimizer `converged` + `iterations`;
   two_step carries neither (it has no optimizer).
"""
import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"


# --------------------------------------------------------------- panel_pmg
def _pmg_battery_panel(seed, kind, n=10, t=150):
    rng = np.random.default_rng(seed)
    ys, xs = [], []
    for _ in range(n):
        ex = rng.standard_normal(t)
        x = ex if kind == "i0" else np.cumsum(ex)
        if kind == "i1x100":
            x = 100.0 * x
        eps = rng.standard_normal(t)
        y = np.zeros(t)
        y[0] = x[0]
        for tt in range(1, t):
            dx = x[tt] - x[tt - 1]
            y[tt] = y[tt - 1] - 0.3 * (y[tt - 1] - x[tt - 1]) + 0.2 * dx + 0.1 * eps[tt]
        ys.append(y)
        xs.append(x.reshape(-1, 1))
    return ys, xs


@pytest.mark.parametrize("kind", ["i1", "i0", "i1x100"])
def test_pmg_battery_converges_at_every_scale(kind):
    """The adversarial battery: DGP dy = -0.3(y - x) + 0.2 dx + 0.1 eps.

    Pre-fix (absolute 1e-12 stopping rule, theta = 0 start only) this
    hard-failed 14/20 (i1), 0/20 (i0), 16/20 (i1x100). Now every seed
    converges and recovers the true common long run theta = 1 at every
    data scale.
    """
    fails = []
    for seed in range(20):
        ys, xs = _pmg_battery_panel(seed, kind)
        try:
            r = tsecon.panel_pmg(ys, xs)
        except Exception as e:  # noqa: BLE001 — counting hard failures
            fails.append((seed, str(e)[:70]))
            continue
        assert abs(r["theta"][0] - 1.0) < 0.2, (kind, seed, r["theta"][0])
        assert r["phi_bar"] < 0
    assert not fails, f"{kind}: {len(fails)}/20 hard failures: {fails}"


def test_pmg_golden_unmoved_and_stops_at_the_same_iterate():
    """The stopping-rule change must not move the golden: same stopping
    iterate (29, measured identical pre/post fix), same values at the
    golden tolerance."""
    fx = json.loads((FIXTURES / "pmg.json").read_text())
    ys = [np.array(u) for u in fx["y"]]
    xk = fx["x"]
    n = fx["design"]["N"]
    xs = [np.column_stack([xk[0][i], xk[1][i]]) for i in range(n)]
    r = tsecon.panel_pmg(ys, xs)
    assert r["iterations"] == 29
    np.testing.assert_allclose(r["theta"], fx["pmg"]["theta"], atol=1e-7)
    np.testing.assert_allclose(r["theta_se"], fx["pmg"]["theta_se"], atol=1e-7)


def test_pmg_tol_and_max_iter_kwargs():
    ys, xs = _pmg_battery_panel(0, "i0")
    # A loose tolerance converges in fewer iterations than the default.
    loose = tsecon.panel_pmg(ys, xs, tol=1e-6)
    tight = tsecon.panel_pmg(ys, xs)
    assert loose["iterations"] < tight["iterations"]
    np.testing.assert_allclose(loose["theta"], tight["theta"], rtol=1e-5)
    # Invalid options are refused with a teaching message.
    with pytest.raises(Exception, match="tol"):
        tsecon.panel_pmg(ys, xs, tol=0.0)
    with pytest.raises(Exception, match="max_iter"):
        tsecon.panel_pmg(ys, xs, max_iter=0)
    # Genuine non-convergence stays reachable, names the knobs, and does
    # not blame the data.
    with pytest.raises(Exception, match="relative tolerance") as exc:
        tsecon.panel_pmg(ys, xs, max_iter=1)
    assert "max_iter" in str(exc.value)
    assert "too weakly cointegrated" not in str(exc.value)


# --------------------------------------------------------------- arima_fit
def test_arima_over_differenced_ma_boundary_is_flagged():
    """Over-differenced white noise: the MA root piles up on the unit
    circle. Pre-fix this reported cov_ok=True with a finite, confident
    bse for ma.L1 and no converged key at all."""
    rng = np.random.default_rng(1)  # theta lands at -1.000000 (measured)
    y = rng.standard_normal(300)
    f = tsecon.arima_fit(y, p=0, d=1, q=1, constant=False)

    assert isinstance(f["converged"], bool)
    theta = float(f["params"][0])
    assert theta <= -0.999, f"expected an MA-boundary pile-up, got {theta}"
    # Packed order [ma.L1, sigma2]: the MA block flagged, sigma2 never.
    assert list(f["boundary"]) == [True, False]
    note = f["boundary_note"]
    assert note is not None and "invertibility" in note and "ma.L1" in note
    assert "over-differenced" in note
    # The trap, closed: the full-vector information inverted (cov_ok True)
    # but the boundary parameter's se is NaN'd with se_valid False.
    assert f["cov_ok"] is True
    bse = np.asarray(f["bse"], dtype=float)
    assert np.isnan(bse[0]) and np.isfinite(bse[1])
    assert list(f["se_valid"]) == [False, True]
    # param_cov itself is still the full-vector matrix (documented).
    assert np.asarray(f["param_cov"]).shape == (2, 2)


def test_arima_interior_fit_flags_clean_and_converged():
    rng = np.random.default_rng(0)
    e = rng.standard_normal(400)
    y = np.zeros(400)
    for t in range(1, 400):
        y[t] = 1.0 + 0.6 * y[t - 1] + e[t]
    f = tsecon.arima_fit(y, p=1, d=0, q=0, constant=True)
    assert f["converged"] is True
    assert not any(f["boundary"])
    assert f["boundary_note"] is None
    assert f["cov_ok"] is True
    assert all(f["se_valid"])
    assert np.isfinite(np.asarray(f["bse"], dtype=float)).all()


def test_auto_arima_shares_the_new_keys():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(160)
    y = np.zeros(160)
    for t in range(1, 160):
        y[t] = 0.5 * y[t - 1] + e[t]
    r = tsecon.auto_arima(y, max_p=2, max_q=1)
    for key in ("converged", "boundary", "boundary_note", "se_valid"):
        assert key in r, key
    # The search never selects near-unit-root candidates.
    assert not any(r["boundary"])


def test_arima_fit_docstring_names_every_returned_key():
    """The runtime docstring must name every returned key (the
    audit-round-3/4 tripwire pattern, extended to arima_fit now that it
    gained flag keys)."""
    import re

    rng = np.random.default_rng(3)
    y = np.cumsum(rng.standard_normal(120))
    keys = set(tsecon.arima_fit(y, 1, 1, 0, forecast_steps=4, conf_alpha=0.1).keys())
    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.arima_fit.__doc__ or ""))
    missing = keys - tokens
    assert not missing, f"arima_fit.__doc__ does not name returned keys: {sorted(missing)}"


# ------------------------------------------- quantile_lp / growth_at_risk
def _lp_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.8 * shock[t] + 0.3 * shock[t - 1] + rng.standard_normal()
    return y, shock


def test_quantile_lp_emits_per_fit_converged_and_reports_real_exhaustion():
    y, shock = _lp_data()
    taus = [0.1, 0.5, 0.9]
    q = tsecon.quantile_lp(y, shock, taus=taus, horizons=4, n_lag_controls=2)
    conv = q["converged"]
    # Shape mirrors irf: [tau][h].
    assert len(conv) == len(taus)
    for row_c, row_i in zip(conv, q["irf"]):
        assert len(row_c) == len(row_i)
        assert all(isinstance(c, bool) for c in row_c)
    # This ordinary AR(1)+shock design genuinely exhausts the IRLS cap at
    # one cell — (tau=0.5, h=1) cycles between vertices and never meets
    # the 1e-6 coefficient tolerance within 1000 iterations (statsmodels
    # QuantReg emits a ConvergenceWarning on the identical design; the
    # binding used to swallow the flag entirely). The fit is
    # deterministic, so the cell is pinned: exactly the situation the
    # per-fit flag exists to surface.
    assert conv[1][1] is False, f"expected the pinned exhaustion cell, got {conv}"
    assert sum(0 if c else 1 for row in conv for c in row) == 1
    # Best-iterate semantics: the unconverged cell still reports finite
    # numbers — the flag, not a NaN, is the signal.
    assert np.isfinite(q["irf"][1][1]) and np.isfinite(q["se"][1][1])
    # And the other cells converge.
    assert all(conv[0]) and all(conv[2])


def test_growth_at_risk_emits_per_tau_converged():
    rng = np.random.default_rng(1)
    n = 220
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
        scale = 0.4 * np.exp(0.4 * x[t - 1])
        y[t] = 0.2 + 0.3 * y[t - 1] - 0.4 * x[t - 1] + scale * rng.standard_normal()
    taus = [0.05, 0.5, 0.95]
    g = tsecon.growth_at_risk(y, x.reshape(-1, 1), horizon=4, taus=taus)
    conv = g["converged"]
    assert len(conv) == len(taus) == len(g["params"])
    assert all(isinstance(c, bool) for c in conv)
    assert all(conv)


# ------------------------------------------------------------ dfm_nowcast
def _dfm_panel(n=120, big_n=6, seed=5):
    rng = np.random.default_rng(seed)
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    load = rng.uniform(0.6, 1.4, big_n)
    return np.outer(f, load) + 0.4 * rng.standard_normal((n, big_n))


def test_dfm_nowcast_mle_reports_converged_and_iterations():
    x = _dfm_panel()
    mle = tsecon.dfm_nowcast(x, n_factors=1, factor_order=1, method="mle")
    assert isinstance(mle["converged"], bool)
    assert isinstance(mle["iterations"], int) and mle["iterations"] > 0


def test_dfm_nowcast_two_step_carries_no_optimizer_keys():
    # No iterative optimizer runs in the two-step route; inventing a flag
    # for it would be dishonest, so the keys are deliberately absent.
    x = _dfm_panel()
    two = tsecon.dfm_nowcast(x, n_factors=1, factor_order=1)
    assert "converged" not in two
    assert "iterations" not in two
