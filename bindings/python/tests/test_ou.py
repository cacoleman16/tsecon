"""Ornstein-Uhlenbeck spread utilities (`ou_fit`, `spread_zscore`) — field item 8.

The validation grade, mirrored in the model card and validation matrix:
closed-form + statsmodels AR(1) golden + MC-measured kappa bias and CI
coverage. Concretely:

* the AR(1) discretization leg is pinned against a LIVE statsmodels
  ``AutoReg(x, lags=1, trend='c').fit()`` here at 1e-12 relative (it is the
  same estimator: OLS with the MLE variance RSS/n; the achieved agreement is
  ~1e-15), and against the frozen fixture in the Rust crate tests;
* the OU mapping layer on top is closed-form and is re-derived in-test from
  the AR(1) numbers (documented-formula check, exact to float round-off);
* the finite-sample kappa bias (Kendall 1954 mapped through -ln(phi)/dt;
  Tang & Chen 2009: ~4 / time span) is MEASURED by a seeded MC smoke here,
  with the full 2000-rep grid in
  docs/examples/coverage/experiments/ou_kappa_bias_coverage.py — the bias is
  documented and quantified, deliberately not corrected away;
* the shipped half-life CI (level-scale kappa interval mapped through
  ln2/kappa, +inf upper endpoint when the interval crosses zero) has its
  coverage measured by the same MC; the smoke here re-measures one healthy
  cell and asserts the honest-uncertainty branches.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
OU = json.loads((FIX / "ou.json").read_text())
CELLS = {c["name"]: c for c in OU["cells"]}


def _fit(name):
    c = CELLS[name]
    return c, tsecon.ou_fit(np.array(c["x"]), dt=c["dt"], level=c["level"])


# --------------------------------------------------------------- goldens


@pytest.mark.parametrize("name", ["daily_fast", "daily_slow", "monthly", "daily_weak"])
def test_ar1_leg_matches_live_statsmodels_autoreg(name):
    """ou_fit's AR(1) leg IS AutoReg(x, lags=1): params, sigma2, bse, llf."""
    sm = pytest.importorskip("statsmodels.tsa.ar_model")
    c, r = _fit(name)
    fit = sm.AutoReg(np.array(c["x"]), lags=1, trend="c").fit()
    c_sm, phi_sm = fit.params
    assert abs(r["phi"] - phi_sm) <= 1e-12 * abs(phi_sm)
    assert abs(r["c"] - c_sm) <= 1e-12 * max(1.0, abs(c_sm))
    assert abs(r["eta2"] - fit.sigma2) <= 1e-12 * fit.sigma2
    assert abs(r["loglik"] - fit.llf) <= 1e-12 * abs(fit.llf)
    assert abs(r["c_se"] - fit.bse[0]) <= 1e-12 * fit.bse[0]
    assert abs(r["phi_se"] - fit.bse[1]) <= 1e-12 * fit.bse[1]
    assert r["n_obs"] == len(c["x"]) - 1


@pytest.mark.parametrize("name", ["daily_fast", "daily_slow", "monthly", "daily_weak"])
def test_ou_mapping_matches_fixture(name):
    """The OU layer reproduces the NumPy transcription of the documented
    closed form (fixture `ou` block) at 1e-10 relative."""
    c, r = _fit(name)
    ou = c["ou"]
    for key in ("kappa", "mu", "sigma", "kappa_se", "mu_se", "sigma_se", "half_life"):
        assert abs(r[key] - ou[key]) <= 1e-10 * max(1.0, abs(ou[key])), key
    lo, hi = r["half_life_ci"]
    flo, fhi = ou["half_life_ci"]
    assert abs(lo - flo) <= 1e-10 * abs(flo)
    if fhi is None:  # the +inf upper branch (kappa interval crosses zero)
        assert math.isinf(hi) and hi > 0
    else:
        assert abs(hi - fhi) <= 1e-10 * abs(fhi)
    assert r["mean_reverting"] is True
    assert abs(r["stationary_sd"] - ou["stationary_sd"]) <= 1e-10 * ou["stationary_sd"]


def test_ou_mapping_closed_form_identities():
    """The mapping identities, re-derived from the returned AR(1) leg:
    kappa = -ln(phi)/dt, mu = c/(1-phi), the stationary-variance identity
    eta2 = sigma^2 (1-phi^2)/(2 kappa), half_life * kappa = ln 2, and
    stationary_sd = sigma/sqrt(2 kappa) — float-exact to ~1 ulp."""
    for name in ("daily_fast", "daily_slow", "monthly"):
        c, r = _fit(name)
        dt, phi = r["dt"], r["phi"]
        assert r["kappa"] == -np.log(phi) / dt
        assert r["mu"] == r["c"] / (1.0 - phi)
        np.testing.assert_allclose(
            r["sigma"] ** 2 * (1 - phi**2) / (2 * r["kappa"]), r["eta2"], rtol=1e-12
        )
        np.testing.assert_allclose(r["half_life"] * r["kappa"], np.log(2.0), rtol=1e-15)
        np.testing.assert_allclose(
            r["stationary_sd"], r["sigma"] / np.sqrt(2 * r["kappa"]), rtol=1e-12
        )
        np.testing.assert_allclose(np.exp(-r["kappa"] * dt), phi, rtol=1e-14)


def test_spread_zscore_golden_and_paths():
    zfx = OU["zscore"]
    c = CELLS[zfx["cell"]]
    x = np.array(c["x"])
    r = tsecon.ou_fit(x, dt=c["dt"])
    head = x[: zfx["n_head"]]
    # explicit-parameter path against the fixture head
    z = tsecon.spread_zscore(head, kappa=r["kappa"], mu=r["mu"], sigma=r["sigma"])
    np.testing.assert_allclose(z["zscore"], zfx["zscore_head"], rtol=1e-10)
    assert z["fitted"] is False
    np.testing.assert_allclose(z["stationary_sd"], r["stationary_sd"], rtol=1e-12)
    # fitted path: fitting on the full series must equal freezing that fit
    zf = tsecon.spread_zscore(x, dt=c["dt"])
    zx = tsecon.spread_zscore(x, kappa=r["kappa"], mu=r["mu"], sigma=r["sigma"])
    np.testing.assert_array_equal(zf["zscore"], zx["zscore"])
    assert zf["fitted"] is True and zf["kappa"] == r["kappa"]
    # the documented formula itself
    np.testing.assert_allclose(
        zf["zscore"], (x - r["mu"]) / (r["sigma"] / np.sqrt(2 * r["kappa"])), rtol=1e-12
    )


# ------------------------------------------------ honesty branches / refusals


def test_explosive_reported_not_raised():
    """phi_hat > 1: the honest mean_reverting=False result, not an error."""
    c, r = _fit_explosive()
    assert r["phi"] > 1.0
    assert r["kappa"] < 0.0
    assert r["mean_reverting"] is False
    assert math.isinf(r["half_life"]) and r["half_life"] > 0
    assert r["half_life_ci"] is None
    assert r["stationary_sd"] is None
    # the AR(1) leg is still fully reported (it is the informative part)
    assert np.isfinite([r["phi_se"], r["c"], r["c_se"], r["eta2"], r["loglik"]]).all()


def _fit_explosive():
    c = CELLS["explosive"]
    return c, tsecon.ou_fit(np.array(c["x"]), dt=c["dt"])


def test_zscore_refuses_non_mean_reverting():
    c = CELLS["explosive"]
    with pytest.raises(ValueError, match="stationary distribution"):
        tsecon.spread_zscore(np.array(c["x"]), dt=c["dt"])
    with pytest.raises(ValueError, match="stationary distribution"):
        tsecon.spread_zscore(np.array(c["x"]), kappa=-0.5, mu=0.0, sigma=1.0)


def test_zscore_refuses_partial_parameters():
    x = np.array(CELLS["daily_fast"]["x"])
    with pytest.raises(ValueError, match="all three"):
        tsecon.spread_zscore(x, kappa=1.0)
    with pytest.raises(ValueError, match="all three"):
        tsecon.spread_zscore(x, mu=0.0, sigma=1.0)


def test_ou_fit_refusals_teach():
    with pytest.raises(ValueError, match="at least 4 observations"):
        tsecon.ou_fit(np.array([1.0, 2.0, 3.0]))
    x = np.array(CELLS["monthly"]["x"])
    with pytest.raises(ValueError, match="dt must be"):
        tsecon.ou_fit(x, dt=0.0)
    with pytest.raises(ValueError, match="level must lie"):
        tsecon.ou_fit(x, level=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.ou_fit(np.array([0.1, np.nan, 0.3, 0.2, 0.1]))
    with pytest.raises(ValueError, match="constant over the sample"):
        tsecon.ou_fit(np.full(16, 2.0))
    # anti-persistent: phi_hat <= 0 has no real kappa
    rng = np.random.default_rng(3)
    alt = np.tile([1.0, -1.0], 32) + 1e-3 * rng.standard_normal(64)
    with pytest.raises(ValueError, match="anti-persistent"):
        tsecon.ou_fit(alt)


def test_weak_cell_ci_upper_is_inf():
    """daily_weak: kappa_hat > 0 but its level-scale interval crosses zero,
    so the shipped half-life CI honestly reports an infinite upper bound."""
    c, r = _fit("daily_weak")
    assert r["mean_reverting"] is True
    lo, hi = r["half_life_ci"]
    assert 0 < lo < r["half_life"]
    assert math.isinf(hi)


# ------------------------------------------------------------ MC (seeded)


def test_mc_kappa_bias_and_shipped_ci_coverage_smoke():
    """Seeded 400-rep smoke of the full experiment
    (docs/examples/coverage/experiments/ou_kappa_bias_coverage.py):
    monthly 20y at kappa=2 — the healthy cell. Asserts (a) the measured
    kappa bias is positive and within [0.3x, 3x] of the Kendall/Tang-Chen
    first-order prediction (1+3phi)/(n phi dt) ~ 4/span, and (b) the
    shipped half-life CI covers within [0.90, 0.98] at nominal 0.95
    (full-grid 2000-rep measurement: 0.948)."""
    kappa, dt, n, reps = 2.0, 1 / 12, 240, 400
    mu, sigma = 0.0, 0.2
    phi = np.exp(-kappa * dt)
    c0 = mu * (1 - phi)
    eta = np.sqrt(sigma**2 * (1 - phi**2) / (2 * kappa))
    rng = np.random.default_rng(20260826)
    x = np.empty((reps, n))
    x[:, 0] = mu
    shocks = rng.standard_normal((reps, n - 1))
    for t in range(1, n):
        x[:, t] = c0 + phi * x[:, t - 1] + eta * shocks[:, t - 1]
    hl_true = np.log(2.0) / kappa
    khat = np.empty(reps)
    cover = 0
    for r in range(reps):
        fit = tsecon.ou_fit(x[r], dt=dt)
        khat[r] = fit["kappa"]
        if fit["mean_reverting"]:
            lo, hi = fit["half_life_ci"]
            cover += lo <= hl_true <= hi
    bias = khat.mean() - kappa
    pred = (1 + 3 * phi) / ((n - 1) * phi * dt)
    assert bias > 0, "the finite-sample kappa bias is upward"
    assert 0.3 * pred < bias < 3.0 * pred, (bias, pred)
    assert 0.90 <= cover / reps <= 0.98, cover / reps


# ------------------------------------------------------------- surface


def test_coercion_pandas_and_list():
    pd = pytest.importorskip("pandas")
    c = CELLS["daily_fast"]
    r_arr = tsecon.ou_fit(np.array(c["x"]), dt=c["dt"])
    r_ser = tsecon.ou_fit(pd.Series(c["x"]), dt=c["dt"])
    r_list = tsecon.ou_fit(list(c["x"]), dt=c["dt"])
    assert r_arr["kappa"] == r_ser["kappa"] == r_list["kappa"]
    z = tsecon.spread_zscore(pd.Series(c["x"]), dt=c["dt"])
    assert z["fitted"] is True


def test_docstrings_name_every_returned_key():
    """House tripwire (audit rounds 3-4, finding 5): every returned key is
    named, backticked, in the runtime docstring."""
    import re

    def tokens(fn):
        return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))

    c = CELLS["daily_fast"]
    keys = set(tsecon.ou_fit(np.array(c["x"]), dt=c["dt"]).keys())
    missing = keys - tokens(tsecon.ou_fit)
    assert not missing, f"ou_fit.__doc__ does not name returned keys: {sorted(missing)}"
    zkeys = set(tsecon.spread_zscore(np.array(c["x"]), dt=c["dt"]).keys())
    zmissing = zkeys - tokens(tsecon.spread_zscore)
    assert not zmissing, f"spread_zscore.__doc__ misses: {sorted(zmissing)}"


# --------------------------------------------------------------------------
# Audit round 10: dt is refused when the OU law is frozen
# --------------------------------------------------------------------------

def test_spread_zscore_dt_refused_when_law_frozen():
    """dt only parameterizes the internal ou_fit(x, dt); with kappa/mu/sigma
    all frozen no fit runs and the z-score is dt-free (verified bit-identical
    before the refusal landed), so explicit dt raises with the cure."""
    c = CELLS[OU["zscore"]["cell"]]
    x = np.array(c["x"])
    with pytest.raises(ValueError, match="dt") as exc:
        tsecon.spread_zscore(x, kappa=0.5, mu=0.0, sigma=1.0, dt=0.25)
    msg = str(exc.value)
    assert "frozen" in msg and "ou_fit" in msg
    # Sentinel resolution: omitted dt == the historical explicit dt=1.0 on
    # the fitted path.
    a = tsecon.spread_zscore(x)
    b = tsecon.spread_zscore(x, dt=1.0)
    np.testing.assert_array_equal(a["zscore"], b["zscore"])
    assert a["kappa"] == b["kappa"]
    # dt stays live on the fitted path (kappa is quoted in 1/dt units).
    d = tsecon.spread_zscore(x, dt=0.25)
    assert d["kappa"] != a["kappa"]
    # The z-score itself is dt-invariant on the fitted path (the law is
    # refit in the new units), which is exactly why dt with a frozen law
    # can never act.
    np.testing.assert_allclose(d["zscore"], a["zscore"], rtol=1e-9)
    # And the frozen path still works with dt omitted.
    z = tsecon.spread_zscore(x, kappa=0.5, mu=0.0, sigma=1.0)
    assert z["fitted"] is False
