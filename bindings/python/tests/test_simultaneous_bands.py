"""Simultaneous (sup-t) bands, from Python.

Every band tsecon shipped before this was POINTWISE. Read as a statement about
a whole path a pointwise band fails badly, and the failure is multiplicity, not
inconsistency, so it does not shrink with the sample: the interval-coverage
audit measured LP's pointwise joint rate moving only 36.5% -> 42.7% when T went
from 240 to 720, against a nominal 90%.

These are the assertions that would fail if the exposure were wrong:

* the simultaneous band contains the SYMMETRIC pointwise band cell by cell
  (`point +/- z*se`, the like-for-like comparator — not the bootstrap
  percentile band, which is a different SHAPE);
* the critical value exceeds the pointwise z and grows with K;
* at K = 1 the two coincide;
* the DEFAULTS are unchanged — the pointwise output is golden-gated and
  published in the audit, so a silent widening would invalidate both;
* the scope label and K come back with every simultaneous band;
* an unknown `band` raises, naming the accepted values;
* the sup-t-from-covariance route is a pure function of its seed;
* lp_iv / lp_multiplier / lp_state REFUSE sup-t, because no cross-horizon
  covariance exists for them — they get Sidak/Bonferroni only.

Every RNG is seeded, every panel is small, and matplotlib is never imported.
"""
import numpy as np
import pytest

import tsecon

# The four accepted spellings of the multiplier, and the two families that make
# a joint promise using only K (so their critical values are analytic).
CLOSED_FORM = ("sidak", "bonferroni")


def _var_panel(n=220, seed=20260809):
    """A stable bivariate VAR(1) — the shape the coverage audit measured."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.55, 0.10], [0.15, 0.45]])
    e = rng.standard_normal((n, 2))
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + e[t]
    return y


def _lp_series(n=400, rho=0.7, seed=20260810):
    rng = np.random.default_rng(seed)
    shock = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + shock[t] + 0.5 * noise[t]
    return y, shock


def _lp_iv_series(n=360, seed=20260811):
    y, shock = _lp_series(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    instrument = shock + 0.3 * rng.standard_normal(n)
    return y, shock, instrument


# ---------------------------------------------------------------------------
# The defaults must not have moved
# ---------------------------------------------------------------------------

def test_var_irf_bands_default_is_untouched_pointwise():
    """Adding `band=` must not widen anything at the default. The pointwise
    arrays are golden-gated and published in the coverage audit."""
    y = _var_panel()
    base = tsecon.var_irf_bands(y, lags=2, horizon=8)
    again = tsecon.var_irf_bands(y, lags=2, horizon=8, band="pointwise")
    for key in ("point", "se", "lower", "upper"):
        assert np.array_equal(np.asarray(base[key]), np.asarray(again[key]))
    # A pointwise call produces no simultaneous band at all: nothing to misread.
    for key in ("sim_lower", "sim_upper", "critical_value", "n_cells"):
        assert key not in base
    assert base["band"] == "pointwise"


def test_var_irf_bands_simultaneous_leaves_the_pointwise_arrays_alone():
    """The simultaneous band changes ONLY the multiplier: same point, same se,
    same pointwise lower/upper, bit for bit."""
    y = _var_panel()
    base = tsecon.var_irf_bands(y, lags=2, horizon=8)
    sim = tsecon.var_irf_bands(
        y, lags=2, horizon=8, band="sup-t", band_n_sim=20_000
    )
    for key in ("point", "se", "lower", "upper"):
        assert np.array_equal(np.asarray(base[key]), np.asarray(sim[key]))


def test_var_forecast_default_is_untouched():
    y = _var_panel()
    base = tsecon.var_forecast(y, lags=2, steps=8)
    again = tsecon.var_forecast(y, lags=2, steps=8, band="pointwise")
    for key in ("point", "lower", "upper"):
        assert np.array_equal(np.asarray(base[key]), np.asarray(again[key]))
    assert "sim_lower" not in base and "critical_value" not in base
    sim = tsecon.var_forecast(y, lags=2, steps=8, band="sup-t", band_n_sim=20_000)
    for key in ("point", "lower", "upper"):
        assert np.array_equal(np.asarray(base[key]), np.asarray(sim[key]))


def test_lp_default_returns_no_band_and_the_same_estimates():
    """`lp` had no band before this; `band=None` must keep it that way, and a
    banded call must not disturb the point path or the standard errors."""
    y, shock = _lp_series()
    base = tsecon.lp(y, shock, horizons=10, n_lag_controls=4)
    assert set(base) == {"horizons", "irf", "se"}
    for method in ("pointwise", "sup-t", "sidak", "bonferroni"):
        got = tsecon.lp(
            y, shock, horizons=10, n_lag_controls=4, band=method, band_n_sim=20_000
        )
        assert np.array_equal(got["irf"], base["irf"])
        assert np.array_equal(got["se"], base["se"])


def test_smooth_lp_default_returns_no_band_and_the_same_estimates():
    y, shock = _lp_series()
    base = tsecon.smooth_lp(y, shock, horizons=10, n_lag_controls=4, lam=1.0)
    assert "lower" not in base and "critical_value" not in base
    got = tsecon.smooth_lp(
        y,
        shock,
        horizons=10,
        n_lag_controls=4,
        lam=1.0,
        band="sup-t",
        band_n_sim=20_000,
    )
    assert np.array_equal(got["irf"], base["irf"])
    assert np.array_equal(got["se"], base["se"])


def test_closed_form_lp_surfaces_default_to_no_band():
    y, shock, inst = _lp_iv_series()
    st = (np.arange(len(y)) % 7 < 3).astype(float)
    assert "lower" not in tsecon.lp_iv(y, shock, inst, horizons=6, n_lag_controls=4)
    assert "lower" not in tsecon.lp_multiplier(
        y, shock, inst, horizons=6, n_lag_controls=4
    )
    assert "lower_state1" not in tsecon.lp_state(
        y, shock, st, horizons=6, n_lag_controls=4
    )


# ---------------------------------------------------------------------------
# Containment: the simultaneous band contains the SYMMETRIC pointwise band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band", ["sup-t", "sidak", "bonferroni"])
def test_var_irf_simultaneous_contains_pointwise_cell_by_cell(band):
    y = _var_panel()
    r = tsecon.var_irf_bands(
        y, lags=2, horizon=8, band=band, band_n_sim=20_000
    )
    lower, upper = np.asarray(r["lower"]), np.asarray(r["upper"])
    sim_lo, sim_hi = np.asarray(r["sim_lower"]), np.asarray(r["sim_upper"])
    # The asymptotic branch's lower/upper ARE the symmetric point +/- z*se, so
    # containment is exact here, not just in expectation.
    assert (sim_lo <= lower + 1e-12).all()
    assert (sim_hi >= upper - 1e-12).all()
    assert (sim_hi >= sim_lo).all()


@pytest.mark.parametrize("band", ["sup-t", "bonferroni"])
def test_bootstrap_simultaneous_contains_the_symmetric_band_not_the_percentiles(band):
    """The bootstrap simultaneous band is `point +/- c*se`; `lower`/`upper` are
    Efron PERCENTILES. Different shapes — so the guarantee is against the
    symmetric comparator `point +/- z*se`, and only against that."""
    y = _var_panel()
    r = tsecon.var_irf_bands(
        y,
        lags=2,
        horizon=6,
        method="bootstrap",
        n_boot=400,
        seed=11,
        band=band,
    )
    point, se = np.asarray(r["point"]), np.asarray(r["se"])
    z = r["pointwise_critical_value"]
    sym_lo, sym_hi = point - z * se, point + z * se
    sim_lo, sim_hi = np.asarray(r["sim_lower"]), np.asarray(r["sim_upper"])
    assert (sim_lo <= sym_lo + 1e-12).all()
    assert (sim_hi >= sym_hi - 1e-12).all()
    # The percentile band is asymmetric, so it is genuinely a different object:
    # if it were the same shape this call would be redundant.
    mid = 0.5 * (np.asarray(r["lower"]) + np.asarray(r["upper"]))
    assert not np.allclose(mid, point, atol=1e-10)


@pytest.mark.parametrize("band", ["sup-t", "sidak", "bonferroni"])
def test_var_forecast_simultaneous_contains_marginal(band):
    y = _var_panel()
    r = tsecon.var_forecast(
        y, lags=2, steps=10, band=band, band_n_sim=20_000
    )
    lower, upper = np.asarray(r["lower"]), np.asarray(r["upper"])
    sim_lo, sim_hi = np.asarray(r["sim_lower"]), np.asarray(r["sim_upper"])
    assert (sim_lo <= lower + 1e-12).all()
    assert (sim_hi >= upper - 1e-12).all()


@pytest.mark.parametrize("band", ["sup-t", "sidak", "bonferroni"])
def test_lp_simultaneous_contains_pointwise(band):
    y, shock = _lp_series()
    pw = tsecon.lp(y, shock, horizons=10, n_lag_controls=4, band="pointwise")
    r = tsecon.lp(
        y, shock, horizons=10, n_lag_controls=4, band=band, band_n_sim=20_000
    )
    assert (r["lower"] <= pw["lower"] + 1e-12).all()
    assert (r["upper"] >= pw["upper"] - 1e-12).all()
    # Anchored on the same centre: only the multiplier moved.
    assert np.allclose(0.5 * (r["lower"] + r["upper"]), r["irf"], atol=1e-12)


@pytest.mark.parametrize("band", CLOSED_FORM)
def test_lp_iv_and_multiplier_simultaneous_contain_pointwise(band):
    y, shock, inst = _lp_iv_series()
    for fn in (tsecon.lp_iv, tsecon.lp_multiplier):
        pw = fn(y, shock, inst, horizons=6, n_lag_controls=4, band="pointwise")
        r = fn(y, shock, inst, horizons=6, n_lag_controls=4, band=band)
        assert (r["lower"] <= pw["lower"] + 1e-12).all()
        assert (r["upper"] >= pw["upper"] - 1e-12).all()


@pytest.mark.parametrize("band", CLOSED_FORM)
def test_lp_state_bands_each_regime_separately(band):
    y, shock = _lp_series(n=400)
    st = (np.arange(len(y)) % 5 < 2).astype(float)
    pw = tsecon.lp_state(
        y, shock, st, horizons=6, n_lag_controls=4, band="pointwise"
    )
    r = tsecon.lp_state(y, shock, st, horizons=6, n_lag_controls=4, band=band)
    for suffix in ("_state1", "_state0"):
        assert (r["lower" + suffix] <= pw["lower" + suffix] + 1e-12).all()
        assert (r["upper" + suffix] >= pw["upper" + suffix] - 1e-12).all()
        assert r["critical_value" + suffix] > r["pointwise_critical_value"]
    assert r["band"] == band and r["band_scope"] == "horizon"
    assert r["n_cells"] == 7


# ---------------------------------------------------------------------------
# The critical value: bigger than z, and growing in K
# ---------------------------------------------------------------------------

def test_lp_critical_values_exceed_z_and_order_as_expected():
    """At K = 13, alpha = 0.10 the audit measured pointwise 1.6449, sup-t
    2.20-2.65 (persistence-dependent), Sidak 2.6490, Bonferroni 2.6653."""
    y, shock = _lp_series()
    cv = {
        m: tsecon.lp(
            y, shock, horizons=12, n_lag_controls=4, band=m, band_n_sim=40_000
        )["critical_value"]
        for m in ("pointwise", "sup-t", "sidak", "bonferroni")
    }
    assert cv["pointwise"] == pytest.approx(1.6449, abs=1e-3)
    assert cv["sidak"] == pytest.approx(2.6490, abs=1e-3)
    assert cv["bonferroni"] == pytest.approx(2.6653, abs=1e-3)
    # sup-t uses the actual cross-horizon dependence, so it is the tightest of
    # the three simultaneous routes and Sidak is (barely) tighter than
    # Bonferroni.
    assert cv["pointwise"] < cv["sup-t"] <= cv["sidak"] < cv["bonferroni"]


def test_lp_critical_value_increases_with_k():
    """Every cell added to the family widens the band for every other cell."""
    y, shock = _lp_series()
    cvs = [
        tsecon.lp(
            y, shock, horizons=h, n_lag_controls=4, band="sup-t", band_n_sim=40_000
        )["critical_value"]
        for h in (2, 6, 12, 20)
    ]
    assert all(a < b for a, b in zip(cvs, cvs[1:])), cvs
    for band in CLOSED_FORM:
        closed = [
            tsecon.lp(y, shock, horizons=h, n_lag_controls=4, band=band)[
                "critical_value"
            ]
            for h in (2, 6, 12, 20)
        ]
        assert all(a < b for a, b in zip(closed, closed[1:])), (band, closed)


def test_at_k_equals_one_simultaneous_coincides_with_pointwise():
    """A family of one cell has nothing to correct for. horizons=0 is K=1."""
    y, shock = _lp_series()
    pw = tsecon.lp(y, shock, horizons=0, n_lag_controls=4, band="pointwise")
    assert pw["n_cells"] == 1
    for band in ("sup-t", "sidak", "bonferroni"):
        r = tsecon.lp(
            y, shock, horizons=0, n_lag_controls=4, band=band, band_n_sim=60_000
        )
        assert r["n_cells"] == 1
        # Sidak and Bonferroni collapse to z exactly; sup-t reaches it up to
        # simulation noise in a tail quantile.
        tol = 0.02 if band == "sup-t" else 1e-12
        assert r["critical_value"] == pytest.approx(
            pw["critical_value"], abs=tol
        )
        assert np.allclose(r["lower"], pw["lower"], atol=tol)
        assert np.allclose(r["upper"], pw["upper"], atol=tol)


def test_var_irf_critical_value_grows_with_the_declared_scope():
    """"Simultaneous over what?" is a user-visible choice with a real price:
    horizon (K = h+1) < shock (K = k(h+1)) < all (K = k^2(h+1))."""
    y = _var_panel()
    out = {
        scope: tsecon.var_irf_bands(
            y,
            lags=2,
            horizon=12,
            band="sup-t",
            band_scope=scope,
            band_n_sim=40_000,
        )
        for scope in ("horizon", "shock", "all")
    }
    assert out["horizon"]["n_cells"] == 13
    assert out["shock"]["n_cells"] == 26
    assert out["all"]["n_cells"] == 52
    cvs = [np.asarray(out[s]["critical_value"])[1, 1] for s in ("horizon", "shock", "all")]
    assert cvs[0] < cvs[1] < cvs[2], cvs
    assert all(np.asarray(o["critical_value"]).shape == (2, 2) for o in out.values())


def test_var_forecast_scope_all_is_wider_than_scope_horizon():
    y = _var_panel()
    per_series = tsecon.var_forecast(
        y, lags=2, steps=12, band="sup-t", band_scope="horizon", band_n_sim=40_000
    )
    joint = tsecon.var_forecast(
        y, lags=2, steps=12, band="sup-t", band_scope="all", band_n_sim=40_000
    )
    assert per_series["n_cells"] == 12
    assert joint["n_cells"] == 24
    assert all(
        j > h for j, h in zip(joint["critical_value"], per_series["critical_value"])
    )
    # scope="all" is one family, so every series shares one multiplier.
    assert len(set(joint["critical_value"])) == 1


# ---------------------------------------------------------------------------
# Every simultaneous result must report its scope and K
# ---------------------------------------------------------------------------

def test_var_irf_reports_scope_k_and_cells_used():
    y = _var_panel()
    r = tsecon.var_irf_bands(
        y, lags=2, horizon=6, band="sidak", band_scope="shock"
    )
    assert r["band"] == "sidak"
    assert r["band_scope"] == "shock"
    assert r["n_cells"] == 14  # k * (horizon + 1) = 2 * 7
    used = np.asarray(r["n_cells_used"])
    # One cell of each shock family is pinned to zero by the Cholesky ordering
    # at h = 0, so the band is simultaneous over fewer cells than it looks.
    assert used.shape == (2, 2)
    assert (used <= r["n_cells"]).all() and (used > 0).all()
    assert r["pointwise_critical_value"] == pytest.approx(1.6449, abs=1e-3)


def test_var_forecast_reports_scope_k_and_cells_used():
    y = _var_panel()
    r = tsecon.var_forecast(y, lags=2, steps=9, band="bonferroni")
    assert r["band"] == "bonferroni"
    assert r["band_scope"] == "all"
    assert r["n_cells"] == 18
    assert list(r["n_cells_used"]) == [18, 18]
    assert len(r["critical_value"]) == 2


def test_lp_reports_scope_k_and_seed_provenance():
    y, shock = _lp_series()
    r = tsecon.lp(
        y, shock, horizons=9, n_lag_controls=4, band="sup-t", band_n_sim=20_000
    )
    assert r["band"] == "sup-t"
    assert r["band_scope"] == "horizon"
    assert r["n_cells"] == 10 and r["n_cells_used"] == 10
    assert r["band_n_sim"] == 20_000
    # Closed forms need no simulation, and say so.
    closed = tsecon.lp(y, shock, horizons=9, n_lag_controls=4, band="sidak")
    assert closed["band_n_sim"] == 0


# ---------------------------------------------------------------------------
# Seeds: the sup-t-from-covariance route is a pure function of its seed
# ---------------------------------------------------------------------------

def test_lp_sup_t_is_a_pure_function_of_the_seed():
    y, shock = _lp_series()
    kw = dict(horizons=10, n_lag_controls=4, band="sup-t", band_n_sim=20_000)
    a = tsecon.lp(y, shock, band_seed=4242, **kw)
    b = tsecon.lp(y, shock, band_seed=4242, **kw)
    c = tsecon.lp(y, shock, band_seed=99, **kw)
    assert a["critical_value"] == b["critical_value"]
    assert np.array_equal(a["lower"], b["lower"])
    assert a["band_seed"] == 4242
    # A different seed differs only by simulation noise in a tail quantile.
    assert c["critical_value"] != a["critical_value"]
    assert c["critical_value"] == pytest.approx(a["critical_value"], rel=0.05)


def test_var_sup_t_routes_are_reproducible_from_their_seeds():
    y = _var_panel()
    kw = dict(lags=2, horizon=8, band="sup-t", band_n_sim=20_000)
    a = tsecon.var_irf_bands(y, band_seed=7, **kw)
    b = tsecon.var_irf_bands(y, band_seed=7, **kw)
    c = tsecon.var_irf_bands(y, band_seed=8, **kw)
    assert np.array_equal(np.asarray(a["sim_lower"]), np.asarray(b["sim_lower"]))
    assert not np.array_equal(
        np.asarray(a["critical_value"]), np.asarray(c["critical_value"])
    )

    f1 = tsecon.var_forecast(y, lags=2, steps=8, band="sup-t", band_seed=3, band_n_sim=20_000)
    f2 = tsecon.var_forecast(y, lags=2, steps=8, band="sup-t", band_seed=3, band_n_sim=20_000)
    assert f1["critical_value"] == f2["critical_value"]


def test_bootstrap_sup_t_is_reproducible_from_the_bootstrap_seed_alone():
    """On the bootstrap branch the replications ARE the draws, so `seed`
    reproduces the band and `band_seed` is inert."""
    y = _var_panel()
    kw = dict(lags=2, horizon=6, method="bootstrap", n_boot=300, band="sup-t")
    a = tsecon.var_irf_bands(y, seed=5, band_seed=1, **kw)
    b = tsecon.var_irf_bands(y, seed=5, band_seed=1, **kw)
    c = tsecon.var_irf_bands(y, seed=5, band_seed=987_654, **kw)
    d = tsecon.var_irf_bands(y, seed=6, band_seed=1, **kw)
    assert np.array_equal(np.asarray(a["sim_lower"]), np.asarray(b["sim_lower"]))
    assert np.array_equal(
        np.asarray(a["critical_value"]), np.asarray(c["critical_value"])
    )
    assert not np.array_equal(
        np.asarray(a["critical_value"]), np.asarray(d["critical_value"])
    )


def test_closed_form_routes_need_no_seed():
    y, shock = _lp_series()
    kw = dict(horizons=8, n_lag_controls=4, band="bonferroni")
    a = tsecon.lp(y, shock, band_seed=1, **kw)
    b = tsecon.lp(y, shock, band_seed=123_456_789, **kw)
    assert a["critical_value"] == b["critical_value"]
    assert np.array_equal(a["lower"], b["lower"])


# ---------------------------------------------------------------------------
# Bad input is refused by name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda y: tsecon.var_irf_bands(y, lags=2, horizon=4, band="sup t"),
        lambda y: tsecon.var_irf_bands(y, lags=2, horizon=4, band="SUP-T"),
        lambda y: tsecon.var_forecast(y, lags=2, steps=4, band="joint"),
    ],
)
def test_unknown_band_raises_naming_every_accepted_value(call):
    y = _var_panel(n=80)
    with pytest.raises(ValueError) as exc:
        call(y)
    msg = str(exc.value)
    for accepted in ("pointwise", "sup-t", "sidak", "bonferroni"):
        assert accepted in msg


def test_unknown_band_raises_on_the_lp_surfaces():
    y, shock = _lp_series(n=200)
    with pytest.raises(ValueError, match="unknown band"):
        tsecon.lp(y, shock, horizons=4, n_lag_controls=2, band="supt-ish")


@pytest.mark.parametrize("scope", ["horizons", "shocks", "everything", "series"])
def test_unknown_band_scope_raises_even_at_the_pointwise_default(scope):
    """A typo in the scope must not be silently ignored: a band whose scope is
    ambiguous is worse than no band."""
    y = _var_panel(n=80)
    with pytest.raises(ValueError, match="band_scope"):
        tsecon.var_irf_bands(y, lags=2, horizon=4, band_scope=scope)
    with pytest.raises(ValueError, match="band_scope"):
        tsecon.var_forecast(y, lags=2, steps=4, band_scope=scope)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_band_alpha_is_validated(bad):
    y, shock = _lp_series(n=200)
    with pytest.raises(ValueError, match="band_alpha"):
        tsecon.lp(y, shock, horizons=4, n_lag_controls=2, band="sidak", band_alpha=bad)


@pytest.mark.parametrize("fn_name", ["lp_iv", "lp_multiplier", "lp_state"])
def test_sup_t_is_refused_where_no_cross_horizon_covariance_exists(fn_name):
    """lp_iv, lp_multiplier and lp_state get Sidak/Bonferroni ONLY. Refusing by
    name is the point: a sup-t number here would be fabricated."""
    y, shock, inst = _lp_iv_series(n=300)
    st = (np.arange(len(y)) % 5 < 2).astype(float)
    calls = {
        "lp_iv": lambda b: tsecon.lp_iv(
            y, shock, inst, horizons=6, n_lag_controls=4, band=b
        ),
        "lp_multiplier": lambda b: tsecon.lp_multiplier(
            y, shock, inst, horizons=6, n_lag_controls=4, band=b
        ),
        "lp_state": lambda b: tsecon.lp_state(
            y, shock, st, horizons=6, n_lag_controls=4, band=b
        ),
    }
    call = calls[fn_name]
    with pytest.raises(ValueError) as exc:
        call("sup-t")
    msg = str(exc.value)
    assert fn_name in msg
    assert "covariance" in msg  # says WHY, not just "unsupported"
    assert "sidak" in msg and "bonferroni" in msg  # says what to use instead
    # The closed-form routes are genuinely available on the same call.
    assert call("bonferroni")["band"] == "bonferroni"


def test_smooth_lp_does_offer_sup_t():
    """Smooth LP already holds the whole-path covariance, so it is NOT in the
    closed-form-only list."""
    y, shock = _lp_series()
    r = tsecon.smooth_lp(
        y,
        shock,
        horizons=10,
        n_lag_controls=4,
        lam=1.0,
        band="sup-t",
        band_n_sim=20_000,
    )
    assert r["band"] == "sup-t"
    assert r["critical_value"] > r["pointwise_critical_value"]
    assert r["n_cells"] == 11


# ---------------------------------------------------------------------------
# Spelling aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spelling", ["sup-t", "supt", "sup_t"])
def test_sup_t_spellings_are_equivalent_and_echo_canonically(spelling):
    y, shock = _lp_series()
    r = tsecon.lp(
        y,
        shock,
        horizons=8,
        n_lag_controls=4,
        band=spelling,
        band_n_sim=20_000,
    )
    assert r["band"] == "sup-t"


def test_facade_and_module_supt_defaults_agree():
    """The two documented routes to the same sup-t band must return the
    same critical values at their defaults. The facade once defaulted
    band_seed=0 against the module's 20260807, so identical inputs gave
    different bands by route (audit round 5)."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal((200, 2)).cumsum(axis=0) * 0.1 + rng.standard_normal((200, 2))
    mod = tsecon.var_irf_bands(y, lags=2, horizon=8, method="asymptotic", band="sup-t")
    res = tsecon.results.var_fit(y, lags=2).irf_bands(
        horizon=8, method="asymptotic", band="sup-t"
    )
    np.testing.assert_array_equal(np.asarray(mod["lower"]), np.asarray(res["lower"]))
    np.testing.assert_array_equal(np.asarray(mod["upper"]), np.asarray(res["upper"]))
