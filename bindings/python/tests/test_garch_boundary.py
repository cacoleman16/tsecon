"""Boundary-aware GARCH standard errors (audit round 7, fixing round 1).

Round 1 found ``garch_fit`` returning a silent all-NaN ``se_mle``/``se_robust``
row whenever a dimensionless coefficient sat at its constraint boundary
(10/120 probe units): the finite-difference Hessian probe crossed the
constraint, the whole covariance computation errored, and the fit swallowed
that into unflagged NaNs. The fix computes a reduced Hessian over the interior
directions and reports per-parameter ``se_valid``/``boundary`` flags plus a
``boundary_note`` teaching string (the ``tsecon-evt`` ``se_valid`` precedent),
so a NaN standard error is now always a *flagged* statement.
"""

import numpy as np
import pytest

import tsecon


def _sim_garch(omega, alpha, beta, n, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 200)
    s2 = omega / max(1e-12, (1 - alpha - beta)) if alpha + beta < 1 else omega * 100
    y = np.empty(n + 200)
    for t in range(n + 200):
        y[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * y[t] ** 2 + beta * s2
    return y[200:]


# --------------------------------------------------------------------- #
# the round-1 reproduction: white noise drives alpha to its sign bound
# --------------------------------------------------------------------- #
def test_alpha_at_sign_bound_is_flagged_with_interior_ses():
    # Seed 2 reproduces the round-1 silent-NaN unit: alpha -> ~1e-14.
    y = _sim_garch(1.0, 0.0, 0.0, 750, 2)
    r = tsecon.garch_fit(y, vol="garch", mean="zero", dist="normal", p=1, q=1)
    names = list(r["param_names"])
    p = dict(zip(names, r["params"]))
    assert p["alpha[1]"] < 1e-6, "the boundary reproduction has drifted"

    bnd = dict(zip(names, r["boundary"]))
    valid = dict(zip(names, r["se_valid"]))
    se_r = dict(zip(names, r["se_robust"]))
    se_m = dict(zip(names, r["se_mle"]))

    # The boundary parameter: flagged, NaN, invalid.
    assert bnd["alpha[1]"] and not valid["alpha[1]"]
    assert np.isnan(se_r["alpha[1]"]) and np.isnan(se_m["alpha[1]"])
    # The interior parameter omega: finite standard errors survive — the
    # round-1 defect was exactly this row coming back NaN.
    assert not bnd["omega"] and valid["omega"]
    assert np.isfinite(se_r["omega"]) and se_r["omega"] > 0
    assert np.isfinite(se_m["omega"]) and se_m["omega"] > 0
    # The teaching note names the flagged parameter and the cause.
    note = r["boundary_note"]
    assert note is not None
    assert "alpha[1]" in note and "sign constraint" in note
    assert "se_valid" in note


def test_no_silent_nan_over_boundary_attracted_battery():
    """Every NaN standard error is flagged: (isnan & se_valid) is empty,
    and any boundary fit carries a note. 20 boundary-attracted units."""
    cases = [("igarch", 0.02, 0.10, 0.90), ("tinyalpha", 0.05, 0.005, 0.90)]
    flagged = 0
    for tag, om, al, be in cases:
        for seed in range(10):
            y = _sim_garch(om, al, be, 750, seed)
            r = tsecon.garch_fit(y, vol="garch", mean="zero", dist="normal")
            nan_any = np.isnan(r["se_mle"]) | np.isnan(r["se_robust"])
            valid = np.asarray(r["se_valid"])
            assert not (nan_any & valid).any(), (tag, seed, "silent NaN")
            if np.asarray(r["boundary"]).any():
                flagged += 1
                assert r["boundary_note"] is not None, (tag, seed)
    assert flagged >= 5, "the boundary battery no longer reaches boundaries"


def test_igarch_boundary_flags_coefficients_and_names_igarch():
    y = _sim_garch(0.02, 0.10, 0.90, 750, 1)  # alpha+beta=1 in the DGP
    r = tsecon.garch_fit(y, vol="garch", mean="zero", dist="normal")
    names = list(r["param_names"])
    pers = sum(v for n, v in zip(names, r["params"]) if n != "omega")
    if pers < 0.9995:
        pytest.skip("this draw was not attracted to the persistence bound")
    bnd = dict(zip(names, r["boundary"]))
    assert bnd["alpha[1]"] and bnd["beta[1]"]
    assert "IGARCH" in r["boundary_note"]


# --------------------------------------------------------------------- #
# interior fits: the flags must not cry wolf
# --------------------------------------------------------------------- #
def test_interior_fit_flags_are_clean_and_converged():
    y = _sim_garch(0.05, 0.08, 0.88, 1000, 0)
    r = tsecon.garch_fit(y, vol="garch", mean="constant", dist="normal")
    assert all(r["se_valid"])
    assert not any(r["boundary"])
    assert r["boundary_note"] is None
    assert r["converged"] is True
    assert np.isfinite(r["se_robust"]).all() and np.isfinite(r["se_mle"]).all()


def test_results_facade_summary_carries_the_note():
    from tsecon.results import GARCHResults

    y = _sim_garch(1.0, 0.0, 0.0, 750, 2)  # the boundary reproduction
    res = GARCHResults.fit(y, vol="garch", mean="zero", dist="normal")
    if not any(res["boundary"]):
        pytest.skip("this draw was not attracted to a boundary")
    s = res.summary()
    assert "Boundary fit" in s
    assert max(len(line) for line in s.splitlines()) <= 72

    # And an interior fit's summary stays note-free.
    y2 = _sim_garch(0.05, 0.08, 0.88, 1000, 0)
    s2 = GARCHResults.fit(y2, vol="garch", mean="zero", dist="normal").summary()
    assert "Boundary fit" not in s2


def test_new_keys_present_for_every_family():
    y = _sim_garch(0.05, 0.08, 0.88, 600, 3)
    for kw in (
        dict(vol="garch", mean="zero", dist="normal"),
        dict(vol="gjr", mean="zero", dist="normal", o=1),
        dict(vol="egarch", mean="zero", dist="normal", o=1),
        dict(vol="garch", mean="constant", dist="t"),
    ):
        r = tsecon.garch_fit(y, **kw)
        k = len(r["params"])
        assert len(r["se_valid"]) == k and len(r["boundary"]) == k
        assert np.asarray(r["se_valid"]).dtype == np.bool_
        assert np.asarray(r["boundary"]).dtype == np.bool_
        assert isinstance(r["converged"], bool)
        assert r["boundary_note"] is None or isinstance(r["boundary_note"], str)


# --------------------------------------------------------------------- #
# 0.6.0: o can no longer be silently discarded under vol="garch"
# (the arch porting trap: arch_model(y, p=1, o=1, q=1) IS GJR there)
# --------------------------------------------------------------------- #
def test_o_with_vol_garch_raises_and_teaches():
    y = _sim_garch(0.05, 0.08, 0.88, 400, 5)
    with pytest.raises(ValueError, match=r'vol="gjr"') as exc:
        tsecon.garch_fit(y, vol="garch", p=1, o=1, q=1)
    msg = str(exc.value)
    assert "no effect" in msg or "silently" in msg
    assert "arch_model" in msg  # names the porting trap it guards
    with pytest.raises(ValueError, match=r'vol="gjr"'):
        tsecon.garch_fit(y, vol="garch", o=2)


def test_o_zero_and_default_are_unchanged_for_garch():
    y = _sim_garch(0.05, 0.08, 0.88, 400, 5)
    default = tsecon.garch_fit(y, vol="garch", mean="zero", dist="normal")
    explicit0 = tsecon.garch_fit(y, vol="garch", mean="zero", dist="normal", o=0)
    np.testing.assert_array_equal(default["params"], explicit0["params"])
    assert default["loglik"] == explicit0["loglik"]
    assert list(default["param_names"]) == ["omega", "alpha[1]", "beta[1]"]


def test_gjr_default_o_is_one_asymmetry_lag():
    # The None sentinel keeps the old default (o=1) for the asymmetric vols.
    y = _sim_garch(0.05, 0.08, 0.88, 400, 5)
    default = tsecon.garch_fit(y, vol="gjr", mean="zero", dist="normal")
    explicit1 = tsecon.garch_fit(y, vol="gjr", mean="zero", dist="normal", o=1)
    np.testing.assert_array_equal(default["params"], explicit1["params"])
    assert "gamma[1]" in list(default["param_names"])
