"""The LP integral-multiplier path, and the trap it exists to close.

`cumulative=True` cumulates ONLY the outcome: the response becomes
`sum_j y_{t+j}` while the impulse stays contemporaneous. That is a legitimate
cumulative impulse response, but it is NOT a multiplier — its denominator
never accumulates, so it inherits the growth of the cumulated impulse instead
of measuring a per-unit effect. `tsecon.lp_multiplier` accumulates both sides and is the
estimator a multiplier question actually wants.

These tests pin the distinction so it cannot silently regress:

1. outcome-only and both-sides cumulation give *different* answers, and the
   outcome-only one climbs with the horizon while the multiplier stays flat;
2. the both-sides IV estimate equals the hand-computed ratio of the two
   cumulative reduced forms (the just-identified IV algebra), computed here in
   plain numpy against the same control set;
3. `cumulative="outcome"` reproduces the legacy `cumulative=True` bitwise, so
   the historical spelling has not silently changed meaning.
"""
import numpy as np
import pytest

import tsecon


def fiscal_system(n=400, seed=11):
    """Spending, output and a news instrument with a KNOWN unit multiplier.

    Output responds one-for-one to government spending contemporaneously, so
    the integral multiplier is 1.0 at every horizon by construction. A
    confounder moves both series, which is what the instrument is for.
    Spending is deliberately persistent (rho = 0.95), as government spending
    is in the data: that is what makes cumulated spending keep growing with
    the horizon, and therefore what makes the outcome-only trap visible.
    """
    rng = np.random.default_rng(seed)
    news = rng.standard_normal(n)
    confound = rng.standard_normal(n)
    g = np.zeros(n)
    for t in range(2, n):
        # Spending responds to the news shock with a hump, plus the confounder.
        g[t] = 0.95 * g[t - 1] + news[t] + 0.8 * news[t - 1] + 0.6 * confound[t]
    y = g + 0.6 * confound + 0.3 * rng.standard_normal(n)
    return y, g, news


def _cumulate(x, h):
    """`sum_{j=0..h} x_{t+j}`, truncated to the usable sample."""
    return np.array([x[t : t + h + 1].sum() for t in range(len(x) - h)])


def _reduced_form_coef(target, instrument, exog):
    """OLS coefficient on `instrument` in `target ~ exog + instrument`."""
    design = np.column_stack(exog + [instrument])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return beta[-1]


def _cumulative_impulse_path(y, g, news, hmax=16):
    """The cumulative spending path — the denominator the trap is missing."""
    return tsecon.lp_multiplier(y, g, news, horizons=hmax, n_lag_controls=4)[
        "cumulative_impulse"
    ]


def test_outcome_only_cumulation_is_not_a_multiplier():
    """The trap: cumulating only the outcome is not a per-unit effect."""
    y, g, news = fiscal_system()
    hmax = 16

    trap = np.asarray(
        tsecon.lp_iv(y, g, news, horizons=hmax, n_lag_controls=4, cumulative=True)["irf"]
    )
    mult = np.asarray(
        tsecon.lp_multiplier(y, g, news, horizons=hmax, n_lag_controls=4)["multiplier"]
    )

    # They are different estimators and must not be confused for one another.
    assert not np.allclose(trap[4:], mult[4:], rtol=0.05)

    # The trap keeps climbing with the horizon: its numerator accumulates and
    # its denominator does not, so it inherits the growth of cumulated
    # spending rather than measuring a per-dollar effect.
    assert np.all(np.diff(trap[2:]) > 0.0)
    assert trap[16] > 2.0 * trap[4]

    # The true integral multiplier is 1.0 by construction, and the one-step
    # estimator stays flat and near it while the trap doubles.
    assert np.allclose(mult[4:], 1.0, atol=0.15)
    assert abs(mult[16] / mult[4] - 1.0) < 0.10

    # And the trap is exactly what its name says: cumulated output per unit of
    # CONTEMPORANEOUS spending, so it moves with the cumulated spending path.
    cum_g = np.asarray(_cumulative_impulse_path(y, g, news))
    assert trap[16] / trap[4] == pytest.approx(cum_g[16] / cum_g[4], rel=0.35)


def test_multiplier_equals_ratio_of_cumulative_reduced_forms():
    """Just-identified 2SLS == ratio of the two reduced forms, exactly.

    Hand-computed in numpy over the same sample and the same control set
    (constant, 4 lags of y, 4 lags of g) that `lp_multiplier` uses.
    """
    y, g, news = fiscal_system()
    p, hmax = 4, 12
    r = tsecon.lp_multiplier(y, g, news, horizons=hmax, n_lag_controls=p)

    n = len(y)
    for h in (0, 3, 8, 12):
        # Sample: t from p to n-1-h, matching the crate's horizon_sample.
        lo, hi = p, n - h
        cum_y = _cumulate(y, h)[lo:]
        cum_g = _cumulate(g, h)[lo:]
        z = news[lo:hi]
        exog = [np.ones(hi - lo)]
        exog += [y[lo - lag : hi - lag] for lag in range(1, p + 1)]
        exog += [g[lo - lag : hi - lag] for lag in range(1, p + 1)]

        ratio = _reduced_form_coef(cum_y, z, exog) / _reduced_form_coef(cum_g, z, exog)
        assert r["multiplier"][h] == pytest.approx(ratio, rel=1e-9)

        # The crate reports the same two legs it divided.
        assert r["cumulative_outcome"][h] == pytest.approx(
            _reduced_form_coef(cum_y, z, exog), rel=1e-9
        )
        assert r["cumulative_impulse"][h] == pytest.approx(
            _reduced_form_coef(cum_g, z, exog), rel=1e-9
        )


def test_legacy_cumulative_true_is_unchanged():
    """`True` still means outcome-only cumulation, bitwise."""
    y, g, news = fiscal_system()
    for kwargs_a, kwargs_b in (
        ({"cumulative": True}, {"cumulative": "outcome"}),
        ({"cumulative": False}, {"cumulative": "none"}),
        ({}, {"cumulative": False}),
    ):
        a = tsecon.lp(y, news, horizons=8, n_lag_controls=4, **kwargs_a)
        b = tsecon.lp(y, news, horizons=8, n_lag_controls=4, **kwargs_b)
        assert np.array_equal(a["irf"], b["irf"])
        assert np.array_equal(a["se"], b["se"])

        a = tsecon.lp_iv(y, g, news, horizons=8, n_lag_controls=4, **kwargs_a)
        b = tsecon.lp_iv(y, g, news, horizons=8, n_lag_controls=4, **kwargs_b)
        assert np.array_equal(a["irf"], b["irf"])


def test_cumulative_both_mode_differs_from_outcome_mode():
    """`"both"` is reachable from lp/lp_iv too, and is a different object."""
    y, g, news = fiscal_system()
    out = tsecon.lp_iv(y, g, news, horizons=10, n_lag_controls=4, cumulative="outcome")
    both = tsecon.lp_iv(y, g, news, horizons=10, n_lag_controls=4, cumulative="both")
    assert not np.allclose(out["irf"][2:], both["irf"][2:], rtol=0.05)
    # lp_iv(cumulative="both") controls only y lags; lp_multiplier also
    # controls impulse lags, so they are close but not identical estimators.
    assert np.allclose(both["irf"][4:], 1.0, atol=0.25)

    lvl = tsecon.lp(y, news, horizons=10, n_lag_controls=4)
    cum = tsecon.lp(y, news, horizons=10, n_lag_controls=4, cumulative="both")
    assert not np.allclose(lvl["irf"][2:], cum["irf"][2:], rtol=0.05)


def test_bad_cumulative_spelling_is_rejected():
    y, g, news = fiscal_system(n=120)
    with pytest.raises(ValueError, match="unknown cumulative"):
        tsecon.lp(y, news, horizons=4, cumulative="cumulative")
    with pytest.raises(ValueError, match="lp_multiplier"):
        tsecon.lp_iv(y, g, news, horizons=4, cumulative="yes")


def test_multiplier_se_and_first_stage_f_are_reported():
    y, g, news = fiscal_system()
    r = tsecon.lp_multiplier(y, g, news, horizons=12, n_lag_controls=4)
    for key in ("horizons", "multiplier", "se", "first_stage_f",
                "cumulative_outcome", "cumulative_impulse", "nobs_per_h"):
        assert key in r
        assert len(r[key]) == 13
    assert np.all(r["se"] > 0.0)
    # Strong instrument by construction.
    assert np.all(r["first_stage_f"][:8] > 10.0)
    # The SE is the multiplier's own: 1.0 sits inside a 95% band everywhere.
    lo = r["multiplier"] - 1.96 * r["se"]
    hi = r["multiplier"] + 1.96 * r["se"]
    assert np.all((lo <= 1.0) & (1.0 <= hi))
