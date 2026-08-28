"""The trend-cycle BN build-out: hamilton_filter extensions (RW variant +
HAC standard errors), the classic ``bn_decomposition``, and the
Kamber-Morley-Wong ``bn_filter``.

Golden re-pins come from ``fixtures/bn_filters.json``:

* ``hamilton_hac`` — statsmodels ``OLS(...).fit(cov_type="HAC")`` on the
  identical design (an independent-package golden: the filter is OLS);
* ``bn_arma`` — the documented Morley-2002 companion-form transcription,
  with psi(1) pinned to statsmodels' cumulative ``arma_impulse_response``;
* ``kmw`` — reference runs of the authors' own R replication code.

Plus: the defaults-unchanged guarantee, the exact BN identities, the
RW-variant equivalence, and error surfaces.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"


def _fx(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def bn_fx():
    return _fx("bn_filters.json")


@pytest.fixture(scope="module")
def gdp():
    return np.asarray(_fx("filters.json")["y_100_log_realgdp"])


@pytest.fixture(scope="module")
def sim(bn_fx):
    return np.asarray(bn_fx["sim_series"])


# --------------------------------------------------------------------------- #
# hamilton_filter: defaults bit-identical, RW variant, HAC inference
# --------------------------------------------------------------------------- #
def test_hamilton_defaults_bit_identical_across_new_surfaces(gdp):
    base = tsecon.hamilton_filter(gdp)
    explicit = tsecon.hamilton_filter(gdp, h=8, p=4, method="regression")
    with_se = tsecon.hamilton_filter(gdp, se="hac")
    assert sorted(base.keys()) == ["beta", "cycle", "first_index", "trend"]
    for key in ("beta", "cycle", "trend"):
        np.testing.assert_array_equal(base[key], explicit[key])
        # Requesting inference must not perturb the filter itself: the
        # decomposition and coefficients are the same BITS.
        np.testing.assert_array_equal(base[key], with_se[key])
    assert base["first_index"] == 11  # h + p - 1
    # And the old fixture still pins the default numbers.
    old = _fx("filters.json")["hamilton_h8_p4"]
    np.testing.assert_allclose(base["beta"], old["beta"], rtol=1e-8, atol=1e-8)


def test_hamilton_hac_matches_statsmodels(bn_fx, gdp):
    block = bn_fx["hamilton_hac"]
    # h-overlap default bandwidth: maxlags resolves to h = 8.
    r = tsecon.hamilton_filter(gdp, se="hac")
    assert r["se_type"] == "hac"
    assert r["maxlags"] == 8
    assert r["use_correction"] is True
    np.testing.assert_allclose(r["bse"], block["hac_h8_corr"]["bse"], rtol=1e-6)
    np.testing.assert_allclose(r["tvalues"], block["hac_h8_corr"]["tvalues"], rtol=1e-6)

    r = tsecon.hamilton_filter(gdp, se="hac", use_correction=False)
    np.testing.assert_allclose(r["bse"], block["hac_h8_nocorr"]["bse"], rtol=1e-6)

    r = tsecon.hamilton_filter(gdp, se="hac", maxlags=4)
    assert r["maxlags"] == 4
    np.testing.assert_allclose(r["bse"], block["hac_l4_corr"]["bse"], rtol=1e-6)

    r = tsecon.hamilton_filter(gdp, se="nonrobust")
    assert r["se_type"] == "nonrobust"
    assert r["maxlags"] is None
    np.testing.assert_allclose(r["bse"], block["nonrobust"]["bse"], rtol=1e-6)
    np.testing.assert_allclose(r["tvalues"], block["nonrobust"]["tvalues"], rtol=1e-6)


def test_hamilton_random_walk_is_the_h_difference_exactly(gdp):
    r = tsecon.hamilton_filter(gdp, h=8, method="random_walk")
    assert r["first_index"] == 8
    assert "beta" not in r
    np.testing.assert_array_equal(r["cycle"], gdp[8:] - gdp[:-8])
    np.testing.assert_array_equal(r["trend"], gdp[:-8])


def test_hamilton_rw_equivalence_on_a_pure_random_walk():
    # On a pure random walk the RW variant IS the population filter:
    # cycle_t = y_t - y_{t-h} for every t, exactly, whatever the draw.
    rng = np.random.default_rng(42)
    y = np.cumsum(0.3 + rng.standard_normal(300))
    r = tsecon.hamilton_filter(y, h=8, method="random_walk")
    np.testing.assert_array_equal(r["cycle"], y[8:] - y[:-8])
    # And the regression variant's slope on the first lag is near 1,
    # intercept near the 8-period drift: the population values.
    reg = tsecon.hamilton_filter(y, h=8, p=4)
    assert abs(reg["beta"][1] - 1.0) < 0.35
    assert abs(np.sum(reg["beta"][1:]) - 1.0) < 0.1  # lag weights sum ~ 1


def test_hamilton_error_surfaces(gdp):
    with pytest.raises(ValueError, match="random_walk"):
        tsecon.hamilton_filter(gdp, method="random_walk", se="hac")
    with pytest.raises(ValueError, match="regression"):
        tsecon.hamilton_filter(gdp, method="banana")
    with pytest.raises(ValueError, match="hac"):
        tsecon.hamilton_filter(gdp, se="banana")
    with pytest.raises(ValueError, match="maxlags"):
        tsecon.hamilton_filter(gdp, maxlags=8)  # bandwidth without se="hac"


# --------------------------------------------------------------------------- #
# bn_decomposition: goldens, identities, closed forms
# --------------------------------------------------------------------------- #
def _series_for(case_name, gdp, sim):
    return gdp if case_name.startswith("gdp") else sim


@pytest.mark.parametrize("case_name", ["gdp_arima212", "sim_arma11_fixed", "sim_ar2_fixed"])
def test_bn_fixed_coefficients_golden(bn_fx, gdp, sim, case_name):
    case = bn_fx["bn_arma"][case_name]
    y = _series_for(case_name, gdp, sim)
    r = tsecon.bn_decomposition(
        y, ar=case["ar"], ma=case["ma"], drift=case["drift"]
    )
    assert r["mode"] == "fixed"
    np.testing.assert_allclose(r["trend"], case["trend"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(r["cycle"], case["cycle"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        r["innovations"], case["innovations"], rtol=1e-8, atol=1e-8
    )
    # psi(1): the closed form, and statsmodels' cumulative impulse response.
    assert r["long_run_multiplier"] == pytest.approx(
        case["long_run_multiplier"], rel=1e-10
    )
    assert r["long_run_multiplier"] == pytest.approx(
        case["long_run_multiplier_sm_cum_irf"], rel=1e-7
    )


def test_bn_identities_hold_exactly(bn_fx, gdp):
    case = bn_fx["bn_arma"]["gdp_arima212"]
    r = tsecon.bn_decomposition(gdp, ar=case["ar"], ma=case["ma"], drift=case["drift"])
    assert r["first_index"] == 1
    # trend + cycle recovers y up to at most one final rounding (trend is
    # stored as y - cycle; observed exact on this series).
    np.testing.assert_allclose(r["trend"] + r["cycle"], gdp[1:], rtol=1e-15, atol=0)
    # The BN trend is a random walk with drift in the series' own shocks.
    np.testing.assert_allclose(
        np.diff(r["trend"]),
        r["drift"] + r["long_run_multiplier"] * r["innovations"][1:],
        rtol=0,
        atol=1e-9,
    )
    # Trend increments are white-ish: innovations of a well-specified fit
    # carry no strong serial correlation (loose sanity band, not a test of
    # the ARMA fit itself).
    d = np.diff(r["trend"])
    ac1 = np.corrcoef(d[1:], d[:-1])[0, 1]
    assert abs(ac1) < 0.25


def test_bn_ima11_textbook_closed_form(gdp):
    theta = 0.4
    r = tsecon.bn_decomposition(gdp, ar=[], ma=[theta], drift=0.8)
    assert r["long_run_multiplier"] == 1.0 + theta
    np.testing.assert_allclose(
        r["cycle"], -theta * r["innovations"], rtol=1e-13, atol=1e-13
    )


def test_bn_fit_path_on_gdp(bn_fx, gdp):
    case = bn_fx["bn_arma"]["gdp_arima212"]
    r = tsecon.bn_decomposition(gdp)  # MNZ default p=2, q=2, library MLE
    assert r["mode"] == "fit"
    assert {"sigma2", "loglik", "aic", "bic", "converged"} <= set(r.keys())
    np.testing.assert_allclose(r["trend"] + r["cycle"], gdp[1:], rtol=1e-15, atol=0)
    # Two optimizers' stopping points: psi(1) and drift land near the
    # statsmodels MLE of the same spec (measured ~1.4e-4 / ~7e-6).
    assert r["long_run_multiplier"] == pytest.approx(
        case["long_run_multiplier"], abs=0.05
    )
    assert r["drift"] == pytest.approx(case["drift"], abs=0.02)


def test_bn_error_surfaces(gdp):
    with pytest.raises(ValueError, match="stationary|unit circle"):
        tsecon.bn_decomposition(gdp, ar=[1.05], drift=0.0)
    with pytest.raises(ValueError, match="invertible|unit circle"):
        tsecon.bn_decomposition(gdp, ma=[-1.2], drift=0.0)
    with pytest.raises(ValueError):
        tsecon.bn_decomposition(np.array([1.0]))  # nothing to difference


# --------------------------------------------------------------------------- #
# bn_filter (KMW): golden re-pins, identities, the amplitude contrast
# --------------------------------------------------------------------------- #
def _kmw_call(y, case):
    kwargs = {"p": case["p"], "demean": "sm" if case["demean"] else "nd"}
    if case["delta_mode"] == "fixed":
        kwargs["delta"] = case["delta"]
    return tsecon.bn_filter(y, **kwargs)


@pytest.mark.parametrize(
    "case_name",
    ["usgdp_p12_auto_sm", "usgdp_p12_fixed025_sm", "sim_p12_auto_sm", "sim_p8_fixed005_nd"],
)
def test_kmw_reference_run_golden(bn_fx, gdp, sim, case_name):
    case = bn_fx["kmw"][case_name]
    y = _series_for(case_name.replace("usgdp", "gdp"), gdp, sim)
    r = _kmw_call(y, case)
    # Auto selection lands on the same 0.0005-spaced grid point as the
    # authors' R code.
    assert r["delta"] == pytest.approx(case["delta"], abs=1e-12)
    np.testing.assert_allclose(r["cycle"], case["cycle"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(r["ar"], case["ar"], rtol=1e-8, atol=1e-8)
    assert r["cycle_se"] == pytest.approx(case["cycle_se"], rel=1e-8)
    assert r["amplitude_to_noise"] == pytest.approx(case["amp_to_noise"], rel=1e-8)
    assert r["drift"] == pytest.approx(case["drift"], abs=1e-12)
    assert r["first_index"] == 1
    # trend is stored as y - cycle, so (trend + cycle) recovers y up to at
    # most one final rounding (observed: exact on 3 of 4 cases, 1 ulp on
    # one element of the 900-level sim series).
    np.testing.assert_allclose(r["trend"] + r["cycle"], y[1:], rtol=1e-15, atol=0)


def test_kmw_ar_sums_to_rho(gdp):
    r = tsecon.bn_filter(gdp, p=12, delta=0.25)
    rho = 1.0 - 1.0 / np.sqrt(0.25)
    assert np.sum(r["ar"]) == pytest.approx(rho, abs=1e-12)


def test_kmw_large_gap_vs_classic_tiny_cycle(sim):
    # The KMW headline: on a drifting series with a persistent cycle the
    # classic freely-estimated BN attributes nearly everything to trend;
    # the pinned-delta filter recovers the large gap (measured ratio ~38x
    # on this fixture series in the Rust tests).
    kmw = tsecon.bn_filter(sim, p=12)
    classic = tsecon.bn_decomposition(sim, p=2, q=0)
    assert np.var(kmw["cycle"]) > 3.0 * np.var(classic["cycle"])
    # And the KMW gap is persistent, not noise.
    c = kmw["cycle"]
    assert np.corrcoef(c[1:], c[:-1])[0, 1] > 0.7


def test_kmw_error_surfaces(gdp):
    with pytest.raises(ValueError, match="p"):
        tsecon.bn_filter(gdp, p=1)
    with pytest.raises(ValueError, match="sm"):
        tsecon.bn_filter(gdp, demean="banana")
    with pytest.raises(ValueError, match="delta"):
        tsecon.bn_filter(gdp, delta=-0.5)
    with pytest.raises(ValueError, match="27"):
        tsecon.bn_filter(gdp[:20], p=12)  # needs 2p + 3 = 27
    bad = gdp.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.bn_filter(bad)


def test_coercion_accepts_lists_and_float32(gdp):
    r64 = tsecon.bn_filter(gdp, p=4, delta=0.25)
    r32 = tsecon.bn_filter(gdp.astype(np.float32), p=4, delta=0.25)
    assert r32["delta"] == r64["delta"]  # same call, degraded input precision
    rl = tsecon.bn_decomposition(list(map(float, gdp[:80])), ar=[0.3], ma=[0.2], drift=0.5)
    assert rl["mode"] == "fixed"
    # hamilton with pandas-like input is covered by the coercion suite;
    # here just check a python list works end to end.
    hl = tsecon.hamilton_filter(list(map(float, gdp)), se="hac")
    assert hl["maxlags"] == 8


# --------------------------------------------------------------------------- #
# statsmodels absence canary (the fixture pins it; re-check live)
# --------------------------------------------------------------------------- #
def test_statsmodels_reference_canary(bn_fx):
    # The fixture pins what was true AT GENERATION TIME (statsmodels 0.14.x,
    # see _meta): no runnable Hamilton or BN reference existed, which is why
    # those goldens are formula transcriptions. That provenance stays pinned:
    assert bn_fx["statsmodels_absence_canary"] == {
        "hamilton": True,
        "beveridge_nelson": True,
    }
    sm = pytest.importorskip("statsmodels.api")
    import statsmodels.tsa.filters as smf

    # BN decomposition: still no statsmodels implementation.
    assert not any(
        "beveridge" in x.lower() or x.lower() == "bn" for x in dir(sm.tsa)
    )

    # Hamilton: the canary fired — statsmodels 0.15.0 added
    # tsa.filters.api.hamilton_filter. When the installed version has it,
    # the absence claim is retired IN FAVOR OF the thing the canary was
    # waiting for: a live third-party cross-check (measured 4.2e-14 max
    # abs on first contact; asserted at 1e-10). On older statsmodels the
    # original absence assertion still holds.
    if any("hamilton" in x.lower() for x in dir(smf)):
        from statsmodels.tsa.filters.api import hamilton_filter as sm_ham

        rng = np.random.default_rng(20260828)
        y = np.cumsum(rng.standard_normal(300)) + 0.05 * np.arange(300)
        ours = tsecon.hamilton_filter(y)  # defaults h=8, p=4
        cycle_sm, trend_sm = sm_ham(y, 8, 4)
        cycle_sm, trend_sm = np.asarray(cycle_sm), np.asarray(trend_sm)
        valid = ~np.isnan(cycle_sm)
        assert valid.sum() == len(ours["cycle"])
        np.testing.assert_allclose(
            np.asarray(ours["cycle"]), cycle_sm[valid], rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(ours["trend"]), trend_sm[valid], rtol=0, atol=1e-10
        )
    else:
        assert not any("hamilton" in x.lower() for x in dir(smf))


# --------------------------------------------------------------------------- #
# Audit round 10: inert HAC/grid kwargs now raise
# --------------------------------------------------------------------------- #
def test_hamilton_maxlags_refused_under_every_non_hac_path(gdp):
    """maxlags is a HAC bandwidth. It was refused under se=None but silently
    swallowed under se="nonrobust" (the returned maxlags key was even None);
    the same guard now covers both non-HAC paths, with the same message."""
    with pytest.raises(ValueError, match="maxlags is a HAC bandwidth"):
        tsecon.hamilton_filter(gdp, maxlags=8)
    with pytest.raises(ValueError, match="maxlags is a HAC bandwidth"):
        tsecon.hamilton_filter(gdp, se="nonrobust", maxlags=8)
    # Still live where documented (already pinned above against statsmodels:
    # test_hamilton_hac_matches_statsmodels's maxlags=4 arm).
    r4 = tsecon.hamilton_filter(gdp, se="hac", maxlags=4)
    r8 = tsecon.hamilton_filter(gdp, se="hac")
    assert r4["maxlags"] == 4 and r8["maxlags"] == 8
    assert not np.array_equal(r4["bse"], r8["bse"])


def test_hamilton_use_correction_refused_where_inert(gdp):
    """use_correction is the HAC n/(n-k) factor; explicit use under se=None,
    se="nonrobust", or method="random_walk" raises instead of being
    silently swallowed."""
    for kwargs in (dict(), dict(se="nonrobust")):
        with pytest.raises(ValueError, match="use_correction") as exc:
            tsecon.hamilton_filter(gdp, use_correction=False, **kwargs)
        assert "se='hac'" in str(exc.value)
    with pytest.raises(ValueError, match="use_correction"):
        tsecon.hamilton_filter(gdp, method="random_walk", use_correction=True)


def test_hamilton_use_correction_sentinel_default_bit_identical(gdp):
    """The sentinel (use_correction=None -> True where HAC applies) keeps
    the default HAC call bit-identical to explicit True, and distinct from
    False (the live check)."""
    d = tsecon.hamilton_filter(gdp, se="hac")
    t = tsecon.hamilton_filter(gdp, se="hac", use_correction=True)
    f = tsecon.hamilton_filter(gdp, se="hac", use_correction=False)
    np.testing.assert_array_equal(d["bse"], t["bse"])
    assert d["use_correction"] is True
    assert not np.array_equal(d["bse"], f["bse"])
    # The decomposition itself never moves with the SE options.
    np.testing.assert_array_equal(d["cycle"], f["cycle"])


def test_kmw_grid_kwargs_refused_under_fixed_delta(gdp):
    """d0/dt lay out the automatic-selection grid; a fixed delta= never
    builds it, so explicit d0/dt raise (they were verified bit-identical
    no-ops before the fix)."""
    for kwargs in (dict(d0=0.5), dict(dt=0.1), dict(d0=0.5, dt=0.1)):
        with pytest.raises(ValueError, match="d0/dt") as exc:
            tsecon.bn_filter(gdp, delta=0.25, **kwargs)
        assert "amplitude-to-noise" in str(exc.value)
    # Sentinel defaults resolve to the historical grid: explicit 0.01/0.0005
    # under auto-selection is bit-identical to the default call.
    a = tsecon.bn_filter(gdp, p=8)
    b = tsecon.bn_filter(gdp, p=8, d0=0.01, dt=0.0005)
    assert a["delta"] == b["delta"]
    np.testing.assert_array_equal(a["cycle"], b["cycle"])
    # And d0/dt stay live under auto-selection: a coarser grid moves the
    # selected delta off the fine grid's stopping point.
    c = tsecon.bn_filter(gdp, p=8, d0=0.05, dt=0.05)
    assert c["delta"] != a["delta"]
