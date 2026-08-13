"""Python-level regression tests for proxy_svar_bands and proxy_ar_sets.

The numerical validation (the Jentsch-Lunsford moving-block transcription, the
brute-force grid inversion of the AR statistic, the Monte Carlo coverage runs)
lives in the tsecon-var and tsecon-ident crate suites. These tests pin the
things a Python caller can be silently wrong about, each written as the
assertion that would fail if the feature were broken:

* the h=0 degeneracy tell — the normalized cell is pinned at `unit` in every
  draw, which is only true if the renormalization is *inside* the bootstrap
  loop.  A non-degenerate value there means it was hoisted out;
* Hall and Efron are both returned and are genuinely different bands (Hall is
  the reflection of Efron about the point estimate);
* `bands="wild"` self-reports as not asymptotically valid, so nobody quotes it
  as inference by accident;
* the failure accounting adds up, so no draw is silently dropped;
* a NaN-prefixed proxy is dropped *by date*, not compacted to the head of the
  residual sample — the misalignment that would follow is invisible in the
  output but wrecks the identification;
* every AR set kind is one of the documented strings, and the point estimate is
  a member of its own set.
"""
import numpy as np
import pytest
import tsecon

A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
KINDS = {"interval", "exterior", "whole", "empty", "point", "ray_below", "ray_above"}
FAILURE_KEYS = {
    "too_few_proxy_obs",
    "zero_proxy_variance",
    "near_zero_gamma_norm",
    "refit_failed",
    "identification_failed",
    "non_finite",
}


def _proxy_var(n=180, seed=0, strength=1.0, noise=0.5, shock=0):
    """Stable VAR(1) DGP plus a proxy for the `shock`-th structural innovation.

    `strength` scales the signal in the instrument; small values give the weak
    instrument the AR sets are for.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 3))
    e = rng.standard_normal((n, 3))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + e[t]
    proxy = strength * e[:, shock] + noise * rng.standard_normal(n)
    return y, proxy


DATA, PROXY = _proxy_var()
H = 6


# --------------------------------------------------------------------------- #
# proxy_svar_bands — the degeneracy tell
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bands", ["moving_block", "wild"])
@pytest.mark.parametrize("norm_var,unit", [(0, 1.0), (0, -1.0), (1, -1.5), (2, 2.5)])
def test_h0_norm_var_band_is_exactly_degenerate_at_unit(bands, norm_var, unit):
    """THE tell: the (h=0, norm_var) cell is [unit, unit] with zero width in
    both the Hall and the Efron band.

    The normalization is re-imposed inside every bootstrap draw, so that cell
    takes the value `unit` in all n_boot replications and every quantile of it
    is `unit`. If the normalization were computed once and hoisted out of the
    loop, the draws would scatter around `unit` and this band would open up.
    Exact equality is the right assertion — there is no arithmetic here to
    lose precision to.
    """
    r = tsecon.proxy_svar_bands(
        DATA, PROXY, lags=2, horizon=H, n_boot=150, seed=4,
        bands=bands, norm_var=norm_var, unit=unit,
    )
    for key in ("point", "lower", "upper", "lower_efron", "upper_efron"):
        got = np.asarray(r[key])[0, norm_var]
        assert got == unit, f"{key}[0, {norm_var}] = {got!r}, expected exactly {unit!r}"
    assert np.asarray(r["se"])[0, norm_var] == 0.0
    # ...and the band is not collapsed everywhere, which would make the above
    # pass for the wrong reason.
    width = np.asarray(r["upper"]) - np.asarray(r["lower"])
    assert (width > 0).any(), "every cell is degenerate — the bootstrap did nothing"


def test_band_shapes_and_diagnostics_contract():
    r = tsecon.proxy_svar_bands(DATA, PROXY, lags=2, horizon=H, n_boot=150, seed=4)
    for key in ("point", "lower", "upper", "lower_efron", "upper_efron", "se"):
        assert np.asarray(r[key]).shape == (H + 1, 3), key
    assert np.all(np.asarray(r["se"]) >= 0.0)
    assert np.all(np.asarray(r["upper"]) >= np.asarray(r["lower"]) - 1e-12)
    assert np.all(np.asarray(r["upper_efron"]) >= np.asarray(r["lower_efron"]) - 1e-12)
    assert r["n_proxy"] == len(DATA) - 2          # the residual sample
    assert r["block_length"] >= 1
    assert r["alpha"] == pytest.approx(0.10)
    assert r["method"] == "moving_block"
    assert r["point_first_stage_f"] > 0.0
    assert 0.0 < r["point_reliability"] <= 1.0


# --------------------------------------------------------------------------- #
# proxy_svar_bands — Hall vs Efron
# --------------------------------------------------------------------------- #
def test_hall_and_efron_are_both_returned_and_differ():
    """Both bands ship, and they are not the same numbers. Returning one under
    two names would silently hide the skew that motivates reporting both."""
    r = tsecon.proxy_svar_bands(DATA, PROXY, lags=2, horizon=H, n_boot=200, seed=11)
    hall_lo, hall_hi = np.asarray(r["lower"]), np.asarray(r["upper"])
    efr_lo, efr_hi = np.asarray(r["lower_efron"]), np.asarray(r["upper_efron"])
    assert not np.array_equal(hall_lo, efr_lo)
    assert not np.array_equal(hall_hi, efr_hi)
    # Hall/basic is the Efron percentile band reflected about the point estimate:
    # a skewed bootstrap distribution therefore pushes them in opposite
    # directions, which is the whole reason the choice matters.
    point = np.asarray(r["point"])
    np.testing.assert_allclose(hall_lo, 2.0 * point - efr_hi, atol=1e-12)
    np.testing.assert_allclose(hall_hi, 2.0 * point - efr_lo, atol=1e-12)


# --------------------------------------------------------------------------- #
# proxy_svar_bands — the validity flag on the wild bootstrap
# --------------------------------------------------------------------------- #
def test_wild_bootstrap_flags_itself_as_not_asymptotically_valid():
    """The wild branch exists to reproduce published Mertens-Ravn /
    Gertler-Karadi bands; it must say so rather than pass as inference."""
    w = tsecon.proxy_svar_bands(DATA, PROXY, lags=2, horizon=H, n_boot=100,
                               seed=5, bands="wild")
    assert w["method"] == "wild"
    assert w["asymptotically_valid"] is False
    assert isinstance(w["validity_note"], str) and w["validity_note"].strip()


def test_moving_block_is_asymptotically_valid_and_mbb_is_an_alias():
    mb = tsecon.proxy_svar_bands(DATA, PROXY, lags=2, horizon=H, n_boot=100, seed=5)
    assert mb["asymptotically_valid"] is True
    alias = tsecon.proxy_svar_bands(DATA, PROXY, lags=2, horizon=H, n_boot=100,
                                   seed=5, bands="mbb")
    assert alias["method"] == "moving_block"
    assert np.array_equal(np.asarray(mb["lower"]), np.asarray(alias["lower"]))


def test_unknown_bands_value_names_both_accepted_values():
    with pytest.raises(ValueError) as exc:
        tsecon.proxy_svar_bands(DATA, PROXY, lags=2, horizon=2, n_boot=20, bands="bogus")
    msg = str(exc.value)
    assert "moving_block" in msg and "wild" in msg, msg


# --------------------------------------------------------------------------- #
# proxy_svar_bands — reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_reproduces_bands_bit_for_bit():
    kw = dict(lags=2, horizon=H, n_boot=150, seed=17)
    a = tsecon.proxy_svar_bands(DATA, PROXY, **kw)
    b = tsecon.proxy_svar_bands(DATA, PROXY, **kw)
    for key in ("point", "lower", "upper", "lower_efron", "upper_efron", "se"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key


def test_different_seed_gives_different_bands():
    kw = dict(lags=2, horizon=H, n_boot=150)
    a = tsecon.proxy_svar_bands(DATA, PROXY, seed=17, **kw)
    b = tsecon.proxy_svar_bands(DATA, PROXY, seed=18, **kw)
    # the point path is the sample estimate and must not move with the seed...
    assert np.array_equal(np.asarray(a["point"]), np.asarray(b["point"]))
    # ...but the resampled band must.
    assert not np.array_equal(np.asarray(a["lower"]), np.asarray(b["lower"]))
    assert not np.array_equal(np.asarray(a["se"]), np.asarray(b["se"]))


# --------------------------------------------------------------------------- #
# proxy_svar_bands — failure accounting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strength", [1.0, 0.05])
def test_failure_counters_are_complete_and_account_for_every_draw(strength):
    """Six named reasons, and every draw is in exactly one bucket. Dropping a
    failed draw without counting it would quietly trim the near-zero-denominator
    tail and shrink the interval."""
    data, proxy = _proxy_var(n=150, seed=2, strength=strength)
    r = tsecon.proxy_svar_bands(data, proxy, lags=2, horizon=4, n_boot=150, seed=8)
    assert set(r["failures"]) == FAILURE_KEYS
    assert r["n_failed"] == sum(r["failures"].values())
    assert r["n_used"] + r["n_failed"] == r["n_boot"] == 150
    assert r["n_used"] > 0
    if r["n_failed"] == 0:
        assert r["failure_warning"] is None
    else:
        assert isinstance(r["failure_warning"], str) and r["failure_warning"].strip()


def test_draw_diagnostics_align_and_rho_is_normalized_in_every_draw():
    """The per-draw diagnostics are all indexed by surviving draw, so they must
    share a length; and because the normalization is re-imposed per draw,
    rho[norm_var] is exactly 1 in every one of them."""
    for norm_var in (0, 1):
        r = tsecon.proxy_svar_bands(DATA, PROXY, lags=2, horizon=4, n_boot=150,
                                   seed=6, norm_var=norm_var)
        gamma = np.asarray(r["gamma_norm_draws"])
        rho = np.asarray(r["rho_draws"])
        assert gamma.shape == (r["n_used"],)
        assert rho.shape == (r["n_used"], 3)
        assert np.asarray(r["first_stage_f_draws"]).shape == (r["n_used"],)
        assert np.asarray(r["reliability_draws"]).shape == (r["n_used"],)
        assert np.all(rho[:, norm_var] == 1.0), "renormalization not applied per draw"
        # gamma_norm is a covariance, so it is signed; a draw that survived is
        # one whose denominator was not near zero.
        assert np.all(np.isfinite(rho)) and np.all(np.abs(gamma) > 0.0)


# --------------------------------------------------------------------------- #
# proxy_svar_bands — NaN handling must not compact the proxy
# --------------------------------------------------------------------------- #
def test_nan_prefix_proxy_is_dropped_by_date_not_compacted():
    """A narrative instrument that only starts partway through the sample is
    passed as a NaN prefix. Those dates must be dropped where they sit.

    If the implementation instead compacted the proxy — stripped the NaNs and
    lined the survivors up against the *first* residuals — the instrument would
    be silently shifted by the length of the prefix and every response would be
    wrong, with nothing in the output to show for it.
    """
    n, m, lags = 220, 40, 2
    data, full = _proxy_var(n=n, seed=1)
    prefixed = full.copy()
    prefixed[:m] = np.nan
    kw = dict(lags=lags, horizon=4, n_boot=100, seed=3)

    padded = tsecon.proxy_svar_bands(data, prefixed, **kw)
    # the same instrument handed over only its available window: the proxy may
    # be passed at the residual-sample length instead of the full length.
    windowed = tsecon.proxy_svar_bands(data, prefixed[lags:], **kw)

    assert padded["n_proxy"] == windowed["n_proxy"] == n - m
    for key in ("point", "lower", "upper", "lower_efron", "upper_efron", "se"):
        assert np.array_equal(np.asarray(padded[key]), np.asarray(windowed[key])), key

    # Decisive check: the identical 180 values moved to the *front* of the
    # residual sample. A compacting implementation cannot tell this apart from
    # the NaN-prefixed version, because it would use the same vector either way.
    shifted = np.full(n, np.nan)
    shifted[lags:lags + (n - m)] = full[m:]
    moved = tsecon.proxy_svar_bands(data, shifted, **kw)
    assert moved["n_proxy"] == padded["n_proxy"]      # same count, different dates
    assert not np.allclose(np.asarray(moved["point"]), np.asarray(padded["point"]))


# --------------------------------------------------------------------------- #
# proxy_ar_sets — set shapes
# --------------------------------------------------------------------------- #
def _all_cells(res):
    return [(h, j, c) for h, row in enumerate(res["cells"]) for j, c in enumerate(row)]


AR_DESIGNS = [
    ("strong", dict(n=180, seed=0, strength=1.0, noise=0.5)),
    ("moderate", dict(n=150, seed=0, strength=0.08, noise=1.0)),
    ("weak", dict(n=150, seed=1, strength=0.02, noise=1.0)),
]


@pytest.mark.parametrize("label,spec", AR_DESIGNS, ids=[d[0] for d in AR_DESIGNS])
@pytest.mark.parametrize("rfu", [True, False])
def test_every_cell_kind_is_a_documented_string(label, spec, rfu):
    data, proxy = _proxy_var(**spec)
    r = tsecon.proxy_ar_sets(data, proxy, lags=2, horizon=4,
                             reduced_form_uncertainty=rfu)
    assert np.asarray(r["impact"]).shape == (3,)
    assert len(r["cells"]) == 5 and all(len(row) == 3 for row in r["cells"])
    for h, j, c in _all_cells(r):
        assert c["kind"] in KINDS, (h, j, c["kind"])
        # `bounded` is exactly "both endpoints are finite"
        assert c["bounded"] == (np.isfinite(c["lower"]) and np.isfinite(c["upper"]))
        # the excluded region is reported only where it means something
        if c["kind"] != "exterior":
            assert c["excluded_lower"] is None and c["excluded_upper"] is None
        if c["kind"] in ("interval", "point"):
            assert c["excludes_zero"] == (not (c["lower"] <= 0.0 <= c["upper"]))


@pytest.mark.parametrize("unit", [1.0, -1.0])
def test_norm_var_h0_cell_is_a_point_at_exactly_unit(unit):
    """The normalized impact response is `unit` by construction, so under
    identification the AR set for that cell collapses to the single point."""
    r = tsecon.proxy_ar_sets(DATA, PROXY, lags=2, horizon=4, norm_var=0, unit=unit)
    c = r["cells"][0][0]
    assert c["kind"] == "point"
    assert c["lower"] == unit and c["upper"] == unit and c["point"] == unit
    assert c["bounded"] is True


@pytest.mark.parametrize("label,spec", AR_DESIGNS, ids=[d[0] for d in AR_DESIGNS])
def test_norm_var_h0_cell_is_never_empty_and_always_contains_unit(label, spec):
    """The discriminant is exactly zero for that cell, so the only two correct
    answers are {unit} and the whole line — never the empty set, which is what
    a naive `A > 0 and D <= 0 => empty` branch would report for the impact
    response of the very variable the caller normalized on."""
    data, proxy = _proxy_var(**spec)
    for unit in (1.0, -1.0):
        r = tsecon.proxy_ar_sets(data, proxy, lags=2, horizon=4, norm_var=0, unit=unit)
        c = r["cells"][0][0]
        assert c["kind"] in ("point", "whole"), (label, unit, c["kind"])
        assert c["point"] == unit
        assert _contains(c, unit)


def test_level_is_a_number_only_when_reduced_form_uncertainty_is_propagated():
    """A set conditional on the estimated reduced form carries no honest
    1-alpha label, so `level` must be None there rather than a comforting
    0.95 the coverage numbers do not support."""
    on = tsecon.proxy_ar_sets(DATA, PROXY, lags=2, horizon=4, alpha=0.05,
                              reduced_form_uncertainty=True)
    off = tsecon.proxy_ar_sets(DATA, PROXY, lags=2, horizon=4, alpha=0.05,
                               reduced_form_uncertainty=False)
    assert isinstance(on["level"], float) and on["level"] == pytest.approx(0.95)
    assert on["reduced_form_uncertainty"] is True
    assert off["level"] is None
    assert off["reduced_form_uncertainty"] is False


def _find_exterior():
    for _, spec in AR_DESIGNS:
        for rfu in (True, False):
            data, proxy = _proxy_var(**spec)
            r = tsecon.proxy_ar_sets(data, proxy, lags=2, horizon=6,
                                     reduced_form_uncertainty=rfu)
            for h, j, c in _all_cells(r):
                if c["kind"] == "exterior":
                    return h, j, c
    return None


def test_exterior_set_is_two_rays_not_an_interval():
    """An exterior set is the COMPLEMENT of (excluded_lower, excluded_upper).
    Reading its lower/upper as an interval would turn "the data reject this
    middle region" into "the response is unbounded", which is backwards."""
    found = _find_exterior()
    if found is None:
        pytest.skip("no exterior AR set arose in the scanned designs")
    h, j, c = found
    assert c["lower"] == -np.inf and c["upper"] == np.inf, (h, j, c)
    assert c["excluded_lower"] < c["excluded_upper"]
    assert c["bounded"] is False
    # the excluded region is what the data reject, so the point estimate — which
    # satisfies the moment exactly — cannot be inside it
    assert not (c["excluded_lower"] < c["point"] < c["excluded_upper"])


# --------------------------------------------------------------------------- #
# proxy_ar_sets — the cheapest correctness invariant
# --------------------------------------------------------------------------- #
def _contains(cell, x, tol=1e-9):
    """Is `x` a member of the set this cell describes?"""
    kind = cell["kind"]
    if kind == "empty":
        return False
    if kind == "whole":
        return True
    if kind == "exterior":
        return x <= cell["excluded_lower"] + tol or x >= cell["excluded_upper"] - tol
    # interval / point / ray_below / ray_above all read off lower..upper,
    # with the open side carrying an infinity.
    return cell["lower"] - tol <= x <= cell["upper"] + tol


@pytest.mark.parametrize("label,spec", AR_DESIGNS, ids=[d[0] for d in AR_DESIGNS])
@pytest.mark.parametrize("rfu", [True, False])
def test_point_estimate_is_always_a_member_of_its_own_set(label, spec, rfu):
    """The AR statistic is zero at the point estimate — it satisfies the moment
    condition exactly — so the point can never be rejected and must lie in its
    own set, whatever shape that set takes. An empty set, or a set that excludes
    its own point, means the inversion picked up a sign or a root the wrong way
    round."""
    data, proxy = _proxy_var(**spec)
    for horizon in (4, 6):
        r = tsecon.proxy_ar_sets(data, proxy, lags=2, horizon=horizon,
                                 reduced_form_uncertainty=rfu)
        for h, j, c in _all_cells(r):
            assert _contains(c, c["point"]), (label, rfu, h, j, c)


def test_hac_lags_without_hac_variance_raises():
    """`hac_lags` parameterizes only the HAC moment variance; under the
    default `variance="hc0"` it used to be a silent no-op (bit-identical
    output with and without it -- audit round 5). It must raise instead."""
    with pytest.raises(ValueError, match='variance="hac"'):
        tsecon.proxy_ar_sets(DATA, PROXY, lags=2, horizon=4, hac_lags=8)
    # The HAC route accepts it, and it is live there.
    base = tsecon.proxy_ar_sets(DATA, PROXY, lags=2, horizon=4, variance="hac")
    with_lags = tsecon.proxy_ar_sets(DATA, PROXY, lags=2, horizon=4,
                                     variance="hac", hac_lags=2)
    assert base["ar_bound_stat"] != with_lags["ar_bound_stat"]
