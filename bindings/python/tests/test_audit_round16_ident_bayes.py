"""Regression pins for adversarial audit round 16: the structural-
identification and Bayesian families.

Each test pins one confirmed finding (or the share of a house-wide finding)
that belongs to these callables:

* A1 / A4 / A6 / A8 -- a horizon, draw, replicate, chain, grid or weight-draw
  count of 2**31 or 2**40 used to abort the interpreter (an infallible
  allocation) or hang; every one is now refused by name, in-process, before
  anything is allocated, quoting the largest value that fits;
* B2 / B3 / B4 -- refusals that named an engine variable, blamed the wrong
  cause, or named no argument at all;
* SVF-4 / H16-07 / API16-5 -- an unknown `identification` with `impact=`;
* SVF-5 -- the pandas float cast that truncated complex frames;
* H16-09 / API16-2 / API16-8 -- `structural_fevd`'s `sigma` and `lags=0`;
* H16-12 -- extreme-scale impact matrices.

Every refusal below happens before an allocation is attempted, so these run
in-process on every platform (the counts, not the allocator, decide).
"""
import time

import numpy as np
import pytest

import tsecon

BIG = 2**31
HUGE = 2**40


def _var(n=200, k=3, seed=0):
    rng = np.random.default_rng(seed)
    a = 0.4 * np.eye(k) + 0.05
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.standard_normal(k)
    return y


X3 = _var()
X2 = _var(k=2, seed=1)
SR3 = [(0, 0, 0, "+"), (1, 0, 0, "+")]
NR = [{"type": "shock_sign", "shock": 0, "period": 10, "sign": "+"}]
PROXY = X3[:, 0] + np.random.default_rng(3).standard_normal(200)
REGIMES = np.where(np.arange(200) < 100, 0, 1)
XH = X2 * np.where(REGIMES[:, None] == 1, 2.0, 1.0) * np.array([1.0, 0.5])


def _nongauss():
    rng = np.random.default_rng(5)
    e = np.column_stack([rng.laplace(size=200), rng.standard_t(4, size=200)])
    return e + 0.3 * e[:, ::-1]


def _panel():
    rng = np.random.default_rng(6)
    f = np.cumsum(rng.standard_normal(80)) * 0.1
    return np.column_stack([f + rng.standard_normal(80) for _ in range(6)])


def _refused(call, arg, value):
    """The call is refused by name before allocating, quoting the largest
    admissible value; returns the message."""
    t0 = time.perf_counter()
    with pytest.raises(ValueError) as info:
        call()
    msg = str(info.value)
    assert f"{arg} = {value} is too large" in msg, msg
    assert "before anything is allocated" in msg, msg
    assert time.perf_counter() - t0 < 30, "the refusal must come before the work"
    return msg


# --------------------------------------------------------------------------
# A1: horizon budgets
# --------------------------------------------------------------------------
HORIZON_CALLS = {
    "bvar_irf_draws": lambda h: tsecon.bvar_irf_draws(X3, lags=2, horizon=h, n_draws=50),
    "bvar_ssvs": lambda h: tsecon.bvar_ssvs(X2, lags=1, n_draws=200, burn=50, horizon=h),
    "sign_restricted_svar": lambda h: tsecon.sign_restricted_svar(
        X3, SR3, lags=1, horizon=h, n_draws=30, max_tries=100
    ),
    "narrative_svar": lambda h: tsecon.narrative_svar(
        X3, SR3, lags=1, horizon=h, n_draws=20, max_tries=100
    ),
    "zero_sign_svar": lambda h: tsecon.zero_sign_svar(
        X3, SR3, [(2, 0, 0)], lags=1, horizon=h, n_draws=20, max_tries=100
    ),
    "fry_pagan_svar": lambda h: tsecon.fry_pagan_svar(
        X3, SR3, lags=1, horizon=h, n_draws=30, max_tries=100
    ),
    "robust_svar_bounds": lambda h: tsecon.robust_svar_bounds(
        X3, SR3, lags=1, horizon=h, n_draws=30
    ),
    "historical_decomposition": lambda h: tsecon.historical_decomposition(
        X3, SR3, lags=1, horizon=h, identification="sign", n_draws=20, max_tries=100
    ),
    "hetero_svar": lambda h: tsecon.hetero_svar(XH, REGIMES, lags=1, horizon=h),
    "max_share_svar": lambda h: tsecon.max_share_svar(X3, lags=1, h1=8, horizon=h),
    "proxy_svar": lambda h: tsecon.proxy_svar(X3, PROXY, lags=2, horizon=h),
    "proxy_ar_sets": lambda h: tsecon.proxy_ar_sets(X3, PROXY, lags=2, horizon=h),
    "proxy_svar_bands": lambda h: tsecon.proxy_svar_bands(
        X3, PROXY, lags=2, horizon=h, n_boot=30, block_length=5
    ),
    "favar": lambda h: tsecon.favar(_panel(), _var(80, 1, 9)[:, 0], n_factors=1, lags=1, horizon=h),
    "nongaussian_svar": lambda h: tsecon.nongaussian_svar(_nongauss(), lags=1, horizon=h),
    "structural_fevd": lambda h: tsecon.structural_fevd(X2, lags=2, horizon=h),
    "structural_fevd(impact)": lambda h: tsecon.structural_fevd(
        X2, lags=2, horizon=h, impact=np.eye(2)
    ),
}


@pytest.mark.parametrize("name", sorted(HORIZON_CALLS))
@pytest.mark.parametrize("h", [BIG, HUGE])
def test_an_absurd_horizon_is_refused_by_name_before_allocating(name, h):
    msg = _refused(lambda: HORIZON_CALLS[name](h), "horizon", h)
    assert "2^28-value (2 GiB)" in msg, msg
    assert "pass horizon <= " in msg, msg


def test_the_quoted_largest_horizon_is_the_budget_boundary():
    # structural_fevd's working set is (horizon + 1) * (2 mat + list) values,
    # with mat(2) = 8 * 2 + 8 = 24 and the list 5 * 4 + 22 + 3 = 45 doubles:
    # 93 per horizon, so the largest admissible horizon is 2^28 // 93 - 1.
    msg = _refused(lambda: tsecon.structural_fevd(X2, lags=1, horizon=BIG), "horizon", BIG)
    assert f"pass horizon <= {2**28 // 93 - 1} " in msg, msg


def test_ordinary_horizons_are_untouched():
    out = tsecon.structural_fevd(X2, lags=2, horizon=40)
    assert np.asarray(out["fevd"]).shape == (41, 2, 2)
    assert len(tsecon.proxy_svar(X3, PROXY, lags=2, horizon=120)["irf"]) == 121


def test_the_proxy_ar_delta_method_work_is_capped_by_name():
    # The delta-method Jacobians are quadratic in horizon; a horizon the
    # memory budget admits is refused before it runs for hours.
    msg = _refused(lambda: tsecon.proxy_ar_sets(X3, PROXY, lags=2, horizon=200_000), "horizon", 200_000)
    assert "2^32 multiply-add work budget" in msg, msg


# --------------------------------------------------------------------------
# A4: draw / replicate / chain counts, named before any stream is spawned
# --------------------------------------------------------------------------
DRAW_CALLS = {
    ("bvar_irf_draws", "n_draws"): lambda d: tsecon.bvar_irf_draws(X3, lags=2, horizon=8, n_draws=d),
    ("bvar_ssvs", "n_draws"): lambda d: tsecon.bvar_ssvs(X2, lags=1, n_draws=d, burn=50, horizon=4),
    ("bvar_ssvs", "n_chains"): lambda d: tsecon.bvar_ssvs(
        X2, lags=1, n_draws=200, burn=50, horizon=4, n_chains=d
    ),
    ("sign_restricted_svar", "n_draws"): lambda d: tsecon.sign_restricted_svar(
        X3, SR3, lags=1, horizon=4, n_draws=d, max_tries=100
    ),
    ("narrative_svar", "n_draws"): lambda d: tsecon.narrative_svar(
        X3, SR3, lags=1, horizon=4, n_draws=d, max_tries=100
    ),
    ("zero_sign_svar", "n_draws"): lambda d: tsecon.zero_sign_svar(
        X3, SR3, [(2, 0, 0)], lags=1, horizon=4, n_draws=d, max_tries=100
    ),
    ("fry_pagan_svar", "n_draws"): lambda d: tsecon.fry_pagan_svar(
        X3, SR3, lags=1, horizon=4, n_draws=d, max_tries=100
    ),
    ("robust_svar_bounds", "n_draws"): lambda d: tsecon.robust_svar_bounds(
        X3, SR3, lags=1, horizon=4, n_draws=d
    ),
    ("historical_decomposition", "n_draws"): lambda d: tsecon.historical_decomposition(
        X3, SR3, lags=1, identification="sign", n_draws=d, max_tries=100
    ),
    ("proxy_svar_bands", "n_boot"): lambda d: tsecon.proxy_svar_bands(
        X3, PROXY, lags=2, horizon=6, n_boot=d, block_length=5
    ),
    ("proxy_ar_sets", "rf_draws"): lambda d: tsecon.proxy_ar_sets(
        X3, PROXY, lags=2, horizon=6, rf_method="second_order", rf_draws=d, rf_seed=1
    ),
}


@pytest.mark.parametrize("key", sorted(DRAW_CALLS))
@pytest.mark.parametrize("d", [BIG, HUGE])
def test_an_absurd_draw_count_is_refused_by_name_before_any_stream_is_spawned(key, d):
    name, arg = key
    msg = _refused(lambda: DRAW_CALLS[key](d), arg, d)
    assert f"pass {arg} <= " in msg, msg
    # Never the RNG layer's unnamed spawn-limit text.
    assert "spawn" not in msg and "random-stream" not in msg, msg


# --------------------------------------------------------------------------
# A6: bvar_hierarchical n_grid
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_grid", [10_001, BIG, HUGE])
def test_bvar_hierarchical_grid_is_capped_by_name(n_grid):
    msg = _refused(lambda: tsecon.bvar_hierarchical(X3, lags=2, n_grid=n_grid), "n_grid", n_grid)
    assert "pass n_grid <= 10000" in msg, msg


# --------------------------------------------------------------------------
# A8: the narrative weight loop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [BIG, HUGE])
@pytest.mark.parametrize("fn", ["narrative_svar", "historical_decomposition"])
def test_an_absurd_weight_draw_count_is_refused_instead_of_hanging(fn, k):
    if fn == "narrative_svar":
        call = lambda: tsecon.narrative_svar(  # noqa: E731
            X3, SR3, NR, lags=1, horizon=4, n_draws=20, max_tries=200, n_weight_draws=k
        )
    else:
        call = lambda: tsecon.historical_decomposition(  # noqa: E731
            X3, SR3, lags=1, identification="sign", narrative_restrictions=NR,
            n_draws=20, max_tries=200, n_weight_draws=k,
        )
    msg = _refused(call, "n_weight_draws", k)
    assert "2^36 multiply-add work budget" in msg, msg
    assert "pass n_weight_draws <= " in msg, msg


@pytest.mark.parametrize("fn", ["narrative_svar", "historical_decomposition"])
def test_n_weight_draws_without_narrative_restrictions_is_refused_as_inert(fn):
    if fn == "narrative_svar":
        call = lambda **kw: tsecon.narrative_svar(  # noqa: E731
            X3, SR3, lags=1, horizon=4, n_draws=20, max_tries=100, **kw
        )
    else:
        call = lambda **kw: tsecon.historical_decomposition(  # noqa: E731
            X3, SR3, lags=1, identification="sign", n_draws=20, max_tries=100, **kw
        )
    for nr in (None, []):
        with pytest.raises(ValueError, match=r"n_weight_draws is only read under narrative_restrictions"):
            call(narrative_restrictions=nr, n_weight_draws=50)
    # The default (and passing it explicitly) is the weight-1 run, unchanged.
    base = call()
    same = call(n_weight_draws=200)
    assert base["weights"] == same["weights"] and all(w == 1.0 for w in base["weights"])
    # Live where documented: with narrative restrictions it is read.
    live = tsecon.narrative_svar(X3, SR3, NR, lags=1, horizon=4, n_draws=20, max_tries=200, n_weight_draws=50)
    assert "weights" in live


# --------------------------------------------------------------------------
# B2: refusals name the Python keyword, not the engine's variable
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call, expect, never",
    [
        (lambda: tsecon.sign_restricted_svar(X3, SR3, lags=1, horizon=4, n_draws=0), "n_draws must be at least 1", "n_posterior_draws"),
        (lambda: tsecon.narrative_svar(X3, SR3, lags=1, horizon=4, n_draws=0), "n_draws must be at least 1", "n_posterior_draws"),
        (lambda: tsecon.zero_sign_svar(X3, SR3, [(2, 0, 0)], lags=1, horizon=4, n_draws=0), "n_draws must be at least 1", "n_posterior_draws"),
        (lambda: tsecon.fry_pagan_svar(X3, SR3, lags=1, horizon=4, n_draws=0), "n_draws must be at least 1", "n_posterior_draws"),
        (lambda: tsecon.historical_decomposition(X3, SR3, lags=1, identification="sign", n_draws=0), "n_draws must be at least 1", "n_posterior_draws"),
        (lambda: tsecon.sign_restricted_svar(X3, SR3, lags=1, horizon=4, max_tries=0), "max_tries must be at least 1", "max_tries_per_draw"),
        (lambda: tsecon.bvar_ssvs(X2, lags=0, n_draws=200, burn=50), "lags = 0", "lag length p"),
        (lambda: tsecon.nongaussian_svar(_nongauss(), lags=0), "lags = 0", "lag length p"),
        (lambda: tsecon.proxy_ar_sets(X3, PROXY, lags=2, horizon=6, rf_method="second_order", rf_draws=0, rf_seed=1), "rf_draws must be an even number", "draws must be an even"),
        (lambda: tsecon.proxy_ar_sets(X3, PROXY, lags=2, horizon=6, variance="hac", hac_lags=BIG), "hac_lags (the HAC bandwidth)", "\0"),
        (lambda: tsecon.proxy_first_stage(X3, PROXY, lags=2, variance="hac", hac_lags=BIG), "hac_lags (the HAC bandwidth)", "\0"),
    ],
)
def test_refusals_name_the_python_keyword(call, expect, never):
    with pytest.raises(ValueError) as info:
        call()
    msg = str(info.value)
    assert expect in msg, msg
    assert never not in msg.replace(expect, ""), msg


# --------------------------------------------------------------------------
# B3: scale_ar is blamed by name, with the largest order the sample supports
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda s: tsecon.bvar_fit(X3, lags=2, scale_ar=s),
        lambda s: tsecon.bvar_irf_draws(X3, lags=2, horizon=8, n_draws=50, scale_ar=s),
        lambda s: tsecon.bvar_hierarchical(X3, lags=2, scale_ar=s),
    ],
)
def test_a_scale_ar_too_long_for_the_sample_names_scale_ar(call):
    with pytest.raises(ValueError) as info:
        call(BIG)
    msg = str(info.value)
    # T = 200 rows support 2 * scale_ar + 2 <= 200, i.e. scale_ar <= 99.
    assert f"scale_ar = {BIG}" in msg and "pass scale_ar <= 99" in msg, msg
    assert "poorly conditioned" not in msg and "Use fewer lags" not in msg, msg


def test_bvar_hierarchical_short_sample_is_not_reported_as_non_convergence():
    short = X3[:9]
    with pytest.raises(ValueError) as info:
        tsecon.bvar_hierarchical(short, lags=1)
    msg = str(info.value)
    assert "scale_ar = 4" in msg and "pass scale_ar <= 3" in msg, msg
    # Same verdict as bvar_fit on the same data and scale_ar.
    with pytest.raises(ValueError, match=r"scale_ar = 4: .* pass scale_ar <= 3"):
        tsecon.bvar_fit(short, lags=1)


def test_callables_without_scale_ar_still_report_a_short_sample():
    # sign_restricted_svar fixes the AR(4) scale convention and has no
    # scale_ar keyword, so it must not tell the user to change one.
    with pytest.raises(ValueError) as info:
        tsecon.sign_restricted_svar(X3[:9], SR3, lags=1, horizon=2)
    msg = str(info.value)
    assert "rows" in msg and "scale_ar" not in msg, msg


# --------------------------------------------------------------------------
# B4: a negative entry inside a restriction spec is named by its position
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call, path",
    [
        (lambda: tsecon.sign_restricted_svar(X3, [(0, 0, -1, "+")], lags=1, horizon=4), "restrictions[0][2]=-1"),
        (lambda: tsecon.sign_restricted_svar(X3, [(0, -1, 0, "+")], lags=1, horizon=4), "restrictions[0][1]=-1"),
        (lambda: tsecon.zero_sign_svar(X3, SR3, [(2, 0, -1)], lags=1, horizon=4), "zero_restrictions[0][2]=-1"),
        (lambda: tsecon.fry_pagan_svar(X3, [(-1, 0, 0, "+")], lags=1, horizon=4), "restrictions[0][0]=-1"),
        (lambda: tsecon.robust_svar_bounds(X3, [(0, 0, -1, "+")], lags=1, horizon=4), "restrictions[0][2]=-1"),
        (lambda: tsecon.historical_decomposition(X3, [(0, 0, -1, "+")], lags=1, identification="sign"), "restrictions[0][2]=-1"),
        (
            lambda: tsecon.narrative_svar(
                X3, SR3, [{"type": "shock_sign", "shock": 0, "period": -1, "sign": "+"}], lags=1, horizon=4
            ),
            "narrative_restrictions[0]['period']=-1",
        ),
        (
            lambda: tsecon.historical_decomposition(
                X3, SR3, lags=1, identification="sign",
                narrative_restrictions=[{"type": "shock_sign", "shock": 0, "period": -1, "sign": "+"}],
            ),
            "narrative_restrictions[0]['period']=-1",
        ),
    ],
)
def test_a_negative_spec_entry_is_named_by_its_path(call, path):
    with pytest.raises(ValueError) as info:
        call()
    msg = str(info.value)
    assert path in msg and "index" in msg and "nonnegative integer" in msg, msg
    assert "counts something" not in msg, msg
    assert isinstance(info.value.__cause__, OverflowError)


def test_a_negative_top_level_count_keeps_its_counting_message():
    with pytest.raises(ValueError, match=r"n_draws=-1 is negative, but this parameter counts"):
        tsecon.sign_restricted_svar(X3, SR3, lags=1, horizon=4, n_draws=-1)


# --------------------------------------------------------------------------
# SVF-4 / H16-07 / API16-5: an unknown identification is refused as unknown
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ident", ["Impact", "impact ", "impcat", "long_run", "chol"])
def test_an_unknown_identification_with_impact_is_refused_as_unknown(ident):
    with pytest.raises(ValueError) as info:
        tsecon.historical_decomposition(X2, lags=1, identification=ident, impact=np.eye(2))
    msg = str(info.value)
    assert msg.startswith(f"unknown identification {ident!r}".replace("'", '"')), msg
    assert 'did you mean identification="impact"' in msg, msg
    assert "identifies its own shocks" not in msg, msg
    with pytest.raises(ValueError, match=r"^unknown identification") as plain:
        tsecon.historical_decomposition(X2, lags=1, identification=ident)
    assert "did you mean" not in str(plain.value)


def test_a_real_mode_with_an_inert_impact_keeps_its_message():
    with pytest.raises(ValueError, match=r'identification="sign" identifies its own'):
        tsecon.historical_decomposition(X2, [(0, 0, 0, "+")], lags=1, identification="sign", impact=np.eye(2))


# --------------------------------------------------------------------------
# SVF-5: a pandas impact gets the verdict of the same values as an ndarray
# --------------------------------------------------------------------------
A0 = [[1.0, 0.2], [0.3, 0.9]]


def _impact_calls(impact):
    yield lambda: tsecon.structural_fevd(X2, lags=1, impact=impact)
    yield lambda: tsecon.historical_decomposition(X2, lags=1, identification="impact", impact=impact)


def _verdict(call):
    try:
        out = call()
    except ValueError as e:
        assert "impact" in str(e), str(e)
        return "refused"
    return np.asarray(out["impact"]).tolist()


def test_object_arrays_of_plain_reals_are_accepted_in_every_container():
    pd = pytest.importorskip("pandas")
    for form in (
        np.array(A0, dtype=object),
        pd.DataFrame(A0).astype(object),
        pd.DataFrame(A0).astype("Float64"),
        A0,
    ):
        for call in _impact_calls(form):
            assert _verdict(call) == A0


@pytest.mark.parametrize("kind", ["complex", "strings", "datetime", "decimal", "bigint"])
def test_malformed_pandas_impact_is_refused_like_the_ndarray(kind):
    pd = pytest.importorskip("pandas")
    import decimal

    arr = np.array(A0)
    forms = {
        "complex": arr + 1j,
        "strings": arr.astype(str),
        "datetime": np.eye(2).astype("datetime64[D]"),
        "decimal": np.array([[decimal.Decimal("1.0"), 0.2], [0.3, 0.9]], dtype=object),
        "bigint": np.array([[10**400, 0.2], [0.3, 0.9]], dtype=object),
    }
    nd = forms[kind]
    df = pd.DataFrame(nd, dtype=object) if nd.dtype == object else pd.DataFrame(nd)
    for call_nd, call_df in zip(_impact_calls(nd), _impact_calls(df)):
        assert _verdict(call_nd) == "refused"
        assert _verdict(call_df) == "refused"


def test_a_complex_dataframe_is_refused_as_data_not_truncated():
    pd = pytest.importorskip("pandas")
    with pytest.raises((TypeError, ValueError)):
        tsecon.structural_fevd(pd.DataFrame(X2 + 1j), lags=1)
    # A nullable-float frame (object to_numpy) is still read as data.
    df = pd.DataFrame(X2).astype("Float64")
    np.testing.assert_array_equal(
        np.asarray(tsecon.structural_fevd(df, lags=1)["fevd"]),
        np.asarray(tsecon.structural_fevd(X2, lags=1)["fevd"]),
    )


# --------------------------------------------------------------------------
# H16-09 / API16-2 / API16-8: structural_fevd's sigma and lags=0
# --------------------------------------------------------------------------
def test_structural_fevd_sigma_is_validated_and_refused_where_inert():
    a0 = np.linalg.cholesky(np.cov(X2.T))
    for kw in ({}, {"impact": a0}):
        for bad in ("bogus", "MLE", ""):
            with pytest.raises(ValueError, match=r"unknown sigma"):
                tsecon.structural_fevd(X2, lags=1, sigma=bad, **kw)
    with pytest.raises(ValueError, match=r'sigma="mle" is only read when impact=None'):
        tsecon.structural_fevd(X2, lags=1, impact=a0, sigma="mle")
    # The default (and the default spelled out) with impact are bit-identical.
    base = tsecon.structural_fevd(X2, lags=1, impact=a0)
    same = tsecon.structural_fevd(X2, lags=1, impact=a0, sigma="dfadj")
    assert base == same
    # sigma stays live where documented: it rescales the reported impact.
    dfadj = tsecon.structural_fevd(X2, lags=1)
    mle = tsecon.structural_fevd(X2, lags=1, sigma="mle")
    assert dfadj["impact"] != mle["impact"]
    np.testing.assert_allclose(dfadj["fevd"], mle["fevd"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("kw", [{}, {"impact": np.eye(2)}, {"sigma": "mle"}])
def test_structural_fevd_lags_zero_names_lags_and_the_model_it_fits(kw):
    with pytest.raises(ValueError) as info:
        tsecon.structural_fevd(X2, lags=0, **kw)
    msg = str(info.value)
    assert msg.startswith("lags = 0: the FEVD needs a VAR with at least one lag"), msg
    assert "BVAR" not in msg and "lag length p" not in msg, msg


# --------------------------------------------------------------------------
# H16-12: extreme-scale impact matrices
# --------------------------------------------------------------------------
def _bq_impact():
    return np.asarray(tsecon.long_run_svar(X2, lags=1)["impact"])


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e-155, 1e155, 1e200, 1e300])
def test_structural_fevd_shares_are_scale_invariant_at_any_finite_scale(scale):
    # Closed form: the shares are invariant to rescaling A0 (each row of
    # Theta_s scales by the same factor in numerator and denominator).
    a0 = _bq_impact()
    ref = np.asarray(tsecon.structural_fevd(X2, lags=1, horizon=12, impact=a0)["fevd"])
    got = np.asarray(tsecon.structural_fevd(X2, lags=1, horizon=12, impact=a0 * scale)["fevd"])
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.sum(axis=2), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-100, 1e100, 1e150])
def test_hd_contributions_are_scale_invariant_where_the_diagnostic_is_representable(scale):
    a0 = _bq_impact()
    ref = tsecon.historical_decomposition(X2, lags=1, identification="impact", impact=a0)
    got = tsecon.historical_decomposition(X2, lags=1, identification="impact", impact=a0 * scale)
    np.testing.assert_allclose(got["hd"], ref["hd"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(got["shock_covariance"]) * scale**2, ref["shock_covariance"], rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("scale", [1e155, 1e200, 1e300, 1e-200, 1e-300])
def test_hd_extreme_scale_is_refused_for_scale_not_called_singular(scale):
    a0 = _bq_impact()
    assert np.linalg.matrix_rank(a0 * scale) == 2
    if scale == 1e155:
        # Inside the range: works (it used to be refused as "singular").
        out = tsecon.historical_decomposition(X2, lags=1, identification="impact", impact=a0 * scale)
        assert np.isfinite(np.asarray(out["shock_covariance"])).all()
        return
    with pytest.raises(ValueError) as info:
        tsecon.historical_decomposition(X2, lags=1, identification="impact", impact=a0 * scale)
    msg = str(info.value)
    assert "rescale impact" in msg and "singular" not in msg, msg


@pytest.mark.parametrize("scale", [1.0, 1e200, 1e-200])
def test_a_singular_impact_is_still_singular_at_any_scale(scale):
    with pytest.raises(ValueError, match=r"impact is singular"):
        tsecon.historical_decomposition(
            X2, lags=1, identification="impact", impact=np.ones((2, 2)) * scale
        )
