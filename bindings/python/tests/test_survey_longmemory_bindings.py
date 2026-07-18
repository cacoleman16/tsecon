"""Binding tests for two carded-but-previously-untested Python entry points:

- ``forecast_disagreement`` (survey expectations, E6): per-period dispersion
  ``std`` (with ``ddof``), quartiles ``p25``/``p50``/``p75``, ``iqr``, and
  forecaster ``counts`` from a ragged panel of per-period cross-sections.
- ``frac_integrate`` (long memory, E7): the exact inverse of ``frac_diff``,
  i.e. the fractional integration filter ``(1 - L)^{-d}``.

Every expected number here is either hand-computed in a comment or built from
a tiny in-test NumPy reference; only ``numpy`` and ``tsecon`` are imported so
the tests exercise the bindings rather than any third-party econometrics code.
"""
import numpy as np
import tsecon


def _raises(fn):
    """Return the exception raised by ``fn()`` (fails if none is raised)."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - we assert on the concrete type below
        return exc
    raise AssertionError("expected the binding to raise, but it returned")


# ============================================================ forecast_disagreement

# A ragged panel: period 0 has four forecasters, period 1 has three. Both
# cross-sections are chosen so every statistic is checkable by hand.
_PANEL = [
    np.array([1.0, 2.0, 3.0, 4.0]),   # count 4
    np.array([10.0, 12.0, 20.0]),     # count 3
]


def test_forecast_disagreement_returns_expected_keys():
    out = tsecon.forecast_disagreement(_PANEL, ddof=0)
    assert set(out.keys()) == {"std", "p25", "p50", "p75", "iqr", "counts"}


def test_forecast_disagreement_std_ddof0_vs_ddof1_hand_values():
    # Period 0 = [1, 2, 3, 4], mean 2.5, sum of squared deviations
    #   = 1.5^2 + 0.5^2 + 0.5^2 + 1.5^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
    #   ddof=0 -> sqrt(5/4) = sqrt(1.25);  ddof=1 -> sqrt(5/3).
    # Period 1 = [10, 12, 20], mean 14, SSD = 16 + 4 + 36 = 56.
    #   ddof=0 -> sqrt(56/3);  ddof=1 -> sqrt(56/2) = sqrt(28).
    std0_hand = np.array([np.sqrt(1.25), np.sqrt(56.0 / 3.0)])
    std1_hand = np.array([np.sqrt(5.0 / 3.0), np.sqrt(28.0)])

    d0 = tsecon.forecast_disagreement(_PANEL, ddof=0)
    d1 = tsecon.forecast_disagreement(_PANEL, ddof=1)

    np.testing.assert_allclose(np.asarray(d0["std"]), std0_hand, atol=1e-12)
    np.testing.assert_allclose(np.asarray(d1["std"]), std1_hand, atol=1e-12)

    # ...and, independently, against NumPy's own ddof-aware estimator.
    np.testing.assert_allclose(
        np.asarray(d0["std"]),
        np.array([np.std(p, ddof=0) for p in _PANEL]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(d1["std"]),
        np.array([np.std(p, ddof=1) for p in _PANEL]),
        atol=1e-12,
    )

    # ddof must not change the (order-statistic based) quartiles or counts.
    for key in ("p25", "p50", "p75", "iqr"):
        np.testing.assert_allclose(np.asarray(d0[key]), np.asarray(d1[key]), atol=0.0)
    assert list(d0["counts"]) == list(d1["counts"])


def test_forecast_disagreement_quartiles_iqr_counts_hand_values():
    out = tsecon.forecast_disagreement(_PANEL, ddof=0)

    # numpy "linear" interpolation, virtual index h = (m-1) * q/100.
    # Period 0 = [1, 2, 3, 4] (m=4):
    #   p25: h = 3*0.25 = 0.75 -> 1 + 0.75*(2-1) = 1.75
    #   p50: h = 3*0.50 = 1.5  -> 2 + 0.50*(3-2) = 2.5
    #   p75: h = 3*0.75 = 2.25 -> 3 + 0.25*(4-3) = 3.25
    #   iqr = 3.25 - 1.75 = 1.5
    # Period 1 = [10, 12, 20] (m=3):
    #   p25: h = 2*0.25 = 0.5 -> 10 + 0.5*(12-10) = 11
    #   p50: h = 2*0.50 = 1.0 -> 12
    #   p75: h = 2*0.75 = 1.5 -> 12 + 0.5*(20-12) = 16
    #   iqr = 16 - 11 = 5
    np.testing.assert_allclose(np.asarray(out["p25"]), [1.75, 11.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(out["p50"]), [2.5, 12.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(out["p75"]), [3.25, 16.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(out["iqr"]), [1.5, 5.0], atol=1e-12)

    # Cross-check the quartiles against numpy's percentile on each cross-section.
    for i, p in enumerate(_PANEL):
        q25, q50, q75 = np.percentile(p, [25.0, 50.0, 75.0])
        assert np.isclose(out["p25"][i], q25)
        assert np.isclose(out["p50"][i], q50)
        assert np.isclose(out["p75"][i], q75)

    # counts is a plain list of the per-period cross-section sizes.
    assert list(out["counts"]) == [4, 3]


def test_forecast_disagreement_empty_panel_raises():
    exc = _raises(lambda: tsecon.forecast_disagreement([], ddof=1))
    assert isinstance(exc, ValueError)


def test_forecast_disagreement_empty_cross_section_raises():
    bad = [np.array([1.0, 2.0]), np.array([])]  # second period has no forecasters
    exc = _raises(lambda: tsecon.forecast_disagreement(bad, ddof=1))
    assert isinstance(exc, ValueError)


def test_forecast_disagreement_ddof_not_smaller_than_count_raises():
    # A one-forecaster period cannot support ddof=1 (divisor count - ddof <= 0).
    exc = _raises(lambda: tsecon.forecast_disagreement([np.array([3.0])], ddof=1))
    assert isinstance(exc, ValueError)


# ==================================================================== frac_integrate


def _frac_integrate_reference(x, d):
    """(1 - L)^{-d} x by the binomial expansion, as a NumPy reference.

    The (1-L)^{-d} weights satisfy w_0 = 1 and w_k = w_{k-1} * (k-1+d)/k,
    and the filtered series is the causal convolution y_t = sum_{j<=t} w_j x_{t-j}.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    w = np.empty(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (k - 1 + d) / k
    y = np.empty(n)
    for t in range(n):
        y[t] = sum(w[j] * x[t - j] for j in range(t + 1))
    return y


def test_frac_integrate_inverts_frac_diff_roundtrip():
    rng = np.random.default_rng(20260717)
    for d in (0.2, 0.45):
        x = rng.standard_normal(12)
        recovered = np.asarray(tsecon.frac_integrate(tsecon.frac_diff(x, d), d))
        assert np.allclose(recovered, x, atol=1e-10)
        # ...and the other composition order recovers x too.
        recovered2 = np.asarray(tsecon.frac_diff(tsecon.frac_integrate(x, d), d))
        assert np.allclose(recovered2, x, atol=1e-10)


def test_frac_integrate_matches_binomial_reference_golden():
    x = np.array([1.0, 0.0, 2.0, -1.0, 3.0])
    d = 0.3
    got = np.asarray(tsecon.frac_integrate(x, d))

    # Hand-expanded golden (weights 1, 0.3, 0.195, 0.1495, 0.12208125):
    #   y0 = 1
    #   y1 = 0.3*1 + 1*0            = 0.3
    #   y2 = 0.195*1 + 0.3*0 + 1*2  = 2.195
    #   y3 = 0.1495*1 + 0.195*0 + 0.3*2 + 1*(-1) = -0.2505
    #   y4 = 0.12208125*1 + 0.1495*0 + 0.195*2 + 0.3*(-1) + 1*3 = 3.2133375
    golden = np.array([1.0, 0.3, 2.195, -0.2505, 3.2133375])
    np.testing.assert_allclose(got, golden, atol=1e-12)

    # And against the independent NumPy binomial-weight reference.
    np.testing.assert_allclose(got, _frac_integrate_reference(x, d), atol=1e-12)


def test_frac_integrate_equals_frac_diff_with_negated_d():
    # frac_integrate(x, d) is defined as frac_diff(x, -d).
    x = np.array([0.5, -1.5, 2.0, 0.0, 1.0, -0.5])
    d = 0.35
    np.testing.assert_allclose(
        np.asarray(tsecon.frac_integrate(x, d)),
        np.asarray(tsecon.frac_diff(x, -d)),
        atol=1e-14,
    )
