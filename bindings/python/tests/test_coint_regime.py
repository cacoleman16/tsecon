"""Golden tests for the cointegration and Markov-switching bindings."""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
COINT = json.loads((FIX / "coint.json").read_text())
REGIME = json.loads((FIX / "regime.json").read_text())
# coint data is stored series-major (3 lists of length T); transpose to T x k.
CDATA = np.array(COINT["data"]).T


def test_johansen_matches_statsmodels():
    r = tsecon.johansen(CDATA, k_ar_diff=COINT["johansen"]["k_ar_diff"])
    np.testing.assert_allclose(r["eig"], COINT["johansen"]["eig"], atol=1e-8)
    np.testing.assert_allclose(r["trace_stat"], COINT["johansen"]["trace_stat"], rtol=1e-6)
    np.testing.assert_allclose(r["max_eig_stat"], COINT["johansen"]["max_eig_stat"], rtol=1e-6)
    # Two cointegrated I(1) series + one stationary series => the trace test
    # detects at least rank 2 (the stationary series is an extra stationary
    # combination, so at 5% this particular draw yields rank 3; at 1% it is 2).
    assert r["rank_trace_5pct"] >= 2


def test_vecm_matches_statsmodels():
    r = tsecon.vecm(CDATA, k_ar_diff=2, coint_rank=1)
    fx = COINT["vecm_rank1"]
    np.testing.assert_allclose(r["alpha"], fx["alpha"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["beta"], fx["beta"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["gamma"], fx["gamma"], rtol=1e-6, atol=1e-8)
    assert r["llf"] == pytest.approx(fx["llf"], rel=1e-6)


def test_markov_switching_recovers_regimes():
    y = np.array(REGIME["y"])
    r = tsecon.markov_switching_ar(y, k_regimes=2, order=1, switching_variance=True)
    # EM must at least reach the fixed-param loglik (up to a small slack).
    assert r["loglik"] >= REGIME["loglike_fixed"] - 1.0
    # The two regime means are recovered (as an unordered set, since labels
    # are identified only up to permutation).
    true_means = sorted(REGIME["fixed_params"]["const"])
    got_means = sorted(r["means"])
    np.testing.assert_allclose(got_means, true_means, atol=0.4)
    # Transition rows/cols are stochastic; durations are positive.
    T = np.array(r["transition"])
    assert np.allclose(T.sum(axis=0), 1.0, atol=1e-8)
    assert (np.asarray(r["expected_durations"]) > 1.0).all()
    assert set(np.unique(r["regimes"])).issubset({0, 1})


def test_markov_switching_returns_full_probability_matrices():
    """Audit fix (rounds 3-4, finding 3): the binding reduced the Kim
    smoother's n x k matrix to its last column, which is unrecoverable at
    k >= 3, and never surfaced the filtered probabilities at all.
    """
    rng = np.random.default_rng(7)
    means = [-2.0, 0.5, 3.0]
    y = np.concatenate(
        [m + 0.4 * rng.standard_normal(50) for _ in range(4) for m in means]
    )
    order = 1
    k = 3
    r = tsecon.markov_switching_ar(y, k_regimes=k, order=order, switching_variance=True)

    n = len(y) - order
    smoothed = np.asarray(r["smoothed_prob"])
    filtered = np.asarray(r["filtered_prob"])
    assert smoothed.shape == (n, k)
    assert filtered.shape == (n, k)
    # Rows are probability distributions over regimes.
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-8)
    # Back-compat: the 0.2.0 scalar path is bit-identical to the last column.
    np.testing.assert_array_equal(
        smoothed[:, -1], np.asarray(r["smoothed_prob_last_regime"])
    )
    # Smoothing conditions on the full sample, filtering on the past only;
    # on a switching series they must genuinely differ in the interior.
    assert not np.array_equal(filtered, smoothed)
    # The MAP regime path is the argmax of the smoothed rows.
    np.testing.assert_array_equal(
        np.asarray(r["regimes"]), smoothed.argmax(axis=1)
    )
    assert set(np.unique(r["regimes"])).issubset(set(range(k)))
