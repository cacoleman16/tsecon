"""Runtime-docstring vs returned-keys tripwire (audit rounds 3-4, finding 5).

The runtime ``__doc__`` is the surface a Python user actually reads
(``help(tsecon.fn)``), and rounds 3-4 found it is also the surface a fix most
often misses: ``long_memory_d``'s docstring still described the pre-fix
two-key return ("the estimate d and its asymptotic se") after the code and the
model card had moved to five keys, and ``predictive_regression``'s named a
``rho`` key that does not exist. These tests diff ``sorted(fn(...).keys())``
against the backticked key names in ``__doc__`` for exactly those two
functions, so that class of drift fails a test instead of shipping.

The check is one-directional on purpose: every *returned* key must be named in
the docstring (a stale docstring fails); the docstring may mention extra
backticked words (parameter names, math symbols), so no attempt is made to
require every token to be a key.
"""
import re

import numpy as np
import tsecon


def _doc_tokens(fn) -> set[str]:
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))


def test_long_memory_d_docstring_names_every_returned_key():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(512)
    tokens = _doc_tokens(tsecon.long_memory_d)

    gph_keys = set(tsecon.long_memory_d(x, method="gph").keys())
    lw_keys = set(tsecon.long_memory_d(x, method="local_whittle").keys())

    assert gph_keys == {"d", "m", "se", "se_asymptotic", "se_regression"}
    assert lw_keys == {"d", "m", "se", "se_asymptotic"}
    missing = (gph_keys | lw_keys) - tokens
    assert not missing, f"long_memory_d.__doc__ does not name returned keys: {sorted(missing)}"
    # The round-1 headline: `se` (bandwidth-exact) and `se_asymptotic` (too
    # narrow) are distinct quantities; the docstring must keep them apart and
    # must not describe the return as just "d and its asymptotic se".
    flat = re.sub(r"\s+", " ", tsecon.long_memory_d.__doc__ or "")
    assert "`d` and its asymptotic `se`" not in flat
    assert "se_asymptotic" in flat


def test_garch_fit_docstring_names_every_returned_key():
    """Round 7: garch_fit's docstring now enumerates its keys (it gained
    `se_valid`/`boundary`/`boundary_note`/`converged`); keep it honest."""
    rng = np.random.default_rng(3)
    z = rng.standard_normal(700)
    y = np.empty(700)
    s2 = 1.0
    for t in range(700):
        y[t] = np.sqrt(s2) * z[t]
        s2 = 0.05 + 0.08 * y[t] ** 2 + 0.88 * s2
    tokens = _doc_tokens(tsecon.garch_fit)
    keys = set(tsecon.garch_fit(y, forecast_horizon=3).keys())
    missing = keys - tokens
    assert not missing, f"garch_fit.__doc__ does not name returned keys: {sorted(missing)}"


def test_predictive_regression_docstring_names_every_returned_key():
    rng = np.random.default_rng(2)
    T = 240
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.95 * x[t - 1] + rng.standard_normal()
    r = 0.05 * x + rng.standard_normal(T)

    out = tsecon.predictive_regression(r[1:], x[:-1])
    tokens = _doc_tokens(tsecon.predictive_regression)

    returned = set(out.keys())
    for sub in ("ols", "stambaugh", "ivx"):
        returned |= set(out[sub].keys())
    assert set(out["stambaugh"].keys()) == {
        "beta_corrected", "beta_ols", "bias_term", "rho_ols", "se"
    }
    missing = returned - tokens
    assert not missing, (
        f"predictive_regression.__doc__ does not name returned keys: {sorted(missing)}"
    )
    # The stale docstring promised a `rho` key that never existed; the actual
    # key is `rho_ols`. A bare backticked `rho` token must not reappear.
    assert "rho" not in tokens, "docstring names a bare `rho` key that is not returned"
    assert "rho_ols" in tokens


def test_theta_forecast_docstring_qualifies_the_statsmodels_match():
    """Audit round 8: tsecon reproduces statsmodels ``ThetaModel`` at
    ``deseasonalize=True, use_test=False``. statsmodels' *default* runs a
    seasonality pre-test and skips deseasonalization when it fails, so the
    bare claim "Matches statsmodels ThetaModel" was an overclaim (measured:
    29/30 iid draws with period=12 diverge from the statsmodels default,
    worst 2.6% relative). The docstring must carry the qualifier."""
    doc = re.sub(r"\s+", " ", tsecon.theta_forecast.__doc__ or "")
    assert "use_test=False" in doc
    # The unqualified sentence must not stand alone as the whole claim.
    assert "Matches statsmodels ThetaModel." not in doc


def test_ou_fit_docstring_names_every_returned_key():
    """Audit round 10: the returned `level` key (the echoed CI level of
    `half_life_ci`) existed since 0.6.0 but was undocumented. The full
    key-vs-docstring diff now gates ou_fit like the functions above."""
    rng = np.random.default_rng(8)
    x = np.empty(300)
    prev = 0.0
    for t in range(300):
        prev = 0.9 * prev + 0.1 * rng.standard_normal()
        x[t] = prev
    out = tsecon.ou_fit(x, dt=0.5, level=0.9)
    assert out["level"] == 0.9 and out["dt"] == 0.5  # the echoes themselves
    tokens = _doc_tokens(tsecon.ou_fit)
    missing = set(out.keys()) - tokens
    assert not missing, f"ou_fit.__doc__ does not name returned keys: {sorted(missing)}"


def test_markov_switching_ar_docstring_names_every_returned_key():
    """Audit round 10: `iterations` (EM steps run) was returned but
    undocumented; it now gates alongside `converged` with the full diff."""
    rng = np.random.default_rng(9)
    n = 240
    y = np.empty(n)
    state = 0
    prev = 0.0
    for t in range(n):
        if rng.random() < 0.05:
            state = 1 - state
        mu = (-1.0, 1.5)[state]
        prev = mu + 0.3 * (prev - mu) + 0.5 * rng.standard_normal()
        y[t] = prev
    out = tsecon.markov_switching_ar(y, max_iter=200)
    assert isinstance(out["iterations"], int) and out["iterations"] >= 1
    tokens = _doc_tokens(tsecon.markov_switching_ar)
    missing = set(out.keys()) - tokens
    assert not missing, (
        f"markov_switching_ar.__doc__ does not name returned keys: {sorted(missing)}"
    )
