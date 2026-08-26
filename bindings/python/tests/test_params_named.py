"""``params_named`` on ``garch_fit`` — named access on the raw dict.

The field report behind this: ``fit["omega"]`` is a ``KeyError`` (parameters
live in the ``params``/``param_names`` parallel arrays), and the natural
``fit.get("omega")`` guard silently yields ``None``/NaN that reads like a
failed fit. The documented named-access route was the results facade
(``GARCHResults.params_named()``), but the raw-dict trap stayed. The raw dict
now carries an additive ``params_named`` key: exactly
``dict(zip(param_names, params))``, in estimator order, on every fit path.

``gas_volatility`` and ``dcs_local_level`` are deliberately NOT given the key:
they already return flat named scalars (``omega``/``a``/``b``[/``nu``] and
``kappa``/``scale``[/``nu``]) and no ``params``/``param_names`` arrays, so
there is no parallel-array trap to fix there (asserted below so the decision
is load-bearing, not folklore).
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


@pytest.fixture(scope="module")
def y():
    return _sim_garch(0.05, 0.08, 0.88, 900, 3)


def _assert_exact_zip(r):
    named = r["params_named"]
    assert isinstance(named, dict)
    names = list(r["param_names"])
    params = np.asarray(r["params"])
    # Same keys, same ORDER, and exactly the same floats (no re-rounding).
    assert list(named.keys()) == names
    for name, value in zip(names, params):
        assert named[name] == value


def test_params_named_matches_parallel_arrays_default(y):
    r = tsecon.garch_fit(y)
    assert set(r["params_named"]) == {"omega", "alpha[1]", "beta[1]"}
    _assert_exact_zip(r)


def test_params_named_on_t_constant_mean_and_gjr(y):
    t = tsecon.garch_fit(y, mean="constant", dist="t")
    assert list(t["params_named"]) == ["mu", "omega", "alpha[1]", "beta[1]", "nu"]
    _assert_exact_zip(t)
    g = tsecon.garch_fit(y, vol="gjr")
    assert "gamma[1]" in g["params_named"]
    _assert_exact_zip(g)
    e = tsecon.garch_fit(y, vol="egarch")
    _assert_exact_zip(e)


def test_params_named_present_on_boundary_fit():
    """The key exists (and is exact) even when the fit lands on a constraint
    — the situation where a .get() guard used to look like a failed fit."""
    yb = _sim_garch(1.0, 0.0, 0.0, 750, 2)  # white noise: alpha -> sign bound
    r = tsecon.garch_fit(yb)
    _assert_exact_zip(r)
    assert r["params_named"]["alpha[1]"] < 1e-6
    assert bool(np.asarray(r["boundary"]).any())


def test_params_named_present_with_forecast_horizon(y):
    r = tsecon.garch_fit(y, forecast_horizon=4)
    assert "variance_forecast" in r
    _assert_exact_zip(r)


def test_params_named_agrees_with_the_results_facade(y):
    from tsecon.results._garch import GARCHResults

    res = GARCHResults.fit(y, vol="garch", mean="zero", dist="normal")
    # The facade method and the raw key are the same mapping.
    assert res.params_named() == {
        k: float(v) for k, v in res["params_named"].items()
    }


def test_gas_and_dcs_keep_flat_named_keys_and_need_no_params_named():
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    y = np.empty(n)
    s2 = 1.0
    for t in range(n):
        y[t] = np.sqrt(s2) * z[t]
        s2 = 0.05 + 0.08 * y[t] ** 2 + 0.88 * s2

    g = tsecon.gas_volatility(y)
    assert {"omega", "a", "b"} <= set(g.keys())          # flat named scalars
    assert "params" not in g and "param_names" not in g  # no parallel arrays
    assert "params_named" not in g                       # so no key needed

    d = tsecon.dcs_local_level(y, density="gaussian")
    assert {"kappa", "scale"} <= set(d.keys())
    assert "params" not in d and "param_names" not in d
    assert "params_named" not in d
