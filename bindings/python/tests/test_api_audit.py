"""Regression tests for the repo-wide API-consistency audit (audit 12).

Two documentation contracts that the audit found broken on the surfaces a
user reads, now pinned so they cannot drift back:

* every returned key of the 33 functions whose keys were named on neither
  the runtime docstring nor the stub is now backticked in ``fn.__doc__``
  (the round-3/11 tripwire rule, extended to this class);
* ``ar_loglik`` treats NaN in ``y`` as a missing observation — the SSM
  crate documents it, the binding did not — so the docstring must say so
  and the behaviour must match what it says (with ``coeffs=[0]`` the
  log-likelihood equals that of the series with the entry deleted, and an
  infinite entry is refused).
"""
import re

import numpy as np
import pytest

import tsecon


def _tokens(fn):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))


def _ar1(T, seed=0, phi=0.5):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    y = np.empty(T)
    prev = 0.0
    for t in range(T):
        prev = phi * prev + e[t]
        y[t] = prev
    return y


def _rw(T, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal(T))


def _var(T, seed=7):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = a @ y[t - 1] + rng.standard_normal(3)
    return y


def _yx(T, seed=4, k=2):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(T), rng.standard_normal((T, k))])
    beta = np.arange(1, k + 2) * 0.5
    return X @ beta + np.random.default_rng(seed + 100).standard_normal(T), X


def _coint(T, seed=11):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(T))
    return np.column_stack([1.5 * x + rng.standard_normal(T), x])


def _garch(T, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(T)
    y = np.empty(T)
    s2 = 1.0
    for t in range(T):
        y[t] = np.sqrt(s2) * z[t]
        s2 = 0.05 + 0.08 * y[t] ** 2 + 0.88 * s2
    return y


def _break(T, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T) + np.where(np.arange(T) < T // 2, 0.0, 2.0), np.ones((T, 1))


T = 200
_RNG = np.random.default_rng(1)

# The 33 functions of the audit's "keys named on neither surface" class,
# each with the audit's canonical call.
CASES = {
    "accuracy": lambda: tsecon.accuracy(_ar1(T)[100:], _ar1(T, 1)[100:], _ar1(T)[:100]),
    "acf": lambda: tsecon.acf(_ar1(T)),
    "adf": lambda: tsecon.adf(_ar1(T)),
    "arch_lm": lambda: tsecon.arch_lm(_garch(T)),
    "backtest": lambda: tsecon.backtest(_ar1(T) + 5.0, train=100, horizon=2),
    "bai_perron": lambda: tsecon.bai_perron(*_break(T), max_breaks=2),
    "bvar_hierarchical": lambda: tsecon.bvar_hierarchical(_var(T)),
    "check_stationarity": lambda: tsecon.check_stationarity(_ar1(T)),
    "chow_test": lambda: tsecon.chow_test(*_yx(T), 100),
    "cusum_test": lambda: tsecon.cusum_test(*_yx(T)),
    "cw_test": lambda: tsecon.cw_test(_ar1(T), _ar1(T, 1), _ar1(T, 2), _ar1(T, 3)),
    "dcs_local_level": lambda: tsecon.dcs_local_level(_rw(T) + np.random.default_rng(0).standard_normal(T)),
    "dm_test": lambda: tsecon.dm_test(_ar1(T), _ar1(T, 1)),
    "growth_at_risk": lambda: tsecon.growth_at_risk(_ar1(T), _ar1(T, 1).reshape(-1, 1), horizon=2, taus=[0.1, 0.5, 0.9]),
    "gw_test": lambda: tsecon.gw_test(_ar1(T) ** 2, _ar1(T, 1) ** 2),
    "har_rv": lambda: tsecon.har_rv(np.exp(_ar1(T, 0, 0.7) * 0.3)),
    "heteroskedasticity_test": lambda: tsecon.heteroskedasticity_test(*_yx(T)),
    "kpss": lambda: tsecon.kpss(_ar1(T)),
    "ljung_box": lambda: tsecon.ljung_box(_ar1(T)),
    "lp_iv": lambda: tsecon.lp_iv(_ar1(T), _ar1(T, 1) + 0.5 * _ar1(T, 2), _ar1(T, 2), horizons=6),
    "lp_multiplier": lambda: tsecon.lp_multiplier(_ar1(T), _ar1(T, 1) + 0.5 * _ar1(T, 2), _ar1(T, 2), horizons=6),
    "mcmc_diagnostics": lambda: tsecon.mcmc_diagnostics(np.array([_ar1(T, i, 0.3) for i in range(4)])),
    "ols": lambda: tsecon.ols(*_yx(T)),
    "panel_fe": lambda: tsecon.panel_fe(
        np.array([_ar1(T, i) for i in range(6)]),
        np.array([np.array([_ar1(T, 10 + i) for i in range(6)]), np.array([_ar1(T, 20 + i) for i in range(6)])]),
    ),
    "phillips_ouliaris": lambda: tsecon.phillips_ouliaris(_coint(T)[:, 0], _coint(T)[:, 1:2]),
    "phillips_perron": lambda: tsecon.phillips_perron(_ar1(T)),
    "quantile_lp": lambda: tsecon.quantile_lp(_ar1(T), _ar1(T, 1), horizons=3, taus=[0.25, 0.5, 0.75]),
    "reset_test": lambda: tsecon.reset_test(*_yx(T)),
    "svensson": lambda: tsecon.svensson(
        np.linspace(3.0, 120.0, 8),
        5.0 - 2.0 * (1 - np.exp(-0.0609 * np.linspace(3.0, 120.0, 8))) / (0.0609 * np.linspace(3.0, 120.0, 8)),
        0.0609,
        0.2,
    ),
    "umidas": lambda: tsecon.umidas(_ar1(T), np.random.default_rng(0).standard_normal((T, 6))),
    "var_granger": lambda: tsecon.var_granger(_var(T), [0], [1]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_returned_key_is_named_in_the_runtime_docstring(name):
    fn = getattr(tsecon, name)
    out = CASES[name]()
    assert isinstance(out, dict)
    missing = set(out.keys()) - _tokens(fn)
    assert not missing, f"{name}.__doc__ does not name returned keys: {sorted(missing)}"


def test_ar_loglik_documents_and_implements_nan_as_missing():
    doc = re.sub(r"\s+", " ", tsecon.ar_loglik.__doc__ or "")
    assert "NaN" in doc and "missing" in doc
    y = np.random.default_rng(0).standard_normal(60)
    y_nan = y.copy()
    y_nan[10] = np.nan
    # with no dynamics the log-likelihood is separable, so skipping the
    # missing entry must equal deleting it
    with_nan = tsecon.ar_loglik(y_nan, [0.0], 1.0)
    dropped = tsecon.ar_loglik(np.delete(y, 10), [0.0], 1.0)
    assert np.isfinite(with_nan)
    assert with_nan == pytest.approx(dropped, rel=1e-12)
    assert with_nan != tsecon.ar_loglik(y, [0.0], 1.0)
    y_inf = y.copy()
    y_inf[10] = np.inf
    with pytest.raises(ValueError):
        tsecon.ar_loglik(y_inf, [0.0], 1.0)


def test_api_reference_preamble_does_not_promise_plain_arrays_everywhere():
    """54 dict-returning functions hand back nested Python lists for matrix
    keys (and `var_irf`/`var_fevd`/`bvar_irf_draws` at the top level); the
    generated reference must not claim every function returns NumPy arrays."""
    from pathlib import Path

    gen = Path(__file__).parents[3] / "docs" / "gen_api_reference.py"
    if not gen.exists():
        pytest.skip("docs tree not present in this checkout")
    text = gen.read_text(encoding="utf-8")
    assert "Every function returns plain NumPy arrays" not in text
    assert "nested Python lists" in text
