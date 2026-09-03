"""Audit round 11 regression pins (docs/roadmap/26-audit-round-11-findings.md).

Round 11 swept the result-object contract, signature/stub/docstring drift,
complexity cliffs and the seed contract over all 162 callables. No severe
class fired; the confirmed findings are contract-documentation defects, and
each test here pins the corrected surface so the drift cannot recur:

* every returned key of the functions whose runtime ``__doc__`` used to
  omit the stub's key list is now named in ``help()`` (the surface the
  brief calls binding);
* ``seed=None`` means seed 0 (not fresh entropy) in the three functions
  that accept it, and says so;
* the EGARCH multi-step forecast refusal is documented on all three GARCH
  surfaces and no longer leaks an internal marker;
* the two imprecise "T" shape claims (``dfm_nowcast.smoothed_factors``,
  ``proxy_svar.shock``) now state the effective row count;
* the card rows the sweeps caught are corrected in the docs tree.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

import tsecon

ROOT = Path(__file__).parents[1]
REPO = ROOT.parent.parent


# --------------------------------------------------------------------------- #
# tiny seeded inputs (mirrors lab/audit/round11/registry.py at T=200)
# --------------------------------------------------------------------------- #
def _ar1(T=200, seed=0, phi=0.5):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    y = np.empty(T)
    prev = 0.0
    for t in range(T):
        prev = phi * prev + e[t]
        y[t] = prev
    return y


def _rw(T=200, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal(T))


def _var3(T=200, seed=7):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = a @ y[t - 1] + rng.standard_normal(3)
    return y


def _coint(T=200, seed=11):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(T))
    return np.column_stack([1.5 * x + rng.standard_normal(T), x])


def _yx(T=200, seed=4, k=2):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(T), rng.standard_normal((T, k))])
    y = X @ (np.arange(1, k + 2) * 0.5) + np.random.default_rng(seed + 100).standard_normal(T)
    return y, X


def _garch(T=200, seed=3, k=1):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T, k))
    y = np.empty((T, k))
    s2 = np.ones(k)
    for t in range(T):
        y[t] = np.sqrt(s2) * z[t]
        s2 = 0.05 + 0.08 * y[t] ** 2 + 0.88 * s2
    return y[:, 0] if k == 1 else y


def _proxy(T=200, seed=7):
    d = _var3(T, seed)
    rng = np.random.default_rng(seed + 50)
    u = np.diff(d, axis=0)[:, 0]
    return d, np.r_[np.nan, 0.8 * u + 0.5 * rng.standard_normal(T - 1)]


def _setar(T=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = (0.6 if y[t - 1] <= 0 else -0.4) * y[t - 1] + rng.standard_normal()
    return y


def _nongauss(T=200, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    B = np.array([[1.0, 0.3, 0.0], [0.2, 1.0, 0.4], [0.0, 0.1, 1.0]])
    e = rng.laplace(size=(T, 3)) * np.array([1.0, 0.7, 1.3])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = a @ y[t - 1] + B @ e[t]
    return y


def _curves(T=200, M=6, seed=2):
    return np.random.default_rng(seed).standard_normal((T, M)) @ np.diag(np.linspace(1.5, 0.3, M))


def _flp_y(T=200, seed=2):
    rng = np.random.default_rng(seed)
    c = _curves(T, 6, seed)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + 0.7 * c[t, 0] - 0.3 * c[t, 1] + rng.standard_normal()
    return y, c


def _factor_panel(T=200, N=8, seed=3):
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.6 * f[t - 1] + rng.standard_normal()
    return np.outer(f, rng.uniform(0.6, 1.4, N)) + 0.5 * rng.standard_normal((T, N))


def _units(N=5, T=200, seed=8):
    rng = np.random.default_rng(seed)
    ys, xs = [], []
    for _ in range(N):
        x = np.cumsum(rng.standard_normal(T)) * 0.3
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + 0.5 * x[t] + 0.5 * rng.standard_normal()
        ys.append(y)
        xs.append(x.reshape(-1, 1))
    return ys, xs


def _iv(T=200, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T, 2))
    v = rng.standard_normal(T)
    x_end = z @ np.array([1.0, 0.5]) + v
    return (
        np.column_stack([np.ones(T), x_end]),
        np.column_stack([np.ones(T), z]),
        1.0 + 0.5 * x_end + 0.5 * v + rng.standard_normal(T),
    )


def _predreg(T=200, seed=0, k=2):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, k))
    for t in range(1, T):
        x[t] = 0.95 * x[t - 1] + rng.standard_normal(k)
    r = 0.05 * x[:, 0] + rng.standard_normal(T)
    return r[1:], x[:-1]


def _break(T=200, seed=0):
    y = np.random.default_rng(seed).standard_normal(T) + np.where(np.arange(T) < T // 2, 0.0, 2.0)
    return y, np.ones((T, 1))


POS = [(0, 0, 0, "+")]
SHOCK = np.random.default_rng(1).standard_normal(200)

# function -> (args, kwargs): one canonical call per function whose runtime
# docstring the round repaired. Every key it returns must be named in
# ``fn.__doc__`` (the help() surface), which is what the sweep found missing.
CALLS = {
    "robust_svar_bounds": ((_var3(), POS), {"horizon": 6, "n_draws": 40, "seed": 0}),
    "fry_pagan_svar": ((_var3(), POS), {"horizon": 6, "n_draws": 40, "seed": 0}),
    "hetero_svar": (
        (np.vstack([_var3(100, 7), 2.0 * _var3(100, 8)]), (np.arange(200) >= 100).astype(np.int64)),
        {"horizon": 6},
    ),
    "historical_decomposition": ((_var3(),), {}),
    "historical_decomposition[sign]": (
        (_var3(),),
        {"identification": "sign", "restrictions": POS, "n_draws": 40, "n_weight_draws": 20, "seed": 0},
    ),
    "narrative_svar": ((_var3(), POS), {"horizon": 6, "n_draws": 40, "seed": 0}),
    "gpd_fit": ((np.abs(np.random.default_rng(0).standard_t(4, 200)),), {}),
    "gev_fit": ((np.abs(np.random.default_rng(0).standard_t(4, 200)),), {"block_size": 5}),
    "adaptive_lasso": ((_yx(k=3)[1][:, 1:], _yx(k=3)[0], 0.1), {}),
    "lasso": ((_yx(k=3)[1][:, 1:], _yx(k=3)[0], 0.1), {}),
    "elastic_net": ((_yx(k=3)[1][:, 1:], _yx(k=3)[0], 0.1), {}),
    "connectedness": ((_var3(),), {"horizon": 6}),
    "factor_model": ((_factor_panel(),), {"n_factors": 2, "kmax": 4}),
    "flp": ((_flp_y()[0], _flp_y()[1][:, :2]), {"horizons": 4, "n_lag_controls": 1}),
    "flp_scenario": (
        (*_flp_y(), np.linspace(1.0, 0.0, 6)),
        {"n_factors": 2, "horizons": 4, "n_lag_controls": 1},
    ),
    "functional_pca": ((_curves(),), {"n_factors": 2}),
    "fvar_scenario": ((*_flp_y(), np.linspace(1.0, 0.0, 6)), {"n_factors": 2, "lags": 1, "horizon": 4}),
    "iv_gmm": (_iv(), {}),
    "ivx_test": (_predreg(), {}),
    "jarque_bera": ((_ar1(),), {}),
    "johansen": ((_coint(),), {}),
    "nongaussian_svar": ((_nongauss(),), {"horizon": 6}),
    "panel_lp": (
        (np.array([_ar1(200, i) for i in range(6)]), SHOCK),
        {"horizon": 6},
    ),
    "panel_pmg": (_units(), {}),
    "panel_unit_root": ((np.array([_ar1(200, i) for i in range(5)]),), {"lags": 1}),
    "proxy_first_stage": (_proxy(), {}),
    "quantile_regression": (_yx(), {"taus": [0.25, 0.5, 0.75]}),
    "setar": ((_setar(), 1), {}),
    "smooth_lp": ((_ar1(), SHOCK), {"horizons": 6, "lam": 1.0}),
    "sup_f_test": (_break(), {}),
    "var_backtest": ((_garch(), np.full(200, -1.7)), {"alpha": 0.05}),
    "local_level_smooth": ((_rw() + np.random.default_rng(0).standard_normal(200), 1.0, 0.5), {}),
    "hp_filter": ((_rw(),), {}),
    "bk_filter": ((_rw(),), {}),
    "cf_filter": ((_rw(),), {}),
    "cg_regression": ((_ar1(), _ar1(200, 1)), {"maxlags": 4}),
    "engle_granger": ((_coint(),), {}),
    "gas_volatility": ((_garch(),), {"horizon": 3}),
    "zero_sign_svar": ((_var3(), [], [(0, 1, 0)]), {"horizon": 6, "n_draws": 40, "seed": 0}),
    "check_series": ((_ar1(),), {}),
    "check_series[multivariate]": ((_var3(),), {}),
}


def _named_in_doc(fn) -> set[str]:
    return set(re.findall(r"`+([A-Za-z_][A-Za-z_0-9]*)`+", fn.__doc__ or ""))


@pytest.mark.parametrize("label", sorted(CALLS))
def test_runtime_docstring_names_every_returned_key(label):
    """Round 11, sweep E(iv): 29 functions' runtime ``__doc__`` (what
    ``help()`` shows) omitted keys the stub named — ``robust_svar_bounds``'s
    help text, for one, was two lines against the stub's full contract
    including its NaN convention. Every returned key must now be backticked
    in the runtime docstring."""
    name = label.split("[")[0]
    fn = getattr(tsecon, name)
    args, kwargs = CALLS[label]
    res = fn(*args, **kwargs)
    assert isinstance(res, dict)
    missing = sorted(set(res) - _named_in_doc(fn))
    assert not missing, f"{name}.__doc__ does not name returned keys: {missing}"


def test_stub_and_runtime_docs_both_name_the_penalized_keys():
    """`max_rel_change` was returned by lasso/elastic_net/adaptive_lasso and
    documented on no surface; both docs now carry it, with `max_change`."""
    stub = (ROOT / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    for name in ("lasso", "elastic_net", "adaptive_lasso"):
        body = re.search(rf"def {name}\(.*?\"\"\"(.*?)\"\"\"", stub, re.S).group(1)
        assert "max_rel_change" in body and "max_change" in body, name
        assert "max_rel_change" in getattr(tsecon, name).__doc__


# --------------------------------------------------------------------------- #
# sweep H: seed=None means seed 0, and says so
# --------------------------------------------------------------------------- #
def _bits_equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_bits_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple, np.ndarray)):
        try:
            aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        except (TypeError, ValueError):  # a list of dicts (proxy_ar_sets' cells)
            return len(a) == len(b) and all(_bits_equal(x, y) for x, y in zip(a, b))
        return aa.shape == bb.shape and aa.tobytes() == bb.tobytes()
    if isinstance(a, float):
        return (np.isnan(a) and np.isnan(b)) or a == b
    return a == b


def test_conformal_seed_none_is_seed_zero_and_documented():
    y = _ar1() + 5.0
    kw = dict(horizon=2, method="enbpi", base="ar", n_boot=10)
    none = tsecon.conformal_forecast(y, seed=None, **kw)
    assert _bits_equal(none, tsecon.conformal_forecast(y, seed=0, **kw))
    assert not _bits_equal(none, tsecon.conformal_forecast(y, seed=1, **kw))
    kw = dict(horizon=1, method="enbpi", base="ar", n_boot=10, n_eval=10, batch=1)
    none = tsecon.conformal_backtest(y, seed=None, **kw)
    assert _bits_equal(none, tsecon.conformal_backtest(y, seed=0, **kw))
    assert not _bits_equal(none, tsecon.conformal_backtest(y, seed=1, **kw))
    for fn in (tsecon.conformal_forecast, tsecon.conformal_backtest):
        flat = re.sub(r"\s+", " ", fn.__doc__)
        assert re.search(r"seed=None.{0,120}seed 0", flat), fn.__name__


def test_proxy_ar_sets_rf_seed_none_is_seed_zero_and_documented():
    d, p = _proxy()
    kw = dict(horizon=6, rf_method="second_order", rf_draws=40)
    none = tsecon.proxy_ar_sets(d, p, rf_seed=None, **kw)
    assert _bits_equal(none, tsecon.proxy_ar_sets(d, p, rf_seed=0, **kw))
    assert not _bits_equal(none, tsecon.proxy_ar_sets(d, p, rf_seed=1, **kw))
    flat = re.sub(r"\s+", " ", tsecon.proxy_ar_sets.__doc__)
    assert re.search(r"rf_seed=None.{0,120}seed 0", flat)
    assert "256" in flat  # rf_draws=None means 256 draws


# --------------------------------------------------------------------------- #
# sweep F(d): the EGARCH multi-step refusal is documented and clean
# --------------------------------------------------------------------------- #
def test_egarch_multistep_forecast_refusal_is_documented_and_clean():
    y = _garch()
    one = tsecon.garch_fit(y, vol="egarch", forecast_horizon=1)
    assert len(one["variance_forecast"]) == 1 and np.isfinite(one["variance_forecast"][0])
    with pytest.raises(ValueError, match="horizon") as exc:
        tsecon.garch_fit(y, vol="egarch", forecast_horizon=2)
    msg = str(exc.value)
    assert "TODO" not in msg and "phase0" not in msg
    assert "forecast_horizon=1" in msg
    r2 = _garch(k=2)
    for fn in (tsecon.ccc_garch, tsecon.dcc_garch):
        with pytest.raises(ValueError, match="horizon") as exc:
            fn(r2, vol="egarch", forecast_horizon=2)
        assert "TODO" not in str(exc.value)
        assert fn(r2, vol="egarch", forecast_horizon=1)["variance_forecast"] is not None
    for fn in (tsecon.garch_fit, tsecon.ccc_garch, tsecon.dcc_garch):
        flat = re.sub(r"\s+", " ", fn.__doc__)
        assert re.search(r"egarch.{0,200}forecast_horizon", flat, re.I), fn.__name__


# --------------------------------------------------------------------------- #
# sweep E(v): the two imprecise "T" shape claims
# --------------------------------------------------------------------------- #
def test_dfm_nowcast_smoothed_factors_rows_are_the_balanced_panel():
    x = _factor_panel()
    x[-2:, :3] = np.nan
    r = tsecon.dfm_nowcast(x)
    assert np.asarray(r["smoothed_factors"]).shape == (198, 1)
    assert "ragged-edge" in tsecon.dfm_nowcast.__doc__


def test_proxy_svar_shock_length_is_the_residual_sample():
    d, p = _proxy()
    r = tsecon.proxy_svar(d, p, lags=2)
    assert len(r["shock"]) == 198
    stub = (ROOT / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    assert "`shock` (length T - lags" in stub


def test_cv_splits_default_train_is_documented_as_refused():
    with pytest.raises(ValueError, match="training window"):
        tsecon.cv_splits(100)
    assert len(tsecon.cv_splits(100, scheme="purged_kfold")) == 5
    assert "defaults to 0" in re.sub(r"\s+", " ", tsecon.cv_splits.__doc__)


# --------------------------------------------------------------------------- #
# the card rows the sweeps caught (docs tree only)
# --------------------------------------------------------------------------- #
def _card(name):
    p = REPO / "docs" / "reference" / "model-cards" / name
    if not p.exists():
        pytest.skip("docs tree not present in this checkout")
    return p.read_text(encoding="utf-8")


def test_forecasting_card_backtest_defaults_match_the_signature():
    """`period` is refused when passed explicitly with a non-seasonal
    forecaster (0.7.0), so a card row reading "default 1" taught the trap; the
    signature default is None for both `period` and `forecaster`."""
    card = _card("forecasting.md")
    sig = inspect.signature(tsecon._core.backtest)
    assert sig.parameters["forecaster"].default is None
    assert sig.parameters["period"].default is None
    assert "| | `forecaster` | `None` (→ `naive`)" in card
    assert "| | `period` | `None` (→ 1)" in card
    assert "raises (0.7.0)" in card


def test_panel_card_lists_the_stamped_keys():
    card = _card("panel.md")
    lp_line = next(l for l in card.splitlines() if l.startswith("- **`panel_lp`** →"))
    for k in ("se_type", "cumulative", "jackknife", "bias_correction"):
        assert f'"{k}"' in lp_line
    did_line = next(l for l in card.splitlines() if l.startswith("- **`lp_did`** →"))
    for k in ("absorbing", "nonabsorbing_lag", "reweight", "pooled", "never_treated_only", "se_type"):
        assert f'"{k}"' in did_line


def test_volatility_card_states_the_egarch_forecast_limit():
    card = _card("volatility.md")
    assert re.search(r"vol=\"egarch\"` accepts `forecast_horizon` 0 or 1 only", card)
