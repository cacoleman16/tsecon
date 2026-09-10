"""Audit round 13 regression pins (docs/roadmap/28-audit-round-13-findings.md).

The post-wave adversarial sweep over the six 0.9.0 callables
(`setar_threshold_ci`, `var_girf`, `threshold_var_girf`, `jsz_fit`,
`jsz_loadings`, `panel_distributed_lag`) and the changed count pre-flight in
`_coerce`. Each test pins one confirmed finding:

* the coercion wrapper rebuilt EVERY failed PyO3 extraction as a rank error
  — `constant=1` (an int where a bool goes) told the user their correct array
  had the wrong shape (sweep S); the rebuild now names the offending
  argument and the genuine `ndarray` downcast keeps the rank text;
* PyO3's integer and float extraction failures ("'float' object cannot be
  interpreted as an integer", "must be real number, not str") named no
  argument (sweep S, 100+ cells); they do now;
* `jsz_fit(seed=None)` — the signature default — silently meant seed 0 while
  `help()`, the stub, the card table and the guide bullet all stated the
  default as `0` (sweeps F and H; the round-11 M4 class);
* both GIRF callables clamp `histories` above the available windows to all
  of them, which no surface said (sweep H);
* `var_girf`'s docstring promised `mc_se`/`draw_sd` "exactly zero" while the
  default `n_draws=2` under `antithetic=True` is one effective draw, so
  `mc_se` is NaN there and `draw_sd` is zero only to rounding (sweep E).
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

import tsecon


def _ar1(T=200, seed=0, phi=0.5):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    y = np.empty(T)
    prev = 0.0
    for t in range(T):
        prev = phi * prev + e[t]
        y[t] = prev
    return y


def _var3(T=200, seed=7):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((T, 3))
    for t in range(1, T):
        y[t] = a @ y[t - 1] + rng.standard_normal(3)
    return y


def _yields(T=200, seed=9):
    rng = np.random.default_rng(seed)
    mats = np.arange(1, 13, dtype=float)
    lam = 0.0609 * 12
    g = (1 - np.exp(-lam * mats)) / (lam * mats)
    h = g - np.exp(-lam * mats)
    load = np.column_stack([np.ones_like(mats), g, h])
    f = np.zeros((T, 3))
    mu = np.array([0.05, -0.02, 0.01])
    f[0] = mu
    for t in range(1, T):
        f[t] = mu + 0.9 * (f[t - 1] - mu) + rng.standard_normal(3) * np.array([0.003, 0.003, 0.004])
    return f @ load.T + 0.0005 * rng.standard_normal((T, 12))


# --------------------------------------------------------------------------- #
# S1: a wrong-typed option is named, not blamed on the array
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "call, offender, want",
    [
        (lambda: tsecon.setar(_ar1(), 1, constant=1), "constant=1", "bool"),
        (lambda: tsecon.var_girf(_var3(), 1, antithetic=2), "antithetic=2", "bool"),
        (lambda: tsecon.var_girf(_var3(), 1, trend=None), "trend=None", "string"),
        (lambda: tsecon.adf(_ar1(), 3), "regression=3", "string"),
        (lambda: tsecon.var_girf(_var3(), 1, bands=[0.16, 0.84]), "bands=", "tuple"),
    ],
)
def test_wrong_typed_option_names_the_argument_not_the_array(call, offender, want):
    """Before: 'var_girf: an array argument is the wrong shape or type (got
    arg0: array(200, 3)) ... want a 2-D array' for `antithetic=2`."""
    with pytest.raises(TypeError) as info:
        call()
    msg = str(info.value)
    assert offender in msg and want in msg, msg
    assert "wrong shape" not in msg and "2-D array" not in msg, msg


@pytest.mark.parametrize(
    "call, offender, want",
    [
        (lambda: tsecon.adf(_ar1(), maxlag=1.5), "maxlag=1.5", "integer"),
        (lambda: tsecon.setar(_ar1(), 1.5), "p=1.5", "integer"),
        (lambda: tsecon.var_girf(_var3(), 1, horizon=None), "horizon=None", "integer"),
        (lambda: tsecon.setar(_ar1(), 1, delays=[1.5]), "delays=[1.5]", "integer"),
        (lambda: tsecon.setar(_ar1(), 1, trim="abc"), "trim='abc'", "real number"),
    ],
)
def test_wrong_typed_count_or_float_names_the_argument(call, offender, want):
    with pytest.raises(TypeError) as info:
        call()
    msg = str(info.value)
    assert re.match(r"^[a-z_]+: ", msg), msg  # prefixed with the function name
    assert offender in msg and want in msg, msg
    assert "Original error" in msg


@pytest.mark.parametrize(
    "call, offender, want",
    [
        (lambda: tsecon.var_girf(_var3(), 1, histories=[1]), "histories=array([1.])", "integer"),
        (lambda: tsecon.adf(_ar1(), maxlag=[2]), "maxlag=array([2.])", "integer"),
        (lambda: tsecon.setar(_ar1(), 1, trim=[0.5]), "trim=array([0.5])", "real number"),
    ],
)
def test_one_element_list_where_a_scalar_goes_names_the_argument(call, offender, want):
    """The wrapper turns a flat numeric list into an array (data everywhere
    else); the boundary then fails in NumPy's words ("only integer scalar
    arrays can be converted to a scalar index"). The data array must never
    be the one blamed.

    NumPy before 2.4 still *converts* a one-element float array to a Python
    float (with a DeprecationWarning since 1.25), so on those versions the
    ``trim=[0.5]`` call reaches the compiled validator as ``trim=0.5`` and is
    refused there, by name; the CI wheel job on Python 3.9 runs such a
    NumPy. Either way the refusal names ``trim``, never the data array.
    """
    if want == "real number" and _one_element_float_converts():
        with pytest.raises(ValueError, match=r"trim = 0\.5") as info:
            call()
        msg = str(info.value)
    else:
        with pytest.raises(TypeError) as info:
            call()
        msg = str(info.value)
        assert offender in msg and want in msg and "one-element list" in msg, msg
    assert "shape=(200" not in msg and "data=" not in msg and "y=" not in msg, msg


def _one_element_float_converts() -> bool:
    """True when this NumPy turns ``array([0.5])`` into ``0.5`` (deprecated
    since 1.25, an error from 2.4)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            float(np.array([0.5]))
        except TypeError:
            return False
    return True


def test_rank_error_names_the_array_parameter_not_its_position():
    """`arg0: array(200,)` became `data: array(200,)` through the cached
    signature; the shape and the advice are unchanged."""
    with pytest.raises(TypeError) as info:
        tsecon.var_fit(_ar1(), 1)
    msg = str(info.value)
    assert "got data: array(200,)" in msg and "arg0" not in msg, msg
    with pytest.raises(TypeError, match=r"got y: array\(200, 3\)"):
        tsecon.adf(_var3())


def test_wrong_type_rebuild_prefers_the_argument_whose_default_has_another_type():
    """`p=1` and `constant=1` are both ints; only `constant` (bool default)
    is blamed."""
    with pytest.raises(TypeError) as info:
        tsecon.setar(_ar1(), 1, constant=1)
    msg = str(info.value)
    assert "constant=1" in msg and "p=1" not in msg, msg


def test_genuine_rank_errors_keep_the_rank_text():
    y = _ar1()
    with pytest.raises(TypeError, match="wrong shape or type"):
        tsecon.var_fit(y, 1)  # 1-D where 2-D is wanted
    with pytest.raises(TypeError, match="wrong shape or type"):
        tsecon.var_fit(_var3().tolist(), 1)  # nested list
    with pytest.raises(TypeError, match="wrong shape or type"):
        tsecon.adf(_var3())  # 2-D where 1-D is wanted


def test_wrong_type_rebuild_keeps_the_original_and_chains_it():
    with pytest.raises(TypeError) as info:
        tsecon.setar(_ar1(), 1, constant=1)
    assert "is not an instance of 'bool'" in str(info.value)
    assert isinstance(info.value.__cause__, TypeError)


# --------------------------------------------------------------------------- #
# F/H: jsz_fit's seed default
# --------------------------------------------------------------------------- #
def test_jsz_fit_seed_none_is_seed_zero_and_every_surface_says_so():
    y = _yields()
    mats = list(range(1, 13))
    a = tsecon.jsz_fit(y, mats, n_starts=2)
    b = tsecon.jsz_fit(y, mats, n_starts=2, seed=0)
    c = tsecon.jsz_fit(y, mats, n_starts=2, seed=None)
    assert a["seed"] == b["seed"] == c["seed"] == 0
    np.testing.assert_array_equal(a["lambda_q"], b["lambda_q"])
    np.testing.assert_array_equal(a["lambda_q"], c["lambda_q"])
    assert inspect.signature(tsecon._core.jsz_fit).parameters["seed"].default is None
    flat = re.sub(r"\s+", " ", tsecon.jsz_fit.__doc__)
    assert "`seed` (None, which means seed 0" in flat, "runtime docstring must state the None default"
    assert "`seed` (0" not in flat
    from pathlib import Path

    root = Path(__file__).parents[1]
    stub = (root / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    stub_doc = stub[stub.index("def jsz_fit(") : stub.index("def jsz_loadings(")]
    assert "`seed` (None, meaning seed 0" in re.sub(r"\s+", " ", stub_doc)
    docs = root.parents[1] / "docs"
    if docs.exists():
        card = (docs / "reference" / "model-cards" / "term-structure.md").read_text(encoding="utf-8")
        assert "| | `seed` | `None` (→ 0) |" in card
        guide = (docs / "guide" / "15-term-structure.md").read_text(encoding="utf-8")
        assert "n_starts=5, seed=None)`" in guide and "n_starts=5, seed=0)`" not in guide


# --------------------------------------------------------------------------- #
# H: the histories clamp is stated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["var_girf", "threshold_var_girf"])
def test_histories_above_the_available_windows_uses_all_and_is_documented(name):
    fn = getattr(tsecon, name)
    data = _var3()
    kw = {"horizon": 4, "n_draws": 4, "seed": 0}
    full = fn(data, 1, **kw)
    clamped = fn(data, 1, histories=10**6, **kw)
    assert clamped["n_histories"] == full["n_histories"] == 199
    np.testing.assert_array_equal(np.asarray(clamped["girf"]), np.asarray(full["girf"]))
    flat = re.sub(r"\s+", " ", fn.__doc__)
    assert "uses all of them, reported in `n_histories`" in flat


# --------------------------------------------------------------------------- #
# E: var_girf's zero-to-rounding contract at the default draw count
# --------------------------------------------------------------------------- #
def test_var_girf_mc_se_is_nan_at_the_default_and_the_docstring_says_so():
    data = _var3()
    r = tsecon.var_girf(data, 2, horizon=6)
    assert r["n_draws"] == 2 and r["n_effective_draws"] == 1
    assert np.isnan(r["mc_se"]).all()
    assert np.abs(r["draw_sd"]).max() < 1e-15
    r4 = tsecon.var_girf(data, 2, horizon=6, n_draws=4)
    assert np.isfinite(r4["mc_se"]).all() and np.abs(r4["mc_se"]).max() < 1e-15
    flat = re.sub(r"\s+", " ", tsecon.var_girf.__doc__)
    assert "zero to rounding" in flat and "NaN below two effective draws" in flat
    assert "exactly zero" not in flat
