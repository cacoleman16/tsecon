"""Repo-audit security sweep regression pins (docs/roadmap/27-repo-audit-2026-09/security.md).

The adversarial-input matrix (lab/audit/repo/security/sweep_adversarial.py)
drove every public callable with corrupted arguments in a memory-capped child
process. One class fired across the surface: an integer count argument of
2**63 reached a `Vec::with_capacity` whose byte size overflows, and the Rust
`capacity overflow` panic escaped to Python as ``pyo3_runtime.PanicException``
— a ``BaseException`` that ``except Exception`` does not catch — while a count
of 2**31 had the allocation attempted for real (16 GB for one index vector)
and aborted the process when the allocator refused. The seal lives at the one
point every wrapper passes through (``tsecon._coerce._call``):

* a count at or beyond 2**48 (2 PiB of f64 — beyond any addressable memory)
  is refused before the call reaches Rust, as a ``ValueError`` naming the
  argument; seeds are exempt (a u64 seed is legitimately any 64-bit value);
* the residual ``capacity overflow`` panic from a *product* of moderate
  counts (a lag length of 2**31 in a squared design) is rebuilt into a
  ``ValueError`` naming the suspect arguments — the panic fires inside the
  allocator's size check before any state is touched, so nothing compiled is
  left inconsistent;
* every other ``BaseException`` (``KeyboardInterrupt``, ``SystemExit``, any
  other panic) passes through unchanged.

The remaining pins guard the repository's stated boundaries: ``import tsecon``
opens no socket and reads no environment variable of its own.

The pre-flight pins are real calls (they never reach Rust). The rebuild pins
use a synthetic panic: the real calls behind them ask for 144 GiB or 72 TiB,
which Linux's allocator refuses up front but macOS commits lazily, killing
the process — that is finding S3, still open, and not something a test may
depend on.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

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


# --------------------------------------------------------------------------- #
# the absurd-count band: refused before Rust, catchable, argument named
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "call, name",
    [
        (lambda: tsecon.bootstrap_indices(2**63, scheme="iid", seed=0), "n"),
        (lambda: tsecon.arch_lm(_ar1(), nlags=2**63), "nlags"),
        (lambda: tsecon.bn_filter(np.cumsum(_ar1()), p=2**63), "p"),
        (lambda: tsecon.ccc_garch(np.column_stack([_ar1(), _ar1(seed=1)]), forecast_horizon=2**63), "forecast_horizon"),
        (lambda: tsecon.bvar_ssvs(_var3(), horizon=2**63, n_draws=20, burn=5, seed=0), "horizon"),
        (lambda: tsecon.boosting(np.random.default_rng(0).standard_normal((100, 3)), _ar1(100), n_steps=2**63), "n_steps"),
        (lambda: tsecon.philox_uniforms(0, 2**63), "n"),
    ],
)
def test_absurd_count_is_a_value_error_not_a_panic(call, name):
    """Each of these escaped as PanicException('capacity overflow') before the
    seal — reproduced by the sweep on the 0.8.0 tree."""
    with pytest.raises(ValueError, match=rf"{name}=\d+ is at or beyond 2\*\*48"):
        call()


def test_absurd_count_inside_an_integer_list_is_refused():
    with pytest.raises(ValueError, match=r"delays=.*at or beyond 2\*\*48"):
        tsecon.setar(_ar1(), 1, delays=[1, 2**63])


def test_absurd_count_is_catchable_by_except_exception():
    """The property the seal exists for: `except Exception` catches it."""
    caught = None
    try:
        tsecon.bootstrap_indices(2**63, scheme="iid", seed=0)
    except Exception as exc:  # noqa: BLE001 — the point of the test
        caught = exc
    assert isinstance(caught, ValueError)


# --------------------------------------------------------------------------- #
# the residual allocation-sizing panic: rebuilt into a ValueError
# --------------------------------------------------------------------------- #
# These pins drive the seal's rebuild half with a synthetic panic rather than
# a real call. The real calls the sweep recorded (`bvar_fit(lags=2**31)`, a
# 144 GiB design; `bvar_fit(lags=2**40)`, 72 TiB) reach the rebuild only where
# the allocator refuses up front — Linux does, and the seal turns the refusal
# into the ValueError below — but macOS commits the request lazily and the
# process is killed while the design is being filled (the CI runner reproduced
# exactly that: exit 137). That platform gap is finding S3, still open, and a
# test must not depend on it. The seal keys on the exception's type name and
# message, so a BaseException subclass of the same name is the real thing as
# far as `_coerce._call` is concerned.


class PanicException(BaseException):
    """Stand-in for `pyo3_runtime.PanicException` (also a BaseException)."""


def _compiled_stand_in(name, message):
    def fn(y, lags=1):
        raise PanicException(message)

    fn.__name__ = name
    return fn


def test_product_capacity_overflow_is_a_value_error_not_a_panic():
    """`Vec::with_capacity` on a product of moderate counts that overflows
    isize panics with "capacity overflow" before any allocation; the rebuild
    names the suspect argument and keeps the panic as the chained cause."""
    from tsecon import _coerce

    fn = _compiled_stand_in("bvar_fit", "capacity overflow")
    with pytest.raises(ValueError, match=r"lags=2147483648") as info:
        _coerce._call(fn, (_var3(),), {"lags": 2**31})
    assert "could not be sized or allocated" in str(info.value)
    assert type(info.value.__cause__).__name__ == "PanicException"


def test_refused_allocation_below_the_line_is_a_value_error_with_the_size():
    """When the allocator refuses and faer's fallible allocation is unwrapped,
    the panic carries the byte count; the rebuild reports it in GiB."""
    from tsecon import _coerce

    fn = _compiled_stand_in(
        "bvar_fit",
        "called `Result::unwrap()` on an `Err` value: AllocError { layout: "
        "Layout { size: 79164837200064, align: 8 } }",
    )
    with pytest.raises(ValueError, match=r"lags=1099511627776.*73,728 GiB was requested"):
        _coerce._call(fn, (_var3(),), {"lags": 2**40})


def test_a_panic_that_is_not_about_allocation_passes_through():
    """Only the three allocation-sizing shapes are rebuilt; any other panic
    stays a BaseException, because it may have touched state."""
    from tsecon import _coerce

    fn = _compiled_stand_in("var_fit", "index out of bounds: the len is 3 but the index is 7")
    with pytest.raises(PanicException):
        _coerce._call(fn, (_var3(),), {"lags": 2})


# --------------------------------------------------------------------------- #
# nothing weakened: seeds, negatives, ordinary counts, other BaseExceptions
# --------------------------------------------------------------------------- #
def test_large_seeds_are_still_accepted():
    idx = tsecon.bootstrap_indices(20, scheme="iid", seed=2**63)
    assert idx.shape == (20,)
    u = tsecon.philox_uniforms(2**64 - 1, 5)
    assert u.shape == (5,)
    r = tsecon.setar_test(_ar1(), 1, n_boot=9, seed=2**63 + 1)
    assert "p_value" in r


def test_negative_counts_keep_their_teaching_error():
    with pytest.raises(ValueError, match="negative"):
        tsecon.arch_lm(_ar1(), nlags=-1)


def test_a_merely_large_count_reaches_the_estimator():
    """2**20 is below the impossibility line: the estimator sees it and
    applies its own sufficiency refusal, not the seal's."""
    with pytest.raises(ValueError) as info:
        tsecon.arch_lm(_ar1(), nlags=2**20)
    assert "2**48" not in str(info.value)


def test_other_base_exceptions_pass_through_the_wrapper_unchanged():
    def kbi(theta):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        tsecon.gmm_nonlinear(kbi, [0.0, 1.0])


# --------------------------------------------------------------------------- #
# stated boundaries: no sockets, no environment reads, on import and in use
# --------------------------------------------------------------------------- #
def test_import_and_use_open_no_socket_and_read_no_environment():
    script = textwrap.dedent(
        """
        import os, socket, sys
        calls = []
        def deny(name):
            def f(*a, **k):
                calls.append(name)
                raise RuntimeError("network use attempted")
            return f
        socket.socket.__init__ = deny("socket")
        for n in ("create_connection", "getaddrinfo", "gethostbyname"):
            setattr(socket, n, deny(n))
        reads = set()
        class Env(dict):
            def __getitem__(self, k):
                reads.add(k); return super().__getitem__(k)
            def get(self, k, d=None):
                reads.add(k); return super().get(k, d)
            def __contains__(self, k):
                reads.add(k); return super().__contains__(k)
        import numpy as np
        os.environ = Env(os.environ)
        import tsecon
        y = np.cumsum(np.random.default_rng(0).standard_normal(200))
        tsecon.adf(y)
        tsecon.setar_test(np.diff(y), 1, n_boot=9, seed=0)
        print(len(calls), sorted(reads))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    n_calls, reads = out.stdout.strip().split(" ", 1)
    assert n_calls == "0"
    assert reads == "[]", reads


# --------------------------------------------------------------------------- #
# audit round 13 OPEN-3 / the sweep-S refusal catalogue: every refusal names
# the offending parameter (in-process replay of the 83 unnamed cells of
# lab/audit/round13/out/sweep_s.txt, now fixed in Rust messages, in the
# bindings, and in the _coerce rebuilds)
# --------------------------------------------------------------------------- #
import inspect
import re as _re


def _names_param(msg, pname):
    return bool(_re.search(rf"(?<![\w.]){_re.escape(pname)}(?![\w])", msg or ""))


def _setar_series(T=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = (0.6 if y[t - 1] <= 0 else -0.4) * y[t - 1] + rng.standard_normal()
    return y


def _tvar_series(T=200, seed=0):
    rng = np.random.default_rng(seed)
    a_low = np.array([[0.8, 0.1], [0.1, 0.7]])
    a_high = np.array([[0.2, 0.0], [0.0, 0.3]])
    y = np.zeros((T + 100, 2))
    for t in range(1, T + 100):
        a = a_low if y[t - 1, 0] <= 0.0 else a_high
        y[t] = a @ y[t - 1] + 0.6 * rng.standard_normal(2)
    return y[100:]


def _yields(T=200, seed=9):
    rng = np.random.default_rng(seed)
    mats = np.arange(1, 13, dtype=float)
    lam = 0.0609 * 12
    g = (1 - np.exp(-lam * mats)) / (lam * mats)
    h = g - np.exp(-lam * mats)
    L = np.column_stack([np.ones_like(mats), g, h])
    f = np.zeros((T, 3))
    mu = np.array([0.05, -0.02, 0.01])
    f[0] = mu
    for t in range(1, T):
        f[t] = mu + 0.9 * (f[t - 1] - mu) + rng.standard_normal(3) * np.array([0.003, 0.003, 0.004])
    return f @ L.T + 0.0005 * rng.standard_normal((T, len(mats)))


def _dl_panel(N=6, T=200, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(20.0, 5.0, N)
    temp = mu[:, None] + rng.standard_normal((N, T))
    growth = (rng.normal(0, 1, N)[:, None] + rng.normal(0, 0.5, T)[None, :] + 0.30 * temp - 0.0075 * temp**2
              + rng.standard_normal((N, T)))
    return growth, temp[None]


BIG = [2**31, 2**47]
_Y = _setar_series()
_V = _var3()
_TV = _tvar_series()
_YL = _yields()
_G, _TEMP = _dl_panel()
_LQ, _KQ, _SX, _MATS = np.array([0.998, 0.96, 0.90]), 1e-4, 1e-6 * np.eye(3), list(range(1, 8))

# (callable, positional args, kwargs, parameter, bad values) — the sweep's
# 83 unnamed cells, one entry per (callable, parameter, value class).
REFUSAL_CATALOGUE = [
    ("setar_threshold_ci", (_Y, 1), {"slope_level": 0.95, "null_threshold": 0.0}, "y",
     [np.empty(0), _Y[:1], float(_Y[0]), _Y > 0, "abc", None, 1.0]),
    ("setar_threshold_ci", (_Y, 1), {}, "p", BIG),
    ("setar_threshold_ci", (_Y, 1), {}, "delay", BIG),
    ("setar_threshold_ci", (_Y, 1), {}, "delays", [[0], [2**31], [2**47], "abc", np.array([1.0, 2.0])]),
    ("var_girf", (_V, 2), {"horizon": 6}, "data", [np.empty((0, 3)), _V[:1], np.ascontiguousarray(_V.T), "abc", None, 1.0]),
    ("var_girf", (_V, 2), {"horizon": 6}, "p", BIG),
    ("threshold_var_girf", (_TV, 1), {"horizon": 6, "n_draws": 20}, "data",
     [_TV[:1], _TV[:, :-1], _TV > 0, "abc", None, 1.0]),
    ("threshold_var_girf", (_TV, 1), {"horizon": 6, "n_draws": 20}, "p", BIG),
    ("threshold_var_girf", (_TV, 1), {"horizon": 6, "n_draws": 20}, "delay", BIG),
    ("threshold_var_girf", (_TV, 1), {"horizon": 6, "n_draws": 20}, "delays",
     [[0], [2**31], [2**47], "abc", np.array([1.0, 2.0])]),
    ("jsz_fit", (_YL, list(range(1, 13))), {"n_starts": 2, "seed": 0}, "yields",
     [np.full_like(_YL, np.nan), np.where(np.arange(_YL.size).reshape(_YL.shape) == 1200, np.nan, _YL),
      np.where(np.arange(_YL.size).reshape(_YL.shape) == 1200, np.inf, _YL), np.empty((0, 12)), _YL[:1],
      np.rint(_YL).astype(np.int64), _YL > 0, "abc", None, 1.0]),
    ("jsz_fit", (_YL, list(range(1, 13))), {"n_starts": 2, "seed": 0}, "maturities", ["abc", np.array([1.0, 2.0])]),
    ("jsz_fit", (_YL, list(range(1, 13))), {"n_starts": 2, "seed": 0}, "periods_per_year", [1e300, 1e-300]),
    ("jsz_fit", (_YL, list(range(1, 13))), {"n_starts": 2, "seed": 0}, "w", ["abc"]),
    ("jsz_loadings", (_LQ, _KQ, _SX, _MATS), {"periods_per_year": 12.0}, "lambda_q",
     [_LQ[:1], _LQ[:2], float(_LQ[0]), "abc", None, 1.0]),
    ("jsz_loadings", (_LQ, _KQ, _SX, _MATS), {"periods_per_year": 12.0}, "sigma_x", ["abc", None, 1.0]),
    ("jsz_loadings", (_LQ, _KQ, _SX, _MATS), {"periods_per_year": 12.0}, "maturities", ["abc", np.array([1.0, 2.0])]),
    ("panel_distributed_lag", (_G, _TEMP), {"lags": 1, "powers": 2, "eval_points": [10.0, 20.0, 30.0]}, "outcome",
     ["abc", None, 1.0]),
    ("panel_distributed_lag", (_G, _TEMP), {"lags": 1, "powers": 2, "eval_points": [10.0, 20.0, 30.0]}, "regressors",
     [np.full_like(_TEMP, np.nan), np.where(np.arange(_TEMP.size).reshape(_TEMP.shape) == 600, np.nan, _TEMP),
      np.where(np.arange(_TEMP.size).reshape(_TEMP.shape) == 600, np.inf, _TEMP), np.empty((1, 6, 0)),
      _TEMP[..., :-1], _TEMP[:, :-1], _TEMP > 0, np.ascontiguousarray(_TEMP.T), "abc", None, 1.0]),
    ("panel_distributed_lag", (_G, _TEMP), {"powers": 2, "eval_points": [10.0, 20.0, 30.0]}, "lags", BIG),
    ("panel_distributed_lag", (_G, _TEMP), {"lags": 1, "powers": 2}, "eval_points", ["abc", 20.0]),
]


def _cells():
    for fn, args, kwargs, pname, values in REFUSAL_CATALOGUE:
        for v in values:
            yield fn, args, kwargs, pname, v


def _cell_id(cell):
    fn, _, _, pname, v = cell
    if isinstance(v, np.ndarray):
        tag = f"array{v.shape}{v.dtype.kind}"
    else:
        tag = repr(v)[:20]
    return f"{fn}.{pname}={tag}"


@pytest.mark.parametrize("cell", list(_cells()), ids=_cell_id)
def test_every_sweep_s_refusal_names_the_offending_parameter(cell):
    """Each cell of the round-13 catalogue was a refusal that did not name
    the mutated argument; every one must now be a ValueError/TypeError whose
    message contains the parameter's name as a word."""
    fn, args, kwargs, pname, v = cell
    call = getattr(tsecon, fn)
    sig_params = list(inspect.signature(call).parameters)
    args = list(args)
    kw = dict(kwargs)
    if pname in sig_params[: len(args)]:
        args[sig_params.index(pname)] = v
    else:
        kw[pname] = v
    with pytest.raises((ValueError, TypeError)) as info:
        call(*args, **kw)
    msg = str(info.value)
    assert _names_param(msg, pname), f"{fn}: refusal does not name {pname!r}: {msg[:300]}"


def test_the_catalogue_covers_the_sweep_count():
    assert sum(len(values) for *_, values in REFUSAL_CATALOGUE) == 83


# --------------------------------------------------------------------------- #
# S3: the GIRF engine's memory budget — a refusal, not an abort
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "call",
    [
        lambda: tsecon.var_girf(_V, 2, n_draws=2**31, antithetic=False),
        lambda: tsecon.var_girf(_V, 2, horizon=10**6, n_draws=10**4, antithetic=False),
        lambda: tsecon.threshold_var_girf(_TV, 1, n_draws=2**30, horizon=20),
        lambda: tsecon.threshold_var_girf(_TV, 1, n_draws=100, horizon=10**6),
    ],
)
def test_girf_beyond_the_memory_budget_is_a_fast_value_error(call):
    import time

    t0 = time.perf_counter()
    with pytest.raises(ValueError) as info:
        call()
    assert time.perf_counter() - t0 < 5.0
    msg = str(info.value)
    for word in ("n_draws", "horizon", "histories", "budget", "GiB"):
        assert word in msg, msg
