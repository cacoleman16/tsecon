"""Repo-audit security sweep regression pins (docs/roadmap/_repo_audit/security.md).

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
# the residual product-overflow panic: rebuilt into a ValueError
# --------------------------------------------------------------------------- #
def test_product_capacity_overflow_is_a_value_error_not_a_panic():
    """`bvar_fit(lags=2**31)` sizes a design of lags x k x T entries — the
    product overflows isize before any allocation, and the panic used to
    escape. 2**31 is below the pre-flight line, so this exercises the rebuild
    path; the message names the suspect argument and keeps the panic as the
    chained cause."""
    with pytest.raises(ValueError, match=r"lags=2147483648") as info:
        tsecon.bvar_fit(_var3(), lags=2**31)
    assert "could not be sized or allocated" in str(info.value)
    assert type(info.value.__cause__).__name__ == "PanicException"


def test_refused_allocation_below_the_line_is_a_value_error_with_the_size():
    """`bvar_fit(lags=2**40)` asks the allocator for a 72 TiB design before
    any sufficiency check runs; the refusal used to escape as an `AllocError`
    unwrap panic. Rebuilt with the requested size in the message."""
    with pytest.raises(ValueError, match=r"lags=1099511627776.*GiB was requested"):
        tsecon.bvar_fit(_var3(), lags=2**40)


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
