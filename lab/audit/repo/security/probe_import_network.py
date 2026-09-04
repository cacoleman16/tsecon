#!/usr/bin/env python
"""Does ``import tsecon`` (or a representative call) open any socket, read any
environment variable of its own, or touch the filesystem outside the package?

Two independent methods, because each has a blind spot:

1. In-process: every socket constructor / resolver / connection entry point in
   the ``socket`` module is replaced with one that raises and records; then
   ``import tsecon`` and a handful of calls run. Blind spot: a native extension
   could call ``connect(2)`` directly, bypassing the Python ``socket`` module.
2. Under ``strace -f -e trace=network`` (run by ``run_all.sh``): catches native
   syscalls too. Blind spot: none for network; noisy for other syscall classes.

Also records the environment variables the process reads during import and a
first call (via a recording ``os.environ`` mapping), and the files opened
outside the interpreter's own tree (via ``sys.setprofile`` on ``builtins.open``
is unreliable for native code, so this part is the ``strace -e trace=file``
pass in ``run_all.sh``).

Exit code 0 means "no socket use observed by method 1".
"""
from __future__ import annotations

import os
import socket
import sys

calls = []


def _forbid(name):
    def f(*a, **k):
        calls.append((name, a[:2]))
        raise RuntimeError(f"network use attempted: socket.{name}")

    return f


# Patch every constructor / resolver / connector before the import.
_orig_socket_init = socket.socket.__init__


def _socket_init(self, *a, **k):
    calls.append(("socket.socket", a[:2]))
    raise RuntimeError("network use attempted: socket.socket()")


socket.socket.__init__ = _socket_init  # type: ignore[method-assign]
for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "socketpair", "fromfd", "create_server"):
    if hasattr(socket, name):
        setattr(socket, name, _forbid(name))

# Record environment reads.
env_reads = set()


class _RecordingEnv(dict):
    def __getitem__(self, k):
        env_reads.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        env_reads.add(k)
        return super().get(k, default)

    def __contains__(self, k):
        env_reads.add(k)
        return super().__contains__(k)


_real_environ = os.environ
os.environ = _RecordingEnv(_real_environ)  # type: ignore[assignment]
_before = set(env_reads)

import numpy as np  # noqa: E402  (numpy's own env reads are attributed to numpy below)

_numpy_reads = set(env_reads) - _before
_before = set(env_reads)

import tsecon  # noqa: E402

_import_reads = set(env_reads) - _before
_before = set(env_reads)

y = np.cumsum(np.random.default_rng(0).standard_normal(300))
tsecon.adf(y)
tsecon.arima_fit(y, p=1, d=1, forecast_steps=3)
tsecon.setar_test(np.diff(y), 1, n_boot=20, seed=0)  # a rayon-parallel path
tsecon.random_forest(np.random.default_rng(1).standard_normal((100, 3)), y[:100], n_trees=5, seed=0)
tsecon.summarize(tsecon.acf(y, nlags=5))
_call_reads = set(env_reads) - _before

os.environ = _real_environ  # type: ignore[assignment]
socket.socket.__init__ = _orig_socket_init  # type: ignore[method-assign]

print(f"tsecon {tsecon.__version__} at {tsecon.__file__}")
print(f"socket entry points invoked during import + 5 calls: {len(calls)} {calls[:5]}")
print(f"env vars read by numpy import (attributed to numpy): {sorted(_numpy_reads)}")
print(f"env vars read by `import tsecon`: {sorted(_import_reads)}")
print(f"env vars read during the 5 calls: {sorted(_call_reads)}")
print(f"tsecon module files: {sorted(os.listdir(os.path.dirname(tsecon.__file__)))}")
sys.exit(1 if calls else 0)
