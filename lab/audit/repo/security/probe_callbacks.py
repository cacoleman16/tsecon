#!/usr/bin/env python
"""Callback-surface probes (repo audit, security): how do the three Python
callback bridges behave under a hostile or broken callable?

Surfaces: ``gmm_nonlinear(moments_fn=...)``, ``backtest(forecaster=<callable>)``,
``conformal_forecast(base=<callable>)`` / ``conformal_backtest(base=<callable>)``.

For each, the questions the brief asks:

* an exception raised inside the callback must surface as a Python exception
  (which class? is the original reachable?) — never a ``PanicException`` and
  never a process death;
* a wrong-shaped / NaN / non-numeric / ``None`` return must be a teaching
  refusal, not a panic;
* mutating the array the engine hands over must not corrupt the engine;
* re-entrancy — calling tsecon (the same function included) inside the
  callback — must work or refuse cleanly;
* ``KeyboardInterrupt`` / ``SystemExit`` raised inside the callback must still
  stop the program (and ideally keep their class);
* a second Python thread calling tsecon while the first is inside a callback
  must not deadlock;
* the GIL: is it released during a long compiled call? (measured: how much
  progress a pure-Python counter thread makes while a 3-second Rust call runs).

Every probe prints one line ``PROBE <name>: <verdict> — <evidence>``; the
script exits 0 regardless (it is a recorder, not a gate). Findings are
classified by hand in ``docs/roadmap/27-repo-audit-2026-09/security.md``.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tsecon  # noqa: E402

PANIC = "PanicException"


class Custom(Exception):
    pass


def report(name, verdict, evidence=""):
    print(f"PROBE {name}: {verdict} — {evidence}", flush=True)


def run(name, thunk, expect=None):
    """Run ``thunk``; classify the exception (or success)."""
    try:
        out = thunk()
    except (KeyboardInterrupt, SystemExit) as exc:
        report(name, f"base-exception {type(exc).__name__} propagated", repr(exc)[:120])
        return exc
    except Exception as exc:  # noqa: BLE001
        cause = type(exc.__cause__).__name__ if exc.__cause__ is not None else None
        verdict = type(exc).__name__ + (f" (cause={cause})" if cause else "")
        if expect is not None:
            ok = isinstance(exc, expect) or isinstance(exc.__cause__, expect)
            verdict += " [as expected]" if ok else f" [EXPECTED {expect.__name__}]"
        report(name, verdict, str(exc)[:160].replace("\n", " "))
        return exc
    except BaseException as exc:  # noqa: BLE001 — PanicException lands here
        report(name, f"!!! {type(exc).__name__} (BaseException — uncatchable by except Exception)", str(exc)[:160])
        return exc
    report(name, "returned normally", f"keys={sorted(out)[:6] if isinstance(out, dict) else type(out).__name__}")
    return out


def y_series(T=120, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(T)) + 50.0


# --------------------------------------------------------------------------- #
# gmm_nonlinear(moments_fn)
# --------------------------------------------------------------------------- #
def gmm_probes():
    y = np.random.default_rng(1).standard_normal(200)

    def good(theta):
        mu, s2 = theta
        return np.column_stack([y - mu, (y - mu) ** 2 - s2])

    run("gmm/baseline", lambda: tsecon.gmm_nonlinear(good, [0.0, 1.0]))

    def raises(theta):
        raise Custom("boom")

    run("gmm/raise-custom", lambda: tsecon.gmm_nonlinear(raises, [0.0, 1.0]), expect=Custom)

    calls = [0]

    def raises_later(theta):
        calls[0] += 1
        if calls[0] > 5:
            raise Custom("boom after 5")
        return good(theta)

    run("gmm/raise-after-5-calls", lambda: tsecon.gmm_nonlinear(raises_later, [0.0, 1.0]), expect=Custom)
    report("gmm/raise-after-5-calls/extra-calls", f"callback invoked {calls[0]} times total", "(how many more after the first raise?)")

    run("gmm/return-1d", lambda: tsecon.gmm_nonlinear(lambda th: y - th[0], [0.0]), expect=TypeError)
    run("gmm/return-none", lambda: tsecon.gmm_nonlinear(lambda th: None, [0.0, 1.0]), expect=TypeError)
    run("gmm/return-string", lambda: tsecon.gmm_nonlinear(lambda th: "abc", [0.0, 1.0]), expect=TypeError)
    run("gmm/return-ragged", lambda: tsecon.gmm_nonlinear(lambda th: [[1.0, 2.0], [3.0]], [0.0, 1.0]), expect=TypeError)
    run("gmm/return-empty", lambda: tsecon.gmm_nonlinear(lambda th: np.empty((0, 2)), [0.0, 1.0]), expect=ValueError)
    run("gmm/return-zero-cols", lambda: tsecon.gmm_nonlinear(lambda th: np.empty((200, 0)), [0.0, 1.0]), expect=ValueError)
    run("gmm/return-nan", lambda: tsecon.gmm_nonlinear(lambda th: np.full((200, 2), np.nan), [0.0, 1.0]))
    run("gmm/return-inf", lambda: tsecon.gmm_nonlinear(lambda th: np.full((200, 2), np.inf), [0.0, 1.0]))
    run("gmm/return-huge", lambda: tsecon.gmm_nonlinear(lambda th: np.full((200, 2), 1e308), [0.0, 1.0]))

    n = [0]

    def shape_shifter(theta):
        n[0] += 1
        return good(theta)[: 200 - (n[0] % 3)]

    run("gmm/shape-changes-between-calls", lambda: tsecon.gmm_nonlinear(shape_shifter, [0.0, 1.0]), expect=ValueError)

    def mutates(theta):
        theta[:] = 1e6  # is the parameter array a private copy?
        return good(theta)

    run("gmm/mutate-theta", lambda: tsecon.gmm_nonlinear(mutates, [0.0, 1.0]))

    def reentrant(theta):
        tsecon.acf(y, nlags=3)
        inner = tsecon.gmm_nonlinear(good, [0.0, 1.0])
        return good(theta) + 0 * inner["objective"]

    run("gmm/reentrant-nested-gmm", lambda: tsecon.gmm_nonlinear(reentrant, [0.0, 1.0]))

    def from_thread(theta):
        box = {}

        def worker():
            box["r"] = tsecon.acf(y, nlags=3)

        t = threading.Thread(target=worker)
        t.start()
        t.join(5)
        box["alive"] = t.is_alive()
        if t.is_alive():
            raise RuntimeError("worker thread did not finish: deadlock?")
        return good(theta)

    run("gmm/tsecon-from-another-thread-inside-callback", lambda: tsecon.gmm_nonlinear(from_thread, [0.0, 1.0]))

    def kbi(theta):
        raise KeyboardInterrupt

    run("gmm/keyboardinterrupt", lambda: tsecon.gmm_nonlinear(kbi, [0.0, 1.0]))

    def sysexit(theta):
        raise SystemExit(3)

    run("gmm/systemexit", lambda: tsecon.gmm_nonlinear(sysexit, [0.0, 1.0]))

    run("gmm/non-callable", lambda: tsecon.gmm_nonlinear(42, [0.0, 1.0]), expect=TypeError)
    run("gmm/initial-empty", lambda: tsecon.gmm_nonlinear(good, []), expect=ValueError)
    run("gmm/initial-nan", lambda: tsecon.gmm_nonlinear(good, [float("nan"), 1.0]))
    run("gmm/weight-wrong-size", lambda: tsecon.gmm_nonlinear(good, [0.0, 1.0], weight=[1.0, 0.0, 0.0]), expect=ValueError)


# --------------------------------------------------------------------------- #
# backtest(forecaster=callable) and conformal(base=callable)
# --------------------------------------------------------------------------- #
def _fc_mean(train, h):
    return np.full(h, float(np.mean(train)))


def backtest_probes():
    y = y_series()
    kw = dict(train=60, horizon=2)
    run("backtest/baseline", lambda: tsecon.backtest(y, forecaster=_fc_mean, **kw))

    def raises(train, h):
        raise Custom("boom")

    run("backtest/raise-custom", lambda: tsecon.backtest(y, forecaster=raises, **kw), expect=Custom)

    def kbi(train, h):
        raise KeyboardInterrupt

    run("backtest/keyboardinterrupt", lambda: tsecon.backtest(y, forecaster=kbi, **kw))

    def sysexit(train, h):
        raise SystemExit(3)

    run("backtest/systemexit", lambda: tsecon.backtest(y, forecaster=sysexit, **kw))

    run("backtest/return-scalar", lambda: tsecon.backtest(y, forecaster=lambda tr, h: 1.0, **kw), expect=TypeError)
    run("backtest/return-wrong-len", lambda: tsecon.backtest(y, forecaster=lambda tr, h: np.ones(h + 1), **kw), expect=ValueError)
    run("backtest/return-none", lambda: tsecon.backtest(y, forecaster=lambda tr, h: None, **kw), expect=TypeError)
    run("backtest/return-string", lambda: tsecon.backtest(y, forecaster=lambda tr, h: "ab", **kw), expect=TypeError)
    run("backtest/return-nan", lambda: tsecon.backtest(y, forecaster=lambda tr, h: np.full(h, np.nan), **kw), expect=ValueError)
    run("backtest/return-inf", lambda: tsecon.backtest(y, forecaster=lambda tr, h: np.full(h, np.inf), **kw), expect=ValueError)
    run("backtest/return-2d", lambda: tsecon.backtest(y, forecaster=lambda tr, h: np.ones((h, 1)), **kw), expect=TypeError)
    run("backtest/return-huge", lambda: tsecon.backtest(y, forecaster=lambda tr, h: np.full(h, 1e308), **kw))

    def mutates(train, h):
        train[0] = 1e9  # the engine promises a read-only copy
        return _fc_mean(train, h)

    run("backtest/mutate-train (read-only contract)", lambda: tsecon.backtest(y, forecaster=mutates, **kw), expect=ValueError)

    seen = []

    def records(train, h):
        seen.append((len(train), h, train.flags.writeable, train.dtype.str))
        return _fc_mean(train, h)

    tsecon.backtest(y, forecaster=records, **kw)
    report("backtest/callable-sees", f"{len(seen)} calls; first {seen[0]}, last {seen[-1]}", "(len, h, writeable, dtype)")

    def reentrant(train, h):
        r = tsecon.arima_fit(train, p=1, forecast_steps=h)
        return np.asarray(r["forecast_mean"])[:h]

    run("backtest/reentrant-arima_fit", lambda: tsecon.backtest(y, forecaster=reentrant, **kw))

    depth = [0]

    def recursive(train, h):
        # One level of re-entry only: the inner backtest uses the plain mean
        # forecaster (an inner call that re-entered again would be exponential
        # in the number of origins — a harness bug, not a library one).
        if depth[0] == 0 and len(train) > 70:
            depth[0] += 1
            try:
                tsecon.backtest(train, forecaster=_fc_mean, train=60, horizon=1)
            finally:
                depth[0] -= 1
        return _fc_mean(train, h)

    run("backtest/recursive-backtest-in-forecaster", lambda: tsecon.backtest(y, forecaster=recursive, **kw))

    def from_thread(train, h):
        box = {}

        def worker():
            box["r"] = tsecon.backtest(train, forecaster=_fc_mean, train=30, horizon=1)

        t = threading.Thread(target=worker)
        t.start()
        t.join(10)
        if t.is_alive():
            raise RuntimeError("worker thread did not finish: deadlock?")
        return _fc_mean(train, h)

    run("backtest/tsecon-from-another-thread-inside-callback", lambda: tsecon.backtest(y, forecaster=from_thread, **kw))

    run("backtest/forecaster-int", lambda: tsecon.backtest(y, forecaster=42, **kw), expect=TypeError)
    run("backtest/forecaster-object-with-__call__", lambda: tsecon.backtest(y, forecaster=type("F", (), {"__call__": lambda self, tr, h: _fc_mean(tr, h)})(), **kw))

    # conformal(base=callable) shares the bridge — same probes, fewer of them.
    ck = dict(horizon=2)
    run("conformal_forecast/baseline", lambda: tsecon.conformal_forecast(y, base=_fc_mean, **ck))
    run("conformal_forecast/raise-custom", lambda: tsecon.conformal_forecast(y, base=raises, **ck), expect=Custom)
    run("conformal_forecast/return-nan", lambda: tsecon.conformal_forecast(y, base=lambda tr, h: np.full(h, np.nan), **ck), expect=ValueError)
    run("conformal_forecast/return-wrong-len", lambda: tsecon.conformal_forecast(y, base=lambda tr, h: np.ones(h + 1), **ck), expect=ValueError)
    run("conformal_forecast/keyboardinterrupt", lambda: tsecon.conformal_forecast(y, base=kbi, **ck))
    run("conformal_backtest/baseline", lambda: tsecon.conformal_backtest(y, base=_fc_mean, n_eval=5, **ck))
    run("conformal_backtest/raise-custom", lambda: tsecon.conformal_backtest(y, base=raises, n_eval=5, **ck), expect=Custom)
    run("conformal_backtest/mutate-train", lambda: tsecon.conformal_backtest(y, base=mutates, n_eval=5, **ck), expect=ValueError)


# --------------------------------------------------------------------------- #
# the GIL during a long compiled call
# --------------------------------------------------------------------------- #
def gil_probe():
    y = np.random.default_rng(0).standard_normal(400)
    # Pick a call that takes a few seconds: a bootstrap with many replications.
    t0 = time.time()
    tsecon.setar_test(y, 1, n_boot=200, seed=0)
    unit = time.time() - t0
    n_boot = max(200, int(200 * 3.0 / max(unit, 1e-3)))
    ticks = [0]
    stop = threading.Event()

    def counter():
        while not stop.is_set():
            ticks[0] += 1

    # Baseline: how fast does the counter tick when nothing else runs?
    t = threading.Thread(target=counter)
    t.start()
    time.sleep(1.0)
    stop.set()
    t.join()
    base_rate = ticks[0] / 1.0

    ticks[0] = 0
    stop.clear()
    t = threading.Thread(target=counter)
    t.start()
    t1 = time.time()
    tsecon.setar_test(y, 1, n_boot=n_boot, seed=0)
    el = time.time() - t1
    stop.set()
    t.join()
    rate = ticks[0] / el
    share = rate / base_rate if base_rate else float("nan")
    verdict = "GIL RELEASED during the compiled call" if share > 0.2 else "GIL HELD for the whole compiled call"
    report("gil/long-call", verdict, f"counter thread ran at {share:.1%} of its idle rate during a {el:.1f}s setar_test(n_boot={n_boot})")


def main():
    print(f"tsecon {tsecon.__version__}", flush=True)
    for name, fn in (("gmm", gmm_probes), ("backtest", backtest_probes), ("gil", gil_probe)):
        try:
            fn()
        except BaseException:  # noqa: BLE001 — record, keep going
            print(f"PROBE {name}/<group>: !!! escaped the group runner")
            traceback.print_exc()


if __name__ == "__main__":
    main()
