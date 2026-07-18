#!/usr/bin/env python3
"""Honest benchmark harness for tsecon.

This runner follows the roadmap's *accuracy before speed* rule. For every
operation that both tsecon and a mature reference (statsmodels / arch) can
compute, it does two things, **in this order**:

1. PARITY (the real deliverable). Assert that tsecon's estimates match the
   reference to a stated numerical tolerance. This is machine-independent:
   it does not depend on your CPU, your build flags, or the phase of the
   moon. If parity fails the script exits non-zero -- a benchmark that
   measures the wrong number is worse than no benchmark.

2. TIMING (indicative only). Time both implementations and print a table.
   Timings are reported *after* and *subordinate to* parity, and they are
   loudly labelled with the detected build mode. A DEBUG build of the Rust
   core (what `maturin develop` installs by default) can be ~10-30x slower
   than a `--release` wheel, so debug timings must never be quoted as
   headline speed numbers.

Run:  python benchmarks/bench.py [--repeats N] [--quick]

Exit code 0 iff every parity check passed.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass, field

import numpy as np

import tsecon

# Reference libraries.
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import scipy
from arch import arch_model
import arch


# --------------------------------------------------------------------------
# Build-mode detection.
#
# `maturin develop` (no --release) installs a DEBUG build of the Rust core.
# Debug builds are large and slow. We detect this by comparing the installed
# extension module against the on-disk debug/release artifacts (by size), and
# fall back to a size heuristic when those artifacts are not present (e.g. a
# published wheel). This is best-effort and clearly labelled as such.
# --------------------------------------------------------------------------
def detect_build_mode() -> tuple[str, str]:
    """Return (mode, detail). mode in {'debug', 'release', 'unknown'}."""
    so_path = None
    pkg_dir = os.path.dirname(os.path.abspath(tsecon.__file__))
    for root, _dirs, files in os.walk(pkg_dir):
        for fn in files:
            if fn.endswith((".so", ".dylib", ".pyd")):
                so_path = os.path.join(root, fn)
                break
        if so_path:
            break
    if so_path is None:
        return "unknown", "no compiled extension found next to tsecon"

    so_size = os.path.getsize(so_path)
    # Locate the repo's target/ dir relative to this file, if we are in-tree.
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    dbg = os.path.join(repo, "target", "debug", "libtsecon.dylib")
    rel = os.path.join(repo, "target", "release", "libtsecon.dylib")
    detail = f"{so_path} ({so_size / 1e6:.1f} MB)"
    if os.path.exists(dbg) and os.path.getsize(dbg) == so_size:
        return "debug", detail + " == target/debug/libtsecon.dylib"
    if os.path.exists(rel) and os.path.getsize(rel) == so_size:
        return "release", detail + " == target/release/libtsecon.dylib"
    # Fall back to a coarse size heuristic. tsecon's debug artifact is ~40+ MB
    # of unoptimised code and symbols; a release build is a couple of MB.
    if so_size > 12_000_000:
        return "debug", detail + " (size heuristic: large -> likely debug)"
    return "release", detail + " (size heuristic: small -> likely release)"


# --------------------------------------------------------------------------
# Tiny timing helper: best (min) of `repeats` wall-clock runs after a warmup.
# min is the standard microbenchmark summary -- it is the run least polluted
# by scheduler noise, GC, and turbo throttling.
# --------------------------------------------------------------------------
def best_time(fn, repeats: int) -> float:
    fn()  # warmup / populate caches
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


@dataclass
class ParityRow:
    op: str
    metric: str
    max_abs_diff: float
    tol: float
    passed: bool


@dataclass
class TimingRow:
    op: str
    tsecon_s: float
    ref_s: float
    ref_name: str

    @property
    def speedup(self) -> float:
        return self.ref_s / self.tsecon_s if self.tsecon_s > 0 else float("nan")


@dataclass
class Case:
    """One operation benchmarked against one reference."""
    op: str
    ref_name: str
    parity: list[ParityRow] = field(default_factory=list)
    timing: TimingRow | None = None
    note: str = ""


def _maxdiff(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.max(np.abs(a - b)))


# --------------------------------------------------------------------------
# Cases. Each builds data, checks parity, then times both sides.
# Tolerances are per-metric and deliberately explicit.
# --------------------------------------------------------------------------
def case_adf(rng, repeats) -> Case:
    c = Case("ADF test (regression='c', fixed lag=4)", "statsmodels.tsa.stattools.adfuller")
    n = 500
    y = np.cumsum(rng.standard_normal(n)) + 0.02 * np.arange(n)
    ts = tsecon.adf(y, regression="c", autolag=None, maxlag=4)
    ref = adfuller(y, maxlag=4, regression="c", autolag=None)  # (stat, p, usedlag, nobs, crit, icbest)
    c.parity.append(ParityRow(c.op, "statistic", _maxdiff(ts["statistic"], ref[0]), 1e-6,
                              abs(ts["statistic"] - ref[0]) <= 1e-6))
    c.parity.append(ParityRow(c.op, "p_value", _maxdiff(ts["p_value"], ref[1]), 1e-6,
                              abs(ts["p_value"] - ref[1]) <= 1e-6))
    crit_diff = _maxdiff([ts["crit"]["1%"], ts["crit"]["5%"], ts["crit"]["10%"]],
                         [ref[4]["1%"], ref[4]["5%"], ref[4]["10%"]])
    c.parity.append(ParityRow(c.op, "crit values", crit_diff, 1e-6, crit_diff <= 1e-6))
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.adf(y, regression="c", autolag=None, maxlag=4), repeats),
        best_time(lambda: adfuller(y, maxlag=4, regression="c", autolag=None), repeats),
        c.ref_name,
    )
    return c


def case_var(rng, repeats) -> Case:
    c = Case("VAR(2) coefficients (2 vars, trend='c')", "statsmodels.tsa.api.VAR")
    k, n = 2, 600
    A1 = np.array([[0.4, 0.1], [0.05, 0.3]])
    A2 = np.array([[0.1, 0.0], [0.1, 0.2]])
    cst = np.array([0.2, -0.1])
    Y = np.zeros((n, k))
    for t in range(2, n):
        Y[t] = cst + A1 @ Y[t - 1] + A2 @ Y[t - 2] + rng.standard_normal(k) * 0.5
    data = Y[50:]
    ts = np.asarray(tsecon.var_fit(data, lags=2, trend="c")["params"], dtype=float)
    ref = np.asarray(VAR(data).fit(maxlags=2, trend="c").params, dtype=float)
    d = _maxdiff(ts, ref)
    c.parity.append(ParityRow(c.op, "coef matrix (5x2)", d, 1e-8, d <= 1e-8))
    ts_llf = tsecon.var_fit(data, lags=2, trend="c")["llf"]
    ref_llf = VAR(data).fit(maxlags=2, trend="c").llf
    c.parity.append(ParityRow(c.op, "log-likelihood", abs(ts_llf - ref_llf), 1e-6,
                              abs(ts_llf - ref_llf) <= 1e-6))
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.var_fit(data, lags=2, trend="c"), repeats),
        best_time(lambda: VAR(data).fit(maxlags=2, trend="c"), repeats),
        c.ref_name,
    )
    return c


def case_ols_hac(rng, repeats) -> Case:
    c = Case("OLS + HAC (Newey-West) SEs (maxlags=4, corrected)", "statsmodels OLS cov_type='HAC'")
    n, L = 400, 4
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    beta = np.array([1.0, 2.0, -0.5])
    e = np.zeros(n)
    u = rng.standard_normal(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + u[t]  # serially correlated errors -> HAC bites
    y = X @ beta + e
    ts = tsecon.ols(y, X, se_type="hac", maxlags=L, use_correction=True)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True})
    dp = _maxdiff(ts["params"], ref.params)
    ds = _maxdiff(ts["bse"], ref.bse)
    c.parity.append(ParityRow(c.op, "params", dp, 1e-8, dp <= 1e-8))
    c.parity.append(ParityRow(c.op, "HAC bse", ds, 1e-8, ds <= 1e-8))
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.ols(y, X, se_type="hac", maxlags=L, use_correction=True), repeats),
        best_time(
            lambda: sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True}),
            repeats,
        ),
        c.ref_name,
    )
    return c


def case_garch(rng, repeats) -> Case:
    c = Case("GARCH(1,1) QMLE (constant mean, normal)", "arch.arch_model")
    c.note = (
        "QMLE optimisers differ; parity is asserted at optimiser tolerance "
        "(loglik rtol 1e-5, params atol 1e-3), not machine precision."
    )
    n = 3000
    omega, alpha, beta = 0.05, 0.08, 0.90
    eps = np.empty(n)
    s2 = np.empty(n)
    s2[0] = omega / (1 - alpha - beta)
    z = rng.standard_normal(n)
    eps[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * z[t]
    y = eps

    ts = tsecon.garch_fit(y, vol="garch", mean="constant", dist="normal", p=1, q=1)
    ref = arch_model(y, vol="GARCH", mean="Constant", dist="normal", p=1, q=1).fit(disp="off")
    ll_diff = abs(ts["loglik"] - ref.loglikelihood)
    ll_tol = 1e-5 * abs(ref.loglikelihood)
    c.parity.append(ParityRow(c.op, "log-likelihood (rtol 1e-5)", ll_diff, ll_tol, ll_diff <= ll_tol))
    # arch order: mu, omega, alpha[1], beta[1] -- matches tsecon param_names.
    ref_params = np.asarray(ref.params, dtype=float)
    dp = _maxdiff(ts["params"], ref_params)
    c.parity.append(ParityRow(c.op, "params (atol 1e-3)", dp, 1e-3, dp <= 1e-3))

    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.garch_fit(y, vol="garch", mean="constant", dist="normal", p=1, q=1),
                  repeats),
        best_time(
            lambda: arch_model(y, vol="GARCH", mean="Constant", dist="normal", p=1, q=1).fit(disp="off"),
            repeats,
        ),
        c.ref_name,
    )
    return c


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------
def hr(char="-", width=88):
    print(char * width)


def print_env(build_mode, build_detail):
    hr("=")
    print("tsecon benchmark harness -- environment & provenance")
    hr("=")
    print(f"  timestamp        : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  python           : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform         : {platform.platform()}")
    print(f"  machine          : {platform.machine()}  cpu_count={os.cpu_count()}")
    print(f"  tsecon           : {getattr(tsecon, '__version__', 'unknown')}")
    print(f"  numpy            : {np.__version__}")
    print(f"  scipy            : {scipy.__version__}")
    print(f"  statsmodels      : {statsmodels.__version__}")
    print(f"  arch             : {arch.__version__}")
    print(f"  tsecon build     : {build_mode.upper()}")
    print(f"    detected via   : {build_detail}")
    if build_mode == "debug":
        hr("!")
        print("  !! DEBUG BUILD DETECTED. Timings below are NOT valid speed claims.  !!")
        print("  !! `maturin develop` installs an unoptimised build. Rebuild with a  !!")
        print("  !! release wheel before quoting any timing:                          !!")
        print("  !!     maturin build --release && pip install --force-reinstall \\    !!")
        print("  !!         target/wheels/tsecon-*.whl                                 !!")
        hr("!")


def print_parity(cases):
    hr("=")
    print("PARITY MATRIX  (the deliverable -- machine-independent, must all PASS)")
    hr("=")
    print(f"  {'operation':<44}{'metric':<26}{'max|diff|':>12}  {'tol':>8}  ok")
    hr()
    all_pass = True
    for c in cases:
        for i, p in enumerate(c.parity):
            op = c.op if i == 0 else ""
            status = "PASS" if p.passed else "FAIL"
            all_pass &= p.passed
            print(f"  {op[:43]:<44}{p.metric:<26}{p.max_abs_diff:>12.2e}  {p.tol:>8.0e}  {status}")
        print(f"  {'':<44}vs {c.ref_name}")
        if c.note:
            print(f"  {'':<44}note: {c.note}")
    hr()
    print(f"  RESULT: {'ALL PARITY CHECKS PASSED' if all_pass else 'PARITY FAILURE(S) PRESENT'}")
    return all_pass


def print_timings(cases, build_mode, repeats):
    label = "DEBUG BUILD, INDICATIVE ONLY -- NOT A SPEED CLAIM" if build_mode == "debug" \
        else f"{build_mode} build"
    hr("=")
    print(f"TIMINGS  (best of {repeats}; {label})")
    hr("=")
    print(f"  {'operation':<44}{'tsecon':>12}{'reference':>12}{'ratio':>10}")
    print(f"  {'':<44}{'(ms)':>12}{'(ms)':>12}{'ref/ts':>10}")
    hr()
    faster = slower = 0
    for c in cases:
        t = c.timing
        if t is None:
            continue
        sp = t.speedup
        tag = "faster" if sp > 1 else "SLOWER"
        if sp > 1:
            faster += 1
        else:
            slower += 1
        print(f"  {c.op[:43]:<44}{t.tsecon_s * 1e3:>12.3f}{t.ref_s * 1e3:>12.3f}{sp:>9.2f}x  {tag}")
    hr()
    print(f"  tsecon faster on {faster}/{faster + slower} ops here.")
    if build_mode == "debug":
        print("  Reminder: this is a DEBUG build. A slower result is expected and does")
        print("  NOT reflect release performance. Re-run against a --release wheel.")
    print("  Honesty note: we publish this ratio for EVERY op, wins and losses alike.")


def main() -> int:
    ap = argparse.ArgumentParser(description="tsecon honest benchmark harness")
    ap.add_argument("--repeats", type=int, default=20,
                    help="timing repeats per op (min is reported); default 20")
    ap.add_argument("--quick", action="store_true",
                    help="fewer repeats for a fast smoke run")
    args = ap.parse_args()
    repeats = 3 if args.quick else args.repeats

    build_mode, build_detail = detect_build_mode()
    print_env(build_mode, build_detail)

    rng = np.random.default_rng(20260717)
    cases = [
        case_adf(rng, repeats),
        case_var(rng, repeats),
        case_ols_hac(rng, repeats),
        # GARCH optimisation is the slow one; cap its repeats so the run stays quick.
        case_garch(rng, min(repeats, 3)),
    ]

    ok = print_parity(cases)
    print_timings(cases, build_mode, repeats)
    hr("=")

    if not ok:
        print("FAIL: at least one parity check did not meet tolerance. Exit 1.")
        return 1
    print("OK: all parity checks passed. (Timings are indicative; see build-mode banner.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
