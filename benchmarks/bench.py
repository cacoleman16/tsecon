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
import warnings
from dataclasses import dataclass, field

import numpy as np

import tsecon

# Reference libraries.
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss as sm_kpss, acf as sm_acf, pacf as sm_pacf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.stattools import jarque_bera as sm_jarque_bera
import scipy
from scipy import signal
from scipy.stats import norm
import sklearn
from sklearn.linear_model import ElasticNet, Ridge
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


def _row(case: "Case", metric: str, got, want, tol: float) -> None:
    """Append one ParityRow comparing `got` vs `want` at an explicit `tol`."""
    d = _maxdiff(got, want)
    case.parity.append(ParityRow(case.op, metric, d, tol, d <= tol))


# --------------------------------------------------------------------------
# Deterministic data generators (all driven by the passed rng).
# --------------------------------------------------------------------------
def _ar1(rng, n, phi=0.6, sigma=1.0):
    y = np.zeros(n)
    u = rng.standard_normal(n) * sigma
    for t in range(1, n):
        y[t] = phi * y[t - 1] + u[t]
    return y


def _var2(rng, n=600, burn=50):
    A1 = np.array([[0.4, 0.1], [0.05, 0.3]])
    A2 = np.array([[0.1, 0.0], [0.1, 0.2]])
    cst = np.array([0.2, -0.1])
    Y = np.zeros((n, 2))
    for t in range(2, n):
        Y[t] = cst + A1 @ Y[t - 1] + A2 @ Y[t - 2] + rng.standard_normal(2) * 0.5
    return Y[burn:]


def _reg(rng, n=200, k=5):
    X = rng.standard_normal((n, k))
    beta = np.linspace(1.5, -1.5, k)
    y = X @ beta + rng.standard_normal(n) * 0.5
    return X, y


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


def case_kpss(rng, repeats) -> Case:
    c = Case("KPSS test (regression='c', auto lags)", "statsmodels.tsa.stattools.kpss")
    c.note = ("p-value is interpolated and clipped to [0.01, 0.10] by BOTH sides "
              "(Kwiatkowski table); parity is on the clipped value.")
    y = _ar1(rng, 500)
    ts = tsecon.kpss(y, regression="c")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels warns when p is clipped
        ref = sm_kpss(y, regression="c", nlags="auto")
    _row(c, "statistic", ts["statistic"], ref[0], 1e-10)
    _row(c, "p_value (clipped)", ts["p_value"], ref[1], 1e-10)
    _row(c, "auto lags", ts["lags"], ref[2], 0.0)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.kpss(y, regression="c"), repeats),
        best_time(lambda: sm_kpss(y, regression="c", nlags="auto"), repeats),
        c.ref_name,
    )
    return c


def case_acf(rng, repeats) -> Case:
    c = Case("ACF (20 lags) + Bartlett SEs", "statsmodels.tsa.stattools.acf")
    y = _ar1(rng, 500)
    ts = tsecon.acf(y, nlags=20, adjusted=False)
    ref, ci = sm_acf(y, nlags=20, adjusted=False, fft=True, alpha=0.05)
    _row(c, "acf (adjusted=False)", ts["acf"], ref, 1e-12)
    # statsmodels reports a confidence band, not the SE: invert it.
    ref_se = (ci[:, 1] - ref) / norm.ppf(0.975)
    _row(c, "Bartlett SE", ts["bartlett_se"], ref_se, 1e-12)
    ts_adj = tsecon.acf(y, nlags=20, adjusted=True)
    ref_adj = sm_acf(y, nlags=20, adjusted=True, fft=True)
    _row(c, "acf (adjusted=True)", ts_adj["acf"], ref_adj, 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.acf(y, nlags=20, adjusted=False), repeats),
        best_time(lambda: sm_acf(y, nlags=20, adjusted=False, fft=True), repeats),
        c.ref_name,
    )
    return c


def case_pacf(rng, repeats) -> Case:
    c = Case("PACF (15 lags, Yule-Walker + OLS)", "statsmodels.tsa.stattools.pacf")
    c.note = "tsecon method='yw' is statsmodels method='ywm' (Yule-Walker, no mean adjustment)."
    y = _ar1(rng, 500)
    _row(c, "pacf (yw / ywm)", tsecon.pacf(y, nlags=15, method="yw"),
         sm_pacf(y, nlags=15, method="ywm"), 1e-10)
    _row(c, "pacf (ols)", tsecon.pacf(y, nlags=15, method="ols"),
         sm_pacf(y, nlags=15, method="ols"), 1e-10)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.pacf(y, nlags=15, method="yw"), repeats),
        best_time(lambda: sm_pacf(y, nlags=15, method="ywm"), repeats),
        c.ref_name,
    )
    return c


def case_ljung_box(rng, repeats) -> Case:
    c = Case("Ljung-Box + Box-Pierce (lags 1..10)", "statsmodels.stats.diagnostic.acorr_ljungbox")
    # Deliberately WEAK autocorrelation (phi=0.15, n=300) so the p-values land in
    # the interesting middle of (0, 1). On a strongly autocorrelated series every
    # p-value underflows to ~1e-40 and an absolute p-value tolerance is vacuous.
    y = _ar1(rng, 300, phi=0.15)
    ts = tsecon.ljung_box(y, nlags=10)
    ref = acorr_ljungbox(y, lags=10, boxpierce=True, return_df=True)
    _row(c, "lb_stat", ts["lb_stat"], ref["lb_stat"].values, 1e-10)
    _row(c, "lb_pvalue", ts["lb_pvalue"], ref["lb_pvalue"].values, 1e-12)
    _row(c, "bp_stat", ts["bp_stat"], ref["bp_stat"].values, 1e-10)
    _row(c, "bp_pvalue", ts["bp_pvalue"], ref["bp_pvalue"].values, 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.ljung_box(y, nlags=10), repeats),
        best_time(lambda: acorr_ljungbox(y, lags=10, boxpierce=True, return_df=True), repeats),
        c.ref_name,
    )
    return c


def case_jarque_bera(rng, repeats) -> Case:
    c = Case("Jarque-Bera normality test", "statsmodels.stats.stattools.jarque_bera")
    x = rng.standard_normal(1000) + 0.3 * rng.standard_normal(1000) ** 2  # skewed on purpose
    ts = tsecon.jarque_bera(x)
    ref = sm_jarque_bera(x)  # (JB, p, skew, kurtosis)
    _row(c, "statistic", ts["statistic"], ref[0], 1e-10)
    _row(c, "p_value", ts["p_value"], ref[1], 1e-12)
    _row(c, "skewness", ts["skewness"], ref[2], 1e-12)
    _row(c, "kurtosis", ts["kurtosis"], ref[3], 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.jarque_bera(x), repeats),
        best_time(lambda: sm_jarque_bera(x), repeats),
        c.ref_name,
    )
    return c


def case_arch_lm(rng, repeats) -> Case:
    c = Case("Engle ARCH-LM test (4 lags)", "statsmodels.stats.diagnostic.het_arch")
    # MILD ARCH: strong ARCH drives the p-value to ~1e-55, where an absolute
    # tolerance proves nothing. Weak ARCH keeps p in a range worth comparing.
    n = 800
    e = np.zeros(n)
    s2 = np.ones(n)
    z = rng.standard_normal(n)
    for t in range(1, n):
        s2[t] = 0.9 + 0.05 * e[t - 1] ** 2 + 0.05 * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    ts = tsecon.arch_lm(e, nlags=4)
    ref = het_arch(e, nlags=4)  # (lm, lmpval, fval, fpval)
    _row(c, "LM statistic", ts["statistic"], ref[0], 1e-9)
    _row(c, "LM p_value", ts["p_value"], ref[1], 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.arch_lm(e, nlags=4), repeats),
        best_time(lambda: het_arch(e, nlags=4), repeats),
        c.ref_name,
    )
    return c


def case_johansen(rng, repeats) -> Case:
    c = Case("Johansen cointegration (3 vars, k_ar_diff=1)",
             "statsmodels.tsa.vector_ar.vecm.coint_johansen")
    k, n = 3, 400
    Y = np.zeros((n, k))
    for t in range(1, n):
        Y[t] = Y[t - 1] + rng.standard_normal(k) * 0.5
    Y[:, 1] = Y[:, 0] + rng.standard_normal(n) * 0.3  # one cointegrating relation
    ts = tsecon.johansen(Y, k_ar_diff=1)
    ref = coint_johansen(Y, 0, 1)  # det_order=0 == tsecon's convention
    _row(c, "eigenvalues", ts["eig"], ref.eig, 1e-10)
    _row(c, "trace stat", ts["trace_stat"], ref.lr1, 1e-8)
    _row(c, "max-eig stat", ts["max_eig_stat"], ref.lr2, 1e-8)
    _row(c, "trace crit (90/95/99)", np.array(ts["trace_crit_90_95_99"]), ref.cvt, 1e-12)
    _row(c, "max-eig crit", np.array(ts["max_eig_crit_90_95_99"]), ref.cvm, 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.johansen(Y, k_ar_diff=1), repeats),
        best_time(lambda: coint_johansen(Y, 0, 1), repeats),
        c.ref_name,
    )
    return c


def case_var_irf_fevd(rng, repeats) -> Case:
    c = Case("VAR(2) orthogonalised IRF + FEVD (h=10)", "statsmodels VARResults.irf/.fevd")
    data = _var2(rng)
    ref = VAR(data).fit(maxlags=2, trend="c")
    _row(c, "orth IRF (11x2x2)", np.array(tsecon.var_irf(data, lags=2, horizon=10, orth=True,
                                                        trend="c")),
         ref.irf(10).orth_irfs, 1e-10)
    _row(c, "FEVD (2x10x2)", np.array(tsecon.var_fevd(data, lags=2, horizon=10, trend="c")),
         np.array(ref.fevd(10).decomp), 1e-10)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.var_irf(data, lags=2, horizon=10, orth=True, trend="c"), repeats),
        best_time(lambda: VAR(data).fit(maxlags=2, trend="c").irf(10).orth_irfs, repeats),
        c.ref_name,
    )
    return c


def case_var_granger(rng, repeats) -> Case:
    c = Case("VAR(2) Granger causality F-test", "statsmodels VARResults.test_causality(kind='f')")
    data = _var2(rng)
    ref_fit = VAR(data).fit(maxlags=2, trend="c")
    ts = tsecon.var_granger(data, caused=[0], causing=[1], lags=2, trend="c")
    ref = ref_fit.test_causality(0, [1], kind="f")
    _row(c, "F statistic", ts["statistic"], ref.test_statistic, 1e-10)
    _row(c, "p_value", ts["p_value"], ref.pvalue, 1e-12)
    _row(c, "df (num, den)", [ts["df_num"], ts["df_den"]], [ref.df[0], ref.df[1]], 0.0)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.var_granger(data, caused=[0], causing=[1], lags=2, trend="c"),
                  repeats),
        best_time(lambda: VAR(data).fit(maxlags=2, trend="c").test_causality(0, [1], kind="f"),
                  repeats),
        c.ref_name,
    )
    return c


def case_hp_filter(rng, repeats) -> Case:
    c = Case("HP filter (lambda=1600, two-sided)", "statsmodels.tsa.filters.hp_filter.hpfilter")
    n = 1000
    y = np.cumsum(rng.standard_normal(n)) * 0.3 + 5 * np.sin(np.arange(n) / 40.0)
    ts = tsecon.hp_filter(y, lamb=1600.0)
    ref_cycle, ref_trend = hpfilter(y, lamb=1600)
    # tsecon solves the pentadiagonal system directly; statsmodels does a sparse
    # spsolve. Tolerance reflects that different-elimination-order difference.
    _row(c, "trend", ts["trend"], ref_trend, 1e-8)
    _row(c, "cycle", ts["cycle"], ref_cycle, 1e-8)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.hp_filter(y, lamb=1600.0), repeats),
        best_time(lambda: hpfilter(y, lamb=1600), repeats),
        c.ref_name,
    )
    return c


def case_bk_filter(rng, repeats) -> Case:
    c = Case("Baxter-King band-pass (low=6, high=32, k=12)",
             "statsmodels.tsa.filters.bk_filter.bkfilter")
    c.note = ("bkfilter drops k observations at each end; tsecon returns the same "
              "trimmed series plus `first_index` = k, so the two align element-wise.")
    n = 400
    y = np.cumsum(rng.standard_normal(n)) * 0.3 + 3 * np.sin(np.arange(n) / 20.0)
    ts = tsecon.bk_filter(y, low=6, high=32, k=12)
    ref = bkfilter(y, low=6, high=32, K=12)
    _row(c, "first_index (== K)", ts["first_index"], 12, 0.0)
    _row(c, "cycle", ts["cycle"], ref, 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.bk_filter(y, low=6, high=32, k=12), repeats),
        best_time(lambda: bkfilter(y, low=6, high=32, K=12), repeats),
        c.ref_name,
    )
    return c


def case_cf_filter(rng, repeats) -> Case:
    c = Case("Christiano-Fitzgerald band-pass (low=6, high=32)",
             "statsmodels.tsa.filters.cf_filter.cffilter")
    n = 400
    y = np.cumsum(rng.standard_normal(n)) * 0.3 + 3 * np.sin(np.arange(n) / 20.0)
    ts = tsecon.cf_filter(y, low=6, high=32)
    ref_cycle, ref_trend = cffilter(y, low=6, high=32)
    _row(c, "cycle", ts["cycle"], ref_cycle, 1e-12)
    _row(c, "trend", ts["trend"], ref_trend, 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.cf_filter(y, low=6, high=32), repeats),
        best_time(lambda: cffilter(y, low=6, high=32), repeats),
        c.ref_name,
    )
    return c


def case_periodogram(rng, repeats) -> Case:
    c = Case("Periodogram PSD (boxcar, n=4096)", "scipy.signal.periodogram")
    x = _ar1(rng, 4096, phi=0.7)
    ts = tsecon.periodogram(x, fs=1.0)
    f, p = signal.periodogram(x, fs=1.0, window="boxcar", detrend=False)
    _row(c, "freqs", ts["freqs"], f, 1e-15)
    _row(c, "psd", ts["psd"], p, 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.periodogram(x, fs=1.0), repeats),
        best_time(lambda: signal.periodogram(x, fs=1.0, window="boxcar", detrend=False), repeats),
        c.ref_name,
    )
    return c


def case_welch(rng, repeats) -> Case:
    c = Case("Welch PSD (Hann, nperseg=256, 50% overlap)", "scipy.signal.welch")
    x = _ar1(rng, 4096, phi=0.7)
    ts = tsecon.welch(x, nperseg=256, fs=1.0)
    f, p = signal.welch(x, fs=1.0, nperseg=256, window="hann", detrend=False)
    _row(c, "freqs", ts["freqs"], f, 1e-15)
    _row(c, "psd", ts["psd"], p, 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.welch(x, nperseg=256, fs=1.0), repeats),
        best_time(lambda: signal.welch(x, fs=1.0, nperseg=256, window="hann", detrend=False),
                  repeats),
        c.ref_name,
    )
    return c


def case_ridge(rng, repeats) -> Case:
    c = Case("Ridge regression (alpha=1.0, no intercept)", "sklearn.linear_model.Ridge")
    X, y = _reg(rng, n=400, k=8)
    _row(c, "coef", tsecon.ridge(X, y, alpha=1.0),
         Ridge(alpha=1.0, fit_intercept=False).fit(X, y).coef_, 1e-10)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.ridge(X, y, alpha=1.0), repeats),
        best_time(lambda: Ridge(alpha=1.0, fit_intercept=False).fit(X, y).coef_, repeats),
        c.ref_name,
    )
    return c


def case_elastic_net(rng, repeats) -> Case:
    c = Case("Elastic net / lasso (coordinate descent)", "sklearn.linear_model.ElasticNet")
    c.note = ("Both minimise (1/2n)||y-Xb||^2 + a*l1*||b||_1 + (a/2)(1-l1)||b||^2. "
              "Tolerance reflects coordinate-descent stopping rules, not a formula difference.")
    X, y = _reg(rng, n=400, k=8)
    for l1 in (1.0, 0.5):
        ts = tsecon.elastic_net(X, y, alpha=0.1, l1_ratio=l1, tol=1e-10)
        ref = ElasticNet(alpha=0.1, l1_ratio=l1, fit_intercept=False,
                         tol=1e-12, max_iter=200000).fit(X, y)
        _row(c, f"coef (l1_ratio={l1})", ts["coef"], ref.coef_, 1e-8)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.elastic_net(X, y, alpha=0.1, l1_ratio=0.5, tol=1e-10), repeats),
        best_time(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False,
                                     tol=1e-12, max_iter=200000).fit(X, y), repeats),
        c.ref_name,
    )
    return c


def _het_data(rng, n=400):
    """Design with MILD heteroskedasticity.

    A strong variance gradient sends every p-value to ~1e-26, where comparing
    p-values at an absolute tolerance is vacuous. exp(0.08 * x1) keeps the LM
    and F p-values around 0.09-0.11 -- small enough to be a real rejection
    region, large enough that the p-value comparison actually has content.
    """
    X = rng.standard_normal((n, 2))
    Xc = np.column_stack([np.ones(n), X])
    y = Xc @ np.array([1.0, 2.0, -1.0]) + rng.standard_normal(n) * np.exp(0.08 * X[:, 0])
    return y, Xc


def case_white(rng, repeats) -> Case:
    c = Case("White heteroskedasticity test", "statsmodels.stats.diagnostic.het_white")
    y, X = _het_data(rng)
    ts = tsecon.heteroskedasticity_test(y, X, test="white")
    ref = het_white(sm.OLS(y, X).fit().resid, X)  # (lm, lm_p, f, f_p)
    _row(c, "LM statistic", ts["statistic"], ref[0], 1e-9)
    _row(c, "LM p_value", ts["pvalue"], ref[1], 1e-12)
    _row(c, "F statistic", ts["fstat"], ref[2], 1e-9)
    _row(c, "F p_value", ts["f_pvalue"], ref[3], 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.heteroskedasticity_test(y, X, test="white"), repeats),
        best_time(lambda: het_white(sm.OLS(y, X).fit().resid, X), repeats),
        c.ref_name,
    )
    return c


def case_breusch_pagan(rng, repeats) -> Case:
    c = Case("Breusch-Pagan test (Koenker studentised)",
             "statsmodels.stats.diagnostic.het_breuschpagan")
    y, X = _het_data(rng)
    ts = tsecon.heteroskedasticity_test(y, X, test="breusch_pagan")
    ref = het_breuschpagan(sm.OLS(y, X).fit().resid, X)
    _row(c, "LM statistic", ts["statistic"], ref[0], 1e-9)
    _row(c, "LM p_value", ts["pvalue"], ref[1], 1e-12)
    _row(c, "F statistic", ts["fstat"], ref[2], 1e-9)
    _row(c, "F p_value", ts["f_pvalue"], ref[3], 1e-12)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.heteroskedasticity_test(y, X, test="breusch_pagan"), repeats),
        best_time(lambda: het_breuschpagan(sm.OLS(y, X).fit().resid, X), repeats),
        c.ref_name,
    )
    return c


def case_reset(rng, repeats) -> Case:
    c = Case("Ramsey RESET (powers of yhat up to 3)",
             "statsmodels.stats.diagnostic.linear_reset")
    y, X = _het_data(rng)
    ts = tsecon.reset_test(y, X, max_power=3)
    ref = linear_reset(sm.OLS(y, X).fit(), power=3, test_type="fitted", use_f=True)
    _row(c, "F statistic", ts["fstat"], ref.statistic, 1e-9)
    _row(c, "p_value", ts["pvalue"], ref.pvalue, 1e-10)
    _row(c, "df (num, den)", [ts["df_num"], ts["df_den"]], [ref.df_num, ref.df_denom], 0.0)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.reset_test(y, X, max_power=3), repeats),
        best_time(lambda: linear_reset(sm.OLS(y, X).fit(), power=3, test_type="fitted",
                                       use_f=True), repeats),
        c.ref_name,
    )
    return c


def case_gjr(rng, repeats) -> Case:
    c = Case("GJR-GARCH(1,1,1) QMLE (constant mean, normal)", "arch.arch_model (o=1)")
    c.note = (
        "Leverage-term QMLE; two different optimisers, so parity is at optimiser "
        "tolerance (loglik rtol 1e-5, params atol 1e-3), not machine precision."
    )
    n = 2000
    omega, alpha, gamma, beta = 0.05, 0.03, 0.10, 0.90
    eps = np.empty(n)
    s2 = np.empty(n)
    s2[0] = omega / (1 - alpha - gamma / 2 - beta)
    z = rng.standard_normal(n)
    eps[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = (omega + alpha * eps[t - 1] ** 2
                 + gamma * eps[t - 1] ** 2 * (eps[t - 1] < 0) + beta * s2[t - 1])
        eps[t] = np.sqrt(s2[t]) * z[t]

    ts = tsecon.garch_fit(eps, vol="gjr", mean="constant", dist="normal", p=1, o=1, q=1)
    ref = arch_model(eps, vol="GARCH", mean="Constant", dist="normal", p=1, o=1, q=1).fit(disp="off")
    ll_diff = abs(ts["loglik"] - ref.loglikelihood)
    ll_tol = 1e-5 * abs(ref.loglikelihood)
    c.parity.append(ParityRow(c.op, "log-likelihood (rtol 1e-5)", ll_diff, ll_tol,
                              ll_diff <= ll_tol))
    _row(c, "params (atol 1e-3)", ts["params"], np.asarray(ref.params, dtype=float), 1e-3)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.garch_fit(eps, vol="gjr", mean="constant", dist="normal",
                                           p=1, o=1, q=1), repeats),
        best_time(lambda: arch_model(eps, vol="GARCH", mean="Constant", dist="normal",
                                     p=1, o=1, q=1).fit(disp="off"), repeats),
        c.ref_name,
    )
    return c


def case_egarch(rng, repeats) -> Case:
    c = Case("EGARCH(1,1,1) QMLE (constant mean, normal)", "arch.arch_model (vol='EGARCH')")
    c.note = ("Same parameterisation (mu, omega, alpha, gamma, beta) on both sides; "
              "parity at optimiser tolerance (loglik rtol 1e-5, params atol 1e-3).")
    y = rng.standard_normal(1500)
    ts = tsecon.garch_fit(y, vol="egarch", mean="constant", dist="normal", p=1, o=1, q=1)
    ref = arch_model(y, vol="EGARCH", mean="Constant", dist="normal", p=1, o=1, q=1).fit(disp="off")
    ll_diff = abs(ts["loglik"] - ref.loglikelihood)
    ll_tol = 1e-5 * abs(ref.loglikelihood)
    c.parity.append(ParityRow(c.op, "log-likelihood (rtol 1e-5)", ll_diff, ll_tol,
                              ll_diff <= ll_tol))
    _row(c, "params (atol 1e-3)", ts["params"], np.asarray(ref.params, dtype=float), 1e-3)
    c.timing = TimingRow(
        c.op,
        best_time(lambda: tsecon.garch_fit(y, vol="egarch", mean="constant", dist="normal",
                                           p=1, o=1, q=1), repeats),
        best_time(lambda: arch_model(y, vol="EGARCH", mean="Constant", dist="normal",
                                     p=1, o=1, q=1).fit(disp="off"), repeats),
        c.ref_name,
    )
    return c


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------
def hr(char="-", width=94):
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
    print(f"  scikit-learn     : {sklearn.__version__}")
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
    print(f"  {'operation':<50}{'metric':<26}{'max|diff|':>12}  {'tol':>8}  ok")
    hr()
    all_pass = True
    for c in cases:
        for i, p in enumerate(c.parity):
            op = c.op if i == 0 else ""
            status = "PASS" if p.passed else "FAIL"
            all_pass &= p.passed
            print(f"  {op[:49]:<50}{p.metric:<26}{p.max_abs_diff:>12.2e}  {p.tol:>8.0e}  {status}")
        print(f"  {'':<50}vs {c.ref_name}")
        if c.note:
            print(f"  {'':<50}note: {c.note}")
    hr()
    print(f"  RESULT: {'ALL PARITY CHECKS PASSED' if all_pass else 'PARITY FAILURE(S) PRESENT'}")
    return all_pass


def print_timings(cases, build_mode, repeats):
    label = "DEBUG BUILD, INDICATIVE ONLY -- NOT A SPEED CLAIM" if build_mode == "debug" \
        else f"{build_mode} build"
    hr("=")
    print(f"TIMINGS  (best of {repeats}; {label})")
    hr("=")
    print(f"  {'operation':<50}{'tsecon':>12}{'reference':>12}{'ratio':>10}")
    print(f"  {'':<50}{'(ms)':>12}{'(ms)':>12}{'ref/ts':>10}")
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
        print(f"  {c.op[:49]:<50}{t.tsecon_s * 1e3:>12.3f}{t.ref_s * 1e3:>12.3f}{sp:>9.2f}x  {tag}")
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
        case_gjr(rng, min(repeats, 3)),
        case_egarch(rng, min(repeats, 3)),
        # Unit-root / stationarity.
        case_kpss(rng, repeats),
        # Serial-correlation diagnostics.
        case_acf(rng, repeats),
        case_pacf(rng, repeats),
        case_ljung_box(rng, repeats),
        # Residual diagnostics.
        case_jarque_bera(rng, repeats),
        case_arch_lm(rng, repeats),
        case_white(rng, repeats),
        case_breusch_pagan(rng, repeats),
        case_reset(rng, repeats),
        # Multivariate.
        case_johansen(rng, repeats),
        case_var_irf_fevd(rng, repeats),
        case_var_granger(rng, repeats),
        # Filters and spectra.
        case_hp_filter(rng, repeats),
        case_bk_filter(rng, repeats),
        case_cf_filter(rng, repeats),
        case_periodogram(rng, repeats),
        case_welch(rng, repeats),
        # Penalised regression.
        case_ridge(rng, repeats),
        case_elastic_net(rng, repeats),
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
