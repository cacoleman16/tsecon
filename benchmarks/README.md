# tsecon benchmarks

An **honest** benchmark harness. tsecon's roadmap has one non-negotiable rule
for benchmarks: **accuracy before speed**. A fast wrong answer is worthless, so
this harness is built to make the *correctness* claim first and loudest, and to
treat timing as a subordinate, heavily-caveated afterthought.

The deliverable here is the **parity matrix**, not a speedup number.

```
python benchmarks/bench.py            # full run
python benchmarks/bench.py --quick    # fast smoke run (fewer repeats)
python benchmarks/bench.py --repeats 50
```

The script exits `0` **iff** every parity check passes, so it doubles as a
cross-library correctness gate you can run in CI.

## What it does, in order

For each operation that both tsecon and a mature reference
(statsmodels / arch / scipy / scikit-learn) can compute, the runner:

1. **Asserts estimate parity first.** It computes the same quantity both ways
   and checks that they agree to a stated tolerance. This is the machine-
   independent, honest core of the benchmark: it does not depend on your CPU,
   your compiler flags, or how the wheel was built. If parity fails, the script
   exits non-zero.
2. **Then times both** (best of *N* runs) and prints a ratio — labelled with the
   detected build mode.

Operations currently covered — **25 cases, 65 parity metrics**, each against a
battle-tested reference:

| Operation | tsecon | Reference |
|---|---|---|
| **Unit root / stationarity** | | |
| ADF test statistic, p-value, critical values | `tsecon.adf` | `statsmodels.tsa.stattools.adfuller` |
| KPSS statistic, p-value, auto lag count | `tsecon.kpss` | `statsmodels.tsa.stattools.kpss` |
| **Serial correlation** | | |
| ACF (adjusted and unadjusted) + Bartlett SEs | `tsecon.acf` | `statsmodels.tsa.stattools.acf` |
| PACF (Yule–Walker and OLS) | `tsecon.pacf` | `statsmodels.tsa.stattools.pacf` |
| Ljung–Box + Box–Pierce, lags 1..10 | `tsecon.ljung_box` | `statsmodels.stats.diagnostic.acorr_ljungbox` |
| **Residual diagnostics** | | |
| Jarque–Bera stat, p, skewness, kurtosis | `tsecon.jarque_bera` | `statsmodels.stats.stattools.jarque_bera` |
| Engle ARCH-LM | `tsecon.arch_lm` | `statsmodels.stats.diagnostic.het_arch` |
| White heteroskedasticity (LM + F forms) | `tsecon.heteroskedasticity_test` | `statsmodels.stats.diagnostic.het_white` |
| Breusch–Pagan, Koenker studentised (LM + F) | `tsecon.heteroskedasticity_test` | `statsmodels.stats.diagnostic.het_breuschpagan` |
| Ramsey RESET | `tsecon.reset_test` | `statsmodels.stats.diagnostic.linear_reset` |
| **Regression** | | |
| OLS + HAC (Newey–West) standard errors | `tsecon.ols` | `statsmodels` OLS `cov_type='HAC'` |
| Ridge coefficients | `tsecon.ridge` | `sklearn.linear_model.Ridge` |
| Elastic net / lasso coefficients | `tsecon.elastic_net` | `sklearn.linear_model.ElasticNet` |
| **Multivariate** | | |
| VAR(p) coefficient matrix + log-likelihood | `tsecon.var_fit` | `statsmodels.tsa.api.VAR` |
| VAR orthogonalised IRFs + FEVD | `tsecon.var_irf`, `tsecon.var_fevd` | `statsmodels` `VARResults.irf/.fevd` |
| VAR Granger-causality F test | `tsecon.var_granger` | `VARResults.test_causality(kind='f')` |
| Johansen cointegration (trace, max-eig, crit) | `tsecon.johansen` | `statsmodels…vecm.coint_johansen` |
| **Filters and spectra** | | |
| Hodrick–Prescott trend + cycle | `tsecon.hp_filter` | `statsmodels…hp_filter.hpfilter` |
| Baxter–King band-pass cycle | `tsecon.bk_filter` | `statsmodels…bk_filter.bkfilter` |
| Christiano–Fitzgerald band-pass trend + cycle | `tsecon.cf_filter` | `statsmodels…cf_filter.cffilter` |
| Periodogram PSD | `tsecon.periodogram` | `scipy.signal.periodogram` |
| Welch averaged-periodogram PSD | `tsecon.welch` | `scipy.signal.welch` |
| **Volatility** | | |
| GARCH(1,1) QMLE log-likelihood + params | `tsecon.garch_fit` | `arch.arch_model` |
| GJR-GARCH(1,1,1) QMLE log-likelihood + params | `tsecon.garch_fit(vol='gjr')` | `arch.arch_model(o=1)` |
| EGARCH(1,1,1) QMLE log-likelihood + params | `tsecon.garch_fit(vol='egarch')` | `arch.arch_model(vol='EGARCH')` |

### What is *not* here, and why

A parity row is only worth having when the two sides compute the **same
estimator under the same convention**. Two categories are therefore absent:

1. **No external reference exists.** Much of tsecon's surface — DSGE solution,
   AFNS/dynamic Nelson–Siegel, sign-restricted SVAR, local projections and
   LP-IV, MIDAS, nowcasting, the realized-measure family — has no equivalent in
   the installed reference stack. Those are covered by the Rust unit tests and
   the replication tier instead; inventing a "reference" for them here would be
   theatre, not evidence.
2. **Conventions differ enough that parity would need a fudge.** Nothing is
   currently excluded on these grounds, but the rule stands: if matching a
   reference requires widening a tolerance past what optimiser or summation-order
   differences justify, the case is dropped and the reason recorded here rather
   than shipped as a green row.

`long_run_variance` sits in between: statsmodels exposes no standalone kernel
LRV estimator with a matching bandwidth rule, so it has no direct row — but it
is exercised indirectly through the HAC standard-error case.

## The honesty rules

These are the rules the harness enforces or the numbers you publish must obey:

1. **Verify identical estimates BEFORE timing.** Timing two functions that
   compute different numbers is meaningless. Parity is asserted first, every
   run, and gates the exit code.
2. **State the tolerance, and never widen it to go green.** Each metric carries
   an explicit tolerance that reflects a *known* source of numerical difference,
   nothing more:
   - **Closed-form linear algebra** (ADF, VAR, OLS/HAC, ACF/PACF, the diagnostic
     tests, Johansen, the filters, the spectra, ridge) matches to machine
     precision. Observed disagreements are `0` to `~8e-11`; tolerances are set at
     `1e-15`–`1e-8` depending only on the conditioning of the op (Johansen's
     `1e-8` covers a generalised eigenproblem; `1e-15` is used where the two
     sides literally produce identical bits).
   - **Iterative fits** are looser *for a stated reason*. Elastic net / lasso
     (`1e-8`) differ by coordinate-descent stopping rules. GARCH, GJR-GARCH and
     EGARCH QMLE match at *optimiser* tolerance (log-likelihood `rtol 1e-5`,
     params `atol 1e-3`) because two optimisers do not land on bit-identical
     parameters.

   A tolerance is never loosened to make a row pass. If a candidate operation
   only agrees under a fudge, the case is **dropped and documented**, not shipped
   green — see "What is *not* here, and why" above.
3. **Publish the runner, hardware, and versions.** The script prints a
   provenance banner (timestamp, OS, CPU, Python, and the version of tsecon,
   numpy, scipy, statsmodels, arch, scikit-learn) on every run. Paste it *with*
   any numbers you quote.
4. **Publish losses, not just wins.** The timing table prints the ref/tsecon
   ratio for **every** op, tagged `faster` or `SLOWER`, and reports the win
   count as a fraction (e.g. "faster on 22/25 ops here"). Cases where tsecon is
   *not* faster are shown, never dropped — the three QMLE volatility fits below
   are all losses, and they are printed in the same table as the wins.
5. **Make the comparison non-trivial where you can.** Several diagnostic cases
   deliberately use *weak* effects (mild ARCH, mild heteroskedasticity, `phi=0.15`
   autocorrelation) so the p-values land in the middle of `(0, 1)`. Under a
   strong effect every p-value underflows to `~1e-40` and an absolute p-value
   tolerance is satisfied by any two implementations, agreeing or not.
6. **Timings are indicative.** `min`-of-N wall-clock, single machine, tiny
   inputs. They indicate order-of-magnitude behaviour, nothing finer.

## ⚠️ The build-mode caveat (read this before quoting any timing)

**`maturin develop` installs a DEBUG build of the Rust core.** Debug builds are
compiled with no optimisation and full symbols — they can be **10–30× slower**
than a release build, and they are the default for local development.

Concretely, in this working tree:

| Artifact | Size |
|---|---|
| `target/debug/libtsecon.dylib` (what `maturin develop` installs) | ~43.4 MB |
| `target/release/libtsecon.dylib` (an optimised wheel) | ~6.0 MB |

The harness **auto-detects** the build mode by comparing the installed
extension against these artifacts (falling back to a size heuristic for a
published wheel) and prints a giant `DEBUG BUILD DETECTED` banner when it finds
one. **Debug-build timings must never be presented as headline speed claims.**

To produce timings worth quoting, build and install a release wheel first:

```
maturin build --release
pip install --force-reinstall target/wheels/tsecon-*.whl
python benchmarks/bench.py
```

Then the provenance banner will read `tsecon build : RELEASE` and the timings
become meaningful (still indicative, still single-machine).

## Reading the output

- **PARITY MATRIX** — the deliverable. `max|diff|` is the largest absolute
  disagreement for that metric; `tol` is the threshold; `ok` must be `PASS`.
- **TIMINGS** — `ratio ref/ts` > 1 means tsecon is faster; the row is tagged
  `faster`/`SLOWER`. The header states the build mode.

### Example run (RELEASE build — these timings *are* quotable)

Verbatim output of `python benchmarks/bench.py`, captured on this working tree
against a release build (`maturin develop -m bindings/python/Cargo.toml
--release`). Hardware and versions as printed by the harness's own provenance
banner:

```
  timestamp        : 2026-07-18 13:59:10 EDT
  python           : 3.12.7 (CPython)
  platform         : macOS-26.5.2-arm64-arm-64bit
  machine          : arm64  cpu_count=14
  tsecon           : 0.0.1
  numpy            : 2.5.1
  scipy            : 1.18.0
  statsmodels      : 0.14.6
  arch             : 8.0.0
  scikit-learn     : 1.9.0
  tsecon build     : RELEASE
    detected via   : /Users/chasecoleman/Time-Series-python/bindings/python/python/tsecon/_core.abi3.so (6.1 MB) (size heuristic: small -> likely release)
```

**Parity — all 65 metrics across 25 operations PASS.** This is the deliverable,
and it is build-independent: a debug build produces the identical parity table.

```
PARITY MATRIX  (the deliverable -- machine-independent, must all PASS)
==============================================================================================
  operation                                         metric                       max|diff|       tol  ok
----------------------------------------------------------------------------------------------
  ADF test (regression='c', fixed lag=4)            statistic                     6.66e-16     1e-06  PASS
                                                    p_value                       2.22e-16     1e-06  PASS
                                                    crit values                   0.00e+00     1e-06  PASS
                                                    vs statsmodels.tsa.stattools.adfuller
  VAR(2) coefficients (2 vars, trend='c')           coef matrix (5x2)             5.55e-16     1e-08  PASS
                                                    log-likelihood                1.14e-13     1e-06  PASS
                                                    vs statsmodels.tsa.api.VAR
  OLS + HAC (Newey-West) SEs (maxlags=4, corrected) params                        4.22e-15     1e-08  PASS
                                                    HAC bse                       2.29e-16     1e-08  PASS
                                                    vs statsmodels OLS cov_type='HAC'
  GARCH(1,1) QMLE (constant mean, normal)           log-likelihood (rtol 1e-5)    1.46e-07     5e-02  PASS
                                                    params (atol 1e-3)            7.82e-06     1e-03  PASS
                                                    vs arch.arch_model
                                                    note: QMLE optimisers differ; parity is asserted at optimiser tolerance (loglik rtol 1e-5, params atol 1e-3), not machine precision.
  GJR-GARCH(1,1,1) QMLE (constant mean, normal)     log-likelihood (rtol 1e-5)    2.47e-09     4e-02  PASS
                                                    params (atol 1e-3)            1.49e-06     1e-03  PASS
                                                    vs arch.arch_model (o=1)
                                                    note: Leverage-term QMLE; two different optimisers, so parity is at optimiser tolerance (loglik rtol 1e-5, params atol 1e-3), not machine precision.
  EGARCH(1,1,1) QMLE (constant mean, normal)        log-likelihood (rtol 1e-5)    2.99e-08     2e-02  PASS
                                                    params (atol 1e-3)            1.19e-05     1e-03  PASS
                                                    vs arch.arch_model (vol='EGARCH')
                                                    note: Same parameterisation (mu, omega, alpha, gamma, beta) on both sides; parity at optimiser tolerance (loglik rtol 1e-5, params atol 1e-3).
  KPSS test (regression='c', auto lags)             statistic                     5.55e-17     1e-10  PASS
                                                    p_value (clipped)             0.00e+00     1e-10  PASS
                                                    auto lags                     0.00e+00     0e+00  PASS
                                                    vs statsmodels.tsa.stattools.kpss
                                                    note: p-value is interpolated and clipped to [0.01, 0.10] by BOTH sides (Kwiatkowski table); parity is on the clipped value.
  ACF (20 lags) + Bartlett SEs                      acf (adjusted=False)          3.33e-16     1e-12  PASS
                                                    Bartlett SE                   2.78e-17     1e-12  PASS
                                                    acf (adjusted=True)           2.22e-16     1e-12  PASS
                                                    vs statsmodels.tsa.stattools.acf
  PACF (15 lags, Yule-Walker + OLS)                 pacf (yw / ywm)               1.13e-15     1e-10  PASS
                                                    pacf (ols)                    2.03e-15     1e-10  PASS
                                                    vs statsmodels.tsa.stattools.pacf
                                                    note: tsecon method='yw' is statsmodels method='ywm' (Yule-Walker, no mean adjustment).
  Ljung-Box + Box-Pierce (lags 1..10)               lb_stat                       5.33e-15     1e-10  PASS
                                                    lb_pvalue                     4.22e-15     1e-12  PASS
                                                    bp_stat                       2.66e-15     1e-10  PASS
                                                    bp_pvalue                     4.39e-15     1e-12  PASS
                                                    vs statsmodels.stats.diagnostic.acorr_ljungbox
  Jarque-Bera normality test                        statistic                     4.26e-14     1e-10  PASS
                                                    p_value                       2.09e-29     1e-12  PASS
                                                    skewness                      8.33e-16     1e-12  PASS
                                                    kurtosis                      8.88e-16     1e-12  PASS
                                                    vs statsmodels.stats.stattools.jarque_bera
  Engle ARCH-LM test (4 lags)                       LM statistic                  5.30e-13     1e-09  PASS
                                                    LM p_value                    8.98e-14     1e-12  PASS
                                                    vs statsmodels.stats.diagnostic.het_arch
  White heteroskedasticity test                     LM statistic                  3.55e-13     1e-09  PASS
                                                    LM p_value                    7.29e-15     1e-12  PASS
                                                    F statistic                   7.02e-14     1e-09  PASS
                                                    F p_value                     5.62e-15     1e-12  PASS
                                                    vs statsmodels.stats.diagnostic.het_white
  Breusch-Pagan test (Koenker studentised)          LM statistic                  0.00e+00     1e-09  PASS
                                                    LM p_value                    2.78e-17     1e-12  PASS
                                                    F statistic                   2.22e-15     1e-09  PASS
                                                    F p_value                     6.69e-15     1e-12  PASS
                                                    vs statsmodels.stats.diagnostic.het_breuschpagan
  Ramsey RESET (powers of yhat up to 3)             F statistic                   1.07e-13     1e-09  PASS
                                                    p_value                       1.62e-13     1e-10  PASS
                                                    df (num, den)                 0.00e+00     0e+00  PASS
                                                    vs statsmodels.stats.diagnostic.linear_reset
  Johansen cointegration (3 vars, k_ar_diff=1)      eigenvalues                   1.21e-13     1e-10  PASS
                                                    trace stat                    7.76e-11     1e-08  PASS
                                                    max-eig stat                  7.77e-11     1e-08  PASS
                                                    trace crit (90/95/99)         0.00e+00     1e-12  PASS
                                                    max-eig crit                  0.00e+00     1e-12  PASS
                                                    vs statsmodels.tsa.vector_ar.vecm.coint_johansen
  VAR(2) orthogonalised IRF + FEVD (h=10)           orth IRF (11x2x2)             2.22e-16     1e-10  PASS
                                                    FEVD (2x10x2)                 3.33e-16     1e-10  PASS
                                                    vs statsmodels VARResults.irf/.fevd
  VAR(2) Granger causality F-test                   F statistic                   2.84e-14     1e-10  PASS
                                                    p_value                       1.30e-17     1e-12  PASS
                                                    df (num, den)                 0.00e+00     0e+00  PASS
                                                    vs statsmodels VARResults.test_causality(kind='f')
  HP filter (lambda=1600, two-sided)                trend                         4.15e-12     1e-08  PASS
                                                    cycle                         4.15e-12     1e-08  PASS
                                                    vs statsmodels.tsa.filters.hp_filter.hpfilter
  Baxter-King band-pass (low=6, high=32, k=12)      first_index (== K)            0.00e+00     0e+00  PASS
                                                    cycle                         1.03e-15     1e-12  PASS
                                                    vs statsmodels.tsa.filters.bk_filter.bkfilter
                                                    note: bkfilter drops k observations at each end; tsecon returns the same trimmed series plus `first_index` = k, so the two align element-wise.
  Christiano-Fitzgerald band-pass (low=6, high=32)  cycle                         2.94e-15     1e-12  PASS
                                                    trend                         3.55e-15     1e-12  PASS
                                                    vs statsmodels.tsa.filters.cf_filter.cffilter
  Periodogram PSD (boxcar, n=4096)                  freqs                         0.00e+00     1e-15  PASS
                                                    psd                           4.26e-14     1e-12  PASS
                                                    vs scipy.signal.periodogram
  Welch PSD (Hann, nperseg=256, 50% overlap)        freqs                         0.00e+00     1e-15  PASS
                                                    psd                           1.07e-14     1e-12  PASS
                                                    vs scipy.signal.welch
  Ridge regression (alpha=1.0, no intercept)        coef                          2.22e-15     1e-10  PASS
                                                    vs sklearn.linear_model.Ridge
  Elastic net / lasso (coordinate descent)          coef (l1_ratio=1.0)           3.09e-13     1e-08  PASS
                                                    coef (l1_ratio=0.5)           1.47e-13     1e-08  PASS
                                                    vs sklearn.linear_model.ElasticNet
                                                    note: Both minimise (1/2n)||y-Xb||^2 + a*l1*||b||_1 + (a/2)(1-l1)||b||^2. Tolerance reflects coordinate-descent stopping rules, not a formula difference.
----------------------------------------------------------------------------------------------
  RESULT: ALL PARITY CHECKS PASSED
==============================================================================================
```

Timings, subordinate and indicative:

```
TIMINGS  (best of 20; release build)
==============================================================================================
  operation                                               tsecon   reference     ratio
                                                            (ms)        (ms)    ref/ts
----------------------------------------------------------------------------------------------
  ADF test (regression='c', fixed lag=4)                   0.014       0.150    11.09x  faster
  VAR(2) coefficients (2 vars, trend='c')                  0.023       0.501    21.46x  faster
  OLS + HAC (Newey-West) SEs (maxlags=4, corrected)        0.024       0.082     3.39x  faster
  GARCH(1,1) QMLE (constant mean, normal)                 19.406       7.936     0.41x  SLOWER
  GJR-GARCH(1,1,1) QMLE (constant mean, normal)           19.081       8.400     0.44x  SLOWER
  EGARCH(1,1,1) QMLE (constant mean, normal)             100.993       6.248     0.06x  SLOWER
  KPSS test (regression='c', auto lags)                    0.004       0.020     4.92x  faster
  ACF (20 lags) + Bartlett SEs                             0.006       0.024     4.24x  faster
  PACF (15 lags, Yule-Walker + OLS)                        0.005       0.284    60.33x  faster
  Ljung-Box + Box-Pierce (lags 1..10)                      0.003       0.104    35.06x  faster
  Jarque-Bera normality test                               0.002       0.262   128.27x  faster
  Engle ARCH-LM test (4 lags)                              0.011       0.162    14.91x  faster
  White heteroskedasticity test                            0.014       0.239    16.50x  faster
  Breusch-Pagan test (Koenker studentised)                 0.009       0.188    21.16x  faster
  Ramsey RESET (powers of yhat up to 3)                    0.013       0.222    17.78x  faster
  Johansen cointegration (3 vars, k_ar_diff=1)             0.022       0.322    14.43x  faster
  VAR(2) orthogonalised IRF + FEVD (h=10)                  0.026       0.660    25.03x  faster
  VAR(2) Granger causality F-test                          0.020       0.745    36.41x  faster
  HP filter (lambda=1600, two-sided)                       0.018       0.331    18.29x  faster
  Baxter-King band-pass (low=6, high=32, k=12)             0.002       0.032    15.76x  faster
  Christiano-Fitzgerald band-pass (low=6, high=32)         0.104       3.039    29.13x  faster
  Periodogram PSD (boxcar, n=4096)                         0.034       0.204     5.93x  faster
  Welch PSD (Hann, nperseg=256, 50% overlap)               0.027       0.187     6.98x  faster
  Ridge regression (alpha=1.0, no intercept)               0.025       0.173     6.85x  faster
  Elastic net / lasso (coordinate descent)                 0.030       0.151     4.98x  faster
----------------------------------------------------------------------------------------------
  tsecon faster on 22/25 ops here.
  Honesty note: we publish this ratio for EVERY op, wins and losses alike.
==============================================================================================
```

**We publish the losses.** All three QMLE volatility fits are still *slower*
than `arch`: GARCH(1,1) at `0.41x`, GJR at `0.44x`, and EGARCH at `0.06x`.

GARCH and GJR used to be worse — `0.23x` and `0.16x`. They improved by **1.8×
and 2.8×** when the estimation-time likelihood was made allocation-free and
given an **analytic gradient**: the optimiser had been driving a plain closure,
so every gradient cost `2k = 8` central-difference probes of the full
likelihood. Profiling put one fit at 2543 likelihood evaluations; the fused
objective also cut the per-evaluation cost from 18.3 µs to 10.9 µs. EGARCH was
deliberately left on the old path (it needs its own fused recursion), so it is
now by a wide margin the worst result in the suite, ~16× slower.

**What we chose not to do.** About two-thirds of the remaining GARCH/GJR time is
the Nelder-Mead polish stage. On these series it moves the log-likelihood by
~1e-12 — it starts at the optimum and finds nothing — and disabling its restarts
would take the total to roughly parity with `arch`. We measured that and did not
take it: the restarts are the documented guard against Nelder-Mead false
convergence, and trading a robustness property for a benchmark number is exactly
the kind of tuning-to-the-test this file exists to refuse. We pay for it in the
timing column instead.

Those three ratios are also the *least* stable numbers in the table, because the
optimiser's iteration count depends on the path it takes, not just on the data.
Treat the QMLE rows as "clearly slower, order-of-magnitude", never as a precise
ratio. The deterministic closed-form rows are stable to a few percent by
comparison.

The parity rows directly above show all three still agree with `arch` on the
*answer*, so these are purely performance deficits. That parity is also what
licensed the optimisation: an analytic gradient is not the central-difference
gradient, so the optimiser now walks a different path and the fitted parameters
moved by ~1e-8 (relative ~5e-7) — far inside the stated optimiser tolerance, and
the goldens caught the question rather than letting it pass unexamined.

### On the DEBUG build

There is deliberately no debug example table here, because publishing one
invites it being quoted. The rule is simpler: if the provenance banner does not
say `tsecon build : RELEASE`, the timing table is not a speed claim. The harness
prints a full-width `DEBUG BUILD DETECTED` banner and repeats the warning under
the timing table when it detects one. (For scale: on an earlier 4-case version
of this suite, the same ops that run 3–21× *faster* than statsmodels in release
ran 2–6× *slower* in debug, with identical parity.)

## Adding a case

Add a `case_*` function that (1) builds deterministic data from the passed
`rng`, (2) appends one `ParityRow` per compared metric with an explicit
tolerance (the `_row(case, metric, got, want, tol)` helper does this), and
(3) sets `.timing` via `best_time(...)`. Register it in the `cases` list in
`main()`. Shared data generators (`_ar1`, `_var2`, `_reg`, `_het_data`) are at
the top of the cases section.

Keep inputs small — the full run is currently **~2.5 s** and should stay well
under 30 s. Cap `repeats` for optimiser-driven cases (`min(repeats, 3)`), as the
three volatility fits do.

Two rules on what deserves a case:

- **Only add one where a trustworthy reference genuinely exists.** Parity is the
  whole point; a case with no independent reference proves nothing.
- **If it only agrees under a fudge, don't add it.** Set the tolerance from the
  known source of numerical difference and see whether it passes. If it needs to
  be widened past that to go green, the two implementations are computing
  different things — drop the case and record why under "What is *not* here".
  A weak green row is worse than an absent one.
