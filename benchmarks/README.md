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
(statsmodels / arch / numpy) can compute, the runner:

1. **Asserts estimate parity first.** It computes the same quantity both ways
   and checks that they agree to a stated tolerance. This is the machine-
   independent, honest core of the benchmark: it does not depend on your CPU,
   your compiler flags, or how the wheel was built. If parity fails, the script
   exits non-zero.
2. **Then times both** (best of *N* runs) and prints a ratio — labelled with the
   detected build mode.

Operations currently covered (all vs a battle-tested reference):

| Operation | tsecon | Reference |
|---|---|---|
| ADF test statistic, p-value, critical values | `tsecon.adf` | `statsmodels.tsa.stattools.adfuller` |
| VAR(p) coefficient matrix + log-likelihood | `tsecon.var_fit` | `statsmodels.tsa.api.VAR` |
| OLS + HAC (Newey–West) standard errors | `tsecon.ols` | `statsmodels` OLS `cov_type='HAC'` |
| GARCH(1,1) QMLE log-likelihood + params | `tsecon.garch_fit` | `arch.arch_model` |

## The honesty rules

These are the rules the harness enforces or the numbers you publish must obey:

1. **Verify identical estimates BEFORE timing.** Timing two functions that
   compute different numbers is meaningless. Parity is asserted first, every
   run, and gates the exit code.
2. **State the tolerance.** Each metric carries an explicit tolerance. Exact
   linear-algebra ops (ADF, VAR, OLS/HAC) match to machine precision
   (`~1e-16` to `1e-13`). QMLE fits (GARCH) match at *optimiser* tolerance
   (log-likelihood `rtol 1e-5`, params `atol 1e-3`) because two optimisers will
   not land on bit-identical parameters — this difference is stated, not hidden.
3. **Publish the runner, hardware, and versions.** The script prints a
   provenance banner (timestamp, OS, CPU, Python, and the version of tsecon,
   numpy, scipy, statsmodels, arch) on every run. Paste it *with* any numbers
   you quote.
4. **Publish losses, not just wins.** The timing table prints the ref/tsecon
   ratio for **every** op, tagged `faster` or `SLOWER`, and reports the win
   count as a fraction (e.g. "faster on 0/4 ops here"). Cases where tsecon is
   *not* faster are shown, never dropped.
5. **Timings are indicative.** `min`-of-N wall-clock, single machine, tiny
   inputs. They indicate order-of-magnitude behaviour, nothing finer.

## ⚠️ The build-mode caveat (read this before quoting any timing)

**`maturin develop` installs a DEBUG build of the Rust core.** Debug builds are
compiled with no optimisation and full symbols — they can be **10–30× slower**
than a release build, and they are the default for local development.

Concretely, in this working tree:

| Artifact | Size |
|---|---|
| `target/debug/libtsecon.dylib` (what `maturin develop` installs) | ~43.4 MB |
| `target/release/libtsecon.dylib` (an optimised wheel) | ~2.2 MB |

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

### Example run (this repo, DEBUG build — timings intentionally not a speed claim)

Captured on an Apple M4 Pro, macOS 26.5.1 (arm64), Python 3.12.7, numpy 2.5.1,
scipy 1.18.0, statsmodels 0.14.6, arch 8.0.0, tsecon 0.0.1 (debug):

```
PARITY MATRIX  (the deliverable -- machine-independent, must all PASS)
  operation                                   metric                       max|diff|       tol  ok
  ADF test (regression='c', fixed lag=4)      statistic                     6.66e-16     1e-06  PASS
                                              p_value                       2.22e-16     1e-06  PASS
                                              crit values                   0.00e+00     1e-06  PASS
  VAR(2) coefficients (2 vars, trend='c')     coef matrix (5x2)             5.55e-16     1e-08  PASS
                                              log-likelihood                1.14e-13     1e-06  PASS
  OLS + HAC (Newey-West) SEs                  params                        4.22e-15     1e-08  PASS
                                              HAC bse                       2.29e-16     1e-08  PASS
  GARCH(1,1) QMLE (constant mean, normal)     log-likelihood (rtol 1e-5)    1.46e-07     5e-02  PASS
                                              params (atol 1e-3)            7.84e-06     1e-03  PASS
  RESULT: ALL PARITY CHECKS PASSED

TIMINGS  (best of 20; DEBUG BUILD, INDICATIVE ONLY -- NOT A SPEED CLAIM)
  operation                                         tsecon   reference     ratio
  ADF test (regression='c', fixed lag=4)             0.259       0.151     0.58x  SLOWER
  VAR(2) coefficients (2 vars, trend='c')            3.428       0.498     0.15x  SLOWER
  OLS + HAC (Newey-West) SEs                         0.430       0.081     0.19x  SLOWER
  GARCH(1,1) QMLE (constant mean, normal)          544.936       8.681     0.02x  SLOWER
  tsecon faster on 0/4 ops here.
```

Note what this example demonstrates: **estimates are identical** (parity to
machine precision, GARCH to optimiser tolerance), yet tsecon is *slower* on
every op — because this is a debug build. That gap between "same answer" and
"slower here" is precisely why we assert parity separately from timing and why
debug timings are not a speed claim. Re-run against a `--release` wheel before
reading anything into the ratios.

### Example run (RELEASE build — these timings *are* quotable)

The same harness against an optimised wheel
(`maturin build --release && pip install --force-reinstall --no-deps target/wheels/tsecon-*.whl`).
Hardware and versions: macOS 26.5.2, arm64, 14 cores, CPython 3.12.7,
numpy 2.5.1, scipy 1.18.0, statsmodels 0.14.6, arch 8.0.0, tsecon 0.0.1.
Parity: **all checks passed** (identical to the debug run — parity is
build-independent, which is the point of separating it from timing).

```
  operation                                         tsecon   reference     ratio
                                                      (ms)        (ms)    ref/ts
----------------------------------------------------------------------------------
  ADF test (regression='c', fixed lag=4)             0.012       0.159    13.21x  faster
  VAR(2) coefficients (2 vars, trend='c')            0.020       0.477    24.04x  faster
  OLS + HAC (Newey-West) SEs (maxlags=4)             0.022       0.087     3.92x  faster
  GARCH(1,1) QMLE (constant mean, normal)           33.806       7.817     0.23x  SLOWER
----------------------------------------------------------------------------------
  tsecon faster on 3/4 ops here.
```

**We publish the loss too.** `GARCH(1,1)` QMLE is ~4× *slower* than `arch` here.
That is a real gap, not a measurement artifact: `arch` has a heavily tuned
likelihood/optimiser path, and tsecon's QMLE has not had the same attention.
The parity row directly above it shows the two agree on the answer (loglik to
`1.5e-07`), so this is purely a performance deficit — and knowing exactly where
we lose is more useful than a headline that hides it.

## Adding a case

Add a `case_*` function that (1) builds deterministic data from the passed
`rng`, (2) appends one `ParityRow` per compared metric with an explicit
tolerance, and (3) sets `.timing` via `best_time(...)`. Register it in the
`cases` list in `main()`. Keep inputs small so a full run stays under a few
seconds, and only add a case where a trustworthy reference exists — parity is
the whole point.
