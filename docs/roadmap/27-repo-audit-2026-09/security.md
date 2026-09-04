# Repository audit — security of the code and the repository

> **Working document.** One of the parallel sweeps of the whole-repository
> audit run on the 0.8.0 tree (`19d308e`). Defensive review of what the
> library does with untrusted input, what the repository exposes, and what
> its history contains. Probe scripts and their outputs are committed under
> `lab/audit/repo/security/`; every number below can be regenerated from
> them. Excluded from the published site.

## Scope and method

Six questions, each answered by a probe that leaves evidence on disk:

1. **Unsafe and panics.** Static inventory of `unsafe` across `crates/` and
   `bindings/`, of `unwrap`/`expect`/`panic!`/casts in
   `bindings/python/src/*.rs`; then the **adversarial-input matrix**
   (`sweep_adversarial.py`): every one of the 173 public callables started
   from its canonical valid call (`registry_ml.py` — the round-11 registry
   plus the eleven machine-learning-wave entries, 173/173 reached) and had
   one argument corrupted at a time — integers set to 0, 1, 2, −1, 2³¹,
   2⁶³, 2⁶⁴ (positional, keyword, and every integer signature default the
   canonical call leaves untouched); floats to NaN, ∞, −1, 0 and the string
   `"abc"`; float arrays to all-NaN, one NaN, one ∞, empty, a single row,
   zero columns, and `"abc"`; ragged panels emptied and NaN-filled; integer
   lists to `[0]`, `[-1]`, `[2**31]`, `[]`; and the whole call rebuilt at
   T = 10⁵. Every cell ran in a child process under a 6 GB virtual-memory
   cap and a per-cell deadline (15 s for the huge-integer cells, 45 s
   otherwise), so a panic, an abort, a signal, or a hang is attributed to the
   exact cell. 4,769 cells before the fix; 4,596 integer/float/array cells
   re-run after it.
2. **Allocation and time bombs.** The 2³¹/2⁶³ and T = 10⁵ cells above, with
   the abort's requested byte count read from the child's stderr, plus three
   uncapped re-runs (no memory limit) to see what happens on a real machine.
3. **Python-side surfaces.** Grep of the shipped package for `pickle`,
   `eval`, `exec`, `subprocess`, `os.system`, sockets, file and environment
   access; then `probe_callbacks.py` — 50 probes of the three callback
   bridges (`gmm_nonlinear(moments_fn=)`, `backtest(forecaster=)`,
   `conformal_forecast`/`conformal_backtest(base=)`): raising inside the
   callback (an `Exception` subclass, `KeyboardInterrupt`, `SystemExit`),
   wrong-shaped / NaN / ∞ / `None` / string / scalar / 2-D / ragged returns,
   shape drift between calls, mutating the handed-over array, re-entrancy
   (tsecon inside the callback, the same function nested, a recursive
   backtest), a second thread calling tsecon from inside the callback, and a
   GIL measurement (how much a pure-Python counter thread progresses during a
   4-second compiled call).
4. **History and secrets.** `scan_history.sh`: 18 credential patterns over
   the full patch text of every commit on every ref, a sensitive-path scan of
   every path ever committed, and the same patterns over every unreachable
   (dangling) blob in the object store. Counts only; no match is ever
   printed.
5. **Supply-chain touchpoints.** `build.rs` files, the maturin/pyproject
   build configuration, the workflows (secrets, `pull_request_target`,
   permissions, action pinning), the fixture generators' data sources, and
   `import tsecon` under three independent network monitors
   (`probe_import_network.py`: an in-process monkeypatch of every socket
   entry point plus a recording `os.environ`; `strace -f -e trace=network`;
   `strace -e trace=file` for out-of-tree file access).
6. **Wheel contents.** `maturin build --release` on this tree, then
   `check_wheel.py` against an explicit allow-list.

Refute-first discipline: every non-refusal cell was re-checked at a
*moderate* too-large value (10⁶, 2⁴⁰) before being called a class, and
every hang was classified as either an unbounded loop or a legitimately
expensive call.

## Totals

**14 findings: 3 severe, 5 moderate, 6 low.** Two of the three severe ones
(the whole `PanicException` class — 113 cells over 67 callables) are fixed
in-branch at one point with 16 regression pins; the third (allocation
failure aborts the process) is a policy question and is proposed, not
applied. The clean bills are the larger story: zero `unsafe`, zero
`unwrap`/`expect` in the bindings, zero dynamic execution or I/O in the
Python package, zero sockets or environment reads on import or in use
(three monitors), zero credential-pattern hits across 248 commits, 1,240
paths and 50 dangling blobs, a wheel that ships exactly the package, and
callback bridges that surface every Python exception as a Python exception.

| severity | count | fixed in-branch |
|---|---:|---|
| severe | 3 | 2 (S1, S2); S3 proposed |
| moderate | 5 | 0 (each has a concrete proposal) |
| low | 6 | 0 (proposals) |

## Findings

### S1 — severe, FIXED: absurd counts escaped as an uncatchable `PanicException` (113 cells, 67 callables)

**Observed (0.8.0).** Any integer count of 2⁶³ reached the compiled core and
panicked instead of erroring, and pyo3 surfaced the panic as
`pyo3_runtime.PanicException` — a `BaseException` that `except Exception`
does not catch. Three mechanisms, all the same class (the byte size or the
sufficiency arithmetic overflowing `usize`):

| mechanism | cells | example | message |
|---|---:|---|---|
| `Vec::with_capacity(n)` size check | 66 | `bootstrap_indices(2**63, scheme="iid", seed=0)` | `capacity overflow` (0.13 s, 30 MB) |
| faer matrix allocation, unwrapped | 14 | `bvar_ssvs(y, lags=2**63)` | ``called `Result::unwrap()` on an `Err` value: CapacityOverflow`` |
| sufficiency check wraps, then a dependency assertion or an index | 24 | `var_fit(y, lags=2**62)`; `stl(y, 2**63)` | `Assertion failed at /root/.cargo/registry/.../faer-0.24.4/src/mat/matref.rs:819 … row_start = 9223372036854775808`; `index out of bounds: the len is 200 but the index is 9223372036854775808` |

**Refuted for the moderate band.** The 24 wrap cells are *not* a missing
check: `stl(y, 10**6)`, `stl(y, 2**40)`, `bk_filter(y, k=2**40)`,
`auto_arima(y, seasonal_period=2**40)`, `var_fit(v, lags=2**40)`,
`connectedness(v, lags=2**40)`, `dfm_nowcast(v, factor_order=10**6)` and
`bvar_fit(v, scale_ar=2**40)` all refuse with the sibling sufficiency
message (`needs at least 2199023255552 observations but got 200`). The
check exists; its arithmetic (`2*period`, `(k+1)*lags`) wraps at ≥ 2⁶².

**Fix.** One seal at the single point every wrapper passes through,
`tsecon._coerce._call` (`bindings/python/python/tsecon/_coerce.py`): a
count at or beyond 2⁴⁸ — 2 PiB of f64, beyond any addressable memory, so
the panic line rather than a policy line — is refused *before* the call
reaches Rust, as a `ValueError` naming the argument
(`bootstrap_indices: n=9223372036854775808 is at or beyond 2**48 — …`).
Seeds are exempt (`seed`, `*_seed`: a u64 seed is legitimately any 64-bit
value; `bootstrap_indices(20, scheme="iid", seed=2**63)` and
`philox_uniforms(2**64 - 1, 5)` still work). Shallow integer lists are
scanned too (`setar(y, 1, delays=[1, 2**63])`). One deliberate consequence:
`max_iter=2**63` as a spelling of "unlimited" is now refused rather than
converging early.

**After the fix:** 0 `PanicException` across all 4,596 re-run cells (was
113); 320 of the 343 cells at 2⁶³ are refusals and the other 23 are seed
parameters returning normally. Pins: `test_security_audit.py::
test_absurd_count_is_a_value_error_not_a_panic` (7 surfaces),
`…inside_an_integer_list…`, `…catchable_by_except_exception`,
`test_large_seeds_are_still_accepted`, `test_a_merely_large_count_reaches_
the_estimator`, `test_other_base_exceptions_pass_through…`.

### S2 — severe, FIXED: product overflow and refused allocations below the line escaped as panics (10 cells)

**Observed.** Below 2⁴⁸, a count can still overflow when *multiplied* — or
be refused by the allocator — and both escaped as panics:
`bvar_fit(v, lags=2**31)` (and `bvar_hierarchical`, `bvar_irf_draws`,
`fry_pagan_svar`, `robust_svar_bounds`, `sign_restricted_svar`,
`zero_sign_svar` at the same value; `var_forecast(steps=2**31)`) →
``unwrap() on … AllocError { layout: Layout { size: 154618822848, … } }``;
`echo_state_network(reservoir_size=2**31)` → `CapacityOverflow`;
`bvar_fit(v, lags=2**40)` → `AllocError { size: 79164837200064 }` (a 72 TiB
design requested before any sufficiency check — unlike `var_fit`, which
refuses at 2⁴⁰); and the brief's own example, **`kernel_ridge` at
n = 10⁵** → `AllocError { size: 80000000000 }` — an 80 GB kernel matrix,
requested and refused, escaping as a `PanicException`.

**Fix.** The same seal's second half: a `PanicException` whose message is
one of the three allocation-sizing failures (`capacity overflow`,
`CapacityOverflow`, `AllocError`) is rebuilt into a `ValueError` naming the
suspect count arguments (those ≥ 2¹⁶) and, when the allocator reported the
byte count, saying how big the request was (`bvar_fit: the working set
implied by lags=1099511627776 could not be sized or allocated — a single
array of 73,728 GiB was requested. …`). The original panic is chained as
`__cause__`. This is sound because all three fire inside the allocator's
size check or on its refusal, before any memory is written or state
changed; every other `BaseException` (`KeyboardInterrupt`, `SystemExit`,
any other panic) passes through untouched, pinned by
`test_other_base_exceptions_pass_through_the_wrapper_unchanged`. Pins:
`test_product_capacity_overflow_is_a_value_error_not_a_panic`,
`test_refused_allocation_below_the_line_is_a_value_error_with_the_size`,
`test_a_panic_that_is_not_about_allocation_passes_through` — driven with a
synthetic panic of the recorded shapes, not the real calls. The integrator's
first push pinned the real `bvar_fit(lags=2**31)` and `lags=2**40` calls,
and the macOS CI wheel job was killed at exit 137: macOS commits a 144 GiB
request lazily and the process dies while the design is filled, where
Linux refuses it up front. That is S3 in the wild, and the reason the pins
cannot depend on the allocator.

**Residual (proposal).** The rebuild couples the wrapper to three panic
message shapes (see L6). The Rust-side hardening is to give the BVAR/SVAR
family the pre-allocation sufficiency check `var_fit` already has, and to
size the faer matrices through `try_*` constructors that return the crate
error instead of unwrapping.

### S3 — severe, NOT FIXED (policy): a refused allocation aborts the process; an accepted one takes the machine

**Observed.** 66 cells over 51 callables at 2³¹ — every `horizon`,
`n_draws`, `n_boot`, `n_trees`, `forecast_steps`, `n_steps`, `n_grid`,
`n_lambdas`, `n_chains`, `n_seeds`, `max_epochs`, `rff_features`,
`n_gamma`/`n_c`, `hidden=[2**31]`, and the count arguments of
`bootstrap_indices`, `philox_uniforms`, `theta_forecast`, `midas_weights`,
`cv_splits` — allocate for real: 16 GB (one f64 vector), 48 GB, 64 GB,
112 GB (an IRF tensor), 144 GB, up to **432 GB** (`robust_svar_bounds
(horizon=2**31)`). When the allocator refuses, Rust's `handle_alloc_error`
**aborts the process** (`memory allocation of 17179869184 bytes failed`,
SIGABRT — no exception, no cleanup, the interpreter dies). Re-run with no
memory cap on this 15 GB machine under default overcommit:
`bootstrap_indices(2**31, scheme="iid", seed=0)` aborts the interpreter;
`bootstrap_indices(2**30, scheme="iid", seed=0)` **succeeds**, taking
8.2 GB of RSS and 21.6 s without a word. NumPy, by contrast, raises a
catchable `MemoryError` on the same refusal.

The same convention bites *valid but enormous* inputs: `bai_perron` at
T = 10⁵ (a 100k-row daily series — plausible) needs a ~40 GB working set
and aborts the process; `kernel_ridge` at 10⁵ needed 80 GB (now a
`ValueError`, per S2); `zivot_andrews`, `kernel_regression`,
`historical_decomposition`, `mcmc_diagnostics`, `bn_decomposition`,
`copula_select` and the two conformal backtests run past 45 s at 10⁵
(legitimately expensive — O(T²) algorithms — not bombs, but nothing warns).

**Why not fixed here.** The line between "refuse" and "run" for a 16 GB
request is a policy choice the brief reserves; see *Policy proposal*. The
abort-versus-`MemoryError` convention is a crate-wide change
(`try_reserve`/`try_new` at every allocation site).

### M1 — moderate: the GIL is held for the whole of every compiled call

**Observed.** During a 4.2 s `setar_test(y, 1, n_boot=98158, seed=0)` a
pure-Python counter thread ran at **0.3 %** of its idle rate
(`probe_callbacks.py::gil/long-call`). No `allow_threads`/`detach` appears
anywhere in `bindings/python/src/`. The good half of this coin: no Python
object is ever touched without the GIL, and the rayon-parallel estimators
still use every core internally — throughput is not lost, only
concurrency with other Python threads (a Dash/Jupyter server, a thread-pool
of independent fits). **Proposal.** Release the GIL around the pure-Rust
section of the long-running estimators (inputs are already copied into
owned `Vec`s by `vec1`/`as_array().to_vec()`, so the section touches no
Python object) — and *never* in the three callback-taking functions, whose
bridge calls back into Python.

### M2 — moderate: after the first exception inside `moments_fn`, the GMM driver keeps calling it

**Observed.** A moment function that raises on its 6th call was invoked
**402 times** in total (`gmm/raise-after-5-calls`): the bridge stashes the
`PyErr` and returns an empty matrix, the Nelder–Mead driver keeps
evaluating, and each further call raises again and overwrites the stash.
The right exception (the first `Custom("boom after 5")`) does surface — but
only after ~396 more invocations of a callable that may be expensive or
side-effectful (a logged model fit). **Proposal** (`bindings/python/src/
lib.rs`, `gmm_nonlinear`): once `err_slot` is set, the closure returns the
empty matrix without calling Python again; the driver then rejects it on
that very evaluation.

### M3 — moderate: `KeyboardInterrupt`/`SystemExit` inside a forecaster become `RuntimeError`

**Observed.** `backtest(y, forecaster=f)`, `conformal_forecast(base=f)` and
`conformal_backtest(base=f)` wrap *every* exception from `f` into
`RuntimeError("forecaster callable … raised at … origin …")` with the
original as `__cause__` — including `KeyboardInterrupt` and `SystemExit`
(`backtest/keyboardinterrupt`, `backtest/systemexit`,
`conformal_forecast/keyboardinterrupt`). Ctrl-C during a long callable
backtest therefore becomes an `Exception` a surrounding `except Exception`
will swallow, and `sys.exit()` inside a forecaster no longer exits. The GMM
bridge gets this right (`gmm/keyboardinterrupt: KeyboardInterrupt
propagated`). **Proposal** (`call_py_forecaster`): re-raise unchanged when
the cause is not an `Exception` subclass.

### M4 — moderate (policy): two iteration counts have no cap and no early exit

**Observed.** `mstl(y, [4, 12], iterate=2**31)` and `reset_test(y, X,
max_power=2**31)` were still running at the 15 s deadline (they would run
for years); every other `max_iter`-style parameter converged early or
refused. Both remain post-fix (below 2⁴⁸). See *Policy proposal*.

### M5 — moderate: the wheel carries no third-party license notices, contrary to the repository's claim

**Observed.** `THIRD-PARTY-LICENSES.md` states that "the full verbatim
copyright notices for each crate are reproduced in released wheels
(generated with `cargo about` at release time)". `cargo about` is wired
nowhere (`.github/`, `scripts/`, `pyproject.toml`, `Cargo.toml`), and the
wheel built here contains no notice file of any kind
(`check_wheel.py`: `third-party license notices in wheel: NONE`) — only
the two tsecon license texts and an SBOM. The statically linked crates are
MIT/BSD/Apache, whose notice-retention clauses apply to binary
redistribution. Cross-reference the supply-chain sweep; the fix is a
release step (`cargo about generate` into the wheel via maturin's
`include`), or correcting the claim.

### L1 — low: the wheel's SBOM embeds the absolute local build path

maturin 1.15 emits `tsecon-0.8.0.dist-info/sboms/tsecon-python.cyclonedx.json`
(133 components, 161 KB); each workspace path-dependency is a
`path+file:///home/user/tsecon/.claude/worktrees/…` purl — **244**
occurrences of the build machine's absolute path (`out/sbom_paths.txt`).
On CI it would be `/home/runner/…`; from a maintainer's laptop it would be
their home directory. Not a secret, but an environment leak in every
artifact. Whether the published 0.8.0 wheel already carries this is *open*
(PyPI unreachable from here). Proposal: build releases only on CI, or
strip the purls' `path+file://` component.

### L2 — low: third-party actions pinned to mutable tags

Every `uses:` in the three workflows is a floating tag (`actions/checkout@v4`,
`PyO3/maturin-action@v1`, `pypa/gh-action-pypi-publish@release/v1`,
`Swatinem/rust-cache@v2`, …); none is a commit SHA, and there is no
Dependabot configuration for actions. The publish job's only credential is
the OIDC token, which limits the blast radius, but a compromised tag in
`maturin-action` would build the wheel that gets published. Proposal: pin
to SHAs with a Dependabot `github-actions` ecosystem entry.

### L3 — low: `#![forbid(unsafe_code)]` is declared nowhere

Zero `unsafe` blocks exist in 43 crates and the bindings today; nothing
locks that in. One attribute per crate root (or a workspace lint) turns the
clean bill into a compile-time guarantee.

### L4 — low: panic messages leak the build machine's cargo registry path

The pre-fix assertion panics carried
`/root/.cargo/registry/src/index.crates.io-…/faer-0.24.4/src/mat/matref.rs:819`
into Python. Post-fix that class is pre-empted; the `AllocError` class still
chains its message as `__cause__` (a `Layout`, no path). `--remap-path-prefix`
in the release profile would close the rest.

### L5 — low: the one network-touching notebook downloads without integrity checking

`notebooks/04_gertler_karadi_src.py` fetches a zip over HTTPS from a
third-party academic server and caches it at a **fixed** path in the shared
temp dir (`tempfile.gettempdir()/ramey_hom_monetary.zip`) with no checksum;
a pre-placed file at that path would be parsed as the data. Members are read
in memory (`zf.read`, no `extractall` — no zip-slip). Notebook only; the
library and the 77 fixture generators make no network calls.

### L6 — low: the seal's reactive half matches panic text

`_coerce._is_alloc_panic` recognises three message shapes. A future Rust
or faer release that rewords one would let that panic escape again as a
`PanicException` (never silently, and the pre-flight half is text-free).
The Rust-side hardening in S2 retires the coupling.

## Adversarial-input matrix summary

Pre-fix (4,769 cells, 173 callables; child process, 6 GB cap):

| mutation | cells | refusal | ok | PANIC | abort/crash | hang |
|---|---:|---:|---:|---:|---:|---:|
| float array (all-NaN / one NaN / one ∞ / empty / 1-row / 0-col / `"abc"`) | 1513 | 1494 | 19 | 0 | 0 | 0 |
| ragged panel (empty / one empty / NaN) | 18 | 18 | 0 | 0 | 0 | 0 |
| float (NaN / ∞ / −1 / 0 / `"abc"`) | 640 | 548 | 92 | 0 | 0 | 0 |
| int 0 | 343 | 179 | 164 | 0 | 0 | 0 |
| int 1 | 343 | 58 | 285 | 0 | 0 | 0 |
| int 2 | 343 | 53 | 290 | 0 | 0 | 0 |
| int −1 | 343 | 340 | 3 | 0 | 0 | 0 |
| int 2³¹ | 343 | 191 | 78 | 9 | 63 | 2 |
| int 2⁶³ | 343 | 160 | 77 | 103 | 1 | 2 |
| int 2⁶⁴ | 343 | 338 | 5 | 0 | 0 | 0 |
| int list (`[0]` / `[-1]` / `[2**31]` / `[]`) | 24 | 21 | 2 | 0 | 1 | 0 |
| whole call at T = 10⁵ | 173 | 0 | 160 | 1 | 2 | 10 |
| **total** | **4769** | **3400** | **1175** | **113** | **67** | **14** |

Post-fix (4,596 cells; the T = 10⁵ row not re-run): **0 PANIC**, 63 aborts
+ 1 crash (all the 2³¹ allocation band, S3), 2 hangs (M4). The 2⁶³ row is
now 320 refusals + 23 seeds returning normally.

Headline: **every array-shaped attack (NaN, ∞, empty, 1×1, zero columns, a
string where an array goes) and every float attack was a refusal or a
normal return — 0 panics in 2,171 cells**; 19 array cells return normally
on a degenerate array (e.g. `ar_loglik` and `local_level_smooth` on an
all-NaN series return a NaN likelihood — a numerical-contract question for
the claims sweep, not a security one). Negative integers are refused in 340
of 343 cells by the existing teaching upgrade. The whole class of
non-catchable failures lived in the huge-integer columns and, post-fix, in
the allocation band only.

The 14 pre-fix hangs: 10 are T = 10⁵ rebuilds of O(T²) estimators
(`arima_fit`, `auto_arima`, `bn_decomposition`, `conformal_forecast`,
`conformal_backtest`, `copula_select`, `historical_decomposition`,
`kernel_regression`, `mcmc_diagnostics`, `zivot_andrews`) — expensive, not
unbounded; 4 are the two uncapped loops of M4 at 2³¹ and 2⁶³.

## History-scan result

**Clean.** `scan_history.sh` over all refs (248 commits, 48.7 MB of patch
text): 0 hits for every one of 18 patterns (AWS, GitHub classic and
fine-grained PATs, PyPI tokens, private-key blocks, Slack tokens and
webhooks, Anthropic/OpenAI/Google keys, URLs with embedded credentials,
`password=`/`token=`/`bearer` assignments, CI secret names). 0 sensitive
paths among the 1,240 paths ever committed (no `.env`, `.pypirc`, `.netrc`,
key files, `settings.local.json`). 50 unreachable blobs, 0 hits. 0 stashes.
Two scratch logs (`maturin.log`, `wstest.log`) were added and removed in the
same day (commit `42ee30e`, 2026-08-17) and contain no paths or hosts. The
only long base64 strings in the history are Cargo.lock checksums. The
ROADMAP's "no secrets in history" holds.

## Wheel-contents check

`tsecon-0.8.0-cp39-abi3-manylinux_2_39_x86_64.whl`, 23 members, 18.2 MB
uncompressed: the package (`__init__.py`, `__init__.pyi`, `py.typed`,
`_coerce.py`, `_inspect.py`, `results/` ×11), `_core.abi3.so` (17.7 MB),
`METADATA` (`License: MIT OR Apache-2.0`; `License-File: LICENSE-MIT,
LICENSE-APACHE`; `Requires-Dist: numpy>=1.22` plus the two extras;
`Requires-Python: >=3.9`), `WHEEL`, `RECORD`, both license texts under
`licenses/`, and the SBOM (L1). **Nothing unexpected**: no tests, fixtures,
notebooks, `.pyc`, paper, lab or scratch material (`exclude =
["**/__pycache__/**", "**/*.pyc"]` in `pyproject.toml` does its job). No
third-party notices (M5). No `build.rs` anywhere in the workspace;
`build-system.requires = ["maturin>=1.14,<2.0"]` is a bounded range, not a
pin.

## Policy proposal on size caps

Today: no count parameter has a ceiling; a request is either refused by a
data-sufficiency rule (when one exists and the count relates to T), sized
and allocated (silently, however large — 8.2 GB for `bootstrap_indices
(2**30)`), or refused by the allocator and the process aborted. Two
uncapped loops (M4) have no ceiling in time either. The seal added here
draws only the impossibility line (2⁴⁸ elements).

Proposed policy — **a memory budget, not per-parameter caps**:

1. **One estimate, one refusal.** Each estimator that allocates a working
   set proportional to a count argument computes its byte size up front
   (`horizon × k² × n_draws × 8`, `T² × 8`, …) and refuses when it exceeds a
   budget, with a teaching message that states the size, the budget, and
   which argument to shrink. Per-parameter caps (`n_boot ≤ 10⁶`) would be
   arbitrary and would still miss the products.
2. **Budget default and override.** Default: half of physical RAM
   (`sysconf(_SC_PHYS_PAGES)`), overridable per call (`memory_budget=`) or
   per process (`TSECON_MEMORY_BUDGET`, bytes; `0` = unlimited). Tests pin
   the estimate, not the machine.
3. **Abort → exception.** Independently of the budget, allocation sites in
   the crates move to `try_reserve`/`try_new` and return the crate error,
   so a refused allocation surfaces as a `MemoryError`-class `ValueError`
   instead of SIGABRT — the NumPy convention.
4. **Time.** `mstl.iterate` and `reset_test.max_power` gain the same
   convergence-or-cap semantics as every other iteration count (a cap with
   a teaching refusal above it), and the O(T²) estimators name their
   complexity in the docstring's size guidance (round 11's sweep G already
   measured the slopes).
5. **Soft, not silent.** A call under budget but above, say, 25 % of RAM
   emits a `ResourceWarning` naming the size — the "valid but enormous"
   case gets a word without a refusal.

## Clean bills

- **`unsafe`: 0 blocks** in 43 crates and the bindings (grep of every
  `.rs`); the only `std::fs` uses are fixture readers inside `#[cfg(test)]`
  modules; no `std::env`, `std::process`, `std::net`; rayon is the only
  concurrency, over owned `Vec<f64>`.
- **Bindings panic surface:** 0 `unwrap`, 0 `expect`, 0 `panic!`, 0
  `unreachable!` in `bindings/python/src/*.rs`; every `as usize` cast (10)
  follows an explicit non-negativity check; the 289 `usize` parameters get
  PyO3's `OverflowError` for negatives, already upgraded to a teaching
  `ValueError` (340/343 cells).
- **Python package:** no `pickle`/`marshal`, `eval`/`exec`/`compile`,
  `subprocess`/`os.system`, sockets, `urllib`/`requests`, `tempfile`,
  `os.environ`, or file writes anywhere in `bindings/python/python/tsecon/`
  (only docstring mentions of `pickle` as a consumer-side compatibility
  promise).
- **Network and environment:** `import tsecon` plus five representative
  calls (a rayon-parallel bootstrap and a forest included) invoke 0 socket
  entry points (in-process monitor), make 0 network syscalls (`strace -f`),
  read 0 environment variables of tsecon's own (NumPy reads three of its
  own), and open only the package's own files (`strace -e trace=file`).
  Pinned by `test_import_and_use_open_no_socket_and_read_no_environment`.
- **Fixture generators:** 77 scripts, 0 network calls; only bundled
  statsmodels datasets and vendored public data, as the README says.
- **Callback bridges:** an `Exception` raised in the callback surfaces as
  that exception (GMM) or as a `RuntimeError` chaining it as `__cause__`
  with origin and window context (forecaster bridge); every bad return —
  scalar, `None`, string, 1-D, 2-D, ragged, empty, zero-column, wrong
  length, NaN, ∞ — is a `TypeError`/`ValueError` naming the callable; the
  training array handed to a forecaster is a read-only private copy
  (mutating it raises inside the callback); calling tsecon inside a
  callback, nesting the same function, a recursive backtest, and a second
  thread calling tsecon from inside the callback all work; a NaN/∞ moment
  matrix is refused by the optimizer's initial-simplex check. 0 panics, 0
  crashes in 50 probes.
- **Workflows:** `permissions: contents: read` at the top of all three; no
  `${{ secrets.* }}` anywhere; no `pull_request_target`; PyPI publishing
  through OIDC trusted publishing only on tags, in a named environment;
  Pages deployment scoped to `pages: write, id-token: write` on the deploy
  job only; the wheel is tested from `site-packages`, never from the source
  tree.
- **Build:** no `build.rs`; nothing downloads at build or import time;
  toolchain pinned (`1.97.1`) for local and CI alike.
- **History:** 0 credential hits, 0 sensitive paths, 0 hits in dangling
  objects.
- **Clippy and fmt:** `cargo clippy -p tsecon-python --all-targets -- -D
  warnings` and `cargo fmt --all --check` pass on this branch (no Rust was
  changed); the full Python suite passes with the seal in place.

## Open

- **S3 and M4** await the policy decision above; the abort-to-exception
  change (item 3) is worth doing regardless of where the budget lands.
- **The published 0.8.0 wheel** was not inspected (PyPI is unreachable from
  this environment): whether it carries the SBOM with CI paths (L1) and
  whether it lacks the notices (M5) should be confirmed with `pip download
  tsecon==0.8.0 --no-deps` and `unzip -l`.
- **T = 10⁵ rebuilds were not re-run post-fix** (the seal does not touch
  them); the T = 10⁶ rebuild of the first pass was dropped after 30
  callables because it only added wall time (results for those 30 are in
  `out/sweep_adversarial_pass1.txt`).
- **Not attempted:** fuzzing the string-typed parameters beyond one bogus
  value (`"abc"` where a float goes was a `TypeError` in all 96 cells; the
  `scheme=`/`kind=`/`method=` string parameters were covered by round 11's
  drift sweep); property-based fuzzing of the array contents beyond
  NaN/∞/empty; the Rust crates' own test-only `std::fs` paths; a
  `cargo audit`/`cargo deny` pass (the supply-chain sweep's remit); the
  `notebooks/` beyond the one download.
- **Two Python-layer proposals (M2, M3)** are small `lib.rs` changes not
  applied here to avoid conflicting edits with the parallel sweeps; each is
  specified above to the line.
