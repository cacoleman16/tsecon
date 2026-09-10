# Adversarial audit, round 13 — findings

> **Working document.** Continuation of
> [round 11](26-audit-round-11-findings.md) and of the repository audit's
> [security sweep](27-repo-audit-2026-09/security.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 13 is the post-wave sweep over the **six callables added in 0.9.0**
(`setar_threshold_ci`, `var_girf`, `threshold_var_girf`, `jsz_fit`,
`jsz_loadings`, `panel_distributed_lag` — 179 public callables at
`9c9927a`) and the one wrapper change of the release, the count pre-flight
in `_coerce.py` that now caches compiled signatures. Every sweep drove one
registry of seeded, tiny canonical inputs
(`lab/audit/round13/registry.py` — the security sweep's 173 entries plus
the six, 179/179 reached), so a finding in one sweep could be re-checked in
another on the identical call. The probe scripts and their summaries are
committed under `lab/audit/round13/`.

**Design.** Six finder/refuter sweeps, each candidate attacked (re-run, a
second seed or size, the promise re-read on the surface that binds —
runtime `__doc__` first, then the stub and the card) before it could be
CONFIRMED:

- **E — result-object contract**: `summarize()` renders; `json.dumps`/
  `pickle` round-trip bit-for-bit; every float finite or its NaN/inf
  documented; returned keys vs the keys `__doc__`, the stub and the card
  name, both directions; array shapes vs the documented shapes (an explicit
  table of 69 expectations); the `tsecon.results` facade (none of the six
  has a typed results class — the generic path applies).
- **F — signature / stub / docstring drift**: `inspect.signature` vs the
  stub for all 179; defaults the prose states on four surfaces vs the
  runtime default; kwargs used in call snippets across docstrings, the six
  card sections, guide chapters 13-16 and `api.md`; every listed string
  value passed; inert-keyword refusal; documented no-ops measured; and,
  for the wrapper change, a count of 2^48 passed *positionally* in every
  positional slot of every callable, cold and warm cache.
- **G — complexity cliffs and resource caps**: three sizes in fresh
  subprocesses with the log-log slope and a re-time of flagged cells, a
  defaults-only pass, then every count argument at 2^47 (just under the
  seal), 2^31, 10^6 and at the documented cap ± 1, each in a child under a
  4 GB `RLIMIT_AS` cap with a deadline.
- **H — seed contract**: same seed twice in-process and across a process
  restart, a different seed, `seed=None`, determinism of all six,
  `RAYON_NUM_THREADS` 1 vs 4 for the two GIRF callables (and `jsz_fit`,
  for the record).
- **S — malformed-input seal**: every parameter of the six corrupted one
  at a time — NaN / inf / empty / wrong rank / one row or column short
  (so paired arguments mismatch) / nested list / int and bool arrays /
  transposed / duplicated row / string / None / scalar for arrays; 0, 1,
  2, −1, 2^31, 2^47, 2^63, 1.5, True, "3", None, [1] for counts; nan,
  ±inf, −1, 0, 1e300, 1e-300, an int, True, "abc", None, [0.5] for floats;
  and the bool, string, integer-list, tuple and Optional slots — 673 cells,
  each in a memory-capped child.
- **C — claims vs reality**: every number in the six card sections, the
  matrix rows, the guide paragraphs (13-16) and `speed.md` re-run through
  the committed test or script it cites and diffed.

**Totals: 211 candidates raised across the six sweeps (plus 8 harness
bugs — two module-name shadowings, the keyword-only stub parser, the
value-window and inert-case slips in F, the positional collision in G's
caps, three f-string backslashes in C — fixed and re-run, not counted), 181 refuted, 19 recorded as known-open classes with
no promise violated, and 11 confirmed items → 6 findings (0 severe, 3
moderate, 3 low) — all 6 fixed in-branch with 22 regression pins.** The
clean bills are the headline: no panic, abort, hang or non-standard
exception over 673 memory-capped malformed-input cells; every count at
2^47 refused before allocation; no seed that failed to reproduce
in-process or across a restart; bit-identity at 1 vs 4 rayon threads;
179/179 signatures matching the stub; 69/69 documented shapes; 2144/2144
positional slots named by the count seal through the new signature
cache; and every Monte-Carlo number quoted on the cards and in the
matrix reproduced by the committed Rust property tests to the printed
digit.

| sweep | raised | refuted | known-open | confirmed items → findings | fixed |
|---|---|---|---|---|---|
| E — result contract | 13 | 12 | 0 | 1 → 1 (L3) | 1 |
| F — signature/doc drift (+ stub parse, + positional seal) | 21 | 10 | 6 (`bands` renders as `...`) | 5 → 2 (M1, M3) | 2 |
| G — cliffs and caps | 12 | 1 | 11 cells → 1 class (S3, GIRF engine) | 0 | — |
| H — seed contract | 3 | 1 | 0 | 2 → 2 (M3 shared with F, L2) | 2 |
| S — malformed input | 160 (154 normal returns + 1 `inf` + 5 refusal classes) | 155 | 1 class (Rust-side text) | 3 classes → 2 (M2, L1) | 2 |
| C — claims vs reality | 2 | 1 | 1 (chapter 16's 0.8.0 banner) | 0 | — |
| **total** | **211** | **181** | **19** | **11 → 6** | **6** |

---

## Severe

None. The three sweeps that could have produced one came back clean:
sweep S found no panic, abort or hang in 673 memory-capped cells; sweep H
found no seed that failed to reproduce in-process or across a restart, no
non-determinism, and bit-identity at 1 vs 4 threads where it is promised;
sweep F's positional check found the count seal naming every offender on
all 179 callables through the new signature cache.

## Moderate

**M1 (F). The shipped type stub did not parse.** The `setar_threshold_ci`
and `panel_distributed_lag` entries appended to `__init__.pyi` by the
hansen-ci and dl-panel slices lost their closing `"""` in the merge — the
append-point-hunk class round 11's integrator notes recorded — so
`ast.parse(stub)` raised `SyntaxError: line 4284: invalid character '—'`
and mypy reported `__init__.pyi:4284: error: Invalid character '—'
(U+2014) [syntax] … errors prevented further checking` for any file
importing tsecon: every type-checker user lost the whole stub, on a
package that ships `py.typed`. The two tripwires were blind — `test_stub_
matches_runtime` only regexes `def` names, and `gen_api_reference.py` is
regex-based too, so it pasted the next entry's section comment and
signature into the previous entry's prose (`api.md` lines 5018-5030 and
5102-5110 carried `# ---- Distributed-lag …`, `def panel_distributed_lag(
…)` and `def jsz_fit(…)` as docstring text). Evidence:
`lab/audit/round13/out/stub_parse_prefix.txt`, `stub_mypy_prefix.txt`.
**Fix** (`5eea18b`): the two closers; `api.md` regenerated (24 leaked
lines removed); the generator now refuses a stub that does not parse; a
new tripwire `test_stub_parses_as_python` runs `ast.parse` and checks no
docstring swallowed a neighbour. After: 360 balanced tokens, mypy
"Success: no issues found", the full signatures revealed
(`stub_parse_postfix.txt`, `stub_mypy_postfix.txt`).

**M2 (S). A wrong-typed option was blamed on the array.** PyO3 renders
*every* failed extraction as `'<type>' object is not an instance of
'<want>'`, and the coercion wrapper's rank rebuild keyed on that phrase
alone, so `setar(y, 1, constant=1)` (an int where a bool goes),
`var_girf(data, 1, antithetic=2)`, `var_girf(data, 1, trend=None)` and
`adf(y, 3)` all raised `TypeError: var_girf: an array argument is the
wrong shape or type (got arg0: array(200, 3)). Estimators that model a
system want a 2-D array …` — naming the user's correct array as the
culprit and advising a reshape. 30 of sweep S's 673 cells across the six
(and every bool/str/tuple option in the library). **Fix**: `_coerce`
parses the message; only the `ndarray` downcast is a rank error, every
other failed extraction names the offending argument through the cached
signature, preferring the argument whose signature default has a
different type (`p=1` and `constant=1` are both ints; only `constant`,
whose default is a bool, is blamed): `setar: constant=1 is of type int,
but this parameter takes a Python bool (True or False — 0/1 are not
accepted as flags). Original error: …`. Pinned by five parametrized cases,
the default-type preference, the chained cause, and the three genuine rank
cases keeping their text (`test_audit_round13.py`).

**M3 (F, H). `jsz_fit(seed=None)` — the signature default — silently meant
seed 0 while four surfaces stated the default as `0`.** `help()` said
"`seed` (0; seeds the perturbed starts …)", the stub "`seed` (0)", the
term-structure card's argument table `0`, and the guide-15 bullet
`seed=0`; `inspect.signature` says `seed=None`, and `None` reaches
`seed.unwrap_or(0)` with the echoed `seed` key reporting `0`. Sweep H
measured `seed=None` ≡ `seed=0` ≠ `seed=1` on the live configuration
(`n_starts=2`). The round-11 M4 class: two "unseeded" runs agreeing
exactly reads as robustness when it is one seed. The `None` sentinel is
load-bearing (it is what lets `seed` with `n_starts=1` raise as inert),
so the fix is on the surfaces: the runtime docstring, the stub, the card
row (`None` (→ 0)) and the guide bullet now say `None` means seed 0, not
fresh entropy, and that the returned `seed` key is the value used;
pinned by `test_jsz_fit_seed_none_is_seed_zero_and_every_surface_says_so`.

## Low

**L1 (S). PyO3's integer and float extraction failures named no
argument.** `adf(y, maxlag=1.5)` → `'float' object cannot be interpreted
as an integer`; `setar(y, 1, trim="abc")` → `must be real number, not
str`; `var_girf(…, horizon=None)` → `'NoneType' object cannot be
interpreted as an integer` — 106 of sweep S's refusals, on every count and
float parameter in the library (the round-6 upgrade covered only the
negative-integer `OverflowError`). Fixed with M2, the same rebuild:
`adf: maxlag=1.5 is of type float, but this parameter takes an integer —
a count such as a lag length, order, horizon, window, or draw count.
Original error: …`; shallow integer lists are scanned too
(`setar: delays=[1.5] …`). Pinned by five parametrized cases.

**L2 (H). Both GIRF callables clamp `histories` above the available
windows to all of them, and no surface said so.** `var_girf(data, 2,
histories=10**6)` returns `n_histories=198` (every window) without a
word; same for `threshold_var_girf` (199) — the value is visible in
`n_histories`, so this is a silent *clamp*, not a silent wrong answer.
Fixed on the runtime docstrings, the stub and both cards ("a count at or
above the number of available windows uses all of them, reported in
`n_histories`"); pinned by
`test_histories_above_the_available_windows_uses_all_and_is_documented`.

**L3 (E). `var_girf`'s docstring promised `mc_se`/`draw_sd` "exactly
zero (NaN with a single draw)"; at the default it is NaN, and otherwise
zero to rounding.** The default `n_draws=2` with `antithetic=True` is
*one* effective draw, so `mc_se` is all-NaN on every default call (sweep
E's one non-finite result); with `n_draws=8`, `mc_se` is 5.9e-18 and
`draw_sd` 2.4e-16 — floating-point summation, not exactly zero. Reworded
on the three surfaces ("zero to rounding (below 1e-15); `mc_se` is NaN
below two effective draws — at the default … there is exactly one");
pinned by `test_var_girf_mc_se_is_nan_at_the_default_and_the_docstring_
says_so`.

## Sweep E — the rest of the ledger

- **(i)/(ii)** `summarize(res).summary()` rendered for 6/6 (54, 85, 121,
  85, 26, 65 lines); `json.dumps` (ndarray default) and `pickle`
  round-tripped 6/6 with every value bit-identical.
- **(iii)** One non-finite result on the canonical calls: `var_girf.mc_se`
  all-NaN at the default draw count (L3, documented as "single draw";
  now stated exactly). `threshold_var_girf(size=1e300)` returns `inf` in
  `mc_se`/`draw_sd` (the variance of a 1e300-scale difference overflows)
  — an absurd input with an honest output; recorded, not counted.
- **(iv)** Every returned key of all six is backticked in `__doc__` and in
  the stub (0 unnamed on the binding surface). The card-side candidates
  refuted: `setar_threshold_ci`'s `k`/`nobs` (echo/count keys the card's
  prose omits; `slope_region_low/high` is one spelling the tokenizer
  split), `threshold_var_girf`'s `history_regimes`, `history_times`,
  `n_effective_draws`, `n_histories`, `shock_vector`, `threshold` (the
  card's "How to read the output" is prose, and the runtime docstring
  names all 26), `panel_distributed_lag`'s `entity_effects`/
  `entity_trends` (the card says "the three effect flags"). The phantom
  candidates (`bands`, `t`, `None`, `B_2`, `acm_term_premium`,
  `sigma_u_mle`) are parameters, math tokens or cross-references, not
  keys — refuted.
- **(v)** 69 array shapes diffed against the docstrings' `[h][variable]`,
  `[history][h][variable]`, `[regressor][power-1][lag]`, `T x M`, `M x N`,
  `2 x k` and `[low, high]` layouts: 69/69 as stated.

## Sweep F — the rest of the ledger

- **(a)** `inspect.signature` vs the stub: 179/179 agree on names, order
  and has-default (keyword-only parameters included). `var_girf` and
  `threshold_var_girf` render `bands=...` (Ellipsis) — round 11's OPEN-1
  class (PyO3 cannot express a tuple default in `__text_signature__`);
  the docstrings and cards state `(0.16, 0.84)`. Known, not counted.
- **(b)** 14 prose-default hits raised: 6 are `setar_threshold_ci.slope_
  region_level` "default 0.80", a documented `None` sentinel ("None: 0.80
  when slope intervals are requested") read across the sentence by the
  parser — refuted; 4 are the `bands` Ellipsis; 4 are M3.
- **(c)** 20 call snippets scanned across the docstrings, the six card
  sections, guide chapters 13-16 and `api.md`: every keyword exists in
  the signature. (`api.md`'s signature blocks match `name(` too and their
  type annotations read as `int=`/`None=` — round 11's tooling noise,
  excluded by skipping `def name(`.)
- **(d)** 8 listed string values probed on the canonical inputs
  (`shock` orthogonal/generalized, `regime` all, `trend` c, `se_type`
  nonrobust/cluster/driscoll_kraay): 8/8 accepted.
- **(e)** Inert-keyword refusal, 5/5 raise: `slope_region_level` without
  `slope_level`; `seed` with `n_starts=1`; `bandwidth` under
  `se_type="cluster"` and `"nonrobust"`; `eval_points` under `powers=1`.
  The three documented no-ops of `var_girf` (`n_draws`, `seed`,
  `antithetic` — "the result does not depend on" them) measured at
  max |Δgirf| = 2.8e-17, 8.3e-17, 6.9e-17: the promise holds to rounding
  (and now says "beyond rounding").
- **(f) The positional-offender check** (`sweep_f_positional.py`): 2144
  positional slots probed (1072 slots × cold and warm cache) with 2^48;
  2086 named correctly as `<name>=281474976710656` by the seal, 58 seed
  slots exempt as designed, **0 offenders**; 177 compiled signatures
  cached after the first pass. The `342b6a2` cache moved nothing.

## Sweep G — complexity cliffs and resource caps

Wall-clock seconds in a fresh subprocess per cell (4 cores, release build,
the machine otherwise idle; the one flagged slope re-timed and agreed
within 0.05). The size axis is T for five callables and the longest
maturity for `jsz_loadings`; "defaults" passes only the required
arguments at the largest size.

| function | T=200 | T=800 | T=3200 | slope | defaults @3200 | note |
|---|---|---|---|---|---|---|
| `setar_threshold_ci` (slope unions, null) | 0.001 | 0.003 | 0.044 | 1.38 (re-time 1.34) | 0.002 | one OLS refit per candidate in the 80% region: T^1.3 by construction, 44 ms |
| `var_girf` (p=2, h=6) | 0.003 | 0.022 | 0.027 | 0.86 | 0.030 | linear in the windows |
| `threshold_var_girf` (h=6, 20 draws) | 0.012 | 0.013 | 0.041 | 0.44 | 1.291 (500 draws, h=20, 3199 windows: 67M simulated periods) | |
| `jsz_fit` (12 maturities, 2 starts) | 0.082 | 0.406 | 1.195 | 0.97 | 2.002 (5 starts) | linear in T; the 1000-start cap runs in 16.8 s |
| `jsz_loadings` (12 maturities to T) | 0.000 | 0.000 | 0.001 | 0.12 | 0.001 | |
| `panel_distributed_lag` (N=6, L=1, quadratic) | 0.001 | 0.002 | 0.007 | 0.77 | 0.004 | |

No cliff: nothing superlinear beyond the documented refit loop, nothing
over 2 s at T=3200 with defaults.

**Caps** (`out/sweep_g.txt`, each cell a child under 4 GB `RLIMIT_AS`
and a 60 s deadline). **Every count at 2^47 — just under the seal — and
at 2^31 is refused before allocation** in ≤ 1 ms with the count in the
message: `p`, `delay`, `delays`, `threshold_index`, `shock_var`
(`requires shock_var < n_series`), `horizon` (`requires horizon <=
1_000_000`), `n_draws` (`n_draws x (horizon + 1) x k … must stay below
2^31 values`), `n_factors`, `n_starts` (`between 1 and 1000`),
`maturities` (`exceeds the supported maximum` 120 000), `lags`
(saturating: `must leave at least two periods`), `powers`; `histories`
at 2^47 and 2^31 clamps to every window (L2) and returns in 5 ms. The
documented caps hold at their edge: `horizon=1_000_001`, `n_starts=1001`
and `maturities=[…, 120_001]` refuse, `n_starts=1000` runs (16.8 s) and
`maturities=[…, 120_000]` runs (55 s for `jsz_fit` at T=200: the
recursion table has one row per period).

**What the GIRF guard admits is the security sweep's S3 class, measured
on the new engine.** The engine's allocation guard is sized at 2^31
*values* per buffer (16 GiB of f64), so a count well inside it is
allocated for real: `var_girf(data, 2, n_draws=2**24)` requests a
2.8 GB per-history buffer (`memory allocation of 2818572288 bytes
failed`, SIGABRT), `n_draws=2**26` an 11 GB one; `horizon=200_000` at
T=200, k=3 needs 0.95 GB in Rust and — because `per_history` is handed
to Python as nested lists — ~3.8 GB of Python floats, and aborts under
the 4 GB cap; `horizon=1_000_000`, the documented cap, aborts the same
way (4.75 GB in Rust, ~19 GB in Python); `n_draws=10**6` and
`threshold_var_girf(horizon=200_000)` run past 60 s (4·10^9 simulated
periods — expensive, not unbounded). No docstring promises a refusal
here, so nothing is violated; the numbers are recorded for the
memory-budget decision the security sweep proposed (OPEN 2).

## Sweep H — the seed ledger

| parameter | live configuration | same seed, in-process | same seed, new process | different seed | `None` |
|---|---|---|---|---|---|
| `var_girf.seed` | linear reduction — documented inert | ✓ | ✓ | identical to 1.2e-16 (documented) | refused |
| `threshold_var_girf.seed` | `n_draws=20` | ✓ | ✓ | ✓ (max Δ 5.5e-3) | refused |
| `jsz_fit.seed` | `n_starts=2` | ✓ | ✓ | ✓ (Δλ^Q 3.9e-8: a different perturbed start, polished separately) | **accepted → 0 (M3)** |

All six callables bit-identical on two consecutive in-process calls;
`var_girf`, `threshold_var_girf` (promised) and `jsz_fit` (not promised)
bit-identical at `RAYON_NUM_THREADS=1` vs `4` in fresh processes. The
`threshold_var_girf` history subsample is keyed by `seed` as the card
says (`history_times` differ between seeds 1 and 2 at `histories=40`);
`histories` above the sample clamps (L2).

## Sweep S — the malformed-input matrix

**0 panics, 0 aborts, 0 hangs, 0 non-standard exceptions over 673 cells**
(6 callables, every parameter, 4 GB `RLIMIT_AS` per child): 519 refusals
(`ValueError`/`TypeError`/`OverflowError`) and 154 normal returns. The
normal returns are all legitimate: valid small counts (`lags=0`, `p=1`,
`horizon=0`), `True` where a count goes (Python's bool is an int — 12
cells), any 64-bit seed (exempt by design), one-row-short or duplicated-row
data for single-array callables (still a valid sample), integer and bool
data arrays and flat numeric lists (coerced by design), boundary floats
inside the documented open ranges (`trim=1e-300`, `level=1e-300`,
`periods_per_year=1e300`), `k_inf_q<0` (any real), `histories` above the
sample (L2), `size=±1e300` (finite paths; `mc_se` overflows to `inf`).

**Refusals that name the parameter.** Before the fix, 280 of the 519
refusals did not name the mutated argument: 74 were the M2 misattribution
(a bool/str/tuple option handed the wrong type and blamed on the array),
92 the L1 unnamed integer/float extraction, 33 a one-element list where a
scalar goes (coerced to an array, then NumPy's "only integer scalar
arrays can be converted to a scalar index"), 49 the genuine rank error
labelling the array by position (`arg0`), and the rest Rust-side text.
After the fix (sweep re-run on the same 673 cells, `out/sweep_s.txt`): 83
— the Rust sufficiency and dimension messages (`insufficient data: 200
observations, at least 6442450948 required` for `p=2^31`; `dimension
mismatch: regressor period dimension …`; `contains a non-finite value`),
PyO3's `Can't extract `str` to `Vec`` (4) and array slots handed a
string, `None` or a scalar ("got no array arguments" — the function is
named, the slot cannot be). 0 panics, 0 aborts, 0 hangs, both runs.

## Sweep C — claims vs reality

Every number was re-run through the test or script that cites it
(`sweep_c_claims.py`; raw outputs in `out/sweep_c_*.txt`, `out/rust_props.txt`):

| claim (where) | cited by | re-run result |
|---|---|---|
| Guide 13 printed block: threshold −0.114, 399/288/111 histories, shock 0.597/0.559, the nine-row table, `h=2` line +0.262/+0.336/+0.179, MC se 0.0003 | `test_guide_chapter_13_example_numbers` | passed, every printed value |
| var-svar card example: `True`, `(9, 3) 398` | `test_var_svar_card_example_prints_what_it_says` + the block itself | passed; the block prints `True` / `(9, 3) 398` |
| Guide 14 printed block: B1 0.3179 (0.1159), B2 −0.0074 (0.0026), turning point 21.60 (2.69), marginal effects ±0.1707/0.0236/−0.1235 | `test_guide_chapter_example_numbers` | passed |
| Panel card + matrix: worst relative error vs linearmodels 7.2e-14 | `test_worst_relative_error_against_linearmodels_is_reported` | 7.24e-14 |
| Panel card + matrix + guide 14: coverage 0.928 / 0.938 (cluster N=50/200), 0.886 / 0.924 (DK T=50/200), means 0.7000/0.6998/0.7020/0.6978, SEs 0.0642/0.0327/0.0611/0.0335 | `panel_dl_properties.rs` (`--nocapture`) | all eight numbers printed exactly; 8 passed in 50 s |
| Cointegration card + matrix + guide 13: the nine-cell `setar_threshold_ci` coverage table (0.886/0.930 … 0.966/0.984), the η² line (468/500, mean 0.617, deciles 0.306/0.629/0.901, 0.915/0.938 conditional, 0.992/0.998 plain) | `setar_ci_properties.rs` | all printed exactly; 9 passed |
| Cointegration card: Hansen Table 1 (4.50/5.94/7.35/10.59) | `test_hansen_2000_table_1_through_lr_crit` | passed |
| TVAR card + matrix: sign 0.149 (t 37), size 0.040 (t 11.5), regime 0.111 (t 60; 340/259), peak 0.690, ratio 4.02, antithetic 1.000 | `tvar_girf_properties.rs` | printed 0.1493 (37.4), 0.0403 (11.5), 0.1111 (60.2; 340 low, 259 high), 0.6905, 4.02, 1.000; 8 passed |
| var-svar matrix row: ratio 4.58, antithetic 0.953 | `girf_properties.rs` | printed 4.58, 0.953; 7 passed |
| TVAR card + guide 13: showcase 0.23 s (Rust release) / 0.63 s (Python, fit included) | `showcase_configuration_timing`, `test_showcase_configuration_timing_is_reasonable` | Python release 0.113 s and 0.116 s on this (quiet) machine — faster than quoted, "indicative"; the Rust figure was re-run in the debug profile only (1.356 s), which the hand-off warns is not the number to quote |
| Term-structure card + matrix + guide 15: GSW λ^Q (0.9965, 0.9624, 0.9092), σ_e 2.69 bp, mean 10y premium 2.30 pp, three basins | `test_jsz_fit_gsw_reproduces_the_fixture_and_the_illustration` | passed |
| Term-structure card + matrix + guide 15: AFNS gaps 9.8e-6 → 2.5e-6 → 6.1e-7, ratios 0.2504/0.2501; recovery 8.7e-5 / 1.2% / 0.3% | `test_jsz_afns_case_…`, `test_jsz_fit_sim_mle_agrees_with_scipy_…` (asserted, not printed) | passed |
| Term-structure card runnable JSZ example, "Expected output" block | the block itself | printed == expected, line for line |
| Cointegration card `setar_threshold_ci` example (no expected block) | the block itself | runs: threshold 0.0, one-candidate 95% set, p(γ₀=0) = 1.0, η² 0.922 |
| Guide 16: the 148-line verbatim block (two PMW tables, the T ladder, the LPW bias/SD/RMSE/coverage tables, λ median 3.53e+03) | `docs/examples/lp_vs_var_head_to_head.py` (107 s) | 147/147 content lines reproduced verbatim; only the `date` line differs |
| Guide 16 provenance: "tsecon 0.8.0 on a release build" | the script's banner | the banner now prints 0.9.0; every number is unchanged (seeded), so the statement records the run that produced them — cosmetic, left to the integrator (version strings) |
| DJO example | `docs/examples/panel_distributed_lag_djo.py` | not run; the header states the network fetch, the temp-dir cache and the mirror, and the card says so |

**`speed.md`** (`bench.py --json` re-run into the scratchpad, 20 repeats,
then `render_dashboard.py --check`): all 65 parity rows PASS and every
`max abs diff`/`tol` pair is bit-identical to the committed
`benchmarks/results/latest.json`; faster on 22 of 25 as the page says;
the per-call wrapper overhead re-measured at 2.6 µs against the
committed 2.85 µs ("~0.003 ms"); the dashboard is up to date with the
JSON. (The harness printed `tsecon build: UNKNOWN` because the re-run
did not set `CARGO_TARGET_DIR`, so its size check looked in a target
directory that does not exist here — an environment note; the committed
JSON records the release build.)

## OPEN — recorded, not fixed

1. **`bands` renders as `...` in `inspect.signature`** for both GIRF
   callables — round 11's OPEN-1 class (PyO3 `__text_signature__` cannot
   express a tuple default); the docstrings, stub and cards state
   `(0.16, 0.84)`. Low, unfixed.
2. **The GIRF engine's allocation guard is sized at 2^31 values per
   buffer** (16 GiB of f64), so a count the guard admits can still be
   refused by the allocator and abort the process — the security sweep's
   S3 class (policy: a memory budget, not per-parameter caps). Measured
   in sweep G; no promise violated.
3. **Wrong-typed refusals that still name no argument** after M2/L1:
   the Rust-side sufficiency messages (`insufficient data: 200
   observations, at least 6442450948 required` for `p=2^31` — the count is
   in the message, the name is not), PyO3's `Can't extract `str` to
   `Vec`` and `expected tuple of length 2, but got tuple of length 3`,
   and NumPy's `only integer scalar arrays can be converted to a scalar
   index` for a one-element list where a count goes. Recorded with counts
   in `out/sweep_s.txt`; a wrapper rebuild for these shapes would need the
   parameter's Rust type, which the Python side does not have.

## Verification record

- `5eea18b` — the stub closers, the generator guard and
  `test_stub_parses_as_python`; the five tripwires: 98 passed.
- `649319c` — the `_coerce` rebuild (M2, L1), the four-surface `seed`
  default (M3), the `histories` clamp (L2) and the `mc_se` wording (L3)
  with `test_audit_round13.py`; the five tripwires plus the four slice
  files (`test_setar_ci`, `test_girf`, `test_jsz`, `test_panel_dl`):
  246 passed. No Rust code changed — doc comments in the binding crate
  only, rebuilt with `maturin develop --release`; `cargo fmt --all
  --check` and `cargo clippy -p tsecon-python --all-targets -- -D
  warnings` clean; `docs/gen_api_reference.py` byte-identical to the
  committed `api.md`; `mkdocs build --strict` clean.
- The five Rust property/golden binaries the cards cite, re-run with
  `--nocapture` for sweep C: 48 passed, 0 failed (`out/rust_props.txt`).
- Full Python suite, once, after every change: **1727 passed, 1 skipped
  in 6:06** (`pytest bindings/python/tests -q -p no:cacheprovider`).

## Lessons

1. **A stub that only a regex reads can be broken for every type-checker
   user while CI is green.** Round 11's integrator notes predicted the
   exact mechanism (a keep-both merge leaving the earlier side
   unterminated) and even said the stub-sync test "only regexes `def`
   names"; nothing was added to parse the file. `ast.parse` is one line
   and now runs in CI and in the generator.
2. **A teaching error is only as good as the pattern that triggers it.**
   The rank rebuild matched a substring PyO3 uses for every failed
   extraction; it taught the right lesson for arrays and the wrong one for
   everything else, and no test passed a wrong-typed *option* through the
   wrapper. Sweep S's bool/str/tuple mutations — absent from the security
   sweep's matrix — are what found it.
3. **`None` is a promise in Python, again.** Round 11's M4 fixed three
   functions; the next wave shipped a fourth with the same `unwrap_or(0)`
   and, this time, four surfaces stating a default the signature does not
   have. The sweep-H harness should keep `None` = entropy as its null.
