# Adversarial audit, round 14 — findings

> **Working document.** Continuation of
> [round 13](28-audit-round-13-findings.md) and of the repository audit's
> [security sweep](27-repo-audit-2026-09/security.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 14 is the post-wave sweep over the **thirteen callables added in
0.10.0** (`unobserved_components`, `tvp_regression`, `ets_fit`, `auto_ets`,
`var_conditional_forecast`, `var_diagnostics`, `var_select_order`,
`spa_test`, `model_confidence_set`, `stepm_test`, `fmols`, `dols`, `ccr` —
192 public callables at `57d0629`), the **`mask=` parameter** the wave added
to `panel_fe`, `panel_distributed_lag`, `panel_lp` and `lp_did`, and the
Blanchard-Quah replication page. Every sweep drove one registry of seeded,
tiny canonical inputs (`lab/audit/round14/registry.py` — round 13's 179
entries plus the thirteen, 192/192 reached, plus four `name@mask` cells that
call the same function with an unbalanced observation mask). The probe
scripts and their summaries are committed under `lab/audit/round14/`.

**Design.** Six finder/refuter sweeps, each candidate attacked (re-run, a
second seed or size, the promise re-read on the surface that binds — runtime
`__doc__` first, then the stub and the card) before it could be CONFIRMED:

- **E — result-object contract**: `summarize()` renders; `json.dumps`/
  `pickle` round-trip bit-for-bit; every float finite or its NaN/inf
  documented; returned keys vs the keys `__doc__`, the stub and the card
  name, both directions; array shapes vs the documented shapes (an explicit
  table of 183 expectations); and a **library-wide container census** — every
  returned key of every registry callable bucketed by (ndarray / nested list /
  Python list, rank, dtype), so the thirteen can be compared with the 179
  that came before.
- **F — signature / stub / docstring drift**: `inspect.signature` vs the
  stub for all 192 and a re-check that no default renders as `Ellipsis`
  anywhere (the hygiene slice took eleven to zero); defaults the prose states
  on four surfaces vs the runtime default; kwargs used in call snippets
  across the docstrings, the card sections, guide chapters 4, 5, 7, 8 and 14,
  the Blanchard-Quah page and `api.md`; every listed string value passed —
  both regex-derived and, new this round, **hand-transcribed** from the
  runtime `__doc__` (101 values, `level`'s twenty-two statsmodels spellings
  included); the inert-keyword contract (35 cases); and the documented
  **mask identity** — a mask of all ones must be bit-identical to no mask.
- **G — complexity cliffs and resource caps**: three sizes in fresh
  subprocesses with the log-log slope and a re-time of every flagged cell, a
  defaults-only pass, then every count that sizes an allocation at 2^47,
  2^31, 10^6 and at the documented cap ± 1 — 137 cells, each in a child
  under a 4 GB `RLIMIT_AS` cap with a 60 s deadline.
- **H — seed contract**: same seed twice in-process and across a process
  restart, a different seed, `seed=None`, determinism of all seventeen
  cells, and `RAYON_NUM_THREADS` 1 vs 4 for every cell — the three
  multiple-comparison tests promise bit-identity at any thread count, the
  rest are measured for the record.
- **S — malformed-input seal**: every parameter of the thirteen and of the
  four masked cells corrupted one at a time — NaN / inf / empty / wrong rank
  / one row or column short (so paired arguments mismatch) / nested list /
  int and bool arrays / transposed / duplicated row / string / None / scalar
  for arrays; 0, 1, 2, −1, 2^31, 2^47, 2^63, 1.5, True, "3", None, [1] for
  counts; nan, ±inf, −1, 0, 1e300, 1e-300, an int, True, "abc", None, [0.5]
  for floats; float lists, integer lists, bool lists, the bool, string and
  Optional slots — **and the `mask=` arrays themselves** (all zero, half
  zero, one entity gone, 2, −1, 0.5, NaN, inf, wrong shape, transposed, a
  string, a scalar) and `var_conditional_forecast`'s nested `conditions`
  (all free, empty, ragged, too wide, NaN-only, a string cell, flat) —
  **1563 cells**, each in a memory-capped child.
- **C — claims vs reality**: every number the wave's thirteen card sections,
  the panel card's mask section, the nine validation-matrix rows, the guide
  paragraphs, the Blanchard-Quah page, the CHANGELOG 0.10.0 entry and ROADMAP
  §0 assert, looked up in a corpus of the committed artifacts (the Rust
  property/golden binaries re-run with `--nocapture`, the wave's pytest files,
  every crate source and test, the Python tests, the examples, the fixtures,
  the generators and the benchmarks); and every runnable card example
  executed and diffed against the output its comments claim.

**Totals: 126 candidates raised across the six sweeps (plus 6 harness bugs —
the `@mask` name resolution, the `also` key round 13 recorded but never
applied, the ragged-`steps` shape expectation, the value-window that let
`optimizer`'s values bleed into `initialization`'s, a wrong `trend="ct"`
entry in the hand-transcribed table and the rounding blind spot in sweep C —
fixed and re-run, not counted), 77 refuted, 2 recorded as known-open with no
promise violated, and 47 confirmed items → 10 findings (0 severe, 2
moderate, 8 low) — all 10 fixed in-branch with 34 regression pins.** The clean bills are
the headline: **no panic, abort, hang or non-standard exception over 1563
memory-capped malformed-input cells**; 103 of the cap sweep's 137 cells
refused before allocating (in ≤ 1 ms, with the count in the message), 33
returned because the count is legitimately inside its cap, and exactly one
neither refused nor finished; **0 of 435 returned keys
unnamed by `__doc__` or the stub**, and 183/183 documented array shapes;
192/192 signatures matching the stub and **0 `Ellipsis` defaults anywhere**;
35/35 inert keywords refusing; the mask identity bitwise on all four panel
callables; no seed that failed to reproduce in-process or across a restart
and **17/17 cells bit-identical at 1 vs 4 rayon threads**; and 1166 of 1173
quoted numbers reproduced by a committed artifact.

| sweep | raised | refuted | known-open | confirmed items → findings | fixed |
|---|---|---|---|---|---|
| E — result contract | 15 | 15 | 0 | 0 | — |
| F — signature/doc drift (+ value table, + mask identity) | 39 | 33 | 0 | 6 → 3 (L4, L5, L6) | 3 |
| G — cliffs and caps | 4 | 2 | 0 | 2 → 2 (M1, M2) | 2 |
| H — seed contract | 1 | 1 | 0 | 0 | — |
| S — malformed input | 40 | 1 | 1 cell | 38 → 4 (L1, L2, L3, L7) | 4 |
| C — claims vs reality | 27 | 25 | 1 (a 0.8.0 sentence) | 1 → 1 (L8) | 1 |
| **total** | **126** | **77** | **2** | **47 → 10** | **10** |

How the columns count. A *candidate* is one line the finder printed: a
prose-default hit, an unnamed refusal, a flagged slope, a number no artifact
carried. Sweep F's 39 are the 38 its counter reached after the harness fixes
plus one raised by hand when the `df_adjust` sentence on the card was read
against the code (L4 — the finder compares a stated default with the runtime
default, and this claim is a formula, not a default). Sweep S's 40 are the
34 refusals that did not name the mutated parameter plus the six
`band_alpha` cells that returned in silence; 38 of them were fixed, one is
refuted as a cascade and one is known-open. Sweep C's 27 are the 26 numbers
the first pass could not find plus the one card-example line that was not
printed; nineteen of the 26 were then matched to a committed number by
rounding.

---

## Severe

None. The three sweeps that could have produced one came back clean: sweep S
found no panic, abort or hang in 1563 memory-capped cells (the wave's five
allocator aborts stayed fixed, and no new one appeared); sweep H found no
seed that failed to reproduce in-process or across a restart, no
non-determinism in seventeen cells, and bit-identity at 1 vs 4 threads
everywhere; and sweep G's cap pass refused every count driven outside its
cap before anything was allocated (103 of its 137 cells, in ≤ 1 ms, with the
count in the message), the other 34 being counts legitimately inside their
caps — one of which is M2.

## Moderate

**M1 (G). An unbalanced two-way panel is CUBIC in T, and no surface said
so.** `panel_distributed_lag` with a mask and the default
`time_effects=True` partials the time effects out through the projected time
dummies — one per observed period — by a rank-revealing least-squares step
(the exact Frisch-Waugh route `PanelOLS` uses, and the one the
`fixtures/panel_unbalanced.json` goldens pin). That is exact, and it is
cubic. Measured at N = 6, `lags=1`, `powers=2`
(`lab/audit/round14/out/sweep_g.txt`, slope 2.68, re-timed 2.64):

| T | no mask | mask of all ones | ragged mask | ragged mask, `time_effects=False` |
|---|---|---|---|---|
| 200 | 0.0008 s | 0.0003 s | 0.023 s | — |
| 400 | 0.0007 | 0.0005 | 0.109 | 0.0005 s |
| 800 | 0.0008 | 0.0007 | 0.719 | 0.0008 |
| 1600 | 0.0014 | 0.0014 | 5.21 | 0.0016 |
| 3200 | 0.0029 | 0.0027 | **38.1** | 0.0033 |

A factor of 8 per doubling, and a factor of **13 000** against the same
panel unmasked at T = 3200. `panel_fe` (entity effects only) and `panel_lp`
are linear under a mask; the cliff is the unbalanced *two-way* projection
alone. Nothing was violated — no surface promised a cost — but a user who
masks a long panel meets a 40-second call with no warning, and the sweep's
slope flag is what found it. **Fix** (docs, no numerical change): the
runtime docstring, the stub, `api.md` and a new **"What it costs"** paragraph
on the panel card state the cubic law with the measured table and the two
escapes (drop the time effects, or trim T); the alternative — alternating
projections — is recorded as an approximation the goldens do not pin, and is
*not* traded for speed here. Pinned by
`test_the_unbalanced_two_way_panel_cost_is_documented_and_bounded` (the
sentence on all three surfaces, plus a 10 s ceiling at T = 400 against a
measured 0.11 s, and a 2 s ceiling on the `time_effects=False` route against
a measured 0.0005 s).

**M2 (G). `var_conditional_forecast` is guarded on memory while its cost is
quadratic in `steps`.** The budget the 0.10.0 hygiene slice added refuses a
`steps` whose working set exceeds 2^24 doubles — which admits `steps` up to
roughly 2^24/(k · m), about 1.9 million on the sweep's k = 3 system. The
work, however, grows with `steps²`:

| steps | 1 000 | 2 000 | 4 000 | 8 000 | 16 000 | 32 000 |
|---|---|---|---|---|---|---|
| seconds | 0.034 | 0.115 | 0.459 | 1.90 | 7.62 | 31.4 |
| peak RSS | 43 MB | 49 | 58 | 80 | 114 | 201 |

×4 per doubling. `steps=200_000` — inside the budget, refused by nothing —
was **the one cell of 137 in the cap sweep that did not finish inside its
60 s deadline**; extrapolated, the largest `steps` the budget admits is about
a day of compute. **Fix** (docs, no behaviour change): the runtime
docstring, the stub, `api.md` and the var-svar card now say the budget is on
memory and not time, quote the measured ladder, and say that forecast
horizons are tens of periods in practice so five figures is a typo. Pinned by
`test_the_conditional_forecast_steps_cost_is_documented_and_bounded` (the
sentence on all three surfaces, a 5 s ceiling at `steps=2000` against a
measured 0.12 s, and the 2^31 refusal).

## Low

**L1 (S). Three `ets_fit` refusals described the concept and never the
keyword.** `smoothing_params` of the wrong length raised *"smoothing
parameters [alpha, beta?, gamma?, phi?] for this spec: expected length 3 but
got 1"*, `initial_states` the same shape, and a `seasonal_periods` larger
than the sample was refused only indirectly, through the parameter count it
inflates: *"y has 200 observations but fitting ETS(A,A,A) with 2147483653
parameters (the AICc needs n − k − 1 > 0) needs at least 2147483655"* — the
number 2147483653 is the only trace of what the user typed. 20 of sweep S's
1563 cells, and the class the wave's own hygiene slice had just closed for
the 0.9.0 surface. **Fix**: the two `what` strings now lead with
`smoothing_params` / `initial_states`, and `check_data` gained a guard on the
*period* — *"y has 200 observations but seasonal_periods = 201 (a seasonal
period the sample never completes is not identified, and it costs m − 1 free
initial states) needs at least 201"* — which also fires at `seasonal_periods
= 250`, where the old message was equally opaque. Pinned by
`test_ets_refusals_name_the_offending_parameter` (six cases) and
`test_a_seasonal_period_longer_than_the_sample_is_refused_by_name`.

**L2 (S). The automatic block-length failure was ungrammatical and blamed a
parameter the caller never passed.** A degenerate or too-short loss panel
raised *"model_confidence_set: `block_size = None` is invalid: requires the
automatic Politis-White block length could not be computed on loss column 0
(…)"* — the requirement slot of the shared template was filled with a
sentence, and the named parameter (`block_size`) is the one the user left at
its default while the offender is `losses`. **Fix**: the requirement now
reads *"a block length the Politis-White rule can compute: it failed on
losses column 0 (…) — pass block_size explicitly (e.g. block_size=5), or
drop the degenerate loss column"*, naming both. Pinned by
`test_automatic_block_length_failure_names_losses_and_reads_as_a_sentence`
for all three entry points.

**L3 (S). The coercion rebuild could not see inside a nested list, and its
fallback read as a tautology.** A string cell in
`var_conditional_forecast`'s `conditions` produced *"var_conditional_forecast:
an argument of type str is of type str, but this parameter takes a real
number"* — round 13's offender scan looked exactly one level into a list
argument (enough for `delays=[1.5]`, not for a table), and when it found
nothing the fallback label repeated the type it was about to print again.
**Fix**: `_nested_path` walks up to three levels and names the **cell by its
position**, and the fallback label is now just *"an argument"*:
`var_conditional_forecast: conditions[0][0]='x' is of type str, but this
parameter takes a real number`. The positional spelling is used only two or
more levels down, so round 13's pinned `delays=[1.5]` wording is byte-for-byte
unchanged. Pinned by
`test_a_bad_cell_inside_a_nested_list_argument_is_named_by_its_position` and
`test_round_13s_shallow_list_message_shape_is_unchanged`.

**L4 (F). `fmols` and `ccr` stated the `df_adjust` factor over the wrong
count.** Four surfaces said the covariance is scaled by `(T-1)/(T-1-k)`,
with `k` defined in the same docstring as *"the (T, k) matrix of I(1)
regressors"*; the card said `T/(T−k)`, wrong in the numerator too. The
implementation scales by `m/(m − nvar)` with `m = T − 1` and
`nvar = k_x + trend.n_det()` — the estimated coefficients, deterministics
included. On the sweep's default call (T = 200, k = 2, `trend="c"`) the
stated factor is 199/197 = 1.01015 and the actual one 199/196 = 1.01531:
the two agree only at `trend="n"`. **Fix**: the runtime docstring, the stub,
`api.md`, the crate module doc and both card sentences now say
`(T−1)/(T−1−p)` with `p` the estimated coefficients (`len(params)`), and the
`dols` formula (`nobs/(nobs − n_params)`, which was already right) is stated
beside it. Pinned by
`test_df_adjust_scales_by_the_estimated_coefficient_count_not_just_k`
(two estimators × three trends, measured against `m/(m − len(params))` at
1e-12 and asserted *different* from `m/(m − n_x)` wherever deterministics
exist) and `test_every_surface_states_the_df_adjust_factor_over_the_coefficient_count`.

**L5 (F). `fmols`/`ccr` called `False` the default of `diff`, and `False` is
a value that raises.** The signature default is the `None` sentinel; the body
said *"`diff` (default False)"* while the tail said *"`diff` (None: False)"*.
Passing the stated default explicitly — `fmols(y, x, diff=False)` at the
default `trend="c"` — **raises**, because `diff` is inert without a trend
term and the library refuses inert keywords. Round 13's M3 class, one
release later. The sentinel is load-bearing (it is what distinguishes "not
asked for" from "asked for where it cannot act"), so the fix is on the
surfaces: *"`diff` (default None, which behaves as False) … and note that
`diff=False` passed EXPLICITLY raises there too, for the same reason: the
default is the `None` sentinel, not `False`."* Pinned by
`test_diff_false_is_refused_where_the_default_runs_and_the_doc_says_so`,
which runs the default, asserts both `True` and `False` raise at
`trend="c"`, asserts `diff=True` runs at `trend="ct"`, and reads the sentence
off `__doc__` and the stub.

**L6 (F). A guide signature bullet advertised a call that raises.** Guide 14's
"Available now in Python" list showed
`tsecon.panel_fe(outcome, regressors, se_type="cluster", bandwidth=4.0)` —
and `bandwidth` under any `se_type` but `"driscoll_kraay"` **raises** by
design, so the advertised call is a refusal. The same list also still carried
the pre-`mask=` signatures for three of the four panel callables the wave had
just extended (the parameter is documented in the chapter's prose, at lines
88, 115 and 156, but not in the signatures). The bullet predates this wave
(`0b54736`); the sweep found it because chapter 14 is one the wave edited.
**Fix**: the bullet now reads `bandwidth=None, mask=None` and says in one
clause that `bandwidth` acts only under Driscoll-Kraay and raises elsewhere;
`panel_distributed_lag` and `panel_lp` gained `mask=None`. Pinned by
`test_the_guide_panel_bullets_are_calls_that_run`, which parses the bullet
out of the markdown, executes it, and asserts the `se_type="cluster"` +
`bandwidth=4.0` combination still raises.

**L7 (S). `band_alpha` accepted NaN in silence whenever no band was
requested.** `panel_lp(y, shock, band_alpha=float("nan"))` — with `band`
left at `None` — returned normally, as did `-1`, `0`, `1` and `1e300`; the
validator lived inside the band builder, which only runs when a band is
asked for. Unlike `bandwidth` or `eval_points`, `band_alpha` has a concrete
default (0.1) rather than a `None` sentinel, so it *cannot* be refused as an
inert keyword without a signature change — but it can be validated, and it
was not. Six LP surfaces shared the hole (`lp`, `lp_iv`, `lp_multiplier`,
`lp_state`, `panel_lp`, `smooth_lp`). **Fix**: a `check_band_alpha` helper
runs unconditionally at the top of all six. **This is a behaviour change a
model card documents**: a call that passed a `band_alpha` outside (0, 1)
with no band used to return and now raises — no valid call is affected, and
the message is the one the band path already used. Pinned by
`test_band_alpha_is_validated_even_without_a_band` (six bad values × two
surfaces, plus the default still running).

**L8 (C). The `unobserved_components` card example did not print what its
"expected output" comment claimed.** The block computes
`np.round(fit["params"], 1)` and zips it into a dict, which under NumPy 2
renders as `{'sigma2.irregular': np.float64(15098.5), 'sigma2.level':
np.float64(1469.2)}` — the comment claims the bare
`{'sigma2.irregular': 15098.5, 'sigma2.level': 1469.2}`. The numbers are
right (they round the fixture's 15098.519 / 1469.176, which
`test_nile_mle_reproduces_durbin_koopman_and_both_optimizers` pins); the
rendering is not, and it is the only one of the eleven runnable blocks of
the new surface that failed the line-for-line diff. **Fix**: the snippet
builds the dict with `round(float(v), 1)`, which prints exactly the claimed
line. Pinned by
`test_the_unobserved_components_card_example_prints_what_it_claims`, which
extracts the block from the card, runs it in a subprocess and asserts the
claimed line is in the output.

## Sweep E — the rest of the ledger

- **(i)/(ii)** `summarize(res).summary()` rendered for 17/17 (832 lines in
  total, 11 to 97 per cell); `json.dumps` (ndarray default) and `pickle`
  round-tripped 17/17 with every value bit-identical.
- **(iii)** Four non-finite paths on the canonical calls, all documented:
  `unobserved_components.se` (3 NaN — the flagged pile-ups, "a NaN standard
  error" is the documented signal) and `.std_resid` (5 NaN — the diffuse
  period), `tvp_regression.std_resid` (3 NaN, same), and `dols.ic_value`
  (NaN — "NaN when both were fixed"). Sweep S adds two absurd-input honest
  infinities, recorded below.
- **(iv)** **0 of 435 returned keys is unnamed in `__doc__`, and 0 in the
  stub** — the whole new surface is enumerated on both binding surfaces. 84
  keys across 15 cells are not backticked in the *card* section, and 48
  backticked tokens in a returns sentence are not keys; every one was
  refuted the way round 13 refuted its own: the cards' "How to read the
  output" is prose that names key *families* ("the component keys without a
  prefix … each with `_var`", "`filtered_*`", "the three effect flags"), and
  the phantom tokens are math (`T_R`, `T_max`, `dbar_k`, `sigma2_HAC`, `b1`,
  `b2`, `B_2`), cross-references (`ets_fit`, `spa_test`, R's `ets`) or the
  keys of a nested dict (`status`, `error` inside `auto_ets.candidates`).
- **(v)** 183 array shapes diffed against the docstrings' stated layouts
  (`nobs x k_states`, `steps x k`, `[h][i][j]`, `[regressor][power-1][lag]`,
  `(1+k) x (1+k)`, the packed smoothing and initial-state vectors, the
  ragged per-step index lists of `stepm_test.steps`, the candidate table of
  `auto_ets`): **183/183 as stated**.
- **(vi) The container census.** 176 callables reached, 1879 returned keys
  bucketed: 470 scalars `float`, 402 `ndarray` 1-D float, 333 `int`, 191
  nested-list 2-D float, 105 `str`, 96 `bool`, 41 nested-list 3-D float, 38
  `dict`, 32 **Python-list 1-D float**, 28 Python-list 1-D int, 27 `ndarray`
  1-D unsigned, 26 list-of-dict, 25 `None`, 20 list of `str`, 9 `ndarray`
  1-D bool, 7 `ndarray` 2-D float, 6 each `ndarray` 1-D int and nested-list
  4-D float, 5 each tuple and Python-list 1-D bool, 3 list-of-list-of-object,
  2 each nested-list 2-D int and 2-D bool. The rule the library follows is
  "2-D and deeper are nested lists, 1-D float payloads are `ndarray`", and
  it is followed 402 times against 32 exceptions — **24 of which predate
  this wave** (`max_share_svar.impact`, `markov_switching_ar.means`,
  `hetero_svar.variance_ratios`, `var_girf.shock_vector`, …). The wave adds
  eight to the minority bucket (`var_diagnostics.roots`,
  `.eigenvalue_moduli`, `.skewness_components`, `.kurtosis_components`, and
  `var_select_order`'s four `*_values`) and three to the Python-list-of-bool
  bucket (`unobserved_components.at_boundary`, `tvp_regression.at_boundary`,
  `.pile_up`) while putting `spa_test.recentered` and `stepm_test.recentered`
  in the `ndarray`-of-bool one — both spellings have precedent
  (`arima_fit.se_valid` is an array, `growth_at_risk.converged` a list), so
  no promise is broken and nothing was changed. Recorded as the standing
  class it is: **the library has no single boolean/1-D-float container rule,
  and 0.10.0 did not make it worse in a new way.**

## Sweep F — the rest of the ledger

- **(a)** `inspect.signature` vs the stub: **192/192** agree on names, order
  and has-default. **0 `Ellipsis` defaults anywhere in the surface** — the
  hygiene slice's 11 → 0 held, and the thirteen new callables added none
  (round 11's OPEN-1 and round 13's `bands` case are closed). No stub
  annotation disagrees with a `None` runtime default.
- **(b)** 27 prose-default hits raised, 24 refuted as parser windows or
  documented sentinels, 3 confirmed (L4, L5, L6):
  - 14 are `unobserved_components`'s four component sentinels
    (`stochastic_seasonal` "default True", `stochastic_freq_seasonal`,
    `damped_cycle`, `stochastic_cycle` "default False") counted twice each
    across `__doc__` and the stub. The signature default is `None` in all
    four — and the sweep **measured** `None` against the stated value on a
    live configuration: identical log-likelihood to the last bit in all four
    (28.289322627837258 for `stochastic_seasonal`, 32.46133247785412 for the
    two cycle flags, 27.59449251668548 for `stochastic_freq_seasonal`). The
    sentinel exists so the option can be refused when its component is
    absent; the stated default is the *effective* one and is correct.
    Refuted, and the contrast with L5 is the point: there the stated default
    is a value that raises.
  - 6 are `ets_fit`'s `level` ("0.95 when omitted"), `n_sim` ("5000 when
    omitted") and the returned key `optimizer` ("\"none\" at fixed
    parameters") read across a sentence — all three correct as written, and
    the tails state the `None` sentinels. Refuted.
  - 2 are `auto_ets`'s `allow_multiplicative_trend=True` in a sentence about
    R's default; 3 are `horizon=40` in a `long_run_svar`/`max_share_svar`
    snippet in the guide corpus, attributed by the parser to whichever
    function was being checked. Refuted.
  - 1 is `panel_lp`'s guide bullet `bandwidth=4.0`, which is correct there:
    `panel_lp` defaults to `se_type="driscoll_kraay"`, where 4.0 is the
    effective default and the call runs (verified). Refuted — and it is
    exactly the same spelling that was wrong on `panel_fe` (L6), which is
    why the bullet is a pin.
  - 1 is `dols`'s guide bullet `ic="bic"` — the effective default behind a
    `None` sentinel, in a snippet that runs and selects the same lags and
    leads as the default call (verified). Refuted.
- **(c)** 67 call snippets scanned across the docstrings, the card sections,
  guide chapters 4/5/7/8/14, the Blanchard-Quah page and `api.md`: **every
  keyword exists in the signature, 0 unknown.**
- **(d)** 59 regex-derived string values probed (6 refusals, all of them the
  window bleeding one parameter's values into another's — `ic="add"` from
  the candidate enumeration, `se_type="dj"` from `bias_correction`; refuted
  and the window narrowed) **plus a hand-transcribed table of 101 documented
  values: 101/101 accepted**, including all twenty-two `level` spellings of
  `unobserved_components` (long and short form), the four ETS optimizers,
  the three bootstrap schemes, the three kernels, both `cov_type`s, both
  `bandwidth_rule`s and the four cointegration trends.
- **(e)** **35/35 inert keywords raise**: the seven `unobserved_components`
  component options; the eleven `ets_fit` cases (`damped` without a trend,
  `seasonal_periods` without a seasonal, `initial_states`/`smoothing_params`
  against the wrong `initialization`, `optimizer`/`max_iter` with
  `smoothing_params`, `level`/`n_sim`/`seed` with `horizon=0`, `n_sim`/`seed`
  on a class-1 model); the three `auto_ets` ones; `bandwidth_rule` with an
  explicit `bandwidth` and `diff` without a trend term on all three
  cointegrating regressions; the four `dols` search sentinels; `bandwidth`
  under `se_type="cluster"`; the three `panel_lp` half-panel jackknives on an
  unbalanced panel; and `lp_did` on an unbalanced mask. Every one a
  `ValueError` whose message names the keyword and the mode.
- **(f) The mask identity.** `panel_fe`, `panel_distributed_lag`, `panel_lp`
  and `lp_did` called with a mask of all ones against the same call with no
  mask: **max |Δ| = 0 on every returned key, all four** — the card's
  "asserted bit-identical" holds at the Python boundary too.

## Sweep G — complexity cliffs and resource caps

Wall-clock seconds in a fresh subprocess per cell (4 cores, release build,
the machine otherwise idle; the one flagged slope re-timed and agreeing to
0.04). "defaults" passes only the required arguments (and the mask, for the
masked cells) at the largest size.

| function | T=200 | T=800 | T=3200 | slope | defaults @3200 | note |
|---|---|---|---|---|---|---|
| `unobserved_components` (lltrend + seasonal 4, 1 start) | 0.428 | 1.288 | 5.030 | 0.89 | 4.824 | linear; the filter is O(T · k_states²) |
| `tvp_regression` (k=2, 1 start) | 0.354 | 0.898 | 5.679 | 1.00 | 17.345 (3 starts) | linear |
| `ets_fit` (A,A,A m=4, h=4) | 0.010 | 0.046 | 0.119 | 0.88 | 0.004 | |
| `auto_ets` (15 candidates, h=4) | 0.125 | 0.493 | 1.977 | 1.00 | 0.274 | linear in T at a fixed candidate set |
| `var_conditional_forecast` (steps=4) | 0.001 | 0.001 | 0.001 | 0.27 | 0.001 | flat in T — the cliff is in `steps` (M2) |
| `var_diagnostics` | 0.001 | 0.001 | 0.002 | 0.38 | 0.001 | |
| `var_select_order` (max_lags=4) | 0.001 | 0.001 | 0.003 | 0.60 | 0.009 (max_lags=8) | |
| `spa_test` (200 reps) | 0.001 | 0.003 | 0.023 | 1.04 | 0.041 (1000 reps) | |
| `model_confidence_set` (200 reps) | 0.002 | 0.004 | 0.015 | 0.81 | 0.060 | |
| `stepm_test` (200 reps) | 0.002 | 0.004 | 0.024 | 0.99 | 0.084 | |
| `fmols` | 0.000 | 0.001 | 0.002 | 0.46 | 0.002 | |
| `dols` (lags=1, leads=1) | 0.001 | 0.001 | 0.002 | 0.48 | 5.549 | defaults SEARCH: `max_lag = ceil(12(T/100)^¼)` = 29 at T=3200, so 900 candidate fits |
| `ccr` | 0.000 | 0.001 | 0.002 | 0.51 | 0.002 | |
| `panel_fe` + mask | 0.001 | 0.001 | 0.004 | 0.59 | 0.014 | |
| `panel_distributed_lag` + mask | 0.023 | 0.780 | **39.0** | **2.68** | 37.714 | **M1** (re-time 2.64) |
| `panel_lp` + mask | 0.002 | 0.008 | 0.039 | 1.02 | 0.046 | |
| `lp_did` + mask | 0.001 | 0.002 | 0.009 | 0.96 | 0.016 | |

Three cells flagged: `panel_distributed_lag@mask`'s slope (M1, confirmed),
and `unobserved_components` / `tvp_regression` crossing 5 s at T = 3200 —
both with slope ≈ 1.0, i.e. a linear filter on a long series, not a cliff;
refuted by the slope itself. `dols`'s 5.5 s "defaults" cell is the
documented information-criterion search over `max_lag × max_lead`
candidates, which grows with T by design.

**Caps** (`out/sweep_g_caps.txt`, each cell a child under 4 GB `RLIMIT_AS`
and a 60 s deadline). **137 cells: 103 refusals, 33 normal returns, 1
hang.** Every count driven outside its cap was refused before anything was
allocated, in ≤ 1 ms, with the count in the message: `seasonal`, each `freq_seasonal` period,
`freq_seasonal_harmonics`, `forecast_steps` (cap 100 000),
`n_starts` (cap 64, on both state-space models), `seasonal_periods`,
`horizon` (cap 1 000 000 on both ETS entry points), `n_sim` (`n_sim ×
horizon ≤ 2^28`), `lags`, `steps`, `nlags`, `max_lags`, `reps` (the
`reps × models` buffer), `block_size`, `leads`, `max_lag`, `max_lead`,
`powers`, `pre_window`, `post_window`. The documented caps hold at their
edge: `forecast_steps=100_001` and `n_starts=65` refuse while 100 000 (0.43 s)
and 64 (24 s) run; `horizon=1_000_001` refuses while 1 000 000 runs in 0.09 s
at 47 MB; `n_sim=2^26` with `horizon=4` is exactly 2^28 and runs. Nothing
aborted the allocator, and no cell returned `memerr` — the five aborts the
wave fixed stayed fixed. `reps=2^24` runs to completion (43–48 s, 0.6–1.9 GB)
rather than being refused, which is the honest outcome for a count inside
the budget.

The **one** cell that was neither refusal nor ok: `var_conditional_forecast`
with `steps=200_000` (M2), a HANG against the 60 s deadline. Two further
cells are recorded, not counted: `fmols`/`ccr` accept `bandwidth = 2^47` and
return in 1 ms (the kernel loop is bounded by the sample, so an absurd
bandwidth simply weights every available lag), and `auto_ets` accepts
`seasonal_periods = 2^47` and returns the best **non**-seasonal model with
nine candidates carrying `status="error"` — which is exactly what the
docstring promises ("failures never abort the search"), visible in
`candidates`, and the reason `auto_ets` does not apply `ets_fit`'s
`n_sim × horizon` guard when the winner turns out to be class 1 (nothing is
simulated, so nothing is allocated; the guard fires as documented the moment
the winner is not class 1 — measured).

## Sweep H — the seed ledger

| parameter | live configuration | same seed, in-process | same seed, new process | different seed | `None` |
|---|---|---|---|---|---|
| `ets_fit.seed` | ETS(M,A,M), h=4, `n_sim=200` | ✓ | ✓ | ✓ (max Δ forecast variance 1.3e-2) | **accepted → 0, documented** |
| `auto_ets.seed` | multiplicative-error winner, h=4, `n_sim=200` | ✓ | ✓ | ✓ (1.1e-1) | **accepted → 0, documented** |
| `spa_test.seed` | 200 reps, 4 models | ✓ | ✓ | ✓ (p 0.565 vs 0.550) | refused, by name |
| `model_confidence_set.seed` | 200 reps, 5 models | ✓ | ✓ | ✓ (max Δp 9.5e-2) | refused, by name |
| `stepm_test.seed` | 200 reps, 4 models | ✓ | ✓ | ✓ (p 0.565 vs 0.550) | refused, by name |

All seventeen cells bit-identical on two consecutive in-process calls, and
**all seventeen bit-identical at `RAYON_NUM_THREADS=1` vs `4` in fresh
processes** — promised for the three multiple-comparison tests ("one Philox
substream per replication, bit-identical at any thread count") and measured
for the other fourteen. `ets_fit(seed=None)` and `auto_ets(seed=None)` are
**seed 0 and say so** on every surface (`seed` (None: 0 when simulating)),
with the returned `seed` key reporting `0` — round 13's M3 lesson applied
correctly by the wave, which is the point of re-checking it. The three
bootstrap tests take a plain `int` and refuse `None` through the round-13
coercion rebuild, naming the parameter.

The one candidate: `auto_ets.seed` first measured as INERT — on a series
whose selected model is class 1, where the intervals are closed forms and
nothing is simulated. That is the documented behaviour ("`n_sim` and `seed`
act only if the selected model is not class 1"); the harness needed a
multiplicative-error configuration, which is now the registry's
`auto_ets@mul` cell. Refuted as a harness-configuration candidate.

## Sweep S — the malformed-input matrix

**0 panics, 0 aborts, 0 hangs, 0 harness errors, 0 non-standard exceptions
over 1563 cells** (17 cells, every parameter, 4 GB `RLIMIT_AS` per child):
1200 refusals (`ValueError`/`TypeError`/`OverflowError`) and 363 normal
returns.

The normal returns are all legitimate: valid small counts (`lags=0`,
`horizon=0`, `reps=1`, `max_lags=1`), `True` where a count goes (Python's
bool is an int), any 64-bit seed where the parameter is one, one-row-short or
duplicated-row data for single-array callables (still a valid sample),
integer and bool data arrays and flat numeric lists (coerced by design),
boundary floats inside the documented open ranges (`level=1e-300`,
`size=1e-300`), `None` for every Optional slot, and — for the mask cells —
`mask` given as an int or bool array, a half-zero mask and a mask that drops
one entity entirely, all of which are legitimate unbalanced panels.
`spa_test`/`stepm_test` accept a 1-D `model_losses` (documented: "a 1-D array
is one model"), and `model_confidence_set` accepts a four-column panel where
five were given (any `m ≥ 2`).

**Refusals that name the parameter.** Before the fixes, 34 of the 1200
refusals did not name the mutated argument. After (sweep re-run on the same
1563 cells, `out/sweep_s14.txt`): **2**.

| class | cells before | after | disposition |
|---|---|---|---|
| `ets_fit` length / sample-size messages | 20 | 0 | **L1, fixed** |
| Politis-White block-length failure | 2 | 0 | **L2, fixed** |
| a string inside `conditions` | 1 | 0 | **L3, fixed** |
| `freq_seasonal[i].harmonics` (named the field, not the keyword) | 3 | 0 | fixed alongside — the message now adds "(the Python keyword is `freq_seasonal_harmonics[0]`)" |
| `panel_lp` / `lp_did` window messages ("the lag order", "LP-DiD pre window") | 6 | 0 | fixed alongside — the messages now spell `n_lag_controls`, `pre_window`, `post_window` |
| a short `benchmark_losses` blamed on `model_losses` | 2 | 0 | fixed alongside — the message now says the expected length is "the length the first column or benchmark_losses sets" |
| a cascade: a one-row `y` reported as `seasonal = 4 with 1 observations` | 1 | 1 | refuted — the message names the constraint that actually failed first |
| `alpha=1e-300` → "argument `p` = 1 outside domain" | 1 | 1 | **known-open**: `1 − alpha/2` rounds to exactly 1 at that scale and the normal quantile refuses by its own symbol. Library-wide and pre-existing (`var_forecast` is identical); round 13's OPEN-3 class |

**The `also` fix, and what it revealed.** Round 13's cell protocol recorded
an `also` key on a cell — the extra keywords that put the call in the mode
where the mutated slot is live — and never applied it; round 14's runner
does. Re-running round 13's own 673-cell sweep on this tree therefore moves
five cells from refusal to normal return: `panel_distributed_lag`'s five
valid `bandwidth` values (`0`, `1e-300`, `3`, `True`, `1e300`) under
`se_type="driscoll_kraay"`, which round 13 measured under the *cluster*
default and recorded as inert-keyword refusals. All five are legitimate
Driscoll-Kraay bandwidths; the sweep's headline is unchanged (673 cells,
**0 unnamed refusals, 0 panics, 0 aborts, 0 hangs**) and
`out/sweep_s.txt` is refreshed with the corrected harness.

Two absurd-input honest infinities, recorded and not counted:
`dols(bandwidth=1e300)` gives every kernel lag weight 1.0 in double
precision, so the long-run variance is the unweighted autocovariance sum —
exactly 0.0 — and the t-statistics are ±inf (`fmols` refuses the same
bandwidth as a non-positive-definite `Omega_22`; a bandwidth ladder confirms
the long-run variance stays strictly positive at every reachable value:
0.99 at 0, 0.16 at 200, 0.0033 at 10 000); and
`var_conditional_forecast` with a cell pinned at 1e300 returns
`mahalanobis = inf`, `mahalanobis_pvalue = 0.0` — the correct answer for a
scenario infinitely far from the forecast.

## Sweep C — claims vs reality

**1173 informative numbers** (a decimal point, an exponent, or three or more
significant digits; years and one- or two-digit counts skipped) across the
thirteen card sections, the panel card's mask section, nine
validation-matrix rows, the added guide and Blanchard-Quah lines, the
CHANGELOG 0.10.0 entry and ROADMAP §0, looked up in a 30.8 MB corpus of
committed artifacts. **1147 found verbatim, 19 more reproduced by rounding a
committed number** (the cards quote `1469.18` for the fixture's
`1469.1763573975115`, `8.8e-4` for `0.0008838554163159363`, `6.1e-13` for
`6.123805974729926e-13`, ten ROADMAP tolerances the same way) — **1166 of
1173**. The Rust property and golden binaries the cards cite were re-run with
`--nocapture` into `out/rust_props.txt`: **96 passed, 0 failed** across
`fmols_golden`, `fmols_properties`, `ets_golden`, `ets_properties`,
`mcs_golden`, `spa_golden`, `spa_mcs_properties`, `unbalanced_golden`,
`uc_golden`, `uc_properties`, `var_cf_golden`, `var_cf_properties`,
`var_diag_golden` and `girf_budget`; the wave's nine pytest files ran
**662 passed**.

The seven that were not found, adjudicated:

| number | where | verdict |
|---|---|---|
| `232.2477` (×1) | UC card, "statsmodels −232.2477, SciPy −232.2446" | **refuted** — `fixtures/uc.json` holds `-232.24769478052863` and `-232.24462399392974`; the card writes a Unicode minus (U+2212), so the finder extracted the magnitude and the sign-sensitive rounding match missed it |
| `351.11` (×4) | forecasting card's runnable MCS example and the guide line quoting it | **refuted** — the block was executed: it prints its expected table line for line, `mean 351.11 0.000 no` included, together with the SPA statistic 2.151, p = 0.019 / 0.046, StepM `[3]` and block length 8 |
| `198.83` (×1) | CHANGELOG 0.10.0, the diffuse-filter bugfix | **refuted** — it is the PRE-fix log-likelihood of a bug this release fixed, so by construction no current artifact reproduces it; the post-fix reference (−253.81) is in the corpus |
| `3.0e-15` (×1) | ROADMAP §0, the **0.8.0** paragraph on `kernel_regression`'s leave-one-out criterion | **known-open, out of this round's scope** — a 0.8.0 claim (round 11's surface). The citing test asserts an *absolute* 1e-10 tolerance and prints nothing; re-running the 12 fixture cases here measures a worst *relative* gap of 2.8e-14 on `cv_loo`, so the sentence's scale is not the one a re-run reproduces. Flagged for the next round rather than touched in this one |

**The runnable card examples.** All eleven `python` blocks of the new card
sections were executed: **11/11 ran with no error**, and of the 23 output
lines their trailing comments claim, **22 were printed verbatim** and one was
not — L8, the `np.float64(…)` wrappers, now fixed. The blocks that carry no
expected-output comment (the cointegration `fmols`/`dols`/`ccr` example, the
ETS example) simply have to run, and do.

## OPEN — recorded, not fixed

The first two are the round's two known-open *candidates* (one sweep-S cell,
one sweep-C number, counted in the table); the third is an observation from
sweep E's census and the fourth an engineering note behind M1.

1. **`alpha` at 1e-300 refuses through an internal symbol.** `1 − alpha/2`
   rounds to exactly 1 in double precision, and the normal quantile refuses
   with *"argument `p` = 1 outside domain: requires 0 < p < 1"* — naming its
   own parameter, not the caller's `alpha`. Library-wide and pre-existing:
   `var_forecast(alpha=1e-300)` gives the identical message. Round 13's
   OPEN-3 class; a fix belongs with the quantile wrapper, not with a single
   callable.
2. **A 0.8.0 sentence in ROADMAP §0 quotes an agreement no committed
   artifact prints.** "`kernel_regression` … pinned to statsmodels `KernelReg`
   at 6.7e-15 with the leave-one-out criterion at 3.0e-15": the first figure
   rounds a fixture value, the second matches nothing, and the citing test
   asserts an absolute 1e-10 tolerance without printing what it achieved.
   Out of this round's surface (round 11 audited 0.8.0); flagged for the
   next round rather than touched here.
3. **The library has no single container rule for 1-D float and boolean
   payloads.** 402 returned keys are `ndarray` 1-D float against 32 Python
   lists, and boolean flags come both ways; 0.10.0 adds eleven to the
   minority buckets with precedent on both sides. Measured in sweep E's
   census; nothing promised, nothing changed.
4. **`panel_distributed_lag`'s unbalanced two-way projection could be
   linearised** by alternating projections instead of explicit time dummies.
   That is an approximation where the current route is exact, and the
   `PanelOLS` goldens pin the exact one; M1 documents the cost rather than
   trading the validation for speed.

## Verification record

- `9223f26` — the Rust refusal messages (L1, L2, and the four named
  alongside), the `check_band_alpha` guard (L7), the `_coerce` nested
  offender and fallback (L3), the six documentation fixes (M1, M2, L4, L5,
  L6, L8), the regenerated `api.md`, and
  `bindings/python/tests/test_audit_round14.py`: **34 pins, all passing**.
  One existing assertion was re-pinned rather than weakened:
  `test_lpdid.py::test_window_exceeding_the_panel_raises` matched
  `"post window"` / `"pre window"` and now matches the keyword spellings
  `"post_window"` / `"pre_window"` the fixed message uses.
- Sweep S re-run on the identical 1563 cells after the fixes: **34 → 2**
  refusals that do not name the mutated parameter, 0 panics/aborts/hangs in
  both runs.
- The fourteen Rust property/golden binaries the cards cite, re-run with
  `--nocapture` for sweep C: **96 passed, 0 failed** (`out/rust_props.txt`).
- Round 13's 673-cell sweep re-run on this tree (`out/sweep_s.txt`, the
  hygiene slice's record, refreshed): **0 unnamed refusals, 0 panics, 0
  aborts, 0 hangs**, with the five `bandwidth@dk` cells moved to normal
  returns by the harness `also` fix described above.
- `cargo fmt --all --check` clean; `cargo clippy --workspace --all-targets --
  -D warnings` clean (0 warnings); `cargo test -p` for the six touched crates
  (`tsecon-ets`, `tsecon-forecast`, `tsecon-panel`, `tsecon-ssm`,
  `tsecon-coint`, `tsecon-python`, release profile): **257 passed, 0
  failed**.
- `docs/gen_api_reference.py` leaves `api.md` byte-identical after the
  regeneration; `mkdocs build --strict` clean.
- Full Python suite, once, after every change: **2378 passed, 1 skipped in
  7:14** (2344 before the round, plus the 34 pins).

## Lessons

1. **A memory budget is not a complexity budget.** Both moderate findings
   are the same shape: the wave added a guard that refuses a count before it
   can abort the allocator, and the guard is sized in *bytes*. `steps` and an
   unbalanced `T` cost time super-linearly, and a count the byte-guard admits
   can run for a day. The cheap countermeasure is the one this round used —
   measure the slope at three sizes and publish it beside the cap.
2. **`None` is a promise in Python, a third time — but the failure mode
   moved.** Round 11 and round 13 found `None` sentinels documented as a
   concrete default that behaved differently. 0.10.0 got the sentinels right
   (`unobserved_components`'s four component flags were measured
   bit-identical to their stated defaults, and `ets_fit(seed=None)` says it
   means 0) and got caught on the *other* side: `fmols`'s `diff` states a
   default, `False`, that the library **refuses** when you type it. State the
   sentinel, and then say what happens when the reader passes the value you
   just named.
3. **A refusal that names a concept is not a refusal that names a
   parameter.** "smoothing parameters", "LP-DiD pre window", "the lag order",
   "loss column 0" all read well and none of them is a keyword the caller
   can search for. Sweep S's rule is mechanical and unforgiving — the exact
   token, word-bounded — and it took 34 messages to 2 in one pass over
   messages that had all been reviewed by a human.
4. **Run the example, don't read it.** The only sweep-C finding was a card
   block whose "expected output" was written before NumPy 2 changed a repr.
   Eleven blocks executed cost 40 seconds and found it; no amount of reading
   would have.
