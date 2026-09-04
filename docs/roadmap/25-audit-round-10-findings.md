# Adversarial audit, round 10 — findings

> **Working document.** Continuation of
> [round 9](24-audit-round-9-findings.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 10 audited the surfaces no round had ever touched: everything the
0.7.0 wave shipped (restricted VECM deterministic cases, Hansen-Seo
threshold cointegration, threshold VAR, the STAR family, plus the freshly
merged OU utilities and callable-forecaster engines from 0.6.0) and the
0.6.0 features that landed after round 9 closed. The round-9 plan had been
to chase that round's unverified sweep tails first — but those lived only
in agent transcripts, and the ephemeral container that held them was
recycled between sessions. Lesson applied below; the loss cost little,
because fresh class sweeps over unaudited surface out-hit tail-chasing on
round 9's own arithmetic.

**Design.** Four read-only class sweeps, all probing the installed wheel,
each finder carrying its own refuter discipline (adversarially attempt to
refute every candidate before granting CONFIRMED; cap fully-verified
findings and record the rest in-doc as OPEN — in the document, not in
transcripts, which is the round-9 lesson):

- **A — docs-vs-behavior**: every checkable claim in the new docstrings,
  cards, and api entries (~250 claims across 20 surfaces).
- **B — silent/inert arguments**: every kwarg of every new surface, probed
  default-vs-explicit on data where the kwarg has room to matter; the
  field report's highest-yield class.
- **C — degenerate inputs and honesty flags**: the constant/short/collinear/
  NaN/extreme battery, plus constructive verification that every honesty
  flag can actually fire.
- **D — cross-surface consistency**: eleven ledger items where two routes
  compute the same estimand (including fresh statsmodels draws, never the
  committed fixture draw).

**Totals: 17 confirmed findings (1 severe), 2 integrator-inflicted
merge-corruption findings, 0 confirmed findings refuted after
verification, all fixed in-branch.** The two clean bills matter as much:
sweep D measured 10 of 11 consistency items at or far inside documented
tolerance, and sweep C's verdict on the new estimator surfaces was
"unusually disciplined — nearly every degenerate cell refuses with an
argument-naming, fix-naming error."

---

## The severe finding: `star_test` let a Rust panic escape

`star_test(y, p, delay=T+1)` — or an empty series, or `delays=[1, T+50]`
— panicked with `capacity overflow` and surfaced in Python as
`pyo3_runtime.PanicException`, which `except Exception` does **not**
catch. Root cause: `star_test` was the single surface in the regime family
that built its design matrix *before* its sufficiency check, so
`t_total - start` wrapped as unsigned arithmetic and `vec![1.0; n]`
exploded. The adjacent boundary (`delay = T-1, T`) failed a different way:
refused, but with the wrong category ("(near-)constant transition
variable" for what is insufficiency).

**Fix**: the estimability check now runs before any design construction,
using the sibling contract (`insufficient data: N observations, at least M
required` — the same message `star`, `setar`, `threshold_var` already
emit), the boundary reports insufficiency, and `build_design` itself now
defends with saturating arithmetic and a debug assertion so no future
caller can reintroduce the wrap. Regression tests pin every panic input
from Python (teaching ValueError, never PanicException) and the exact
`needed` count in Rust.

## Sweep B — the inert-argument class, second harvest

Thirteen kwarg groups were accepted, documented as doing something, and
provably changed nothing (each verified bit-identical default-vs-explicit
before the fix; each live in its documented mode):

| surface | inert kwarg(s) | inert condition |
|---|---|---|
| `ccc_garch` / `dcc_garch` / `dcc_test` | `o` | `vol="garch"` (the default) — the exact trap `garch_fit` guards since 0.6.0, on the siblings whose docstrings promise "the same knobs" |
| `conformal_forecast` / `conformal_backtest` | `order` | any base but `"arima"` (validated by the parser, then dropped) |
| `conformal_forecast` | `n_eval` | `split` / `enbpi` |
| both conformal entry points | `calib` | `enbpi` |
| `conformal_backtest` | `batch` | `split` / `aci` |
| both conformal entry points | `gamma`; `seed`/`n_boot`/`optimize_beta`; `lags` | non-aci; non-enbpi; non-`"ar"` base |
| `hamilton_filter` | `maxlags` | `se="nonrobust"` (the guard lived only in the `se=None` arm — refused there, swallowed here) |
| `hamilton_filter` | `use_correction` | any `se` but `"hac"`, including `method="random_walk"` which refuses `se`/`maxlags` two lines away |
| `bn_filter` | `d0`, `dt` | fixed `delta=` |
| `backtest` | `period` | `naive`/`drift`/`mean` and every Python callable (`insample_period` verified live everywhere) |
| `spread_zscore` | `dt` | all three parameters frozen |
| `threshold_vecm` | `n_grid_beta`, `beta_span` | `beta=` supplied |
| `vecm` | `first_season` | `seasons=0` (and the modulo-`seasons` wrap was undocumented) |

All now follow the `garch_fit` sentinel convention: explicit-where-inert
refuses with a teaching error naming the mode that would use it; the
default call is asserted bit-identical; the kwarg is asserted still live
where documented. `proxy_ar_sets` was measured **fully clean** across its
twelve kwargs — the round-5 fix that established the refusal convention is
now the in-repo gold standard the rest of the surface has been brought to.

## Sweep C — degenerate inputs (beyond the panic)

- The TVECM/`hansen_seo_test` minimum-sample error misstated its own
  requirement in mixed units (claimed "12 usable rows", succeeded at 10)
  and explained itself with the Johansen per-equation count rather than
  the TVECM one. Fixed with an exact, bisection-verified minimum
  (feasibility is non-monotone in n because `ceil(trim·n)` grows — the
  fix scans) in consistent units.
- Three teaching errors leaked Rust internals (`beta = Some(..)`,
  `fit_vecm_det(.., Constant)`, `psi_reduced_form_cov`, and a
  `n_grid_gamma` name on a surface whose kwarg is `n_grid`); all now name
  the Python argument and a working Python remedy.
- Message honesty repairs: `bn_filter` called a linear ramp "constant"
  (it is the differences that are constant); the OU surfaces reused
  eigenvalue-flavored NaN text from the coint crate; `spread_zscore`
  refused `kappa=inf` via a positivity message that never stated
  finiteness; `vecm`'s seasonal-heavy insufficiency hint recommended a
  `k_ar_diff` reduction while ignoring that the seasonal dummies consume
  the degrees of freedom; `backtest`'s constant-window MASE refusal
  advised "an unscaled measure" no `backtest` parameter can select.
- Proxy surfaces silently dropped **inf** proxy values as if they were
  missing. NaN-as-missingness is a documented convention (now documented
  on every proxy surface, not just `proxy_svar`); inf is corruption and
  is now refused with a teaching error, family-wide through the shared
  alignment path (`proxy_svar`, `proxy_svar_bands`, `proxy_first_stage`,
  `proxy_ar_sets`) — a behavioral change, flagged loud in the CHANGELOG.
- **Flag constructibility, resolved**: sweep C could not trip
  `star.converged=False` or the bottom-wall `gamma_at_boundary` from
  random data. The fix agent proved both reachable (an objective-resolution
  route for `converged`; a deterministic construction landing standardized
  γ within 5.2e-10 of the bottom wall for the flag) and pinned each with a
  test — the sweep missed them because the bottom-wall detection band
  (1e-9) is narrower than the optimizer's resolution (1e-8). No dead
  flags; no code change needed.

## Sweep A — docs vs behavior

Two undocumented return keys (`ou_fit.level`, `markov_switching_ar.
iterations`) — documented now, and the docstring-keys gate extended to
full-diff so the class cannot silently recur on those surfaces. One
fixture-scoped card sentence that read as universal (star's
"smooth data leaves the boundary flag False") — scoped. The
`hansen_seo_test`-accepts-k>2-while-`threshold_vecm`-refuses contrast,
previously documented only on one side — now on both. Everything else
checked exact, including all nine VECM deterministic cases against
statsmodels at ≤1e-10 on fresh draws and every worked example's printed
numbers.

## Sweep D — cross-surface consistency (the measured ledger)

Ten of eleven items clean, with margins: vecm-vs-statsmodels ≤6.8e-14
(alpha) on fresh seeds; `vecm("co")` β vs `johansen` evec cosine exactly
1.0; the documented-asymptotic `"colo"`↔`det_order=1` gap shrinking
monotonically 7.0e-10 → 2.2e-16 as T quadruples; `hansen_seo_test`'s null
model bit-equal to `vecm("co")` on every seed; the STAR large-γ limit at
2.1e-15 once the convention (G=0.5 at s=c) is respected; OU vs `AutoReg`
at 3.3e-16; BN identities at 7.1e-15 with an independent long-run-forecast
route at 2.6e-13; callable-vs-built-in conformal paths at 4.7e-15; the
second_order/second_order_bc boundedness statistics bit-identical as
documented; all GARCH reconstructions at 0.0. The eleventh item was the
mgarch `o` finding (fixed above) — the numerics themselves were clean.

## Two integrator-inflicted findings, and the lesson

The threshold-block merge used union-style conflict resolution on files
where git had split *functions* across conflict hunks that shared interior
context lines (four identical signature lines appearing in both `fn star`
and `fn threshold_var`). The union interleaved the fragments. Caught in
`lib.rs` at compile time; **missed** in two artifacts that do not compile:
the shipped `.pyi` stub (syntactically invalid — type checkers received
nothing for the entire module; would have failed CI's mypy gate) and the
cointegration-regime card (doubled roster, a dangling mid-sentence
fragment, two worked examples merged into one fence sharing RNG state).
Sweeps A and B found both within hours. Both repaired by reconstructing
the region from the two merge parents with the shared context duplicated
into each side, then verified (`ast.parse`; both card examples executed).

**Integrator rule adopted**: union-regex resolution is safe only for
append-style text (CHANGELOG, matrix rows). For code, stubs, and
structured docs, resolve from the parents — and after any merge, run
`ast.parse` on the stub and execute the touched cards' fences, not just
the compilers and test suites.

## A canary fired — and paid for the whole canary system

statsmodels 0.15.0 (released between sessions) added
`tsa.filters.api.hamilton_filter`. The absence canary in the bn_filters
fixture — written when no reference existed, precisely so its failure
would announce one — failed the suite on the fresh environment. Response:
the canary test now runs a live, version-gated cross-check of our full
cycle/trend decomposition against the new statsmodels implementation —
**measured max abs 4.2e-14 on first contact, asserted at 1e-10** — while
older statsmodels environments keep the absence assertion and the fixture
keeps its generation-time provenance. `hamilton_filter`'s validation grade
gains a genuine third-party leg it could not have had when it shipped.

## Lessons

1. **The inert-argument class is structural, not incidental.** Round 9
   found it on old surfaces; round 10 found thirteen more groups on
   surfaces written *after* the convention was established, including by
   agents told to follow it. The sentinel-refusal pattern is now enforced
   by per-surface tests (explicit-inert raises / default bit-identical /
   live-where-documented), which is the only form that holds.
2. **Record sweep tails in the findings doc.** Round 9's unverified
   candidates lived in transcripts and died with the container. Round
   10's OPEN items are in this document. (Remaining OPEN, all low: the
   conformal enbpi default-base ergonomics — `base` omitted always
   refuses; `bn_decomposition` p/q fixed-path no-op now documented; no
   others survived the fix batch.)
3. **Merge tooling has a failure class of its own.** The audit's
   finder/refuter machinery caught the integrator's damage the compilers
   missed. Auditing the *process* products (stubs, cards, generated docs)
   is as necessary as auditing the estimators.
4. **Canaries are cheap and they fire.** One dependency release turned an
   honest absence claim into a machine-precision third-party validation,
   automatically, because a test existed whose only job was to notice.
