# The consolidated ledger — every open, deferred, or named follow-up item the project has recorded, re-verified against main at `19d308e` (tsecon 0.8.0)

> **Working document**, part of the whole-repository audit (September 2026).
> Excluded from the published site like everything else under
> `docs/roadmap/`. It is a *ledger*, not a findings doc: nothing here is a new
> defect. Every row is something an earlier round, release note, card, or
> report **already recorded** as open, deferred, or a follow-up — read
> together for the first time and checked against today's tree and wheel.

## Scope and method

**Sources read in full** (the "to date" record of open tails): the audit brief
and ten findings documents (`docs/roadmap/16`–`18`, `20`–`26`), the two
engineering follow-ups (`15-proxy-svar-bands.md`, `21-long-horizon-and-joint-
inference.md`), the research scan (`19-research-contributions.md`),
`ROADMAP.md` §0 (build-later / deferred / on deck / every "named follow-up"),
`CHANGELOG.md` 0.2.0–0.8.0 (every "follow-up", "not yet", "deferred",
"remains", "still", "Not in this release"), `docs/reference/validation-matrix.md`
(every "follow-up", "not runnable", "named", "property-MC until"), every model
card (`follow-up`, `not yet`, `deferred`, `not implemented`, `open item`),
`docs/examples/interval-coverage.md` and the two Monte-Carlo pages (unmeasured
surfaces and standing recommendations), `lab/REPORT.md` with
`lab/experiments/results/exp06.md`, and `docs/roadmap/10-machine-learning.md`
Tier 3/4 (the module `ROADMAP.md` names as next).

**Extraction rule.** One row per recorded item, with the file and line it was
recorded at, the version or date it was recorded against, and the severity or
tier the source itself assigned. Items recorded in several places are one row
with the duplicates noted under `evidence` and their own rows marked
`SUPERSEDED (tracked as …)` so the per-source counts still reconcile to the
sources.

**Re-verification.** Each row was checked against main today by one of three
routes, named in the `evidence` column: (i) a CHANGELOG bullet or commit; (ii)
a grep of the tree (`crates/`, `bindings/`, `docs/`); (iii) a probe on the
freshly built wheel (`lab/audit/repo/ledger/probes.py`, log in
`lab/audit/repo/ledger/probes.log`, probe ids `P01`–`P58`). Where none was
possible the row says `UNVERIFIED` and why. The wheel was built in this
worktree (`maturin develop --release`, own `CARGO_TARGET_DIR`) from `19d308e`.

**Status vocabulary.**

| status | meaning |
|---|---|
| `FIXED` | closed on main; the evidence column cites the release, commit, grep, or probe that proves it |
| `OPEN` | still open on main today; carries a `value` / `cost` estimate and a suggested next step in §4 |
| `SUPERSEDED` | the recorded item no longer stands as written — replaced by a shipped alternative, or tracked under another row (named) |
| `N/A` | not applicable: refuted, by-design with a documented contract, unrecoverable, or a recorded observation with no action attached |
| `UNVERIFIED` | could not be checked here; the reason is stated |

Severity is inherited from the source (`silent-wrong-answer` / `silent-noop` /
`trap` / `overclaim` / `cosmetic`, the round-11 `moderate` / `low`, the roadmap
tiers, or the release's own grading words). `value` and `cost` are this
audit's estimates (low / medium / high) for OPEN rows only.

**What this ledger does not do.** No code was changed. The only edit outside
this file and the probe outputs is the correction recorded in §6 — a source
document that still said an item was open when main provably closed it.

## 1. Totals per source

Counting rule (mechanical — the table below is generated from the ledger rows
by the status cell, so it cannot drift from them): a row whose status cell
names any remaining open work (`OPEN`, `OPEN (half)`, `FIXED (core) / OPEN
(frontier)`, …) counts as **open**; a row containing `UNVERIFIED` counts as
**unverified**; otherwise the leading word counts. `SUPERSEDED` rows are the
same item recorded in a second place (or replaced by a shipped alternative)
and are listed so the per-source counts reconcile to the sources; the
**distinct** open items are the `OPEN` rows, grouped in §3.

| source | recorded | fixed | open | superseded | n/a | unverified |
|---|---|---|---|---|---|---|
| `16-adversarial-audit-brief.md` | 15 | 7 | 5 | 2 | 1 | 0 |
| `17-audit-round-2-findings.md` | 20 | 14 | 2 | 2 | 2 | 0 |
| `18-audit-rounds-3-4-findings.md` | 18 | 12 | 3 | 0 | 3 | 0 |
| `20-audit-round-6-findings.md` | 12 | 9 | 1 | 2 | 0 | 0 |
| `22-audit-round-7-findings.md` | 9 | 8 | 0 | 0 | 1 | 0 |
| `23-audit-round-8-findings.md` | 11 | 4 | 1 | 0 | 6 | 0 |
| `24-audit-round-9-findings.md` | 14 | 8 | 2 | 0 | 3 | 1 |
| `25-audit-round-10-findings.md` | 10 | 8 | 1 | 0 | 1 | 0 |
| `26-audit-round-11-findings.md` | 23 | 8 | 10 | 0 | 5 | 0 |
| `21-long-horizon-and-joint-inference.md` | 5 | 3 | 1 | 0 | 1 | 0 |
| `15-proxy-svar-bands.md` | 9 | 2 | 4 | 1 | 1 | 1 |
| `19-research-contributions.md` | 21 | 5 | 16 | 0 | 0 | 0 |
| `ROADMAP.md` §0 (+ `14-packaging-distribution.md`) | 19 | 0 | 17 | 0 | 1 | 1 |
| `CHANGELOG.md` 0.2.0–0.8.0 | 20 | 5 | 8 | 3 | 4 | 0 |
| `docs/reference/validation-matrix.md` | 16 | 0 | 6 | 3 | 7 | 0 |
| model cards | 18 | 0 | 4 | 10 | 4 | 0 |
| `interval-coverage.md` / Monte-Carlo pages | 9 | 1 | 4 | 2 | 2 | 0 |
| `lab/REPORT.md` + `exp06.md` | 10 | 3 | 5 | 0 | 2 | 0 |
| `10-machine-learning.md` Tier 3/4 | 16 | 1 | 14 | 0 | 1 | 0 |
| **all sources** | **275** | **98** | **104** | **25** | **45** | **3** |

## 2. The ledger (all items)

Columns: `id` · `source` (file:line) · `recorded` (what the source said) ·
`as of` (version / date it was recorded against) · `severity / tier` (the
source's own) · `status` · `evidence` (release, grep, or probe id) ·
`value / cost` (OPEN rows only).

### 2.1 `16-adversarial-audit-brief.md` (round 1 brief; "Known open" and lens targets)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R1-01 | 16:374-375 | The SSM diffuse period terminates on a norm test over `P_inf`; "the fix is rank-counting termination" | 0.3.0 | class-2 lineage (absolute tolerance) | OPEN | `crates/tsecon-ssm/src/filter.rs:113-141` still uses the absolute `TOLERANCE_RANK` Frobenius test and argues at length it is *deliberately* absolute ("`P_inf` is a dimensionless rank indicator"); round 2 refuted the element-level sibling on the same grounds (17:465-472). Design-defended but the brief's item was never formally closed | low / medium |
| R1-02 | 16:376-377 | `lp_iv`, `lp_multiplier`, `lp_state` have no cross-horizon covariance → Šidák/Bonferroni only, sup-t refused | 0.3.0 | trap-class ergonomics | OPEN | P24: all four (`panel_lp` too) still refuse `band="sup-t"` with a teaching error; LP card :201-206, :270-272, :327-328; panel card :255-261 | medium / high |
| R1-03 | 16:378-382 | `proxy_ar_sets` propagated variance is estimate-correlated at long horizons; "a re-engineered long-horizon correction is the open item" | 0.3.0 | trap | SUPERSEDED | Opt-in repairs shipped: `rf_method="second_order"` (0.3.0, CHANGELOG:1766-1777) and `"second_order_bc"` (0.6.0, CHANGELOG:1258-1270); P04 confirms the menu. The *default* is still `"delta"` — that decision is tracked as N21-03 | — |
| R1-04 | 16:383-384 | `ivx_test`'s joint Wald size in k; "a size-restoring joint test is the open item" | 0.3.0 | trap | FIXED | `joint="bonferroni"` shipped 0.3.0 (CHANGELOG:1778-1790) and became the **default** in 0.5.0 (CHANGELOG:1450-1460); P03: default `joint='bonferroni'`, `wald_scalar` returned | — |
| R1-05 | 16:385-387 | `flp`'s per-element `se` for estimated scores; "a generated-regressor correction (or bootstrap route) is the open item" | 0.3.0 | trap | OPEN | Disclosed on card (functional-shocks.md:125), guide (09:395) and docstring (P25: docstring says "generated"); no correction kwarg exists (P25). `flp_scenario`'s `w'β` route is the documented workaround | medium / high |
| R1-06 | 16:388-390 | A minimum-cycles advisory for `nsdiffs`/`seasonal_strength`; "whether the advisory belongs in the output itself is open" | 0.3.0 | trap → doc | OPEN | Card documents the saturation table (diagnostics.md:298-300); P23: `nsdiffs` output carries no advisory-like key; docstring has no "cycle" text | low / low |
| R1-07 | 16:391-400 | Unmeasured-seven closed; "still outside the registry: `growth_at_risk`, `proxy_svar`/`proxy_ar_sets`, `nongaussian_svar`, GARCH forecast intervals, `flp`" | 0.3.0 | lens 7 | FIXED | 0.6.0 `docs/examples/coverage/proxy_garch_tail.py` (13 rows, registry 50 → 63; CHANGELOG:1273-1290; interval-coverage.md:1119-1180); `nongaussian_svar`/GARCH `variance_forecast` verified interval-free by per-run tripwire (P50) | — |
| R1-08 | 16:401-402; interval-coverage.md:1454-1458 | "Only two nominal levels are swept anywhere" | 0.3.0 | lens 7 | OPEN | The coverage page still says so (:1454); round 2's methodological result (17 "A methodological result worth keeping") shows scale-type shortfalls peak at 70–82 % nominal, so a 95 %-only sweep understates them | medium / medium |
| R1-09 | 16:322-323 | Lens-7 room: the `bvar_*` family as Bayesian calibration (draw from the prior, check the credible set) | 0.2.0 | lens 7 | FIXED | Round 6 ran full SBC: found and fixed the ML-II collapse, measured the GLP plug-in at 0.82–0.85 (20:25-61); `bvar_fit` core machine-exact (20:161-166) | — |
| R1-10 | 16:323 | Lens-7 room: `bai_perron`'s Bai-1997 break-date CIs | 0.2.0 | lens 7 | FIXED | Round 6 finder 3 swept them "within its disclosed scoping and found consistent" (20:170-172); Bai-Perron replication pins the published 90 % CIs (22:278-281) | — |
| R1-11 | 16:324 | Lens-7 room: `adl_midas` | 0.2.0 | lens 7 | N/A | No callable of that name exists; the MIDAS surface is `midas_weights`/`umidas`/`weighted_midas` (P56). `umidas` HAC intervals were measured in 0.3.0 (CHANGELOG:1756-1758); `weighted_midas` ships no interval (tripwired) | — |
| R1-12 | 16:324 | Lens-7 room: `proxy_ar_sets` | 0.2.0 | lens 7 | FIXED | Round 6 finding 8 measured it (20:109-122); the repair lineage is R1-03 | — |
| R1-13 | 16:67; 22:174-193 | Nelder-Mead's absolute `f_tol` certificate and the "x-side mixed-scale" concern | 0.1.0 | class 2 | FIXED | Round 7 F1b: initial-simplex edge floored at scipy's `zdelt` (22:174-193; CHANGELOG:1599-1603) | — |
| R1-14 | 16:106-107 | "Roughly twenty [doc claims] were found and fixed in two days, and that sweep was not exhaustive" | 0.2.0 | class 5 | SUPERSEDED | Rounds 8, 10 and 11 ran the class to completion over their surfaces (23; 25 sweep A; 26 sweeps E/F, 162/162 callables); the durable-generator follow-up is R11-L1 | — |
| R1-15 | 16:153, 335 | `growth_at_risk` covering 0.61 at h=12 (missing HAC correction) | 0.1.0 | silent-wrong-answer | FIXED | Fixed "since" 0.2.0 (16:334-335); measured in the 0.6.0 registry — the Newey-West correction is "the whole story at the median" (interval-coverage.md:1171-1180) | — |

### 2.2 `17-audit-round-2-findings.md` (round 2, run at 0.2.0)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R2-01 | 17:58-129 | `lp(cumulative="both")` reports an inconsistent (flat-in-h) standard error; coverage 0.507 at h=12 | 0.2.0 | silent-wrong-answer | FIXED | 0.3.0: HAC is the default for that mode and the `lag_augmented` pairing raises (CHANGELOG:1957ff, 16:340-342); measured 0.507 → 0.920 at T=400 (CHANGELOG:1752-1753); P09 confirms both halves on the wheel | — |
| R2-02 | 17:133-176 | `bvar_ssvs` not scale-invariant (absolute `gamma_b`); posterior inclusion flips; `ssvs.rs:36` overclaim | 0.2.0 | silent-wrong-answer + overclaim | FIXED | 0.3.0 "scale-carrying hyperpriors (now semi-automatic)" (16:341-342); `crates/tsecon-bayes/src/ssvs.rs:51-69` now documents the per-equation `GAMMA_B_REL * s2_j` rate and the invariance it buys | — |
| R2-03 | 17:200-243 | `panel_fe` accepts a rank-deficient design its own guard exists to reject (guard anti-monotone in collinearity) | 0.2.0 | silent-wrong-answer | FIXED | 0.3.0: two scale-invariant checks + the linearmodels rank criterion, monotone (CHANGELOG:1909-1922); P08: an entity-constant share regressor is refused | — |
| R2-03b | 17:232-241 | Sub-claim: unpivoted QR returns a point off the LS manifold — "plausible and unverified" | 0.2.0 | (unverified) | N/A | Moot: the guard now refuses the rank-deficient design before the solve; never verified, no longer reachable | — |
| R2-04 | 17:247-293 | `ols(se_type="hac")` misses its documented statsmodels match at defaults; the `use_correction` default story stated backwards on two surfaces; `har_rv` defaults the other way | 0.2.0 | trap + overclaim | FIXED | 0.3.0 "the HAC `use_correction` default story (`har_rv` flipped, every claim corrected)" (16:342-343); today: migration guide :95 states the opposite defaults and the exact `sqrt(n/(n−k))` gap; cookbook :8, :133-140; expectations card :56-61; `crates/tsecon-hac/src/ols.rs:118` now says statsmodels is "internally" inconsistent | — |
| R2-05 | 17:297-317 | The published coverage tables silently drop a harvested row (the `hc3` row) | 0.2.0 | trap | FIXED | 0.3.0: tables regenerated, `check_page.py` + a binding test diff every row against the registry (CHANGELOG:1975-1983) | — |
| R2-06 | 17:321-340 | `panel_lp(jackknife=True)` costs 8 pp coverage and the cookbook recommends it where it costs most | 0.2.0 | trap | FIXED | 0.3.0: cookbook/card/guide/docstring re-scoped to moderate-to-long T (CHANGELOG:1998-2004; cookbook :170); the estimator-side answer, SPJ `bias_correction="spj"`, shipped the same release (P57) | — |
| R2-07 | 17:344-371 | `smooth_lp`'s default CV λ grid is absolute; endpoint pinning; card example lands at grid index 15/16 while saying "mid-grid" | 0.2.0 | trap | FIXED | 0.3.0 "`smooth_lp`'s absolute λ grid" (16:346); LP card :432-447 now states the grid is exactly rescaling-invariant and what an endpoint `lambda_used` means | — |
| R2-08a | 17:377-385 | Two pages (and `lp_family.py`) still say no simultaneous band exists | 0.2.0 | overclaim | FIXED | grep today: no hit for "No function in the library reports a simultaneous" (testing.md), "No simultaneous band exists anywhere" (SI card), "none of these functions reports" (`lp_family.py`); CHANGELOG:1947-1950 | — |
| R2-08b | 17:386-392 | Test counts wrong in three of four places | 0.2.0 | cosmetic | FIXED | 0.3.0: "re-measured and corrected (README now uses resilient phrasing)" (CHANGELOG:2019-2021). Today's ROADMAP:13 claims 1775 Rust / 1526 Python; **not re-counted here** (a full test collection is outside this sweep's budget) | — |
| R2-08c | 17:393-395 | `docs/quickstart.md` ships stray harness markup (`</invoke>`) | 0.2.0 | cosmetic | FIXED | grep count of `</invoke>` / `</content>` in quickstart.md = 0; CHANGELOG:2019 | — |
| R2-08d | 17:396-404 | `panel_pmg` blames the panel for a floating-point failure (absolute `TOL = 1e-12` on θ) | 0.2.0 | cosmetic (misdiagnosis) | FIXED | 0.6.0 round 9: relative rule at 3e-13 plus a deterministic restart from the PSS start; 13–16/20 hard failures → 0/20 (24:92-100; CHANGELOG:1140ff) | — |
| R2-08e | 17:405-412 | Missing shape preconditions panic (`engle_granger.rs:202`, `lib.rs:2926`) as `PanicException` | 0.2.0 | cosmetic (uncatchable) | FIXED | 0.3.0 "the panic pair" (16:349; CHANGELOG:1923-1928); P39: `engle_granger` on a `(1, k)` sample raises a catchable `ValueError` | — |
| R2-08f-i | 17:413-425 | Smaller drift: `testing.md` tallies; `results.md` omits `CheckSeriesResults`/`CoefficientFrame`; `dfm_nowcast` "T × r" docstring | 0.2.0 | cosmetic | FIXED | tallies corrected 0.3.0 (CHANGELOG:1947-1950); `docs/reference/results.md:123,130` list both classes; the `smoothed_factors` shape was fixed and pinned in round 11 L7 (CHANGELOG:437-439) | — |
| R2-08f-ii | 17:423-425 | `__init__.pyi` says `link` is "probit" or "logit" with no dynamic-path caveat, contradicting the runtime docstring ("probit only") | 0.2.0 | cosmetic | OPEN | Stub `bindings/python/python/tsecon/__init__.pyi:2533` still reads `link is "probit" or "logit"`; the runtime doc (`lib.rs:10384-10385`) says "probit only" for `dynamic=True`; P53 | low / low |
| R2-09 | 17:431-433 | Residue: `recession_probit(link="banana")` silently accepted on the dynamic path (validation only in the `else` branch) | 0.2.0 | cosmetic | OPEN | P19: `recession_probit(y, x, link="banana", dynamic=True)` **returns a dict** on the 0.8.0 wheel — the `link` validation (`lib.rs:10421`) still lives only in the static branch; the dynamic model is probit-only and says so, but an unknown `link` string is not refused there | low / low |
| R2-10 | 17:487-489 | Residue: `results/_predreg.py` titles the forest plot "95% intervals" with no naive-normal caveat | 0.2.0 | cosmetic | FIXED | 0.3.0 `plot_estimates` renders the caveat on the figure (CHANGELOG:2017-2019) | — |
| R2-11 | 17:427-506 | Thirteen refuted candidates (recession `link`, `panel_unit_root` LRV kwargs, `var_irf_bands` seed on asymptotic, `gmm_nonlinear` start, MS-AR variance floor, CCC/DCC singular R, `panel_fe` N=1, `dfm_nowcast` interior NaN, `ffbs.rs` tolerance, `qreg.rs` floors, `max_share_svar(sign=)`, `panel_lp(jackknife=)` se, `proxy_svar_bands(robust_f=)`, `ols(use_correction=)`, `panel_lp` Driscoll-Kraay N=100, `quantile_lp` tail) | 0.2.0 | refuted | N/A | Recorded so no round re-derives them; the brief's "Refuted" pointer (16:298-307) | — |
| R2-12 | 17 "A methodological result worth keeping" | Sweep α, not just 95 % — a scale-type miss is smallest at 95 % | 0.2.0 | method | SUPERSEDED | Tracked as R1-08 (the same open item, recorded on the brief and the coverage page) | — |
| R2-13 | 17 "Reproducing this audit" | Probe scripts were scratchpad-only and are gone; the four harnesses worth rebuilding (fingerprint, switch sweep, scale sweep, convention survey) | 0.2.0 | method | SUPERSEDED | Round 11 committed its registry-driven harnesses under `lab/audit/round11/` (sweeps E–H); the lens-1–3 harnesses of rounds 2–4 were never committed and no later round rebuilt them as durable code — see R34-13 | — |

### 2.3 `18-audit-rounds-3-4-findings.md` (rounds 3–4, run at 0.2.0)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R34-00 | 18:18-46 | The audit's own tooling: 16/59 axes and 6/12 seed cases never compared; 13 functions unreached by the scale sweep | 0.2.0 | method | FIXED | Both holes closed in the same document (18:40-46); "report comparisons made, never attempted" is a hard constraint in the brief (16:226) | — |
| R34-01 | 18:50-113 | `panel_fe` reports a t-statistic for a regressor the fixed effects annihilate (19.2 % nominally significant) | 0.2.0 | silent-wrong-answer | FIXED | Same fix as R2-03 (CHANGELOG:1909-1922); P08 | — |
| R34-02 | 18:117-186 | `flp`'s standard errors condition on the estimated eigenfunctions (se/sd ≈ 0.67, flat in T); no page warns | 0.2.0 | trap | FIXED | The disposition the round asked for — disclosure on card/guide/docstring — shipped in 0.3.0 (CHANGELOG:1984-1991; functional-shocks.md:125; guide 09:395). The engineering correction is R1-05 (OPEN) | — |
| R34-03a | 18:190-226 | `growth_at_risk` computes `bse_powell` and drops it at the binding | 0.2.0 | overclaim | FIXED | 0.3.0 (CHANGELOG:1964-1967); P05: `bse_powell` returned | — |
| R34-03b | 18:229-244 | `markov_switching_ar` reduces the Kim smoother's `n × k` matrix to one column; `filtered_prob` never surfaces | 0.2.0 | trap | FIXED | 0.3.0 (CHANGELOG:1968-1971); P06: `smoothed_prob` is `(n, k)`, `filtered_prob` and (0.6.0) `ar` present | — |
| R34-03c | 18:246-256 | `lp(se="hac", band=…)` computes `cov_se_max_rel_diff` and drops it | 0.2.0 | overclaim | FIXED | 0.3.0 (CHANGELOG:1958-1963); P07 | — |
| R34-04 | 18:260-322 | `ivx_test`'s joint Wald loses its size in k and n does not fix it; only k=1 is tested | 0.2.0 | trap | FIXED | Documented 0.3.0 (CHANGELOG:1992-1998), Bonferroni route 0.3.0, default flipped 0.5.0, k=5 chi-square size regression pinned (21:217-219); P03. The `joint="chi2"` route keeps the defect by design and says so | — |
| R34-05a | 18:329-339 | `gmm_nonlinear` blames `initial` for a fault in the moment function | 0.2.0 | cosmetic | FIXED | 0.3.0 (CHANGELOG:1972-1976) | — |
| R34-05b | 18:341-353 | `long_memory_d.__doc__` calls `se` "asymptotic"; `predictive_regression.__doc__` names a `rho` key that does not exist | 0.2.0 | cosmetic | FIXED | 0.3.0 + `test_docstring_keys.py` tripwire (CHANGELOG:2013-2016); P38 | — |
| R34-05c | 18:360-363 | `which-model-when.md` contradicts itself about IVX on one page | 0.2.0 | cosmetic | FIXED | grep today: the stale "until IVX lands" sentence is gone; CHANGELOG:1996-1997 | — |
| R34-06 | 18 "Refuted" | Six refuted candidates (`cg_regression` intercept / MZ Wald size, `zero_sign_svar(weighted=)`, `growth_at_risk(rearrange=)`, `var_forecast(band_scope=)`, Stambaugh `se`, `factor_model` criteria) | 0.2.0 | refuted | N/A | Recorded with the evidence that killed them | — |
| R34-07 | 18 "Refuted" (the `quantile_lp` note) | "The real `quantile_lp` interval gap sits … on the h axis": the Powell sandwich is not HAC under overlapping multi-step outcomes | 0.2.0 | trap (self-declared) | OPEN | quantile card :43-48 still says "`growth_at_risk` carries the Newey-West correction … `quantile_lp` does not yet"; P30: `quantile_lp` has no `se`/`hac_lags` kwarg | medium / medium |
| R34-08 | 18 "Swept and found sound" | `lp(cumulative=True)` is at nominal but "worth a docs line that it is materially worse-calibrated than the level path at T=200" | 0.2.0 | cosmetic | OPEN | No such sentence on the LP card (grep "worse-calibrated" / "cumulative=True … T=200": none); the 0.3.0 registry does measure the cumulated-outcome mode (CHANGELOG:1752-1754), so the number exists, the sentence does not | low / low |
| R34-09 | 18 "Incidental" | `interval-coverage.md`'s "not measured" list omits `flp`/`flp_scenario` | 0.2.0 | cosmetic | FIXED | Named in 0.3.0 (CHANGELOG:1990-1991), measured in 0.6.0 (R1-07) | — |
| R34-10 | 18 "Incidental" | Surfaces that ship no interval (`nelson_siegel`, `svensson`, `dynamic_ns`, `weighted_midas`, `favar`, `bvar_fit`, `structural_fevd`) | 0.2.0 | observation | N/A | Nothing to measure; `favar` bands and `bvar_fit` posterior uncertainty have since shipped and are measured (CHANGELOG:1754-1756; 24:73-76); NS/`weighted_midas` are key-set-tripwired (CHANGELOG:1759-1760). The un-tripwired two are IC-05 | — |
| R34-11 | 18 "Incidental" | `iv_gmm`'s positional order is `(x, z, y)` — "worth a keyword-only signature or a docstring warning" | 0.2.0 | cosmetic | FIXED | The docstring route was taken in 0.3.0 (CHANGELOG:2010-2012); P17: first params `x, z, y`, docstring leads with the order. Keyword-only was not adopted (a signature break) | — |
| R34-12 | 18 "Swept and found sound" (`long_memory_d`) | New observation: LW's box pins `d̂ = 0.999999985` in up to 30 % of draws at d=0.9, so coverage is non-monotone in α | 0.2.0 | observation | N/A | Recorded as an observation with no action; the card discloses LW's narrowness | — |
| R34-13 | 18 "Reproducing" | "If a future round wants [the harnesses] durable, the place to put them is `docs/examples/coverage/`"; two more worth rebuilding (coverage checker, constant-diagnostic detector) | 0.2.0 | method | OPEN | Round 11's harnesses are committed (`lab/audit/round11/`) and the note-21 experiments live in `docs/examples/coverage/experiments/`, but the lens-1–3 switch/scale/degenerate sweeps and the constant-diagnostic detector exist nowhere in the tree — a round that re-runs them starts from the prose | medium / medium |

### 2.4 `20-audit-round-6-findings.md` (round 6, run at 0.3.0)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R6-01 | 20:25-49 | `bvar_hierarchical`'s ML-II default collapsed to the search floor; bands at the selection covered 6 % | 0.3.0 | trap (headline) | FIXED | Default `hyperprior="glp"` (BREAKING, 0.3.0; CHANGELOG:2065ff); P12: default is `'glp'` | — |
| R6-02 | 20:51-61 | The GLP plug-in's own calibration: 90 % bands cover 0.82–0.85 | 0.3.0 | trap → doc | FIXED | Card calibration paragraph + docstring (20:60-61); interval-coverage.md:1440-1446 points to it | — |
| R6-03 | 20:63-74 | `seasonal_strength` returned a float-noise ratio ≈ 0.64 on constant series | 0.3.0 | silent-wrong-answer (narrow) | FIXED | `ConstantSeries` refusal (CHANGELOG:2028-2033); P11 | — |
| R6-04 | 20:76-86 | `bvar_ssvs` blamed missing values for its own overflow | 0.3.0 | cosmetic | FIXED | CHANGELOG:2034-2039 | — |
| R6-05 | 20:88-92 | `har_rv`'s docstring halved its own breaking change | 0.3.0 | cosmetic | FIXED | CHANGELOG:2040-2042 | — |
| R6-06 | 20:94-99 | `zivot_andrews` documented `trim ∈ [0, 1/3]` but `trim=0` can never run | 0.3.0 | cosmetic | FIXED | CHANGELOG:2043-2045 | — |
| R6-07 | 20:101-105 | `proxy_ar_sets`' `kind` enumeration disagreed across surfaces | 0.3.0 | cosmetic | FIXED | CHANGELOG:2046-2048 | — |
| R6-08 | 20:109-122 | `proxy_ar_sets`' propagated coverage keeps declining through the default horizon (0.876–0.894 at h=12; 0.80–0.85 on a VAR(1)); "a re-engineered long-horizon correction (or a joint band) is future work" | 0.3.0 | trap | SUPERSEDED | Same lineage as R1-03: `second_order` (0.964 / 0.932 at h=12) and `second_order_bc` (0.982 / 0.966) shipped opt-in; measured every run by the registry (interval-coverage.md:1150-1169). The default-flip decision is N21-03 | — |
| R6-09 | 20:124-135 | The seasonal-strength rule saturates below ~4 cycles; "whether a minimum-cycles advisory … belongs in the `nsdiffs` output itself" is open | 0.3.0 | trap (doc-gap) | SUPERSEDED | Tracked as R1-06 (same item, recorded on the brief) | — |
| R6-10 | 20:139-142 | Residue: `bvar_ssvs["diagnostics"]` echo keys `burn`/`thin` not in the card's list | 0.3.0 | residue | FIXED | Card list updated the same round (20:141) | — |
| R6-11 | 20:143-148 | Residue: negative integer arguments raise raw `OverflowError` library-wide; "a candidate small-fix for a future round" | 0.3.0 | cosmetic | FIXED | Round 7 F3, 0.4.0: one central teaching `ValueError` at `_coerce._call` (22:121-139; CHANGELOG:1604-1607) | — |
| R6-12 | 20:184-193 | Reproducing: the generative designs worth rebuilding — the NIW prior-draw SBC harness, the `proxy_ar_sets` coverage harness, `postfix_hier.py` (scratchpad) | 0.3.0 | method | OPEN | The proxy-AR harness is committed (`docs/examples/coverage/experiments/proxy_ar_long_horizon.py`, 21:17); the NIW SBC harness that produced the round's headline and the post-fix collapse probe were never committed (grep `sbc` under `docs/examples/coverage/`, `lab/`: none) | medium / low |

### 2.5 `22-audit-round-7-findings.md` (round 7, post-0.3.0)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R7-F1 | 22:30-74 | `garch_fit` silent all-NaN SEs at an active boundary → reduced-Hessian SEs with `se_valid`/`boundary`/`boundary_note`/`converged` | 0.4.0 | silent-wrong-answer (round-1 backlog) | FIXED | CHANGELOG:1583-1589; P13: all five keys present | — |
| R7-F2 | 22:76-119 | `garch_fit` not scale-robust → standardize-and-map-back | 0.4.0 | silent-wrong-answer (round-1 backlog) | FIXED | CHANGELOG:1590-1598 | — |
| R7-F3 | 22:121-139 | Negative integer arguments → one central teaching `ValueError` | 0.4.0 | cosmetic | FIXED | CHANGELOG:1604-1607 | — |
| R7-C1 | 22:151-172 | `dcs_local_level(density="laplace")` converged to unit-dependent points | 0.4.0 | silent-wrong-answer | FIXED | Same standardize-and-map-back repair; CHANGELOG:1595-1598 | — |
| R7-C1b | 22:174-193 | Latent Nelder-Mead initial-simplex hole (near-zero coordinate below `x_tol`) | 0.4.0 | silent-wrong-answer (latent) | FIXED | CHANGELOG:1599-1603 (tracked as R1-13 too) | — |
| R7-C2 | 22:195-204 | `forecasting.md` described `gpd_fit` as unbuilt; a stale printed digit | 0.4.0 | cosmetic | FIXED | CHANGELOG:1607-1608 | — |
| R7-R | 22:206-224 | Refuted: `var_backtest` zero-violation "over-refusal"; `panel_lp` `"dj"` echo alias; two probe artefacts | 0.4.0 | refuted | N/A | Recorded with evidence | — |
| R7-P1 | 22:288-305 | Proposed CHANGELOG entries "not applied; CHANGELOG is off-limits this round" | 0.4.0 | process | FIXED | Applied in the 0.4.0 section (CHANGELOG:1581-1608) | — |
| R7-P2 | 22:307-315 | Proposed validation-matrix entries "not applied; matrix is off-limits" | 0.4.0 | process | FIXED | `validation-matrix.md:162` "GARCH boundary/scale conventions (round 7) — internal-property grade" carries them | — |

### 2.6 `23-audit-round-8-findings.md` (round 8, post-0.4.0)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R8-C1 | 23:35-59 | `theta_forecast`'s "Matches statsmodels ThetaModel" only at `use_test=False` | 0.4.0 | overclaim | FIXED | Fixed in-round with a regression test (23:54-59; CHANGELOG:1462ff) | — |
| R8-C2 | 23:61-83 | Copula invariance claimed for "any strictly monotone transform"; decreasing flips the fit | 0.4.0 | overclaim | FIXED | In-round, Rust + Python regression tests (23:77-83) | — |
| R8-C3 | 23:85-93 | `acm_term_premium` returns three undocumented echo keys | 0.4.0 | cosmetic | FIXED | In-round with a returned-keys ⊆ doc-words tripwire (23:91-93) | — |
| R8-R | 23:95-135 | Five refuted candidates (Frank MLE, residual theta mismatches, BNS power, DCS-t `converged`, ACM percent units) | 0.4.0 | refuted | N/A | Recorded with evidence | — |
| R8-res1 | 23:245-251 | Residue: the Rust `ThetaForecast` computes `alpha`, `b0`, `one_step`, `seasonal`; the binding returns the bare array — "exposing `alpha`/`b0` would be a reasonable future enhancement" | 0.4.0 | residue (enhancement) | OPEN | P18: `theta_forecast` still returns a bare `ndarray`; `lib.rs:2511` `r.forecast.into_pyarray(py)` | low / low |
| R8-res2 | 23:252-257 | Residue: t-copula ν̂ ≈ 424 with `se_valid=True` on near-Gaussian data; "nothing dishonest observed" | 0.4.0 | residue | N/A | Recorded, no action attached; the card's "ν at the barrier" failure mode covers it | — |
| R8-nm1 | 23:261-265 | Not made: `scale_ar=4` default vs the 0.3.0 wheel bit-identity (no 0.3.0 wheel in the environment) | 0.4.0 | comparison not made | N/A | Unverifiable here for the same reason; the default-vs-explicit half was made and is pinned | — |
| R8-nm2 | 23:266-269 | Not made: `ln_gamma_half_ratio` through the Rust binary | 0.4.0 | comparison not made | N/A | The committed Rust unit tests carry the in-binary assertions (23:150-151) | — |
| R8-nm3 | 23:270-273 | Not made: `lp_did` vs a live R/fixest run — "no R in this environment" | 0.4.0 | comparison not made | FIXED | Made since: `validation-matrix.md:161` "reference-run golden: an R/fixest execution of the authors' published example code, fixest compiled from the CRAN mirror"; `fixtures/generate_lpdid_fixtures.R` is the script. **Silently closed** — the round-8 doc still records it as not made (annotated, §6) | — |
| R8-nm4 | 23:274-277 | Not made: a full independent re-implementation of the ACM three-step pipeline | 0.4.0 | comparison not made | N/A | The committed generator *is* that re-implementation (read and verified non-circular) | — |
| R8-nm5 | 23:278-280 | Not made: GAS/DCS-t against an external reference — none exists | 0.4.0 | comparison not made | N/A | Still none (validation matrix, DCS row) | — |

### 2.7 `24-audit-round-9-findings.md` (round 9, field report + class sweeps, 0.6.0)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R9-F2 | 24:42-47 | Field item 2: `markov_switching_ar` never returned the AR coefficients | 0.5.0 | footgun | FIXED | 0.6.0 (CHANGELOG:41-53 of the 0.6.0 block); P06: `ar` returned | — |
| R9-F11 | 24:48-53 | Field item 11: `cv_splits(scheme="purged_kfold")` absorbed the embargo into the purge | 0.5.0 | footgun (BREAKING fix) | FIXED | 0.6.0 (CHANGELOG:789-803); P10: gap (21,10) − gap (21,0) = 10 | — |
| R9-F12 | 24:54-60 | Field item 12: `vecm` fit the no-deterministic case while `johansen` documents an unrestricted constant; restricted cases refused (documented follow-up) | 0.5.0 | correctness-trap | FIXED | `"n"`/`"co"` in 0.6.0 (CHANGELOG:1058-1068); the restricted cases and seasonal dummies in 0.7.0 (CHANGELOG:543-548); P42: `deterministic="ci"` runs | — |
| R9-F13 | 24:61-66 | Field item 13: `ivx` localizing sequence indexed by n instead of N = n−1 (fixture generator shared the misreading) | 0.5.0 | correctness | FIXED | 0.6.0 (CHANGELOG:957-972) | — |
| R9-S1 | 24:70-78 | Class "Rust-computed-but-never-bound": `dfm_nowcast` loadings, `bvar_fit` posterior uncertainty, `var_fit` residuals, `dcc_garch` stage-1 | 0.5.0 | footgun | FIXED | 0.6.0; P28: `loadings`, `omega_bar`, `resid` all returned | — |
| R9-S2 | 24:80-90 | Class "inert/absorbed/mis-indexed": `var_fevd` transposed (BREAKING), spectral `detrend` default (BREAKING), `garch_fit` silent `o` discard, `panel_fe`/`panel_lp` absorbed `bandwidth` | 0.5.0 | silent-noop / trap | FIXED | 0.6.0 (CHANGELOG:1195-1250); P27 `(7, 3, 3)` horizon-first; P58 `bandwidth` under cluster refused, `welch` `detrend='constant'`; P13 `o=1` refused | — |
| R9-S3 | 24:92-109 | Class "missing convergence/boundary signals": `panel_pmg` divergence, `arima_fit` dropped `converged` + boundary pileup, `quantile_lp`/`growth_at_risk` IRLS flag, `dfm_nowcast(method="mle")` certificates | 0.5.0 | silent-wrong-answer / trap | FIXED | 0.6.0 (CHANGELOG:1140-1194); P14: `arima_fit` carries `converged`/`boundary`/`se_valid`/`boundary_note`; P05: `growth_at_risk` carries `converged` | — |
| R9-S3b | 24:104-106; CHANGELOG:1177-1180 | `arima_fit` "tier-2 reduced-Hessian SEs recorded as a stated follow-up" — interior `bse` still come from the full-vector observed information at a boundary | 0.6.0 | tier-1 honesty (follow-up) | OPEN | `arima.md:160-164` still says "Reduced-Hessian standard errors over the free directions only (what `garch_fit` does) are a documented follow-up"; `crates/tsecon-arima/src/results.rs:408` repeats it; no `boundary_mask`/reduced-Hessian code in `tsecon-arima` (grep) | medium / medium |
| R9-S4 | 24:111-120 | Class "docs-vs-cited-paper": `cg_series` fixed-event vs fixed-horizon; `bns_jump_test` missing Huang-Tauchen factors | 0.5.0 | correctness | FIXED | 0.6.0 (CHANGELOG:1097-1139) | — |
| R9-R | 24:124-128 | Items 15, 16 upheld as documented contracts; `params_named` added anyway | 0.5.0 | by design | N/A | P13: `params_named` present | — |
| R9-T | 24:128-130 | The finders' unverified tails (4 per sweep) "listed in the sweep transcripts and remain open for a future round" | 0.5.0 | open tail | N/A | Lost with the container (25:12-17 records the loss); round 10 re-swept the same classes over new surface and round 11 committed its probes under `lab/audit/round11/` so it cannot recur | — |
| R9-A | 24:15-24 | Field items 5, 8, 9 "absent features"; ROADMAP:48 says "the three promoted features built" | 0.5.0 | feature request | FIXED / UNVERIFIED | Item 8 = `ou_fit`/`spread_zscore` (0.6.0, CHANGELOG 0.6.0 "field report item 8"); item 9 = callable forecasters (CHANGELOG:835). **Item 5's identity is not recoverable from the committed record** (the field report itself is not in the repo, and no CHANGELOG bullet is tagged "item 5"); ROADMAP:48 claims all three were built | — |
| R9-L1 | 24:134-136 | Lesson: "short-sample robustness batteries and cited-convention diffs are now standing checks in the brief" | 0.6.0 | process | OPEN | grep of `16-adversarial-audit-brief.md` for "short-sample", "cited-convention", "field probe": no hit — the brief was never updated with the round-9 standing checks | low / low |
| R9-L2 | 24:137-140 | Lesson: a golden's generator that transcribes the same paper needs a second reader | 0.6.0 | process | N/A | Recorded lesson; no artefact to verify | — |

### 2.8 `25-audit-round-10-findings.md` (round 10, 0.7.0)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R10-01 | 25:47-66 | `star_test` let a Rust panic escape (`capacity overflow` → `PanicException`) | 0.7.0 | severe | FIXED | CHANGELOG 0.7.0; P40: `delay=T+1` → catchable `ValueError` "insufficient data: 240 observations, at least 255 required" | — |
| R10-02 | 25:68-95 | Thirteen inert kwarg groups (mgarch `o`, conformal `order`/`n_eval`/`calib`/`batch`/`gamma`/…, `hamilton_filter` `maxlags`/`use_correction`, `bn_filter` `d0`/`dt`, `backtest` `period`, `spread_zscore` `dt`, `threshold_vecm` grid kwargs, `vecm` `first_season`) | 0.7.0 | silent-noop | FIXED | CHANGELOG:671-724; P44: `hamilton_filter(maxlags=…, se="nonrobust")` refuses | — |
| R10-03 | 25:97-130 | Sweep C: TVECM minimum-sample message, three leaked Rust internals, five message-honesty repairs, inf proxy values now refused family-wide, STAR flag constructibility proven | 0.7.0 | cosmetic / behavioural | FIXED | CHANGELOG:725-741 (inf proxies), 762-786 (docs); flags pinned by tests (25:123-130) | — |
| R10-04 | 25:132-143 | Sweep A: `ou_fit.level` / `markov_switching_ar.iterations` undocumented; one over-general card sentence; the k>2 contrast one-sided | 0.7.0 | cosmetic | FIXED | CHANGELOG:762-786 | — |
| R10-05 | 25:145-157 | Sweep D: 10/11 cross-surface items clean; the eleventh was the mgarch `o` finding | 0.7.0 | clean bill | N/A | Measured ledger, no action | — |
| R10-06 | 25:159-178 | Two integrator-inflicted merge corruptions (stub `SyntaxError`, doubled card roster); rule adopted | 0.7.0 | process | FIXED | Repaired in-round; rule restated in 26 "Integrator notes" | — |
| R10-07 | 25:180-191 | The statsmodels-0.15.0 canary fired → `hamilton_filter` gained a live cross-check | 0.7.0 | validation upgrade | FIXED | CHANGELOG:742-761 | — |
| R10-O1 | 25:203-205 | OPEN (low): conformal EnbPI default-base ergonomics — `base` omitted always refuses | 0.7.0 | low | OPEN | P22: `conformal_forecast(y, method="enbpi", horizon=1)` refuses with a teaching error naming its own AR ensemble | low / low |
| R10-O2 | 25:205 | OPEN (low): `bn_decomposition` p/q fixed-path no-op "now documented" | 0.7.0 | low | FIXED | P52: the docstring states `p`/`q` are ignored on the fixed path (`lib.rs:2144`) | — |
| R10-L2 | 25:200-206 | Lesson: record sweep tails in the findings doc, not transcripts | 0.7.0 | process | FIXED | Round 11 did exactly that (26 "OPEN — recorded, not fixed") | — |

### 2.9 `26-audit-round-11-findings.md` (round 11, 0.8.0)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| R11-M1 | 26 M1 | Five structural-identification `help()` texts carried none of their return contract | 0.8.0 | moderate | FIXED | CHANGELOG:389-393; P37: `robust_svar_bounds.__doc__` is 1299 chars | — |
| R11-M2 | 26 M2 | `gpd_fit`/`gev_fit` `help()` lacked the `Keys:` line | 0.8.0 | moderate | FIXED | CHANGELOG:393-394; P37: every `gpd_fit` key named | — |
| R11-M3 | 26 M3 | EGARCH multi-step forecasts refused with a leaked internal marker; no surface said so | 0.8.0 | moderate | FIXED | CHANGELOG:418-428; P41: horizon 2 refused, message clean | — |
| R11-M4 | 26 M4 | `seed=None` silently means seed 0 on three surfaces | 0.8.0 | moderate | FIXED | Documented + pinned (CHANGELOG:414-417); P43 | — |
| R11-M5 | 26 M5 | The forecasting card's `backtest` table taught the 0.7.0 `period` trap | 0.8.0 | moderate | FIXED | CHANGELOG:429-433 | — |
| R11-M6 | 26 M6 | Twenty-two runtime docstrings missing keys the stub named | 0.8.0 | moderate (group) | FIXED | CHANGELOG:395-402; gated by the M1 test | — |
| R11-L1 | 26 L1 | `max_rel_change` documented on no surface | 0.8.0 | low | FIXED | P36; CHANGELOG:403-406. The ML *card* half is R11-O3 | — |
| R11-L2–L8 | 26 L2-L8 | `local_level_smooth` keys; HP/BK/CF filter keys; `engle_granger`/`factor_model`/`gas_volatility`/`zero_sign_svar` keys; `cg_regression` phantom keys; panel card key lists; two shape claims; `cv_splits(n)` default always raises | 0.8.0 | low | FIXED | CHANGELOG:407-442; P37: `cg_regression` keys all named | — |
| R11-E | 26 "Sweep E — the rest" | `summarize`/JSON/pickle 162/162; 30 phantom-key candidates refuted; 45 shape checks | 0.8.0 | clean bill | N/A | Measured ledger | — |
| R11-F | 26 "Sweep F — the rest" | Signature-vs-stub 162/162; 12 prose-default candidates refuted; card default rows | 0.8.0 | clean bill | N/A | Measured ledger | — |
| R11-G-ref | 26 "Flags refuted by analysis" | `adf`/`check_stationarity` T^1.75, `cf_filter` O(T²), `cv_splits` O(T²) output, `bai_perron` O(n²) documented | 0.8.0 | refuted | N/A | Algorithmic by construction | — |
| R11-H | 26 "Sweep H" | Seed contract clean on 21/21 parameters, restart-stable | 0.8.0 | clean bill | N/A | — | — |
| R11-O1 | 26 OPEN 1 | `inspect.signature` renders eight defaults as `Ellipsis` (PyO3 `__text_signature__` limits) | 0.8.0 | low | OPEN | P20: all eight still `Ellipsis` (`adf`/`zivot_andrews`/`engle_granger.autolag`, `box_cox_lambda.bounds`, `historical_decomposition.restrictions`, `narrative_svar.sign_restrictions`, `predictive_regression`/`ivx_test.cz`) | low / medium |
| R11-O2 | 26 OPEN 2 | `check_stationarity` returns `adf_statistic`/`adf_p_value`/`kpss_statistic`/`kpss_p_value`/`alpha` named on no surface | 0.8.0 | low | OPEN | P21: those five keys are absent from `check_stationarity.__doc__` | low / low |
| R11-O3 | 26 OPEN 3 | The ML card's `adaptive_lasso` key list omits `max_rel_change` (card owned by the ML wave) | 0.8.0 | low | OPEN | `machine-learning.md:112-113` still lists `{"coef", "n_iter", "max_change"}`; the ML wave (0.8.0) shipped without touching it | low / low |
| R11-O4 | 26 OPEN 4 | No model card covers `hp_filter`/`bk_filter`/`cf_filter` | 0.8.0 | low | OPEN | grep of `docs/reference/model-cards/*.md` for `hp_filter`: only `ml-convex.md` (as the `l1_trend_filter` comparison); no filter card | low / medium |
| R11-O5 | 26 OPEN 5 | The ARIMA exact-MLE engine's constant (~2 ms/obs; `arima_fit` 6 s, `auto_arima` 62 s at T=3200) | 0.8.0 | performance | OPEN | P55: `arima_fit` 3.56 s at T=3200 on this machine (linear; still ~1 ms/obs) | medium / high |
| R11-O6 | 26 OPEN 6 | The t-copula MLE's constant (26 s at n=3200; `copula_select`'s default menu 130× its second-slowest member) | 0.8.0 | performance | OPEN | Not re-timed here (the round's table stands); no change to `tsecon-copula` since 0.8.0 | low / medium |
| R11-O7 | 26 OPEN 7 | `historical_decomposition` is O(T²) (cumulated contributions summed over all past shocks) | 0.8.0 | performance | OPEN | P55: 0.49 s at T=3200 (round 11: 0.43 s) — a companion-form recursion would make it O(T) | low / low |
| R11-O8 | 26 OPEN 8 | `mcmc_diagnostics` scales as T^1.7 (autocorrelation without an FFT) | 0.8.0 | performance | OPEN | P55: 0.057 s at 3200 draws (round 11: 0.09 s) — an FFT autocorrelation would make it T log T | low / low |
| R11-L-1 | 26 "Lessons" 1 | "The durable fix is a generator that emits both [docstring and stub] from one source" | 0.8.0 | tooling | OPEN | No such generator exists (`bindings/python/python/tsecon/__init__.pyi` is hand-written; the stub-sync test only regexes `def` names) | medium / medium |
| R11-L-3 | 26 "Lessons" 3 | The sweep-H harness should carry the NumPy `None` = entropy expectation as its null | 0.8.0 | tooling | OPEN | `lab/audit/round11/sweep_h_seed.py` was not revised after the round (its verdict logic still treats determinism under `None` as pass) | low / low |
| R11-INT | 26 "Integrator notes" | Four merge-hygiene rules (keep-both interleaving, shared enums, hunks through `match`, `commit-tree` signing) | 0.8.0 | process | N/A | Rules recorded; nothing to verify on main | — |

### 2.10 `21-long-horizon-and-joint-inference.md` (the two re-engineering follow-ups)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| N21-01 | 21:84-98 | `second_order` shipped; "honest residual … 0.932 at h=12, not 0.95"; "a combination arm (second_order at bias-corrected coefficients) is the natural next candidate" | 0.3.0 | engineering follow-up | FIXED | The 2026-08-23 follow-up shipped `rf_method="second_order_bc"` (21:252-312; CHANGELOG:1258-1270); measured every run (interval-coverage.md:1150-1169); P04 | — |
| N21-02 | 21:197-207 | `joint="bonferroni"` shipped; "the chi-square default stays for small k or ρ safely below 1" | 0.3.0 | engineering follow-up | FIXED | Default flipped to `"bonferroni"` in 0.5.0 on the library's own measurements (CHANGELOG:1450-1460); P03 | — |
| N21-03 | 21:234-236 | "Default flips (making `second_order` and/or `bonferroni`-at-large-k the defaults) are deliberate future decisions for an audit round" | 0.3.0 | decision | OPEN (half) | The IVX half was decided (N21-02). The `proxy_ar_sets` half was not: P04 `default rf_method='delta'`; the 0.6.0 note says "the default remains `"delta"`" (CHANGELOG:1268-1269). The evidence for a flip is already committed (three arms measured on 1000 reps, `second_order` at-or-near nominal, `second_order_bc` a conservative floor at ~2× width) | medium / low |
| N21-04 | 21:99-111, 208-216 | Discarded with evidence: boot-c, floor, bcvar, boot-v (Problem A); demeaned variance, FM normaliser, wild bootstrap, the alpha ladder (Problem B) | 0.3.0 | recorded dead ends | N/A | Recorded so nobody re-runs them on faith | — |
| N21-05 | 21:217-219 | The suite now pins the chi-square defect itself (k=5 size regression ≥ 0.10) | 0.3.0 | test | FIXED | Stated as shipped; `crates/tsecon-predreg` property suite | — |

### 2.11 `15-proxy-svar-bands.md` (design spec)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| S15-00 | 15:3-8 | "**Status: specification only. Not yet implemented.** … it is not a claim that the library implements any of it yet" | pre-0.3.0 | status banner | FIXED | Both surfaces shipped in 0.3.0 (`proxy_svar_bands` moving-block/wild; `proxy_ar_sets` — CHANGELOG:2190-2270); P02. **Silently closed** — the banner was never updated (corrected, §6) | — |
| S15-U1 | 15:10-13, 47, 153 | Citations reconstructed from memory; "check them against the papers before any of this reaches user-facing documentation or `CITATION.cff`" (JL 2022 JBES volume/pages; BJT 2016 title; MOSW 2021 volume/pages) | pre-0.3.0 | verification | OPEN | The SI card cites JL 2019 / MR 2013 / GK 2015 / SW 2018 / Hall 1992 and "Montiel Olea, Stock & Watson (2021)" (:310, :588-589) without volume/page detail; `CITATION.cff` carries no references block (grep); no document records that the flagged details were checked against the primary sources | low / low |
| S15-U2 | 15:59, 123 | Block-length rule `ell = round(5.03 · T^{1/4})` — constant, exponent and rounding "moderate confidence … verify before hardcoding the default" | pre-0.3.0 | verification | UNVERIFIED | P49: the shipped default at T_eff = 238 is `block_length = 20`, which equals `round(5.03 · 238^{1/4}) = round(19.76)` — so the rule *as reconstructed* is what ships; whether it matches Jentsch-Lunsford's paper was never recorded as checked (the card :526 says only "`block_length=None` picks a default from T") | — |
| S15-U3 | 15:79, 125 | Hall (basic) vs Efron percentile recommendation uncertain | pre-0.3.0 | verification | SUPERSEDED | Both are returned (`lower`/`upper` Hall, `lower_efron`/`upper_efron`; P49 keys) and the question was settled by measurement, not citation: on the card DGP Efron beats Hall at h=12 (0.885 vs 0.787 pooled; interval-coverage.md:1128-1140; card :564-567) | — |
| S15-U4 | 15:65 | JL's deterministic proxy scale correction — "exact algebraic form" unknown; diagnostics-only refinement "to be pinned from JL's replication code" | pre-0.3.0 | refinement | OPEN | grep `rescal` / `scale adjust` in `crates/tsecon-ident/src/proxy.rs`: none — the diagnostics-only correction was never pinned; IRF bands are provably unaffected (the scalar cancels in `rho*`) | low / low |
| S15-U5 | 15:85 | Multi-instrument (k > 1) note: "the structure generalizes … JL's exact treatment of the k > 1 normalization is flagged uncertain" | pre-0.3.0 | scope | OPEN | P49: a two-column proxy is refused as a shape error — the bands are single-proxy only; `n_proxy` is echoed | low / medium |
| S15-U6 | 15:229-233 | Which variance MOSW use (null-imposed with Ψ fixed / joint delta / bootstrap) is "the single biggest attribution gap" | pre-0.3.0 | attribution | OPEN | The card attributes the sets to "Montiel Olea, Stock & Watson (2021, weak-IV-robust bands)" (:310) and documents the implemented variance (`variance="hc0"`/`"hac"`, the reduced-form propagation) — but nothing records that the attribution gap was closed by reading the paper | low / low |
| S15-U7 | 15:129 | Horizon profile of the wild bootstrap's under-coverage: "I will not claim a universal direction — measure it in the MC and report what you find" | pre-0.3.0 | measurement | FIXED | Measured in 0.6.0: the wild arm collapses to 0.19–0.24 at impact and looks almost reasonable at h ≥ 1 (interval-coverage.md:1128-1145) | — |
| S15-V | 15:24-40 | Four load-bearing claims "verified numerically before implementation" | pre-0.3.0 | verified | N/A | Pre-implementation checks; the shipped tests carry the claims (CHANGELOG:2255-2270) | — |

### 2.12 `19-research-contributions.md` (research scan, 2026-08-17)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| RC19-G | 19:12 | Grounding note: "EVT and conformal (ACI/EnbPI) are on the roadmap but unshipped"; JOSS 6-month gate not met until ~mid-January 2027 | 0.2.0 | grounding | FIXED (half) / OPEN (half) | EVT shipped 0.3.0 (`gpd_fit`/`gev_fit`), conformal 0.5.0 — **silently closed**, annotated (§6). The JOSS clock is time-gated and still running (public history began 2026-07-17) | — |
| RC19-01 | 19:20-24 | #1 Split-panel-jackknife panel LP (S effort) | 0.2.0 | S | FIXED | 0.3.0 `panel_lp(bias_correction="spj")` (CHANGELOG:1725-1741); P57 | — |
| RC19-02 | 19:26-30 | #2 LP-DiD core (M); doubly-robust variant speculative, "sequence it second" | 0.2.0 | M | FIXED / N/A | Core shipped 0.4.0 (`lp_did`, R/fixest golden). The DR (AIPW) variant is declared out of scope until a reference exists (panel.md:462) — N/A by policy | — |
| RC19-03a | 19:32-36 | #3 Joint-across-horizon LP inference: sup-t bands | 0.2.0 | M | FIXED | 0.3.0 sup-t for `lp`/`smooth_lp` (+ VAR) (CHANGELOG:2281ff); the closed forms for the other LP entry points; the sup-t gap on those is R1-02 | — |
| RC19-03b | 19:32-36 | #3 Jordà significance bands | 0.2.0 | M | OPEN | grep `significance band` in `docs/`, `crates/tsecon-lp/src`: only the research doc itself | low / medium |
| RC19-03c | 19:32-36 | #3 Wild block bootstrap for LP / LP-IV bands | 0.2.0 | M | OPEN | `lp`/`lp_iv` signatures carry no bootstrap route (`band`, `band_alpha`, `band_seed`, `band_n_sim` only — sup-t is a Gaussian simulation on the HAC covariance) | medium / medium |
| RC19-04a | 19:38-42 | #4a Weak-proxy robust inference: ACF moving-block pre-test, AR-type / MSW confidence sets, Lewis-Mertens generalized first-stage test | 0.2.0 | M | FIXED (2 of 3) / OPEN (1) | `proxy_ar_sets` (0.3.0) and `proxy_first_stage` with the Montiel Olea-Pflueger effective F (0.4.0) shipped; the Lewis-Mertens generalized first-stage test and the Angelini-Cavaliere-Fanelli pre-test did not (card :439 cites LM only as context; grep of the stub: none) | low / medium |
| RC19-04b | 19:38-42 | #4b Gertler-Karadi (2015) replication | 0.2.0 | M | FIXED | 0.4.0 `docs/examples/replication-gertler-karadi.md` ("first stage verbatim", ROADMAP:44) | — |
| RC19-04c | 19:38-42, 99 | #4c Doko Tchatoka-Haque post-1984 split across the proxy zoo; JCRE submission by end Q1 2027 | 0.2.0 | M (paper) | OPEN | No post-1984 arm in the GK replication; no paper draft beyond `paper/paper.md` (JOSS) | medium / high |
| RC19-05 | 19:44-48 | #5 Conformal module: split-CP, ACI, decaying-step online CP, EnbPI, SPCI, multi-horizon | 0.2.0 | M | FIXED (core) / OPEN (frontier) | split / EnbPI / ACI shipped 0.5.0 (CHANGELOG:1301-1336); decaying-step-size ACI, SPCI and multi-horizon variants did not — the lab's exp06 frontier list names SPCI, conformal PID and quantile-conformal-on-GARCH-residuals as next (LAB-08) | — |
| RC19-06 | 19:50-54 | #6 Independent Monte-Carlo verification of the LP-vs-VAR primer (Montiel Olea et al. 2025) | 0.2.0 | S/M | OPEN | `docs/examples/monte-carlo-frontier.md` runs an LP-vs-VAR bias/variance experiment on the library's own DGPs; the primer's DGPs, horizons and bootstrap variants are not reproduced (grep "Plagborg" in `docs/examples/`: none) | medium / medium |
| RC19-07 | 19:56-60 | #7 Fast "soft" sign-restriction posterior sampling (Read-Zhu) | 0.2.0 | M/L | OPEN | `sign_restricted_svar` is accept-reject (card); no penalty-MCMC sampler in `tsecon-ident` | medium / high |
| RC19-08 | 19:62-66 | #8 GaR robustness horse-race (quantile GaR vs EVT tails vs calibrated quantiles) | 0.2.0 | M | FIXED (EVT) / OPEN (horse race) | EVT shipped 0.3.0; no horse-race study in `docs/examples/` or `lab/` (grep "Adrian" + "gpd" across examples: none together) | medium / medium |
| RC19-09 | 19:68-72 | #9 Bayesian quantile VAR via multivariate asymmetric-Laplace likelihood | 0.2.0 | M/L | OPEN | No QVAR surface (grep `qvar` / `quantile_var`: none) | low / high |
| RC19-10 | 19:74-78 | #10 Climate-damages reconciliation (Bilal-Känzig vs Nath-Ramey-Klenow) | 0.2.0 | L | OPEN | Depends on #1/#3 (shipped) and on the distributed-lag climate regressions ROADMAP:54 lists as build-later (RM-05) | low / high |
| RC19-HM | 19:80-86 | Honorable mentions: GSULP joint-LP, generalised-Bayes robust Kalman update, distribution-generic GAS engine, HD-LP desparsified lasso, NY Fed Nowcast 2.0, TSFM contamination harness | 0.2.0 | tracked, not ranked | OPEN | None shipped; the desparsified lasso and the contamination harness recur as ML10 Tier 3/4 rows (ML10-11, ML10-10); the GAS engine has a partial base in `gas_volatility`/`dcs_local_level` | low / high |
| RC19-Q4A | 19:92-93 | Next-quarter A: "LP inference hardening" — SPJ + sup-t + MC coverage audit + spot-verify IJK / JG papers | 0.2.0 | plan | FIXED (3 of 4) / OPEN (1) | SPJ, sup-t and the coverage audit shipped 0.3.0; the spot-verification of the Inoue-Jordà-Kuersteiner and Jordà-Gadea papers against their primary sources is recorded nowhere | — |
| RC19-Q4B | 19:95-96 | Next-quarter B: LP-DiD core with Stata/R goldens, card, gallery example | 0.2.0 | plan | FIXED | 0.4.0 | — |
| RC19-Q4C | 19:98-99 | Next-quarter C: proxy-SVAR workstream — MSW sets + first-stage diagnostics, GK gallery doc, post-1984 split, JCRE draft | 0.2.0 | plan | FIXED (2 of 4) / OPEN (2) | Sets and diagnostics (0.3.0/0.4.0) and the GK doc (0.4.0) shipped; the post-1984 split and the JCRE draft are RC19-04c | — |
| RC19-V | 19:105-120 | Venue map: JOSS Q1–Q2 2027 (gates: 6 months public history + research-use evidence); SoftwareX / JSS / JCRE / I4R / SciPy timing | 0.2.0 | plan | OPEN | Time-gated; nothing submitted (no `paper/` artefact beyond the JOSS draft) | medium / low |
| RC19-cite | 19:7-8, 120 | "Every citation below … must be re-verified against the primary source before any submission is drafted" | 0.2.0 | verification | OPEN | No record of a verification pass over the scan's citations | low / low |

### 2.13 `ROADMAP.md` §0 (build-later / deferred / on deck / named follow-ups)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| RM-01 | ROADMAP:21 | "Built in Rust, awaiting Python bindings: the `smooth_fixed` fixed-parameter DFM state-space entry point" | 0.8.0 | binding gap | OPEN | P26: `hasattr(tsecon, "smooth_fixed") = False` | low / low |
| RM-02 | ROADMAP:54 | Build-later: Hansen (1997/2000) threshold confidence sets for SETAR | 0.8.0 | build-later | OPEN | grep `crates/tsecon-regime/src/setar.rs` + stub for threshold CI: none | medium / medium |
| RM-03 | ROADMAP:54; cointegration-regime.md:936-943; CHANGELOG:655-659 | Build-later: regime-dependent generalized impulse responses (Koop-Pesaran-Potter) for the shipped threshold VAR | 0.8.0 | build-later | OPEN | P48: no `girf`-like callable; card declares GIRFs deferred | medium / high |
| RM-04 | ROADMAP:54; 12-extensions.md:36-41 | Build-later: JSZ canonical affine term structure | 0.8.0 | build-later (E2) | OPEN | grep `jsz` / `joslin` in the stub: none | low / high |
| RM-05 | ROADMAP:54 | Build-later: distributed-lag climate-impact regressions | 0.8.0 | build-later | OPEN | grep `climate` in the stub: none (E11 "climate econometrics (docs deliverable)" also deferred at ROADMAP:250) | low / medium |
| RM-06 | ROADMAP:56 | Deferred: HEGY seasonal unit roots (R-only, location-dependent tables) | 0.8.0 | deferred | OPEN | Not in the stub; R `uroot` reference | low / medium |
| RM-07 | ROADMAP:56; unit-root-cointegration-tests.md:417 | Deferred: Lee-Strazicich two-break minimum-LM test (the card flags it "for honesty") | 0.8.0 | deferred | OPEN | Not in the stub | low / medium |
| RM-08 | ROADMAP:56, 100, 260 | Deferred: X-13ARIMA-SEATS wrapper (needs the external Census binary; reimplementing is a non-goal) | 0.8.0 | deferred | OPEN | No wrapper; the no-network/no-binary boundary makes this a companion-package question | low / high |
| RM-09 | ROADMAP:56; copulas.md:59-61 | Deferred: dynamic / time-varying copulas (and `d > 2`, rotated variants) | 0.8.0 | deferred | OPEN | P47: `d > 2` refused as "bivariate in this slice" | low / high |
| RM-10 | ROADMAP:56 | Deferred: cointegrated energy-climate systems | 0.8.0 | deferred | OPEN | Not built | low / high |
| RM-11 | ROADMAP:58, 227; 11-docs-ux-adoption.md:159 | On deck: head-to-head explainer chapters (LP vs VAR, BVAR priors compared, the GARCH zoo, identification schemes compared) | 0.8.0 | adoption polish | OPEN | grep `explainer` / `head-to-head` in `mkdocs.yml`: none; `docs/guide/` has the 15 teaching chapters, no comparative essays | medium / medium |
| RM-12 | ROADMAP:58, 232, 275 | On deck: a public speed dashboard (v1 metric: "benchmark dashboard showing order-of-magnitude Monte Carlo speedups") | 0.8.0 | adoption polish | OPEN | `benchmarks/` holds the parity-first harness (`bench.py`), no published dashboard (grep `dashboard` in `benchmarks/README.md`, `README.md`: none) | medium / medium |
| RM-13 | ROADMAP:58, 200, 275 | On deck: more published replications (v1 gate ≥ 15) | 0.8.0 | adoption polish | OPEN | `docs/examples/replication-*.md` = 8 | medium / high |
| RM-14 | ROADMAP:52, 54; validation-matrix.md:147-149 | Named follow-up: the tsDyn reference-run for `threshold_vecm`/`hansen_seo_test`/`threshold_var`/`star` ("CRAN unreachable from the build container") | 0.7.0 | validation upgrade | OPEN | Still transcription + MC grade on all three rows. Note R 4.3.3 *was* runnable for two other fixtures (`generate_bn_filter_fixtures.R`, `generate_lpdid_fixtures.R`, fixest "compiled from the CRAN mirror", matrix :161, :177, :209) — the block is a per-container egress limit, not a repository one | medium / medium |
| RM-15 | 14-packaging-distribution.md:12; ROADMAP:203, 269 | conda-forge feedstock ("Still open") | 0.8.0 | packaging | OPEN | No feedstock recipe in the tree; no CHANGELOG mention | medium / medium |
| RM-16 | ROADMAP:200 | v1.0 gate: API freeze + deprecation policy, model cards for every core estimator, guide complete, replication gallery ≥ 15, public benchmark dashboard, tiering published, conda-forge live | 0.8.0 | milestone | OPEN | Model cards and the guide are in place; the other five are RM-12/13/15 plus the freeze/deprecation policy (no `DEPRECATION.md` / policy section in `GOVERNANCE.md` — grep `deprecat`: not found as a policy) | — |
| RM-17 | ROADMAP:50-51 | "Build-next — high value, runnable golden: the list is empty" | 0.8.0 | statement | N/A | Statement of fact; the build-later list is RM-02..05 | — |
| RM-18 | ROADMAP:48-49 (0.6.0) | The field report: "three promoted to feature requests" and built | 0.6.0 | claim | UNVERIFIED | See R9-A — item 5's identity is not in the record | — |
| RM-19 | ROADMAP:46 (0.8.0) | Machine-learning wave: "Module 10's Tier 2 in one release" — Tier 3/4 are what remains | 0.8.0 | tiering | OPEN | The Tier 3/4 rows are ML10-01..16 | — |

### 2.14 `CHANGELOG.md` 0.2.0–0.8.0 (items not already carried by a findings doc)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| CL-01 | CHANGELOG:2510-2513 (0.2.0), 2274-2280 (0.3.0) | "Not in this release": SARIMA seasonal orders `(P, D, Q, s)` | 0.2.0 | absent feature | FIXED | 0.3.0 "Added — seasonal ARIMA" (CHANGELOG:1881-1906); P51: `arima_fit(seasonal=…)` | — |
| CL-02 | CHANGELOG:2511 (0.2.0), interval-coverage.md:1418-1424, 1440-1445 | Anderson-Rubin / weak-IV-robust confidence sets for `iv_gmm` (and `lp_iv`); the coverage page's rec. #4 "half done" | 0.2.0 | absent feature | OPEN | P31: no AR/CD/KP key or callable for the IV regression estimators (`proxy_ar_sets` exists for the proxy-SVAR estimand only); the coverage page still says "no bounded Wald set can be honest" under weak identification | medium / medium |
| CL-03 | CHANGELOG:2512, 2277 | Angrist-Pischke, Cragg-Donald, Kleibergen-Paap statistics | 0.2.0 | absent feature | OPEN | P31: none in `dir(tsecon)`; `iv_gmm` returns `first_stage_f` only | low / low |
| CL-04 | CHANGELOG:2274-2277 (0.3.0) | "Simultaneous (joint) bands — both new surfaces are pointwise … every other band in the library is pointwise too" | 0.3.0 | absent feature | SUPERSEDED | sup-t shipped the same release for `var_irf_bands`/`var_forecast`/`lp`/`smooth_lp` (CHANGELOG:2281ff); the remaining gap (closed forms only on the other LP entry points; `proxy_svar_bands`/`proxy_ar_sets` pointwise) is tracked as R1-02 | — |
| CL-05 | CHANGELOG:2278-2280; structural-identification.md:23-25 | Bootstrap bands for the other point-identification schemes (`long_run_svar`, `max_share_svar`, `hetero_svar`, `nongaussian_svar` "remain point-only") | 0.3.0 | absent feature | OPEN | P29: none of the four takes a band/bootstrap kwarg; the card still says "their bands remain an open item" | medium / medium |
| CL-06 | CHANGELOG:1556-1560 (0.4.0); copulas.md:59-61, 95 | Copulas: `d > 2`, rotated/survival variants, dynamic copulas "deferred and stated"; tau-fit SEs "deferred, not faked" | 0.4.0 | deferred | OPEN | P47 (`d > 2` refused); tau-method fits return `se_valid=False` NaN SEs (card :95) | low / medium |
| CL-07 | CHANGELOG:1177-1180 (0.6.0) | `arima_fit`: "reduced-Hessian interior SEs over the free directions are a stated follow-up" | 0.6.0 | follow-up | SUPERSEDED | Tracked as R9-S3b | — |
| CL-08 | CHANGELOG:1062-1064 (0.6.0) | `vecm` restricted cases "refused with an error naming what is supported (a documented follow-up in ROADMAP build-later)" | 0.6.0 | follow-up | FIXED | 0.7.0 (CHANGELOG:543-548); P42 | — |
| CL-09 | CHANGELOG:1345-1349 (0.5.0); panel.md:255-261 | `panel_lp` cross-horizon influence-function covariance under entity clustering / Driscoll-Kraay — "a documented follow-up in `tsecon-panel`" (so `band="sup-t"` is refused) | 0.5.0 | follow-up | OPEN | P24: `panel_lp(band="sup-t")` refuses; grep `cross-horizon` in `crates/tsecon-panel/src`: none built. Same shape as R1-02 (the panel half) | medium / high |
| CL-10 | CHANGELOG:655-659 (0.7.0) | TVAR "regime-dependent generalized impulse responses are named as deferred in the model card rather than shipped half-right" | 0.7.0 | deferred | SUPERSEDED | Tracked as RM-03 | — |
| CL-11 | CHANGELOG:146-150 (0.8.0); validation-matrix.md:113 | `pds_lasso` coverage is Monte-Carlo grade: "R `hdm` and Stata `pdslasso` are not runnable in the reference environment" | 0.8.0 | validation grade | OPEN | Still MC grade; `hdm` is on CRAN (the same egress limit as RM-14) | low / medium |
| CL-12 | CHANGELOG:290-294 (0.8.0); ml-convex.md:225-231; validation-matrix.md:117 | `boosting`: "R mboost is not runnable in the build environment … cross-checking against it is an open follow-up" | 0.8.0 | validation grade | OPEN | Transcription grade at 1e-12; `mboost::glmboost` is the named target | low / medium |
| CL-13 | CHANGELOG:418-428 (0.8.0) | EGARCH multi-step variance forecasts: "no closed-form multi-step EGARCH forecast exists; the simulation route is not shipped" | 0.8.0 | absent feature (documented) | OPEN | P41: `forecast_horizon=2` refused cleanly with the two remedies | low / medium |
| CL-14 | CHANGELOG:1258-1270 (0.6.0) | `second_order_bc` — "the roadmap-note-21 follow-up" | 0.6.0 | follow-up | FIXED | Tracked as N21-01 (closed) | — |
| CL-15 | CHANGELOG:1273-1298 (0.6.0) | The interval-coverage audit now measures the five previously unmeasured families; the registry's stale pre-0.5.0 HAR windows caught and fixed | 0.6.0 | measured | FIXED | Tracked as R1-07 (closed) | — |
| CL-16 | CHANGELOG (0.5.0 `auto_arima`) ; ROADMAP:45 | `auto_arima`: "parity with R/`pmdarima` was deliberately NOT the target, since those two disagree with each other" | 0.5.0 | by design | N/A | Documented grade (MC-recovery with candidate-level statsmodels pins) | — |
| CL-17 | CHANGELOG (0.5.0 `ng_perron`); validation-matrix.md:142 | `ng_perron`: no runnable implementation exists in statsmodels, arch, or CRAN; a canary pins the absence | 0.5.0 | by design | N/A | "NOT a reference-run golden, because none can exist" — the canary will fire if one appears | — |
| CL-18 | CHANGELOG:1000-1006 (0.6.0) | `garch_fit` `params_named`; "the results facade's `GARCHResults.params_named()` … return flat named scalars and deliberately gain no such key" | 0.6.0 | by design | N/A | P13 | — |
| CL-19 | CHANGELOG 0.8.0 (`kernel_regression`); ml-kernel.md:84-90 | Compact-support kernels deferred because statsmodels' `tricube` gives out-of-support points full weight ("a bug in the reference") | 0.8.0 | deferred (reference defect) | N/A | Nothing honest to pin to; Gaussian kernel only, documented | — |
| CL-20 | CHANGELOG:1416-1447 (0.5.0) | "Fixed — reported from the field": `garch_fit(dist="t")` short samples, `har_rv` windows, MS-AR transition orientation, GARCH filter timing | 0.5.0 | field | FIXED | Round-9 stage-1 "6 already fixed" items | — |

### 2.15 `docs/reference/validation-matrix.md` (follow-ups, not-runnable references, property-graded rows)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| VM-01 | validation-matrix.md:106 | Conformal EnbPI/ACI: "`mapie` 1.5.0 shipped `regression.TimeSeriesRegressor` whose `method=` takes exactly `"enbpi"` and `"aci"` — a reference now exists, and cross-checking against it is a named follow-up … the grade stays property-MC until that run happens" | 0.5.0 | named follow-up | OPEN | P45: `mapie 1.5.0` with `TimeSeriesRegressor(estimator, method='enbpi', cv, …)` is importable in the audit venv — the run is feasible today and still not made | high / low |
| VM-02 | validation-matrix.md:87 | `lp_state` is not golden-pinned (property tests only) | 0.2.0 | property-graded | OPEN | An interacted OLS per horizon is reproducible in statsmodels; no fixture block exists | low / low |
| VM-03 | validation-matrix.md:107 | `backtest` is not golden-pinned ("an evaluation engine, not an estimand") | 0.2.0 | by design | N/A | Alignment/guardrail tests + the 0.6.0 bitwise string-path snapshot carry it | — |
| VM-04 | validation-matrix.md:108 | `adaptive_lasso`, `lasso_path` are not golden-pinned | 0.2.0 | property-graded | OPEN | scikit-learn's `lasso_path` is a runnable reference for the path (the ML wave pinned `alpha_y`/`alpha_d` against it for `pds_lasso`, :113) — the gap is a fixture, not a reference | low / low |
| VM-05 | validation-matrix.md:109 | `kernel_ridge(rff_features=…)` random-Fourier mode property-graded | 0.8.0 | by design | N/A | A seeded Monte-Carlo object with no reference value | — |
| VM-06 | validation-matrix.md:113 | `pds_lasso` coverage leg MC grade (R `hdm` / Stata `pdslasso` not runnable) | 0.8.0 | validation grade | SUPERSEDED | Tracked as CL-11 | — |
| VM-07 | validation-matrix.md:117 | `boosting` vs R `mboost::glmboost` — "an open follow-up, stated on the card" | 0.8.0 | validation grade | SUPERSEDED | Tracked as CL-12 | — |
| VM-08 | validation-matrix.md:119 | `echo_state_network` estimator property-graded (mechanics third-party-confirmed by `reservoirpy`) | 0.8.0 | by design | N/A | No reference estimand for the public estimator | — |
| VM-09 | validation-matrix.md:120 | `panel_lp` is not golden-pinned (known-IRF recovery inside 4-σ bands) | 0.2.0 | property-graded | OPEN | R `lpirfs` / the pLP repository are references at the R-egress limit; the SPJ leg is transcription-pinned (:153) | low / medium |
| VM-10 | validation-matrix.md:121 | `mean_group_var` is not golden-pinned (simulation recovery) | 0.2.0 | property-graded | OPEN | No package reference named; a per-entity `VAR` mean in statsmodels is buildable | low / low |
| VM-11 | validation-matrix.md:132 | `recession_probit` dynamic (Kauppi-Saikkonen) property-only | 0.2.0 | by design | N/A | No reference implementation exists | — |
| VM-12 | validation-matrix.md:142 | `ng_perron` — "NOT a reference-run golden, because none can exist" | 0.5.0 | by design | N/A | Tracked as CL-17 | — |
| VM-13 | validation-matrix.md:147-149 | STAR / TVECM / TVAR — "no third-party reference was runnable" (tsDyn; CRAN unreachable) | 0.7.0 | validation grade | SUPERSEDED | Tracked as RM-14 | — |
| VM-14 | validation-matrix.md:153 | SPJ panel LP — "the R-only reference commits no numeric outputs, so transcription + seeded MC — stated" | 0.3.0 | validation grade | OPEN | `panelLP.R` was fetched verbatim and transcribed; running it under the R 4.3.3 that generated `bn_filters.json`/`lpdid.json` would upgrade the row to reference-run | medium / low |
| VM-15 | validation-matrix.md:161, 177, 209 | R 4.3.3 reference runs exist for `lp_did` (fixest from the CRAN mirror) and `bn_filter` (the authors' R code) | 0.4.0 / 0.6.0 | evidence | N/A | Recorded here because it bounds the "R/CRAN blocked" theme: the block is per-container egress, and two fixtures prove the path exists | — |
| VM-16 | validation-matrix.md:162 | GARCH boundary/scale conventions — "internal-property grade, no external reference defines these conventions" | 0.4.0 | by design | N/A | Tracked as R7-P2 (closed) | — |

### 2.16 Model cards (`docs/reference/model-cards/*.md`)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| MC-01 | arima.md:160-164 | Reduced-Hessian SEs over the free directions "are a documented follow-up" | 0.6.0 | follow-up | SUPERSEDED | Tracked as R9-S3b | — |
| MC-02 | cointegration-regime.md:724 | Escribano-Jordá (2001) alternative STAR selection rule "not implemented" | 0.7.0 | alternative | N/A | No promise attached; the Teräsvirta H-sequence is the documented rule | — |
| MC-03 | cointegration-regime.md:936-943 | TVAR: "no regime-dependent (generalized) impulse responses — deferred" | 0.7.0 | deferred | SUPERSEDED | Tracked as RM-03 | — |
| MC-04 | copulas.md:59-61, 95 | `d > 2` deferred; rotated/survival variants deferred; tau-fit moment-based SE "deferred, not faked" | 0.4.0 | deferred | SUPERSEDED | Tracked as CL-06 | — |
| MC-05 | diagnostics.md:392-396 | `mstl`: statsmodels' Box-Cox `lmbda` option "deliberately not implemented — pre-transform `y` yourself" | 0.5.0 | by design | N/A | Documented boundary | — |
| MC-06 | diagnostics.md:616-621 | `bn_filter`: dynamic demeaning (the authors' later work) "not implemented here — the 2018 baseline is" | 0.6.0 | not implemented | OPEN | `demean="sm"`/`"nd"` only (docstring); the reference R code (`bnf_fcns.R`) carries the dynamic option, so a reference-run golden is available in the same environment that produced `bn_filters.json` | low / low |
| MC-07 | ml-convex.md:225-231 | `boosting` vs `mboost` "an open follow-up" | 0.8.0 | follow-up | SUPERSEDED | Tracked as CL-12 | — |
| MC-08 | ml-kernel.md:84-90 | Compact-support kernels deferred (reference bug) | 0.8.0 | deferred | SUPERSEDED | Tracked as CL-19 (N/A) | — |
| MC-09 | panel.md:255-261 | `panel_lp` cross-horizon covariance "a documented follow-up in `tsecon-panel`" | 0.5.0 | follow-up | SUPERSEDED | Tracked as CL-09 | — |
| MC-10 | panel.md:455-462 | LP-DiD: "Covariates / regression adjustment, the composition-effects correction (DGJT §2.10), pre-mean-differenced baselines (`pmd`), and the IV variant are not yet implemented" | 0.4.0 | not yet implemented | OPEN | P46: `lp_did` params carry none of them | medium / medium |
| MC-11 | panel.md:462 | Doubly-robust (AIPW) LP-DiD "has no reference implementation anywhere and is out of scope until one exists" | 0.4.0 | out of scope | N/A | Policy: no reference, no build | — |
| MC-12 | quantile.md:43-48 | `quantile_lp`'s Powell sandwich is not HAC under overlapping outcomes — "`growth_at_risk` carries the Newey-West correction … `quantile_lp` does not yet" | 0.1.0 | not yet | SUPERSEDED | Tracked as R34-07 | — |
| MC-13 | realized-vol.md:128-133; 03-volatility.md:115 | Noise-robust realized estimators (two-scale RV, realized kernels, pre-averaging) "not implemented here" | 0.3.0 | not implemented (Tier 2/3) | OPEN | The spec names R `highfrequency` + BNHLS tables as the gate | medium / high |
| MC-14 | structural-identification.md:23-25 | "the other four schemes are still point-only, and their bands remain an open item" | 0.3.0 | open item | SUPERSEDED | Tracked as CL-05 | — |
| MC-15 | unit-root-cointegration-tests.md:412-418 | Lee-Strazicich minimum-LM "not yet shipped, flagged here for honesty" | 0.3.0 | not yet | SUPERSEDED | Tracked as RM-07 | — |
| MC-16 | var-svar.md:430-440; `crates/tsecon-ident/src/zero.rs:309-312` | `zero_sign_svar`: for zeros at horizon ≥ 1 "this build does not yet apply the exact ARW volume-element correction — it returns the conditionally-uniform (unit) weight … the exact ARW weight for non-impact zeros is a roadmap swap-point" | 0.2.0 | not yet | OPEN | P32: `weighted=` kwarg present; `zero.rs:309-312` documents the "single, isolated swap point"; round 2's refutation (17:459-464) notes a horizon ≥ 1 zero under `weighted=True` **raises** rather than substituting 1.0, while the card says the unit weight is returned — the two surfaces disagree on the mode and should be reconciled when the weight lands | low / high |
| MC-17 | machine-learning.md:112-113 | `adaptive_lasso` key list omits `max_rel_change` | 0.8.0 | low | SUPERSEDED | Tracked as R11-O3 | — |
| MC-18 | check-series.md:26, 55 | "Every recommendation is a *starting point* that names the follow-up" (routing language, not an open item) | 0.5.0 | — | N/A | Not an open item | — |

### 2.17 `docs/examples/interval-coverage.md`, `monte-carlo.md`, `monte-carlo-frontier.md`

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| IC-01 | interval-coverage.md:1418-1424, 1440-1445 | "No weak-instrument-robust set exists to measure for these two rows" (`iv_gmm`, `lp_iv`); rec. #4 "half done" | 0.2.0 | recommendation | SUPERSEDED | Tracked as CL-02 | — |
| IC-02 | interval-coverage.md:1447-1450 | Rec. #5: "A per-τ convergence flag for `quantile_regression`. *Still open.*" (the shared bool trips on 232/3000 replications as an iteration cap) | 0.2.0 | recommendation | OPEN | P15: `converged` is a single scalar bool; quantile card :73 "`converged` is a single …"; `iterations` is per tau | low / low |
| IC-03 | interval-coverage.md:1432-1438 | Rec. #3 sup-t: "left two *coverage* gaps standing, both inherited: the IRF band's marginal coverage is itself 85.3 % at h=12, and `var_forecast`'s pooled per-cell marginal rate is 93.4 %" | 0.3.0 | inherited coverage gap | OPEN | The page still publishes 0.789 for the asymptotic per-horizon band at h=12, T=200 (:398) with the bootstrap + `bias_correct=True` route as the caller-side fix (:1477-1480); no estimator-side change since | medium / high |
| IC-04 | interval-coverage.md:1440-1446 | `bvar_*` measured as frequentist diagnostics only; Bayesian calibration lives in the round-6 doc and the card, not the registry | 0.3.0 | registry gap | OPEN | No SBC row in `docs/examples/coverage/`; the harness was never committed (R6-12) | low / medium |
| IC-05 | interval-coverage.md:1447-1452 | `svensson` and `dynamic_ns` "return the same interval-free shape … but are not under a per-run tripwire" | 0.3.0 | registry gap | OPEN | `factor_midas.py` tripwires `nelson_siegel` only | low / low |
| IC-06 | interval-coverage.md:1454-1458 | "Only two nominal levels are swept" | 0.2.0 | registry gap | SUPERSEDED | Tracked as R1-08 | — |
| IC-07 | interval-coverage.md:1425-1440 | `lp(cumulative=…)`, `growth_at_risk`, `proxy_svar`, `nongaussian_svar`, GARCH intervals, `flp` "not in this audit" | 0.2.0 | registry gap | FIXED | Struck through on the page ("Measured now"); P50 tripwire for `nongaussian_svar` | — |
| IC-08 | interval-coverage.md:1460-1470 | The nine caller-side recommendations | 0.3.0 | advice | N/A | Advice, not open work | — |
| IC-09 | monte-carlo.md; monte-carlo-frontier.md | No "not measured" / follow-up language on either page (grep) | 0.3.0 | — | N/A | Nothing recorded as open on the two Monte-Carlo pages | — |

### 2.18 `lab/REPORT.md` and `lab/experiments/results/exp06.md`

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| LAB-01 | lab/REPORT.md:347-367 | Graduation #1: DCS robust local level — PROPOSE (MC-recovery graded, Gaussian limit golden) | 2026-08-17 | graduation | FIXED | 0.3.0 `dcs_local_level` (CHANGELOG:1708-1724); P54 | — |
| LAB-02 | lab/REPORT.md:369-379 | Graduation #2: VaR backtest battery — PROPOSE | 2026-08-17 | graduation | FIXED | 0.3.0 `var_backtest` (CHANGELOG:1692-1707); P54 | — |
| LAB-03 | lab/REPORT.md:381-395, 259-262 | Graduation #3: AL-GAS dynamic quantile — HOLD; "the niche claim — wins when the volatility model is wrong — was not tested and must be demonstrated before promotion (vol-break / non-GARCH DGPs, next iteration)" | 2026-08-17 | hold | OPEN | `lab/laplace/al_gas.py` unchanged; no vol-misspecification experiment in `lab/experiments/` (exp01–06); P54: no `dynamic_quantile` | low / medium |
| LAB-04a | lab/REPORT.md:409-412 | prophet_lite salvage (a): a Fourier deterministic-terms builder (S) | 2026-08-17 | salvage | OPEN | P54: no Fourier-terms callable (the only "fourier" in the stub is `kernel_ridge`'s random features) | low / low |
| LAB-04b | lab/REPORT.md:413-420 | prophet_lite salvage (b): exact L1 trend filter, "promote as `l1_trend(y, n_knots, tau)` if Module 01 wants a changepoint-flavored filter" | 2026-08-17 | salvage | FIXED | 0.8.0 `l1_trend_filter` (Kim-Koh-Boyd, duality-gap certificate; ROADMAP:46); P54. **Silently closed** in the lab report (annotated, §6) | — |
| LAB-05 | lab/REPORT.md:422-429 | LAD-ARMA — DEFER; "should arrive as an `innovations="laplace"` option on ARIMA, MC-recovery graded" if demanded | 2026-08-17 | deferred | OPEN | `arima_fit` has no innovations-distribution kwarg | low / medium |
| LAB-06 | lab/REPORT.md:293-299 | Failure mode 1: `dm_test` refuses at long h with few origins (rectangular LRV non-PSD); "a Bartlett/NW variance option on `dm_test` would remove the sharp edge" | 2026-08-17 | actionable | OPEN | P16: `dm_test(e1, e2, h, loss)` — no variance-kernel option | medium / low |
| LAB-07 | lab/REPORT.md:300-303 | prophet_lite fit cost (a Rust port would erase it) | 2026-08-17 | cost | N/A | The forecaster was not promoted ("DO NOT promote", :397) | — |
| LAB-08 | lab/REPORT.md:448-466 (exp06) | Frontier for the next lab cycle: SPCI, conformal PID, quantile conformal on GARCH-standardized residuals, BSTS | 2026-08-25 | next-cycle | OPEN | None started (`lab/` holds `laplace`, `prophet_lite`, `experiments`, `audit` only) | medium / medium |
| LAB-09 | lab/REPORT.md:331-336 | Scope limits of the study (20 origins on CO2; correctly-specified-GARCH world only; additive symmetric outliers at one size) | 2026-08-17 | scope | N/A | Recorded limits; the actionable half is LAB-03 | — |

### 2.19 `docs/roadmap/10-machine-learning.md` Tier 3 / Tier 4 (the module `ROADMAP.md:46` says is next after the Tier-2 wave)

| id | source | recorded | as of | severity / tier | status | evidence | value / cost |
|---|---|---|---|---|---|---|---|
| ML10-01 | 10:118-124 | Square-root LASSO (validate vs R RPtests/scalreg) | plan | Tier 3 | OPEN | Not in the stub | low / medium |
| ML10-02 | 10:126-130 | Three-pass regression filter / PLS (Kelly-Pruitt; validate vs authors' MATLAB, PLS vs sklearn) | plan | Tier 3 | OPEN | Not in the stub; sklearn PLS is a runnable golden for the PLS half | medium / medium |
| ML10-03 | 10:132-136 | Quantile regression forests / distributional forests (Meinshausen) | plan | Tier 3 | FIXED | 0.8.0 `random_forest(quantiles=…)` "quantile regression forests" (ml-trees.md:10, 20; stub :3213-3220) at property grade | — |
| ML10-04 | 10:138-143 | Complete subset regressions (Elliott-Gargano-Timmermann) | plan | Tier 3 | OPEN | Not in the stub | medium / medium |
| ML10-05 | 10:143 | Dynamic model averaging / selection (Raftery; Koop-Korobilis) | plan | Tier 3 | OPEN | Not in the stub; validation target is MATLAB output | medium / high |
| ML10-06 | 10:145-149 | Gaussian processes with state-space (SDE) representation | plan | Tier 3 | OPEN | Not in the stub; GPy/GPflow exact-GP golden named | low / high |
| ML10-07 | 10:151-155 | Double/debiased ML with dependent-data cross-fitting | plan | Tier 3 | OPEN | Not in the stub; DoubleML named as the iid golden | medium / high |
| ML10-08 | 10:157-161 | Lag-grouped Shapley attributions (TreeSHAP interop) | plan | Tier 3 | OPEN | Not in the stub | low / high |
| ML10-09 | 10:163-167 | GETS model selection and indicator saturation (R `gets`) | plan | Tier 3 | OPEN | Not in the stub; R-only reference | low / high |
| ML10-10 | 10:169-173 | Contamination-aware benchmark harness for external forecasters (retained in core) | plan | Tier 3 | OPEN | Not built; overlaps RC19-HM | medium / medium |
| ML10-11 | 10:175-181 | Desparsified / debiased LASSO under serial dependence (gate: reproduce R `desla`) | plan | Tier 4 | OPEN | Not in the stub; R-only reference | low / high |
| ML10-12 | 10:187 | Factor-adjusted regularized selection (FarmSelect; gate: R package) | plan | Tier 4 | OPEN | Not in the stub | low / medium |
| ML10-13 | 10:193 | Macroeconomic Random Forest (Goulet Coulombe; gate: MacroRF outputs) | plan | Tier 4 | OPEN | Not in the stub | low / high |
| ML10-14 | 10:199 | Illusion-of-sparsity diagnostic (GLP 2021 spike-and-slab; gate: paper's six applications) | plan | Tier 4 | OPEN | Not in the stub | low / high |
| ML10-15 | 10:200 | Random subspace / random projection forecasting (Boot-Nibbering; gate: FRED-MD results) | plan | Tier 4 | OPEN | Not in the stub | low / medium |
| ML10-16 | 10:82-83, 219 | Gradient-boosted-tree adapters, Inoue-Kilian bagging, foundation-model adapters (companion package by scope ruling) | plan | Tier 2 adapters / companion | N/A | Out of core by the module's scope ruling (ROADMAP:46) | — |


## 3. Open items grouped by theme, ranked

Every `OPEN` row above appears exactly once below. Within a theme, rows are
ranked by value first and cost second (this audit's estimates; the source's
severity is on the row). Each carries one suggested next step. The six theme
names the audit brief asked for are kept; two more were needed for what the
sources actually hold (estimator-side inference tails, and the release
milestones).

### 3.1 Inference engineering the audits left standing (estimator-side)

| rank | id | item | value / cost | next step |
|---|---|---|---|---|
| 1 | R1-02, CL-09 | No cross-horizon covariance for `lp_iv`, `lp_multiplier`, `lp_state`, `panel_lp` → sup-t refused, closed forms only | medium / high | Build the influence-function cross-horizon covariance once in `tsecon-lp` (the `lp` HAC path already has it) and thread it to the four; `panel_lp` needs the entity-cluster / Driscoll-Kraay variant the panel card names |
| 2 | IC-03 | The asymptotic IRF band's marginal coverage is 0.789 at h=12, T=200; `var_forecast` pooled 0.934 — "inherited" gaps under every sup-t band | medium / high | Make `method="bootstrap", bias_correct=True` the long-horizon default the card already recommends, or ship a Kilian-corrected asymptotic band; re-measure with the existing `irf_bands.py` |
| 3 | R34-07 | `quantile_lp`'s Powell sandwich is not HAC under overlapping multi-step outcomes (the card's own "does not yet") | medium / medium | Port `growth_at_risk`'s Newey-West correction (`hac_lags = h − 1`) to `quantile_lp` as `se="hac"` and add a registry row |
| 4 | N21-03 | `proxy_ar_sets` default still `"delta"` (0.881 / 0.828 at h=12) although `second_order` (0.974 / 0.935) is measured every run | medium / low | Decide the flip in a release note: default `second_order`, `"delta"` kept as the explicit escape hatch; the width price (1.6×) is already on the card |
| 5 | R1-05 | `flp` per-element `se` conditions on estimated eigenfunctions (se/sd ≈ 0.67, flat in T) | medium / high | A bootstrap-the-two-steps route (`n_boot=`) over `functional_pca → flp`, measured on the committed FLP registry rows |
| 6 | R9-S3b | `arima_fit` interior `bse` at a boundary come from the full-vector observed information | medium / medium | Reuse `tsecon-garch`'s free-coordinate mask + reduced Hessian in `tsecon-arima/src/cov.rs`; the flags and `boundary_note` already exist |
| 7 | CL-02 | No Anderson-Rubin / weak-IV-robust set for `iv_gmm` and `lp_iv` (coverage rec. #4 "half done") | medium / medium | The closed-form quadratic inversion in `proxy_ar.rs` is the same object; expose it for the IV regressions and add the two coverage rows the page already names |
| 8 | CL-05 | Bootstrap bands for `long_run_svar`, `max_share_svar`, `hetero_svar`, `nongaussian_svar` (still point-only) | medium / medium | A residual/moving-block bootstrap over the reduced form + re-identification per draw, reusing `proxy_svar_bands`' machinery; measure before shipping |
| 9 | LAB-06 | `dm_test` refuses at long h with few origins (rectangular LRV non-PSD); no Bartlett/NW option | medium / low | Add `kernel="rectangular"|"bartlett"` to `dm_test`, default unchanged; the lab's `common.py` fallback is the reference arithmetic |
| 10 | RC19-03c | Wild block bootstrap bands for LP / LP-IV | medium / medium | After rank 1: a `band="bootstrap"` route on `lp`/`lp_iv` using `tsecon-bootstrap`'s block schemes; measure against sup-t on `lp_family.py` |
| 11 | RC19-03b | Jordà significance bands | low / medium | Only after RC19-03c; needs the same bootstrap |
| 12 | MC-16 | Exact ARW volume-element weight for non-impact zeros in `zero_sign_svar`; the card ("returns the unit weight") and round 2 ("raises") disagree on the current mode | low / high | First reconcile the two statements with a probe on a horizon-1 zero; then the swap point `zero.rs:309-312` |
| 13 | CL-13 | EGARCH multi-step variance forecasts (simulation route unshipped) | low / medium | Simulation forecast with a seed, returning mean and quantile paths; refusal text already names the remedy |
| 14 | MC-10 | LP-DiD covariates, composition correction, `pmd`, IV variant | medium / medium | Covariates first (the fixest example code covers them, so the R golden route exists) |
| 15 | RC19-04a | Lewis-Mertens generalized first-stage test; ACF moving-block pre-test | low / medium | Matlab-only references; transcription + MC grade |
| 16 | R1-06 | A minimum-cycles advisory in `nsdiffs` output | low / low | Add `n_cycles` and an `advisory` string key (not a refusal — the reference has none) |
| 17 | CL-06 | Copulas: `d > 2`, rotated/survival variants, tau-fit SEs | low / medium | Gaussian/t correlation-matrix parameterization first (statsmodels golden exists) |
| 18 | R1-01 | Diffuse period terminates on an absolute norm test over `P_inf` | low / medium | Close formally: either record the design defence in the brief's "Still open" list or implement rank counting and prove bit-identity on the DK examples |
| 19 | CL-03 | Angrist-Pischke, Cragg-Donald, Kleibergen-Paap statistics | low / low | Cheap on the `iv_gmm` first stage; linearmodels goldens exist |

### 3.2 Reference runs blocked on R / CRAN (or other unrunnable references)

The block is a per-container egress limit, not a repository one: R 4.3.3 ran
for `bn_filters.json` (the authors' own R code) and `lpdid.json` (fixest built
from the CRAN mirror) — `validation-matrix.md:161, :177, :209` (VM-15). Every
row below is an existing generator waiting for that environment.

| rank | id | item | value / cost | next step |
|---|---|---|---|---|
| 1 | RM-14 (VM-13) | tsDyn reference-run for `threshold_vecm`/`hansen_seo_test`/`threshold_var`/`star` — the 0.7.0 "named follow-up" | medium / medium | Run `tsDyn::TVECM`, `TVECM.HStest`, `TVAR`, `star` in the fixest-capable container on the committed fixture draws; expect grid-resolution agreement only (the matrix row says why) |
| 2 | VM-14 | SPJ panel LP: `panelLP.R` fetched verbatim, never run | medium / low | `Rscript` the vendored transcription source on `panel_spj.json`'s inputs; upgrade the row to reference-run |
| 3 | CL-11 (VM-06) | `pds_lasso` coverage vs R `hdm` / Stata `pdslasso` | low / medium | `hdm::rlassoEffects` on the committed design; one fixture block |
| 4 | CL-12 (VM-07, MC-07) | `boosting` vs R `mboost::glmboost` | low / medium | One `glmboost` run on `convex.json`'s cases with `nu`, `mstop` matched |
| 5 | MC-06 | `bn_filter` dynamic demeaning (the reference R code carries it) | low / low | Extend `generate_bn_filter_fixtures.R` with `dynamic demean` cases; add `demean="dynamic"` |
| 6 | VM-09 | `panel_lp` not golden-pinned (R `lpirfs` / pLP) | low / medium | Same container as rank 2 |
| 7 | RM-06, RM-07 | HEGY and Lee-Strazicich (R-only, location-dependent tables) | low / medium | Stay deferred until the container exists; `uroot`/`urca` goldens then make them build-next |
| 8 | ML10-09, ML10-11, ML10-12 | GETS (`gets`), desparsified LASSO (`desla`), FarmSelect — Tier 3/4 with R-only gates | low / high | Not before the Tier-3 sequencing decision (§3.5) |
| 9 | VM-02, VM-04, VM-10 | Property-graded rows whose golden needs **no R at all**: `lp_state` (statsmodels OLS on the interacted design), `lasso_path` (scikit-learn `lasso_path`), `mean_group_var` (per-entity statsmodels `VAR` averaged) | low / low | Three fixture blocks in the existing generators; the cheapest matrix upgrades after VM-01 |

### 3.3 The mapie EnbPI / ACI cross-check

| rank | id | item | value / cost | next step |
|---|---|---|---|---|
| 1 | VM-01 | EnbPI/ACI "property-MC until that run happens"; `mapie 1.5.0` `TimeSeriesRegressor(method="enbpi"|"aci")` is importable in this audit's venv (P45) | **high / low** | Write `fixtures/generate_conformal_fixtures.py` (no `import tsecon`) calling `TimeSeriesRegressor` on the `test_conformal.py` DGP with the AR base handed in as a scikit-learn-style estimator; pin the interval endpoints at the tolerance the residual arithmetic allows; upgrade the matrix row. The single cheapest validation-grade upgrade left in the repository |
| 2 | R10-O1 | EnbPI: `base` omitted always refuses (ergonomics) | low / low | Accept `base=None` as "the built-in AR ensemble" for `method="enbpi"` only, keeping the refusal for a supplied callable |
| 3 | RC19-05 (frontier), LAB-08 | Decaying-step ACI, SPCI, conformal PID, quantile conformal on GARCH-standardized residuals, BSTS | medium / medium | Conformal PID first (smallest surface, natural ACI extension; `lab/REPORT.md:459-466`); the GARCH-residual variant fixes the conditional-coverage weakness exp06 measured |

### 3.4 Interval-coverage surfaces never measured, and registry gaps

| rank | id | item | value / cost | next step |
|---|---|---|---|---|
| 1 | R1-08 (R2-12, IC-06) | Only two nominal levels are swept; scale-type shortfalls peak at 70–82 % nominal | medium / medium | Add a `levels=(0.68, 0.90, 0.95)` axis to `run_all.py` for the regression-SE and LP families first (they are the cheap rows) |
| 2 | R34-13, R6-12 | The lens-1–3 harnesses (switch sweep, scale sweep, degenerate sweep, constant-diagnostic detector) and the NIW SBC harness exist only as prose | medium / medium | Commit them beside `lab/audit/round11/` driving the same registry; the round-11 `registry.py` already gives every callable a canonical call |
| 3 | IC-04 | The `bvar_*` Bayesian calibration (SBC) has no registry row | low / medium | After rank 2: one SBC row per prior (`"glp"`, `"none"`) at the round-6 design |
| 4 | IC-02 | `quantile_regression` per-τ `converged` flag (rec. #5 "Still open"; the shared bool trips on 232/3000 replications as an iteration cap) | low / low | Return `converged` per tau like `quantile_lp` does (`[tau][h]`) — additive key, keep the scalar as `converged_all` |
| 5 | IC-05 | `svensson` / `dynamic_ns` not under the interval-free key-set tripwire | low / low | Two lines in `factor_midas.py` |
| 6 | R34-08 | The LP card lacks the "cumulative=True is materially worse-calibrated at T=200" sentence the 0.3.0 registry can now source | low / low | One sentence citing the registry row |

### 3.5 Tier 3 / 4 roadmap gates, build-later and deferred features

| rank | id | item | value / cost | next step |
|---|---|---|---|---|
| 1 | RM-03 (CL-10, MC-03) | Regime-dependent GIRFs (Koop-Pesaran-Potter) for `threshold_var` | medium / high | Simulation GIRF over shock/history draws with the seed contract; MC-graded (no reference) — the card already declines to fake it with `var_irf` |
| 2 | RM-02 | Hansen (1997/2000) threshold confidence sets for SETAR | medium / medium | LR-inversion set on the existing concentrated-LS scan; R tsDyn golden once §3.2 rank 1 runs |
| 3 | MC-13 | Noise-robust realized measures (two-scale RV, realized kernels, pre-averaging) | medium / high | Two-scale RV first (closed form, BNHLS tables as the gate) |
| 4 | ML10-02, ML10-04, ML10-07, ML10-10 | Tier 3 with runnable goldens or clear gates: 3PRF/PLS (sklearn PLS), complete subset regressions (Medeiros et al. numbers), DML with dependent-data cross-fitting (DoubleML iid golden), the contamination-aware benchmark harness | medium / medium–high | These four are the build-next candidates of Module 10 under the validation-first bar; sequence PLS → CSR → harness → DML |
| 5 | ML10-01, ML10-05, ML10-06, ML10-08 | Tier 3 with weaker gates: square-root LASSO (R), DMA/DMS (MATLAB), GP-SSM (GPy), lag-grouped Shapley (shap) | low / medium–high | Sequence after rank 4 |
| 6 | ML10-13, ML10-14, ML10-15 | Tier 4: MRF, illusion-of-sparsity, random subspace | low / high | Gated on the named paper tables; not before Tier 3 |
| 7 | RM-04, RM-05, RM-08, RM-09, RM-10 | JSZ affine term structure; distributed-lag climate regressions; X-13 wrapper; dynamic copulas; energy-climate systems | low / high | Remain build-later / deferred as ROADMAP states; JSZ has the ACM precedent (NY Fed series) as a possible golden |
| 8 | RC19-07, RC19-09 | Read-Zhu soft sign-restriction sampler; Bayesian QVAR | medium–low / high | Research-grade; the sampler swap is the one with an existing stack to slot into |
| 9 | RC19-06, RC19-08, RC19-10, RC19-HM | The study-shaped items: LP-vs-VAR primer verification, GaR horse-race, climate reconciliation, honorable mentions | medium–low / medium–high | The primer verification is the cheapest (the frontier MC page is the seed) |
| 10 | LAB-03, LAB-04a, LAB-05 | AL-GAS hold (needs vol-misspecification DGPs), the Fourier-terms builder, LAD-ARMA as `innovations="laplace"` | low / low–medium | The Fourier builder is S-effort with a closed-form golden and unblocks the covariates contract |
| 11 | RM-01 | `smooth_fixed` awaiting a Python binding | low / low | Bind it; it is already validated and used by `dfm_nowcast` |
| 12 | RM-19 | The Tier 3/4 sequencing decision itself | — | Record it in ROADMAP §0 once ranks 4–6 are ordered |

### 3.6 Documentation follow-ups

| rank | id | item | value / cost | next step |
|---|---|---|---|---|
| 1 | R11-L-1 | A generator emitting docstring and stub from one source (the round-11 lesson; 29 drifted docstrings were the symptom) | medium / medium | Generate `__init__.pyi` from the `///` comments (or vice versa) in CI; the M1 key-gate test then becomes redundant |
| 2 | R11-O4 | No model card for `hp_filter` / `bk_filter` / `cf_filter` | low / medium | A short filters card; the diagnostics card's trend-cycle section is the template |
| 3 | R9-L1 | The brief never gained round 9's "standing checks" (short-sample batteries, cited-convention diffs) | low / low | Two bullets in `16-adversarial-audit-brief.md` §"Techniques worth reusing" |
| 4 | R11-O2 | `check_stationarity` returns five keys named on no surface | low / low | One `Keys:` sentence in `_inspect.py` |
| 5 | R11-O3 | ML card's `adaptive_lasso` key list omits `max_rel_change` | low / low | One list edit |
| 6 | R2-08f-ii | Stub says `link` is "probit" or "logit" with no dynamic caveat | low / low | One stub sentence |
| 7 | S15-U1, S15-U6, RC19-cite, RC19-Q4A | Citations reconstructed from memory (JL 2022 volume/pages, BJT 2016 title, MOSW 2021 volume/pages, the MOSW variance attribution, the research scan's citations, the IJK/JG spot-check) never recorded as verified | low / low | One verification pass, recorded in the spec's "Flagged uncertain" sections; add a `references` block to `CITATION.cff` |
| 8 | S15-U4 | JL's diagnostics-only proxy scale correction never pinned | low / low | Read JL's replication code once; IRF bands are provably unaffected |
| 9 | R11-L-3 | The sweep-H harness should treat `None` = entropy as the null | low / low | Flip the verdict logic in `lab/audit/round11/sweep_h_seed.py` |

### 3.7 API polish and performance

| rank | id | item | value / cost | next step |
|---|---|---|---|---|
| 1 | R2-09 | `recession_probit(link="banana", dynamic=True)` returns a result (P19) — the `link` validation lives only in the static branch | low / low | Validate `link` before the branch; a one-line move in `lib.rs:10421` |
| 2 | R11-O1 | `inspect.signature` renders eight defaults as `Ellipsis` | low / medium | Express the defaults in `#[pyo3(signature)]` where PyO3 allows (`cz=-1.0` can be a literal; the tuple/`vec![]` cases need `None` sentinels) |
| 3 | R11-O5 | The ARIMA exact-MLE engine's ~1–2 ms/observation constant (`auto_arima` ≈ 1 min at T=3200) | medium / high | Profile the per-observation Kalman step; a steady-state switch after convergence of `P_t` is the standard fix |
| 4 | R8-res1 | `theta_forecast` returns a bare array; `alpha`/`b0` computed and unexposed | low / low | Additive: a `theta_forecast_fit` or a `return_details=` flag — the bare-array contract is documented, so keep it as the default |
| 5 | S15-U5 | `proxy_svar_bands` is single-proxy (`k > 1` refused as a shape error) | low / medium | Generalize the block unit to `(u_t', m_t')` with the k×k normalization inversion the spec sketches |
| 6 | R11-O7, R11-O8 | `historical_decomposition` O(T²); `mcmc_diagnostics` T^1.7 | low / low | Companion-form recursion; FFT autocorrelation |
| 7 | R11-O6 | The t-copula MLE constant makes `copula_select`'s default menu 130× its next member | low / medium | Profile the ν profile step; or drop `"t"` from the default menu with a note |

### 3.8 Adoption and release milestones

| rank | id | item | value / cost | next step |
|---|---|---|---|---|
| 1 | RM-15 | conda-forge feedstock | medium / medium | A staged-recipes PR after the next tagged release |
| 2 | RM-13 | Replication gallery 8 → the v1 gate of ≥ 15 | medium / high | The three cheapest are already scoped: Blanchard-Quah, Kilian oil VAR, Primiceri (ROADMAP:189) |
| 3 | RM-11 | Head-to-head explainer chapters (LP vs VAR, BVAR priors, GARCH zoo, identification schemes) | medium / medium | The LP-vs-VAR one can be written from `monte-carlo-frontier.md` today |
| 4 | RM-12 | Public speed dashboard | medium / medium | Publish `benchmarks/bench.py` output as a docs page with parity checks first |
| 5 | RC19-V, RC19-G (JOSS half), RC19-04c, RC19-Q4C | JOSS submission (gate ~January 2027), the JCRE proxy-zoo replication paper (the post-1984 split and the draft are the two unfinished steps of next-quarter item C) | medium / low–high | Nothing to do before the clock runs; the JCRE draft is the research-use evidence JOSS needs |
| 6 | RM-16 | v1.0 gate: API freeze + deprecation policy, tiering published | — | A `DEPRECATION.md` policy is the only piece with no precursor in the tree |

## 4. Silently-closed items (recorded as open somewhere, actually fixed on main)

These are the rows whose *source* still reads as open although main closed
them — the ones a reader of that source alone would get wrong.

| id | where it still reads open | what closed it | corrected in place? |
|---|---|---|---|
| S15-00 | `15-proxy-svar-bands.md:3-8` "Status: specification only. Not yet implemented." | `proxy_svar_bands` + `proxy_ar_sets` shipped in 0.3.0 (CHANGELOG:2190-2270); P02 | yes (§5) |
| R1-04 | `16-adversarial-audit-brief.md:383-384` "a size-restoring joint test is the open item" | `joint="bonferroni"` 0.3.0, default since 0.5.0 (CHANGELOG:1450-1460); P03 | yes (§5) |
| R1-03 / R6-08 | `16:378-382` "a re-engineered long-horizon correction is the open item"; `20:120-122` "future work" | `rf_method="second_order"` (0.3.0) and `"second_order_bc"` (0.6.0), measured every run; P04 — the *default* is the remaining decision (N21-03) | yes (§5) |
| N21-02 | `21:234-236` "Default flips … are deliberate future decisions" | The IVX half was decided in 0.5.0 | yes (§5) |
| R8-nm3 | `23:270-273` "`lp_did` vs a live R/fixest run — no R in this environment" | `validation-matrix.md:161` reference-run golden via `fixtures/generate_lpdid_fixtures.R` | yes (§5) |
| RC19-G | `19:12` "EVT and conformal (ACI/EnbPI) are on the roadmap but unshipped" | EVT 0.3.0; conformal 0.5.0 | yes (§5) |
| RC19-01, RC19-02, RC19-04b, RC19-05 | `19` ranks #1, #2, #4(b), #5 read as proposals | SPJ 0.3.0; LP-DiD 0.4.0; GK replication 0.4.0; conformal core 0.5.0 | covered by the same note (§5) |
| LAB-01, LAB-02, LAB-04b | `lab/REPORT.md:347, 369, 413-420` "PROPOSE" / "promote as `l1_trend`" | `dcs_local_level` and `var_backtest` 0.3.0; `l1_trend_filter` 0.8.0 | yes (§5) |
| R10-O2 | `25:205` listed under "Remaining OPEN" | Its own text says "now documented"; P52 confirms | no — the sentence already carries the closure |
| R9-F12 (the 0.6.0 half) | `CHANGELOG:1062-1064` "a documented follow-up in ROADMAP build-later" | 0.7.0 shipped the restricted cases | no — the CHANGELOG is chronological and 0.7.0's own entry says "the ROADMAP build-later follow-up that finishes field-report item 12" |

Not silently closed, but worth saying: every "OPEN" item in rounds 10 and 11
is still open exactly as written (P20–P22, P55), and the brief's `flp`,
`nsdiffs`, diffuse-period and LP-family items are still open as written
(P23–P25).

## 5. Sources corrected

Per the audit's rule — only where a source states an item is open that main
provably closed — each edit adds a short "closed in …" note in place and
leaves the original text intact:

| file | line(s) | note added |
|---|---|---|
| `docs/roadmap/15-proxy-svar-bands.md` | 3-8 (status banner) | shipped in 0.3.0 as `proxy_svar_bands` (moving-block / wild) and `proxy_ar_sets`; the spec is retained as the design record |
| `docs/roadmap/16-adversarial-audit-brief.md` | 378-384 (`proxy_ar_sets`, `ivx_test` bullets) | the correction shipped opt-in in 0.3.0/0.6.0 (default flip still open); the size-restoring joint test shipped in 0.3.0 and became the default in 0.5.0 |
| `docs/roadmap/19-research-contributions.md` | 12 (grounding note) | EVT (0.3.0) and conformal (0.5.0) shipped; #1, #2, #4(b), #5 core shipped |
| `docs/roadmap/20-audit-round-6-findings.md` | 120-122 (finding 8 "Open") | opt-in repairs shipped; default unchanged |
| `docs/roadmap/21-long-horizon-and-joint-inference.md` | 234-236 (default flips) | the IVX default flipped in 0.5.0; the `proxy_ar_sets` default is still `"delta"` |
| `docs/roadmap/23-audit-round-8-findings.md` | 270-273 (`lp_did` vs R) | the R/fixest reference run was made; matrix row 161 |
| `lab/REPORT.md` | 347, 369, 413 (graduation candidates) | `dcs_local_level` and `var_backtest` shipped 0.3.0; `l1_trend_filter` shipped 0.8.0 |

No other source was edited. In particular the round-10/11 OPEN lists,
ROADMAP §0's build-later / deferred lists, the validation matrix's named
follow-up and the coverage page's "Still open" recommendation were all
re-verified as still open and left as they are.

## 6. Open — what this ledger did not get to

- **The probes are closure probes, not re-measurements.** Every coverage
  number quoted in an OPEN row is the source's own; nothing was re-run at
  Monte-Carlo scale (the registry's consolidated run is ~40 min and was
  outside this sweep's budget). The one timing probe (P55) is single-machine.
- **Field-report item 5** (round 9) could not be identified from the committed
  record; R9-A / RM-18 stay `UNVERIFIED`.
- **The JL block-length rule** (S15-U2) is implemented as the spec
  reconstructed it (`round(5.03 · T^{1/4})` reproduces the shipped 20 at
  T_eff = 238), but whether that is Jentsch-Lunsford's rule was never checked
  against the paper here either.
- **Test counts** (R2-08b) were not re-collected; the ROADMAP's 1775 / 1526
  figures are reported, not verified.
- **The mapie cross-check itself** (VM-01) was not run — it is a fixture and a
  test, i.e. code, which this sweep may not add. It is the highest-value /
  lowest-cost row in the ledger.
- **Module specs other than 10** carry Tier 3/4 inventories too
  (`01`–`09`, `12`); ROADMAP §0 names only Module 10 as next, so only its
  tiers were extracted. Rows for the others would add ~200 planned-but-ungated
  items without changing the picture.
