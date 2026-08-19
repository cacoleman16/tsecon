# Adversarial audit, round 7 — findings

> **Working document.** Continuation of
> [round 6](20-audit-round-6-findings.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 7 ran against the post-0.3.0 tree with two jobs, in order: **fix** the
two oldest confirmed-open findings (both round-1 `garch_fit` items, plus the
round-6 negative-integer cosmetic), then a **finder + refuter pass** over the
surface no audit has touched — the 0.3.0-late additions (`var_backtest`,
`dcs_local_level`, `panel_lp(bias_correction=)`, `proxy_ar_sets(rf_method=)`,
`ivx_test(joint=)`) and the five replication suites' claims, under lenses
1–3 and 5. Every candidate went through an adversarial refutation step
(independent DGP/seed re-derivation, promise-surface check against `__doc__`
and the model card), defaulting to refuted. Probes were scratchpad-only;
fixes landed in the same branch, each with a pre/post reproduction.

**Part 1: 3 backlog items fixed. Part 2: 236 lens-1–3 comparisons attempted,
236 made, across the five new surfaces; 2 confirmed findings (1 code — fixed,
plus the latent optimizer defect it exposed — fixed; 1 cosmetic doc drift —
fixed), 3 candidates refuted, everything else swept clean, including four
card snippets reproduced byte-for-byte and the five replication suites'
claims verified against their test sources.**

---

# Part 1 — the round-1 `garch_fit` backlog, retired

## F1 — Silent all-NaN standard errors at an active boundary → boundary-aware reduced-Hessian SEs with per-parameter flags

**Was:** `silent-wrong-answer` (round 1: 10/120 probe units; reproduced this
round at **24/50** on boundary-attracted DGPs — near-integrated, IGARCH,
tiny-alpha, white-noise, pure-beta; T=750). Mechanism, from source: at an
active constraint (`alpha` at its sign bound — the optimizer lands at
~1e-14 — or persistence at 1) the central-difference Hessian probe crosses
the constraint, `validate_params` refuses the probe point, the whole
covariance computation errors, and `fit()` swallowed that into an
**unflagged all-NaN `se_mle`/`se_robust` row** (old `model.rs`
`unwrap_or(NaN)`).

**Fix (cause, not symptom):** the observed information is singular in the
constrained direction *by construction*, so no full-vector covariance
exists. `inference::std_errors` now takes a free-coordinate mask and
computes the reduced Hessian / score covariance over the interior
directions only (bit-identical arithmetic when everything is free — the
all-free path was verified bit-identical on the arch fixture fits through
the Python surface before the scale fix landed). `GarchModel::boundary_mask`
detects active constraints at exactly the probe resolution: coordinate sign
bounds, the GJR pair bound `alpha_i + gamma_i >= 0`, and the joint
persistence / EGARCH `|sum(beta)|` bound over the remaining free
coefficients. The results (Rust and the Python dict) gain per-parameter
**`se_valid`** and **`boundary`** flags — the `tsecon-evt` `se_valid`
precedent — a **`boundary_note`** teaching string naming the constraint,
and the previously unexposed **`converged`** flag; `GARCHResults.summary()`
renders the note; docstring/`.pyi`/card carry the contract, and
`test_docstring_keys.py` gained a `garch_fit` key-enumeration tripwire.

Pre/post, on the same 50-unit battery:

| | pre-fix | post-fix |
|---|---|---|
| fits reached | 50/50 | 50/50 |
| silent (unflagged) NaN SE rows | **24** | **0** |
| boundary-flagged fits | (no flags existed) | 24, every one with a note |
| flagged fits keeping finite interior SEs | — | 24/24 |
| interior fits with clean flags | 26 | 26 |

When even the reduced problem is degenerate (with `alpha = 0` the pair
`(omega, beta)` can sit on a flat likelihood ridge), the report is honestly
all-invalid — still flagged, never silent (pinned in the white-noise
property test). Golden constraint honored: all five arch-pinned fixture
cases were **bit-identical** pre/post this fix (params, SEs, loglik,
conditional-volatility path, verified through the Python surface).

## F2 — Fitted parameters not scale-robust → standardize-and-map-back estimation

**Was:** round 1 measured 52/330 cross-scale comparisons converging to a
different point; the round-2 fix repaired only the SE steps. Reproduced this
round with the audit's own design (4 DGPs × 2 means × 5 seeds × 16 decades,
T=500): **93/320** comparisons disagreed (>1e-3 mapped relative difference),
with a diagnostic the original count did not have: **every disagreement had
a mapped log-likelihood gap ≤ 2.4e-11** — these are landings at different
points of an `alpha≈0` / low-persistence likelihood *ridge*, not
convergence failures. On a well-identified battery (5 spec families × 8
seeds × 8 decades, T=1000) the pre-fix count was 0/320: the units leak
through the optimizer's *path* (the loglik shifts by −T ln c, the working
`ln ω` coordinate translates, and termination arithmetic sees both), which
only matters where the surface is nearly flat.

**Fix:** `GarchModel::fit` now optimizes on the internally standardized
series `y / rms` (the RMS of the starting residuals — the quantity the
backcast and starting-value grid already use) and maps the optimum back
through the exact reparameterization; loglik/paths/SEs are re-evaluated at
the mapped parameters on the original data. `arch`'s own `rescale=True`
trick, applied unconditionally so it cannot be forgotten.

Pre/post, same probe design (comparisons attempted = made = 320 each):

| battery | pre-fix different-point | post-fix |
|---|---|---|
| well-identified (5 families × 8 seeds × 8 decades) | 0/320 | **0/320** |
| boundary-attracted (4 DGPs × 2 means × 5 seeds × 16 decades) | 93/320, unflagged | **75/320, every one flag-covered; 0 on unflagged fits** |

"Flag-covered" = at least one side of the comparison is a fit the output
itself now marks (`boundary` set, or an invalid-SE flag) — measured
directly: 75/75. The mapped-loglik gaps on the residual 75 are ≤ 0.03: the
ridge coordinate is unidentified and `c * y` itself rounds, so bit-level
agreement there is not achievable even in principle; the honest posture is
the flag plus the note's ridge warning. Where bit-level *is* achievable it
is delivered and pinned: **power-of-two rescalings commute bit-exactly**
(standardization is a pure exponent shift — a same-run invariant test per
the brief's own technique), decade rescalings agree to < 1e-6 relative on
well-identified fits (property tests, Rust and Python).

Golden constraint honored: fitted fixture parameters moved ≤ 7.8e-7
relative (normal cases) / 4.9e-6 (the documented flat-`nu` t ridge), loglik
by ~1e-11 — far inside the 1e-3/2e-2 pins; **no tolerance was touched**.
The volatility card's snippet output is unchanged at its printed precision.

## F3 — Negative integer arguments raise a raw `OverflowError` → one central teaching `ValueError`

**Was:** `cosmetic`, round 6: `lags=-1`, `outer_iter=-1`, `p=-1`, a negative
seed — every negative int passed to an unsigned Rust parameter surfaced as
PyO3's bare `OverflowError: can't convert negative int to unsigned`,
naming neither function nor parameter, library-wide.

**Fix, central by construction:** `_coerce._call` — the same single
choke-point that already upgrades PyO3 rank errors — matches exactly that
conversion message and rebuilds it into a `ValueError` naming the function
and the offending parameter(s) (positional args mapped through the compiled
signature; shallow tuples scanned), stating the nonnegative-count contract,
chaining the original. Deliberately narrow: genuine numeric overflows
inside estimators still surface as themselves, and a boundary that already
raises a better error (`arima_fit`'s seasonal tuple) is not shadowed.
Tests cover a cross-crate sample (`var_fit`, `stl`, `garch_fit`, `lp`,
numpy ints), and that negative values destined for `f64` parameters are
untouched.

---

# Part 2 — finder + refuter over the never-audited surface

Comparisons **made** (not merely attempted), per target, lenses 1–3:
`var_backtest` 26/26 · `dcs_local_level` 32/32 · `panel_lp` 34/34 (+8
follow-ups) · `proxy_ar_sets` 15/15 (+1 corner) · `ivx_test` 37/37 —
**236/236 attempted comparisons reached**, plus doc-surface key diffs for
all five, four card snippets re-run, and the five replication suites read
against their CHANGELOG claims.

## Confirmed 1 — `dcs_local_level(density="laplace")` converged to different points depending on the units of `y` (fixed)

**`silent-wrong-answer`** (class 2 — the garch-F2 shape on the round's own
new surface). Lens-2 sweep: on the *same* contaminated level series,
rescaling across eight decades moved the Laplace fit's `kappa` between
distinct optima — measured incidence **11/20 seeded series**, `kappa`
moving up to **57%**, mapped log-likelihood gaps up to **4.6**, every run
`converged=True`. Treatment separated from seed per the brief: the smooth
`"t"`/`"gaussian"` fits moved **0/20** with mapped-loglik gaps 0.0000
(negative control), so this is not Monte-Carlo noise. Reproduced by finder
(seed 42 DGP) and refuter (independent DGP, seed 777 — which also showed
the incidence is data-dependent: that series was stable, confirming basin
proximity as the mechanism). The card documents the Laplace likelihood's
kink basins ("best basin found, not global optimality") — but nowhere did
any surface say the basin depends on the *units*.

**Fix:** the same standardize-and-map-back repair as garch-F2
(`DcsModel::fit` optimizes on `y / s_rob` and maps back exactly), pinned
bit-exact under power-of-two rescalings for all three densities. Post-fix
incidence on the identical probe: **4/20** (from 11/20) — the residual is
the irreducible rounding of `c * y` itself on a kinked surface, stated on
the card. `t`/`gaussian`: 0/20 before and after.

## Confirmed 1b — the latent Nelder-Mead initial-simplex hole the fix exposed (fixed)

Standardization parks the working `ln(scale)` coordinate at `ln(1) ≈ 0`,
and the DCS *Gaussian* golden fits promptly stalled at their starting
scale with `converged=true` — which turned the brief's open **"Nelder-Mead
x-side mixed-scale" concern from theoretical to realized**. scipy's
initial-simplex rule (copied faithfully) displaces coordinate `i` by
`0.05·x0_i`, reserving the absolute `0.00025` for *exactly* zero: a
coordinate starting at `1e-9` got a `5e-11` simplex edge — smaller than an
exact zero's, and **below the default `x_tol = 1e-8`, so the simplex-size
test held in that direction before the first iteration**. Restarts rebuild
the simplex with the same rule, so they cannot recover it. **Fix:** the
displacement is now `max(0.05·|x0_i|, 0.00025)` — bit-identical to scipy's
vertex whenever `|x0_i| ≥ 0.005` and at exactly zero — with a regression
test sweeping the near-zero band at both signs. All 13 `tsecon-optim`
dependents' crate suites pass unchanged except one garch boundary property
test, extended (not weakened) because the floored simplex now legitimately
reaches interior points on some white-noise draws: every-NaN-is-flagged is
asserted on every draw, and ≥3 boundary fits must still deliver finite
interior SEs.

## Confirmed 2 — `forecasting.md` described `gpd_fit` as unbuilt (fixed)

**`cosmetic`** (lens 5's roadmap-drift class). The card's "when to use
`var_backtest`" bullet pointed at "an EVT tail (the POT/GPD `gpd_fit`
slice scoped build-next on the roadmap, Module 03/E11)" — `gpd_fit`
shipped in 0.3.0, in the same release as `var_backtest` itself. Reworded
to reference the shipped McNeil-Frey POT VaR. (Same class, rolled into the
Part-1 work: the volatility card's expected-output comment printed `nu`
as `8.37` where the snippet prints `8.3708` — corrected while touching the
card.)

## Refuted (kept out, with the evidence)

- **`var_backtest` refuses a zero-violation sequence** — looked like an
  over-refusal (Kupiec is computable there), but the error is a teaching
  refusal that *computes and reports the Kupiec continuity limit*
  (`LR_uc = −2n ln(1−α) = 76.94` at n=750) and explains that the
  independence/DQ statistics are undefined with no violations. Honest,
  actionable, and the card documents it with the same number. Not a trap.
- **`panel_lp` echoes `bias_correction="dj"` as `"dhaene_jochmans"`** —
  candidate non-round-trippable echo; refuted: the long name is an accepted
  input alias (`"dj" | "dhaene_jochmans"` at the binding), so the echo
  feeds back cleanly, and the card lists the canonical spelling.
- **Two probe artifacts, recorded so the next round does not re-derive
  them:** `p_dq` flagged as a "non-DQ key moved by `dq_lags`" (it *is* the
  DQ p-value — the probe's own key classifier was wrong), and
  `proxy_ar_sets` cells "failing" to scale with the data (the sets are
  unit-normalized responses — dimensionless ratios — and the correct
  expectation, invariance, holds exactly; my first probe asserted
  equivariance).

## Swept and found sound

- **`var_backtest`** (26/26): `dq_lags` moves only the DQ block;
  hits+VaR reproduces the returns path bit-for-bit (docstring promise);
  hits-only drops exactly the VaR regressor with honest df; statistics
  scale-invariant over 9 decades to <1e-6; all-violations/zero-violations/
  constant-VaR/NaN/short-sample/bad-`input` all refuse with teaching
  messages (the constant-VaR case runs with the documented rank-aware df
  and says so in the verdict); non-0/1 hit sequences are refused, so a
  forgotten `var_forecasts` cannot silently grade returns as hits.
- **`dcs_local_level`** (32/32 post-fix): density axis alive (three
  distinct fits), unknown density refused; `t`/`gaussian` scale-clean
  before and after; constant/near-constant/short/NaN inputs refused with
  the documented errors; the 1e12-outlier fit pins `nu` at its lower bound
  exactly as the card's failure-modes paragraph describes.
- **`panel_lp(bias_correction=)`** (34/34 + 8): the three modes differ
  pairwise (bit-compare); `jackknife=True` ≡ `bias_correction="dj"`
  **bit-identically**; `dj` keeps the full-sample SEs and `spj` recomputes
  them (both documented); `spj`+`jackknife` raises;
  `se_type="nonrobust"` under `spj` refused as documented; echo keys
  present; joint and outcome-only rescaling exact to 1e-6 for both `none`
  and `spj`; N=1/short-T/odd-T corner cases refuse or run with teaching
  messages (odd-T median split runs, N=1 names the cluster requirement).
- **`proxy_ar_sets(rf_method=)`** (15/15 + 1): `second_order` differs from
  `delta` while the boundedness `kind`s are **bit-identical** (the
  CHANGELOG's claim, verified); `rf_draws`/`rf_seed` under `"delta"`
  **raise** (the `hac_lags` lesson, honored); any `rf_method` with
  propagation off raises; default seed deterministic, seed/draws axes
  alive; odd/small `rf_draws` refused ("even, ≥ 32", teaching); the sets
  are exactly scale-invariant (unit-normalized), kinds stable across 8
  decades. Explicit `rf_method="delta"` with propagation off is accepted
  (indistinguishable from the default — noted, not a finding).
- **`ivx_test(joint=)`** (37/37): the two modes differ; bonferroni
  arithmetic exact (`pvalue = min(1, k·min pⱼ)` to 1e-15, `wald` = max
  scalar); each scalar column equals `predictive_regression`'s ivx block
  to 1e-10 (the docstring's "exactly" claim, verified per column); k=1
  consistency; every statistic invariant under joint/r-only/x-only
  rescaling over 12 decade-cells; unknown `joint` refused with a teaching
  enumeration; collinear/constant predictor columns refused (correctly
  attributed to the collinear cross-product).
- **Doc surfaces and cards**: returned-key sets diffed against `__doc__`
  and each card for all five functions — no documented-but-never-returned
  key anywhere; the cards' key enumerations (var_backtest's full list,
  proxy's cell keys, ivx's mode-dependent extras) match the observed
  returns. Four card snippets with printed expected output re-run on the
  final build: **all reproduce byte-for-byte** (var_backtest's
  GARCH-vs-flat comparison, the DCS contamination two-liner, both IVX
  snippets), plus the volatility garch snippet at its printed precision.
- **The five replication suites**: all five named fixtures exist and are
  loaded by the named test files; `grep -l "import tsecon" fixtures/*.py`
  returns nothing (the circularity tripwire); the CHANGELOG's headline
  claims are pinned by real assertions at the stated tolerances (Hamilton
  +1.16/−0.36 at 0.02/0.03 abs with the statsmodels cross-fit at 1e-3;
  Bai-Perron break dates asserted **exact** as quarter labels, m=3 set and
  the published 90% CIs included; Hansen d=2 exact and threshold asserted
  to *round to* the published 7.4 — the honest formulation; Uhlig's
  acceptance rate banded around the published ~5.9% with the no-price-
  puzzle quantile sign asserted through month 60); 0 skips in the suite
  run (862 passed).

---

## Proposed CHANGELOG entries (0.4.0 — not applied; CHANGELOG is off-limits this round)

- **Fixed — `garch_fit` boundary fits are reported, never silently NaN**:
  per-parameter `se_valid`/`boundary` flags, a `boundary_note`, the
  `converged` flag exposed, interior parameters keep finite SEs from a
  reduced Hessian (was: unflagged all-NaN rows on 24/50 boundary-attracted
  fits). BREAKING only in the additive-keys sense.
- **Fixed — `garch_fit` and `dcs_local_level` estimation is
  scale-adaptive**: internally standardized, mapped back exactly;
  power-of-two rescalings commute bit-exactly; cross-scale disagreements
  93/320 → 0 on unflagged garch fits, 11/20 → 4/20 on the Laplace DCS.
- **Fixed — Nelder-Mead initial simplex floors its edge at 0.00025**: a
  near-zero starting coordinate could begin below `x_tol` and certify
  convergence at its starting value; bit-identical to scipy's vertices for
  `|x0| ≥ 0.005`.
- **Fixed — negative integer arguments raise a teaching `ValueError`**
  naming the function and parameter, centrally at the coercion layer
  (was: bare PyO3 `OverflowError`, library-wide).

## Proposed validation-matrix entries (not applied; matrix is off-limits)

- `garch_fit` boundary flags: white-noise/IGARCH property tests
  (`crates/tsecon-garch/tests/properties.rs`), Python battery
  (`bindings/python/tests/test_garch_boundary.py`) — internal properties,
  no external reference exists for boundary-fit reporting conventions.
- `garch_fit`/`dcs_local_level` scale commutation: bit-exact
  power-of-two same-run invariants + 1e-6 decade agreement
  (`test_garch_scale.py`, `test_dcs_scale.py`, crate property tests).

## Reproducing

Probe scripts were scratchpad-only (`scratchpad/round7/`): the 50-unit
boundary battery (`probe_a_boundary_nan.py`/`probe_a_post.py`), the two
cross-scale batteries with mapped-parameter comparison and flag-coverage
counting (`probe_b_scale.py`, `probe_b_stats.py`, `probe_b_flags.py`), the
DCS incidence probe with its t-density negative control
(`p2_dcs_incidence.py`), and one lens-1–3 probe per new surface
(`p2_*.py`), each printing comparisons attempted vs made. The generative
designs worth rebuilding: boundary-attracted GARCH DGPs (IGARCH truth,
tiny-alpha, white noise) at T=750, and the contaminated-level DCS DGP
(5% additive 8-sigma outliers) whose seeds land near Laplace kink-basin
boundaries.
