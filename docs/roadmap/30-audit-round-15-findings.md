# Adversarial audit, round 15 — findings

> **Working document.** Continuation of
> [round 14](29-audit-round-14-findings.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 15 is the pre-merge sweep over the **0.11.0 batch-1 surface**. It
covers the ten new callables (`variance_ratio`, `breusch_godfrey_test`,
`durbin_watson_test`, `panel_re`, `panel_between`, `panel_fd`,
`panel_fama_macbeth`, `ewc_lrv`, `ewc_default_b`, `ardl_bounds_test`). It
also covers the extended `ols(se_type="ewc")`, `var_forecast` (`cov`,
`param_uncertainty=`), `unobserved_components` (`autoregressive=`,
`cycle_init="stationary"`, `mle_regression=False`) and the shared
input-coercion layer's wrong-type message. That is 202 public callables at
`3431249`, the head of PR #30 when the round ran. The wheel under test was
built from that head and installed into a venv with the reference libraries
the fixtures pin (numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, arch 8.0.0,
linearmodels 7.0, scikit-learn 1.9.1, pandas 3.0.6, mapie 1.5.0). The repro
script of every finding is committed under
[`lab/audit/round15/`](../../lab/audit/round15/README.md).

Every feature had already been reviewed adversarially once, by a separate
agent that fixed what it found before the branch was integrated. This round
looks for what those per-feature reviews could not see: disagreements that
appear only on random designs, interactions between features, and claims
measured once and then quoted.

**Design.** Six finder lenses, each blind to the others. Every candidate
went to an independent refuter, told to default to *refuted* unless it
could reproduce the defect and show that nothing in the docstring, the
card or the known-issues list already accounts for it.

- **Fuzz A — differential fuzzing against arch and statsmodels.**
  `variance_ratio` against `arch.unitroot.VarianceRatio`: 1,400 randomized
  calls (about 3,000 horizon comparisons, n from 5 to 10,000, every
  option, scales 1e-8..1e8) plus 940 boundary cases. `breusch_godfrey_test`
  against `acorr_breusch_godfrey`: about 1,730 calls. `ardl_bounds_test`
  against statsmodels `UECM`/`ARDL`: about 895 random cases (T 25..2000,
  k 1..11, cases 1-5, orders 1..5).
- **Fuzz B — differential fuzzing against linearmodels, statsmodels and
  closed forms.** The panel menu against `linearmodels` `RandomEffects` /
  `BetweenOLS` / `FirstDifferenceOLS` / `FamaMacBeth`: 370 randomized
  designs with ragged masks and every `se_type`, which is 15,210 fits, plus
  1,351 edge fits. Also 2,400 `ewc_lrv` and 500 `ols(se_type="ewc")` cases
  against an independent transcription of the Lazarus-Lewis-Stock-Watson
  estimator, 1,400 `var_forecast` calls against statsmodels'
  `forecast_cov` / `mse`, and `unobserved_components` against statsmodels
  at fixed parameters.
- **Hostile — malformed-input sweep.** 1,488 calls plus about 230
  targeted probes, each in a child process under a 4 GB `RLIMIT_AS` cap
  with a 60-90 s deadline. The inputs were NaN/±inf, empty, tiny, the
  wrong rank, mismatched shapes, wrong dtypes, pandas with non-default
  indices, 2^63/2^64/-1 counts, unknown option strings, absurd sizes and
  inert keyword combinations.
- **Docs truth.** All 16 new or changed card examples were run. Every
  quoted number in the new card sections, the CHANGELOG `[Unreleased]`
  bullets, the validation-matrix rows and the ROADMAP status paragraph was
  checked against the committed artifact that should produce it.
- **API consistency.** Signatures against the stub, returned keys against
  both docstrings across 39 option variants, `json`/`pickle` round trips,
  `tsecon.summarize`, pandas against numpy inputs, and calls from 4 threads.
  Names and conventions were compared with the neighbouring callables.
- **Statistical claims.** Every size, power, coverage and recovery figure
  was re-measured with fresh seeds and far more replications (for example
  2 seeds × 20,000 replications per variance-ratio DGP, 1,000,000 null
  draws per exact-Durbin-Watson design, 8,000/4,000 size/power replications
  per Breusch-Godfrey cell).

**Outcome.** 36 candidates. **9 refuted, 27 confirmed: 1 severe, 11
moderate, 15 minor.** All 27 are fixed, each with a regression test, and
each fix went through a second adversarial review before integration.
That review amended 11 of the 27 fixes:
- in three (F1, F3, API15-2) the first version had introduced a regression
  of its own;
- in four (B15-UC-1, B15-EWC-1, F2, SWEEP-7) it was incomplete or its
  golden was mis-built. The **severe** finding is not a regression: it is a
defect in the exact-diffuse Kalman filter that shipped in 0.10.0 and
earlier, which the wave's new options made easier to reach.

## Severe

### B15-UC-1 — the exact-diffuse filter refused genuine diffuse observations

*Surface:* every model on the shared exact-diffuse Kalman filter
(`crates/tsecon-ssm/src/filter.rs`): `unobserved_components`,
`tvp_regression` and the rest. *Found by:* Fuzz B.

The diffuse-branch test compared `F_inf` against
`TOLERANCE_CANCEL · |Z|' P_mag |Z|`, where `P_mag` is a cancellation-free
magnitude bound carried forward as `|T| P_mag |T|'`. Under the
dummy-seasonal transition `|T|` has spectral radius near 2. The bound
therefore grew about 4× per step, although `P_inf` itself does not grow.
In a long diffuse period a genuine `F_inf` was eventually refused: at
t = 26 a true `F_inf = 1.09` was compared against 204. `P_inf` then never
washed out, `nobs_diffuse` became the whole sample, `std_resid` was NaN
everywhere, and the log-likelihood, the smoothed states and the MLE were
silently wrong.

Two same-calendar-month missing values were enough. Local level plus
seasonal 12 with NaN at [2, 14] returned −187.051 with
`nobs_diffuse = 120`, against statsmodels' −184.181 with 27. Every
[2, 14] case in 60 seeds failed. The reviewer found a second common
trigger: a year or more missing at the *start* of a monthly series. Under
the wave's `mle_regression=False` it was reachable with no missing values
at all, for example local linear trend plus seasonal 12 plus two N(0, 1)
regressors.

*Fix.* The element test now reads `F` from a square-root factor `B` of
the diffuse covariance (`B ← T B`). Resolved directions are annihilated by
orthogonal projection, and the remaining magnitude is compared against
`|z|'|U||z|`, with `U` carried by the real transition `U = T U T'`. The
projection algebra reproduces the covariance downdate exactly, the update
arithmetic is unchanged, and every 0.10.0 pin holds, including
`defaults_are_bit_identical_to_0_10_0`.

*Evidence.* The reviewer wrote an independent 90-digit Decimal exact-diffuse
filter on statsmodels' system matrices (`scripts/uc_exact_diffuse_reference.py`)
and compared 1,200 fresh designs against it. The designs covered
same-period NaN runs for seasonals 4, 7 and 12, trigonometric seasonals,
leading-NaN runs, state regression, AR and cycle components. The old
filter got 336 wrong (295 `nobs_diffuse` mismatches, log-likelihood errors
up to 2.9e3). The new one gets none wrong outside one refused weak design
and the period-6 Nyquist harmonic, where exact arithmetic on the double
`sin(π)` is the artefact and tsecon agrees with statsmodels. Elsewhere the
largest gap to statsmodels is 2e-8 relative, and `nobs_diffuse` always
matches.

*Pins.* New statsmodels goldens in `fixtures/uc_extensions.json`
(`long_diffuse`):
- repeated same-month NaNs, fixed and stochastic seasonal, with and
  without `autoregressive=1` (`nobs_diffuse` 27 and 39);
- the two leading-gap cases (`nobs_diffuse` 24 and 29);
- the state-regression designs from the report.

These are checked by `uc_extensions_golden.rs` and `test_uc_extensions.py`.

## Moderate

- **B15-UC-2** — `unobserved_components(exog=..., mle_regression=False)`
  refused a unit-scale N(0, 1) regressor as "columns' scales too far
  apart". That was B15-UC-1 misfiring on one of the two runs the
  regressor-scale guard compares, and rescaling could not help. About 1%
  of local-level + seasonal-12 + one-regressor designs were refused. With
  the root cause fixed, 1,500 fresh seeds across five designs give 0
  refusals. The units message now fires only for column scales more than
  a factor of 100 apart. Otherwise the refusal says the units are not the
  cause and points to dropping the weakly seen column or to
  `mle_regression=True`.
- **F1** — `ardl_bounds_test`, case 2. The bounds F lost accuracy under a
  level offset: 22% off at level/increment ratio ~6e6, and 9.5% off on a
  fuzz case at ~1e4. At larger offsets valid data was refused with a false
  "exact linear combination" message. The cause was an explicit inverse of
  an ill-conditioned covariance block in the Wald form, which statsmodels
  avoids. The F is now the SSR difference from one Householder QR with the
  restricted columns last. Under a constant, every other restricted column
  is centred first (an exact change of basis), and no inverse is formed.
  On the auditor's design the F is 12.93501798 at every offset up to 1e6
  (relative error ≤ 1.5e-15). Over 900 fresh fits against an
  exact-rational referee (level/sd ratios up to 1e11) the worst F error is
  1.4e-14 with no refusals; before, case 2 was off by up to 2.18 relative
  with 25 refusals. The 121 fixture cases now agree with the exact F to
  5.5e-14, against 5.7e-11 before. The reviewer caught a regression in the
  first version of the fix: removing the inverse also removed the only
  guard against data near 1e±160, which then returned NaN or infinite
  statistics in 247 cells of an 18×18×5 scale grid. An explicit scale check
  now refuses those by name.
- **F2** — `breusch_godfrey_test`. A design that comes close to spanning
  a constant without spanning it fell into a band where every `nlags` was
  refused, with a message blaming the lagged residuals. The auxiliary
  regression now adds the constant as `M_X 1` (power-of-two rescaled)
  instead of a literal column of ones. A new `conditioning_cases` golden
  is checked against exact rational arithmetic. In review, the generator's
  one-ulp sensitivity turned out to use `1 ± 2^-53`, which rounds to 1
  half the time; it now uses `np.nextafter`.
- **SWEEP-1** — `breusch_godfrey_test` had no size or work guard on
  `nlags`, so a valid large order aborted the process (SIGABRT). A memory
  budget (2^26 entries) and a work budget (2^34 multiply-adds) are now
  checked before any lag column is built. The refusal names `nlags`, the
  budget that tripped and the largest order that fits.
- **SWEEP-2** — Driscoll-Kraay accepted any finite bandwidth. Because the
  per-period score sums add up to `X'e = 0`, each standard error at
  `bandwidth + 1 ≥ S` (S periods carrying scores) is its edge value times
  `sqrt(S / (bandwidth + 1))`, so standard errors shrank silently and
  reached 0 (t = inf) at huge bandwidths. Every Driscoll-Kraay path now
  refuses `bandwidth + 1 > S`, naming `bandwidth`, and says when the value
  was the default. linearmodels raises at the same point, and the edge is
  pinned against it. **This reaches released callables:** a default
  `panel_lp` call raises when a horizon leaves fewer than five periods (any
  `T ≤ 14` at the defaults), where it used to return shrunk standard
  errors. The CHANGELOG records it under both Changed and Fixed.
- **SWEEP-4** — `panel_re` refused an entity-constant outcome by blaming
  the regressors, because its within-fit guard compared rounding noise
  with rounding noise. The outcome's within variation is now checked
  against its own scale, and the refusal names `outcome` and points to
  `panel_between`.
- **SWEEP-8** — the `var_forecast(param_uncertainty=True)` budget refusal
  named `steps` and quoted a cost formula that left out the lag-driven
  terms that actually trip it. The message now names the driver and gives
  the full formula (`m = n_trend + k·lags`).
- **DOC15-1** — the Durbin-Watson card still said a long sample is
  refused, but the integration commit made the statistic available above
  T = 4000 with `pvalue = None`. The failure-modes paragraph now says so.
- **DOC15-2** — two precision figures in the ROADMAP status paragraph,
  written by the integrator, had no committed artifact behind them. They
  now cite what the fixtures record: the documented-formula golden for the
  exact Durbin-Watson p-value, and the 1.04e-15 relative agreement in
  `fixtures/var_forecast_cov.json`.
- **API15-1 / SWEEP-7** (one root cause) — the coercion layer's
  wrong-type message still blamed innocent integer counts whose default is
  a `None` sentinel, including every ARDL count and UC's new
  `autoregressive=`. Blame is now decided by the type the shipped stub
  declares. In review, NumPy scalars (`np.int64(3)`) were found to be
  named by their own type and were also mapped.
- **API15-2** — `tsecon.summarize` labelled any vector with a name list
  whenever the lengths matched, so in the basic ARDL(1,1) case the bounds
  were printed under coefficient names. Labelling is now by explicit
  pairing. The review scanned 27 representative calls and restored the
  legitimate labels the first version dropped: per-parameter
  `se_valid`/`boundary` flags, `tvp_regression`'s per-coefficient
  variances, and ETS `final_states`.

## Minor

- **F3** — `variance_ratio` refused a float horizon as a scalar but
  accepted it inside a sequence. Horizons now follow one integer rule in
  both forms, naming `lags[j]`. Taking `lags` out of the coercion layer
  briefly let a DataFrame be read by its column labels and a
  bytearray/memoryview as small ints. The review closed both: anything with
  `ndim > 1` and any binary buffer is refused. The integrator then found
  the same DataFrame hole in `ardl_bounds_test(order=)` and closed it the
  same way.
- **B15-EWC-1** — `ewc_default_b(n)` was off by one just above
  half-integer ties for large n, so it did not equal `round(0.4 n^(2/3))`.
  It is now exact. The review found that the golden's "closest to a tie"
  selection picked only cases above a tie. It now pins the ten closest on
  each side in 1e6..3e10, plus the auditor's cases.
- **SWEEP-9** — a mistyped `se_type` together with `ewc_b` was reported as
  an inert `ewc_b`. The unknown `se_type` is now reported first.
- **SWEEP-10 / API15-3 / API15-4** — the variance-ratio stub said `lags`
  comes back as a 1-D array, but it is a list (as the runtime docstring
  said). The stub annotation also rejected the documented integer-array
  input under `mypy --strict`. Both are fixed.
- **API15-6** — `ardl_bounds_test(order=b"3")` was read as order [51]. It
  is now refused like `order="3"`, as are `bytearray` and `memoryview`.
- **DOC15-3** — the testing page's test, file and fixture counts were
  incremented by hand per branch and had drifted. They are recounted once
  at integration from one logged run.
- **DOC15-4** — a Breusch-Godfrey size range in the CHANGELOG (4.5–5.1%)
  did not match its committed output (4.6–5.1%).
- **DOC15-5** — the validation matrix called two new fixtures "the first
  to record statsmodels 0.15.0"; eight earlier ones already do.
- **DOC15-6** — the opening paragraphs of the `var_forecast` and `ols`
  docstrings had gone stale and disagreed with the stub.
- **DOC15-7** — the ARDL docstring said the response-surface p-value at a
  tabulated critical value is within 0.001 of nominal. It is within
  0.0013 at 90%, 0.0010 at 95%, 0.0003 at 99% and 0.0001 at 99.9%, now
  stated and recomputed by a test from the fixture.
- **EWC-PRECISION-1** — the EWC coverage table's Monte Carlo standard
  error was computed at the nominal 0.95, not at the measured coverage,
  and two cells did not reproduce at the stated precision. Both corrected.
- **UC-CYCLE-1** — stationary-cycle recovery was quoted to four decimals
  from 8 seeds, and the median damping error happened to be about half the
  typical value. Now quoted from 400 seeds
  (`scripts/mc_uc_stationary_cycle_recovery.py`), with standard errors
  and the misses named.

## Refuted

Each of these reproduced as described. They were refuted because the
behaviour is documented, matches the reference, predates the wave outside
its surface, or is ordinary Monte Carlo noise at a replication count the
card states.

- **B15-VAR-1** — a singular `Σ_u` from too few observations for the lag
  order is blamed on duplicated series. Real, but pre-existing in
  `var_fit`, which the wave does not touch. Recorded under Open.
- **SWEEP-3** — Fama-MacBeth and between-estimator t-statistics on an
  exactly fitted outcome are rounding noise. linearmodels `BetweenOLS`
  returns the same kind of noise, and its `FamaMacBeth` raise is an
  incidental `ZeroDivisionError`, not a guard.
- **SWEEP-5** — wide Durbin-Watson designs are slow. Almost all of the
  time is the OLS fit that every call runs. The part the 4,000-observation
  cap bounds, the exact p-value, grows only modestly with k.
- **SWEEP-6** — collinear regressors under `mle_regression=False` leave
  a diffuse period that never ends. The card documents this exact
  behaviour, and every number matches statsmodels.
- **API15-5** — `variance_ratio` accepts the integer-valued float array
  `[4.0]` but refuses the scalar `4.0`. This was the deliberate,
  test-pinned convention at the time. F3 has since unified the rule in the
  stricter direction.
- **VR-SIZE-1, PANEL-PETERSEN-1, ARDL-SIZE-1, BG-NOCONST-1** — the quoted
  sizes and coverages reproduce exactly at their committed seeds and
  replication counts. The fresh-seed values differ by Monte Carlo error at
  the precision the cards state. The software decisions agree with the
  references case by case: 0 disagreements in 12,000 variance-ratio
  decisions per DGP against arch.

## Open — recorded, not fixed

- Weakly resolved or near-collinear state-space designs remain
  ill-conditioned in double precision, in the old filter, the new one and
  statsmodels alike. Examples are regressors collinear to 1e-6 under
  `mle_regression=False`, a near-constant regressor beside a level, and a
  diffuse cycle at frequency ≤ 1e-4. After a weakly resolved element, the
  covariance-form washout test can keep `nobs_diffuse` open well past the
  exact value. A rank-count period end would weaken the state-regression
  guard on near-collinear designs, so it is not done here. The card says
  so qualitatively and recommends `mle_regression=True`.
- Under `mle_regression=False`, a regressor with `max|x| ≳ 1e155` ends in
  an unnamed `NoInformation` refusal. This predates the wave.
- In ARDL, data near 1e±155 with a constant is refused as numerically
  singular. This is on the known list. With `x` alone near 1e-155,
  `long_run_se` can overflow to inf while every other output stays finite.
  It is left out of the scale guard deliberately, because a near-zero `π`
  can legitimately give a huge long-run standard error.
- `arch_lm(nlags=)` and `adf(maxlag=)` have the same unguarded-allocation
  abort that SWEEP-1 fixed for Breusch-Godfrey. They predate the wave and
  are out of its surface.
- B15-VAR-1's misleading `Σ_u` message (above).
- The PSS (2001) published 5% bounds carried in `fixtures/ardl.json` are
  marked `verified_against_pdf=false`; egress to the paper was blocked.
  Web reprints confirm two rows.

## Verification record

After the 27 fixes were merged onto the integration branch:

- `cargo fmt --check`, the notices and inventory check, and
  `cargo clippy --workspace --all-targets -D warnings`: clean.
- Each fix branch passed `cargo test -p` over every crate it touched and
  every dependent. The filter fix ran `tsecon-ssm`, `-arima`, `-bayes`,
  `-nowcast` and `-ident`: 49 test binaries.
- Each fix branch also passed the full Python suite twice, in a full venv
  and in a minimal one mirroring the CI wheel jobs (only numpy, scipy and
  pytest).

The integrated run's totals are recorded on the
[testing page](../reference/testing.md).

## Lessons

- **A cancellation bound must not be carried through `|T|`.** A
  magnitude bound transported by the absolute transition grows like
  ρ(|T|)^2t even when `T` preserves norms, as rotations and seasonal
  dummies do. Any guard built on such a bound has a horizon after which it
  refuses true signal. The square-root form makes resolved mass
  structurally zero, so no threshold has to separate dust from signal.
- **Fuzz against the reference, not only against the fixture.** Every
  severe or moderate numerical finding (B15-UC-1, F1, F2) passed its
  fixture and its per-feature review. Each surfaced only on randomized
  designs compared with an independent implementation, and in two cases
  with exact arithmetic.
- **A fix's review is a second audit.** The second adversarial pass
  amended 11 of the 27 fixes before integration. Three of them had
  introduced a regression of their own: F1's lost scale guard, F3's
  DataFrame and buffer holes, and API15-2's dropped legitimate labels.
- **Test where CI tests.** Three CI failures on PR #30 came from
  environment differences the per-feature checks could not see:
  - a module-level `import pandas` in a CI job that does not install it;
  - NumPy 2.0's `np.bool_.__index__` on the Python 3.9 leg;
  - macOS arm64 rounding amplified by an ill-conditioned diffuse period.

  The integration checklist now runs the suite in minimal Python 3.9 and
  3.13 venvs before every push.
