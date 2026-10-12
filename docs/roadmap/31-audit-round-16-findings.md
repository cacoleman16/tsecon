# Adversarial audit, round 16 — findings

> **Working document.** Continuation of
> [round 15](30-audit-round-15-findings.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 16 is the pre-merge sweep over the **0.11.0 batch-2 surface**. It
covers the four new callables (`long_run_svar_bands`,
`multivariate_normality`, `dfm_smooth`, `adl_midas`) and the extended ones:
`long_run_svar(impact_signs=)`,
`historical_decomposition(identification="impact")`,
`structural_fevd(impact=<nested list>)`, the Doornik-Hansen keys of
`var_diagnostics`, `var_conditional_forecast(condition_sd=, soft=)`,
`arima_fit(method="css")` with the new `ARIMAResults.nobs`, and the engines
`var_irf_bands` now shares with `long_run_svar_bands`. It also ran the
first **house-wide abort/hang sweep** over all 206 public callables. The
round ran at `fdd90d3`, the head of PR #30 after batch 2 was integrated,
with the wheel built from that head in a venv holding the reference
libraries (numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, arch 8.0.0,
linearmodels 7.0, scikit-learn 1.9.1, pandas 2.3.3, statsforecast 2.1.1).
The repro script of every finding, and the sweep harness itself, are
committed under [`lab/audit/round16/`](../../lab/audit/round16/README.md).

**Design.** Seven finder lenses, each blind to the others. Every
candidate went to an independent refuter, told to default to *refuted*
unless it could reproduce the defect and show that nothing in the
docstring, the card or the known-issues list already accounts for it.

- **Fuzz SVAR — differential fuzzing of the structural-VAR additions.**
  700 random `long_run_svar` cases (k 2..5, p 1..8, companion radius up to
  0.9995, column scales up to 1e±7, permuted restriction patterns) against
  an independent NumPy Blanchard-Quah map; 400 delta-arm cases against a
  complex-step Jacobian composed with statsmodels' asymptotic covariance,
  settled by a 60-digit mpmath Jacobian where they disagreed; 677
  bootstrap cases (250 Kilian) against a replay of the documented RNG
  stream; 330 `historical_decomposition(identification="impact")` cases
  against the closed form; and 400 `var_irf_bands` cases bit-compared with
  the 0.10.0 wheel.
- **Fuzz stats — the statistical and state-space additions.** 1,311
  randomized and 400 degenerate `multivariate_normality` cases against
  NumPy and 60-digit mpmath transcriptions of Doornik-Hansen (2008) and
  both Jarque-Bera forms; 800 soft-condition cases against the Gaussian
  closed form and 36 against statsmodels' VARMAX smoother; 600 ragged
  `dfm_smooth` panels against statsmodels `DynamicFactor`; 434
  `adl_midas` cases against statsmodels OLS/HAC; 496 random SARIMA CSS fits
  against statsforecast's `arima_css`.
- **Hostile — malformed-input sweep of the batch-2 surface.** 1,708
  subprocess calls under a 4 GB address-space cap and a 90 s deadline,
  plus about 190 targeted probes: non-finite and empty inputs, every rank
  and dtype, pandas with reversed indices, strided views, ragged lists, every
  integer argument at 2^31…2^64 and −1, unknown option strings, and every
  inert keyword combination.
- **House sweep — abort/hang sweep of all 206 callables.** A valid
  baseline call was built for every callable (278 variants, none skipped),
  then each of 521 integer- or size-like parameters (including the
  components of list, tuple and dict arguments) was set to 2^31, 2^40,
  2^62, −1 and 0, one call per process under `ulimit -v 4000000` and a 60 s
  deadline: 2,710 calls.
- **Docs truth.** Every new card example re-run, and every quoted number
  in the batch-2 cards, the Blanchard-Quah page, guide chapter 07, the
  CHANGELOG, the validation matrix and the ROADMAP checked against the
  committed artifact that should produce it.
- **API consistency.** Neighbouring-callable conventions, `json`/`pickle`
  round trips, `tsecon.summarize` and the results classes, pandas against
  numpy, determinism across 4 threads, and `mypy --strict` on every
  batch-2 signature.
- **Statistical claims.** Every size, power and coverage figure
  re-measured with fresh seeds: the `long_run_svar_bands` coverage tables
  on 428 cells per design, the Doornik-Hansen sizes, the StepM
  family-wise-error study, the soft-condition distribution claims.

**Outcome.** 59 candidates. **10 refuted, 49 confirmed: 14 severe, 22
moderate, 13 minor.** The batch-2 lenses found two severe numerical
defects, both older than the wave and reached through it: unbalanced
companion eigenvalues (SVF-1) and roundoff in the shared univariate Kalman
filter (DF16-1). The house sweep found the largest class of defects any
round has: **136 calls that killed the interpreter and 12 that hung**
(56 callables abort, 4 hang; 79 callable-parameter pairs), almost all of
them outside batch 2. Counts between 2^31 and
the coercion layer's 2^48 pre-flight reached Rust with no budget, and
Rust's allocation-failure handler aborts the process.

All 49 are fixed, together with the pre-existing items that round 15 and
the batch-2 reviews had queued (`garch_fit(forecast_horizon=2**36)`, the O(H²)
asymptotic `var_irf_bands`, `seed=2**64`, unbudgeted ARMA orders, the
`arch_lm`/`adf` allocation aborts, B15-VAR-1's misleading `Σ_u`
message). The fixes were split into eight crate-scoped groups. Each fix
was followed by a second, adversarial review that re-ran the repro,
probed both sides of every new boundary and checked an independent
reference. **The reviews amended the fixes 51 times**; the most important
amendments are named below.

## Severe

### SVF-1 — VAR roots depended on the units of the series

*Surface:* every eigenvalue solve on a companion matrix:
`var_fit`'s `min_root`/`is_stable`, `var_diagnostics`, the
`n_unstable_draws` of both bootstrap callables, and the Kilian
stationarity shrink (`delta -= 0.01` until the radius is below 1) shared by
`var_irf_bands` and `long_run_svar_bands`. *Found by:* Fuzz SVAR.

`tsecon_linalg::spectral_radius` and `VarResults::roots_moduli` called
faer's real-Schur eigenvalues on the raw companion matrix. A QR iteration
is accurate relative to the matrix norm, and rescaling a series by `s` is
a diagonal similarity that leaves the roots unchanged but spreads the
companion's entries over the square of the unit spread. With three series
in units `[1e7, 1, 1e-7]` (realistic: a level in currency units beside a
rate in decimals), a VAR whose exact minimum root is 1.000887 reported
0.997715 and `is_stable=False`; `n_unstable_draws` was 90 against 47, and
the Kilian bands moved by 1.25e-2 relative. The OLS coefficients
themselves were equivariant to 1e-15; LAPACK's unbalanced `dgees` drifts
the same way, and numpy's balanced `dgeev` is exact.

*Fix.* One LAPACK `dgebal`-style balancing (a diagonal similarity by
powers of two, exact in floating point) in `tsecon_linalg::balance` /
`balanced_eigenvalues`, used by every such site, including the JSZ
starting eigenvalues. *Evidence.* On 40 fresh VARs with unit spreads
1e3..1e8, `min_root`/`max_root` agree with `dgeev` on tsecon's own
coefficients to 1.4e-14 with no stability disagreement; the base wheel
was off by up to 1.87 relative with 7 wrong verdicts. The Kilian bands
move at most 7.5e-15 under rescaling (5.3e-3 before). 190 valid-call keys
are bit-identical to the base wheel, and `echo_state_network` moves at
1e-13 (its reservoir radius comes from the balanced solve).

### DF16-1 — roundoff entered the DFM log-likelihood

*Surface:* `dfm_smooth`, and every multi-series model on the shared
univariate Kalman filter of `tsecon-ssm`. *Found by:* Fuzz stats.

When more than `r` series with zero idiosyncratic variance are observed in
one period, the sequential update pins the factor directions with the
first `r` of them. The rest have an exactly zero innovation variance
`F_*`, but in floating point a residue of order 1e-17 remains, and the
filter treated that "dust" as an informative observation. The smoothed
factor was right to 3.5e-15, but `-0.5·(ln F + v²/F)` of the dust entered
the log-likelihood: 18.657 where statsmodels and the documented
skip-a-singular-element rule give −1.737, and gaps of up to 29,500 points
on fuzz panels. The test compared `F_*` against a scale read off the
already-collapsed `P_*`, so both sides were roundoff. Against a dense
Gaussian reference, 93 of 160 fresh panels were wrong.

*Fix.* `F_*` is now measured against the period's predicted `P_*`
instead of collapsing within the period. **The review found the first
version incomplete**: a pin that is itself nearly redundant (two
zero-variance series whose loadings are 0.02 rad apart) leaves dust of
order eps/θ that cleared the test, and 40 of 120 column orders came out
115 to 839 log-likelihood points high. An `F_*` within `SUSPECT_DUST` of
the reference after an earlier update in the period is now re-read off a
square-root companion (a pivoted Cholesky root of the predicted `P_*`,
replayed through the period's updates with Potter downdates). The
companion only makes the decision, and it is built only in periods that
contain such an element. *Evidence.* 0 of 160 fresh panels wrong (worst
relative error 6e-11), 0 of 40 at every angle down to 1e-3 rad, and every
ssm/UC/tvp/nowcast golden holds.

### The house-wide abort class (A1–A10, H16-01–H16-03)

*Surface:* the 56 callables that aborted and the 4 that hung in the house sweep;
H16-01–H16-03 are the batch-2 instances. *Found by:* House sweep and
Hostile.

The coercion layer refuses counts at 2^48 or above and negatives. Between
2^31 and 2^48 a count reached Rust with no budget, so a `horizon`,
`n_draws`, `n_boot`, `band_n_sim`, grid size, forecast horizon, iteration
count or output length sized an allocation directly: 16 GB for one f64
vector, 120 GB for an IRF cube, 155 GB for one RNG stream per draw. When the
allocator refuses, Rust's `handle_alloc_error` **aborts the interpreter**
(`memory allocation of 120259084344 bytes failed`, SIGABRT), and when it
accepts, the process takes the machine. The classes were:

- **A1** `horizon` in 22 VAR/SVAR callables (the MA/IRF arrays);
- **A2** `horizons` in seven local-projection callables;
- **A3** forecast horizons in `garch_fit`, `ccc_garch`, `dcc_garch`,
  `gas_volatility`, `theta_forecast` and `conformal_forecast`;
- **A4** draw, bootstrap and replicate counts in 15 callables, because
  `SeedSequence::spawn` materialised one stream per draw before any
  check (2^40 gave an unnamed spawn-limit error);
- **A5** `band_n_sim` in the four sup-t band builders;
- **A6** grid, path and network sizes in seven ML and regime estimators;
- **A7** output lengths in five utilities, and `acm_term_premium`
  maturities;
- **A8** iteration counts that hung (`stl`/`mstl`, and the narrative
  sampler's `n_weight_draws`);
- **A9** `reset_test(max_power)`, which built power columns until memory
  ran out;
- **A10** `mlp_regression(hidden=[2**62])`, which escaped the pre-flight as
  a list and raised a `PanicException` (a `BaseException`, so
  `except Exception` did not catch it);
- **H16-01–H16-03** the batch-2 instances: `long_run_svar` and
  `structural_fevd` horizons, the `var_irf_bands` bootstrap working set
  (`n_boot=10**7` panicked in a worker thread and then hung), and a
  `long_run_svar_bands` budget that under-counted real storage about 2x.

*Fix.* Every such argument is now checked against a budget **in the
shared engine**, with checked arithmetic, before anything is allocated.
The refusal names the Python keyword and states the largest value that
fits at the other arguments. The budgets follow one idiom: a 2^28-value
(2 GiB) working set counted as real storage (padded matrices and the
Python lists the binding returns included), with a work cap where the
cost is superlinear (2^36 for the sup-t simulation, 2^28 for the
quadratic delta-method Jacobians, 2^37 for an ARMA likelihood). The
1,000,000-step forecast cap that ARIMA and ETS already had now covers the
GARCH, GAS, Theta and conformal forecasts. `tsecon-rng` derives each
replicate's stream lazily inside its task, with a typed refusal above 2^24
materialised streams, so no seeded output changed.

Each reviewer ran both sides of every new boundary in a capped
subprocess: the quoted largest value must run, the next must be refused
by name. That check found counts the first fixes under-counted. The inner
Kilian bootstrap cloned each replication's coefficients into matrices of
their own, which fragmented the heap to 2.8x the count, so
`var_irf_bands(bias_correct=True)` aborted at its admitted maximum. `cv_splits`
counted the Rust indices but not the Python ints and dicts it returns, so
its admitted maximum aborted. `zivot_andrews`' fixed-lag break-date scan
had no budget at all. The `mlp_regression` validation split left
`max_epochs` unbudgeted. And EnbPI's quoted maximum ran past 290 s
because of a cache-hostile loop order. All of them are fixed.

## Moderate

- **SVF-2 + B15-VAR-1** — a bootstrap pseudo-sample whose refit residual
  covariance is singular aborted the call with a message blaming the
  user's data. It is now reported by replication index. **The review
  found the first version platform-dependent**: a resample with at most k
  distinct residual rows has an exactly singular `Σ_u`, but its Cholesky
  catches that only by rounding, and 313 of 400 such pseudo-samples
  passed, so which replication was named, or whether a meaningless draw
  was kept, depended on the floating-point kernels. Such a resample is now
  recognised by counting its distinct rows, and the replication index is
  pinned against an independent replay of the index stream. The inner
  Kilian bootstrap and `var_irf_bands(orth=False)` no longer fail on a
  `Σ_u` they never read. A sample with fewer residual degrees of freedom
  than series is now refused for what it is (B15-VAR-1), naming the rows
  needed; `var_select_order` names `max_lags`.
- **DF16-2 + H16-11** — the shared OLS engine (`ols`, `umidas`,
  `adl_midas`, `lp`, `har_rv`, `predictive_regression`,
  `breusch_godfrey_test` and the other least-squares callables) solved
  the normal equations by Cholesky, which squares the condition number:
  `adl_midas` on a persistent level target missed its documented 1e-10
  agreement with statsmodels, and finite inputs near 1e77 returned NaN
  standard errors. It now solves by blocked Householder QR of the
  power-of-two-equilibrated design (resolving a `TODO(phase0)`). Against
  a 60-digit mpmath reference with cond(X) up to 1.5e8, the worst
  parameter error fell from 1.2e-5 to 5.7e-14 and the worst HC error from
  1.5e-2 to 8.4e-13; every golden holds, and tsecon is now below
  statsmodels' own error on every audit design.
- **H16-04** — `multivariate_normality` had no size or work cap (an
  11.4 MiB residual matrix ran 110 s with the GIL held). It validates
  once and is budgeted. The review found the first budget under-counted
  faer's column padding (the largest admitted shape asked for 5.7 GB) and
  replaced the copy with an unpadded buffer, bit-identical.
- **H16-05** — `dfm_smooth` factored its diagonal `N × N` observation
  covariance through a dense jittered Cholesky, an unpriced O(N³) one-off:
  N = 8192 took 18-20 s and 3.1 GB. A nonnegative diagonal is now
  accepted by inspection; the same call takes 0.9 s at 550 MiB.
- **H16-06** — complex `dfm_smooth` loadings and soft-condition arrays
  were silently truncated to their real part. One real-numeric gate now
  refuses them by name. The review found complex pandas objects still
  truncated by the input layer, which now hands them on uncast.
- **SVF-5** — an object-dtype `impact` was refused as an ndarray but
  accepted as a DataFrame. The input layer now casts a pandas object to
  float64 only when its values are real numbers.
- **SVF-3** — `normalize="long_run"` was documented as making the
  long-run diagonal positive; under a permuted `restrictions` pattern it
  makes each shock's long-run pivot positive. The card, docstrings and
  refusal now say so.
- **H16-08 / API16-3 / API16-8** — `long_run_svar(lags=0)` and
  `structural_fevd(lags=0)` named engine internals (`coefs`, "a BVAR").
  Both name `lags`.
- **H16-09 / API16-2** — `structural_fevd` accepted `sigma` silently, and
  unvalidated, whenever `impact` was given. It is now refused by name.
- **API16-1** — `var_irf_bands` silently ignored `n_boot`/`seed` under
  `method="asymptotic"` and `band_scope` under `band="pointwise"`, while
  its engine-sharing sibling refused them. One shared inert-keyword table
  now serves both, and the results facade's `plot_irf`.
- **API16-4** — `umidas` ignored `maxlags` under `se_type="nonrobust"`.
  It is refused, as in `adl_midas`.
- **B1 / B2 / B3** — about 60 refusals named no keyword ("series has N
  observations but needs at least M"), the engine's variable
  (`n_posterior_draws`, `coefs`, `lag length p`, `n_sim`,
  `max_seasonal_p`, `nlags`, `r`, `k^(order+1)`), or the wrong cause (a
  `scale_ar` of 2^31 reported as an ill-conditioned posterior,
  `max_iter=0` as non-convergence after one sweep). Each now names the
  Python keyword and, where a size is the cause, its largest admissible
  value. The reviews closed several second-order cases: `dcc_test` quoted
  a largest `lags` that then failed as a singular design, `zivot_andrews`
  blamed `lags` when the search had chosen it under `max_lags`, and
  `markov_switching_ar` offered an order the fit refuses.
- **B4** — a −1 inside a restriction tuple or narrative dict got the
  generic "the parameter counts something" text. It is now named by its
  path (`restrictions[0][2]=-1`).
- **B5** — `check_series` validated neither `lags` nor `max_breaks`
  (`lags=-1` became 1 silently, 2^62 became n−2). Both are now validated
  by name.
- **DOC16-1** — `dfm_smooth`'s quoted agreement (6.0e-11) left out the
  standardized case (2.6e-10 / 1.2e-10). All five places now quote the
  worst case.
- **DOC16-3** — under `band="pointwise"` the asymptotic budget refusal
  told the caller to change `band_scope` or `band_n_sim`, both refused as
  inert there. Refusals are now term-specific.
- **Pre-existing, queued** — `seed`/`band_seed`/`rf_seed` outside
  `[0, 2**64 - 1]` are refused by name with that range in every callable
  (including the results facade, which called the raw binding);
  `arima_fit` refuses an absurd ARMA specification before allocating,
  naming `p`, `q` and `seasonal`; the queued `arch_lm`/`adf` aborts are
  fixed with the A-class budgets.

## Minor

- **DF16-4** — the symmetric-root Jarque-Bera lost up to 5 digits as
  column scales diverged inside the accepted range. Its inverse root now
  comes from a Cholesky factor and a one-sided Jacobi SVD, accurate to
  7.6e-15 at every scale ratio against a 50-digit reference (1.5e-5 at
  1e5 before), with no slowdown.
- **SVF-4 / H16-07 / API16-5** — a mistyped `identification` passed with
  `impact=` was refused as if the typo were a real mode. The mode is now
  validated first, with a hint when `impact` was passed.
- **H16-12** — an extreme-scale invertible `impact` was called singular
  "by the numpy convention" or gave inf/NaN. Scales from 1e-300 to 1e300
  now agree with statsmodels to 2.2e-16.
- **H16-13 / B6** — the `long_run_svar_bands` budget refusal blamed
  `horizon and band_scope` when `band_n_sim` alone was over. The review
  found one more case: a sup-t band past the Jacobian cap was blamed on
  the family covariance, which fit.
- **H16-14** — the CSS sample-size refusal for a seasonal model said
  "lower p, d or q" and never named the seasonal part, which was the
  cause.
- **DOC16-2** — the validation matrix's 0.11 provenance paragraph named 13
  of the 15 NumPy 2.5.3 fixtures and listed `var_diag` among the pre-wave
  ones.
- **DOC16-4** — `help(long_run_svar)` said indices outside `0..k` are
  refused; the stub (correctly) said `0..k-1`.
- **DOC16-5** — `multivariate_normality`'s non-finite refusal blamed "the
  VAR estimator", although it fits nothing.
- **C1** — `dsge_solve`'s docstring said the returned `verdict` can read
  indeterminate or no stable solution; both raise.

## Refuted

Each reproduced as described and was refuted because the behaviour is
documented, deliberate, house convention, or Monte Carlo noise at the
replication count the card states.

- **DF16-3** — `multivariate_normality` refuses the whole call when only
  the unit-dependent symmetric block is unreliable, where `var_diagnostics`
  returns that block as `None`. Every surface documents the refusal and
  the difference.
- **DF16-5** — a signal-mode `condition_sd` below 1e-162 returns `se = 0`
  while still reported as soft. That is the correctly rounded exact
  variance (checked in 80-digit arithmetic), not a silent hard cell.
- **H16-10** — `alpha` below about 1e-16 fails with an unnamed
  special-function message. A documented house-wide residue, older than
  the wave.
- **B7** — `auto_ets` accepts an `n_sim` that `ets_fit` refuses: its
  docstring explains that the winning class is not known in advance.
- **DOC16-6** — `lab/README`'s map does not list every directory; it has
  never claimed to.
- **API16-6** — `skewness`/`kurtosis` mean sample moments in `jarque_bera`
  and chi-square components in `multivariate_normality`. Both are
  documented exactly, and the latter must sum to `jarque_bera`.
- **API16-7** — the stub rejects pandas forms the runtime accepts: the
  deliberately narrow, house-wide array alias.
- **API16-9** — a Python scalar `resid` raises `TypeError` rather than
  `ValueError`: the house wrong-type convention, which a 0-d ndarray does
  not hit.
- **STAT16-1** — the normality size table differs from a fresh 20k-rep
  re-measurement. The table is the exact output of its seeded study, and
  the gap is within the combined Monte Carlo error.
- **STAT16-2** — StepM's family-wise error "slightly over nominal" did not
  reproduce on one fresh 1,500-panel seed set. On 20,000 fresh panels
  it reproduces (tsecon 0.112-0.118, arch 0.109-0.123 at a nominal 0.10).

## Open — recorded, not fixed

- **Long but bounded runs.** The budgets bound memory, and work where the
  cost is superlinear in the argument. Linear loops over the caller's own
  counts are not work-capped, because a cap would refuse legitimate long
  runs: `boosting` (O(n·n_steps²) is capped by memory only),
  `robust_svar_bounds` at its largest `n_draws` (about 7 minutes),
  `bvar_ssvs` with a huge `thin`, `mlp_regression` without early stopping,
  `auto_arima` seasonal searches without R's `floor(n/3/s)` order caps,
  `umidas` with a huge HAC `maxlags`, and default-rule orders of
  `zivot_andrews` and `stl` on very long series. They run, slowly; none
  aborts or hangs on a bounded input.
- **Lower-bound counts.** The `random_forest` budget is a documented lower
  bound (node counts depend on the data); its admitted maximum at n = 200
  aborted under a 4 GB cap in review, because an upper bound would refuse
  valid forests. The ARIMA value count ignores faer's per-matrix padding,
  which matters only for data-driven sizes near 1e7.
- **DF16-1 residual risk.** Two consecutive pins each near the 1e-14 floor
  could compound dust past the companion's 1e-6 band. Cross-period dust
  was probed and could not be triggered.
- `panel_unit_root(test="llc")` with `lrv_bandwidth ≥ T − 1` returns a
  numerically meaningless statistic when rounding lands above zero (it is
  refused when it lands at or below). `threshold_vecm(n_grid_beta > 1,
  beta_span=0)` accepts an inert combination. Refusing either would change
  an accepted call.
- The house-wide allocation-failure fallback names only integer arguments.

## Verification record

**The house sweep, re-run.** The same harness, cases and values were run
against the wheel built from the merged fixes. The first re-run found no
abort, timeout, `MemoryError` or panic, and 78 refusals still classed as
unnamed. Two of them were the sweep's own fault: its `narrative_svar`
baselines passed `n_weight_draws` without narrative restrictions, which
the round now refuses as inert, so the baselines were corrected. The rest
were nine wording classes that the findings had not listed cell by cell,
and they were fixed at integration:
- the engine's own words for `block_length` (`bootstrap_indices`,
  `proxy_svar_bands`), `coint_rank` (`vecm`), `n_factors` and
  `slow_indices` (`favar`), and `return_periods` (`gev_fit`);
- an index past its bound inside a restriction spec, which the
  identification engines reported as "restriction out of range" without
  the keyword or the entry. The input layer now names it as
  `restrictions[1][2] = 9`;
- a conformal grid blamed for what the base forecaster's own `period`
  asks (no training window can hold `2 * period` observations);
- `proxy_ar_sets` blaming the default `horizon` for an absurd `lags`;
- `dsge_solve`'s `n_predetermined`. Its Blanchard-Kahn refusals also gave
  their remedies backwards, which is corrected.

Integration also caught one interaction between groups. The `band_n_sim`
budget that var_budgets added to `lp_band` ran before the horizon
pre-scan that ml_lp added, so `horizons=2**40` was refused as a
band-simulation budget with no admissible count. The band family is now
sized by the horizons the sample can reach.

After those fixes the sweep was run once more, on the final wheel:

| Outcome | `fdd90d3` (round 16 ran here) | merged fixes |
|---|---|---|
| Refusal naming the parameter | 1,830 | 2,177 |
| Result returned | 538 | 528 |
| Unnamed refusal | 194 | **0** |
| Abort (SIGABRT, interpreter killed) | 136 | **0** |
| Timeout (60 s) | 12 | **0** |
| `MemoryError` or panic | 0 | 0 |
| Calls | 2,710 | 2,705 |

The re-run makes five fewer calls: the corrected `narrative_svar` baseline
no longer sweeps `n_weight_draws`, which the `narrative_svar#narr` variant
sweeps with narrative restrictions in place. A "result returned" at 2^62
is mostly a seed, for which any 64-bit value is legitimate; the rest are
counts the callable accepts by design, such as `max_epochs` under early
stopping.

After the eight fix branches were merged onto the integration branch:

- `cargo fmt --check` and `cargo clippy --workspace --all-targets -D
  warnings`: clean.
- Each fix branch passed `cargo test -p` over every crate it touched and
  its dependents; the branches that changed `tsecon-linalg`, `tsecon-ssm`
  and `tsecon-rng` ran the whole workspace.
- Each fix branch passed the full Python suite twice, in a full venv and in
  a minimal one mirroring the CI wheel jobs.

The integrated run's totals are recorded on the
[testing page](../reference/testing.md).

## Lessons

- **A budget at the coercion layer is not a budget.** A flat 2^48
  pre-flight stopped only the counts no one would pass by accident. The
  sizes that kill a process are products (draws × horizon × k²) that only
  the engine can count, so every budget now lives in the shared engine,
  next to the allocation it protects, and names the keyword.
- **Count what is stored, not what is meant.** Four first-version budgets
  under-counted: faer's column padding, the heap fragmentation of
  per-replication matrices, and the Python ints and dicts a binding
  returns. Only running the quoted maximum, in a capped subprocess, showed
  it. Every boundary in this round is now pinned on both sides.
- **Balance before solving.** An eigenvalue solver is accurate relative to
  the matrix norm. When the units of the data enter the matrix
  quadratically, as they do in a companion form, a diagonal similarity by
  powers of two costs nothing and removes the dependence.
- **Decide singularity by structure where structure exists.** Whether a
  resample's covariance is singular was left to a Cholesky's rounding, so
  the answer depended on the platform's kernels. Counting distinct rows
  gives the exact answer on every platform.
- **Integration is a third audit.** Eight groups fixed eight crates in
  parallel, and one interaction (`lp_band`'s budget order) and a residue
  of unnamed refusals surfaced only on the merged wheel. Re-running the
  finder's own harness after the merge is what found them.
- **The review is the second audit.** As in round 15, the adversarial
  review of each fix found what the fix missed: 51 amendments across 8
  groups, including two incomplete severe fixes (DF16-1, SVF-2) and four
  under-counted budgets.
