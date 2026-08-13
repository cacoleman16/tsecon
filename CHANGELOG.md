# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows the
pre-1.0 policy in [ROADMAP.md](ROADMAP.md) (minor = breaking allowed, patch =
fixes) until 1.0, then strict [SemVer](https://semver.org/).

## [Unreleased]

### Added — seasonal ARIMA

- **`arima_fit(..., seasonal=(P, D, Q, s))`** — the multiplicative
  SARIMA(p,d,q)(P,D,Q)_s, closing the gap the docs have been honest about
  since 0.1.0. The seasonal and regular lag polynomials are multiplied into a
  single dense ARMA and run through the existing exact-MLE state-space engine;
  seasonal differencing follows the statsmodels order (seasonal first, then
  regular; `simple_differencing=True` semantics, `d + D*s` observations lost);
  forecasts are re-cumulated to levels through the augmented state — one
  cumulator per regular difference, one `s`-long delay line per seasonal
  difference — so the reported forecast variance is the exact cumulative one,
  seasonal stages included (the pure seasonal random walk reproduces
  `se_h = sigma * sqrt(ceil(h/s))` bit-for-bit). Seasonal parameters are named
  statsmodels-style (`ar.S.L12`, `ma.S.L12`), stationarity/invertibility is
  enforced per factor polynomial through the same Monahan transform, and
  Hannan-Rissanen starting values gain the seasonal lags. Golden-pinned to
  statsmodels `SARIMAX` in `fixtures/sarima.json`: the airline model
  `(0,1,1)(0,1,1)_12` on the real log Series G (fixed-parameter log-likelihood
  at 1e-8 relative, fit at the textbook `theta ~ -0.40`, `Theta ~ -0.56`,
  `cov_type='approx'` standard errors at 1e-4, 24-step levels forecasts against
  the statsmodels levels state-space form at 1e-6), a quarterly
  SAR(1)x(1)_4-with-constant, and the mixed `(1,1,1)(1,1,1)_4`. The Rust
  `ArimaSpec` gains `seasonal(P, D, Q, s)` and the results object
  `seasonal_ar()` / `seasonal_ma()`; CSS (`fit_css`) conditions on the expanded
  `p + s*P` presample.

### Fixed — audit findings (rounds 2–5)

- **`panel_fe` / `panel_lp` now refuse a design the fixed effects have
  absorbed** instead of returning a publishable t-statistic for it. The old
  guard was a Cholesky positive-definiteness test that fired only when the
  within-demeaned residue was bit-exactly zero: entity constants stored as
  ordinary doubles (log land area, a share in `[0, 1]`) slipped through, and
  the default cluster covariance turned the `O(1e-16)` residue into
  t-statistics that reached nominal 5% significance in 19.2% of audit draws —
  a statistic that moved when a constant was added to the data. The guard is
  now two scale-invariant checks (per-column absorption of the demeaned norm
  relative to the raw norm, plus the numpy/linearmodels singular-value rank
  criterion), matching the linearmodels `AbsorbingEffectError` refusal the
  docstring promises — and, unlike its predecessor, it is monotone: a merely
  ill-conditioned near-duplicate still fits while an exact duplicate raises.
  `panel_lp` additionally rejects an infeasible `max_horizon` up front as a
  sample-size error, rather than tripping over whatever symptom the shrunken
  per-horizon window produced first.
- **`engle_granger` raises instead of panicking when the sample has no more
  rows than step-1 design columns** — the `(0, k)` / `(1, k)` shapes escaped
  `except Exception` as `PanicException` through the Python boundary.
- **`proxy_svar_bands`'s proxy alignment can no longer panic** when `lags`
  exceeds the observation count (the same `PanicException` class).
- **`proxy_ar_sets(hac_lags=...)` with the default `variance="hc0"` now
  raises** instead of silently ignoring the argument — output used to be
  bit-identical with and without it. `hac_lags` parameterizes only the
  `variance="hac"` route, and the structural-identification model card now
  says so instead of implying `hac_lags` itself switches the estimator.
- **The `bvar_fit` model card had its two shrinkage dials swapped**: it called
  `lambda0` "overall tightness" and `lambda1` the "own-lag scale", while in
  the implementation (deliberately, and consistently with the conjugate
  Kronecker form) `lambda0` scales only the intercept prior and `lambda1` is
  the overall tightness on every lag coefficient, own and cross alike. A
  card-following user lowering `lambda0` for shrinkage got a pinned intercept
  and untouched dynamics. The card now matches the code, and states that the
  conjugate form cannot express the classic cross-variable `lambda2`.
- **The ARIMA model card mis-stated `arima_fit`'s defaults** (`p`, `d`, `q`
  "all default 0", `constant = False`): the shipped defaults are `p = 1` and
  `constant = True`, so with `d >= 1` the default fits a drift. The card now
  matches the code and says to turn the constant off deliberately.
### Changed — BREAKING

- **`long_memory_d`'s `se` changed meaning, and the number it reports moves.**
  It used to be the large-*m* asymptotic closed form — `pi/sqrt(24m)` for GPH,
  `1/(2*sqrt(m))` for local Whittle — a **constant that ignored the data**
  entirely: two different series with the same bandwidth got the same standard
  error. The data-dependent one was computed and then discarded. `se` is now
  the standard error at the bandwidth actually used, and the closed form is
  reported alongside it as `se_asymptotic` (plus `se_regression` for GPH).

  This matters because the old value was too **narrow**. Measured at `n=512`,
  `m=22`: `se` = 0.17037 against `se_asymptotic` = 0.13672, so intervals built
  from the old number were about **25% too tight at the library's own default
  bandwidth** — and the model card claimed the opposite, telling readers the
  bands were conservatively wide. Verified against exact ARFIMA(0,d,0) draws
  (Davies-Harte circulant embedding, 3000 replications): the new `se` tracks
  the realised sampling dispersion at a ratio of 0.99–1.04 with 94.5–95.7%
  coverage, where `se_asymptotic` ran 0.76–0.88 and covered 87–92%.

  If you were reading `se`, your intervals were too narrow and will now widen.
  If you need the old value for continuity, it is `se_asymptotic`.

- **`var_irf_bands(bias_correct=True)` now raises on `method="asymptotic"`**
  instead of silently discarding the flag. Kilian's correction re-centres
  bootstrap draws and the delta-method arm draws none, so the flag never had
  anything to act on — and this library's own coverage audit *instructed*
  callers to set it, without saying `method="bootstrap"` was required. Both
  instructions now name the method, and the flag is echoed in the result.

- **`historical_decomposition` rejects the five sampling arguments the
  `identification="cholesky"` path never reads** (`n_draws`, `max_tries`,
  `seed`, `lambda1`, `n_weight_draws`), naming the ones that were set. An
  accepted-and-inert `seed` makes a single point decomposition look like a
  seeded draw from a set-identified posterior.

- **`zero_sign_svar(weighted=True)` raises on a zero at horizon >= 1.** The ARW
  importance weight was hardcoded to 1.0 on every path, so the flag could never
  change the output, while the docstring implied non-impact patterns received a
  non-unit weight. The volume element for those patterns is not implemented;
  the refusal is now loud rather than a silent downgrade. New `arw_weighted`
  key says whether the weighting actually applied.

- **`panel_unit_root(test="llc", lrv_kernel="truncated")` raises** instead of
  returning `statistic=nan, p_value=nan`. The truncated kernel is not positive
  semi-definite, so the long-run variance can come out negative and `sqrt`
  mints a NaN — it did on 57 of 60 test panels. The error names the kernel and
  the PSD alternatives.

---

`proxy_svar` identified an impulse response and told you nothing about how
precisely. This release attaches uncertainty to it — twice, because one answer
is not enough. When the instrument is strong you want a band; when it is weak a
band is the wrong object no matter how you build it, and you want a set.

### Added — proxy-SVAR inference

- **`proxy_svar_bands`** — Jentsch-Lunsford moving-block bootstrap confidence
  bands for the external-instrument SVAR impulse response. The joint pair
  `(u_t, m_t)` is resampled under **one** set of block starts, and inside every
  draw the VAR is reconstructed, re-estimated, re-identified, and the unit-effect
  normalization is re-imposed. Returns the Hall/basic band (`lower`/`upper`,
  recommended) alongside the Efron percentile band (`lower_efron`/`upper_efron`)
  that Mertens-Ravn and Gertler-Karadi report; the two separate when the
  bootstrap distribution is skewed.

  **The `h = 0` cell of `norm_var` is degenerate at `unit` by construction** —
  verified `[1.000000, 1.000000]`. That is not a bug, it is the free proof that
  the normalization is re-imposed inside the loop rather than hoisted out of it.

  **`bands="wild"` reproduces the published Mertens-Ravn / Gertler-Karadi bands
  and is not asymptotically valid.** With the common Rademacher draw those papers
  use, `e_t` hits the residuals and the proxy alike, so `m*_t u*_t' = e_t^2 m_t
  u_t' = m_t u_t'` and the identifying moment is **bit-identical in every draw** —
  verified 200/200, maximum deviation exactly `0.000e+00`. The step that does the
  identifying contributes no variance at all. Measured on a known-truth DGP at
  nominal 0.90, the wild arm covers **0.113** at impact against the moving
  block's 0.860, with a mean width of 0.018 against 0.173. The arm is shipped
  because reproducing a published figure is a legitimate thing to want; it sets
  `asymptotically_valid=False` and carries a `validity_note`, and it is not
  inference.

  Failed draws are **counted by reason, never dropped**: six counters
  (`too_few_proxy_obs`, `zero_proxy_variance`, `near_zero_gamma_norm`,
  `refit_failed`, `identification_failed`, `non_finite`), a total in `n_failed`,
  and a `failure_warning`. Dropping them would trim exactly the
  near-zero-denominator tail and shrink the interval precisely when the
  instrument is weakest. A nonzero `n_failed` means reach for `proxy_ar_sets`
  instead.

  These are strong-instrument asymptotics. The moving-block arm's own shortfall
  at longer horizons (0.78-0.81 for a nominal 0.90) is inherited from the
  reduced-form VAR bootstrap, not introduced by the proxy layer — the Cholesky
  reference lands within 0.07 at every horizon on the same replications — and
  there is no Kilian bias correction on this path.

- **`proxy_ar_sets`** — weak-instrument-robust Anderson-Rubin confidence **sets**
  for the same impulse response, inverted in closed form rather than searched
  over a grid. Under weak identification no bounded set can be honest (Dufour
  1997), so a cell may come back as a bounded `"interval"`, an `"exterior"` set
  (the **complement** of a rejected middle region — two rays), the `"whole"`
  line, `"empty"`, a degenerate `"point"`, or a one-sided ray. The shape is the
  answer. Branch on `kind`: an `"exterior"` set reports `lower`/`upper` as
  ±infinity precisely so it cannot be mistaken for an interval, and
  **`excludes_zero` on an unbounded set does not establish a sign** — both signs
  can be members while zero sits in the rejected middle.

  **Reduced-form (VAR coefficient) uncertainty is propagated by default**,
  because omitting it is not a drift but a collapse. Measured at nominal 0.95 on
  an estimated VAR, `T = 300`, VAR(2), excluding the degenerate `(norm_var, 0)`
  cell, coverage at `h = 0..8` runs `.952 .529 .458 .315 .247 .195 .163 .135
  .119` omitted against `.952 .953 .954 .947 .941 .936 .930 .922 .913`
  propagated. **The propagation is conservative under a weak instrument**: the
  weak arm goes from `.9413` omitted to **`.9908`** propagated, because the extra
  variance turns exterior sets into the whole line. That is the correct direction
  to err, and it is disclosed rather than tuned. The price is width — the paired
  median set-width ratio at `h = 8` is **13.5x**. With
  `reduced_form_uncertainty=False` the returned `level` is `None`: a set
  conditional on the reduced form has no honest 1-alpha label.

### Validation, and what kind it is

Neither function is pinned to an independent package, and neither is described
as if it were. No external package implements Jentsch-Lunsford moving-block
proxy-SVAR bands, so the band golden is a **documented-formula NumPy
transcription** with the block starts pinned in the fixture (the RNG becomes a
shared input, and everything downstream is compared cell for cell); it pins the
arithmetic, and the theory is carried by property tests and seeded Monte-Carlo
coverage. The Anderson-Rubin "golden" is likewise a **co-derived NumPy
transcription by the same author from the same specification** — not a
third-party reference. Its load-bearing validation is instead a **brute-force
grid inversion** that re-tests `AR(lambda) <= c` directly at thousands of
candidate values per cell for every shape the set can take, plus a numerical
Jacobian check on the reduced-form correction. See the
[validation matrix](docs/reference/validation-matrix.md) for the row-by-row
grading.

### Not in this release

Named so they are not read into the above: **simultaneous (joint) bands** — both
new surfaces are **pointwise**, covering each `(horizon, variable)` cell at the
nominal rate and saying nothing about the path as a whole, and every other band
in the library is pointwise too. Also still absent: SARIMA seasonal orders
`(P, D, Q, s)`; Angrist-Pischke, Cragg-Donald, and Kleibergen-Paap statistics;
and bootstrap bands for the other point-identification schemes (`long_run_svar`,
`max_share_svar`, `hetero_svar`, `nongaussian_svar` remain point-only).
### Added — simultaneous (sup-t) bands

- **A band you can read as a statement about a whole path.** Every band this
  library produced was **pointwise**: it covered one horizon, or one
  `(horizon, series)` cell, at the level you asked for and promised nothing
  about the path a reader actually traces with their eye. The
  [interval-coverage audit](docs/examples/interval-coverage.md) measured what
  that costs — a nominal 90% pointwise IRF band contained the whole `h = 0..12` path
  in **72.2%** of samples at T=500, and nominal 95% forecast bands contained
  every horizon of every series at once in **40.9%** at T=100 and still only
  **48.1%** at T=800. The gap is **multiplicity, not sample size**: it does not
  shrink as the data grows, which is what makes it a design gap rather than a
  small-sample caveat. `var_irf_bands`, `var_forecast`, `lp` and `smooth_lp` now
  take a band selector — `band="sup-t"`, `"sidak"` or `"bonferroni"` — that
  changes the multiplier and **nothing else**: same point estimate, same
  standard errors, a larger `c` in `point ± c·se` chosen so the whole declared
  family is covered at once. The sup-t construction is Montiel Olea and
  Plagborg-Møller's. **Nothing that already reads these functions changes
  meaning**: `var_irf_bands` and `var_forecast` keep returning pointwise
  `lower`/`upper` whatever you pass and hand the simultaneous edges back as
  extra `sim_lower`/`sim_upper` keys (with `critical_value`,
  `pointwise_critical_value`, `band_scope`, `n_cells` and `n_cells_used`), and
  the LP family returns no band at all unless `band` is set. `lp` and
  `smooth_lp` take their own `band_alpha`; the sup-t simulation is a pure
  function of `band_seed` and `band_n_sim`.
- **Measured, with both arms scored on the same replications** — joint coverage
  of the whole family, pointwise band against sup-t band:

```text
  object                        nominal  design                                pointwise   sup-t
  var_forecast                      95%  T=100, 12 horizons x 2 series, 6000    41.2+-0.6   90.5+-0.4
  var_irf_bands (asymptotic)        90%  T=500, h=0..12, 3000 reps              70.4+-0.8   84.8+-0.7
  lp (lag-augmented)                90%  T=240, 13 horizons, 400 reps           36.5        81.8
  lp (lag-augmented)                90%  T=720, 13 horizons, 400 reps           42.7        89.5
```

Tripling T moved LP's *pointwise* joint rate from 36.5% to 42.7%. It is not
converging.

- **Neither VAR simultaneous rate reaches nominal — 84.8% against 90%, 90.5%
  against 95% — and the tests say so rather than tuning.** A sup-t band fixes
  multiplicity and **inherits every other defect of the band it widens**,
  because it reuses the same standard errors. The IRF band's own *marginal*
  coverage on that design is 88.7% at h=0 falling to 85.2% at h=12, so what that
  cell needs is a better standard error, not a bigger multiplier; `var_forecast`
  is a plug-in band that ignores coefficient sampling error, so its marginal
  rate is 93.3% rather than 95%. LP is the clean case and it demonstrates the
  mechanism: at T=720, where the marginals sit on nominal, sup-t lands on
  nominal.
- **The cell family is a user-visible choice, not a detail.** "Simultaneous over
  what?" has several defensible answers that give different critical values, and
  a band whose scope is ambiguous is worse than no band. `var_irf_bands` takes
  `band_scope="horizon"` (the default: `K = horizon+1`, one family per
  response-shock pair — the object the audit measured), `"shock"`
  (`K = k(horizon+1)`), or `"all"` (`K = k²(horizon+1)`); `var_forecast` takes
  `"all"` (the default, `K = steps·k`) or `"horizon"` (`K = steps`); `lp` and
  `smooth_lp` band the horizons of the one response. **Every result reports its
  scope and its `K`.**
- **`lp_iv`, `lp_multiplier` and `lp_state` get Šidák and Bonferroni only.**
  `band="sup-t"` is **refused with an error naming the reason** — no
  cross-horizon covariance exists for them, and none was invented. Their bands
  must not be described as sup-t.
- **What the four routes cost.** At `K = 13` and `alpha = 0.10`: pointwise
  1.6449, sup-t 2.20–2.65 depending on the persistence of the path, Šidák
  2.6490, Bonferroni 2.6653. Only sup-t uses the dependence across cells, which
  is why it is the one that moves; the closed forms pay for a worst case that a
  smooth response path does not present.
- **Two shapes that must not be blurred.** On `var_irf_bands(method="bootstrap")`
  the simultaneous band is symmetric `point ± c·se`, while the reported
  percentile band is asymmetric Efron. The simultaneous band is therefore **not**
  guaranteed to lie outside `lower`/`upper` cell by cell — only outside the
  symmetric `point ± z·se`.
- **Nothing you already plotted changed.** The pointwise output is
  bit-identical, verified rather than assumed:
  FNV-1a fingerprints of the raw f64 bit patterns were captured against the
  unmodified tree and re-checked after every edit, and the statsmodels goldens
  still pass.
- **Measured again, from Python, by the audit that asked for this.** The
  interval-coverage harness now carries a simultaneous arm in
  `docs/examples/coverage/irf_bands.py` and
  `docs/examples/coverage/forecast_intervals.py`. Both read the pointwise and
  the simultaneous arm off the **same call** — the pointwise band is
  bit-identical whether or not a band is requested, and the harness asserts
  that — so the comparison is paired and every replication that gains the whole
  path gains it because the multiplier grew, never because the point estimate
  moved. It reproduces the crate numbers: joint coverage 42.0% → 90.5% for
  `var_forecast` (K=24, 2000 reps) and 71.7% → 85.2% for `var_irf_bands` (K=13,
  1000 reps), with the shortfall-to-nominal asserted rather than smoothed. The
  closed forms are worth a look there too: on the IRF cell Šidák and Bonferroni
  land at 91.7% and 92.0% against a nominal 90%, over the line because
  over-widening cancelled a marginal shortfall of a different origin. `lp` and
  `smooth_lp` have no harness arm yet.
- **Implementation.** One owner for the critical value, `tsecon_stats::simultaneous`,
  with four routes (sup-t from bootstrap draws, sup-t from a covariance by
  Gaussian simulation, Šidák, Bonferroni). LP's cross-horizon covariance was
  **built**, not faked — the Frisch-Waugh-Lovell influence representation, with
  `sqrt(diag)` matching the reported standard errors to 1.3e-15 and ragged
  samples handled so the matrix is PSD by construction. `smooth_lp` already
  computed a joint covariance and discarded all but the diagonal, so its band is
  exact and nearly free.

## [0.2.0] - 2026-08-05

An interval-coverage audit took the library's interval-producing surfaces,
pointed each at its own nominal level, and measured what fraction of intervals
actually covered. It found four real defects. This release fixes them. **One
fix is breaking**: it changes numbers callers were already getting, and
changing those numbers is the point of the release.

### Changed — BREAKING
- **`iv_gmm(weight="hac")` was a silent no-op, and its standard errors will
  move.** `bandwidth` defaulted to `0.0`, and a Bartlett kernel truncated at
  zero lags *is* the White estimator — so `weight="hac"` returned results
  bit-identical to `weight="robust"` (max |Δ se| = 0.000e+00 over 3000
  replications) while the caller believed they had bought serial-correlation
  robustness. `bandwidth` now defaults to `None`, which selects the
  Newey-West rule of thumb `floor(4 * (n/100)^(2/9))`; an **explicit**
  `bandwidth=0.0` now raises instead of silently degrading to White; and the
  truncation actually used comes back as `hac_bandwidth`. On an AR(1)-error
  design at n = 250 the slope standard error moves from 0.10522 (`robust`, and
  the old `hac`) to 0.09393 (`hac`, automatic bandwidth 4) or 0.09228 (`hac`,
  `bandwidth=10`). If you passed `weight="hac"` before, you were reporting
  White standard errors under a HAC label and your numbers will change.
  **This does not restore coverage.** The audit measured 0.868 ± 0.006 against
  a nominal 0.95 at `bandwidth=10`, and the automatic rule picks *fewer* lags
  (4 at T = 250). A defensible default is not a remedy: those intervals are
  still too narrow, and the fix is that you can now see and set the bandwidth
  rather than that the bandwidth is now right.
- **`iv_gmm(method="2sls", weight="hac")` now raises.** 2SLS fixes its weight
  matrix at `(Z'Z/n)^-1` by construction, so accepting a weight argument there
  was the same silent no-op in a second place.

### Added — inference
- **`ols(se_type="hc2")` and `ols(se_type="hc3")`** — the leverage-corrected
  heteroskedasticity-robust standard errors, matched to statsmodels HC2/HC3 to
  2.96e-15 on the audit's own T = 25 chi2(1)-regressor high-leverage design. On
  that design tsecon's own slope standard error runs 0.1749746 (nonrobust),
  0.1751173 (hc0), 0.1825724 (hc1), 0.2095910 (hc2), 0.2629148 (hc3): HC3 is
  44% larger than HC1, which is exactly the leverage correction that hc1's
  `n/(n-k)` factor does not buy. An observation whose leverage is numerically
  equal to 1 is refused, not returned as a near-infinite standard error.
- **`iv_gmm` returns `first_stage`** — a list of dicts with keys `regressor`,
  `fstat`, `dof_num`, `dof_den`, `pval`: a heteroskedasticity-robust
  per-regressor first-stage F. Entries are **omitted** where the statistic is
  undefined (an exogenous regressor, no excluded instruments, a regressor the
  instruments reproduce exactly, rank-deficient `Z`, a non-finite statistic),
  so the list can be shorter than the regressor count and must be indexed by
  `regressor`, never by position; a missing entry is not a failed fit. **With
  two or more endogenous regressors this is not a weak-identification test** —
  every regressor can clear 10 while the system is under-identified, because
  the instruments may predict only one common combination of them. The right
  objects are Angrist-Pischke (per regressor) and Cragg-Donald /
  Kleibergen-Paap against Stock-Yogo (joint), and none of those are
  implemented. Even with a single endogenous regressor, F > 10 is not a safety
  threshold: the audit measured 0.915 coverage at a median first-stage F of
  10.5.
- **`arima_fit(drift_uncertainty=True)`** — with `d >= 1` and `constant=True`
  the h-step forecast contains an estimated drift whose uncertainty grows like
  h², and the default omits it entirely (the reported se is exactly
  `sigma*sqrt(h)`). Opt in and the se matches the closed form
  `sigma*sqrt(h + h²/(T-1))` to 5.22e-09. It is **opt-in** so the default path
  stays bit-identical and keeps matching the statsmodels `get_forecast` golden
  at 1e-6: the two are different estimands, not a right one and a wrong one.
- **`arima_fit` returns `bse`, `param_cov`, and `cov_ok`** — parameter standard
  errors from the observed information (statsmodels `cov_type="approx"`).
  ARIMA previously reported no parameter standard errors at all. `bse` and
  `param_cov` are `None` with `cov_ok=False` when that matrix is too
  ill-conditioned to invert honestly, rather than reporting a number the
  numerics do not support.

### Added — ergonomics
- **Forgiving input**: every estimator now accepts a pandas `DataFrame`/`Series`
  (or any `.to_numpy` array-like), off-dtype/non-contiguous float arrays,
  **integer and boolean arrays** (data read as `int`, a 0/1 dummy, a `y > 0`
  mask), and a **plain list of numbers**. All are converted to `float64` at the
  boundary instead of raising. Coercion is *parameter-aware*, so it never
  touches an argument that is not data: the four audited integer label/index
  parameters (`hetero_svar.regime_labels`, `var_granger.caused`/`causing`,
  `favar.slow_indices`), restriction-tuple specs, tuple-valued options, and
  callables pass through untouched, and ragged panel lists keep their container
  while each per-unit array is converted. A *nested* Python list is
  deliberately left alone — `[(0, 1), (0, 2)]` is a restriction spec and
  `[[1.0, 2.0], [3.0, 4.0]]` is a matrix, and the values are indistinguishable.
- **Errors that teach**: the messages a first run is most likely to hit now say
  what happened, why, and what to try, with the offending numbers included. For
  example, too few observations for the requested lags now reports the required
  row count and the degrees-of-freedom arithmetic behind it, and suggests a
  concrete smaller `lags`. A wrong-rank array argument reports the shapes it
  received instead of the low-level `'ndarray' object is not an instance of
  'ndarray'`.
- **`tsecon.summarize(result)`**: a uniform, opt-in renderer for *any* function's
  output. Plain dicts get a generic aligned `.summary()`; the six bespoke
  `tsecon.results` objects pass through unchanged. Still a `dict` subclass, so
  the plain-data contract is preserved.
- **Cookbook**: short single-task recipes under `docs/cookbook/`, each a
  self-contained page with executed output.

### Added — estimators
- `ndiffs` — how many differences a series needs, with the per-order test
  evidence rather than just the integer.
- `box_cox_lambda` — variance-stabilising Box-Cox lambda (MLE, matched to
  `scipy.stats.boxcox_normmax`; Guerrero as a documented-formula alternative).
- `engle_granger` — the two-step cointegration test now returns p-values and
  critical values from the MacKinnon response surfaces, not just the statistic.

### Fixed

The four defects the interval-coverage audit found, with measured coverage
before and after (nominal 0.95 throughout):

- **`ols` robust intervals under-covered badly under high leverage.** On the
  T = 25 chi2(1) design: **hc1 0.682 → hc3 0.863**. Fixed by adding hc2/hc3.
  hc1 is still available and is still the wrong choice on that design; hc3 does
  not reach nominal there either, and the honest reading is that a T = 25
  tail-heavy design is hard, not that hc3 solves it.
- **`arima_fit` forecast intervals ignored drift uncertainty.** With `d >= 1`
  and `constant=True`, at h = 24 and T = 60: **0.902 → 0.945** with
  `drift_uncertainty=True`.
- **`iv_gmm(weight="hac")` never applied a HAC correction** (bit-identical to
  `weight="robust"`, max |Δ se| = 0.000e+00 over 3000 replications). See the
  breaking note above. **Not fixed by this release:** coverage on the audited
  AR(1)-error design is 0.868 at `bandwidth=10`, and the new automatic default
  picks fewer lags than that.
- **`iv_gmm` reported no first-stage evidence at all.** Now returns
  `first_stage`, with the caveats above — this is a diagnostic, not a
  weak-identification test.

### Not in this release

Named so they are not read into the above: SARIMA seasonal orders `(P, D, Q, s)`;
Anderson-Rubin and other weak-IV-robust confidence sets for `iv_gmm`;
Angrist-Pischke, Cragg-Donald, and Kleibergen-Paap statistics; and simultaneous
(joint) bands anywhere — every band in the library is pointwise.

## [0.1.0] - 2026-07-23

First tagged release, published to PyPI as `tsecon`. Pre-1.0 and under active
development: minor versions may make breaking changes, patch versions are
fixes, until 1.0.

### Added — foundations and first model classes
- **Foundations**: Philox RNG (bit-identical to NumPy), special functions and
  the distribution zoo, structured linear algebra (Levinson-Durbin, Toeplitz,
  discrete Lyapunov), the resampling/bootstrap engine, the exact-diffuse
  linear-Gaussian state-space (Kalman) engine, the numerical optimizer suite
  with the Monahan stationarity transform, and the HAC/robust-inference module.
- **Diagnostics**: ACF/PACF, Ljung-Box, Jarque-Bera, ARCH-LM; the full
  unit-root workflow (ADF with MacKinnon p-values, KPSS, `check_stationarity`);
  the semiparametric Phillips family — `phillips_perron` (Z-tau/Z-alpha
  unit-root test) and `phillips_ouliaris` (single-equation residual
  cointegration test), matched to `arch` to < 1e-10 with MacKinnon
  response-surface p-values; spectral analysis (periodogram, Welch, coherence).
- **One-call battery**: `check_series` — the Module 01 flagship — runs the
  diagnostic families in order (outlier screen, ADF+KPSS quadrant with
  analysis-scale routing, Ljung-Box/ACF/PACF, ARCH-LM, Jarque-Bera, a
  sup-F/Bai-Perron mean-shift scan, GPH long memory, seasonality evidence;
  for a 2D panel: per-series integration, Johansen, and VAR lag selection
  with a stability check) and ends in recommendations that route to concrete
  tsecon calls — every hypothesis test on the record in `tests_run` with the
  multiple-testing arithmetic shown, never silently corrected.
  `tsecon.results.check_series` adds `.summary()` and `.plot_diagnostics()`.
- **Univariate models**: exact-MLE ARIMA; GARCH/GJR/EGARCH with normal and
  Student-t QMLE, Bollerslev-Wooldridge robust standard errors, and a fused
  allocation-free likelihood with analytic gradients; GAS/DCS score-driven
  volatility (Gaussian and Student-t); Markov-switching AR; trend-cycle
  filters (HP, one-sided HP, Baxter-King, Christiano-Fitzgerald, Hamilton);
  long memory (fractional differencing/integration, GPH, local Whittle).
- **Multivariate and structural**: reduced-form VAR with IRF/FEVD/Granger/
  forecasting, frequentist IRF confidence bands (`var_irf_bands` — Lütkepohl
  (1990) delta-method SEs validated against statsmodels to machine precision,
  and a Kilian (1998) residual bootstrap with optional bias correction), and an
  honest stability block (`is_stable`/`min_root`); sign-restricted Bayesian
  SVARs; `zero_sign_svar` — the corrected Rubio-Ramírez-Waggoner-Zha (2010) /
  Arias-Rubio-Ramírez-Waggoner (2018) **zero + sign** restricted SVAR (a
  superset of the sign-only sampler that reproduces the recursive Cholesky IRF
  as its degenerate impact-only-zero corner, with the weight-invariant
  identified-set envelope as the prior-robust deliverable); and four closed-form
  point-identification schemes —
  `long_run_svar` (Blanchard-Quah long-run restrictions), `max_share_svar`
  (Uhlig/Francis maximum-FEV-share and Barsky-Sims news shocks), `proxy_svar`
  (external-instrument SVAR-IV with a first-stage-F report and NaN-window
  handling), and `hetero_svar` (Rigobon two-regime identification through
  heteroskedasticity, with a Box's-M covariance-equality gate); FAVAR;
  Diebold-Yilmaz connectedness; the PCA factor model with Bai-Ng selection;
  Johansen cointegration and VECM; multivariate GARCH (CCC/DCC).
- **Statistical (non-Gaussian) SVAR identification**: `nongaussian_svar` —
  independent-component / ICA identification (Lanne-Meitz-Saikkonen 2017;
  Gouriéroux-Monfort-Renne 2017) that point-identifies the *whole* structural
  impact matrix from the reduced-form residuals alone — no sign, zero, long-run,
  proxy, or variance-regime restriction — by exploiting the mutual independence
  and non-Gaussianity of the shocks. A deterministic symmetric FastICA fixed
  point (Hyvärinen log-cosh contrast, identity-initialized — bit-reproducible, no
  RNG) rotates the whitened residuals to maximal non-Gaussianity and returns
  `B = Σ_u^{1/2} Q`; column sign and order are conventions, and the scheme
  **fails under Gaussianity** — a boundary the returned `shock_kurtosis`
  diagnostic flags (near zero ⇒ weakly identified). Validated by an independent
  NumPy FastICA golden (itself cross-checked to `sklearn.decomposition.FastICA`)
  for the core, plus seeded Monte-Carlo recovery of the true `B` up to sign and
  permutation.
- **Post-identification and prior-robust SVAR tools**: a layer that *takes* an
  identification (any impact matrix `A0`, or a sign-restricted set) and answers
  what comes after — `structural_fevd` (forecast-error variance decomposition
  for an arbitrary structural `A0`, the gap the recursive-only `var_fevd`
  leaves; reproduces `var_fevd`/statsmodels exactly for the Cholesky case,
  rows sum to 1 by the rotation-invariant-denominator identity);
  `historical_decomposition` (per-`(time, variable, shock)` contributions with
  the exact `y = baseline + Σ_j hd` adding-up identity, in a Cholesky point mode
  and an importance-weighted sign-identified set mode); `fry_pagan_svar`
  (Fry-Pagan 2011 median-target — the single accepted, coherent draw closest to
  the pointwise-median band, the answer to "medians mix models");
  `robust_svar_bounds` (Giacomini-Kitagawa 2021 prior-robust identified-set
  bounds via the Gafarov-Meier-Montiel-Olea 2018 active-set closed form, exact
  for a single restricted shock and a conservative marginal outer bound for
  jointly-restricted shocks — removing the Haar-prior artifact that pointwise
  sign-restricted bands carry); and `narrative_svar` (Antolín-Díaz-Rubio-Ramírez
  2018 narrative sign restrictions — shock-sign and "most/least important
  contributor" episode statements imposed by importance-reweighting with weight
  `1/P̂(N|S)`, reporting `ess`/`min_ptilde`; a strict superset of
  `sign_restricted_svar` that reproduces it bit-for-bit with no narrative
  restrictions).
- **Local projections**: `lp` (lag-augmented inference by default), `lp_iv`
  with a per-horizon first-stage F, state-dependent `lp_state`, a three-valued
  `cumulative` mode, and `lp_multiplier` — the one-step Ramey-Zubairy integral
  multiplier as a first-class entry point (because outcome-only cumulation is
  a cumulative IRF, not a multiplier).
- **Bayesian**: a Minnesota-NIW Bayesian VAR with closed-form posterior,
  posterior impulse-response draws, and ArviZ-exact convergence diagnostics;
  `bvar_hierarchical` — empirical-Bayes (ML-II / GLP MAP-II) selection of
  the prior tightness by maximizing the Giannone-Lenza-Primiceri (2015)
  marginal likelihood, then refitting the posterior at the optimum; and
  `bvar_ssvs` — the George-Sun-Ni (2008) spike-and-slab **stochastic-search
  variable selection** BVAR (a four-block Gibbs sampler returning per-coefficient
  posterior inclusion probabilities, optional error-precision selection, and
  Cholesky-orthogonalized IRF draws), MC-recovery-validated on a sparse VAR.
- **Forecasting and evaluation**: Diebold-Mariano (HLN), Clark-West,
  Giacomini-White, Theta, accuracy measures, and the rolling/expanding
  backtest engine.
- **GMM**: linear IV-GMM (2SLS/two-step/iterated, Hansen J) and nonlinear GMM
  with Python-callback moment conditions.
- **Predictive regressions**: OLS/Stambaugh/IVX in one call plus the joint
  IVX test — Monte-Carlo-validated to hold size through an exact unit root.
- **Panels**: fixed effects with clustered/Driscoll-Kraay SEs, panel LP,
  mean-group VAR, the heterogeneous-panel trio (mean group, CCE-MG, PMG), and
  `panel_unit_root` — the three first-generation panel unit-root tests
  (Levin-Lin-Chu, Im-Pesaran-Shin, Fisher/Maddala-Wu-Choi) of the joint
  unit-root null, validated to R `plm::purtest` (and, for Fisher, statsmodels).
- **Nowcasting and mixed frequencies**: MIDAS (weights/U-MIDAS/weighted),
  DFM nowcasting (two-step and exact one-step MLE) with a ragged edge and the
  Bańbura-Modugno news decomposition.
- **Term structure**: Nelson-Siegel, Svensson, dynamic Nelson-Siegel, and the
  arbitrage-free (AFNS) yield adjustment of Christensen-Diebold-Rudebusch.
- **Applied-macro extensions**: recession-probability models (static and
  Kauppi-Saikkonen dynamic probit/logit); survey-expectations tools
  (Coibion-Gorodnichenko, Mincer-Zarnowitz, disagreement); the specification
  and stability battery (White, Koenker-Breusch-Pagan, RESET, Chow, CUSUM);
  and a linear rational-expectations (DSGE-lite) solver via Blanchard-Kahn.
- **Python layer**: maturin mixed layout (`tsecon._core` + a pure-Python
  package); the opt-in `tsecon.results` rendering layer — `dict` subclasses
  with `.summary()`/`.plot_*()` that preserve the plain-dict contract;
  complete type stubs with `py.typed`.
- **Configurable inference**: a uniform `se_type=` on regression estimators;
  configurable interval coverage; cumulative IRF views.
- **Evidence beyond fixtures**: a seeded Monte Carlo validation suite (size /
  coverage / consistency) and frontier experiments (LP vs VAR; weak-IV LP-IV);
  a 25-case cross-library parity benchmark harness (statsmodels, SciPy,
  scikit-learn, `arch`) that gates CI; two replications of published results —
  Ramey-Zubairy (2018) government-spending multipliers and Estrella-Mishkin
  (1998) yield-curve recession prediction — running offline from committed
  public data; Rust and Python coverage tooling.
- **Docs**: a 15-chapter teaching guide, model cards for every estimator
  family, a generated API reference with a drift guard, a validation matrix,
  a testing-and-validation map, migration guides (statsmodels/R/Stata) with a
  Rosetta glossary, a worked figure gallery, and an interactive demo.
- **Packaging**: complete `pyproject.toml` metadata, abi3-py39 wheels tested
  on Python 3.9 and 3.13 in CI, GitHub Actions CI (Rust gates, a three-OS
  wheel matrix, a mypy stub check, and an evidence job running the Monte Carlo
  suites and the parity gate), and a tag-triggered release pipeline with PyPI
  trusted publishing.

### Removed
- **Data-fetching loaders** (`tsecon.datasets`): built, then deliberately
  removed before release. A library that hardcodes external data URLs owns
  their breakage (the widely-cited FRED-MD URL had already moved), so tsecon
  ships no network code — the only runtime dependency is NumPy, and the
  replications run on small public datasets committed to the repository.

Every estimator is validated against a reference implementation (statsmodels,
SciPy, NumPy, `arch`, `linearmodels`, scikit-learn, ArviZ) or a documented
closed form in the test suite.
