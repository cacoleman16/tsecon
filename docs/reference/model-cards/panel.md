# Model card — Panel time series

**Family:** `panel_fe`, `panel_distributed_lag`, `panel_lp`, `lp_did`,
`mean_group_var`, `panel_mean_group`, `panel_pmg`

Many entities, each observed over time. The methods here span the two ends of
the panel spectrum: **pooled** estimators that assume a common slope and
difference out fixed effects (`panel_fe`, `panel_lp`), a **causal event-study**
estimator built on the local-projection idea (`lp_did`), and **heterogeneous**
estimators that let every unit have its own dynamics and then average or pool
carefully across them (`mean_group_var`, `panel_mean_group`, `panel_pmg`). The
recurring theme is honest inference: cross-sectional and serial correlation
both bias naïve standard errors, so the defaults reach for Driscoll-Kraay and
cluster covariances.

| Function | Slope assumption | Delivers |
|----------|------------------|----------|
| `panel_fe` | common | Fixed-effects OLS with robust SEs |
| `panel_distributed_lag` | common | Distributed-lag (climate-impact) regression: cumulative effect, turning point |
| `panel_lp` | common | Panel local-projection IRF of a common shock |
| `lp_did` | ATT (VW or EW) | LP-DiD event-study DiD with clean controls |
| `mean_group_var` | heterogeneous | Mean-group panel VAR + orthogonalized IRFs |
| `panel_mean_group` | heterogeneous | Mean-group / CCE-MG average slope |
| `panel_pmg` | pooled long run, free short run | Pooled Mean Group ARDL(1,1) |

## What it estimates

- **`panel_fe(outcome, regressors)`** — the within (fixed-effects) estimator:
  entity means are swept out and a common slope vector is estimated by OLS,
  with clustered or Driscoll-Kraay standard errors. `outcome` is N×T;
  `regressors` is k×N×T. An unbalanced panel is passed with `mask=` (an
  N×T array of 0/1 flags, 1 where the entity is observed) — see
  [the mask section](#unbalanced-panels-the-observation-mask-mask).
- **`panel_distributed_lag(outcome, regressors, lags)`** — the distributed-lag
  panel regression of the climate-impact literature: `L` lags of each
  regressor (and of its square under `powers=2`) with entity and time effects
  and optional entity trends, read off as the **cumulative effect** Σ_l β_l
  with a delta-method SE and interval and, for the quadratic response, the
  marginal effect at chosen points and the turning point of the cumulative
  response. See the [dedicated section below](#distributed-lag-panel-regressions-panel_distributed_lag).
- **`panel_lp(outcome, shock)`** — a panel local projection: at each horizon h,
  regress the h-step-ahead outcome on a **common** shock with entity fixed
  effects, tracing a dynamic causal response averaged across units.
- **`lp_did(outcome, treatment)`** — LP-DiD (Dube-Girardi-Jordà-Taylor 2025):
  per-horizon regressions of the long difference `y[i,t+h] − y[i,t−1]` on the
  treatment-switch indicator with period effects, restricted to **clean
  controls**, tracing the dynamic average treatment effect on the treated
  around unit-level treatment events. `treatment` is an N×T binary indicator.
  See the [dedicated section below](#lp-did-lp_did-event-study-did-with-clean-controls).
- **`mean_group_var(entities)`** — fits a separate VAR to each entity's Tᵢ×k
  matrix and averages the coefficients and orthogonalized IRFs (Pesaran-Smith
  1995). Robust to slope heterogeneity that a pooled panel VAR would bias.
- **`panel_mean_group(ys, xs)`** — the mean-group estimator: per-unit OLS
  slopes averaged across units, with the cross-unit standard deviation giving
  the standard error. `method="cce"` adds Pesaran (2006) common-correlated-
  effects terms (cross-sectional averages) to purge a common factor.
- **`panel_pmg(ys, xs)`** — the Pooled Mean Group ARDL(1,1) estimator
  (Pesaran-Shin-Smith 1999): the **long-run** coefficient θ is pooled (common
  across units) by maximum likelihood, while the error-correction speed and
  short-run dynamics stay unit-specific.

## Assumptions

- **`panel_fe` / `panel_lp` assume a common slope.** If the true response
  differs across units, the pooled estimate is a variance-weighted average that
  need not equal the cross-sectional mean effect — reach for the mean-group
  estimators instead.
- **Cross-sectional dependence.** With a common shock or common factor, errors
  are correlated across entities at each date; cluster-by-entity SEs do not
  address this. `se_type="driscoll_kraay"` is the default for `panel_lp`
  precisely because it is robust to both serial and cross-sectional
  correlation.
- **`panel_pmg` requires a genuine ARDL / error-correction structure**: the
  long-run regressors must be non-degenerate and not collinear across the panel
  once short-run dynamics are partialled out, or θ is not identified (the
  estimator raises rather than returning a meaningless number). Feed it *level*
  series with real dynamics, not, say, a shock and its own lag.
- **Mean-group estimators need enough time per unit** to estimate each unit's
  regression; they trade the efficiency of pooling for robustness to
  heterogeneity, and are noisy when Tᵢ is small.
- **`panel_mean_group(method="mg")` is a static regression** of y on
  contemporaneous x — its average slope is *not* the ARDL long-run coefficient.
  Use `panel_pmg` when the object of interest is a common long-run relationship.

## When to use

- **`panel_fe`** — the workhorse when you believe in a common slope and want
  clustered or Driscoll-Kraay inference (e.g. the effect of a policy variable
  across countries).
- **`panel_lp`** — dynamic causal responses to a *common* shock (a global oil
  or monetary shock hitting many countries), fixed effects for level
  differences, Driscoll-Kraay bands.
- **`lp_did`** — dynamic causal effects of a *unit-level binary* treatment
  (a policy adopted by different states in different years): event-study
  coefficients with pre-trends, a pooled ATT, and none of the TWFE
  negative-weighting pathologies. If your "shock" is a common aggregate
  series, use `panel_lp`; if it is a 0/1 adoption indicator per unit, use
  `lp_did`.
- **`mean_group_var`** — impulse responses in a heterogeneous panel where a
  pooled VAR would be misspecified.
- **`panel_mean_group`** — the average marginal effect across heterogeneous
  units; `method="cce"` when an unobserved common factor contaminates OLS.
- **`panel_pmg`** — long-run equilibrium relationships (growth-savings,
  consumption-income) where theory says the long run is common but adjustment
  speeds differ by country.

## Key arguments and defaults

| Call | Argument | Default | Notes |
|------|----------|---------|-------|
| `panel_fe` | `se_type` | `"cluster"` | `"nonrobust"`, `"cluster"` (by entity), `"driscoll_kraay"` |
| | `mask` | `None` | N×T 0/1 (or bool) flags, 1 where the entity is observed in that period — an **unbalanced** panel; cells outside the mask are ignored and may hold NaN; without a mask a NaN anywhere is refused (0.10.0; see [the mask section](#unbalanced-panels-the-observation-mask-mask)) |
| | `bandwidth` | `None` | Driscoll-Kraay kernel truncation, `4.0` when omitted under `se_type="driscoll_kraay"` — the only `se_type` it acts on. Passing it explicitly with any other `se_type` **raises** (0.6.0; it used to be silently absorbed, so `panel_fe(..., bandwidth=8)` under the default `"cluster"` was a complete no-op) |
| `panel_distributed_lag` | `lags` | required | `L`; lags `0..L` of every regressor enter and the first `L` periods of each entity are dropped (under a mask, a lagged row enters only when the entity is observed in every period `t−L..t`) |
| | `powers` | `1` | `2` adds the lags of `x²` — the Burke-Hsiang-Miguel quadratic response |
| | `entity_effects` / `time_effects` | `True` / `True` | the DJO/BHM two-way design; at least one effect is required |
| | `entity_trends` | `False` | entity-specific linear trends (BHM's robustness spec); requires `entity_effects=True` |
| | `se_type` | `"cluster"` | by entity — the DJO default; `"driscoll_kraay"` for cross-sectionally dependent weather (needs a long `T`); `"nonrobust"` |
| | `bandwidth` | `None` | same contract as `panel_fe`: `4.0` when omitted under `"driscoll_kraay"`; explicit under any other `se_type` **raises** |
| | `eval_points` | `None` | points at which the marginal effect of the cumulative quadratic response is evaluated; acts only under `powers=2` (`None` there = each regressor's pooled mean over the observed cells) and **raises** under `powers=1` |
| | `mask` | `None` | same contract as `panel_fe`; `nobs` counts the lagged rows whose every lag is observed |
| `panel_lp` | `horizon` | `8` | IRF horizons |
| | `n_lag_controls` | `2` | lags of outcome/shock included as controls |
| | `se_type` | `"driscoll_kraay"` | robust to cross-sectional dependence |
| | `bandwidth` | `None` | same contract as `panel_fe`: `4.0` when omitted under `"driscoll_kraay"`, and an explicit bandwidth with `"cluster"`/`"nonrobust"` **raises** instead of being silently absorbed |
| | `cumulative` | `False` | `True` for cumulative IRFs |
| | `jackknife` | `False` | Dhaene-Jochmans half-panel (time-split) jackknife; corrects the **points only**, SEs stay the full-sample plug-in (measured: 95% coverage 0.880 → 0.804 at T=60 — see the [panel-LP cookbook](../../cookbook/panel-lp-standard-errors.md#gotchas)) |
| | `bias_correction` | `"none"` | `"spj"` = Mei-Sheng-Shi split-panel jackknife: corrected points **and** the reference adjusted-score SEs; `"dj"` = alias for `jackknife=True`. Setting `jackknife=True` together with `"spj"` raises |
| | `band` | `None` | `"pointwise"`, `"sidak"` or `"bonferroni"` adds a band over the horizons; `"sup-t"` is refused — see [the band section](#simultaneous-bands-over-the-horizons-panel_lp) |
| | `band_alpha` | `0.1` | the band's own level (a 90% band); a band is not the same object as an `se` |
| | `mask` | `None` | same contract as `panel_fe`: the horizon-h regression keeps the rows whose target (or cumulated window) and lagged-outcome controls are observed, so `nobs` shrinks with the gaps too; `jackknife=True` and `bias_correction="dj"`/`"spj"` **raise** on an unbalanced panel |
| `lp_did` | `pre_window` / `post_window` | `4` / `8` | event window: horizons −pre..−2 (pre-trends) and 0..post; −1 is the omitted baseline |
| | `absorbing` | `True` | treatment never reverses (raises on a reversal); `False` requires `nonabsorbing_lag` |
| | `nonabsorbing_lag` | `0` | stabilization window L for non-absorbing treatments (units re-enter the control pool L quiet periods after a status change) |
| | `reweight` | `False` | `True` = equally-weighted ATT; default OLS = variance-weighted ATT |
| | `mask` | `None` | accepted for symmetry; an unbalanced mask **raises** — LP-DiD needs a balanced panel (see the mask section) |
| | `pooled` | `False` | also report single-number pooled post/pre estimates |
| | `never_treated_only` | `False` | restrict controls to never-treated units |
| `mean_group_var` | `lags` | `1` | per-entity VAR order |
| | `trend` | `"c"` | deterministic terms |
| | `horizon` / `response` / `impulse` | `10` / `0` / `0` | IRF horizon and the response/shock variable indices |
| `panel_mean_group` | `method` | `"mg"` | or `"cce"` (common-correlated-effects) |
| `panel_pmg` | `tol` | `3e-13` | *relative* stopping tolerance of the ML back-substitution: stop when `|dθ|_inf ≤ tol·(1+|θ|_inf)` — see [the convergence note](#panel_pmg-convergence-the-i1-repair) |
| | `max_iter` | `1000` | iteration budget per pass (θ=0 start, then the unrestricted-ARDL restart) |

## How to read the output

- **`panel_fe`** → `{"params", "bse", "tvalues", "se_type"}`, one entry per
  regressor. The stamped `se_type` tells you which covariance produced `bse`.
- **`panel_lp`** → `{"irf", "se", "nobs", "se_type", "cumulative", "jackknife", "bias_correction"}` — the first three each length `horizon+1`, the last four the resolved settings stamped on the result (`band=` adds the band keys); plot `irf`
  ±1.96·`se` for the usual per-horizon read, and note that band is
  **pointwise** — for a statement about the whole path, ask for a
  [simultaneous band](#simultaneous-bands-over-the-horizons-panel_lp).
  `irf[0]` is the impact response. The method metadata is stamped
  on the result — `se_type`, `cumulative`, `jackknife`, and
  `bias_correction` (`"none"` / `"dhaene_jochmans"` / `"spj"`) — so a saved
  result records which estimator and covariance produced it. With `band=`
  set, the band keys (`lower`/`upper`, `critical_value`,
  `pointwise_critical_value`, `band`, `band_alpha`, `band_scope`,
  `n_cells`, `n_cells_used`) are added; without it the result is exactly
  the historical three-plus-metadata dict.
- **`lp_did`** → `{"horizons", "coef", "se", "nobs", "n_switchers", "absorbing", "nonabsorbing_lag", "reweight", "pooled", "never_treated_only", "se_type"}` (the last six are the resolved settings stamped on the result; `pooled=True` adds the `pooled_post_*`/`pooled_pre_*` keys) aligned
  per event-time horizon (−pre_window..post_window; the −1 row is the omitted
  baseline, stored as exact zeros), plus `pooled_post_att`/`pooled_post_se`/
  `pooled_post_nobs`/`pooled_post_n_switchers` (and the `pooled_pre_*` four)
  when `pooled=True`, and the stamped options (`absorbing`,
  `nonabsorbing_lag`, `reweight`, `pooled`, `never_treated_only`,
  `se_type="cluster_entity"`). **Read `nobs` and `n_switchers`**: the
  clean-control samples shrink as |h| grows, and an event-study point
  estimated from a handful of switchers is noise wearing a confidence
  interval.
- **`mean_group_var`** → per-entity-averaged `intercept`, `coefs`
  (lags × neqs × neqs) and their SEs, plus `orth_irfs`
  (horizon+1 × response × shock) with SEs and a convenience `irf_path`
  (the `response`/`impulse` cell) and `irf_path_se`. Also `n_entities`,
  `neqs`, `lags`.
- **`panel_mean_group`** → `{"coef", "se", "tstat", "coef_per_unit",
  "n_units", "k"}`. `coef_per_unit` (n_units × k) lets you inspect the spread
  of individual slopes behind the average.
- **`panel_pmg`** → `{"theta", "theta_se", "phi_bar", "phi", "sigma2",
  "loglik", "iterations", "n_units", "k"}`. `theta` is the pooled long-run
  coefficient; `phi_bar` is the average error-correction speed (negative and
  bounded by −1 for stable adjustment); `phi` is the per-unit speed vector.

## Unbalanced panels — the observation mask (`mask=`)

Until 0.9.0 every panel callable refused a NaN anywhere and required a
balanced N×T layout. Since 0.10.0 an **unbalanced** panel — entities entering
late, leaving early, or with internal gaps — is declared through `mask=`, an
N×T array of 0/1 (or bool) flags with 1 where the entity is observed in that
period. Nothing is skipped silently: without a mask a NaN anywhere is still
refused (the message now points at `mask=`), a NaN in an *observed* cell is
refused naming the input, and a mask that is 1 everywhere takes exactly the
balanced code path (asserted bit-identical). Cells outside the mask are
never read and may hold anything.

What the estimators do with it, and what they refuse:

- **`panel_fe`** (and every effects menu reachable through
  `panel_distributed_lag`): the per-entity projection — demeaning, or
  residualising on `[1, t]` under `entity_trends` — runs over each entity's
  observed cells; time effects are then partialled out **exactly**, the way
  linearmodels' `PanelOLS` does it on unbalanced data (the Frisch-Waugh
  route: the time dummies of the observed periods, projected the same way,
  are residualised out by a rank-revealing least-squares step), and
  time-only effects demean per period over the entities observed in it.
  Effect counts use the entities and periods that carry an observation
  ($N_\text{obs} + T_\text{obs} - 1$ under two-way effects, trend slopes
  $N_\text{obs}$ or $N_\text{obs} - 1$ alongside time effects), the
  cluster score sums run over each entity's observed cells, and the
  Driscoll-Kraay per-period sums over the entities observed in each period
  with lags measured in **calendar** periods of the layout (a period nobody
  observes contributes a zero score and still counts as elapsed time —
  linearmodels, which only knows the periods present in its frame, would
  skip it; the fixtures observe every period, so the two agree).
- **`panel_distributed_lag`**: a lagged row $(i, t)$ enters only when entity
  $i$ is observed in every period $t-L, \dots, t$ — exactly what a
  within-entity `shift` followed by dropping incomplete rows does — so
  `nobs` counts those rows, the default `eval_points` (the pooled regressor
  mean) runs over the observed cells, and a mask with no run of $L+1$
  observed periods is refused with the longest run in the message.
- **`panel_lp`**: the horizon-$h$ regression keeps the rows whose target
  ($y_{i,t+h}$, or the whole cumulated window) and lagged-outcome controls
  are observed; `nobs[h]` therefore shrinks with the gaps as well as the
  horizon, and the entity composition can change across horizons — read it.
  The two half-panel jackknives (`jackknife=True`, `bias_correction="dj"` /
  `"spj"`) **raise** on an unbalanced panel: their bias reduction assumes
  each half carries the full panel's incidental-parameter bias, which
  entry, exit and gaps break, and the split-panel reference implementation
  (`pLP`) is written for balanced panels.
**What it costs.** The per-entity projection and the one-way menus stay
linear in the panel. The **unbalanced two-way** case does not: the time
effects are partialled out through the projected time dummies, one per
observed period, by a rank-revealing least-squares step, so the work grows
with the cube of the number of periods. Measured on `panel_distributed_lag`
at $N = 6$, `lags=1`, `powers=2` (audit round 14,
`lab/audit/round14/out/sweep_g.txt`): **0.11 s at $T = 400$, 0.69 s at
$T = 800$, 5.2 s at $T = 1600$, 38 s at $T = 3200$** — a factor of 8 per
doubling — against **3 ms** at $T = 3200$ for the same panel with no mask, or
with a mask and `time_effects=False`, both of which are linear. A mask of all
ones takes the balanced path and is bit-identical to no mask at any size. So:
a long unbalanced panel is practical without time effects, or after trimming
$T$; with two-way effects, keep $T$ in the hundreds. (The alternative —
alternating projections — is an approximation; the exact route is what the
`PanelOLS` goldens pin, and it is not being traded for speed here.)

- **`lp_did`** accepts `mask=` for symmetry but **raises** on an unbalanced
  one: the clean-control windows and the long differences are defined on
  contiguous outcome paths, and the fixest reference run that validates the
  estimator was made on balanced panels (R is not runnable in the build
  container, so a masked LP-DiD would ship without its reference).
- **`mean_group_var`** takes a *list* of per-entity matrices, so entry and
  exit are simply shorter matrices; an internal NaN is refused naming
  `entities[i]` (a VAR's lags must be contiguous — trim each entity to a
  contiguous span).

**Validation (honest grade).** `fixtures/panel_unbalanced.json`
(`generate_panel_unbalanced_fixtures.py`) pins linearmodels 7.0 `PanelOLS`
on the Arellano-Bond **`EmplUK`** panel (`plm`; 140 firms, 1976–1984, 1031
firm-years — 103 firms observed 7 years, 23 for 8, 14 for 9) and on a seeded
N = 20 × T = 30 panel with random entry, exit and 8% internal gaps: the
within estimator under every effects menu × three covariances (10 cases),
the lagged design (12 cases, incl. the delta-method leg) and the panel LP
per horizon on exactly the rows the crate keeps (3 cases × 3 covariances),
all at **1e-10 relative** with `nobs`/`df_resid` exact
(`unbalanced_golden.rs`, `test_panel_unbalanced.py`). Every balanced call is
pinned **bitwise** against the 0.9.0 build through
`fixtures/panel_balanced_snapshot.json` (a float-hex self-snapshot in the
`backtest_string_snapshot.json` mould), so the mask changed no balanced
number.

## Nickell bias and the two half-panel corrections (`panel_lp`)

Fixed effects + dynamics + short T biases a panel local projection even when
no lagged dependent variable appears among the regressors: the within
transformation correlates the demeaned regressors with the demeaned
horizon-`h` error, the bias is `O(1/T)` and horizon-amplified (roughly
`O(h/T)`), it does **not** shrink with the number of entities, and it
invalidates the t-test (Nickell 1981; Mei-Sheng-Shi 2026). `panel_lp` offers
two half-panel corrections. Both replace the point estimates with
`2·θ_full − (θ_half1 + θ_half2)/2`; they differ in bookkeeping and —
decisively — in the standard error:

- **`jackknife=True`** (equivalently `bias_correction="dj"`) — the
  Dhaene-Jochmans (2015) jackknife as originally shipped: each half is
  re-estimated strictly inside its own time window (lags re-burnt, leads
  truncated at the split; halves overlap by one period when T is odd), and
  the reported `se` is the **unchanged full-sample plug-in** — asymptotically
  justified (DJ Theorem 3.1) but a finite-T approximation with a measured
  cost: the round-2 audit found that at N=20, T=60, h=8 the correction
  removes the bias but inflates the estimator's sd by ~36% while `se` is
  bit-identical, costing 8pp of coverage (0.880 → 0.804); the equivalence
  arrives by T≈240. Use it when T is moderate, and read the bands
  accordingly.
- **`bias_correction="spj"`** — the Mei-Sheng-Shi (2026, *J. International
  Economics*) split-panel jackknife for panel LPs, matching their reference
  implementation (the `pLP` R package): leads and lags are computed on the
  full panel and only the per-horizon regression rows are split, at the floor
  of the median usable period (odd row counts give the extra row to the first
  half; no overlap), so nothing is burnt at the boundary — and the SE is
  **recomputed for the corrected estimator**: residuals at the SPJ
  coefficients, sandwich meat from the jackknife-adjusted scores
  `2·x̃ − x̃_half`, full-sample bread. Cluster-by-entity and Driscoll-Kraay
  variants ship (`se_type="nonrobust"` is refused — the reference provides no
  homoskedastic SPJ variance, and a plug-in SE would repeat the DJ trap).
  Following `pLP`, the SPJ cluster covariance uses the Stata-style
  `(N/(N−1))·((n−1)/(n−k))` factor and the SPJ Driscoll-Kraay applies no
  small-sample factor — deliberately different conventions from the
  uncorrected route's linearmodels ones; `pLP` hardcodes the DK lag
  truncation to `floor((T−h)^(1/4))` while tsecon honours your `bandwidth`
  (set it to that value to reproduce `pLP`).

The two corrections coincide in the **points** only where the two
half-samples are the same rows (horizon 0, no lag controls, even T — a pinned
test); the SPJ SEs differ by construction everywhere. Requesting both at once
raises.

**Measured Monte Carlo evidence** (seed 20260818, 300 replications per T,
N=50, `y_{i,t} = α_i + 0.8·y_{i,t−1} + 0.8·s_t + e_{i,t}` with a common iid
shock; one shock lag + one outcome lag; Driscoll-Kraay `bandwidth=2`; true
IRF `0.8·0.8^h`; `crates/tsecon-panel/tests/spj_properties.rs` reproduces
this table):

| T | h | bias FE | bias SPJ | \|FE\|/\|SPJ\| | 95% cov FE | 95% cov SPJ |
|---|---|---------|----------|---------------|------------|-------------|
| 20 | 1 | −0.094 | +0.023 | 4.1x | 0.770 | 0.817 |
| 20 | 2 | −0.137 | +0.009 | 15.2x | 0.743 | 0.823 |
| 40 | 1 | −0.035 | +0.015 | 2.3x | 0.900 | 0.873 |
| 40 | 2 | −0.054 | +0.019 | 2.9x | 0.870 | 0.863 |
| 80 | 1 | −0.009 | +0.012 | 0.7x | 0.920 | 0.907 |
| 80 | 2 | −0.013 | +0.019 | 0.7x | 0.900 | 0.877 |

A 2000-replication rerun (independent seed) sharpens the noisy cells: SPJ
bias at h=2 is +0.007 / +0.002 / +0.003 at T=40/80/160 against FE's
−0.060 / −0.025 / −0.010 — FE shrinks like `O(1/T)` and SPJ removes ~85-95%
of the bias; the 300-rep T=80 SPJ entries are Monte-Carlo noise (the columns
share draws). Read the coverage honestly: at T=20 the FE t-interval is
clearly invalid and SPJ improves it only modestly — the 300-rep table above
shows 0.743 → 0.823, but the interval-coverage audit's 2500-rep re-measurement
at the same design puts the paired gain at **+2.5pp (se 0.6), 0.713 → 0.761**,
so treat SPJ's short-T coverage benefit as small; its real value at short T is
the bias removal (which the re-measurement corroborates almost exactly:
−0.141 → +0.015). **Neither estimator reaches the nominal 95% at T=20**,
because with a common shock the horizon-h residual contains common future
shocks, Driscoll-Kraay is the right covariance family, and DK is itself a
short-T approximation (the same caveat this card and the cookbook attach to DK
generally; see the [interval-coverage audit](../../examples/interval-coverage.md)
for the full (N, T) table). From T=40 both sit in the high-0.80s/low-0.90s
with the bias gone from SPJ.

## Simultaneous bands over the horizons (`panel_lp`)

A panel-LP impulse response is a *path*, and `irf ± z·se` is a **pointwise**
band: it covers each horizon separately at the nominal rate and promises
nothing about the path as a whole. `panel_lp(..., band=)` adds the same
closed-form band selector as `lp_iv`/`lp_multiplier`/`lp_state` —
`"pointwise"`, `"sidak"`, `"bonferroni"` at level `band_alpha`, default
`band=None` returning exactly the historical result — with the band keys
(`lower`/`upper`, `critical_value`, `pointwise_critical_value`,
`band_scope="horizon"`, `n_cells = horizon+1`, `n_cells_used`) added only
when a band is requested. The simultaneous-band framework is Montiel Olea
and Plagborg-Møller's (see the
[LP card's band section](local-projections.md#simultaneous-bands-over-the-horizons-lp)).

**`band="sup-t"` is refused, with an error naming the reason.** Sup-t needs
the covariance of the IRF *across horizons*, and tsecon estimates no such
covariance for the panel LP (each horizon is its own within regression; a
cross-horizon influence-function covariance under entity clustering /
Driscoll-Kraay weighting is a documented follow-up in `tsecon-panel`). Šidák
and Bonferroni need nothing but `K`, and are simply wider than a sup-t band
would be. Never describe a band from `panel_lp` as sup-t.

**Measured joint coverage** (seeded MC, seed 20260823, 200 reps per cell,
known-truth DGP `y_{i,t} = α_i + Σ_j ψ_j s_{t-j} + ε_{i,t}` with an iid
common shock, so the estimand is exactly `ψ_h = 0.8·0.6^h`; K = 9 horizons,
two outcome/shock lag controls, Driscoll-Kraay `bandwidth=10`; nominal 90%;
"joint" = the whole 9-horizon path inside the band at once; MC se ≈ 2–3.5pp;
the N=24/T=160 cell runs in CI as
`test_simultaneous_bands.py::test_panel_lp_joint_coverage_pointwise_fails_and_closed_forms_repair_it`):

| design | pointwise, joint | Šidák | Bonferroni | pointwise marginals |
|---|---|---|---|---|
| N=24, T=160 | **0.305** | 0.765 | 0.765 | 0.815–0.880 |
| N=30, T=400 | **0.405** | 0.840 | 0.845 | 0.845–0.920 |
| N=30, T=800 | **0.425** | 0.880 | 0.890 | 0.840–0.905 |

Read it the way the library reads all its band measurements. (1) The
pointwise joint rate is not converging to 0.90 — quintupling T bought twelve
points — because the failure is multiplicity, not consistency. (2) The union
bounds fix multiplicity **only**: at T=160 they sit at 0.765 because the
per-horizon Driscoll-Kraay standard errors themselves run a few points short
at short T (the same documented DK caveat as everywhere else on this card),
and they rise to ≈nominal exactly as the DK marginals do. A wider multiplier
cannot repair a short standard error. (3) Šidák and Bonferroni are
conservative on a smooth IRF path in principle (they ignore the positive
cross-horizon correlation a sup-t band would exploit); the price at K=9,
`alpha=0.10` is a multiplier of 2.5229/2.5392 against the pointwise 1.6449.

**Validation status, stated honestly.** The critical values themselves are
the same SciPy-pinned closed forms as every other band surface (see the
[validation matrix](../validation-matrix.md)); the joint-coverage claim is
graded **property-MC (joint coverage measured)** — there is no third-party
reference for simultaneous local-projection bands in Python (statsmodels
ships none), so there is nothing external to pin a golden against.

## Distributed-lag panel regressions (`panel_distributed_lag`)

The workhorse of the climate-economy literature — Dell, Jones & Olken (2012),
Burke, Hsiang & Miguel (2015), reviewed in Hsiang (2016) — is a fixed-effects
regression of an outcome on the current and lagged values of a weather
variable:

$$
y_{it} = \sum_{l=0}^{L} \beta_l\, x_{i,t-l} \;\Big[+ \sum_{l=0}^{L} \gamma_l\, x^2_{i,t-l}\Big]
\;+\; \alpha_i + \delta_t \;[+\; g_i\, t] \;+\; \varepsilon_{it}.
$$

The individual lag coefficients are rarely the object of interest. The
**cumulative effect** $B = \sum_l \beta_l$ is: it is the long-run impact of a
permanent one-unit change in $x$ once $L$ periods have elapsed, and the
contrast between $\beta_0$ (a level effect that reverses) and $B$ (a growth
effect that persists) is the DJO test of whether temperature affects the level
or the growth rate of output. Under the quadratic response the same logic
gives the **marginal effect of the cumulative response** at a point $x$,
$B_1 + 2B_2 x$, and its **turning point** $x^* = -B_1/(2B_2)$ — the BHM
"optimal temperature" around 13 °C — with delta-method standard errors from the
joint covariance of the lag coefficients (the gradient of $x^*$ is
$-1/(2B_2)$ in every power-1 lag and $B_1/(2B_2^2)$ in every power-2 lag).

### What it assumes, and when not to use it

- **Strict exogeneity of the regressor.** Weather is the canonical case: it is
  not caused by output, and its lags are not caused by past output shocks.
  The design deliberately has **no lagged dependent variable** — putting one
  in `regressors` would reintroduce Nickell (1981) bias into a short-$T$
  within estimator (the [section above](#nickell-bias-and-the-two-half-panel-corrections-panel_lp)
  explains the mechanism); use `panel_lp` with a bias correction for dynamic
  panels. Distributed lags of a strictly exogenous regressor carry no such
  bias.
- **Missing cells are declared, never guessed.** A NaN in an observed cell
  is refused; an unbalanced panel says which cells are missing through
  `mask=` (0.10.0; [the mask section](#unbalanced-panels-the-observation-mask-mask)),
  and a lagged row enters only when every one of its $L+1$ cells is
  observed, so entities entering late or leaving early cost nothing but
  rows while an internal gap costs the $L$ rows after it. Trimming to a
  balanced window remains a *choice* — say so when you make it, because the
  balanced subsample is a different population than the full panel (the
  DJO example below keeps its balanced subsample so its printed numbers stay
  comparable across releases; with `mask=` the full country-year file can
  now be used directly).
- **A common lag polynomial.** The within estimator pools $\beta_l$ across
  entities; DJO's rich-poor split is a hand-built interaction (pass
  `temperature × poor` as a second regressor).
- **Inference.** Clustering by entity (the DJO default) is a large-$N$
  approximation that ignores cross-sectional dependence; weather is
  spatially correlated, so with few entities or a strong regional component
  prefer `se_type="driscoll_kraay"`, which is robust to it but is a large-$T$
  approximation — the measured coverage below shows the price at $T = 50$.
  The 95% intervals use the normal critical value 1.959964.
- **Time effects absorb any regressor common to every entity** (a global
  temperature series, a year trend): such a column is refused as absorbed,
  not returned with a spurious coefficient.

### What is returned

`params`, `names` (columns ordered regressor-major, then power, then lag:
`x0_L0`, `x0_L1`, …, `x0^2_L0`, …), `bse`, `tvalues`, the full `cov`;
`lag_effects` / `lag_se` indexed `[regressor][power-1][lag]`;
`cumulative_effect`, `cumulative_se` ($\sqrt{\mathbf{1}'V\mathbf{1}}$),
`cumulative_ci_low`, `cumulative_ci_high` indexed `[regressor][power-1]`; under
`powers=2` the `eval_points` used, `marginal_effect` and `marginal_se`
(`[regressor][point]`), `turning_point` and `turning_point_se`
(`[regressor]`; NaN when $B_2$ is exactly zero) — `None` under `powers=1`;
and `nobs` ($N(T-L)$), `n_entities`, `n_periods_used` ($T-L$), `lags`,
`powers`, `df_resid`, `se_type` and the three effect flags. The call
`panel_distributed_lag(y, x, lags=0, time_effects=False)` is bit-identical to
`panel_fe(y, x)`.

### Validation and its honest grade

- **Independent-package golden** (`fixtures/panel_dl.json`,
  `generate_panel_dl_fixtures.py`): nine cases — linear responses at
  $L = 0, 1, 2, 3$ under two-way, entity-only and time-only effects, entity
  trends with and without time effects, two regressors, and the quadratic
  response with explicit and default evaluation points — each pinned at
  **1e-10 relative** against linearmodels 7.0 `PanelOLS` on the explicitly
  lagged design for slopes, standard errors, t-statistics and the full
  covariance under `cov_type` `"unadjusted"`, `"clustered"` (entity) and
  `"kernel"` (Bartlett, bandwidth 4); `nobs` and `df_resid` exact. The trends
  variant is pinned via explicit entity × trend regressors ($N-1$ columns when
  time effects are present — the common trend already lies in their span).
  The inputs are stored at full double precision, so unlike `panel.json` the
  ceiling is the estimator, not the fixture (worst measured relative error
  is reported by `test_panel_dl.py`).
- **Documented-formula golden**: the cumulative effect, its delta-method SE
  and interval, the marginal effects and the turning point against the NumPy
  transcription in the generator's docstring at 1e-10 — a check that the
  delta method is the documented one, *not* an independent authority for it
  (no Python package reports these objects).
- **Property-MC, coverage measured** (`panel_dl_properties.rs`, seed
  20260910, 500 replications per cell, $L = 2$, true long-run impact 0.70,
  two-way effects, nominal 95%):

  | covariance | cell | coverage | mean estimate | mean SE |
  |---|---|---|---|---|
  | cluster (entity) | $N = 50$, $T = 30$ | 0.928 | 0.7000 | 0.0642 |
  | cluster (entity) | $N = 200$, $T = 30$ | 0.938 | 0.6998 | 0.0327 |
  | Driscoll-Kraay (bw 4) | $N = 25$, $T = 50$ | 0.886 | 0.7020 | 0.0611 |
  | Driscoll-Kraay (bw 4) | $N = 25$, $T = 200$ | 0.924 | 0.6978 | 0.0335 |

  The clustered cells use AR(1) errors with entity-specific scales and no
  cross-sectional dependence; the Driscoll-Kraay cells add a common AR(1)
  weather component with heterogeneous loadings and a common AR(1) error
  factor with heterogeneous loadings (so the time effects absorb only the
  mean loading). Read honestly: the estimator is unbiased in every cell, the
  clustered interval runs 1–2 points short of nominal (the usual
  finite-cluster shortfall, shrinking with $N$), and Driscoll-Kraay is
  6 points short at $T = 50$ — the short-$T$ kernel caveat — recovering to
  0.92 at $T = 200$. Same seed, bit-identical output; entity relabelling and
  (at $L = 0$) period relabelling leave the fit unchanged to 1e-10.
- **Unbalanced panels** (`fixtures/panel_unbalanced.json`): 12 lagged-design
  cases on the Arellano-Bond `EmplUK` panel and a seeded ragged panel, pinned
  at 1e-10 against `PanelOLS` with `nobs`/`df_resid` exact — see
  [the mask section](#unbalanced-panels-the-observation-mask-mask).
- **Real data** — see the DJO example below.

### Worked example — the Dell-Jones-Olken panel

DJO's replication panel (`climate_panel.dta`, the AEJ:Macro data archive;
fetched in the build container from a public course mirror on GitHub because
the AEA and Stanford hosts are unreachable through its proxy) has
population-weighted temperature `wtem` and WDI GDP `gdpLCU` for 1950–2006.
The committed script
[`docs/examples/panel_distributed_lag_djo.py`](../../examples/panel_distributed_lag_djo.py)
builds growth as $100\,\Delta\log(\text{GDP})$, keeps the countries observed
in every year of 1971–2003 (a **balanced subsample** of the DJO panel — DJO
use the full unbalanced panel with region × year and poor × year effects, so
these are not their published coefficients) and runs the linear DJO
specification and the BHM quadratic one with country and year effects and
country-clustered standard errors. Its measured output (growth in percentage
points per °C; standard errors in parentheses):

```text
balanced subsample: N = 100 countries, T = 33 years (1971-2003), 3300 country-years of 5758 with growth and temperature in the file
temperature: mean 19.67 C, sd 7.01; growth: mean 1.28, sd 5.63
linear  L=0: beta_0 = -0.380 (0.228)  cumulative = -0.380 (0.228)  95% [-0.826, +0.066]  nobs = 3300
linear  L=3: beta_0 = -0.301 (0.250)  cumulative = -0.412 (0.383)  95% [-1.162, +0.338]  nobs = 3000
quadratic L=0: B1 = +0.081 (0.348)  B2 = -0.0146 (0.0127)  turning point = 2.8 C (10.0)  marginal: 10C -0.212 (0.201), 20C -0.505 (0.299), 30C -0.798 (0.517)
quadratic L=3: B1 = -0.913 (0.540)  B2 = +0.0163 (0.0184)  turning point = 28.0 C (20.8)  marginal: 10C -0.587 (0.326), 20C -0.261 (0.440), 30C +0.065 (0.743)
```

Read it the way the callable is meant to be read. The linear contemporaneous
effect of a 1 °C warmer year is −0.38 growth points with a country-clustered
standard error of 0.23 — the sign and rough size of the DJO pooled effect,
though not their estimate, because the sample and the effects differ as
described above. Adding three lags moves the *cumulative* effect to −0.41
(0.38): the lags do not reverse the impact, which is the DJO
"growth-not-level" reading, but on 100 countries the interval on the sum of
four coefficients comfortably includes zero. The quadratic form is where the
balanced subsample and the plain two-way design run out of information: the
turning point is 2.8 °C with a delta-method standard error of 10 °C at
$L = 0$ and undefined in any useful sense at $L = 3$ (28 °C ± 21). BHM's
13 °C optimum is estimated on 166 countries with country trends and a much
richer control set; a two-way design on a third of the country-years cannot
pin the curvature, and the honest output says so through `turning_point_se`
rather than through a point estimate alone.

## LP-DiD (`lp_did`) — event-study DiD with clean controls

The local-projections difference-in-differences of Dube, Girardi, Jordà &
Taylor (2025, *J. Applied Econometrics*). Per horizon `h`, one regression:

```text
y[i,t+h] − y[i,t−1] = β_h ΔD[i,t] + δ_t + e[i,t]     estimated ONLY on
    { newly treated (ΔD[i,t] = 1) } ∪ { clean controls }
```

with period effects `δ_t` and standard errors clustered by entity. Pre-event
horizons (`h = −pre_window..−2`) run the same regression on
`y[i,t+h] − y[i,t−1]` and display pre-trends; `h = −1` is the omitted
baseline. The **clean-control condition** is the entire point: a two-way
fixed-effects event study implicitly compares newly treated units against
*already-treated* ones whose own dynamic effects sit in the "control"
outcome — the forbidden comparisons that give TWFE its negative weights
(Goodman-Bacon 2021; de Chaisemartin & D'Haultfœuille 2020). LP-DiD only ever
compares switchers with units that are:

- **not yet treated** through `t+h` (absorbing treatment, the default);
- **never treated** in the observed sample (`never_treated_only=True`);
- **stabilized** (`absorbing=False` + `nonabsorbing_lag=L`): treatment may
  turn on and off, and an observation is clean if its status did not change
  in `[t−L, t−1]` (post horizons also require no change through `t+h`; pre
  horizon `−j` widens the lag window to `[t−L−(j−1), t−1]`). This is the
  DGJT §3.2 effect-stabilization assumption: `L` periods after a change the
  dynamic effect is assumed settled, so previously-treated units re-enter the
  control pool. Exits (`ΔD = −1`) are excluded from both groups, and a
  status change outside the observed panel counts as clean (the reference
  implementation's Stata missing-value semantics — documented, not hidden).

**Variance-weighted vs equally-weighted ATT.** Plain OLS on the clean sample
gives a *variance-weighted* ATT: every period's clean 2×2 comparison enters
with non-negative weight proportional to `n_t p_t (1−p_t)` — no negative
weights, but precisely-estimated cohorts count more. `reweight=True`
reweights each period cell by the inverse of its switcher share (the DGJT
§2.5 construction, transcribed from the authors' `get_reweights`), which is
exactly equivalent to weighting each cell's comparison by its number of
switchers: the *equally-weighted* ATT across treated observations. Under
reweighting, period cells with no switcher drop from the sample, and a
switcher cell with no clean control raises (its equally-weighted
contribution is undefined) rather than silently degrading.

**Pooled ATT.** `pooled=True` adds two single-number estimates: the post
regression replaces the regressand with `mean(y[t..t+H]) − y[t−1]` on the
horizon-`H` clean sample (the average effect over the post window), and the
pre regression uses `mean(y[t−Q..t−2]) − y[t−1]` (a one-number pre-trend
test). Unlike the R port (where `pooled` replaces the event study), tsecon
always reports the event study and adds the pooled rows.

### Measured: recovery, coverage, and the naive contrast

Seeded Monte Carlo (seed 20260819, 300 replications;
`crates/tsecon-panel/tests/lpdid_properties.rs`). Recovery DGP: N = 60 (20
never-treated), T = 36, five adoption cohorts, homogeneous effect ramping to
6, true ATT(h) = h+1; variance-weighted LP-DiD with cluster-by-entity 95%
z-intervals:

| h | true | bias | coverage |
|---|------|------|----------|
| 0 | 1.0 | +0.008 | 0.970 |
| 1 | 2.0 | −0.002 | 0.953 |
| 2 | 3.0 | +0.002 | 0.970 |
| 3 | 4.0 | +0.001 | 0.963 |
| 4 | 5.0 | +0.011 | 0.960 |

The contrast that motivates the method (asserted in CI on every run, not
just described): heterogeneous cohort effects `θ_c · (e+1)` growing with
event time, only 4 never-treated units in 44. A *naive all-controls* variant
of the same horizon-3 regression — identical except previously-treated units
stay in the control pool — loses more than half the effect, because the
already-treated "controls" are still on their own effect trajectory
(200 replications, true equally-weighted ATT 5.20):

| estimator | mean | bias |
|-----------|------|------|
| LP-DiD (`reweight=True`) | 5.207 | +0.1% |
| naive all-controls LP | 2.260 | −56.5% |

### Worked example — staggered adoption, heterogeneous effects

```python
import numpy as np
import tsecon

rng = np.random.default_rng(31184)   # the working-paper number
N, T = 200, 40

# Staggered adoption: 60 never-treated units; the rest adopt in five
# cohorts (t = 8, 13, 18, 23, 28). Effects ramp in over 4 periods and are
# LARGER for early adopters (theta from 3.0 down to 2.0, mean 2.5) — the
# setting where TWFE event studies break and LP-DiD is designed to work.
adopt = np.full(N, -1)
adopt[60:] = np.tile([8, 13, 18, 23, 28], 28)
theta = np.where(adopt > 0, 3.0 - 0.05 * (adopt - 8), 0.0)

alpha, delta = rng.normal(0, 1, N), rng.normal(0, 1, T)
y = alpha[:, None] + delta[None, :] + rng.normal(0, 0.8, (N, T))
d = np.zeros((N, T))
for i in range(N):
    if adopt[i] > 0:
        d[i, adopt[i]:] = 1.0
        e = np.arange(T - adopt[i])
        y[i, adopt[i]:] += theta[i] * np.minimum(e + 1, 4) / 4.0

# LP-DiD event study: variance-weighted and equally-weighted.
vw = tsecon.lp_did(y, d, pre_window=4, post_window=6, pooled=True)
ew = tsecon.lp_did(y, d, pre_window=4, post_window=6, reweight=True,
                   pooled=True)

print("h    VW ATT   (se)    EW ATT   (se)    nobs   switchers")
for k, h in enumerate(vw["horizons"]):
    if h == -1:
        print("-1   (omitted baseline)")
        continue
    print(f"{h:+d}   {vw['coef'][k]:+.3f}  ({vw['se'][k]:.3f})  "
          f"{ew['coef'][k]:+.3f}  ({ew['se'][k]:.3f})  {vw['nobs'][k]:5d}   "
          f"{vw['n_switchers'][k]}")
print(f"pooled post ATT: VW {vw['pooled_post_att']:+.3f} "
      f"({vw['pooled_post_se']:.3f})   EW {ew['pooled_post_att']:+.3f} "
      f"({ew['pooled_post_se']:.3f})")
print(f"pooled pre     : VW {vw['pooled_pre_att']:+.3f} "
      f"({vw['pooled_pre_se']:.3f})")
```

```text
h    VW ATT   (se)    EW ATT   (se)    nobs   switchers
-4   +0.022  (0.111)  +0.039  (0.110)   4260   140
-3   -0.131  (0.098)  -0.127  (0.100)   4460   140
-2   +0.051  (0.097)  +0.049  (0.098)   4660   140
-1   (omitted baseline)
+0   +0.685  (0.106)  +0.678  (0.104)   4860   140
+1   +1.220  (0.099)  +1.202  (0.100)   4660   140
+2   +1.769  (0.109)  +1.749  (0.109)   4460   140
+3   +2.446  (0.107)  +2.409  (0.108)   4260   140
+4   +2.608  (0.094)  +2.588  (0.095)   4060   140
+5   +2.557  (0.108)  +2.528  (0.108)   3860   140
+6   +2.394  (0.113)  +2.370  (0.114)   3660   140
pooled post ATT: VW +1.944 (0.079)   EW +1.922 (0.079)
pooled pre     : VW -0.019 (0.084)
```

Reading it: pre-trends flat (all within ±1.5 se of zero), the effect ramps
0.63 → 2.5 over four horizons exactly as built (true EW ATT is
`2.5·min(h+1,4)/4`: 0.625, 1.25, 1.875, then 2.5), the equally-weighted
column sits slightly below the variance-weighted one because early adopters
have the larger effects *and* the more precisely estimated cohorts, and the
pooled post ATT matches its truth (2.5·22/28 = 1.96). The clean samples
shrink from 4,860 rows at h = 0 to 3,660 at h = 6 while all 140 switchers
remain usable — on real data watch both columns.

### Standard errors and the conventions transcribed

Cluster-by-entity, in the exact small-sample convention the authors' code
fixes (`setFixest_ssc(ssc(adj = TRUE, cluster.adj = TRUE))`, matching Stata
`reghdfe`): `(n−1)/(n−K) · G/(G−1)` times the cluster sandwich, where `K`
counts the slope **plus every absorbed period effect** (period effects are
not nested in entity clusters) and `G` is the number of entities in that
horizon's clean sample — deliberately different from the nested-cluster
`n/(n−k)` convention `panel_fe`/`panel_lp` inherit from linearmodels, and
verified against fixest at machine precision during fixture generation.
Covariates / regression adjustment, the composition-effects correction
(DGJT §2.10), pre-mean-differenced baselines (`pmd`), and the IV variant are
not yet implemented; doubly-robust (AIPW) LP-DiD has no reference
implementation anywhere and is out of scope until one exists to validate
against.

## `panel_pmg` convergence: the I(1) repair

The PSS estimator is a fixed-point iteration (back-substitute between the
pooled long-run GLS solve and the per-unit `φ_i`, `σ²_i`), and two defects
used to make it hard-fail exactly the panels PMG exists for — cointegrated
I(1) regressors. Measured on a stable error-correction DGP
(`dy = −0.3(y − x) + 0.2 dx + 0.1 eps`, N=10, T=150, 20 seeds):
**14/20 hard failures with I(1) x, 0/20 with I(0) x, 16/20 with the same
I(1) x scaled ×100** — the same dynamics, only the scale moved.

- **Divergence from the pinned start (the dominant failure).** The
  iteration ran from `θ = 0`; the map is only *locally* convergent, and on
  the failing panels θ walked monotonically **away** from the fixed point
  at ~0.7 per iteration (to −179 by iteration 260 on one traced seed) — no
  stopping tolerance can repair that (a relative rule at 1e-9 still failed
  14–16/20). Fix: when and only when the `θ = 0` pass fails, the identical
  iteration is rerun once from the Pesaran-Shin-Smith unrestricted-ARDL
  start (per-unit unrestricted EC regressions pooled by one GLS solve —
  the initialization PSS themselves recommend, and a consistent estimator,
  hence inside the basin of attraction). Converging panels never reach the
  restart, so previously-working fits are untouched to the bit.
- **A scale-dependent stopping rule.** The old rule was *absolute*
  (`|dθ|_inf < 1e-12`), unreachable whenever θ is large because solve
  updates carry rounding noise of `|θ|·O(eps)`. The rule is now *relative*
  (`|dθ|_inf ≤ tol·(1+|θ|_inf)`); the default `tol=3e-13` is the old
  rule's measured effective relative stringency on the O(1)-θ panels it
  was validated on (golden fixture: the old rule stopped at iteration 29
  with relative updates 5.890e-13 → 1.947e-13 across that iterate, so any
  default in [1.95e-13, 5.89e-13) keeps the golden **bit-identical**;
  3e-13 sits mid-window, and the battery converges at every tolerance down
  to 1e-14).

Measured post-fix: **0/20 failures on all three variants**, θ within 0.2 of
the true common long run on every panel. Genuine non-convergence (both
starts exhaust `max_iter`) still raises, with a message that names `tol`
and `max_iter` rather than blaming the data; the last iterate is not
returned because it is not a verified fixed point.

## Failure modes

- **Pooling heterogeneous slopes.** `panel_fe` on data with genuinely
  different unit responses returns a hard-to-interpret weighted average. If a
  Hausman-style comparison of pooled vs mean-group estimates diverges, trust
  the mean-group one.
- **Cluster SEs under cross-sectional dependence.** With a common shock,
  `se_type="cluster"` understates uncertainty. Use `driscoll_kraay`.
- **`panel_pmg` collinearity error.** If the partialled long-run regressors are
  collinear across the panel, θ is unidentified and the call raises. This is a
  correct refusal, not a bug — supply level regressors with real, non-redundant
  long-run variation.
- **Small Tᵢ with mean-group.** Per-unit regressions become unstable and the
  cross-unit average inherits the noise; prefer pooling (with heterogeneity
  tested) when time series are short.
- **`lp_did` refusals are the method speaking.** A treatment reversal under
  `absorbing=True` raises (choose `absorbing=False` with a stabilization
  lag); `never_treated_only=True` with no never-treated units raises; a
  horizon where no period cell mixes switchers with clean controls raises
  (there is no clean comparison — not a bug, a fact about the design); under
  `reweight=True` a switcher cell with no clean control raises. None of
  these are silently patched over.
- **`lp_did` pre-trends with thin cohorts.** The pre-window regressions use
  the same switcher count as h = 0 but a shrinking control window; a
  "significant" pre-trend at −Q built on a few dozen switchers is fragile —
  check `n_switchers` before reading the pre-trend test.
- **Reading `panel_mean_group(method="mg")` as a long run.** It is a static
  average slope; the ARDL long run comes from `panel_pmg`.

## Validated against

`panel_fe` matches `linearmodels` `PanelOLS` for the within estimator under
nonrobust, cluster-by-entity, and Driscoll-Kraay (Bartlett kernel) covariances —
on balanced panels (`fixtures/panel.json`) and, since 0.10.0, on unbalanced
ones through `mask=` (`fixtures/panel_unbalanced.json`: the Arellano-Bond
`EmplUK` panel and a seeded ragged panel, every effects menu, 1e-10; the
balanced paths pinned bitwise against the 0.9.0 build by
`fixtures/panel_balanced_snapshot.json`).
`panel_distributed_lag` (and the two-way / entity-trend menu it added to the
within estimator) is pinned at 1e-10 against `PanelOLS` on the explicitly
lagged design for nine cases × three covariances, its delta-method objects
against the documented transcription, and its interval coverage is measured
(the table in [its section](#distributed-lag-panel-regressions-panel_distributed_lag)).
`panel_lp` is a documented-formula golden built on the same within-plus-DK
machinery with a known simulated IRF. The `bias_correction="spj"` route is a
**transcription golden + Monte Carlo**, stated honestly: the method's
reference implementation is R-only (the authors' `pLP` package,
`github.com/zhentaoshi/panel-local-projection`, which commits datasets but no
numeric outputs), so `fixtures/panel_spj.json` pins tsecon at 1e-10 against
an independent NumPy reimplementation of the `panelLP.R` algebra (split
convention, combination, both adjusted-score sandwiches — see the generator's
docstring for provenance), and the statistical claims are separately measured
in the seeded Monte Carlo above — not matched against a stored run of the R
package. `lp_did` is a **reference-run golden**: `fixtures/lpdid.json` pins
six cases (absorbing VW / equally-weighted / never-treated-only, each with
pooled estimates; non-absorbing VW / reweighted / never-treated-only with a
stabilization lag) at 1e-10 against an actual R/fixest run of the authors'
own example implementations (github.com/danielegirardi/lpdid — the VW and EW
R scripts and the non-absorbing do-file, transcribed file:line in
`fixtures/generate_lpdid_fixtures.R`), cross-checked at generation time
against an independent NumPy reimplementation (max deviation 5.3e-15); the
one stated caveat is that the SSC-only Stata ado itself could not be fetched
in the build environment, so the pin is to the authors' published example
code, not the packaged command. The statistical claims (unbiasedness,
coverage, and the naive-contrast table above) are separately measured in the
seeded Monte Carlo. `panel_lp`'s closed-form simultaneous bands reuse the
SciPy-pinned critical values shared by every band surface, and their joint
coverage is property-MC measured (the seeded table in the band section
above); no third-party reference for simultaneous LP bands exists in
Python, so no golden is possible for that claim. `mean_group_var`, `panel_mean_group`
(MG and CCE-MG), and `panel_pmg` are documented-formula goldens reproducing the
Pesaran-Smith (1995), Pesaran (2006), and Pesaran-Shin-Smith (1999)
estimating equations, and are additionally property-validated: on data with a
known common long run, PMG recovers it and pools far more tightly than a free
mean-group of per-unit long runs. Fixtures:
[`fixtures/panel.json`](../../../fixtures/panel.json),
[`fixtures/panel_dl.json`](../../../fixtures/panel_dl.json),
[`fixtures/panel_unbalanced.json`](../../../fixtures/panel_unbalanced.json),
[`fixtures/panel_balanced_snapshot.json`](../../../fixtures/panel_balanced_snapshot.json),
[`fixtures/panel_spj.json`](../../../fixtures/panel_spj.json),
[`fixtures/lpdid.json`](../../../fixtures/lpdid.json),
[`fixtures/tsecon-panelts.json`](../../../fixtures/tsecon-panelts.json),
[`fixtures/pmg.json`](../../../fixtures/pmg.json).

## References

- Pesaran, M. H. & Smith, R. (1995). "Estimating long-run relationships from
  dynamic heterogeneous panels." *J. Econometrics* 68.
- Pesaran, M. H., Shin, Y. & Smith, R. (1999). "Pooled Mean Group Estimation of
  Dynamic Heterogeneous Panels." *JASA* 94.
- Pesaran, M. H. (2006). "Estimation and Inference in Large Heterogeneous
  Panels with a Multifactor Error Structure." *Econometrica* 74.
- Driscoll, J. & Kraay, A. (1998). "Consistent Covariance Matrix Estimation
  with Spatially Dependent Panel Data." *Rev. Econ. Stat.* 80.
- Jordà, Ò. (2005). "Estimation and Inference of Impulse Responses by Local
  Projections." *AER* 95.
- Nickell, S. (1981). "Biases in Dynamic Models with Fixed Effects."
  *Econometrica* 49.
- Dhaene, G. & Jochmans, K. (2015). "Split-Panel Jackknife Estimation of
  Fixed-Effect Models." *Rev. Econ. Studies* 82.
- Mei, Z., Sheng, L. & Shi, Z. (2026). "Nickell bias in panel local
  projection: Financial crises are worse than you think." *J. International
  Economics* (arXiv:2302.13455). Reference implementation: the `pLP` R
  package, github.com/zhentaoshi/panel-local-projection.
- Dube, A., Girardi, D., Jordà, Ò. & Taylor, A. M. (2025). "A Local
  Projections Approach to Difference-in-Differences." *J. Applied
  Econometrics* 40(7) (doi:10.1002/jae.70000; NBER WP 31184). Reference
  implementations: the authors' example code at
  github.com/danielegirardi/lpdid, the Stata `lpdid` package (SSC), and the
  R port at github.com/alexCardazzi/lpdid.
- Dell, M., Jones, B. F. & Olken, B. A. (2012). "Temperature Shocks and
  Economic Growth: Evidence from the Last Half Century." *AEJ: Macroeconomics*
  4(3).
- Burke, M., Hsiang, S. M. & Miguel, E. (2015). "Global non-linear effect of
  temperature on economic production." *Nature* 527.
- Hsiang, S. (2016). "Climate Econometrics." *Annual Review of Resource
  Economics* 8.
- Goodman-Bacon, A. (2021). "Difference-in-differences with variation in
  treatment timing." *J. Econometrics* 225.
- de Chaisemartin, C. & D'Haultfœuille, X. (2020). "Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects." *AER* 110.

See the guide: [Panel Time Series](../../guide/14-panel-time-series.md); for
the wider local-projection family (`lp`, `lp_iv`, state-dependent and smooth
LPs), see the [local projections model card](local-projections.md).

## Runnable example

```python
import numpy as np
import tsecon

rng = np.random.default_rng(88)
N, T = 20, 100

# ---- a balanced panel with entity fixed effects and a common observed shock ----
shock = rng.standard_normal(T)
alpha = rng.normal(0, 2.0, N)                 # entity fixed effects
psi = 0.8 * 0.6 ** np.arange(8)               # true dynamic response to the shock
y = np.empty((N, T))
for i in range(N):
    u = np.empty(T); u[0] = rng.standard_normal()
    for t in range(1, T):
        u[t] = 0.3 * u[t - 1] + rng.standard_normal()
    y[i] = alpha[i] + np.convolve(shock, psi)[:T] + u + 0.3 * rng.standard_normal(T)

# 1. Fixed-effects panel OLS. outcome is N x T; regressors is k x N x T.
s0 = np.tile(shock, (N, 1))
s1 = np.tile(np.r_[0.0, shock[:-1]], (N, 1))
regressors = np.stack([s0, s1])               # 2 x N x T
fe = tsecon.panel_fe(y, regressors, se_type="driscoll_kraay")
print("FE params:", np.round(fe["params"], 3), " (Driscoll-Kraay SEs)")

# 2. Panel local projection of the common shock (dynamic causal response).
plp = tsecon.panel_lp(y, shock, horizon=8, se_type="driscoll_kraay")
print("panel-LP IRF h=0..2:", np.round(plp["irf"][:3], 3))

# 3. Mean-group panel VAR (Pesaran-Smith): per-entity VARs, averaged.
entities = [np.column_stack([y[i], np.r_[0.0, shock[:-1]]]) for i in range(N)]
mg = tsecon.mean_group_var(entities, lags=2, horizon=8)
print("MG-VAR orthogonalized IRF path h=0..2:", np.round(mg["irf_path"][:3], 3))

# ---- a heterogeneous ARDL(1,1) panel with a COMMON long run (for MG / PMG) ----
theta0 = np.array([1.5, -0.8])
def sim_unit():
    lam = rng.uniform(0.2, 0.7); mu = rng.normal(0.5, 1.0)
    d0 = rng.normal([0.6, -0.3], [0.25, 0.25]); d1 = theta0 * (1 - lam) - d0
    burn, tt = 50, 90 + 50; K = 2
    x = np.empty((tt, K)); rho = rng.uniform(0.3, 0.6, K); xm = rng.normal(0, 1, K); x[0] = xm
    for t in range(1, tt):
        x[t] = xm * (1 - rho) + rho * x[t - 1] + rng.normal(0, 1, K)
    yy = np.empty(tt); yy[0] = mu / (1 - lam)
    for t in range(1, tt):
        yy[t] = mu + lam * yy[t - 1] + d0 @ x[t] + d1 @ x[t - 1] + rng.normal(0, 0.5)
    return yy[burn:], x[burn:]
ys = []; xs = []
for _ in range(25):
    yy, xx = sim_unit(); ys.append(yy); xs.append(xx)

# 4. Mean-group / CCE-MG estimator: the average of per-unit static slopes.
#    (A static contemporaneous regression, so this is NOT the ARDL long run.)
mgest = tsecon.panel_mean_group(ys, xs, method="mg")
print("MG average slope:", np.round(mgest["coef"], 3), " t:", np.round(mgest["tstat"], 2))

# 5. Pooled Mean Group: pool the long-run coefficient, keep short-run dynamics
#    free. This IS the estimator that targets the common long run of the DGP.
pmg = tsecon.panel_pmg(ys, xs)
print("PMG long-run theta:", np.round(pmg["theta"], 3),
      " (true", theta0, "),  adjustment speed phi_bar:", round(pmg["phi_bar"], 3))
```

Expected output:

```
FE params: [0.827 0.454]  (Driscoll-Kraay SEs)
panel-LP IRF h=0..2: [0.819 0.383 0.303]
MG-VAR orthogonalized IRF path h=0..2: [1.289 0.424 0.205]
MG average slope: [ 0.878 -0.418]  t: [ 25.53 -10.53]
PMG long-run theta: [ 1.5 -0.8]  (true [ 1.5 -0.8] ),  adjustment speed phi_bar: -0.583
```
