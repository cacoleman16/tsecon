# Lab findings memo — frontier forecasting study, 2026-08-17

**What this is.** The lab's first end-to-end, seeded comparison study of the
two frontier method families implemented this iteration —
`lab/prophet_lite` (Taylor-Letham decomposable forecaster) and
`lab/laplace` (AL score-driven quantiles, DCS robust filtering, LAD-ARMA) —
against the shipped tsecon baselines (`arima_fit` incl. SARIMA,
`theta_forecast`, `quantile_regression`, `garch_fit`,
`local_level_smooth`, seasonal-naive/mean). Five experiments, all seeded
and deterministic, all rerunnable:
`cd lab/experiments && <venv>/python run_all.py` (~10-13 min total; tables
regenerate into `experiments/results/` and are embedded verbatim below).

**Honesty ground rules used throughout.** Every model is refit at every
rolling origin with parameters frozen before the evaluation window; no
model sees test data at estimation time; DGP-truth comparisons use the
clean truth; the correctly-specified competitor is always included where
one exists (and it wins where it should); negative results are reported
with the same prominence as positive ones.

---

## Verdict summary (one line per method)

| Lab method | Verdict | One-line evidence |
|---|---|---|
| `robust_filter` DCS-t local level | **Clear win — strongest graduation case** | −22%/−31% level RMSE vs the Kalman pipeline at 5%/10% contamination, zero measurable tax on clean data, Gaussian-nesting golden holds to ~1e-3 |
| `robust_filter` DCS-Laplace | Works; dominated by DCS-t | robust, but 3-10% behind DCS-t at every contamination level here |
| `al_gas` AL-GAS dynamic quantile | **Qualified win** — beats static decisively and is calibrated; loses to any GARCH-implied quantile on a GARCH DGP | pinball 0.1125 vs static 0.1184 and Kupiec 0/5 vs 2/5 rejections; but GARCH-t 0.1096 and even misspecified GARCH-normal 0.1099 beat it |
| `prophet_lite` point forecasts | **Loses to shipped baselines** except long-horizon MAE on its home-turf DGP | CO2: DM p<0.001 vs SARIMA at h=1; GDP growth: indistinguishable from AR(1)/mean; home turf: best h=6/12 MAE but not DM-significant |
| `prophet_lite` intervals | Better calibrated than SARIMA's under estimation-window outliers — but both over-cover | pooled 80% coverage 0.897 vs SARIMA's 0.949, at half the width (4.3 vs 8.3) |
| `al_arima` LAD-ARMA | **Defer** — parameter gains don't convert to forecast gains | one-step OOS RMSE ratio LAD/Gauss 0.993 under t(2.5): a ~1% gain despite ~31% parameter-RMSE gains in the unit tests |

---

## Experiment 1 — point-forecast horse race

Rolling-origin, expanding window, every model fully refit at every
origin; RMSE/MAE at h ∈ {1, 6, 12}; Diebold-Mariano (tsecon `dm_test`,
squared loss, HLN correction) on the headline pairs.
Script: `experiments/exp01_point_horse_race.py`.

### (a) Synthetic piecewise-trend + seasonal + outliers (prophet_lite's home turf; T=240, 55 origins, first origin at the trend changepoint)

| model | RMSE h=1 | MAE h=1 | RMSE h=6 | MAE h=6 | RMSE h=12 | MAE h=12 |
|---|---|---|---|---|---|---|
| prophet_lite | 2.5647 | 1.8285 | 3.9017 | 2.4508 | 5.5393 | 3.3618 |
| sarima (0,1,1)(0,1,1)12 | 2.3668 | 1.7639 | 3.8471 | 2.9204 | 5.9995 | 4.7354 |
| theta | 2.0789 | 1.4604 | 3.6824 | 3.3680 | 6.3258 | 6.1080 |
| seasonal_naive | 4.9249 | 4.0950 | 3.9096 | 3.5979 | 4.0422 | 3.7813 |

| pair (squared loss) | h | DM (HLN) | p | mean d |
|---|---|---|---|---|
| prophet_lite vs sarima | 1 | 0.98 | 0.3293 | +0.976 |
| prophet_lite vs sarima | 12 | −0.96 | 0.3420 | −5.310 |
| prophet_lite vs theta | 1 | 1.64 | 0.1065 | +2.256 |
| prophet_lite vs theta | 12 | −0.35 | 0.7314 | −9.332 |

**Reading, honestly.** Even on the DGP built for it, prophet_lite does
not win h=1 (theta and SARIMA are better) — its edge appears exactly
where the piecewise-trend prior should help: h=6/12 **MAE**, where it
beats every competitor (3.36 vs 3.78-6.11 at h=12), and h=12 RMSE vs
SARIMA/theta. None of the DM tests reach significance at these origin
counts, so the honest claim is "directionally better at long horizons,
not statistically established". Two humbling details: (i) the first
origin sits at the trend changepoint, so all models forecast through a
regime change — seasonal-naive's implicit zero-trend turns out closer to
the post-break slope (−0.3) than freshly estimated trends, handing it
the best h=12 RMSE; (ii) SARIMA's double differencing spreads each
outlier into several large innovations, which shows up in its
long-horizon MAE (4.74 at h=12 vs prophet's 3.36).

### (b) CO2 monthly means, interpolated (T=526, 20 origins; integer index + (12,5) Fourier seasonality — calendar months are irregularly spaced in days, so prophet_lite's dated path doesn't apply)

| model | RMSE h=1 | MAE h=1 | RMSE h=6 | MAE h=6 | RMSE h=12 | MAE h=12 |
|---|---|---|---|---|---|---|
| prophet_lite | 0.9651 | 0.8430 | 1.0502 | 0.8955 | 1.2226 | 1.0323 |
| sarima (0,1,1)(0,1,1)12 | 0.3465 | 0.2914 | 0.5851 | 0.4434 | 0.7566 | 0.5695 |
| theta | 0.3468 | 0.2941 | 0.7886 | 0.6518 | 1.1534 | 0.9660 |
| seasonal_naive | 1.6911 | 1.5382 | 1.6745 | 1.5090 | 1.6999 | 1.5437 |

| pair (squared loss) | h | DM (HLN) | p | mean d |
|---|---|---|---|---|
| prophet_lite vs sarima | 1 | 3.91 | 0.0009 | +0.811 |
| prophet_lite vs sarima | 12 | 2.55 | 0.0197 | +0.922 |
| prophet_lite vs theta | 1 | 3.99 | 0.0008 | +0.811 |
| prophet_lite vs theta (NW fallback) | 12 | 0.48 | 0.6279 | +0.164 |

**Reading.** A decisive, statistically significant loss for prophet_lite
against the shipped SARIMA at every horizon (2.8x RMSE at h=1, DM
p=0.0009 even with only 20 origins). On a smooth, strongly seasonal,
persistence-dominated series, a global-trend + fixed-Fourier fit cannot
match a differencing model's local adaptation — the known structural
weakness of the Prophet family, reproduced here on real data. The "NW
fallback" row exists because `tsecon.dm_test` *refused* the h=12
rectangular-window variance as non-PSD at n=20 (see failure modes).
With 20 origins the insignificant rows should be read as signs only.

### (c) Real GDP growth, quarterly 400·Δlog (T=202, 71 origins) — no seasonality, where a trend/seasonality model should lose to ARIMA

| model | RMSE h=1 | MAE h=1 | RMSE h=6 | MAE h=6 | RMSE h=12 | MAE h=12 |
|---|---|---|---|---|---|---|
| prophet_lite (trend only) | 2.1385 | 1.6190 | 2.2165 | 1.7061 | 2.5853 | 1.8636 |
| ar1 (arima_fit 1,0,0 + const) | 2.0738 | 1.5766 | 2.2099 | 1.6526 | 2.6066 | 1.8055 |
| theta | 2.1045 | 1.6459 | 2.4771 | 1.9184 | 2.9529 | 2.2275 |
| mean | 2.1585 | 1.6168 | 2.2073 | 1.6500 | 2.6029 | 1.8022 |

| pair (squared loss) | h | DM (HLN) | p | mean d |
|---|---|---|---|---|
| prophet_lite vs ar1 | 1 | 0.79 | 0.4295 | +0.273 |
| prophet_lite vs ar1 | 12 | −0.09 | 0.9273 | −0.111 |
| prophet_lite vs mean | 1 | −0.26 | 0.7966 | −0.086 |
| prophet_lite vs mean | 12 | −0.08 | 0.9390 | −0.091 |

**Reading.** The expected loss shows up at h=1 (worst RMSE of the four,
positive loss differential vs AR(1)) but is *not* significant, and there
is no blow-up: the L1 prior shrinks nearly all changepoints on a
stationary series, so prophet_lite degrades gracefully toward the mean
forecaster instead of hallucinating trends. It has uniformly worse MAE
than AR(1)/mean at every horizon. Practical summary: on non-seasonal,
non-trending data prophet_lite buys nothing over a one-line AR(1), at
~10x the fit cost.

---

## Experiment 2 — interval calibration (300 seeded replications)

Home-turf DGP, training window T=120 with 3% outliers at 6-10σ, clean
12-step future — the design isolates how estimation-window outliers
distort each method's intervals. prophet_lite simulation intervals (500
draws, per-rep seed) vs SARIMA (0,1,1)(0,1,1)_12 parametric Gaussian
intervals (tsecon's documented statsmodels-matching default:
innovation+filtering uncertainty, parameters treated as known;
`constant=False`, so the documented drift-uncertainty under-coverage
path is not in play). Binomial MC standard errors in parentheses
(R=300). Script: `experiments/exp02_interval_calibration.py`.

| model | nominal | cov h=1 (se) | cov h=6 (se) | cov h=12 (se) | pooled h=1..12 | mean width |
|---|---|---|---|---|---|---|
| prophet_lite | 80% | 0.887 (0.018) | 0.893 (0.018) | 0.890 (0.018) | 0.897 | 4.26 |
| prophet_lite | 95% | 0.967 (0.010) | 0.980 (0.008) | 0.987 (0.007) | 0.981 | 6.49 |
| sarima | 80% | 0.900 (0.017) | 0.933 (0.014) | 0.957 (0.012) | 0.949 | 8.27 |
| sarima | 95% | 0.987 (0.007) | 0.993 (0.005) | 0.993 (0.005) | 0.994 | 12.65 |

prophet_lite fits converged: 300/300.

**Reading.** *Both* methods over-cover under contamination — neither is
"calibrated" here — but they fail by very different amounts. The
mechanism: 6-10σ outliers in the training window inflate the estimated
noise variance while the evaluated future is clean, so all intervals are
too wide. SARIMA suffers roughly twice as badly (pooled 80% coverage
0.949 at width 8.27, vs prophet_lite's 0.897 at width 4.26): its double
differencing turns one additive outlier into several large innovations,
inflating σ̂² much more than a single residual in prophet_lite's
regression fit. prophet_lite's own clean-window audit (module README:
0.797 @ 80%, 0.950 @ 95%) confirms the interval *scheme* is calibrated —
the contamination is the driver, and the miscalibration direction
(conservative) is at least the safe one. SARIMA's 80% band is
effectively a 95% band by h=12. Neither method models outliers; a robust
scale estimate would fix most of this for both.

---

## Experiment 3 — robust trend filtering under additive outliers

Gaussian local level (σ_eta=0.1, σ_eps=1.0, T=500), 0/5/10% additive
outliers at 8σ, 30 reps per level. All DCS variants are MLE fits; the
tsecon Kalman rows run `local_level_smooth` at Gaussian UC-MLE variances
(statsmodels `UnobservedComponents` MLE — the realistic contaminated
pipeline). Timing is fair: all "filter" rows are one-step-*predicted*
levels; the smoother row uses the full sample and is a look-ahead
reference, not a competitor. RMSE vs the clean truth. Script:
`experiments/exp03_robust_filtering.py`.

| method (one-step-predicted level unless noted) | RMSE 0% (sd) | RMSE 5% (sd) | RMSE 10% (sd) |
|---|---|---|---|
| DCS-t (robust) | 0.321 (0.037) | 0.342 (0.044) | 0.354 (0.052) |
| DCS-Laplace (robust) | 0.354 (0.043) | 0.360 (0.043) | 0.375 (0.049) |
| DCS-Gaussian (nested control) | 0.321 (0.036) | 0.441 (0.052) | 0.514 (0.078) |
| tsecon Kalman predicted @ UC-MLE | 0.321 (0.037) | 0.448 (0.051) | 0.516 (0.076) |
| tsecon Kalman SMOOTHED @ UC-MLE (look-ahead ref) | 0.225 (0.025) | 0.345 (0.056) | 0.365 (0.061) |

Fitted gain κ (the Gaussian gain-collapse failure mode):

| method | mean κ 0% | mean κ 5% | mean κ 10% |
|---|---|---|---|
| DCS-Gaussian (nested control) | 0.0886 | 0.0406 | 0.0295 |
| DCS-t (robust) | 0.0908 | 0.1288 | 0.1322 |
| DCS-Laplace (robust) | 0.1161 | 0.0816 | 0.0617 |

Nesting check on clean data (first 5 reps): DCS-Gaussian vs steady-state
Kalman at UC-MLE variances —

| rep | DCS-Gaussian κ | steady-state Kalman gain | abs diff | path RMSE vs Kalman predicted |
|---|---|---|---|---|
| 0 | 0.1105 | 0.1111 | 5.8e-04 | 0.0019 |
| 1 | 0.0927 | 0.0948 | 2.1e-03 | 0.0059 |
| 2 | 0.0959 | 0.0971 | 1.3e-03 | 0.0037 |
| 3 | 0.0547 | 0.0489 | 5.8e-03 | 0.0249 |
| 4 | 0.0537 | 0.0561 | 2.4e-03 | 0.0082 |

**Reading.** The cleanest result in the study.
(i) *No robustness tax*: on clean data DCS-t matches the Gaussian/Kalman
filter to three decimals (0.321 both) — the redescending score costs
nothing when there are no outliers.
(ii) *Large, monotone gains under contamination*: −22% RMSE at 5%, −31%
at 10% vs the Kalman pipeline; at 10% contamination the DCS-t *filter*
(no look-ahead) matches the contaminated Kalman *smoother* (0.354 vs
0.365) — a robust one-step filter is worth as much as full-sample
smoothing done non-robustly.
(iii) *The mechanism is visible*: the Gaussian MLE absorbs outliers by
collapsing its gain (0.089 → 0.030), going nearly blind to genuine level
shifts; DCS-t instead *raises* κ because its bounded score already
discounts the outliers.
(iv) *The nesting golden holds on every seed*: DCS-Gaussian reproduces
the steady-state Kalman gain to 5.8e-4–5.8e-3 (the worst seed is a
near-flat-likelihood small-gain case) and the predicted path to
0.002-0.025 RMSE — exactly the runnable validation target a promotion
would be gated on.

---

## Experiment 4 — dynamic 5% tail forecasting, GARCH(1,1)-t DGP

T=3000 (train 2000 / test 1000), parameters frozen at the training fit,
5 seeds. The DGP is exactly GARCH(1,1)-t, so the GARCH-t competitor is
*correctly specified and should win* — the question is how close the
semiparametric AL-GAS recursion gets without any volatility model.
Kupiec column = number of seeds where the 5% unconditional-coverage LR
test rejects. Script: `experiments/exp04_tail_quantiles.py`.

| model | mean pinball (τ=.05) | mean hit rate | Kupiec rej. @5% | RMSE vs true quantile |
|---|---|---|---|---|
| AL-GAS dynamic quantile (lab) | 0.1125 | 0.052 | 0/5 | 0.350 |
| GARCH(1,1)-t implied (tsecon, correctly specified) | 0.1096 | 0.050 | 0/5 | 0.055 |
| GARCH(1,1)-normal implied (tsecon) | 0.1099 | 0.045 | 0/5 | 0.101 |
| static quantile_regression (tsecon) | 0.1184 | 0.058 | 2/5 | 0.483 |

| pinball loss differential vs AL-GAS (NW t, mean over seeds) | mean t | signif @5% |
|---|---|---|
| GARCH-t implied − AL-GAS | −0.64 | 0/5 |
| GARCH-normal implied − AL-GAS | −0.37 | 0/5 |
| static quantile_regression − AL-GAS | +1.72 | 2/5 |

**Reading, stated honestly in both directions.**
*Where AL-GAS wins*: against the static quantile it is better on every
metric — 5% lower pinball, 28% lower quantile-path RMSE (0.350 vs
0.483), and calibrated where the static quantile fails Kupiec on 2/5
seeds. It achieves this with no volatility model, no tail-distribution
assumption, and 3 parameters.
*Where it loses*: on a GARCH DGP, any GARCH-implied quantile — including
the *misspecified normal-innovation* one — beats it on average pinball,
and the correctly-specified GARCH-t tracks the true quantile path 6x
better (RMSE 0.055 vs 0.350). The indicator score updates only on
hit/no-hit, so AL-GAS adapts far more slowly than a model that sees the
full squared-return signal. None of the AL-GAS-vs-GARCH differences are
NW-significant at these sample sizes, but the sign is consistent across
all 5 seeds.
*Fair conclusion*: AL-GAS is a robust default when you distrust the
volatility model; it is not a replacement for one when GARCH is roughly
right. The promotion case must rest on vol-model-misspecification DGPs,
which this study did not run — flagged as next-iteration work.

---

## Experiment 5 (supplementary) — LAD/median ARMA one-step forecasts

ARMA(1,1), φ=0.6, θ=0.3, train 300 / test 50 one-step forecasts with
frozen parameters, 20 reps per innovation type; LAD vs the
identical-pipeline Gaussian-CSS twin (spot-verified against
`tsecon.arima_fit` exact MLE on a seeded series: twin φ=0.6223,
θ=0.4264 vs arima_fit φ=0.6223, θ=0.4251). Script:
`experiments/exp05_lad_arima.py`.

| DGP | LAD RMSE | Gauss RMSE | ratio | LAD MAE | Gauss MAE | ratio |
|---|---|---|---|---|---|---|
| t(2.5) innovations | 1.8421 | 1.8542 | 0.993 | 1.1889 | 1.2023 | 0.989 |
| Laplace innovations | 1.4672 | 1.4687 | 0.999 | 1.0414 | 1.0441 | 0.997 |
| Gaussian innovations | 1.0158 | 1.0132 | 1.003 | 0.8094 | 0.8063 | 1.004 |

**Reading.** The direction is exactly as theory predicts (LAD wins under
heavy tails, loses under Gaussian) — but the magnitudes are ~1%. The
module's unit tests show ~31% *parameter*-RMSE gains under t(2.5); this
experiment shows those do not convert into point-forecast gains at
T=300, because one-step forecast error is dominated by the innovation
itself, not by parameter noise. An honest negative result: LAD-ARMA is
about robust *estimation and inference*, not better point forecasts.

---

## Failure modes observed during the study

1. **`tsecon.dm_test` refusal at long h with few origins** (exp01b): with
   20 origins at h=12, the rectangular-window (uniform-weight) long-run
   variance went negative and `dm_test` raised instead of returning a
   junk statistic. That is the library behaving as designed — but the
   study needed a Bartlett fallback (hand-rolled Newey-West t on the
   loss differential in `experiments/common.py`). *Actionable*: a
   Bartlett/NW variance option on `dm_test` would remove the sharp edge.
2. **prophet_lite fit cost** grows to ~4-6 s per fit at T≈500 (pure-
   Python coordinate descent inside the σ↔lasso alternation); fine at
   T≤250 (~0.3-0.6 s). This capped exp01b at 20 origins. A Rust port
   would erase it, but it is a real cost of the lab prototype.
3. **prophet_lite's default prior is weak on smooth series** (known,
   documented in its README): on CO2 windows 21-23 of 25 candidate
   changepoints go active with tiny deltas (λ = σ²_scaled/τ is small
   when σ ≪ y_scale). Not a solver failure (KKT gap ~1e-10; upstream
   Prophet behaves the same) — but a user reading `n_active` as
   "detected breaks" would be misled.
4. **Interval miscalibration under contamination is generic, and worse
   for SARIMA** (exp02): estimation-window outliers made *both* methods
   over-cover; SARIMA's differencing multiplies one outlier into several
   large innovations, so its 80% band is a de-facto 95% band by h=12 at
   twice prophet_lite's width. Conservative, so safe-direction — but
   miscalibrated is miscalibrated.
5. **Gaussian gain collapse under contamination** (exp03): the Gaussian
   local-level MLE — the UC-MLE variances feeding tsecon's
   `local_level_smooth` included — absorbs additive outliers by
   shrinking the Kalman gain ~3x, then misses genuine level movement.
   This is the concrete failure the robust filters fix.
6. **DCS-t ν̂ is not a tail estimate under contamination** (module tests:
   ν̂ → ~1.6 at 7% outliers): the fat tail is doing outlier duty. Fine
   for filtering; wrong to report as the clean-noise tail index.
7. **Optimizer fragility only off the defaults**: the hard-indicator
   AL-GAS and hard-sign Laplace variants (`bandwidth=0`, `smooth=0`) can
   stall L-BFGS-B on kinks. With the default smoothing and the built-in
   deterministic multi-starts, no convergence failures surfaced in this
   study (prophet_lite reported `converged` on 300/300 exp02 fits; the
   laplace fits were spot-checked, and their own seeded suites assert
   convergence).
8. **Scope limits of the study itself**: 20 origins on CO2 (see #2)
   makes those DM p-values fragile; exp04 tests only a correctly-
   specified-GARCH world; exp03's outliers are additive, symmetric and
   at a single size; exp01a's origins start exactly at the trend break
   (deliberately hard). None of the conclusions should be quoted beyond
   these designs.

---

## Graduation candidates (proposals against ROADMAP §0's validation-first bar)

The bar (ROADMAP, "Scoped next work"): build-next requires a golden that
*runs in the test environment* (statsmodels/arch/SciPy/scikit-learn or a
documented closed form); R-only references grade lower and are said so;
MC-recovery grading is the declared fallback where no reference runs.

### 1. DCS robust local level — PROPOSE for next iteration (strongest case)

*(Shipped in 0.3.0 as `dcs_local_level` — repository audit, September 2026.)*

- **Evidence**: exp03 — no clean-data tax, −22/−31% RMSE under 5/10%
  contamination, mechanism understood (gain collapse vs bounded score),
  nested-Gaussian equivalence held on every seed.
- **Crate / API**: extend the score-driven family that already ships
  (`gas_volatility`, Creal-Koopman-Lucas 2013) — same crate or a sibling
  `models-dcs`: `dcs_level(y, density="t"|"gaussian"|"laplace")` → dict
  with `kappa`, `scale`, `nu`, `level` (one-step-predicted path),
  `loglik`, `aic`/`bic`, `converged`, standard results convention.
  Reference: Harvey-Luati (2014, JASA); Harvey (2013).
- **Validation target, graded honestly**: the *Gaussian limit* has a
  true runnable golden — equality with the steady-state Kalman filter
  (statsmodels `UnobservedComponents` MLE gain, and tsecon's own exact
  `local_level_smooth` path) at tight tolerance, plus the ν→∞ collapse
  of DCS-t onto it. The t/Laplace filters themselves have **no
  independent reference implementation in the test environment** (DCS
  reference code is R/Matlab), so they are **MC-recovery graded**
  (seeded recovery of κ/scale/ν; contamination regression test: RMSE
  ratio vs Gaussian ≤ 0.8 at 10% outliers) — the same honest grade the
  roadmap assigns auto.arima. Difficulty M.

### 2. VaR backtest battery (Kupiec / Christoffersen / DQ) — PROPOSE (a need this study surfaced)

*(Shipped in 0.3.0 as `var_backtest` — repository audit, September 2026.)*

- Not lab code, but the study had to *hand-roll Kupiec* in
  `experiments/common.py` to evaluate exp04, and Module 03's roadmap
  already lists the battery (difficulty Low) with closed-form LR
  statistics — a fully runnable golden by the roadmap's own definition
  (plus the Kuester-Mittnik-Paolella 2006 study as a published fixture).
  `var_backtest(y, var_path, tau)` → Kupiec POF, Christoffersen
  independence/conditional coverage, Engle-Manganelli DQ. Cheapest
  high-leverage item here, and the natural acceptance harness for any
  later dynamic-quantile promotion.

### 3. AL-GAS dynamic quantile — HOLD in lab one more iteration

- **For**: calibrated out of the box (Kupiec 0/5), decisively better
  than the shipped static `quantile_regression` tail (exp04), 3
  parameters, no distributional assumption.
- **Against**: loses (in sign, consistently across seeds) to *any*
  GARCH-implied quantile on a GARCH world, including the misspecified-
  normal one. The niche claim — "wins when the volatility model is
  wrong" — was not tested and must be demonstrated before promotion
  (vol-break / non-GARCH DGPs, next iteration).
- If promoted later: into the quantile family beside
  `quantile_regression`/`growth_at_risk` as `dynamic_quantile` (adaptive
  CAViaR). **No runnable reference exists** (Engle-Manganelli code is
  Matlab; nothing in statsmodels/arch) → MC-recovery graded, gated on
  the #2 battery as its acceptance harness.

### 4. prophet_lite — DO NOT promote as a forecaster; salvage two parts

- The forecaster fails the wedge test: decisively worse than shipped
  SARIMA on stable-seasonal real data (exp01b, DM p<0.001),
  indistinguishable from AR(1)/mean where there is no trend/seasonal
  structure (exp01c), home-turf advantage (long-horizon MAE, exp01a)
  never significant; its one genuine head-to-head win — narrower,
  better-calibrated intervals under contamination (exp02) — is an
  argument about SARIMA's outlier sensitivity, not for shipping a new
  forecaster. There is also no runnable golden: the reference
  implementation (cmdstan-backed `prophet`) is not installable in the
  test environment, so even a faithful port could only be self-graded.
- **Salvage (a): Fourier deterministic-terms builder.** The covariates
  contract (ROADMAP §3) already plans "trends, seasonal dummies, Fourier
  terms, holidays" builders; prophet_lite's Fourier block is exactly
  that, with a trivial closed-form golden. Difficulty S.
- **Salvage (b): exact L1 trend filter.** *(Shipped in 0.8.0 as
  `l1_trend_filter`, Kim-Koh-Boyd on the banded dual with a duality-gap
  certificate — repository audit, September 2026.)* The FWL + coordinate-descent
  lasso trend solver is an exact 1-D trend-filtering engine (Kim-Koh-
  Boyd 2009 family, adjacent to the shipped `hp_filter` and
  `bai_perron`), and it validates *exactly* against scikit-learn `Lasso`
  on the reduced problem — a true runnable golden (KKT gap ~1e-10
  already demonstrated). Promote as `l1_trend(y, n_knots, tau)` if
  Module 01 wants a changepoint-flavored filter; otherwise drop.
  Difficulty S-M.

### 5. LAD-ARMA (`al_arima`) — DEFER

- Real parameter-efficiency gains (module tests), ~1% forecast gains
  (exp05), no runnable golden (no LAD-ARMA in statsmodels/arch;
  Davis-Dunsmuir/Ling references are theory papers with R/Matlab code).
  If robust ARIMA estimation is ever user-demanded it should arrive as
  an `innovations="laplace"` option on ARIMA, MC-recovery graded — not
  as a separate estimator now.

---

## Reproduction

```bash
VENV=/home/user/tsecon/.venv/bin/python
cd /home/user/tsecon/lab/experiments && $VENV run_all.py     # ~10-13 min
# individually: $VENV exp01_point_horse_race.py ... exp05_lad_arima.py
# unit tests:
$VENV -m pytest /home/user/tsecon/lab/prophet_lite/tests.py -q      # 8 passed
cd /home/user/tsecon/lab/laplace && $VENV -m pytest tests.py -q     # 7 passed
```

All seeds are hard-coded in the scripts; `results/expNN.md` /
`results/expNN.json` regenerate byte-comparable tables (up to the
runtime footers).

## 2026-08-25 — exp06: conformal interval wrappers, and what graduated

Head-to-head of the three conformal wrappers now shipped publicly as
`conformal_forecast`/`conformal_backtest` (split / EnbPI / ACI, all over the
same AR base). Two measured settings (`experiments/results/exp06.md`):
on GARCH(1,1)-t returns all three hold marginal 90% coverage (0.894–0.905)
but only ACI passes the Kupiec independence screen at a 7% rejection rate
(split 26%, EnbPI 33% — clustered violations, as expected for methods that
do not condition on volatility); under a mid-window variance shift the
fixed-level methods collapse (split 0.71, EnbPI 0.51 post-shift) while ACI
at γ=0.05 holds 0.89. Frontier scan for the next lab cycle, with honest
feasibility under the library's validation bar: **SPCI** (sequential
predictive conformal — random-forest QRF on residuals; heavy dependency,
property-MC only), **conformal PID** (Angelopoulos et al. 2023 — the control-
theoretic extension of ACI; small surface, natural next step), **quantile
conformal on GARCH-standardized residuals** (would fix the conditional-
coverage weakness measured above; buildable on the existing garch engine),
and **BSTS** (big surface; the SSM engine makes it feasible but it is a
release-scale project, not a lab sketch).
