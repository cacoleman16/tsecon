# Model card — Exponential smoothing (ETS)

**Family:** `ets_fit`, `auto_ets`

The innovations state-space exponential-smoothing family of Hyndman,
Koehler, Snyder & Grose (2002) and Hyndman, Koehler, Ord & Snyder (2008):
thirty models ETS(Error, Trend, Seasonal) with additive (`A`) or
multiplicative (`M`) error, none / additive / multiplicative trend
(optionally damped, `Ad` / `Md`), and none / additive / multiplicative
seasonality — simple exponential smoothing, Holt, the damped trend and the
two Holt-Winters methods among them — fitted by maximum likelihood, with
prediction intervals and the information-criterion search that R's
`forecast::ets` runs.

| Function | Role |
|----------|------|
| `ets_fit` | One member of the taxonomy: MLE (or evaluation at fixed parameters), states, forecasts with exact or simulated prediction intervals |
| `auto_ets` | The candidate-set search: every admissible member fitted, ranked by AICc/AIC/BIC, the winner returned with the table |

## What it estimates

Every member is an innovations state-space model: **one** error $e_t$
drives the observation and every state. For ETS(A,Ad,A),

$$
\begin{aligned}
y_t &= \ell_{t-1} + \phi b_{t-1} + s_{t-m} + e_t \\
\ell_t &= \ell_{t-1} + \phi b_{t-1} + \alpha e_t \\
b_t &= \phi b_{t-1} + \beta e_t \\
s_t &= s_{t-m} + \gamma e_t ,
\end{aligned}
$$

and the other members replace `+` by `*` component by component (Hyndman
et al. 2008, Tables 2.2 and 2.3; under a multiplicative error $e_t$ is the
*relative* one-step error). The crate implements the recursion exactly as
R's `etscalc.c` does it — the state update is the same for both error
types; only the residual and the likelihood differ. The parameters are
Hyndman's $\alpha, \beta, \gamma, \phi$, **not** the classical
$\beta^* = \beta/\alpha$ and $\gamma^* = \gamma/(1-\alpha)$ of the
Holt-Winters recursions (statsmodels reports the same convention as this
card; R's `ets` too).

The log-likelihood is the concentrated Gaussian one of Ord, Koehler &
Snyder (1997), $-\tfrac{n}{2}\bigl(\ln(2\pi\hat\sigma^2) + 1\bigr) -
\sum_t \ln|\hat y_t|\,[\text{multiplicative error}]$ with $\hat\sigma^2$
the mean squared residual — statsmodels' `ETSModel.loglike`; R's `ets`
reports the same quantity without the constant $-\tfrac{n}{2}(\ln(2\pi/n)
+ 1)$, so R's `loglik` differs by a constant at fixed $n$ and its
criteria rank models identically.

**Initial states.** `initialization="estimated"` (the default, R's and
statsmodels') treats the level, trend and seasonal states as free
parameters started from the Hyndman (2008, §2.6.1) heuristic; the $m$
seasonal indices are identified only up to a shift (additive) or a scale
(multiplicative) absorbed by the level, so the crate normalises them to
sum to zero / average one — R's convention — and counts $m-1$ free
seasonal states in the information criteria (statsmodels pins the index
of the first observation at 0/1 instead and counts all $m$; the fixture
records both counts, and the two identifications are converted exactly in
the tests). `"heuristic"` holds the heuristic values fixed (a centred
moving average of order $m$ over the first cycles, detrended and averaged
by season; the level and trend from a regression of its first ten values
on a linear time index); `"known"` takes `initial_states` from the caller.

## Assumptions

- Gaussian, homoskedastic innovations (relative ones under a
  multiplicative error) — the likelihood and the class-1 intervals are
  built on it; the simulated intervals draw Gaussian innovations too
  (residual bootstrap is not offered).
- Any multiplicative component needs **strictly positive** data (`y > 0`,
  refused otherwise naming the offending observation).
- No missing values: the innovations form conditions every state update
  on the observed error and has no NaN mechanism (R's `ets` refuses NaN
  as well) — interpolate first, or use the Kalman-filter models
  (`local_level_smooth`), which handle gaps exactly. The fixture's `co2`
  series is the bundled Mauna Loa data aggregated to monthly means with
  its five missing months linearly interpolated, and says so.
- The smoothing parameters live in the traditional box $0 < \alpha < 1$,
  $0 < \beta < \alpha$, $0 < \gamma < 1 - \alpha$, $0.8 \le \phi \le 0.98$
  (R's and statsmodels' default bounds, statsmodels' numerical margins
  $10^{-4}$). R's larger *admissible* (forecastable) region is not
  offered; a fitted parameter sitting on the box edge is reported as is,
  and `converged=False` there means the search ended on the edge.

## When to use

- Level, trend and seasonal forecasting of a single series with a
  likelihood, proper prediction intervals and automatic selection —
  the default first model for most business and macro series, and the
  M3-competition-grade benchmark any elaborate model must beat (see
  `theta_forecast`, which is ETS(A,N,N) with drift).
- ETS(M,·,M) — the multiplicative Holt-Winters — when the seasonal swing
  and the noise grow with the level (retail, passengers, energy).
- `auto_ets` when you have many series and want a disciplined, auditable
  choice among the taxonomy rather than a hand-tuned one.

**When not.** Data with zeros or negatives under any multiplicative
component (use the additive models, or transform); series with gaps
(Kalman-filter models); long-memory or explicit unit-root dynamics you
want to test rather than smooth (`auto_arima`, `check_stationarity`);
weekly or daily seasonality with a long period on a short sample — the
$m$ seasonal states have to be initialised from at least two cycles and
$10 + 2\lfloor m/2 \rfloor$ observations.

## Key arguments and defaults

| Argument | Default | Why |
|---|---|---|
| `error`, `trend`, `damped`, `seasonal`, `seasonal_periods` | `"add"`, `None`, `False`, `None`, `None` | The taxonomy letters. `damped` without a trend and `seasonal_periods` without a seasonal are **refused** (they would be inert); a seasonal without `seasonal_periods` is refused too |
| `initialization` | `"estimated"` | R's and statsmodels' default; `"heuristic"` is faster and adds only the smoothing parameters to $k$; `"known"` needs `initial_states` |
| `smoothing_params` | `None` (estimate) | Evaluate at fixed parameters instead — statsmodels' `smooth(params)`; needs `"heuristic"` or `"known"` (with `"estimated"` nothing would be estimated), and `optimizer`/`max_iter` are then refused |
| `optimizer` | `None` = `"auto"` | L-BFGS and Nelder-Mead from a staged start (the smoothing parameters first, at the heuristic states), then a BFGS polish of the better — two searches because the surface can hold a boundary optimum at $\alpha \to 0$ or a ridge at $\alpha \to 1$ that traps one of them. `"nelder_mead"` (R's choice), `"bfgs"`, `"lbfgs"` run one search; the quasi-Newton ones can stall where a parameter runs into a bound (the logistic transform flattens the working-space gradient there) — diagnostic options, not the default |
| `horizon`, `level` | `0`, `None` = 0.95 | Forecast steps and interval coverage; `level` (and `n_sim`, `seed`) with `horizon=0` are refused |
| `n_sim`, `seed` | `None` = 5000, `None` = 0 | Simulated intervals for the non-class-1 models; refused for a class-1 model, where the interval is exact and nothing is simulated. Allocation guards (not modelling limits): `horizon` at most $10^6$, `n_sim` $\times$ `horizon` at most $2^{28}$ simulated values held at once, both refused by name, and the buffer is reserved fallibly so a machine that cannot supply a permitted one also gets an error rather than an abort |
| `auto_ets(seasonal_periods, ic, allow_multiplicative_trend, restrict, damped)` | `None`, `"aicc"`, `False`, `True`, `None` | R's defaults: no multiplicative trend, the infinite-variance / mis-scaled combinations dropped, damped and undamped both tried |

## How to read the output

`ets_fit` returns the specification (`spec`, `short_name`, the letters), the
smoothing parameters (`alpha`, `beta`, `gamma`, `phi`, `None` when the
component is absent; `params`/`param_names` packed), the initial states
(`initial_level`, `initial_trend`, `initial_seasonal` with
`initial_seasonal[j]` the index in force for observation `j`;
`initial_states`/`initial_state_names` packed), the one-step `fitted`
values and `resid` (relative under a multiplicative error), the state
paths `level_path` / `trend_path` / `seasonal_path` (the states *after*
each update; statsmodels' `level` / `slope` / `season`), the forecast
anchor `final_level` / `final_trend` / `final_seasonal` (`[j]` for
forecast step `j`), `loglik`, `sigma2`, `nobs`, `k_params`, `aic`,
`aicc`, `bic`, the search record (`converged`, `n_iterations`,
`n_fevals`, `optimizer`), `class1`, and with `horizon=h` the `forecast`,
`forecast_variance`, `forecast_lower`, `forecast_upper`, `interval_level`,
`interval_method` (`"exact"` or `"simulated"`), `n_sim`, `seed`.

The **point forecast is the zero-innovation path** from the final state
(R's and statsmodels' convention) — for the multiplicative models it is
therefore the median-like path of the recursion, not the simulation mean.
For the six **class-1** models (additive error, additive or no trend and
seasonal) the interval is Gaussian with the closed-form variance $v_h =
\sigma^2\bigl[1 + \sum_{j=1}^{h-1} c_j^2\bigr]$, $c_j = w'F^{j-1}g$
(Hyndman et al. 2008, chapter 6, Table 6.1); it ignores parameter
uncertainty, so it runs a little narrow in short samples (measured
below). For every other model the bounds are the empirical quantiles of
`n_sim` simulated paths and `forecast_variance` is their sample variance.

`auto_ets` returns the winner's `ets_fit` dict — the very fit the search
scored, so refitting the reported specification reproduces every number
bit for bit — plus `ic`, `ic_value`, and `candidates`: every candidate
with its `spec`, `loglik`, `aic`/`aicc`/`bic`, `ic_value`, `k_params`,
`converged`, `status` and `error`, ranked by the criterion with failures
last (a failing candidate is recorded, never fatal). Candidates within
about 2 of the best criterion are near-ties the data do not distinguish;
the winner's standard errors (not offered here) would not know a search
happened.

## Failure modes

- **Boundary optima.** $\beta \to 0$, $\gamma \to 0$ or $\alpha \to 1$
  are common and legitimate (a deterministic trend, a fixed seasonal, a
  random walk); the search reports them at the box edge with
  `converged=False` when it ended on the edge. Where statsmodels'
  L-BFGS-B stalls on an edge with a lower likelihood — log-UKgas
  ETS(A,A,A) with estimated initial states, where it returns $\alpha =
  10^{-4}$ and a log-likelihood 0.89 below the interior optimum — the
  staged two-search default finds the interior one; a single quasi-Newton
  search from the same start can stall the same way, which is why one
  alone is not the default.
- **Multiplicative components on data near zero** make the recursion
  divide by tiny states; the crate refuses non-positive data up front and
  reports a degenerate recursion (naming the observation) if it happens
  inside the search.
- **Undamped multiplicative trends** extrapolate exponentially; R's
  default `allow.multiplicative.trend = FALSE` is the default here too.
- **Short seasonal samples.** The heuristic needs two full cycles and
  $10 + 2\lfloor m/2 \rfloor$ observations; below that the estimated
  initialisation starts from the "simple" rule (first cycle) and the
  criteria have few degrees of freedom — treat selections on
  $n < 3m$ as guesses.
- **Multiplicative-seasonal parity with statsmodels is not expected**:
  see the next section.

## Validated against

The fixture is `fixtures/ets.json`, from
`fixtures/generate_ets_fixtures.py` (statsmodels 0.15; the generator never
imports tsecon). Graded leg by leg:

1. **Independent package — fixed parameters** (`ets_golden.rs`,
   `test_ets.py`). For the **twenty models without a multiplicative
   seasonal**, statsmodels `ETSModel(initialization_method="known").smooth(params)`
   at stated parameters and initial states — log-likelihood, fitted
   values, residuals, the level/trend/seasonal paths, `forecast(h)`, and
   `simulate(h, anchor="end")` along stated innovations — all at
   **1e-10**; for **all thirty**, `simulate(h, anchor="start")` along
   stated innovations at 1e-10 (statsmodels' simulator is written in the
   innovations form). For the six class-1 models the forecast-error
   variance against statsmodels' exact `get_prediction` at 1e-10.
2. **Documented-formula transcription** (1e-12): the Hyndman et al.
   (2008) recursion in R's `etscalc.c` arithmetic, transcribed in NumPy
   in the generator for all thirty models, and the Table 6.1 closed-form
   variances, which the generator first checks against the general
   $w'F^{j-1}g$ formula built from explicit matrices — the crate evaluates
   the general formula through the same recursion that produces the
   point forecast, so the six per-model formulas are never typed into
   Rust. **For the ten multiplicative-seasonal models statsmodels'
   Cython smoother is not the innovations form**: it updates the
   seasonal with the *post-update* level and $\gamma/(1-\alpha)$ (the
   classical Holt-Winters recursion, which agrees with the published
   state-space form only to first order in the error; its own `simulate`
   uses the innovations form). Measured on the airline series at fixed
   parameters (all ten models): fitted values differ by up to 0.65
   passengers and the log-likelihood by 0.28–0.49. The gap is recorded
   in the fixture as a convention difference and those ten models are
   pinned to the transcription (and, exactly, to statsmodels'
   `simulate`).
3. **Heuristic initialisation** against `holtwinters.ExponentialSmoothing(initialization_method="heuristic")`
   (cross-checked in the generator against `ETSModel`'s) at 1e-10 on 30
   (series, trend, seasonal) combinations, plus the "simple" short-sample
   rule on 6 more at 1e-12 — 36 fixture cases in all.
4. **Maximum likelihood** against statsmodels `ETSModel.fit()` (L-BFGS-B)
   on 21 (series, model, initialisation) cases, each run through two
   optimizers (`auto` and `bfgs`): match-or-beat on the log-likelihood at
   1e-5 relative — measured worst shortfall **2.67e-6**, on co2
   ETS(A,Ad,A) with estimated initial states (17 free parameters) — and
   parameters within 1e-3 where the optima coincide, measured worst
   $|\Delta\alpha| = 6.11 \times 10^{-5}$ over the 40 of 42 (case,
   optimizer) pairs that do coincide. On the other two — log-UKgas
   ETS(A,A,A) with estimated initial states under both optimizers, where
   statsmodels' L-BFGS-B stalls at the lower bound $\alpha = 10^{-4}$ —
   the crate's optimum is **better by 0.89 log-likelihood units**, so the
   parameter comparison is not meaningful and is skipped. The criteria
   reproduce the documented formulas with the crate's parameter count.
5. **`auto_ets`** — graded as `auto_arima` was, the selection loop having
   no runnable third-party reference (the M3-competition parity of the
   method is R-only): the candidate set reproduces R's `ets.R` loop
   exactly on 48 option combinations; every candidate's criterion is the
   golden-pinned `ets_fit` criterion; the winner is the table minimum and
   its refit is bitwise identical; and seeded Monte-Carlo recovery of the
   generating component form (below).
6. **Seeded Monte Carlo** (`ets_properties.rs`, run with `--nocapture` to
   print the numbers; the values below are from the committed seeds):

```text
Parameter recovery, T = 800, 40 replications per design (seed 101; gate:
|mean - truth| < 0.05 on every non-zero parameter):
  ETS(A,N,N)   alpha  0.500 -> 0.4964 (sd 0.0328)
  ETS(A,A,N)   alpha  0.400 -> 0.3958 (sd 0.0306)   beta  0.100 -> 0.0973 (sd 0.0103)
  ETS(A,N,A)   alpha  0.300 -> 0.2969 (sd 0.0254)   gamma 0.200 -> 0.1950 (sd 0.0227)
  ETS(M,N,N)   alpha  0.500 -> 0.4961 (sd 0.0326)
  ETS(M,Ad,M)  alpha  0.300 -> 0.2939 (sd 0.0389)   beta  0.030 -> 0.0303 (sd 0.0205)
                                                    gamma 0.150 -> 0.1457 (sd 0.0213)
  worst bias 0.0061 (ETS(M,Ad,M) alpha). Spread shrinks with T (seed 202,
  40 reps): ETS(A,N,N) alpha sd 0.0948 at T = 100 -> 0.0300 at T = 800
  (gate: sd(800) < 0.6 sd(100)).

Class-1 prediction-interval coverage, T = 300, h = 1..8, 300 replications
(seed 303; gate: 0.72-0.88 at 80%, 0.90-0.99 at 95%):
  model        80% (h=1, h=8)          95% (h=1, h=8)
  ETS(A,N,N)   0.812 (0.807, 0.793)    0.956 (0.953, 0.940)
  ETS(A,A,N)   0.804 (0.800, 0.807)    0.955 (0.957, 0.957)
  ETS(A,Ad,N)  0.794 (0.807, 0.787)    0.952 (0.950, 0.947)
  ETS(A,N,A)   0.812 (0.820, 0.800)    0.952 (0.947, 0.940)
Simulated vs exact interval width, ETS(M,N,N) against ETS(A,N,N) on the same
T = 400 series at a 1% relative error (seed 404, n_sim = 20000, seed 7;
gate: within 8%): measured |width ratio - 1| <= 0.016 over h = 1..6.

auto_ets component-form recovery, T = 600, 30 replications per design
(seed 505; gate: >= 0.7; a hit is the generating TREND and SEASONAL shape,
either error type and damped or not):
  generating form   hits counted            rate   picks
  ETS(A,N,N)        ANN, MNN                0.93   ANN 28, AAN 1, AAdN 1
  ETS(A,A,N)        AAN, AAdN, MAN, MAdN    1.00   AAN 29, AAdN 1
  ETS(A,N,A) m=4    ANA, MNA                0.93   ANA 28, AAA 1, AAdA 1
  ETS(M,Ad,M) m=4   MAM, MAdM               0.90   MAdM 25, MNM 3, MAM 2
```

   The class-1 intervals ignore parameter uncertainty (Hyndman et al.
   2008, §6.4), so plug-in coverage sits close to but not above nominal —
   measured 0.79–0.81 at a nominal 80% and 0.95–0.96 at a nominal 95%,
   with the shortfall growing mildly in h. The simulated interval of
   ETS(M,N,N) at a 1% relative error tracks the exact ETS(A,N,N) interval
   on the same data to within 1.6% in width (gated at 8%), which is the
   check that the simulator and the closed form agree where the two models
   nearly coincide. The auto-selection recovery counts the damped and
   undamped forms of the generating trend, and both error types, as hits:
   AICc is not consistent for either distinction at these sample sizes and
   does not try to be — what is graded is the component form.

## References

- Hyndman, R. J., A. B. Koehler, R. D. Snyder & S. Grose (2002), "A state
  space framework for automatic forecasting using exponential smoothing
  methods", *International Journal of Forecasting* 18, 439–454.
- Hyndman, R. J., A. B. Koehler, J. K. Ord & R. D. Snyder (2008),
  *Forecasting with Exponential Smoothing: The State Space Approach*,
  Springer — Tables 2.2/2.3 (the recursions), §2.6.1 (initialisation),
  chapter 6 / Table 6.1 (forecast variances), §7.2 (model selection).
- Ord, J. K., A. B. Koehler & R. D. Snyder (1997), "Estimation and
  prediction for a class of dynamic nonlinear statistical models",
  *JASA* 92, 1621–1629 (the likelihood).
- Hyndman, R. J. & Y. Khandakar (2008), "Automatic time series
  forecasting: the forecast package for R", *JSS* 27(3).

## Runnable example

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n, m = 160, 4                                # quarterly, multiplicative seasonal
t = np.arange(n)
season = np.array([0.85, 1.10, 1.15, 0.90])[t % m]
y = (100 + 0.6 * t) * season * (1 + 0.03 * rng.standard_normal(n))

fit = tsecon.ets_fit(y, error="mul", trend="add", damped=True,
                     seasonal="mul", seasonal_periods=m, horizon=8, seed=1)
print(fit["spec"], f"alpha={fit['alpha']:.3f} beta={fit['beta']:.3f} "
      f"gamma={fit['gamma']:.3f} phi={fit['phi']:.3f} aicc={fit['aicc']:.1f}")
print("intervals:", fit["interval_method"],        # "simulated" (not class 1)
      np.round(fit["forecast_lower"][:3], 1), np.round(fit["forecast_upper"][:3], 1))

auto = tsecon.auto_ets(y, seasonal_periods=m, horizon=8)
print("selected:", auto["spec"], f"(aicc={auto['ic_value']:.1f}, "
      f"{auto['n_fitted']}/{auto['n_candidates']} candidates fitted)")
for c in auto["candidates"][:4]:                 # near-ties: within ~2 of the best
    print(f"  {c['short_name']:5s} aicc={c['aicc']:.1f}")
# Exact intervals for the linear additive models:
lin = tsecon.ets_fit(np.log(y), trend="add", seasonal="add", seasonal_periods=m, horizon=8)
print(lin["interval_method"], np.round(lin["forecast_variance"][:3], 5))   # "exact"
```
