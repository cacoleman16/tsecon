# Structural time-series models (unobserved components) & TVP regression

`unobserved_components` · `tvp_regression`

Harvey's structural time-series models and the random-walk-coefficient
regression, both on the crate's exact-diffuse Kalman filter and smoother
(`tsecon-ssm`, the engine that also carries `local_level_smooth`,
`ar_loglik` and the ARIMA exact likelihood). The two callables share one
maximum-likelihood layer — the square-root / logistic working space of
statsmodels' `UnobservedComponents`, BFGS + Nelder-Mead from a deterministic
ladder of starts, scale-adaptive estimation, observed-information standard
errors, and the *pile-up* check that reports a variance the likelihood cannot
tell from zero as a boundary flag rather than as a tiny number. The
validation story is the same for both: every fixed-parameter quantity is
pinned to statsmodels at 1e-8, the optimum to the better of two independent
optimizers, and the statistical claims are measured by seeded Monte Carlo.

---

## `unobserved_components` — Harvey's structural models by exact-diffuse MLE

**What it estimates.** The decomposition

```
y_t = mu_t + gamma_t + c_t + beta' x_t + eps_t,   eps_t ~ N(0, sigma2.irregular)
```

with the blocks assembled exactly as statsmodels' `UnobservedComponents`
enumerates them (Harvey 1989, ch. 2; Durbin & Koopman 2012, ch. 3):

- **level / trend** `mu_t`, chosen by `level=` in statsmodels' vocabulary —
  `"llevel"` (random-walk level + irregular), `"lltrend"` (random-walk level
  and slope), `"strend"` (integrated random walk: smooth trend),
  `"rwdrift"`, `"dtrend"` (deterministic line + noise), `"lldtrend"`,
  `"rwalk"`, `"rtrend"`, `"dconstant"`, `"ntrend"` (white noise only), and
  the noise-free `"fixed intercept"` / `"fixed slope"` (which need another
  stochastic component); the long names parse too;
- **dummy seasonal** `gamma_t` of period `s` (`seasonal=s`, `s-1` states):
  the dummies sum to a disturbance of variance `sigma2.seasonal`, or exactly
  to zero when `stochastic_seasonal=False`;
- **trigonometric seasonal** (`freq_seasonal=[p, ...]`): for each period
  `p` and `h` harmonics (default `floor(p/2)`), `h` rotation pairs at
  frequencies `2 pi j / p` driven by two disturbances of a common variance
  `sigma2.freq_seasonal_p(h)` (or deterministic);
- **stochastic cycle** `c_t` (`cycle=True`): the rotation by the cycle
  frequency `lambda` — a free parameter confined to `(2 pi/max, 2 pi/min)`
  by `cycle_period_bounds=[min, max]`, default `[2, len(y)]` — scaled by a
  damping `rho in (0, 1)` when `damped_cycle=True`, with a common
  disturbance variance when `stochastic_cycle=True`;
- **regressors** `exog` (T x k) with time-invariant coefficients `beta.x1,
  ...` estimated jointly by MLE (statsmodels `mle_regression=True`).

Every state is initialized *exactly* diffuse (Koopman 1997), the cycle
included, which is statsmodels' own convention under
`use_exact_diffuse=True` — so the log-likelihoods, AIC/BIC (its
`k_params + k_diffuse` degrees of freedom) and every state moment agree with
it to the digit. NaN in `y` is a missing period (the filter skips the
update, the smoother bridges the gap and the variance widens there);
regressors must be finite.

**Assumptions.** Gaussian disturbances, time-invariant variances, a
linear-Gaussian state space — no regime switching, no stochastic
volatility, no outlier robustness (see `dcs_local_level` for a level that
discounts outliers). The trigonometric seasonal assumes a fixed period; the
cycle a single frequency. Regression coefficients are constant (for drifting
coefficients use `tvp_regression`).

**When to use (and when not).** Use it when the question is *what are the
components* — trend extraction with honest uncertainty, a seasonal that is
allowed to evolve, a business-cycle component with an estimated period and
damping, an intervention effect measured against a structural background
(the seat-belt example below), or a forecast that carries the state
uncertainty through the horizon. It is the model-based alternative to
`hp_filter`/`stl`: the components come with variances, missing data costs
nothing, and the signal-to-noise ratios are *estimated* rather than fixed
by a smoothing constant. Do **not** use it for pure short-horizon point
forecasting of a stationary series (`arima_fit`/`auto_arima` are cheaper and
the ARIMA reduced form is equivalent), for series shorter than roughly three
times the state dimension (the diffuse period eats `k_states`
observations), or when you need the components to be robust to additive
outliers.

**Key arguments and defaults (and why).** `level="llevel"` because the
local level is the "hello world" of the family and the safe default for a
noisy level; add `"lltrend"` when the series drifts, `"strend"` when you
want a smooth trend (the HP-filter-like case). `stochastic_seasonal=True`,
`stochastic_cycle=False`, `damped_cycle=False` follow statsmodels so a
spec written for it fits here unchanged. **`cycle_period_bounds` defaults
to `[2, len(y)]`, and this is a deliberate divergence from statsmodels**,
which leaves the upper period at infinity when the series carries no
frequency information. Under exact-diffuse initialization the
log-likelihood of a stochastic cycle *diverges* as the frequency goes to
zero: at `lambda = 0` the second cycle state is unobservable and stays
diffuse for free, but at a small `lambda > 0` it is weakly observable
through `rho sin lambda`, so its diffuse direction resolves with
`F_inf ~ (rho sin lambda)^2` and contributes `-(ln 2 pi + ln F_inf)/2`,
which grows without bound. Measured in
`crates/tsecon-ssm/tests/uc_properties.rs`: on a seeded local-level series
the same model at `lambda = 1e-2, 1e-4, 1e-6` scores −188.15, −183.55,
−162.53 — 25 log-likelihood points of pure singularity, and a free
optimizer walks straight into it and reports a "cycle" of period `10^6`.
statsmodels does not meet the singularity because its diffuse tolerance is
an absolute `1e-10` on `F_inf`, which clips it. A cycle longer than the
sample is not identified by the sample in any case, so the sample length is
the honest bound; pass the plausible period range (e.g. `[6, 32]` quarters
for a business cycle) to identify the cycle properly and keep the optimizer
off the seasonal frequencies. `n_starts=3` runs the start ladder (the
heuristic start, then the state variances scaled by 0.1 and 10; for a
cycle, the periodogram peak first, then the bound midpoint and quarter
points) — the cycle likelihood is multimodal in the frequency, and the
periodogram start is what finds the right mode. `fixed_params=` evaluates
the model at given
parameters (statsmodels' order, listed in `param_names`) — the
fixed-parameter goldens, and the way to run a Durbin-Koopman example at the
book's values. Options that act only under a component **raise** when
passed without it (`stochastic_seasonal` without `seasonal`, the
`freq_seasonal_*` companions without `freq_seasonal`, `damped_cycle` /
`stochastic_cycle` / `cycle_period_bounds` without `cycle=True`,
`forecast_exog` without `exog` or `forecast_steps`).

**How to read the output.** `params`/`se`/`param_names` are the MLE with
observed-information standard errors — *conditional* on the flagged
parameters sitting exactly at their boundary, i.e. the information matrix
is inverted over the free parameters only. statsmodels' `cov_type="approx"`
inverts the full matrix instead; the two coincide exactly when nothing is
flagged and differ by definition when something is (on the seat-belt model
below its full Hessian is indefinite — its own `bse` for `sigma2.trend`
comes back NaN — and its `sigma2.level` standard error is 26 times smaller
than the conditional one). `at_boundary[i]` is the pile-up flag
— setting that variance to exactly zero (everything else at the estimate)
lowers the log-likelihood by less than 1e-4, so the data cannot tell the
estimate from zero, and its `se` is NaN because a boundary has no
curvature (Shephard & Harvey 1990). The component keys without a prefix
(`level`, `slope`, `seasonal`, `freq_seasonal[i]`, `cycle`, each with
`_var`) are the **smoothed** two-sided estimates — what you want for
historical decomposition; `filtered_*` are the real-time one-sided ones.
`smoothed_state`/`filtered_state` hold the full state (`state_names`).
`fitted` is the one-step-ahead prediction of `y_t`, `resid` the prediction
error, `std_resid` its standardized version — NaN inside the diffuse
period (`nobs_diffuse` periods, where no finite prediction variance exists)
and at missing periods; use it for the usual Ljung-Box / normality checks.
`forecast`/`forecast_var` are the `forecast_steps` out-of-sample means and
variances (state uncertainty plus the irregular); `1.96 * sqrt(forecast_var)`
is the nominal 95% band. `converged` is the optimizer's certificate, not a
fit grade: read `at_boundary` first.

**Failure modes.** Pile-up is the family's signature: with a true small
variance the MLE lands on exactly zero with substantial probability, and a
short series will do it to the slope or seasonal variance routinely (the
seat-belt example below does it to both) — the flag tells you, and the
right response is to accept the deterministic component, not to read the
tiny estimate. Unidentified combinations (a dummy seasonal *and* a
trigonometric seasonal of the same period; a cycle whose period range
overlaps a seasonal frequency) give flat likelihood directions, a
non-converged flag or NaN standard errors. A constant series is refused
(the likelihood is unbounded as the variances go to zero), as is a
specification with no stochastic component at all, and a sample with fewer
than `k_states + 2` observed values. A forecast whose variance is still
diffuse — infinite — at the forecast origin is refused rather than
returned; the sample-size check makes that hard to reach, and it is a
guard, not a routine outcome.

**Validated against.** statsmodels `UnobservedComponents(...,
use_exact_diffuse=True)` — an independent package — at *fixed parameters*:
`loglike`, filtered and smoothed states and their variances, one-step
predictions and residuals, standardized residuals (after the diffuse
period), 8-step forecasts and forecast variances, AIC/BIC and the component
paths, for 26 component combinations (every level/trend specification,
dummy seasonals stochastic and deterministic, trigonometric seasonals with
one, two and three harmonics and with two blocks, undamped/damped and
deterministic/stochastic cycles, regressors, and the combinations of them)
on a seeded simulated series and on three NaN-inserted variants, all at
**1e-8 relative** (`fixtures/uc.json`,
`crates/tsecon-ssm/tests/uc_golden.rs`) — with one measured exception,
stated rather than papered over. The exact-diffuse *smoother* is the
ill-conditioned part of this family, and only inside the diffuse period:
on the eight-diffuse-state combination (smooth trend + two harmonics +
damped stochastic cycle + two regressors) statsmodels' own univariate and
conventional smoothers disagree with **each other** by up to 4.5e-3
relative on the smoothed state variances there, while their filters agree
to 2.9e-11. The fixture records that internal spread per case
(`smoother_spread`, computed by the generator) and the tests use it as the
tolerance for smoothed variances over the diffuse period and the two rows
after it: tsecon must be at least as close to the reference as the
reference is to itself. On 19 of the 26 combinations the two reference
paths agree bit for bit and the tolerance is exactly 1e-8; the largest
tsecon-vs-statsmodels gap anywhere in that window is 8.8e-4 on the same
eight-state case (against its 4.5e-3 reference spread), and every smoothed
variance from `nobs_diffuse + 3` onwards holds 1e-8. Filtered variances,
means, residuals, forecasts, log-likelihood and AIC/BIC are at 1e-8
throughout. The MLE is pinned to the **better
of two optimizers** — statsmodels' own `fit` and a SciPy Nelder-Mead +
L-BFGS-B re-optimization of the identical criterion — on six cases: the
Rust optimum is never below the better reference by more than 1e-5, and at
a shared interior optimum the parameters agree to 2e-3, the pile-up flags
agree exactly with an independent implementation of the same documented
criterion in the fixture generator, and the standard errors agree to 2e-2
with statsmodels' own complex-step Hessian inverted over the non-boundary
parameters (`se_conditional` in the fixture; the full-Hessian
`cov_type="approx"` numbers are recorded beside it so the difference stays
visible). On the bounded cycle case the two reference optimizers land
0.0031 log-likelihood apart (statsmodels −232.2477, SciPy −232.2446) and
the Rust fit reaches the better of them. The Nile
local level reproduces Durbin & Koopman (2012, §2.2) as printed —
15098.5 / 1469.18 against their 15099 / 1469.1. The Harvey-Durbin (1986)
basic structural model on the UK `Seatbelts` data (`log(drivers)` on a
local linear trend, a stochastic monthly dummy seasonal, log petrol price,
log kms and the January-1983 law dummy) — fetched from Rdatasets by the
fixture generator and again by the Python test, never redistributed (R's
`datasets` is GPL), so the fixture holds only the derived optimum — is
re-estimated from scratch and reaches the statsmodels/SciPy optimum: the
slope and seasonal variances pile up at zero (statsmodels' optimum has them
at 3e-21 and 4e-19; both flagged, NaN standard errors) and the law
coefficient is −0.240 (se 0.047) — a 21% reduction in drivers killed or
seriously injured — with the fixed-parameter evaluation at that optimum
matching a live statsmodels run at 1e-8; no published number is asserted,
only what the committed generator reproduces. Seeded Monte Carlo
(`crates/tsecon-ssm/tests/uc_properties.rs`, ten tests, every number below
reproducible at the seeds in that file): local-level variance recovery at
T = 1200 over 12 seeds — median |relative error| **0.036** on
`sigma2.irregular` and **0.096** on `sigma2.level`, mean relative bias
−0.018 and +0.015, and no spurious boundary flag in any of the 12;
95% forecast-interval coverage on a local-linear-trend DGP (T = 150, 12
steps ahead, 80 seeds) — **[0.925, 0.963, 0.975, 0.975, 0.938, 0.950,
0.963, 0.963, 0.950, 0.938, 0.925, 0.938]** at horizons 1…12, pooled
**0.950** against a nominal 0.95 (a Monte-Carlo standard error of 0.024 per
horizon at 80 replications, so these are all within one standard error);
scale equivariance of the whole fit under `y -> 1000 y` — estimates
2.9e-10, log-likelihood 2.2e-15 (against the *exact-diffuse* shift
`-(n - d) ln c`, not `-n ln c`), level and forecast paths 3.1e-11,
standardized residuals 3.3e-10, and 2.3e-8 on the observed-information
standard errors, which are a numerical Hessian and amplify the last ulp of
the standardization scale; the deterministic dummy seasonal is exactly
4-periodic and sums to zero over every period (8.9e-16 and 5.6e-16 against
an amplitude of 1.4); a frequency-π (Nyquist) harmonic state cannot move
the log-likelihood (difference exactly 0.0 against a hand-built model with
the state removed); the cycle log-likelihood's divergence as the frequency
goes to zero is measured rather than asserted (−188.15, −183.55, −162.53 at
λ = 1e-2, 1e-4, 1e-6) and the default period bound is shown to keep the fit
out of it; the cycle frequency lands inside its bounds and recovers a
period-8 cycle within 1.5 periods. Base-R `StructTS` was *not* used as a second reference:
it fits by a different (non-diffuse, `optim`) likelihood and would only
have added a lower-graded number.

**References.** Harvey, A. C. (1989), *Forecasting, Structural Time Series
Models and the Kalman Filter*, CUP; Harvey & Durbin (1986), "The Effects of
Seat Belt Legislation on British Road Casualties", *JRSS A* 149; Durbin &
Koopman (2012), *Time Series Analysis by State Space Methods*, 2nd ed., OUP
(ch. 2–3 and the exact-diffuse initialization of ch. 5); Koopman (1997),
"Exact Initial Kalman Filtering and Smoothing for Nonstationary Time Series
Models", *JASA* 92; Shephard & Harvey (1990), "On the Probability of
Estimating a Deterministic Component in the Local Level Model", *J. Time
Series Analysis* 11.

```python
import numpy as np, statsmodels.api as sm, tsecon

nile = sm.datasets.nile.load_pandas().data["volume"].values.astype(float)
fit = tsecon.unobserved_components(nile, forecast_steps=5)
print(dict(zip(fit["param_names"], np.round(fit["params"], 1))), fit["at_boundary"])
# {'sigma2.irregular': 15098.5, 'sigma2.level': 1469.2} [False, False]
band = 1.96 * np.sqrt(fit["level_var"])          # smoothed level ± band
print(np.round(fit["forecast"], 1), np.round(np.sqrt(fit["forecast_var"]), 1))
```

---

## `tvp_regression` — random-walk coefficients with the pile-up check

**What it estimates.** The time-varying-parameter regression

```
y_t = x_t' beta_t + eps_t,    beta_{t+1} = beta_t + eta_t,
eps_t ~ N(0, sigma2_eps),     eta_t ~ N(0, diag(sigma2_beta)),   beta_1 diffuse
```

(Harvey 1989, §8.3; Durbin & Koopman 2012, §3.6) — the state-space form
with the per-period design row `x_t'`, identity transition, and one
innovation variance per coefficient, all `k` coefficients exactly diffuse.
`constant=True` (default) prepends a random-walk intercept. The `k + 1`
variances are estimated by MLE; `beta_filtered`/`beta_smoothed` (T x k,
with `_var`) are the real-time and two-sided coefficient paths. With every
state variance at zero the filter *is* recursive least squares: the
filtered path is the expanding-window OLS estimate (statsmodels
`RecursiveLS`), which `fixed_params=[sigma2_eps, 0, ..., 0]` reproduces.

**Assumptions.** Gaussian errors, regressors that are finite and
predetermined, coefficients that drift as independent random walks (no
mean reversion — for a coefficient that returns to a mean the random walk
overstates long-run drift), constant observation variance.

**When to use (and when not).** Use it when a relationship is suspected to
drift — a pass-through, a beta, a Phillips-curve slope — and you want the
path with a variance rather than a rolling window's arbitrary width; the
smoothed path is the object for description, the filtered path for
real-time questions. The estimator's known weakness is what the pile-up
check is for: with modest true variation the MLE of a state variance is
exactly zero with high probability (Stock & Watson 1998), so a flagged
coefficient means *the data cannot show it moving*, not that it is
constant. Do **not** read a flagged coefficient's `se` (it is NaN on
purpose), do not use it for regressors with a unit root without thinking
about spurious drift, and do not expect it to separate coefficient drift
from omitted-variable bias.

**Key arguments and defaults (and why).** `constant=True` because a
regression without an intercept is rarely intended; `n_starts=3` (state
variances at 1%, 0.01% and 100% of the OLS residual variance) because the
likelihood is often flat near zero in a state variance and the ladder
brackets the pile-up. `fixed_params` runs the filter at given variances —
the RLS limit above, or a variance ratio taken from the literature (the
Stock-Watson median-unbiased estimator is a roadmap follow-up).

**How to read the output.** `sigma2_eps`, `sigma2_beta` (and the same in
`params`/`se`/`param_names`); `pile_up[j]` (= `at_boundary[j+1]`) flags a
coefficient variance the likelihood cannot tell from zero (zeroing it
costs < 1e-4 log-likelihood), with a NaN `se`; `beta_*_var` are the
diagonal coefficient variances (finite part inside the `nobs_diffuse`-long
diffuse period for the filtered ones, exact for the smoothed ones);
`fitted`/`resid`/`std_resid` are the one-step quantities as for
`unobserved_components`; `aic`/`bic` follow statsmodels'
`k_params + k_diffuse` convention.

**Failure modes.** Collinear regressors give a slow, badly conditioned
diffuse period and NaN standard errors; a column that is identically zero
is refused. With fewer than `k + 2` observed values the diffuse period
cannot end and the fit is refused. Pile-up at zero is the expected
behaviour on constant coefficients (measured below), so a study that
*needs* an unbiased variance should use the median-unbiased approach; this
callable reports the MLE honestly.

**Validated against.** A statsmodels `MLEModel` transcription of the
documented state-space form (per-period design, exact diffuse) at fixed
parameters — log-likelihood, filtered and smoothed coefficient paths and
variances, one-step predictions, residuals and standardized residuals,
AIC/BIC — at **1e-8**, on a seeded series and on a NaN-inserted variant;
statsmodels `RecursiveLS` in the zero-state-variance limit — its filtered
coefficients at 1e-8 and its concentrated log-likelihood equal to the TVP
log-likelihood at `sigma2_eps = scale`, with the last filtered coefficient
equal to full-sample OLS; the MLE pinned to the better of statsmodels'
`fit` and a SciPy re-optimization (1e-5 in log-likelihood, 2e-3 in the
interior parameters, 2e-2 in their standard errors) with the true-zero
third variance flagged (statsmodels' optimum has it at 3e-17). Seeded Monte
Carlo (`uc_properties.rs`): with constant true coefficients (T = 200, 24
seeds, 48 coefficient variances) the share flagged at zero is **0.604**;
with clearly moving coefficients (`sigma2_beta = 0.05`) it is **0.000**,
and those 48 estimates run from 0.0029 through a **median of 0.0417** to
0.109 against a truth of 0.05 — the median is close, the individual draws
are not, which is the attenuation the pile-up section warns about and the
reason the *path* is the deliverable. The zero-variance filter equals
expanding-window OLS at every checked date (closed form, 1e-8).

**References.** Harvey (1989, §8.3); Durbin & Koopman (2012, §3.6); Stock,
J. H. and M. W. Watson (1998), "Median Unbiased Estimation of Coefficient
Variance in a Time-Varying Parameter Model", *JASA* 93; Shephard & Harvey
(1990).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
T = 300
x = rng.normal(size=(T, 1))
beta = 0.5 + np.cumsum(rng.normal(0, 0.05, T))     # a drifting slope, true var 0.0025
y = 1.0 + beta * x[:, 0] + rng.normal(0, 0.3, T)
r = tsecon.tvp_regression(y, x)
print(np.round(r["sigma2_beta"], 4), r["pile_up"])
# [0.     0.0012] [True, False]   -> the intercept cannot be shown to move; the slope can
path = np.asarray(r["beta_smoothed"])[:, 1]         # smoothed slope, with r["beta_smoothed_var"]
np.corrcoef(path, beta)[0, 1]                       # 0.882
```

Read that output the way the failure-mode section says to. The intercept
really is constant and its variance is flagged, which is the right answer.
The slope variance really is 0.0025 and the MLE says 0.0012: the
random-walk-variance MLE is severely attenuated at this sample size, and
the smoothed path still tracks the truth at a correlation of 0.88. With the
same slope drift against a *unit* noise standard deviation instead of 0.3,
the same call returns `3.3e-5` — a seventy-fold understatement, verified
against statsmodels to be the true maximum of the likelihood, not an
optimizer failure. The path is the deliverable; the variance is a nuisance
parameter with a badly behaved estimator (Stock & Watson 1998).

---
