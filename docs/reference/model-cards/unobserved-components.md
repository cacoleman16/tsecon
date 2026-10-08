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
y_t = mu_t + gamma_t + c_t + a_t + beta' x_t + eps_t,   eps_t ~ N(0, sigma2.irregular)
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
- **autoregressive component** `a_t` (`autoregressive=p`): an AR(p) block
  after the cycle, `a_t = ar.L1 a_{t-1} + ... + ar.Lp a_{t-p} + eta_t`, in
  statsmodels' row-companion form, with the disturbance variance
  `sigma2.ar`; it is the stationary, serially correlated deviation that a
  random-walk level and white noise cannot express;
- **regressors** `exog` (T x k) with time-invariant coefficients `beta.x1,
  ...` estimated jointly by MLE (statsmodels `mle_regression=True`, the
  default), or — with `mle_regression=False` — carried in the state vector
  as `k` exactly diffuse constant states with the design row `x_t'`.

Under the defaults every state is initialized *exactly* diffuse (Koopman
1997), the cycle included, which is statsmodels' own convention under
`use_exact_diffuse=True` — so the log-likelihoods, AIC/BIC (its
`k_params + k_diffuse` degrees of freedom, `k_diffuse` counting the diffuse
states only) and every state moment agree with it to the digit. An AR block
is the one exception statsmodels makes too: it starts from its stationary
distribution (the Lyapunov solution of its companion block), and so does it
here, so AR specifications are directly comparable with statsmodels as
well. NaN in `y` is a missing period (the filter skips the update, the
smoother bridges the gap and the variance widens there); regressors must be
finite.

!!! warning "Two switches change what the log-likelihood *is*"

    `cycle_init="stationary"` and `mle_regression=False` are not tuning
    knobs: each changes the definition of the likelihood, so `loglik`,
    `aic` and `bic` are **not comparable across either switch**, not even at
    the same parameters. On 100 log US GDP the local level + damped cycle at
    the same five parameters scores −454.98 with the stationary cycle and
    −444.16 with the diffuse one (one diffuse element instead of three). The
    state-regression likelihood is the *marginal* likelihood, the MLE form
    the *profile* one, and at the GLS coefficients they differ by exactly
    `ln det(regression_coefficients_cov) / 2` (−4.34 to −5.09 on the six
    specifications in the fixture). And under `mle_regression=False` the
    likelihood also depends on the **units of the regressors**: multiplying
    one column by 100 moves it by exactly −ln 100 (−186.5505 → −191.1557)
    while no estimate moves; the MLE form does not care. Compare models
    within one setting of both switches, with the regressors in fixed
    units.

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
`stochastic_cycle` / `cycle_period_bounds` / `cycle_init` without
`cycle=True`, `mle_regression` — even `True` — without `exog`,
`forecast_exog` without `exog` or `forecast_steps`).

*`autoregressive=p`* (`None` by default; `0` is refused with "pass None",
and `p` may not exceed `len(y)`) adds the AR(p) component. Its parameters
follow statsmodels' order — `sigma2.ar` after the other state variances,
`ar.L1`…`ar.Lp` after `damping.cycle` — and the coefficients are searched
inside the stationarity region (the partial autocorrelations through `tanh`
and the Levinson-Durbin recursion; statsmodels uses `x/sqrt(1+x^2)` for the
same partial autocorrelations, which changes the path, not the optimum). A
fixed non-stationary polynomial is refused, naming the partial
autocorrelation that leaves (−1, 1); statsmodels has no such check and
returns a number there (on Nile, local level + AR(1) at `[8000, 1400, 6000,
phi]`: −88.02 at φ = 1.2 and −1403.49 at φ = 1.0), whereas at φ = 0.9999,
still stationary, both agree to 1.6e-15. The AR-UC likelihood is multimodal
— the AR block can trade places with the irregular, absorb a trend near the
unit root, or die out into a piled-up zero variance — so the start ladder
walks the first partial autocorrelation over 0.5, 0.9, 0, −0.5, 0.5, and
the second rung of an AR(p ≥ 2) starts hump-shaped at φ = (1.35, −0.5),
the autocorrelation of a business cycle around a trend (see failure modes).
The cost is `O(len(y) p^2)` per likelihood evaluation, so a very long AR is
slow rather than refused.

*`cycle_init="stationary"`* (`"diffuse"` when None, statsmodels'
convention; parsed trimmed and case-insensitive) starts the two cycle
states from the damped cycle's unconditional distribution,
`sigma2.cycle / (1 - damping^2)` on each, instead of a diffuse prior. A
damped stochastic cycle *is* a stationary process, so this is the prior
the model itself implies, and it removes the low-frequency singularity
above at the root: no cycle state is diffuse, so there is no
`-ln F_inf` term to run away with. At the US-GDP optimum (smooth trend plus
a stationary cycle of 6–32 quarters) moving only the frequency to 1e-2,
1e-4 and 1e-6 gives −266.559, −266.657, −266.657 — flat — where the diffuse
cycle at the same parameters climbs −258.509, −240.094, −238.025 in tsecon
(and −258.509, −253.642, −253.059 in statsmodels, whose absolute 1e-10
tolerance on `F_inf` clips it). It needs `damped_cycle=True` (an undamped
rotation has no stationary distribution) and `stochastic_cycle=True` (a
deterministic damped cycle has stationary variance zero, so it would be
identically zero), and a fixed damping of exactly 1 is refused. statsmodels
has no keyword for it; the reference is statsmodels' own block
`Initialization` (graded below).

*`mle_regression=False`* puts the regression coefficients in the state
vector — statsmodels' `mle_regression=False` — as `k` exactly diffuse
constant states after the AR block (`state_names` end with `beta.x1`…),
with design row `x_t'` and no `beta.x*` parameters. The variance parameters
are then estimated from the *marginal* (diffuse, REML-like) likelihood,
which accounts for the degrees of freedom spent on the coefficients, and
the coefficients come back as states: `regression_coefficients` (smoothed,
`nobs x k`, constant in `t` and equal to the GLS estimate given the
variances), `filtered_regression_coefficients` (the recursive GLS path),
their `_var`s, and `regression_coefficients_cov`, the `k x k` covariance
`Var(beta_gls | theta)`. With white noise only this is exactly OLS:
coefficients, `sigma2 (X'X)^-1`, forecasts `x' beta_hat` with variance
`sigma2 (1 + x'(X'X)^-1 x)`, pinned in closed form. `forecast_exog` is
still required for a forecast.

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
(`level`, `slope`, `seasonal`, `freq_seasonal[i]`, `cycle`,
`autoregressive` — the first AR state — each with `_var`; the regression
keys above under `mle_regression=False`; every key present, None when its
component is absent) are the **smoothed** two-sided estimates — what you want for
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
than `k_states + 2` observed values. Counts that size an allocation are
bounded so that a mistyped integer is a teaching error and not an allocator
abort: `seasonal` and each `freq_seasonal` period at most `len(y)`,
`forecast_steps` at most 100000, `n_starts` at most 64. A forecast whose variance is still
diffuse — infinite — at the forecast origin is refused rather than
returned; the sample-size check makes that hard to reach, and it is a
guard, not a routine outcome.

The AR component brings failure modes of its own. *Multimodality*: on US
inflation (local level + AR(2)) statsmodels' own L-BFGS stops at −455.6236,
1.89 log-likelihood points short of the optimum (−453.7374, a
near-unit-circle AR(2) at φ = (1.858, −0.939)) that SciPy's re-optimization
and this fit both reach at the default `n_starts=3`. *The AR block can die
out*: on 100 log US GDP (local linear trend + AR(2)) every start of the
first-PACF ladder alone lands on a local optimum at −259.87 where
`sigma2.ar` piles up at zero and the level variance absorbs everything; the
hump-shaped second rung reaches the genuine −250.18. When `sigma2.ar` is
flagged like that, a zero-variance stationary block is identically zero, so
`ar.L1`… carry no information and get NaN standard errors (the same rule
drops `frequency.cycle` and `damping.cycle` when `sigma2.cycle` piles up
under the stationary cycle) instead of every standard error going NaN — or
coming back as a huge number read off a near-singular pivot. *Near the unit
root*: the Hessian probe of an AR coefficient is halved until every probe
point is stationary, so an estimate within 1e-4 of the boundary still
leaves the other standard errors finite; and a stationary block whose root
sits within roundoff of the unit circle — where `tanh` or the logistic map
saturate — passes the Lyapunov solve but has an unconditional variance some
1e15 times its one-step variance, at which point the filter can only skip
the later observations as numerically singular and the "log-likelihood"
collapses to its diffuse terms (on Nile, local level + AR(1) at φ = 1 −
1e-15, the raw filter returns exactly one diffuse term, −ln(2π)/2 = −0.919,
against statsmodels' own −637.04 at φ = 0.9999). Such a likelihood is
undefined in the search and refused at fixed parameters.

Under `mle_regression=False`, *collinear regressors* leave one diffuse
direction (the difference of the collinear coefficients) that the data can
never resolve. Nothing is refused — the likelihood and every forecast that
does not load on that direction stay defined, exactly as for the Nyquist
harmonic above — but the diffuse period never ends: `nobs_diffuse` equals
the sample length and `std_resid` is NaN throughout. Drop the redundant
column.

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

**Validated against — the 0.11.0 options** (`fixtures/uc_extensions.json`,
generator `fixtures/generate_uc_extensions_fixtures.py`;
`crates/tsecon-ssm/tests/uc_extensions_golden.rs`, the in-module unit tests
of `uc.rs` and `mle.rs`, `uc_properties.rs`,
`bindings/python/tests/test_uc_extensions.py`). The fixed-parameter
tolerance is the same 1e-8 with the same recorded `smoother_spread`
relaxation inside the diffuse period — and it engages once: on the
combined specification below statsmodels' two smoothers disagree by 6.3e-8
there.

- *Autoregressive component* — an independent-package golden: statsmodels'
  own `autoregressive=` at fixed parameters on nine specifications (Nile
  local level + AR(1) and AR(2), random walk + AR(1), inflation local level
  + AR(2), GDP local linear trend + AR(2), GDP growth white noise + AR(3)
  with every state stationary and `nobs_diffuse = 0`, and simulated fixed
  intercept + AR(1), local level + seasonal + diffuse cycle + AR(1), local
  linear trend + seasonal + AR(2) + regressors), with the AR component path;
  the initial covariance against the closed forms `sigma2.ar / (1 - phi^2)`
  and the AR(2) autocovariances; the MLE on Nile (−631.3802544693),
  inflation (−453.7373687669) and GDP (−250.1813180960) reaching the better
  of two optimizers at the default `n_starts=3`, parameters 2e-3, pile-up
  flags exact, standard errors 2e-2 against statsmodels' complex-step
  Hessian over the free set (which mirrors the dropped-coefficient rule).
- *Stationary cycle* — graded as statsmodels machinery under a documented
  initialization, since statsmodels has no keyword for it: a subclass whose
  `initialize_default` builds statsmodels' own block `Initialization`
  ("stationary" on the cycle pair and any AR block, "diffuse" elsewhere;
  the cycle located through the public `state_names`), so the Lyapunov
  solve, the exact-diffuse filter and the smoother are statsmodels'. Fixed
  parameters on GDP (−454.978475273841), Nile (−631.619075075044), a
  simulated local linear trend + seasonal + regressors, and GDP smooth trend
  + stationary cycle + AR(1) as one block-diagonal Lyapunov system
  (−265.424367261473), at 1e-8; the closed form `P_* = sigma2.cycle /
  (1 - damping^2)` (10.526315789474 at 2 and 0.9) in the crate's unit test;
  the GDP MLE (smooth trend, 6–32 quarters) reaching SciPy's −252.1147178886
  (statsmodels −252.1147179444) with the irregular variance piled up; the
  flat `lambda -> 0` sweep at 1e-8.
- *Regressors in the state* — statsmodels' `mle_regression=False` at fixed
  parameters on four specifications, with the coefficient paths and their
  covariance: a local level, white noise (the closed-form case), a local
  linear trend + AR(2) with NaN at t = 5, 6, 50 and 119 (the forecast
  origin missing), and one specification with every new option at once
  (smooth trend, a 12-period two-harmonic trigonometric seasonal, the
  stationary cycle, an AR(1) and the regressors). Closed forms at 1e-10:
  white noise plus regressors is OLS — coefficients, `sigma2 (X'X)^-1`,
  the diffuse log-likelihood (−479.125183355077), forecasts and their
  variances — and it equals `tvp_regression(constant=False)` at zero
  coefficient variances to 1e-12; the marginal/profile identity on six
  specifications and the regressor-units identity (shift exactly −ln 100)
  at 1e-10; the MLE reaching the better of two optimizers
  (−185.1819788397).
- *Seeded Monte Carlo and identities* (`uc_properties.rs`): the stationary
  cycle recovers a period-12, damping-0.9 cycle (8 seeds, T = 200: median
  |period error| 0.637, median |damping error| 0.0117); local level +
  AR(1) recovers φ = 0.8 (10 seeds, T = 200: median |error| 0.0556); at the
  true variances the state-regression intervals cover the coefficients at
  0.9475 and 0.9515 (2000 seeds, nominal 0.95); scale equivariance holds
  with an AR component (estimates 7.3e-8, log-likelihood 2.5e-16) and
  under state regression (4.5e-8, 6.1e-16); the stationary cycle's
  log-likelihood moves by 3.6e-3 over frequencies 1e-2 to 1e-8, where the
  diffuse cycle on the same series climbs 25 points between 1e-2 and 1e-6;
  and the default specifications reproduce
  0.10.0 **bit for bit** — the exact IEEE-754 patterns of three `uc.json`
  MLE fits and the `tvp_regression` MLE, recorded before the shared
  likelihood layer was touched.

**References.** Harvey, A. C. (1989), *Forecasting, Structural Time Series
Models and the Kalman Filter*, CUP; Harvey & Durbin (1986), "The Effects of
Seat Belt Legislation on British Road Casualties", *JRSS A* 149; Durbin &
Koopman (2012), *Time Series Analysis by State Space Methods*, 2nd ed., OUP
(ch. 2–3 and the exact-diffuse initialization of ch. 5; secs. 5.7 and 7.2
for regression coefficients in the state); Koopman (1997),
"Exact Initial Kalman Filtering and Smoothing for Nonstationary Time Series
Models", *JASA* 92; Shephard & Harvey (1990), "On the Probability of
Estimating a Deterministic Component in the Local Level Model", *J. Time
Series Analysis* 11; de Jong, P. (1991), "The Diffuse Kalman Filter",
*Annals of Statistics* 19; Monahan, J. F. (1984), "A Note on Enforcing
Stationarity in Autoregressive-Moving Average Models", *Biometrika* 71.

```python
import numpy as np, statsmodels.api as sm, tsecon

nile = sm.datasets.nile.load_pandas().data["volume"].values.astype(float)
fit = tsecon.unobserved_components(nile, forecast_steps=5)
print({k: round(float(v), 1) for k, v in zip(fit["param_names"], fit["params"])},
      fit["at_boundary"])
# {'sigma2.irregular': 15098.5, 'sigma2.level': 1469.2} [False, False]
band = 1.96 * np.sqrt(fit["level_var"])          # smoothed level ± band
print(np.round(fit["forecast"], 1), np.round(np.sqrt(fit["forecast_var"]), 1))
```

The 0.11.0 options. An AR(1) beside the Nile level — the optimum both
reference optimizers reach (−631.3802544693), every variance interior:

```python
import numpy as np, statsmodels.api as sm, tsecon

nile = sm.datasets.nile.load_pandas().data["volume"].values.astype(float)
ar = tsecon.unobserved_components(nile, autoregressive=1)
print({k: round(float(v), 4 if k == "ar.L1" else 1) for k, v in zip(ar["param_names"], ar["params"])})
# {'sigma2.irregular': 7873.5, 'sigma2.level': 521.4, 'sigma2.ar': 8519.3, 'ar.L1': 0.4733}
print({"loglik": round(ar["loglik"], 4), "k_diffuse": ar["k_diffuse"], "at_boundary": ar["at_boundary"]})
# {'loglik': -631.3803, 'k_diffuse': 1, 'at_boundary': [False, False, False, False]}
```

A smooth trend plus a stationary business cycle on 100 log US real GDP,
periods confined to 6–32 quarters. The irregular variance piles up at zero
(flagged, NaN standard error), the cycle runs about seven years with
damping 0.94, and only the two trend states are diffuse:

```python
import numpy as np, statsmodels.api as sm, tsecon

md = sm.datasets.macrodata.load_pandas().data
lgdp = 100 * np.log(md["realgdp"].values.astype(float))
gdp = tsecon.unobserved_components(lgdp, level="strend", cycle=True, damped_cycle=True,
                                   stochastic_cycle=True, cycle_period_bounds=[6, 32],
                                   cycle_init="stationary")
p = dict(zip(gdp["param_names"], gdp["params"]))
print({"period": round(float(2 * np.pi / p["frequency.cycle"]), 2), "damping": round(float(p["damping.cycle"]), 4),
       "loglik": round(gdp["loglik"], 4), "k_diffuse": gdp["k_diffuse"]})
# {'period': 28.86, 'damping': 0.9401, 'loglik': -252.1147, 'k_diffuse': 2}
print({k: b for k, b in zip(gdp["param_names"], gdp["at_boundary"]) if b})
# {'sigma2.irregular': True}
```

Regressors in the state: the coefficients come back as smoothed states with
their covariance, and the log-likelihood is the marginal one — 6.3 points
below the MLE form's profile likelihood here, which is a difference of
definition, not of fit:

```python
import numpy as np, tsecon

rng = np.random.default_rng(7)
T = 200
X = rng.normal(size=(T, 2))
y = np.cumsum(rng.normal(0, 0.3, T)) + X @ [1.0, -0.5] + rng.normal(0, 0.5, T)
st = tsecon.unobserved_components(y, exog=X, mle_regression=False)
beta = np.asarray(st["regression_coefficients"])[-1]            # GLS, constant over t
se = np.sqrt(np.diag(np.asarray(st["regression_coefficients_cov"])))
print({"beta": np.round(beta, 3).tolist(), "se": np.round(se, 3).tolist()})
# {'beta': [0.963, -0.525], 'se': [0.044, 0.042]}
ml = tsecon.unobserved_components(y, exog=X)                      # the MLE form, for contrast
print({"beta_mle": np.round(ml["params"][2:], 3).tolist(), "loglik_gap": round(st["loglik"] - ml["loglik"], 3)})
# {'beta_mle': [0.963, -0.525], 'loglik_gap': -6.309}
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
