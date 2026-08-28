# Model card — Cointegration and regime switching

`johansen` · `vecm` · `ou_fit` · `spread_zscore` · `markov_switching_ar` ·
`setar` · `setar_test` · `star` · `star_eval` · `star_test`
`johansen` · `vecm` · `threshold_vecm` · `hansen_seo_test` · `ou_fit` ·
`spread_zscore` · `markov_switching_ar` · `setar` · `setar_test` ·
`threshold_var` · `threshold_var_test`

Two ways the tidy linear-stationary world breaks. First, series can be
individually nonstationary yet move together — share a long-run equilibrium
(cointegration); differencing away the trends throws that equilibrium away, and
the vector error-correction model keeps it. Once a cointegrating spread is in
hand, `ou_fit` / `spread_zscore` quantify what a trading workflow does with it
— how fast it mean-reverts (half-life) and how far it sits from equilibrium
(z-score); they live in this card, not the forecasting one, because the spread
*is* the cointegrating residual and the estimator is the continuous-time twin
of the error-correction speed `alpha`. Second, the parameters themselves
can switch between regimes — either *unobserved* states governed by a hidden
Markov chain (`markov_switching_ar`), or *observed* states triggered when a
lagged value of the series itself crosses a threshold — abruptly (`setar`,
with `setar_test` deciding whether a threshold exists at all) or smoothly
(`star`, the smooth-transition family, with `star_test` running the
Terasvirta modeling cycle: is there nonlinearity at all, and is it logistic
or exponential?).
lagged value of the series itself crosses a threshold (`setar`, with
`setar_test` deciding whether a threshold exists at all; `threshold_var` /
`threshold_var_test` are the same story for a *system* of series). The two
breaks meet in **threshold cointegration** (`threshold_vecm` /
`hansen_seo_test`): the long-run equilibrium exists, but the *correction
toward it* switches regimes when the equilibrium error itself crosses a
threshold — arbitrage that only kicks in once the spread is wide enough to
cover transaction costs.

---

## `johansen` — cointegration rank test

**What it estimates.** How many independent long-run equilibrium relations tie a
set of I(1) series together — the cointegrating rank `r`. Runs Johansen's trace
and maximum-eigenvalue tests sequentially against tabulated critical values.

**Assumptions.** Each series is I(1) (test first — `check_stationarity` on each
column); the VECM lag order `k_ar_diff` is correct; the deterministic-term
convention matches the critical values (this is the classic cross-package
replication trap — five conventions give five critical-value families).

**When to use (and when not).** Use to decide `r` *before* fitting a VECM, when
several series trend together and you suspect a shared equilibrium (spot/futures,
short/long rates, consumption/income). Not for a single series (that is a unit-
root test — `adf`/`kpss`), and not on series that are already stationary (fit a
VAR in levels).

**Key arguments and defaults (and why).** `data` is T×k; `k_ar_diff` is the
number of lagged differences (one less than the VAR level lag order — choose it
as you would a VAR lag length). The deterministic convention is fixed: an
**unrestricted constant** in the data (statsmodels `coint_johansen`
`det_order=0`). That is `vecm`'s `deterministic="co"` case, **not** `vecm`'s
default `"n"` — fit the VECM this test ranks with `vecm(...,
deterministic="co")`.

**How to read the output.** `trace_stat` and `max_eig_stat` (one per null
`r ≤ i`), each with critical values in `trace_crit_90_95_99` /
`max_eig_crit_90_95_99` (columns are the 90/95/99% levels — take column 1 for
the 5% test). `rank_trace_5pct` / `rank_max_eig_5pct` apply the sequential rule
for you. `eig` are the ordered eigenvalues; `evec` (k×k, S₁₁-orthonormal
columns, sign-arbitrary) holds the estimated cointegrating directions — the
first `r` columns span the space a rank-`r` `vecm(..., deterministic="co")`
fit estimates. Reject `r = 0` but not `r ≤ 1` ⇒ rank 1.

**Failure modes.** Using the wrong deterministic convention silently shifts the
critical values — and silently changes the estimated cointegrating vectors:
pairing this test with `vecm`'s `deterministic="n"` default on drifting data
gives betas that visibly disagree (the shipped regression fixture pins a
cosine of ~0.63 between the two on one drifting draw). Testing series that are
not actually I(1); the trace and max-eigenvalue tests can disagree at the
margin — report both.

**Validated against.** statsmodels `coint_johansen` (`det_order=0`,
`k_ar_diff=2`), statistics and critical values (`fixtures/coint.json`);
eigenvalues and eigenvectors on drifting cointegrated data
(`fixtures/vecm_deterministic.json`).

**References.** Johansen (1988, 1991); Engle & Granger (1987).

---

## `vecm` — vector error-correction model

**What it estimates.** Given the rank `r`, the ML estimate of the VECM: the
cointegrating vectors `beta` (the long-run equilibria — the "leashes"), the
adjustment speeds `alpha` (how fast each equation corrects a disequilibrium),
the short-run dynamics `gamma`, the deterministic coefficients — `det_coef`
for terms in the short-run equations, `det_coef_coint` for terms restricted
to the cointegration relation — the residual covariance, and the
log-likelihood.

**Assumptions.** The rank `coint_rank` is correct (take it from `johansen`);
Gaussian innovations for the ML/log-likelihood; the same deterministic
convention as the rank test — which means `deterministic="co"` whenever the
rank came from `johansen` (see below).

**When to use.** After `johansen` returns `0 < r < k`. It keeps the levels
information a differenced VAR discards, and `alpha`/`beta` are directly
interpretable — which series bear the burden of adjustment back to equilibrium.

**Key arguments and defaults (and why).** `data` (T×k), `k_ar_diff`,
`coint_rank` (from the Johansen test), `deterministic` naming the
statsmodels VECM case (all nine accepted), and `seasons`/`first_season`
(statsmodels-style centered seasonal dummies, `0` = none).

Deterministic-case guidance — "restricted" means *inside the cointegration
relation*: the term is appended to the lagged-levels block, so the
reduced-rank step estimates a **widened** cointegrating matrix and its extra
rows come back as `det_coef_coint` (statsmodels' own split); unrestricted
terms live in the short-run equations (`det_coef`):

| `deterministic` | Johansen case | Model it answers | Use when |
|---|---|---|---|
| `"n"` (default) | I | no deterministic terms at all | means/drifts truly zero (rare); the default only because it is what this function has always computed |
| `"ci"` | II | equilibrium error has a free mean; **no drift** in the data | non-drifting levels whose equilibrium is not mean-zero |
| `"co"` | III | unrestricted constant: drifting data, mean-stationary equilibrium error | **drifting data, and whenever the rank came from `johansen` (`det_order=0`)** |
| `"coli"` | IV | drift + a linear trend *inside* the relation | trending data whose equilibrium relation is trend-stationary |
| `"colo"` | V | unrestricted constant + trend | even the equilibrium error trends; the analogue of `coint_johansen(det_order=1)` |
| `"lo"`, `"li"`, `"cilo"`, `"cili"` | — | the remaining statsmodels-valid combinations | complete the grid; statsmodels forbids the same term on both sides (`"co"`+`"ci"`, `"lo"`+`"li"`) and so does `vecm` |

`seasons=s` adds `s-1` **centered** seasonal dummies to the short-run
equations (they sum to zero over a cycle, shifting the seasonal profile
without moving the level, so they combine with every case above);
`first_season` is the 0-based season of the first row.

**How to read the output.** `beta` (k×r, each column a cointegrating vector —
normalized on the first variable(s): the widened matrix
`[beta; det_coef_coint]` has the identity as its leading r×r block),
`det_coef_coint` (n_coint×r — the restricted deterministic rows of the
widened cointegrating matrix, constant row first then trend row; column `j`
completes cointegrating relation `j`: the equilibrium error is
`beta[:,j]'y + det_coef_coint[:,j]'[1; t]`; empty unless `"ci"`/`"li"` is in
the case), `alpha` (k×r adjustment speeds; a large negative entry means that
equation does most of the correcting, a near-zero entry means that variable
is weakly exogenous), `gamma` (short-run lag coefficients), `det_coef`
(k×n_det unrestricted deterministic coefficients, statsmodels column order:
constant, `seasons-1` seasonal dummies, trend — empty for `"n"`), `sigma_u`,
`llf`.

**Failure modes.** A wrong rank propagates everywhere; imposing cointegration
on series that are not cointegrated fabricates a spurious equilibrium; and the
deterministic-case trap this card exists to flag: reading `vecm`'s `"n"`
default against `johansen`'s unrestricted constant on drifting levels gives
cointegrating vectors that genuinely disagree (the shipped fixture pins a
beta cosine of ~0.63 between the two cases on one drifting draw — a field
report measured ~0.57) — that is two different models, not noise. Match the
cases before comparing.

**Validated against.** statsmodels `VECM` (ML estimation; `k_ar_diff=2`,
`coint_rank=1`, `deterministic="n"`) — `alpha`, `beta`, `gamma`, `sigma_u`,
`llf` (`fixtures/coint.json`); **every deterministic case** on
`fixtures/vecm_deterministic.json` — `"n"` and `"co"` plus the
`"co"`-reconciles-with-`johansen` / `"n"`-diverges relationship on seeded
drifting data, all nine cases (`alpha`, `beta`, `det_coef_coint`, `gamma`,
`det_coef`, `sigma_u`, `llf` at 1e-6; measured deviations ≤ ~1e-11) on
seeded *trending* data where the case choice visibly moves `beta` (the
cross-case β cosines are pinned), and two `seasons=4` fits (including a
nonzero `first_season`) on a seeded quarterly pair. The `"colo"` ↔
`coint_johansen(det_order=1)` correspondence is pinned as *asymptotic* (β
cosine ~1−6e-9 on the trending draw — statsmodels' `det_order=1` detrends
over the full sample, a different finite-sample projection), unlike the
exact `"co"` ↔ `det_order=0` identity. The pre-existing `"n"`/`"co"` paths
are additionally pinned **bit-identical** to their 0.6.0 output
(`crates/tsecon-coint/tests/vecm_bit_identity.rs`).

**References.** Johansen (1995); Lütkepohl (2005, ch. 6–7).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 400
common = np.cumsum(rng.standard_normal(n))    # one shared stochastic trend
y1 = common + rng.standard_normal(n)
y2 = common + rng.standard_normal(n)          # y1 - y2 is stationary -> rank 1
y3 = np.cumsum(rng.standard_normal(n))        # an independent I(1) series
data = np.column_stack([y1, y2, y3])

joh = tsecon.johansen(data, k_ar_diff=2)
crit5 = np.asarray(joh["trace_crit_90_95_99"])[:, 1]
print("trace:", np.round(joh["trace_stat"], 1), " 5% crit:", np.round(crit5, 1),
      " -> rank", joh["rank_trace_5pct"])

# The rank came from johansen (unrestricted constant, det_order=0), so fit
# the matching deterministic case — "co" — not the "n" default: on drifting
# data the two estimate visibly different cointegrating vectors.
fit = tsecon.vecm(data, k_ar_diff=2, coint_rank=1, deterministic="co")
print("beta :", np.round(np.asarray(fit["beta"])[:, 0], 3))   # ~[1, -1, 0]: y1 - y2
print("alpha:", np.round(np.asarray(fit["alpha"])[:, 0], 3))
print("const:", np.round(np.asarray(fit["det_coef"])[:, 0], 3))
```

---

## `ou_fit` / `spread_zscore` — Ornstein-Uhlenbeck mean reversion for spreads

**What it estimates.** The continuous-time mean-reversion law of a stationary
spread — `dX = kappa (mu − X) dt + sigma dW` — by the **exact-discretization
Gaussian MLE**: observed at step `dt`, an OU process is *exactly* the AR(1)
`X_{t+1} = c + phi X_t + eps` with `phi = e^{−kappa dt}`, `c = mu(1 − phi)`,
`Var(eps) = sigma²(1 − phi²)/(2 kappa)`, so the MLE is the closed-form AR(1)
OLS (with variance `RSS/n`) mapped back through that bijection — no iterative
optimizer, no convergence question. Delta-method standard errors for
`(kappa, mu, sigma)` come from the AR(1) information (the formulas are written
out in the crate docs, `tsecon-coint/src/ou.rs`). `spread_zscore` scores the
spread against the stationary law `N(mu, sigma²/(2 kappa))` — the entry/exit
signal of a pairs trade.

**Assumptions.** The spread is a stationary Gaussian OU process sampled at a
*fixed* step `dt` — in the pairs workflow that means cointegration has already
been established (`engle_granger` / `johansen`; the estimator will *tell* you,
via `mean_reverting = False`, when the "spread" you gave it does not revert,
but it cannot tell you the hedge ratio was wrong). Constant `kappa`, `mu`,
`sigma` over the sample; SEs are asymptotic (conditional MLE).

**When to use (and when not).** Use on the residual of a cointegrating
regression (or any spread you intend to trade) to get the half-life — the
number that decides whether the reversion is tradable at your horizon — and
the z-score bands. Not a test for cointegration (it conditions on
stationarity rather than testing it); not for irregularly-sampled data
(`dt` is fixed); and if all you want is the discrete AR(1), `arima_fit(x,
order=(1,0,0))` is the direct tool — `ou_fit` buys the continuous-time
parametrization (`kappa` per unit time, comparable across sampling
frequencies) at the cost of requiring `0 < phi`.

**Key arguments and defaults (and why).** `dt=1.0` quotes `kappa` and the
half-life in observation units; pass `dt=1/252` (daily) or `1/12` (monthly) to
quote them in years. `level=0.95` sets the half-life CI. `spread_zscore`
takes all three of `kappa`/`mu`/`sigma` (score new data against a frozen fit)
or none (fit-then-score); a partial set is refused rather than silently mixed.

**How to read the output.** `kappa`/`mu`/`sigma` with `*_se`; `half_life =
ln 2 / kappa` (expected time for a deviation to halve, in `dt` units);
`half_life_ci`; `stationary_sd = sigma/sqrt(2 kappa)`; the honest flag
`mean_reverting`; and the AR(1) leg (`phi`, `phi_se`, `c`, `c_se`, `eta2`,
`loglik`, `n_obs`) so the discrete fit is never hidden behind the mapping.
When `phi_hat >= 1` the result is **returned, not raised**: the AR(1) root at
or over unity is how a non-cointegrated "spread" announces itself, so you get
`mean_reverting=False`, `half_life=inf`, `half_life_ci=None`,
`stationary_sd=None` — and `spread_zscore` refuses such a fit (no stationary
distribution exists to score against). `phi_hat <= 0` (anti-persistent at
this sampling interval — no real `kappa` exists) is the one genuine refusal.

**The kappa bias — documented, not hidden.** The AR(1) slope is biased down
(`E[phi_hat] − phi ≈ −(1+3phi)/n`, Kendall 1954), which maps to an **upward**
bias in `kappa_hat` of roughly `(1+3phi)/(n phi dt) ≈ 4 / (time span)` for
persistent spreads (Tang & Chen 2009; Yu 2012): five years of data biases
`kappa_hat` up by ~0.8/year *regardless of sampling frequency* — only a longer
span shrinks it. Measured on the shipped seeded Monte Carlo (2000 reps/cell,
`docs/examples/coverage/experiments/ou_kappa_bias_coverage.py`, DGP `mu=0`,
`sigma=0.2`), together with the coverage of the shipped 95% half-life CI and
of the log-scale alternative:

| cell | true kappa | bias (measured) | bias (≈4/span) | RMSE | CI coverage (shipped) | log-scale alt. |
|---|---|---|---|---|---|---|
| daily, 5y span | 5.0 | +0.82 | +0.80 | 1.84 | 0.939 | 0.892 |
| daily, 5y span | 2.0 | +0.91 | +0.80 | 1.57 | 0.911 | 0.797 |
| daily, 5y span | 0.5 | +1.08 | +0.80 | 1.50 | 0.820 | 0.527 |
| daily, 5y span | 0.1 | +1.10 | +0.80 | 1.43 | 0.713 | 0.210 |
| monthly, 20y span | 5.0 | +0.26 | +0.23 | 0.99 | 0.952 | 0.943 |
| monthly, 20y span | 2.0 | +0.24 | +0.21 | 0.61 | 0.948 | 0.898 |
| monthly, 20y span | 0.5 | +0.25 | +0.20 | 0.42 | 0.906 | 0.785 |
| monthly, 20y span | 0.1 | +0.28 | +0.20 | 0.38 | 0.804 | 0.467 |

**The half-life CI is level-scale — a measured choice.** `half_life_ci` maps
the symmetric kappa interval `kappa_hat ± z·SE` through the monotone
`ln2/kappa`; when that interval crosses zero the upper endpoint is reported as
`inf` — the data cannot rule out *no mean reversion* at that confidence, and
the interval says so instead of fabricating a finite bound. The a-priori
argument favored a log-scale interval (positive by construction), and the
table above is why it ships level-scale instead: `kappa_hat` centers *above*
the truth, and a multiplicative interval around an upward-biased center never
reaches down to a small true `kappa` (0.21 coverage at `kappa=0.1`), while the
level interval — precisely by conceding the `inf` branch — covers closer to
nominal in **every** cell. Neither attains nominal in the slow-reversion
cells; that residual under-coverage is the bias itself, and the table is the
honest statement of it. Read a wide or infinite `half_life_ci` as "this span
does not identify the reversion speed", not as noise to be tuned away.

**Failure modes.** Short spans with slow reversion: `kappa_hat` can exceed
its truth several-fold (see `daily_weak` in the fixture: `kappa_hat = 5.7` on
a true 0.3 in one year of daily data) — the CI's `inf` branch is the guard.
Scoring `spread_zscore` with a `kappa` from a different `dt` convention than
the data quietly rescales nothing (the z-score is `dt`-free once the
parameters are consistent) but the *half-life* is only comparable across
frequencies if `dt` was passed correctly. The stationary-law z-score uses
`sigma/sqrt(2 kappa)`, not the sample standard deviation of the spread — on a
finite sample of a slowly-reverting spread the two differ materially, and the
sample sd underestimates the stationary sd.

**Validated against.** Grade: **closed-form + statsmodels AR(1) golden +
MC-measured kappa bias and CI coverage.** The AR(1) leg (`c`, `phi`, `eta2`,
both SEs, loglik) is pinned to statsmodels `AutoReg(x, lags=1)` — the same
estimator through an independent lstsq path — at 1e-10 in the Rust golden
(`fixtures/ou.json`) and 1e-12 live in the Python suite (achieved ~1e-15);
the closed-form mapping is asserted **bit-for-bit** against a
summation-order-identical reimplementation in the crate test, and at 1e-10
against the NumPy transcription in the fixture; `half_life·kappa = ln 2`,
`e^{−kappa·dt} = phi`, and the stationary-variance identity are asserted at
float round-off; the kappa bias and both CI constructions are MC-measured
(table above, 2000 reps, seeded).

**References.** Uhlenbeck & Ornstein (1930); Vasicek (1977); Kendall (1954,
Biometrika 41); Tang & Chen (2009, J. Econometrics 149); Yu (2012, J.
Econometrics 169).

```python
import numpy as np, tsecon
rng = np.random.default_rng(7)
# A cointegrated pair with a *persistent* spread: each price is a shared
# random walk plus an AR(1) mispricing (phi = 0.93 daily -> the spread
# mean-reverts with a half-life of about ln 2 / (252 ln(1/0.93)) ~ 10 days).
common = np.cumsum(rng.standard_normal(1000))
e1 = np.zeros(1000); e2 = np.zeros(1000)
for t in range(1, 1000):
    e1[t] = 0.93 * e1[t-1] + 0.3 * rng.standard_normal()
    e2[t] = 0.93 * e2[t-1] + 0.3 * rng.standard_normal()
y1, y2 = common + e1, common + e2

eg = tsecon.engle_granger(np.column_stack([y1, y2]), trend="c")
print(f"EG p = {eg['pvalue']:.4f}")            # cointegrated -> small
spread = eg["resid"]                            # the tradable spread

fit = tsecon.ou_fit(spread, dt=1/252)           # daily data, kappa per year
print(f"kappa = {fit['kappa']:.1f}/yr  half-life = {fit['half_life']*252:.1f} days",
      f" CI(days) = {tuple(round(v*252, 1) for v in fit['half_life_ci'])}")
print(f"mean-reverting: {fit['mean_reverting']}")

z = tsecon.spread_zscore(spread, dt=1/252)      # fit-then-score
print("today's z:", round(z["zscore"][-1], 2))  # |z| > 2: stretched
```

---

## `markov_switching_ar` — Markov-switching AR

**What it estimates.** A Hamilton (1989) regime-switching autoregression: an
AR(p) whose mean (and optionally variance) jumps between `k_regimes` hidden
states, with the state following a first-order Markov chain. Fit by EM; returns
the transition matrix, per-regime parameters, and filtered/smoothed regime
probabilities.

**Assumptions.** The regimes are discrete and Markovian; the number of regimes
`k_regimes` is chosen a priori (the likelihood-ratio test for `k` is
non-standard — do not read the LR p-value naively); Gaussian innovations within
regime.

**When to use (and when not).** Use when a series plausibly alternates between
persistent states with different means/volatilities — business-cycle expansions
and recessions, low- and high-volatility markets. Not for smooth nonlinearity
(use STAR/threshold models) or when the "regimes" are really an omitted
covariate you could just include.

**Key arguments and defaults (and why).** `k_regimes=2` (the workhorse),
`order` (AR lag order), `switching_variance=True` lets volatility differ across
regimes (usually essential — regimes often *are* volatility states),
`max_iter`/`tol` govern EM convergence.

**How to read the output.** `transition` is the k×k Markov matrix in the
**column-stochastic** orientation: `transition[i][j] = P(S_t = i | S_{t-1} = j)`,
so each **column** sums to 1 (matching statsmodels' `regime_transition`, *not*
the row-stochastic textbook convention). The one-step forward propagation of a
probability vector `p` over regimes is therefore `P @ p` — **not** `p @ P`;
transposing by habit silently swaps the entry/exit probabilities. Also
`means`, `variances` (per regime), `ar` — the estimated AR coefficients
`(phi_1, …, phi_p)`, a length-`order` array **shared across regimes** (the
binding fits Hamilton's common-AR specification, in which the AR applies to
deviations `y_t − mu_{S_t}`; with `means`, `transition`, and `variances` it
reproduces and forecasts the fitted model) — `expected_durations` (average spell length
in each regime — the persistence read, `1 / (1 - transition[i][i])`),
`loglik`, `converged`, the full probability matrices `smoothed_prob`
(Kim 1994, `P(S_t | Y_T)`) and `filtered_prob` (Hamilton filter,
`P(S_t | Y_t)`) — each `(n, k_regimes)` with `n = len(y) - order`, rows
summing to 1 — and the `regimes` series (the most-likely regime per period,
the argmax of each smoothed row). One timing warning: `smoothed_prob`
conditions on the **full sample** (Kim 1994 runs backward from `T`), so it
must not be used for real-time regime dating — a "recession probability for
March" computed with December's data in hand is not a real-time call;
`filtered_prob` is the real-time object. `smoothed_prob_last_regime` is
`smoothed_prob[:, -1]`, kept because 0.2.0 returned only that column
(recoverable at `k_regimes = 2` as `1 - p`, not at `k_regimes >= 3`). Label
regimes by their `means`/`variances`, not their index (EM does not order
them).

**Failure modes.** EM converges to local optima — try multiple starts; regime
labels are arbitrary across runs; too many regimes on a short sample gives empty
or degenerate states.

**Validated against.** statsmodels `MarkovAutoregression` (`k_regimes=2`,
`order=1`, `switching_variance=True`) — fixed-parameter log-likelihood and
filtered/smoothed regime probabilities (`fixtures/regime.json`).

**References.** Hamilton (1989); Kim & Nelson (1999).

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
y = np.concatenate([0.5 + 0.3 * rng.standard_normal(150),    # calm regime
                    -0.5 + 1.2 * rng.standard_normal(150),   # volatile regime
                    0.5 + 0.3 * rng.standard_normal(150)])
ms = tsecon.markov_switching_ar(y, k_regimes=2, order=1, switching_variance=True)
print("regime means    :", np.round(ms["means"], 3))
print("regime variances:", np.round(ms["variances"], 3))
print("expected durations:", np.round(ms["expected_durations"], 1))
```

---

## `setar` — self-exciting threshold autoregression

**What it estimates.** A two-regime SETAR(p) (Tong & Lim 1980): an AR(p) whose
coefficients switch when the *observed* lagged value `y_{t-d}` crosses a
threshold `γ`. Fit by concentrated least squares (Hansen 1997): for each
candidate threshold — the order statistics of `y_{t-d}` with a `trim` fraction
excluded at each end — OLS in each regime; the threshold (and, when `delays`
is a list, the delay) minimizes the pooled SSR. The workhorse observable-regime
nonlinear benchmark (sunspots, unemployment asymmetry, floor/ceiling dynamics).

**Assumptions.** Two regimes with an abrupt switch on a *lagged own value*
(smooth transitions want STAR; switching on an unobserved state wants
`markov_switching_ar`); iid errors within regime for the classical SEs; the
threshold variable visits both sides of `γ` often enough (trimming plus the
`k + 1` per-regime minimum enforce this mechanically, not statistically).

**When to use (and when not).** Use when the *level* of the series itself
plausibly triggers the regime — asymmetry over the cycle, floor/ceiling
dynamics — and you want interpretable per-regime dynamics. Not for
volatility-driven or unobserved regimes, and not before checking a threshold
exists: run `setar_test` first — the split fit *always* lowers the SSR, so an
unvalidated SETAR fit on linear data will happily report two regimes.

**Key arguments and defaults (and why).** `p` (AR order, per regime);
`delay=1` (the standard first try); `delays=[1, 2, 3]` searches the delay
jointly with the threshold — all candidates then share the common sample
`t ≥ max(p, max(delays))` so pooled SSRs are comparable; `trim=0.15` (Hansen's
15% trimming — each regime keeps at least 15% of the sample, and never fewer
than `k + 1` observations); `constant=True`; `ic="aic"|"bic"` selects which
criterion is *reported* under the `ic` key (with `p` fixed, both rank
candidates exactly as the SSR does, so the fit itself never depends on it).

**How to read the output.** `threshold`, `delay`; `params_low`/`params_high`
(constant first, then lags 1..p) with classical `bse_low`/`bse_high`;
`n_low`/`n_high` (regime occupancy — a tiny regime means the threshold sits in
a data-sparse corner even after trimming); pooled `ssr` and
`sigma2 = SSR/(n−2k)`, per-regime `sigma2_low`/`sigma2_high`; `aic`/`bic`
(`n·ln(SSR/n) + penalty·m`, `m = 2k+1` counting the threshold); the full
candidate grid `thresholds` with its `ssr_path` — plot it: a sharp V says the
threshold is well identified, a flat valley says it is not.

**Failure modes.** Fitting SETAR to linear data "finds" a threshold (use
`setar_test`); the threshold is superconsistent but its *sampling* distribution
is nonstandard — the reported SEs are for the regression coefficients, not
`γ`; near-empty regimes make per-regime SEs meaningless; delay search over
many candidates on short samples overfits.

**Validated against.** No third-party SETAR exists in the test venv (no R
`tsDyn`), so the golden is an *independent NumPy transcription of the published
algorithm* — explicit regime-split design matrices, per-regime `lstsq`, the SSR
profile over the candidate grid (Hansen 1997 notation; the generator header
states this honestly): threshold, per-regime coefficients and SEs, SSRs,
variances, and ICs pinned at 1e-10 over six cases including delay search and a
no-constant fit (`fixtures/setar.json`). Statistical correctness is established
by seeded Monte Carlo (`setar_properties.rs`): over 200 replications of a
two-regime DGP (`y = 1.0 + 0.6y₋₁` below 0, `−1.0 + 0.2y₋₁` above, T = 400),
the threshold's median absolute error is **0.008** and per-coefficient biases
are **|bias| ≤ 0.012** (c_low +0.012, φ_low +0.007, c_high −0.010, φ_high
+0.008); on data where both regimes are identical the regime fits reproduce the
plain AR OLS fit, and coefficients/threshold are scale/location-equivariant.

**References.** Tong & Lim (1980); Hansen (1997, 2000); Tong (1990).

---

## `setar_test` — Hansen (1996) bootstrap linearity test

**What it estimates.** Whether a threshold exists at all: sup-F =
`n·(S0 − S1)/S1` (S0 the linear-AR SSR, S1 the SETAR SSR at the concentrated
optimum), with a p-value from Hansen's fixed-regressor wild bootstrap. Run it
*before* interpreting any `setar` fit.

**Assumptions.** The null is a linear AR(p) with a constant; the alternative a
two-regime SETAR at the given `delay`. The bootstrap conditions on the
regressors and reweights the null residuals (`y* = ê·η`, `η` iid N(0,1)), so it
is robust to heteroskedasticity of the errors.

**When to use (and when not).** Whenever a SETAR (or any regime story keyed to
an observed lag) is on the table. Never replace the bootstrap p-value with a
chi-squared tail: the threshold is unidentified under the null (the Davies
problem), so the sup-F statistic does *not* have a chi-squared distribution —
the library refuses to report one by design.

**Key arguments and defaults (and why).** `n_boot=499` (odd B makes
`(B+1)·α` an integer at the usual levels; the p-value lattice is
`{1/(B+1), ..., 1}`); `seed=0` — the bootstrap is embarrassingly parallel
(rayon) and bit-identical for a given seed at any thread count, so results
replicate exactly across machines.

**How to read the output.** `stat`, `p_value` (small ⇒ reject linearity ⇒ a
threshold model is warranted), `threshold` (where the sup is attained — a
preview of `setar`'s estimate), `f_path` over `thresholds` (the pointwise F
profile), `boot_stats` (the null distribution actually used — histogram it
against `stat`).

**Failure modes.** Too few bootstrap draws make the p-value lattice coarse
(with B = 99 the smallest possible p is 0.01); rejecting linearity does not
choose *which* nonlinear model — STAR or Markov-switching may fit better;
power falls when the true delay is not the one tested (search delays in
`setar` but test at the chosen delay honestly: pre-testing distorts size).

**Validated against.** The sup-F statistic is pinned at 1e-10 against the
independent NumPy transcription (four cases, `fixtures/setar.json`); the
bootstrap p-value is validated by *property*, not fixture: over 200 seeded
linear-AR series (T = 100, B = 199), the test rejects at rate **0.08 at the 5%
level** and **0.11 at the 10% level** with mean p-value **0.50** (approximately
uniform); on a strongly separated SETAR (T = 500) it rejects with p ≤ 0.01.
Determinism at any thread count is asserted by test.

**References.** Hansen (1996, Econometrica); Hansen (1997); Davies (1987).

```python
import numpy as np, tsecon
rng = np.random.default_rng(1)
# Simulate a two-regime SETAR: mean-reverting pushes across the threshold 0.
y = np.zeros(400)
for t in range(1, 400):
    if y[t-1] <= 0.0:
        y[t] = 1.0 + 0.6 * y[t-1] + rng.standard_normal()
    else:
        y[t] = -1.0 + 0.2 * y[t-1] + rng.standard_normal()

lin = tsecon.setar_test(y, p=1, delay=1, n_boot=499, seed=0)
print(f"sup-F = {lin['stat']:.1f}, bootstrap p = {lin['p_value']:.3f}")

if lin["p_value"] < 0.05:
    fit = tsecon.setar(y, p=1, delays=[1, 2], trim=0.15)
    print("threshold:", round(fit["threshold"], 3), " delay:", fit["delay"])
    print("low  regime:", np.round(fit["params_low"], 2), " n =", fit["n_low"])
    print("high regime:", np.round(fit["params_high"], 2), " n =", fit["n_high"])
```

---

## `star` — smooth-transition autoregression (LSTAR / ESTAR)

**What it estimates.** A two-regime STAR(p) (Teräsvirta 1994): an AR(p) whose
coefficients move *smoothly* between two extremes as the observed lagged value
`s_t = y_{t−d}` crosses a location `c`,

```
y_t = φ₁'x_t + G(γ, c; s_t)·φ₂'x_t + ε_t,   x_t = (1, y_{t−1}, …, y_{t−p})',
```

with `G = 1/(1+exp(−γ(s−c)))` (`model="lstar"`: regimes differ by the *level*
of `s_t` — expansions vs. recessions) or `G = 1 − exp(−γ(s−c)²)`
(`model="estar"`: regimes differ by the *distance* from `c`, symmetric — the
classic real-exchange-rate / transaction-band shape). SETAR is the `γ → ∞`
limit of LSTAR. Estimation is concentrated NLS: for fixed `(γ, c)` the model
is OLS in `(φ₁, φ₂)`, so a grid over `(γ, c)` locates the basin and
Nelder-Mead refines the best cell; `star_eval` exposes the same concentrated
fit at *fixed* `(γ, c)` for scoring a published parameterization
(SSR/log-likelihood comparison is robust to optimizer differences;
parameter-level comparison is not — the auto_arima precedent).

**Gamma-scaling convention (read this before comparing packages).** The
reported `gamma` is **raw** — the value inside the transition function — which
is R `tsDyn::lstar`'s convention (its sigmoid is
`plogis(s, location=th, scale=1/gamma)`, no standardization). Teräsvirta
(1994) instead standardizes the exponent by `sd(s)` (LSTAR) or `var(s)`
(ESTAR) so that γ is scale-free; that value is reported as
`gamma_standardized` (= `gamma·s_sd` for LSTAR, `gamma·s_sd²` for ESTAR,
population sd over the usable sample). The internal grid is built in
*standardized* units — log-spaced over [0.5, 100] — so the search is
scale-equivariant, then mapped back to raw γ.

**Assumptions.** Two extreme regimes with a *smooth, monotone (LSTAR) or
symmetric (ESTAR)* transition on a lagged own value; iid errors for the
Gauss-Newton SEs; `s_t` must actually vary (a near-constant transition
variable is refused) and visit both sides of `c` (trimming of the `c` grid
enforces this mechanically).

**When to use (and when not).** Use when theory says adjustment is gradual —
aggregation over heterogeneous agents, adjustment costs, transaction bands —
and the regime trigger is an observed lag. Prefer `setar` when the switch is
genuinely abrupt (and note the SSR surface often *prefers* the abrupt limit in
small samples: see `gamma_at_boundary` below). Not for unobserved-state
switching (`markov_switching_ar`) and not before `star_test` — the STAR fit on
linear data will happily report a transition.

**Key arguments and defaults (and why).** `p`; `model="lstar"`; `delay=1`,
or `delays=[1, 2, 3]` to search the delay by refined SSR on the common sample
`t ≥ max(p, max(delays))`; `trim=0.15` (the `c` grid spans the 15%–85% order
statistics of `s_t`); `constant=True`; `n_gamma=25`, `n_c=25` (the grid —
625 concentrated OLS fits — is cheap and the refinement polishes the best
cell, so finer grids buy little).

**How to read the output.** `gamma`, `gamma_standardized`, `c`, `delay`;
`params_linear` (φ₁ — the `G = 0` regime) and `params_nonlinear` (φ₂ — the
*difference*; the `G = 1` regime is φ₁+φ₂) with Gauss-Newton `bse_linear` /
`bse_nonlinear` / `se_gamma` / `se_c` over all 2k+2 parameters
(`se_valid=False` with NaN SEs when `J'J` degenerates — typically at huge γ,
where the SSR surface carries no curvature in γ; conditional-on-(γ,c) OLS SEs
would *understate* uncertainty, so the library reports the honest NaN
instead); `transition` (the fitted `G_t` path — plot it: a path stuck at 0/1
is a threshold model in disguise); the grid surface (`grid_gamma`, `grid_c`,
`ssr_grid`, `best_cell`) — a flat valley in the γ direction is the visual of
weak γ identification; `converged` (Nelder-Mead verdict) and
**`gamma_at_boundary`** — True when standardized γ ends at the top (≥ 100:
numerically a hard threshold; read γ as a *lower bound*, the Teräsvirta
large-γ advice, and consider `setar`) or pinned at the bottom wall (0.5:
numerically linear in `s`; γ and φ₂ are separately unidentified — only their
product is — so read γ as an *upper bound* and take the φ₂ block with a grain
of salt).

**Failure modes.** γ is the notoriously hard parameter: its likelihood is
flat for large values (accurate estimation needs many observations *near*
`c` — Teräsvirta 1994), so estimates routinely run to a bound; that is what
the flag is for, and it mirrors tsDyn's routine "gamma reached its bound"
warning rather than pretending precision. On boundary draws the φ₂ block is
attenuated (the step approximation mixes transition-zone observations), so
mean bias/RMSE tables over MC replications look alarming while medians over
identified fits are clean — see the numbers below. `c` and γ trade off when
the transition is smooth, spreading `c`. ESTAR at large γ *and* ESTAR at
tiny γ both degenerate (to inner/outer indicators and to quadratic drift);
the same flag covers both edges.

**Validated against.** No third-party STAR is reachable from the test
environment (R/tsDyn needs CRAN, which the sandbox egress policy denies —
r-base installed but tsDyn's dependency tree is unbuildable offline;
statsmodels has no STAR), so the golden is an *independent NumPy/SciPy
transcription of the published closed forms* (`fixtures/star.json`, generator
header states the grading honestly): the concentrated OLS at fixed `(γ, c)`
with Gauss-Newton SEs, log-likelihood and ICs, the transition path, and the
full `(γ, c)` grid surface (including the c-grid order-statistic and γ-scaling
conventions) pinned at 1e-10 over six eval cases and three grid cases (LSTAR
and ESTAR, with and without constant, `d > p`). The Nelder-Mead refinement is
deliberately *not* pinned (optimizer-dependent); properties assert refined
SSR ≤ grid SSR and `star_eval(fit) == fit`. Statistical correctness by seeded
MC (`star_properties.rs`, 200 reps of the LSTAR DGP φ₁=(1, 0.6),
φ₂=(−2, −0.4), γ_std ≈ 2.9, c = 0): **medians over identified
(non-boundary) fits** at T = 250: (−0.07, +0.00, −0.26, +0.29) over 103/200
fits; at T = 500: (+0.04, +0.00, −0.28, +0.04) over 143/200; `c` median error
−0.06 (T=250) / −0.01 (T=500) with median |c| 0.46 / 0.32; **standardized γ
median 35.9 [IQR 2.3, 1000] at T = 250 collapsing to 2.73 [IQR 1.6, 24.7] at
T = 500 (truth ≈ 2.9)** — γ at a boundary in 97/200 resp. 57/200 fits;
convergence 199/200 resp. 200/200. The **LSTAR → SETAR limit** is asserted
against the test's *own* split-OLS transcription (never against
`tsecon.setar` — no circularity): at γ = 10⁶ with `c` between two order
statistics, the concentrated fit equals the hard-threshold two-regime OLS to
1e-7 and `se_valid` flips to False. The grid stage is exactly
scale/location-equivariant by test; hard-threshold data trips
`gamma_at_boundary = True`, smooth data leaves it False.

**References.** Teräsvirta (1994, JASA); Luukkonen, Saikkonen & Teräsvirta
(1988, Biometrika); van Dijk, Teräsvirta & Franses (2002, Econometric
Reviews); Franses & van Dijk (2000), ch. 3.

---

## `star_test` — Teräsvirta modeling-cycle battery (LM3 + H-sequence)

**What it estimates.** The two specification questions of the STAR modeling
cycle, both answered by *closed-form auxiliary regressions* (unlike
`setar_test`, no bootstrap is needed — the auxiliary regression is linear, so
there is no Davies problem and the null distributions are standard):

1. **Is there STAR-type nonlinearity at all?** The LM3 test
   (Luukkonen-Saikkonen-Teräsvirta 1988): regress `y_t` on
   `[w_t, x̃s_t, x̃s_t², x̃s_t³]` (`w` the null AR design, `x̃` the lag block,
   both augmented with `y_{t−d}` when `d > p` — Teräsvirta's redefinition) and
   test the 3q interaction coefficients. `lm3_stat` is the χ² form
   `n(SSR0−SSR3)/SSR0` (df = 3q); `lm3_f_stat` the **F form, recommended in
   small samples** (the χ² form over-rejects; the F form holds size).
2. **LSTAR or ESTAR?** The nested H-sequence: H03 (`s³` block = 0), H02
   (`s²` | no cubic), H01 (`s` | neither). Decision rule: `suggested="estar"`
   iff the H02 p-value is strictly the smallest — the even terms carry ESTAR's
   symmetric transition; odd terms carry LSTAR's. Only meaningful when LM3
   rejects.

`delays` runs the battery per candidate delay (each on its own usable sample)
and `best` marks the smallest F-form LM3 p-value — Teräsvirta's rule for
choosing `d`.

**When to use (and when not).** Always before `star`, and as the cheap
first-line nonlinearity screen even when `setar` is the goal (LM3 has power
against threshold alternatives too). The H-sequence is a heuristic: it
misfires in appreciable finite-sample fractions (see the numbers), so treat
`suggested` as a tiebreak, not a verdict — when in doubt fit both and compare
`aic`/out-of-sample.

**How to read the output.** Top level = the selected delay's battery:
`lm3_f_stat`/`lm3_f_p_value` (use these; the χ² pair is reported for
completeness), `h1_*`/`h2_*`/`h3_*`, the SSR ladder `ssr0..ssr3`, `q`, `k0`,
`suggested`; plus `tests` (all candidate delays) and `best`.

**Failure modes.** Heteroskedastic errors inflate LM-type linearity tests
(GARCH masquerades as STAR — check `arch_test` first); the cubic block's `y³`
terms are heavy-tailed in small samples (the F form is the mitigation); low
power when the tested `d` is wrong (search `delays`); the H-sequence's
LSTAR/ESTAR split is fragile when both even and odd terms are strong.

**Validated against.** Transcription golden: every statistic, p-value, SSR,
and the suggested-model verdict pinned at 1e-10 against the independent
NumPy/SciPy implementation over five case families including `d > p`
augmentation and delay selection (`fixtures/star.json`). Statistical
properties by seeded MC (400 reps): **size** of the F form under an AR(1)
null — 0.060 at the 5% level (MC se 0.011) and 0.100 at 10% at T = 200;
0.028 / 0.065 at T = 500 (slightly conservative, the documented small-sample
behavior of the F form; mean null p-value 0.51/0.53); **power** at T = 250 —
0.81 (se 0.02) against the LSTAR DGP above, 0.91 against an ESTAR
random-walk-band DGP; **selection given rejection** — ESTAR chosen 98% of the
time on the ESTAR DGP, LSTAR 55% on the LSTAR DGP (the known asymmetry of the
sequence: strong even terms appear in both families).

**References.** Luukkonen, Saikkonen & Teräsvirta (1988); Teräsvirta (1994);
Escribano & Jordá (2001) for an alternative selection rule (not implemented).
## `threshold_vecm` — Hansen-Seo (2002) threshold cointegration

**What it estimates.** A two-regime threshold VECM: the series share one
long-run equilibrium `w_t = beta'y_t`, but the error-correction dynamics
switch when the *equilibrium error itself* crosses a threshold —
`Δy_t = A₁'X_{t−1} 1{w_{t−1} ≤ γ} + A₂'X_{t−1} 1{w_{t−1} > γ} + u_t` with
`X_{t−1} = (1, w_{t−1}, Δy lags)`. Estimation is Hansen & Seo's concentrated
Gaussian MLE: grid search over `(beta, γ)` with per-cell two-regime OLS,
minimizing `ln det` of the pooled residual covariance. The canonical story is
transaction-cost arbitrage: inside the band the spread wanders (weak
correction), outside it snaps back.

**Assumptions.** Exactly **one** cointegrating relation (rank 1 — the
Hansen-Seo setting; take the rank from `johansen` first); two regimes with an
abrupt switch on the *lagged equilibrium error* (regimes keyed to an outside
variable want `threshold_var`; smooth transitions are out of scope); an
unrestricted constant per regime; iid-within-regime errors for the concentrated
MLE (the reported Eicker-White SEs tolerate heteroskedasticity).

**When to use (and when not).** After cointegration is established
(`johansen`/`engle_granger`) and `hansen_seo_test` rejects linear adjustment —
never as a first look: the split fit *always* improves the criterion, so an
unvalidated threshold fit on linearly-cointegrated data will happily report two
regimes. Not for more than two regimes, and not for `k > 2` unless you supply
`beta` (the (k−1)-dimensional grid is not searched — the error says exactly
this).

**Key arguments and defaults (and why).** `k_ar_diff=1` lagged differences;
`trim=0.05` — Hansen-Seo's π₀, *their* suggested value (each regime keeps at
least `max(m+1, ceil(0.05·n))` observations; note this is looser than SETAR's
0.15 default because the paper's own applications run at 0.05);
`n_grid_gamma=300` (the paper's grid resolution) evenly-spaced feasible order
statistics of `w_{t−1}`; `beta=None` estimates the cointegrating vector on a
bivariate grid of `n_grid_beta=50` points spanning the linear Johansen ML
estimate ± `beta_span=10` first-order standard errors (a *search region*
centered on the consistent linear estimate, not an inference interval);
`beta=[1, ...]` fixes it (normalized so `beta[0] = 1` — order the series with
the normalized one first).

**How to read the output.** `beta` and `threshold` locate the regime split;
`ect` is `w_{t−1}` itself (threshold it to see which periods sat in which
regime); `params_low`/`params_high` (rows = equations, columns `[const, ect,
Δy lags]`) — the `ect` column is the error-correction speed per regime, and
the typical threshold-cointegration finding is a near-zero low-regime loading
with a strong high-regime one; `frac_low` flags a threshold parked in a
data-sparse corner; `llf` vs `llf_linear` shows the (always nonnegative)
improvement over the linear VECM; `beta_grid` is the region actually searched
— an estimate pinned at its edge means widen `beta_span`.

**Failure modes.** Fitting on linearly-cointegrated data "finds" a threshold
(run `hansen_seo_test` first); the threshold's own sampling distribution is
nonstandard — the SEs are for coefficients, not `γ` or `beta`; a `beta` grid
too narrow silently truncates the estimate to its edge; near-empty regimes
make the per-regime SEs meaningless; with `k > 2` and a *wrong* fixed `beta`
the "equilibrium error" is not stationary and the regimes are fiction.

**Validated how (honest grade).** **No reference implementation was runnable**:
R installs in the build container but CRAN is unreachable through its egress
proxy, so `tsDyn::TVECM` could not be run (and no Python package implements
the estimator). The golden is therefore an *independent NumPy transcription of
the published algorithm* (`fixtures/generate_tvecm_fixtures.py`, header states
the grade): fixed-`beta` cases pinned at **1e-10** (threshold, per-regime
coefficients, Eicker-White SEs, Σ, criterion, llf), estimated-`beta` cases at
**1e-8** (the Johansen eigensolver enters), four cases including k = 3 and the
grid-subsample rule. Statistical correctness by seeded Monte Carlo
(`tvecm_properties.rs`): over 200 replications of a threshold-cointegrated DGP
(γ = 0, β = (1, −1), regimes `[1.0, 0.7]`/`[−1.0, 0.3]`, T = 300, `beta`
estimated), median |γ̂ − γ| = **0.025** and median |β̂₂ − β₂| = **0.0046**;
the two-regime fit provably nests the linear fit (`llf ≥ llf_linear`
asserted), and the reported split reproduces direct OLS exactly. Point
estimates against tsDyn would in any case only match at grid resolution —
tsDyn's default γ/β grids are coarser (50×50) and its `beta` region is
constructed differently.

**References.** Hansen & Seo (2002, J. Econometrics 110); Balke & Fomby
(1997, IER — threshold cointegration); Johansen (1995).

---

## `hansen_seo_test` — sup-LM test for threshold cointegration

**What it estimates.** Whether the error-correction dynamics really switch:
H₀ *linear* cointegration (`A₁ = A₂`) against the two-regime threshold VECM,
with `beta` fixed at the null (linear Johansen ML) estimate as the paper
prescribes. The pointwise statistic is the coefficient-difference quadratic
form with Eicker-White covariance (their eq. 10-12), maximized over the
trimmed grid of `w̃_{t−1}` order statistics; the p-value comes from their
Section-4 **fixed-regressor bootstrap** (Hansen 1996): regressors, threshold
variable, and null residuals held fixed, `y* = ũ·η` with scalar `η ~ N(0,1)`,
re-residualized, same sup over the same grid.

**Assumptions.** The series ARE cointegrated under *both* hypotheses — this
tests linear vs threshold **adjustment**, not cointegration vs none (test
cointegration first; on non-cointegrated inputs the "equilibrium error" is
spurious and the verdict meaningless). Rank 1; the bootstrap tolerates
heteroskedastic errors (the statistic is Eicker-White weighted).

**When to use (and when not).** Between the rank decision and any
`threshold_vecm` interpretation. Never read the sup-LM against a chi-squared
table: `γ` is unidentified under the null (the Davies problem) — the library
deliberately reports no asymptotic p-value.

**Key arguments and defaults (and why).** `trim=0.05` (π₀), `n_grid=300`
(Hansen-Seo's resolution), `n_boot=499` (odd B keeps `(B+1)·α` integral),
`seed=0` — one Philox substream per replication via the library's shared
`par_replicate` engine (the same contract as `setar_test`), so the p-value is
bit-identical at any thread count. `beta=` runs their known-cointegrating-
vector variant.

**How to read the output.** `stat`, `p_value` (small ⇒ threshold
cointegration), `threshold` (where the sup lands — a preview of
`threshold_vecm`'s estimate), `lm_path` over `thresholds` (plot it),
`boot_stats` (the bootstrap null actually used).

**Failure modes.** **Small-sample over-rejection, measured and documented**:
at T = 150 the seeded null Monte Carlo rejects at **0.100** at the 5% level
(see below) — in short samples treat a marginal rejection (p ∈ [0.01, 0.10])
with suspicion and prefer a larger `trim`; fixing `beta` at the null estimate
ignores its (superconsistent, hence small) estimation error; a non-rank-1
system violates the setting; power falls if the true threshold sits inside
the trimmed tails.

**Validated how (honest grade).** Statistic and per-candidate path pinned
against the independent NumPy transcription (fixed-`beta` cases 1e-10,
estimated-`beta` 1e-8; `fixtures/tvecm.json`) — the same no-runnable-reference
caveat as `threshold_vecm`. The bootstrap p-value is validated by *property*
(`tvecm_properties.rs`): over 200 seeded linearly-cointegrated draws
(B = 199, `beta` re-estimated each draw), rejection rates are **0.100 at the
5% level** and **0.160 at 10%** at T = 150 (MC se ≈ 0.02), falling to
**0.065** and **0.120** at T = 400 — mildly liberal in small samples, in the
direction Hansen-Seo's own simulations show, converging toward nominal; mean
null p-value 0.44/0.47. On a strongly threshold-cointegrated DGP it rejects
with p ≤ 0.01. Determinism at any thread count is asserted by test.

**References.** Hansen & Seo (2002); Hansen (1996); Davies (1987); Seo
(2006, J. Econometrics — tests against SETAR cointegration).

```python
import numpy as np, tsecon
rng = np.random.default_rng(2)
# Simulate an LSTAR: smooth mean-reversion flip around 0.
y = np.zeros(500)
for t in range(1, 500):
    G = 1.0 / (1.0 + np.exp(-2.0 * y[t-1]))
    y[t] = 1.0 + 0.6*y[t-1] + G*(-2.0 - 0.4*y[t-1]) + rng.standard_normal()

# The Teräsvirta cycle: linearity -> family -> fit -> flags.
battery = tsecon.star_test(y, p=1, delays=[1, 2])
print(f"LM3-F p = {battery['lm3_f_p_value']:.4f} at d = {battery['delay']}, "
      f"suggested: {battery['suggested']}")

if battery["lm3_f_p_value"] < 0.05:
    fit = tsecon.star(y, p=1, model=battery["suggested"], delay=battery["delay"])
    print("gamma (raw):", round(fit["gamma"], 2),
          " standardized:", round(fit["gamma_standardized"], 1),
          " c:", round(fit["c"], 2))
    print("G=0 regime:", np.round(fit["params_linear"], 2),
          " G=1 regime:", np.round(fit["params_linear"] + fit["params_nonlinear"], 2))
    if fit["gamma_at_boundary"]:
        print("gamma at boundary: transition is numerically a step -> "
              "compare with tsecon.setar")
# Spread corrects only when |spread| is pushed past the threshold band.
T = 400
w = np.zeros(T)                      # equilibrium error (two-regime TAR)
for t in range(1, T):
    if w[t-1] <= 0.0:
        w[t] = 1.0 + 0.7 * w[t-1] + rng.standard_normal()
    else:
        w[t] = -1.0 + 0.3 * w[t-1] + rng.standard_normal()
y2 = np.cumsum(0.5 * rng.standard_normal(T))   # common stochastic trend
data = np.column_stack([w + y2, y2])           # beta = (1, -1)

hs = tsecon.hansen_seo_test(data, k_ar_diff=1, n_boot=499, seed=0)
print(f"sup-LM = {hs['stat']:.1f}, bootstrap p = {hs['p_value']:.3f}")
if hs["p_value"] < 0.05:
    fit = tsecon.threshold_vecm(data, k_ar_diff=1)
    print("beta:", np.round(fit["beta"], 3), " gamma:", round(fit["threshold"], 3))
    print("ect loadings low :", np.round([r[1] for r in fit["params_low"]], 3))
    print("ect loadings high:", np.round([r[1] for r in fit["params_high"]], 3))
```

---

## `threshold_var` — two-regime threshold VAR

**What it estimates.** The multivariate SETAR: a VAR(p) whose *entire*
coefficient matrix switches when the delay-`d` lag of one chosen series
(`threshold_index`) crosses a threshold —
`y_t = A₁'x_t 1{z_t ≤ γ} + A₂'x_t 1{z_t > γ} + u_t`,
`z_t = y_{threshold_index, t−d}`. Fit by concentrated least squares / Gaussian
MLE: per-candidate two-regime OLS over the trimmed order-statistic grid,
minimizing `ln det` of the pooled residual covariance (the multivariate
analogue of SETAR's pooled SSR). Lives next to `setar` because it *is* that
model with a vector response; it is deliberately not part of the `var_*`
family, whose IRF/FEVD/forecast surface assumes one linear regime.

**Assumptions.** Two regimes, abrupt switch, threshold variable = an observed
lag of one of the modeled series; enough visits to both sides of `γ`
(`trim` and the `m + 1` per-regime minimum enforce this mechanically); iid
errors within regime for the classical SEs.

**When to use (and when not).** Regime-dependent *system* dynamics keyed to an
observable — output growth above/below a stall speed, spreads in/out of a
stress band — when you want per-regime coefficient matrices you can read. Not
for unobserved regimes (Markov-switching), not for smooth transitions, and
not before `threshold_var_test` says a threshold exists. **Scope honesty:**
two regimes only, and **no regime-dependent (generalized) impulse responses**
— GIRFs à la Koop-Pesaran-Potter (1996) require simulating the fitted
nonlinear system over shock/history distributions and are *deferred*; pointing
the linear `var_irf` machinery at one regime's matrices would answer a
question nobody asked, so the library declines to.

**Key arguments and defaults (and why).** `p` (lags per regime — remember each
regime spends `m = k·p + 1` coefficients *per equation*); `threshold_index=0`,
`delay=1`; `delays=[...]` searches the delay jointly on the common sample
(criteria comparable, first strict improvement wins, exactly as `setar`);
`trim=0.10` (tsDyn's TVAR default; a working compromise between Hansen-Seo's
0.05 and SETAR's 0.15 — the multivariate fit burns k× more parameters per
regime than SETAR, so regimes need more mass than 0.05 buys); `constant=True`.

**How to read the output.** `threshold`/`delay`/`threshold_index` define the
split; `params_low`/`params_high` (rows = equations, columns `[const?,
y_{t−1}…, y_{t−p}…]`) with classical `bse_low`/`bse_high`;
`sigma_low`/`sigma_high` (per-regime ML covariances — a volatility regime
shows up here even when coefficients barely move); `logdet_path` over
`thresholds` — plot it, a sharp V means a well-identified threshold; `aic`/
`bic` (`n·ln det Σ + penalty·q`, `q = 2km + 1` counting the threshold) for
comparing `p` or delay choices across fits on the same sample.

**Failure modes.** All of SETAR's, multiplied by k: the split *always*
improves the criterion (test first); per-regime parameter counts explode with
`k` and `p` (k = 4, p = 2 spends 36 coefficients per regime — regimes of 50
observations give SEs that are noise); the delay search over many candidates
on short samples overfits; a threshold series that rarely crosses `γ` leaves
one regime's dynamics estimated from a handful of episodes.

**Validated how (honest grade).** Same reference landscape as the TVECM: **no
runnable third-party TVAR** (R present, CRAN blocked by the container proxy,
so no `tsDyn::TVAR`), hence an *independent NumPy transcription* golden
(`fixtures/generate_tvar_fixtures.py`): thresholds at 1e-12, coefficients,
classical SEs, per-regime and pooled covariances, criterion path, llf, and ICs
at **1e-10**, over five cases including delay search, an over-specified `p`, a
no-constant fit, and a non-default `threshold_index`. Statistical correctness
by seeded Monte Carlo (`tvar_properties.rs`): 200 replications of a strongly
separated two-regime VAR(1) (γ = 0, T = 400) give threshold median |error|
**0.008** and coefficient biases **|bias| ≤ 0.009** (intercepts and own-lags,
both regimes); the refit criterion equals the scan minimum exactly and the
pooled Σ is the regime-size-weighted mix of the per-regime ones (both
asserted).

**References.** Tong (1983); Tsay (1998, JASA); Lo & Zivot (2001,
Macroeconomic Dynamics); Hubrich & Teräsvirta (2013, survey).

---

## `threshold_var_test` — bootstrap linearity test for the TVAR

**What it estimates.** Whether the VAR's coefficients really switch: H₀ one
linear VAR(p) against the two-regime TVAR at the given delay. The pointwise
statistic is the coefficient-difference quadratic form with **Eicker-White**
covariance evaluated at the null residuals — the heteroskedasticity-robust
sup-Wald in its score (LM) form, the exact multivariate analogue of
`hansen_seo_test`'s statistic (the Wald and LM numerators coincide because the
null residuals are orthogonal to the full-sample regressors) — maximized over
at most `n_grid` trimmed order statistics of `z_t`; p-value by the Hansen
(1996) fixed-regressor wild bootstrap.

**Assumptions.** The null is a linear VAR(p) with a constant; the alternative
switches *all* coefficients at one threshold of one lagged series. The
bootstrap conditions on the regressors and reweights null residuals, so
conditional heteroskedasticity is tolerated.

**When to use (and when not).** Before interpreting any `threshold_var` fit.
Never against a chi-squared table (Davies problem — no asymptotic p-value is
reported, by design). Note the naming honestly: **R `tsDyn`'s `TVAR.LRtest`
is a different convention** — a sup-LR `T(ln det Σ₀ − ln det Σ₁)` with a
residual bootstrap. The two tests answer the same question and both are
bootstrap-calibrated, but their statistic *values* are not comparable numbers.

**Key arguments and defaults (and why).** `trim=0.10` (matches the
estimator), `n_grid=300` (the sup over a subsampled grid, Hansen-Seo's
convention — the estimator's own scan uses every candidate), `n_boot=499`,
`seed=0` — the same reproducible-parallel contract as every bootstrap test in
the library.

**How to read the output.** `stat`, `p_value` (small ⇒ a threshold model is
warranted), `threshold` (the sup's location), `wald_path` over `thresholds`,
`boot_stats` (the realized bootstrap null).

**Failure modes.** **Small-sample over-rejection, measured and documented**:
at T = 150 the seeded null Monte Carlo rejects at **0.100** at the 5% level
(below) — the HC0-weighted quadratic form is liberal when regimes near the
trimming edge hold few observations relative to `k·m` tested coefficients;
prefer larger `trim` in short samples. Rejection does not choose *which*
nonlinearity (STAR and Markov-switching are competitors); power falls when the
tested delay is wrong; pre-testing the delay then testing at the winner
distorts size.

**Validated how (honest grade).** Statistic and path pinned at **1e-10**
against the independent NumPy transcription (three cases,
`fixtures/tvar.json`) — same no-runnable-reference caveat. The bootstrap
p-value by *property* (`tvar_properties.rs`): over 200 seeded linear-VAR
draws (B = 199), rejection **0.100 at 5%** / **0.180 at 10%** at T = 150 (MC
se ≈ 0.02), falling to **0.085** / **0.130** at T = 400; mean null p-value
0.44/0.49; on a strong TVAR it rejects with p ≤ 0.01; bit-identical at any
thread count (asserted).

**References.** Hansen (1996); Hansen & Seo (2002); Tsay (1998); Lo & Zivot
(2001).

```python
import numpy as np, tsecon
rng = np.random.default_rng(3)
# Two-regime bivariate VAR: dynamics switch on the sign of y0's lag.
T, y = 400, np.zeros((400, 2))
for t in range(1, T):
    if y[t-1, 0] <= 0.0:
        A, c = np.array([[0.5, 0.1], [0.2, 0.4]]), np.array([1.0, 0.3])
    else:
        A, c = np.array([[0.1, 0.0], [-0.1, 0.5]]), np.array([-1.0, -0.3])
    y[t] = c + A @ y[t-1] + rng.standard_normal(2)

lin = tsecon.threshold_var_test(y, p=1, threshold_index=0, n_boot=499, seed=0)
print(f"sup-W = {lin['stat']:.1f}, bootstrap p = {lin['p_value']:.3f}")
if lin["p_value"] < 0.05:
    fit = tsecon.threshold_var(y, p=1, delays=[1, 2])
    print("threshold:", round(fit["threshold"], 3), " delay:", fit["delay"])
    print("low  regime:", np.round(fit["params_low"], 2))
    print("high regime:", np.round(fit["params_high"], 2))
```
