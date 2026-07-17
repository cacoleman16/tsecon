# Model card — Cointegration and regime switching

`johansen` · `vecm` · `markov_switching_ar`

Two ways the tidy linear-stationary world breaks. First, series can be
individually nonstationary yet move together — share a long-run equilibrium
(cointegration); differencing away the trends throws that equilibrium away, and
the vector error-correction model keeps it. Second, the parameters themselves
can switch between unobserved regimes — expansion and recession, calm and
crisis — governed by a hidden Markov chain.

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
as you would a VAR lag length).

**How to read the output.** `trace_stat` and `max_eig_stat` (one per null
`r ≤ i`), each with critical values in `trace_crit_90_95_99` /
`max_eig_crit_90_95_99` (columns are the 90/95/99% levels — take column 1 for
the 5% test). `rank_trace_5pct` / `rank_max_eig_5pct` apply the sequential rule
for you. `eig` are the ordered eigenvalues. Reject `r = 0` but not `r ≤ 1` ⇒
rank 1.

**Failure modes.** Using the wrong deterministic convention silently shifts the
critical values; testing series that are not actually I(1); the trace and
max-eigenvalue tests can disagree at the margin — report both.

**Validated against.** statsmodels `coint_johansen` (`det_order=0`,
`k_ar_diff=2`), statistics and critical values (`fixtures/coint.json`).

**References.** Johansen (1988, 1991); Engle & Granger (1987).

---

## `vecm` — vector error-correction model

**What it estimates.** Given the rank `r`, the ML estimate of the VECM: the
cointegrating vectors `beta` (the long-run equilibria — the "leashes"), the
adjustment speeds `alpha` (how fast each equation corrects a disequilibrium),
the short-run dynamics `gamma`, the residual covariance, and the log-likelihood.

**Assumptions.** The rank `coint_rank` is correct (take it from `johansen`);
Gaussian innovations for the ML/log-likelihood; the same deterministic
convention as the rank test.

**When to use.** After `johansen` returns `0 < r < k`. It keeps the levels
information a differenced VAR discards, and `alpha`/`beta` are directly
interpretable — which series bear the burden of adjustment back to equilibrium.

**Key arguments.** `data` (T×k), `k_ar_diff`, `coint_rank` (from the Johansen
test).

**How to read the output.** `beta` (k×r, each column a cointegrating vector —
normalized on the first variable), `alpha` (k×r adjustment speeds; a large
negative entry means that equation does most of the correcting, a near-zero
entry means that variable is weakly exogenous), `gamma` (short-run lag
coefficients), `sigma_u`, `llf`.

**Failure modes.** A wrong rank propagates everywhere; imposing cointegration on
series that are not cointegrated fabricates a spurious equilibrium.

**Validated against.** statsmodels `VECM` (ML estimation; `k_ar_diff=2`,
`coint_rank=1`, `deterministic="n"`) — `alpha`, `beta`, `gamma`, `sigma_u`,
`llf` (`fixtures/coint.json`).

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

fit = tsecon.vecm(data, k_ar_diff=2, coint_rank=1)
print("beta :", np.round(np.asarray(fit["beta"])[:, 0], 3))   # ~[1, -1, 0]: y1 - y2
print("alpha:", np.round(np.asarray(fit["alpha"])[:, 0], 3))
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

**How to read the output.** `transition` (k×k; column-stochastic Markov matrix),
`means`, `variances` (per regime), `expected_durations` (average spell length in
each regime — the persistence read), `loglik`, `converged`, and the
`smoothed_prob_last_regime` / `regimes` series (the smoothed probability path and
the most-likely regime per period). Label regimes by their `means`/`variances`,
not their index (EM does not order them).

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
