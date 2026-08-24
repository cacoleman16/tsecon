# Model card — Cointegration and regime switching

`johansen` · `vecm` · `markov_switching_ar` · `setar` · `setar_test`

Two ways the tidy linear-stationary world breaks. First, series can be
individually nonstationary yet move together — share a long-run equilibrium
(cointegration); differencing away the trends throws that equilibrium away, and
the vector error-correction model keeps it. Second, the parameters themselves
can switch between regimes — either *unobserved* states governed by a hidden
Markov chain (`markov_switching_ar`), or *observed* states triggered when a
lagged value of the series itself crosses a threshold (`setar`, with
`setar_test` deciding whether a threshold exists at all).

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

**How to read the output.** `transition` is the k×k Markov matrix in the
**column-stochastic** orientation: `transition[i][j] = P(S_t = i | S_{t-1} = j)`,
so each **column** sums to 1 (matching statsmodels' `regime_transition`, *not*
the row-stochastic textbook convention). The one-step forward propagation of a
probability vector `p` over regimes is therefore `P @ p` — **not** `p @ P`;
transposing by habit silently swaps the entry/exit probabilities. Also
`means`, `variances` (per regime), `expected_durations` (average spell length
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
