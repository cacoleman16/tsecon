# Model card — Local projections

`lp` · `lp_iv` · `lp_state`

The modern impulse-response workhorse. Instead of inverting a fitted VAR, a
local projection runs one regression *per horizon* — regress the outcome `h`
periods ahead on today's shock (plus controls) — and reads the sequence of
slope coefficients as the impulse response. Robust to misspecification of the
long-run dynamics, and honest about uncertainty at each horizon separately.

---

## `lp` — local projection IRFs

**What it estimates.** For each horizon `h = 0..H`, the coefficient on the shock
in a regression of `y_{t+h}` on `shock_t` and lagged controls. The collected
coefficients are the impulse response; `cumulative=True` gives the cumulated
(level) response.

**Assumptions.** The shock is exogenous conditional on the controls (already an
identified shock — a monetary surprise, a narrative series, a Cholesky
innovation). Serial correlation in the horizon-`h` residuals is expected and
must be handled by the standard errors, not assumed away.

**When to use (and when not).** Use when you want horizon-robust responses,
state dependence, or a shock series you trust more than a full VAR
identification; LP responses need no stability or invertibility assumption. Use
a VAR instead when you need a tight, model-consistent long-horizon response
from short samples — LP standard errors widen with the horizon and can be noisy
far out.

**Key arguments and defaults (and why).** `horizons` (H). `n_lag_controls` sets
how many own-lags enter as controls. `se="lag_augmented"` is the **default and
the recommendation** (Montiel Olea & Plagborg-Møller 2021): it augments the
regression with an extra lag so the response is inference-robust even under
persistence, without hand-tuning a bandwidth; `se="hac"` gives Newey-West with
`maxlags`. `cumulative=True` for multipliers.

**How to read the output.** `horizons`, `irf` (the response path), and `se` (one
standard error per horizon — build bands as `irf ± z·se`). Plot `irf` against
`horizons`; the per-horizon `se` widening is a feature, not a defect.

**Failure modes.** Feeding a *non*-identified shock (a raw endogenous variable)
returns a correlation, not a causal response — use `lp_iv`. Too few
`n_lag_controls` leaves the shock endogenous; very long horizons on short
samples give wide, unstable bands.

**Validated against.** statsmodels OLS with HAC (Newey-West) standard errors,
horizon by horizon (`fixtures/lp.json`).

**References.** Jordà (2005); Montiel Olea & Plagborg-Møller (2021,
lag-augmented inference); Plagborg-Møller & Wolf (2021, LP = VAR).

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 400
shock = rng.standard_normal(n)
y = np.zeros(n)                              # y_t = sum_h 0.9^h * shock_{t-h} + noise
for t in range(n):
    y[t] = sum(0.9 ** h * shock[t - h] for h in range(min(t, 20) + 1))
y += 0.3 * rng.standard_normal(n)

out = tsecon.lp(y, shock, horizons=12, n_lag_controls=4)   # lag-augmented SEs (default)
print("IRF (h=0..3):", np.round(out["irf"][:4], 3))        # ~[1.0, 0.9, 0.81, 0.73]
print("SEs (h=0..3):", np.round(out["se"][:4], 3))
```

---

## `lp_iv` — instrumented local projections (LP-IV)

**What it estimates.** The same horizon-by-horizon response, but the impulse
variable is *endogenous* and instrumented — the coefficient is identified by an
external instrument (a proxy: high-frequency surprise, narrative shock). The
natural home for **cumulative multipliers** (e.g. the fiscal multiplier: the
cumulated response of output to a cumulated response of spending).

**Assumptions.** Instrument relevance (a strong first stage) and exogeneity
(the instrument affects `y` only through the impulse). Weak instruments bias
the response and understate uncertainty.

**Key arguments.** `impulse` (endogenous), `instrument`, `horizons`,
`n_lag_controls`, `cumulative=True` for multipliers.

**How to read the output.** `horizons`, `irf`, `se`, and **`first_stage_f`** —
the first-stage F at each horizon. Treat `first_stage_f` below ~10 as a
weak-instrument warning: the point estimate and band at that horizon are not to
be trusted.

**Failure modes.** Weak instruments (low `first_stage_f`) are the dominant
failure; a proxy correlated with other shocks violates exogeneity silently.

**Validated against.** `linearmodels` IV2SLS with a Bartlett-kernel HAC
covariance, horizon by horizon (`fixtures/lp.json`).

**References.** Stock & Watson (2018); Ramey & Zubairy (2018); Jordà, Schularick
& Taylor (2015).

---

## `lp_state` — state-dependent local projections

**What it estimates.** Ramey-Zubairy (2018) interacted local projections: the
shock is interacted with a state indicator so the impulse response is allowed to
differ across regimes (e.g. recession vs expansion, slack vs tight). One
regression per horizon delivers a separate IRF and SE for each state.

**Assumptions.** Same exogeneity requirement as `lp`, plus a state indicator
that is predetermined (does not itself respond to the shock within the period).

**Key arguments.** `state_indicator` (per-period 0/1, or a continuous transition
weight), `horizons`, `n_lag_controls`, `se` (lag-augmented default),
`cumulative`.

**How to read the output.** `horizons` and, per regime, `irf_state1`/`se_state1`
and `irf_state0`/`se_state0`. Compare the two paths — a gap that exceeds the
combined bands is the state-dependence finding.

**Failure modes.** Thin regimes (few periods in one state) give noisy,
unreliable per-state estimates; a state that reacts to the shock contaminates
the split.

**Validated against.** Built on the validated `lp` OLS-HAC machinery (the
interacted design of Ramey-Zubairy 2018); shares the `fixtures/lp.json` golden.

**References.** Ramey & Zubairy (2018); Tenreyro & Thwaites (2016).

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
n = 400
shock = rng.standard_normal(n)
y = np.zeros(n)
for t in range(n):
    y[t] = sum(0.9 ** h * shock[t - h] for h in range(min(t, 20) + 1))
y += 0.3 * rng.standard_normal(n)
state = (np.arange(n) % 2).astype(float)          # a toy alternating regime
out = tsecon.lp_state(y, shock, state, horizons=8, n_lag_controls=2)
print("state 1 IRF (h=0..3):", np.round(out["irf_state1"][:4], 3))
print("state 0 IRF (h=0..3):", np.round(out["irf_state0"][:4], 3))
```
