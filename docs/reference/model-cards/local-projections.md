# Model card — Local projections

`lp` · `lp_iv` · `lp_multiplier` · `lp_state`

The modern impulse-response workhorse. Instead of inverting a fitted VAR, a
local projection runs one regression *per horizon* — regress the outcome `h`
periods ahead on today's shock (plus controls) — and reads the sequence of
slope coefficients as the impulse response. Robust to misspecification of the
long-run dynamics, and honest about uncertainty at each horizon separately.

---

## `lp` — local projection IRFs

**What it estimates.** For each horizon `h = 0..H`, the coefficient on the shock
in a regression of `y_{t+h}` on `shock_t` and lagged controls. The collected
coefficients are the impulse response. `cumulative` selects which side(s)
accumulate over the horizon: `False`/`"none"` (level response), `True`/
`"outcome"` (the cumulated outcome `sum_j y_{t+j}` on the *contemporaneous*
shock — a cumulative impulse response), or `"both"` (cumulated outcome on
cumulated shock). See [the multiplier trap](#lp_multiplier-integral-multipliers)
before reaching for a multiplier.

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
`maxlags`. `cumulative` as above — note `True` is a cumulative *impulse
response*, not a multiplier.

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
external instrument (a proxy: high-frequency surprise, narrative shock). For a
**multiplier** (e.g. the fiscal multiplier) use `lp_multiplier` below, not
`lp_iv(..., cumulative=True)`.

**Assumptions.** Instrument relevance (a strong first stage) and exogeneity
(the instrument affects `y` only through the impulse). Weak instruments bias
the response and understate uncertainty.

**Key arguments.** `impulse` (endogenous), `instrument`, `horizons`,
`n_lag_controls`, `cumulative` (`False`/`"none"`, `True`/`"outcome"`,
`"both"`). The instrument stays contemporaneous under every cumulation mode.

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

## `lp_multiplier` — integral multipliers

**What it estimates.** The Ramey-Zubairy (2018) **integral multiplier** by
one-step LP-IV. At each horizon `h`:

```text
sum_{j=0..h} y_{t+j} = m_h * sum_{j=0..h} x_{t+j} + c
                     + sum_{l=1..p} (phi_l y_{t-l} + psi_l x_{t-l}) + u_{t,h}
```

with the cumulated impulse instrumented by the **contemporaneous** instrument.
Both sides accumulate over the same window, so `m_h` is extra cumulated outcome
per extra cumulated impulse — a multiplier, in the units of the two series.

**Why this is its own function.** A cumulative response and a multiplier differ
only in whether the *denominator* accumulates too, and nothing about the call
site tells you which one you got. `lp_iv(..., cumulative=True)` accumulates only
the outcome: its coefficient is cumulated output per unit of *contemporaneous*
spending, so it inherits the growth of the spending path instead of measuring
anything per-dollar. On the Ramey-Zubairy data that quantity runs from 7.4 at
h = 4 to 48.7 at h = 20, with a first-stage F of 1.68 — while the actual
multiplier sits flat around 0.7 with F above 10 throughout. Giving the correct
estimator its own name makes the correct thing the easy thing to write.

**Assumptions.** Instrument relevance and exogeneity, as for `lp_iv`. Additional
to `lp_iv`, the design controls for `n_lag_controls` lags of the **impulse** as
well as the outcome: the denominator is now an endogenous quantity whose own
dynamics have to be soaked up for the ratio to be interpretable.

**Key arguments.** `y` (outcome), `impulse` (the endogenous quantity being
accumulated, e.g. government spending), `instrument`, `horizons`,
`n_lag_controls`, `maxlags` (overrides the default HAC bandwidth `h + p`).

**Standard errors — what `se` is.** The multiplier is estimated as a **single
2SLS coefficient**, not as a ratio of two separately estimated responses, so
`se` is the kernel-HAC standard error of the parameter actually being reported.
It is not a delta-method approximation to a ratio, and it is not one leg's
standard error relabelled. The two reduced-form legs are returned as
`cumulative_outcome` and `cumulative_impulse` for transparency and carry **no**
standard errors; by the just-identified IV algebra their ratio equals
`multiplier` to numerical precision.

**How to read the output.** `horizons`, `multiplier`, `se`, `first_stage_f`,
`cumulative_outcome`, `cumulative_impulse`, `nobs_per_h`. Treat `first_stage_f`
below ~10 as a weak-instrument warning at that horizon.

**Failure modes.** A weak instrument in the *cumulated* first stage; an impulse
that is not measured in the same units as the outcome (the coefficient is then
an elasticity-like object, not a multiplier — this is why Ramey-Zubairy divide
by potential output rather than logging).

**Validated against.** The published Ramey & Zubairy (2018) headline: 0.64-0.74
across h = 4..20 on the authors' own data, inside their reported 0.6-0.8 range
— see the [replication](../../examples/replication-ramey-zubairy.md).

**References.** Ramey & Zubairy (2018); Gordon & Krenn (2010, the potential-output
normalisation); Stock & Watson (2018).

```python
r = tsecon.lp_multiplier(y, g, news, horizons=20, n_lag_controls=4)
r["multiplier"][8]      # dollars of output per dollar of spending through h=8
r["se"][8]              # standard error OF the multiplier
r["first_stage_f"][8]   # weak-instrument diagnostic
```

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
`cumulative` (`False`/`"none"`, `True`/`"outcome"`, `"both"`).

**How to read the output.** `horizons` and, per regime, `irf_state1`/`se_state1`
and `irf_state0`/`se_state0`. Compare the two paths — a gap that exceeds the
combined bands is the state-dependence finding.

**Failure modes.** Thin regimes (few periods in one state) give noisy,
unreliable per-state estimates; a state that reacts to the shock contaminates
the split.

**Validated against.** Built on the validated `lp` OLS-HAC machinery (the
interacted design of Ramey-Zubairy 2018); shares the `fixtures/lp.json` golden.

**References.** Ramey & Zubairy (2018); Tenreyro & Thwaites (2016).

The DGP below builds in real state dependence — the shock's effect is 1.5 in
regime 1 and 0.5 in regime 0, decided by the regime the shock *landed* in — so
the two estimated IRFs have something genuine to disagree about:

```python
import numpy as np, tsecon
rng = np.random.default_rng(0)
n = 600
shock = rng.standard_normal(n)
state = ((np.arange(n) // 40) % 2).astype(float)   # 40-period spells: 0,1,0,1,...
y = np.zeros(n)
for t in range(n):
    # The multiplier depends on the regime the shock LANDED in:
    # 1.5 in state 1 (slack), 0.5 in state 0 (tight), both decaying at 0.9.
    y[t] = sum(0.9 ** h * (1.5 if state[t - h] == 1.0 else 0.5) * shock[t - h]
               for h in range(min(t, 20) + 1))
y += 0.3 * rng.standard_normal(n)

out = tsecon.lp_state(y, shock, state, horizons=8, n_lag_controls=2)
print("state 1 IRF (h=0..3):", np.round(out["irf_state1"][:4], 3))
print("state 0 IRF (h=0..3):", np.round(out["irf_state0"][:4], 3))
print("state 1 SE  (h=0..3):", np.round(out["se_state1"][:4], 3))
print("state 0 SE  (h=0..3):", np.round(out["se_state0"][:4], 3))
# state 1 IRF (h=0..3): [1.489 1.505 1.35  1.154]
# state 0 IRF (h=0..3): [0.545 0.396 0.338 0.336]
# state 1 SE  (h=0..3): [0.026 0.08  0.122 0.143]
# state 0 SE  (h=0..3): [0.024 0.042 0.049 0.059]
```

The impact responses recover the true regime multipliers (1.489 vs 0.545
against true 1.5 vs 0.5), each path decays at roughly the true 0.9 rate, and
the gap between them dwarfs the combined standard errors at every horizon —
the state-dependence finding, read exactly as described above.
