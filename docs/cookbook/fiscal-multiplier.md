# Estimate a fiscal multiplier from an instrumented shock

A multiplier is dollars of output per dollar of spending. It is **not** an
impulse response, and the difference is where most of the applied literature's
confusion lives. `tsecon.lp_multiplier` implements the Ramey-Zubairy (2018)
one-step integral estimator: at each horizon it regresses the *cumulated*
outcome on the *cumulated* impulse, instrumented by the contemporaneous shock.
Both sides accumulate over the same window, so the coefficient is a ratio of
dollars to dollars.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(11)                    # 80 years of quarterly data
n = 320
news = rng.standard_normal(n)                      # the exogenous shock (e.g. military news)
u = np.zeros(n)
g = np.zeros(n)
for t in range(1, n):
    u[t] = 0.5 * u[t - 1] + rng.standard_normal()  # business-cycle noise
    g[t] = 0.7 * g[t - 1] + news[t] + 0.8 * u[t]   # spending also REACTS to the cycle
y = 1.2 * g + u                                    # true multiplier = 1.2

m = tsecon.lp_multiplier(y, g, news, horizons=8, n_lag_controls=4)
for h in (0, 2, 4, 6, 8):
    print(f"h={h}   multiplier {m['multiplier'][h]:+.3f}   se {m['se'][h]:.3f}"
          f"   first-stage F {m['first_stage_f'][h]:7.1f}")
```

```
h=0   multiplier +1.274   se 0.054   first-stage F   454.2
h=2   multiplier +1.267   se 0.055   first-stage F   120.2
h=4   multiplier +1.255   se 0.050   first-stage F    76.5
h=6   multiplier +1.224   se 0.061   first-stage F    47.1
h=8   multiplier +1.203   se 0.079   first-stage F    24.6
```

## Reading it

`multiplier[h]` is the cumulative multiplier through horizon `h`: 1.20 at
`h=8` recovers the true 1.2 built into the simulation. `se[h]` is the
kernel-HAC standard error **of that single 2SLS coefficient** — inference on
the multiplier itself, not a delta-method ratio of two impulse responses and
not one leg's standard error relabelled. Build a band the usual way:
`multiplier ± 1.96 · se`.

**`first_stage_f` is not optional reading.** It falls from 454 at `h=0` to 25
at `h=8` — instruments always weaken as the accumulation window grows, and a
multiplier estimated off a weak first stage is a number with a confidence
interval you cannot trust. The conventional `F > 10` rule of thumb is a floor,
not a target. `nobs_per_h` tells you how much sample each horizon actually used.

## Why not just run OLS on the cumulated pair?

Because spending responds to the cycle, so cumulated `g` is correlated with the
cumulated error:

```python
cy = np.cumsum(y)[4:]                              # what OLS on the same cumulated pair says
cg = np.cumsum(g)[4:]
naive = tsecon.ols(cy, np.column_stack([np.ones(cy.size), cg]), se_type="hac")
print(f"OLS on cumulated y and g : {naive['params'][1]:+.3f}   (true 1.2)")
print(f"LP-IV at h=4             : {m['multiplier'][4]:+.3f}")
```

```
OLS on cumulated y and g : +1.375   (true 1.2)
LP-IV at h=4             : +1.255
```

OLS overstates the multiplier by 15% here because it attributes to spending the
part of output that was driving spending. The instrument is what buys you the
causal reading; the accumulation is what makes it a multiplier.

## The trap this function exists to avoid

The other route people take is `tsecon.lp_iv(..., cumulative=True)` and then
dividing the peak outcome response by the peak impulse response. That ratio is
**not** a multiplier: the two peaks generally occur at different horizons, and
the ratio of two estimated responses has no standard error you can read off
either regression. Ramey and Zubairy's point is that the one-step estimator
gets both the estimand and the inference right. Use `lp_multiplier` when you
want dollars per dollar; use `lp_iv` when you want a response path.

## Gotchas

- **`n_lag_controls` controls both series.** It adds that many lags of the
  outcome *and* the impulse; too few and the shock is not exogenous conditional
  on the controls.
- **The instrument must be contemporaneous** with the impulse, and exogenous.
  Feeding a raw endogenous variable gives you a correlation with an
  IV-shaped standard error.
- Units matter. If your data are in logs, convert to a dollar scale (the usual
  sample mean of `Y/G` rescaling) before calling this the multiplier.

## See also

- Model card: [Local projections](../reference/model-cards/local-projections.md)
- Guide: [9. Local Projections](../guide/09-local-projections.md)
- Replication: [Ramey-Zubairy (2018)](../examples/replication-ramey-zubairy.md) —
  the same estimator on the authors' real data
- Recipes: [Get HAC standard errors on a regression](hac-standard-errors.md) ·
  [Driscoll-Kraay and clustered SEs for a panel local projection](panel-lp-standard-errors.md)
