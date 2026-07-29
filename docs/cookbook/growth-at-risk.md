# Compute growth-at-risk: the left tail of future GDP growth

A point forecast of GDP growth answers the wrong question for a policymaker.
The question is *how bad could it get*. Growth-at-risk (Adrian, Boyarchenko &
Giannone 2019) answers it by running a quantile regression of `h`-ahead growth
on today's financial conditions, so the whole conditional distribution — not
just its centre — moves with the state of the economy.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(6)                      # 300 quarters
n = 300
nfci = np.zeros(n)                                  # a financial-conditions index
for t in range(1, n):
    nfci[t] = 0.85 * nfci[t - 1] + rng.standard_normal()
g = np.zeros(n)                                     # GDP growth
for t in range(1, n):
    scale = 1.0 + 0.8 * max(nfci[t - 1], 0.0)       # tight conditions -> fatter left tail
    g[t] = 0.3 * g[t - 1] - 0.5 * nfci[t - 1] + scale * rng.standard_normal()

out = tsecon.growth_at_risk(g, nfci.reshape(-1, 1), horizon=4,
                            taus=[0.05, 0.25, 0.5, 0.75, 0.95])
for tau, b in zip(out["taus"], np.asarray(out["params"])):
    print(f"tau={tau:.2f}   const {b[0]:+.3f}   NFCI {b[1]:+.3f}   own growth {b[2]:+.3f}")
```

```
tau=0.05   const -4.163   NFCI -1.270   own growth -0.248
tau=0.25   const -1.419   NFCI -0.796   own growth -0.078
tau=0.50   const +0.047   NFCI -0.495   own growth +0.018
tau=0.75   const +1.181   NFCI -0.299   own growth +0.055
tau=0.95   const +3.041   NFCI -0.083   own growth +0.329
```

## Reading it

This coefficient column *is* the growth-at-risk result. The slope on financial
conditions is `-1.27` at the 5th percentile and `-0.08` at the 95th: a one-unit
tightening drags the bad outcome down by 1.3 percentage points and barely moves
the good one. **Conditions move the left tail, not the mean** — a
conditional-mean regression would have reported the single number `-0.50` and
thrown that asymmetry away.

Each row is a separate quantile regression of `y_{t+4}` on
`[const, conditions, y_t]`, fitted at every observation. `bse` carries the
standard errors in the same layout.

## Today's risk read

```python
print("4-quarter-ahead distribution today:", np.round(out["current"], 2))
print("growth-at-risk (5th percentile)   :", round(out["current"][0], 2))
print("raw quantile paths crossed?       :", out["crossing"])
```

```
4-quarter-ahead distribution today: [-4.84 -1.78 -0.13  1.11  3.26]
growth-at-risk (5th percentile)   : -4.84
raw quantile paths crossed?       : True
```

`current` is the predicted distribution evaluated at the most recent
observation — the headline "growth-at-risk is -4.8%" number. `crossing=True`
reports that the raw fitted quantile paths crossed somewhere in the sample
(a 5th percentile above a 25th, or similar). Separate quantile regressions
carry no monotonicity constraint, so this happens routinely in finite samples.
`rearrange=True` (the default) applies the
Chernozhukov-Fernandez-Val-Galichon (2010) monotone rearrangement, and `fitted` is
the rearranged object while `fitted_raw` preserves the unsorted fits. The flag
is reported rather than hidden because heavy crossing is a specification
warning.

## The whole history, not just today

```python
fitted = np.asarray(out["fitted"])                  # [tau][t], monotone-rearranged
t_tight, t_loose = int(np.argmax(nfci)), int(np.argmin(nfci))
for label, t in (("tightest", t_tight), ("loosest ", t_loose)):
    print(f"{label} (NFCI {nfci[t]:+.2f})   5% {fitted[0, t]:+.2f}"
          f"   50% {fitted[2, t]:+.2f}   95% {fitted[4, t]:+.2f}")
```

```
tightest (NFCI +5.15)   5% -10.24   50% -2.54   95% +1.99
loosest  (NFCI -7.05)   5% +3.55   50% +3.63   95% +5.26
```

At the tightest date the predicted distribution spans 12 percentage points; at
the loosest it spans under 2. The distribution does not merely shift — it
*stretches downward*. Plotting `fitted[0]` against `fitted[4]` over time
reproduces the widening-left-tail chart that made this literature.

## Gotchas

- **`taus` must be strictly increasing** and `horizon >= 1`.
- **`conditions` is 2-D** (`n × k`); reshape a single index with
  `.reshape(-1, 1)`. Do not add a constant column — the function builds
  `[const, conditions, y_t]` itself.
- **Quantile-regression standard errors are the weak link.** `bse` is the
  Powell kernel sandwich; extreme taus in short samples are noisy, and a
  bootstrap is worth the compute for a headline number.
- The impulse-response analogue — how a *shock* moves each quantile — is
  `tsecon.quantile_lp`. Plain cross-sectional quantile regression is
  `tsecon.quantile_regression`.

## See also

- Model card: [Quantile regression and growth-at-risk](../reference/model-cards/quantile.md)
- Guide: [5. Forecasting](../guide/05-forecasting.md) ·
  [3. Honest Inference](../guide/03-inference-toolkit.md)
- Recipes: [Model fat-tailed volatility with GARCH](garch-fat-tails.md) ·
  [Forecast a VAR with prediction intervals](var-forecast-intervals.md)
