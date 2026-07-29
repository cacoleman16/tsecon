# Test whether a series has a unit root

Running ADF alone is the classic mistake: failing to reject a unit root is not
evidence *of* one, because ADF has low power near the boundary. The fix is the
confirmatory pair — ADF (null: unit root) **and** KPSS (null: stationary) —
read together. `tsecon.check_stationarity` runs both, classifies the result
into one of four quadrants, and tells you what to do next.

## The recipe

```python
import numpy as np, textwrap, tsecon

rng = np.random.default_rng(7)
walk = np.cumsum(rng.standard_normal(300))          # a genuine unit root

r = tsecon.check_stationarity(walk)
print(f"{r['quadrant']}  ->  {r['recommendation']}")
print(f"ADF   stat {r['adf_statistic']:+.3f}   p {r['adf_p_value']:.3f}")
print(f"KPSS  stat {r['kpss_statistic']:+.3f}   p {r['kpss_p_value']:.3f}")
print(textwrap.fill(r["interpretation"], 72))
```

```
UnitRoot  ->  Difference
ADF   stat -1.691   p 0.436
KPSS  stat +2.625   p 0.010
At the 5% level ADF cannot reject a unit root and KPSS rejects
stationarity — the tests agree the series looks I(1). Difference it once
and re-run this battery on the differences before modeling; regressing
I(1) levels on each other risks spurious regression unless you are
explicitly testing for cointegration.
```

## Reading the quadrant

`quadrant` is the whole point. The two tests have *opposite* nulls, so their
four combinations mean four different things:

| Quadrant | ADF | KPSS | What it means |
|---|---|---|---|
| `Stationary` | rejects | does not reject | Both agree on I(0). Proceed in levels. |
| `UnitRoot` | does not reject | rejects | Both agree on I(1). Difference. |
| `Conflict` | rejects | rejects | Often a deterministic trend or a structural break, not a root. |
| `Inconclusive` | neither rejects | neither rejects | The sample cannot tell. Say so. |

Only the two agreeing quadrants are conclusions. `Conflict` and `Inconclusive`
are the honest outcomes the single-test workflow hides from you, and the
`interpretation` string names the follow-up in each case.

Then confirm the differences are what you thought:

```python
d = tsecon.check_stationarity(np.diff(walk))
print(f"first differences:  {d['quadrant']}  ->  {d['recommendation']}")
```

```
first differences:  Stationary  ->  Proceed
```

## When to reach for Phillips-Perron

`tsecon.phillips_perron` tests the same null as ADF but gets there differently:
it runs the Dickey-Fuller regression with **no** lagged differences and then
corrects the statistic with a Bartlett kernel estimate of the residual long-run
variance. Same nonstandard null distribution, so the same MacKinnon p-values
apply.

```python
pp = tsecon.phillips_perron(walk, regression="c", test_type="tau")
print(f"PP Z-tau {pp['stat']:+.3f}   p {pp['pvalue']:.3f}   bandwidth {pp['lags']}")
print("5% critical value:", round(pp["crit"]["5%"], 3))
```

```
PP Z-tau -1.671   p 0.446   bandwidth 16
5% critical value: -2.871
```

Use it when you would rather not choose an augmentation lag length, when the
errors are heteroskedastic in an unknown way, or as a robustness cross-check —
agreement between ADF and PP is reassuring. **Prefer ADF when you suspect a
large negative MA component**, where PP's size distortion is worst. And quote
`lags`: a PP result without its bandwidth is not reproducible.

## Gotchas

- **Match the deterministics to your alternative.** `regression="c"` (default)
  tests against a stationary-around-a-constant alternative; use `"ct"` when the
  series trends, or a genuine trend will masquerade as a root.
- **A break looks like a root.** A stationary series whose mean jumps
  mid-sample fools ADF (Perron 1989). If the quadrant comes back `UnitRoot` or
  `Conflict` on a series you suspect has a break, run
  [the break scan](structural-breaks.md) before differencing.
- **KPSS p-values are interpolated and clamped to `[0.01, 0.10]`.** A reported
  `p = 0.01` means "at most 0.01", and `p = 0.10` means "at least 0.10" — they
  are table endpoints, not measurements. Setting `alpha` outside that range
  makes the quadrant verdict meaningless even though the call succeeds.
- Testing several series at once? `tsecon.check_series` on a 2-D array gives
  per-series verdicts plus a Johansen cointegration test.

## See also

- Model cards: [Diagnostics and the stationarity workflow](../reference/model-cards/diagnostics.md) ·
  [Phillips-Perron and Phillips-Ouliaris](../reference/model-cards/unit-root-cointegration-tests.md)
- Guide: [2. Exploring and Diagnosing](../guide/02-exploration-and-diagnostics.md)
- Recipes: [Screen a new series before you model it](screen-a-series.md) ·
  [Find structural breaks when you don't know the dates](structural-breaks.md)
