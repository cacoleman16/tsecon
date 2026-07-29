# Find structural breaks when you don't know the dates

If you already know the date, `tsecon.chow_test` is your answer. You usually do
not — and searching for the break yourself invalidates the standard critical
values, because you have implicitly run one test per candidate date. Two
functions handle the search honestly: `sup_f_test` for a single break at an
unknown date, and `bai_perron` when there may be several.

## Step 1: is there a break at all?

```python
import numpy as np, tsecon

rng = np.random.default_rng(4)
n = 300
mu = np.repeat([0.0, 2.0, -1.0], 100)          # true mean shifts after t=99 and t=199
y = mu + rng.standard_normal(n)
X = np.ones((n, 1))                            # intercept-only: a pure mean-shift scan

s = tsecon.sup_f_test(y, X)
print(f"sup-F {s['stat']:.2f}   p {s['p_value']:.4f}   most likely break at t={s['break_date']}")
```

```
sup-F 122.66   p 0.0000   most likely break at t=199
```

The Andrews (1993) sup-F statistic is the largest Chow statistic over all
candidate dates in the trimmed interior; the p-value comes from Hansen's (1997)
response surfaces and already accounts for the search. `f_path` holds the whole
scan if you want to plot it.

Note what it reports: **one** break, at `t=199`. There are two. A single-break
test finds the most prominent one, which is why a rejection is a reason to keep
going, not to stop.

## Step 2: how many, when, and how sure?

```python
bp = tsecon.bai_perron(y, X, max_breaks=5)
print("breaks selected:", bp["n_breaks"], " dates:", np.asarray(bp["break_dates"]).tolist())
for k, d in enumerate(bp["break_dates"]):
    print(f"  break {k+1} at t={d}   95% CI [{bp['ci_lower_95'][k]}, {bp['ci_upper_95'][k]}]")
for k in range(bp["n_breaks"] + 1):
    print(f"  regime {k+1}: t={bp['regime_starts'][k]}..{bp['regime_ends'][k]}"
          f"   mean {bp['params'][k][0]:+.3f} (se {bp['bse'][k][0]:.3f})")
```

```
breaks selected: 2  dates: [99, 199]
  break 1 at t=99   95% CI [95, 103]
  break 2 at t=199   95% CI [196, 202]
  regime 1: t=0..99   mean -0.063 (se 0.102)
  regime 2: t=100..199   mean +2.159 (se 0.097)
  regime 3: t=200..299   mean -0.790 (se 0.102)
```

Both true dates recovered, with Bai (1997) confidence intervals a few
observations wide, and per-regime OLS estimates that reproduce the true means
(0, 2, -1). **Report the break-date confidence intervals.** A break date is an
estimate like any other; quoting it as a fact is the most common way this
literature is misread.

## How the count was chosen

```python
print("sequential supF(l+1|l):", np.round(bp["sup_f_seq"], 2))
print("5% critical values   :", np.round(bp["sup_f_crit"], 2))
```

```
sequential supF(l+1|l): [122.66 248.82   1.45   1.11   0.38]
5% critical values   : [ 8.58 10.13 11.14 11.83 12.25]
```

The sequential procedure tests `l+1` breaks against `l`. The first two
statistics blow past their critical values; the third (1.45 against 11.14) does
not, so the procedure stops at two. `break_dates_by_m` carries the optimal
partition for *every* `m`, so you can see what a three- or four-break model
would have claimed before deciding.

## Gotchas

- **`x` is the design whose coefficients all switch**, and you supply the
  constant. `np.ones((n, 1))` scans for mean shifts; add columns to let slopes
  break too. Every coefficient in `x` switches at every break — there is no
  partial-break option.
- **Autocorrelation inflates sup-F.** The scan assumes serially uncorrelated
  errors under the null. On a persistent series, a rejection may be persistence,
  not a break.
- **An unremoved trend looks like a cascade of mean shifts.** Detrend, or put
  the trend in `x`.
- **`trim=0.15`** leaves 15% of the sample at each end unsearchable — breaks
  near the endpoints cannot be found, by construction.
- Breaks and unit roots are confusable in both directions: a broken-mean
  stationary series fools ADF (Perron 1989), and a random walk fools a break
  scan. Run [the stationarity workflow](unit-root-test.md) alongside this one
  and reconcile the two.

## See also

- Model card: [Structural breaks](../reference/model-cards/structural-breaks.md) ·
  [Specification and diagnostic tests](../reference/model-cards/specification-tests.md)
- Guide: [2. Exploring and Diagnosing](../guide/02-exploration-and-diagnostics.md)
- Recipes: [Test whether a series has a unit root](unit-root-test.md) ·
  [Screen a new series before you model it](screen-a-series.md)
